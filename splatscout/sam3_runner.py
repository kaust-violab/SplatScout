"""SAM3 同进程封装 (normlift 环境里 sam3 可直接 import, 已实测): 权重只载一次.

一次调用 = 把模型点的几帧按帧号排好当一段"视频"开会话, 文本提示传播, 返回 SAM3 自己的
物体编号与逐帧掩码. 切段 / 并组在 lifting.py 做, 这里只管出掩码.
"""
import os
import shutil
import tempfile

import cv2
import numpy as np
import torch

from . import config
from . import imaging


class Sam3Runner:
    def __init__(self, version=config.SAM3_VERSION):
        self.version = version
        self._pred = None

    def _predictor(self):
        if self._pred is None:
            from sam3.model_builder import build_sam3_predictor
            self._pred = build_sam3_predictor(version=self.version)
        return self._pred

    def warmup(self, image_path):
        """开题前把一次性开销付掉: 载权重 + 首次推理的 CUDA 预热 (第七次 demo 里第一次 segment 花了 40 s, 之后 1-3 s).
        用场景的一帧做一次文字切割, 结果丢掉. 返回秒数."""
        import time
        t0 = time.time()
        self.run([image_path], "object")
        return time.time() - t0

    def run(self, frame_paths, text=None, whiten_masks=None, wh=None, box=None):
        """frame_paths: 有序的图片路径, 首帧是提示帧. text: 概念词 (可 None). box: 首帧上的示例框, 原图像素
        (x0,y0,x1,y1); SAM3 要归一化 xywh, 用 wh 换算. 文字+框 = 概念+示例 (找这个词里长得像框中物的);
        只给框 = 只按长相找 (词不对也能切到); 只给文字 = 老用法. SAM3 基础版视频接口没有点提示, 只有框.
        whiten_masks: {局部下标: bool mask} 掩码外刷白 (切部件).

        返回 {局部下标: [(obj_id, prob, mask bool HxW)]}; 掩码统一到 wh=(W,H).
        """
        if text is None and box is None:
            raise ValueError("sam3 needs a text and/or a box prompt")
        prompt = dict(type="add_prompt", frame_index=0)
        if text is not None:
            prompt["text"] = text
        if box is not None:
            prompt["bounding_boxes"] = [_norm_xywh(box, wh, frame_paths[0])]
            prompt["bounding_box_labels"] = [1]
        tmp = tempfile.mkdtemp(prefix="sam3_agent_")
        try:
            for i, p in enumerate(frame_paths):
                dst = os.path.join(tmp, f"{i:05d}.jpg")
                if whiten_masks and i in whiten_masks:
                    img = cv2.imread(p)
                    if wh and (img.shape[1], img.shape[0]) != tuple(wh):
                        img = cv2.resize(img, tuple(wh))
                    cv2.imwrite(dst, imaging.whiten_outside(img, whiten_masks[i]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    os.symlink(os.path.abspath(p), dst)
            pred = self._predictor()
            resp = pred.handle_request(request=dict(type="start_session", resource_path=tmp,
                                                    offload_video_to_cpu=True, offload_state_to_cpu=True))
            sid = resp["session_id"]
            per_frame = {}
            try:
                # SAM3 在构建模型时把 bf16 autocast 打开并留给整个进程 (sam3_tracking_predictor 51 行), 传播
                # 路径靠它才跑得通 —— 所以这里不关它, 只在外面再套一层 (与生产 sam3_process 一致); 我们自己的
                # GPU 几何 (投票 / 渲染 / 并组) 各自显式 autocast(enabled=False), 两边互不干扰.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    r = pred.handle_request(request=dict(prompt, session_id=sid))
                    per_frame[0] = r["outputs"]
                    if len(frame_paths) > 1:
                        gen = pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid))
                        try:
                            for r in gen:
                                per_frame[r["frame_index"]] = r["outputs"]
                        finally:
                            gen.close()          # 在本层 with 内关掉生成器, 退出顺序才正确
            finally:
                pred.handle_request(request=dict(type="close_session", session_id=sid))
                torch.cuda.empty_cache()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        out = {}
        for li, o in per_frame.items():
            ids = _np(o["out_obj_ids"]).astype(np.int64).tolist()
            probs = _np(o["out_probs"]) if "out_probs" in o else np.ones(len(ids), np.float32)
            masks = _np(o["out_binary_masks"])
            rows = []
            for k, oid in enumerate(ids):
                m = masks[k].astype(bool)
                if not m.any():
                    continue
                if wh is not None:
                    m = imaging.resize_mask(m, tuple(wh))
                rows.append((int(oid), float(probs[k]), m))
            out[li] = rows
        return out


def _norm_xywh(box, wh, path0):
    """原图像素 (x0,y0,x1,y1) -> SAM3 的归一化 [x, y, w, h], 全部夹进 0-1 且 x+w, y+h 不超 1."""
    if wh is None:
        h, w = cv2.imread(path0).shape[:2]
    else:
        w, h = wh
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = max(0.0, min(x0, x1)), min(float(w), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(h), max(y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f"box {box} is empty after clipping to the {w}x{h} image")
    return [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]


def _np(x):
    """SAM3 输出在 autocast 下可能是 bf16 的 torch 张量, np.asarray 直接转会报 'unsupported ScalarType
    BFloat16' (第二次 demo 全部分割因此失败). 统一: 张量 -> float32/bool -> cpu -> numpy."""
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.dtype == torch.bool:
            return x.cpu().numpy()
        if x.is_floating_point():
            return x.float().cpu().numpy()
        return x.cpu().numpy()
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
        return np.stack([_np(t) for t in x])
    return np.asarray(x)
