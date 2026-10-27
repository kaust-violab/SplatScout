"""viser 可视化: 既能回放已跑完的题, 也能在 agent 跑的同时实时跟着看 (run_bench --live).

三维: 整个场景的高斯 (splat / 点云) + 本步用到的帧 (蓝色带图视锥; 看附近 / 放大看的参照帧橙色; 之前看过的帧灰色)
      + 本步的三维结果 (segment / nearby 的分割组粗成员 = 彩色点带组名, 放大看的区域名叫 region; lift 的实例成员 = 红点;
      答案 = 绿点). 模型画的框描在发给它的图上 (黄), 像素坐标也列在 干了 页签.
侧栏: 题目与状态 / 输入框现场提问 (live) / 步进滑条与前后按钮 / 显示开关 / 三个页签 看到 · 想到 · 干了.

回放:  python -m splatscout.viser_view splatscout/runs/f9f95681fd/cold/demo_5404474b [--port 8080] [--points]
实时:  python -m splatscout.run_bench --scene f9f95681fd --mode cold --live [--port 8080] [--query "..."]
       (不给 --query 就只起服务, 在网页输入框里提问)
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

from . import config

PALETTE = [(230, 60, 60), (60, 180, 60), (255, 150, 0), (240, 110, 200), (80, 80, 210), (90, 210, 210),
           (210, 210, 80), (160, 100, 40)]


def load_splats_cpu(scene):
    """ckpt -> (means, cov 3x3, rgb 0-1, opacity) numpy, 全在 CPU (可视化不占 GPU)."""
    import torch
    from . import legacy
    legacy.bind_scene(scene)
    sp = legacy.scene_paths()
    ck = torch.load(sp.ckpt_path(scene, True), map_location="cpu", weights_only=False)["splats"]
    means = ck["means"].numpy().astype(np.float32)
    q = ck["quats"].numpy().astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)
    s = np.exp(ck["scales"].numpy().astype(np.float64))
    cov = np.einsum("nij,nj,nkj->nik", R, s * s, R).astype(np.float32)
    rgb = np.clip(0.5 + 0.2820948 * ck["sh0"].numpy()[:, 0, :], 0, 1).astype(np.float32)
    opa = 1 / (1 + np.exp(-ck["opacities"].numpy())).astype(np.float32)
    return means, cov, rgb, opa


def read_img(path, long_side=None):
    img = cv2.imread(path)
    if img is None:
        return None
    if long_side:
        h, w = img.shape[:2]
        sc = long_side / max(h, w)
        if sc < 1:
            img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Viewer:
    """一个场景一个 viser 服务. load_run() 回放; 实时模式下由 loop 调 on_start / on_step / on_done."""

    def __init__(self, scene, port=8080, points=False, max_splats=1_500_000, runner=None):
        import viser
        self.scene_name = scene
        self.runner = runner                          # runner(text, mode) -> 在后台跑一题 (实时模式才有)
        prep = config.prep_dir(scene)
        with open(os.path.join(prep, "meta.json")) as f:
            self.meta = json.load(f)
        cams = np.load(os.path.join(prep, "cams.npz"))
        self.c2w, self.K = cams["c2w"], cams["K"]
        self.W, self.H = self.meta["width"], self.meta["height"]
        self.up = np.array(self.meta["up"])
        self.prep = prep
        print(f"[viser] 载入 {scene} 高斯 (CPU) ...", flush=True)
        self.means, cov, rgb, opa = load_splats_cpu(scene)
        N = len(self.means)
        keep = np.random.RandomState(0).choice(N, max_splats, replace=False) if N > max_splats else np.arange(N)

        self.server = viser.ViserServer(port=port, verbose=False)
        self.server.scene.set_up_direction(tuple(self.up.tolist()))
        if points:
            self.base = self.server.scene.add_point_cloud("/scene/points", self.means[keep], (rgb[keep] * 255).astype(np.uint8),
                                                          point_size=0.006, point_shape="circle")
        else:
            self.base = self.server.scene.add_gaussian_splats("/scene/splats", self.means[keep], cov[keep], rgb[keep],
                                                              opa[keep][:, None])
        self.run_dir, self.steps, self.roster, self.result = None, [], "", None
        self.lock = threading.Lock()
        self.state = {"gui": [], "scene": [], "cur_frames": []}
        self.ans_handle = None
        self._build_gui()
        print(f"[viser] 就绪: {len(keep)} 个高斯, http://localhost:{port}", flush=True)

    # ── 侧栏 ──
    def _build_gui(self):
        g = self.server.gui
        self.md_head = g.add_markdown(f"**{self.scene_name}** · 等待任务")
        if self.runner is not None:
            with g.add_folder("现场提问"):
                self.in_query = g.add_text("句子", "")
                self.in_mode = g.add_dropdown("缓存口径", list(config.MODES), "cold")
                self.b_run = g.add_button("开跑")
                self.b_run.on_click(lambda _e: self._launch())
        with g.add_folder("回放"):
            self.slider = g.add_slider("step", 1, 1, 1, 1)
            self.cb_follow = g.add_checkbox("自动跟到最新一步", True)
            self.b_prev = g.add_button("◀ 上一步")
            self.b_next = g.add_button("下一步 ▶")
            self.b_jump = g.add_button("跳到该帧视角")
        with g.add_folder("显示"):
            self.cb_scene = g.add_checkbox("整个场景", True)
            self.cb_seen = g.add_checkbox("之前看过的帧 (灰视锥)", True)
            self.cb_answer = g.add_checkbox("始终显示最终答案 (绿)", False)
            self.pt_size = g.add_slider("高亮点大小", 0.002, 0.03, 0.001, 0.008)
        tabs = g.add_tab_group()
        self.tab_saw, self.tab_think, self.tab_act = tabs.add_tab("看到"), tabs.add_tab("想到"), tabs.add_tab("干了")

        self.slider.on_update(lambda _e: self.show(int(self.slider.value)))
        self.b_prev.on_click(lambda _e: setattr(self.slider, "value", max(1, int(self.slider.value) - 1)))
        self.b_next.on_click(lambda _e: setattr(self.slider, "value", min(max(1, len(self.steps)), int(self.slider.value) + 1)))
        self.b_jump.on_click(lambda _e: self.jump_camera())
        self.cb_scene.on_update(lambda _e: setattr(self.base, "visible", self.cb_scene.value))
        for h in (self.cb_seen, self.cb_answer, self.pt_size):
            h.on_update(lambda _e: self.show(int(self.slider.value)))

    def _launch(self):
        text = self.in_query.value.strip()
        if not text or self.runner is None:
            return
        self.b_run.disabled = True
        mode = self.in_mode.value

        def go():
            try:
                self.runner(text, mode)
            finally:
                self.b_run.disabled = False
        threading.Thread(target=go, daemon=True).start()

    # ── 数据源 ──
    def load_run(self, run_dir):
        """回放一个已跑完的题目录."""
        with open(os.path.join(run_dir, "trajectory.json")) as f:
            d = json.load(f)
        self.on_start(run_dir, d["result"]["text"], d["result"]["qid"], d.get("roster", ""), d["result"].get("mode"))
        self.steps = [s for s in d["steps"] if "step" in s]
        self.on_done(d["result"])
        self.slider.max = max(1, len(self.steps))
        self.slider.value = 1
        self.show(1)

    def on_start(self, run_dir, text, qid, roster, mode):
        with self.lock:
            self.run_dir, self.steps, self.roster, self.result = run_dir, [], roster, None
            self.text, self.qid, self.mode = text, qid, mode
        self._clear()
        if self.ans_handle is not None:
            self.ans_handle.remove()
            self.ans_handle = None
        self.slider.max = 1
        self.slider.value = 1
        self.md_head.content = f"**“{text}”**  \n{qid} · {mode} · 进行中…"
        self._md(self.tab_saw, "开题消息 = 名册 + 句子:\n```\n" + roster + "\n```")

    def on_step(self, rec):
        """外壳每步落完轨迹后调一次 (来自 agent 线程)."""
        with self.lock:
            self.steps.append(rec)
            n = len(self.steps)
        self.slider.max = n
        self.md_head.content = f"**“{self.text}”**  \n{self.qid} · {self.mode} · 进行中… step {n}: {rec.get('tool', 'text')}"
        if self.cb_follow.value:
            self.slider.value = n                     # 触发 show(n)
            if int(self.slider.value) == n:
                self.show(n)

    def on_done(self, result):
        self.result = result
        fin = result["final"]
        ids = " ".join("#" + str(i) for i in (fin.get("ids") or fin.get("candidates") or []))
        self.md_head.content = (f"**“{result['text']}”**  \n{result['qid']} · {result.get('mode')} · {result['steps']} steps · "
                                f"{result['seconds']} s · in {result['usage'].get('input_tokens_this_query')} tokens  \n"
                                f"final: **{fin.get('status')} {ids}**")
        idx = np.asarray(fin.get("gaussians", []), np.int64)
        if self.ans_handle is not None:
            self.ans_handle.remove()
        self.ans_handle = self.server.scene.add_point_cloud(
            "/answer", self.means[idx] if len(idx) else np.zeros((1, 3), np.float32), (40, 200, 60),
            point_size=0.008, point_shape="circle", visible=False)
        if self.steps:
            self.show(int(self.slider.value))

    # ── 绘制 ──
    def _frustum(self, name, fid, color, with_image, scale):
        import viser.transforms as tf
        pose = tf.SE3.from_matrix(self.c2w[fid])
        fov = 2 * np.arctan(self.H / (2 * self.K[fid][1, 1]))
        img = read_img(os.path.join(self.prep, "thumbs", f"f{fid:05d}.jpg"), 384) if with_image else None
        self.state["scene"].append(self.server.scene.add_camera_frustum(
            name, fov=float(fov), aspect=self.W / self.H, scale=scale, color=color, image=img,
            wxyz=pose.rotation().wxyz, position=pose.translation(), line_width=1.5))

    def _clear(self):
        for h in self.state["gui"] + self.state["scene"]:
            try:
                h.remove()
            except Exception:                        # noqa: BLE001
                pass
        self.state["gui"], self.state["scene"] = [], []

    def _md(self, tab, text):
        with tab:
            self.state["gui"].append(self.server.gui.add_markdown(text))

    def _img(self, tab, path, label):
        im = read_img(path, 640)
        if im is None:
            return
        with tab:
            self.state["gui"].append(self.server.gui.add_image(im, label=label, format="jpeg", jpeg_quality=80))

    def jump_camera(self):
        import viser.transforms as tf
        frames = self.state["cur_frames"]
        if not frames:
            return
        pose = tf.SE3.from_matrix(self.c2w[frames[0]])
        for client in self.server.get_clients().values():
            client.camera.wxyz = pose.rotation().wxyz
            client.camera.position = pose.translation()
            client.camera.fov = float(2 * np.arctan(self.H / (2 * self.K[frames[0]][1, 1])))

    def show(self, i):
        with self.lock:
            steps = list(self.steps)
        if not steps or i < 1 or i > len(steps):
            return
        self._clear()
        s = steps[i - 1]
        viz = s.get("viz") or {}
        frames = viz.get("frames", [])
        self.state["cur_frames"] = frames
        is_text = "tool" not in s
        tool = s.get("tool", "text")
        means, up, run_dir = self.means, self.up, self.run_dir

        # 三维
        if self.cb_seen.value:
            seen = []
            for prev in steps[:i - 1]:
                seen += (prev.get("viz") or {}).get("frames", [])
            for fid in sorted(set(seen) - set(frames))[:80]:
                self._frustum(f"/step/seen/f{fid}", fid, (150, 150, 150), False, 0.10)
        for fid in frames:
            self._frustum(f"/step/cur/f{fid}", fid, (20, 120, 240), True, 0.22)
        if viz.get("ref_frame") is not None:
            self._frustum(f"/step/ref/f{viz['ref_frame']}", viz["ref_frame"], (240, 160, 20), True, 0.22)
        for gi, (nm, g) in enumerate(sorted((viz.get("groups") or {}).items())):
            idx = np.load(os.path.join(run_dir, g["file"]))
            if len(idx) == 0:
                continue
            c = PALETTE[gi % len(PALETTE)]
            self.state["scene"].append(self.server.scene.add_point_cloud(f"/step/group/{nm}", means[idx], c,
                                                                         point_size=self.pt_size.value, point_shape="circle"))
            self.state["scene"].append(self.server.scene.add_label(f"/step/label/{nm}", f"{nm} ({g['n']})",
                                                                   position=means[idx].mean(axis=0) + up * 0.05))
        if viz.get("lift"):
            L = viz["lift"]
            idx = np.load(os.path.join(run_dir, L["file"]))
            self.state["scene"].append(self.server.scene.add_point_cloud("/step/lift", means[idx], (235, 40, 40),
                                                                         point_size=self.pt_size.value * 1.2, point_shape="circle"))
            self.state["scene"].append(self.server.scene.add_label(
                "/step/lift_label", f"#{L['id']} {L['word']} ({L['n']}, {L['grade']})", position=means[idx].mean(axis=0) + up * 0.08))
        if self.ans_handle is not None:
            self.ans_handle.visible = self.cb_answer.value or tool in ("answer", "report")

        # 看到
        saw = s.get("saw") or {}
        lines = [f"### step {i} · 输入",
                 f"工具集: {'只剩 answer/report' if any('Budget' in n for n in saw.get('notes', [])) else '全部'}"]
        for n in saw.get("notes", []):
            lines.append(f"> **外壳注入**: {n}")
        if saw.get("omitted_steps"):
            lines.append("已换成一行摘要的图: " + ", ".join(f"step {x}" for x in saw["omitted_steps"]))
        if i == 1:
            lines.append("开题消息 = 名册 + 句子:\n```\n" + (self.roster or "") + "\n```")
        self._md(self.tab_saw, "\n\n".join(lines))
        by_step = {x["step"]: x.get("images", []) for x in steps}
        shown = saw.get("image_steps", [])
        if not saw:
            self._md(self.tab_saw, "_旧轨迹, 未记录本步上下文_")
        elif not shown:
            self._md(self.tab_saw, "_上下文里还没有图_")
        for st_ in shown:
            for im in by_step.get(st_, []):
                self._img(self.tab_saw, os.path.join(run_dir, im["file"]), f"step {st_}: {im['caption']}")

        # 想到
        th = s.get("thought") or {}
        t = [f"### step {i} · 模型文字输出"]
        if th.get("reasoning"):
            t.append("**思考摘要**\n\n" + "\n\n".join(th["reasoning"]))
        elif th:
            t.append(f"_无思考摘要 (端点未返回); 思考 token {th.get('reasoning_tokens') or 0}_")
        else:
            t.append("_旧轨迹, 未记录_")
        txt = th.get("text") or s.get("model_text")
        if txt:
            t.append("**模型说**: " + txt)
        if not is_text:
            t.append("**决定调用**: `" + tool + "(" + json.dumps(s.get("args"), ensure_ascii=False) + ")`")
        self._md(self.tab_think, "\n\n".join(t))

        # 干了
        a = [f"### step {i} · {tool}"]
        if is_text:
            a.append("_本步没有调工具_")
        else:
            a.append("`" + tool + "(" + json.dumps(s.get("args"), ensure_ascii=False) + ")`" + (" **← 出错**" if s.get("error") else ""))
            a.append(f"耗时 工具 {s.get('tool_seconds')} s · LLM {s.get('llm_seconds')} s · token in {s.get('input_tokens')} / out {s.get('output_tokens')}")
            a.append("```\n" + (s.get("result") or "") + "\n```")
            if frames:
                a.append("三维里蓝色带图视锥 = 本步用到的帧 " + " ".join(f"f{f}" for f in frames))
            if viz.get("box"):
                a.append(f"模型画的框 (原图像素, 已描黄在图上): f{viz.get('box_frame')} {viz['box']}")
            if viz.get("groups"):
                a.append("三维里彩色点 = 各分割组的粗认领成员 (" + ", ".join(f"{nm} {g['n']}" for nm, g in viz["groups"].items())
                         + "; 'region' = 放大看的区域)")
            if viz.get("lift"):
                a.append(f"三维里红点 = 提升出的 #{viz['lift']['id']} 的 {viz['lift']['n']} 个高斯"
                         + (f" (来自 {viz['lift']['source']} = {' '.join(viz['lift'].get('groups', []))})" if viz['lift'].get('source') else ""))
            if tool in ("answer", "report") and self.result:
                a.append(f"三维里绿点 = 最终答案的 {len(self.result['final'].get('gaussians', []))} 个高斯")
        self._md(self.tab_act, "\n\n".join(a))
        for im in s.get("images", []):
            self._img(self.tab_act, os.path.join(run_dir, im["file"]), im["caption"])

    def serve_forever(self):
        while True:
            time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--points", action="store_true", help="场景用点云而不是 splat 渲染 (更轻)")
    ap.add_argument("--max-splats", type=int, default=1_500_000)
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "trajectory.json")) as f:
        d = json.load(f)
    scene = d["result"].get("scene") or os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    v = Viewer(scene, port=args.port, points=args.points, max_splats=args.max_splats)
    v.load_run(run_dir)
    print("[viser] 回放就绪. 侧栏滑条 / 按钮翻步; Ctrl+C 退出.", flush=True)
    v.serve_forever()


if __name__ == "__main__":
    main()
