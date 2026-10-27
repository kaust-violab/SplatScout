"""预定义的视角无关空间关系函数 (3D grounding 路线 1, 仿 ZSVG3D step_interpreters).

输入: 锚点物体的名字 + 目标相对锚点的关系  →  输出: 场景中目标物体的名字.

设计要点 (相对 ZSVG3D 的改动都源于真实失效案例):
  1. 排序用 OBB 表面距离 (27 点阵互测), 不用中心距 — 否则大物体 (桌子) 系统性失效;
  2. 双粒度: SAM3 的 2D 实例身份经反投影完整保留 (labels.npy / instances.json),
     候选枚举默认**实例级** ("离烤面包机最近的那把刀"必须回答具体哪把, 厨房有14把) —
     grounding 的核心正是在同类干扰项里挑一个; 名字级聚合 (碎饼干/桌子碎片并为一个
     逻辑物体) 降级为可选视图 (granularity="name" / CLI --by-name / viewer 勾选).
     锚点可用 "#id" 或 "名字#id" 锚定到具体实例 (如 knife#7), 锚定后同名兄弟
     不再被排除, 可以问"离 7 号刀最近的刀";
  3. 精判用高斯点级复核 (两段式): OBB 紧盒 (2-98 分位) 会给贴合物体制造 ~0.2m 假间隙
     (bear nose 长在熊上却测出 0.18m), touching 先 OBB 粗筛 top-k 再回到实例点云
     (256 个 FPS 采样高斯中心, 减去两端高斯尺度) 精测, 阈值相对物体尺度而非场景对角线;
  4. inside 用包含率 (目标点落在锚点盒内的比例) 而非中心判据 — 治凹形锚点 (桌下空腔)
     和跨放物体; 0.3~0.7 的中间带白送谓词 inserted (部分插入);
  5. 每条结果带 conf (绝对满足分 [0,1], 每候选独立) 和 flags (支撑面/容器几何证据):
     conf 高 = 更满足该关系; 唯一性由上层 margin 判; flags scene_scale/contains_anchor 供跳过判据.

重力方向估计 (首次用到重力谓词时自动算并缓存 gravity.json, --gravity 强制重算):
  1. 所有帧相机 up 轴 (COLMAP/OpenCV 系: 相机系 -y) 变到世界系取平均 -> u0,
     打印前 5 个相机 up 做 sanity check (应大致同向且相对场景内容朝上);
  2. RANSAC 在实例点云上多次拟合大平面得候选集; 每个候选取法向 n 与 ±u0
     夹角的较小者, 保留 < 45° 的, 其中选内点最多的平面 (桌面/地面), 内点 SVD 精修;
  3. dot(n, u0) < 0 则 n = -n, 得朝上的 up; 重力方向 = -up, 所有谓词按 up 书写;
  4. 元数据 {up, method: "plane_snap", agreement_deg: angle(up, u0), confidence};
  5. 兜底: 没有平面过 45° 门 -> up = u0, method="cam_only", confidence 调低.

支持的关系 (同义词共用实现):
  closest / nearest / near / next to / beside / by   离锚点最近的
  farthest / furthest                                 离锚点最远的
  touching / adjacent / against                       与锚点表面接触 (点级精判)
  inside / in / within                                包含率 >= 0.7 在锚点盒内
  inserted / stuck in / partially inside              包含率 0.3~0.7 (部分插入)
  contains / containing / holding                     锚点的点大部分落在其盒内 (inside 反向)
  between                                             在两个锚点之间 (需第二个名字, 面感知垂距)
  largest / biggest, smallest                         同名候选按 OBB 体积取极值 ("all"=全场)
  distance / dist                                     报告两物体的表面距离 (工具)
  ── 重力相关 (按估出的 up 书写) ──
  above / over                                        在锚点上方 (垂直隔开 + 水平投影重叠)
  below / under / beneath                             在锚点下方 (对称)
  on / atop / on top of                               搁在锚点上 (接触 + 底贴锚点顶 + 水平重叠)
  higher / lower                                      中心高度高/低于锚点 (纯比较, 无水平约束)
  highest / topmost, lowest                           同名候选里最高/最低 (锚点名当过滤词)
  ── 视角相关 (判定方法仿 ZSVG3D view_interpreters, 但观察者不用它的
     "场景中心正交伪视角", 而是当前渲染相机: viewer 里 = 你此刻的视角,
     CLI 里 = --frame 指定的数据集相机) ──
  left / right                                        锚点的左/右侧 (紧邻优先;
                                                        conf = 正左对齐度 x 近邻衰减)
  front / behind                                      锚点的前/后 (比视线深度, 贴着锚点的优先)
  leftmost / rightmost                                同名候选里屏幕最左/最右 (锚点名当过滤词)
  middle / mid                                        当前视野里最居中的那个 (距画面中心角偏最小)

数据依赖:
  ckpts/<scene>/sam3_lift/instances.json        OBB/名字/状态/axis_conf (export 产出)
  ckpts/<scene>/sam3_lift/instance_points.npz   实例点缓存 (缺失时降级纯 OBB 并告警)
  scenes/LeRF/<scene>/cameras.json              相机位姿 (重力估计的 u0 先验)
  ckpts/<scene>/sam3_lift/gravity.json          重力估计缓存 (自动生成)

用法 (任何带 numpy 的环境, CPU):
  python pre_relation.py --list                              # 看场景物体
  python pre_relation.py closest "stuffed bear"              # 默认实例级: 答案带 #id
  python pre_relation.py closest "knife#7"                   # 锚定7号刀, 其他刀是合法候选
  python pre_relation.py on plate --by-name                  # 名字级视图 (碎片并集)
  python pre_relation.py touching plate
  python pre_relation.py between sheep "stuffed bear"
  python pre_relation.py inside table
  python pre_relation.py inserted table
  python pre_relation.py largest all
  python pre_relation.py distance sheep apple
  python pre_relation.py --gravity                           # (重)估重力并打印诊断
  python pre_relation.py on table                            # 什么搁在桌上
  python pre_relation.py above plate                         # 什么在盘子正上方
  python pre_relation.py highest all                         # 全场最高的物体
  SCENE=figurines python pre_relation.py closest jake        # 换场景
  python pre_relation.py left plate --frame frame_00002      # 以该帧相机为视角
  python pre_relation.py --viewer                            # 可视化 (normlift 环境,
      # 浏览器 :8080, 选关系+填锚点 -> 红色分割出可信度最高的物体, 锚点蓝色;
      # 视角相关关系直接用你当前的渲染视角; "滤除桌面"勾选后按几何证据
      # (大+扁+水平) 排除支撑面, 不读名字)

作为库 (后续 agent 工具就是包这个):
  from pre_relation import relate
  relate("closest", "stuffed bear", topk=5)
  # -> [{'name','id','dist','conf','status','flags':{...}}, ...]  conf=绝对满足分[0,1], 高=更满足
"""

import argparse
import json
import os
import re

import numpy as np

# ─── 数据 ────────────────────────────────────────────────────────────────────
from . import scene_paths

SCENE = os.environ.get("SCENE", "")
LIFT_DIR = scene_paths.lift_dir(SCENE)
INSTANCES_JSON = os.path.join(LIFT_DIR, "instances.json")
POINTS_NPZ = os.path.join(LIFT_DIR, "instance_points.npz")
CAMERAS_JSON = os.path.join(scene_paths.scene_dir(SCENE), "cameras.json")
GRAVITY_JSON = os.path.join(LIFT_DIR, "gravity.json")

# touching: 点级距离 <= TOUCH_SCALE_FRAC x 两物体特征尺寸的较小者 (相对阈值, 跨场景稳定)
TOUCH_SCALE_FRAC = 0.10
TOUCH_TOPK = 10          # OBB 粗筛进入点级精判的实例数 (便宜的先上, 精确的按需)
# inside / inserted 的包含率分界
INSIDE_HI = 0.7
INSIDE_LO = 0.3
# 细长物体插入的轴延长判定 (勺在杯里): 重建拍不到容器内部, 勺只有露出杯口那截有点,
# 包含率被压低. 包含率在 [INSERT_AXIS_LO, INSIDE_HI) 的细长物体, 用主轴延长线是否穿过
# 容器内部来补判 —— 0.1 下限天然排除"平躺着指向杯子"的假阳性 (那种包含率≈0).
INSERT_AXIS_LO = 0.10       # 启用延长线判定的包含率下限
INSERT_AXIS_ASPECT = 2.5    # "细长"门: 最长轴 / 次长轴 >= 此值 (勺 3~9, 杯盘 <2)
# between: 投影参数 t 的合法区间 与 面感知垂距 / 两锚距离 的上限
BETWEEN_T = (0.10, 0.90)
BETWEEN_PERP = 0.35
# flags: 体积超过全场中位数体积的这么多倍 -> 标 scene_scale (支撑面/大结构嫌疑).
# 不用体积分位: N 小时 p95 贴顶, 且稀疏碎片的虚胖 OBB 会挤占头部分位
SCENE_SCALE_MULT = 20
# 重力估计 (RANSAC 平面吸附)
RANSAC_ITERS = 1000
PLANE_EPS_FRAC = 0.005     # 平面内点带宽 = 场景对角线 x 此值
PLANE_ANGLE_GATE = 45.0    # 平面法向与 ±u0 夹角门 (度)
PLANE_MIN_INLIER = 0.15    # 内点占比下限, 不到不算"大平面"
# 重力谓词容差 (相对物体特征尺寸)
ABOVE_VGAP_SLACK = 0.50    # above/below: 判定门允许的负间隙(交叠)上限 (x 锚点尺寸).
                           # 放宽原因: 3DGS 实例的盒/点边界歪斜, 锚点顶面易估高, 真在上方也常出假负间隙
ABOVE_OVERLAP_MIN = 0.30   # above/below: 目标点有锚点垫底(/盖顶)的比例下限
ON_OVERLAP_MIN = 0.15      # on: 同上但更宽 (接触已保证邻近)
ON_ALIGN_TOL = 0.30        # on: |目标底 - 锚点局部顶| 上限 (x s_min, 另加采样稀疏度松量)


def load_objects():
    """读 instances.json + 点缓存 -> 实例列表.

    每项: {id, name, status, centroid, c/R/h (OBB), s (特征尺寸=体积^(1/3)),
           pts (K,3) 采样高斯中心, pscale (K,) 各点高斯尺度}  (缓存缺失时 pts=None)
    """
    with open(INSTANCES_JSON) as f:
        instances = json.load(f)["instances"]
    pts = None
    if os.path.exists(POINTS_NPZ):
        pts = np.load(POINTS_NPZ)
    else:
        print(f"[warn] 找不到 {POINTS_NPZ}, touching/inside 降级为纯 OBB 判定 "
              f"(跑 back_project.py 可生成)")
    objs = []
    for inst in instances:
        obb = inst["obb"]
        h = np.array(obb["extent"], dtype=np.float64) / 2
        objs.append({
            "id": inst["id"],
            "name": inst["names"][0],
            "names": list(inst["names"]),              # 全部别名 (2026-08-12 跨词并回归: 主名在首)
            "status": inst.get("status", "?"),
            "centroid": np.array(inst["centroid"], dtype=np.float64),
            "c": np.array(obb["center"], dtype=np.float64),
            "R": np.array(obb["axes"], dtype=np.float64),      # (3,3) 行向量=主轴
            "h": h,
            "s": float(np.cbrt(np.prod(np.maximum(h * 2, 1e-6)))),  # 特征尺寸
            "pts": pts[f"pts_{inst['id']}"].astype(np.float64) if pts is not None else None,
            "pscale": pts[f"scale_{inst['id']}"].astype(np.float64) if pts is not None else None,
        })
    return objs


def scene_diag(objs):
    P = np.stack([o["centroid"] for o in objs])
    return float(np.linalg.norm(P.max(0) - P.min(0)))


# ─── 重力方向估计 ────────────────────────────────────────────────────────────
def camera_up_prior(verbose=True):
    """步骤1: 所有帧相机 up 轴 (相机系 -y, COLMAP/OpenCV 约定) 变到世界系取平均.

    world_up_i = R_c2w @ [0,-1,0] = -R_c2w 的第二列.
    """
    with open(CAMERAS_JSON) as f:
        cams = json.load(f)
    ups = []
    for c in cams:
        R_c2w = np.array(c["rotation"], dtype=np.float64)   # C2W 旋转
        ups.append(-R_c2w[:, 1])
    ups = np.stack(ups)
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    u0 = ups.mean(axis=0)
    u0 /= np.linalg.norm(u0)
    if verbose:
        print(f"[gravity] sanity check: 前 5 个相机的 up (世界系), 应大致同向:")
        for i, u in enumerate(ups[:5]):
            print(f"  cam{i}: [{u[0]:+.3f} {u[1]:+.3f} {u[2]:+.3f}]  dot(u0)={u @ u0:+.3f}")
        agree = float((ups @ u0).mean())
        print(f"[gravity] u0 = [{u0[0]:+.3f} {u0[1]:+.3f} {u0[2]:+.3f}], "
              f"{len(ups)} 台相机平均一致性 dot={agree:.3f}")
    return u0


def ransac_plane_snap(P, u0, diag, verbose=True):
    """步骤2/3: RANSAC 拟合大平面, 法向与 ±u0 夹角<45° 的候选里取内点最多者,
    内点 SVD 精修法向, 再按 u0 定向 (dot<0 翻转). 返回 (up, inlier_frac) 或 None."""
    eps = PLANE_EPS_FRAC * diag
    rs = np.random.RandomState(0)
    best = None   # (n_inlier, normal)
    for _ in range(RANSAC_ITERS):
        i, j, k = rs.choice(len(P), 3, replace=False)
        n = np.cross(P[j] - P[i], P[k] - P[i])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        # 与 ±u0 夹角的较小者过 45° 门 (先过门再数内点, 省时且排除竖直墙面)
        cos = abs(float(n @ u0))
        if cos < np.cos(np.deg2rad(PLANE_ANGLE_GATE)):
            continue
        n_in = int((np.abs((P - P[i]) @ n) < eps).sum())
        if best is None or n_in > best[0]:
            best = (n_in, n, P[i].copy())
    if best is None or best[0] < PLANE_MIN_INLIER * len(P):
        return None
    n_in, n, p0 = best
    # SVD 精修: 内点的最小奇异向量
    inl = P[np.abs((P - p0) @ n) < eps]
    _, _, Vt = np.linalg.svd(inl - inl.mean(axis=0), full_matrices=False)
    n = Vt[-1] / np.linalg.norm(Vt[-1])
    if float(n @ u0) < 0:          # 定向: 朝上
        n = -n
    if verbose:
        print(f"[gravity] plane_snap: 内点 {n_in}/{len(P)} ({n_in/len(P):.1%}), "
              f"精修后 up = [{n[0]:+.3f} {n[1]:+.3f} {n[2]:+.3f}]")
    return n, n_in / len(P)


def estimate_gravity(objs, verbose=True):
    """完整流程 -> {up, method, agreement_deg, confidence, u0}. 写入 gravity.json."""
    u0 = camera_up_prior(verbose)
    P = (np.concatenate([o["pts"] for o in objs if o["pts"] is not None])
         if any(o["pts"] is not None for o in objs) else None)
    snap = ransac_plane_snap(P, u0, scene_diag(objs), verbose) if P is not None else None

    if snap is not None:
        up, inlier_frac = snap
        agreement = float(np.rad2deg(np.arccos(np.clip(up @ u0, -1, 1))))
        meta = {
            "up": [round(float(x), 6) for x in up],
            "method": "plane_snap",
            "agreement_deg": round(agreement, 2),
            "confidence": round(float(np.cos(np.deg2rad(agreement))), 3),
            "inlier_frac": round(float(inlier_frac), 3),
            "u0": [round(float(x), 6) for x in u0],
        }
    else:
        meta = {   # 步骤5 兜底: 只有相机先验
            "up": [round(float(x), 6) for x in u0],
            "method": "cam_only",
            "agreement_deg": 0.0,
            "confidence": 0.4,
            "u0": [round(float(x), 6) for x in u0],
        }
        if verbose:
            print("[gravity] 没有平面通过 45° 门, 兜底 up=u0 (cam_only, 置信度调低)")
    os.makedirs(os.path.dirname(GRAVITY_JSON), exist_ok=True)
    with open(GRAVITY_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[gravity] method={meta['method']}, 与相机先验夹角 "
              f"{meta['agreement_deg']}°, confidence={meta['confidence']} -> {GRAVITY_JSON}")
    return meta


def get_up(objs, force=False, verbose=True):
    """读缓存的重力估计 (没有则现算). 返回 (up (3,), meta dict)."""
    if not force and os.path.exists(GRAVITY_JSON):
        with open(GRAVITY_JSON) as f:
            meta = json.load(f)
    else:
        meta = estimate_gravity(objs, verbose)
    return np.array(meta["up"], dtype=np.float64), meta


# ─── OBB / 点级几何 ──────────────────────────────────────────────────────────
def point_to_obb(p, obb):
    """点到 OBB 的距离 (盒内为 0). 精确."""
    q = obb["R"] @ (p - obb["c"])
    d = np.maximum(np.abs(q) - obb["h"], 0.0)
    return float(np.linalg.norm(d))


def points_in_obb(P, obb):
    """一批点是否在 OBB 内 -> bool (N,). 矢量化."""
    Q = (P - obb["c"]) @ obb["R"].T          # 世界 -> 局部 (行向量轴)
    return (np.abs(Q) <= obb["h"]).all(axis=1)


def obb_lattice(obb):
    """OBB 表面 3x3x3 点阵 (角/棱中/面心/中心, 27 点)."""
    g = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
    return obb["c"] + (g * obb["h"]) @ obb["R"]


def obb_distance(a, b):
    """两 OBB 表面距离近似 (双向 27 点阵互测取最小; 相交为 0). 只偏大不偏小."""
    d1 = min(point_to_obb(p, b) for p in obb_lattice(a))
    d2 = min(point_to_obb(p, a) for p in obb_lattice(b))
    return min(d1, d2)


def point_level_dist(a, b):
    """点级表面距离: 两实例采样点最小对距 - 两端高斯尺度, 截到 >=0.

    分位缩盒的假间隙 (bear nose 0.18m) 在点级自动归零. 需要点缓存."""
    if a["pts"] is None or b["pts"] is None:
        return obb_distance(a, b)   # 降级
    D = np.linalg.norm(a["pts"][:, None, :] - b["pts"][None, :, :], axis=-1)
    i, j = np.unravel_index(int(D.argmin()), D.shape)
    return max(0.0, float(D[i, j] - a["pscale"][i] - b["pscale"][j]))


def point_seg_dist(p, a, b):
    """点到线段 ab 的垂距与投影参数 t."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a)), 0.5
    t = float((p - a) @ ab / L2)
    perp = float(np.linalg.norm(p - (a + np.clip(t, 0, 1) * ab)))
    return perp, t


def seg_to_obb_dist(obb, a, b):
    """线段到 OBB 表面的近似距离: 盒面 27 点阵到线段垂距取最小 (面感知, 治大而扁的物体)."""
    return min(point_seg_dist(p, a, b)[0] for p in obb_lattice(obb))


# ─── 名字检索 / 聚合 ─────────────────────────────────────────────────────────
def find_by_name(objs, name):
    """按名字子串检索实例; 支持 "#7" / "knife#7" 锚定到具体实例.

    锚定 (#id) 的实例带 _pinned 标记 (浅拷贝, 不污染共享对象):
    _excluded 据此放行同名兄弟 — "离 7 号刀最近的刀"才问得出来."""
    q = name.strip().lower()
    if q in ("all", "*", ""):
        return list(objs)
    m = re.match(r"^(.*?)#(\d+)$", q)

    def _names(o):
        return [n.lower() for n in (o.get("names") or [o["name"]])]

    if m:
        prefix, oid = m.group(1).strip(), int(m.group(2))
        hits = [o for o in objs
                if o["id"] == oid and (not prefix or any(prefix in n for n in _names(o)))]
        if not hits:
            raise SystemExit(f'找不到实例 "{name}" (#id 与名字不匹配或不存在)')
        return [{**o, "_pinned": True} for o in hits]
    hits = [o for o in objs if any(q in n for n in _names(o))]
    if not hits:
        avail = sorted({o["name"] for o in objs})
        raise SystemExit(f'找不到 "{name}". 场景里的物体: {avail}')
    return hits


def _excluded(o, anchors):
    """候选排除判据 (实例级):
    - 锚点实例本身永远排除;
    - 未锚定 (按名字选锚) 时同名兄弟也排除 — 防止"离桌子最近的"返回另一块桌子碎片;
    - #id 锚定时同名兄弟放行 (它们正是合法候选)."""
    if any(o["id"] == a["id"] for a in anchors):
        return True
    if anchors and anchors[0].get("_pinned"):
        return False
    return o["name"] in {a["name"] for a in anchors}


def group_char_size(insts):
    return float(np.mean([o["s"] for o in insts]))


def dedupe_by_name(ranked):
    """按名字去重, 保留每个名字最好的一条 (ranked 已按好->差排序)."""
    seen, out = set(), []
    for r in ranked:
        if r["name"] not in seen:
            seen.add(r["name"])
            out.append(r)
    return out


def make_flags(o, anchors, vol_thr):
    """支撑面/容器的几何证据 (给 agent 读, 不靠名字猜):
       contains_anchor: 有锚点中心落在该候选盒内 (候选可能是支撑面/容器)
       scene_scale:     体积 > 20x 全场中位数 (大结构嫌疑)"""
    return {
        "contains_anchor": bool(any(point_to_obb(a["centroid"], o) == 0.0 for a in anchors)),
        "scene_scale": bool(float(np.prod(o["h"] * 2)) > vol_thr),
    }


def _vol_threshold(objs):
    return SCENE_SCALE_MULT * float(np.median([np.prod(o["h"] * 2) for o in objs]))


def _near_score(dist, ref):
    """接近度: dist=0 -> 1, dist=ref -> 0.5, 越远越小. ref=参考尺度 (锚点特征尺寸等)."""
    ref = max(float(ref), 1e-6)
    return 1.0 / (1.0 + max(float(dist), 0.0) / ref)


def _minmax_score(vals, v, hi_best=True):
    """把 v 在候选集 [min,max] 里线性归一; hi_best=True 大端=1, False 小端=1. 全并列->1."""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return 1.0
    f = (float(v) - lo) / (hi - lo)
    return f if hi_best else 1.0 - f


# ─── 关系实现 ────────────────────────────────────────────────────────────────
def _rank_instances(objs, anchors, reverse=False):
    """OBB 表面距对逐实例排序 (不去重), touching 等下游在此之上精判."""
    out = []
    for o in objs:
        if _excluded(o, anchors):
            continue
        d = min(obb_distance(o, a) for a in anchors)
        out.append((d, o))
    out.sort(key=lambda x: x[0], reverse=reverse)
    return out


def rel_closest(objs, anchors, reverse=False):
    vol95 = _vol_threshold(objs)
    s_ref = group_char_size(anchors)
    ranked = [
        {"name": o["name"], "id": o["id"], "status": o["status"],
         "dist": round(d, 4), "flags": make_flags(o, anchors, vol95)}
        for d, o in _rank_instances(objs, anchors, reverse)
    ]
    dists = [r["dist"] for r in ranked]
    zero_tie = sum(1 for d in dists if d == 0.0)
    for r in ranked:
        # 绝对满足分: closest=接近度(贴着=1), farthest=远离度(=1-接近度)
        near = _near_score(r["dist"], s_ref)
        r["conf"] = round((1.0 - near) if reverse else near, 3)
        if r["dist"] == 0.0 and zero_tie > 1:
            r["flags"]["zero_dist_tie"] = zero_tie    # d=0 并列, 唯一性由上层 margin 判
    return ranked


def rel_touching(objs, anchors):
    """两段式: OBB 粗筛 top-k -> 点级精测, 阈值 = 0.1 x 两物体特征尺寸的较小者."""
    vol95 = _vol_threshold(objs)
    coarse = _rank_instances(objs, anchors)[:TOUCH_TOPK]
    out = []
    for _, o in coarse:
        d_pt = min(point_level_dist(o, a) for a in anchors)
        thr = TOUCH_SCALE_FRAC * min(o["s"], min(a["s"] for a in anchors))
        if d_pt <= thr:
            out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                        "dist": round(d_pt, 4),
                        "conf": round(abs(d_pt - thr) / (thr + 1e-9), 3),
                        "flags": make_flags(o, anchors, vol95)})
    return sorted(out, key=lambda r: r["dist"])


def _containment_ratio(target, anchors):
    """target 的采样点落在任一锚实例 OBB 内的比例."""
    if target["pts"] is None:
        # 降级: 中心判据
        return 1.0 if any(point_to_obb(target["centroid"], a) == 0.0 for a in anchors) else 0.0
    inside = np.zeros(len(target["pts"]), dtype=bool)
    for a in anchors:
        inside |= points_in_obb(target["pts"], a)
    return float(inside.mean())


def _axis_chord(o, c):
    """细长物体 o 的主轴直线在容器 c 的 OBB 内的弦长 (米). o 不细长或不相交返回 0.

    用途: 插入判定的几何外推 —— 勺在杯里时, 杯内那截拍不到没有点, 但可见段的主轴
    延长线必然穿过杯内部; 弦长就是"轴向还能容纳多少隐藏部分"的量.
    """
    hs = sorted(o["h"], reverse=True)
    if hs[0] < INSERT_AXIS_ASPECT * (hs[1] + 1e-9):
        return 0.0                                   # 不细长, 主轴无意义
    d = o["R"][int(np.argmax(o["h"]))]               # 主轴方向 (单位向量, 世界系)
    # 直线 (过 o 质心, 方向 d) 与 c 的 OBB 求交: 变到 c 局部系做 slab test
    q = c["R"] @ (o["centroid"] - c["c"])
    v = c["R"] @ d
    t0, t1 = -np.inf, np.inf
    for k in range(3):
        if abs(v[k]) < 1e-9:
            if abs(q[k]) > c["h"][k]:
                return 0.0                           # 平行且在该 slab 外
            continue
        a, b = (-c["h"][k] - q[k]) / v[k], (c["h"][k] - q[k]) / v[k]
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return 0.0
    return float(max(0.0, t1 - t0))


def _insert_conf(ratio, o, chord):
    """插入满足分: 可见段已在容器内的比例 + 延长线可容纳的隐藏比例 (弦长/物体全长), 封顶 1."""
    return float(min(1.0, ratio + chord / (2 * max(o["h"]) + 1e-9)))


def rel_inside(objs, anchors, band):
    """band='inside': 包含率>=0.7;  band='inserted': 0.3~0.7 (部分插入).

    细长物体补判: 包含率在 [0.1, 0.7) 且主轴延长线穿过锚点内部 -> 也算 inserted
    (重建盲区: 容器内部拍不到, 包含率被压低; 见 _axis_chord).
    """
    out = []
    for o in objs:
        if _excluded(o, anchors):
            continue
        r = _containment_ratio(o, anchors)
        if band == "inside" and r >= INSIDE_HI:
            conf = (r - INSIDE_HI) / (1 - INSIDE_HI)          # 0=擦线, 1=全在里面
        elif band == "inserted" and INSIDE_LO <= r < INSIDE_HI:
            conf = min(r - INSIDE_LO, INSIDE_HI - r) / ((INSIDE_HI - INSIDE_LO) / 2)
        elif band == "inserted" and INSERT_AXIS_LO <= r < INSIDE_LO:
            # 细长补判: 灰区包含率 + 主轴延长线穿过锚点内部 -> 部分插入 (盲区外推)
            chord = max(_axis_chord(o, a) for a in anchors)
            if chord <= 0:
                continue
            conf = _insert_conf(r, o, chord)
        else:
            continue
        out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                    "dist": round(r, 3), "conf": round(conf, 3), "flags": {}})
    return sorted(out, key=lambda x: -x["dist"])


def rel_contains(objs, anchors):
    """锚点的采样点大部分落在候选盒内 (inside 的反向). >=0.7 按原方案;
    细长锚点 (勺) 部分插入本来就露柄 + 容器内部是重建盲区, 包含率在 [0.1, 0.7) 时
    用锚点主轴延长线是否穿过候选内部补判 (用户方案: 包含率当门, 延长线只在灰区启用)."""
    out = []
    for o in objs:
        if _excluded(o, anchors):
            continue
        best = None                              # 对多锚实例取最有利的一个 (ratio, conf)
        for a in anchors:
            r = _containment_ratio(a, [o])       # 锚点 a 的点在候选 o 盒内的比例
            if r >= INSIDE_HI:
                conf = (r - INSIDE_HI) / (1 - INSIDE_HI)
            elif r >= INSERT_AXIS_LO:
                chord = _axis_chord(a, o)        # 细长锚点的轴延长线在候选内的弦长
                if chord <= 0:
                    continue
                conf = _insert_conf(r, a, chord)
            else:
                continue
            if best is None or conf > best[1]:
                best = (r, conf)
        if best is not None:
            out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                        "dist": round(best[0], 3), "conf": round(best[1], 3), "flags": {}})
    return sorted(out, key=lambda x: -x["dist"])


def rel_between(objs, anchors_a, anchors_b):
    """目标在两锚中心连线之间: 面感知垂距 (盒面点阵到线段) 小, 且中心投影在中段."""
    ca = np.mean([a["centroid"] for a in anchors_a], axis=0)
    cb = np.mean([b["centroid"] for b in anchors_b], axis=0)
    seg_len = float(np.linalg.norm(cb - ca))
    perp_lim = BETWEEN_PERP * seg_len
    out = []
    for o in objs:
        if _excluded(o, anchors_a) or _excluded(o, anchors_b):
            continue
        _, t = point_seg_dist(o["centroid"], ca, cb)   # 语义门: 中心投影在中段
        perp = seg_to_obb_dist(o, ca, cb)              # 面感知垂距 (治大而扁的 plate)
        if BETWEEN_T[0] <= t <= BETWEEN_T[1] and perp <= perp_lim:
            out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                        "dist": round(perp, 4), "t": round(t, 2),
                        "conf": round((perp_lim - perp) / (perp_lim + 1e-9), 3),
                        "flags": {}})
    return sorted(out, key=lambda r: r["dist"])


def rel_extreme_size(cands, largest=True):
    """同名候选按 OBB 体积取极值 (锚点名当过滤词, "all"=全场)."""
    ranked = sorted(
        ({"name": o["name"], "id": o["id"], "status": o["status"],
          "dist": round(float(np.prod(o["h"] * 2)), 6), "flags": {}} for o in cands),
        key=lambda r: r["dist"], reverse=largest,
    )
    vols = [r["dist"] for r in ranked]
    for r in ranked:                       # 绝对满足分: 体积在候选集里归一, largest 大端=1
        r["conf"] = round(_minmax_score(vols, r["dist"], hi_best=largest), 3)
    return ranked


# ─── 重力相关关系 ────────────────────────────────────────────────────────────
def heights(o, up):
    """物体沿 up 的 (底, 中心, 顶). 点级取 P10/P90 分位而非 min/max:
    实例分割沿邻居表面的渗漏点会把 min 拉低 (鸭子沿魔方侧面渗漏 ~0.14),
    分位高度才代表真实的落座面/顶面. 无点缓存时退化为 OBB 支撑函数."""
    if o["pts"] is not None:
        hs = o["pts"] @ up
        return (float(np.percentile(hs - o["pscale"], 10)),
                float(o["centroid"] @ up),
                float(np.percentile(hs + o["pscale"], 90)))
    span = float(np.abs(o["R"] @ up) @ o["h"])   # OBB 沿 up 的半跨度
    ch = float(o["c"] @ up)
    return ch - span, float(o["centroid"] @ up), ch + span


def point_spacing(o):
    """锚点采样点的中位近邻距 (稀疏度): 桌子 256 点摊 5 个单位跨度, 间距 ~0.3,
    局部表面估计的邻域半径和对齐容差都要为它留松量. 结果缓存在对象字典里."""
    if o["pts"] is None:
        return 0.0
    if "_spacing" not in o:
        D = np.linalg.norm(o["pts"][:, None, :] - o["pts"][None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        o["_spacing"] = float(np.median(D.min(axis=1)))
    return o["_spacing"]


def overlap_and_ref(o, a, up, top=True):
    """目标点足迹与锚点点足迹的水平重叠, 以及垫底(/盖顶)锚点的局部表面高度.

    质心邻域方案的两个失效被它同时治掉:
      - 锚点稀疏 (桌子): 配对半径含 2x 采样间距, 邻居够多, P90 稳;
      - 宽目标 (熊在盘子旁边被算 above): 按"目标各点有没有锚点垫底"计比例,
        旁边≠上方, 比例过不了门.
    返回 (overlap_frac, ref_h 或 None)."""
    if o["pts"] is None or a["pts"] is None:
        b, _, t = heights(a, up)                # 降级: 全局顶/底
        return 1.0, (t if top else b)
    r = 2 * point_spacing(a) + 0.25 * min(o["s"], a["s"])
    po = o["pts"] - np.outer(o["pts"] @ up, up)
    pa = a["pts"] - np.outer(a["pts"] @ up, up)
    D = np.linalg.norm(po[:, None, :] - pa[None, :, :], axis=-1)   # (No,Na) 水平距
    near = D <= r
    if not near.any():
        return 0.0, None
    # 双向重叠取大: 小压大看"目标点被锚点垫底的比例", 大压小看"锚点点被目标罩住的比例".
    # "旁边"两个方向都低 -> 仍过不了门 (熊在盘子旁不会被算 above); "大柜在小烤箱上"锚点侧接近 1.
    frac = max(float(near.any(axis=1).mean()), float(near.any(axis=0).mean()))
    sel = near.any(axis=0)                      # 处在目标足迹下的锚点点
    hs = a["pts"][sel] @ up
    ps = a["pscale"][sel]
    # 局部参照面取中位数: footprint 内多数点是真实垫底表面, P90 会取到
    # 少数凸起 (盘沿实测比盘心高 0.06-0.15, 害 cookie 过不了 on 的对齐门)
    ref = np.median(hs + ps) if top else np.median(hs - ps)
    return frac, float(ref)


def rel_above_below(objs, anchors, up, above=True):
    """above: 目标与锚点水平重叠(双向取大, 罩住或被罩住) >= 30%, 且目标底面高于锚点局部顶面
    (负间隙容忍 0.5 x 锚点尺寸); below 完全对称.
    满足分 conf = 重叠覆盖比例(罩得越全 = 越在正上/下方); dist = 垂直间隙."""
    out = []
    for o in objs:
        if _excluded(o, anchors):
            continue
        ob, _, ot = heights(o, up)
        best = None   # 对多锚实例取覆盖比例最高的那个 (vgap, frac)
        for a in anchors:
            frac, ref = overlap_and_ref(o, a, up, top=above)
            if frac < ABOVE_OVERLAP_MIN or ref is None:
                continue                                  # 水平不重叠 -> 在旁边, 不算正上/下方
            vgap = (ob - ref) if above else (ref - ot)
            # 负间隙(交叠)容忍度 = 锚点尺寸的 50%: 3DGS 盒/点边界歪斜, 很容易和 target 交叉出假负间隙,
            # 松量太小会把真正在上/下方的目标误杀. 再加两侧高斯尺度(静置接触时点带互相嵌入 ~2 个尺度).
            slack = ABOVE_VGAP_SLACK * a["s"]
            if o["pscale"] is not None and a["pscale"] is not None:
                slack += 1.5 * float(np.median(o["pscale"]) + np.median(a["pscale"]))
            if vgap < -slack:
                continue
            if best is None or frac > best[1]:            # 取覆盖比例最高的锚点
                best = (vgap, frac)
        if best is not None:
            vgap, frac = best
            out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                        "dist": round(max(vgap, 0.0), 4),
                        "conf": round(frac, 3),           # 满足分 = 门1 覆盖比例 (罩得越全越上方)
                        "flags": {}})
    return sorted(out, key=lambda r: -r["conf"])          # 覆盖最好的在前


def rel_on(objs, anchors, up):
    """搁在锚点上 = 点级接触 + 足迹重叠 + 目标底 ≈ 垫底锚点的局部顶面.
    对齐容差 = 0.3 x s_min + 0.5 x 锚点采样间距 (稀疏锚点留松量).
    dist = |目标底 - 局部顶面|, conf = 1 - 对齐误差/容差."""
    out = []
    for o in objs:
        if _excluded(o, anchors):
            continue
        ob, oc, _ = heights(o, up)
        # 容差的尺寸项封顶在 2x 全场中位特征尺寸: 巨型目标 (坐在椅子上探过桌沿的
        # 羊, s=1.4) 否则容差膨胀到 0.6, 把真实的 0.6 错位也放进来
        s_cap = 2 * float(np.median([x["s"] for x in objs]))
        best = None
        for a in anchors:
            s_min = min(o["s"], a["s"])
            if point_level_dist(o, a) > TOUCH_SCALE_FRAC * s_min:
                continue                                  # 没接触
            frac, lt = overlap_and_ref(o, a, up, top=True)
            if frac < ON_OVERLAP_MIN or lt is None:
                continue                                  # 脚下没有锚点表面
            tol = ON_ALIGN_TOL * min(s_min, s_cap) + 0.5 * point_spacing(a)
            if o["pscale"] is not None and a["pscale"] is not None:
                tol += float(np.median(o["pscale"]) + np.median(a["pscale"]))
            align = abs(ob - lt)
            # oc 门只挡"粘在面下": 阈值放到半个目标尺寸, 否则薄目标 (平躺的
            # 饼干, 中心只比表面高 1-2 个高斯尺度) 会被误杀
            if align > tol or oc <= lt - 0.5 * o["s"]:
                continue
            if best is None or align < best[0]:
                best = (align, tol)
        if best is not None:
            align, tol = best
            out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                        "dist": round(align, 4),
                        "conf": round(1.0 - align / (tol + 1e-9), 3),
                        "flags": {}})
    return sorted(out, key=lambda r: r["dist"])


def rel_higher_lower(objs, anchors, up, higher=True):
    """纯高度比较 (无水平约束): 目标中心高度 vs 锚点中心高度, dist = 高差 (米)."""
    a_ch = max(heights(a, up)[1] for a in anchors) if higher else \
           min(heights(a, up)[1] for a in anchors)
    out = []
    for o in objs:
        if _excluded(o, anchors):
            continue
        ch = heights(o, up)[1]
        diff = (ch - a_ch) if higher else (a_ch - ch)
        if diff > 0:
            out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                        "dist": round(diff, 4), "flags": {}})
    out = sorted(out, key=lambda r: -r["dist"])
    s_ref = group_char_size(anchors)
    for r in out:              # 绝对满足分: 高差越大越"明显更高", diff/(diff+s_ref)
        r["conf"] = round(1.0 - _near_score(r["dist"], s_ref), 3)
    return out


def rel_extreme_height(cands, up, highest=True):
    """同名候选里最高/最低 (锚点名当过滤词, "all"=全场). dist = 中心高度."""
    ranked = sorted(
        ({"name": o["name"], "id": o["id"], "status": o["status"],
          "dist": round(heights(o, up)[1], 4), "flags": {}} for o in cands),
        key=lambda r: r["dist"], reverse=highest,
    )
    vals = [r["dist"] for r in ranked]     # dist = 中心高度
    for r in ranked:                       # 绝对满足分: 高度在候选集里归一, highest 高端=1
        r["conf"] = round(_minmax_score(vals, r["dist"], hi_best=highest), 3)
    return ranked


# ─── 视角相关关系 (观察者 = 当前渲染相机 / --frame 指定相机) ─────────────────
def view_feats(objs, c2w):
    """所有实例中心在相机系下的 (nx, ny, z, ang):
       nx/ny = x/z, y/z 归一化屏幕坐标 (OpenCV: x右 y下 z前; nx 越小越靠屏幕左)
       z     = 视线深度
       ang   = 物体的角半径 (s/2 / z), 屏幕量的天然参考尺度."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    P = np.stack([o["centroid"] for o in objs])
    pc = (P - t) @ R                       # 世界 -> 相机 (R 列 = 相机轴)
    z = pc[:, 2]
    nx = pc[:, 0] / np.maximum(np.abs(z), 1e-9)
    ny = pc[:, 1] / np.maximum(np.abs(z), 1e-9)
    ang = np.array([o["s"] / 2 for o in objs]) / np.maximum(np.abs(z), 1e-9)
    return nx, ny, z, ang


def _view_prep(objs, anchors, c2w):
    """公共准备: 特征 + 可见性 (z>0, 排除相机身后) + 锚点聚合值."""
    nx, ny, z, ang = view_feats(objs, c2w)
    idx = {o["id"]: i for i, o in enumerate(objs)}
    vis = z > 1e-3
    a_is = [idx[a["id"]] for a in anchors]
    a_nx = float(np.mean(nx[a_is]))
    a_ny = float(np.mean(ny[a_is]))
    a_z = float(np.mean(z[a_is]))
    a_ang = float(np.mean(ang[a_is]))
    return nx, ny, z, ang, vis, idx, a_nx, a_ny, a_z, a_ang


def rel_left_right(objs, anchors, c2w, left=True):
    """锚点的屏幕左/右侧, 紧邻优先 (与 ZSVG3D 的最左优先不同 — 它的候选被
    查询词预过滤过, 我们的候选是全场, "兔子左边的东西"应取紧挨着的那个).

    入选: 屏幕横坐标在锚点左(/右)侧;
    排序: 与锚点的总张角 δ 升序 (紧邻优先);
    conf: cos(偏轴角) x (δ_1/δ_i) — 偏轴角 = 候选相对锚点的连线偏离水平轴
          的角度, 正左且最近 -> conf≈1, 斜上/斜下方或更远 -> 衰减.
    dist = δ (度), off_axis = 偏轴角 (度)."""
    nx, ny, z, ang, vis, idx, a_nx, a_ny, a_z, a_ang = _view_prep(objs, anchors, c2w)
    rows = []
    for o in objs:
        i = idx[o["id"]]
        if _excluded(o, anchors) or not vis[i]:
            continue
        if (nx[i] < a_nx) if left else (nx[i] > a_nx):
            dx = np.degrees(np.arctan(nx[i]) - np.arctan(a_nx))   # 横向张角差
            dy = np.degrees(np.arctan(ny[i]) - np.arctan(a_ny))   # 纵向张角差
            delta = float(np.hypot(dx, dy))                       # 总张角
            phi = float(np.degrees(np.arctan2(abs(dy), abs(dx)))) # 偏离水平轴
            rows.append((delta, phi, o))
    rows.sort(key=lambda r: r[0])                                 # 紧邻优先
    out = [{"name": o["name"], "id": o["id"], "status": o["status"],
            "dist": round(d, 1), "off_axis": round(p, 1), "flags": {}}
           for d, p, o in rows]
    # 绝对满足分: cos(偏轴角)=有多正左/右, × 接近度(以锚点角半径为"紧邻"尺度) —— 不再看别的候选
    ref = max(float(np.degrees(np.arctan(max(a_ang, 1e-6)))), 1.0)   # 锚点角半径(度), 至少 1°
    for r in out:
        r["conf"] = round(float(np.cos(np.radians(r["off_axis"])))
                          * _near_score(r["dist"], ref), 3)
    return out


def rel_front_behind(objs, anchors, c2w, front=True):
    """仿 ZSVG3D FRONT/BEHIND: 比视线深度, 贴着锚点的优先排序.
    front = 比锚点更靠近观察者. dist = 深度差, conf = 间隙 / 锚点尺寸."""
    nx, ny, z, ang, vis, idx, a_nx, a_ny, a_z, a_ang = _view_prep(objs, anchors, c2w)
    rows = []
    for o in objs:
        i = idx[o["id"]]
        if _excluded(o, anchors) or not vis[i]:
            continue
        if (z[i] < a_z) if front else (z[i] > a_z):
            rows.append((abs(float(z[i] - a_z)), o))
    rows.sort(key=lambda r: r[0])                     # 贴着锚点的优先 (仿 ZSVG3D)
    out = [{"name": o["name"], "id": o["id"], "status": o["status"],
            "dist": round(d, 4), "flags": {}} for d, o in rows]
    s_ref = group_char_size(anchors)
    for r in out:              # 绝对满足分: 贴着锚点深度=1 (仿 ZSVG3D 紧邻优先), 越远越小
        r["conf"] = round(_near_score(r["dist"], s_ref), 3)
    return out


def rel_view_extreme(cands, c2w, mode):
    """同名候选里屏幕最左/最右/最居中 (锚点名当过滤词, 仿 ZSVG3D LEFT_MOST/MID).

    绝对满足分: 屏幕量在候选集范围里归一 ——
      rightmost = 右边缘(nx+角半径)最靠右的 =1 (即离画面右缘最近);
      leftmost  = 左边缘(nx-角半径)最靠左的 =1;
      middle    = 距当前视野中心的角偏最小的 =1 (2026-08-13 用户拍板: 居中 =
                  在**画面**里居中, 横竖两轴一起量 sqrt(nx²+ny²), 不再是候选横排取中位).
    dist = 横向角度(度, 负=左); middle 行的 dist = 距画面中心的径向角偏(度). 供展示参考.
    """
    nx, ny, z, ang = view_feats(cands, c2w)
    rows = [(float(nx[i]), float(ny[i]), float(ang[i]), o)
            for i, o in enumerate(cands) if z[i] > 1e-3]
    if not rows:
        return []
    if mode == "rightmost":
        pos = [v + a for v, _n, a, _ in rows]         # 右边缘
    elif mode == "leftmost":
        pos = [v - a for v, _n, a, _ in rows]         # 左边缘
    else:                                             # middle: 距视野中轴的径向偏 (越小越居中)
        pos = [float(np.hypot(v, n)) for v, n, _a, _ in rows]
    order = sorted(range(len(rows)), key=lambda i: pos[i],
                   reverse=(mode == "rightmost"))     # 最极端在前
    out = []
    for i in order:
        v, _n, _a, o = rows[i]
        shown = pos[i] if mode == "middle" else v
        out.append({"name": o["name"], "id": o["id"], "status": o["status"],
                    "dist": round(float(np.degrees(np.arctan(shown))), 1),
                    # rightmost 大端=1; leftmost 小端(左边缘最负)=1; middle 偏差小端=1
                    "conf": round(_minmax_score(pos, pos[i],
                                                hi_best=(mode == "rightmost")), 3),
                    "flags": {}})
    return out


def load_view_from_frame(frame_name):
    """从 cameras.json 取指定帧的 C2W (CLI 用; viewer 直接用当前渲染相机)."""
    with open(CAMERAS_JSON) as f:
        cams = json.load(f)
    for c in cams:
        if c["img_name"] == frame_name:
            c2w = np.eye(4)
            c2w[:3, :3] = np.array(c["rotation"], dtype=np.float64)
            c2w[:3, 3] = np.array(c["position"], dtype=np.float64)
            return c2w
    raise SystemExit(f'cameras.json 里没有 "{frame_name}" (例: frame_00002)')


# ─── 注册表 (同义词 -> 实现) ─────────────────────────────────────────────────
SYNONYMS = {
    "closest": ["closest", "nearest", "near", "next to", "beside", "by"],
    "farthest": ["farthest", "furthest"],
    "touching": ["touching", "adjacent", "against", "contacting"],
    "inside": ["inside", "in", "within"],
    "inserted": ["inserted", "stuck in", "partially inside"],
    "contains": ["contains", "containing", "holding"],
    "between": ["between"],
    "largest": ["largest", "biggest"],
    "smallest": ["smallest", "tiniest"],
    "distance": ["distance", "dist"],
    # 重力相关
    "above": ["above", "over"],
    "below": ["below", "under", "beneath", "underneath"],
    "on": ["on", "atop", "on top of", "resting on"],
    "higher": ["higher"],
    "lower": ["lower"],
    "highest": ["highest", "topmost"],
    "lowest": ["lowest", "bottommost"],
    # 视角相关
    "left": ["left", "left of", "to the left of"],
    "right": ["right", "right of", "to the right of"],
    "front": ["front", "in front of"],
    "behind": ["behind", "at the back of"],
    "leftmost": ["leftmost"],
    "rightmost": ["rightmost"],
    "middle": ["middle", "mid"],
}
CANON = {syn: k for k, syns in SYNONYMS.items() for syn in syns}
GRAVITY_RELS = {"above", "below", "on", "higher", "lower", "highest", "lowest"}
VIEW_RELS = {"left", "right", "front", "behind", "leftmost", "rightmost", "middle"}
VIEW_FILTER_RELS = {"leftmost", "rightmost", "middle"}   # 锚点名只是过滤词


def relate(relation, anchor_name, anchor2_name=None, topk=5, objs=None, view=None,
           granularity="instance"):
    """主入口 (agent 工具直接包它).

    返回 [{name, id, dist, conf, status, flags}, ...] 按好->差排序:
      dist  closest/touching=米; inside/inserted/contains=包含率; largest=体积;
            left/right/leftmost...=横向角度(度); front/behind=深度差
      conf  绝对满足分 [0,1]: 该候选"有多满足这条关系" (每候选独立算, 不看别人的间隙).
            如 closest=接近度(贴着=1), rightmost=右边缘归一(最右=1), on=1-对齐误差/容差.
            "是不是唯一满足"由上层的 margin(top1-top2)判断, 不再压进 conf.
      flags contains_anchor / scene_scale -> 支撑面/容器的几何证据 (跳过判据读它, 别读名字)
            zero_dist_tie=n -> n 个候选都贴着锚点, 排序不可信
      view  视角相关关系必填: 观察者相机的 C2W (4,4). viewer 传当前渲染相机,
            CLI 用 --frame 从 cameras.json 取.
      granularity  "instance"(默认) 每个实例一条, 答案是具体实例 (grounding 语义);
                   "name" 同名去重取每名最优 (碎片并集/集合名场景, 如 three cookies).
      anchor_name  支持 "#7" / "knife#7" 锚定具体实例 (锚定后同名兄弟成为合法候选).
    """
    rel = CANON.get(relation.strip().lower())
    if rel is None:
        raise SystemExit(f'不认识的关系 "{relation}". 支持: {sorted(SYNONYMS)}')
    objs = objs if objs is not None else load_objects()
    anchors = find_by_name(objs, anchor_name)

    if rel == "closest":
        out = rel_closest(objs, anchors)
    elif rel == "farthest":
        out = rel_closest(objs, anchors, reverse=True)
    elif rel == "touching":
        out = rel_touching(objs, anchors)
    elif rel in ("inside", "inserted"):
        out = rel_inside(objs, anchors, band=rel)
    elif rel == "contains":
        out = rel_contains(objs, anchors)
    elif rel == "between":
        if not anchor2_name:
            raise SystemExit("between 需要两个锚点: pre_relation.py between <名字A> <名字B>")
        out = rel_between(objs, anchors, find_by_name(objs, anchor2_name))
    elif rel in ("largest", "smallest"):
        out = rel_extreme_size(anchors, largest=(rel == "largest"))
    elif rel == "distance":
        if not anchor2_name:
            raise SystemExit("distance 需要两个名字: pre_relation.py distance <名字A> <名字B>")
        others = find_by_name(objs, anchor2_name)
        d = min(point_level_dist(a, b) for a in anchors for b in others)
        out = [{"name": f"{anchor_name} <-> {anchor2_name}", "id": -1, "status": "-",
                "dist": round(d, 4), "conf": 1.0, "flags": {}}]
    elif rel in GRAVITY_RELS:
        up, gmeta = get_up(objs, verbose=not os.path.exists(GRAVITY_JSON))
        if rel in ("above", "below"):
            out = rel_above_below(objs, anchors, up, above=(rel == "above"))
        elif rel == "on":
            out = rel_on(objs, anchors, up)
        elif rel in ("higher", "lower"):
            out = rel_higher_lower(objs, anchors, up, higher=(rel == "higher"))
        else:  # highest / lowest (锚点名当过滤词)
            out = rel_extreme_height(anchors, up, highest=(rel == "highest"))
        # 重力估计的可靠度进 flags (不再乘进 conf): conf 是"绝对满足分", up 不确定属于"可靠度",
        # 均匀打折会压掉排名信号 (highest 的最高者也被压到 0.4). 上层要用它降信任就读这个 flag.
        for r in out:
            r["flags"]["gravity"] = gmeta["method"]
            r["flags"]["gravity_conf"] = round(float(gmeta["confidence"]), 3)
    elif rel in VIEW_RELS:
        if view is None:
            raise SystemExit(
                "视角相关关系需要观察者相机: viewer 里直接查询 (用当前渲染视角), "
                "CLI 加 --frame frame_00002 (用该帧相机)")
        if rel in ("left", "right"):
            out = rel_left_right(objs, anchors, view, left=(rel == "left"))
        elif rel in ("front", "behind"):
            out = rel_front_behind(objs, anchors, view, front=(rel == "front"))
        else:  # leftmost / rightmost / middle (锚点名当过滤词)
            out = rel_view_extreme(anchors, view, mode=rel)
    if granularity == "name":       # 名字级视图: 同名去重, 每名保留最优一条
        out = dedupe_by_name(out)
    return out[:topk]


# ─── 可视化 (仿 reviewer.py) ─────────────────────────────────────────────────
# 输入 锚点 + 相对关系, 分割出可信度最高的物体: 点云模式红色高亮答案(锚点蓝色),
# RGB 模式只渲染答案+锚点的高斯. 需要 normlift 环境 (torch/viser/nerfview/gsplat).
PORT = 8080
CKPT_PATH = scene_paths.ckpt_path(SCENE, gs2d=os.environ.get("GS2D", "0") == "1" or
                                  scene_paths.is_bench(SCENE))
LABELS_NPY = os.path.join(LIFT_DIR, "labels.npy")
NO_ANCHOR_RELS = {"largest", "smallest", "highest", "lowest"}   # 锚点名只是过滤词
ANSWER_RGB = (255, 30, 30)
ANCHOR_RGB = (40, 120, 255)


def _is_support_surface(o, up, vol_thr):
    """滤桌面的判据: 大 + 扁 + 水平 三个几何证据, 不读名字.

    contains_anchor 不行 (搁在桌上的杯子质心在桌盒上方, 不在其内);
    只看大小也不行 (巨型毛绒玩具会被误滤). 支撑面的签名是水平的大扁板:
      大:   体积 > 20x 全场中位
      扁:   最小轴跨度 / 最大轴跨度 < 0.35
      水平: 最小轴方向 (板的法向) 与 up 夹角 < 30°"""
    if float(np.prod(o["h"] * 2)) <= vol_thr:
        return False
    i = int(np.argmin(o["h"]))
    flat = o["h"][i] / (o["h"].max() + 1e-9) < 0.35
    horizontal = abs(float(o["R"][i] @ up)) > np.cos(np.deg2rad(30))
    return flat and horizontal


def run_viewer():
    import time
    import torch
    import viser
    import nerfview
    from gsplat import rasterization

    device = "cuda"
    sh0_c = 0.28209479177387814

    objs = load_objects()
    vol_thr = _vol_threshold(objs)
    up, _gmeta = get_up(objs, verbose=False)   # 支撑面判据要用 up (大+扁+水平)
    ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    sp = ck["splats"]
    splats = dict(
        means=sp["means"].to(device).float(),
        quats=sp["quats"].to(device).float(),
        scales=torch.exp(sp["scales"].to(device).float()),
        opacities=torch.sigmoid(sp["opacities"].to(device).float()),
        sh=torch.cat([sp["sh0"], sp["shN"]], dim=1).to(device).float(),
    )
    labels_t = torch.from_numpy(np.load(LABELS_NPY).astype(np.int64)).to(device)
    means_np = splats["means"].cpu().numpy()
    base_rgb = ((0.5 + sh0_c * splats["sh"][:, 0, :].cpu().numpy()).clip(0, 1)
                * 255).astype(np.uint8)
    print(f"[viewer] {means_np.shape[0]} 个高斯, {len(objs)} 个实例 <- {SCENE}")

    state = {"mode": "PointCloud", "answer_idx": None, "anchor_idx": None,
             "c2w": None}   # c2w = 最近一次渲染的相机, 视角相关关系的观察者
    server = viser.ViserServer(port=PORT, verbose=False)
    base_cloud = server.scene.add_point_cloud(
        "/cloud/base", points=means_np[::2], colors=base_rgb[::2], point_size=0.01)
    marks = {}   # name -> 高亮点云句柄

    def ids_to_idx(ids):
        if not ids:
            return None
        m = torch.isin(labels_t, torch.tensor(ids, device=device))
        return m.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def render_fn(camera_state, img_wh):
        W, H = img_wh
        state["c2w"] = np.array(camera_state.c2w, dtype=np.float64)  # 当前视角
        if state["mode"] == "PointCloud":
            return np.full((H, W, 3), 0.08, dtype=np.float32)
        sels = [s for s in (state["answer_idx"], state["anchor_idx"]) if s is not None]
        sel = torch.cat(sels).unique() if sels else None   # RGB: 只渲染答案+锚点
        if sel is None:
            means, quats = splats["means"], splats["quats"]
            scales, opac, sh = splats["scales"], splats["opacities"], splats["sh"]
        else:
            means, quats = splats["means"][sel], splats["quats"][sel]
            scales, opac, sh = (splats["scales"][sel], splats["opacities"][sel],
                                splats["sh"][sel])
        if means.shape[0] == 0:
            return np.zeros((H, W, 3), dtype=np.float32)
        colors, _, _ = rasterization(
            means, quats, scales, opac, sh,
            torch.from_numpy(np.linalg.inv(camera_state.c2w)).float().to(device)[None],
            torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)[None],
            W, H, sh_degree=3, render_mode="RGB")
        return colors[0].clamp(0, 1).cpu().numpy()

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")

    with server.gui.add_folder("关系查询"):
        mode_dd = server.gui.add_dropdown("模式", ("PointCloud", "RGB"),
                                          initial_value="PointCloud")
        rel_dd = server.gui.add_dropdown("关系", tuple(sorted(SYNONYMS)),
                                         initial_value="closest")
        anchor_txt = server.gui.add_text("锚点 anchor (可 knife#7 锚定)", initial_value="")
        anchor2_txt = server.gui.add_text("第二锚点 (between/distance)", initial_value="")
        no_table = server.gui.add_checkbox("滤除桌面 (大结构)", initial_value=False)
        by_name = server.gui.add_checkbox("按名字聚合 (碎片并集)", initial_value=False)
        query_btn = server.gui.add_button("查询并分割")
        reset_btn = server.gui.add_button("重置 (显示全部)")
        psize = server.gui.add_slider("点大小", min=0.001, max=0.05, step=0.001,
                                      initial_value=0.01)
        info = server.gui.add_markdown("选关系、填锚点名字, 点查询")

    def set_mark(name, idx, color, size_mult):
        if name in marks:
            marks[name].remove()
            del marks[name]
        if idx is not None and idx.numel() > 0:
            marks[name] = server.scene.add_point_cloud(
                f"/cloud/{name}", points=means_np[idx.cpu().numpy()],
                colors=np.tile(np.array(color, np.uint8), (int(idx.numel()), 1)),
                point_size=psize.value * size_mult)
            marks[name].visible = state["mode"] == "PointCloud"

    def apply_mode():
        pc = state["mode"] == "PointCloud"
        base_cloud.visible = pc
        for h in marks.values():
            h.visible = pc
        viewer.rerender(None)

    @query_btn.on_click
    def _(_) -> None:
        try:
            out = relate(rel_dd.value, anchor_txt.value,
                         anchor2_txt.value.strip() or None, topk=10, objs=objs,
                         view=state["c2w"],    # 视角相关关系用当前渲染视角
                         granularity="name" if by_name.value else "instance")
        except SystemExit as e:            # find_by_name 找不到名字等
            info.content = f"⚠ {e}"
            return
        dropped = []
        if no_table.value:
            # 滤除桌面: 大+扁+水平 的几何签名 (见 _is_support_surface), 不读名字
            kept = []
            for r in out:
                o = next((x for x in objs if x["id"] == r["id"]), None)
                surf = o is not None and _is_support_surface(o, up, vol_thr)
                (dropped if surf else kept).append(r)
            out = kept
        if not out:
            note = f" (滤掉大结构: {[d['name'] for d in dropped]})" if dropped else ""
            info.content = f"关系不成立, 没有物体满足{note} — 可降级换关系/问 VLM"
            return

        ans = out[0]                       # 可信度最高的物体
        if by_name.value:                  # 名字级: 同名并集 (碎片一起亮)
            ans_ids = [o["id"] for o in objs if o["name"] == ans["name"]]
        else:                              # 实例级: 只亮答案那一个实例
            ans_ids = [ans["id"]] if ans["id"] != -1 else []
        state["answer_idx"] = ids_to_idx(ans_ids)
        no_anchor = NO_ANCHOR_RELS | VIEW_FILTER_RELS   # 锚点名只是过滤词的关系
        if rel_dd.value not in no_anchor and ans["id"] != -1:
            try:
                state["anchor_idx"] = ids_to_idx(
                    [a["id"] for a in find_by_name(objs, anchor_txt.value)])
            except SystemExit:
                state["anchor_idx"] = None
        else:
            state["anchor_idx"] = None
        set_mark("answer", state["answer_idx"], ANSWER_RGB, 1.6)
        set_mark("anchor", state["anchor_idx"], ANCHOR_RGB, 1.2)

        ans_label = ans["name"] if by_name.value or ans["id"] == -1 \
            else f'{ans["name"]}#{ans["id"]}'
        lines = [f"**答案: {ans_label}**  (conf={ans.get('conf', '-')}, "
                 f"值={ans['dist']}, 红色; 锚点蓝色)"]
        lines += [f"- {r['name']} #{r['id']} 值={r['dist']} conf={r.get('conf', '-')}"
                  for r in out[:5]]
        if dropped:
            lines.append(f"已滤除大结构: {', '.join(d['name'] for d in dropped)}")
        info.content = "\n".join(lines)
        viewer.rerender(None)

    @reset_btn.on_click
    def _(_) -> None:
        state["answer_idx"] = None
        state["anchor_idx"] = None
        for h in list(marks.values()):
            h.remove()
        marks.clear()
        info.content = "已重置"
        viewer.rerender(None)

    @mode_dd.on_update
    def _(_) -> None:
        state["mode"] = mode_dd.value
        apply_mode()

    @psize.on_update
    def _(_) -> None:
        base_cloud.point_size = psize.value
        for name, mult in (("answer", 1.6), ("anchor", 1.2)):
            if name in marks:
                marks[name].point_size = psize.value * mult

    apply_mode()
    print(f"\n✅ 打开浏览器访问: http://localhost:{PORT}\n")
    while True:
        time.sleep(1.0)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="视角无关空间关系查询: <关系> <锚点名字> [第二锚点]")
    ap.add_argument("relation", nargs="?", help=f"关系词: {sorted(SYNONYMS)}")
    ap.add_argument("anchor", nargs="?", help='锚点物体名字 (子串匹配), largest/smallest 可用 "all"')
    ap.add_argument("anchor2", nargs="?", default=None, help="between/distance 的第二个名字")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--list", action="store_true", help="列出场景物体")
    ap.add_argument("--gravity", action="store_true", help="(重新)估计重力方向并打印诊断")
    ap.add_argument("--viewer", action="store_true",
                    help="启动可视化 (viser 端口 8080): 输入锚点+关系, 分割可信度最高的物体")
    ap.add_argument("--frame", default=None,
                    help="视角相关关系的观察者相机, 取 cameras.json 里的帧名 (如 frame_00002)")
    ap.add_argument("--by-name", action="store_true",
                    help="名字级聚合视图 (同名去重取每名最优); 默认实例级, 每个实例一条")
    args = ap.parse_args()

    if args.viewer:
        run_viewer()
        return

    objs = load_objects()
    print(f"[scene] {SCENE}: {len(objs)} 个实例, 场景对角线 {scene_diag(objs):.2f}\n")

    if args.gravity:
        get_up(objs, force=True, verbose=True)
        if not (args.relation and args.anchor):
            return

    if args.list or not (args.relation and args.anchor):
        byname = {}
        for o in objs:
            byname.setdefault(o["name"], []).append(o)
        for name, lst in sorted(byname.items()):
            tags = ", ".join(f"#{o['id']}({o['status'][:4]})" for o in lst)
            print(f"  {name:<18} x{len(lst)}  [{tags}]")
        if not args.list:
            print("\n用法: python pre_relation.py closest 'stuffed bear'  (--help 看全部关系)")
        return

    view = load_view_from_frame(args.frame) if args.frame else None
    out = relate(args.relation, args.anchor, args.anchor2, topk=args.topk,
                 objs=objs, view=view,
                 granularity="name" if args.by_name else "instance")
    if not out:
        print(f'没有物体满足 "{args.relation} {args.anchor}'
              f'{" " + args.anchor2 if args.anchor2 else ""}" — 关系不成立 (上层可降级: 换关系/问VLM)')
        return

    rel = CANON[args.relation.lower()]
    unit = {"inside": "包含率", "inserted": "包含率", "contains": "包含率",
            "largest": "体积m^3", "smallest": "体积m^3",
            "above": "垂直间隙m", "below": "垂直间隙m", "on": "贴合误差m",
            "higher": "高差m", "lower": "高差m",
            "highest": "中心高度", "lowest": "中心高度",
            "left": "横向角度°", "right": "横向角度°", "leftmost": "横向角度°",
            "rightmost": "横向角度°", "middle": "横向角度°",
            "front": "深度差", "behind": "深度差"}.get(rel, "距离m")
    ans_label = out[0]["name"] if args.by_name or out[0]["id"] == -1 \
        else f'{out[0]["name"]}#{out[0]["id"]}'
    print(f'答案: {ans_label}   (conf={out[0].get("conf", "-")})\n')
    print(f"  {'名字':<20}{'实例':>5}  {unit:>9}  {'conf':>6}  状态/标记")
    for r in out:
        marks = []
        f = r.get("flags", {})
        if f.get("scene_scale"):
            marks.append("大结构")
        if f.get("contains_anchor"):
            marks.append("含锚点")
        if f.get("zero_dist_tie"):
            marks.append(f"{f['zero_dist_tie']}个并列贴合")
        if f.get("gravity") == "cam_only":
            marks.append("重力仅相机先验")
        if "t" in r:
            marks.append(f"t={r['t']}")
        if "off_axis" in r:
            marks.append(f"偏轴{r['off_axis']}°")
        mark = (" [" + ", ".join(marks) + "]") if marks else ""
        print(f"  {r['name']:<20}#{r['id']:>4}  {r['dist']:>9}  {r.get('conf', 0):>6}  "
              f"{r['status']}{mark}")


if __name__ == "__main__":
    main()
