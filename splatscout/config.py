"""全部常数与路径. 数值出处见 施工方案_v2.md / 施工方案_v3.md 对应小节."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("SPLATSCOUT_DATA") or os.path.join(REPO_ROOT, "data")   # scenes/ ckpts_2dgs/ lift/ cache/ eval/ (see scene_paths.py)
PREP_ROOT = os.path.join(DATA_ROOT, "cache", "agent_prep")       # preprocessing package per scene (scene_prep.py)
MEM_ROOT = os.path.join(DATA_ROOT, "cache", "agent_mem")         # scene-graph memory (disabled by default)
RUNS_ROOT = os.environ.get("SPLATSCOUT_RUNS") or os.path.join(REPO_ROOT, "runs")   # trajectories and answers
QUERIES_DIR = os.path.join(REPO_ROOT, "benchmark", "data", "queries")               # PartRef-ScanNet++ queries
PARSE_ROOT = REPO_ROOT                                             # old name, kept for compatibility

DEVICE = "cuda"

# ── 模型 (§2) ──
MODEL = os.environ.get("AGENT_MODEL") or None   # None -> llm_config.LLM_MODEL; override with the AGENT_MODEL environment variable
LLM_BACKEND = os.environ.get("LLM_BACKEND", "responses")   # responses = OpenAI Responses 接口 (工具+思考); chat = chat/completions (本地 Qwen 服务或任何兼容端点)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")             # chat backend URL, e.g. http://127.0.0.1:8008/v1; default = OPENAI_BASE_URL (llm_config.py)
LLM_KEY = os.environ.get("LLM_KEY")                       # chat 后端的 key; 本地服务随便填
CHAT_MAX_TOKENS = 700
CHAT_MAX_TOKENS_THINKING = 16384  # 思考版的 max_tokens 含思考正文 (一步实测思考 3k token), 给足, 否则工具调用还没吐出来就被截断
LLM_STREAM = os.environ.get("LLM_STREAM", "0") == "1"   # chat 后端走流式 (思考版自动开)
LLM_SAMPLING = os.environ.get("LLM_SAMPLING", "greedy")   # greedy = 温度 0 (主跑次); card = Qwen3.5 模型卡不思考推荐 temperature 0.7 / top_p 0.8 / presence 1.5, seed 0
LLM_THINKING = os.environ.get("LLM_THINKING", "")       # on|off: 混合模型 (qwen3.5 等) 的思考开关, 传 enable_thinking; 空 = 按模型名判断
REASONING_EFFORT = "medium"     # Responses 接口: 工具 + 思考可并用 (chat/completions 不行, 已实测)
REASONING_SUMMARY = "auto"      # 向端点要思考摘要 (可视化用); 不支持时客户端自动去掉
IMAGE_DETAIL = "low"            # 全程 low: 整帧看布局, 局部一律裁剪后再发
REQUEST_TIMEOUT = 300
LLM_MAX_RETRIES = 5             # 一步最多再试几次 (429 每次多等 20 s, 上限 90 s; 5xx 短等); 09-14 之前是 2

# ── 外壳预算 (§3) ──
ABLATION = ""                   # the ablation variants of the paper (prompt / tool-schema swaps) are not part of this release
# 反馈消融 (09-25): "5" 导航器不返图 (开题 explore 除外, 无放大); "6" 分割器只回组号 (无 BOXED / 叠图 / 抠图 / 重叠); "7" 提升器只回实例号 (无尺寸 / 验收图); "8" 无 relate (尺子)
MAX_STEPS = 16
KEEP_IMAGE_GROUPS = 0           # v3: 0 = 一题之内图全留 (每张低清图约 220 token); >0 沿用 v2 "只留最近几组"
IMAGE_HARD_CAP = 40             # 单次调用图数硬上限, 超过才略去最早的组 (替换文字用模型自己写的 seen)
TOKEN_BUDGET = 150_000          # 单题累计输入 token 上限
NO_PROGRESS_STEPS = 3
MAX_IMAGES_PER_TOOL = 4
THUMB_LONG = 768                # 发给模型的图长边
CROP_MIN_SIDE = 384             # 裁剪最小边 (像素, 原图坐标)
CROP_PAD = 0.3                  # 裁剪外扩比例

# ── 预处理 (§5) ──
VIS_DEPTH_TOL = 0.05            # |z - D| < tol * z 判未被遮挡
VIS_MIN_RADIUS_PX = 3.0         # 投影半径 >= 3 像素才算"看见" (远处几像素的东西不算看过)
VIS_MIN_ALPHA = 0.5             # 该像素渲染 alpha 需 >= 此值 (有表面)
PREP_FRAME_CHUNK = 32           # 帧间交并比分块

# ── explore / nearby (§4.1, §4.2) ──
EXPLORE_N = 4
EXPLORE_MAX = 4                 # = MAX_IMAGES_PER_TOOL: 说了 6 帧只发 4 张图, 模型会看不见后两张 (首次 demo)
NEARBY_JACCARD = (0.2, 0.7)
NEARBY_MIN_VISIBLE = 0.3        # 看某个东西附近: 帧里可见成员占比下限
NEARBY_AREA_FLOOR = 0.25        # 方向分散挑帧时, 面积不低于最大者的这个比例

# ── segment (§4.3) ──
SEG_MAX_FRAMES = 6
SEG_MIN_SCORE = 0.5
GAP_SPLIT = 2                   # 帧号间隔 >= 2 处切段 (沿 merge3d.GAP_SPLIT)
WITHIN_MIN_INSIDE = 0.8         # 候选掩码 >= 80% 像素在父掩码内才保留
WITHIN_DILATE_PX = 5            # 渲染得到的父掩码外扩像素
SAM3_VERSION = "sam3"

# ── 框提示 (segment / nearby(frame=, box=)) ──
BOX_GRID = 1000                 # 模型画框的坐标网格: [x0,y0,x1,y1] 按所见图的宽高 0-1000; <=1 的小数也收, 当比例
BOX_MATCH_IOU = 0.25            # SAM3 返回的掩码与模型框的 IoU 达到此值才算"切到了框里的东西"
SAME_HINT_IOM = 0.3             # segment 返回里提示"与此前哪组三维重叠" (粗成员 交/较小者) 的下限; 是否同一个东西由模型判断

# ── lift (§4.4; 投票规则沿 merge3d / back_project) ──
CLAIM_MIN_W1 = 0.3              # 粗认领绝对票下限
MERGE_IOM = 0.5                 # 并组线: 交 / 较小者
MIN_CONSENSUS_VIEWS = 2         # 帧数 >= 2 时, 中心须落在 >= 2 帧的掩码内
MIN_INST_GAUSSIANS = 10         # 少于此数的提升结果判为碎块, 不入库. 原 200 (沿全帧流水线) 会误杀小件: 杯子真值 351 个,
                                # 多数票+两帧共识后只剩 131-187, 第七次 demo 四次 lift 全被拒 (2026-09-05 降到 10)
CH_MAX = 16                     # 单次光栅化最大通道数 (2DGS 反传核共享内存限制)
THOROUGH_ROUNDS = 5
THOROUGH_FRAMES_PER_ROUND = 6   # 抽屉这类小件 4 张就够, 大件多几张; 8 偏多 (第五次 demo 一次精提 40 s)
THOROUGH_MIN_VISIBLE = 0.3
THOROUGH_STOP_DELTA = 0.05      # 每轮成员涨幅低于此即停; 3% 太紧, 边缘碎屑每轮缓涨 5% 会把 5 轮跑满
THOROUGH_BOX_PAD = 0.1
FPS_POINTS = 256
DEDUP_MUTUAL_FIT = 0.7          # 入库去重: 与库内实例 交 / 较大者 >= 此值 -> 同一个东西 (别名/升级). 沿 merge3d 的互吻合线;
                                # 不能用 交/较小者: 部件对整体永远接近 1, 第五次 demo 抽屉顶替了柜子 #2

# ── relate (§4.5) ──
RELATE_MAX_ROWS = 30
PREDICATES = ["closest", "farthest", "touching", "inside", "inserted", "contains", "between",
              "largest", "smallest", "above", "below", "on", "higher", "lower", "highest",
              "lowest", "left", "right", "front", "behind", "leftmost", "rightmost", "middle"]
VIEW_PREDICATES = ["left", "right", "front", "behind", "leftmost", "rightmost", "middle"]

# ── 场景图记忆 (v3 §2, §5, §6) ──
MEMORY = False                  # 记忆与缓存总开关 (用户 09-12 停用). off: 每题空图且不落盘、开题只给句子、无 recall、不算 CLIP 向量、记忆 agent 不起;
                                # run_bench --memory on 打开, 下面 MODES / 记忆 agent 的开关才生效
MODES = ("session", "cold", "prewarm")   # session: 一个场景一场会话, 题间不清; cold: 每题换空图 (对照); prewarm: 开场灌旧管线全场库
ORDERS = ("given", "reverse", "shuffle")
OPEN_IMAGES = 2                 # 开题带几张描边帧 (命中节点所在), 0 关
OPEN_MAX_HITS = 6               # 开题消息里最多列几个命中节点
GRAPH_MAX_EDGES_PER_NODE = 4
KEYFRAMES = 6                   # 没命中时给的总览: 按覆盖贪心挑的固定关键帧数
NEAR_M = 1.0                    # 命中节点若提升过, 带上它一米以内提升过的东西
RECALL_MAX_THUMBS = 6
# ── 查询 -> 节点 的匹配 (v3 改: 名字对上 或 CLIP 图文相似度过线; 相似块不参与) ──
CLIP_ERODE_PX = 3               # 给 CLIP 的抠图: 掩码先腐蚀几个像素, 去边缘背景
CLIP_FEAT_FRAMES = 3            # 每组取掩码最大的几帧各算一个向量取平均
CLIP_MATCH_T = 0.6              # CLIP 相关度 (对"object/things/stuff/texture"四个空泛词标定后的 0-1 分, agent.clip_channel.calibrate_sim)
                                # 过线且比第二名高出 CLIP_MARGIN 才算按长相命中. 0908 抠图上标定: 同名中位 0.55, 异名 p90 0.51,
                                # 记忆里没有的词对最像的节点也常到 0.6-0.8 —— 所以 CLIP 只当次要通道, 名字和别名对上的优先
CLIP_MARGIN = 0.1
QUERY_PARSER = "dashscope"      # 句子 -> 物体短语: dashscope (小模型, 环境变量 DASHSCOPE_API_KEY) / main (主模型无图) / none (片段法)
QUERY_PARSER_MODEL = os.environ.get("QUERY_PARSER_MODEL", "qwen-turbo")
QUERY_PARSER_TIMEOUT = 20
# ── 记忆 agent (v4): 常驻的小 VLM, 题中跟看整理图, 换题时交假设页; 关掉就退到上面的程序检索 ──
MEMORY_AGENT = True
MEMORY_AGENT_MODEL = os.environ.get("MEMORY_AGENT_MODEL", "gpt-5.6")   # gpt* 走主 agent 的 key / 网址; 其他名字走 DashScope (qwen3-vl-plus 试过, 每题 5.4 s)
MEMORY_AGENT_REASONING = os.environ.get("MEMORY_AGENT_REASONING", "low")   # gpt 后端的 reasoning_effort: none / low / medium; DashScope 后端不用
MEMORY_AGENT_REASONING_HEADROOM = 3000  # gpt 后端思考 token 也算在输出上限里, 开思考时在 MAX_TOKENS 之上再留这么多
MEMORY_AGENT_HTTP_TIMEOUT = 90
MEMORY_AGENT_MAX_TOKENS = 1800          # 交接页 JSON 可能长 (每条匹配带核对); 700 会截断
MEMORY_AGENT_MAX_HANDOFF_CALLS = 1      # 换题交接: 1 = 只读图的文字, 一次调用交子图 (用户 09-10 定); >1 才允许它要缩略图 / 帧 / 距离
MEMORY_AGENT_EVENT_TIMEOUT = 8.0        # 题中每个事件的处理时限 (秒); 排队超过 3 倍就丢, 下次碰到再补
MEMORY_AGENT_DRAIN_TIMEOUT = 20.0       # 换题前最多等后台多久
MEMORY_AGENT_ROUGH_LIFT = True          # 确认过没提升的相似块在显卡空闲时粗提
THUMB_SIDE = 256                # 拼图每格边长
SNAPSHOT_MIN_VISIBLE = NEARBY_MIN_VISIBLE


def prep_dir(scene):
    return os.path.join(PREP_ROOT, scene)


def mem_dir(scene, tag=None):
    return os.path.join(MEM_ROOT, scene, tag) if tag else os.path.join(MEM_ROOT, scene)


def runs_dir(scene, qid):
    return os.path.join(RUNS_ROOT, scene, qid)
