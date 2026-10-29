"""Project configuration for the SAM3-only ModuSeg pipeline."""
from __future__ import annotations
import os
import torch

# 删除的实验参数不得静默失效；旧命令请改为本版单库入口。
_removed_overrides = [key for key in os.environ if key.startswith(
    ('OVERRIDE_REGION_POOLING', 'OVERRIDE_BUILD_MASK_EROSION'))]
if _removed_overrides:
    raise ValueError(f'Removed experiment options: {_removed_overrides}; use the SMFA-only configuration')

from configs.config_helpers import (
    env_get_str, env_get_int, env_get_float, env_get_bool
)


# ==========================================================================
# 1. 运行环境与数据集
# ==========================================================================

# 随机种子；用于采样与可复现设置。
SEED = env_get_int("OVERRIDE_SEED", 42)
# 数据集：voc/coco/ade20k/cityscapes。
DATASET_TYPE = env_get_str("OVERRIDE_DATASET_TYPE", 'voc')
# 固定为 sam3；其他特征骨干不再支持。
FEATURE_BACKBONE = env_get_str("OVERRIDE_FEATURE_BACKBONE", 'sam3').lower()
if FEATURE_BACKBONE != "sam3":
    raise ValueError("This SAM3-only project requires OVERRIDE_FEATURE_BACKBONE=sam3 (or unset it)")
# MLP 权重路径；空字符串表示原始 SAM3 特征，必须与库签名匹配。
FEATURE_ADAPTER_CKPT = env_get_str("OVERRIDE_FEATURE_ADAPTER_CKPT", "")
# 按 CUDA 可用性自动选择进程内设备。
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 数据加载工作进程数。
NUM_WORKERS = env_get_int("OVERRIDE_NUM_WORKERS", 8)

# ==========================================================================
# 2. 区域建库与特征过滤（仅 SMFA，无腐蚀）
# ==========================================================================

# 伪标签版本名，用于默认路径及记录。
PSEUDO_MASK_TYPE = env_get_str("OVERRIDE_PSEUDO_MASK_TYPE", 'sam3_active')
# 区域最小像素面积；建库按建库图片坐标，查询按原始提议坐标。
MIN_MASK_AREA = env_get_int("OVERRIDE_MIN_MASK_AREA", 300)

# 同类不连通区域分别建库；背景保留整图一个区域。
BUILD_SPLIT_CONNECTED_COMPONENTS = env_get_bool(
    "OVERRIDE_BUILD_SPLIT_CONNECTED_COMPONENTS", True
)

# 前景异常特征过滤比例；0 不过滤。
FILTER_FOREGROUND_DROP_RATIO = env_get_float("OVERRIDE_FILTER_FOREGROUND_DROP_RATIO", 0.1)
# 背景异常特征过滤比例；0 不过滤。
FILTER_BACKGROUND_DROP_RATIO = env_get_float("OVERRIDE_FILTER_BACKGROUND_DROP_RATIO", 0.0)
# 识别背景特征类的目录前缀。
FILTER_BACKGROUND_PREFIX = 'background'

# ==========================================================================
# 3. 提议缓存、检索分类与最终评测
# ==========================================================================

# 默认推理可视化位置；正式实验使用入口 --output-dir 指定新目录。
INF_SAVE_DIR = 'experiments/inference_vis'
# 最多保存的可视化图片数；0 不保存。
INF_VIS_LIMIT = env_get_int("OVERRIDE_INF_VIS_LIMIT", 20)
# 分段推理：提议阶段与检索阶段分离
# 从磁盘读取提议缓存，跳过模型提议生成。
INF_LOAD_PROPOSALS = env_get_bool("OVERRIDE_INF_LOAD_PROPOSALS", False)  # True=从磁盘加载预生成提议，跳过SAM3提议器
# SAM3 supplies both masks and visual features; MLP adaptation is optional.
# 即时提议类型：sam3_grid 点网格或 sam3_query 文本提议。
INF_MASK_BACKEND = env_get_str("OVERRIDE_INF_MASK_BACKEND", "sam3_grid").lower()
_default_proposals_dir = (
    f"storage/proposals/{DATASET_TYPE}/{INF_MASK_BACKEND}_raw_v1"
)
# 提议 NPZ 缓存目录；原始与清理后缓存须用不同版本。
INF_PROPOSALS_DIR = env_get_str("OVERRIDE_INF_PROPOSALS_DIR", _default_proposals_dir)
# 分类后实例分数下限，与原始提议清理 min_score 不同。
INF_INSTANCE_SCORE_THR = env_get_float(
    "OVERRIDE_INF_INSTANCE_SCORE_THR",
    0.02,
)

# 分类后实例去重 IoU 阈值，与提议清理 NMS 不同。
INF_NMS_IOU = env_get_float("OVERRIDE_INF_NMS_IOU", 0.4)
from configs.experiment import GEOMETRIC_COMPLETION as _GEOMETRIC_COMPLETION_DEFAULTS
INF_GEOMETRIC_COMPLETION = dict(_GEOMETRIC_COMPLETION_DEFAULTS)
for _key, _value in list(INF_GEOMETRIC_COMPLETION.items()):
    _reader = env_get_str if isinstance(_value, str) else env_get_bool if isinstance(_value, bool) else env_get_int if isinstance(_value, int) else env_get_float
    INF_GEOMETRIC_COMPLETION[_key] = _reader('OVERRIDE_INF_GEOMETRIC_COMPLETION_' + _key.upper(), _value)
# 评测忽略像素标签 ID。
INF_IGNORE_INDEX = 255
# 基础检索近邻数量 K。
INF_TOPK_NEIGH = env_get_int("OVERRIDE_INF_TOPK_NEIGH", 25)
# balanced 为类别平衡计分；legacy 为传统近邻投票。
INF_RETRIEVAL_MODE = env_get_str("OVERRIDE_INF_RETRIEVAL_MODE", 'balanced').lower()
# balanced 模式每类使用的最强近邻数量。
INF_CLASS_TOPM = env_get_int("OVERRIDE_INF_CLASS_TOPM", 5)
# 扩大候选检索集合的数量倍数。
INF_RETRIEVAL_SEARCH_MULTIPLIER = env_get_int(
    "OVERRIDE_INF_RETRIEVAL_SEARCH_MULTIPLIER", 2
)
# 类别中心分数的融合权重；其余权重用于近邻分数。
INF_CENTROID_WEIGHT = env_get_float("OVERRIDE_INF_CENTROID_WEIGHT", 0.2)
# 最佳与次佳类别的分数差下限；0 不额外限制。
INF_RETRIEVAL_MARGIN_THR = env_get_float(
    "OVERRIDE_INF_RETRIEVAL_MARGIN_THR", 0.0
)
# 最终区域得分中原始提议置信度的权重。
INF_PROPOSAL_SCORE_WEIGHT = env_get_float(
    "OVERRIDE_INF_PROPOSAL_SCORE_WEIGHT", 0.2
)
# 是否允许无签名的旧库；正式实验保持 False。
INF_ALLOW_LEGACY_FEATURE_BANK = env_get_bool("OVERRIDE_INF_ALLOW_LEGACY_FEATURE_BANK", False)

# 使用 FAISS 检索实现。
INF_USE_FAISS = True
# 请求 GPU FAISS；不可用时按实现回退。
INF_FAISS_USE_GPU = env_get_bool("OVERRIDE_INF_FAISS_USE_GPU", True)
# 进程内 GPU 编号；物理 GPU 由 CUDA_VISIBLE_DEVICES 选择。
INF_FAISS_DEVICE = env_get_int("OVERRIDE_INF_FAISS_DEVICE", 0)
# 小于此样本量的库使用精确检索。
INF_FAISS_EXACT_SEARCH_THRESHOLD = env_get_int(
    "OVERRIDE_INF_FAISS_EXACT_SEARCH_THRESHOLD", 100000
)

# ==========================================================================
# 4. SAM3 特征编码、点提议及文本提议
# ==========================================================================

# SAM3: pseudo masks, online proposals and frozen visual features.
# 基础 SAM3 权重路径。
SAM3_CKPT = env_get_str("OVERRIDE_SAM3_CKPT", os.path.join("weights", "sam3", "sam3.pt"))  # 本地权重优先，缺失时回退 HF 下载
# 特征编码输入边长；默认 1008，高分辨率要求兼容 336 的倍数。不是伪标签分割头尺寸。
SAM3_RESOLUTION = env_get_int("OVERRIDE_SAM3_RESOLUTION", 1008)
# trunk、fpn 或拼接 trunk_fpn；改变后须重新建库和训练 MLP。
SAM3_FEATURE_SOURCE = env_get_str("OVERRIDE_SAM3_FEATURE_SOURCE", "trunk_fpn").lower()
# build_sam3_image_model 会丢掉 stride-28 层，因此 -1=stride14，-2=stride7，-3=stride3.5。
# 可见 FPN 层索引；-1 为 stride14，-2 为 stride7，-3 为 stride3.5。
SAM3_FEATURE_LEVEL = env_get_int("OVERRIDE_SAM3_FEATURE_LEVEL", -1)
# token 归一化方式：none/layernorm/l2。
SAM3_TOKEN_NORM = env_get_str("OVERRIDE_SAM3_TOKEN_NORM", "layernorm").lower()
# 区域特征扣除图像上下文的系数；0 关闭。
SAM3_CONTEXT_SUBTRACT = env_get_float("OVERRIDE_SAM3_CONTEXT_SUBTRACT", 0.0)
# 通用 SAM3 processor 置信度阈值。
SAM3_CONF_THR = env_get_float("OVERRIDE_SAM3_CONF_THR", 0.5)
# 允许模型路径使用 BF16；实际硬件支持和显存须服务器验证。
SAM3_USE_BF16 = env_get_bool("OVERRIDE_SAM3_USE_BF16", False)
SAM3_POOLING = env_get_str("OVERRIDE_SAM3_POOLING", "hard_center_gmp").lower()
SAM3_GMP_RHO = env_get_float("OVERRIDE_SAM3_GMP_RHO", 0.1)
if SAM3_POOLING not in {"hard_center_gmp", "smfa"}:
    raise ValueError("SAM3_POOLING must be hard_center_gmp or smfa")
if SAM3_POOLING == "hard_center_gmp" and (SAM3_USE_BF16 or SAM3_CONTEXT_SUBTRACT != 0):
    raise ValueError("Mainline hard-center GMP requires FP32 and context_subtract=0")
# 点提示网格每边点数，总数为平方。
SAM3_GRID_SIZE = env_get_int("OVERRIDE_SAM3_GRID_SIZE", 48)  # 在线均匀点提示网格（48×48=2304 点）
# 每批点提示数，影响速度与显存。
SAM3_POINTS_PER_BATCH = env_get_int("OVERRIDE_SAM3_POINTS_PER_BATCH", 64)
# 点提议预测 IoU 质量下限。
SAM3_PRED_IOU_THRESH = env_get_float("OVERRIDE_SAM3_PRED_IOU_THRESH", 0.3)
# 点提议稳定性分数下限。
SAM3_STABILITY_THRESH = env_get_float("OVERRIDE_SAM3_STABILITY_THRESH", 0.5)
# 生成提议的小区域后处理面积；0 关闭。
SAM3_MIN_MASK_REGION_AREA = env_get_int("OVERRIDE_SAM3_MIN_MASK_REGION_AREA", 0)
# 点提议最大掩码占图比例。
SAM3_MAX_MASK_AREA_RATIO = env_get_float("OVERRIDE_SAM3_MAX_MASK_AREA_RATIO", 0.9)
# 历史变量名；实现使用 mask IoU 去重。
SAM3_BOX_NMS_THRESH = env_get_float("OVERRIDE_SAM3_BOX_NMS_THRESH", 0.7)  # 实际为 mask NMS 阈值
# 点提议裁剪层数，0 仅全图，1 增加 2×2 裁剪。
SAM3_CROP_N_LAYERS = env_get_int("OVERRIDE_SAM3_CROP_N_LAYERS", 1)          # 0=只整图；1=+2×2；2=+4×4（更全但更慢）
# 相邻裁剪区域重叠比例。
SAM3_CROP_OVERLAP_RATIO = env_get_float("OVERRIDE_SAM3_CROP_OVERLAP_RATIO", 512 / 1500)
# 每增加一层裁剪的点网格降采样倍数。
SAM3_CROP_N_POINTS_DOWNSCALE = env_get_int("OVERRIDE_SAM3_CROP_N_POINTS_DOWNSCALE", 2)
# 跨裁剪结果去重 IoU 阈值。
SAM3_CROP_NMS_THRESH = env_get_float("OVERRIDE_SAM3_CROP_NMS_THRESH", 0.7)
# 过滤接触裁剪人工边界的提议。
SAM3_FILTER_CROP_EDGE = env_get_bool("OVERRIDE_SAM3_FILTER_CROP_EDGE", True)
# 点提议数量上限；0 不限制。
SAM3_MAX_MASKS = env_get_int("OVERRIDE_SAM3_MAX_MASKS", 0)
# SAM3 原生 200-query 提议器；多个 prompt 使用 | 分隔。
# 文本 query 提议的提示词；多条用 | 分隔，不做同义词搜索。
SAM3_QUERY_PROMPTS = env_get_str("OVERRIDE_SAM3_QUERY_PROMPTS", "object")
# 文本提议置信度下限。
SAM3_QUERY_CONF_THRESH = env_get_float("OVERRIDE_SAM3_QUERY_CONF_THRESH", 0.15)
# 文本提议生成阶段去重 IoU 阈值。
SAM3_QUERY_NMS_THRESH = env_get_float("OVERRIDE_SAM3_QUERY_NMS_THRESH", 0.7)
# 文本提议最大掩码占图比例。
SAM3_QUERY_MAX_MASK_AREA_RATIO = env_get_float(
    "OVERRIDE_SAM3_QUERY_MAX_MASK_AREA_RATIO", 0.95
)
# 文本提议数量上限；0 不限制。
SAM3_QUERY_MAX_MASKS = env_get_int("OVERRIDE_SAM3_QUERY_MAX_MASKS", 0)


def _validate_runtime_config():
    if INF_MASK_BACKEND not in {"sam3_grid", "sam3_query"}:
        raise ValueError("INF_MASK_BACKEND must be 'sam3_grid' or 'sam3_query'")
    if not 0.0 <= SAM3_QUERY_CONF_THRESH <= 1.0:
        raise ValueError("SAM3_QUERY_CONF_THRESH must be in [0, 1]")
    if not 0.0 <= SAM3_QUERY_NMS_THRESH <= 1.0:
        raise ValueError("SAM3_QUERY_NMS_THRESH must be in [0, 1]")
    if not 0.0 <= SAM3_QUERY_MAX_MASK_AREA_RATIO <= 1.0:
        raise ValueError("SAM3_QUERY_MAX_MASK_AREA_RATIO must be in [0, 1]")
    if INF_RETRIEVAL_MODE not in {"legacy", "balanced"}:
        raise ValueError("INF_RETRIEVAL_MODE must be 'legacy' or 'balanced'")
    if INF_TOPK_NEIGH < 1 or INF_CLASS_TOPM < 1:
        raise ValueError("INF_TOPK_NEIGH and INF_CLASS_TOPM must be >= 1")
    if INF_RETRIEVAL_SEARCH_MULTIPLIER < 1:
        raise ValueError("INF_RETRIEVAL_SEARCH_MULTIPLIER must be >= 1")
    if not 0.0 <= INF_CENTROID_WEIGHT <= 1.0:
        raise ValueError("INF_CENTROID_WEIGHT must be in [0, 1]")
    if not 0.0 <= INF_PROPOSAL_SCORE_WEIGHT <= 1.0:
        raise ValueError("INF_PROPOSAL_SCORE_WEIGHT must be in [0, 1]")
    if SAM3_FEATURE_SOURCE not in {"trunk", "fpn", "trunk_fpn"}:
        raise ValueError("SAM3_FEATURE_SOURCE must be trunk, fpn, or trunk_fpn")
    if SAM3_TOKEN_NORM not in {"none", "layernorm", "l2"}:
        raise ValueError("SAM3_TOKEN_NORM must be none, layernorm, or l2")
    if not 0.0 <= SAM3_CONTEXT_SUBTRACT <= 1.0:
        raise ValueError("SAM3_CONTEXT_SUBTRACT must be in [0, 1]")
    if SAM3_FEATURE_SOURCE != "trunk" and SAM3_FEATURE_LEVEL not in {
        -3, -2, -1, 0, 1, 2,
    }:
        raise ValueError("SAM3_FEATURE_LEVEL must select one of the 3 visible FPN levels")


_validate_runtime_config()

# ==========================================================================
# 5. 数据集路径、类别映射与派生输出目录
# ==========================================================================
from configs.config_helpers import load_classes

CLS_VOC21_PATH = os.path.join('configs', 'cls_voc21.txt')
CLS_COCO_PATH = os.path.join('configs', 'cls_coco_object.txt')
CLS_ADE20K_PATH = os.path.join('configs', 'cls_ade20k.txt')
CLS_CITYSCAPES_PATH = os.path.join('configs', 'cls_city_scapes.txt')


def _load_classes_no_background(name_file: str):
    """Load class names from a file without a background line."""
    with open(name_file, "r", encoding="utf-8") as f:
        return [line.strip().split(";")[0].strip() for line in f if line.strip()]

if DATASET_TYPE == "coco":
    COCO_ROOT = 'data/COCO2014'
    COCO_TRAIN_SPLIT = 'train2014'
    COCO_VAL_SPLIT = 'val2014'
    COCO_CLASSES = load_classes(CLS_COCO_PATH)
    COCO_NUM_CLASSES = len(COCO_CLASSES) + 1
    COCO_IMAGELEVEL_TRAIN_JSON = os.path.join(COCO_ROOT, 'annotations', 'train_imagelevel.json')
    COCO_IMAGELEVEL_VAL_JSON = os.path.join(COCO_ROOT, 'annotations', 'val_imagelevel.json')
    COCO_TRAIN_LIST = os.path.join(COCO_ROOT, 'ImageSets', 'coco_train.txt')
    COCO_VAL_LIST = os.path.join(COCO_ROOT, 'ImageSets', 'coco_val.txt')
    COCO_OUTPUT_ROOT = os.environ.get("OVERRIDE_COCO_OUTPUT_ROOT", "feature_bank_coco")

    DATASET_ROOT = COCO_ROOT
    DATASET_TRAIN_SPLIT = COCO_TRAIN_SPLIT
    DATASET_VAL_SPLIT = COCO_VAL_SPLIT
    CLS_PATH = CLS_COCO_PATH
    DATASET_CLASSES = COCO_CLASSES
    NUM_CLASSES = COCO_NUM_CLASSES
    IMAGELEVEL_JSON_PATH = COCO_IMAGELEVEL_TRAIN_JSON
    OUTPUT_ROOT = env_get_str("OVERRIDE_OUTPUT_ROOT", COCO_OUTPUT_ROOT)
elif DATASET_TYPE == "ade20k":
    ADE20K_ROOT = "data/ADEChallengeData2016"
    ADE20K_TRAIN_SPLIT = "training"
    ADE20K_VAL_SPLIT = "validation"
    ADE20K_CLASSES = _load_classes_no_background(CLS_ADE20K_PATH)
    if len(ADE20K_CLASSES) != 150:
        raise ValueError(f"ADE20K class file must contain 150 classes, got {len(ADE20K_CLASSES)}")
    ADE20K_NUM_CLASSES = len(ADE20K_CLASSES) + 1
    ADE20K_IMAGELEVEL_TRAIN_JSON = os.path.join(ADE20K_ROOT, 'ImageSets', 'ImageLevel', 'train_imagelevel.json')
    ADE20K_IMAGELEVEL_VAL_JSON = os.path.join(ADE20K_ROOT, 'ImageSets', 'ImageLevel', 'val_imagelevel.json')
    ADE20K_OUTPUT_ROOT = os.environ.get("OVERRIDE_ADE20K_OUTPUT_ROOT", "feature_bank_ade20k")

    DATASET_ROOT = ADE20K_ROOT
    DATASET_TRAIN_SPLIT = ADE20K_TRAIN_SPLIT
    DATASET_VAL_SPLIT = ADE20K_VAL_SPLIT
    CLS_PATH = CLS_ADE20K_PATH
    DATASET_CLASSES = ADE20K_CLASSES
    NUM_CLASSES = ADE20K_NUM_CLASSES
    IMAGELEVEL_JSON_PATH = ADE20K_IMAGELEVEL_TRAIN_JSON
    OUTPUT_ROOT = env_get_str("OVERRIDE_OUTPUT_ROOT", ADE20K_OUTPUT_ROOT)
elif DATASET_TYPE == "cityscapes":
    CITYSCAPES_ROOT = "data/CityScapes"
    CITYSCAPES_TRAIN_SPLIT = "train"
    CITYSCAPES_VAL_SPLIT = "val"
    CITYSCAPES_CLASSES = _load_classes_no_background(CLS_CITYSCAPES_PATH)
    if len(CITYSCAPES_CLASSES) != 19:
        raise ValueError(f"CityScapes class file must contain 19 classes, got {len(CITYSCAPES_CLASSES)}")
    CITYSCAPES_NUM_CLASSES = len(CITYSCAPES_CLASSES) + 1
    CITYSCAPES_IMAGELEVEL_TRAIN_JSON = os.path.join(CITYSCAPES_ROOT, 'ImageSets', 'ImageLevel', 'train_imagelevel.json')
    CITYSCAPES_IMAGELEVEL_VAL_JSON = os.path.join(CITYSCAPES_ROOT, 'ImageSets', 'ImageLevel', 'val_imagelevel.json')
    CITYSCAPES_OUTPUT_ROOT = os.environ.get("OVERRIDE_CITYSCAPES_OUTPUT_ROOT", "feature_bank_cityscapes")

    DATASET_ROOT = CITYSCAPES_ROOT
    DATASET_TRAIN_SPLIT = CITYSCAPES_TRAIN_SPLIT
    DATASET_VAL_SPLIT = CITYSCAPES_VAL_SPLIT
    CLS_PATH = CLS_CITYSCAPES_PATH
    DATASET_CLASSES = CITYSCAPES_CLASSES
    NUM_CLASSES = CITYSCAPES_NUM_CLASSES
    IMAGELEVEL_JSON_PATH = CITYSCAPES_IMAGELEVEL_TRAIN_JSON
    OUTPUT_ROOT = env_get_str("OVERRIDE_OUTPUT_ROOT", CITYSCAPES_OUTPUT_ROOT)
else:
    DATASET_ROOT = env_get_str("OVERRIDE_DATASET_ROOT", "data/VOC2012")
    DATASET_TRAIN_SPLIT = env_get_str("OVERRIDE_VOC_TRAIN_SPLIT", "trainaug")
    if DATASET_TRAIN_SPLIT not in {"train", "trainaug"}:
        raise ValueError("OVERRIDE_VOC_TRAIN_SPLIT must be train or trainaug")
    DATASET_VAL_SPLIT = "val"
    CLS_PATH = CLS_VOC21_PATH
    DATASET_CLASSES = load_classes(CLS_VOC21_PATH)
    NUM_CLASSES = 21
    IMAGELEVEL_JSON_PATH = os.path.join(DATASET_ROOT, 'ImageSets', 'ImageLevel', f'{DATASET_TRAIN_SPLIT}_imagelevel.json')
    OUTPUT_ROOT = env_get_str("OVERRIDE_OUTPUT_ROOT", "feature_bank_voc")

# 未显式指定路径时统一落在 storage；正式实验应显式选择一个新版本。
if "OVERRIDE_OUTPUT_ROOT" not in os.environ:
    bank_kind = "mlp_banks" if FEATURE_ADAPTER_CKPT else "raw_banks"
    OUTPUT_ROOT = os.path.join("storage", bank_kind, DATASET_TYPE, f"sam3_{SAM3_FEATURE_SOURCE}_v1")

# Image-level label JSON key names
IMAGELEVEL_IMG_ID_KEY = "img_id"
IMAGELEVEL_LABELS_KEY = "labels"

# Derived paths
FEATURES_DIR = os.path.join(OUTPUT_ROOT, "features")
INDEX_DIR = os.path.join(OUTPUT_ROOT, "index")
META_DIR = os.path.join(OUTPUT_ROOT, "meta")

# 伪标签根目录；VOC PNG 平铺，其余数据集含划分子目录。
PSEUDO_MASK_ROOT = env_get_str("OVERRIDE_PSEUDO_MASK_ROOT", os.path.join('storage', 'pseudo_labels', DATASET_TYPE, PSEUDO_MASK_TYPE))
# 仅建库图片目录；SPAN 图片必须与伪标签尺寸匹配。
BUILD_IMAGE_DIR = env_get_str("OVERRIDE_BUILD_IMAGE_DIR", os.path.join(DATASET_ROOT, 'JPEGImages'))

INF_DATASET_ROOT = env_get_str("OVERRIDE_INF_DATASET_ROOT", DATASET_ROOT)
# 查询使用的单库路径；启用 MLP 时必须指定其导出库。
INF_FEATURE_BANK_ROOT = env_get_str("OVERRIDE_INF_FEATURE_BANK_ROOT", OUTPUT_ROOT)
INF_NUM_CLASSES = NUM_CLASSES

print(f"[Config] DATASET_TYPE={DATASET_TYPE}, BACKBONE={FEATURE_BACKBONE}, OUTPUT={OUTPUT_ROOT}, PSEUDO={PSEUDO_MASK_TYPE}")
