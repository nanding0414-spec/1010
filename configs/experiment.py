"""实验配方：CLI 可覆盖；不读取历史实验恢复默认值。数值不是历史最优成绩的保证。"""

# 1. Active 伪标签。面积均为输入图像像素；生成器内部仍为 1008。
PSEUDO = dict(
    dataset='voc',                 # 数据集与类别映射
    split='trainaug',               # 仅使用训练图片生成标签
    prompt_backend='active',        # 唯一保留的标签生成路线
    no_bf16=True,                   # 伪标签默认 FP32，兼容 RTX6000；False 表示请求 BF16
    conf_thr=0.5,                   # 掩码置信度下限
    min_mask_area=100,              # 输出掩码最小面积
    bucket_size=4,                  # Active 类别分桶大小
    preview=True,                  # 启用低分辨率预览
    preview_resolution=672,         # 预览编码分辨率
    presence_threshold=0.2,         # 类别存在分数下限
    active_keep_ratio=0.4,          # 保留活跃类别比例
    min_active_classes=1,           # 最少活跃类别数量
    max_active_classes=0,           # 最多活跃类别；0 不限制
    neighbor_tokens=2,              # Active 邻近类别上下文 token 数
    hypernym_tokens=2,              # Active 上位类上下文 token 数
    context_strength=1.0,           # Active 上下文强度
    min_mask_region_area=0,         # Active 后处理碎片面积阈值；0 关闭
    nms_threshold=0.7,              # 生成阶段重复掩码 IoU 阈值
    max_mask_area_ratio=0.0,         # 最大掩码占图比例；0 不限制
    force_presence_one=True,        # 已知训练类别存在概率固定为1；验证提议不使用此默认值
    edge_repair=True,               # 类别并集在合成语义标签前修复边缘
    mask_logit_threshold=0.4,        # sigmoid后的概率阈值
    close_kernel=3,
    close_iterations=1,
    dilate_pixels=1,
    diningtable_close_iterations=2,
)

# 2. MLP：先硬中心GMP聚合区域，再投影。v2 的特征扰动正则不是聚合分支。
TRAINING = dict(
    recipe='v1',                   # v1 基础训练；v2 增加正则化默认值
    classifier=None,               # None 按 recipe 选择 linear/cosine
    selection_metric='balanced',   # LSA: production-aligned macro first strict max
    feature_dropout=None,          # 特征维度随机置零比例；None 跟随 recipe
    consistency_weight=None,       # 扰动前后特征一致性损失权重
    label_smoothing=None,          # 分类标签平滑比例
    cosine_scale=16.0,             # cosine 分类器 logits 缩放
    eval_class_topm=5,             # 内部检索评估每类近邻数
    eval_centroid_weight=0.2,       # 与最终检索一致
    eval_search_multiplier=2,      # 内部候选检索数量倍数
    eval_topk=25,                  # 生产 balanced K
    epochs=50,                     # 训练轮数
    steps_per_epoch=100,           # 每轮采样批次数
    classes_per_batch=8,           # 每批抽取类别数
    instances_per_class=4,         # 每类抽取区域数
    hidden_dim=512,                # 隐藏层宽度
    num_layers=2,                  # 主线1280→512→256；3层仅显式对照
    output_dim=256,                # 导出特征维度
    lr=0.001,                      # 学习率
    temperature=0.1,               # 监督对比损失温度
    contrastive_weight=0.5,        # 监督对比损失权重
    val_fraction=0.2,              # 按图片划分内部验证比例
    seed=42,                       # 采样和划分随机种子
)

# 3. 原始提议的内存清理，坐标保持为原始验证图像坐标。
CLEANUP = dict(
    min_score=0.7,                 # 提议分数下限
    min_mask_area=100,             # 清理后的掩码最小像素面积
    min_island_area=64,            # 删除小于此面积的孤岛
    max_hole_area=32,              # 可填补孔洞的最大像素面积
    max_hole_fraction=0.01,        # 可填补孔洞相对掩码面积上限
    nms_iou=0.85,                  # 清理阶段去重 IoU，区别于分类后 NMS
)

# 4. 训练语义标签只使用同一清理程序的几何部分，无实例分数或跨类NMS。
PSEUDO_CLEANUP = {key: CLEANUP[key] for key in
                  ('min_mask_area', 'min_island_area', 'max_hole_area', 'max_hole_fraction')}
MAINLINE = dict(recipe_id='lsa_hcgmp_prr_v1',
                recipe='configs/recipes/lsa_hcgmp_prr_v1.json',
                semantic_cleanup=True, save_image_features=True,
                prefilter_percent=0, postfilter_percent=5,
                classified_min_mask_area=100, prr=True)

# Optional, class-agnostic completion after clean rasterization and HCR.
# Retained clean instances reuse their existing acceptance thresholds/scores.
GEOMETRIC_COMPLETION = dict(
    enabled=False, mode='legacy', kernel_ratio=0.10, min_kernel=3, max_kernel=31,
    min_mask_density=0.15, min_component_area=16,
    max_component_ratio=0.10, max_total_add_ratio=0.15, min_enclosure=0.35,
    bbox_only=True, background_only=True,
    debug_visualize=False, debug_image_limit=5,
    local_min_thickness=6.0, local_max_gap_ratio=1.0, local_min_alignment=0.70,
    support_search_pixels=31, support_depth_pixels=16, support_alignment_padding=3,
    support_thickness_quantile=0.25, support_gap_quantile=0.75,
)


def cli_options(settings):
    """将配方转成生成器 CLI；布尔值使用 --foo / --no-foo。"""
    result = []
    for name, value in settings.items():
        flag = name.replace('_', '-')
        if isinstance(value, bool):
            if flag.startswith('no-'):
                if value:
                    result.append('--' + flag)
            else:
                result.append('--' + ('' if value else 'no-') + flag)
        elif value is not None:
            result.extend(['--' + flag, str(value)])
    return result
