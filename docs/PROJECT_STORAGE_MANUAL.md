# ModuSeg SAM3 · presence=1＋边缘修复＋几何清理＋GMP2＋HCR 主线手册

更新：2026-10-11。本文是本项目唯一维护手册；当前正式操作以第46节为准，供应商SAM3的许可证、接口说明仍以 `sam3-main/` 为准。

### 2026-10-10 当前独立从头实验项目

用户本次指定并已通过SSH确认的根目录为 `/data/home/zystu2025/nd/1010-main`，
环境Python为 `/data/home/zystu2025/.conda/envs/moduseg_sam3/bin/python`。
本次使用项目内 `data/VOC2012` 和 `weights/sam3/sam3.pt`，当前双卡完整入口与主线激活见第46节。
下述旧项目迁移、旧缓存约定及1–42节历史命令保留；本次明确从头生成新标签、提议和库，
不使用旧缓存约定或历史模型作为输入。


> 当前主线按用户要求采用第45节已验证的presence=1＋边缘修复＋逐类别几何清理：原图Active标签 → 冻结SAM3 → 单相位硬中心GMP（ρ=0.1）→ 两层MLP → 单库检索 → clean/raw提议与HCR。当前正式操作以第46节为准。VOC val单seed42为86.7350%，属于开发验证分数；此前三seed均值86.4316%属于旧配方。1–45节保留历史操作和实验记录，旧代码/标签/库/权重/指标不改写。

### 历史服务器项目路径（2026-10-07 用户确认迁移）

- 当前项目根目录：`/data/home/zystu2025/nd/nd/ModuSeg_SAM3`。其中连续的 `nd/nd` 为用户明确指定，后续服务器命令以此路径为准。
- 旧项目根目录：`/home/zystu2025/nd/ModuSeg_SAM3`。下文旧命令及历史实验中的绝对路径属于迁移前记录，不代表当前位置；不批量改写历史 manifest、指标或完成标记。
- 2026-10-08已连接服务器核验新目录；旧目录不存在，不存在旧路径软链接。历史JSON路径由本次导入工具在内存重定位，历史记录不改写。
- 外部 VOC 数据目录和 Conda 环境是否迁移尚未确认，不据项目迁移推断其新路径。
- 已清洗提议的项目内相对路径仍为 `storage/proposals/voc/active_clean_20260928_151808`，按新项目根目录定位，运行前检查实体完整性。

### 历史服务器提议缓存约定（2026-09-28 用户指定）

后续新实验默认只读使用已清洗缓存 `/home/zystu2025/nd/ModuSeg_SAM3/storage/proposals/voc/active_clean_20260928_151808`，不重新生成提议、不重复清洗。使用 `OVERRIDE_INF_LOAD_PROPOSALS=true` 和上述 `OVERRIDE_INF_PROPOSALS_DIR`，调用 `run_inference.py` 时不传 `--cleanup-config`。分类后的实例筛选和 NMS 仍正常执行，它们不是缓存清洗。

该缓存对应原始提议 `storage/proposals/voc/pmfa_dual_20260923_162706/attempt_001`，清洗配方为 min_score=0.7、min_mask_area=100、min_island_area=64、max_hole_area=32、max_hole_fraction=0.01、nms_iou=0.85。服务器运行前检查 cleanup_config.json 的配方、errors=0、total=1449，以及实际 NPZ 与评测 ID 的完整对应关系；当前仅记录用户指定路径，未在本地验证服务器实体。

主线新增可选HCR（高置信检索救援）：在上述命令追加 `--rescue --strict-eval`，读取clean缓存metadata记录的原始提议目录，构造附加候选并只补背景；原clean缓存仍只读且不重新清洗。HCR对原始附加池在内存应用min_score=0的清理，其余清理参数不变。无开关保持旧结果。详见第31节。

下文原始提议加内存清洗的命令保留为旧协议示例。旧 PMFA/RPG/GMP 入口会显式启用清洗并校验历史输入身份，不可仅把它们的提议路径改成此缓存后直接运行或续跑。后续新实验入口应明确支持已清洗输入、跳过清洗并记录缓存来源和哈希；旧实验源码、manifest 和结果不改写。

## 1. 版本边界与算法流程

```text
训练图片 + 图像级类别
  → Active SAM3 伪标签 PNG
  → 按类别/连通区域提取掩码（不腐蚀）
  → 冻结 SAM3 token + SMFA 覆盖比例加权平均
  → 原始区域特征库 → 异常特征过滤
  → 训练 MLP → 导出 MLP 单库

验证图片 + 原始提议缓存
  → 提议分数/面积/孤岛/孔洞/NMS 清理
  → 同配置 SAM3 + SMFA + 同一 MLP
  → 单库检索分类 → 实例筛选与像素合成 → 混淆矩阵与 IoU
```

已删除：逐同义词/扩展提示词选择路线、建库腐蚀、distance_core/consistency 等非 SMFA 聚合、Global–Local 双库，以及依赖它们的旧消融入口。不会用兼容参数静默恢复这些算法。

保留 Active 的标准类名、邻居与上位类上下文，这是 Active 提议实现的一部分，不是已经删除的逐同义词候选选择路线。保留 MLP v2 特征扰动一致性正则；它不属于已删除的 consistency 聚合。SMFA 可选上下文扣除仍在，默认系数 0。

本版本已改变建库区域及部分默认配置，不能宣称复现历史 85.2553% 或 Global-only 85.6338%。不从历史实验、旧标签库或旧训练报告自动恢复“最优配置”。重新建库与训练后，以新验证结果判断效果。

## 2. 目录与不覆盖规则

```text
configs/config.py                  模型、区域建库、检索、提议和路径参数
configs/experiment.py              Active 标签、MLP、清理配方
src/                               核心算法
tools/                             生成、训练、清理、诊断及实验入口
project_utils/                     公共记录、随机种子、指标工具
tests/                             CPU 回归测试（不替代真实模型精度）
sam3-main/                         随包提供的 SAM3 源码和许可证
data/VOC2012/                      外部数据集，可链接到服务器数据目录
storage/pseudo_labels/voc/<版本>/   类别 PNG；0 背景，1–20 前景
storage/raw_banks/voc/<版本>/       原始区域特征库
storage/mlp_banks/voc/<版本>/       MLP 导出单库
storage/proposals/voc/<版本>/       提议 NPZ，raw/clean 分开命名
weights/sam3/sam3.pt                外部基础权重
weights/sam3_adapter/<版本>/        新训练 adapter.pt 和 training_report.json
experiments/<版本>/                有效配置、代码哈希、指标、诊断图
logs/                             后台启动日志和 PID
```

- 默认物理 GPU1，`CUDA_VISIBLE_DEVICES=1` 后进程内是 `cuda:0`。双卡入口明确使用物理 GPU0/1；运行前检查占用。
- 不覆盖已有库、权重或指标。每次变更参数、输入、代码都用新版本目录。
- 不修改、删除历史原始提议、伪标签、特征库和实测指标。旧实验不纳入代码压缩包。
- 数据迁移需保留旧路径软链接并记入迁移日志；运行中数据不迁移。`tools/tidy_project_root.py` 对已完成实验的归档是例外：移入 experiments，不留根目录实验链接。先看计划再执行。
- 历史 JSON 的路径是当时记录，不能直接据此续跑归档实验。必要时通过迁移记录定位并验证实体与签名。
- `--resume` 仅恢复同一精简版入口、相同输入/代码/参数的实验。旧版本实验不可续接到新代码。

## 3. 模块职责与修改位置

| 模块 | 主要文件 | 输入 → 输出及注意事项 |
|---|---|---|
| 类别列表 | tools/generate_voc_imagelevel_train_val.py | train/trainaug 划分与标注 → 图像级类别 JSON；模型生成时仅消费类别信息 |
| Active 标签 | tools/generate_pseudo_masks_sam3.py、tools/proposal_demo/ | 图片+允许类别 → PNG+生成元数据；生成失败不能计作成功 |
| 双卡标签 | tools/run_pseudo_two_gpus.py | JSON 交错分片，独立 metadata，全部完成后全量评测 |
| 区域提取 | src/region_extractors.py | 类别掩码 → 前景连通域，背景整图一域；仅面积过滤，无腐蚀 |
| 特征提取 | src/feature_extractor_sam3.py、src/feature_extractor.py | 默认硬中心GMP，SMFA仅显式兼容；原图区域映射到token网格 |
| 高分辨率 | src/sam3_resolution.py、tools/probe_sam3_resolution.py | RoPE 适配与全局注意力 query 分块；只适配特征路径 |
| 原始库 | run_build_feature_bank.py、src/build_feature_bank.py、src/filter_stage.py | 区域向量 → 分类别 NPZ、索引、类别映射、特征签名 |
| MLP | tools/train_feature_adapter.py、src/feature_adapter.py、src/adapter_training.py | 原库 → checkpoint、训练报告、投影单库；默认2层 |
| 提议 | run_generate_proposals.py、run_generate_proposals_active.py | grid/query/Active 提议 → 可复用缓存；Active 缓存供单库推理读取 |
| 清理 | tools/refine_proposals.py、src/proposal_cleanup.py | 保留原始 score，删除低分/小区域、去孤岛、填小洞、去重 |
| HCR候选救援 | src/semantic_rescue.py | 原clean＋原始候选 → 高检索证据筛选 → 只补背景；run_inference.py --rescue开启 |
| 分类与评测 | run_inference.py、src/inference.py、src/retrieval.py | 区域检索 → 语义结果、逐类 IoU、mIoU、混淆矩阵 |
| 可视化 | tools/visualize_feature_bank.py、tools/visualize_proposals.py | 特征分布、类间分离、提议可视化；不等于验证精度 |
| 参数扫描 | tools/sweep_proposal_cleanup.py | 固定库/权重/原始提议，内存清理并比较检索设置 |
| 完整双卡 | tools/run_mainline_full.py | presence1/edge标签 → 几何清理 → 原图GMP → GPU1新训MLP2 → 双卡HCR |
| 标签几何清理 | tools/clean_pseudo_labels.py、src/pseudo_label_cleanup.py | 原始语义PNG → 独立清理后PNG及逐图SHA；保护其它类别 |
| 公共实验函数 | project_utils/experiment.py | 原子 JSON、哈希、阶段重试、迁移定位、MLP 命令；无旧消融依赖 |

`tools/run_pipeline.py` 是可选 tmux “建库+推理”启动器，不负责标签生成或 MLP 训练；不要将它与完整双卡入口混淆。`debug_*` 用于局部排查，不能代替严格全量评测。

## 4. 配置如何生效

`configs/config.py` 按运行环境、建库过滤、推理检索、SAM3、数据路径分节。可调参数旁有中文说明。`OVERRIDE_<参数名>` 环境变量优先于文件默认值，例如 `OVERRIDE_SAM3_RESOLUTION=2016`。Python 导入时读取环境，启动后改环境不会修改已加载模块。

`configs/experiment.py` 中：

- `PSEUDO` 是 Active 生成 CLI 的默认值；默认 VOC trainaug、FP32 标签生成，CLI 可覆盖普通选项。请求 BF16 可将 `no_bf16` 改为 False；老卡保持 True。
- `TRAINING` 是独立 MLP CLI 与完整双卡入口共同使用的配方；值 None 表示按 v1/v2 决定。v1 默认 linear 分类、KNN 选模；v2 默认 cosine 分类与特征正则。内部验证按训练图片划分，不使用最终 VOC val 调训练参数。
- `CLEANUP` 是验证提议清理默认值；`PSEUDO_CLEANUP`是相同几何参数用于训练语义标签的子集。正式完整入口分别保存原标签和清理后标签、raw和clean提议。`MAINLINE`默认启用语义几何清理和整图FP32缓存。

主要默认值及含义：

| 模块 | 默认值 | 含义 |
|---|---|---|
| 特征 | 1008、trunk_fpn、level=-1、layernorm | 默认特征编码尺寸与特征源；改变后库和 MLP 必须匹配 |
| 区域 | 面积 300、拆连通域 | 输入坐标中的像素面积；没有腐蚀参数 |
| 库过滤 | 前景 0.10、背景 0 | 类内异常特征过滤比例 |
| MLP | 2层、隐藏512、输出256 | Linear→GELU→Linear，输出归一化 |
| 训练 | 50 轮×100 步、8 类×4 区域 | 类平衡采样，学习率 0.001，对比温度 0.1、权重 0.5 |
| 划分 | seed=42、内部验证 0.2 | 可用 --split-file 指定已有训练报告来复用图片划分 |
| 清理 | 0.7 / 100 / 64 / 32 / 0.01 / 0.85 | 依次为分数、最小掩码、孤岛、孔洞、孔洞比例、清理 NMS |
| 检索 | balanced、K=25、每类 M=5、搜索倍数 2 | 类平衡近邻计分；中心权重默认 0.4 |
| 最终筛选 | 实例分数 0.02、NMS 0.4 | 与原始提议清理的分数/NMS 区分 |

所有面积阈值都必须说明坐标。×4 宽高意味着面积 ×16；完整 2016 脚本建库面积默认 4800，验证仍按原图区域面积 300，清理按原图面积 100。不能把这些值混写为一个“面积阈值”。

## 5. 安装与部署

压缩包只含代码、配置、必要词表、测试、手册和第三方源码；不含图片、标签、特征库、提议、模型权重、历史结果。

请解压到新目录，如 `/home/zystu2025/nd/ModuSeg_SAM3_SMFA_Clean`。不要覆盖运行中的旧工程。准备数据与权重后：

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3_SMFA_Clean
conda activate moduseg_sam3
# 沿用已验证的 torch/torchvision CUDA 环境，再安装本地 SAM3 和项目依赖
python -m pip install -r requirements.txt
python -c "import sam3; print(sam3.__file__)"
nvidia-smi
mkdir -p logs experiments weights/sam3
```

SAM3 导入路径必须指向本次代码包的 `sam3-main/sam3`，避免仍加载旧工程外部 editable 安装。依赖没有包含 CUDA 驱动或 GPU 版 PyTorch；不要用本地 CPU 测试环境替换服务器 CUDA 环境。基础权重放 `weights/sam3/sam3.pt`；也可显式传路径。

VOC 需要 JPEGImages、ImageSets/Segmentation/{trainaug,val}.txt、对应图像级 JSON 与 GT。完整双卡预检要求 SegmentationClassAug 覆盖 trainaug 和 val。不自动拷贝或改写数据；已有存储可以明确指定绝对路径或创建链接。

## 6. 历史单卡示例（当前正式主线见第46节）

以下在项目根目录执行，物理 GPU1。`clean_v1` 必须是未使用过的新版本。已有提议示例路径需要按实际服务器实体检查。

```bash
export CUDA_VISIBLE_DEVICES=1
export OVERRIDE_VOC_TRAIN_SPLIT=trainaug
export OVERRIDE_SAM3_RESOLUTION=1008
export OVERRIDE_FEATURE_ADAPTER_CKPT=''

# 1) Active 伪标签；明确输入清单，保存原输入尺寸
python -u tools/generate_pseudo_masks_sam3.py \
  --dataset voc --split trainaug --no-bf16 \
  --ckpt weights/sam3/sam3.pt \
  --imagelevel-json data/VOC2012/ImageSets/ImageLevel/trainaug_imagelevel.json \
  --output-dir storage/pseudo_labels/voc/clean_v1 \
  --experiment-dir experiments/clean_v1_pseudo

# 2) 严格建原始库，不允许已有输出目录
export OVERRIDE_PSEUDO_MASK_ROOT=storage/pseudo_labels/voc/clean_v1
export OVERRIDE_OUTPUT_ROOT=storage/raw_banks/voc/clean_v1
python -u run_build_feature_bank.py --strict-build

# 3) 从新库训练 MLP 并导出配对单库
python -u tools/train_feature_adapter.py \
  --source-bank storage/raw_banks/voc/clean_v1 \
  --train-list data/VOC2012/ImageSets/Segmentation/trainaug.txt \
  --output-dir weights/sam3_adapter/clean_v1 \
  --export-bank storage/mlp_banks/voc/clean_v1 --num-layers 3 --device cuda

# 4) 将清理配方写成普通推理接受的 JSON
mkdir -p experiments/clean_v1_eval
python -c "import json; from pathlib import Path; from configs.experiment import CLEANUP; Path('experiments/clean_v1_eval/cleanup.json').write_text(json.dumps(CLEANUP, indent=2))"
export OVERRIDE_FEATURE_ADAPTER_CKPT=weights/sam3_adapter/clean_v1/adapter.pt
export OVERRIDE_INF_FEATURE_BANK_ROOT=storage/mlp_banks/voc/clean_v1
export OVERRIDE_INF_LOAD_PROPOSALS=true
export OVERRIDE_INF_PROPOSALS_DIR=/home/zystu2025/nd/ModuSeg_SAM3_Pure/storage/proposals/voc/proposals_query_voc_recall_gpu1
python -u run_inference.py --strict-eval \
  --cleanup-config experiments/clean_v1_eval/cleanup.json \
  --output-dir experiments/clean_v1_eval/predictions \
  --results-json experiments/clean_v1_eval/metrics.json
```

长期任务使用 `nohup ... > logs/<新版本>.log 2>&1 &`，并记录 `$!`。`nohup: 忽略输入` 是正常提示；错误需要看后续堆栈和阶段日志。

## 7. SPAN ×4 / 2016 双卡完整流程

这条入口不生成超分图像。训练图片由外部 SPAN 预先放大，宽高严格 ×4；用户确认本轮验证只有原图，查询端原图直接编码至 2016。

```bash
python -u tools/run_span2016_full.py --help
nohup python -u tools/run_span2016_full.py \
  --version spanx4_2016_clean_v1 \
  --train-images /home/zystu2025/datasets/VOC2012/JPEGImages_SPAN_x4 \
  --val-images /home/zystu2025/datasets/VOC2012/JPEGImages \
  --val-image-scale 1 \
  --proposals-dir /home/zystu2025/nd/ModuSeg_SAM3_Pure/storage/proposals/voc/proposals_query_voc_recall_gpu1 \
  --checkpoint weights/sam3/sam3.pt \
  > logs/spanx4_2016_clean_v1.log 2>&1 &
echo $! > logs/spanx4_2016_clean_v1.pid
```

阶段为 pseudo、build、train、eval，每阶段保留 attempt 编号。标签、建库、验证分别按图片分片使用两卡；MLP 在 GPU1 重新训练。先合并未过滤的原始库，再统一过滤一次。验证先合并混淆矩阵再计算 IoU，绝不平均两张卡的 mIoU。

标签 PNG 保持 ×4 输入尺寸，但标签生成器内部正式解码仍为 1008；2016 用于建库和查询特征，典型 stride14 网格 144×144。query 分块 256 行仍与全部 key/value 交互，数学上保持全局注意力，浮点结果及速度可能有差异。它不是切图，也不保证 24GB 显卡一定能跑完。

该实验同时存在库端 SPAN、查询端原图的处理差异，应写进结论。未经重新建库与新 MLP 训练，不能混用旧 1008 库。最终指标在 `experiments/<版本>/result.json`；每阶段详细日志在 attempt 子目录。

## 8. 诊断、扫描和签名

### 在线提议与 GT 整图覆盖差异（2026-09-28）

独立 CPU 入口 `tools/visualize_proposal_gt_diff.py`，仅依赖 NumPy 和 Pillow，不加载 SAM3、MLP 或 GPU。日常输入提议文件夹即可：

```bash
python tools/visualize_proposal_gt_diff.py --proposals-dir /实际提议文件夹
# 数据不在项目默认目录时，显式覆盖：
python tools/visualize_proposal_gt_diff.py --proposals-dir /实际提议文件夹 \
  --image-dir /实际VOC目录/JPEGImages --gt-dir /实际VOC目录/SegmentationClassAug
```

默认图片/GT 为本项目根目录下 `data/VOC2012/JPEGImages` 和 `data/VOC2012/SegmentationClassAug`，不依赖当前工作目录；用户输入的相对路径按当前工作目录解析。按文件夹顶层全部 NPZ 的文件名匹配同名图片和 GT PNG；可用 `--image-id 2007_000039` 或 `--limit 20` 缩小范围，limit=0 为全部。只读取 NPZ 中 masks/scores，不增加分数过滤、清理或 NMS，全部提议取并集 P。GT 标签 1–20 合并为前景 G，0 为背景，255 忽略。

每图仅输出一张无损差异覆盖 PNG，以原图为底图，不输出原图、GT 或提议并集的独立面板。差异以 50% 透明度叠加：绿色为正确覆盖 P∩G，红色为多分割 P−G，蓝色为少分割 G−P，灰色为忽略区域；所有计数排除 255。黄色轮廓和 #编号展示缓存中的全部非空提议，编号为 NPZ 原始顺序从 1 开始；包含与 GT 完全不相交、嵌套及位于忽略区域的提议，不按 GT 或分数筛选。完全重合的轮廓/编号可能重叠，空掩码无轮廓。GT 外的有效覆盖保持红色，255 区域仍为灰色且不计误分割。图例使用英文 TP/FP/FN/Ignore，标题显示图片 ID、提议数、前景 IoU。此处仅评估前景覆盖，不区分类别、实例粘连、重复提议，也不是最终语义 mIoU。

自动新建 `experiments/proposal_gt_diff_<时间戳>/`，重名追加序号；`--output-dir` 可指定新目录，已有目录拒绝覆盖。`overlays/` 保存单张覆盖图（旧版本 comparisons 四联图不会被修改），`summary.csv` 保存逐图 TP/FP/FN、有效/忽略像素数、Precision/Recall/IoU 和错误。`run.json` 记录参数、所选 ID、脚本及成功读取输入的路径/大小/mtime/SHA256、成功/失败数和像素汇总指标。Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，IoU=TP/(TP+FP+FN)；分母为零时 JSON 为 null、CSV 留空、图中为 undefined。汇总先累加成功图片像素计数再计算比值，不平均逐图比值。

严格核对二值掩码、scores 长度/有限值、GT 合法标签及图片/GT/非空掩码尺寸；保持文件原始像素方向，不做 EXIF 旋转或缩放。兼容历史 (0,1,1) 空提议，此时全部 GT 前景为漏分割。逐图缺失/损坏/尺寸不符会记错误并继续；最终 status=incomplete 且退出码为 1，不能作为完整评测结果。目录缺失等启动错误直接停止。不修改原始输入，不自动生成提议或打包。

合成数据验收：`python -B -m unittest tests.test_visualize_proposal_gt_diff -v`。测试只验证覆盖计算、着色和数据流，不代表实际在线提议质量。

### 其他诊断入口

```bash
# 明确指定一张有前景、原尺寸标签对齐的图做探针
python tools/probe_sam3_resolution.py --cuda 1 \
  --image data/VOC2012/JPEGImages/2007_000039.jpg \
  --mask /实际路径/原尺寸标签/2007_000039.png \
  --span-image-dir /home/zystu2025/datasets/VOC2012/JPEGImages_SPAN_x4 \
  --resolutions 1008 1344 2016 --output-dir experiments/probe_clean_v1

# 特征可视化不再依赖旧聚合实验目录
python tools/visualize_feature_bank.py \
  --bank raw=storage/raw_banks/voc/clean_v1 \
  --bank mlp=storage/mlp_banks/voc/clean_v1 \
  --output-dir experiments/bank_vis_clean_v1
python tools/sweep_proposal_cleanup.py --help
python tools/refine_proposals.py --help
```

原始库与 MLP 库维度可以不同，比较时工具分别投影；降维图只用于定性分析，类内紧密度和类别余量也不是最终验证精度。

特征签名包含骨干、特征源、层、分辨率、归一化、上下文、维度、网格、精度，以及可选 MLP 架构/权重哈希和分辨率适配。读取时必须匹配；拒绝旧非 SMFA 或双库签名。不能仅凭文件夹名或向量维度选库。记录的基础权重哈希及实际导入 SAM3 源码位置也要核对。

## 9. 复现与验收要求

1. 运行前核对实际 CLI：不使用删除模块的旧命令，也不把无效环境变量当作仍生效的参数。
2. 全量验证必须 `--strict-eval`、`--eval-limit 0`；建库使用 `--strict-build`。缺失、重复 ID、提取失败都需要处理，不能静默跳过后报告完整精度。
3. 记录有效参数、输入、划分、代码和权重哈希、库签名。完整双卡入口写 manifest、阶段命令环境、训练报告和汇总指标；独立运行需保留各模块报告与启动命令。
4. 分清伪标签质量、MLP 内部检索验证与最终 VOC val 分割 mIoU。指标差使用“百分点”，不同数据/配置的差值不能单因素归因。
5. CPU 检查：`python -m unittest discover -s tests`；Active 专项：`python -m unittest tools.proposal_demo.test_active`。其中模拟输出验证的是数据流，不代表 SAM3 checkpoint 的真实预测。
6. 未连接服务器就明确说明未启动服务器任务。高分辨率显存、耗时及真实精度只能以服务器日志为据。

## 10. 第三方来源

SAM3 源码与许可位于 `sam3-main/`。Active 部分参考 VILA-Lab/ActiveSAM，提交 `41cad43aa340d15876d2dcf1eeda92355fc21695`，MIT 许可保存在 `tools/proposal_demo/ActiveSAM_LICENSE.txt`。

Active 实现保留低分辨率类别筛选、标准提示与上下文、提示分桶；未实现论文的完整双分支、类别竞争或背景校准。它只用实例掩码，伪标签按允许类别的顺序填充，不能宣称复现论文成绩。供应商源码中的通用形态学或其他模型内部功能不属于项目已移除的建库腐蚀/聚合分支，不应为了字面清理而破坏 SAM3 依赖。

## 11. 可移除的 SMFA / PMFA 对照实验（2026-09-23）

用户明确要求的独立研究入口：`tools/experimental_pmfa_ab.py`。生产主线仍为 SMFA 单库版；本入口仅在自己的 worker 子进程内启用 PMFA 和统一的 FP32 修正，不编辑 `src/`、`configs/` 或供应商源码，不影响已运行任务。新增的 `tests/test_experimental_pmfa_ab.py` 是实验专用测试。

### 运行

在服务器项目根目录、`moduseg_sam3` 环境中先预检。`--version` 必填，每个新实验使用新名称：

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B tools/experimental_pmfa_ab.py --help
python -B -m unittest tests.test_experimental_pmfa_ab -v
nvidia-smi
python -B tools/experimental_pmfa_ab.py --version pmfa_ab_v1 --preflight-only
python -u -B tools/experimental_pmfa_ab.py --version pmfa_ab_v1
```

若需要后台运行，确保启动日志也使用未占用的新名字：

```bash
mkdir -p logs
(set -o noclobber; nohup python -u -B tools/experimental_pmfa_ab.py --version pmfa_ab_v1 > logs/pmfa_ab_v1_launcher.log 2>&1 &)
```

默认物理 GPU1，脚本自行在每个子进程设置 `CUDA_VISIBLE_DEVICES=1`，程序内为 `cuda:0`；顺序运行，不使用另一张卡。每个新 worker 前重新检查 GPU，占用超过 1000 MiB 或利用率超过 10% 则停止。完整任务先执行一张真实图片的 FP32 特征前向探针；`--preflight-only` 为只读检查，不加载模型、不写实验目录。驱动版本以实时 `nvidia-smi` 为准。

默认数据根目录 `/home/zystu2025/datasets/VOC2012`，原图目录 `/home/zystu2025/datasets/VOC2012/JPEGImages`，训练使用 trainaug，验证使用 val；权重默认本工程 `weights/sam3/sam3.pt`。可以显式指定：

```bash
python -u -B tools/experimental_pmfa_ab.py --version pmfa_ab_v2 \
  --gpu 1 --dataset-root /home/zystu2025/datasets/VOC2012 \
  --image-dir /home/zystu2025/datasets/VOC2012/JPEGImages \
  --checkpoint weights/sam3/sam3.pt \
  --pseudo-dir /实际存在的完整训练标签目录 \
  --proposals-dir /实际存在的完整验证提议目录
```

上例最后两项是占位路径，必须替换为真实路径，或直接省略。完整有效的目录只读复用；缺失、不完整或内容校验失败时，在本实验新目录重新生成**全量**缓存一次，不往旧目录补写。两版共享最终选定的同一份输入。

其他输入覆盖参数为 `--train-list`、`--val-list`、`--imagelevel-json`、`--gt-dir`。标签生成需要已存在的训练图像级 JSON，脚本不从 GT 自动生成它；缺少划分、JSON、GT 或权重时列出错误并停止，不自动下载。验证 GT 优先选择完整的 `SegmentationClassAug`，否则完整的 `SegmentationClass`，也可显式传入。自定义 train-list 必须与训练 JSON 完全对应，且不与 val-list 重叠。

### 双卡后台模式

GPU 检查补充：显存超过 1000 MiB 立即停止；显存不超过此值但利用率超过 10% 时，每 2 秒复查，最多等待 30 秒，仍超标则停止。这用于处理上一 worker 退出后利用率采样窗口的短暂残留，不放宽启动阈值。更新实验脚本会改变代码哈希，旧版本不能直接 `--resume`，应使用新实验版本并保留旧产物。

添加 `--parallel-gpus 0 1`：物理 GPU0 跑 SMFA，物理 GPU1 跑 PMFA，每个 worker 内都是 `cuda:0`。这是两个独立实验，不是一个模型跨卡。预检检查两卡，各卡分别完成真实 FP32 探针。共用标签与提议仍先在 `--gpu`（默认1）上准备一次，然后两版并行建库；PMFA 训练等待 SMFA 已完成并校验的训练报告，以复用内部验证图片成员，可与 SMFA 全量评测重叠。准备和等待阶段可能有一张卡空闲。

上传更新后的实验脚本，在服务器执行：

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
nvidia-smi
version="pmfa_dual_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_pmfa_ab.py \
    --version "$version" --gpu 1 --parallel-gpus 0 1 \
    > "logs/${version}_launcher.log" 2>&1 &
  echo $! > "logs/${version}_launcher.pid"
)
echo "版本：$version"
tail -f "logs/${version}_launcher.log"
```

启动日志及 PID 不覆盖旧文件。`tail -f` 用 Ctrl+C 退出不会停止后台实验。要中止实验，核实 PID 后执行 `kill -TERM "$(cat logs/${version}_launcher.pid)"`；启动器只清理自己启动的 worker。双卡任一侧失败会取消另一侧尚未完成的工作，已完成阶段和完整结果保留，状态标记 incomplete；必须两版完整评测成功才生成有效对比。续跑保持原版本、GPU 参数和其他参数不变并添加 `--resume`，启动日志另取新名字。下文先原版后新版的顺序及失败规则仅适用于默认单卡模式。

### 固定协议与参数

第二轮启动入口 `tools/experimental_pmfa_round2.py`（Linux）：读取上一轮完整默认 PMFA 实验的 manifest 和完成标记，校验原输入/权重哈希，复用相同标签与提议。GPU0 顺序运行新 SMFA 基线和 alpha=0.25 PMFA；GPU1 顺序运行新 SMFA 基线和 steps=0 PMFA，其余默认参数不变。它不会复用上一轮的模型成绩代替基线。各任务使用独立版本和后台进程；不向 A/B 子进程传入 `--parallel-gpus`。两个独立对照互不取消，单个对照内部保持原失败规则。

```bash
python -B tools/experimental_pmfa_round2.py --preflight-only
python -B tools/experimental_pmfa_round2.py
# 若发现多个历史默认 PMFA 实验，显式选择：
# python -B tools/experimental_pmfa_round2.py --previous experiments/实际上一轮版本
```

启动器只在唯一完整默认 PMFA 实验时自动选择；缓存不完整则停止，不分别再生输入。启动后输出两个 PID 和日志路径，可退出终端。`experiments/pmfa_round2_<时间>/` 记录启动命令和 PID；两份结果分别位于同名前缀加 `_alpha025`、`_steps0` 的实验目录。启动器退出不代表实验完成，以各子实验 status/comparison 为准。停止时向记录的 A/B 启动器 PID 发送 SIGTERM。专用测试为 `tests/test_experimental_pmfa_round2.py`；不再使用时可删除第二轮脚本和测试，不影响原实验入口和生产模块。第二轮未新增逐图混淆及提议分数导出。

PMFA 覆盖率允许 FP32 的 1e-6 边界误差：INTER_AREA 映射可能将全覆盖 token 算为略大于 1。超出容差或非有限值仍报错并显示范围；容差内仅在 PMFA 运算中裁剪到 [0,1]，记录 coverage_roundoff_regions/tokens。alpha=0 与 SMFA 回退保留原始权重计算。实验签名的 experimental_pooling.version=2 记录此策略，拒绝混用修复前的实验库。代码变化后必须使用新实验版本；完整旧标签和提议可经完整性校验后只读复用。

流程为：输入准备 → 原版 SMFA 建库/过滤/新训 MLP/全量验证 → 新版 PMFA 同样四步 → 对比。两版都是 1008、trunk_fpn、level=-1、layernorm、context_subtract=0、FP32、原图输入。脚本清除继承的 `OVERRIDE_*` 后显式设置实验参数，避免旧终端环境污染。FP32 同时关闭 BF16 autocast 并应用 `use_standard_vit_mlp()`，处理供应商 ViT MLP 内部强制 BF16 路径；没有自动降分辨率、改精度或 CPU 推理回退。

PMFA 默认 `--core-threshold 0.9 --steps 2 --rel-margin 0.12 --temperature 0.2 --alpha 0.5 --min-support 3`。高覆盖 token 初始化、相似度迭代、覆盖率乘注意力、与 SMFA 均值融合、最终 L2 归一化；小支持/零 Proxy 回退 SMFA，alpha=0 调用原 SMFA。单区域与批量路径使用同一核心，同图只编码一次，最多每块 16 个掩码，不扩张成区域×token×通道三维张量。

两版分别过滤前景最远 10%，背景不删；这是完整流程对照，**变化包含过滤后的库成员变化**。MLP 均重新训练，v1、三层、512 隐藏维、256 输出维、50轮×100步、8类×4区域、seed42；新版使用原版报告的内部验证图片成员，报告实际过滤后覆盖，不自动调整缺类划分。两版共享 CLEANUP 配方与 balanced 检索参数，严格全量 val 评测，不自动进行 val 参数扫描。

新生成训练标签使用 PSEUDO Active 配方（FP32、训练允许类别）；新生成验证提议使用独立 Active 导出入口默认参数并显式 FP32，使用整个 VOC 前景词表，**不读取验证图片的真值类别**。标签准备不另做标签 mIoU，以免混淆本次最终对比；最终验证必须有完整 GT。

### 结果、失败与移除

最终读取 `experiments/<版本>/comparison.md`、`comparison.json`、`per_class.csv`。包含两版 mIoU、百分点变化、逐类与背景 IoU、混淆矩阵、评测 ID、过滤成员差异、内部划分实际覆盖、阶段耗时、worker 峰值显存及 PMFA 支持/回退累计统计。一个 seed 的结果不代表统计显著。JSON 中存储 IoU 比例，`delta_pp` 已乘100。

标签/提议/原库/MLP 库仍各自在 `storage` 的对应分类，权重在 `weights/sam3_adapter/<版本>`，worker 日志在 `logs/<版本>`；每阶段保留 attempt 和 worker_report。`artifacts.json` 列出本次拥有的输出根目录与只读复用输入。默认不覆盖，完全相同的运行可添加 `--resume`；输入和代码 SHA256、阶段产物 SHA256 必须匹配。失败重试创建新 attempt。原版失败不运行新版；新版失败保留原版结果，`status.json` 标为 incomplete。

`launcher.lock` 防止同版本同时启动。正常退出和中断会删除锁；强制杀进程可能留下锁，必须核实其中 PID 已结束，才手动移除。中断仅终止本启动器拥有的 worker 进程组。脚本不提供自动清理命令。

实验特征签名使用 format_version=3001，并记录 experimental_pooling（算法、参数、FP32策略、分块）；实验 worker 严格比较该字段和已有签名、MLP 权重哈希。普通生产推理会拒绝实验格式，不能绕过签名混用；原版在本脚本中也记录实验精度身份，其算法仍为原始 SMFA。

不采用该方案时，移除 `tools/experimental_pmfa_ab.py` 和 `tests/test_experimental_pmfa_ab.py` 即停止提供实验功能，无需恢复任何生产代码。结果目录按 `artifacts.json` 人工审查后再决定是否清理，不能删除只读复用的标签、缓存或其他历史实验。本地模拟测试不代表服务器 CUDA 运行成功或真实分割精度。

## 12. RPG-SMFA 独立消融实验（2026-09-24）

恢复入口补充：若误将修复后的 RPG Python 文件也上传，原实验会因源码哈希改变而拒绝续跑。更新后的 `experimental_rpg_resume.sh` 会读取 B/C 原 manifest 中记录的源码身份；只有将 verify_records 恢复为历史实现后，整个文件的 SHA256 和大小都精确匹配原记录，才备份现文件到总调度目录并原子恢复源码，再在父进程内应用正确的比较函数。它不改旧 manifest 或算法。额外编辑、组间记录冲突或存在 launcher.lock 时拒绝自动恢复；备份与恢复凭据保存在实验目录。上传此 shell 文件即可，不需要再次覆盖任何 Python 文件。

**旧实验复核排序错误的恢复（版本 rpg_20260924_163342）：** 原 verify_records 将多个阶段拼接后的未排序清单与 identities 返回的全局排序清单直接比较，会误报 Baseline artifacts files changed or missing。已改为按路径比较 size/SHA256，重复一致记录允许，冲突、修改或缺失仍拒绝。若需要保留旧运行的源码身份并续跑，服务器仅新增 `tools/experimental_rpg_resume.sh`，不要覆盖原 RPG/PMFA Python 脚本。这个 shell 入口只在父进程内替换清单比较函数，不修改 manifest、不关闭哈希校验、不改变 worker 算法。`.sh` 不属于原 Python 源码清单，能保持已完成阶段的续跑身份。执行 `bash tools/experimental_rpg_resume.sh --previous experiments/pmfa_fixed_20260924_101553 --version rpg_20260924_163342 --gpus 0 1 --variants B C D --resume`；后台运行可加 nohup 并使用新的恢复日志名。B 的已完成四阶段经校验后跳过并重新生成比较报告，C 保留失败 attempt、重新执行未完成的建库阶段，D 正常启动。建库没有逐图断点，不应称为从 C 的最后一张图继续。若实际文件变动，恢复入口会指出路径并停止。新版本实验直接使用修复后的 RPG Python 脚本。

入口 `tools/experimental_rpg_ab.py`，专用测试 `tests/test_experimental_rpg_ab.py`。复用第 11 节实验脚本的预检、FP32 修正、建库/过滤、MLP 训练、严格验证和阶段校验工具；两个脚本均须保留。没有修改生产模块、原有 PMFA 脚本或配置默认值。删除 RPG 脚本及其专用测试即可移除该实验。

### 与参考方案的区别及固定定义

项目实际顺序为 SAM3 token → 区域聚合并 L2 归一化 → 三层 MLP；不是参考附件所写的先对 patch 做 MLP。本实验保持实际顺序，B/C/D 各自建单库并重新训练同结构、同 v1 配方的 MLP，不能复用 SMFA 的 MLP 权重，也不新增模块训练阶段。分别过滤前景最远 10%，背景不删；结果包含过滤成员和训练特征变化。

附件正文与参考代码的 q0 不一致，本实验采用正文：对处理后的原始 token 做覆盖率加权均值，再 L2 归一化。相似度仍使用 L2 token，最终 Value 使用原处理后 token。B 使用 q0 和 M×sigmoid；C 从核心候选中选择与 q0 最相似的 top-k 后生成 q1，再使用 M×sigmoid；D 使用相同 q1 和 M×[lambda+(1-lambda)×sigmoid]。三组 temperature=0.1、core_threshold=0.8、core_ratio=0.3，D 的 lambda_base=0.5，B/C 的有效 lambda 固定为 0。C/D 只更新一次 Proxy。中位数采用 torch.median（偶数个元素时取较低中位数），记录在签名中。

核心不足 3 个时从阈值 0.8 放宽至 0.5，再放宽至全部正覆盖 token。有效 token 少于 3、零 Proxy 或零权重回退原始 SMFA；空区域返回零。D 的 lambda=1 精确调用原 SMFA。覆盖率保留 1e-6 浮点容差，仅裁剪容差内误差，真正越界/NaN/Inf 报错。门控计算不回传梯度，最终 Value 聚合保持梯度连接。建库、单区域及批量查询共享同一核心；同图共享前向，批量按最多 16 个掩码分块。

实验使用 format_version=3002，并严格匹配全部参数、q0 定义、median 定义、MLP 前聚合位置、FP32策略和 MLP 来源签名。不同组、参数、PMFA 或生产库不可混用。上下文扣除为 0，输入为原图 1008，沿用现有 mask 网格映射、标签、提议、清理和检索。

### 复用基线与运行

用户已完成的 SMFA 作为 A 组只读复用，**本入口不启动任何 SMFA worker**。`--previous` 必须指向之前完整的 PMFA/SMFA 实验目录，不能只给 comparison.json。预检核对该实验的原输入和权重哈希、已记录源码、SMFA 建库/训练/验证的产物哈希和签名、FP32及环境配方。允许新增加独立 experimental_ 工具文件，但不能更改已记录源码或增加未验证生产代码。完整旧标签和提议必须存在；不完整则停止，不重新生成。源码或产物不同需定位差异，不应绕过检查。

默认 GPU0 依次 B→D，GPU1 跑 C；三个变体使用同一份旧 SMFA 内部验证图片划分。某组失败会取消本次启动器尚未完成的其他 worker，保留完成阶段和报告。两卡都必须空闲，原任务是否停止以实时 nvidia-smi 为准。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_rpg_ab -v
nvidia-smi
python -B tools/experimental_rpg_ab.py \
  --previous experiments/pmfa_fixed_20260924_101553 --preflight-only

version="rpg_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_rpg_ab.py \
    --previous experiments/pmfa_fixed_20260924_101553 \
    --version "$version" --gpus 0 1 --variants B C D \
    > "logs/${version}_launcher.log" 2>&1 &
  echo $! > "logs/${version}_launcher.pid"
)
tail -f "logs/${version}_launcher.log"
```

上面的 previous 来自用户已完成实验日志；若迁移或更名，改为实际完整目录。只跑 D 可指定 `--gpus 1 --variants D`。预检为只读，不运行模型；真正执行时每组先做一张图的真实 FP32 特征探针，OOM 不降精度或分辨率。

### 输出及续跑

总调度目录 `experiments/<version>/` 保存 plan、锁与最终 summary.json；各组 `experiments/<version>_{B,C,D}/` 保存 manifest、阶段报告、comparison.json/md、per_class.csv、status 和只读基线来源。各组 raw_banks/mlp_banks 在 storage 对应分类，权重在 weights/sam3_adapter/<组版本>，worker 日志在 logs/<组版本>。artifacts.json 列出各组自有根目录及只读输入；不自动删除任何数据。

阶段 attempt 内的 region_diagnostics.jsonl 记录图片 ID、区域顺序、有效/core/选中 token 数、回退原因、q0/q1 余弦相似度、门控均值/极值、乘法因子极值、RPG 与 SMFA 输出余弦相似度和聚合耗时；B 没有核心更新，核心计数为 0。image_diagnostics.jsonl 记录逐图处理耗时和累计聚合耗时，处理计时范围为建库区域提取或验证实例预测，不等同于完整端到端时间。probe 的诊断图片 ID 为空。诊断不缓存完整 token，也不输出类别置信度。额外诊断开销包含在本次耗时中，不能当作纯算法性能基准。

完成汇总包含两版 mIoU/差值、逐类和背景 IoU、混淆矩阵、评测 ID、库成员差异、内部划分实际覆盖、阶段耗时和峰值显存。仅在完整评测且输入/源码/产物复核通过后生成有效 comparison；单种子只标提高、持平、下降。统计显著性未检验。

续跑使用完全相同 version、previous、GPU、组别和参数并加 `--resume`，启动日志另取新名；只跳过哈希校验通过的阶段，失败阶段新建 attempt。发送 SIGTERM 给 launcher PID 会仅停止其 worker；不要用 pkill python。强杀后残留锁需核实 PID 已结束才人工处理。本地缺 PyTorch 时张量/GPU测试跳过，不代表服务器运行通过。

## 13. SMFA 信息保留文献与机制核验（2026-09-27）

本轮交付是研究报告与 CPU 数学核验，不是新增生产聚合入口或真实 A/B 实验。报告在 [experiments/smfa_information_retention_20260927_v1/report.md](../experiments/smfa_information_retention_20260927_v1/report.md)，同时保存 sources.json、source_snapshot.json 和 mechanism_checks_v1 下的数值结果。ApET/GMP/LRCP 为优先参考；CAR/IRN/NTR 的新增训练路线未实现。

同目录 mechanism_probe.py 是仅依赖 NumPy 的低维合成探针，实现覆盖率归一化 GMP、线性残差和明确中心化政策的 SVD 诊断；不是生产实现或论文全流程复现。已在本地 CPU float64 通过 30 项检查，覆盖重复、互补、污染、小支持、求解一致性及分组合并等价性。结果不是 SAM3 精度、VOC IoU 或 GPU 性能。未改变 src、configs、原 PMFA/RPG 脚本，也未启动服务器任务。

复核时使用新的结果目录（已存在则拒绝写入），或省略 --output-dir 仅打印 JSON：

```bash
python -B experiments/smfa_information_retention_20260927_v1/mechanism_probe.py --help
python -B experiments/smfa_information_retention_20260927_v1/mechanism_probe.py \
  --output-dir experiments/smfa_information_retention_20260927_v1/mechanism_checks_v2
```

探针只消费自身构造的 token，不读取数据、标签、提议或权重。后续真实聚合实验仍需独立入口、新版本、新库及匹配签名、重新训练的 MLP；普通已聚合向量库不能用于恢复原始 token。不要直接将此探针注册为生产聚合方法。

## 14. 覆盖率加权 GMP 后台实验（2026-09-27）

独立入口 `tools/experimental_gmp_ab.py`，测试 `tests/test_experimental_gmp_ab.py`。依赖现有 experimental_pmfa_ab.py 和 experimental_rpg_ab.py 的通用流程函数，不修改它们或生产文件；服务器只需增加 GMP 脚本和测试，不要覆盖旧实验源码。入口在父进程使用按路径核对的哈希校验，不依赖旧 RPG 校验函数的排序行为。

算法定义：p=m/sum(m)，S=Fᵀdiag(p)F，mu=Fᵀp，lambda=rho×tr(S)，求解 (S+lambda I)z=mu，最后 L2 归一化。固定 rho=1.0 和 0.1 两候选。lambda 是项目尺度适配，不沿用合成探针固定 0.1；输入为现有处理后的原 token，不额外归一化 token、不加 ReLU、不裁剪负系数。FP32 Cholesky 按 N/C 选择较小的原始/对偶系统，不显式求逆，相对正规方程残差上限 1e-4。空/全零输入返回零；非空建库区域输出零则停止。覆盖率允许 1e-6 舍入误差，真正越界、非有限值、分解失败、OOM 不静默回退或降分辨率。签名 format_version=3004 记录目标、尺度正则、原 token、求解器和边界规则，不能与其他库混用。

流程：只读复核完整旧 SMFA 基线和输入 → 重建固定区域清单 → 内部训练图片 seed42 固定抽取128张真实探针 → GPU0/1 分片提取完整训练集 → 合并两个固定成员库 → 双卡分别训练候选 MLP → 仅按内部验证 best_macro_accuracy 选候选 → 只有超过旧 SMFA 内部成绩的胜者执行完整 VOC val。完全同分优先 rho=1；两组都未超过则输出 complete_no_candidate_improvement，不读取最终 val 成绩挑参数，不自动追加搜索。仍预检完整 val GT/输入，以确保胜者可评测。

不重跑 SMFA、伪标签或提议。固定旧 SMFA 过滤后的 sample ID；按原始训练顺序、类别全局计数重建所有区域 ID，记录掩码哈希。每个保留区域重新算 SMFA 并与旧原库向量比较（atol=rtol=5e-5），验证区域匹配；两候选不再独立过滤。各候选新训三层 MLP，沿用 v1、50×100、seed42、原内部图片划分和检索参数。报告是固定成员对照，不能与分别过滤的实验混为同一协议。

每图一次 SAM3 前向，同时计算两种 GMP。storage/raw_banks/voc/<版本>/image_cache 保存已聚合向量和逐图完成校验，不保存完整 token。中断后仅跳过校验通过的图片；未完成图片使用新 attempt。探针图片缓存可用于后续建库。探针额外对每图第一个保留区域做残差诊断：均匀抽至最多512个正覆盖 token 构造覆盖率加权 SVD 基准，rank=8/16，分别未中心化和中心化，在区域全部正覆盖 token 上计算平方残差并记录最高16个网格索引；它只是采样基准诊断，不代表 ApET/LRCP 复现，不影响聚合和选择。

默认探针平均每图时间预算120秒（包括诊断），超过则停止、不进入全量提取。可在新版本启动前用 --max-probe-mean-seconds 明确设定预算；不自动提高预算。probe_summary.json 记录真实耗时、最大方程残差和图片清单。预检只检查输入/设备，不加载模型。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_gmp_ab -v
nvidia-smi
df -h .
python -B tools/experimental_gmp_ab.py \
  --previous experiments/pmfa_fixed_20260924_101553 --preflight-only

version="gmp_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_gmp_ab.py \
    --previous experiments/pmfa_fixed_20260924_101553 --version "$version" --gpus 0 1 \
    > "logs/${version}_launcher.log" 2>&1 &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

输出 experiments/<版本>/ 下的 selection.json、status.json、probe_summary.json；胜者完成严格全量评测后生成 comparison.json/md、per_class.csv。记录基线来源、固定成员、一致划分、内部成绩、全量ID/混淆矩阵、阶段耗时与显存；单种子无显著性声明。缓存 diagnostics.json 包含每区域正则尺度、支持数、求解残差、负系数比例、低覆盖 token 系数绝对值占比和 SMFA 余弦相似度。训练完成不等于全量评测完成。报告出错后可复用完成的 eval 阶段重做报告。

库按 storage 分类，权重 weights/sam3_adapter/<版本>，日志 logs/<版本>，artifacts.json 列出本次自有根目录和只读输入。已有版本拒绝覆盖；相同参数/源码/输入使用原版本加 --resume，恢复日志另取名字。SIGTERM 启动器仅终止自有 worker；失败保留 attempt。没有自动删除。只移除新增 GMP 脚本及测试即可停用，不需恢复主线；未使用服务器时不声称获得真实精度。源码锁定后不要在运行或续跑前覆盖任意旧脚本。

## 15. SMFA/GMP 配对种子验证（2026-09-27）

新增独立脚本 `tools/experimental_gmp_seeds.py`，测试 `tests/test_experimental_gmp_seeds.py`，依赖既有 PMFA/RPG/GMP 实验脚本；不修改旧入口、生产源码或历史产物。服务器只上传这两个新增文件。固定 GMP rho=0.1、原先过滤后完全相同的区域成员和内部图片划分，复用已完成 seed42 结果。GPU0 顺序执行 SMFA seed123 训练/验证 → GMP seed123 训练/验证；GPU1 同样执行 seed2026。只改现有训练 CLI 的 --seed，强制 --split-file 指向原 SMFA 的 training_report.json，记录实际训练 argv；其他 v1 配方沿用原实验，评测参数保持原值。

不重新生成标签/提议，不执行原始建库、GMP训练集求解或区域过滤，不重新挑 rho。每个新模型只训练 MLP、导出独立 MLP 库、完整 VOC val 评测。seed42 的现有权重及结果只读复用；没有历史逐图混淆文件就明确标记缺失，不为补齐它自动重跑。新种子的逐图混淆矩阵保存于 eval attempt 的 per_image_confusion.npz（ids、confusions[N,21,21]），合计必须与最终 metrics.json 完全一致。没有新增 token 缓存。

`--previous` 指向刚完成的 GMP 实验目录；省略时只允许唯一完整、固定成员且胜者 rho01 的 comparison.json 自动匹配。多个候选必须显式指定。预检验证旧 GMP 源码和输入哈希、所选 raw/train/eval 产物、SMFA 源与产物、精度/签名、库成员、内部划分、最终评测覆盖、两卡占用。rho01 非内部选优结果时拒绝；不把任意历史最高分作基线。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_gmp_seeds -v
nvidia-smi
python -B tools/experimental_gmp_seeds.py --preflight-only

version="gmp_seeds_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_gmp_seeds.py \
    --version "$version" --gpus 0 1 > "logs/${version}_launcher.log" 2>&1 &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

若需要明确来源，给预检和启动命令同时追加 `--previous experiments/实际GMP实验版本`。结果 experiments/<版本>/summary.json、summary.csv 和 comparison_seed42/123/2026.json 包含三个配对差值、均值、样本标准差、最小值、正收益种子数、各类差值及 worker 耗时/显存。summary 完成前会再次复核阶段、输入、源码及引用产物；报告错误可以续跑完成阶段后再汇总。三个种子不自动宣称统计显著，VOC val 已参与开发，不是独立最终测试证据。

权重 weights/sam3_adapter/<版本>/<方法_seed>，导出库 storage/mlp_banks/voc/<版本>/<方法_seed>，日志 logs/<版本>；不新增 raw_banks。artifacts.json 区分自有目录和只读原库/引用实验。相同配置使用原版本加 --resume；失败阶段新 attempt、完整校验后跳过完成阶段。原始源码和输入变化拒绝续跑，不自动迁移签名。任一任务失败会取消该启动器未完成的 worker；已完成计算保留。SIGTERM 启动器停止自有 worker，禁止 pkill python。删除新增脚本及测试即可移除此入口，无需恢复主线。

## 16. MPSA 完全替换聚合的独立实验（2026-09-28）

预检兼容修复：旧基线将新增加的 `tools/visualize_proposal_gt_diff.py` 误判为生产模块。MPSA 入口现在仅在旧基线的新增文件检查期间接受这一已审阅的独立可视化工具，要求其规范化换行/BOM后的源码SHA256精确等于已审阅版本；不同内容仍停止。旧基线全部源码/输入/产物原始字节SHA256校验保留，其他新增非实验生产文件仍拒绝。新实验manifest仍完整记录该可视化工具的原始字节哈希，结束/续跑照常核验。无需删除工具，不修改旧RPG/PMFA入口或历史manifest。用户2026-09-28日志显示首次13项服务器测试全部通过，但在基线预检处停止，尚未启动真实模型阶段；上传修复后的MPSA脚本及测试后重新预检并使用新版本启动。

入口 `tools/experimental_mpsa_ab.py`，测试 `tests/test_experimental_mpsa_ab.py`。用户明确要求的独立研究实验：新方法的所有区域聚合都使用 MPSA，没有覆盖率权重、SMFA 混合或 SMFA 回退；不修改生产源码、配置或旧 PMFA/RPG/GMP/配对种子脚本。它复用这些旧脚本的预检、固定成员重建、阶段管理、评测核验和逐图混淆导出函数，因此旧文件必须保留。服务器只增加 MPSA 脚本及测试，不覆盖运行中的旧源码。

固定三个变体：phase1={(0,0)}、phase2={(0,0),(7,7)}、phase4={(0,0),(7,0),(0,7),(7,7)}。先调用项目原预处理将 RGB 缩放到 1008×1008 并归一化，再在编码输入上向右/下平移 7 像素；不是把任意尺寸原图平移 7 像素。图像左/上 reflect 填充，右/下裁切，禁止循环平移。每相位独立顺序 FP32 编码，同图所有 mask 共享该相位 token，按16个 mask 分块聚合，不构造 mask×token×channel 大副本、不把四张图堆成 SAM3 batch4。1008/72=14，对应中心坐标14i+7。

mask 保持原图二值坐标，中心取值等价于最近邻缩放再同步平移：floor((center-offset)×原尺寸/1008)，输入画布外为0，不求 coverage。中心落在 mask 内的 token 等权平均→相位内 L2→选定相位等权平均→最终 L2，保留原 trunk/FPN token 处理和1280维输出。非空 mask 无采中点时选距离平移后原始 mask 质心最近的有效中心，距离并列按行优先；即使小目标被右/下裁出画布也应用此确定规则，并记录 nearest_fallback。空 mask 输出0；零特征输出0，非空训练区域输出0则停止。phase1也先后经过同样归一化流程。相位均值/最终均值为0时不换回 SMFA。该方法检验采样相位假设，不预先宣称边界污染必然抵消或±7编码完全等价。

默认只读来源：`experiments/pmfa_fixed_20260924_101553` 已完成 SMFA，以及已清洗提议 `/home/zystu2025/nd/ModuSeg_SAM3/storage/proposals/voc/active_clean_20260928_151808`。核验 cleanup_config.json、原始来源SHA256、1449张清单/尺寸/二值mask/保留score和source_indices、清洗后总提议数。整个实验不调用 clean_proposals，也不重生成提议或伪标签。启动时仅使用旧 SMFA 权重/库复评一次新缓存，混淆矩阵必须与历史 SMFA 完全一致，否则停止查输入或执行差异；不重训 SMFA、不重建 SMFA 库。复评也导出逐图混淆矩阵，后续比较同一缓存、cleanup=null。分类后原实例筛选/NMS仍保留。

复评通过后，重建原 SMFA 过滤后固定区域成员并核对 mask 哈希；不重新过滤。先8张内部训练图片真实四相位探针，然后GPU0/1分片完整提取，两卡每张图均顺序执行四相位，三种变体共用中间结果。缓存只存三套区域向量，不保存完整token。逐图失败保留旧attempt，续跑只跳过校验通过的图片。合并三个新原库后，GPU0顺序训练/全量验证phase1、phase2，GPU1训练/全量验证phase4；三个都预先固定、完整评测，不用最终val挑参数。不变项为v1三层MLP、50轮×100步、seed42、旧内部图片划分、库成员、检索参数、原图输入和FP32标准ViT MLP修正；不复用旧SMFA的MLP作为新方法权重。

签名format_version=3005记录相位集合、坐标映射/边界政策、归一化、FP32执行策略及基础checkpoint SHA256。worker内扩展严格签名校验，保留adapter来源签名验证；不同相位、基础权重和其他算法库不能混用，普通生产入口也会拒绝MPSA签名。不改变生产默认配置。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_mpsa_ab -v
nvidia-smi
df -h .
python -B tools/experimental_mpsa_ab.py --preflight-only

version="mpsa_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_mpsa_ab.py \
    --version "$version" --gpus 0 1 > "logs/${version}_launcher.log" 2>&1 &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

如路径不同，用 `--previous`（完整旧PMFA/SMFA实验目录）及 `--clean-proposals-dir` 显式覆盖。预检只检查数据、哈希、依赖、GPU和磁盘，不执行模型。两卡空闲检查沿用旧实验；OOM停止，不降精度或分辨率。特征提取总计算量明显高于单相位，不能因为一次处理多个mask而声称接近单次前向成本。

产物：`storage/raw_banks/voc/<版本>/{image_cache,phase1,phase2,phase4}`，`storage/mlp_banks/voc/<版本>/train_{1,2,4}`，`weights/sam3_adapter/<版本>/train_{1,2,4}`，`logs/<版本>`，`experiments/<版本>`。后者保存manifest、proposal_policy、成员清单、probe_summary、三个comparison_phaseN.json、summary.json/csv、status及阶段attempt；各eval attempt有per_image_confusion.npz。报告包含21类IoU、全量ID和混淆矩阵、各方法相对SMFA的百分点差、phase4相对phase1差、worker耗时/峰值显存和支持/回退统计；逐图提取诊断另记录相位与phase0的余弦相似度。耗时含诊断和I/O，不是纯算法性能基准。只有全部完成并复核哈希后才写complete汇总；单seed不声称统计显著。

已有版本拒绝覆盖。相同代码/参数/输入使用 `--version 原版本 --resume`；失败阶段新attempt，保留已完成计算，不自动删除旧产物。SIGTERM launcher仅终止自有worker。artifacts.json列自有根目录和只读输入。不用时移除新增MPSA脚本及测试即可，产物由用户按清单另行清理。无服务器连接时只报告本地测试结果，不声称已经启动或取得真实精度。

## 17. 48小时双卡 MPSA 组合队列（2026-09-29）

独立入口 `tools/experimental_mpsa_combo_queue.py`，专用测试 `tests/test_experimental_mpsa_combo_queue.py`。仅新增入口及测试，不修改既有实验/生产模块。依赖原 PMFA、RPG、GMP、GMP种子、MPSA实验脚本，服务器只上传新增两个文件。默认来源为 `experiments/mpsa_20260928_172820` 与 `experiments/gmp_seeds_20260927_194351`，从manifest解析相同SMFA来源。只读复用指定的active_clean_20260928_151808缓存，不重清洗、生成标签或提议，不修改单库结构。

固定A/B=1/2相位硬中心均值，C/D=1/2相位硬中心GMP，E/F=1/2相位硬中心温和门控。两相位固定(0,0)/(7,7)，坐标、padding、空mask、最近中心规则与第16节相同。GMP仅对硬选中的token使用均匀权重1/N，rho=0.1、lambda=rho×tr(S)，FP32 Cholesky和残差检查；门控proxy为原token均值归一化，sim为单位token余弦，a=0.5+0.5 sigmoid((sim-torch.median(sim))/0.1)，不迭代、不定义coverage core。零proxy返回硬中心均值，空mask返回0，非空训练区域最终为0则停止；A至F均无SMFA回退。新C至F签名format_version=3006，A/B保持原3005、参考SMFA保持3001；签名隔离组合/相位/参数/基础权重，MLP仍校验完整来源签名。

seed固定42/123/2026；A/B seed42复用旧模型，其余16次新训MLP。三个seed的旧SMFA模型均只读复用，不训练SMFA。每个新方法固定旧SMFA过滤后的区域成员及内部图片划分，不重新过滤；v1三层MLP50×100步配方不变。先检查历史源码/输入/产物哈希，仅接受第16节已核验的可视化工具例外，不放宽其他生产代码检查。

两卡分片提取训练和val，每张图顺序编码两个相位，同一份token供均值/GMP/门控使用。训练缓存只保存C至F聚合向量；同步计算A/B并逐区域allclose旧原库(atol=rtol=5e-5)。val缓存保存六组及参考SMFA的MLP前向量，绑定图片文件、提议文件、mask顺序/内容及特征空间哈希，不保存完整token。SMFA参考向量维持原批量matmul归约顺序；coverage仅用于该参考，不进入A至F。

真实探针固定4张内部训练图片加2张val图片，检查共享/单mask提取及原MPSA一致性；val缓存落盘后与内存向量精确核对。随后完整提取、合并新原库，使用缓存复评A/B seed42与三个SMFA模型，要求全量混淆矩阵与历史完全一致。通过后才训练新组合。缓存评测通过独立worker内的提取器包装供原推理入口使用，仍执行相同MLP、检索、NMS和像素合成；不得把缓存评测耗时称为端到端编码耗时。

调度按seed分轮：C/D/E/F seed42→A至F seed123→A至F seed2026，每轮固定顺序领取，两卡各最多一个自有worker。单任务为训练加完整验证；可校验跳过已完成阶段。任务级失败最多重试一次(跨续跑累计)，OOM不重试；失败任务最终记入summary，继续其他独立任务。历史复评失败、共享提取失败、缓存/输入/签名/源码错误全局停止。GPU运行时被外部任务占用则该lane等待，另一lane继续，不杀外部进程；预检要求两卡满足空闲条件。

时间预算从首次正式启动的budget.json时间戳算起，默认48小时；到期不领取新任务/新提取图片，已启动训练任务允许完成其验证。不会自动重置预算；48小时之后需显式用更大的 `--max-hours` 配合原版本 `--resume` 才能延长。5 GiB为空闲磁盘保留底线；新运行预检估算四套raw的缓存/合并副本、16套投影库、验证缓存和2 GiB额外余量，再加5 GiB底线。运行中每张提取图片、领取任务和启动worker前检查空间。到期或磁盘不足写deferred并保留结果，不删除历史或自有产物。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_mpsa_combo_queue -v
nvidia-smi
df -h .
python -B tools/experimental_mpsa_combo_queue.py --preflight-only

version="mpsa_combo_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_mpsa_combo_queue.py \
    --version "$version" --gpus 0 1 --max-hours 48 > "logs/${version}_launcher.log" 2>&1 &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

CLI另外提供 `--mpsa-previous`、`--smfa-seeds-previous`、`--clean-proposals-dir` 指定完整历史目录；无自动下载。查看 `experiments/<版本>/status.json`、`progress.json` 和 `summary.json`；逐worker日志在 `logs/<版本>/`，launcher日志在 `logs/<版本>_launcher.log`。失败/预算未耗尽的续跑用相同 `--version 原版本 --resume --max-hours 48`，后台日志须取新名字；延长为自首次启动起96小时可改 `--max-hours 96`。SIGTERM launcher停止自有worker，禁止pkill python；强杀残留锁需先核实原进程已退出才人工处理。

新产物在storage/raw_banks/voc/<版本>/{region_cache,C,D,E,F}、storage/mlp_banks/voc/<版本>/<组_seed>、weights/sam3_adapter/<版本>/<组_seed>及experiments/logs。artifacts.json列自有根和只读输入；无默认自动删除。逐图缓存有完成校验和新attempt；缺失或身份不同不静默重用。report阶段也独立校验，续跑或预算停止不会丢失已经完成的comparison报告。

每个comparison_<组>_<seed>.json包含严格1449张ID、21类IoU、混淆矩阵、相对同seed SMFA的百分点差和逐图文件位置。summary汇总三个seed的均值/样本标准差、完成情况，C−A/D−B/E−A/F−B及两条交互差值；报告缺少seed则明确不完整，不自动选优部署、不声称统计显著。workers按probe/extract/train/eval分别记录耗时及峰值显存，逐图提取诊断记录支持数、最近中心回退、GMP残差/负系数和门控范围。五张关键图片另存预测NPZ及检索类别/分数/间隔、margin筛选、NMS前后、最终合成像素数；只用于诊断、不调参。GT不参与聚合或提议筛选。

本地测试模拟两卡调度、全流程16次新训练、局部重试、全局失败、预算/磁盘边界、取消自有进程、签名与缓存错配、已有版本拒绝及完整续跑。张量测试需要PyTorch；真实SAM3/FAISS/历史模型复评只能在服务器确认，不以模拟结果代替真实精度。

## 18. SMFA 固定聚合的 MLP 结构对照（2026-10-01）

独立入口 `tools/experimental_smfa_mlp_queue.py`，专用测试 `tests/test_experimental_smfa_mlp_queue.py`。服务器上传这两个新增文件；本手册同步记录使用方法。保留全部旧实验依赖，不覆盖旧入口或生产源码。所有组都用原版 SMFA，不加入 GMP、MPSA、门控或 mask 分数条件。

默认 `--previous experiments/mpsa_combo_20260929_105543`，仅把已完成组合实验作为**历史 SMFA 权重和 SMFA 查询向量的来源**；只读其中 SMFA 列，完全不使用 A–F 的表示。该目录必须 status=complete，三个 replay_SMFA 阶段完整，历史源码、输入、引用库/权重哈希和 1449 张逐图向量缓存全部通过核验。没有完整旧缓存就停止，不自动生成标签、提议或重提 token。已有 clean 提议固定为 active_clean_20260928_151808，不再次清洗。

固定八组，每组三个 seed=42/123/2026：

| 组别 | Linear 层数 | 隐藏维度 | 变化 | Adapter 参数量 |
|---|---:|---:|---|---:|
| d3 | 3 | 512 | 历史基线，只读复用权重并复评 | 1,049,856 |
| d2 | 2 | 512 | 普通两层，重新训练 | 787,200 |
| d4 | 4 | 512 | 普通四层 | 1,312,512 |
| d5 | 5 | 512 | 普通五层 | 1,575,168 |
| res4 | 4 | 512 | 四层，隐藏层残差 | 1,312,512 |
| res5 | 5 | 512 | 五层，隐藏层残差 | 1,575,168 |
| ln3 | 3 | 512 | 普通三层，每个隐藏 Linear 后加 LayerNorm | 1,051,904 |
| wide3 | 3 | 704 | 宽三层，与普通五层参数量相差约 0.22% | 1,578,624 |

参数量不含训练用分类头。所有模型输入 1280、输出 256，输入/输出 L2，GELU 激活，无新增 Dropout。普通模型为 Linear→GELU 重复，最后 Linear→L2；ln3 为隐藏 Linear→LayerNorm→GELU。残差组先 Linear(1280,512)→GELU，中间每个等宽层采用 h←h+GELU(Linear(h))，最后 Linear(512,256)→L2；没有添加 LayerNorm，残差本身不增加参数或额外投影。层数均指 Linear 数。

共21次新 MLP 训练，不重训三个 d3 历史模型。首先在当前环境重新执行三个 d3 的缓存全量评测，混淆矩阵必须与历史完全一致，才能领取新训练。随后按 seed42→123→2026 分轮；每轮按 d2/d4/d5/res4/res5/ln3/wide3 排队，GPU0/1各最多一个自有worker。每项训练＋完整1449图评测。固定原 SMFA 过滤后库成员、内部训练/验证图片划分、v1损失和选模、50轮×100步、学习率0.001、类均衡8×4及所有检索/像素合成参数。只改结构与相应参数量；不根据VOC val临时扩大搜索。

复用历史1280维训练库和SMFA查询缓存，因此无新的SAM3编码和原库提取。仍新建每组/seed匹配的MLP库和权重。worker内临时扩展4/5层CLI及模型构造/加载，完整架构写进checkpoint和adapter签名(type=smfa_mlp_ablation_v1)，保留来源签名与权重SHA256验证；不同架构/权重不能混用，普通生产加载器拒绝实验架构。FP32、TF32关闭，历史特征精度签名不变。

预检与启动：

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_smfa_mlp_queue -v
nvidia-smi
df -h .
python -B tools/experimental_smfa_mlp_queue.py \
  --previous experiments/mpsa_combo_20260929_105543 --preflight-only

version="smfa_mlp_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_smfa_mlp_queue.py \
    --previous experiments/mpsa_combo_20260929_105543 \
    --version "$version" --gpus 0 1 --max-hours 48 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

如果历史组合版本不同，预检和启动同时修改 --previous。必须指向完整的服务器原实验目录，不能使用只有拷贝summary/comparison的文件夹。--clean-proposals-dir可显式填写，但必须与该历史复评使用的缓存路径及身份相同。

查看状态与结果：

```bash
cat "experiments/${version}/status.json"
cat "experiments/${version}/summary.md"
ls "logs/${version}/"
```

退出终端后，先手工把 version 设为实际版本名，再执行以上命令。逐worker日志在logs/<版本>/；summary.json包含每组均值/样本标准差、完整性、对三层基线配对差值，以及res4−d4、res5−d5、ln3−d3、d5−wide3等预定对比。comparison_<组>_<seed>.json包含全量ID、21类IoU、混淆矩阵和逐图矩阵位置。耗时为训练或缓存评测耗时，不含历史SAM3提特征时间；workers报告峰值显存。不自动宣称统计显著、不自动部署最高分模型。

产物只新增到 experiments/<版本>、logs/<版本>、weights/sam3_adapter/<版本>/<组_seed>/attempt_NNN、storage/mlp_banks/voc/<版本>/<组_seed>/attempt_NNN。artifacts.json列自有目录和只读来源；不新增原始库、不自动删除。磁盘预检按21套MLP库及2GiB额外余量估算，并保留5GiB底线；领取任务/启动worker前再次检查。预检要求两卡空闲，运行中外部占卡则等待，不终止外部进程。

默认自首次正式启动计48小时，截止后不领取新任务；已领取的训练/验证允许完成。预算耗尽写deferred，保留断点。普通局部任务失败最多重试一次，OOM不重试；共享输入/缓存/签名或历史复评出错全局停止。SIGTERM/SIGINT只终止本启动器启动的worker。已有版本无--resume拒绝复用，有--resume也必须身份一致、产物校验通过；失败使用新attempt。

续跑示例（时间预算改为自首次启动起96小时；不需要延长时仍填48）：

```bash
version="填写上次实际的smfa_mlp版本"
resume_log="logs/${version}_resume_$(date +%Y%m%d_%H%M%S).log"
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_smfa_mlp_queue.py \
    --previous experiments/mpsa_combo_20260929_105543 \
    --version "$version" --gpus 0 1 --max-hours 96 --resume \
    > "$resume_log" 2>&1 < /dev/null &
  echo "恢复启动器 PID：$!"
)
tail -f "$resume_log"
```

已耗尽两次失败额度或曾OOM的任务不会因反复--resume无限重试。需要改代码/结构/参数时用新版本。强杀残留launcher.lock必须先人工确认旧启动器和worker都已退出再处理，不自动破锁。

本地模拟检查不代表真实CUDA训练。服务器须运行上述unittest，其中张量测试核对三层初始化/输出等价、所有结构参数量/梯度/归一化、checkpoint往返及错配拒绝；随后真实三层历史复评仍是正式实验前置条件。

## 19. 单相位硬中心 GMP＋两层 MLP（2026-10-02）

用户要求结合已完成实验验证组合是否更好。独立入口 `tools/experimental_gmp_mlp2.py`，测试 `tests/test_experimental_gmp_mlp2.py`；只新增这两个文件及更新手册，旧源码和入口不修改。依赖前述旧实验脚本（包括 SMFA MLP 队列），服务器保留原文件。此实验补齐 SMFA/GMP × 两层/三层的第四格，不扩大参数搜索，不改变主线默认配置。

候选固定为原组合实验 C 的单相位(0,0)、硬中心等权 token、GMP rho=0.1，再接普通两层1280→512→GELU→256（实际为Linear(1280,512)→GELU→Linear(512,256)，输入/输出L2）。没有覆盖率权重、双相位、门控、残差、LayerNorm或mask分数条件。复用原C组过滤后固定成员的原始库和1449图查询向量，不重新过滤、提特征、生成伪标签或清洗提议。固定1008、FP32/关闭TF32、v1配方50×100、图片划分和原检索参数。

默认来源：`--combo-previous experiments/mpsa_combo_20260929_105543` 和 `--mlp-previous experiments/smfa_mlp_20261001_221703`。两个目录必须完整且身份一致，不能以仅拷贝summary的文件夹代替。先核验历史源码、输入、基础权重、库成员、划分、缓存及签名，随后复评GMP＋三层和SMFA＋两层的三个seed，共6次全量评测；与历史混淆矩阵完全一致后才执行3次新训练（seed42/123/2026）。SMFA＋三层保留已核验的历史参考。三套新权重都基于GMP原始特征训练，不将SMFA的两层权重直接用于GMP。

模型加载和两层参数替换只在独立worker生效。两层adapter复用第18节架构标签，同时严格绑定C组GMP来源签名和基础权重；建库与查询均匹配，禁止通过关闭签名检查混用。父进程复用第18节进程监督器，将其内存中的worker入口指向本脚本，结束时恢复；磁盘旧脚本不改写。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_gmp_mlp2 tests.test_experimental_smfa_mlp_queue -v
nvidia-smi
df -h .
python -B tools/experimental_gmp_mlp2.py --preflight-only

version="gmp_mlp2_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_gmp_mlp2.py \
    --version "$version" --gpus 0 1 --max-hours 24 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

路径不同时，预检与启动均显式指定--combo-previous、--mlp-previous。--clean-proposals-dir默认用户指定的active_clean_20260928_151808，必须与历史一致。

双卡各最多一个自有worker，预检检查占用，运行中外部占卡等待，不杀外部任务。24小时预算从首次正式运行开始，截止不领取新任务，已领取的训练与完整评测允许完成。空间估算为3套导出库加1GiB余量，另保留5GiB底线。此次小批次失败不自动重试，保留attempt和其他已完成结果；局部训练/评测失败允许其它seed继续，历史复评、共享输入/签名/缓存错误全局停止。OOM不自动降精度或配置。

`experiments/<版本>/summary.json`、`summary.md`汇总四格平均值和样本标准差；`comparison_seed42/123/2026.json`保存全量ID、逐类IoU、混淆矩阵和逐图矩阵路径。报告给出GMP2−SMFA2、GMP2−GMP3、GMP2−SMFA3，以及交互项(GMP2−GMP3)−(SMFA2−SMFA3)。仅三个seed齐全才给出是否超过这三种参考配置的平均成绩/单次最高成绩，部分结果不宣称创新高。当前参考GMP3均值85.5544%、最高单次85.7924%；SMFA2均值85.5462%。这是基于已看过VOC val结果选择的开发实验，不作为独立测试或统计显著性证据。

新权重在weights/sam3_adapter/<版本>/<seed>/attempt_NNN，导出库在storage/mlp_banks/voc/<版本>/<seed>/attempt_NNN，日志在logs/<版本>，无新增raw库。artifacts.json列自有及只读路径。SIGTERM/SIGINT只停止本启动器的worker。已有版本拒绝覆盖；人工确认问题解决后，使用原版本加--resume跳过校验通过的阶段并用新attempt重试未完成阶段。代码/输入/参数变化必须用新版本。

查看：`cat experiments/$version/status.json`、`cat experiments/$version/summary.md`。后台续跑用相同两个历史来源、`--version 原版本 --resume --max-hours 48`，另取新的launcher日志名；48表示自首次启动起48小时，不是额外增加48小时。不自动删除历史数据或部署模型。

## 20. SAM3 分数：训练加权 × 推理融合（2026-10-02）

独立入口 `tools/experimental_mask_score_queue.py`，专用测试 `tests/test_experimental_mask_score_queue.py`。不修改生产代码或旧实验脚本。服务器须保留前述实验依赖入口。本轮以 `experiments/gmp_mlp2_20261002_141439` 的单相位硬中心 GMP rho=0.1、普通两层 MLP 为基线，固定 seed42/123/2026、原库成员、内部图片划分、v1 50×100 配方。只新增三个加权 CE 模型，不重新提取区域特征、不建 raw 库、不重新过滤。

实验为“历史原训练/置信度加权训练” × “推理分数融合 alpha=0、0.1、0.2、0.4” × 三个 seed，共24个全量1449图评测单元。历史原训练 alpha=0.2 的三个单元先复评，要求历史总混淆矩阵及逐图矩阵完全一致。然后完成其余9个历史模型的融合评测，再补采训练分数，最后训练三个新模型并各评测四个 alpha。不根据 val 成绩追加参数。

推理公式仍为 `(1-alpha)*retrieval_score + alpha*proposal_score`，融合进入原类别内 NMS、排序和像素合成。检索类别与 margin 筛选不变。原实验已经使用 alpha=0.2；alpha=0 仅关闭最终融合，不撤销此前基于分数的提议清理。固定读取 `active_clean_20260928_151808`，不生成/再清洗验证提议。评测复用原 C 组的查询向量，耗时不包含历史 SAM3 编码。

训练置信度必须补采：从历史 PNG 目录的 `generation_*.json` 恢复 Active 设置、词表、类别顺序、图片与权重来源；缺失或冲突时只阻断加权训练分支，原模型推理对照仍可完成。不使用当前默认值猜测历史参数。原 CLI 未暴露的 max_masks=0 绑定到已核验的历史源码默认值。仅支持本实验历史原图、FP32、无额外标签尺寸转换的生成协议；使用标准 ViT MLP 修正，关闭 BF16/TF32。

补采对完整训练划分双卡分片，每张图在内存重放“同类合并、先允许类别优先”，与旧 PNG 逐像素比较。任何不一致保存 image/pixel-count/shape 诊断并停止该分支，不覆盖原图、不重建库、不用近似匹配。分数标记为重放获得，不能称为当时未保存的历史分数。Active 分数实际是 sigmoid(pred_logits) × sigmoid(presence_logit_dec)，不是经过校准的 mask IoU。

同一个旧前景区域内，每个像素取覆盖它的同类候选最高分，再对区域求均值得到 q；前景像素无候选覆盖报错。前景训练权重 `w=0.5+0.5*q`，背景无 SAM3 实例分数，记录 q=null、w=1。仅 CE 改为 `sum(w*ce)/sum(w)`；SupCon、平衡采样、优化器、内部验证指标及 checkpoint 选择不变。训练报告记录各类分数/权重分布与采样权重贡献。分数按原库类别、样本 ID、mask 哈希对齐，错序不靠位置猜测。新 adapter 保留原 GMP 来源签名，额外绑定加权公式及分数缓存身份；主线不能直接误加载该实验模型。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_mask_score_queue -v
nvidia-smi
df -h .
python -B tools/experimental_mask_score_queue.py \
  --previous experiments/gmp_mlp2_20261002_141439 --preflight-only
```

预检不运行 SAM3，不代表真实重放已通过；若显示 `[Score branch]` 元数据问题，方案二会阻断，方案三仍可运行。历史目录须是服务器完整原实验目录，不能指向只有拷贝 summary 的目录。确认本地测试和服务器预检后，后台启动：

```bash
version="mask_score_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_mask_score_queue.py \
    --previous experiments/gmp_mlp2_20261002_141439 \
    --version "$version" --gpus 0 1 --max-hours 48 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

每张物理卡最多一个自有 worker，按现有阈值（显存<=1000MiB、利用率<=10%）检查，运行中占用超阈值则等待，不终止其他进程。48小时从首次正式启动计时；到期停止领取图片/任务，已领取训练及其四个评测允许完成。新空间按三套256维导出库、分数sidecar及1GiB余量估算，另保留5GiB底线，运行中持续检查。SIGTERM/SIGINT只终止自有worker。

新增目录：`storage/pseudo_labels/voc/<版本>/score_sidecars`（逐图分数、成员映射和校验，不保存候选mask或token）、`storage/mlp_banks/voc/<版本>/<seed>`、`weights/sam3_adapter/<版本>/<seed>`、`experiments/<版本>`、`logs/<版本>`。`artifacts.json` 区分自有及只读路径，不自动删除。`--train-score-cache` 仅接受完整、代码/输入/配方/区域身份一致的sidecar目录；直接用旧 PNG 或验证提议目录不行。

`summary.json`、`summary.md` 汇总八组结果与预先指定的训练、推理及交互对比；`comparison_<original或weighted>_<seed>_a<0或1或2或4>.json` 包含全量ID、逐类IoU、混淆矩阵及逐图NPZ路径。少于三个seed的组别不报告完整均值优势。`workers` 分开记录补采、训练、缓存评测耗时与峰值显存；`score_status` 明确说明待补采、完成或分支阻断。不宣称统计显著，不自动部署最高分配置。

```bash
cat "experiments/${version}/status.json"
cat "experiments/${version}/summary.md"
```

局部训练/评测失败保留attempt，其他独立任务继续；共享输入损坏或历史复评不一致全局停止。无自动重试或OOM降配置，人工解决问题后用同一版本续跑；更改代码、输入或配方必须新版本。失败attempt保留，重试创建新attempt，已校验的完整单元跳过。停止后先确认旧启动器及worker已退出；不自动解除残留启动锁。

```bash
version="填写上次实际版本名"
resume_log="logs/${version}_resume_$(date +%Y%m%d_%H%M%S).log"
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_mask_score_queue.py \
    --previous experiments/gmp_mlp2_20261002_141439 \
    --version "$version" --gpus 0 1 --max-hours 96 --resume \
    > "$resume_log" 2>&1 < /dev/null &
  echo "恢复启动器 PID：$!"
)
tail -f "$resume_log"
```

96小时表示从首次正式启动起的总预算，不是额外96小时。外部sidecar路径若首次设置，续跑也须提供相同参数。本地缺少torch时相关张量测试会明确跳过；服务器必须执行它们，并通过真实PNG探针和历史全量复评，才能得到有效实验结论。

## 21. SAM3 解码器 query 特征可行性探针（2026-10-03）

新增独立入口 `tools/experimental_query_probe.py` 和 `tests/test_experimental_query_probe.py`，不修改生产代码或旧实验。取 `sam3_image.py` 输出的 `out['queries']`（最后一层解码后的目标查询，不是初始可学习查询），在原训练图片的固定内部划分上进行前景区域检索诊断。此轮不训练MLP、不处理背景分类、不新增VOC val预测，也不自动启动全量实验；结果不是语义分割mIoU。预检复用历史完整性检查，因此会读取历史验证指标及校验原缓存。

基于 `experiments/gmp_mlp2_20261002_141439`。每个前景类在原内部训练集最多选8张、内部验证集最多选4张；以SHA256("42:"+图片ID)排序固定选择并去重，最多240张。同一图片的所有保留前景区域都纳入。每类至少需要两张gallery图片和一张heldout图片，否则停止。划分、选中ID、源码及输入哈希写入manifest。

在每张图上重放原训练Active配方，严格复现原类别PNG。只在独立worker内包装 `_ground`/`_filter`，按类别与query槽位携带向量，通过逐项比较保证区域清理、NMS、稳定排序、max_masks截断后的mask与query一致。查询维度从真实输出读取，数值非有限、零向量或对应关系错误立即停止。使用FP32、标准ViT MLP修正，关闭BF16/TF32。

训练伪标签合并/拆分区域的query由同类候选构造：区域中每个像素归属覆盖它且分数最高的候选，同分按原候选顺序；对候选L2 query按归属像素数加权平均后归一化。不将背景编造为query，也不从GT选候选。区域mask哈希与原成员计划核对，GMP原向量复用原C组训练缓存。

内部验证图片分别执行两种提示条件：conditional沿用已知图像类别，仅用于说明提示条件的影响；full使用完整20类词表，不传该图片类别。full候选与旧伪标签区域按**不看类别**的最高IoU匹配，固定IoU>=0.5，未匹配项明确报告。它仍使用伪标签区域几何做诊断，不能当作真实部署预测。query变化同时包含候选几何、提示及上下文变化，余弦变化不能独立归因于文本。

固定top5余弦近邻投票，比较原GMP、query、两个分支各L2后拼接（等权）的原始向量，不额外训练投影层。在同一个full匹配子集比较三者，单列coverage/逐类coverage/未匹配数量，避免通过删掉难例制造优势。同时报告对应候选的原提示类别作为直接对照。结果包含各类准确率、存在样本类别上的宏平均及类别数、GMP/query错误交集、条件变化余弦分位数、实际query维度、耗时及峰值显存。准确率对照的是伪标签，不是真值。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_query_probe -v
nvidia-smi
df -h .
python -B tools/experimental_query_probe.py --preflight-only
```

预检通过后启动双卡后台探针（先一张gallery和一张heldout通过真实前向，再双卡分片）：

```bash
version="query_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_query_probe.py \
    --previous experiments/gmp_mlp2_20261002_141439 \
    --version "$version" --gpus 0 1 --max-hours 12 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

缓存只写入 `storage/raw_banks/voc/<版本>/query_probe`，其中每图保存聚合后的query/GMP向量、区域/候选对应关系及校验，不保存完整token网格或全量mask。配置、status、summary.json、summary.md和产物清单放 `experiments/<版本>`，日志放 `logs/<版本>`，不新增权重。读旧库不改旧库，不重清理验证提议。

磁盘预留5GiB底线，预检另要求1GiB探针余量。两卡各最多一个worker，外部占用超过既有阈值则等待，不杀其他任务。截止后不再领取图片，当前图片允许完成；默认12小时从首次正式启动计算。错误或OOM停止，不自动降精度或替换候选；SIGTERM/SIGINT只停止自有worker。逐图原子提交可续跑，不覆盖完成缓存；新代码/设置必须新版本。

查看 `cat experiments/$version/summary.md`。续跑使用原 `--version` 加 `--resume --max-hours 24`，另取新的后台日志名；24为首次启动起总预算。运行完成先提供 `summary.json`、`status.json`、`manifest.json`，据覆盖率、提示依赖和错误互补结果决定是否开展背景表示及全量新MLP实验。本地无torch时张量槽位测试明确跳过，服务器需执行该测试。

## 22. VOC 误差拆解与真值辅助诊断（2026-10-03）

独立入口 `tools/experimental_error_oracles.py`，测试 `tests/test_experimental_error_oracles.py`。基线固定为 `experiments/gmp_mlp2_20261002_141439` 中普通训练的单相位硬中心 GMP rho=0.1＋两层 MLP，三个 seed42/123/2026，融合权重0.2。不是后续分数加权模型。不修改生产代码、旧实验或缓存，不生成标签、提议、特征库或权重。

只读复用已清洗提议和 C 组查询向量，通过现有严格签名、adapter、历史产物校验后，双卡每卡一个worker执行三个完整1449图评测。每个seed必须逐项复现历史总混淆矩阵和逐图混淆矩阵，且诊断内的baseline矩阵必须相同，才能提交该阶段。GT只在获得原模型预测后进入诊断，不进入特征、检索、选模或训练。

固定诊断口径：

- baseline：原检索类别、margin、NMS、评分和像素覆盖。
- class_selected：保留最终候选成员与顺序，每个mask改为有效GT像素多数类别（包含背景，平票取最小类ID；255不计），不重新NMS。因此隔离的是最终候选上的类别替换，不能恢复早先被过滤的候选。
- pixel_selected：保留最终mask及预测类别；逐像素只要任一候选提供正确类别，就采用该类别，否则保留baseline；允许背景拒识。这是乐观的像素归属oracle，不保证能用一个全局mask排序实现。
- class_all_proposals：所有已清洗候选按SAM3分数稳定排序，GT多数类别，跳过分类margin及分类后NMS。与class_selected候选集合不同，差值不可直接视为纯分类收益。
- class_and_pixel_all：上述候选和GT多数类别，再允许逐像素正确归属。
- best_region_fixed_order：VOC语义GT前景按类别和四连通域划分区域，每区选择忽略255后IoU最大的原候选（同分取缓存首项，零IoU不选）；以类别/连通域固定顺序合成。语义连通域不是实例GT；相邻同类实例可能合并。
- best_region_pixel：最佳区域候选额外允许逐像素正确归属。属于复合oracle，不能作为可部署精度或严格的全局最优mIoU。

还报告每区域最佳IoU、IoU>=0.5/0.75的区域覆盖率、逐类区域统计、baseline前景FP（GT背景预测前景）、FN（GT前景预测背景）、前景类别混淆。三者构成有效像素错误总量。边界用有效GT相邻异类像素作为种子，以四邻域膨胀3次；忽略255，不把图像边框本身当边界；分别保存边界/内部混淆矩阵及错误数。所有差值不可直接相加，oracle不保证提高mIoU。

历史提议读取API只有mask与score，缺少已核验的原提示类别来源，所以SAM3原始类别直接对照明确标记unavailable，本入口不重放SAM3、不猜类别。需另行恢复有来源证明且与清洗mask一一对应的类别记录后才能做该对照。这不阻断上述三类oracle。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_error_oracles -v
nvidia-smi
df -h .
python -B tools/experimental_error_oracles.py --preflight-only

version="error_oracles_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_error_oracles.py \
    --previous experiments/gmp_mlp2_20261002_141439 \
    --version "$version" --gpus 0 1 --max-hours 24 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

需要服务器已有torch、scipy及原项目依赖。仅上传本次新脚本、测试和手册，不替换正在运行任务的旧代码。预检核验历史/缓存，要求至少6GiB可用（1GiB余量＋5GiB底线）。运行中逐图检查5GiB底线，空间不足保留attempt；不是逐图续跑，恢复以完整seed阶段为单位。24小时从首次正式启动算，到期不领取新seed，已开始的seed允许完成。外部显存/利用率超过既有阈值则等待；SIGTERM只终止本启动器worker，不杀其他任务。不自动重试、降配置或删除数据。

结果在 `experiments/<version>/summary.json`、`summary.md`、`status.json`、`manifest.json`。各 `seed_<seed>/attempt_NNN/diagnostics.json` 含逐类IoU（顺序由classes给出）、总矩阵、逐图误差与区域IoU，`oracle_per_image.npz` 含完整有序ID及各诊断逐图矩阵；旧流程同时保存baseline metrics和逐图矩阵。workers的queue_worker.json含耗时和显存。artifacts.json列自有目录（experiments和logs），只读历史不删除。全部三个seed通过才写完整summary；失败保留完成阶段和attempt，status列completed。

```bash
cat "experiments/$version/status.json"
cat "experiments/$version/summary.md"
# 确认旧启动器已退出后，续跑原版本；24是总预算，不是新增时长。
resume_log="logs/${version}_resume_$(date +%Y%m%d_%H%M%S).log"
( set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_error_oracles.py \
    --version "$version" --gpus 0 1 --max-hours 48 --resume \
    > "$resume_log" 2>&1 < /dev/null &
  echo "恢复启动器 PID：$!"
)
```

代码、输入或参数改变须新版本；manifest身份和完成产物校验失败不绕过。没有服务器连接时本地测试不能代替真实三seed复评，不能声称已获得oracle结果。

## 23. 后续实验邮件通知约定（2026-10-03）

用户已授权后续实验结束后通知 2858372757@qq.com。新增公共工具 `project_utils/experiment_notify.py`，测试 `tests/test_experiment_notify.py`。后续新实验入口必须在父启动器写入最终status/summary之后调用 `notify_experiment(version, status, result_dir, seconds)`，完成、失败、截止暂停或中断均应通知；只由父进程调用一次，不让worker重复发信。预检不发信。邮件失败不能覆盖实验异常或改变实验完成状态。SIGKILL/断电无法保证发送。

SMTP使用QQ邮箱TLS连接。按用户2026-10-03明确要求，通知工具内置授权码默认值；可通过 `MODUSEG_SMTP_PASSWORD` 覆盖，设为空字符串可禁用通知。授权码不得输出到manifest、日志或邮件正文，也不记录完整环境变量。默认发件人及收件人2858372757@qq.com；可用MODUSEG_SMTP_USER和MODUSEG_NOTIFY_TO覆盖。历史脚本和已运行任务不改动；公共工具尚不会自动给没有接入它的旧入口发信。

默认无需输入授权码。如需临时覆盖内置值，可在当前shell输入，nohup子进程继承环境：

```bash
read -rsp 'QQ SMTP授权码: ' MODUSEG_SMTP_PASSWORD; echo
export MODUSEG_SMTP_PASSWORD
```

不要把授权码明文放入shell命令历史。可单独调用通知工具（status必须真实）：

```bash
python -B -m project_utils.experiment_notify --version 实际版本 \
  --status complete --result-dir experiments/实际版本
```

仅记录SMTP接受状态，不保证收件箱投递成功；网络/认证失败打印无敏感信息的提示，不自动重试避免重复邮件。此功能不上传数据、权重或完整日志。当前仅完成本地模拟SMTP测试，没有服务器实际发信验证。


## 24. 错误归因第二轮（2026-10-03）

入口 `tools/experimental_error_attribution.py`，测试 `tests/test_experimental_error_attribution.py`。依赖第22节脚本，但不修改它。基线仍为普通GMP＋两层MLP、三个seed、融合0.2，复用原向量、权重、已清洗提议；不重新训练、生成或清洗。原始提议目录从cleanup_config.json的input_dir恢复，经原历史来源校验并新增完整文件哈希，原提议只用于覆盖统计。

新增background_clear（仅真值背景误报清零）、foreground_conflict（只修复预测为其他前景类且有正确类别最终mask覆盖的像素）、foreground_recover（只修复预测背景且有正确类别最终mask覆盖的像素）、pixel_combined。前三者在baseline上分别执行，改动像素集合互斥，但mIoU增益仍不可相加。foreground_recover在当前first-wins实现中应接近零，可作为流程一致性检查。GT仅用于诊断。

每个类别的前景→背景像素按首次缺失分层：原始提议无覆盖、清洗后无覆盖、无有效特征候选覆盖、没有预测正确类别的候选覆盖、正确类别候选被margin筛除、被NMS筛除、最终覆盖阶段。按此顺序互斥分配并验证总数等于FN。不代表严格因果；清洗改变几何时前后覆盖不一定嵌套。“无正确类别”包括背景预测和其他类别，不等同于原始提议漏检。

每张图保存全部清洗后候选的索引、原始source_index、mask哈希、分数、GT类别像素直方图、检索类别/分数/margin及实际检索返回的前10个近邻（bank全局行号、类别、相似度）。近邻来自原检索搜索调用，不替换检索实现；不是完整的类别均衡分数解释，也未将bank行号假称为训练图片ID。发现椅子真值被预测为沙发时，保存原图与紫色错误像素标记的并排JPG，位于attempt/ chair_sofa目录。GT不参与候选选择或排序。

边界改为直接有效类别转变带与255忽略区域附近有效像素的并集，四邻域膨胀3次，255始终不评测。名为boundary_proxy，不是标准Boundary IoU；大块忽略区也可能进入邻近带。它解决旧版跨忽略带漏计，但不能当精确物体边界。边界/内部指标不再计算与全图mIoU的提升。区域报告补充>=1/100/300/1000面积分层和面积加权R50；语义连通区域不是实例。

每seed必须历史总矩阵、逐图矩阵、诊断baseline全部一致才提交。双卡每卡一个worker；截止不领新seed，完整seed允许完成；空间预检6GiB、运行逐图保留5GiB。续跑单位是seed，不是图片，失败保留attempt。代码/输入变化用新版本。产物只在experiments/<version>和logs/<version>，不自动删除。复用第22节源码身份、启动锁、子进程终止和严格续跑规则。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_error_attribution tests.test_experimental_error_oracles tests.test_experiment_notify -v
nvidia-smi
df -h .
python -B tools/experimental_error_attribution.py --preflight-only
version="error_attribution_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_error_attribution.py \
    --version "$version" --gpus 0 1 --max-hours 24 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

summary.json含三seed指标、逐类FN去向、面积分层、完整1449 IDs；各seed/attempt_NNN/diagnostics.json含逐图候选和近邻；oracle_per_image.npz含逐图诊断矩阵。运行完成提供summary.json、status.json、manifest.json，必要时补seed42 diagnostics.json。summary.md为简表，细项在JSON中。

父进程正式运行结束后，最终状态落盘再调用第23节邮件工具；完成、失败、截止暂停、SIGINT/SIGTERM中断均尝试发送，预检不发送，worker不发送。默认使用用户指定QQ凭据，需将project_utils/experiment_notify.py一并上传。邮件失败不更改实验状态，断电/SIGKILL无法保证通知。初始化/预检失败发生在正式运行前时不发送。当前仅模拟邮件测试，未验证服务器网络投递。

续跑：旧启动器退出后，用原版本加 `--resume --max-hours 48`，nohup日志取新名字；48是首次启动起总预算。三seed实际历史复评只能在服务器验证，本地单元测试不能替代。

第24节补充修复：历史combo的生产源码集合早于邮件工具，直接放入project_utils会触发Unverified new production code。新入口仅在历史核验期间排除**固定路径且SHA256匹配已审阅版本**的experiment_notify.py（校验时统一CRLF/LF）；不会放行其他新增生产文件，历史记录内文件仍逐项校验。核验结束立即恢复完整源码枚举，新manifest和运行结束校验仍包含邮件工具。邮件工具内容改变会拒绝，需对应更新审核后的入口，不关闭签名校验。不修改历史入口或manifest。上传新版experimental_error_attribution.py及专用测试；邮件工具须为配套版本，更新代码后正式运行取新版本。

## 25. 清洗逐规则审计＋检索完整分数分解（2026-10-03）

新增 `tools/experimental_mechanism_audit.py`、`tests/test_experimental_mechanism_audit.py`。参考外部建议，本批先做其无需重新编码的两个机制验证；**不包含patch原型验证、语义救援、熵门控推理或背景聚类**。历史缓存仅为区域向量，没有完整patch tokens，后续前景验证必须重新提取SAM3特征，不能声称零编码成本。

阶段1在CPU按1449图读取原始提议，只在内存重放清洗。原配置输出mask、score、source_indices必须与用户指定clean缓存逐项完全一致。固定七组：baseline、min_score=0、min_mask_area=0、min_island_area=0、max_hole_area=0（不填洞）、max_hole_fraction=1（取消比例限制但保留32像素绝对限制）、nms_iou=1。一次仅改一项，不根据val调整。将max_hole_fraction设为0也会禁用填洞，并非取消该约束，因此不增加重复组。

每图每配置输出候选数、清理操作计数、逐类别GT像素总数/候选并集覆盖/相对baseline新增与丢失覆盖，排除255。背景覆盖不等于最终模型误报；恢复前景覆盖也不保证mIoU提升。无新mask缓存落盘，不改变生产清理配方，不自动选择最好配置。

阶段2双卡调度普通GMP2的seed42/123/2026原模型，复用原查询向量，只记录不修改分类。截取实际检索搜索结果，为每个候选导出全部类别的local、center、fused、邻居支持数和三种top1。局部分数严格保留生产代码的规则：某类没有搜索邻居时以center回退，所以local_top1不是完全无中心的消融。逐候选核对fused类别、分数和margin，再核对全量历史总矩阵、逐图矩阵与诊断baseline。不能将这些候选排名当成替代模型的全量mIoU。

使用GT多数类别（含背景）仅作事后诊断，报告local正确→final错误和反向翻转，以及chair→sofa候选全分数。少数类纯净度可通过gt_histogram检查，不把多数类当无噪声实例标签。所有候选分数保存 `seed_<seed>/attempt_NNN/all_candidate_scores.json`，summary收录翻转计数和chair候选。仍未将bank行号映射回训练区域，不据此宣称背景污染。

默认来源、完整性、签名、两卡占用、5GiB运行底线、6GiB预检、启动锁、阶段续跑、邮件通知沿用第24节。依赖旧error_oracles/error_attribution及其历史工具，必须保留。邮件工具固定哈希例外同第24节，不改历史入口。预检不发信。正式阶段结束通知，网络失败不改变结果；当前无实际SMTP投递验证。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_mechanism_audit -v
nvidia-smi
df -h .
python -B tools/experimental_mechanism_audit.py --preflight-only
version="mechanism_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_mechanism_audit.py \
    --version "$version" --gpus 0 1 --max-hours 24 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

CPU清洗审计期间GPU空闲正常；之后双卡各一个评测worker。24小时从首次正式启动计，清洗阶段逐图检查截止，评测阶段截止不领新seed，已有seed允许完成。清洗中断后保留attempt，续跑重做整个清洗阶段；完成阶段校验通过则跳过，不是逐图续跑。源码或参数变化用新版本。

产物位于experiments/<version>及logs/<version>。cleanup_audit/attempt_NNN/coverage.jsonl保存逐图七组，cleanup_summary.json为总表；summary.json含七组覆盖统计和三seed检索分解，status、manifest和artifacts.json保留完整身份与路径。后续先提供summary.json、status.json、manifest.json。后台续跑使用原version加--resume --max-hours 48及新日志名；不自动删除历史数据。

本地数值和覆盖测试4项通过，OpenCV相关1项因本地缺依赖跳过；服务器必须执行全部测试并通过真实clean缓存一致性及历史评测检查。无服务器连接，尚未运行真实实验。

## 26. 类别保持的patch前景验证（2026-10-03）

入口 `tools/experimental_foreground_verifier.py`，测试 `tests/test_experimental_foreground_verifier.py`。这轮是实际方法对照，不是oracle。保留普通GMP＋两层MLP、原库成员、检索评分、SAM3融合0.2与已清洗提议。固定三个seed，3次baseline历史复评＋6次前景验证全量评测；不新增MLP训练，不改生产代码或旧实验，不包含候选救援。

两组固定配置：每前景类4个球面原型，背景分别1或8个；温度0.1，删除证据D<-0.05的位置。D=0.1*logmeanexp(cos(f,fg)/0.1)-0.1*logmeanexp(cos(f,bg)/0.1)。以mean而非sum消除原型数量造成的额外logK偏置。不能将D解释为校准概率。两组固定报告，不根据VOC val追加搜索或部署。

仅在历史内部train_images中，以SHA256('42:'+ID)固定选每类最多16图、去重（最多320图），类别成员来源为历史membership。内部validation_images和VOC val不参与原型学习。读取原训练伪标签缩放至1008方形，逐类四邻域腐蚀7步，缩至72网格后核心覆盖>=0.95才采样；背景同样远离非背景/ignore及图片边缘。每图每类最多32个token，全类最多2048个，固定随机数42。前景至少8个样本，否则停止，不自动换配置。伪标签核心不等于真实高质量监督，背景仍可能含漏标前景。原型为SAM3处理后的1280维token的L2空间，不经过区域MLP。

球面k-means使用固定最远点初始化、20轮Lloyd，空簇保留上一中心，零向量或非有限值拒绝；没有新神经网络训练，但存在无监督原型拟合。样本与统计、原型全部保存并做哈希。共享原型用于全部seed。

本轮必须重跑SAM3：过去缓存没有完整patch token。使用1008、trunk_fpn、level=-1、layernorm、FP32，标准ViT MLP修正在模型初始化及任何特征探针前应用，关闭TF32。先真实一图训练探针；随后双卡提取训练样本，CPU拟合原型；再真实一图查询探针，并逐查询图检查前两个mask由在线token重新得到的GMP向量与旧缓存一致（atol/rtol 5e-5），不一致即停。之后双卡提取完整1449图。每图编码一次，同时生成两组20类72x72 FP32证据图，保存分数图、不存全量token；三个seed复用证据图。

分类仍使用原始mask对应的GMP缓存。分类后、类别内NMS前，对预测前景类取对应证据图，bilinear align_corners=False插值到原图，仅与原mask相交，不能新增像素或改类别/分数。空mask丢弃，无SMFA/原mask回退。NMS参数不变，但几何改变可能改变NMS成员，所以结果包含这一作用。背景候选保持原状。patch插值无法保证恢复细车架，应检查前景损失。每组seed固定保存按评测顺序最先20张有预测bicycle候选的原图/紫色被删像素并排图（bicycle_removed），标记候选被删像素，不代表最终预测净变化，不以GT选择样例。

原型、采样和证据图位于storage/raw_banks/voc/<版本>/foreground_verifier；配置、指标、报告在experiments/<版本>；日志在logs/<版本>。不新建语义检索库或权重。版本manifest绑定全部源码、历史输入、选中图片、伪标签、基础权重、配方；原型文件hash绑定查询提取job；逐图完成产物校验和独占启动锁支持恢复。三次基线必须复现历史总及逐图矩阵，再运行新方法。完整统计含每类IoU、总和逐图矩阵、前景漏失/背景误报及相对baseline差值、三seed均值/样本标准差；峰值显存/耗时在各worker queue_worker.json。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_foreground_verifier -v
nvidia-smi
df -h .
python -B tools/experimental_foreground_verifier.py --preflight-only
version="fg_verifier_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_foreground_verifier.py \
    --version "$version" --gpus 0 1 --max-hours 48 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

默认历史来源experiments/gmp_mlp2_20261002_141439，默认用户clean缓存active_clean_20260928_151808。保留此前error_attribution.py（包含邮件文件固定hash例外）、error_oracles.py及历史依赖。预检要求8GiB可用（3GiB余量＋5GiB底线），运行逐图检查底线，不自动删除。两卡最多各一worker，外部占卡等待，不杀其他任务；OOM不降精度或分辨率，失败全局停止并保留attempt。48小时从首次正式运行计，提取逐图检查截止；已领取评测允许完成，原型CPU拟合允许完成。完成、失败、中断或截止暂停时父进程尝试邮件通知，预检不发信。

续跑旧version加--resume --max-hours 96并另取日志名。已完成图片可跳过（提取新attempt会重新加载SAM3），已完成模型评测按阶段跳过；源码/参数变化必须新版本。先发summary.json、status.json、manifest.json。没有服务器连接，本地测试不能代替真实FP32、CUDA、缓存复现或全量评测。本地3项通过，2项因torch/OpenCV缺失跳过，服务器须执行。

## 27. 12小时双卡固定救援批次（2026-10-04）

独立入口 `tools/experimental_overnight_rescue.py`，测试 `tests/test_experimental_overnight_rescue.py`。用户休息期间无人值守，固定5组×seed42/123/2026，最多15次全量评测，无新训练、不追加搜索。基线仍为普通GMP＋两层MLP，SAM3评分融合0.2，原库和原标签不变。不继续负结果patch原型分支。

组别：baseline（原清洗与中心融合0.4）、local_only（原候选，中心权重0；无局部邻居类别保留原中心回退，非完全无中心算法）、loose（原候选并入min_score=0清洗后候选）、rescue、rescue_strict。后3组输入池相同，以mask哈希去重，原候选及其分数优先保留，附加候选仍保留其他清洗参数；并非关闭全部清洗。运行中在内存重新执行固定放宽规则，不覆盖用户原clean缓存。

救援规则：原strong候选按原分类后NMS保留；新增候选要求预测前景、SAM3分数>=0.4，温和组retrieval_score>=0.85且margin>=0.05，严格组>=0.90且margin>=0.10。按原融合分数稳定排序，要求相对当前保留前景的新增覆盖比例>=0.1，仅将新覆盖部分填入背景，不能覆盖原预测前景。背景候选不占据覆盖。原strong路径的逐图混淆矩阵必须与历史一致，最终预测在原strong前景位置逐像素不变。保留与裁剪后的形状不同，但检索用完整候选特征；这是保守的背景填充救援，不宣称复现外部建议所有集合选择公式。阈值固定，未在VOC val优化，检索分不是概率。无生成prompt类别可核验，因此不加入prompt一致性。

执行顺序：严格历史核验→3次基线完整复评→3次local_only（先留下低成本结果）→一图真实FP32/GMP探针→双卡分片附加候选特征提取→按seed42、123、2026，每seed按loose/rescue/rescue_strict领取任务。每图SAM3前向共享，原strong特征复用旧缓存，只持久化附加候选1280维GMP向量、mask哈希、source_indices。不保存全量token。没有附加候选的图片除探针外不编码。实际编码图像先验证前两个strong的在线GMP与历史一致，FP32修正先于模型初始化，TF32/BF16关闭，OOM不降配置。

所有输出按版本隔离：新向量storage/raw_banks/voc/<version>/rescue，状态/指标/产物清单experiments/<version>，日志logs/<version>。身份绑定源码、原始及清洗提议、基础权重、旧库与adapter。保留第24节邮件工具固定hash例外；不修改旧脚本、历史manifest或任何旧数据。变体指标显式允许已声明的候选数及local_only中心权重差异，其余配置、全量IDs、签名、逐图矩阵与IoU一致性严格检查。result.json说明有效候选池为内存扩展，metrics原提议路径只是strong输入来源。

每卡最多一个worker，外部占卡则等待。预检要求7GiB空闲（2GiB余量＋5GiB底线），运行逐图检查底线。12小时从首次正式运行计；截止不领新图/任务，已开始评测允许完成，退出可能略晚于12小时。输入/签名/缓存/历史复评异常全局停止；变体普通运行错误或OOM记录失败，继续独立任务，无自动重试。提取失败阻断救援组，已完成local结果保留。SIGTERM只终止本启动器子进程。已有图片/完整阶段可续跑，失败用新attempt，不覆盖。

每完成一单元更新summary.json和summary.md，列IoU、逐类/总矩阵、逐图矩阵路径、FP/FN、接受救援数、配对差值；不足3seed不计算均值。结束状态complete/partial/deferred/failed/interrupted，不自动选最高分部署。每worker queue_worker.json记录耗时、峰值显存；worker本身不发信，父进程最终落盘后尝试邮件通知（正式运行前预检失败不发信，断电/SIGKILL不保证）。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_overnight_rescue -v
nvidia-smi
df -h .
python -B tools/experimental_overnight_rescue.py --preflight-only
version="overnight_rescue_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_overnight_rescue.py \
    --version "$version" --gpus 0 1 --max-hours 12 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

查看 `cat experiments/$version/summary.md`。原进程退出后可用原version加--resume --max-hours 24继续（总预算24小时），日志另取时间戳，代码参数改变则新version。起床后提供summary.json、status.json、manifest.json，即使partial/deferred也可分析已完成项。测试5项本地通过，主要覆盖救援阈值、空池、不重叠及原前景保留；无本地torch/CUDA，真实加载、两卡提取及全量复评必须服务器验证，尚未启动真实任务。不承诺12小时内一定完成15个单元。

## 28. 救援机制与误报控制：48次固定评测（2026-10-04）

新增 tools/experimental_rescue_mechanisms.py、tests/test_experimental_rescue_mechanisms.py。复用上一轮 overnight_rescue_20261004_005647 的附加候选GMP向量、普通GMP两层MLP三个seed和原strong缓存。无新训练、无SAM3前向、无新伪标签、无新特征库；运行中只在内存重放原先min_score=0的扩展候选池，不写用户提议。依赖第27节旧脚本及其历史依赖，旧文件不修改。

固定16组，每组seed42/123/2026，共48次1449图评测。先完成baseline三次和rescue_strict三次历史总矩阵、逐图矩阵完全复评；任意不一致全局停止，再运行其他42次。

| 方法 | 相对严格救援的定义 |
|---|---|
| baseline | 原clean候选和原推理 |
| rescue_strict | q>=0.4、retrieval>=0.9、margin>=0.1、新增比例>=0.1；只填背景 |
| background_only | 只填背景，无新增q/检索/分差/比例门槛 |
| score_only | 只填背景，只有q>=0.4 |
| score_retrieval | q>=0.4、retrieval>=0.9，无额外分差和比例门槛 |
| score_margin | q>=0.4、margin>=0.1，无额外检索和比例门槛 |
| no_sam_score | 严格救援取消q门槛 |
| no_novelty | 严格救援取消新增比例门槛 |
| novelty30 | 新增比例改为0.3 |
| score60 | q改为0.6 |
| margin20 | margin改为0.2 |
| conflict10 | 增加其他类别原前景冲突比例<=0.1 |
| conflict30 | 增加其他类别原前景冲突比例<=0.3 |
| vote2 | 增加同类候选至少两票的像素支持 |
| conflict10_vote2 | 同时使用冲突0.1和两票支持 |
| retrieval_order | 新增候选按纯检索分降序分配像素 |

q为SAM3原候选分数，检索分不是概率。“无额外门槛”仍保留原检索器筛选和扩展池的其他清洗条件，不能称无过滤候选。除baseline外都保护原strong预测前景；不按类别特调瓶子/椅子阈值。冲突分母为完整新增候选面积，分子为其与原strong实际可见的其他类别前景交集（按原融合分first-wins合成，忽略被遮挡的strong类别），不是与未裁剪strong mask简单取并。冲突只看原前景，不随救援改变。

两票支持先对通过语义门槛和冲突规则的新增候选按预测类别逐像素计票；当前候选本身计一票，所以至少另有一个不同mask支持。重复mask拒绝；不同类别不投票。投票候选来自同一模型，不能解释成独立模型证据或校准置信度。只保留满足票数、原来未被前景占据的像素；新增比例分母始终为完整候选面积。随后按排序贪心分配，原始融合分不改，已接受新增mask彼此不重叠，所以后续按原分排序不会改变其像素归属。原候选及分类后NMS保持原样。

缓存来源必须complete、协议/配方/模型来源/清洗设置/1449 IDs相同；历史源码、全部输入、每个附加向量及完成标记校验hash，mask顺序与重放池精确匹配。保留adapter来源签名、原指标设置检查、第24节固定hash邮件工具例外。旧缓存只读，没有则停止，不静默重提取。新源码绑定新manifest，旧源码与历史结果不能覆盖。

执行：两卡各最多一个worker；6次复评通过后，按表顺序优先跑完每组三个seed。外部占卡时该worker等待，另一卡继续。默认12小时从首次正式运行计，到期不领新评测，已启动评测允许结束。全部48个单元完成就退出，不刻意占满12小时。预检6GiB可用（1GiB余量加5GiB保留底线），每图检查底线，不自动删除。输入/缓存/签名/复评异常全局停止；其他评测局部失败或OOM留attempt，继续独立任务，不自动重试或降配置。续跑失败阶段用新attempt，完成阶段校验后跳过。SIGTERM只终止本启动器子进程。

产物只有experiments/<version>和logs/<version>；artifacts.json分别列owned_roots和read_only_cache。summary含完整/逐图矩阵路径、逐类IoU、三seed均值/样本标准差、相对baseline和strict的配对差值、FP/FN、救援数和门槛计数。新增像素另按预测类别记录：背景误报、正确找回的前景、错分成其他前景；逐图统计在各attempt/result.json，255忽略。便于区分瓶子/椅子损失原因。缓存评测耗时和峰值显存保存在queue_worker.json。

此轮是根据VOC val既有诊断制定的开发消融及敏感性实验，不能把48次中最高分当成独立测试成绩；不自动部署、不声称统计显著。只在三个seed全完成时报告组均值。最终complete/partial/deferred/failed/interrupted状态落盘后沿用邮件通知，预检不发邮件。

```bash
cd /home/zystu2025/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_rescue_mechanisms tests.test_experimental_overnight_rescue -v
nvidia-smi
df -h .
python -B tools/experimental_rescue_mechanisms.py --preflight-only
version="rescue_mechanisms_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
(
  set -o noclobber
  nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_rescue_mechanisms.py \
    --version "$version" --gpus 0 1 --max-hours 12 \
    > "logs/${version}_launcher.log" 2>&1 < /dev/null &
  echo $! > "logs/${version}_launcher.pid"
)
echo "$version"
tail -f "logs/${version}_launcher.log"
```

查看 `cat experiments/$version/summary.md`。续跑必须先确认旧进程退出，保留原version，另取日志名：

```bash
# version设为需要继续的原版本；24是首次启动以来的总小时预算
nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_rescue_mechanisms.py \
  --version "$version" --resume --gpus 0 1 --max-hours 24 \
  > "logs/${version}_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

若来源目录不同，显式传--rescue-previous，预检和正式运行保持一致。不要覆盖旧脚本或改其完成标记来绕过核验。完成后提供summary.json、status.json、manifest.json；逐图定位再提供相关attempt/result.json与per_image_confusion.npz。本地17项测试通过（新12项加旧5项）及--help检查通过；本地无torch/CUDA，尚未执行服务器真实缓存/模型/48次评测，不代表已启动后台任务。

### 第28节补充：FP32指标核验恢复入口

2026-10-04服务器score_retrieval_2026两次完整评测中，mIoU与整数混淆矩阵float64重算差1.2693e-7、逐类最大差1.3891e-7，超出旧1e-7门槛；逐图矩阵之和、ID顺序及独立诊断矩阵全部一致。新增 experiments/recovery_tools/resume_rescue_fp32.py，不更改生产源码、旧实验脚本、manifest或历史metrics。该文件是实验恢复工具，位于source_identity枚举之外，其本身SHA256、目标manifest SHA256及数值策略单独记录在目标版本fp32_recovery_policy.json；重复启动要求策略身份相同。

仅父进程包装validate_eval：逐类和总体指标必须有限、合法且与矩阵float64重算绝对差不超过4个FP32 epsilon（4.76837158203125e-7），否则拒绝；通过后只在内存副本上统一指标并调用原核验器。矩阵、1449 IDs、设置、签名、输入哈希、逐图与独立矩阵一致性检查全部保留。worker仍用原文件，旧指标JSON不重写。已完成阶段跳过，未完成阶段新attempt重新评测；不伪造完成标记。原始异常会立即打印，避免被另卡取消信息遮住。邮件沿用原入口。

上传恢复脚本到服务器同相对路径，执行 python -B -m unittest tests.test_rescue_fp32_recovery -v（同时上传测试）。后台：

```bash
version="rescue_mechanisms_20261004_122809"
nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B experiments/recovery_tools/resume_rescue_fp32.py \
  --version "$version" --max-hours 24 \
  > "logs/${version}_fp32_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

24为首次启动以来总预算；如已经超过，显式增加。不要同时启动旧恢复进程。后续恢复仍用本入口。本地4项测试通过，无服务器连接，尚未执行真实续跑。

恢复入口v2修复（2026-10-04）：v1错误地依赖per_class_iou字典顺序；old.canonical按字母排序后与VOC矩阵顺序不同，导致虚假的class_error=0.443。改用 experiments/recovery_tools/resume_rescue_fp32_v2.py，按显式VOC类别名称与矩阵索引对齐，类别集合不符拒绝，数值误差上界保持不变。测试 tests/test_rescue_fp32_recovery_v2.py 共6项通过，包含非均匀逐类IoU的字典排序等价、实际类别值交换拒绝。v1代码和fp32_recovery_policy.json保留，新策略记录fp32_recovery_policy_v2.json，不覆盖旧审计记录。v2启动worker前检查该实验已保存的全部attempt指标（包括canonical排序副本）及已存在的逐图矩阵，尽早发现对齐问题。服务器上传v2脚本和v2测试，用v2入口替换上方后台命令中的文件名，其余version和预算不变。未连接服务器执行续跑。

## 29. 强证据单候选＋弱证据两票支持（2026-10-04）

独立入口tools/experimental_adaptive_rescue.py，测试tests/test_experimental_adaptive_rescue.py。固定8组×seed42/123/2026=24次完整验证，无训练、SAM3重提取或新库。默认来源和旧缓存同第28节，额外核验第28节完整结果的vote2对照。原基线、strict和vote2共9次复评总矩阵和逐图矩阵必须与历史完全一致，随后才运行15次组合评测。

全部组合先通过strict基础门槛q>=.4、retrieval>=.9、margin>=.1，强证据候选允许单独补背景，其余候选只保留至少两个同类合格候选共同覆盖的像素。强证据固定五组：margin>=.2、margin>=.3、retrieval>=.95、q>=.6、retrieval>=.95且margin>=.2。保留新增面积比例.1、原融合分排序和原前景保护，票数包含当前候选自身；不同mask不是独立模型证据。无按类阈值，无验证过程中追加参数。特征、MLP、清洗、检索全部保持；GT只参与评测及诊断。

新入口内置v2按类别名称对齐的4*FP32 epsilon内存核验副本；逐图、签名、来源、原指标检查仍保留。原始父进程异常即时打印。每卡最多1worker，12小时停止领取新任务，已开始评测可完成；5GiB运行保留底线，6GiB预检。新版本experiments/logs隔离，不改历史脚本或数据，不自动部署最优结果。完成、失败或中断按原邮件工具发送通知。原server缺少本手册，本次同步创建docs/PROJECT_STORAGE_MANUAL.md。

启动：python -B tools/experimental_adaptive_rescue.py --preflight-only；通过后nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_adaptive_rescue.py --version adaptive_rescue_<时间> --gpus 0 1 --max-hours 12 > logs/<版本>_launcher.log 2>&1 < /dev/null &。续跑同版本加--resume，总预算按首次开始计。查看experiments/<版本>/status.json、summary.md及logs；结束提供summary.json/status.json/manifest.json。实验速度不能代替三seed全量核验，不声称统计显著。

## 30. 高门槛与投票支持的机制分解（2026-10-04）

新增tools/experimental_rescue_evidence.py及tests/test_experimental_rescue_evidence.py。固定8组×3seed=24次完整验证。按顺序先复评baseline、rescue_strict、hybrid_retrieval95，要求总及逐图混淆矩阵完全复现；上一轮最佳混合组来自adaptive_rescue_20261004_164559，历史源码/输入和完成标记绑定新manifest。其余五组：high95_only只接受retrieval>=.95且原基础门槛合格候选；weak_vote2_only仅接受.9<=retrieval<.95合格候选的两票像素；hybrid_strong_first将>=.95的强候选排在其他候选前，各层内仍按原融合分降序；hybrid_weak_voters强候选仍可单独恢复，但投票地图仅由弱候选生成；hybrid_vote3强候选单独恢复，弱候选需3个同类不同mask支持（含自身）。所有组保留q>=.4、margin>=.1、新增面积比例.1以及保护原前景；弱候选计票支持不是独立模型证据。无GT决策、无按类阈值、无中途调参。

重点配对对比hybrid95−high95_only确认弱候选共识的边际价值；weak-only提供单独贡献对照，不能把两者mIoU简单相加；strong-first检查像素竞争；weak-voters检查强候选能否提供必要支持；vote3检查支持数量敏感性。报告沿用逐类IoU、逐图矩阵、背景误报/前景漏检与新增像素诊断。不新增训练或SAM3编码，输出仅新版本experiments/logs。固定12小时预算、双卡独占自有worker、磁盘5GiB保留、阶段续跑和邮件通知沿用第29节。源码、模型、输入、缓存和FP32具名核验仍严格保留；不自动部署最高分配置。

运行：python -B tools/experimental_rescue_evidence.py --preflight-only；通过后nohup env CUDA_VISIBLE_DEVICES=0,1 python -u -B tools/experimental_rescue_evidence.py --version rescue_evidence_<时间> --gpus 0 1 --max-hours 12 > logs/<版本>_launcher.log 2>&1 < /dev/null &。续跑同version加--resume并扩大从首次启动计的总预算；原进程须已退出。已有脚本和历史结果不修改。新7项测试加相关11项测试共18项本地通过，真实对照复评仍以服务器结果为准。


## 31. 主线HCR模块：高置信检索救援（2026-10-04）

用户授权将已验证的high95_only合入生产主线。新增src/semantic_rescue.py；修改src/inference.py和run_inference.py；新增tests/test_semantic_rescue.py。模块不依赖实验脚本，不新增可训练参数、不改变区域聚合/MLP/特征库签名。生产默认仍为SMFA单库；本次历史精度验证使用已有GMP＋两层MLP缓存与权重，不能把其精度直接当成任意SMFA配置的成绩。

开启方式：在现有匹配库/权重/数据的严格推理命令追加 `--rescue`。当前仅接受VOC、--strict-eval、INF_LOAD_PROPOSALS=true、已清洗缓存且不传--cleanup-config；不支持在线提议或ADE/Cityscapes的隐式套用。不加--rescue保持旧行为；这是主线正式模块的显式开关，不是运行时补丁。--rescue-raw-dir可显式指定原始提议目录，否则读取clean/cleanup_config.json的input_dir。原始池必须存在，不能把已清洗池伪装成原始池，缺失直接报错。

固定HCRConfig参数：proposal_score=.4、retrieval_score=.95、margin=.1、novel_fraction=.1。流程：保留全部原clean mask及score → 原raw池按元数据清理配方仅将min_score设0生成relaxed池 → 按mask几何去重并优先保留clean → 共用同一特征提取/MLP/检索 → 原clean候选正常类别内NMS → 附加候选应用HCR门槛、按原融合分稳定排序，只补未被前景占据的位置 → 新增面积占原候选面积至少.1才保留 → 原光栅化。附加候选不参加原候选NMS、不覆盖原前景，不引入两票投票或跨类冲突过滤。原检索margin等全局筛选仍适用。背景及非前景类别不参与救援。完整候选用于提特征，裁剪仅发生在最终像素合成前。

结果JSON增加semantic_rescue：模块版本、有效门槛、raw/clean目录、清理metadata哈希、每图raw/clean文件哈希、strong/extra数量、接受数及新增像素。原proposal_count_before/after仍表示clean输入；附加数在semantic_rescue.images中明确记录。文本成绩也记录HCR启用状态和配置。GT只参与评测，无任何救援决策读取GT。

合入前已确认服务器没有运行中的项目worker；旧生产源码及其他Python源码快照保存在experiments/hcr_mainline_20261004_214541/source_before，旧手册同目录docs下。原标签、提议、库、权重、指标全部保留。新验证的prepared.json绑定原源码/输入/历史任务；deployment.json只允许run_inference.py和src/inference.py两处既有源码变更及src/semantic_rescue.py新增，其余历史输入继续逐项校验。旧实验启动器有源码严格绑定，生产变更后拒绝直接续跑是预期行为；不要改旧manifest或关闭签名检查。历史源码快照用于复现，后续实验需基于新代码另立版本。

验证工具tools/validate_semantic_rescue.py只用于主线回归，依赖旧实验兼容层加载其GMP2签名和缓存，不改变生产签名规则。它调用真实新run_inference、predict_instances和HCR模块；只替换特征编码器为已核验的历史pre-MLP向量，正常执行原MLP、检索、NMS、救援和合成。三个seed各做关闭/开启共6次1449图评测；检查总及逐图混淆矩阵完全一致，关闭对照baseline、开启对照high95_only。--prepare必须在替换源码之前执行；之后同version运行全量回归或重试失败阶段，完成阶段哈希校验后跳过。完成/失败邮件通知沿用旧工具。回归产物位于experiments/hcr_mainline_20261004_214541，日志logs/同版本。

服务器18项相关测试已通过（HCR5项、检索2项、清理8项、签名3项）；全量回归启动后以该版本status.json/summary.json为准，不用单元测试代替实测。主线开启示例（库和adapter须换成匹配的生产路径）：

```bash
export CUDA_VISIBLE_DEVICES=1
export OVERRIDE_INF_LOAD_PROPOSALS=true
export OVERRIDE_INF_PROPOSALS_DIR=/home/zystu2025/nd/ModuSeg_SAM3/storage/proposals/voc/active_clean_20260928_151808
python -u -B run_inference.py --strict-eval --rescue \
  --output-dir experiments/<新版本>/predictions \
  --results-json experiments/<新版本>/metrics.json
```

不要把历史实验GMP库直接交给生产SMFA编码器；正常签名校验会拒绝。HCR本身与聚合方法解耦，本次不顺带将其他实验聚合方案并入主线。

第31节实测验收（服务器2026-10-04）：hcr_mainline_20261004_214541最终status=complete，6/6项完整1449图回归通过，所有总矩阵及逐图矩阵与历史完全相同；最终源码和输入哈希复核通过。关闭HCR三个seed mIoU：.8550376044603231、.8564289827324358、.8563366602544759，均值85.5934%；开启：.8627229329769592、.8644032270471317、.8658216628667466，均值86.4316%。该验收复用历史GMP查询向量并执行实际MLP/主线推理/HCR，没有重跑SAM3编码。不能将结果泛化为所有特征配置的保证。主线模块已部署；实际使用通过--rescue显式开启，关闭保留原行为。

## 32. 特征库错误溯源（2026-10-05）

用户暂停HCR后续优化，新增独立入口tools/experimental_bank_provenance.py、测试tests/test_experimental_bank_provenance.py。生产源码和历史结果不改动。固定现有GMP两层MLP的42/123/2026三个seed，诊断原clean候选，HCR不参与本轮重放；结果不代表HCR新增候选的归因。

从第31节已通过的部署记录核验生产源码，并保留旧输入及未修改源码的SHA256检查；新脚本绑定新manifest，不关闭签名检查。先遍历训练划分，按历史members.json中的类别、样本ID、区域index重建伪标签区域，逐一核对mask SHA256、图片/伪标签哈希及训练缓存成员顺序。读取训练GT只产生区域纯度诊断，不修标签、不删样本、不训练。样本ID后缀不能当作区域index。

每个seed复用GMP缓存，验证投影库与原始区域向量经过配套MLP后的结果吻合，精确核对线上库行顺序。真实主线推理同时导出MLP前后两套近邻、各类别局部/中心/融合分数、检索支持数量、top20近邻及预测类/GT主要类在候选检索集内最多5条近邻来源。没有检索支持的类别保留原中心回退。三个seed均必须复现完整1449图总混淆矩阵及逐图矩阵，不把候选层诊断准确率当作语义mIoU。

GT纯度>=0.9的候选用于固定诊断统计，混合候选仍完整导出。重复候选统计不等于去重错误像素；邻居频繁出现和标签纯度只能作为证据，不能独立证明因果。输出catalog/attempt_*/regions.json、seed_*/attempt_*/query_provenance.jsonl及diagnostics.json、summary.json、status.json。来源记录在catalog中绑定哈希。当前不输出新语义方法或宣称提升。

运行python -B tools/experimental_bank_provenance.py --preflight-only；正式运行用新--version及nohup，默认先GPU1后GPU0分配三个seed，各卡至多一个自有worker。CPU区域溯源阶段不占GPU。预检至少6GiB空闲，运行保留5GiB；默认24小时预算。已有版本仅--resume且身份一致可恢复，失败attempt保留；信号只停止本启动器子任务。完成或失败邮件通知沿用既有工具。新产物仅experiments/<version>及logs/<version>，不自动删除或打包。该脚本不会绕过其他旧启动器的源码变更检查。

本地5项纯CPU测试通过，包含真实检索分解及无近邻类别的中心回退一致性。服务器版本bank_provenance_20261005_101808已后台启动，完成预检后进入10582张训练图的区域溯源阶段；真实完成情况以该版本status.json为准，不以启动成功代替全量验收。

## 33. 库可信度与训练分布拒识（2026-10-05）

新增独立tools/experimental_bank_trust_queue.py、tests/test_experimental_bank_trust_queue.py。固定现有GMP两层MLP三个seed、原clean候选和单库成员，不训练、不改生产源码、不研究HCR。六组baseline/trust/reject01/reject05/trust_reject01/trust_reject05，每组三seed，共18次1449图全量评测；先复评baseline三个seed总及逐图混淆矩阵，通过后才做新组。保留特征签名和权重来源核验、代码/缓存哈希，原数据只读。

可信度：在MLP前GMP特征空间，对每条库样本排除同图全部区域，计算每类余弦top5均值；r=sigmoid((同类均值-最强异类均值)/0.05)。使用原伪标签类别，GT不参与计算，三个seed共用r。推理时保留原检索候选与每类top5集合，将其相似度s改为s-0.05*(1-r)后求均值。类中心、中心权重0.4和无近邻类别的中心回退不变。全r=1的数值等价性测试保留，每个批次还将未修正公式与生产分类器核对。

拒识：每个seed的已投影训练库排除同图区域，计算各类别top5支持分数。只用样本自身伪标签类别的支持分布，固定取第1%或5%分位数；原分数及可信度修正后的分数分别计算门槛。查询预测为前景，但该类局部支持低于门槛或检索集中无该类支持时退回背景。对已经预测为背景的候选不做救回；不变更SAM3分数融合、原筛选与NMS。这里是已学习特征上的排除同图邻居统计，不是独立留出精度，也不是校准错误概率；不读取训练GT或验证标签计算r和门槛。

可信度计算、三个seed校准阶段用FP32分块矩阵运算，关闭TF32；同卡最多一个自有worker，GPU1优先，双卡分配。复用已有区域向量，不再SAM3编码，不写新库/权重。训练缓存身份由历史members.json、mask哈希、样本ID与类名顺序绑定。GT只在原推理评测中读取。脚本不读上一轮regions.json的GT纯度作为输入。

运行python -B tools/experimental_bank_trust_queue.py --preflight-only；正式nohup python -u -B tools/experimental_bank_trust_queue.py --version <新版本>，后台日志放logs/<版本>_launcher.log。查看experiments/<版本>/status.json和summary.json；失败attempt保留，--resume须身份相同且旧进程退出。每次启动24小时预算，预检6GiB、运行保留5GiB；信号只终止本启动器worker。完成/失败沿用邮件。产物只有experiments及logs，不自动清理。

summary记录六组三seed、均值/样本标准差、配对差值以及组合交互差值。未完成三seed不报告组均值。metrics含1449 IDs、逐类IoU、混淆矩阵，per_image_confusion.npz保存逐图矩阵。decision_stats仅候选拒识计数，不等于去重像素错误。作为VOC val开发实验，不宣称显著性、不根据结果临时添加参数或自动部署。

第33节启动记录：本地4项通过、1项因无torch跳过；服务器5项全部通过，预检通过，本地/服务器脚本SHA256一致。后台版本bank_trust_20261005_112906，启动器PID 676337。完成状态以该版本status.json为准；此记录仅表示已启动，不代表完成18次评测。

## 34. 图像标签约束：训练审计与原配方受控重训（2026-10-05）

新增独立tools/experimental_tag_constraint_queue.py、tests/test_experimental_tag_constraint_queue.py，生产源码和旧实验不改。依据第33节已完成实验核验历史源码/输入，并绑定历史GMP2 train job、图像级JSON及内部划分。只复用历史单相位GMP原库、原clean查询缓存，不启用HCR。当前实施三组：historical（原模型复评）、control（原配方新训）、absent01（新训并加图中不存在类别约束），每组42/123/2026，共6次训练、9次1449图评测。不实现MIL、cross-fit或软成员库；下一阶段依据本轮机制证据另定，不在运行期间追加参数。

先审计固定库对训练图片图像级类别的覆盖，列出缺失正类和伪区域类别违反图像标签的记录。标签JSON采用原images/img_id/labels结构，ID通过原class_mapping转换，background始终允许；缺失、重复或非法标签停止。此覆盖只描述现存库成员，不能证明对象没有被区域表示。训练数据/库行顺序从原load_training_bank读取，balanced_batch输出的原行号用于绑定每条区域的图像标签；不通过特征猜测身份，不读取训练像素GT。

新增损失采用稳定形式L_abs=mean(logsumexp(all logits)-logsumexp(allowed logits))，即-log P(属于图像允许类别集合)，允许集合包括背景。absent01：L=原CE+0.1 L_abs+0.5 SupCon。control完全返回原CE优化图；原hard label、线性head、SupCon、类平衡采样、50×100步、AdamW、原学习率/scheduler及内部KNN选模全部保留。所有patch只在实验worker内生效，调用原trainer。该约束不能直接解决同图允许类之间的混淆，不能保证纠正漏标前景。

原trainer没有保存临时分类head，本轮在best adapter选中时另保存classifier.pt，类别顺序/epoch/图像标签哈希绑定；训练后验证head与best adapter配套，在原内部train/validation图片上报告不存在类概率质量、top1违规数。这里只是临时分类head的监督诊断，不是检索精度或校准概率。记录5000步原CE、辅助损失和概率质量。新control与历史checkpoint逐tensor比较、记录best epoch及bitwise equal；若新control平均精度偏离历史超过0.1个百分点，停止后续absent组，先排查管线，不把它当作方法结论。

历史模型先复评总/逐图矩阵；新库成员、内部划分、查询数、特征签名、MLP来源和逐图混淆矩阵继续核验。lambda=0的短训练轨迹等价、全允许类别零损失、极端logits有限梯度、标签错序及缺项由专用测试覆盖。本地3项通过、2项缺torch待服务器执行。

运行python -B tools/experimental_tag_constraint_queue.py --preflight-only；正式nohup python -u -B tools/experimental_tag_constraint_queue.py --version <新版本>，默认GPU1优先、双卡每卡一个worker，外部占卡等待。启动24小时预算、磁盘5GiB底线，预检估算6套投影库加256MiB余量；已有版本仅--resume且身份一致、旧进程退出可恢复，失败保留attempt。新权重/分类head放weights/sam3_adapter/<版本>/<方法>/<seed>/<attempt>，新投影库放storage/mlp_banks/voc同结构，阶段审计/配置/指标放experiments，日志放logs。完成/失败邮件通知，不自动删数据或部署。

最终summary含原模型/新control/absent01三seed及配对absent-control、control-historical差值；完整与逐图矩阵保存各eval attempt。内部审计来自图像级标签和伪标签，最终VOC val GT只用于全量评测。本轮属于VOC val开发实验，不将结果视为独立测试，也不宣称统计显著。

第34节启动记录：服务器6项专用测试全部通过（含采样行绑定/最佳分类头保存及零系数短训练轨迹）；本地3项通过、3项因无torch跳过。服务器预检通过，额外空间估算0.46GiB，本地/服务器脚本SHA256一致。后台版本tag_constraint_20261005_124131，启动器PID 677765。以status.json为准，此记录不代表已完成新训练或获得新精度。

第34节失败与恢复（2026-10-05）：原版本三个历史模型总/逐图矩阵复评通过；control seed42/123完成训练，最佳权重逐tensor与历史一致。评测环境仍引用历史投影库，而缓存extractor加载新权重，文件SHA不同触发签名拒绝；未获得新方法结果，summary的Queue cancelled是另一卡取消的后续报错。

新增独立tools/experimental_tag_constraint_recovery.py和tests/test_experimental_tag_constraint_recovery.py。原启动器及worker文件保持冻结。恢复入口先核验原manifest的源码/输入和全部完成阶段，检查已训练库adapter SHA与权重文件相同，在新版本只读引用已完成训练产物；失败评测不复用。仅将评测环境的OVERRIDE_FEATURE_ADAPTER_CKPT、OVERRIDE_INF_FEATURE_BANK_ROOT、OVERRIDE_INF_PROPOSALS_DIR绑定到当前job。签名和历史复评继续严格执行，算法、损失、seed、查询及训练配方不变；新identity记录恢复来源和完整SHA记录，不修改旧manifest或完成标记。

预检：python -B tools/experimental_tag_constraint_recovery.py --recover-previous experiments/tag_constraint_20261005_124131 --preflight-only。正式运行用新--version和nohup，日志写logs/<新版本>_launcher.log；该入口的--resume只恢复其自身新版本且身份一致。旧权重/库不复制、不覆盖；尚未完成的训练输出到新版本。完成/失败仍邮件通知。专用测试覆盖多seed库/权重路径绑定、已完成训练只读复用及哈希损坏拒绝、未完成训练新attempt。

恢复启动记录：本地3项恢复测试通过，服务器恢复及原实验合计9项通过，严格预检通过，原脚本SHA保持5705c4a1f2aa4d0ca54b46f6a95a4967810d5206cd1d744273ef09a6b3574bd7。恢复脚本本地/服务器SHA一致。后台新版本tag_constraint_fixed_20261005_133142，启动器PID678364，仍需完成后核验九个评测单元；此记录不代表获得新方法精度。

第34节完成核验：tag_constraint_fixed_20261005_133142状态complete，9/9评测均完整1449图；完成阶段产物哈希通过，逐图矩阵之和/评测ID通过。三个新control最佳权重逐tensor和历史相同，总/逐图混淆矩阵也完全复现。control均值85.5934416%，absent01均值85.6086944%，配对均值+0.0152528个百分点；42/123/2026分别+0.0005338/+0.1087877/-0.0635632个百分点，方向不稳定。原内部验证分类head不存在类别概率质量均值3.4431%→2.7101%，top1违规0.8590%→0.6476%，这些不是检索正确率。最终每seed平均背景误报像素不变，前景漏检减少826像素、前景类别混淆减少5196.7像素；逐类均值chair下降1.1054个百分点，plant不变。不能据此宣称修复库污染或方法有稳定优势，未并入主线。当前仅检验absent loss，不是MIL/cross-fit/soft-membership方案的完整否定。事后分析保存本版本analysis_20261005/findings.json，本地副本位于experiments/remote_results同版本目录。

## 35. 数量匹配的背景结构实验（2026-10-05）

独立入口tools/experimental_background_partition_queue.py，专用测试tests/test_experimental_background_partition_queue.py；不改生产代码、旧脚本、伪标签及验证提议。固定单相位硬中心GMP rho=0.1、两层MLP、v1原训练配方、内部图片划分、检索与clean查询向量，不启用HCR。默认来源tag_constraint_fixed_20261005_133142，严格核验历史源码/输入/完成产物，先复评历史模型三个seed的完整总/逐图矩阵。实验共四组，各42/123/2026：historical（完整历史库、只复评），merged_matched（数量匹配合并背景），split_matched（数量匹配局部背景），safe_matched（隔离疑似对象后的局部背景）。共9次新训练、12次1449图完整评测。

训练侧没有可复用的全量提议缓存。本轮用新的训练图像级类别限制Active候选类，FP32，标准ViT MLP修正先于模型真实探针，关闭BF16/TF32。Active固定preview=false、bucket_size=4、neighbor/hypernym_tokens=2、context_strength=1、confidence_threshold=0.15、mask_nms_threshold=0.7、min_mask_area=100、min_mask_region_area=0、max_mask_area_ratio=0.95、max_masks=0。这是新训练提议，不声称重放原标签过程，不写入或覆盖旧PNG。Active与特征提取共享同一个冻结SAM3模型实例，依次生成候选及原特征token；不是共享同一次前向。原验证clean缓存不重新生成或清洗。

unknown仅由候选几何确定：候选重心取最近像素落在伪背景，且候选位于伪背景的面积比例严格大于0.5。unknown只从背景监督中隔离，不标为任何前景，不使用候选类别做重标注；Active候选仍源于SAM3，不能当作独立真值或校准objectness。safe=原伪背景减unknown并集。

先对固定训练库前景掩码统计硬中心token数中位数，局部方块边长=ceil(sqrt(median))个token，映射到72×72网格对应原图空间，边界截断。对每图用SHA256(image:tile-row:tile-column)固定块顺序，选择第一个实际背景像素>=300的块，每图只留一个局部向量。split和safe使用相同网格及顺序，safe无合格块时不回退污染背景。只有split/safe均有合格块的图片进入三个新组背景库；三个新组背景样本ID、图片集合、总数完全一致。没有合格safe的图，其背景从三个新组同时排除，所有原前景成员和向量保留。原完整库作为额外参考，因此merged_matched-historical单独量化背景成员减少的影响；split_matched-merged_matched量化局部表示作用；safe_matched-split_matched量化隔离作用。不是保存所有局部块的扩库方法，也不能把结果泛化为所有背景划分方案。

背景全图参考向量只读复用原C缓存；两个新背景向量按原硬中心选择、现有处理后token、等权GMP与原归一化顺序生成，空/非有限/最终零向量明确失败。真实probe先核对重新计算的原背景GMP与历史缓存allclose(atol/rtol=5e-5)、共享token重复聚合一致，失败不进入全量提取。两卡按训练图片交错分片，一卡最多一个worker；逐图原子保存两个所选mask的packbits、三条背景向量、mask哈希/统计，不存完整token或全量提议mask。已完成图片仅哈希核验一致才跳过。

新原库前景NPZ通过只读符号链接引用旧文件，绝不在链接位置写入；新背景矩阵保存到本版本原库。独立background_protocol签名记录组别、几何、提议参数及匹配规则，worker扩展严格签名比较；保留原聚合签名、基础权重、adapter来源和文件SHA检查。缓存查询使用相同前MLP C向量、新模型及配套库；环境中的库/adapter/提议路径明确绑定job，避免沿用历史路径。内部验证图片只能来自原划分；样本排除导致实际图片覆盖减少时显式记录，不改划分。缺训练类别或空内部验证停止。

库成员/掩码全部固定并提交后，独立diagnostic阶段才读取训练像素GT，报告三组背景的前景占比、>=10%和>=50%污染区域数及每区域直方图；诊断数据不进入训练、候选选择、参数或检索。随后按seed42三个组、seed123三个组、seed2026三个组训练评测。报告三seed均值/样本标准差及预设配对差值，完整/逐图混淆矩阵保存在eval attempt。VOC val为开发结果，不宣称显著、不自动部署。

命令：python -B tools/experimental_background_partition_queue.py --preflight-only；正式nohup python -u -B tools/experimental_background_partition_queue.py --version <新版本> > logs/<新版本>_launcher.log 2>&1 < /dev/null &。默认物理GPU1/0，可--gpus 0 1，默认--max-hours 48；到期停止新图片/任务，已开始训练/评测可完成。--resume要求相同身份、产物校验及旧启动器退出；预算从本版本首次运行计，可显式增大--max-hours。保留5GiB磁盘底线，预检估算新背景、九套投影库与384MiB掩码/报告余量；不足则保留断点，不自动清理。

新背景缓存与原库：storage/raw_banks/voc/<版本>/background_cache和各组/<merge attempt>；MLP库及权重分别storage/mlp_banks/voc/<版本>/<组>/<seed>/<attempt>、weights/sam3_adapter同结构；参数、诊断、指标在experiments，日志在logs。输入/缓存/签名全局失败停止；局部训练/评测失败保留attempt并继续独立任务，OOM不自动降配置。SIGTERM只停止自己的worker，完成/失败沿用邮件通知。原始提议、标签、历史库/指标不删除。

第35节启动记录：本地5项通过，GMP数值测试因缺torch、符号链接合并测试因Windows账户权限而跳过；服务器7项全部通过，严格预检通过，空间估算0.95GiB。脚本本地/服务器SHA256一致：b1c2b2e1bad46f6126aefc04a13d5e463393a6ea4e97ce0d7da8c99409b41ac0。后台版本background_partition_20261005_183438，启动器PID680002，已进入双卡历史模型完整复评；真实probe和全量新方法结果仍以阶段完成标记/status.json为准，不能以启动成功代替完成。

第35节接入修正：background_partition_20261005_183438完成三个历史复评后，在geometry阶段因旧train job省略plan/pseudo/source字段而停止。失败源码保存其source_snapshot目录，本地副本同样归档；旧完成标记和指标不改。新脚本extraction_context从已核验且cache/checkpoint/train_ids/raw_signature相同的baseline job补充这三个字段，预检核验区域计划完整覆盖训练ID；新增测试覆盖此差异。服务器8项测试通过，本地6项通过、2项环境限制跳过。新预检通过，SHA本地/服务器一致：2cc110da4f53cc285177894e5300ae7c14ba848c94ac1a84db3ca9e63c429953。正式新版本background_partition_fixed_20261005_184155，启动器PID680349；默认48小时，不能把已启动视为新方法已完成。

第35节序列化修正及正式恢复：fixed版本完成三seed历史复评和10582图geometry审计，前景支持中位数237，块边长固定16 token。FP32模型初始化前向通过，但probe的Active词表比较因tuple经JSON变list触发误拒绝，未提取全量新背景。原源码保存fixed版本source_snapshot，旧产物不改。比较统一JSON规范化，内容变化仍拒绝；服务器9项测试通过，本地7项通过、2项环境限制跳过。新增--recover-previous只读复用geometry和历史复评阶段，不复用失败probe；除本独立启动器外全部旧源码SHA保持严格校验，启动器旧SHA必须由原样归档源码匹配，恢复来源和记录绑定新manifest。恢复不关闭签名检查。

最新正式版本background_partition_ready_20261005_185337，启动器PID680725，启动命令追加--recover-previous experiments/background_partition_fixed_20261005_184155。源SHA本地/服务器一致：cede963dfa10e508b9a342fb8ced9aa0a062a7c926e6f7b2e524ebfa4e9a2da9。状态以该最新版本为准；需probe通过才进入双卡全量背景提取。

最新版本真实probe已通过：2007_000032重新计算的全图背景GMP与原缓存max_abs=2.9802322387695312e-08，局部/隔离后掩码均128个硬中心token，重复共享token聚合一致；FP32标准ViT MLP修正32实例，probe峰值allocated约4.353GiB。三个历史模型总/逐图复评仍通过，只读复用对应完成阶段及geometry。两卡已正式开始5291/5291张交错分片提取，核查时两lane分别处理29/33张，状态running/error=null。尚未完成新库合并、九次训练及九次新组评测，不报告新精度。

第35节部分完成与恢复（2026-10-06）：ready版本全量背景提取、合并和事后纯度诊断完成；因运行中磁盘不足5GiB，状态deferred，12评测只完成6个（历史3、新组3），4次新训练完成。合并/局部/safe背景均保留10578成员，前景不变。新组42：merged 85.3966%、split 85.4534%；123：merged 85.4804%；不得报告新组三seed均值。三组背景区域前景占比均值6.0208%/9.0797%/8.8445%，>=50%污染区域187/709/684；GT仍仅事后诊断。原已记录文件逐一SHA/size核验通过。

新增独立tools/experimental_background_partition_recovery.py、tests/test_experimental_background_partition_recovery.py，四项测试本地/服务器通过。旧ab.Stages将三个原库的前景符号链接解析为重复路径，重新排序/去重后误判列表不等；恢复入口只改实验进程内阶段产物比较为绝对路径→size/SHA映射，重复身份必须完全相同，文件变化/缺失/冲突继续拒绝。不修改旧启动器、生产源码、签名或历史完成标记。要求原提取/合并/诊断全部提交，所有旧源码/输入/产物核验后，在新版本只读复用全部完成阶段，未完成任务新attempt。命令：python -B tools/experimental_background_partition_recovery.py --recover-previous experiments/background_partition_ready_20261005_185337 --version <新版本> --preflight-only；正式去掉--preflight-only并用nohup，已有恢复版本用相同参数加--resume。不降低5GiB底线，不清理历史产物。

第35节恢复启动记录：严格预检通过，原独立脚本SHA仍为cede963dfa10e508b9a342fb8ced9aa0a062a7c926e6f7b2e524ebfa4e9a2da9；新恢复入口SHA为3f1e70d3d7039b8ced12dbf62f4f3167e7a0a45b82bb1d5da854ceb19eeab989。后台新版本background_partition_resume_20261006_002500，PID682466，命令追加--max-hours 24。正式状态已进入running/error=null，只读复用历史三个评测、geometry及probe，继续校验和引用完整提取产物。仍须补完5次训练、6次评测；原ready版本保持deferred，不覆盖其summary/status。最终完成以新版本status.json为准，邮件沿用。

第35节完整结果（2026-10-06）：background_partition_resume_20261006_002500状态complete，12/12全量1449图评测，failures为空；完成产物size/SHA、冻结源码、逐图矩阵核验通过，历史三seed总/逐图矩阵精确复现。historical/merged_matched/split_matched/safe_matched三seed均值分别85.5934%/85.4737%/85.5431%/85.6042%。safe相对完整基线仅+0.0107个百分点，seed42/123/2026分别+0.1647/-0.0095/-0.1231，方向不稳定；相对数量匹配merged为+0.1305，不得混用两种对照。safe相对完整基线chair -2.0068、sofa -1.3662个百分点，pottedplant/bottle不变；背景误报像素均值不变，前景漏检-49552.7，前景类别混淆+37183.3。九次新训练内部图片覆盖均保持；seed2026 split/safe逐图矩阵相同，但adapter SHA不同。建议保留原clean基线，当前背景划分方案不并入主线，不继续自动扫参。完整报告归档experiments/remote_results/background_partition_resume_20261006_002500/report.md，服务器事后分析位于该版本analysis_20261006/findings.json；原始完成阶段不修改。本次未启动下一轮实验。


## 36. SAM3 区域语义辅助检索（独立开发实验）

入口 `tools/experimental_sam3_semantic_queue.py`，测试 `tests/test_experimental_sam3_semantic_queue.py`。只读复用原 clean hard-center GMP rho=0.1、两层 MLP、三个 seed=42/123/2026、固定 clean mask 与缓存查询向量；不启用 HCR，不重训、不改库成员。默认历史核验来源 background_partition_resume_20261006_002500；比较的是其 historical 组，不采用新背景库。

四组共12次1449图评测：baseline 原检索；sam_label 仅由 SAM3 决定类别；rank_bonus 语义排序奖励辅助原分类；guarded 只纠正犹豫且语义证据明确的区域。为隔离类别判定，本轮所有组保留原检索 winner 的标量分数和 margin，用原0.8检索+0.2提议分数排序，NMS/覆盖算法不变；类别改变仍会影响类别内NMS。这是 label-only 机制实验，sam_label 不是原生SAM3整套分割基线。

旧 raw/clean NPZ 没有候选类别ID，旧元数据只提供图片级类别筛选信息，不能恢复逐mask语义。新建语义证据：Active 关闭preview，对完整VOC20前景词表解码，不读验证图片类别标签。固定bucket=4、neighbor/hypernym=2、context_strength=1、confidence_threshold=0.15、per-class NMS=0.7、min_area=100、min_region_area=0、max_area_ratio=0.95、max_masks=0。FP32、TF32关闭，真实前向前应用标准ViT MLP修正，1008输入。新候选只用于为固定查询mask提供证据，不加入查询集合，不重新清洗旧提议。

对原查询R，E[R,c]=max_j(q_j*IoU(R,M_j))，仅接受同类候选且IoU>=0.5。q是SAM3条件检测/存在性置信度，不是校准的类别概率或mask IoU；它和检索均来源于SAM3，并非独立真值。无匹配证据时联合组完全回退原检索；sam_label弃权、该区域不作为前景，不能解释成SAM3明确预测背景。rank_bonus对正证据类别按E降序给原检索类别分加0.05/rank，背景不加；不会直接线性混合两种未校准分数。guarded只在原检索top1-top2<=0.05、E最高>=0.5且最高与第二名差>=0.1时采纳SAM类别，否则原判定；阈值预先固定，不按val结果搜索。语义同分按VOC ID稳定排序。

先完成原模型三个seed总/逐图混淆矩阵精确复评，再做首张固定val图片的数值/分块/缓存等价性探针（不读取GT、不按精度调整规则），通过后两卡交错分片完成1449图语义缓存。缓存仅保存N×20证据、对应IoU/置信度和统计，不保存新mask/token；绑定原图/提议SHA、mask顺序哈希、完整源码身份、权重、词表、规则、FP32及特征签名。随后执行三新组三seed共9次缓存评测。GT仅在分类与像素合成完成后读取作诊断，不传入语义生成或decision函数。

报告完整/逐图矩阵、逐类IoU、三seed均值与样本标准差、相对baseline配对百分点、纯度>=90%候选的纠正/误伤、检索错SAM对/检索对SAM错、弃权覆盖率、背景误报/前景漏检/前景类间混淆像素。语义提取与缓存评测计时/峰值显存分阶段记录。只有三个seed齐全才给组均值，VOC val仍属开发结果，不宣称显著、不自动部署。

命令：`python -B -m unittest tests.test_experimental_sam3_semantic_queue -v`；`python -B tools/experimental_sam3_semantic_queue.py --preflight-only`。后台：`nohup python -u -B tools/experimental_sam3_semantic_queue.py --version <新版本> --gpus 1 0 --max-hours 24 > logs/<新版本>_launcher.log 2>&1 < /dev/null &`；续跑相同参数加`--resume`，需旧启动器已退出且没有有效运行锁。默认GPU1/0，每卡一个worker，外部占卡等待；截止停止领取新图片/任务，已开始的评测可完成。预检保留2GiB新增空间预算及5GiB底线，运行中保持底线；不自动删除。全局输入/签名/历史复评失败停止，局部评测失败保留attempt并继续其它组，人工续跑新attempt，OOM不降配置；SIGTERM只结束自己的worker。

新产物：`storage/proposals/voc/<版本>/semantic_evidence`，`experiments/<版本>`，`logs/<版本>`；根目录清单记录在artifacts.json。历史输入身份按路径/size/SHA去重，冲突拒绝，避免继承百万条重复记录；旧签名和adapter来源检查保留。完成/失败沿用邮件通知。不改生产源码或旧实验脚本。当前脚本已写，正式运行结果另记。

## 37. 生成相机视角增强：只训练 MLP，真实区域单库保持（2026-10-07）

新增独立入口 `tools/experimental_view_augmentation_queue.py`，测试 `tests/test_experimental_view_augmentation_queue.py`。不改生产源码或旧实验。用户使用 inspatio-world-v1.5 对 trainaug 生成左右30°/60°视角；这些角度是生成条件/命名，不是已测量的真实三维相机标定。文件目录尚未提供，未上传、未启动服务器实验。

固定基线为已验证的单相位硬中心 GMP rho=0.1＋两层 MLP；关闭 HCR，查询使用原 clean 提议及 GMP 查询向量。四组：baseline（历史三个模型精确复评）、view30（±30°）、view_all（±30°/±60°等概率）、view_weighted（L30/R30/L60/R60抽样权重1/1/0.25/0.25）。有完整支持时，后两组生成样本中的60°占比分别50%/20%；按类别和实际可用角度重归一化，训练报告记录实际抽样数。这个权重控制生成样本出现频率，同时影响CE和SupCon，不是将特征乘以0.25。三seed固定42/123/2026，共9次新训练、12次完整VOC val评测，固定50×100步、不自动搜索。

每个前景类别通常取2个原图区域＋2个生成区域；背景仍全部取原图。不同区域按原图source_id分组，同一个batch同类尽量来自不同原图，避免把同一原图的四个视角当四张独立图片。少见类别不足时保留可用原图样本并记录synthetic_shortfall；不足两个生成原图家族的类别直接停止，不静默跳过。保留旧CE＋SupCon，不假设跨视角的区域具有实例一一对应关系。

只对历史内部训练图片的生成视角重跑Active伪标签并提取GMP向量，内部验证图片及其全部后代不参与训练；内部验证gallery/query都只用原图。图像级标签只限制允许类别，不强制已消失类别出现。新视角同类mask合并、类别先后优先与现有伪标签逻辑一致，前景连通区域最小300像素，不使用生成背景，不新增过滤。FP32标准ViT MLP修正在真实前向前生效，关闭BF16/TF32。共享一张图的token聚合多个区域，不持久化token。所有新MLP只导出原始真实库成员，严格检查成员不变；生成图不是新增查询或最终库成员。

命名支持 `<原图stem>_L30/R30/L60/R60.jpg/jpeg/png/webp`，大小写角度兼容、可递归子目录；同stem/view多文件拒绝。目录里的原图副本忽略，仍只读使用历史原图。正式预检要求全trainaug四视角文件完整可解码，拒绝VOC val及其后代、未知命名，绑定图片SHA。生成中使用inventory只检查清单，不读未完成图片、不占GPU。后续若使用其它命名，先调整独立入口并启用新版本。

预检从已完成 `experiments/background_partition_resume_20261006_002500` 及 `experiments/hcr_mainline_20261004_214541/prepared.json` 核验原模型/库/划分/查询缓存。旧项目绝对路径仅在新实验读取时按完整前缀映射到当前根目录，外部VOC/Conda路径不改；历史文件内容不修改，原SHA/签名/adapter来源检查继续执行。旧目录迁移不完整或源码与历史不一致时明确停止。

执行顺序：严格预检→三seed历史矩阵/逐图矩阵精确复评→一张原图GMP及一张生成图真实probe→双卡分片生成训练区域缓存→按seed42/123/2026执行三个新配置训练及全量评测。每卡最多一个worker，启动后外部占卡时等待；预检GPU需空闲。共享输入/历史复评失败全局停止，普通局部训练或评测失败保留attempt并继续；OOM不降配置、不自动重试。时间到期不领取新图/任务，已开始图或训练/评测可完成。5GiB磁盘底线及新增空间估计，不自动删除。SIGTERM只终止本启动器子进程。完成/失败沿用邮件通知，无新增明文凭据。

服务器文件同步后，生成中清点（替换真实目录）：

```bash
cd /data/home/zystu2025/nd/nd/ModuSeg_SAM3
conda activate moduseg_sam3
python -B -m unittest tests.test_experimental_view_augmentation_queue -v
python -B tools/experimental_view_augmentation_queue.py --views-dir /实际生成目录 --inventory-only
```

清点可用 `--train-list`、`--val-list` 指定外部VOC划分；正式实验使用已核验历史划分，不被这两个清点选项覆盖。

生成完再预检与后台启动：

```bash
nvidia-smi
df -h .
python -B tools/experimental_view_augmentation_queue.py --views-dir /实际生成目录 --preflight-only
VER=view_aug_$(date +%Y%m%d_%H%M%S)
mkdir -p logs
nohup python -u -B tools/experimental_view_augmentation_queue.py \
  --views-dir /实际生成目录 --version "$VER" --gpus 0 1 --max-hours 48 \
  > "logs/${VER}_launcher.log" 2>&1 < /dev/null &
echo "version=$VER launcher_pid=$!"
tail -f "logs/${VER}_launcher.log"
```

续跑相同版本和全部参数加 `--resume`，只跳过身份及产物SHA一致的完成阶段，失败用新attempt。48小时预算从首次启动算起；确需续跑增加时间时显式加 `--extend-hours 48`（每次调用会再增加48小时，不要重复用于预检或重复启动）。启动锁不自动抢占；异常断电残留锁须确认原PID及worker已退出后人工处理。

产物：`storage/pseudo_labels/voc/<版本>/generated` 新视角PNG；`storage/raw_banks/voc/<版本>/generated_regions` 区域向量/诊断缓存；`storage/mlp_banks/voc/<版本>` 原成员投影库；`weights/sam3_adapter/<版本>` 新MLP；`experiments/<版本>` manifest/status/summary/阶段指标；`logs/<版本>` 日志。artifacts.json列出本版本根目录，手工清理即可，不删除输入。报告三seed完整性、mIoU/逐类IoU/总与逐图混淆矩阵、配对差值、样本标准差、各worker耗时/峰值显存、实际角度采样统计。VOC val仍为开发集，未满三seed不报告组平均优势，不自动部署。

本地检查：无torch时保留明确跳过标记；采样、文件命名/泄漏拒绝、原图划分/导出成员钩子、缓存身份、续跑/损坏拒绝可在numpy环境测试。真实SAM3前向、完整训练及历史复评只能在服务器正式验证，不将本地通过表述成已获得精度。

第37节启动前核验补充（2026-10-08）：用户提供生成目录 `/data/home/zystu2025/nd/IW/inspatio/inspatio-world-v1.5/voc_aug/JPEGImages`，清单为同级 `trainaug_aug.txt`。新增 `--aug-list` 将该清单纳入成员及SHA核验；新增 `--validation-only`，完成三个历史全量复评、8张生成图真实提取及1轮100步短训练/导出/全量评测后停止，正式运行同版本加`--resume`复用验证产物。短训练仅验证接口，不参与四组精度比较。正式训练仍50轮×100步。

已连接并确认迁移后项目目录可访问，外部VOC与Conda旧路径仍有效；2026-10-08检查两卡空闲，/data约3.3TiB可用。52,910个文件与清单对应，四个生成角度各10,582张；正式预检全部生成图解码/SHA、历史源码/输入校验通过，内部训练可用33,864张生成视角。服务器14项专用测试全部通过（含PyTorch短轨迹）；验证版本`view_aug_20261008_checked`已进入三个历史模型复评。后续真实probe/训练检查与正式启动结果另记，当前条目不代表全量增强实验已完成。

第37节验证完成（2026-10-08）：版本view_aug_20261008_checked的三个历史模型总/逐图混淆矩阵均精确复现。原图GMP重提最大绝对误差2.9802322387695312e-08；8张生成图完成Active→GMP→缓存回读，其中一张R60无合格前景按空样本记录，未强行赋类。probe用时31.10秒、峰值allocated约4.40GiB。短训练1轮100步成功，实际抽到87个生成区域样本；导出仍为37,200个真实库成员，随后完整1449图评测及逐图矩阵校验通过。短训练成绩不参与实验对比。验证status=validated、error=null；已使用相同版本--resume发起nohup正式任务，实际Python启动器PID166294（外层shell作业166293），日志logs/view_aug_20261008_checked_launcher.log。正式提取进度另记。

第37节完成结果（2026-10-08）：服务器view_aug_20261008_checked状态complete、12/12完整评测、failures为空。关闭HCR，baseline/view30/view_all/view_weighted三seed均值85.5934%/85.7186%/85.7915%/85.7953%，相对baseline分别+0.1251/+0.1981/+0.2019个百分点。weighted三个seed均改善，较all仅+0.0038个百分点，不能宣称更优；样本标准差分别0.0778/0.1320/0.1847/0.0661个百分点。仅训练增强，真实检索库成员不变；尚未测试与HCR组合，不与86.4316% HCR基线直接比较，不自动部署。

## 38. 视角降权增强＋固定 HCR（2026-10-08）

独立入口 tools/experimental_view_hcr.py，只复用第37节已完成的view_weighted三seed模型/真实投影库与历史GMP查询/附加候选缓存，不重训或生成提议。固定HCR proposal_score=0.4、retrieval_score=0.95、margin=0.1、novel_fraction=0.1；只填背景。先精确复评历史三个HCR模型（86.4316%均值），再执行三个增强模型HCR评测，共六次完整1449图评测。历史复评矩阵不一致立即停止；保留完整签名、逐图矩阵、原图及mask身份检查。仅在独立worker内调用现有validate_semantic_rescue.worker；迁移后的原始提议位置通过主线已有--rescue-raw-dir显式指定，不改旧元数据。启动前验证历史源码/输入及新权重产物SHA，结果单独保存，不修改主线或旧实验。

已启动后台版本view_hcr_20261008，PID168265，命令 `nohup python -u -B tools/experimental_view_hcr.py --version view_hcr_20261008 > logs/view_hcr_20261008_launcher.log 2>&1 < /dev/null &`。GPU0/1每卡最多一个worker，12小时预算，空间5GiB底线，结束邮件通知。现阶段启动器进行严格预检，实际评测状态/结果以experiments/view_hcr_20261008/status.json和summary.json为准。需要续跑时同参数加--resume；不自动清理、不自动部署。

第38节完成结果（2026-10-08）：view_hcr_20261008状态complete、6/6完整1449图评测，error=null。服务器再次核验逐图矩阵求和与总矩阵完全一致，HCR参数固定，原HCR三个模型精确复评通过。原HCR/视角降权增强+HCR三seed均值86.4316%/86.5178%，配对提升+0.0863个百分点；seed42/123/2026新成绩86.4791%/86.4593%/86.6152%，配对提升+0.2068/+0.0189/+0.0331。三个seed方向均正但提升有限，不宣称统计显著；关闭HCR时增强收益+0.2019点、开启后+0.0863点，收益部分重叠，不能简单相加。仅VOC val开发结果，尚未部署或修改主线默认配置。

## 39. 同数量多视角检索扩库（2026-10-08）

独立入口tools/experimental_multiview_bank.py，专用测试tests/test_experimental_multiview_bank.py。用户授权下一步扩库实验。缓存已核对：33,864个generated_regions完成标记及33,864张生成伪标签PNG，均来自原内部训练图片。PNG位于storage/pseudo_labels/voc/view_aug_20261008_checked/generated，区域向量位于storage/raw_banks/voc/view_aug_20261008_checked/generated_regions；不重复生成或提特征。

固定view_weighted的三seed MLP，不重训；四组real/duplicate/view30/view_weighted。real为原37,200成员；duplicate复制同类同原图已有真实向量，是数量/密度对照；view30加入缓存L30/R30区域；view_weighted按角度1/1/0.25/0.25无放回抽取L30/R30/L60/R60区域。每个原图家族/类别的新增配额=min(原真实区域数,可用30度区域数)，三个新组按每个家族/类别严格匹配数量，背景完全保持原库。生成区域不与真实区域宣称实例对应，只用source_id和伪类别控制数量。选择seed=42固定，与MLP种子无关，三个模型复用同一选择计划。新图不进内部验证或VOC val训练。没有30度支持的家族不扩展，明确报告覆盖。

每组均评测clean及原固定HCR，三seed，共24次全量1449图评测；先精确复现real的6个历史结果（clean85.7953%、HCR86.5178%）后再扩库。HCR仍为proposal=.4、retrieval=.95、margin=.1、novel_fraction=.1；local/center融合、候选和像素合成不调参。对比view30/weighted与duplicate判断视角覆盖，duplicate与real判断数量效果；weighted与view30判断增加60度作用。只比较VOC val开发成绩，不宣称显著、不自动部署。

服务器4项测试全通过，含真实d2模型、实际缓存区域向量的投影/导出试验，原行保留、背景保留、类数量匹配核验通过。本地3项通过、真实模型测试按条件跳过。新版本multiview_bank_20261008后台启动；GPU0/1每卡一worker、24小时预算，原输入/源码/缓存/模型阶段SHA核验、严格特征签名及adapter来源保留。新库使用相同特征空间签名，独立meta/multiview_gallery.json和阶段SHA绑定成员方案。补丁仅worker内复用旧HCR缓存评测入口；不改旧实验、原标签或主线。启动日志logs/multiview_bank_20261008_launcher.log，产物storage/mlp_banks/voc/<版本>/<seed>/<方法>/<attempt>、experiments/<版本>、logs/<版本>；selection.json保存家族配额/角度/特征行，artifacts.json列出人工清理根目录。结束邮件通知，5GiB底线加3GiB预算；截止不领取新任务；SIGTERM只终止自己的worker；输入损坏全局停止，普通局部新组失败可继续，人工--resume使用新attempt，不自动降精度或删除。

启动命令：python -u -B tools/experimental_multiview_bank.py --version multiview_bank_20261008 --max-hours 24。续跑使用同一参数加--resume，必须确认旧启动器退出、锁未被占用，身份和产物SHA相同才复用。

## 40. 与代码对齐的论文框架图（2026-10-08）

新版位于experiments/paper_framework_20261008_v3/output，含中英文两页可编辑PPT、英文矢量PDF、两种语言PNG；代码对应说明位于同版本code_alignment.md。延续上一版ModuSeg风格，将Active掩码与图像编码器合并到同一冻结SAM3系统，两输出共同进入GMP；增加离线区域类别→MLP监督、过滤位置、在线clean+extra pool统一提取/检索、NMS→HCR已有前景占用及只补背景关系。使用既定GMP＋两层MLP＋HCR论文配置，生产默认SMFA及独立多视角实验不被改写。当前缓存流程分阶段执行SAM3掩码/特征前向，该实现细节放备注，图不宣称一次前向同时返回两者。最终PPT结构/字体/几何校验通过，两页重导入渲染和PDF均人工检查；结果图片属示意。仅生成本地文件，未改生产源码或运行中任务。

第39节完成结果（2026-10-08）：服务器multiview_bank_20261008状态complete，24/24完整1449图评测、failures为空；再次核验逐图矩阵总和与总矩阵一致。三个扩库每组新增18,613行，总55,813行，配额覆盖10,985个原图家族/类别组合；seed42原37,200行逐元素保持，两个视角库新增行均不同于重复对照。固定训练增强MLP，real/duplicate/view30/view_weighted的clean均值85.7953%/85.7810%/85.7837%/85.6556%，HCR均值86.5178%/86.5091%/86.5221%/86.3852%。30度扩库相对真实库HCR仅+0.0042点且seed方向不一致；含60度扩库HCR -0.1327点。weighted seed2026羊/牛IoU分别-5.2618/-3.1687点，具体原因尚未溯源。保留训练增强+真实库+HCR，不自动并入扩库方案，不宣称显著。

## 41. 按家族组织多视角证据（2026-10-08）

独立入口tools/experimental_family_set.py，专用tests/test_experimental_family_set.py。固定增强训练MLP、GMP查询缓存及上一轮同一30度扩库55813行，关闭HCR，不重训、重生成、扩增或过滤成员，不使用60度。六组real_flat（真实库原检索）、view30_flat_reference（原30度扩库检索及其旧中心）、view30_flat_fixed（30度扩库原检索但固定真实库中心）、real_set（真实库家族集合）、view30_family_set（同一30度库真实家族集合）、view30_random_set（同类同视角打乱家族）。三seed42/123/2026，共18次完整1449图评测。

先精确复评前两组三seed总/逐图矩阵；新组中心均固定真实库中心，局部/中心仍0.6/0.4。所有检索复用原top210向量候选范围；集合方法从该候选中取集合ID，读取每个候选集合全部成员得分，避免只计算被检索到的视角。集合内按视角等质量，再在每视角内部平均exp相似度，使用数值稳定logmeanexp，temperature=.05预先固定。每类取最强5个不同家族集合均分；不存在该类候选时仍以该类原中心分兜底。背景逐区域单成员，保持背景证据定义。成员分组使用原sample_id及旧selection.json的生成来源/类别/视角，严格核对实际行/ID和原行不变，不宣称实例对应。随机控制在每个类别/视角内置换组ID，精确保留各集合类别、大小和视角组成，seed42与MLPseed无关。

只在worker包装分类调用，主线源码/旧实验不改。查询和库签名/模型来源、历史阶段产物SHA、库行顺序严格核对；候选分数、mask哈希、margin及集合候选数量记录retrieval_trace.json。全部预测及最终评测完成后才读取GT作纯度>=.9的事后候选分类、纯背景误报、前景漏检与类间混淆统计；GT不参与分组/参数/预测。报告原图分组作用、中心变化作用、同向量flat→set作用、多视角vs原图集合、真实vs随机分组配对差值；未满三seed不报平均优势，不宣称显著，不自动部署。

服务器5项测试全通过，含真实55813/37200行库顺序与随机分组验证；本地4项通过，真实历史测试按条件跳过。后台新版本family_set_20261008，入口nohup python -u -B tools/experimental_family_set.py --version family_set_20261008 --max-hours 24。GPU0/1每卡一个worker，24小时预算，磁盘5GiB底线、截止停止领取新任务，失败保留attempt；全局输入损坏停止，普通局部新组失败其他可继续，OOM不降配置；SIGTERM只停止自己的进程。身份一致续跑用--resume，结束邮件通知。产物仅experiments/<版本>和logs/<版本>，不新增特征库；artifacts.json列出人工清理根目录。

第41节完成结果（2026-10-08）：服务器family_set_20261008 complete、18/18完整1449图评测、failures为空；再次核验逐图矩阵求和等于总矩阵。六组real_flat/view30_flat_reference/view30_flat_fixed/real_set/view30_family_set/view30_random_set均值85.7953%/85.7837%/85.7810%/85.8411%/85.8412%/85.8384%。原图集合+0.0458点、三seed均正；30度相比原图集合仅+0.000126点，真实家族比随机仅+0.002764点且seed方向不一致。随机分组80.86%成员改变归属，类别/大小/视角组成守恒。每seed3609查询，新视角集合相比原图集合判类仅变1/0/1个，平均分数绝对变化约0.00130/0.00136/0.00140，新增证据确实参与但当前判类作用很小。关闭HCR，不能与86.5%直接比较；不自动部署或追加过滤/60度搜索。


## 42. 原图最高纪录主线与GitHub发布（2026-10-08）


当前主线：Active SAM3 → 单相位硬中心 GMP（ρ=0.1）→ 两层 MLP
（1280→512→256）→ 单一投影库 → 已清洗候选检索 → HCR 只补背景。
Active 根据训练图片的图像级标签选择类别，并增加类别上下文提示。
推理候选使用完整 VOC 前景词表，不输入验证图片真值类别。

GMP只选择中心落在掩码内的stride-14 patch；非空小目标没有命中时取
距区域中心最近的patch。选中token等权求二阶统计与正则线性系统，
ρ=0.1，FP32 Cholesky；不使用覆盖率加权或SMFA回退。SAM3冻结，
标准ViT MLP精度修正在第一次探针前生效，BF16/TF32关闭。

训练：v1配方、50×100步、8类×4区域、CE+0.5 SupCon；仅内部训练图片
划分选模。过滤前景最远10%，背景不删；每种算法训练匹配的MLP。
历史纪录沿用SMFA过滤后的37,200个固定区域成员；新建GMP库再过滤
可能改变成员，不能把历史分数直接归给新库。

检索：balanced，K=25，每类最强5个邻居，搜索倍数2；局部/类别中心
权重0.6/0.4。实例分数=0.8检索分+0.2 SAM3分，NMS=0.4，实例阈值0.02。
HCR：提议分≥0.4，检索分≥0.95，类别间隔≥0.1，新像素比例≥0.1。
只有一个特征库，类别中心从该库计算。

## 从头使用正式入口

先提供原图、trainaug/val划分、训练图像级类别和本地SAM3权重。
所有版本名使用新值；以下占位路径替换为真实路径。

```bash
export CUDA_VISIBLE_DEVICES=1
export OVERRIDE_DATASET_ROOT=/path/to/VOC2012
export OVERRIDE_SAM3_CKPT=$PWD/weights/sam3/sam3.pt
export OVERRIDE_SAM3_POOLING=hard_center_gmp
export OVERRIDE_SAM3_USE_BF16=false

# 已有完整Active标签时只读复用，不必重复生成。
python -B tools/generate_pseudo_masks_sam3.py --help
# 按实际CLI指定新输出和训练图像级JSON。
export OVERRIDE_PSEUDO_MASK_ROOT=$PWD/storage/pseudo_labels/voc/YOUR_ACTIVE_VERSION
export OVERRIDE_OUTPUT_ROOT=$PWD/storage/raw_banks/voc/YOUR_GMP_VERSION
python -B run_build_feature_bank.py --strict-build

python -B tools/train_feature_adapter.py \
  --source-bank "$OVERRIDE_OUTPUT_ROOT" \
  --train-list /path/to/VOC2012/ImageSets/Segmentation/trainaug.txt \
  --output-dir weights/sam3_adapter/YOUR_VERSION/42 \
  --export-bank storage/mlp_banks/voc/YOUR_VERSION/42 \
  --recipe v1 --num-layers 2 --seed 42

python -B tools/run_mainline.py \
  --dataset-root /path/to/VOC2012 \
  --bank storage/mlp_banks/voc/YOUR_VERSION/42 \
  --adapter weights/sam3_adapter/YOUR_VERSION/42/adapter.pt \
  --clean-proposals-dir storage/proposals/voc/YOUR_CLEAN_VERSION \
  --raw-proposals-dir storage/proposals/voc/YOUR_RAW_VERSION \
  --output-dir experiments/YOUR_NEW_EVAL_VERSION --gpu 1
```

正式入口默认HCR，`--no-hcr`用于clean消融。它不重新清理clean缓存，
也不覆盖已存在的输出目录。严格复现历史纪录应使用下节固定成员库
导入；上面的全新建库按GMP重新过滤，属于新实验。

## 历史纪录导入与服务器回归

`tools/promote_original_mainline.py`只接受历史单相位硬中心GMP C＋d2
权重。校验完整来源签名、基础权重及adapter哈希，在新版本复制库，
保持区域向量与状态字典不变，显式转换为正式签名。不覆盖历史数据。
普通加载器仍拒绝实验签名，不允许关闭签名检查来迁移。

执行一次真实SAM3/GMP特征探针，再用已核验历史聚合向量做六个1449张
完整正式检索/HCR回归，比较整体和逐图混淆矩阵。这是部署等价性
验证，不是新精度实验，也不是全量重跑SAM3的速度测试。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sam3-main python -B \
  tools/promote_original_mainline.py \
  --root /data/home/zystu2025/nd/nd/ModuSeg_SAM3 \
  --version YOUR_NEW_MAINLINE_VERSION
```

历史路径仅在读取JSON时重定位，不改写旧报告。不同算法、ρ、基础
权重、精度或MLP权重不能混用。`tools/run_mainline.py`使用正式SAM3
在线提取，不依赖历史实验模块或向量缓存。

## 提议、存储与验证

clean缓存`storage/proposals/voc/active_clean_20260928_151808`对应raw
`storage/proposals/voc/pmfa_dual_20260923_162706/attempt_001`。
清理配方0.7/100/64/32/0.01/0.85；HCR附加池在内存以min_score=0
处理，其余配方相同。不要再次清理clean缓存。

storage存数据与库，weights存模型，experiments存参数指标，logs存日志。
GitHub只提交代码、测试、文档、必要SAM3词表及许可证。邮箱授权码通过
环境变量传入，本地`.env`不提交。多视角实验继续保留各自版本，不入主线。

```bash
PYTHONPATH=sam3-main python -B -m unittest \
  tests.test_mainline_gmp tests.test_feature_signature \
  tests.test_retrieval tests.test_proposal_cleanup tests.test_semantic_rescue \
  tests.test_sam3_precision -v
```

86.4316%是无多视角HCR三seed开发成绩；最高单seed为86.5822%，不要
将二者混称。新开发实验与最终独立测试分开报告。

通知功能仍保留：除环境变量外，可读取用户主目录下`.config/moduseg/notify.json`私有配置（password/sender/recipient），服务器文件权限设600，不进入项目或GitHub。显式设置空的MODUSEG_SMTP_PASSWORD可关闭通知。

2026-10-08部署验收完成：真实查询图2007_000033的4个区域向量与历史缓存误差为0；六个1449张clean/HCR回归的整体及逐图混淆矩阵完全一致。部署后46项测试通过；37,200成员原库与三套MLP库/权重均另存为`mainline_gmp2_hcr_20261008`正式签名。详细摘要见同目录`mainline_validation.json`。

## 43. 原图 Active/GMP/MLP2/HCR 双卡从头完整实验（2026-10-10）

入口 `tools/run_mainline_full.py`，专用测试 `tests/test_mainline_full.py`。
固定原图VOC trainaug=10,582、val=1,449，两张物理GPU0/1，每卡最多一个worker。
无历史输入：重新生成训练Active标签、验证Active原始提议，独立清理一次；
重新提取单相位硬中心GMP rho=.1、1008、trunk_fpn、FP32特征，再全库统一过滤前景10%、背景0%。
训练seed42普通两层1280→512→256 MLP，v1、50×100步、8类×4区域；内部图片划分重新生成。
最终两卡分片在线提取GMP并完成HCR评测，混淆矩阵相加后计算IoU，不能平均两卡mIoU。

新增FP32提议生成修正在第一次图像前向之前替换供应商ViT MLP内部强制BF16路径。
验证提议使用完整20类VOC词表（Active仍按存在分数选择活跃类别），不读取验证图像级类别。
所有面积保持原图像素坐标，参数来自当前PSEUDO/TRAINING/CLEANUP，在线提议沿用Active导出默认配方并显式FP32。

执行前先查看真实CLI及运行测试：

```bash
cd /data/home/zystu2025/nd/1010-main
conda activate moduseg_sam3
PYTHONPATH=sam3-main python -B -m unittest tests.test_mainline_full tests.test_experiment_notify tests.test_mainline_gmp tests.test_feature_signature tests.test_retrieval tests.test_proposal_cleanup tests.test_semantic_rescue tests.test_sam3_precision tests.test_bulk_active_proposals tests.test_pseudo_two_gpus tools.proposal_demo.test_active
python -B tools/run_mainline_full.py --help
python -B tools/run_mainline_full.py --version YOUR_NEW_VERSION --preflight-only
python -u -B tools/run_mainline_full.py --version YOUR_NEW_VERSION --probe-only
```

只读预检全量解码并SHA256绑定图片、GT、清单、训练图像级JSON、本地权重和代码；
检查GPU占用（显存不超过1000MiB、利用率不超过10%）及保守存储估计，并保留5GiB。
`--probe-only`创建新版本及代码快照，两卡分别完成真实训练标签、验证提议、GMP前向，
用真实GMP向量配合明确标记的合成诊断类别运行短MLP训练/导出和一图严格HCR推理。
探针的synthetic类别和指标只检查接口，不进入正式库、不报告为实验精度。
成功状态ready，探针不发送完成邮件。随后同版本`--resume`复用已核验探针，开始全量生成：

```bash
mkdir -p logs
# YOUR_NEW_VERSION 与上面的探针版本一致；日志/PID均使用新名字。
nohup python -u -B tools/run_mainline_full.py --version YOUR_NEW_VERSION --gpus 0 1 --seed 42 --resume > logs/YOUR_NEW_VERSION_launcher.log 2>&1 < /dev/null &
# 记录启动器PID，避免覆盖历史PID文件。
```

不用单独探针时直接运行新版本（不带--resume），入口也先完成全部真实探针，失败不进入全量阶段。
无整体时间截止。每阶段开始重新核验输入SHA，失败阶段保留attempt；后续相同代码/参数/输入可--resume。
标签和提议沿用本次分片缓存；GMP特征逐图原子NPZ及绑定签名/输入哈希的receipt，只复用校验通过的检查点。
不完整/被修改的GMP检查点拒绝重用，不静默覆盖。清理、合库、训练、评测失败使用新attempt。
完成阶段凭据绑定产物SHA256，不通过时拒绝续跑。代码/参数改变必须新版本。

storage/pseudo_labels/voc/<版本>保存训练PNG及分片meta；storage/proposals/voc/<版本>/raw与clean/attempt_NNN保存原始及清理NPZ。
storage/raw_banks/voc/<版本>/image_cache保存逐图向量，bank/attempt_NNN保存合并过滤库；
weights/sam3_adapter/<版本>/42/attempt_NNN和storage/mlp_banks/voc/<版本>/42/attempt_NNN为正式权重及导出库。
probe/attempt_NNN/gpuN属于独立诊断产物。experiments/<版本>/code保存不可变代码快照，worker从快照运行，
storage/runtime/<版本>保存/data上的临时目录和模型缓存。logs/<版本>/<阶段>/<attempt>/worker_N.log保存worker日志。

experiments/<版本>保存manifest、artifacts、progress、status、summary、metrics、per_class.csv及各阶段attempt；
worker报告含物理GPU、耗时及本进程峰值显存。最终summary区分训练标签质量与1,449张HCR语义mIoU。
只有全量标签、raw/clean提议、50轮训练、全量严格评测全部成功才complete，不承诺复现历史86.4316%。

通知仅由父启动器在最终状态落盘后发送，正文含mIoU、21类IoU、数量和最佳轮次；
附件只含summary.json、metrics.json及per_class.csv，失败附已存在的实际结果，不打包数据模型。
邮箱配置可通过MODUSEG_NOTIFY_CONFIG指向主目录私有JSON，或沿用MODUSEG_SMTP_*环境变量。
凭据不进入实验manifest、代码快照或命令记录。SMTP最多尝试3次，失败独立记录notification.json；
SMTP accepted不代表确认收件箱投递。SIGTERM只停止本启动器拥有的worker进程组，禁止pkill python；
断电/SIGKILL无法保证邮件，遗留锁须确认原启动器及worker均退出后人工处理。

2026-10-10服务器实施与启动记录：新增完整入口及缓存严格校验，65项相关测试全部通过。
两卡真实Active/GMP/短MLP导出/严格HCR探针均complete，每卡探针峰值分配5230.60MiB，
探针缓存属diagnostic_only，不作为真实精度。新实验版本`mainline_full_20261010_172757`，
正式后台启动器PID198857，GPU0/1伪标签worker为198879/198880，均从不可变代码快照运行。
已核验运行状态running/pseudo，首批训练PNG及逐图哈希完成记录正在增加。
启动日志`logs/mainline_full_20261010_172757_launcher.log`，PID文件同名前缀`_launcher.pid`。
最终结果位于`experiments/mainline_full_20261010_172757`；当前没有完整实验指标。
结果邮件配置已通过SMTP认证，最终评测及结果落盘后由父启动器发送摘要及三个结果附件。
标签/提议worker启用每图磁盘保留检查、逐图完成凭据和PNG/NPZ SHA校验；
被修改的缓存拒绝重用，不自动覆盖原缓存。新手册变更不影响正在运行的代码快照。

## 44. 已知类别 presence=1＋逐类别边缘修复＋整图特征缓存（2026-10-10）

本轮用户明确指定的新实验，不覆盖第43节版本或其产物。完整入口新增可选参数
`--edge-repair --force-presence-one --save-image-features`，默认不启用，旧入口行为保持。
概率阈值参数名`--mask-logit-threshold`，语义始终为sigmoid后的概率阈值，默认0.4，可显式0.25。
SAM3 Active直接读取forward_grounding的原始pred_masks logits，上采样到原图后sigmoid、按阈值二值化，
沿用实例分数/面积/NMS筛选，再对同类别实例取并集，每类别分别进行close与轻微dilate，最后合成伪标签。
close为先膨胀再腐蚀，普通类kernel3/iterations1/dilate1；VOC的table对应diningtable，默认iterations2。
每类别mask是uint8的0/1，最终VOC语义PNG仍0..20，重叠仍由图像级允许类别的原顺序先到先占。
纯NumPy模块src/mask_edge_repair.py的dilate外部padding=False、erode外部padding=True，遵照用户代码。

presence=1仅用于已知训练图像级类别：关闭Active预览筛类，全部允许前景类均解码，
实例分数=object sigmoid×1，保留原实例conf_thr；不是把原始presence logit设为1（后者概率只有约0.731）。
强制存在模式要求明确class_indices，不能对无已知类别的全词表提议启用。
验证提议依旧使用完整VOC词表、正常预测presence/Active预览，不读取验证图像级类别或GT指导筛选。
新选项/类别阈值/修复配置写入伪标签配置、指纹和manifest，不能续接旧版标签。

整图缓存位于storage/image_features/voc/<新版本>/{train,val}/<image_id>.npz及对应JSON。
保存的是本项目GMP实际使用的trunk_fpn整图空间特征：72×72×1280、FP32，融合/归一化后、GMP和MLP之前。
NPZ不压缩以保持快速写入，原始浮点值无损；包含tokens和grid，JSON绑定原图SHA、实际RGB SHA、
尺寸、基础权重及编码配置、文件SHA。每图约25.3MiB，12,031图约297GiB；启动估算额外计入缓存与余量。
不保存全部Transformer层中间状态。缓存签名独立于掩码/GMP rho/MLP，便于今后显式复用相同编码空间。
生成训练GMP向量和验证在线编码时保存缓存；同图共享内存tokens，恢复时可核验并直接读取。
即便某图无有效区域或候选也保存其整图特征。坏文件、改图、配置不匹配或不完整NPZ/JSON对拒绝使用，保留原文件。
逐图区域向量、原始库、MLP库仍按第43节目录保存，完成阶段绑定整图缓存SHA并严格核对全量ID。

例：先执行新增及原有测试，再用新版本做两卡真实探针；探针包括落盘读回后的GMP向量一致性检查。

```bash
PYTHONPATH=sam3-main python -B -m unittest tests.test_edge_feature_cache tests.test_mainline_full tests.test_mainline_gmp tests.test_feature_signature tests.test_pseudo_mask_evaluation tools.proposal_demo.test_active
python -u -B tools/run_mainline_full.py --version YOUR_NEW_EDGE_VERSION --edge-repair --force-presence-one --save-image-features --probe-only
# 探针ready后，同参数、同版本追加--resume进行全量后台运行；启动日志使用新名字。
```

仍固定seed42、原图GMP rho=.1、MLP2 v1 50×100、单库和HCR；伪标签、原始提议、清理、原库、MLP及评测均新建。
结束通知沿用用户指定QQ邮箱配置，附summary/metrics/per_class.csv；图像特征不进入邮件，不自动打包。

第44节服务器启动记录：72项相关测试全通过，两卡真实探针ready，均对餐桌图片2007_000250
确认允许类别bottle/table存在概率1，餐桌close_iterations2；餐桌通道49402→51917像素，
以上只是通道覆盖诊断，不是精度提升。每卡保存5份真实整图NPZ，形状72×72×1280、FP32，
落盘读回GMP向量一致性检查通过；短训练/导出/严格HCR流程通过。
全新版本mainline_edge_presence1_features_20261010_224233，后台启动器PID200757，
双卡伪标签worker 200779/200780，状态running/pseudo。估算新增433.2GiB，预检/data空闲约3334GiB。
运行日志logs/mainline_edge_presence1_features_20261010_224233_launcher.log；
结果experiments/mainline_edge_presence1_features_20261010_224233；整图缓存storage/image_features/voc/同版本/train和val。
上一轮mainline_full_20261010_172757已complete：1449张HCR mIoU86.5016%，最佳MLP轮次44，
SMTP首次接受结果通知及三个附件。旧输入、产物和代码快照不改写；新版本最终指标尚未获得。

## 45. 第二轮语义伪标签几何清理与缓存复用消融（2026-10-11）

入口 `tools/run_pseudo_cleanup_ablation.py`，测试 `tests/test_pseudo_cleanup_ablation.py`。
第44节实验已complete：伪标签mIoU86.2840%，完整HCR mIoU86.2694%，MLP最佳13轮，实际训练满50轮。
用户要求在第二轮伪标签之后加入在线提议清理，复用可用产物，再做完整建库/训练/评测。
第二轮只保存合成语义PNG及提示记录，没有逐实例mask/score；本入口明确是几何清理消融。
复用 `src.proposal_cleanup.clean_mask`：每前景类别单独删除小于64像素的8连通小岛，
填不超过32像素且不超过该类别面积1%的封闭孔洞，再执行该类别100像素面积门槛。
孔洞只可填输入背景，不覆盖其它原始前景类别；语义标签仍0..20。背景由清理后前景的补集给出。
不虚构实例分数，不声称执行0.7分数过滤或逐实例NMS；已有语义通道互斥，也不做跨类NMS。
清理不腐蚀边缘，不保证能纠正上一轮已过度膨胀的轮廓。GT仅用于生成后的质量统计和最终评测。

复用源必须是complete的第44节实验，原标签、原始及clean验证提议、整图FP32 token缓存、
过滤前逐图GMP向量全部只读。新标签放storage/pseudo_labels/voc/<新版本>；
新逐图区域检查点和一次全库前景10%过滤后的库放storage/raw_banks/voc/<新版本>；
新MLP权重放weights/sam3_adapter/<新版本>/42；新投影库放storage/mlp_banks/voc/<新版本>/42。
不复制约297GiB整图缓存，不重新运行SAM3编码器或生成验证提议。owned_roots和referenced_roots明确分开。
训练图像的类别和GMP选中token索引完全相同时，原始区域向量可逐区域复用；
改变token支持的区域用已保存整图特征重新做相同FP32 GMP及二次L2归一化。
重新合并所有未过滤区域后只过滤一次，不直接拼接旧过滤库，不复用旧MLP权重。
验证重算GMP并应用本次新MLP，读取上一轮raw/clean提议，保持检索和HCR参数。

预检核验源完成凭据、全部图片/GT/清单/权重SHA、PNG/逐图GMP/验证提议SHA、
全量token文件ID/大小和JSON凭据SHA/编码签名；大token NPZ实际SHA在使用前核验。
`CachedGMPExtractor`严格只读；缺失或损坏缓存立即失败，不回退生成或修改历史。
固定第二轮内部训练/验证图片成员，用--split-file训练seed42、MLP2 v1、50×100步；
训练后再核对两组图片成员完全一致，1449张严格HCR混淆矩阵相加后统计mIoU。
每图新检查点绑定源SHA、清理配置和代码摘要；阶段完整性凭据、不可变代码快照、
新attempt恢复及5GiB空间底线沿用完整主线。无整体时间到期，不覆盖历史输出。

```bash
PYTHONPATH=sam3-main python -B -m unittest tests.test_pseudo_cleanup_ablation tests.test_edge_feature_cache tests.test_mainline_full tests.test_proposal_cleanup
python -B tools/run_pseudo_cleanup_ablation.py --version YOUR_NEW_VERSION --preflight-only
python -u -B tools/run_pseudo_cleanup_ablation.py --version YOUR_NEW_VERSION --probe-only
# 双卡探针通过后，同版本固定代码/配置/输入，再后台--resume：
nohup python -u -B tools/run_pseudo_cleanup_ablation.py --version YOUR_NEW_VERSION --resume > logs/YOUR_NEW_VERSION_launcher.log 2>&1 < /dev/null &
```

探针对实际清理标签的向量复用与重新计算做1e-6一致性核对，两卡分别完成短MLP导出和严格HCR联通；
探针标签/区域/库仍分别位于storage下，诊断权重位于weights，各根目录使用probe/attempt_NNN/gpuN；
探针参数、指标和日志位于experiments和logs，不混入正式库。
短训练的合成诊断类别不进入正式库、不报告为实验精度。结果附清理像素统计、向量复用数量、
伪标签质量、库统计、最佳轮次、完整1449张总体/逐类IoU及对第44节源实验的差值。
结束或失败后沿用用户指定QQ通知配置，附件summary.json、metrics.json、per_class.csv，不打包特征或模型。

第45节启动验收：78项相关测试通过，目录调整后新增8项再次通过。用户确认采用几何清理及缓存复用。
新版本`mainline_edge_cleanreuse_20261011_114857`，启动器PID203763，后台脱离SSH。
全量源输入/区域/提议SHA及12,031份token凭据预检通过；两卡短流程均complete，
每卡4图14个区域中11个向量复用、3个重新计算，复用与直接计算的最大误差为0。
诊断HCR子进程峰值分配82.16MiB，指标仅为接口验证。
验收时状态running/pseudo_cleanup，CPU分片worker 204071/204072均存活，日志推进；
已有5,959份正式标签完成凭据，抽检20张PNG的哈希、尺寸、类值和代码绑定均通过。
后续建库/评测使用物理GPU0/1，50轮MLP训练使用GPU1。正式结果尚未获得。
启动日志`logs/mainline_edge_cleanreuse_20261011_114857_launcher.log`；
结果目录`experiments/mainline_edge_cleanreuse_20261011_114857`，含deployment_validation.json。
早期`mainline_edge_cleanreuse_20261011_114355`只完成短流程诊断，状态ready，未进入全量实验；
诊断历史保留，正式版本按storage/weights分类存储探针产物，两个版本不混用。

第45节完成验收（北京时间2026-10-11 12:45:28）：正式版本complete，50轮训练与全部1,449张
严格HCR评测通过；再次核对两卡图像ID互斥且完整、混淆矩阵之和等于总矩阵、指标/训练报告SHA。
清理后伪标签mIoU86.2884%，较第二轮86.2840%增加0.0045个百分点。
最终HCR mIoU86.7350%，比第二轮86.2694%增加0.4656点，比第一轮86.5016%增加0.2334点。
内部5-NN原始宏准确率仍92.8271%，新MLP最佳宏准确率98.3774%、最佳38轮，末轮loss0.601585。
内部训练/验证图片成员和raw/clean验证提议SHA均与第二轮一致。公交、火车、鸟IoU分别
增加3.5642/3.6544/1.7807点；公交→火车混淆从157,082像素降为0；汽车下降1.4864点。
清理改变5,285张图、350,017像素，删除226,198小岛像素，填123,648背景孔洞像素，
保护7,281其它前景像素，33个类别通道未达到100像素门槛。过滤后前景仍24,311个区域。
原始GMP向量复用32,699个，重算4,872个；整图缓存未复制，SAM3编码器未重跑。
主流程耗时3,269.64秒（约54分30秒，不含初始化预检）；QQ SMTP首次接受结束通知及
summary.json/metrics.json/per_class.csv三个附件，不声称确认收件箱投递。
这是固定seed42的单次清理对照结果，未自动替换默认主线或追加多seed实验。

## 46. 当前正式主线与GitHub同步（2026-10-11）

用户明确将第45节完成版本作为正式主线。仓库为 `https://github.com/nanding0414-spec/1010`。
当前模型版本 `mainline_edge_cleanreuse_20261011_114857`，VOC val单seed42 mIoU86.7350%，
训练50×100步、内部最佳38轮、完整1,449图严格HCR；这是开发验证成绩，不是独立测试或三seed均值。
历史原始标签、提议、库、模型、指标及代码快照不改写。

正式配方：训练图片的已知类别presence概率固定1，全部允许类别解码；SAM3原图logits上采样并sigmoid，
概率阈值0.4，普通类3×3 close1/dilate1，餐桌close2/dilate1。同类实例合并、原类别顺序合成语义PNG后，
再逐类别做几何清理：去小于64像素的8连通小岛，填≤32像素且≤类别面积1%的封闭孔洞，类别面积门槛100。
填洞仅写原背景、不覆盖其他类别，不伪造语义PNG不存在的实例分数，不对互斥语义通道做跨类NMS。
验证提议保持全词表、正常预测presence及原清理配方0.7/100/64/32/0.01/0.85，不使用验证真值类别。
冻结SAM3，原图1008/trunk_fpn/FP32，单相位硬中心GMP rho=.1，原库前景最远10%过滤一次、背景保留。
MLP2 1280→512→256、v1/seed42、50×100步，按训练图片内部验证5-NN宏准确率选模。
单库balanced检索K25/每类top5/中心权重0.4，加原HCR参数，仅补背景。

`configs/experiment.py`的PSEUDO默认启用presence1和边缘修复；PSEUDO_CLEANUP明确几何参数。
`tools/run_mainline_full.py`默认启用独立语义清理阶段和训练/验证整图FP32缓存。
新主线标签目录为storage/pseudo_labels/voc/<版本>/raw与clean/attempt_NNN；原始PNG始终保留。
`--no-edge-repair --no-force-presence-one --no-semantic-cleanup --no-save-image-features`用于显式基线消融。
新默认不能续接旧实验；旧实验恢复只使用当时的不可变代码快照。

### 双卡从头完整运行

```bash
cd /data/home/zystu2025/nd/1010-main
conda activate moduseg_sam3
PYTHONPATH=sam3-main python -B -m unittest tests.test_mainline_promotion tests.test_pseudo_cleanup_ablation tests.test_edge_feature_cache tests.test_mainline_full tests.test_mainline_gmp tests.test_feature_signature tests.test_retrieval tests.test_proposal_cleanup tests.test_semantic_rescue
python -B tools/run_mainline_full.py --version YOUR_NEW_VERSION --preflight-only
python -u -B tools/run_mainline_full.py --version YOUR_NEW_VERSION --probe-only
# ready后，同代码/输入/参数恢复，日志使用新名字；此命令必须手动给新版本。
nohup python -u -B tools/run_mainline_full.py --version YOUR_NEW_VERSION --gpus 0 1 --resume > logs/YOUR_NEW_VERSION_launcher.log 2>&1 < /dev/null &
```

新完整入口分开生成raw标签、清理标签、评价清理后标签；建库仅用clean。
大token缓存约297GiB，预检仍估算额外空间并保留5GiB；保存逐图SHA、原始区域向量、原库及投影库。
运行流程/失败新attempt/阶段校验/邮件通知仍按第43–45节约定。
若只对现有raw语义标签清理，独立入口为：

```bash
python -B tools/clean_pseudo_labels.py --input-dir storage/pseudo_labels/voc/YOUR_RAW_VERSION --output-dir storage/pseudo_labels/voc/YOUR_NEW_CLEAN_VERSION --image-list data/VOC2012/ImageSets/Segmentation/trainaug.txt
```

### 激活已验证模型并推理

`tools/activate_mainline.py`只接受完成50轮、10582训练图与1449验证图、配方相符的seed42版本。
核验已登记库/MLP/基础权重/验证提议SHA及匹配签名，不复制或重训模型。
版本化激活记录放experiments/mainline_selections/<模型版本>.json，当前选择写experiments/mainline_active.json。
这些是服务器运行产物，不提交GitHub；正式推理入口只在缺少显式资源路径时读取激活记录。

```bash
python -B tools/activate_mainline.py --version mainline_edge_cleanreuse_20261011_114857
python -B tools/run_mainline.py --output-dir experiments/YOUR_NEW_EVAL_VERSION --gpu 1 --preflight-only
python -u -B tools/run_mainline.py --output-dir experiments/YOUR_NEW_EVAL_VERSION --gpu 1
```

新安装没有激活记录时，显式提供资源：

```bash
python -u -B tools/run_mainline.py --bank storage/mlp_banks/voc/YOUR_VERSION/42/attempt_001 --adapter weights/sam3_adapter/YOUR_VERSION/42/attempt_001/adapter.pt --checkpoint weights/sam3/sam3.pt --dataset-root data/VOC2012 --clean-proposals-dir storage/proposals/voc/YOUR_PROPOSALS_VERSION/clean/attempt_001 --raw-proposals-dir storage/proposals/voc/YOUR_PROPOSALS_VERSION/raw --output-dir experiments/YOUR_NEW_EVAL_VERSION --gpu 1
```

服务器当前被选模型使用第45节的新库/新MLP及第44节的raw/clean验证提议；可复用整图缓存的实验
仍可显式使用tools/run_pseudo_cleanup_ablation.py，但它是复用对照入口，不取代从头完整入口。
GitHub更新只含代码、测试、配置、必要SAM3词表/源码/许可证及文档；.gitignore继续排除data/storage/
weights/experiments/logs和通知凭据，不自动打包。默认分支更新采用已有提交的后继提交，保留全部历史。

第46节发布验收：86项相关测试全部通过。新默认双卡真实探针
`mainline_releasecheck_20261011_132203`达到ready，完整验证raw标签→几何清理→真实GMP→短MLP导出→严格HCR；
两卡生成的餐桌图片2007_000250清理后PNG均与第45节实际标签逐像素完全一致。
第45节正式库和MLP已通过2,930个已登记输入/控制文件的SHA及签名核验，
激活为experiments/mainline_active.json的当前选择；版本化激活记录保留在mainline_selections下。
原库、原权重、原实验指标和快照未复制或改写。发布清单覆盖388个代码/文档/词表/许可证文件。
