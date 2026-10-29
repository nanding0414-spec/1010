# LSA + HC-GMP + PRR · SAM3弱监督语义分割主线手册

更新：2026-10-29。本文是唯一项目操作与模块说明手册。**当前正式主线以第109节为准**：LSA＋HC-GMP＋PRR，balanced选模、MLP前不删/后删5%、分类面积100。用户本次明确授权接入主线并发布GitHub源码和文档。第46/84节及其它早期记录保留为历史，不覆盖本条当前操作。供应商SAM3许可证和接口说明仍以sam3-main为准。

### 历史：2026-10-10独立从头实验项目

用户本次指定并已通过SSH确认的根目录为 `/data/home/zystu2025/nd/1010-main`，
环境Python为 `/data/home/zystu2025/.conda/envs/moduseg_sam3/bin/python`。
本次使用项目内 `data/VOC2012` 和 `weights/sam3/sam3.pt`，当前双卡完整入口与主线激活见第46节。
下述旧项目迁移、旧缓存约定及1–42节历史命令保留；本次明确从头生成新标签、提议和库，
不使用旧缓存约定或历史模型作为输入。


> 历史记录（2026-10-13；现行操作以第109节为准）：当时主线按用户要求采用第45节已验证的presence=1＋边缘修复＋逐类别几何清理：原图Active标签 → 冻结SAM3 → 单相位硬中心GMP（ρ=0.1）→ 两层MLP → 单库检索 → clean/raw提议与HCR。2026-10-13按用户要求将推理中心权重设为0.2；第48节seed42对照为86.8217%，原中心权重0.4基准为86.7350%，均为开发验证分数，尚未完成新配方三seed验证。当前正式操作以第46节为准。此前三seed均值86.4316%属于旧配方。1–45节保留历史操作和实验记录，旧代码/标签/库/权重/指标不改写。

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

## 46. 当前正式主线与GitHub同步（2026-10-13更新）

用户明确将第45节完成版本作为正式主线。仓库为 `https://github.com/nanding0414-spec/1010`。
当前模型版本 `mainline_edge_cleanreuse_20261011_114857`，原中心权重0.4的VOC val单seed42 mIoU86.7350%，
训练50×100步、内部最佳38轮、完整1,449图严格HCR；这是开发验证成绩，不是独立测试或三seed均值。
历史原始标签、提议、库、模型、指标及代码快照不改写。

正式配方：训练图片的已知类别presence概率固定1，全部允许类别解码；SAM3原图logits上采样并sigmoid，
概率阈值0.4，普通类3×3 close1/dilate1，餐桌close2/dilate1。同类实例合并、原类别顺序合成语义PNG后，
再逐类别做几何清理：去小于64像素的8连通小岛，填≤32像素且≤类别面积1%的封闭孔洞，类别面积门槛100。
填洞仅写原背景、不覆盖其他类别，不伪造语义PNG不存在的实例分数，不对互斥语义通道做跨类NMS。
验证提议保持全词表、正常预测presence及原清理配方0.7/100/64/32/0.01/0.85，不使用验证真值类别。
冻结SAM3，原图1008/trunk_fpn/FP32，单相位硬中心GMP rho=.1，原库前景最远10%过滤一次、背景保留。
MLP2 1280→512→256、v1/seed42、50×100步，按训练图片内部验证5-NN宏准确率选模。
单库balanced检索K25/每类top5/中心权重0.2，加原HCR参数，仅补背景。

2026-10-13用户指定将推理中心权重从0.4调整为0.2，计分为0.8×近邻均值＋0.2×类别中心。
`configs/config.py`、单次推理入口`tools/run_mainline.py`和完整入口`tools/run_mainline_full.py`
均使用0.2；训练选模、MLP权重、区域几何及HCR参数不随此修改改变。
第48节固定seed42/同一MLP与单库的0.2对照为86.8217%，比该节0.4基准提高0.0867个百分点。
该结果不是三seed稳定性结论，也不改写既有模型版本的历史指标；后续评测使用新输出版本。

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

## 47. 清理标签缓存上的独立优化对照（2026-10-11）

入口tools/run_cached_optimization.py；当前主线选择和86.7350%历史结果不改写，不自动激活优化结果。
训练标签明确复用第45节clean PNG，验证raw/clean提议复用第44节。native分支只读复用第44节
FP32整图token；letterbox分支保留宽高比、居中、归一化零填充，在新版本storage/image_features下
重新保存整图FP32 token。新几何签名包括分辨率、实际网格、映射、padding；不能混用旧square token。

对照顺序：baseline精确回放（要求1449图混淆矩阵与第三轮逐元素一致）、hcr_geom、multi4、
calibrated_multi4、bg4_ce、bg4_gce、letterbox_ce。后三项各50轮×100步、seed42、原内部图片划分，
仍只在全库合并后过滤前景最远10%、背景不删。bg4用每图背景token外观聚成至多4个互斥区域，
划分按参考报告的图片归属固定；新几何过滤后若某图无剩余区域，记录split_coverage，不将它换到另一折。
完整覆盖原背景，不改变前景PNG；前景1280维原向量直接复用。CE/GCE使用相同库隔离损失影响。
letterbox_ce保留原前景连通域和每图背景union，隔离缩放几何变化，不同时增加背景分块。

hcr_geom对clean和附加候选统一要求原图面积300、与clean/已接受raw池IoU>=0.9去重，救援同类NMS=0.4。
核对发现live生成有300门槛，旧缓存加载路径只有清理时的100门槛，因此该对照同时过滤小clean候选。
原始relaxed raw池本来就做过NMS=0.85和几何清理；这些新增控制不改变历史默认HCR行为。
救援仍只补背景；这里只对原候选面积设300，不对补出的有效新像素设300，以保留合理边缘补全。
multi4每类至多4个确定性球面中心，近邻均值与最大中心相似度仍按0.6/0.4组合。
calibrated_multi4用内部留出TRAIN伪区域做温度缩放和每类99%经验精度阈值（至少20例）。
不足支持的类禁用救援。标签有噪声，经验精度不代表GT校准保证；最终全库包含内部留出区域，
校准池与最终全库存在分布差异，记录来源。验证GT仅用于最终mIoU和完成后的提议覆盖诊断。

当前缓存不含逐实例mask/score。它无法恢复真实同类实例边界、原跨类score竞争或置信度权重。
未来生成可显式使用--overlap-policy score --save-instances，保存SAM原始、修复后、分配后masks、
class_ids和真实SAM scores，逐实例边缘修复后按分数优先合成。默认历史生成规则保留。
src/pseudo_instances.instance_regions读取assigned masks，不按连通域重切；若后续语义清理修改PNG，
它拒绝不匹配的实例缓存，要求显式重新协调，不能猜测置信度。训练支持--pseudo-loss gce --gce-q 0.7，
以及--sample-weights JSON（class/sample_id到真实(0,1]权重；必须覆盖全部保留成员）。本次GCE不使用
虚构SAM权重。GCE只替换分类损失；对比损失保留，不能视为所有噪声源都已解决。

raw历史生成阈值0.15不能由现有NPZ降低。本次统计raw/clean前景覆盖和GT连通块最佳候选IoU，
作为提议瓶颈诊断，不能宣称测试了0.05生成阈值或得到可达mIoU上限。降低生成阈值需要新raw版本，
旧缓存只读保留；本次遵循复用要求，不重生成提议。

长任务先检查全量输入SHA、缓存receipt/encoder签名和GPU，再双卡真实BG/letterbox提取、短GCE
训练导出、严格1图HCR。所有探针通过才做完整对照。各阶段使用不可变代码快照；相同输入、参数、
代码才允许--resume，失败阶段新attempt。数据/库/全图tokens在storage，权重在weights，结果在
experiments，PID和日志在logs，缓存临时目录在/data，保留5GiB。七组均评测1449图才complete。
最终summary.json记录全部对照、差值和限制，metrics.json记录各组21类IoU和混淆矩阵，per_class.csv
给出逐类差值；结束/失败按已有通知配置发邮件，附三份结果，不附模型、数据或凭据。

    python -B tools/run_cached_optimization.py --version cached_opt_YOUR_TIMESTAMP --preflight-only
    python -u -B tools/run_cached_optimization.py --version cached_opt_YOUR_TIMESTAMP --probe-only
    nohup python -u -B tools/run_cached_optimization.py --version cached_opt_YOUR_TIMESTAMP --resume > logs/cached_opt_YOUR_TIMESTAMP.launch.log 2>&1 < /dev/null &

方法出处：[GCE原论文](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)、
[温度缩放原论文](https://proceedings.mlr.press/v70/guo17a)。这些方法的论文结果不等于本项目提升。

后续真实实例库入口为 tools/build_saved_instance_bank.py：显式提供 --pseudo-dir、--instances-dir、--image-features、--reference-bank（原始未适配GMP签名）、--train-list、--images-dir、--output-bank 和 --checkpoint-dir。默认 GPU1、background-clusters=4，前景保持assigned实例边界；--min-confidence可按真实分数过滤。全库统一过滤后生成meta/sample_weights.json，传给train_feature_adapter.py --sample-weights。背景采用固定单位权重并注明来源，不伪称SAM背景置信度。实例NPZ、PNG及receipt SHA不一致会拒绝；此入口不应用于第三轮只有PNG的旧缓存。

### 第47节完整实验结果（2026-10-11）

`cached_opt_20261011_143933`于北京时间20:58完成，status=complete、completed_variants=7。
七组均完整评测VOC val 1449张；bg4_ce、bg4_gce、letterbox_ce均完成50轮×100步。

| 对照 | mIoU (%) | 相对第三轮 (百分点) | best epoch |
|---|---:|---:|---:|
| baseline | 86.7350 | 0 | 38（复用） |
| hcr_geom | 86.7462 | +0.0112 | 38（复用） |
| multi4 | 86.6322 | -0.1028 | 38（复用） |
| calibrated_multi4 | 86.0499 | -0.6851 | 38（复用） |
| bg4_ce | 86.4987 | -0.2363 | 36 |
| bg4_gce | 86.5185 | -0.2165 | 38 |
| letterbox_ce | 86.5930 | -0.1421 | 31 |

这是同一seed42的开发验证对照，只有HCR几何控制微涨，其它组均下降；不自动激活新结果，
当前mainline_active仍保持第三轮模型。基准回放的混淆矩阵与第三轮逐元素相同。
新letterbox整图FP32缓存已保存train=10582、val=1449，位置为
storage/image_features/voc/cached_opt_20261011_143933/letterbox/{train,val}。
结果位于experiments/cached_opt_20261011_143933/{summary.json,metrics.json,per_class.csv}。

后验诊断：raw前景像素覆盖95.5772%，clean覆盖92.4382%；这是几何覆盖，不能解释为分类精度
或可达mIoU上限；本次未测试低于0.15的原始生成阈值。
结束邮件3次自动发送失败。23:11以LOGIN机制重试确认QQ SMTP返回535认证拒绝，尚未成功发送；
诊断追加在notification_LOGIN_retry_20261011_2312.json，原notification.json与实验结果保留。
三个结果文件已同步到本地experiments/cached_opt_20261011_143933。
23:20使用新授权码补发成功，QQ SMTP已接受发送，附summary.json、metrics.json和per_class.csv；
这表示服务器接收成功，不代表已确认收件箱投递。补发记录为
notification_retry_20261011_232011.json，原失败通知记录与实验结果保留。

## 48. 从头主线及固定参数/网格/多尺度对照（2026-10-11）

独立入口tools/run_mainline_tuning.py，沿用run_mainline_full.py的CLI、签名预检、双卡监督、
代码快照、逐图检查点、阶段attempt和结束/失败邮件。当前主线不变，不自动激活或向GitHub发布。
不从历史标签、提议、原库、MLP权重或查询特征恢复输入；首先完整从头运行第46节配方。
该完整基准的产物版本为<队列版本>_fresh，阶段和指标位于experiments/<队列版本>/fresh，
日志位于logs/<队列版本>_fresh。训练标签10582、验证提议1449、新原库、50×100步MLP2、
严格1449图HCR均须完成；默认保存本次全新1008整图FP32 tokens。

随后只复用本队列生成的标签、提议、token和模型，按预先固定列表执行单因素开发验证。
cases()包含基准精确回放、关闭HCR、纯实例检索、topm=3/7/10、中心权重=0/.2/.6/.8/1、全局margin=.05/.1/.15、
分类NMS=.35/.45、面积150/450、HCR q=.35/.45、检索=.92/.97、margin=.08/.12、
novel_fraction=.15/.2/.25、救援同类NMS=.35/.4/.45，以及五项独立按类策略。
注意主线全局margin为0、HCR margin为.1，原HCR novel_fraction为.1；不存在已接入Active的grid参数。
训练类别频次只取训练图像级JSON：低于20类频次中位数视为稀有类，topm=10（其余5）、
margin=.05（其余.15）、动态面积max(64,300*H*W/(500*375)*稀有系数.5或1)、
合成实例分数门槛稀有.02/其余.1；频繁类HCR门槛q=.45/检索=.97/margin=.12，稀有保留原值。
这些是固定假设，不从VOC val调出阈值；动态面积仅改变原clean实例筛选，HCR附加候选仍用原策略。
INF_INSTANCE_SCORE_THR用于在线提议，缓存推理不直接应用；本次adaptive_score显式筛合成分数，不能
将它描述为同一个SAM提议门槛。INF_PER_CLASS_NMS_IOU实际配置名为INF_NMS_IOU。
原weight=0仍保留无近邻类别的中心回退；新增pure_instance同时将回退关掉，无支持类别局部分数设-1。
当前固定队列为fresh基准1组＋推理35组＋GMP6组＋网格10组，共52组，均完整1449图。

六项GMP独立新库/新训练：rho=.05/.15、spatial_core、672单尺度、1344单尺度、672+1008+1344融合。
spatial_core以mask内部距离产生0.5+distance/max权重，是空间代理，非Transformer attention或学习门控。
融合为各尺度L2 GMP向量均值再L2，仍1280维再训练相同MLP2；不混用旧模型。
各新库合并后仅一次全库前景最远10%过滤、背景保留，固定本次基准的内部图片划分，独立训练50轮。
非1008尺度使用现有全局RoPE适配与query分块256，FP32；保存全部train/val整图token至
storage/image_features/voc/<版本>/scale{672,1344}/{train,val}。签名aggregation.version=3、
square_gmp_tuning_v1显式包含各尺度encoder、基础权重、rho、权重规则和融合，库/MLP/查询严格匹配。
历史native和letterbox签名不改变，未知或混合签名拒绝。几何仍方形缩放，不加入上一轮letterbox。

网格点提示为独立提议器对照，不假称调整Active参数。先用grid16/32/64、crop_layers=1、
points_per_batch=32、IoU>=.75、stability>=.88、NMS关闭保存真实候选及predicted_iou、
stability_score、crop_boxes/layers；这些真实模型质量字段不由Active置信度冒充。
从各自候选分别重放每crop及跨crop mask NMS，得到十项完整验证：grid32基准、grid16/64、
IoU=.85/.90、stability=.92/.95、mask NMS=.6/.8、仅原图crop_layers=0。
每项新raw/clean保留并执行原清理配方和同一新基准模型；不把高IoU分数等同分类置信概率。
裁剪0对照从同一原图分支候选抽取；不重新把crop图片提取结果当原图分支。

保存的查询为MLP前区域向量，绑定图片SHA、全部mask SHA、特征签名和整图token receipt；
复用只在完全匹配时发生，不用查询缓存代替新的图片编码。诊断输出包括提议数量直方图、
各类margin分布、GT多数类候选分类匹配率、几何前景覆盖、HCR接受数/新增像素和21类IoU。
GT多数类诊断在该图预测完成后计算，不回传决策，不把该候选指标当像素分类精度。
全量混淆矩阵先合并再计算IoU；baseline缓存回放必须逐元素复现。
summary.json和per_class.csv包含各组差值，metrics.json汇总各组矩阵/IoU/诊断；详细逐图
记录留在各eval attempt。所有组完整1449图、全部七次新MLP（含fresh）50轮才complete。
邮件附汇总、compact metrics和逐类CSV，通知工具优先用QQ支持的LOGIN，DATA接受后QUIT断连不误报失败。

    python -B tools/run_mainline_tuning.py --version fresh_tuning_YOUR_TIMESTAMP --preflight-only
    python -u -B tools/run_mainline_tuning.py --version fresh_tuning_YOUR_TIMESTAMP --probe-only
    nohup python -u -B tools/run_mainline_tuning.py --version fresh_tuning_YOUR_TIMESTAMP --resume > logs/fresh_tuning_YOUR_TIMESTAMP.launch.log 2>&1 < /dev/null &

不设置整体到期时间。预估除fresh约297GiB之外增加672/1344约661GiB以及网格缓存400GiB保守余量，
实际按预检输入计算；每阶段保留5GiB。双卡真实六种GMP短训练/导出/检索/HCR联通与grid64真实提议，
再加原完整入口双卡Active短流程，全部通过才从头生成标签。错误停止队列并保留日志/attempt。

第48节启动验收（2026-10-12）：80项相关单元测试通过，真实双卡六种特征配方均完成短训练、
导出、检索和HCR检查，两卡逐元素最大差均为0；grid64真实提议3090个，质量字段来源已验证。
两卡探针峰值allocated各7632.7MiB，约7.45GiB；全部VOC val图片至多500×500。
双卡Active标签/清理/短MLP/严格HCR及所有新特征、按类策略的实际1图评测路径通过后进入ready。
此前诊断版本fresh_tuning_20261012_000018因probe库meta路径失败，000657因双卡同图查询缓存
来源冲突失败，均保留失败产物与SMTP接受的通知；全量数据阶段未启动。新实现将完整来源绑定
纳入查询文件名，不覆盖旧checkpoint；新增测试验证独立库映射和不同encoder receipt来源隔离。

正式版本fresh_tuning_20261012_001942于北京时间00:34:10通过nohup/独立会话从ready恢复启动，
启动器PID212393；实际从头标签worker为GPU0 PID212422、GPU1 PID212423，各5291张且互不交叉。
00:50核验1673张新PNG：完成凭据与PNG SHA256匹配、尺寸为原图、类别只取图像级允许集合，
presence=1、概率阈值.4；两卡日志推进。当前是pseudo阶段，50轮训练及52组完整评测仍待执行，
不能把探针的合成类别分数当实验精度。当前激活主线仍是第三轮模型。
启动记录、缓存验收和对照参数为experiments/<版本>/{background_launch.json,
startup_validation_receipts.json,experiment_plan.json}，已同步本地；主进度看status.json和fresh/status.json，
阶段worker PID看对应progress.json，父层progress可能仍记录上一次探针，不能据它误判长任务停止。
完整日志logs/fresh_tuning_20261012_001942_launcher.log；标签阶段日志在
logs/fresh_tuning_20261012_001942_fresh/pseudo/attempt_001。
私有通知配置已使用新授权码并认证通过；完成/失败邮件附本版本summary.json、compact metrics.json和per_class.csv。

第48节中断恢复记录（2026-10-12）：fresh_tuning_20261012_001942于03:02:15在fresh/build
启动前失败，原因是父调度器创建fresh子流程时没有写experiments/<版本>/fresh/manifest.json，
原完整入口建库需要该文件的SHA256。故障发生在建库worker启动之前，并非训练或显存故障。
已完成10582张raw/clean标签及伪标签评估、1449张raw/clean提议；伪标签mIoU为86.2884458885%。
恢复前完整核验原始输入、279个源码文件及不可变快照、全部已完成stage的artifact SHA256，
并重新检查全部PNG尺寸/类别和NPZ结构、clean/raw来源绑定。所有校验通过。

从未改变的suite_manifest（父manifest加code_digest、子版本后缀_fresh）恢复缺失的子manifest，
没有修改实验代码、参数、输入、已完成标签/提议或指标。原失败summary/status/notification
及子流程状态复制到experiments/<版本>/recovery/20261012_085548，旧失败日志继续保留。
08:55:48使用同版本--resume以独立后台会话启动，PID213807；新的启动日志为
logs/fresh_tuning_20261012_001942_resume_20261012_085548.log，对应.pid为新启动器PID。
恢复预检记录recovery_preflight_20261012_0855.json，启动记录background_recovery_20261012_085548.json。
恢复会复用已校验完成阶段，失败建库进入新attempt_002，后续仍按原52组计划继续，不重复生成标签/提议。

本地入口新增prepare_fresh_launcher：在构造子Launcher之前写入其完整manifest，恢复时只接受
完全相同的manifest，存在差异立即拒绝且不覆盖旧记录。新增三项回归测试覆盖先写后构造、
匹配复用和错配拒绝；连同检查点、双卡分片及环境绑定测试共六项通过。
该源码修正供后续新实验版本部署；本次服务器使用原不可变代码快照，通过补齐缺失记录恢复。

09:01恢复验收：建库attempt_002双卡worker PID213920/213921存活、日志持续推进，已生成
150余张逐图区域NPZ及整图token NPZ。两卡各抽查一个已完成checkpoint，验证PNG/图片来源、
子manifest的code_hash、区域向量非零有限、整图token SHA256及72×72×1280 float32形状均通过。
恢复验收记录recovery_startup_validation_20261012_0900.json；MLP训练尚未开始。
恢复期间summary.json和notification.json可能仍显示上次失败记录，直到本次运行结束更新；
当前状态以status.json、fresh/status.json和fresh/progress.json为准。

2026-10-13用户要求完成42组后暂停：15:38确认启动器和两个网格64 worker已退出，
两卡已释放；42组均严格评测1449图，partial_results.json未改变，网格16/32完成缓存与
网格64部分缓存保留。状态interrupted为人工暂停，剩余10组网格对照未获得最终结果。
暂停请求和退出核验保存在experiments/<版本>/user_pauses/20261013_153732。
未经用户明确要求继续，不恢复这套实验或启动配对三seed验证。

同日用户指定推理中心权重0.2作为当前配置，详见第46节；本地和服务器当前入口同步更新。
修改前的服务器配置、两个入口和手册备份位于experiments/config_updates/centroid02_20261013/before，
变更核验记录位于同目录。原52组实验的manifest、指标和不可变代码快照保留原0.4基准配方；
新配置用于新版本，旧实验续跑仍须满足原源码/参数/输入身份，不将0.2混入原52组基准。

## 49. 固定库成员的责任加权GMP对照（2026-10-13）

用户提供五种token responsibility方案并要求复用缓存。独立入口为
`tools/experimental_gmp_responsibility.py`，默认只读来源为第48节已完成的
`fresh_tuning_20261012_001942/fresh`，原52组网格队列继续暂停。
复用清理伪标签、原始/清理提议、训练和验证1008/72×72/1280整图FP32 token，
复用原内部训练图片划分；不运行SAM3编码器，不复制整图token，不生成提议。
固定来源过滤后的34,880个区域成员及每类行顺序（前景24,311、背景10,569），
不重新做全库异常过滤。这里的区域来自既有语义PNG，不能称为原始SAM实例边界。

固定seed42、rho=0.1、中心权重0.2、主路径margin=0、原HCR和50×100步MLP2/v1。
已完成的同库/同MLP原Hard-Center中心0.2评测86.8217%作为只读基准；
七个新配置各自重训MLP和导出匹配单库，再严格评测1449张，合并混淆矩阵后统计IoU。
每类成员ID、标签和行顺序必须与来源相同，模型和查询签名必须匹配；不自动替换主线。

| 新配置 | token责任/聚合规则 |
|---|---|
| coverage_only | 仅用精确mask覆盖率；区分软几何支持本身的影响 |
| geometry_consistency | 覆盖率×单原型视觉一致性sigmoid |
| multi_prototype3 | 区域核心token确定性3原型，取最大余弦一致性 |
| iterative2 | 覆盖率初始化，2轮原型/softmax责任更新，最后只求解一次GMP |
| robust_huber | 覆盖率×Huber风格余弦残差权重 |
| core_boundary_gate | 核心/边界分别等权GMP，由边界与核心GMP向量的一致性门控融合 |
| multi_consistency_huber | 多原型一致性×Huber残差抑制的组合 |

首轮固定gamma=1、核心覆盖率≥0.8、特征一致性beta=0.5/tau=0.1、K=3、
K-means10轮、Huber delta=0.3、迭代2轮。核心/边界融合alpha=0.5、
gate beta=0/tau=0.1；这些是预先固定的实验参数，不是经验证集选优的参数。
有核心时原型数不超过核心token数；无核心时单原型方法用覆盖率均值构造原型，
多原型方法在正覆盖率token上做同一确定性聚类，K不超过可用token数，
各中心仍为覆盖率加权均值；core_boundary无核心时使用覆盖率GMP。
核心缺失、有效token数、原型数和门控记录在逐区域诊断中。
覆盖率通过原像素积分图的分数坐标面积积分计算，不用近似adaptive pooling；
正覆盖率token进入支持集。余弦用于责任权重，不解释为语义正确概率。
保持FP32及rho×加权二阶矩trace的岭项，Cholesky残差≤1e-4；失败立即停止，
不降精度、分辨率或跳过图片。

双卡按图片交错分片；每张图只读取并校验一次旧token，然后同时计算七套训练向量。
验证端同样一次读token，按原clean＋HCR relaxed raw的完整候选池计算七套查询；
后续七轮评测读取新查询检查点，不重复编码或求解查询GMP。新缓存逐图绑定
来源图片、伪标签/候选mask、token凭据、算法参数、固定成员摘要和代码SHA，
完成后可同签名恢复；失败attempt和日志保留，不覆盖旧产物。
新区域/查询检查点和库放storage/raw_banks/voc/<新版本>；权重放weights/sam3_adapter，
投影库放storage/mlp_banks，参数/结果放experiments，启动及worker日志放logs。
GPU0/1分别完成真实缓存、原Hard-Center一致性、七种短训练/导出/严格HCR联通探针后，
才允许长任务。结果和失败统一由启动器按已授权QQ邮件通知。

```bash
python -B -m unittest tests.test_responsibility_gmp tests.test_responsibility_protocol
python -B tools/experimental_gmp_responsibility.py --version YOUR_NEW_VERSION --preflight-only
python -u -B tools/experimental_gmp_responsibility.py --version YOUR_NEW_VERSION --gpus 0 1 --probe-only
# ready后，同代码、参数和输入恢复；新启动日志名：
nohup python -u -B tools/experimental_gmp_responsibility.py --version YOUR_NEW_VERSION --gpus 0 1 --resume > logs/YOUR_NEW_VERSION_launcher.log 2>&1 < /dev/null &
```

此节记录方案和运行协议；没有完成正式评测时不声称获得提升。

启动验收（2026-10-13）：18项相关测试通过，包含面积守恒、加权正规方程、
多模态原型、固定成员顺序、签名错配拒绝、适配器训练/加载显式许可和原恢复/分片/矩阵合并。
早期探针版本gmp_responsibility_20261013_163818被适配器加载的严格签名许可拦下，
保留失败日志；修正只在新实验worker内显式启用新签名，不改主线模型加载规则。
正式版本gmp_responsibility_20261013_164316的两卡七方法短流程共14项全部完成，
原Hard-Center向量与来源缓存的一致性检查通过，探针不作为正式精度。
长任务使用同签名--resume后台启动，启动器PID228203，区域向量worker PID228238/228239。
启动日志logs/gmp_responsibility_20261013_164316_launcher.log，运行与验收记录为
experiments/gmp_responsibility_20261013_164316/background_launch.json和startup_validation.json。
16:51服务器确认shared_build运行、两worker存活、日志推进、已生成530份七方法联合检查点；
正式训练和评测尚未完成，第48节剩余网格实验保持人工暂停。

完成验收（北京时间2026-10-13 20:19:02）：gmp_responsibility_20261013_164316全部complete，
7个新方案均训练50轮并严格评测1449张，加复用基准共8组；长流程耗时12601.02秒，
约3小时30分。每组固定34880个区域及原行顺序、内部图片划分保持一致，原数据/提议只读。
全部结果相对中心权重0.2的原Hard-Center基准比较，不相对历史0.4基准比较。

| 配置 | mIoU (%) | 相对86.8217的差值（百分点） | 最佳轮次 |
|---|---:|---:|---:|
| native_center02（只读复用基准） | 86.8217 | 0 | 38 |
| coverage_only | 86.5425 | -0.2792 | 32 |
| geometry_consistency | 86.7080 | -0.1137 | 38 |
| multi_prototype3 | 86.6402 | -0.1815 | 23 |
| iterative2 | 85.7059 | -1.1159 | 37 |
| robust_huber | 86.4715 | -0.3503 | 29 |
| core_boundary_gate | 86.5648 | -0.2569 | 38 |
| multi_consistency_huber | 86.6103 | -0.2114 | 32 |

本轮固定参数、单seed42没有新方案超过原Hard-Center，保持第46节原GMP及中心权重0.2。
geometry_consistency相对纯覆盖率对照提高0.1655个百分点，但相对原Hard-Center仍低0.1137；
不能将其解释为已优于主线。组合方案汽车提高2.8725点，但椅子/瓶子分别下降4.1866/3.6462点。
此结论限于本轮固定配方，不否定其他责任权重/参数配方；没有自动追加新实验。
结束邮件首次被QQ SMTP接受，附件summary.json、metrics.json、per_class.csv；不声称确认收件箱投递。
启动器及worker已退出，两卡已释放；最终记录为同版本completion_validation.json，
保留原42组暂停状态与全部历史代码、缓存、模型及指标。

## 50. 交叉检索一致性反馈与中心权重配对三seed（2026-10-13）

独立入口 `tools/experimental_crossfit_feedback.py`，测试
`tests/test_experimental_crossfit_feedback.py`；不修改 `src/`、`configs/`、主线入口或旧实验。
固定第45/46节 `mainline_edge_cleanreuse_20261011_114857` 的已过滤1280维原库、seed42两层MLP、
原导出库、原内部图片划分和图像级JSON，第44节raw/clean验证提议全部只读。
主线当前中心权重0.2；历史86.7350%核对仍显式使用0.4。固定现有原始查询向量缓存，
逐图校验图像SHA、全部候选mask哈希及顺序、原生特征签名、编码凭据和NPZ SHA；
只选择匹配第44节候选的已有缓存。缺失停止，不运行SAM3、不重新聚合GMP。

用户在本次会话明确取消“第48节52组全部完成”门槛，改为先检查旧启动器及worker退出、
锁释放及所用GPU空闲；第48节保持42/52暂停，不自动续跑。服务器已确认未完成10组全部属于
网格提议对照，6组GMP及35组推理参数均完成，不能把旧17:09进度当成当前状态。
旧版本只保存总混淆矩阵，用户批准替代核验：完整1449图先逐元素复现历史总矩阵，
再独立回放一次，两次新回放逐图矩阵逐元素一致。此核验不能证明不存在的历史逐图记录相同。
CLI默认仍保留严格旧门槛和历史逐图要求；本次授权通过显式参数绑定进不可变manifest。

交叉拟合按 `SHA256("crossfit42:"+image_id)` 的整数奇偶分A/B，同图全部区域同折，
每个前景类每折至少两张不同图片，否则停止。两折各用原v1配方50轮×100步、
8类×4区域、1280→512→256、seed42训练；内部选模只在本折图片生成固定划分。
使用本折模型投影另一折查询及本折全部库，直接调用生产balanced分类器，
K25/top5/搜索倍数2/中心权重0.2、单中心、无温度校准。
保存每条 `类别目录/sample_id`、21类分数（VOC ID顺序）、margin、
g=伪类别分数−最强异类分数、预测、折及图像级允许类别。

两折检查完整提交并核验SHA之前不读取训练像素GT，包括不在预检提前计算训练GT SHA。
预检只复制历史训练GT身份凭据并检查存在性/大小；诊断阶段才按历史SHA读取和验证像素。
以有效非255像素中“GT等于伪类别”的比例作为伪类别纯度，另外记录GT多数类别、
多数纯度和是否匹配伪类别。报告一致率、两组纯度分布、g相关系数和允许类别比例。
唯一预设继续门槛：前景一致组平均伪类别纯度−不一致组平均纯度≥0.05。
不满足写 `complete_no_signal` 并发通知；GT诊断数据不传入样本权重、剔除或改标签。

A：背景及前景g≥0权重1，前景g<0权重0.3，权重覆盖每个原库成员，
使用原trainer `--sample-weights`，只作用CE，SupCon不变；`--split-file`固定原内部图片划分。
原库成员完全不变，重训50轮后导出同一成员集合。本次先用 `--a-only`，信号成立只执行A。
B已实现但本次不启动：g<−0.05按g最差优先，每类最多剔除15%，至少保留两图，
背景不动；写新子集库，无第二次前景过滤，训练与检索库同时变化，报告split_coverage。
C默认关闭，显式 `--enable-c` 才运行：预测为图像级允许前景、margin≥0.1、g<−0.1、
不一致才改标签；改入和改出分别保守限制为各类原区域数的10%，保留两图及原sample_id，
记录relabel_from。背景在三个方案中权重1、不删除、不改标签。

`--mode center-paired` 用同一原库、伪标签、提议、查询向量和冻结内部图片划分；
seed42复用原MLP，只补seed123/2026两次50×100训练。每个seed用同一MLP和导出库比较
中心0.4/0.2，固定候选，不按seed选配置。报告配对增益均值、样本标准差及n=3/df=2的
Student-t 95%区间；注明三seed仅初步稳定性检查，不自动部署。两个模式各占一张物理卡，
交叉拟合两折在同卡依次训练，双卡模式则每卡最多一个worker；可单卡或显式双卡。

```bash
python -B -m unittest tests.test_experimental_crossfit_feedback -v
python -B tools/experimental_crossfit_feedback.py --version NEW_CROSSFIT --gpus 0 \
  --gate-policy explicit-bypass --reference-policy repeated --a-only --preflight-only
python -u -B tools/experimental_crossfit_feedback.py --version NEW_CROSSFIT --gpus 0 \
  --gate-policy explicit-bypass --reference-policy repeated --a-only --probe-only
# 探针通过后原参数加 --resume，后台脱离SSH；另一个版本用 --mode center-paired --gpus 1。
```

新原库与折库放 `storage/raw_banks/voc/<版本>`，导出库放 `storage/mlp_banks/voc/<版本>`，
权重放 `weights/sam3_adapter/<版本>`，参数、检查、诊断及总/逐图矩阵在 `experiments/<版本>`，
worker日志在 `logs/<版本>`。启动时冻结完整代码及专用测试，绑定输入/参数SHA；
`--resume`要求完全一致，已完成阶段逐一核验，失败阶段新attempt，不覆盖旧attempt。
运行保留5GiB空间，SIGTERM仅结束本启动器所创建的worker；无整体到期时间，不降低配方。
完成、无诊断信号、失败或中断后由启动器向用户指定QQ邮箱发送实际状态，
附件summary.json、metrics.json、per_class.csv，SMTP最多3次，发送失败不更改实验结果。
本地/服务器14项专用CPU测试通过；正式预检与启动记录另行追加，当前未宣称获得新精度。

第50节启动验收：31,396项固定输入身份通过，原库34,880成员，全部1,449图存在匹配的
原生1280维查询缓存；预检不读取训练GT像素。两卡各完成2图真实缓存MLP/检索/HCR探针，
严格图像覆盖及逐图矩阵求和检查通过，没有SAM3前向或GMP重新提取。
正式后台版本 `crossfit_feedback_20261013_204924` 使用GPU0，启动器PID231388，
`--mode crossfit --gpus 0 --gate-policy explicit-bypass --reference-policy repeated --a-only --resume`；
版本 `centroid02_paired_20261013_204924` 使用GPU1，启动器PID231389，
`--mode center-paired --gpus 1`，其余授权策略与前者相同。
两版本均使用固定代码SHA的 `--reuse-preflight experiments/preflight_audits/parallel_20261013_204924/validated_final_manifest.json`，
仅复用已核验的候选几何对应关系；启动/阶段仍重新校验全部消费输入SHA。
同版本probe的ready摘要保存在该preflight_audits目录，正式attempt与探针输出分开。
2026-10-13 21:11:40验收时，两个启动器及自有worker231434/231433均存活，
分别处于historical_replay，日志已超过750/1449图；两卡各占用约1.8GiB。
不可变快照SHA及生产src/configs未修改核验通过，结果尚未完成，不将启动验收视为新精度。
启动日志为 `logs/<版本>/launcher_attempt_001.log`，状态、PID及验收证据分别在
`experiments/<版本>/status.json`、`background_launch.json`、`startup_validation.json`。
完成/无信号/失败通知已由两个独立启动器配置；B、C本次均不执行，不自动升级主线。

启动后核验：两版本的1449图历史总混淆矩阵均逐元素复现，第二次独立回放的1449份
逐图矩阵也全部相同，`replay_proof`已passed。原seed42模型在当前中心0.2下均得到
86.8217170759%，随后GPU0进入fold_training、GPU1进入train_seed123，尚无新方案精度。
单独核验证据为各版本 `historical_total_replay_verified.json`。
运行入口及快照SHA为 `ae9a38f2a55c4beda9d0a11c5fb934f535ba9804774591b9a25ab7a00b0024e4`。
本地入口后续仅修复日志wave目录名，加上阶段attempt编号以防重试日志冲突，
SHA为 `6287a70124b02d540dd9b541b3afd46c844262eaa87d7a971f89cf6c4b874fc0`，14项CPU测试再次通过。
该日志修订另存代码修订目录，不替换两条正在运行的服务器入口或快照；后续启用时须新版本。

## 51. A诊断恢复及B/C结束后衔接（2026-10-13）

用户明确授权“等这个跑完就开始BC”。B与C算法已实现；新增 `--feedback-source <完整A版本>`
和 `--variants B` / `--variants C`，分别建立独立新版本，不重训A、两折模型或重新计算检查结果。
两个方案各自从第46节同一原始库出发，不将B剔除后的库传给C。背景保持原向量、sample_id和类别。
阶段只读引用旧产物，逐项SHA校验并写新版本的reused_from完成凭据；原始源码/模型/库/指标不覆盖。
生产代码须与供体快照相同，仅本独立入口与专用测试允许在新版本更新，记录两个代码摘要。

检查时发现原 `crossfit_feedback_20261013_204924` 已因
`ValueError: Region has no valid diagnostic GT pixels` 停在diagnostic；两折50轮模型和全部交叉检查已完成，
失败邮件已被SMTP接受。原因是某些区域GT全部为255忽略标签，纯度不可测，原诊断错误地中止。
现在为其记录null纯度、valid_pixels=0及ignored_only，只从纯度分布、相关系数和5个百分点门槛的
统计样本中排除；一致率仍统计全部区域，A/B/C成员、权重、剔除和改标签规则不读取此逐区域GT信息。
GT尺寸/类别/SHA异常仍拒绝，不把全255区域虚构成纯度0或1。

新增 `--recover-source <诊断失败版本> --variants A`：只接受已退出、锁释放且失败阶段为diagnostic的
供体；原7个已完成共同阶段核验后直接引用，新版本重新执行诊断与条件成立后的A训练/评测。
恢复版本 `crossfit_feedback_recover_20261013_214723`，物理GPU0，第二次启动器PID232666；
CPU预检通过，原两折模型和34880条区域检查不重做。首次恢复启动在正式产物创建前因预建logs版本
目录触发目录保护而退出，启动日志保留；修正为logs根目录文件后用同一尚未创建的版本正式启动。
日志 `logs/crossfit_feedback_recover_20261013_214723_launcher_attempt_002.log`；
启动记录与18项专用CPU测试结果在 `experiments/preflight_audits/crossfit_feedback_recover_20261013_214723`。
三seed任务继续使用自己的冻结快照，不中断。旧诊断失败记录、旧源码和所有产物保留。

后续冻结计划：`experiments/crossfit_bc_pending_20261013_215903/plan.json`，包含两个新版本：
`crossfit_feedback_B_20261013_215903` 使用GPU0；`crossfit_feedback_C_20261013_215903` 使用GPU1。
先等待A恢复版和 `centroid02_paired_20261013_204924` 均complete、锁释放、进程退出和GPU空闲，
再校验A诊断纯度差≥0.05、完整50轮/1449图及所有输入、签名、检查、折模型、基线和A结果SHA。
每组额外训练一次seed42/50×100并完整1449图严格HCR，中心权重0.2；附原A评测只读参考。
基线及总/逐图回放、折划分、折模型、交叉检查和诊断全部复用，不重新SAM3/GMP或清理缓存。

```bash
python -B tools/experimental_crossfit_feedback.py --version NEW_B \
  --feedback-source crossfit_feedback_recover_20261013_214723 --variants B --gpus 0 \
  --gate-policy explicit-bypass --reference-policy repeated --preflight-only
# C使用另一个新版本、--variants C --gpus 1。预检通过后后台去掉--preflight-only。
```

本对话已创建每5分钟条件检查的heartbeat自动任务，ID `b-c`，名称“当前实验结束后衔接B/C”。
未变化时静默；条件满足且未启动时按冻结计划各占一张卡后台启动，记录PID与验收，避免重复启动。
如果complete_no_signal，不启动B/C而记录skipped_no_signal；异常先查日志并保留历史。
用户后续暂停/取消时停用自动衔接；B/C均完成后核验、报告并停用自动任务，不自动部署或加三seed。
自动任务需要本地Codex保持运行；模型训练和已启动的服务器进程自身脱离SSH执行，邮件沿用已授权配置。
后台B/C启动器日志必须用计划指定的logs根目录文件，不在正式启动前预建logs/<版本>。

恢复验收：七个共同阶段的reused_from和源SHA全部核验，诊断已完成并进入train_A。
共26个前景区域仅含255，计入不可测统计；其余前景一致组23914区域、平均伪类别纯度
81.8566%，不一致组371区域、平均69.3147%，差12.5418个百分点，超过预设5个百分点门槛。
该统计不赋给样本任何GT权重，也不移除26个不可测区域；A仍按原g阈值写全库sample_weights。
三seed同时推进至eval_seed2026_center04。B/C等待两个供体完整结束后才启动，没有抢占当前GPU。
后续冻结计划及自动任务信息分别为pending目录的plan.json、automation.json；本地已有plan.json副本。

前置实验完成验收（北京时间2026-10-14）：A恢复版与中心权重配对三seed均complete，
各阶段输入/代码快照/产物SHA通过，启动器与worker退出、锁释放；结束邮件均被SMTP接受。
A训练完整50轮，best_epoch=23，保持34,880个原库成员、原内部图片划分与背景权重1；
373个前景区域仅按交叉检索g<0降权至0.3。严格评测1,449图，逐图矩阵求和等于总矩阵。

| 方案 | mIoU (%) | 相对中心0.2原基准（百分点） | 最佳轮次 |
|---|---:|---:|---:|
| 原基准 seed42 | 86.821717 | 0 | 38 |
| A：交叉检索不一致区域CE降权 | 86.473929 | -0.347788 | 23 |

A背景误报2,479,065像素（原基准2,411,611），前景漏检4,543,486（原4,359,445），
前景类间混淆472,964（原486,861）。诊断能区分平均纯度不等于该固定降权规则改善最终分割；
本轮A单seed没有收益，不自动部署，也不据此否定所有交叉检索反馈配方。

| seed | 中心0.4 mIoU (%) | 中心0.2 mIoU (%) | 配对差（百分点） | 最佳轮次 |
|---|---:|---:|---:|---:|
| 42（复用原MLP） | 86.735021 | 86.821717 | +0.086696 | 38 |
| 123 | 86.383729 | 86.321692 | -0.062036 | 29 |
| 2026 | 86.607327 | 86.374583 | -0.232744 | 37 |

配对均值-0.069361个百分点、样本标准差0.159846、Student-t n=3/df=2的95%区间
[-0.466440, +0.327717]个百分点，仅1/3 seed正向。0.4/0.2三seed均值分别86.575359%与
86.505997%；当前没有中心0.2稳定提升的证据，小样本区间不能证明两配置等价。
seed123/2026均完整50轮、固定原图片划分与导出成员顺序，每组两配置共用同一MLP和库，
全部正式评测覆盖1,449张；2图probe不计入正式评测。历史总矩阵及两次新回放逐图矩阵精确核验。
本轮B/C仍按事先冻结的中心0.2比较，不因三seed结果中途改参数或更改主线。

B/C CPU预检均passed，A继续门槛12.541848个百分点通过。新后台启动器于2026-10-14
创建：B PID235175（GPU0）、C PID235176（GPU1），正在引用共同阶段并再次校验SHA；
启动器脱离SSH，邮件配置沿用计划。正式worker启动验收另行记录，不将启动器创建当成训练完成。
前置完整性证据在pending目录`audit_20261014_090439`，冻结plan.json不改写；
调度状态、PID和命令分别新增`status.json`、`launches.json`，避免重复启动。

B/C正式启动验收（北京时间2026-10-14 09:14）：两启动器存活，各卡仅一个训练worker，
B worker PID235355、C worker PID235354，均已完成第5/50轮，训练日志持续推进。
B按冻结规则剔除186个前景区域，新库34,694成员；C独立从原库改标签14个区域，
保留34,880成员。两组10,569个背景区域的ID、类别及原始向量逐元素不变；
全部保留向量与来源一致、sample_id保留，B每类15%与至少两图、C图像级允许前景及
改入/改出10%上限核验通过，C记录relabel_from。八个共同阶段均只读引用，不再次训练折模型或A。
每组固定seed42、原内部图片划分、50×100步，后续评测全部1,449图，中心权重0.2；
此时尚无B/C正式mIoU。不可变快照及来源SHA通过，后台GPU进程分别为235355/235354。
各版本`startup_validation.json`及pending目录同名文件保存验收；pending状态更新为running，
heartbeat继续仅在完成、失败或需要用户处理时报告，完成后停止，不重复启动或追加实验。

B/C完成验收（北京时间2026-10-14）：两组于09:36结束，分别耗时1767.842/1767.741秒，
完整50轮×100步，seed42，中心0.2；每组严格评测1,449图。输入/阶段/快照SHA、原始向量一致性、
导出成员顺序、冻结内部图片划分、逐图矩阵求和及mIoU重算通过。B/C保护规则、背景不动、
C的允许前景和relabel_from均通过；启动器与worker退出，锁释放，两卡空闲。

| 方案 | mIoU (%) | 相对原基准（百分点） | 相对A（百分点） | 最佳epoch |
|---|---:|---:|---:|---:|
| 原基准 | 86.821717 | 0 | +0.347788 | 38 |
| A：降权 | 86.473929 | -0.347788 | 0 | 23 |
| B：剔除186区域 | 86.461033 | -0.360684 | -0.012896 | 26 |
| C：改标签14区域 | 85.860918 | -0.960799 | -0.613011 | 18 |

| 方案 | 背景误报像素 | 前景漏检像素 | 前景类间混淆像素 |
|---|---:|---:|---:|
| 原基准 | 2,411,611 | 4,359,445 | 486,861 |
| A | 2,479,065 | 4,543,486 | 472,964 |
| B | 2,390,508 | 4,463,309 | 553,978 |
| C | 2,323,723 | 4,410,368 | 837,242 |

B相对基准椅子/公交/船分别下降3.5446/3.3852/2.0250点，汽车提高2.0273点。
C羊/船/火车/椅子下降4.9216/3.6062/3.4697/3.4148点，盆栽提高4.7006点。
背景误报减少没有抵消前景漏检和类别混淆。最佳epoch由训练图片内部的伪标签检索指标选择，
不使用VOC val；较早最佳轮次不是提前停止，三组均训练满50轮。
本轮固定规则、单seed42的A/B/C均没有提升，建议保留原训练和建库方式，不采用这三种反馈规则；
不将结果解释为全部交叉反馈方向无效，不自动部署或追加三seed。

最终验收及四组21类IoU、按类修改数量、错误像素对比在
`experiments/crossfit_bc_pending_20261013_215903/completion_20261014_093849`：
`comparison.json`、`B_completion_validation.json`、`C_completion_validation.json`、
`results_report.md`、`per_class_comparison.csv`。各B/C版本另有`completion_validation.json`；
pending状态为complete，冻结plan.json保持不变。两封结果邮件均首次被SMTP接受，
附件summary.json、metrics.json、per_class.csv，不声称已确认收件箱投递。
自动任务`b-c`已通过Codex工具停用（PAUSED），第48节仍保持42/52人工暂停。

## 52. GMP＋边缘patch＋高覆盖率内部patch三库实验（2026-10-14）

用户确认第三库为高覆盖率内部patch，并授权后台实验。独立入口
`tools/experimental_threebank_retrieval.py`，专用测试`tests/test_threebank_retrieval.py`；
不修改生产src/configs、原GMP/MLP、旧入口或历史产物。默认双卡物理GPU0/1、seed42，
原主线中心0.2，基准86.8217170759%。来源为完整A恢复版中的原主线输入与原生查询缓存，
整图token只读取第49节已完成来源；不采用A/B/C反馈模型、剔除或改标签。

原GMP经现有MLP保留256维；边缘/内部库保存L2归一化的原1280维SAM3 token，不经过区域MLP。
只从原内部train_images的保留区域建辅助库，原validation_images不进入图库。
原图像素积分给出准确patch覆盖率，距离在1008坐标计算，中心采用原stride14/floor映射。
边缘：中心在区域内、覆盖≥0.5、内侧边界距离≤14；内部：覆盖≥0.95、距离>14。
两集合互斥，不把图像边框作为类别边界；无合格支持时弃权，不强行选最近token。
背景作为类别0负样本使用同一规则，训练GT不参与选patch或赋类。

每区域最多8个、每图每类最多16个、每类每库最多4096个patch，使用固定SHA排序选择，
记录图片/父区域/sample_id/token位置/覆盖率/距离及来源凭据。不会把多个patch平均成图库向量。
每个query patch按类先取每张图库图片的最高相似度，再均值最高3张不同图片；
同一图片不能因重复patch占据多票。区域查询最多32个patch，≥10个时按类截去两端10%后取均值。
同图重复token仅检索一次，score检查点由五组共享。

原内部留出图片按固定SHA分校准与诊断两部分；温度仅在校准部分，用伪标签类别的平衡CE拟合
80个预设对数温度0.01–1，不读取训练像素GT，不用VOC val选温度、权重或阈值。
不足3张图库图片的类别不提供patch证据；该类概率质量保持原GMP，不能当作反对证据。
查询支持少于3个时弃权，达到8个时满权重；edge最大0.1、interior最大0.2，余下归GMP。
固定五组：baseline、gmp_interior、gmp_edge、three_fixed、three_guarded。
guarded改类要求原GMP分差≤0.05、两个辅助分支均可用且共同支持融合后的新类别。

三库仅改变clean候选的分类；排序使用新类别的原GMP分数与原SAM分数，融合概率只用于选类及
clean margin，不能直接与原HCR的0.95门槛比较。raw/HCR附加候选保留原GMP类别、分数、margin
和阈值；原mask几何、候选、NMS参数、rasterize规则不改。clean改类可能影响NMS分组和HCR占用，
属于最终流程效果，不声称像素排序或预测不变。

先检查两卡空闲、输入和旧快照SHA、全部12031个token凭据、图片/编码签名与缓存尺寸，
大token内容SHA在消费前检查；预检保留16GiB新增空间加5GiB底线。真实两卡探针各使用8张原训练图，
构建小patch库、做去重检索、缓存读回；验证图前两个mask的重聚GMP须与原1280查询缓存一致，
随后每卡做一图原MLP/融合/严格HCR。探针不作为正式精度。
正式首先完整回放1449图基准，总和逐图矩阵必须与来源中心0.2记录精确相同，之后建全辅助库、
提取内部留出查询、拟合温度、生成全val共享patch分数，再做四组1449图严格评测。
全部预测完成后才做验证GT辅助诊断，报告原GMP错误被纠正/原本正确被误伤及纯度≥0.9子集统计。
诊断针对候选多数类，不等同实例GT或最终像素因果归因；不将诊断反馈到参数。

数据和patch库/逐图检查点在`storage/raw_banks/voc/<版本>`，参数/校准/指标/逐图矩阵在
`experiments/<版本>`，worker日志在`logs/<版本>`；不新训MLP或复制原权重。
每图检查点采用独立attempt，完成记录校验SHA后恢复；残缺attempt保留，不能覆盖完成缓存。
阶段失败新attempt，`--resume`要求代码/规则/输入/图片划分完全一致，运行使用不可变快照。
后台脱离SSH，每卡至多一worker，SIGTERM仅结束自有worker，无整体实验到期时间，保留5GiB底线。
完整/失败/中断后由启动器发送邮件，附件summary.json、metrics.json、per_class.csv，SMTP有限重试。
不自动部署、追加阈值搜索或三seed，也不恢复第48节的剩余网格实验。

```bash
python -B -m unittest tests.test_threebank_retrieval -v
python -B tools/experimental_threebank_retrieval.py --version NEW_VERSION --gpus 0 1 --preflight-only
python -u -B tools/experimental_threebank_retrieval.py --version NEW_VERSION --gpus 0 1 --probe-only
# ready后同版本同参数加--resume，移除--probe-only，日志采用logs根目录的新文件。
```

初步实现已有13项服务器CPU测试通过；正式预检、两卡探针与后台启动验收另行追加，
未完成正式评测时不声称获得新精度。

第52节启动验收（北京时间2026-10-14 11:57）：正式版本
`threebank_20261014_112935`，启动器PID241957，后台命令为
`python -u -B tools/experimental_threebank_retrieval.py --version threebank_20261014_112935 --gpus 0 1 --resume`。
日志`logs/threebank_20261014_112935_launcher.log`，通知沿用已授权QQ配置。
13项服务器CPU测试通过，12,031个token凭据和全部原输入/源码/权重/参数签名预检通过。
两卡各8张训练图的真实patch建库/检索/严格HCR探针通过，原GMP重聚与旧查询一致；
额外8张留出图完成真实温度校准联通检查，固定划分中校准与诊断图片不重叠，不读训练像素GT。
最初小样本恰好全部落在诊断半折，失败fixture和已生成检查点保留；改为固定半折各4张完成测试，
未改变正式划分、算法或阈值。完整留出集为校准1052图、诊断1064图。
探针ready摘要与启动/测试记录在`experiments/preflight_audits/threebank_20261014_112935`。

正式已完整回放1449图基准，86.8217170759%，历史总/逐图矩阵逐元素一致，随后进入gallery_build。
GPU0/1 worker PID242141/242142，每卡仅一worker；验收时已有1,402张训练图片的patch检查点，
首批数组/父区域/图库图片集合/SHA检查通过，日志推进；新增方法精度尚未产生。
原MLP/原始区域库/伪标签/raw及clean提议保持只读，整图token不复制，不运行SAM3或重训MLP。
不可变代码摘要`19b5f10efb97e761cf4d0fb4730b2c4c6fb9fb699f6cc7f997c6eddec30c8ba2`，
入口SHA`b3e56fb9e222567655c878592a98185f2513bdc09e5de2e1851a5730cb433ff7`。
当前状态、命令、验收分别在同版本`status.json`、`background_launch.json`、`startup_validation.json`；
最终四个融合组与基准共五组，全部严格1449图后才标记完成，随后邮件发送指标和逐类CSV。

第52节完成验收（北京时间2026-10-14 13:04完成，随后独立复核）：
`threebank_20261014_112935`五组全部complete，每组严格覆盖同一1,449张验证图，
总耗时约73.75分钟；原GMP基准总和逐图混淆矩阵与历史中心0.2记录逐元素相同。

| 配置 | mIoU (%) | 相对原GMP (百分点) |
|---|---:|---:|
| 原GMP＋MLP | 86.821717 | 0 |
| GMP＋内部patch | 86.755176 | -0.066541 |
| GMP＋边缘patch | 86.765263 | -0.056454 |
| 三库固定融合 | 86.646325 | -0.175392 |
| 三库保护式融合 | 86.579516 | -0.242202 |

边缘库80,887个patch，内部库80,277个patch，各含4,096个背景负样本；21类均有至少3张
不同图库图片。图库8,466图，留出校准1,052图、诊断1,064图，不重叠。
正式温度GMP/edge/interior分别0.05747694/0.01060026/0.01000000，均用固定温度网格和
训练伪标签平衡CE拟合，无训练像素GT或VOC val选参；内部温度达到预设下界，未扩大搜索。
3,609个clean候选中，edge至少3个支持的有3,598个，interior和双分支均至少3个支持的有3,166个；
edge/interior达到8个满权重支持的分别3,436/2,994个。

GT仅作全部预测完成后的候选多数类诊断：有有效GT的clean候选3,481个，纯度≥0.9的3,296个。
内部库改判7个，纠正2、误伤5；边缘库改判5个，纠正1、误伤4；固定三库改判11个，纠正3、误伤8；
保护式三库改判10个，纠正2、误伤8。所有改判也在高纯度子集中；这些候选统计不等同最终像素因果归因。
双分支共同同意仍有8个误伤，当前保护规则不足以提供可靠纠错。椅子IoU在两种三库融合中均下降3.9951点。
固定三库虽使汽车提高1.3727点、减少55,630个背景误报，却新增82,720个前景类间混淆像素；
保护式没有保留汽车与背景误报收益，前景类间混淆同样新增82,720像素。

独立复核通过42,146项输入SHA、284个源码及快照SHA、12阶段和31,752项阶段产物SHA；
全部逐图矩阵之和等于总矩阵，21类IoU和mIoU复算一致。逐图图库/校准/验证查询凭据分别覆盖
8,466/2,116/1,449图。五组5,184个raw/HCR附加候选的类别、原GMP分数、margin及顺序完全一致。
原MLP、原输入保持不变，未运行SAM3或新训MLP；启动器和worker退出，锁释放，两卡空闲。
结果邮件首次被SMTP接受，附件summary.json、metrics.json、per_class.csv，未确认收件箱投递。

本轮固定配方、单seed42开发评测没有收益，保留原GMP＋MLP主线，不采用三库融合；
不能据此否定全部边缘/内部patch表示或多库方向。未自动部署、追加搜索/三seed或恢复第48节。
完整五组21类IoU、像素错误、候选诊断、验收与对比在同版本
`completion_20261014_131210/results_report.md`、`per_class_comparison.csv`、
`completion_validation.json`、`comparison.json`、`support_statistics.json`；
原始summary.json、metrics.json、per_class.csv、逐图矩阵、检查点和不可变快照均保留。

## 53. 三库互补性事后诊断（2026-10-14）

独立CPU入口`tools/diagnose_threebank_complementarity.py`，测试
`tests/test_threebank_complementarity_diagnostic.py`。默认只分析第52节已完成的
`threebank_20261014_112935`，不新训练、不运行SAM3、不生成提议、不改参数或主线。
新版本产物放`experiments/<诊断版本>`，启动日志放`logs/<诊断版本>_launcher.log`，
源码快照、输入身份、分阶段attempt及完成产物SHA记录在诊断版本；不覆盖原实验。

原辅助分数可直接读缓存，但val的原GMP完整21类分数没有保存，因此用已有1280维区域向量、
原MLP、原导出库，在CPU调用生产balanced分类器补算。必须全部复现原候选类别，
原GMP分数与margin误差不超过4e-6，完成提交后才读取val像素GT作诊断。
该过程不重新提取特征；诊断GPU可见性清空，计算线程固定4。

只将GT多数类纯度≥0.9的clean候选纳入主统计；raw/HCR附加候选单独排除。
同时报告前景子集、面积≥300子集及有效GT覆盖率≥0.9子集。
按辅助分支是否有至少3个patch支持区分可用与弃权，报告在GMP判错时的辅助正确率、
在GMP正确时辅助判错率、三分支二值错误的phi/Pearson相关性及配对四格计数。
报告GMP错误候选中GT类别位于top2、margin≤0.05和两者同时满足的比例，
以及oracle候选正确率上限；这些候选指标不等于语义mIoU。

全部11个固定三库改判候选导出类别、GT直方图、margin、支持量和原图覆盖图；
不按val挑候选或阈值。语义GT可检查不同类别混合，不能判断同类实例合并或真实感受野。
训练端只复用第51节的region_quality记录并核对原chair/sofa保留区域几何，
读取已有GT计算这两个伪类别对21类的像素混淆；统计范围为过滤后原库成员，不冒充全PNG质量。
GT不进入新权重、成员、改标签、训练或参数。

已有五组1449张逐图混淆矩阵作2000次seed42配对图片bootstrap，给出mIoU差的百分位95%区间。
该区间只反映当前seed和开发验证图像抽样，不能表示训练seed方差或独立测试泛化。
3项CPU测试验证弃权分母、错误相关性常量处理和配对bootstrap一致性。

```bash
python -B -m unittest tests.test_threebank_complementarity_diagnostic -v
python -B tools/diagnose_threebank_complementarity.py --version NEW_DIAGNOSTIC --preflight-only
python -u -B tools/diagnose_threebank_complementarity.py --version NEW_DIAGNOSTIC
# --resume仅复用相同输入、源码及固定诊断参数的完成阶段。
```

第53节完成（`threebank_diag_20261014_133658`）：3项CPU测试通过，预检通过，
CPU启动器PID244549，4个阶段完成；阶段计算约147.13秒（不含预检）。
8,793个候选原GMP类别全部复现，分数和margin最大误差4.7684e-7；
20项阶段产物SHA、286项源码/快照SHA及源manifest/summary/metrics/region_quality核验通过。
原预测、模型、库、伪标签、提议不变，任务退出且锁释放，不占GPU。

高纯度clean候选3,296个，含前景3,095个。全部候选GMP错273个：
edge可用错误候选中正确9/270（3.333%），interior正确13/191（6.806%）；
至少一个正确辅助分支17/273（6.227%）。仅前景GMP错72个：edge正确7/71（9.859%），
interior正确6/56（10.714%），任一辅助正确9/72，双辅助均正确4/72。
相反在GMP正确的前景里，edge单独错277/3016、interior单独错117/2747；弃权不计为正确证据。
共同可用前景2,803个的错误phi为GMP–edge 0.3471、GMP–interior 0.5026、edge–interior 0.5548。
这些是候选分类诊断，不能当作最终mIoU。

前景72个GMP错误中，GT在top2为27/72，margin≤0.05为9/72，两者同时为8/72；
仅top2且小margin的重排序范围有限，但top2缺失不证明所有全类别方法都不能纠正。
全部11个固定三库改判已保存覆盖图：误伤为7个椅子→沙发和1个牛→狗，
纠正为椅子→沙发、摩托车→自行车、汽车→背景各1个。
所有11个有效GT多数纯度≥99.2963%，不能把这些具体误伤归因于混合了两个语义类别。
有效GT覆盖率另有单独统计，语义GT不能判断同类实例合并。

原库可测椅子2,485区域、沙发892区域（共1,416训练图），与交叉拟合缓存纯度逐项一致。
伪椅子GT多数为沙发22/2485（0.885%），伪沙发多数为椅子3/892（0.336%）；
椅子区域中沙发像素1.452%、背景14.313%，沙发区域中椅子像素0.0245%、背景5.692%。
椅子/沙发区域平均自身类别纯度67.023%/87.071%，像素加权自身纯度82.706%/93.815%。
直接互相错标不多，椅子背景污染更突出；不将区域污染直接等同所选patch污染或证明因果。
内部5-NN原区域宏准确率92.8271%，MLP best_epoch38为98.3774%，目标是伪标签类别；
不能据此推出逐patch必然更弱，也不能当作val像素精度。

2000次配对图片bootstrap的mIoU差95%百分位区间（百分点）：
gmp_interior[-0.550650,+0.484441]，gmp_edge[-0.468737,+0.468993]，
three_fixed[-0.698356,+0.419041]，three_guarded[-0.763246,+0.328901]。
均跨0；只是当前训练seed下的开发图像抽样区间，不表示训练seed方差或独立测试泛化。
当前原始patch辅助互补弱、额外错误多，不继续val权重/阈值扫描；
拼接多区域GMP、结构/上下文或混淆对判别器仍待训练集交叉拟合验证，本轮未自动启动。
交叉反馈A/B/C已无收益，不因本次标签诊断自动重跑旧规则。

报告`experiments/threebank_diag_20261014_133658/results_report.md`，
验收`completion_validation.json`，详情在`summary.json`及4个阶段attempt。
11个候选CSV及总览图在`validation_diagnostic/attempt_001/changed_candidates.csv`、
`changed_candidates_contact_sheet.png`，逐候选图在`candidate_overlays/`，无选择地保留全部改判。

## 54. 超算集群本地代码同步与Slurm入口（2026-10-14）

用户授权使用提供的私钥连接`zhengyuan@hpc.chinahpc.com:1014`，ED25519主机指纹
`SHA256:2DUyHqSOJXeGStmxEzPt89dP55+UL8fZCLGe2H5h2zo`已匹配。
项目`/data/home/zhengyuan/25-nd/1010`，环境
`/data/home/zhengyuan/miniforge3/envs/moduseg_sam3`，分区`v100g32`。
先更新集群代码；检索回填损失的新实验按用户选择在原服务器运行，复用原服务器缓存。

同步版本`hpc_code_sync_20261014_162002`，审计和旧代码备份位于同名experiments目录。
本地目录无Git元数据，因此以本地逐文件SHA清单为代码来源；集群原Git提交
`315de02763df81892fe43d10ff29d700769b2dad`及.git历史保留，本次工作区同步不等于GitHub发布。
只同步代码、测试、配置、文档和必要SAM3词表/许可证；data、storage、weights、旧experiments/logs、
Conda环境、SSH密钥及邮件配置不作为上传内容。改动文件先备份，失效的旧受Git跟踪代码可归入
旧代码备份；不处理服务器未知的用户文件，不覆盖旧探针或实验完成凭据。

旧`deploy_20261013_sam3voc`仍仅为旧代码双卡探针ready，历史输入、快照和结果保留。
更新后的代码应选新版本，在计算节点重做preflight/probe；不能拿旧ready冒充新版探针通过。
集群上没有原服务器完整GMP库、MLP和提议缓存；复制代码不等于迁移这些历史产物。

新入口`tools/hpc_mainline.sbatch`沿用2GPU/8CPU/96GiB/48小时资源；
`tools/hpc_gpu_allocation.py`解析Slurm分配，核对CUDA可见设备与nvidia-smi物理ID。
GPU数量、UUID、物理ID不一致时停止，不猜测共享节点设备。没有Slurm任务环境时拒绝运行。
48小时到期后只在旧任务退出、代码/参数/输入相同并获得相容GPU分配时按同版本续跑。
完整运行前检查个人配额；df文件系统总剩余不代表个人配额，quota无输出不视为已确认配额充足。

```bash
cd /data/home/zhengyuan/25-nd/1010
sbatch --export=ALL,MODE=preflight,VERSION=NEW_CLUSTER_VERSION tools/hpc_mainline.sbatch
# preflight成功后，同一VERSION提交MODE=probe；probe成功后，经用户授权提交MODE=full。
```

`tools/hpc_validate.sbatch`用于短代码验收任务：1GPU/2CPU/8GiB/15分钟，测试时禁用CUDA，
验证主线、检索、HCR、清理、通知附件、交叉反馈、三库和互补诊断的CPU测试及真实CLI。
该任务不运行SAM3或模型训练，不把CPU测试当作真实GPU探针或mIoU。

```bash
sbatch --export=ALL,AUDIT_DIR=/data/home/zhengyuan/25-nd/1010/experiments/hpc_code_sync_20261014_162002 \
  tools/hpc_validate.sbatch
```

同步清单、原代码SHA/备份、应用记录与逐文件验收分别为同名审计目录内
`local_manifest_initial.json`、`remote_before.json`、`apply_report.json`、`verification.json`；
Slurm测试任务与日志以`validation_submission.json`、`tests.log`、`validation.json`为准。
私钥只由本机SSH客户端读取，不进入同步清单、普通提示词、日志或Git。

同步验收：首轮Slurm任务3672435完成、81项CPU测试通过；补同步检索回填实验后的
`hpc_code_sync_20261014_171938_followup`核对419个文件SHA全部一致，任务3672969
COMPLETED、退出0:0，87项CPU测试和新入口CLI通过。更新不启动集群全量训练。
旧代码备份与首轮审计保留，操作记录不替代新版SAM3真实GPU探针。

## 55. 检索回填误差直接监督MLP（2026-10-14）

用户确认目标为：隐藏查询区域类别，用库检索出21类分数，再把回填分布与保留的原伪类别比较，
检索误差直接反传到MLP。与第50/51节离线降权/剔除/改标签不同，本轮不改标签、权重或成员。
按用户选择在原服务器`/data/home/zystu2025/nd/1010-main`运行，新集群先同步最新代码。

独立入口`tools/experimental_retrieval_reconstruction.py`；训练器
`tools/train_retrieval_reconstruction.py`；可微分数在`tools/retrieval_reconstruction_core.py`；
专用测试`tests/test_retrieval_reconstruction.py`，不修改生产src/configs或旧实验入口。
固定复用原34880成员、1280维原GMP向量、内部图片划分、原clean/raw提议和1449图原始查询向量。
SAM3冻结，不重新提取特征，不读取训练像素GT，不使用第50/51节GT诊断门槛或其逐区域结果。

两组seed42从同一初始化配方重新训练，两卡各一组：GPU0为原v1 CE＋0.5 SupCon的生产训练器；
GPU1在完全相同50轮×100步、8类×4区域、1280→512→256、AdamW lr=0.001、梯度裁剪5与余弦学习率上
添加1.0×retrieval CE。固定temperature=0.1，不在VOC val上选损失权重或温度。
原内部留出图不参加梯度或训练图库，两组沿用同一内部伪标签5-NN宏准确率选模；不使用val选epoch。

每10步用当前MLP无梯度投影内部训练区域，刷新detached图库。查询区域用可训练MLP投影；
查询原类别不传入检索，仅在分数计算后作为CE目标。按生产balanced规则：
K25、每类top5、搜索倍数2、中心权重0.2，最强邻居均值与类中心融合。
同图全部成员从近邻候选和类中心两处排除；某类排除后无成员则拒绝，不用同图回退。
没有局部邻居时保留生产类中心回退。图库和类中心停止梯度，梯度经选中的top-k分数传到查询MLP；
硬类别argmax只作报告，不作为可微损失。背景仍参加原CE及retrieval CE，SupCon背景规则不变。

预检绑定原库、模型、提议、查询顺序/几何、代码和内部划分SHA，缺失停止，不自动重建SAM3。
真实两卡短训练与匹配单库导出后，各做一图严格HCR；检索组额外验证真实梯度非零，
4个真实查询与排除同图后生产分类器的21类分数最大差≤6e-6。
正式先1449图精确回放旧中心0.2基准的总和逐图矩阵，再双卡训练两组，最后每组1449图严格HCR。
报告相对历史及同轮control的mIoU、21类IoU、错误像素、最佳epoch、loss分项、耗时及显存。
仅固定seed42开发对照，不自动部署、调阈值或追加seed，不恢复第48节。

数据库输出`storage/mlp_banks/voc/<版本>/<case>/<attempt>`，新权重
`weights/sam3_adapter/<版本>/<case>/<attempt>`；每个检索组完整epoch保存模型、classifier、
optimizer、scheduler及随机状态检查点。中断后新attempt从同版本相同binding的完整epoch继续，
图库在epoch边界重新投影；原control失败阶段新attempt重训，所有旧attempt保留。
参数、指标、不可变源码与阶段凭据在experiments，日志在logs，保留5GiB磁盘底线，无整体到期时间。
完成/失败/中断有限重试邮件，附件summary.json、metrics.json、per_class.csv，邮件失败不改结果。

```bash
python -B -m unittest tests.test_retrieval_reconstruction -v
python -B tools/experimental_retrieval_reconstruction.py --version NEW_VERSION --gpus 0 1 --preflight-only
python -u -B tools/experimental_retrieval_reconstruction.py --version NEW_VERSION --gpus 0 1 --probe-only
# probe达到ready后，同版本同参数去掉--probe-only，加--resume后台启动。
```

测试覆盖生产分数等价、同图近邻/中心排除、只有查询模型获得梯度、背景监督及空类拒绝。
本节的实现与探针不等于正式提升，最终状态以新版本summary.json为准。

本次正式版本`retrieval_reconstruction_20261014_164724`。19,358个固定输入SHA预检通过，
两卡探针116.750秒完成；检索梯度范数0.513270，真实生产分数最大差2.38419e-7，
两组均导出34,880成员并各完成一图严格HCR，完整epoch检查点安全加载通过。
北京时间2026-10-14 17:22:58脱离SSH后台启动正式任务，启动器PID246694；
先双卡历史回放，再GPU0 control/GPU1 retrieval训练及全量评测。
根日志`logs/retrieval_reconstruction_20261014_164724_launcher.log`；预检与启动凭据
`experiments/preflight_audits/retrieval_reconstruction_20261014_164724`。
启动状态以对应status.json为准；尚无新mIoU结论，不用两步探针结果判断收益。
启动验收时旧基准1,449图的总矩阵及逐图矩阵均逐元素一致，mIoU=86.8217170759%；
两组正式worker已进入50轮训练，loss日志和检索组逐epoch检查点持续推进。

本轮于北京时间2026-10-14 17:50:41完成，正式入口计时1641.004秒（27分21秒，不含先前探针）。
2026-10-14 19:28完成事后验收：6个阶段共296项产物SHA通过；两组均50轮×100步、
保留34,880成员及原内部图片划分；历史、control和retrieval均覆盖1,449图，逐图矩阵之和
等于各自总矩阵，control的总矩阵和逐图矩阵均与历史逐元素一致；启动器退出且锁释放。

| seed42方案 | mIoU | 相对同轮control（百分点） | 最佳epoch | 内部5-NN宏准确率 |
|---|---:|---:|---:|---:|
| control：原CE＋0.5 SupCon | 86.821717% | 0 | 38 | 98.377400% |
| retrieval：增加1.0检索CE | 86.253598% | -0.568119 | 32 | 98.424244% |

retrieval相比control的背景误报像素增加135,988、前景漏检增加86,642、前景类间混淆增加35,530。
主要逐类下降为chair -5.526135点、bottle -4.008988点、bird -1.756173点、ship（VOC boat）
-1.507772点；sofa +0.650220点、tvmonitor +0.567487点、sheep +0.486375点。
反馈训练的retrieval CE从首轮0.796792降至末轮0.029568，训练批检索准确率从82.34375%
升至99.5%，确认损失生效，但这不等于真实像素语义或在线分割改善。
这是固定权重1.0、温度0.1、seed42的开发集对照结果，不否定所有检索反馈配方；本轮不推广。

服务器结果在`experiments/retrieval_reconstruction_20261014_164724/`，包含`summary.json`、
`metrics.json`、`per_class.csv`及`completion_acceptance_20261014.json`。
本地验收副本在同版本的`completion_verified_20261014/`；下载文件SHA须与验收凭据一致。
`notification.json`记录首次SMTP接受成功，未确认收件箱投递；不重复发送。

## 56. 集群主线基线从头计时实验（2026-10-14）

用户明确授权在集群后台完整跑现行主线，测量从头生成到评测的实际耗时。
新版本`cluster_mainline_baseline_20261014_175149`，项目与环境沿用第54节；
不引用旧`deploy_20261013_sam3voc`的ready，也不迁移原服务器实验的库或模型。
配方完全沿用第46节：presence1、概率0.4类别并集边缘修复、几何清理、
冻结SAM3/FP32/1008、rho0.1硬中心GMP、前景统一10%过滤、MLP2/v1/seed42/50×100，
中心权重0.2、原HCR，全10582 trainaug和1449 val；保存整图FP32特征。
这是同配方的新从头运行，不保证复现旧单seed开发分数。

新增调度包装`tools/hpc_mainline_baseline.sbatch`和`tools/run_hpc_mainline_baseline.py`，
不改主线生产代码。一个Slurm分配连续执行preflight→真实两卡probe→full --resume，
使用实际分配的两张V100 32GiB物理GPU，每卡至多一个worker。
集群设备隔离会将宿主机Slurm编号重新映射为作业内NVML/CUDA的0、1。
`hpc_gpu_allocation.py`仅在可见设备恰为两卡、当前Slurm job cgroup确认、
CUDA与NVML的实际UUID逐项一致时接受该映射；重复、缺失、较宽设备集合或UUID错配仍拒绝。
宿主机编号、作业内编号和UUID证明记录在`hpc_launch_audits/<版本>/allocation_<JobID>.json`。
2GPU/8CPU/96GiB，单作业48小时；提前180秒发SIGTERM，包装只通知自己的启动器退出，
主线启动器负责停止自身worker、保留检查点与失败/中断记录。
续跑必须原作业和worker退出、锁已安全释放、相同代码/输入/参数且GPU分配相容；
不得修改旧manifest或降低精度。旧尝试与日志保留。

GPFS用户quota/limit为0，表示该层不限制；所属zhengyuan组限额1000GiB，
启动前用量约32GiB、剩余约968GiB；root fileset无额外限额。
正式全量前按manifest的estimated_bytes检查真实组配额，续跑扣除本次已分配的产物空间；
运行每60秒检查组剩余不少5GiB，主线本身继续检查文件系统空间。
任何配额读取失败或不足均停止自己任务，不用df的全系统剩余冒充个人可用空间。

```bash
sbatch --export=ALL,VERSION=cluster_mainline_baseline_20261014_175149 tools/hpc_mainline_baseline.sbatch
```

提交与预检依据在`experiments/hpc_launch_audits/<版本>`；作业包装计时在
`experiments/hpc_jobs/<版本>/job_<JobID>/timing.json`，分别记录preflight/probe/full。
主线结果仍在`experiments/<版本>/summary.json`，模型、库、提议和日志按第46节目录。
排队时间使用提交记录和sacct的Submit/Start单独计算，实际GPU分配耗时用Start/End及包装计时；
正式流水线耗时用summary.seconds和阶段worker报告，不把两张卡并行worker耗时相加当总耗时。

Slurm启用END/FAIL通知2858372757@qq.com。登录节点QQ SMTP域名解析和直接IP外连实测失败，
尚未确认计算节点SMTP出口；Slurm邮件依赖控制节点MailProg，启用参数不等于确认投递。
结果SMTP失败不改变实验结果。测试覆盖真实组配额和未落账块扣除，以及probe失败不启动full。
完成必须由主线核验10582标签/全部提议/50轮/1449严格评测后标为complete。

首次作业3673337因旧GPU编号核对在23秒内失败，尚未创建实验manifest或训练产物，历史日志保留。
独立两卡映射探针3673345完成、退出0:0，宿主机1/6在作业内映射为0/1，CUDA和NVML UUID一致。
修复限于集群设备启动适配，原服务器运行代码和主线算法不改动；新提交继续通过完整预检及模型探针。

正式作业3673359：北京时间18:12:30开始，节点c2v100m2-2，宿主机GPU2/4映射为作业内0/1。
完整数据预检311.317秒通过；双卡真实probe全部complete，单卡峰值分配5229.599MiB。
全量入口18:32:38进入，伪标签双卡worker为1216255/1216256，18:52:44验收时已保存425张。
首批PNG原尺寸和取值、分片不交叉覆盖、输入/探针产物SHA与生成检查点绑定均分别验收；
日志正在推进，不将synthetic probe的mIoU作为正式结果。
启动、检查点和探针凭据分别为同名hpc_launch_audits目录内startup_validation.json、
checkpoint_startup_validation.json和real_probe_acceptance.json。

本对话跟踪任务`automation`每30分钟核查一次，仅报完成、异常或需用户行动，健康运行静默。
Slurm邮件未确认投递时，获用户本对话已有授权，通过原服务器私有SMTP配置补发结果及三个指标附件；
另存notification_relay.json，不修改集群原notification.json，不重复已被SMTP接受的发送。

### 集群全量完成与计时结果（2026-10-15验收）

作业3673359于北京时间2026-10-14 23:44:45 COMPLETED，退出0:0；启动器与worker已随Slurm分配退出，锁释放。
最终HCR mIoU **86.8217224092%**，伪标签mIoU **86.2884465695%**，MLP完整50轮×100步，最佳epoch **38**。
与原服务器中心0.2的86.8217170759%相差+0.0000053333个百分点，数值基本一致；没有声称两环境逐像素或矩阵完全相同。
本次是seed42/VOC val的从头复跑，不是多seed或独立测试集验证，不更改主线。

| 时间口径 | 实际耗时 |
|---|---|
| 排队 | 1秒 |
| Slurm分配占用（Start→End） | 5小时32分15秒 |
| 数据预检 | 5分11.32秒 |
| 双卡真实探针全过程 | 14分21.92秒 |
| 正式进程（包括初始化/通知） | 5小时12分7.27秒 |
| 正式流水线summary.seconds | 5小时4分16.19秒 |

| 阶段 | 最长worker耗时（双卡并行取最大） |
|---|---|
| 伪标签生成 | 1小时1分14.81秒 |
| 伪标签几何清理 | 2分30.51秒 |
| 伪标签质量评测 | 1分5.03秒 |
| raw提议生成 | 22分40.01秒 |
| 提议清理 | 1分3.05秒 |
| GMP/整图特征提取 | 1小时7分43.48秒 |
| 统一合并与一次前景过滤 | 4分20.91秒 |
| 50轮MLP训练与单库导出 | 14分16.17秒 |
| 1449张严格HCR | 11分57.34秒 |

阶段worker时间不包含启动器输入/产物SHA、检查与调度，不能将并行worker时间相加当作总耗时。
worker外开销约1小时57分24.87秒。先前build日志停止更新时，worker已经完成，启动器仍在校验大缓存及逐图产物。

所有10582张raw/clean伪标签、1449张raw/clean提议齐全；train 10582份与val 1449份整图FP32特征已保存，72×72×1280。
原始库37571区域，经一次前景10%过滤保留34880，背景10569不丢弃；MLP库保持同一成员与sample_id。
两分片1449评测ID无重复/遗漏，总混淆矩阵为两分片总矩阵逐元素之和。该生产入口没有保存逐图混淆矩阵，不补造历史记录。
全流程最高allocated显存5249.83MiB；运行结束组剩余配额约370.79GiB，高于5GiB底线。

原始summary/metrics/per_class及通知结果保留；独立验收位于
`experiments/hpc_launch_audits/cluster_mainline_baseline_20261014_175149/completion_acceptance_20261015.json`。
整图token创建和阶段提交均有全内容SHA；事后核对所有阶段SHA凭据和当前文件尺寸/覆盖，关键代码、权重、库、指标重新核验SHA，固定大缓存样本核验内容/绑定/真实数组头；不重复读取全部大缓存。
21类IoU、完整计时与小文件SHA核对保存在本地
`experiments/cluster_mainline_baseline_20261014_175149/completion_verified_20261015_0122/`。

集群原notification.json为SMTP未接受（三次尝试因无密码跳过）；不等于Slurm END邮件已投递。
按既有授权只将summary.json、metrics.json、per_class.csv复制至原服务器新目录
`experiments/hpc_results/cluster_mainline_baseline_20261014_175149`并校验SHA，使用既有私有SMTP配置转发，
独立notification_relay.json记录状态、防重复。SMTP接受仅表示服务器接收，不证明收件箱投递。
完成验收与一次结果通知后关闭本节automation，不追加实验，不干预原服务器实验。

独立验收：289份冻结与当前代码SHA通过，92,189份产物凭据/当前尺寸通过；253份关键产物和6份大缓存样本重新核验内容SHA。
原服务器notification_relay.json记录smtp_accepted=true、attempts=1，收件人2858372757@qq.com；未确认收件箱投递。

## 57. 固定反馈模型的HCR开关配对回放（2026-10-14）

用户授权按第55节下降诊断继续原服务器实验；先隔离HCR贡献，不重新训练或调整阈值。
独立入口`tools/experimental_retrieval_hcr_ablation.py`，测试
`tests/test_retrieval_hcr_ablation.py`；不修改src/configs、旧入口、模型或旧指标。
版本`retrieval_hcr_ablation_20261014_194946`，来源
`retrieval_reconstruction_20261014_164724`的control与retrieval最佳MLP及对应34,880成员单库。
沿用seed42、原clean/raw提议、原1280维GMP查询、中心权重0.2和其余推理参数。

开启HCR的1,449图结果直接复用经SHA及逐图矩阵校验的第55节结果。两卡各做一图开启HCR
精确回放探针，分别与本模型历史逐图矩阵逐元素比较；再做各一图关闭HCR探针。
关闭HCR只取clean提议，按原mask哈希从已签名的clean+extra查询缓存选择相同向量，
验证原缓存签名、SHA、clean几何及顺序；缺失或重复不明确的哈希拒绝，不重新跑SAM3/GMP。
正式GPU0评测control关闭HCR，GPU1评测retrieval关闭HCR，每组完整1,449图严格评测。
两组模型、库与提议冻结，唯一操作变量为HCR开关；不从VOC val选epoch或阈值。

参数、不可变源码、阶段attempt、指标及逐图/总矩阵在`experiments/<版本>`，
日志在`logs/<版本>`；仅复用原模型/缓存路径，不复制或新建训练库及模型。
支持`--preflight-only`、`--probe-only`、`--resume`；续跑必须代码、输入和参数完全一致，
只复用SHA通过的本版本阶段。每卡至多一worker，保留5GiB，SIGTERM只停止自有worker。
结束统一邮件通知，附件summary.json、metrics.json、per_class.csv，SMTP失败有限重试并独立记录。

```bash
python -B -m unittest tests.test_retrieval_hcr_ablation -v
python -B tools/experimental_retrieval_hcr_ablation.py --version NEW_VERSION --gpus 0 1 --preflight-only
python -u -B tools/experimental_retrieval_hcr_ablation.py --version NEW_VERSION --gpus 0 1 --probe-only
# probe ready后，同版本同参数--resume进行完整关闭HCR回放。
```

报告四组mIoU和21类IoU、背景误报/前景漏检/类间混淆像素、带/不带HCR时反馈与control的差，
以及HCR分别对两模型的净增益和二者差值。该差值是固定模型HCR开关效应，不解释为整个方向
已有效或无效；不自动部署，也不自动追加损失权重、校准或多seed实验。

本轮后台启动器PID247974，于北京时间2026-10-14 19:56:27脱离SSH运行，19:58:37完成；
入口计时112.717秒，另有启动前预检。4项CPU测试、19,419项签名输入及全部1,449图clean查询
子集检查通过。两模型开启HCR的一图探针精确复现历史，关闭HCR的一图探针与正式对应逐图
矩阵一致。两组关闭HCR均完整1,449图；3阶段48项产物SHA验收通过，启动器和worker退出、锁释放。

| 固定seed42模型 | 开启HCR mIoU | 关闭HCR mIoU | HCR净收益（百分点） |
|---|---:|---:|---:|
| control原模型 | 86.821717% | 85.841489% | +0.980228 |
| retrieval反馈模型 | 86.253598% | 85.719246% | +0.534352 |
| retrieval-control | -0.568119点 | -0.122243点 | HCR收益差-0.445876点 |

按本轮固定模型开关对照，0.445876点对应HCR收益差，约占净下降0.568119点的78.5%；
其余0.122243点在关闭HCR后仍存在。不是将错误候选全部归因于HCR训练，也不是反馈方向被否定。
关闭HCR时，两模型背景误报均2,132,653像素、前景漏检均5,020,719像素；
前景类间混淆由452,394增至487,908，说明仍有分类退步。鸟、船、瓶子的两模型IoU在关闭
HCR时相同，而椅子仍低3.524333点。HCR对反馈模型仍有正收益，不将关闭HCR设为主线。
开启/关闭的原JSON精度略有差别，显示值均保留足够有效位；不覆盖历史指标或从val选门槛。

原始结果与`completion_acceptance.json`在服务器本版本experiments目录；本地验收副本在
同版本`completion_verified_20261014/`，下载SHA与验收记录核对。
邮件首次SMTP接受成功，未确认收件箱投递。仅完成本轮开关对照，不追加训练或部署。

## 58. 检索回填CE弱权重0.1单因素实验（2026-10-14）

用户授权继续尝试第55节反馈训练，只将retrieval CE权重由1.0改为固定0.1。
独立入口`tools/experimental_retrieval_weak_feedback.py`，专用测试
`tests/test_retrieval_weak_feedback.py`；复用现有`tools/train_retrieval_reconstruction.py`，
不修改旧训练器、src/configs或旧实验。版本`retrieval_weak01_20261014_202026`，在原服务器运行。

从原1280维GMP原库重新初始化seed42 MLP，沿用两层1280→512→256、50轮×100步、
8类×4区域、v1 CE＋0.5 SupCon、AdamW lr0.001、余弦学习率、裁剪5和原内部图片划分。
温度0.1、图库每10步刷新、同图近邻与类中心排除、图库停止梯度、背景反馈及内部伪标签5-NN选模
均不变；损失为CE＋0.5 SupCon＋0.1 retrieval CE，逐轮验证日志损失分项一致。
不新增warmup、teacher约束、margin损失、校准或阈值搜索。

复用第55节原始库、伪标签、提议、查询缓存、control和权重1.0的MLP/导出库；
复用第57节关闭HCR的历史基准/权重1.0矩阵。GPU1训练新0.1模型，GPU0同时分别回放原control
开启和关闭HCR的全部1,449图，要求总矩阵及逐图矩阵与历史逐元素一致，不重新训练control。
新模型完成后，GPU0开启HCR、GPU1关闭HCR各评测全部1,449图，推理参数仍为中心0.2及原HCR。
源库34,880成员与sample_id保持一致；不重新前景过滤或跑SAM3/GMP，不读取训练像素GT。

入口复用第57节的签名查询读取协议，manifest.protocol保留该评测schema，
manifest.experiment_protocol和最终summary.protocol为retrieval_weak_feedback_v1；
独立签名绑定本版本代码、权重0.1和成员，不修改或绕过旧来源签名。
preflight/probe/resume、不可变快照、阶段新attempt、5GiB底线、自有worker SIGTERM清理沿用现有框架。
完整epoch检查点保存模型、优化器、随机状态及binding；同版本恢复仅接受相同代码/参数/输入。

权重在`weights/sam3_adapter/<版本>/weak01/<attempt>`，导出库在
`storage/mlp_banks/voc/<版本>/weak01/<attempt>`，参数/矩阵/指标在experiments、日志在logs。
开始正式训练前，GPU1做真实两步训练/导出/梯度与生产分数等价探针，GPU0回放baseline一图开/关HCR；
再两卡分别验证新探针模型的一图开/关HCR。探针不用于判断正式收益。
完成邮件附summary.json、metrics.json、per_class.csv，有限重试，不改变实验结果。

```bash
python -B -m unittest tests.test_retrieval_weak_feedback tests.test_retrieval_hcr_ablation tests.test_retrieval_reconstruction -v
python -B tools/experimental_retrieval_weak_feedback.py --version NEW_VERSION --gpus 0 1 --preflight-only
python -u -B tools/experimental_retrieval_weak_feedback.py --version NEW_VERSION --gpus 0 1
# 正式入口内部先执行所有真实probe；--probe-only可单独止于ready，--resume仅复用已校验本版本阶段。
```

报告control/0.1/历史1.0的开/关HCR共六组mIoU和21类IoU、错误像素、HCR净收益、最佳epoch与loss分项。
统一由混淆矩阵重算float64 IoU，保留原阶段和历史指标。只作固定seed42开发对照；
不自动部署、增加seed或继续尝试其它权重。真实状态以本版本status/summary和worker日志为准。

本轮12项CPU测试通过，19,439项固定输入签名通过。北京时间2026-10-14 20:29:58由启动器
PID248420脱离SSH后台运行；两步真实训练梯度非零、生产分数最大差2.38419e-7，
34,880成员导出与一图开/关HCR探针通过，共62项probe产物SHA通过。
20:31:55进入正式训练与基准回放，GPU1训练worker248572，GPU0基准worker248573。
20:34:39验收时第4轮完整检查点已通过SHA及实际0.1权重损失验证，正式日志持续推进。
启动和验收凭据在`experiments/preflight_audits/<版本>`的launch.json、startup_validation.json；
跟踪任务`0-1`每15分钟核对，正常运行静默，结束核验和通知后删除；独立于集群跟踪任务。

本轮于北京时间2026-10-14 20:54:50完成，入口计时1473.318秒（24分33秒，不含此前CPU预检）。
20:58验收：19,439项输入与源码快照SHA通过，4阶段222项产物SHA通过；
50轮×100步、真实0.1权重损失、34,880成员/sample_id和原内部图片划分均通过；
基准开/关HCR总矩阵及逐图矩阵精确复现，新模型开/关HCR均覆盖1,449图且逐图之和等于总矩阵。
启动器和worker已退出、锁释放。验收文件为本版本`completion_acceptance.json`，原指标不覆盖。

| 固定seed42模型 | 开启HCR mIoU | 关闭HCR mIoU | HCR净收益（百分点） | 最佳epoch |
|---|---:|---:|---:|---:|
| control原模型 | 86.821717% | 85.841489% | +0.980228 | 38 |
| retrieval权重1.0（复用） | 86.253598% | 85.719248% | +0.534350 | 32 |
| weak01权重0.1 | 86.670578% | 85.766902% | +0.903675 | 43 |

0.1相对原模型：开启HCR -0.151139点，关闭HCR -0.074587点；
相对1.0模型：开启HCR +0.416980点，关闭HCR +0.047655点。
HCR收益差从1.0版本约-0.445878点缩小至-0.076553点，但主路径分类仍退步，不推广。
上述旧off末位与第57节原JSON略不同，是本轮统一float64从同一混淆矩阵重算，未更改原指标。

开启HCR时0.1相比control：背景误报+41,004像素、前景漏检-23,095、前景类间混淆+21,791。
主要逐类变化为chair -2.367009点、ship（VOC boat）-1.585266点、cow -0.498863点、
dog -0.368485点；car +1.481379点、sofa +0.676942点。bird恢复为与基准相同，
bottle差距缩小至-0.011635点。内部伪标签5-NN最佳宏准确率98.345438%；
回填CE首轮0.855565→末轮0.036906，训练批检索准确率84.25%→99.46875%。
这些仅是seed42开发对照，不能作为多seed提升或具体噪声机制的证明。

服务器本版本summary.json、metrics.json、per_class.csv和完成凭据已同步本地
`experiments/<版本>/completion_verified_20261014/`，下载文件SHA与完成凭据核对。
邮件首次SMTP接受成功，未确认收件箱投递；本轮结束，删除`0-1`跟踪，不追加训练或部署。

## 59. 整图类别存在性联合判断：先做CPU oracle门槛（2026-10-14）

用户授权独立实验入口 `tools/experimental_image_presence.py`，测试
`tests/test_experimental_image_presence.py`。本次先实现并运行阶段0；不修改src/configs、主线入口、
原标签、提议、模型和指标。若oracle不足固定+0.3个百分点，状态为`complete_no_headroom`，
不池化大token缓存、不训练整图分类器，不启动V1–V4。通过时记`oracle_passed`，再独立启动分类器阶段，
保存阶段0不可变快照与结果，不能将oracle成绩计入方法结果。

本次按用户方案的历史中心0.4基准86.7350211582%执行，区别于当前主线中心0.2的86.8217170759%。
同一原MLP和导出库、同一clean/raw提议；读取第44节整图FP32缓存的全部12,031个SHA凭据与签名。
oracle阶段不消费整图token数据；数据SHA须在后续池化消费前逐文件核验。原生区域查询读取既有缓存，
CPU执行原MLP与src/retrieval.py balanced分类器K25/top5/搜索2/中心0.4，不运行SAM3，不训练MLP。

先落盘全部候选分数及无GT的完整1,449张基准预测并记录SHA，再读取验证像素GT，
完美存在集合只作用于诊断：clean预测类别不在GT图像前景集合时判背景；raw/HCR附加候选不变。
重新调用生产同类NMS、原HCR和rasterize，保留原融合排序分数，不能在最终PNG上简单抹类代替候选重放。
总混淆矩阵必须精确等于原版本历史；逐图必须等于第50节已获授权且两次一致的历史回放记录。
原始实验未存逐图矩阵，后来的逐图回放不冒充原始记录。

背景诊断沿用有效GT多数背景且纯度≥0.9，分clean/raw_extra，分别统计分类前、margin后、NMS后、
最终可见像素阶段的数量、原预测背景/前景数量、并集覆盖和重复计数面积；预测背景的候选保留在
分类前统计中。raw_extra尚未参与clean NMS，所以其NMS阶段不应解释为自身已通过HCR。
oracle与基准保存1,449张逐图矩阵、总矩阵、21类IoU、像素错误、按类拒识数及2000次seed42配对图片bootstrap。
GT只用于最终评测及oracle诊断；不进入训练、存在性标定或方法成绩。

所有实验按新版本隔离，代码快照、源码/输入/完成产物SHA、失败新attempt、相同签名resume、
5GiB磁盘底线、自有worker的SIGTERM退出和结束邮件沿用完整入口。CPU worker不占GPU；
预检核实两张物理卡空闲。后台日志写logs根目录新文件，不提前建立logs/<版本>。
方法结果表始终不包含oracle。本节结果由真实运行验收后追加。

```bash
python -B -m unittest tests.test_experimental_image_presence -v
python -B tools/experimental_image_presence.py --version NEW_ORACLE_VERSION --preflight-only
python -u -B tools/experimental_image_presence.py --version NEW_ORACLE_VERSION --probe-only
# probe ready后使用相同版本、相同代码和参数，追加--resume完成阶段0；通过后再进入分类器阶段。
```

第59节阶段0完成（版本`image_presence_oracle_20261014_212232`）：
1,449张CPU严格回放，总矩阵等于86.7350211582%历史结果，逐图等于既有双重回放记录。
oracle仅对clean拒识并重新执行原NMS/HCR，得到88.2362174361%，诊断提升+1.5011962779个百分点，
2000次配对图片bootstrap 95%区间[+1.006299,+2.028304]，超过固定+0.3门槛。
这不是方法成绩，也不是训练seed置信区间。门槛通过，允许继续原方案的分类器阶段。
背景口径：纯度≥0.9的clean背景候选在分类前已为201个、原预测背景0个；
margin后201、NMS后160、最终可见160，不存在先排除预测背景导致201/201错的统计选择。
raw额外池中有138个正确预测背景，背景候选总数2551；它们与clean群体分别报告。
阶段0约4分28秒探针（包含全部非GT预测）＋2分56秒正式oracle续跑，预检耗时另计。
源阶段结果和代码保留，分类器采用新的版本，不能用修改后的入口续接阶段0旧代码。

阶段1/2现由同一独立入口提供`--phase presence --oracle-source ORACLE_VERSION`。
全图均值保留带符号token；GeM分支先ReLU再求p=3幂均值，拼接后L2，固定2560维。
池化缓存放storage/image_presence/<版本>/pooled/<train或val>/<图片>/attempt_NNN，并逐图绑定
原token SHA、原receipt SHA、编码签名和池化规则；每份token消费前核验实际内容SHA。
完整训练固定内部train_images/validation_images，不读训练像素GT、不读VOC val类别；
20前景多标签2560→512 GELU→20，seed42/BCE/AdamW lr0.001/weight_decay0.01/batch256，
完整30轮，使用最后一轮，不在VOC val选epoch。每类阈值取内部留出正例召回≥0.98的最高阈值；
正例少于10个时t=0永不拒。训练JSON标签为VOC语义ID 0..20，0仅背景，20前景对应输出位置0..19。
权重与30份epoch检查点放weights/image_presence/<版本>/classifier/attempt_NNN。

V0–V4完全按用户固定规则；V3仅clean改判到第二名的前景p≥0.9且原margin≤0.05，
所有改判保留原融合排序分数。V4支持集合为原始clean和raw额外池的原分类结果，
必须有另一候选达到retrieval≥0.95/margin≥0.1，不能自我支持、不随改判迭代。
所有修改后仍调用生产NMS/HCR/rasterize。V0完整像素预测与已提交基准逐图比较，
再核验历史总/逐图矩阵。五组全部预测和SHA凭据提交后才读取GT评测及候选诊断，
oracle始终单列diagnostic，不进入method_results或方法CSV。
拒识事后诊断区分背景正确拒识与拒掉真实前景，另记原检索类别是否正确；
类别改判记录纠正/误伤/wrong_to_wrong。每组都有1449逐图矩阵、总矩阵、21类IoU、
三类错误像素、按类操作数和2000次配对bootstrap，不据VOC val重调参数或自动部署。

```bash
python -B -m unittest tests.test_experimental_image_presence -v
python -B tools/experimental_image_presence.py --version NEW_PRESENCE_VERSION --phase presence --oracle-source image_presence_oracle_20261014_212232 --preflight-only
python -u -B tools/experimental_image_presence.py --version NEW_PRESENCE_VERSION --phase presence --oracle-source image_presence_oracle_20261014_212232 --probe-only
# probe ready后，完全相同代码和参数，移除--probe-only并加--resume进行后台全量。
```

第59节分类器阶段预检与探针（新版本`image_presence_joint_20261014_214827`）：
服务器13项CPU测试全通过；57,109项固定输入与12,031个第44节token凭据预检通过。
真实18图探针（内部train8＋留出8＋val2）完成原token实际SHA核对、池化落盘/读回、
GPU1两轮短BCE训练、内部阈值标定与V0–V4两图严格评测；t全0时五组总矩阵逐元素相同。
GPU1探针峰值分配显存90.366MiB。正式与探针代码digest
`f8987be53ac47401638aa6e1f0cd307cf881270adc2bb9cab20c6ff64e4b1e197`。
完成探针后已后台提交全量，启动器PID249878，日志
`logs/image_presence_joint_20261014_214827_launcher.log`；GPU1仅在整图分类器训练阶段使用，
池化和缓存候选评测使用CPU。不干预集群的主线基线任务。
正式启动验收及最终结果另行追加；probe ready不能当作本次已完成。

第59节正式启动验收：启动器249878与CPU池化worker249896/249897真实存活，
训练/验证worker均已输出100张进度；验收时train159、val153，共312份缓存，
抽查6份完整凭据/数组SHA、2560维FP32形状与L2归一化全部通过，/data剩余约1737.91GiB。
启动凭据在experiments/preflight_audits/image_presence_joint_20261014_214827/；
全部计算从新不可变快照运行，原服务器现有数据和原模型只读。自动完成验收`automation-2`
仅在完成、失败或需要用户行动时通知，正常运行静默，完成验收后删除；不干预集群监控。
QQ邮件由实验启动器统一发送，通知成功须以notification.json的SMTP接受记录为准。
正式方法指标目前尚未产生；阶段0结果已下载到本地对应版本completion_verified_20261014并逐文件SHA核对。

第59节完成结果（2026-10-14，本地验收副本`completion_verified_20261014_2318`）：
服务器状态complete且锁释放，V0–V4均覆盖1,449图。V0/V1/V2/V3/V4的mIoU分别为
86.735021%/86.358534%/86.407967%/86.407967%/86.519348%，均使用该节历史中心权重0.4。
V4相对V0下降0.215673个百分点，高纯度改动候选中正确拒识背景270个、误拒真实前景178个。
V3没有类别改判，等于V2。保持原主线；完美存在性oracle88.236217%单列诊断，不能作为方法成绩。

## 60. 类别条件区域语义验证（2026-10-15）

用户授权新增模块并运行独立实验。模块为`src/region_semantic_verifier.py`，独立入口
`tools/experimental_region_verifier.py`，专用测试`tests/test_region_semantic_verifier.py`及
`tests/test_experimental_region_verifier.py`。
原MLP、检索库、标签、提议、主线入口和历史实验保持原身份；新版本按以下固定协议运行。

第一阶段只用CPU重算当前中心权重0.2的21类检索分数，提交全部1,449张无GT基准预测后，
才读取验证像素GT。基准须与既有中心0.2回放的总/逐图混淆矩阵精确一致（mIoU86.821717%）。
背景oracle仅撤销有效非255像素中背景占比至少0.9的clean候选的前景指派，保留原检索/融合分数，
重新执行原NMS、HCR及像素合成。raw附加池不改；诊断收益不足固定0.3个百分点时结束为
`complete_no_headroom`。该oracle不同于第59节整图存在性oracle，只作为启动门槛和诊断。

通过门槛后默认使用物理GPU1，读取第44节整图FP32 token缓存。区域1280维原GMP向量只读复用；
上下文用固定外环的带符号token均值（外环宽度为0.1×区域面积平方根，限定3–32原图像素），
不执行SAM3，不修改原GMP。空外环或无token支持时弃权保留原类别，不使用最近token伪造上下文。
区域和外环分别L2归一化，与21类检索分数、6维固定几何特征拼接，默认输入2587维。
几何定义和逐区域mask/token身份保存在新实验manifest和特征凭据中。

沿用原MLP内部8,466训练图/2,116留出图，验证器拟合与负例挖掘只使用训练图。
冻结原adapter，训练图按图片hash分两折，每折查询另一训练折图库；留出图查询全部训练图库。
所有图库均排除内部留出图，查询和图库图片不得相交；不复用包含留出图训练的旧折adapter。
伪类别c仅提供一个带噪正例；图像级JSON明确缺失的类别提供可靠“非该类别”负例。
图内其它已存在类别是未知，忽略其损失；“非c”包含其它前景，不能统一当成背景负例。
仅在拟合图中，按每类交叉检索分数选择最高25%的可靠负例，至少128个、不超过可用数量；
相同分数按唯一区域key排序，保留全部伪正例。保存每类选中key、分数下限及数量，留出图不参与挖掘。
不读取训练像素GT，不使用VOC val图像类别或像素GT进行拟合、挖掘、选模或标定。

固定2587→256 GELU→20的二层验证头，seed42、30轮、batch256、AdamW lr0.001/weight_decay0.01，
每类正负样本等质量的masked BCE，使用最后一轮。只在内部留出伪正例上取保留率≥0.995的最高阈值；
该类留出正例少于20个或训练正/负例不足则阈值0，永不拒识。这个经验保留率不保证真实前景召回。
固定三组：V0原基准，V1仅对clean撤销不支持的类别指派，V2对clean及raw附加池都应用验证。
不主动改成其它前景类别；撤销后重跑原候选NMS/HCR/合成，不在最终PNG上直接抹类。
所有方法预测和SHA提交后才读取GT评测，oracle不进入方法成绩。

新特征/预测放`storage/region_verifier/<版本>`，新权重放`weights/region_verifier/<版本>`，
配置/指标/诊断放`experiments/<版本>`，启动日志放`logs/<版本>_launcher.log`。
每次消费token前核验实际NPZ SHA、receipt和编码签名；不可变源码快照、阶段attempt、输入/产物SHA
和同签名resume沿用完整入口，保留5GiB磁盘余量，只停止自己worker。
先做真实小样本token读取、GPU1短训练和候选重放探针；probe分数不作为正式精度。
完整报告1,449张逐图/总矩阵、21类IoU、三类错误像素、背景正确拒识和真实前景误拒，
以及2000次seed42配对图片bootstrap。当前只做seed42开发对照，不自动部署或追加val阈值扫描。

```bash
python -B -m unittest tests.test_region_semantic_verifier tests.test_experimental_region_verifier -v
python -B tools/experimental_region_verifier.py --version NEW_REGION_VERIFIER_VERSION --preflight-only
python -u -B tools/experimental_region_verifier.py --version NEW_REGION_VERIFIER_VERSION --probe-only
# ready后，同版本、代码、参数和输入去掉--probe-only，加--resume执行完整实验。
python -u -B tools/experimental_region_verifier.py --version NEW_REGION_VERIFIER_VERSION --resume
```

以上为已授权固定实验协议；实现验收、服务器预检、真实探针和正式运行状态另行追加，
不能由本节说明推断已启动或已经得到新方法精度。

第60节服务器实施（版本`region_verifier_20261015_014227`）：本地44项检查34通过、
10项因缺少项目库跳过；原服务器44项全部通过。新模块、入口、两份测试和手册初次部署SHA一致。
预检通过12,031份token凭据和34,880个原库成员，原内部8,466/2,116图划分不变；
源码快照digest为`1e6838b169f271150cfe8ffd94133a88838eb67eccfb00078919ddc2c60406e7`。

阶段0完成1,449图严格回放，原总/逐图矩阵精确一致，中心0.2基准86.821717%：
高背景占比clean候选拒识201个，oracle88.152700%，诊断增益1.330985个百分点，
2000次配对图片bootstrap的95%百分位区间为[+0.855678,+1.855015]，通过固定0.3门槛。
这不是方法成绩或训练seed置信区间，不与第59节中心0.4的存在性oracle混作同一个诊断。

真实GPU1探针完成6训练图/4内部留出图/2验证图、两轮短训练、外环特征及V0–V2重放。
训练区域复用GMP最大绝对误差2.9802322387695312e-08，验证区域误差0；
探针永不拒识阈值下三组逐图矩阵完全相同，阶段产物SHA全部核验，probe ready且锁释放。
同版本同源码/输入/参数已后台提交正式30轮与全量三组评测，启动器PID251758，
日志`logs/region_verifier_20261015_014227_launcher.log`。完整启动验收与方法最终结果另行追加；
probe ready与后台提交不代表正式实验完成。审计文件在`experiments/preflight_audits/<版本>`。

第60节完整启动验收：启动器251758、GPU1特征worker251815真实存活，完整阶段为
`feature_extraction`；验收时已有355份训练特征检查点。抽查6份数组SHA、2587维FP32形状、
有限值及区域分支L2归一化通过，294项不可变源码快照SHA匹配，/data剩余约1737.31GiB。
验收在`experiments/preflight_audits/region_verifier_20261015_014227/startup_validation.json`；
正式验证头尚未完成训练，当前没有新方法mIoU。完整任务已脱离SSH，原模型、库、提议和标签只读。

第60节完成结果（2026-10-15，`region_verifier_20261015_014227`）：服务器complete，
启动器已退出、锁释放。固定30轮训练，V0/V1/V2各完整评测1,449张图；正式流水线耗时
4783.307秒（79分43秒，不含之前独立探针）。

| seed42方案 | mIoU (%) | 相对V0（百分点） |
|---|---:|---:|
| V0原基准 | 86.821717 | 0 |
| V1仅验证clean候选 | 86.521443 | -0.300274 |
| V2验证clean及raw附加池 | 86.521443 | -0.300274 |

V1撤销13个clean候选，包含3个高纯背景、10个高纯前景（其中9个原类别正确）。
相比基准，背景误报像素-842、前景漏检+348634、前景类间混淆-46953；仅13张图的矩阵改变。
V2额外拒识527个raw候选，其中469个有有效GT、58个无有效GT；两组全部1,449张最终PNG的SHA一致，
附加筛选没有进一步改变最终预测。候选拒识计数不等于独立像素收益。
羊/人IoU分别下降5.262929/1.439552点，餐桌提高0.748744点；其它主要类别不变。
配对图片bootstrap差值95%百分位区间[-0.841048,+0.032539]点，跨0；
仅反映当前seed的开发图像抽样，不是训练seed方差或独立测试结论。

31,401项完成产物及294项源码快照SHA核验，30轮、三组逐图矩阵之和和mIoU复算、
V0与旧中心0.2基准逐图精确一致均通过；原始指标不覆盖。
本地验收副本在`experiments/<版本>/completion_verified_20261015`，含summary/metrics/per_class、
三组逐图矩阵、`completion_acceptance.json`、`local_validation.json`及`results_report.md`。
当前单seed实测没有收益，原主线保持；不追加训练、阈值扫描或部署。88.152700%的oracle仍仅为诊断。

## 61. 当前主线完整Oracle与模块误差诊断（2026-10-15）

用户授权以当前主线重新运行真值辅助诊断。独立入口`tools/diagnose_mainline_oracles.py`，
纯诊断核心`tools/mainline_oracle_core.py`，专用测试`tests/test_mainline_oracles.py`。
固定第46节原模型/单库/clean及raw提议、中心0.2、seed42，基准86.821717%；
第60节新验证器仅作为已完成缓存来源，不启用验证头或其拒识结果。

完整核验已选主线六项资源路径、来源快照、输入SHA和编码签名后，CPU重新执行原MLP与
balanced检索并提交全部1,449张无GT基准预测及SHA，再读取VOC val像素GT做诊断。
总矩阵和逐图矩阵必须精确等于现有0.2基准。只读缓存，不运行SAM3、不重训或改变主线；
CUDA_VISIBLE_DEVICES固定空，CPU线程4，读取GPU占用不要求GPU空闲，不干预GPU0其它任务。

像素干预分别修复背景误报、前景漏检、前景类间混淆、边界邻近带或内部错误；
它们表示对应错误的乐观空间，不是可部署方法。边界使用直接类别变化及255忽略带附近有效像素，
四邻域半径3，报告为boundary_proxy，不能称标准Boundary IoU。
候选干预包括固定最终掩码/排序的前景类别纠正和全部多数类替换，clean背景90%拒识复现、
背景多数类拒识、clean前景类别替换、clean或全扩展池多数类替换后重跑原NMS/HCR。
所有类别替换保留原检索分数、margin与融合排序；全void候选在生产重放中保留原类别。
单独关闭clean分类NMS时仍保留面积300、原HCR；不添加缓存推理原本没有的实例分数筛选。

固定预测类别的完美像素归属分别在最终裁剪后候选、HCR裁剪前已接受候选、全部clean和
全扩展池比较。raw多数类赋值和像素仲裁另列；raw、clean、扩展池的几何覆盖cap允许每像素真值
赋类及背景拒识，只是乐观几何诊断。多数类替换可能降低mIoU，任何组差值不可直接相加，
不把这些数值称为统一的可达或数学最优mIoU上限，也不假设低于既有raw生成阈值的新候选存在。

针对基准错误的GT前景像素，使用真正嵌套的正确类别支持集合分层：扩展池几何→正确检索类别→
margin→clean面积→clean NMS→HCR基本门槛→HCR占据/新颖比例/裁剪→像素竞争。
各层互斥计数之和必须等于前景漏检加前景类间混淆。背景误报按最终像素所属clean/HCR拆分。
raw/clean几何可能因清理不嵌套，单列几何覆盖与描述性clean链，不能用它冒充严格因果。
语义GT采用4连通域，报告各区域最佳候选IoU、R50/R75、面积≥1/100/300/1000分层及逐类统计；
语义连通域不是实例GT。候选纯度、类别混淆及top2范围与最终mIoU分开报告。

新版本`current_mainline_oracles_20261015_122200`：配置/矩阵/诊断在experiments，
新基准PNG在`storage/oracle_diagnostics/<版本>/baseline`，日志在`logs/<版本>`。
不可变快照、严格同输入/代码resume、阶段attempt及产物SHA沿用完整入口；保留5GiB磁盘余量，
只终止自有worker。先完成2图真实CPU诊断探针，再同版本恢复1,449图完整诊断。
每组保存逐图/总矩阵、21类IoU和2000次图片配对bootstrap；这仍是单训练seed的VOC开发诊断。

```bash
python -B -m unittest tests.test_mainline_oracles -v
python -B tools/diagnose_mainline_oracles.py --version current_mainline_oracles_20261015_122200 --preflight-only
python -u -B tools/diagnose_mainline_oracles.py --version current_mainline_oracles_20261015_122200 --probe-only
python -u -B tools/diagnose_mainline_oracles.py --version current_mainline_oracles_20261015_122200 --resume
```

实际预检、探针、完整结果与模块优先级在真实验收后追加。所有GT只用于诊断，不进入模型、库或训练。

第61节完成验收（2026-10-15，`current_mainline_oracles_20261015_122200`）：
服务器36项测试通过，55,653项输入绑定预检通过；CPU重新检索8,793个查询，库成员34,880。
所有1,449张新基准PNG在读取GT前逐像素匹配原主线，并提交完整SHA；完整总/逐图矩阵精确重放。
25组全图对照及2组子集指标完成；4,450项登记产物、295份新快照和294份来源快照验收通过。
正式恢复入口613.11秒，其中全量Oracle worker542.86秒；此前检索/基准/2图探针另281.51秒。
本地30项下载产物SHA核对、27组矩阵mIoU重算通过；全部worker无可见GPU，不干预GPU0其它任务。

主要mIoU（均为百分比）：基准86.821717；理想补全前景漏检93.770285；完美清GT背景90.965750；
理想修复前景类间混淆88.167612；固定最终候选前景改类88.001957；clean背景90%拒识88.152702；
最终候选原类别像素仲裁91.013086；完整扩展池原类别像素仲裁93.329761；
clean/完整池/raw几何cap分别91.259440/95.265376/95.297114；关闭clean分类后NMS86.821954。
背景90%拒识逐图矩阵精确重放第60节oracle，增量1.330985个百分点，图片配对95%区间
[0.855678,1.855015]；所有组2000次图片配对bootstrap，只针对单训练seed的VOC开发集。

全部错误7,257,917像素：FN4,359,445，背景FP2,411,611，前景错类486,861。
嵌套前景错误分母4,846,306：完整池无几何2,909,117(60.03%)，无正确预测类别666,260(13.75%)，
margin0，clean面积10,211，clean NMS127，HCR基础门槛1,235,079(25.49%)，
HCR占据/新颖/裁剪5,391，像素竞争20,121；各项精确加总。
背景FP归属clean2,132,653(88.43%)、HCR278,958(11.57%)。
面积≥300的3,905个GT语义连通区域，clean/完整池/raw R50分别69.577%/75.314%/75.288%。

模块优先级：候选完整性和缺失区域补全；掩码内部前景/背景结构细化；HCR候选质量和类别可靠性准入；
再针对chair等混淆类改进语义。table当前47.729%，raw几何cap58.428%，直接补FN可到93.018%；
bicycle当前55.153%，完美清GT背景95.825%，区域背景90%拒识仍55.153%。
不优先改NMS/排序：关闭NMS仅+0.000237点，最终候选像素仲裁除清背景外只多0.047336点。
extra噪声很大，不能由HCR门槛归因直接推导降低0.95阈值有收益；本次不启动新训练、扫描或部署。

完整分析`experiments/<版本>/results_report.md`，定义`case_definitions.md`；
服务器原始矩阵在`full_oracles/attempt_001/worker_0`，基准PNG在`storage/oracle_diagnostics/<版本>/baseline`；
本地证据在`experiments/<版本>/completion_verified_20261015`，含全部27组NPZ、逐图诊断、汇总、逐类表及双端验收。
这些GT辅助空间不可当作方法精度、收益不可相加；几何cap允许逐像素拆mask，众数类替换不是最优mIoU保证。

## 62. Oracle后续改进实验设计（2026-10-15，仅设计，未启动）

用户请求设计改进实验。具体计划与固定参数保存在
`experiments/plans/oracle_guided_improvements_20261015_v1/plan.md`和`protocol.json`，
状态`designed_not_started`；这些不是已有可运行的新CLI或实际实验成绩。

首批6组为B原基准、R1全20类原图补充提议、R2固定上下文的餐桌文本别名补充、
P1原FP32空间token线性像素头、P_uniform同头均匀传播、P2同头RGB传播。
R3中心80%crop是预登记第二批；Active当前无crop实现。R保持原MLP/单库/GMP/中心0.2，
新增mask从原图token重聚合，原输出后只补背景；P用训练Active伪标签，不读密集GT，
只在最后把有充分背景证据的前景像素改BG，不让HCR填回，像素阈值按原图分辨率标定。

H先拆proposal/retrieval/margin三门槛，不能把第61节25.49%全部当作0.95损失。
H1/H2仅救援原Active extras中0.90≤retrieval<0.95的灰区，分别为直接准入/新增区域质量准入；
R新增灰区不混入H，训练候选与原Active来源匹配，不复用旧验证器权重。
训练/标定均遵守原8,466/2,116图划分；内部验证再哈希分C_cal/C_check各1,058图，
训练侧检索排除同源图片，所有真值GT仅用于提交预测后的评测及单列Oracle。

首轮资源门槛为真实mIoU至少+0.2个百分点，并检查机制对应的FP/FN/覆盖指标；
所有数字是预登记筛选标准，不是收益承诺。通过者做完整R/P/H因子消融及预算对照，
锁定后以当前配方seed42/123/2026逐seed配对确认。已有VOC val全量被Oracle查看，
只能称开发集；最终泛化需未参与设计的隐藏测试或独立同协议数据。
本地只新增计划文件，不启动服务器任务；执行时另建新版本并重新核验CLI、签名、输入及GPU占用。

## 63. Oracle后续R/P多卡实验（2026-10-15）

用户已授权按第62节计划在集群执行。首批版本为`oracle_guided_rp_20261015_160701`，
独立入口`tools/experimental_oracle_guided.py`，Slurm脚本`tools/hpc_oracle_guided.sbatch`；
R、P和只读评测分别在`tools/oracle_guided_r.py`、`tools/oracle_guided_p.py`、
`tools/oracle_guided_evaluation.py`。不修改src、configs、主线或旧实验。

首批为B/R1/R2/P1/P_uniform/P2六组，固定原模型、伪标签、内部划分和中心0.2。
R1/R2各一次20类FP32补充生成，R2只将餐桌主提示换成table，实际上下文张量冻结共用；
新增mask在原图token上重聚合，再用原MLP和单库检索，只补原预测背景。
P共用一个10轮末轮线性像素头，只用原训练伪标签监督；内部C_cal按原图像素标定阈值，
C_check单列诊断，三个P变体只删除最终前景像素，不再回填。
全部六组1449张预测提交SHA后才读取验证GT；B需精确匹配原总矩阵和逐图矩阵。

集群根目录`/data/home/zhengyuan/25-nd/1010`，环境`moduseg_sam3`。
不可变代码和计划在`experiments/hpc_launch_audits/<版本>/code`与`plan`；
输入清单和阶段凭据在`experiments/<版本>`，新缓存`storage/oracle_guided/<版本>`，
权重`weights/oracle_guided/<版本>`，worker日志`logs/<版本>`，Slurm日志`logs/oracle-guided-<JobID>.out`。

复用已完成集群FP32整图特征，来源阶段签名、图像身份及编码器参数需一致；
预检不重新读取12031份特征数组或逐份receipt，内容SHA在worker实际消费时检查。
原模型、库、精确基准及候选由原服务器以逐文件流式SHA方式补齐约472MiB，
原伪标签有35份需用独立副本，验证GT文件身份不一致也使用独立原文件副本约4.2MiB，
仅用于预测全部提交后的评测；不覆盖集群数据，不据此推断差异仅是文件编码。
桥接目录为`storage/oracle_guided_bridge/oracle_guided_inputs_20261015_154701`。
前一版本`oracle_guided_rp_20261015_154701`只因GT身份不一致在CPU预检停止，未提交GPU。

当前预检62.27秒通过，关键2987项SHA、原模型/库及数据身份已绑定，
未读取token数组、未解码GT；本地47项测试44通过、3项依赖Torch/SciPy的测试留待集群。
Slurm探针`3687233`申请4卡，正式R1=`3687234`申请8卡、R2=`3687235`申请8卡、
P=`3687236`申请4卡，三者均afterok依赖探针；CPU评测`3687249`在debug分区，
afterok依赖三项正式任务。GPU任务48小时、CPU评测2小时为分区时限。
探针需真实模型、原基准精确回放及协议测试全部通过；正式任务由Slurm自动衔接。
具体提交命令、配额及代码摘要见启动审计目录`jobs.json`，不能把排入队列当作正式启动验收。

首批结束后按原计划固定门槛判断：R需结合几何和真实收益，P_uniform只作对照，
H空间只作oracle资源门槛；未锁定R不能凭其H空间自动进入组合。
通过者才进入已登记H及必要组合/多seed，不扫描阈值、不自动部署。
集群无SMTP凭据，结果仅转发summary.json、metrics.json、per_class.csv到原服务器，
用既有私有邮件配置通知2858372757@qq.com；独立记录SMTP接受状态并防重复，最多3次重试。
真实探针、正式启动验收与后续结果在本节继续追加。

恢复记录（2026-10-15 20:21）：原probe作业3687233完成47项集群测试、原B精确回放与真实R/P探针；
R1/R2全1449图完成，Slurm占用分别13分00秒、14分30秒。P在采样阶段因GPFS配额查询
15秒超时失败，不是OOM或模型前向失败；旧评测依赖作业3687249已取消。
同时发现失败日志路径拼接错误。旧目录及全部产物保留，未读取本批验证GT。

新恢复版本`oracle_guided_rp_recover_20261015_201700`，新增准备入口
`tools/prepare_oracle_guided_recovery.py`；仅修复瞬时quota失败最多3次重试、失败日志路径，
以及新目录中导入SHA和输入绑定均通过的原P逐图采样检查点。配额不足仍立即停止，
连续查询失败仍停止，不使用过期配额代替实测。方法参数、R/P训练规则及内部划分不变。
来源manifest需匹配原提交记录，旧源码、原probe/R完成记录和R预测提交均绑定SHA；
R输出作为只读来源，P已完成的样本复用，未完成样本在新版本重算。
恢复probe还须真实验证旧sample导入、B1449精确回放、R预测SHA和32图P技术流程。

本地48项测试45通过、3项待集群依赖。恢复probe=`3688876`（1卡/4CPU/24GiB），
P=`3688877`（4卡/16CPU/128GiB，afterok probe），
CPU评测=`3688878`（debug分区8CPU/48GiB，afterok P）。
一次1卡8CPU的请求因集群每卡最多6CPU而在提交前被拒，无GPU作业启动，已改为1卡4CPU。
恢复目录`jobs.json`记录实际请求，原R计算不重复申请GPU。
自动监测任务`oracle`每15分钟核查，正常运行静默，完成或失败通知；根据固定门槛继续登记后续。

正式恢复启动验收：probe3688876以0:0完成，用时5分31秒，48项集群测试全部通过；
B1449图逐像素一致、两R预测SHA及原sample导入、32图P真实技术流程通过。
P3688877已在c2v100m2-7运行并分配4张GPU，当前为8个CPU采样worker阶段，
后续轻量头训练使用一张卡，标定及预测分四卡并行，不据此声称采样时GPU已满载。
通过Slurm节点内/proc精确命令核对worker身份，确认日志推进、6328份完成采样凭据，
首批3份NPZ的SHA、FP32形状、标签范围及128/32采样上限通过。
凭据为`experiments/hpc_launch_audits/oracle_guided_rp_recover_20261015_201700/startup_validation.json`，
`formal_start_reported=true`，不重复报告首次启动。CPU评测3688878仍等待P成功依赖。
R1/R2目前是预测已完成，首批六组汇总指标需待全部预测提交后统一读取GT计算。

第二次恢复记录（2026-10-15 21:32）：P3688877已完成8,466张fit图片的全部采样和1,061,797条样本mmap，但在正式训练首轮前失败。原因是PyTorch 2.7的CUDA allocator尚未初始化就调用reset_peak_memory_stats，报Invalid device argument；不是OOM。原作业已退出、角色锁释放，旧依赖评测3688878已取消，历史不删除。
新独立版本`oracle_guided_p_cuda_recover_20261015_213000`只修训练入口CUDA设备初始化顺序，固定方法参数未变；保留原R几何来源路径，直接只读复用旧完整sample阶段及mmap，消费时核SHA，不再扫描原token或重做采样。新增回归测试验证CUDA初始化先于显存统计，CPU不访问CUDA；本地28项相关测试27通过、1项Torch依赖待集群。
新探针额外真实运行曾失败的train_job入口，以一张fit图的缓存训练轻量头；原32图P几何探针凭据可复用，避免重复读取token。probe3689130（1卡4CPU24GiB）、P3689131（4卡16CPU128GiB）及debug CPU finalize3689132（8CPU48GiB）已按afterok依赖提交，均以新审计目录jobs.json为准。当前仅代表成功提交，正式P恢复待新探针通过并核验日志后验收。不改变六组全1449预测提交后才读取GT的顺序，不干预第64节Margin任务。

第二次恢复正式启动验收（2026-10-15 21:39）：probe3689130以0:0完成4分01秒，49项集群测试全部通过，B1449精确回放、R预测SHA及实际train_job入口小训练通过，已签名32图几何探针只读复用。P3689131运行于c2v100m2-7并分配4卡；当前train阶段仅一张卡训练轻量像素头，后续标定/检查/预测四卡分片。worker2808638经Slurm节点/proc核对完整命令，启动日志推进；直接复用8466图和1061797条采样，不再采样或重读原token。新审计startup_validation.json为passed/formal_start_reported=true；这不是10轮或1449预测已完成。CPU评测3689132仍依赖P完成，自动任务已更新至新版本。



汇总调度修复（2026-10-16 01:36）：R1/R2/P均已完成，六组预测各1449张。旧汇总3689132从未运行；debug每CPU最多3000M，48GiB请求被扩成17CPU，超出debugQOS的8CPU/24000M限制。仅取消此排队作业，保留历史；同冻结脚本/manifest/方法参数，以debug 8CPU/23000M/2h重新提交finalize3691486，无GPU前向或重训。新增finalizer_resubmission_001.json为当前调度凭据，不改旧jobs.json，不重复提交旧/新作业。



首批完成验收（2026-10-16）：六组均1449张，B86.821717%、R1 86.822603%（+0.000886点）、R2 86.544122%（-0.277595点）、P1 86.770844%（-0.050873点）、P_uniform 86.766653%（-0.055064点）、P2 86.767712%（-0.054005点）。R1区间跨零，R2下降；R/P全部未达继续门槛，R1/R2 cap后几何增益仅0.000028/0.027661点，餐桌0/0.375076点。本配方按停止规则结束，不扫描阈值、不部署。H原B灰区乐观空间+0.249652点仅是诊断，直接放宽回放86.475600%（-0.346117点）；不能冒充方法提升或绕过本轮停止条件。P完成10轮固定末轮，8466 fit/1058 cal/1058 check、1061797 token样本，训练head/小产物SHA及六组总/逐图矩阵由completion_acceptance核对；未重新读token或解码GT。最终汇总3691486完成0:0、12分39秒，01:48:49结束；R1/R2各13分/14分30秒，P恢复35分45秒。全流程墙钟9小时22分含故障恢复和调度等待，不累加并行时间。本地experiments/hpc_results/oracle_guided_p_cuda_recover_20261015_213000/verified_20261016保存summary/metrics/21类CSV/analysis_report及验收；邮件以独立notification_relay为准。

## 64. Hard-Class Prototype Margin三seed独立实验（2026-10-15）

用户授权在集群新增实验，要求少量必要检查、复用缓存。版本
`hard_class_margin_20261015_205100`，入口`tools/experimental_hard_class_margin_queue.py`，
损失与训练`tools/hard_class_margin_core.py`，测试`tests/test_experimental_hard_class_margin_queue.py`，
提交脚本`tools/hpc_hard_class_margin.sbatch`。本轮按用户方案固定检索中心0.4/local0.6，
三个旧模型与九个新模型统一口径；当前正式主线中心0.2不变。
用户文本中的旧服务器路径不作为部署路径，实际使用集群`/data/home/zhengyuan/25-nd/1010`。

复用原34880个1280维GMP成员、原8466/2116内部图片划分、原两层MLP、v1的CE+.5SupCon、
AdamW(.001/1e-4)、50×100、8类×4区域和原选模指标knn5 macro。
唯一训练项为前景prototype hinge margin：mild(.05,.2)、main(.10,.2)、strong(.10,.5)，
各seed42/123/2026，共九次训练。前5轮只用原损失，5/10/…/45轮末刷新20类前景中心，
中心detach、L2归一化，按VOC1–20顺序；不建立背景prototype，不人工指定chair/sofa负类。
“完整训练库”解释为内部train图片的全部成员，内部留出不进入prototype更新或训练。
不重新SAM3/GMP，不更换伪标签、成员、采样、optimizer/scheduler、提议、NMS、HCR或检索参数。

部署采用薄代码快照：只上传新增入口、损失、测试和sbatch，生产模块引用第63节已签名的冻结快照，
仅核对11份直接相关生产文件和新增文件，不重复扫描整套token或全部工程。
桥接`storage/margin_bridge/<版本>`：从原服务器补传4432个文件共177241358字节，
包含三seed历史MLP/导出库、原查询GMP、clean提议及HCR历史总/逐图矩阵；逐文件流式SHA通过。
原1280维训练库、raw提议、图像与独立原验证GT继续只读引用第63节缓存，不搬整图特征。
查询receipt在新`storage/margin_inputs/<版本>`重定位，并保留原artifact内容SHA和mask哈希绑定。

执行依赖为：新增8项损失/协议测试及6轮×2步真实技术探针、2图clean/HCR联通→
三seed历史模型完整clean/HCR复评→九个1卡训练评测任务并行→CPU汇总。
历史HCR必须逐元素匹配总矩阵及1449张逐图矩阵，不符则阻断九次新训练。
历史没有三seed clean逐图参考，clean先建立本次同条件基准，不声称它是历史精确复现。
每个新模型先1449图clean，再同模型原固定HCR；总共18次新评测，加6次基线评测。
每卡最多一个计算worker；probe/基线1卡每任务，九组通过Slurm array上限9并行，
不会中断或修改第63节Oracle作业。每个GPU任务3CPU/12GiB，48小时时限；汇总用debug CPU分区。

权重`weights/sam3_adapter/<版本>/<方法_seed>/attempt_*`，
导出库`storage/mlp_banks/voc/<版本>/<方法_seed>/attempt_*`，
任务参数、逐图矩阵、指标在`experiments/<版本>`；
worker日志`logs/<版本>`，Slurm日志`logs/hard-margin-<ArrayID>_<TaskID>.out`。
不可变代码与提交命令在`experiments/hpc_launch_audits/<版本>`，阶段失败保留新attempt，
完整epoch检查点绑定代码/输入/参数，可在同条件新attempt恢复，不覆盖旧结果。
每epoch保留模型/分类头/optimizer/scheduler/RNG/prototype状态；保留5GiB真实配额，瞬时查询失败限3次。

记录逐epoch原CE/SupCon/margin、自动hard negative分布、正负相似度/gap/激活率，
每次刷新20×20中心矩阵及chair-sofa/top10相似类对；另存最终50轮及选中模型的中心/内部留出gap。
评测后才统计GT多数类候选的local/center/fused top1、chair-sofa混淆、三seed均值/配对差、
baseline前景错类top10及逐类变化，单类下降>1点单列；这些诊断不进入训练、阈值或负类选择。
输出summary.md/json、metrics.json、per_class.csv。不自动判成功、不加扫参或部署。
完成后沿用原服务器既有SMTP配置转发三个小结果附件，记录SMTP接受状态，不声称收件箱投递。
实际Slurm作业号及启动状态在验收后追加。

提交记录及路径恢复：首版probe3689005的8项测试及6×2步真实训练通过，第5轮刷新prototype，
第6轮margin loss=0.07709；评测因薄快照工作目录缺少相对`configs/cls_voc21.txt`失败。
原baseline数组3689006、九任务数组3689007未开始，已取消，保留首版全部记录。
修正仅让子worker以冻结生产代码目录为cwd。新版本为
`hard_class_margin_recover_20261015_211100`，通过`--prepared-source`复用首版已验证输入准备、
169MiB小缓存和6轮技术训练，不再复制数据、重建链接或重训短探针；训练实现SHA必须不变。
新probe仍跑8项测试及真实2图clean/HCR，验证修复后的实际推理路径。

当前Slurm依赖链：probe=`3689044`；三基线array=`3689045`（0–2，最多3并行）；
九任务array=`3689046`（0–8，最多9并行）；汇总=`3689047`。
旧debug分区因用户提交名额上限未接受首版汇总，恢复版汇总改在1卡Slurm分配内仅用CPU，
依赖九任务成功后执行并立即释放资源，不占用debug队列名额，不在登录节点计算。
新版本`jobs.json`保存全部命令、依赖和配额；未把排入队列当作九卡训练已开始。
自动监测`hard-class-margin`每15分钟检查，运行正常静默，完成/失败或需用户行动时通知。

恢复启动验收（北京时间21:17）：8项测试及修复后的2图clean/HCR均通过，
probe3689044完成，三基线3689045_0/1/2在c2v100m2-3各占1卡运行。
节点内/proc精确命令核验worker3615360/3615561/3615560，日志均持续推进；
启动凭据`experiments/hpc_launch_audits/hard_class_margin_recover_20261015_211100/startup_validation.json`
为passed、formal_start_reported=true。此时phase=historical_baseline_replay，
new_training_started=false，九任务3689046仍等待严格基线通过，不能声称九卡已在训练。
后续正常阶段变化静默，全部完成/失败或需要用户行动时通知并发结果邮件。


完成验收（2026-10-15 22:04）：九任务3689046_0–8及汇总3689047全部COMPLETED 0:0。九个模型各完成50轮/5000 optimizer step，34880成员/sample_id、8466/2116图片划分、20个detach归一化前景中心、5轮warmup和5至45轮刷新全部通过；24组各1449严格评测，总/逐图矩阵一致，三基线HCR与历史逐元素一致。独立CPU验收作业3689256完成1分03秒（脚本验证36.15秒），仅读结果/模型/小库，不重读token、密集GT或重新前向；检查最后完整checkpoint、选中adapter、库和指标SHA，历史epoch保留签名sidecar。验收凭据experiments/hard_class_margin_recover_20261015_211100/completion_acceptance.json，额外机制/候选诊断analysis_completion.json，原summary/metrics未修改。

三seed配对HCR结果（统一中心0.4，不能与正式中心0.2的seed42 86.821717%混用）：

|配方|三seed HCR均值%|相对基准均值(pp)|配对增益样本std(pp)|best epoch 42/123/2026|
|---|---:|---:|---:|---|
|baseline|86.575359|0|0|38/29/37|
|mild(.05,.2)|86.507394|-0.067965|0.117717|38/28/37|
|main(.10,.2)|86.575315|-0.000044|0.000076|38/29/37|
|strong(.10,.5)|86.572904|-0.002455|0.002637|38/29/37|

mild在seed123下降0.203893点；main三个seed均未正向，strong仅seed42提升0.000408点，另外两seed下降0.004784/0.002990点。clean均值baseline85.663944%、mild85.622210%、main85.663944%、strong85.663945%，同样没有明显收益。main/strong的chair↔sofa错误像素基本不变，mild虽减少部分chair→sofa，却增加sofa→chair；seed123 clean chair/sofa各下降约1.334/1.375点。
非warmup训练hinge激活率mild/main/strong约0.957%/1.370%/1.364%，固定全局prototype hinge对多数训练区域无额外梯度；选中模型的chair/sofa中心cosine由基准均值0.577681变为0.585874/0.577981/0.578482，未呈预期下降。不能据此证明MLP或整个margin方向无效，仅说明本轮固定配方没有稳定收益，不继续扫margin，不部署。
原恢复流水线21:12:31至21:55:00墙钟42分29秒（含探针；基线回放约15分24秒，九模型并行训练评测最长25分39秒，汇总23秒，不累加九任务时间）。本地小结果和验收保存在experiments/hpc_results/hard_class_margin_recover_20261015_211100/acceptance_20261015_215800，含analysis_report.md、21类CSV及SHA验收。三附件经原服务器新hpc_results目录转发，2026-10-15T14:02:35Z首次SMTP尝试已接受，独立notification_relay.json记录，未确认收件箱投递。本轮结束，停止自动监测，不追加实验；Oracle任务保持独立运行。


## 65. Class-agnostic geometric completion（2026-10-15）

用户授权实现并在集群做固定配方开/关对照，新增src/geometric_completion.py、tests/test_geometric_completion.py及独立入口tools/experimental_geometric_completion.py。生产接入为run_inference.py的clean rasterization → 原HCR → postprocess_semantic_map几何补全 → 保存/评测。src/inference.py仅在capture_clean_sources=True时为NMS保留的clean实例标记来源；不改SAM3、GMP、MLP、库、检索、HCR函数或其规则。HCR新增实例不能作为扩张源。默认关闭，关闭时不添加来源标记，原rasterization返回图不变。

配置在configs/experiment.py的GEOMETRIC_COMPLETION和configs/config.py的INF_GEOMETRIC_COMPLETION。固定默认kernel_ratio=.10、min/max_kernel=3/31（正奇数）、min_mask_density=.15、min_component_area=16、max_component_ratio=.10、max_total_add_ratio=.15、min_enclosure=.35、bbox_only/background_only=True。后两个是本版本硬约束，设False拒绝运行，不允许覆盖已有前景。沿用clean现有接受门槛，不额外创造新的score阈值，不加餐桌专用规则或GT触发条件。

水平(1×kw)和垂直(kh×1)closing分别计算后union，原mask不变。原bbox内仅当前背景生成8连通candidate，enclosure定义为component内边界像素中位于原mask一像素8邻域膨胀内的比例。先面积/包围度过滤，再按enclosure降序、连通域ID稳定排序，整块接受或跳过，超过单实例15%预算不裁块凑数。实例按原score稳定降序，每次写图即时保护占据。所有类别使用同一几何规则。

开启示例：python run_inference.py --strict-eval --rescue --geometric-completion --geometric-completion-config PARAMS.json；关闭用--no-geometric-completion。JSON可覆盖全部配置，也支持OVERRIDE_INF_GEOMETRIC_COMPLETION_<KEY>环境变量。--geometric-completion-debug开启调试，debug_image_limit默认5。每图/实例JSON记录原面积、density、bbox、kernel、candidate/accepted面积和每块enclosure/拒绝原因；按类汇总触发数、新增像素、触发实例及全部实例的平均新增比例。debug保存原mask、envelope、candidate、accepted completion及before/after semantic PNG。

独立实验版本geometric_completion_20261015_221600，复用第63节原模型mainline_edge_cleanreuse_20261011_114857、中心.2、seed42的全部候选与检索分数，原HCR最终B=86.821717%作为精确对照。新输出storage/geometric_completion/<版本>，指标experiments/<版本>，启动/worker日志logs；代码/输入清单在experiments/hpc_launch_audits/<版本>。不重做模型、特征或库，不读取训练/验证GT供补全决策。所有1449图开/关预测与SHA提交后才统一读取原验证GT评测，检查B历史总/逐图精确和全部原前景不变。输出两组21类IoU、FP/FN/错类、正确/错误新增像素、逐类触发/面积、2000次配对图片bootstrap及耗时，VOC val仅属单seed开发证据，不自动推广主线。

仅8项模块测试、2张真实缓存联通探针；本地缺cv2，语法检查通过，8测试在集群实际环境执行。不重扫token、模型权重或全部工程。Slurm v100g32申请1GPU/6CPU/24GiB，几何计算用4个CPU分片，GPU不做新前向；此申请是GPU分区资源配额，不表示几何算法由CUDA加速。probe通过afterok自动运行全量。复用已签名来源/小缓存，消费时核候选SHA，至少保留5GiB真实GPFS配额；SIGTERM只停自己的worker，逐图完整凭据可在相同条件新attempt恢复，旧产物不覆盖。源码和参数更改须新版本，失败不能降阈值、跳图或改精度规避。

实际JobID、探针及正式启动验收在后续追加。邮件沿用已授权原服务器私有配置，仅转发summary.json/metrics.json/per_class.csv；独立记录SMTP接受，正常阶段静默，完成/失败通知，不追加参数扫描。


正式完成（2026-10-15 22:24）：probe3689350完成0:0、17秒，8测试及2图真实缓存联通通过；正式3689351完成0:0、44秒（脚本41.53秒），1GPU配额内4CPU分片执行。全部1449图开/关预测先提交SHA再读取GT，关闭输出及历史总/逐图矩阵精确，开启保护所有原HCR后前景。验收completion_acceptance.json及本地三结果文件SHA通过，无需额外全工程测试或模型前向；没有后续参数扫描。
基准86.821717%，本配方86.616168%，下降0.205549点，2000次配对图片bootstrap 95%区间[-0.317144,-0.103121]点。共2324实例新增1669577像素：381538正确、472280背景误报、22492前景错类、793267忽略区。餐桌/沙发IoU分别+2.147739/+0.552562点，自行车/椅子/摩托车分别-2.525583/-0.845682/-0.810068点。各类完全同规则；bbox/background保护不能辨别真实孔洞和缺失前景，密度阈值未充分保护细结构。本次结果仅否定当前固定配方的整体收益，模块默认关闭，不自动推广或追加阈值/类别特例。
本地小结果及验收为experiments/hpc_results/geometric_completion_20261015_221600/verified_20261015_222400，含analysis_report.md/summary.json/metrics.json/per_class.csv。集群逐图诊断和debug在storage/geometric_completion/<版本>；预检与提交在hpc_launch_audits/<版本>。邮件通知状态在独立notification_relay.json，以实际SMTP接受为准。


邮件验收（2026-10-15 22:26）：三结果附件SHA核验通过，原服务器既有SMTP配置首次发送已被QQ SMTP接受（2026-10-15T14:25:55Z）；独立notification_relay.json绑定三附件SHA，未确认收件箱投递。本实验全部结束，未创建后续监测或追加实验。

## 66. Geometric completion local structural support（2026-10-15）

用户授权在第65节上做第二版。只修改src/geometric_completion.py及该模块配置configs/experiment.py、configs/config.py；run_inference.py、src/inference.py、HCR/GMP/MLP/库/检索等其它主线模块不变，旧入口和旧快照不改。mode默认legacy，模块enabled默认False。新增独立tools/experimental_geometric_local_support.py和tests/test_geometric_local_support.py，旧8项测试加新7项测试、2张真实缓存联通后才自动运行。

首轮固定：local_min_thickness=6px、local_max_gap_ratio=1.0、local_min_alignment=.70、support_search_pixels=31、support_depth_pixels=16、support_alignment_padding=3、support_thickness_quantile=.25、support_gap_quantile=.75。原closing、density=.15、component/total预算、enclosure=.35、bbox/background-only全部保留，不随类别变化。所有数字在运行前注册，之后不根据VOC val修改或追加扫描。字符串mode支持legacy/local_support及对应环境变量。

实现定义：按连通域与horizontal/vertical新增区域交集像素数选择主来源方向，平票取horizontal，门控失败不换另一方向试过。先对原mask（bbox外补0）做精确Euclidean EDT。每条component扫描线在两侧31像素内找最近原前景，向外仅沿连续原前景采至多16像素，遇背景立即停止，避免借用不连通厚区域；扫描线厚度为2×该支持段最大EDT，分别取左右/上下厚度的25%分位，并以较薄侧作为support_thickness。gap为两原支持间缺失像素数的75%分位。两侧均须存在且有成对扫描线。

alignment取“局部正交窗口内两侧支持投影IoU”和“component扫描线成对支持覆盖率”的较小值。窗口在component正交bbox两端各扩3像素，避免只看一个碰巧对齐点；gap/厚度、最小厚度、alignment、原enclosure门槛需全部通过，仍按原enclosure顺序/整块预算加入，不覆盖原HCR后前景。仅使用原source mask的EDT和支持，已补全像素不能成为新支持。component日志新增direction、来源像素数、gap_length、left/right_thickness（vertical对应top/bottom）、support_thickness、gap_to_thickness_ratio、alignment、projection IoU、paired coverage、support gate原因、enclosure和accepted。

版本geometric_local_support_20261015_223800；复用第65节同一seed42/中心.2候选及检索结果，三组B关闭、legacy、local_support各1449图。所有预测先完整提交SHA，随后才读GT统一评测。B必须逐像素及逐图矩阵精确复现86.821717%，legacy必须逐像素及逐图矩阵精确复现第65节86.616168%，否则停止。源v1完整凭据/参数/产物SHA只读绑定，不再提SAM3/GMP/MLP特征，不重训，不读GT作门控、不做SAM3 feature similarity或类别特例。

产物storage/geometric_completion/<版本>，指标experiments/<版本>，新增代码/注册参数/提交记录experiments/hpc_launch_audits/<版本>，日志logs。Slurm v100g32申请1GPU/6CPU24GiB，实际是4个CPU几何分片，不声称CUDA计算。保存三组总/逐图矩阵、21类CSV、背景FP/前景FN/前景错类、按类触发和新增比例、各组新增像素的正确/背景/错类/忽略区GT事后分解；2000次配对图片bootstrap分别给local相对B和相对legacy的区间。单seed已查看VOC开发证据，不自动替换主线。

复用小缓存签名，不重扫token/模型或全工程测试，真实GPFS配额至少保留5GiB；自己的launcher/worker可安全停止，相同签名逐图完整检查点才可resume新attempt。源码改动另建版本，不改旧快照/原缓存。结束由原服务器既有私有SMTP配置转发summary.json/metrics.json/per_class.csv，不打印改配置，不传模型数据日志，SMTP接受与收件箱投递区分。实际作业及结果后续追加。



完成验收（2026-10-16核对）：probe3689545完成0:0、27秒，15项相关测试及2图缓存联通通过；正式3689546完成0:0、72秒（脚本70.08秒）。三组各1449图、全部预测SHA提交后读GT；B及legacy像素/逐图矩阵与历史精确，全部原HCR后前景保持不变。下载三结果及完成验收SHA通过，不重复大缓存或全工程检查。

B=86.821717%，legacy=86.616168%，local_support=86.801727%。第二版相对B为-0.019990个百分点，2000次配对图片bootstrap 95%区间[-0.048143,+0.008181]；相对legacy为+0.185559点，区间[+0.094415,+0.286546]。说明比旧补全明显改善，但未证明优于关闭基准，不推广主线、不追加扫参。图片bootstrap不是多seed训练稳定性证据。

第二版相对B的bicycle/chair/motorbike分别-0.065295/-0.099130/-0.151831点（旧版-2.525583/-0.845682/-0.810068），table为+0.216060点（旧版+2.147739）。新增284825像素：正确70932、背景误报74657、前景错类4460、GT忽略134776；非忽略新增像素正确率47.27%，旧版43.54%。背景FP=2486268、前景FN=4284053、前景错类=491321。新门控大幅减少误填，也移除了大量正确补全，仍未形成总体收益。

本地结果experiments/hpc_results/geometric_local_support_20261015_223800/verified_20261016含summary.json、metrics.json、完整21类per_class.csv、analysis_report.md及download_verification.json。模型/库/提议不变，各类同规则；模块默认关闭，mode=legacy与local_support可显式切换。邮件状态以独立notification_relay.json为准。

## 67. 关机后自动衔接的条件R3裁剪实验（2026-10-16）

用户授权今晚由集群自主运行、次日看结果。独立版本oracle_r3_overnight_20261016_014500，入口tools/experimental_oracle_r3_overnight.py，5项控制/裁剪回映射测试tests/test_experimental_oracle_r3_overnight.py。只新增调度入口与测试，生产模块、原Oracle代码和产物不改；薄改动叠加第63节冻结源码形成独立不可变快照。

固定第62节已登记R3：中心80%宽高裁剪后1008 FP32编码，canonical全部20类和已冻结上下文，正常presence，固定seed42、每生成类最多1个新提议。映回原图后从原整图token计算rho=.1 GMP，复用原MLP和单库/中心.2。严格原HCR阈值只补原B剩余背景。不会因本轮几何补全负结果改阈值、改类别规则或扩充原计划。

先等待首批finalize3691486退出，然后在集群读取首批完整签名结果：至少一个R1/R2同时达到真实mIoU+B的.2点及几何门槛（macro cap+.3点，或table cap+3点且macro不降）；R2还须优于同预算R1。仅几何通过而真实失败不能触发本R3，仍按原计划留待H路线。门槛不满足写skipped_no_signal，发布首批六组交接结果并释放资源，不消耗新的SAM3前向；不是伪造R3失败或成绩。

通过门槛才运行5项小测试和固定ID两图R3真实联通，再4卡分片全1449图。第二个独立CPU作业在前项退出后自动汇总B/R1/R3；B/R1复用原预测，全部提交SHA才读GT。只计算新R3几何cap，旧R1/R2几何复用已有报告。报告21类IoU、FP/FN/错类、新增像素、几何cap/R50/R75、2000配对bootstrap、相对B/同预算R1及既有R2的差，单seed已查看VOC开发证据，不自动推广或追加参数。

两个作业都由Slurm依赖推进，关本地电脑不影响执行。4卡阶段v100g32/16CPU/128GiB/48h；汇总debug/8CPU/23000M/2h，避免48GiB被自动扩核造成QOS阻塞。日志logs/r3-overnight-<JobID>.out；新storage/oracle_guided/<版本>和experiments/<版本>隔离，审计hpc_launch_audits/<版本>含注册参数、jobs和代码签名。每阶段保留5GiB实测GPFS配额，消费时校验缓存SHA，不重扫所有token；失败不覆盖历史。Slurm配置END/FAIL通知2858372757@qq.com，不能视为已送达；详细三附件SMTP转发等本机重新上线后执行，不把本地heartbeat作为夜间依赖。

夜间作业提交：R3条件运行3691535（afterany:3691486，4GPU）；汇总3691563（afterany:3691535）。debug分区因每用户提交数上限拒绝第二个汇总请求，没有生成作业；已保留失败记录，改为v100g32单卡配额内6CPU/24GiB、CUDA隐藏的CPU汇总。仅调整资源，不改实验代码或参数。以jobs.json的finalize_retry及finalizer_submission_002.json为实际汇总凭据，两个作业已提交，不重复。


夜间链完成（2026-10-16 01:49:36）：条件运行3691535与汇总3691563均COMPLETED 0:0，24秒/21秒；首批R1/R2真实及几何门槛均不达标，状态skipped_no_signal、R3_started=false。仅检查条件及整理首批结果，没有进行R3模型前向或产生R3方法成绩，不重复启动。完整停止本轮并处理一次结果邮件；原主线保持不变。


Oracle首批与R3跳过状态邮件已于2026-10-16 11:04:40北京时间被QQ SMTP首次接受，三附件SHA绑定在独立notification_relay.json；未确认收件箱投递。本轮自动监测结束。

## 68. 用户直接授权完整R3（2026-10-16）

用户明确要求“把R3跑完”，覆盖第67节“首批达标才启动”的执行前提。新版本oracle_r3_direct_20261016_111900，仍用tools/experimental_oracle_r3_overnight.py，但独立冻结新源码、manifest、注册参数和执行授权；旧skipped_no_signal目录和全部历史不改。默认conditional行为不变，仅新manifest中execution_authorization.run_R3_to_completion=true且recipe_changed=false、用户请求非空并与签名protocol一致，才直接运行。保留原未达标证据registered_gate_was_met=false，不将其改写为通过。

R3配方完全沿用第62/67节：中心80%宽高裁剪，1008 FP32、canonical20类/原邻近上位上下文、正常presence、seed42；mask映回原图后从原整图token计算rho=.1 GMP，原两层MLP/单库/中心.2，严格proposal .4、retrieval .95、margin .1、novel .1，只补原B背景。不训练，不重新生成B/R1，不改主线或阈值，不按类别特设规则。

8项执行授权/默认门槛/裁剪回映射测试通过，集群预检只核相关冻结源码和小凭据，不重扫全部token；两图真实R3生成/查询/救援联通通过后4GPU分片完成1449张。汇总复用B/R1全量预测，先确认三组预测SHA完备再读GT；输出21类IoU、FP/FN/错类、新增像素、R3 cap前后几何/R50/R75、2k配对bootstrap、对B/R1和原R2差异。新原始产物storage/oracle_guided/<版本>，指标experiments/<版本>，代码及作业记录hpc_launch_audits/<版本>，日志logs/r3-direct-<JobID>.out。

Slurm正式请求4GPU/16CPU/128GiB/48h，后接debug 8CPU/23000M/2h的CPU评测，避开48GiB扩核造成QOS阻塞；全部由集群依赖衔接，电脑离线不影响执行。保留至少5GiB真实GPFS配额，缓存消费时查SHA；失败保留attempt，旧任务退出/锁释放且相同签名才续跑，方法实现修复另开版本。Slurm END/FAIL通知已请求，详细邮件按既有授权由原服务器私有SMTP转发三个小附件，不传模型数据日志。仅固定seed42的VOC开发实验，不自动部署或追加其它尝试。实际JobID和启动验收后续追加。

第68节实际提交：R3=3697857（4GPU16CPU128GiB/v100g32），finalize=3697858（debug8CPU23000M2h，afterany:3697857），均提交成功，不重复。8项集群CPU测试及预检通过，GPFS余量356550688768字节。R3已分配c2v100m2-6，正式全量启动仍须两图探针和worker/首批缓存验收；不能仅凭RUNNING认作完成。自动监测r3每15分钟跟进，正常运行静默。

R3探针修复：首次direct版本3697857在两图探针读取canonical context时失败，原因是缓存仍绑定原oracle_guided_rp_20261015_160701的version/code，而新调用者版本已变；正式1449未启动。其后置汇总3697858同样失败，二者均已退出，保留全部日志和旧产物。不是OOM或配方失败。

恢复版本oracle_r3_context_recover_20261016_113400，仅新增tools/import_oracle_r_context.py与3项导入测试，并在新prepare中执行受控导入；生产tools/oracle_guided_r.py和SAM3代码不改。要求原context生成者manifest SHA匹配原提交记录，原binding版本/源码摘要、权重SHA、canonical词表及所有相关SAM3/Active/精度代码摘要与新版本一致；源checkpoint内容SHA必须匹配原direct已签名identity。复制原始context字节到新storage版本，保留原tensor_sha256、生成者binding、manifest/原receipt/artifact身份和导入依赖清单，仅新副本写入新归属binding。新metadata身份也进入manifest，原缓存与原receipt不修改；真实探针仍重算每个tensor哈希。不是放松签名或重新计算上下文。8项原控制测试+3项导入测试共11项通过，R3 recipe和用户直接执行授权不变。旧作业不重启，新作业号待记录。

CPU准备113400在新增导入工具的模块搜索路径处失败，未创建manifest/jobs、未申请GPU。恢复准备113700改为从新快照绝对路径加载导入工具；生产快照以只读硬链接复用，任何补丁先删除新目标链接再写入，原文件不修改。该差异仅在部署准备层，R3算法与张量内容不变，保留113400的CPU失败历史。后续以oracle_r3_context_recover_20261016_113700为当前版本。


最终恢复版oracle_r3_context_recover_20261016_113700提交成功：R3=3697970（4GPU16CPU128GiB/v100g32），汇总=3697971（debug8CPU23000M2h，afterany:3697970）。上下文迁移来源校验通过，内容SHA仍为a9d9c13e65c34003ddce3fbf4d606470dc5779479da98d0ccc48631c3600990d；11项集群CPU测试与预检通过。manifest SHA de989140dac7138124a99c2d127b67da80c55c5bd6e5cb4373dee6f00b4a36bc，实测GPFS余356677763072字节。旧作业不重启，新两个作业不重复提交；全量验收仍需新两图真实探针、4worker和首批R3缓存。

R3恢复版首次正式全量验收：两图真实探针通过，context实际tensor哈希一致、原有前景保持；3697970在c2v100m2-6正常RUNNING，四卡worker PID677331/677332/677333/677334，日志已分别推进到90/88/91/93张（合计362/1449生成缓存，并非最终评测完成）。各卡首份NPZ的内容SHA、原图尺寸、二值mask、每类cap及FP32/1008配置通过必要检查。startup_validation.json记录formal_start_reported=true；后置CPU评测3697971按依赖等待，不重复提交。后续正常进度静默，完成或失败再通知。

第68节完成结果（2026-10-16）：3697970与3697971均COMPLETED 0:0，分别41分29秒和14分39秒，连续墙钟56分08秒；正式流水线2441.29秒、实际评测821.07秒，单卡峰值5174.93MiB，不累加并行worker耗时。B/R1/R3各1449张严格评测，R3全部预测SHA提交后才读取GT；B总及逐图矩阵精确复现、逐图求和、原有前景保持、产物SHA验收通过，锁已释放。

| 方案 | mIoU | 相对B百分点 | 背景FP | 前景FN | 前景错类像素 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | 86.821717% | 0 | 2411611 | 4359445 | 486861 |
| R1 | 86.822603% | +0.000886 | 2412870 | 4356932 | 486864 |
| R3 | 86.933660% | +0.111943 | 2456424 | 4269553 | 487497 |

R3比同预算R1高0.111057点、比原R2高0.389538点。2000次配对图片bootstrap的R3-B 95%区间为[-0.034871,+0.306296]百分点，包含0；仅为seed42且已查看VOC开发集证据，不能当作跨seed稳定收益。chair提升1.200861点、bird提升1.196071点、sheep提升0.329973点；pottedplant下降0.342608点、table下降0.113624点、ship下降0.094009点。

64个候选获准补全，原图总新增154745像素；其中有效GT像素134705，正确前景89256、背景误报44813、前景错类636，另20040像素属于忽略GT。几何oracle上限从B95.297114%到R3 cap前95.710883%/cap后95.509270%，分别+0.413769/+0.212156点，均为诊断值。面积至少300的3905个语义区域，B R50/R75=75.288092%/66.555698%，R3 cap前76.414853%/67.580026%，cap后75.492958%/66.837388%。真实+0.2点及cap后几何+0.3点登记门槛仍未通过，不自动替换主线或追加实验。

本地结果：experiments/hpc_results/oracle_r3_context_recover_20261016_113700/verified_20261016，含summary.json、metrics.json、per_class.csv、analysis_report.md（21类完整表）和小验收JSON。summary/metrics/CSV SHA分别dc26e5d0b9beece90de508ed4607a72bb731aa1553de8dd7347992332a036dca、63ffcea1afc79771f041189c5a89559ae0551b08014f4d1b0fafa727db3b8813、c53548a0464cbfefa0243b60f6edafa480c38f0ff7531aa5fcb307cb6f6699cf。completion_verification SHA c63c13fab74292580e2684d704d58d7e110f471abb2e831e8205519d7eb34f8e。

详细邮件通过原服务器私有配置发送2858372757@qq.com，仅转发上述三个结果附件；2026-10-16 12:54:26北京时间首次尝试被SMTP接受，不代表已进收件箱。notification_relay SHA b1caee9a109c24b88e0a48acf1ddc4d85761698e11e6f0df99e7fa0774a75c25，禁止重复发送。完成验收后删除r3自动监测。

## 69. 固定R3三seed配对稳定性验证（2026-10-16）

用户在第68节R3完成后明确授权“去做吧”，补42/123/2026的配对验证，不受原首批+0.2点启动门槛限制。版本r3_paired_20261016_142200，独立入口tools/experimental_r3_paired.py，8项CPU小测试tests/test_experimental_r3_paired.py，Slurm脚本tools/hpc_r3_paired.sbatch。不修改src/configs或旧快照，不调阈值，不自动替换主线。

复用第68节全部1449张R3提议及查询缓存中的原始1280维GMP，不用旧投影向量和旧检索类别。复用第64节桥接的三个原始基线MLP及其各自导出库、签名查询GMP和固定8466/2116内部图片划分。没有SAM3前向、GMP重新提取或MLP训练，背景/成员/伪标签/提议几何不变。每seed只重新投影和生产balanced检索：K25、每类top5、搜索倍数2、中心0.2；原clean NMS/HCR后得到该seed自己的B，再以原R3严格proposal.4/retrieval.95/margin.1/novel.1仅补该B背景。第64节评测中心0.4不混入本次；123/2026建立新的匹配中心0.2基准。seed42全1449张B和R3必须与第68节预测逐像素相同，否则停止汇总。

小预检复用历史签名，核模型/库、内部划分、相关源码和完成凭据；缓存消费时核内容SHA及mask身份，不扫整图token或全工程。两图seed42真实回放和8项小测试通过后，Slurm array 0-2%3三卡各一个worker执行三seed。每seed4h、3CPU/12GiB；CPU汇总debug8CPU/23000M/2h依赖所有seed成功。每图可校验恢复检查点，失败保留attempt，新代码启用新版本。GPFS真实配额保留5GiB，SIGTERM仅作用本Slurm worker，不启动其他实验。

各seed均提交B/R3两组1449预测SHA、共8694份预测后才解码GT。输出六组mIoU/21IoU/总和逐图混淆矩阵、FP/FN/前景错类、新增像素类型、每seed2000次配对图片bootstrap；汇总三seed配对增益均值、样本std、df2的t95区间及正向seed数，说明小样本与已查看VOC开发集限制，不以2/3正向直接判成功。数据及PNG在storage/r3_paired/<版本>，参数/指标在experiments/<版本>，冻结源码/协议/JobID在experiments/hpc_launch_audits/<版本>，日志logs/r3-paired-<ArrayID>_<TaskID>.out；不新增模型权重。完成后仅三个结果附件经原服务器私有SMTP转发，并结束监测。

第69节已完成：probe3699681、三seed array3699682_0/1/2、finalize3699683均COMPLETED 0:0，耗时4:16、13:29/17:37/17:37、5:20（并行不累加）。seed42的B/R3=86.821717%/86.933660%，+0.111943点；seed123=86.321692%/86.413496%，+0.091804点；seed2026=86.374583%/86.474971%，+0.100388点。配对均值+0.101378点、样本std0.010106点，df2 t95区间[+0.076273,+0.126483]点，3/3正向；这是当前VOC开发集三个训练seed的小幅稳定收益，不证明跨数据集收益，不自动部署或把R3加进后续主线消融。

seed42两组全量逐像素精确回放通过，六组各1449、全部预测提交后才读GT、签名和本地三个结果SHA通过，worker锁释放。结果experiments/hpc_results/r3_paired_20261016_142200/verified_20261016；summary/metrics/CSV SHA为bd836079a0732da532e5a6c2ac565c3aced3f1e5702662d9575d804c5c7338f3、253e13f99cfe28e2245f066b04bebd6d74df2992b19687dfa6ef60b2d804ba5d、e725535f4a096d0a92650417c32a0537689a0eb16ce08dfe83b9e5a1f824a24a。详细三附件邮件已被QQ SMTP首次接受，独立notification_relay SHA8b37860f7ba9cbd40d18330d0c38e88096002d5ee744b9d4e22b9960a6ef7c1d；不确认入收件箱。

## 70. 四组主线模块消融（2026-10-16）

用户授权集群后台对照：1完整原主线，2仅把GMP换成SMFA，3仅移除MLP，4仅关闭HCR。版本mainline_ablation_20261016_230700，入口tools/experimental_mainline_ablation.py，测试tests/test_experimental_mainline_ablation.py，提交tools/hpc_mainline_ablation.sbatch。固定seed42、10582原训练图、1449验证图、原清理标签和raw/clean提议、FP32/1008、中心0.2、原检索/NMS/HCR参数，R3和几何补全均不加入。不修改src/configs、原模型、原库、旧结果及运行中快照。

完整原主线复用原MLP/原导出库并重新回放1449预测，须逐像素匹配86.821717%历史；去MLP直接使用原34880成员的1280维GMP原库与原始GMP查询，完全没有adapter，HCR阈值不重新标定；去HCR保留原GMP/MLP及clean主路径排序、同类NMS、像素合成，额外提议不进入结果。

SMFA采用严格固定成员的聚合替换：保持GMP已过滤后34880个sample_id、类别、区域顺序和原8466/2116图片划分，**不再次执行10%过滤**，从而排除成员变化这个混杂因素。读取原伪标签重建同一个区域，使用生产BaseFeatureExtractor的INTER_AREA覆盖率和加权平均/L2实现SMFA（context_subtract=0）；整图token只读缓存、实际消费时核SHA，不构建SAM3编码器。新建SMFA原库并新训匹配的MLP2 1280→512→256，v1/seed42/50×100/CE+.5SupCon/AdamW.001和原内部选模；不能套用GMP的MLP。验证侧同样重新SMFA聚合原候选并使用匹配的新MLP/导出库。SMFA签名为原签名去除aggregation，严格与GMP空间分开。本组属于固定成员消融，不宣称是SMFA独立重新筛库的最优成绩。

CPU仅检查相关源码/小凭据与固定成员，原始token消费时SHA；本地4项CPU通过，另2项真实生产SMFA运算在集群torch环境执行。probe3705115（一卡，6项测试+一训练图/一验证图真实缓存SMFA+两图完整/无MLP/无HCR联通）通过后，extract3705116 array0-3%4四卡分片训练及验证token池化，与controls3705117一卡并行。train3705118依赖全部池化完成，完成一次50轮训练及SMFA全量预测；finalize3705119依赖controls和train，debug8CPU/23000M/2h汇总四组。GPU各3CPU/12GiB/12h、每GPU一个worker，不占登录节点。JobID及manifest签名以hpc_launch_audits/<版本>/jobs.json为准，不重复提交。

产物storage/mainline_ablation/<版本>，SMFA权重weights/sam3_adapter/<版本>，指标experiments/<版本>，日志logs/main-ablation-<ArrayID>_<TaskID>.out，均新版本隔离；运行保留真实GPFS至少5GiB。四组各1449预测SHA全部提交后才读取GT，输出mIoU、21类IoU、总和逐图矩阵、相对完整主线百分点、背景FP/前景FN/错类、新增删除像素、每组2000次配对图片bootstrap、阶段耗时/峰值及SMFA最佳epoch。当前为单seed开发集消融，不自动改变主线、加seed或调阈值。结束后按既有授权转发summary.json/metrics.json/per_class.csv，通知一次并结束监测。

第70节正式启动验收：probe3705115 COMPLETED 0:0、2分39秒；集群6项测试全部通过，真实SMFA一训练/一验证图、两图完整/无MLP/无HCR联通及baseline逐像素匹配通过。extract3705116四卡worker PID1664877/3951546/3951545/2784709（array实际child3705122/3705123/3705124/3705116），在c2v100m2-2/3/4运行，各53/127/146/55份首批训练缓存，合计381份；FP32 Nx1280、原sample_id及SHA抽验通过。controls3705117 PID2784708已完成8图检查点，三个组首批PNG有效且完整主线匹配原预测。共五卡正式工作，SMFA50轮训练与统一GT评测仍按依赖等待，不能当作已完成。

manifest SHA9c8d9b878bda4839200030d0ea6b5e4f69fb5dc3a0b0431e932af334aea74116；startup_validation.json passed/formal_start_reported=true，SHA4e26c839e9bef0918c549b8fbdec9bf9ba0e26884e22f523b177ca1a5c7452fd。GPU/配额实测后启动，GPFS余356680302592字节。自动监测automation每15分钟，正常阶段静默，失败或完成才通知，不重复提交任务。

第70节完成验收（2026-10-17）：probe3705115、extract3705116四个child、controls3705117、train3705118和finalize3705119全部COMPLETED 0:0，无失败/遗留worker锁。北京时间10月16日23:12:55开始probe，10月17日00:41:31完成汇总，总墙钟1小时28分36秒；正式计算自23:15:34起1小时25分57秒。SMFA池化四卡关键路径42:24、MLP训练/导出及SMFA预测38:55、最终CPU汇总4:38；controls21:20与池化并行，不累加worker时间。池化和预测有GPU峰值记录，训练子进程未单独记录显存峰值，不声称完整流水线峰值已测。

| 方案 | mIoU | 相对完整主线百分点 | 背景FP | 前景FN | 前景错类像素 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整GMP＋MLP＋HCR | 86.821717% | 0 | 2411611 | 4359445 | 486861 |
| 仅GMP换SMFA | 86.681025% | -0.140692 | 2400411 | 4328617 | 535924 |
| 仅去MLP | 71.443980% | -15.377737 | 1410526 | 6622600 | 5665977 |
| 仅去HCR | 85.841489% | -0.980228 | 2132653 | 5020719 | 452394 |

2000次配对图片bootstrap相对完整主线95%区间：SMFA[-0.629920,+0.415239]点，去MLP[-17.979469,-12.546796]点，去HCR[-1.586246,-0.411159]点。固定配方下MLP/HCR移除下降明确；GMP相对SMFA仅小幅点估计优势，区间包含0，不能宣称已证实GMP普遍优于SMFA。此处SMFA是原GMP过滤成员固定对照，不重新挑SMFA成员、不标定无MLP的HCR阈值；四组均seed42且是已查看VOC开发集，不自动部署或追加实验。

SMFA新MLP完整50轮×100步，固定原8466/2116图片划分，34880个原sample_id在1280维原库和256维导出库中顺序一致，最佳epoch36、内部macro accuracy0.9832996504。完整主线1449张逐像素匹配历史，总和逐图混淆矩阵精确复现；四组各1449、逐图求和、全部预测SHA提交先于GT、模型/库/指标SHA验收通过。SMFA相对GMP的chair -3.013968点、car +1.485813点、pottedplant -0.936161点；去HCR会多漏661274个真实前景像素，同时少278958个背景误报和34467个前景错类像素，净效果mIoU下降0.980228点。

本地小结果experiments/hpc_results/mainline_ablation_20261016_230700/verified_20261017，含summary.json、metrics.json、per_class.csv、analysis_report.md的21类完整表及小验收JSON。summary/metrics/CSV SHA为33597120ada5bc1b0045e83e38383c1d8ef83ffbde06d2a87590e27f3337e356、6347577e252ae85480b1018f490cc808d19fb3cc2c2f8048facdcc7fbda7a66f、95f8069102a311392a4f885d8e1815aa0c9484e87528d651f9c95072667c4c8a。completion_verification SHA52d2a14b3d559926bf4b20cafbf220da871061498f6a3322d83143b1d729ceae。结束邮件独立转发凭据和监测关闭状态另行追加。

第70节结果邮件于2026-10-17 09:40:51北京时间经原服务器私有SMTP首次接受，收件人2858372757@qq.com，三个结果附件SHA绑定在notification_relay.json；不确认收件箱投递，禁止重复发送。凭据SHA3772514336e62528903483730b0fe6e9d822667ba65bbfff1dbe31aae595e202。全部验收与通知处理完成，删除automation自动监测，停止本轮后续尝试。

## 71. 局部负上下文约束GMP固定对照（2026-10-17）

用户已授权按 experiments/plans/gmp_context_objective_20261017_v1/plan.md 实施。版本gmp_context_20261017_111500，入口tools/experimental_gmp_context.py、数学实现tools/gmp_context_core.py、测试tests/test_gmp_context_core.py、调度tools/hpc_gmp_context.sbatch。独立薄代码快照放experiments/hpc_launch_audits/<版本>/code，生产模块只读复用第70节冻结代码和oracle_r3_context_recover_20261016_113700/code。不修改src/configs、主线或历史产物。协议protocol.json预先冻结全部门槛，不扫描参数。

目标函数mean(Fz-1)^2 + beta*mean(Bz)^2 + lambda*||z||²；beta=.1，lambda=.1乘内部平均能量。F保持原hard-center；B是1008坐标下mask外距离(14,42]像素环带的token中心，排除训练伪标签其它前景或验证其它clean候选，不使用GT。B少于8个退回生产原GMP。FP32较小原始/对偶Cholesky、残差<=1e-4；beta0/无外部支持精确走原生产路径。简单对照为normalize(nativeGMP-.1*normalize(mean(B)))。特征签名在独立进程中仅注册原签名派生的两个完全相等的新签名，不更改生产文件、不放宽未知签名。

先单卡7项相关测试（本地4项通过，torch/scipy的3项在集群运行）、一训练图及两验证图真实池化、原GMP查询逐元素与两图原HCR逐像素回放，并对两个新签名做GPU两步优化/保存加载技术探针。通过才四卡分片研究探针：原内部训练划分按SHA256("gmp_context_probe42:"+ID)取512图作为干净库；原内部留出取128图查询，0/5/10/20%确定性嵌套替换内部token，三方法使用相同污染。直接调用生产balanced分类器K25/top5/search2/中心.2；图片完全隔离，不重训MLP，不读VOC val GT。宏平均固定为留出存在的类别，要求每类在fit库有支持，并报告全部计数/缺失类。无污染context对原GMP不低于-.1点，10%/20%平均至少+.2点且严格优于简单扣除时才继续；否则complete_no_signal，结束本配方并通知，不占用全量训练预算。

门控通过后由集群CPU gate自动提交四卡全量联合池化、两个单卡匹配MLP训练与1449预测，再CPU统一评测；本机离线不影响Slurm依赖。四组A/B复用第70节SMFA和原GMP的已签名完整预测，C/D各新训一次MLP2/v1/seed42/50x100，原34880成员、sample_id顺序、8466/2116图片划分、中心.2、原HCR/NMS/清理不变；不再次过滤、不重跑SAM3、不加R3/几何补全。背景同规则，通常外部支持不足退回原GMP。失败保留attempt；逐图可复用校验通过的本版本检查点；训练无完整epoch恢复接口时新attempt完整50轮，不能伪造续训。最多两个新完整训练，不自动追加seed或调参。

数据/查询storage/gmp_context/<版本>；权重weights/sam3_adapter/<版本>/<方法>；参数/阶段/指标experiments/<版本>；日志logs/gmp-context-<ArrayID>_<TaskID>.out。CLI支持--preflight-only/--probe-only/--resume；所有研究计算在Slurm，每GPU一个worker，GPFS真实配额至少余5GiB。大token在消费时SHA，不重复全量扫描。阶段attempt与PID/JobID/锁隔离，SIGTERM仅停自己的worker/训练子进程。正式启动和完成依据jobs.json、阶段完整凭据及真实进程，不能把technical ready当研究或全量完成。

四组1449预测全部提交SHA后才作新GT评测，输出21IoU、FP/FN/错类像素、逐图/总矩阵、2k配对图片bootstrap、支持不足比例、残差、阶段时间和峰值。D对原GMP小于+.2点时结束配方；后续三seed是另行冻结实验，0.6点目标不是收益承诺。正常运行静默，完成/失败通知；结果邮件沿用原服务器私有SMTP转发三个小结果文件并独立防重记录，不传模型/数据、不打印配置。当前代码已准备，JobID及实际启动验收另行追加。

第71节已提交：technical3709822（一卡3CPU/16GiB）、research3709823 array0-3%4（四个一卡worker，afterok technical）、gate3709824（debug8CPU/23000M，afterok research）。manifest SHA0bb7ca6037c2f3fd098914ec3f274af19511165e3fa7f3a997856832d91642e0，binding00241360e6a05cf3d2bf52d8f72710c447bae4083a2de2814eaae1f2ca083eb5；真实CLI --preflight-only通过。technical已在c2v100m2-6运行，研究探针依赖等待，不把排队当四卡已运行。gate通过才自动登记并提交full_extract/train/finalize；当前未申请全量训练。监测gmp每15分钟，正常运行静默；实际研究启动凭据另行追加。所有作业已通过Slurm提交，本机关闭不影响集群依赖推进。

第71节技术恢复：首版technical3709822的7测试全部通过，一训练图缓存成功，但新入口漏掉CachedGMPExtractor/SAM3生产路径最终的single-phase batch L2，历史raw查询逐元素核对失败。research3709823和gate3709824在运行前取消；旧失败和缓存保留，锁释放，没有研究/全量训练。恢复版本gmp_context_norm_recover_20261017_112500仅补齐三个分支相同的生产最终归一化，并新增该操作次序测试；不是调整beta、环带、门槛或算法目标。技术核对改为同设备生产aggregate+phase-fusion逐元素一致；跨硬件历史查询记录bitwise与max_abs<=1e-5界限，最终两图原baseline预测仍必须逐像素精确。新签名记录phase-fusion，不伪装成旧代码。

恢复版已提交technical3709866、research3709867 array0-3%4、gate3709868，资源和依赖同前；真实jobs.json为准，不重启旧任务。manifest SHA543d468af0bebe58ce3f9b42f194be5ef64bb7a1955cc5301248074babfc2b42，binding92070f931e00edff6f6c81c5ddf98dad791e33709dc7b35ffc540e412c1bd524。恢复版technical须8项测试、真实图流程及两个新签名GPU优化保存加载，通过才开始研究阶段；尚不把提交当正式研究已启动。

第71节正式研究启动验收（2026-10-17 11:27北京时间）：technical3709866 COMPLETED 0:0、Slurm1:40，8项测试全过；真实一训练图/两验证图池化、两个新签名GPU两步优化保存加载、两图原baseline逐像素复现通过。同设备生产aggregate+final phase-fusion逐元素一致；历史查询非bitwise，最大绝对误差2.6193447411e-7和2.5331974030e-7，按预登记跨硬件界限通过，不称历史特征逐元素精确。

research3709867四卡已真实运行，实际child3709871/3709872/3709873/3709867，worker PID925996/1476201/1448870/1448871，分别在c2v100m2-6/11/17/17；通过Slurm节点/proc核对本版本实际命令。四份首批NPZ的SHA、FP32、[4,3,N,1280]、sample_id数量和归一化检查通过；验收时62/51/87/87共287份研究图片缓存有效（不是最终检索成绩）。startup_validation.json passed/formal_start_reported=true，SHAa6878b7046a0f35a27ebfc0a17325e47e51e6bf3ef6c8323a4c45a43d313f7aa。CPU gate3709868仍等待全部640图；未启动全量池化或两次50轮训练，不重复启动通知。集群依赖自动推进，本机离线不影响；监测gmp保持正常阶段静默。

第71节完成验收（2026-10-17）：恢复版状态complete_no_signal，属于固定研究门槛未通过，不是计算失败。technical3709866、research3709867四child及gate3709868全部COMPLETED 0:0；技术探针至门控墙钟619秒（10分19秒），不累加并行worker。512 fit图/128 heldout图全部完成，分别1724/453个区域、21类有支持，无图片交叉；640份收据签名/阶段绑定/覆盖与消费时产物SHA通过。fit/heldout分别544/136个区域支持不足退回原GMP，最大求解相对残差4.95993277e-7。进程退出、锁释放，无full_extract/train/finalize新提交，无新全量权重；没有读取VOC val GT、重跑SAM3、改变门槛或追加seed。

以下为内部伪标签代理上的原始1280维区域宏检索准确率，不是mIoU：

| 外部token替换比例 | 原GMP | 简单扣除 | 新约束GMP | 新约束对原GMP百分点 |
| --- | ---: | ---: | ---: | ---: |
| 0% | 83.1967% | 83.6115% | 85.4776% | +2.2809 |
| 5% | 79.8974% | 80.3551% | 82.1178% | +2.2203 |
| 10% | 80.7704% | 80.2757% | 79.4821% | -1.2884 |
| 20% | 79.0769% | 78.7905% | 79.2446% | +0.1677 |

无污染提升2.280884点，但预先规定的10%/20%平均对原GMP为-0.560348点（要求至少+.2），对简单扣除为-0.169789点（要求>0），所以未进入两次50轮匹配MLP或全1449张评测。仅说明当前固定配方没有通过抗污染继续门槛；干净代理正向不等同最终mIoU收益，留出部分类仅2—5区域也限制宏平均解释。不能宣称达到SMFA+.6点，也不据此否定所有外部上下文方法；本轮结束，不自动改门槛/扫参数。

本地结果experiments/hpc_results/gmp_context_norm_recover_20261017_112500/verified_20261017，含三结果、probe_report、完成验收和analysis_report.md。summary/metrics/CSV SHA为407173d169cc192d3465b53195c760af30cd8a7a51117402e61691a92692153a、c3e8a1d539328e1d394b65f7f7b5e8daa66fd7db2bdae4147dc826e9989f5a4f、ca8e04a53a4470270e01299fc2f1868b96847e5f56e5446abcb1b4605dd53f2f；完成核验SHA689e98563f832faa2483ea29adaf37517c574bc5bc1d8c1fb1413b323e3044fd。三附件邮件于北京时间11:38:27经原服务器QQ SMTP首次接受，不确认收件箱投递，独立notification_relay SHAec2a7805beb56ada87de9de7995d055a74ffb70b588a5dca20de18dd204c7312。停止本配方并删除gmp自动监测，保留旧失败及所有研究产物。

## 72. 用户新授权：上下文GMP完整分割验证（2026-10-17）

用户在知晓第71节抗污染门槛未通过、但干净代理有正向结果后，明确回复“来吧”，授权完整验证最终分割。新版本gmp_context_full_20261017_120500；旧门槛、旧complete_no_signal和旧产物全部保留，不把旧结果改为passed。协议experiments/plans/gmp_context_full_20261017_v1/protocol.json，execution_protocol=context_gmp_full_user_authorized_v1，method/data schema沿用fixed_local_context_gmp_v1。两者分别记录执行授权与特征算法，不混淆。

新增入口tools/experimental_gmp_context_full.py和提交tools/hpc_gmp_context_full.sbatch，测试tests/test_gmp_context_full.py。新入口只负责显式授权和源检查点受控导入；旧数学/MLP训练/检索/评测代码按原SHA从第71节冻结快照复制至新快照，src/configs及原始产物不改。640张旧fit/holdout图片仅复用污染索引0的干净三分支向量，核原receipt/NPZ SHA、来源binding、图片和成员类别/sample_id顺序；新版本另存，不覆盖源。其它训练/验证图仍读现成FP32 token，不执行SAM3前向。

四组为第70节已验收SMFA、原GMP，以及新训练的简单外部均值扣除、新约束GMP两组匹配MLP。参数不变：beta=.1/rho=.1、外部(14,42]像素环带排除其它前景、少于8支持退原GMP、生产最终单相位L2。原34880成员/sample_id顺序、不再次过滤、8466/2116图片划分、MLP2 1280/512/256、v1 seed42 50×100/CE+.5SupCon、中心.2及原HCR/NMS/提议清理固定。不加R3、几何补全、参数扫描或额外seed。

复用第71节8项数学/签名及真实探针凭据；仅新增4项受控导入测试，并用一train缓存对比原GPU聚合、两val原baseline像素回放和相同的短adapter保存加载联通。通过后四卡full_extract、两个单卡train array并行50轮和1449预测，再debug CPU8/23000M统一评测。每GPU一个worker，仅Slurm计算；分配GPU检查空闲，真实GPFS余量至少5GiB，预计新增8GiB。CLI沿用--preflight-only/--probe-only/--resume，逐图检查点校验后复用，训练无完整epoch恢复时新attempt完整50轮，SIGTERM仅停止自己的worker与训练子进程。

storage/gmp_context/<版本>放新库/查询/预测，weights/sam3_adapter/<版本>放权重，experiments/<版本>放参数/阶段/指标，logs/gmp-context-full-%A_%a.out为调度日志，代码及提交审计放hpc_launch_audits/<版本>。四组全部1449预测SHA提交后才作新GT评测，原A/B总及逐图矩阵须精确复现，输出21IoU、mIoU、相对GMP和SMFA百分点、FP/FN/错类、路径计数、支持/残差/耗时峰值、最佳epoch和2k配对图片bootstrap。当前只授权这一批两个新完整训练，完成后报告，不自动部署或追加三seed。结束邮件沿用私有SMTP三附件与独立防重凭据。当前准备完成，JobID和实际启动验收另行追加。

第72节提交与完成（2026-10-17）：technical3710292、full_extract3710293 array0-3%4、train3710294 array0-1%2、finalize3710295全部COMPLETED 0:0。manifest SHA49fd9eb0a62b845ad7c2c8b8a848add2cb45f60b92dc98c8006d7e0f238ff9a5，binding480351ab2492a3a4474cf8e3890b286b24edce8f7afd88ad671db4a62415cd70。技术探针4项新增测试通过，旧640图中一例导入与实际GPU重新聚合bitwise相同，max_abs=0；两图原baseline回放和短adapter保存加载通过，原8项数学测试按冻结凭据复用。

北京时间12:04:06至13:41:39完成，完整墙钟5853秒（1小时37分33秒），不累加并行时间；技术探针3:27，四卡池化关键路径56:55，双卡训练/导出/预测关键路径33:41，CPU汇总3:30。四卡各自完成对应10582训练图/1449验证图分片；640张旧图仅复用无污染切片，全部来源绑定和成员顺序通过。简单扣除/新约束MLP均完成50轮×100步，best epoch分别23/19，内部macro最佳分别0.98328716/0.98351649。best epoch较早不代表训练提前中断。两个训练子进程峰值分配显存均45.5073MiB，不当作包含池化/预测的全流程峰值。

| 方法 | mIoU | 对原GMP百分点 | 对SMFA百分点 | 背景FP | 前景FN | 前景错类 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原GMP主线 | 86.821717% | 0 | +0.140692 | 2411611 | 4359445 | 486861 |
| SMFA | 86.681025% | -0.140692 | 0 | 2400411 | 4328617 | 535924 |
| 简单上下文扣除 | 86.564911% | -0.256806 | -0.116114 | 2566435 | 4400803 | 422486 |
| 新约束GMP | 86.010575% | -0.811142 | -0.670450 | 2419169 | 4559536 | 602216 |

两新方法均没有表现出最终分割提升；新约束对原GMP增加7558个背景FP、200091个前景FN、115355个前景错类像素。简单扣除增加154824个背景FP和41358个前景FN，但减少64375个前景错类。新约束对原GMP的逐类变化中，sheep -5.0892、bus -3.3820、chair -3.1290、cow -2.9073点；car +1.4633、sofa +0.3762、tvmonitor +0.3459点。21类完整表在analysis_report.md和CSV，不只报告有利类别。

2000次配对图片bootstrap对原GMP95%区间：简单扣除[-0.861012,+0.351088]点，新约束[-1.826479,+0.073049]点；新约束对SMFA[-1.600815,+0.035898]点。区间包含0，这是seed42已查看VOC开发集，不代表训练seed方差，也不能宣称普遍显著下降。但当前固定配方没有收益，未达到SMFA+.6点目标，按计划结束，不自动扫参数、加seed或部署。此前原始特征代理+2.28点没有转化为最终分割收益；有关上下文抑制/分布或HCR分数变化的机制解释仍是假设，未做新的消融证明。

四组各1449张，全部预测SHA提交先于本轮GT评测；逐图矩阵求和、原GMP/SMFA历史总及逐图矩阵精确一致。两组34880成员身份/顺序、原8466/2116图片划分、匹配原库/MLP/导出库签名、模型与导出产物SHA通过；原始新库SHA也独立记录。旧complete_no_signal和旧门槛保留；当前新授权完整实验状态complete。无残留worker锁或失败作业。

本地结果experiments/hpc_results/gmp_context_full_20261017_120500/verified_20261017，含summary.json、metrics.json、per_class.csv、analysis_report.md、完成核验和通知凭据。三结果SHA为27cff61b3d80cbe716cf8693aae24895e02afd1ad6c768f8dab0a5d9b67a3388、857b99252606050921a20b717b0d840c36535c2cfe820c3e1e52afcf70912bf4、47313d9fef9bf3556ca62c2a4dc7d72076b19c05cb631bf4c05f1d8e04eb0325；completion_verification SHA346e5461364cb1f1363942e8463c5b64e11cf057f9b551e9bdb0873e21ce0345。结果三附件邮件于北京时间14:56:42被QQ SMTP首次接受，不确认收件箱投递，notification_relay SHA377cde9f6d5cb8d3a7ab3b20fac8db6804bfdbda77270028da70451d719c5934。停止该配方并删除gmp自动监测，不继续追加实验。

## 73. Foreground–Context Relational Representation 三seed clean独立实验（2026-10-18）

用户授权按粘贴方案实现并在集群后台运行，随后明确选择“当前34,880成员＋中心0.2”，替代方案中的旧37,200成员与中心0.4；旧主线和旧指标不改。独立新增src/context_relational_gmp.py、tools/experimental_context_relational_gmp.py、tests/test_context_relational_gmp.py及tools/hpc_context_relational_gmp.sbatch，生产模块和configs不修改。注册文件experiments/plans/context_relational_gmp_v1/protocol.json及plan.md。版本context_relational_20261018_121800，冻结代码和提交凭据在experiments/hpc_launch_audits/<版本>。

原hard-center选择与小目标fallback不变，context为72×72上原foreground选择的8邻域1-token环带；不排除其它对象，不称GT背景。FP32解(Sf+.25Sb+.1trace(Sf)I)z=mean(F)，lambda只取Sf；primal/dual较小Cholesky、相对残差<=1e-4、空context或alpha0回原GMP。原背景成员raw向量保持原库逐元素不变，context为zero/invalid；不重新过滤成员。Relational对象1280维、context均值L2 1280维，object库格式不变，独立sidecar按生产加载顺序绑定row/class/sample/region/mask hash。查询使用相同几何，不做SAM3前向。

A原GMP；B Relational GMP；C原GMP+context重排；D Relational GMP+context重排。seed42/123/2026三次新MLP2 1280→512→256/v1/CE+.5SupCon/50×100，原8466/2116图片划分；B/D每seed共享一个MLP，A/C复用原三个模型。检索保持K25/topm5/search2/中心.2，仅原Top2都为前景、query context有效且margin<.1时辅助；候选类别的邻居必须为object similarity同类top5，再去掉无效context，不补取其它邻居，任一类有效集合为空则跳过。d_final=d_obj+.1*clip((.1-margin)/.1,0,1)*d_ctx，仅允许Top2交换。翻转后保留原object score和margin用于原候选筛选、NMS与像素排序，避免再改变置信分口径。首批仅clean，不执行HCR实验、不添加R3或几何补全。

中心.2的seed42 clean精确复用第70节no_hcr历史预测/总及逐图矩阵；123/2026当前缺中心.2历史clean，建立匹配新clean参考，每图独立调用冻结原生产检索并核对合成像素。不与旧中心.4 clean混比，也不宣称不存在的历史逐图精确性。所有12×1449预测提交SHA后才解码GT，训练和预测仅用原伪标签及冻结token。事后诊断候选翻转、GT多数类/纯度、错误→正确/正确→错误/错→另错、不可测GT、margin区间及逐类；context净纠正像素分别比较C-A与D-B。池化保存Nfg/Nctx、Sf/Sb trace、alpha能量比、cos(native,relational)，按类汇总。完整21IoU/FP/FN/错类、三seed均值/样本std/配对差、2000配对图片bootstrap都报告，不只挑最好seed。

若C或D三seed均值高于A，按注册触发对应方法的same-class/cross-class shuffle，query/MLP固定、所有seed共用seed42固定双射，背景不动；same-class禁止self配对，cross-class保证异类，无法构造时显式失败。没有该机制对照不宣称实例级关系有效。没有正均值则记录skipped_no_positive_mean_context_signal，直接复用首批评测，不重复GT/bootstraps。此授权不包含第二阶段HCR、alpha/beta扫描、额外seed或部署。

集群仅12项针对测试及一train/两val真实GPU探针。链为technical单卡→extract四卡分片→merge CPU→train三卡→predict三卡→evaluate CPU→条件shuffle三卡→finalize CPU；Slurm依赖自动推进，电脑离线不影响已提交作业。每GPU一worker、真实GPFS余量>=5GiB；token只在消费时核SHA，预检复用小签名不全扫12031数组。各阶段attempt、逐图完整receipt可在身份一致时复用；训练没有完整epoch恢复，失败保留旧产物，新attempt完整50轮，不伪造续训。新实现更改必须新版本，不改旧快照。产物storage/context_relational/<版本>、storage/raw_banks/voc/<版本>、storage/mlp_banks/voc/<版本>、weights/sam3_adapter/<版本>、experiments/<版本>、logs/context-relational-%A_%a.out。

正式提交、探针和首次worker验收在后续追加；在验收之前不把CPU预检或排队称作全量开始。完成后通过原服务器既有私有SMTP仅转发summary.json/metrics.json/per_class.csv到2858372757@qq.com，独立receipt防重复，SMTP接受不等于收件箱投递。正常运行静默，完成/失败/需用户行动通知。

提交及首次正式启动验收（2026-10-18）：manifest SHA8737afdd7f0a887f03345cb9d0e51a0e089d50469171078d25e6fe7b28a8f579，binding b7e5eb4ae9a9c163b78bdde3d28b39d91a812bd3bd9cccca84fed5e1e2394c10。technical3722742完成0:0，12项测试及上述真实探针通过。extract3722743_0–3已四卡运行；merge3722744、train3722745_0–2、predict3722746_0–2、evaluate3722758、条件shuffle3722759_0–2、finalize3722760均已依赖提交，不重复。原evaluate debug请求因QOSMaxSubmitJobPerUserLimit拒绝且没有JobID，只对该未提交请求和后续finalize改用v100g32单GPU资源配额6CPU24GiB执行CPU统计（CUDA隐藏）；方法/参数/冻结源码不变，旧失败请求保留submission_resource_fallback_001.json，不把调度拒绝当训练失败。

首次验收读取四个worker对应Slurm作业下/proc实际命令，核四份首批NPZ内容SHA、FP32[N,1280]、固定sample/class、有效context归一化、背景context无效；四分片进度250/250/75/75共650训练图片缓存。startup_validation.json passed/formal_start_reported=true，SHA1f6ef6d4c4f0238855c19a8e5edaa496e2a16e87add5b0fe2484454d7a7ca1c8。此时尚未开始新50轮训练，不能称作12组结果完成。已建relational-gmp-seed自动任务接续验收，正常阶段不重复启动通知；Slurm依赖已独立提交，本地离线不影响计算。

科学流水线完成（2026-10-18）：全部Slurm作业及各array child均COMPLETED 0:0，三个新模型各训满50轮；A/B/C/D十二组及D的same/cross-class六组，合计18组各1449图完成。squeue已无本批作业，summary.status=complete、final_evaluation完成，不能与旧HCR基线86.8217%混比。本地小结果已下载并逐字节核对completion_acceptance记录的三文件SHA，保存在experiments/hpc_results/context_relational_20261018_121800/verified_20261018。

| 方法 | seed42 | seed123 | seed2026 | 均值 | 对A均值差（百分点） |
|---|---:|---:|---:|---:|---:|
| A 原GMP clean | 85.841489% | 85.558371% | 85.699829% | 85.699896% | 0 |
| B Relational GMP | 85.813308% | 85.618101% | 85.999881% | 85.810430% | +0.110534 |
| C 原GMP+context重排 | 85.729824% | 85.595633% | 85.699829% | 85.675095% | -0.024801 |
| D Relational GMP+context重排 | 85.734852% | 85.618101% | 85.999881% | 85.784278% | +0.084381 |
| D同类context打乱 | 85.712832% | 85.656825% | 86.100209% | 85.823289% | +0.123392 |
| D异类context打乱 | 85.651465% | 85.536571% | 85.923630% | 85.703888% | +0.003992 |

B相对A逐seed为-0.028181/+0.059730/+0.300052点，2/3为正，但均值主要由seed2026贡献；B/D各seed图片bootstrap区间都含0，不宣称稳定提升。context重排C-A均值-0.024801点，D-B均值-0.026152点；真实配对未优于同类打乱，不支持实例级foreground-context绑定有效。D正均值触发的既定shuffle已做完，没有再加参数/seed/HCR或部署。

C跨三个seed只翻转3个候选：纠正1个、误伤2个，净纠正像素-29066；D只翻转1个且为误伤，净纠正像素-28381，全部在原margin 0–0.02。C的像素变化按C-A，D按D-B，不能把D-A的表示变化归到context辅助。21类均值、逐seedIoU/FP/FN/错类及bootstrap详见analysis_report.md、metrics.json和per_class.csv。

三结果SHA：summary a07d96cb755fa083efb0d98712d3de0b464dbdfbdde56aa55be5c4553b33aacc；metrics 9ce2e1e370f257f1c0566e919ec03e55ebe29a19bc8a8ae34f57c3cbd6ae1d5d；CSV bf251fe1828257fb8e8a8f20e86b23f9d0e3ca27bb0382e7b14d03d73bfa7183。独立最终模型/sidecar/矩阵验收仅消费结果和小库，不重读token或GT；其凭据完成后另记，不因验收追加研究计算。

详细邮件尚未转发：原服务器183.175.12.79:9470的现有本地SSH认证不可复用，BatchMode返回Permission denied (publickey,password)；没有读取或修改私有SMTP配置，也未发起SMTP尝试。独立notification_relay.json记blocked_ssh_authentication/smtp_attempts=0，不把Slurm邮件或awaitingrelay当投递成功。实验结果不受邮件失败影响。

最终独立验收及归档完成（2026-10-18）：completion_verification.json status=passed，SHA38e67d0b3fc3bc17a9b20adf6b8d55c15dea98fa90ce35bb7f8bf16d21847227，已下载到上述本地验收目录并核对SHA。核验三模型各50轮5000步、实际CE+.5SupCon损失、原图片划分、34880成员身份/顺序、raw/MLP库与adapter匹配签名、全部模型/库/sidecar/矩阵SHA；原背景raw逐元素不变，sidecar的row/class/sample/region/mask摘要绑定、有效context归一化与same/cross-class双射均通过。18组逐图矩阵和总矩阵一致，seed42 clean历史精确，所有对应预测提交先于该轮GT；17个实际作业均COMPLETED 0:0，无残留worker锁，不重读token或GT、不重复推理。

正式Slurm链从北京时间12:22:16至13:42:15，整体墙钟4799秒（1小时19分59秒），不累加并行任务时间。三个seed42/123/2026的最佳epoch分别38/17/45，内部macro accuracy分别0.9848401777/0.9853858816/0.9852822301；三者实际均训满50轮，不按VOC val选epoch。主结果、shuffle和邮件认证阻塞已向用户报告；最终验收后结束relational-gmp-seed收尾监测，不追加实验或无凭据邮件重试。

## 74. Relational GMP稳定性与机制对照（2026-10-18）

用户新方案见`experiments/plans/relational_gmp_mechanism_v1/plan.md`和`protocol.json`。只研究表示，不继续Context Retrieval/Rerank、paired-context分类、HCR联动或参数扫描。独立入口`tools/experimental_relational_gmp_mechanism.py`，几何/随机donor规则`src/relational_gmp_mechanism.py`，8项针对测试`tests/test_relational_gmp_mechanism.py`，Slurm入口`tools/hpc_relational_gmp_mechanism.sbatch`。不改生产模块和第73节冻结快照。

固定34880成员、原sample_id/顺序、8466/2116图片划分、rho.1、alpha.25、8邻域1-token ring、FP32、MLP1280/512/256、v1 CE+.5SupCon、50×100、单库center.2、clean only。新增Original/Real各seed7/3407共4次，Distant/Random各seed42/123/2026共6次；历史6组Original/Real只读复用模型、预测和逐图矩阵，不重新训练。候选、背景10569条、SAM3/token缓存均沿用。

Distant按距前景token集合的Euclidean距离降序取原Nb个，row-major打破平局，严格排除前景和local ring。训练侧已发现24个区域几乎铺满整图，剩余不同token不足，部分为0，不能同时满足严格等量与两个不重叠约束。已向用户提出仅这些例外显式回退原GMP的澄清问题；在用户回答前，不运行受影响的完整Distant池化。技术探针仅选满足等量条件的真实图；稳定性4次不依赖此决策，可以先运行。例外策略须用独立`distant_policy_authorization.json`绑定manifest与原始答复，不能默默重复token或侵入ring。

Random的donor严格来自内部fit训练图片，不来自内部留出或VOC val；对所有正Nb候选建立不读类别的sample目录，default_rng42固定排列，用SHA256(phase/image/mask_hash)取起始位置，循环寻找另一图片且Nb足够者，row-major截取。所有MLP seed共享映射，train/query定义相同。先从已有token一次性保存各fit donor ring的FP32值，后续按SHA消费小ring缓存，避免每个query重复读取整幅donor token。完整provenance保留图片、sample、mask摘要和token indices。

目标版本`relational_mechanism_20261018_160500`，新产物在storage/relational_mechanism、storage/raw_banks/voc、storage/mlp_banks/voc、weights/sam3_adapter、experiments下按版本分离；日志`logs/relational-mechanism-%A_%a.out`。冻结登记和实际JobID以`experiments/hpc_launch_audits/<版本>/jobs.json`为准。一次性prepare不得重复；先8项小测试、一train/两val真实池化、两图历史Original/Real精确回放、4种签名短GPU保存加载，然后正式。已有小完成凭据复用，不预扫12031个token数组。

计划调度：technical→四分片donor缓存；technical→4任务稳定性训练/预测；donor完成后四分片control池化→merge→6任务control训练/预测；全部新1449预测SHA提交后统一CPU GT评测。完整control池化在例外规则澄清前以Slurm hold保护，不影响独立稳定性任务。CPU统计用v100g32单GPU资源配额6CPU24GiB、CUDA隐藏，避免debug提交数限制。每GPU一worker、真实GPFS保留至少5GiB；正常阶段不重复通知。

报告5seed Original/Real均值、样本std和配对方向，3seed Real/Distant/Random机制表，21IoU、FP/FN/前景错类、2000共享图片bootstrap、pottedplant新seed、去掉最大收益类别的诊断、Nfg/Nctx/trace/能量比/cos及新control的rf/rb。历史Real未记录可选rf/rb，保留空值并说明，不为补统计重读大token。固定成功口径为稳定性4/5正且均值正，locality需多数seed优于两个control；图片bootstrap不代替seed方差。无论结果如何，本批完成即停止，不自动追加实验或部署。

详细邮件仍受第73节原服务器SSH认证阻塞，未恢复授权前不反复重试，不读取或修改私有SMTP配置；Slurm END/FAIL请求不作为投递证明。正式提交及启动验收后续追加；代码已实现不等于实验已运行或完成。

已提交初版`relational_mechanism_20261018_160500`：manifest SHAceb7e8303f7a906d3cfb253ae27549518836665745353260ebb7c58a25a25510，binding5f6f0ca3f8c124ab65cbf5cd1188105eda190c4425db6820e79046b46c7f9368。technical3724333通过8项测试、一train两val真实池化、两图历史Original/Real精确回放与四签名GPU保存加载并成功退出。donors3724334四分片、stability3724335四任务随后实际运行。首次验收逐节点/proc核对真实命令、四训练epoch日志已至21/22附近，四首批donor NPZ的内容SHA、FP32[N,1280]、sample/图片/索引、每ring无重复均通过，首份形状256/132/508/202×1280。startup_validation.json passed/formal_start_reported=true，SHA c73b3d64b6454c059bcb2f6a6d654cc0f19d26aca4c564de1bd270dce6d067b5，已下载本地audit目录。已报告4次稳定性+4个donor缓存worker启动，不再重复；这不表示10次训练已全部开始或50轮已完成。

Random边界修复使用独立`relational_mechanism_control_recover_20261018_161500`：唯一最大ring的sample `2011_002661_000000`请求Nb935，但另一内部fit图片可用最大值是836，donor为`2009_000575_000003`。先前估计871不在内部fit，不能使用。按用户Random“尽量匹配”口径，只有遍历完整目录仍找不到足够大其它图片donor时，使用最大可用ring、catalog顺序打破平局，显式记935→836，无重复token、不改F/μ/Sf/λ。其余精确匹配。新manifest SHA4aa34e00c2735c794ca5e6cb536565d00b75f1f4577fe0f76f20d85a2f58d8e6，binding3238a4c56f464008eab2bc1f7a24877e01e622d7baa21f5f2cf2add38647675b。冻结code不覆盖；原始版四稳定性train/predict及四donor阶段按原binding+manifest+catalog digest只读导入，绝不重训或重新生成。新technical只需9项测试及这个最大ring实际GPU探针/新Random签名，复用初版已通过的其它真实检查。

恢复准备最初的871固定断言在CPU阶段失败，当时无manifest/jobs/GPU，已保留prepare_failed_before_registration_001.py；登记前改为冻结fit目录计算的实际数量，算法不依赖该错误估计。恢复准备核实PENDING后取消原先尚未运行的旧extract3724336/merge3724337/controls3724338/evaluate3724339，保留历史；technical/donors/stability不动。恢复版首个sbatch请求因旧technical已完成且从Slurm活动依赖表过期而返回Job dependency problem，没有JobID，不是GPU科研失败。冻结代码、参数及输入未变，新的调度重试脚本只在旧父任务仍活动时保留afterok；已过期成功任务必须用精确完整阶段凭据确认后省略该无效调度边。失败请求留在原jobs.json，实际新控制JobID以最新jobs_retry_001.json为准；重试正在进行时不可重复提交。

Distant的24个例外仍等待已发出的用户选择。未得到明确答复前不创建授权、不释放最新extract，不把沉默视同意。推荐选项确认后，只在最新恢复audit写distant_policy_authorization.json，绑定最新binding、policy=native_gmp_explicit、explicit_user_reply=true及原答复，再release最新held extract并保存凭据；不能释放已取消旧3724336。若选择重复补足方案，须新版本针对性实现，不改旧快照；原4次稳定性与合法donor仍可复用。监测automation为`relational-gmp`，正常运行不重复更新，完结后删除。大token和GT不参与收尾预检。

调度重试已全部提交并下载核对`jobs_retry_001.json`：新technical3724626（依赖仍活动的donors3724334），extract3724627四分片held，merge3724628，controls3724629 array4–9%6，evaluate3724630（依赖controls和仍活动的stability3724335）。原过期technical边以已验收完成凭据替代，未重跑它。冻结算法/manifest不变；独立planning_correction_001.json记Random实际935→836与早期计划文字871的差别，不修改旧快照。所有prepare/controller已返回，不再重复调用。接下来仅用户明确选择推荐例外策略后才能写新授权并release3724627；Slurm仍会自动等新technical成功，电脑离线不影响已经提交的合法作业。

用户例外策略已确认（2026-10-18）：通过原async问题明确选择“不足时回退原GMP（推荐）”。这取代上文待确认状态；24个不可行训练区域使用原GMP、保留成员并逐例记录，其余区域严格等量且排除F与local ring。操作脚本`audit/authorize_release_001.py`只核冻结manifest、现有JobID3724627的owner/name/command后登记独立授权和release，不重提任何实验。以最新audit的`distant_policy_authorization.json`及`distant_release_001.json`为正式执行凭据，不改冻结code/manifest。后续无需再向用户询问该策略。另已实测stability3724335四child均COMPLETED0:0，四模型训练及各1449预测完成；统一GT评测仍按全部10组预测提交后执行。

策略已登记并完成放行：`scontrol release 3724627`返回0，解除JobHeldUser，后置阶段继续由既有afterok依赖保护；没有新sbatch或训练重提交。授权receipt SHA7550feb31a54a51c8ef801123436ba782ba7c14acbea82ae44c3d5dad76b55c7；释放receipt SHA59e1bdf6325581481425879ba266652f44152b73ad153aa0461de877631f03c3。后续仍按donors→technical3724626→extract3724627→merge3724628→controls3724629→evaluate3724630自动推进，不再因例外规则等待用户。放行不等于六模型已经开始训练；先等缓存及技术探针。


Relational机制研究最终结果（2026-10-18 21:19完成）：最新恢复版本`relational_mechanism_control_recover_20261018_161500`已complete，22个实际作业/child全部COMPLETED0:0，两版本无锁。10个新模型均50轮5000步，10组各1449预测全部SHA提交后才GT；历史六组逐图矩阵精确复用。独立小结果验收核16组矩阵SHA、逐图求和、mIoU、10份训练报告/固定划分，不扫描token或模型大库。

五seed Original均值85.669823%（样本SD0.116689点），Real85.737870%（SD0.169875点），平均+0.068047点，仅3/5正向，未达4/5稳定性门槛。新增seed7差−0.096410点，seed3407差+0.105042点。三seed机制均值Original85.699896%、Real85.810430%、Distant85.713851%、Random85.740634%；Real比Distant+0.096579、比Random+0.069795点，各2/3 seed胜出；相关2000次图片bootstrap区间均跨0，不能据此宣称充分证明局部关系。最大收益类pottedplant五seed+1.077530点，新两个seed均0；去掉此类其余20类平均仅+0.017572点。只做clean、中心.2，不能混比旧HCR86.8217%。

24个Distant训练区域按用户明确授权回原GMP，验证0例外；Random唯一935→836记真实donor。区域明细115467行（Real/Distant/Random各38489）；历史Real可选rf/rb缺失保持空值，未重扫token补填。最终21类、FP/FN/错类、逐seedCI、最佳epoch、机制统计及耗时见本地`experiments/hpc_results/relational_mechanism_control_recover_20261018_161500/verified_20261018_2125/analysis_report.md`。大CSV及context_mapping仅保留HPC，已核小结果下载SHA。

墙钟跨度15:58:51–21:19:09共5小时20分18秒；donor最长58分37、池化最长2小时57分04、合并32分15、最终评测18分48；并行阶段不累加。池化torch峰值最大156.899MiB，训练GPU峰值未记录，不以主机内存代替。completion_verification SHA5d2f39f60c08fab4c5a92ff9a651e16fd7822e8b1e34899fcfe72c7bee1353e4；独立completion_acceptance SHA851cd06e98679856b45ea1db462b30404edf094b69709e0271acd9426f723aea。summary/metrics/per_class SHA分别63b046fa38f44efcfac3fd68f9e27adf8fb9a233bd9a774cc1cdb8778634a3bd、4dfbdfde736b77a3486fdb5f52e044e3871a2c3698d17168ab29367ea40d262f、c608c59a7d7e656cccb898e04df77addd46de6d024fc31745762b2e2b7c9f305。原服务器SSH仍阻塞，notification_relay记录SMTP0次/未发送，不读取私有配置、不无凭据重试；不影响实验。固定本批结束，不追加实验或部署；完成归档后删除relational-gmp监测。

## 75. Retrieval-Guided Patch Carving：先验证patch可分性（2026-10-18）

用户完整新方案保存在`experiments/plans/retrieval_patch_carving_v1/request.md`，执行计划和固定规则见同目录`plan.md`/`protocol.json`。先执行独立诊断版本`retrieval_patch_carving_diag_20261018_225913`，新增`src/retrieval_patch_carving.py`、`tools/experimental_retrieval_patch_carving.py`、`tests/test_retrieval_patch_carving.py`、`tools/hpc_retrieval_patch_carving.sbatch`；不改生产模块、configs、原库、模型、旧快照。诊断阶段代码不实现mask修改。

固定当前34880成员、原GMP/原MLP seed42、center.2、clean、HCR关闭、1008/FP32。用原始清理/NMS后所有前景proposal，逐图原输出必须与第73节已验收A clean像素精确一致。72×72的stride14/center7按原图floor映射，严格中心在mask内，不借用mask外fallback。原256维query决定类别后，找该类真实Top5 projected行，以class/sample/row精确映回1280维raw；归一化各行后求均值再L2得前景参考。背景使用原库全部10569条raw向量，GPU批量矩阵乘法取Top5均值；记录s_fg/s_bg/d与提议、token索引、邻居身份。mask/token原数据以只读SHA引用保留，不再复制整图token。

先提交1449张原输出与patch分数并核全部内容SHA，再读取GT做诊断。GT=预测类、GT背景、其它前景、ignore255四类分开，前两类计算AUROC/AUPRC；报告20类及整体分布、前景比例和自行车d最低/最高各20处可视化。统计单位proposal/patch，重叠区域不宣称独立图片样本。固定门槛整体AUROC>=.65且bicycle>=.70，任一缺失或不足即停止，不实现carving、不扫kappa/阈值/TopK。VOC已看过，GT只作明确授权的开发诊断和后续资源门槛，不构成独立泛化证据。

只有通过后才用新版本实现正式RPC：score>=.95、margin>=.1、median−2.5MAD、sb>sf、4连通至少2patch、排除接触mask外部组件、双线性72→1008→原图/.5阈值、只删原mask内部、原图删去面积>.2时整提议回退。保留原NMS/分数/类别，原NMS后最终raster前carve，不再填孔或重算特征。原seed42/123/2026三模型各clean/RPC两组1449，0训练，不开HCR。记录提议挖孔与最终像素因遮挡重合而不同的变化，完整IoU/FP/FN/错类、删除像素GT分解、2000配对图片bootstrap及有益/有害可视化。当前尚未执行该条件阶段。

提交链为Slurm单卡technical（10项CPU测试＋两val真实GPU/原预测精确/背景Top5小CPU参考）→四个单卡score分片→CPU诊断（v100g32单GPU配额6CPU24G但隐藏CUDA）。所有研究只在计算节点，每GPU一worker；真实GPFS余量>=5GiB、token仅消费时核SHA，不扫12031缓存预检。代码/参数变更用新冻结版本；失败保留attempt，不能重复prepare/sbatch。提交、实际JobID、SHA及首次正式验收在后续追加。在作业确认前不称已运行。用户已说明邮件不重要，本批不继续SSH/SMTP排障。

实际提交已成功：technical3730846→score3730847（array0–3%4）→diagnose3730848，均afterok衔接。manifest SHA890d1d5e5c6ea8ecf7e2e4b2e7d4cb49edac2326d583dc5a3f36665d6127b605，binding563ec0fc4a49ae5e3fe891312585f26de782e0682d5c21555c3eb77346049370。以audit/jobs.json实际命令和JobID为准，绝不重复prepare/sbatch。提交前CPU预检通过，完整技术GPU探针及正式首批验收仍须随后检查，不把提交成功当1449完成。已登记rpc-patch监测，首阶段失败按固定门槛停止，通过后才启用新版本正式RPC，正常排队运行静默。

首次正式诊断启动已验收（2026-10-18 23:21）：technical3730846 COMPLETED0:0、5分14秒，10项小测试、两val真实原clean像素精确、GPU背景Top5对小CPU参考通过。score实际child3730908/3730909/3730910/3730847，worker3177559/3811745/2525434/2525433，分别在c2v100m2-4/-7/-8/-8；逐节点/proc实际命令、每worker单GPU及四份首批NPZ内容SHA、FP32、d=sf−sb、1280维前景参考单位范数、Top5原raw行/class/sample身份通过，首份patch数813/2929/351/2016。当前progress合计450/1449（225/125/50/50），只是分数生成进度，不是AUROC结果或mask修改完成。startup_validation.json passed/formal_start_reported=true，SHA6de1a05179e2daf006fedf708e1729d7310f3df5d232acd26210a3c4d665c7dd，已下载并核SHA；registration_acceptance额外核四个节点/GPU对应不重复。GPFS实测余318814601216字节。diagnose3730848仍等待所有score完成；今后正常阶段不重复启动通知。

四个worker的CUDA_VISIBLE_DEVICES都显示0，这是集群每作业GPU隔离命名空间，不能把两个同节点的local0误判为共用物理GPU。已另核原allocation记录：host分配0/5/3/4，四个NVML UUID互异；后三个的CUDA/NVML UUID一致和job cgroup限制均由冻结allocation helper实测确认。GPU背景Top5与小CPU参考最大误差1.1920928955078125e−7。审计脚本verify_startup_002支持分次保存各首worker读取，避免较快分片退出后无法读取旧/proc；只读小产物，不修改冻结实验代码。

第一阶段已于2026-10-18 23:39:35完整结束，次日收尾核验：technical、四个score child及diagnose六个实际作业均COMPLETED0:0，本批squeue无作业、无残留锁或失败记录。1449图、3418个NMS后前景proposal、1965964个proposal/patch观测完整；原clean输出1449图历史像素精确，分数/预测全SHA提交在GT读取之前。阶段只做诊断，mask修改0、训练0、SAM3前向0。墙钟23:10:58–23:39:35共28分37秒，不累加并行秒数；四分片torch峰值最大297.139MiB。

固定门槛未通过，状态complete_no_patch_separation：整体AUROC0.6434111553<.65，bicycle0.5596588414<.70。整体AUPRC0.9766425723，前景比例基准已0.9664140330；不能把高AUPRC解释成可靠挖背景。自行车前景22374patch、内部背景16998patch，中位d分别+.0657844/+.0542510，分布明显重叠；其它前景155、ignore17322独立统计。chair/motorbike/pottedplant的AUROC分别.6308760/.6025230/.4613151。完整20类分布、AUROC/AUPRC和各20张极值定位已保留。结论限于该seed42、1280维raw patch与region参考的固定配方和已查看VOC开发集；上下文/尺度错配解释仍是假设。按用户原停止规则结束，不实现正式carving，不运行条件三seed，也不扫描参数；没有RPC后的mIoU或删除像素精度结果。

独立verify_completion_001只读冻结代码、小索引、SHA提交凭据、三结果与40张PNG，未重扫token/raw库/GT或重跑推理；核四索引不重不漏、图片/提议/patch及四类统计求和、CSV/JSON一致、固定门槛、40张极值、Slurm完成状态。completion_verification SHAd07e9e0683908d70829528ea0d433317e80c4ad079ab9aff9f3485a9c7285d47；独立completion_acceptance_001 SHA68c982c7e3d0edce22f7779d115d0461af4cc9b26c9162fe23d5060238cac4cd。summary/metrics/per_class SHA分别b257f84ede2c6a35ae14ff8f3bfe719552cbc12fda2111542ee008c0aa45e49a、a53160b70defa52e461903f9f74270ad16bd91a6fe056ab7876387055862b1b8、d7cc342b0217af45e3be1f6e8b3309d3b424de3e1c8af7a93fa675897dbb9aaf。本地experiments/hpc_results/retrieval_patch_carving_diag_20261018_225913/verified_20261019已核全部下载SHA，含analysis_report.md、三结果/验收/提交凭据及visualizations的40张PNG。结果已通知用户，rpc-patch自动监测已删除，不再追加实验或邮件排障。

## 76. RPC patch参照与图内背景诊断（2026-10-19）

用户在第75节失败后接受新想法并明确要求“按照你的想法安排实验”。新独立版本`rpc_patch_reference_20261019_014000`，计划`experiments/plans/rpc_patch_reference_v1/plan.md`和`protocol.json`。新增`src/rpc_patch_reference.py`、`tools/experimental_rpc_patch_reference.py`、`tests/test_rpc_patch_reference.py`及Slurm脚本`tools/hpc_rpc_patch_reference.sbatch`。原RPC v1结果和停止决定保留，不改生产src/configs、原34880成员、模型或旧快照；本批只分析分数，不实现mask carving。

两项研究并行：一项复用旧1449完整已提交分数，做逐proposal AUROC、至少5FG/5BG的混合支持统计、每图macro，以及各proposal中位数居中/MAD归一化后的pooled分数诊断；此事后分析不重启旧配方。另一项用SHA256("pilot:42:"+image_id)固定选256张val，原GMP/MLP seed42、center.2、clean/HCR关闭、1008/FP32，完全复用旧类别/score/NMS/原clean预测和原Top5训练成员。所有方法使用相同提议和patch，A分数必须逐元素与旧缓存相等。

A原GMP前景参照＋原10569背景区域向量；B把前景换为Top5训练成员内实际patch；C再把背景换成训练背景patch；D再使用同图局部预测背景patch。前景每member最多32，center在原mask且原图cell覆盖>=.90；背景每FIT图最多64、伪标签0覆盖>=.95。全局BG参照固定10569条，先取SHA固定256张FIT，有效支持不足时才按同一hash顺序补到最多512张，不重复token、不扫阈值。mask必须与原row/class/sample/region及已验收sidecar的mask hash相同；仅部分读取原sidecar的身份字段并核独立paired摘要，不重读其context大向量。原pseudo-fallback来源及不在批准pseudo_labels树下的输入不进入新patch参照，Top5对应项跳过不补取，原主库成员不变。

D局部候选在原mask center-grid外8邻域3token环、原clean预测背景cell覆盖>=.95，最多64，少于8则回退C；这仍是背景候选，不是真GT。若五训练成员合计前景支持不足5patch，B/C/D整提议回退完整A分数。固定随机映射，无类别例外。辅助A_pairwise用原5个raw GMP各自cosine均值，区分均值归一化影响；C_matched用与每proposal局部BG相同数量的全局patch，区分局部信息与字典规模。所有辅助均不增加SAM3前向或训练。

全256四组score/元数据/复用baseline内容SHA提交后才GT；旧1449分析只复用旧已验收全量commit。预测类FG/内部BG/其它FG/ignore分开，报告各类pooled AUROC/AP及前景比例、分布、逐proposal AUROC/每图macro、共同覆盖子集、回退/排除数量与同图配对2000bootstrap（对象为每图within-proposal AUROC宏平均差，不冒充pooled AUROC CI）。固定score>=.95/margin>=.1/median−2.5MAD/sb>sf只做候选patch误筛风险，不执行连通域、像素挖孔或报告新mIoU。潜在正向方法须同256上整体>=.65、自行车>=.70、自行车比A>=+.03且整体不降，前景支持全体/自行车均>=80%；D还需两者局部覆盖>=50%。缺失失败，所有方法完整报告，不按GT调参数。本固定批诊断完成即止，后续完整诊断或carving须新版本登记，不自动加HCR/三seed。

13项针对测试已本地通过。Slurm链technical单卡真实训练参照＋两val旧A/clean精确回放及新GPU Top5对小CPU参考→4卡提取必要训练参照→CPU合并→4卡256图评分→CPU汇总；旧1449逐proposal CPU分析在technical后独立并行，汇总同时等两支。CPU阶段用v100g32单卡资源配额6CPU24G但隐藏CUDA；每研究GPU最多一worker。仅消费所需token时核SHA，不预扫12031数组，不训练MLP、不运行SAM3。实际JobID/manifest SHA及首次worker验收后续追加；在真实启动前不把本地测试或提交叫作正式完成。失败保留attempt，源码修复用新版本，逐图相同签名可安全复用；用户取消立即停止自己的任务，不自动恢复。只归档小结果与验收，用户已说邮件不重要，不做SMTP排障。

2026-10-19已提交六阶段依赖链，实际记录`experiments/hpc_launch_audits/rpc_patch_reference_20261019_014000/jobs.json`：technical3731699、extract3731700（array0–3%4）、within3731701、merge3731702、score3731703（array0–3%4）、finalize3731704。manifest内容SHA`5020efa1ffbcf7e6957b5c50553fdca8a4f60c87c74034c675c3c4e163e2e7fa`，binding`2c0f4204bd116edc0318531f30e558807703b57a866001edfbe90d7fb3f806d6`。预检真实GPFS余318798069760字节，256 FIT图有16384个合法BG候选；实际需2596训练图片/2774前景成员，六个来源不合格的Top5参照排除不补取，最终全局BG仍固定10569。预检未读token、密集GT或sidecar大context；约75.7GB是正式计算实际所需缓存读取量估算，并非额外全库扫描。集群13项测试已通过，真实GPU探针及四worker首次产物仍待验收。监测自动任务`rpc-reference`已创建，正常排队运行静默，诊断完成停止并删除监测，不自动进入carving。独立`verify_startup_001.py`位于audit外层，不属于冻结算法snapshot；正式启动凭据通过后只报告一次。

正式启动已验收：technical3731699为COMPLETED0:0、3分51秒；集群13项测试、11张实际训练参照图及两val原clean逐像素回放通过。重算A最大绝对误差1.7881393432617188e-7，新FG Top5 GPU对CPU误差0、BG误差2.9802322387695312e-8。extract四实际child为3731728/3731729/3731730/3731700，PID2107035/1452976/1196786/1196784；核实际冻结脚本/manifest/shard命令、四份首批NPZ内容SHA、FP32 Nx1280/单位范数/唯一token/sample-class-mask绑定及四个不同node+GPU UUID。`startup_validation.json`为passed/formal_start_reported=true，SHA`7e13b5a5c9555eff9ef4653a49dca33b537a707d6eb730c4c91f36e00101ce59`，已下载核对，勿重报启动。within3731701的旧1449事后分析角色已complete，原整体及自行车AUROC精确复现断言通过；新256评分尚待正式提取和合并完成，不把within完成当新四组完成。后续仅完成、失败或需行动通知。

本批已于2026-10-19 02:44:55完成，起始01:57:28，整体墙钟47分27秒，不累加并行秒数。12个实际Slurm作业（含8个array child）全部COMPLETED0:0、squeue为空，无失败或残留锁。旧1449分析和新256图/656提议/349686个proposal-patch观测完整；全256 score/复用baseline SHA在新GT解码前提交，A原缓存全图逐元素相等。所有foreground支持656/656，自行车23/23；local支持643/656，自行车23/23。失败并非支持不足。global BG字典10569条唯一patch、源FIT/排除/身份及四辅助参照对齐已验收。

同256上pooled AUROC整体/自行车：A .6466313631/.5890463185，B .6023711899/.4783364716，C .6319173921/.4774037578，D .5454858295/.4265361288。A_pairwise .6740560287/.5502296264，整体改善而自行车更差；数量匹配C_matched .5645828520/.4999123114。B/C/D均未通过预登记门槛，promising_cases=[]，next_step=stop_no_reference_signal。本固定批停止，不进入全量新参照/carving/三seed/HCR，不扫描参数。固定筛选规则下B/C/D候选patch真背景比例仅2.50%/1.82%/3.24%（有效候选排除ignore），不是实际删除像素precision；mask修改0、训练0、SAM3前向0，不报告新mIoU。

新256中有51个提议/44张图满足至少5FG和5BG。每图within-proposal AUROC均值A/B/C/D=.726501/.680976/.672672/.725607；相对A的2000共享图片bootstrap差：B−.045525，CI[−.092044,+.002078]；C−.053829，CI[−.101420,−.003255]；D−.000894，CI[−.057497,+.062885]。这些不是pooled AUROC CI或训练seed方差；D图内排序近似持平，不能笼统声称每个提议内部都更差。旧1449原AUC精确复现.6434111553/.5596588414；median-centered后整体/自行车.5511088121/.5699991687，MAD后.5495788389/.5700435966；原338混合提议/301图的图内宏平均.688092，未形成通过证据。

独立`verify_completion_001.py`在audit外层，只读小结果/全部256新score内容SHA、四分片原A独立抽查、唯一BG身份/源码/提交及Slurm凭据，无GT/token/raw bank读取或研究计算；全部阶段与CSV/JSON求和、历史AUC精确、固定门槛通过验收。`completion_acceptance_001.json` SHA`cbc387ba7cf9d74a22dd15e11b6a458e47c6879886bf5c16b97b09866d0f5e39`；summary/metrics/per_class SHA分别`8bef145fa22a92b70f8ccfc078753cc47ff2fb9c2a86de3c764dcc37293b8acb`、`4a5dec6df45cdb07b028b2ff1473a9cb6857dfdae46d64980919f2414f837905`、`8cac14ace93746582615820a49c10eb9e22414f17a7df501802d30576c47a919`。本地`experiments/hpc_results/rpc_patch_reference_20261019_014000/verified_20261019`已核三结果、within_analysis、提交与验收SHA，另含analysis_report.md/download_verification.json。完成汇报后删除rpc-reference监测，不做邮件排障，不追加实验。结论限于该固定raw patch参照和VOC开发诊断，不否定所有patch表示或GMP。

## 77. RPC受限几何Oracle诊断（2026-10-19）

用户在第76节失败后明确回复“做吧”，授权先检查固定patch删除操作的可修复空间。新版本`rpc_restricted_oracle_20261019_123200`，独立计划`experiments/plans/rpc_restricted_oracle_v1/plan.md`、`protocol.json`，新增`src/rpc_restricted_oracle.py`、`tools/experimental_rpc_restricted_oracle.py`、`tools/hpc_rpc_restricted_oracle.sbatch`、`tests/test_rpc_restricted_oracle.py`。冻结薄快照在本版本audit/code，生产src/configs及旧缓存、模型、指标只读不改。完整1449张，原seed42/中心.2/clean/HCR关闭，重用已验收原NMS后的前景提议顺序/分数/类别/mask，不重新检索或NMS，不读取token、库或模型，不训练或SAM3前向。每图原clean PNG及第73节42/A逐图矩阵必须精确匹配，基线85.8414891527%，不可与86.821717% HCR混比。

GT明确仅用于事后Oracle，原1449预测已在父实验提交SHA后才解码GT；新Oracle输出本身使用GT，绝不称GT-free预测或可部署收益。固定72格、stride14/center7/floor、retrieval>=.95/margin>=.1；用中心GT背景替代不可靠相似度，4连通>=2、接触proposal中心格外部的整个组件拒绝，图像格外也算外部；CPU FP32 torch bilinear72→1008→原图/align_corners=False/严格>.5、交原mask；删除面积超过原mask20%整提议回退。保留原raster优先级及遮挡，不填孔、不改类别。

五个对照：B原基线；pixel_upper把最终可见GT背景误报全部改背景，作为不受网格约束的无前景损伤背景删除上限；grid_budget_upper使用erode4后去singleton的最大合法种子包络，按原可见owner分区的可达GT背景数与floor(.2原mask面积)取min并更新混淆矩阵，放宽隐藏重叠mask成本及网格组件/像素联合选择，是宽松上界而非可实现mask；grid_center执行中心理想GT标签后的真实固定删除算子，记录混合patch和回映的前景损伤；grid_safe额外用GT拒绝映射后触及前景/ignore的组件，union映射仍触及非BG则整提议回退，再守同样20%面积限制，是保守可行对照而非严格最优解。宽松上界只针对无前景损伤的GT背景删除目标。要求pixel_upper>=grid_budget_upper>=grid_safe的逐图及总体mIoU，grid_safe前景GT预测逐像素不变/FN和错类不变。

输出21IoU/mIoU/相对B百分点、FP/FN/错类、提议删除与最终raster删除/露出其它类的区别、BG/FG/ignore分解、组件/面积/安全回退、原总及逐图矩阵/SHA、2000共享配对图片bootstrap、角色与整体墙钟不累加并行。若grid_budget_upper<+.2点则stop_insufficient_grid_headroom；否则grid_safe>=+.2点为fixed_geometry_feasible_evidence_missing；其余relaxed_headroom_only_no_safe_fixed_recipe。宽松上界高不证明实际可实现。本批所有结局均停止，不自动训练/点提示/新decoder/扫描参数或加seed。

16项针对测试本地15通过、1项需Torch在集群核；含重叠提议预算上界的50组随机可行删除验证。先集群16测试+4固定非空真实图CPU探针，再1个debug8CPU/23000M作业内4个CPU rank做1449，随后CPU汇总，CUDA隐藏，无需GPU计算。debug请求如无JobID且因明确QOS限制拒绝，仅可登记resource-only fallback到v100g32单卡配额/隐藏CUDA，不能重提已提交任务。GPU/训练均0，真实GPFS余>=5GiB，小凭据快速核验、实际消费核mask/GT内容SHA，不扫12031token。正常排队运行静默，仅首次正式验收、完成、失败、需行动通知；probe失败取消自己未运行依赖；同版恢复须旧进程退出锁释放/完整代码参数输入产物SHA同，修源码用新版本。用户暂停取消立即停自己作业/监测，不自动恢复。只下载小结果/验收，完成后删除监测，不做邮件排障。实际JobID及签名待提交后追加，不能把本地测试当全量启动。

本批已完成验收，版本manifest SHA`78bd2a50869ca2924b76639b104165bac91f4c5db3cb57a65eed8f2600ccdf22`，binding`663bdd89c5d74e8636244bb75f2c0442b8197928c7a31d40773d29e8852d5304`。最新实际调度记录为audit/jobs_resource_retry_002.json：technical3738008、run3738024、finalize3738038，均debug CPU执行/COMPLETED0:0，分别2分10秒、1分14秒、1分20秒，整体墙钟7分20秒（440秒，包含中间等待，不累加并行秒数），实际GPU分配0。初始debug后续请求受QOSMaxSubmitJobPerUserLimit拒绝无JobID，后备run3738009分配4CPU而冻结srun要求8CPU，已在启动前hold；technical完成后按相同冻结代码/参数/输入重提匹配debug8CPU的3738024，更新现有final依赖并取消旧pending3738009。后备final3738010也未启动，因等待GPU资源配额，在全量完成后仅改回debug8CPU提交3738038并取消旧pending。resource_resubmission_001.json/finalizer_resubmission_002.json保存实际替换凭据；旧jobs.json及零耗时CANCELLED历史保留，不是方法失败，绝不重启旧3738009/3738010。

16项集群测试无skip、4固定图CPU技术探针通过，原PNG/历史矩阵精确、安全与上界检查通过；这四图最终carved0，不能称作已成功挖孔。全量4rank在74秒完成，来不及捕获实时/proc，startup_validation未写passed，不补造现场启动凭据。以四分片索引/进度、完整1449小产物SHA、冻结源码和Slurm完成状态验收；原基线1449逐图及总矩阵精确匹配第73节42/A。五组完整1449，3418提议，2951通过固定score/margin门槛，mask/GT实际消费核SHA；无token/模型/库读、无训练/SAM3前向，GT明确只用于Oracle。

总体mIoU及相对B百分点：B85.841489%；pixel_upper89.548917%（+3.707428点）；grid_budget_upper86.858693%（+1.017204点）；grid_center86.022745%（+0.181256点）；grid_safe85.842025%（+0.000536点）。pixel_upper不受网格/面积约束；grid_budget_upper是针对无前景损伤BG删除的宽松分析上界，不是可实现mask；grid_center是完美中心标签下固定组件/面积规则的结果，不是严格最优上限；grid_safe是额外严格的可行对照，不是最优子集。decision=relaxed_headroom_only_no_safe_fixed_recipe，固定批结束，不自动追加训练/decoder/点提示/扫描参数。

B背景FP2132653、FG FN5020719、FG错类452394。grid_center实际保留119个挖孔提议、22个整提议面积回退，最终删除99479个GT背景与14932个ignore，未删除有效GT前景、未露出其它前景类别，FP变2033174。grid_safe仅3个提议/317个GT背景删除，FP2132336。两者FN/错类均与B不变；严格safe禁止触及ignore显著压缩收益，不能把其近零解释为本批实际误删了有效GT前景，也不能把ignore当已确认背景。grid_budget_upper的宽松BG删除计数498675，剩FP1633978。自行车IoU：B55.273070%，grid_center57.826099%（+2.553029点），grid_budget_upper70.953986%，pixel_upper95.689432%，grid_safe55.280915%。这些均为GT诊断，不是可部署模型提升。

2000次共享配对图片bootstrap的总体mIoU差95%CI（百分点）：pixel_upper[+3.104617,+4.329505]，grid_budget_upper[+.911492,+1.133186]，grid_center[+.116826,+.273936]，grid_safe[0,+.001277]。图片不确定性不是训练seed方差；grid_center点估计+.181256仍未达预登记+.2门槛，CI跨过门槛不改写点估计通过状态。宽松上界大仅说明仍不能排除几何空间，不证明实际算法可实现1点。

独立只读验收001远端异常退出（REMOTE_EXIT=-1），检查发现历史NPZ矩阵在字典推导中反复完整解压且被行视图保留；已确认旧审计进程退出，002只修审计为一次读取矩阵后复用，未改任何研究结果、冻结源码或重算Oracle。verify_completion_002.py核1449结果内容SHA、全部B历史逐图矩阵/总和、原baseline运行时精确凭据、逐图上界>=safe、安全前景行不变、CSV/JSON/删除统计一致、Slurm实际CPU完成，审计不读GT/token/模型。completion_acceptance_001.json SHA`c67f9cd745530c4133adbcc66275624b95ca059bf9f70f03442119901f20c7a4`。summary/metrics/per_class SHA分别`7539ffa9529cd77243cc497c5dc9da5c934dd2a78ad2a98f0f10d4a1a5fab2e5`、`3c7b0cfd176e1d24d86a9ad6dfaa9e892ce3d4d89bb526d124b00fbdc5c71566`、`18f5bff436ccd5595427768759bc564142d1a17658440729dc36ed3f294bb627`。本地experiments/hpc_results/rpc_restricted_oracle_20261019_123200/verified_20261019保存三结果、验收与analysis_report.md/download_verification.json。完整验收及一次结果汇报后删除rpc-oracle监测，不做邮件排障或追加实验。

## 78. RPC背景误报分解及单约束CPU诊断（2026-10-19）

用户明确选择“执行修正版CPU实验1”，仅此批CPU事后Oracle；不授权SAM3重放、背景词前向、负点解码或实验2。版本`rpc_error_decomposition_20261019_133000`，计划`experiments/plans/rpc_error_decomposition_v1/plan.md`、`protocol.json`，独立core/runner/test分别为`src/rpc_error_decomposition.py`、`tools/experimental_rpc_error_decomposition.py`、`tests/test_rpc_error_decomposition.py`。生产模块及旧快照/指标只读。

固定原clean42/中心.2/HCR关闭、1449图；按最终raster owner把背景FP互斥穷尽分到整候选有效GT背景纯度>=.9、距真实外边界<=14原图像素的边界带、深部封闭/外接/ignore桥接不确定三类及other残差。外边界距离计算先只在诊断几何中填原mask已有孔洞，避免把孔洞自身边界误当外边界；GT0四连通和允许ignore桥接的连通分别报告，不把ignore称作背景。保留完整mask纯度/多数GT类，另列所属proposal预测类与多数GT一致的内部/封闭孔洞Oracle。每组只改该组最终可见FP为背景，增益不能相加。

复现第77节grid_center逐图矩阵，分别仅去外接拒绝、仅去20%整提议回退、仅min2改min1；另外原生像素组件用ceil(2*H*W/72^2)保留等效最小面积，并加保留原两中心支持的native对照，不把两个patch偷换成两个原图像素。原生连通改变本身是空间表示对照，不冒充严格相同算子的单插值参数替换；另列仅移除score/margin门槛以量化2951/3418覆盖。所有几何版本重放原提议顺序，分别报告proposal删除、原可见owner删除、最终背景及露出下层同类/异类前景。

标准21IoU/mIoU/FP/FN/错类、分解比例、2000共享图片bootstrap和ignore改变数均报告；共同修改ignore支持赋原baseline前景类作两组共用伪GT重新算mIoU，是baseline-correct-ignore压力测试，不冒充未知21类标签的严格全局最坏界，不能用背景数减ignore数推断mIoU。内部三类总Oracle<+.3点则stop；否则原分辨率等效面积对照须正向且>=2*max(grid_center,0)才记空间信号供复核，任何结果均结束本批，不自动做实验2。

拟单个debug 8CPU/23000M/2h作业：12项小测试→4张包含实际挖孔/面积回退/ignore的CPU技术探针→4CPU rank完整1449→CPU汇总，CUDA隐藏且无GPU申请；原PNG/历史矩阵及旧grid_center精确、输入实际消费SHA、真实GPFS余>=5GiB。只下载小结果及验收，不扫token/模型/库、不训练/前向/扫描参数、不邮件排障。实际提交JobID与签名、真实测试及全量结果待验收后追加；当前代码准备不代表已启动。

已成功提交唯一CPU作业3738385，实际8CPU/23000M/debug、无gres/gpu，状态RUNNING。audit/jobs.json及registered_manifest.json为提交依据；manifest SHA`4088c305937840147ff6ff97a2e57f5818249939c3f56c8744a4f38f856339f4`，binding`b10eeb3a2163680b5309e6e4602ad57a2627c6564c681d491645ca137e9e44ba`。真实GPFS余317651288064字节；probe_ids为2009_003224/2008_008711/2008_005727/2007_000033，按旧已验收挖孔/面积回退/ignore统计固定选择作技术覆盖，不据此调算法。尚未验收12测试、四图及全量，不把提交等同完成；不得重复prepare/sbatch。

正式全量已验收：集群12项测试无skip、4图真实CPU探针通过，原baseline像素/历史矩阵及旧grid_center逐图精确，分解穷尽互斥；四worker73866/73867/73868/73869在c21507运行，Slurm重叠只读step核/proc冻结脚本、manifest和phase shard命令，四份首批18×21×21矩阵及receipt SHA有效。startup_validation.json passed/formal_start_reported=true，SHA`8df42390d6417fb0812c38a1ee860978656bc622b677f46d45e96150bd7082e5`，已下载核SHA并报告，不重报启动。验收时730/1449推进，非全量完成；后续正常阶段不重复通知。

本批已完成并独立验收：3738385为COMPLETED0:0、8分38秒（518秒），实际8CPU/23000M，无GPU分配；完整1449图/3418提议/18组矩阵，原baseline85.841489%及旧grid_center逐图精确。背景FP2132653互斥穷尽：整候选BG纯度>=.9组1265439（59.3364%）、外边界带259618（12.1735%）、深部封闭378864（17.7649%）、深部外接99537（4.6673%）、ignore桥接不确定129195（6.0579%）、其它0。各组分别只删除该组最终FP的Oracle增益为+1.335793/+.430185/+.891522/+.172866/+.272681点，不能相加，也不是实际候选拒识成绩。当前160个高BG纯度最终可见候选，不复用旧201候选口径。

深部总607596个FP（28.49%）的联合Oracle+1.662506点；以所属mask有效GT多数类匹配预测作为“正确类别”代理后，深部237769个FP/+0.454324点，严格封闭146627个FP/+0.275650点。多数类不一致不等于语义对象分类必错，背景占多数的自行车mask仍可能确实含自行车；不可把所有深部空间当作正确候选孔洞空间。

固定几何相对B：原grid_center+.181256；仅取消外接拒绝+.321605；仅取消20%整提议回退+1.194382；仅min2改min1+.170210；native等效面积+.190197；native保留两原中心+.195045；仅取消score/margin门槛+.204596。native等效面积仅为grid收益1.0493倍，未达2倍门槛。22个面积回退候选中5个满足多数类匹配代理；不能据GT Oracle直接去掉真实方法面积保护。所有case最终有效GT前景预测未变，FN5020719/错类452394与B相同；仅无外接拒绝版露出9个下层异类前景，均属ignore，未错算最终BG删除。

共同ignore支持压力测试（不是未知21类GT赋值的严格最坏界）下，grid_center增益+.139211点，无面积回退+1.058814点，native等效面积+.189570点；标准grid_center删99479BG/14932ignore，ignore占13.05%，把ignore删像素算错误时BG precision86.95%，不能以BG数减ignore数替代mIoU。decision=internal_headroom_but_no_twofold_native_signal_stop_this_batch；不启动实验2，不自动SAM3/decoder/背景词/负点或训练。本轮未创建后台监测，无待删新自动任务。

独立验收核1449全部结果内容SHA、原/旧grid逐图精确、18组求和、FP互斥穷尽、压力矩阵与CSV/JSON、12测试/4探针及CPU完成；无重算Oracle/GT/token/模型读。completion_acceptance_001.json SHA`2cdaf82899785f78ad3fa8acd359ea55096cba72bec12f791ddffd5a38fced58`。summary/metrics/per_class SHA分别`b5b0190b6b3f80656c254ba983d9a0e4a44767fc7b09f6913b888b3c6d770b95`、`f15bc0bd436839ece0249ed647ca8a20d0f2658d9c63cea993c60509c4d7d639`、`b00684c0aaafd70687dfda1c7978bf5ed8c7d4367c0b77ad8ba9fbaa114d5fca`。本地experiments/hpc_results/rpc_error_decomposition_20261019_133000/verified_20261019三结果/验收/analysis_report.md/download_verification.json下载SHA通过；单seed已查看VOC开发Oracle，不是可部署收益或训练seed方差。

## 79. 椅子／沙发混淆的固定检索对照与事后库纯度诊断（2026-10-19）

用户明确要求运行检查实验、定位两类混淆原因及应修改哪一部分。版本`chair_sofa_diagnosis_20261019_155000`，计划`experiments/plans/chair_sofa_diagnosis_v1/plan.md`、`protocol.json`，独立代码`src/chair_sofa_diagnosis.py`、`tools/experimental_chair_sofa_diagnosis.py`、`tools/hpc_chair_sofa_diagnosis.sbatch`和12项小测试。冻结快照在audit/code；生产src/configs、旧库、模型、标签和结果不改。原34880成员（10569背景）、原GMP查询缓存、三个已有原MLP seed42/123/2026、中心.2、原提议/清理/NMS/raster/HCR；无训练、SAM3或token前向/读取，不重跑旧margin训练。

固定B原210全局候选内逐类Top5/.2；raw去MLP但其余相同；local中心权重0；center权重1；full_class每类全库精确Top5；image_balanced每类全库Top5尽量来自5个不同训练图片（同图只取最高）。所有21类同规则，无椅子／沙发专用推理规则。raw不依赖MLP，只在42计算后由123/2026共享，明确不是三个独立raw训练seed。三模型各比较clean及完整HCR，共32个唯一1449图预测集，合并保存为4347个每图NPZ及SHA凭据。原三seed clean PNG逐像素精确，42HCR逐像素及总矩阵必须精确原86.821717%；raw42HCR精确第70节no_mlp。123/2026HCR是中心.2匹配重放，不冒充旧中心.4历史。

先单GPU技术探针（集群12测试、两实际图、原分类分数逐元素及baseline/raw像素精确、精确分组Top5对CPU参考），再三GPU单worker/卡做三模型全量，随后CPU debug8CPU/23000M提交全部预测SHA后，4个图分片评测并汇总。真实GPFS余>=5GiB；瞬时quota最多3次；仅Slurm研究计算。记录每query的原样本/mask/索引、MLP前后类分数、局部与中心、210支持数、chair/sofa真实Top5行/图片，原clean和HCR最终owner对应可见错分像素，区分候选纯度、extras和栅格重叠影响。

全部预测提交后才读GT。对全体chair/sofa库成员重建原伪区域，必须类别/索引/maskhash匹配第73节已验收sidecar配对身份；仅解压小身份字段，不扫context向量。训练GT只登记路径/大小，首次实际消费在预测提交之后记内容SHA再解码。桥接副本来源被保守排除的伪标签、不可重建区域或缺GT明确unavailable及覆盖率，不替代或剔除任何推理库成员。比较错误/正确query的Top5成员GT纯度及同图重复，GT纯子集类中心仅作Oracle组成诊断，不用于过滤或新推理。最多10幅query+Top5红mask覆盖图按事后错分像素固定排序保存。来源澄清见第80节：pseudo_fallback_ids不是GT替代标签，旧统计及冻结指标不改。

报告所有21IoU/mIoU、FP/FN/错类、两混淆方向、三seed均值/样本SD/配对方向及2000共享图片bootstrap。固定改法需完整HCR均值>=+.2点、至少2/3正向、两类互错合计降>=20%且各方向不增、chair/sofa平均IoU不降、其它类平均IoU最大下降<=1点、BG FP和FG FN各增加<=1%，才记开发集上promising，仍不自动部署；全不通过则明确尚无验证解决方案。类中心接近或GT纯度相关不能单独证明因果。本固定批结束停止，无追加训练/参数搜索/新seed/邮件排障。

已提交technical3740144、score3740145 array0-2%3 afterok technical、evaluate3740146 afterok score，全部成功；audit/jobs.json为实际命令及JobID。manifest SHA`0716c56294d33516bd1eaae814c02adf483cbb92e1f7c27ca60353e95f771f8e`，binding`2f1a3349452bef97957315e62955756eadb424a2d773f288f23445859417d864`。提交时technical为PENDING(Resources)，不能把已提交称作技术通过或正式全量启动；不重复prepare/sbatch，接下来先验收真实探针及三worker首批内容SHA。

正式启动已验收：technical3740144 COMPLETED0:0、1分11秒；集群12测试无skip、两图2007_000452/2007_000661真实clean/HCR/raw像素精确、原分类score逐元素及精确分组Top5对CPU参考maxabs7.48784e-8通过。三score worker分别PID213623/300368/1642431，实际child3740252/3740255/3740145，在c2v100m2-5/4/18运行；Slurm重叠只读step核/proc真实冻结脚本/manifest/score seed命令，三首份FP32类分数、语义图、owner及Top5行类别/34880身份内容SHA有效。startup_validation.json passed/formal_start_reported=true，SHA`d161414929d3279ba4da74e6072a13ea1abbce18bc228ffb0947eaf6aef8f0ef`，不重复报告启动。提交GPFS实测余317389471744字节；当前仍在预测阶段，未读新GT、无最终指标，不能把技术通过当实验完成。独立只读验收verify_completion_001.py在audit外层已上传，尚未执行，需全量完成/无审计运行后只执行一次；自动跟进id=automation，正常阶段静默，完成验收后删除。

本批已完成并独立验收：technical3740144、三个score实际child3740252/3740255/3740145及CPU evaluate3740146全部COMPLETED0:0，分别71/613/594/558/538秒；实际3原模型worker各1GPU，CPU评测无GPU分配。整体墙钟16:09:06–16:29:38共1232秒，不累加并行秒数；三worker记录峰值各65.1436MiB。4347个每图预测NPZ/32唯一1449预测集已完整提交SHA后才GT；36逻辑矩阵含共享raw别名、全部逐图求和与CSV、三seed clean历史逐图及42HCR原总精确，raw42HCR精确原no_mlp。独立verify_completion_001.py已执行一次通过，无锁/失败，不再重跑预测、评测或审计。

完整HCR三原模型均值mIoU：B86.505997%、raw71.443980%（−15.062018点，单一共享对照）、local85.889739%（−.616259点）、center86.555136%（+.049139点）、full_class86.492370%（−.013627点）、image_balanced86.483617%（−.022381点）。B三seed样本SD.274697点；42仍精确86.821717%，不要把三seed均值与旧42单值混比。相对B的两类互错总数变化依序raw+23.32%、local+27.82%、center−3.32%、full+1.38%、image+2.75%；center有2/3总体正向，但42下降、chair→sofa方向增加、两类IoU/FG FN也不满足限制，五项均不通过固定成功门槛，promising_cases=[]、decision=no_validated_fix_from_fixed_retrieval_controls。clean均值B85.699896/raw71.443980/local85.159426/center85.745342/full85.699896/image85.689626%。无新训练或自动部署/追加。

库纯度后验覆盖：chair2487中2477可评（99.60%），等成员平均本类有效GT纯度66.98%；563个有效GT多数非chair，其中489BG、21sofa。sofa894中886可评（99.11%），平均87.14%；60个多数非sofa，其中45BG、3chair。18个不可评保留原推理身份，未过滤；GT多数不等于对象完全不存在，不能把这些统计冒充密集监督清库结果。错误query Top5引用对方类别GT像素均值三seed仅约.0123%/.000063%/.4104%，不是两库普遍互换；其本类纯度74.13%/78.16%/83.39%对正确纯query79.90%/82.58%/82.54%，第三seed没有下降，因此“污染直接导致错分”并非跨seed已证因果。错误Top5来自4.75/5/5个不同图，图片去重无收益。原raw两类中心cos.8813，投影后.5553/.5965/.5779；GT纯>=.9子集（chair829/sofa667）的raw中心.9183、投影.5629/.6180/.5931，仅Oracle组成诊断，不生成GT过滤推理。

最终两类互错像素中94.39%/94.89%/94.89%由GT多数为chair/sofa且纯度>=.9的原query承担，不支持“主要是候选同时圈进两类”解释。每seed仅16/20/20个可见query承担互错，前5占86.52%/76.55%/76.49%。共同最大难例2011_000661 query1的132003有效GT像素全chair，三MLP全判sofa，单例占互错46.38%/42.28%/42.24%，固定投影后改法均未纠正；raw判BG。seed42第二大2007_009655（99.43%sofa却判chair，margin.00205）可由center-only纠正，但其它seed/其它难例无普遍收益。已有证据不应优先删除MLP或继续扫Top5/中心权重；后续若立项应研究上游伪区域质量和区域表示的形状/实例结构区分，尚未验证出解决方案，不能承诺有效或自动训练。10幅可视化红色是原mask叠色，不据此认定原颜色相同。

独立completion_acceptance_001.json SHA`7f65fd6aaf1cd5cba3ee541e72beb68e4dad1046f441e3843b112a04bbb350dc`；summary/metrics/per_class SHA分别`263ff18bf33e21b0bf0ae676ea45a310bce16a863f8738a58a5bd17789ebdd82`、`8de6c2d5f98923c4d6c67b6e30ec78a97ddb054fa29aa7877de7d7612fd93c98`、`df9b7504370c605c4a6f6b5e1be257fc619dd22cb167549bb99443d059aa7984`。本地experiments/hpc_results/chair_sofa_diagnosis_20261019_155000/verified_20261019已核三结果/验收/proof/10图以及1.37MB已有query诊断SHA，analysis_report.md和posthoc_analysis.json为独立事后汇总不覆盖正式指标。2000配对图片bootstrap保留各seed/mode区间，不当训练seed方差；用户不重视邮件，不排障。完成汇报后删除automation监测。

本批验收归档完成后，automation监测已通过工具删除；无后续实验或邮件任务。

## 80. 全局GMP＋2×2空间表示及容量匹配对照（2026-10-19）

用户明确回复“你试试吧”，授权前述三组独立实验。当前有效版本`spatial_gmp_runtime_recover_20261019_172500`，计划`experiments/plans/spatial_gmp_v1/plan.md`、`protocol.json`，新增独立`src/spatial_gmp.py`、`tools/experimental_spatial_gmp.py`、`tools/hpc_spatial_gmp.sbatch`和12项小测试；冻结代码在audit/code。生产src/configs、原伪标签、原库/模型/指标只读。原基线B复用第79节三已有原MLP及中心.2 clean/HCR预测，不重训或混入旧中心.4。

固定原34880成员/10569背景、sample/class/region/maskhash与顺序、8466fit/2116内部伪标签留出划分，原1008 FP32/72×72 token。复用原签名1280D全局GMP字节，再按原mask紧包围框半开边界中点，把已选stride14中心精确分成TL/TR/BL/BR；原全局nearest fallback只保留一次，分区不新增fallback或重复token。各非空区使用原rho.1 FP32 Cholesky GMP和最终phase L2，空区全零。空间组S输入为全局＋四区concat L2的6400D；容量组C把原全局向量复制五份再concat L2，不引入新信息。两新MLP均6400→512→256、两层、adapter3408640参数＋21类linear classifier5397参数，同v1 CE+.5SupCon/AdamW.001/50×100步、8类×4实例和原留出集选bestepoch。所有21类包括背景同规则，匹配新签名库及查询；检索K25/top5/search2/中心.2/NMS/raster/HCR不变。无新SAM3前向、新标签/过滤/损失、上下文重排或patch分类/挖孔。

首批seed42只新增C/S两次完整50轮训练，原B只读复用；新C/S各1449、clean/HCR合并NPZ的内容SHA全部提交后才读GT。S须完整HCR对B和C均>=+.2点、两类互错各降>=20%且两方向不增、chair/sofa IoU均不降、其它类最大下降<=1点、FP/FN各增<=1%，且clean对两参考不降，才自动提交123/2026各C/S四次训练；固定最大6次新训练，所有分区缓存/库共用，不扫描参数。全不通过则完成seed42开发结果停止。最终三seed还需相对B/C各至少2/3正向；图片bootstrap不能当seed方差。通过也只是已查看VOC开发信号，不自动部署。

技术阶段：集群12测试无skip、一train两val真实池化，原全局重新聚合相对已签名raw maxabs<=1e-5并报告bitwise，原42 clean/HCR像素精确；另用最多4张trainaug伪标签图片做C/S实际训练入口1轮×2步、保存/加载/导出新签名，不能称正式50轮或质量结果。通过后四GPU单worker/卡消费10582train＋1449val原token；仅实际消费核SHA，本批确需建新空间表示，不额外预扫12031大token。CPU合并两个6400D原始库，两个单GPU训练预测任务，CPU评测和2000共享配对图片bootstrap。GPU4CPU24GiB/worker；CPU优先debug8CPU23000M/2h、CUDA隐藏。debug因明确QOSMaxSubmitJobPerUserLimit且未发JobID才保存失败请求、换v100g32单GPU配额6CPU24GiB/相同单进程CPU脚本、CUDA隐藏，不能重提已获JobID任务。真实GPFS余>=5GiB，瞬时quota最多3次，不以df代替。

来源澄清：核对原`prepare_oracle_guided_cluster.resolve_pseudo_records`，`pseudo_fallback_ids`35张/131成员是两服务器伪标签SHA不同时回退到**原已签名伪标签桥接副本**，不是GT标签替代。新实验按原class/region/maskhash核一致并完整保留；不新增密集GT训练。第79节纯度审计保守排除了桥接来源，其中18个chair/sofa成员计unavailable，因此旧纯度覆盖/指标保持原样；此前文档“GT回退伪标签”描述已纠正，不改变任何旧冻结结果。

准备修复历史：初版`spatial_gmp_20261019_171500`的technical3740973、extract3740974、merge3740975、train3740976均已核PENDING后取消，全部0秒，无GPU或正式训练；evaluate的debug请求被拒、无JobID。旧jobs及superseded_pending_cancellation_001.json保留，不重启。静态检查发现runpy返回模块globals不包含main内部model变量；新版本仅把训练后统计收集改读已保存checkpoint，训练器源码SHA仍`f1bbdec37250fc4cd96eaf9c848d26d9fc7b251fd7a9f984f3101074566d2ba8`，算法/参数/规则不变，另登记CPU资源后备；不得把旧取消或无JobID拒绝当科研负结果。

当前版本五阶段全部成功提交：technical3741010、extract3741011 array0-3%4、merge3741012(debug)、train_predict3741013 array0-1%2、evaluate3741014(v100g32单GPU配额但隐藏CUDA CPU)。最新manifest SHA`5144a25237353639dbbaab3a4b9726f9f7d917338f7c03cdfea249bda7f679fd`，binding`d96121df510a50f1f22303f0557369a954e0ae4ef3e9da236bd5e931a80f542f`，audit/jobs.json为实际命令/JobID依据。resource_fallback_evaluate_001.json SHA`5b1b92f60342daf5e62c50526b598de6d5cdb191840b491e4abc0e2a53c0c4d5`保存原debug无JobID拒绝和唯一后备请求。全部afterok串联；条件追加以conditional_jobs_001.json为准，写提交中状态后再sbatch，绝不重复已提交任务。

初次状态核对时technical3741010已RUNNING、集群12测试全过，真实池化和实际新签名短训练仍在技术验收中；当时未写startup passed，不把测试或提交当正式全量/训练开始。通过后需核四worker/proc实际命令、四首份6400D FP32/支持数/空区/身份内容SHA，audit外层validate_startup_001.py已准备；formal_start_reported=true后不重报启动。独立verify_completion_001.py只在完成/无同审计进程时启动一次，核新模型50轮/5000步/相同参数量与原身份划分、所有小产物SHA/每图矩阵求和/原B逐图精确/CSV/Slurm/耗时峰值/固定门槛，不读GT/token或重算实验。只下载三结果/小验收和必要诊断到新本地目录，更新本节。正常阶段静默，完成失败需操作报告；失败先日志、保留attempt，旧worker退出锁释放且代码参数输入完成产物SHA同才显式resume；训练无optimizer续训支持则新attempt完整50轮。实现更改用新版本真实针对探针，不改旧快照；用户暂停取消立即停自己任务和监测，不自动恢复。固定批完成验收汇报后删除监测，不排障邮件或自动追加其它算法。

正式四卡提取已验收并在本回合报告：technical3741010 COMPLETED0:0、185秒，12集群测试无skip、一train两val真实池化/原42clean及HCR像素精确、两6400签名实际1轮2步训练保存加载导出通过。原全局重算maxabs1.6167759895324707e-6、bitwise=false（允许FP32跨硬件技术差异）；实际拼接仍复用原签名global字节，不将重算误差冒充输入变化或逐元素相同。两短训练adapter参数均3408640、记录peak各82.6333MiB，仅技术连通不代表质量。预提交GPFS实测余317038854144字节。

四提取worker经Slurm只读overlap step核/proc真实冻结命令、四份首缓存6400D FP32/归一化/原样本-类别-maskhash、支持数合计和空区零/内容SHA通过，startup_validation.json passed/formal_start_reported=true，SHA`f19860a6fb0c75e497f2f6816e0c797c3a7fc8d49cb92bdc8dd2a4e2cbd26542`，凭据已下载核SHA；不再重复启动通知。当前仅提取阶段，正式两次50轮仍等待全部分片及合并，未得到新mIoU。后台监测id=gmp已启用，按固定门槛继续与收尾；正常阶段静默，完成验收后删除。

首批seed42计算已完成：四extract/merge以及两train_predict child均COMPLETED0:0，C/S各50轮和各1449 clean/HCR预测完成。3741014已完成全部GT评测及2000配对bootstrap，evaluate_first/complete.json和group_metrics.json（SHA`3bda9ba26f4fb0b522a2c8e1a859de13abf9aaf7c180f4a92962104d3850da47`）、逐图矩阵（SHA`467ccd602989112fd0c76966937ff472d4bc0770b7a14d97e9fa7a9037b843a2`）完整保存，但随后写first_stage_gate.json时numpy.bool_无法JSON序列化，作业FAILED1:0。原failure记录和94字节partial gate保留，不能把该Slurm失败冒充训练/预测失败或丢弃已完成评测。

用户询问完成情况时发现上述汇总异常。独立audit外层reporting_recovery_001.py与reporting_recovery_job_001.json登记唯一CPU debug8CPU23000M恢复3743054，只把JSON输出中的numpy scalar转换为等值Python scalar后调用冻结finalize；不改code/manifest/算法/参数，不重训、重预测、重新读GT或重跑bootstrap。原partial gate移入reporting_recovery_attempt_001保留SHA，原failed job和failure文件不删除。恢复proof reporting_recovery_001.json SHA`b7a50bfac17b41e99c5009bdff8975ee74635fac91607951c5b01c0b8d4cc43e`显式绑定原指标/矩阵/预测提交凭据不变以及恢复脚本/作业。原verify_completion_001.py不改，独立verify_completion_002.py保留全部原核验，仅允许这一SHA绑定的报告失败历史，并要求成功恢复job；未声称原3741014变为COMPLETED。audit外层一次启动凭据和进程防重已登记，正在收尾验收。

已核SHA下载的首批结果：clean B85.841489%/C85.798648%/S85.737179%；完整HCR B86.821717%/C86.733002%/S86.735792%。S−B完整HCR−.085925点，S−C+.002790点；配对图片bootstrap对应95%区间分别[−.617258,+.530754]和[−.483126,+.506570]点，均跨0。S完整HCR chair66.772782→64.456431%、sofa63.892376→63.197273%；两方向互错总像素284599→287834，chair→sofa增加、sofa→chair减少，不能当解决混淆。固定首批门槛失败，停止此单seed硬分区批，不追加123/2026，不修改独立第81节SA-GMP的三seed授权。

summary/metrics/per_class已下载至experiments/hpc_results/spatial_gmp_runtime_recover_20261019_172500/verified_20261019，SHA依序`7d3c5d56c8ebb37a21e6fe7a4ba4a3a67f71225fba4a0dbf7249002835df1de9`、`94db1dc6ffb35dbb810e118b94a2631e220c6779e852bd32068cb084d2b8c7f9`、`af2e2159dfd26b5579897abd4203af30fed963b479c50be8bc7e1d982c4d4944`，completion_verification SHA`0c4c3148722b5fb32dcffb584accd63f13678ec0170d7ea75e3b19f333d3215f`。独立最终验收和归档凭据完成后再追加；本轮为单seed已查看VOC开发证据，不宣称所有空间表示无效。此时gmp监测已为PAUSED，未擅自恢复，完成收尾后删除；第81节独立监测不更改。audit/independent_audit_launch_002.json已防重一次启动只读审计PID2707399，在c2ln9核/proc真实verify_completion_002.py命令正常、共享log已推进。外部SSH可能轮换到c2ln7/8/9，不能因另一个登录节点/proc不见PID就判断审计退出或重启；只读共享log/验收文件并核正确节点，不重复审计。

最终独立验收已通过：verify_completion_002.py由既有防重进程完成，completion_acceptance_001.json SHA`b4cd46419061cf63459ac7ec202b0ef6134cc3a01d57c6bff1c8d020c1180efc`，已下载核SHA。它保留并明确接受上述报告故障历史，完整核12031图片缓存/34880成员/两新模型各50轮5000步/两个1449预测集/矩阵与CSV/历史B及成功恢复作业，不读GT/token、不重算实验。两训练真实峰值均138.132324MiB，训练秒数C1027.775773/S1050.204455，best epoch38/48；含报告修复等待的整体墙钟13698秒，不累加并行。独立analysis_report.md和posthoc_analysis.json已更新为accepted。本批固定停止，按授权删除已暂停的gmp监测，不恢复运行。

## 81. Structure-Aware GMP：软空间分支与三标量残差单库实验（2026-10-19）

用户明确要求按附件完整方案“设计实验，并运行”。独立版本`sagmp_singlebank_20261019_174000`，计划`experiments/plans/sagmp_singlebank_v1/plan.md`、`protocol.json`和原文`request.md`。按用户此前已确认的当前34880成员/中心.2修正附件过时的37200/中心.4描述；10569背景、原8466/2116伪标签图片划分、sample/class/region/maskhash及顺序不变。35张/131成员仍为原签名伪标签桥接副本，不是GT替代。第80节6400维硬分区研究是另一独立授权实验，继续原计划，不取消、不更改或合并结果。

新增独立`tools/sagmp_singlebank_core.py`、`tools/experimental_sagmp_singlebank.py`、`tools/hpc_sagmp_singlebank.sbatch`及16项小测试；不修改生产src/configs、旧库/模型/标签/提议/指标。复用原1008 FP32/72×72×1280缓存token，严格原hard-center选择和唯一nearest fallback。token坐标采用原floor投影的原图像素位置，mask半开紧bbox计算相对u/v；四个双线性soft权重逐token合计1，各分支按权重总和归一后用rho.1 FP32原GMP normal equation、较小primal/dual Cholesky、residual<=1e-4和原phase/final L2。若单分支权重总和<=1e-12，显式复用原global、记录例外，不补token/SMFA或改precision。global分支原签名1280字节复用。缓存[5,1280]而非直接6400输入MLP。

rx=.5(TR+BR−TL−BL)，ry=.5(BL+BR−TL−TR)，rxy=.5(TL+BR−TR−BL)，最终L2(global+(tanh(gx)rx+tanh(gy)ry+tanh(gxy)rxy)/3)。三gate零初始化，与原1280→512 GELU→256两层MLP联合训练，原adapter787200参数只加3个=787203，classifier5397；v1 CE+.5SupCon、背景negative-only、8类×4图、AdamW.001/weight_decay.0001、cosine scheduler/clip5、50轮×100步、原伪标签内部留出kNN k5 macro选bestepoch。三个seed42/123/2026无条件各一次新完整训练，最大3次，不套旧context或第80节首seed继续门槛，不重训B、扫参或加seed。

每seed导出一个匹配256D库；自定义checkpoint格式`sagmp_singlebank_v1`绑定gate/MLP/规则/SAM3签名/成员SHA/seed/binding，普通adapter loader必须拒绝。新库签名额外绑定完整checkpoint SHA，库与查询使用同checkpoint。原balanced K25/top5/search2/中心.2、score融合/NMS/raster和HCR不变，全部21类含背景同规则。对照复用第79节三seed原B中心.2 clean/HCR预测，42HCR必须精确86.821717%，不混旧中心.4。无新SAM3前向/提议/伪标签/重新过滤/GT训练。

先单GPUtechnical：集群16项测试无skip，覆盖soft权重和为1、零gate/MLP回归、位置置换、相对平移/缩放、weighted与原uniform、primal/dual参考、单token、梯度/参数量/加载拒绝；真实一train两val检查global对原签名maxabs<=1e-5并报告bitwise、zero-gate allclose、原42clean/HCR像素精确。最多4张原伪标签trainaug图实际短训练1轮2步、保存/加载/导出；不是正式50轮或精度。随后双卡各一worker提取10582训练图+1449val共享branch缓存、CPU合并、三单GPU训练预测、CPU评测。仅实际消费token时核SHA，不轮询预扫12031token/全工程测试。CPU首选debug8CPU23000M；仅明确提交数QOS拒绝且无JobID才登记失败请求后备v100g32单GPU配额6CPU24G且CUDA隐藏、同CPU脚本；不重复已提交任务。真实GPFS余>=5GiB。

全部三个新1449预测内容SHA提交后才GT；12逻辑矩阵（B/SA×三seed×clean/HCR）逐图求和，B历史逐图精确。报告全部21IoU、FP/FN/错类、chair/sofa两方向、seed配对差/均值/样本SD/正向数和2000共享配对图片bootstrap（不替代seed方差）、bestepoch/角色耗时/整体墙钟不累加并行/真实GPU峰值。诊断包含每轮及best checkpoint raw/tanh gate、global四分支cos/三方向norm的mean/std/p10/p50/p90、空间残差占比整体/按类/固定支持数分箱、fallback和solver误差。gate magnitude不是因果归因；接近零不证明所有空间设计无效；零gate保证表示回归容差，不保证重训后旧模型预测相同。

独立startup验收需两worker Slurm/proc真实命令和首批FP32 [N,5,1280]/原样本身份/global字节/内容SHA通过；startup formal_start_reported=true后不重报启动。独立final验收只在完成且无同审计进程时执行一次，不读GT/token或重算。小结果summary/metrics/per_class/summary.md及验收到本地新目录核SHA，更新本节，固定三seed后停止并删除本批监测，不自动部署/追加/排障邮件。失败保留attempt，旧worker退出锁释放且源码/参数/输入/完整产物SHA一致才显式resume；训练无optimizer恢复，失败新attempt完整50轮；代码修复新版本真实针对探针，不改旧快照。用户暂停取消立即停本批与对应监测，不自动恢复。

初次准备时本地实现与语法检查完成，本地Python无torch，因此16项真实数学测试待集群technical；不将本地语法检查称为实验启动或结果。随后实际提交与验收记录如下。

已实际成功提交：technical3741400、extract3741401 array0-1%2、merge3741402、train_predict3741403 array0-2%3、evaluate3741404，均afterok自动衔接。manifest SHA`bf3912dd084c3fe68c1045a4f777f93cb1cf30f1fa9a03fdefa8fdc173b51231`，binding`9ee0387fbffe81e91df32779afbea6424ea38e0c3df778ab5e85709a615f162c`，audit/jobs.json为实际命令/JobID依据。初prepare已写空ledger，随后因A.write排他创建不能更新ledger而在首次sbatch前退出，无JobID/无GPU；历史controller_failure_001.json保留，audit外层新submit_registered_001.py核空ledger/无原prepare进程/无sagmp作业后只修调度ledger原子更新，scheduler_recovery_001.json记录，不改任何算法冻结code/manifest，绝不重复prepare或该恢复controller。两个CPU debug请求均因明确QOSMaxSubmitJobPerUserLimit被拒无JobID，分别resource_fallback_merge_001.json SHA`b1a51d55400ff25e65d4a36ad12ec2a9441d88262e40f155c92af6cc8d7f7be8`、evaluate SHA`5e62a2435ac6be0bb28fd584a56703763e275eeb7ef53ded636035fd71d19975`登记唯一v100g32配额/CUDA隐藏6CPU24G后备。预提交真实GPFS余313648766976字节。technical已RUNNING且16项集群测试无skip全过（7.337秒）；此时真实图像和短训练探针仍进行中，不能称正式提取/50轮或精度完成。独立后台监测id=sa-gmp-seed，正常状态静默，固定批验收后删除；第80节gmp监测不变。

正式双卡提取已完成首次验收并报告：technical3741400 COMPLETED0:0、183秒；16项测试及真实一train两val、零gate allclose、原42clean/HCR逐像素回放、实际1轮2步gate+MLP保存加载/单库导出及普通loader拒绝通过。重算global对签名raw maxabs1.6167759895324707e-6、bitwise=false（FP32跨硬件技术差异），实际新分支缓存global仍复用原签名字节；technical_report SHA`2ec306531f6b13e669383915ca49c19ccaae135498b041b9d22bc52b0aa176fa`。短训练两步的.5内部准确率仅技术连通，不能当方法质量或正式50轮。

两extract实际child3741424/3741401，在c2v100m2-2/7，PID2199353/3908121；经Slurm只读overlap step核/proc真实冻结入口/manifest/shard，并核首批FP32[N,5,1280]、原sample-class-maskhash、global原字节SHA、归一化/支持数/soft权重和/solver误差/内容SHA通过。startup_validation.json status=passed/formal_start_reported=true，SHA`c18e9d3e6ae6d923259c9926543611c9da15ec954c4b3c4c38f1afd8327406da`，不重复启动通知。当前只进入正式分支提取，三次50轮尚待提取和合并，未得到新mIoU。固定后台Slurm链与监测继续，完成后独立验收与小结果归档，不自动追加任何实验。

本批训练、预测、评测已全部结束：technical/两extract/merge/三train_predict/evaluate共8个计算child均COMPLETED0:0、squeue为空；3个新模型各50轮5000步、各1449图clean/HCR，全部4347个预测NPZ提交SHA后才GT。原B中心.2六组历史逐图精确、12逻辑矩阵求和及CSV均由正式completion_verification核通过，SHA`105f8303f44ee03db07b858deacaff6118382db6deead1f86bbcab9065ec1554`。summary/metrics/per_class/summary.md已下载至experiments/hpc_results/sagmp_singlebank_20261019_174000/verified_20261019并核SHA，依序`e18b6a5b6850e2cbe239f949a1a0c82c4e8db886fc456b2ba8feb2c3cfb9008b`、`625ab3d6b3c3a74670b9043746004d1dda72dc3694c823e15509e3f4c6130ac3`、`76db580bd1db2acb498637291c9aafcf875b5d46ed718cf0f808977c9a3cce0d`、`9797bc72a409f0975f622234f524544e6bf740f4587f7407d58760075fa4862a`；all_predictions_committed SHA`259ffbc5bc5287a49d01bb891da8b7cc2a84c6d4e4b6ada3211960528f4b9e4c`，不重跑训练/预测/GT/bootstrap。

三seed均值±样本SD：clean原B85.699896±.141559%，SA85.406932±.196135%，配对差−.292965±.280248点、1/3正向；完整HCR原B86.505997±.274697%，SA86.170405±.085409%，配对差−.335592±.255564点、0/3正向。HCR各seed42/123/2026差值−.606012/−.098070/−.302694点；2000共享配对图片bootstrap固定三seed均值差95%区间[−.808074,+.076879]点跨0，不称统计显著下降，也没有正收益证据，不替代训练seed方差。固定三seed完成停止，不追加实验或部署。

目标类别HCR均值chair64.302969→63.497706%（−.805263点）、sofa63.960734→62.086738%（−1.873995点）；每seed平均chair→sofa像素255355.33→246114.67（−3.6187%），反向47758.67→61483.33（+28.7375%），两向合计+1.4793%。整体BG FP均值−2.0554%，FG FN+3.0971%，FG错类+16.1518%；sofa/train/bird/sheep/cow退步，bottle/car改善不足抵消。不是对椅子／沙发的有效解决方案。

best checkpoint tanh gate按seed42/123/2026：x为−.043252/−.009994/−.012640，y为−.331568/−.341446/−.420972，xy为−.029707/−.043024/−.112929。残差/全局平均范数3.7840%/3.8714%/4.7780%；四分支对global平均cos约.970–.972。三seed均为上下gate绝对值最大，均值绝对值排序y>xy>x，不当因果贡献；gate没有全关闭，不能将负结果解释为空间从未参与，也不外推所有空间编码无效。34880训练区域零权重分支/nearest fallback均0，max solver residual7.02758939e−7。真实训练秒数881.069174/856.058284/863.672120，best epoch14/13/24，峰值均68.172363MiB，787203+5397参数匹配；每轮gate、逐类/支持数分布在小报告中保留。

独立只读SHA验收verify_completion_001.py已由共享排他凭据independent_audit_launch_001.json唯一启动，host=c2ln5、PID2240841、脚本SHA`fd8de582571e69a00cfdff062c73fc8ae82fd2068969e4d3a8e8ee4e140e7c04`；不重复启动、不读GT/token/重算。待completion_acceptance_001.json后补最终凭据。两监测在App均为PAUSED，未擅自恢复；第80节验收归档已完成后删除gmp，第81节独立验收归档后删除sa-gmp-seed。analysis_report.md/posthoc_analysis.json仅作小结果独立事后分析，不覆盖正式指标。邮件不排障。

第81节最终独立SHA验收已通过：原唯一进程完成全部两shard缓存与三模型/4347预测/12矩阵/CSV/历史及8个Slurm计算child核验，completion_acceptance_001.json SHA`ed1838c9d6543f9db278b06e09c0392702331b5a70d3667bc41132ea9913e60f`已下载核SHA。计算整体墙钟14401秒（并行不相加），小analysis_report/posthoc/download_verification已更新为accepted并归档；不重算任何实验。旧sa-gmp-seed监测按授权删除，不恢复或追加。第82节只读复用的source版本/向量/成员签名保持不变。

## 82. Spatial Residual Probe：内部伪标签线性可分性、KNN与位置shuffle诊断（2026-10-19）

用户附件明确授权实现并运行独立诊断；版本`spatial_residual_probe_20261019_231500`，计划`experiments/plans/spatial_residual_probe_v1/plan.md`、`protocol.json`、`request.md`。新增`tools/spatial_residual_probe_core.py`、`tools/experimental_spatial_residual_probe.py`、`tools/hpc_spatial_residual_probe.sbatch`和14项测试；不改生产src/configs、旧模型/结果。只复用第81节已完成SA-GMP的merged_branches训练成员缓存，不生成新branch/token/伪标签/提议或bank，不读1449张VOC最终val查询缓存、GT或val split文件。旧manifest SHA`bf3912dd084c3fe68c1045a4f777f93cb1cf30f1fa9a03fdefa8fdc173b51231`固定；34880成员/10569背景、原8466/2116内部图片划分。按catalog核sample/image/class/region/maskhash，按原merge member_digest核逐类成员顺序，原始与三份SA训练report划分精确；缺失或不符停止，不以数量代替身份。

不乘旧gate：rx=.5(TR+BR−TL−BL)、ry=.5(BL+BR−TL−TR)、rxy=.5(TL+BR−TR−BL)。A=global1280D；B=分别L2的三个方向concat3840D；C=global+B的5120D。norm<1e−12保留零并记数，线性输入不额外整体L2；KNN输入整体L2。各为单一Linear(input,21)，参数26901/80661/107541；无隐藏层/激活/LayerNorm/Dropout/SupCon/scheduler。9个常规单元A/B/C×seed42/123/2026，加2个seed42的B_shuffle/C_shuffle，共11个正式轻量训练。AdamW lr.001/weight_decay.0001、50个完整数据epoch、batch512，fit类频率N/(21*n_c)加权CE，内部heldout macro accuracy首次最优epoch选模。各seed相同数据遍历顺序SHA；OOM最多3次减半batch，从seed完整重训50轮并保留不完整checkpoint/OOM凭据，不伪造续训。本地14项Numpy测试已通过，Torch/GPU路径仍待集群实际技术检查，不把本地测试当已启动或质量结果。

非参数cosine KNN各A/B/C一次、K5多数票，票数平局取该类邻居平均cos，再平局canonical class_id；相同相似度用原gallery行顺序稳定排序。fit gallery与internal heldout query严格按图片隔离。shuffle固定42+sample_id派生随机数，每个region仅在自己的TL/TR/BL/BR间换identity，global/四向量multiset不变，不跨region/图片交换；允许identity排列并报比例。shuffle只与相应原B/C seed42配对，不拿三seed均值对单shuffle。

报告overall/macro accuracy、macro F1、21类混淆及支持、top1、softmax与logit top2 margin，三seed均值/样本SD/C-A配对方向，uniform1/21与fit多数类的heldout baseline。残差norm/cos按类及fit/heldout/all汇总，零向量cos pair排除；A/C正确错误四象限按类统计。car/bottle/sofa/train/bird/sheep/cow/chair为预指定诊断，不能据其结果调参。线性维度/容量与块数范数不同，C>A不能单独证明旧SA-GMP因scalar融合失败；probe伪标签分类不等于VOC分割mIoU或稠密GT泛化。

预注册1pp宏准确率为实践分辨阈值，B明显有信息须高于uniform和majority macro较高者至少10pp。Case4优先：seed42 B/C均与各自shuffle差在±1pp内，只记该控制未分辨identity收益，不冒充统计等效。Case2还需C-A均值>1pp、>=2/3正且KNN C>A；Case3须C-A均值<−1pp、>=2/3负；Case1须B未达信息阈值且|C-A|<=1pp。原四类非穷尽，未覆盖或冲突组合明确inconclusive，不强迫归因。任何结果本批结束，不自动adaptive gate/rerank/新实验。

HPC仅Slurm：technical单GPU先14测试/全部真实merged成员SHA身份/实际2步classifier保存加载/小KNN对CPU参考；通过后3seed GPU array（42顺序5单元，其余各3），独立GPU KNN，CPU finalize afterok全部。每GPU一worker，最多4卡并行，真实GPFS余>=5GiB且瞬时quota最多3次；debug明确QOS拒绝且无JobID才记录v100g32配额/CUDA隐藏CPU后备，不重复已提交任务。冻结audit/code，作业/参数/签名以audit/jobs.json为准；同版resume须完整SHA同且旧worker退出锁释放，只复用完整单元，不完整训练从头50轮。technical失败仅取消自己未启动依赖。输出experiments/<版本>/linear、shuffle、knn、diagnostics、checkpoints和summary，日志logs/<版本>；只下载小summary/diagnostics/验收，不传模型或数据。实际提交及真实技术验收凭据待返回后追加，当前准备不代表已提交或通过GPU技术检查。

已成功一次提交：technical3744143、train3744144 array0-2%3、KNN3744145、CPU finalize3744146，均有JobID，afterok技术→训练/KNN→汇总。debug CPU申请本次成功，无资源后备或重复提交。manifest SHA`e53106d16ff337b27bdf90fc871a575b34b834e479b8c7301aef883bc6ae4c19`，binding`03c5a68bf869b1e9eb133a243dc234808ea95a6c4f947ec7813f3d02aaa6a0c9`；源码和实际命令以audit/jobs.json为准。本地verified_20261019已核下载manifest/jobs/submission_complete/实际GPFS凭据；mmlsquota余306670714880字节。首次查询technical在c2v100m2-11 RUNNING，训练与KNN及汇总等待依赖；尚未验收真实GPU技术结果，不把提交当probe训练已完成或准确率。controller已返回0，不重复prepare/sbatch。

真实技术验收已完成：3744143 COMPLETED0:0、311秒，14项集群测试无skip、全部21 merged文件内容SHA及34880逐成员身份/顺序/原三训练report划分、实际GPU两步保存加载和小KNN对CPU参考通过。preflight SHA`5267c7de51efd9c0c868a18ac78be5c283c5bbda5eb02e22bb2b63f33d436ffa`，source membership SHA`702759f136d5c43152fa56b3ca237443e8bd23db1cf9c5d494c6a8487af6bfe5`、source order digest`6e6a81ea8c33fdce5abac26ead544dbc6b00e07ce5814b0fdd7eb838f10b6c7e`、split identity`f4182b224ddabfe323a97417ac1f3824c982f61e4eb87e64c71e72583c2c472b`。实际27939 fit区域/6941 heldout区域；图片仍严格8466/2116，全部类别支持非零。

正式3seed worker PID3941969/1806003/2855854在c2v100m2-7/11/12，实际Slurm child3744161/3744162/3744144；经Slurm overlap只读ps核两慢节点真实冻结命令，D态为共享文件读取等待，未重启或修改参数。随后3个seed的A/B/C和42的两shuffle共11个单元各50个完整数据epoch均完成；不是旧MLP的50×100步，batch512时每单元2750次optimizer.step。3seed作业355/54/337秒，KNN3744145 COMPLETED0:0、23秒。独立startup_validation passed/formal_start_reported=true，SHA`624f9f8575ef6517774b08513ca4302f2a43fd26521efd94aaeb27bf3ff65a06`，核真实日志命令及完成产物SHA；不重复执行或启动通知。当前CPU finalize3744146仍在汇总，不能把已完成训练当全部汇总/最终验收完成。

本批全部完成且独立验收通过：CPU finalize3744146 COMPLETED0:0、126秒；全部6个Slurm计算child COMPLETED0:0，无锁/失败，squeue为空。11个单元均完整50epoch/2750实际optimizer步，batch均512、没有OOM重试；A/B/C参数26901/80661/107541及同seed数据遍历SHA匹配。14本地与14集群测试、真实GPU技术检查、固定成员/图片隔离、预测内容SHA/矩阵求和/类别计数和KNN均核过。audit外层verify_completion_001.py只启动一次，独立验收SHA`517850d2683a5c3f37200f04f84e60b6422b2666161c9b78d72b4e8d49b04833`；最终proof SHA`8eb097631d12d2934c657a11ac992942712523b79cee9570971c309ad0a5116b`。整体Slurm计算墙钟792秒，不累加并行；各线性unit峰值A239.5835/B586.0288/C760.0015MiB，首单元包含冷启动IO，不把GPFS启动延迟当训练算法吞吐。

内部伪标签heldout宏准确率（三seed均值±样本SD）：A Global97.309218±.021533%，B Spatial86.645026±.085896%，C Global+Spatial96.908312±.048711%；C−A均值−.400906点、配对样本SD.060021点，各seed42/123/2026为−.331637/−.433568/−.437512点，0/3正。KNN A92.827106%/B68.475044%/C78.566841%，C−A−14.260265点。多数类由fit确定为BG，heldout overall30.456707%、macro4.761905%；均匀随机期望macro同为4.761905%。不能把这些内部伪标签区域分类指标叫VOC分割mIoU。

seed42真实B86.552335%对B_shuffle14.152390%，相差+72.399945点；真实C96.960683%对C_shuffle96.227999%，相差+.732684点。因此残差有内部类别信息且B依赖一致quadrant identity，不能归为“完全无信息”或Case4。但原四种情形不覆盖B强、C仅在1pp实践阈值内小幅下降的组合，预注册判断inconclusive；线性与KNN的C−A方向均负，不是二者方向相反。证据未显示净macro互补收益，不能把以前SA-GMP失败因果归结为三个scalar gate；本批不自动adaptive gate/rerank/新loss/新seed或扫参。

交换计数提醒：42/123/2026每seed的C正确A错误为50/50/52，A正确C错误为45/46/47；C overall平均98.079047%略高于A98.011814%，但macro下降，说明不能只看全体净纠正数代替类等权指标。指定类别C−A线性准确率均值：car−1.0274、bottle−.1873、sofa+.7286、train−2.2222、bird−.5128、sheep−2.5641、cow+.8032、chair+.5219点；这些均为内部伪标签region分类，不外推VOC GT改善。三个残差方向cos均值约.0041/.0546/.0069，std约.2762/.3428/.3335，没有整体高度共线迹象，但不证明判别信息独立或对global互补。完整逐类norm/cos分布及零向量排除计数保留在residual_stats。

本地experiments/hpc_results/spatial_residual_probe_20261019_231500/verified_20261019已下载核summary.json SHA`cfd333803e2d763566b2dfaf3c72f387b779d587ff7c26f1ad558ebd0afe25c2`、summary.md SHA`860b3574e3c57d3e044e37f34d9bc6c73255e5a221c3575d2e81d9b4ee2bd113`及小diagnostics/验收；只传小结果，无模型或branch数据。analysis_report.md/posthoc_analysis.json及含线性与KNN的294行per_class_complete.csv为独立事后汇总，不覆盖正式CSV/指标。实现、测试、集群运行与归档已完成，本轮没有创建新后台监测，无待删新自动任务；旧第80/81节监测均已验收删除。固定批结束，用户先看summary再决定下一步。

## 83. Spatial Late Fusion v1：冻结空间分类头与原检索概率融合（2026-10-20）

用户附件要求实现并运行独立实验；本回合已明确选“按新附件改用 center 0.4，并新建匹配基线”。新版本`spatial_late_fusion_20261020_002500`，计划`experiments/plans/spatial_late_fusion_v1/{plan.md,protocol.json,request.md}`，独立`tools/experimental_spatial_late_fusion.py`、`spatial_late_fusion_core.py`、`spatial_late_fusion_schedule.py`及26项小测试。生产入口/src/configs、旧实验/库/权重/指标不改；不是将正式主线中心.2改成.4。

来源显式固定第79节原三模型、第81节SA分支缓存、第82节Probe B三已验收checkpoint；注册时核各manifest和独立验收SHA。所有34880成员/10569BG、8466/2116原图片划分保持。三空间head均完整，零新训练、零SAM3/token前向、零新增bank。内部gallery只含27939fit区域/8466图片，6941内部验证区域不进图库或类别中心；原MLP/production balanced K25/top5/search2/中心.4，临时gallery在RAM。完整原34880库仅VOC阶段及两图技术回归消费，不作内部gallery。

内部2116图按SHA256("spatial_late_fusion_v1:"+ID)排序平分1058calibration/1058holdout；每seed各GMP/Spatial scalar温度在calibration上NLL优化，固定数值logT界[-12,12]且最优触边即停不拓界。共用alpha只[0,.05,.1,.2,.3]，三seed calibration macro均值最佳、完全同分较小；温度/alpha先SHA冻结再评holdout。重要监督边界：复用的GMP/Probe B checkpoint均曾用完整2116图选epoch，此次是**选模后的内部伪标签holdout诊断**，只隔离本轮校准/alpha，不称完全未参与选模的独立验证集。

三seed holdout macro均值严格提升且>=2/3正向才自动提交3seed VOC既有1449query分支预测及CPU评测。失败状态complete_no_internal_improvement、VOC NOT RUN，固定停止不追加alpha/gate。通过也只用冻结参数，新建center.4原GMP clean/HCR对照；center.2历史只作像素回归身份控制，不混入效果比较。Fusion使用PF类别/semantic score及margin，SAM3 score融合.2/NMS/raster不变；HCR可信性保持旧GMP前景类别/score.95/margin.1/proposal.4，绝不拿新概率套旧阈值，fused label仅用于分配类别。候选原mask/source_index/order不改，缓存val无sample/class字段不伪造，以query_index/maskhash/原global字节严格绑定。

本地26测试已通过；集群技术、正式内部结果以audit/jobs.json及complete凭据为准，尚未把本地测试当GPU精度。完整链不需电脑在线。内部不读VOC GT；只有通过后全部3×1449新预测SHA提交才decodeGT。报告内部校准/holdout逐类与disagreement、固定margin箱和8重点类；若VOC执行则12逻辑矩阵、21IoU/FP/FN/wrongFG/chair双向/三seed均值样本SD和配对方向。无论结果固定停，不自动部署/扩参，不排障邮件。真实GPFS余>=5GiB，一GPU一worker；CPU仅明确debug提交QOS拒绝无JobID才一次后备单GPU配额但隐藏CUDA。独立最终验收只读已完成小产物，归档SHA及真实JobID后补本节。

本版本已唯一注册提交：technical3744647、scores3744648 array0-2%3、internal3744649 debug8CPU23000M，全部afterok。manifest SHA294546d2985fb894a6c6f4e2172b091072a1b91528afec36b28d9a3dd71c99f1，binding1a646a80ec75bebf0089aeb18fbac5903400c42db80588a8f69d3e6b26d2e4e7。technical已COMPLETED0:0、261秒，26集群测试和真实两图回归通过，technical_report SHA507cdb872e3690af4cd74ab7f6a690f83c2db56f09b954513e108a89f795b18a。seed42和2026已各完成6941查询，123仍在运行；这不是最终融合结果，尚未触发VOC。新代码冻结，后续只按既有依赖推进，不重复prepare/sbatch。

随后三seed scores均已COMPLETED0:0（17/278/152秒），各6941查询、图库27939区域，无新训练/GT。CPU internal3744649已RUNNING，温度校准与holdout结论尚未完成，不把检索完成当实验完成。后台heartbeat spatial-late-fusion已设置15分钟检查本版本，正常等待不通知；完成后只读独立验收/小结果归档、一次通知并删除，不重启旧实验。

本批全部计算已完成。内部3744649 COMPLETED0:0、163秒，按calibration固定候选选出共享alpha=.05；TG按seed42/123/2026为.073873076907/.070475627486/.074428424412，TS为.620187941635/.620062410755/.619596262093。先写parameters_frozen.json SHA`a44647da7a3fa9b5db169824a50e967f768ba68513b33e61bfebf7332160cac8`再评holdout。内部GMP macro三seed均值98.591644%→Fusion98.633823%，+.042179点，配对样本SD.033151点，3/3正向（+.055159/+.004501/+.066875点），满足事先门槛。只净纠正2/1/4个内部区域，无GMP正确→Fusion错误；GMP错Spatial对有15/22/23个，反方向有256/250/256个。属于选模后的内部伪标签证据，不是VOC分割收益或完全独立验证。

固定门槛通过后，既有internal脚本仅一次提交conditional_jobs_001.json：VOC3744774 array0-2%3、CPU evaluate3744775 afterok VOC。debug评测因明确QOSMaxSubmitJobPerUserLimit无JobID拒绝，登记resource_fallback_voc_evaluate_001.json SHA`2c5a97935c67fd888aecf46fd92255f75b3234a1879d4e1ada561d683dbe3054`后唯一后备v100g32一GPU配额6CPU24G/CUDA隐藏，未改变算法或重复提交已获ID任务。三个VOC实际child3744776/3744777/3744774均COMPLETED0:0、899/1173/926秒，CPU evaluate3744775 COMPLETED0:0、273秒。全部9个计算child成功，队列为空、无本批锁/失败；计算整体墙钟00:32:35–01:08:24共2149秒，不累加并行任务。

三seed center.4匹配VOC均值：clean GMP85.663944%→Fusion85.659200%（−.004744点），HCR GMP86.575359%→Fusion86.571568%（−.003791点），均仅1/3正向。clean配对差依42/123/2026为+.119190/−.070512/−.062911点、样本SD.107397点；HCR为+.116794/−.067908/−.060258点、样本SD.104499点。均值基本持平，未验证出稳定净改善，不将这个微小负均值宣称统计显著下降。新center.4原seed42 HCR .8673502115822558与历史.4标量精确，所有新对照每图原production tail独立像素精确，旧center.2历史只核身份回归，不作本批效果比较。

逐类解释仅基于已提交矩阵：clean三seed平均chair+.755615、sofa+.594597点，但cow−1.448670点抵消；HCR对应+.723190/+.591425/−1.448610点。seed42纠正78772个sofa→chair像素，123/2026该方向不变；chair→sofa三个seed均不变。因此平均两类IoU增加主要由42驱动，不称跨seed稳定解决混淆。三个seed clean均新增44048个cow→BG及2479个cat→BG像素，foreground FN均净增46517；平均wrongFG虽减少27624.67，整体仍无净mIoU收益。总矩阵不能证明是单张图/某一proposal。融合PF也改变原实例排序数值，不能把这些变化单独因果归为类别翻转；HCR仍旧GMP可信性阈值，非PF套.95。

3×1449完整预测NPZ（12逻辑预测集）SHA提交后才GT；独立audit外层verify_completion_001.py在c2ln8/PID1927803防重启动一次，最终completion_acceptance_001.json SHA`f94da0648a24d85733af14adcb011026e99461bd188c03080b7564c1901f9a32`通过。它核原冻结代码/来源/3空间head字节、gallery隔离/三score身份、内部冻结/指标/CSV、全部4347预测SHA、12逐图矩阵求和与Slurm；不解码GT/token/模型或重跑实验。source独立proof和controller completion_verification区分保留。另对已下载结果只读核252行VOC CSV精确对应指标、12矩阵GT行支持一致及从矩阵算IoU精确，不重读GT。

本地`experiments/hpc_results/spatial_late_fusion_20261020_002500/verified_20261020`已核下载21小结果/验收，另两独立事后汇总共23项SHA。summary SHA`e4357a52302d5a00632cebe53cbc55b7892663a9a3b403908392492b7f8ae928`、VOC metrics SHA`772a8c542d4f50706f73b84c7a98e9cf2533738de532ec21fe961a6353274d34`、VOC CSV SHA`59643587f5399de4cf793d828ce1f3c3ac5484a4f8af8ca4be67d81b3456a651`；analysis_report.md SHA`0840dbd78f5bf4c4c417b31effdcf83b2e70ac267737857e450f811c9d4f85fb`、posthoc_analysis.json SHA`8c2b548dfa7dc82c6d06223f4c020e88dab926b660a49ea9a5a1154445dd8c33`，不覆盖正式指标/summary。仅归档小结果和分析，无模型/数据/token/日志传输。固定批结束，不自动新增alpha、uncertainty gate、训练、seed或部署；邮件不重要不排障。验收归档及一次结果通知后删除spatial-late-fusion监测。

独立验收和小结果/分析归档均已成功，三份归档SHA已在HPC再次核对；spatial-late-fusion自动监测已通过工具删除。本批完成，无继续实验或邮件任务。

## 84. 最高记录R3配置与最新实验手册的GitHub发布（2026-10-20）

用户在确认最高记录后明确要求“把这个配置更新到github上，还有最近的实验手册也是”。本次据此发布固定R3配置和复现入口，并同步完整手册至第83节及本节；不受第68–69节“不得自动部署/发布”的旧限制阻挡，但不启动新训练、SAM3研究实验或服务器部署。目标为 `https://github.com/nanding0414-spec/1010` 的main分支，保留原 `315de02763df81892fe43d10ff29d700769b2dad` 及全部历史。工作区无.git，使用独立clone审查后正常fast-forward push，不强推、不提交其它checkout内容。

发布配方为 `configs/recipes/r3_record_v1.json`，独立CLI为 `tools/run_r3_record.py`，底层原实验 `tools/oracle_guided_r.py` 字节保持不变，SHA `90dc80fe59ee7666be1b2217966fa24d624aa0cbe82000a6a5e9d9fdee974668`。普通原图入口不静默加R3计算，配置和两个主线入口只将旧GitHub中心默认0.4同步为本地已验证0.2。`src/retrieval.py`增加balanced完整class_scores输出供原worker记录；label/score/margin及排序算式不改，legacy返回None。发布回归测试不等于新的GPU质量验证，精度仍引用第68–69节真实完成记录。

最高非Oracle单seed为seed42的 **86.933659766%**，原匹配基线 **86.821717076%**，+0.111942690个百分点。42/123/2026配对均值：原86.505997478%→R3 **86.607375466%**，+0.101377987点、样本SD0.010105985点、3/3正向。不是把最高单seed当均值，也不混center.4、旧37200成员或后续空间融合结果。已查看VOC开发集，未达到原+0.2点推广门槛；发布是用户本次明确选择，不意味着创新或独立测试达标。原三seedsummary SHA `bd836079a0732da532e5a6c2ac565c3aced3f1e5702662d9575d804c5c7338f3`、metrics SHA `253e13f99cfe28e2245f066b04bebd6d74df2992b19687dfa6ef60b2d804ba5d`、CSV SHA `e725535f4a096d0a92650417c32a0537689a0eb16ce08dfe83b9e5a1f824a24a`，结果标量与证据SHA进入配方；原运行文件不提交。

固定算法：中心80%宽×80%高裁剪，1008 FP32 SAM3，5桶×4个规范VOC类名、无preview，验证presence正常预测；object×presence>.15、mask概率>.5，原图面积100至95%、同生成类NMS .7、精确hash去重后每生成类cap1。原图mask映射严格保留。新mask用原图签名FP32 72×72×1280 token做rho=.1硬中心GMP，不用裁剪token；原MLP2与对应单库、K25/top5/search2、center.2、融合.8retrieval+.2proposal。生成类仅作来源，不替代最终检索类别。只补原clean+HCR背景，proposal≥.4、retrieval≥.95、margin≥.1、相对当前未占用像素比例≥.1，原有前景完整保持。

### 84.1 运行资源与manifest契约

GitHub不含SAM3权重、MLP、原始数据、token、库、proposal、运行日志或私有配置。新安装应提供与配方一致的原始资源；仅下载代码不能得到记录分数。三个seed分别用对应原MLP/单库及同seed的center.2 clean+HCR预测，不训练新模型。若重新生成标签/库或换权重、版本、排序，则属于新实验，不能把该运行直接称历史记录精确复现。

调用方先创建 `experiments/R3_NEW/sources.json`，提供以下真实字段；所有路径用绝对路径，所有SHA取原签名凭据，不填写假值、不通过临时重签改变内容：

- `version`：新的唯一版本；`seed`：42/123/2026之一。
- `sam3_weights`、`source_adapter`、`source_mlp_bank`：原SAM3权重、原MLP checkpoint、含features和meta的对应bank根。配方固定原SAM3权重SHA和原hard-center GMP签名；MLP1280→512→256、两层，bank签名中的adapter SHA必须等于实际checkpoint。
- `token_records`：按完整image ID索引的对象，每项含 `path,sha256,receipt_path,receipt_sha256,geometry`；geometry为 `original_hw:[H,W],token_hw:[72,72]`。receipt的 `binding.encoder_signature` 必须等于source_signature去掉aggregation/adapter，`binding.original_shape` 等于原图HW、sha256等于token记录。NPZ为FP32 `tokens[72,72,1280]` 和 `grid[72,72]`。
- `image_records`、`baseline_records`：同一完整ID集合，各项为 `path,sha256`（可加bytes），分别指原始JPEG和已提交的clean+HCR uint8 PNG。baseline_dir中对应文件名应为 `<ID>.png`，禁止GT或其它seed/center配置充当baseline。

以下示例只冻结已经准备好的资源身份、写新manifest及四个job，不做模型前向或实验评测。`sources.json`的三类记录应已来自原签名清单。1449全量必须用完整同顺序ID列表，不能将小探针当全量。

```python
import hashlib, json
from pathlib import Path

root = Path.cwd().resolve()
run = root / 'experiments/R3_NEW'
src = json.loads((run / 'sources.json').read_text(encoding='utf-8'))
profile_path = root / 'configs/recipes/r3_record_v1.json'
profile = json.loads(profile_path.read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def identity(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

bank = Path(src['source_mlp_bank'])
inputs = [Path(src['sam3_weights']), Path(src['source_adapter'])]
inputs += sorted(p for sub in ['features', 'meta'] for p in (bank / sub).rglob('*') if p.is_file())
m = dict(src, recipe_id=profile['recipe_id'], baseline_recipe=profile['baseline'],
         source_signature=profile['source_signature'],
         code_digest=sha(root / 'release_manifest.json'),
         parameter_digest=sha(profile_path), input_artifacts=[identity(p) for p in inputs])
ids = list(m['image_records'])
assert len(ids) == 1449 and set(ids) == set(m['baseline_records']) == set(m['token_records'])
assert len({Path(v['path']).parent for v in m['baseline_records'].values()}) == 1
base = run / 'attempt_001'
context = base / 'context.pt'
write_new(run / 'manifest.json', m)
for stage in ['context', 'generate', 'query', 'rescue']:
    job = dict(arm='R3', stage=stage, image_ids=ids,
               context_path=str(context), output_dir=str(base / stage))
    if stage in ['query', 'rescue']:
        job['candidates_dir'] = str(base / 'generate')
    if stage == 'rescue':
        job['queries_dir'] = str(base / 'query')
        job['baseline_dir'] = str(Path(m['baseline_records'][ids[0]]['path']).parent)
    write_new(run / (stage + '.json'), job)
```

### 84.2 四阶段CLI与恢复边界

```bash
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/context.json --preflight-only
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/context.json --gpu 1
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/generate.json --gpu 1
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/query.json --gpu 1
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/rescue.json
```

preflight核配方/worker SHA、文件身份、原SAM3权重与bank/MLP签名、输出隔离，不载模型或GT、不重复预扫token。context保存canonical真实语言/邻居/上位类张量与内容SHA；generate逐图额外裁剪前向；query逐图消费原token核内容SHA和几何；rescue逐图核proposal/query ID、hash和上游版本/源码/参数归属，仅写新prediction.png和complete.json。输出为 `<stage>/<ID>/{proposals.npz|queries.npz|prediction.png,complete.json}`。所有阶段使用同一新manifest，不能把旧版本context改binding冒充新产物。完整已完成逐图凭据的相同版本/代码/输入可只读核SHA后复用；半成品保留，新attempt处理，不覆盖历史。

CLI仅为原worker的固定配方包装，不是全流程调度器。调用方负责隔离源码快照、核完整release_manifest SHA、GPU占用和锁、Slurm资源、真实GPFS余量和互斥；本地默认第二张物理GPU，Slurm已有CUDA_VISIBLE_DEVICES不覆盖。每GPU最多一个计算worker，context只运行一次，后续分片互斥且不同时写同图/同stage报告。全部最终预测SHA提交后再运行单独GT评测，明确VOC开发边界；该入口自身无GT、训练、自动作业提交或邮件逻辑。

发布验收使用原worker八项契约、新八项配方/签名/输出隔离/分数元数据契约及既有retrieval/HCR/主线回归。Windows缺torch/OpenCV时不能宣称目标环境全通过；目标环境CPU检查及Git提交状态以本节后续凭据为准。不为这次发布重做GPU精度、参数扫描或研究试验。

源码发布清单为 `release_manifest.json`，列出每个已发布文件的字节数和SHA（清单自身不列入以免循环依赖）。Git仅提交明确列出的代码/配方/测试/手册；忽略数据/模型/NPZ/通知凭据，提交前定向检查敏感字段。运行资料只留本地和集群对应目录，不自动ZIP、不修改生产/冻结快照。

目标环境发布CPU回归已完成：Slurm debug作业3750333，c21507，COMPLETED 0:0、90秒；36项测试全部通过，无skip。该检查只使用隔离的187个源码/配置/测试文件及环境依赖，不解码GT、加载研究权重或重跑SAM3精度。Windows同配方18项中17通过、1项因无torch跳过，以目标环境完整36项通过为准。发布入口与底层原worker的验证范围已分开记录。

## 85. R3 Cross-View Semantic Agreement：固定三seed跨视角一致性扩展（2026-10-20）

用户附件明确授权实现并运行r3_crossview_agreement_v1；本轮零训练，不修改第84节正式R3三文件、原SAM3/MLP/库/GMP/center.2/HCR/提议/裁剪比例。版本r3_crossview_agreement_20261020_171000；计划与原始请求在experiments/plans/r3_crossview_agreement_v1，运行目录experiments/r3_crossview_agreement_v1、冻结源码hpc_launch_audits/该版本/code、新缓存storage/r3_crossview/该版本、日志logs/r3_crossview_agreement_v1。完整规约见plan.md与protocol.json。

旧R3没有crop tokens，每图仅重放一次原80%中心裁剪前向，并同时通过只读hooks捕获该forward的trunk1024/stride14-FPN256，按原layernorm/分块L2/concat生成FP32[5184,1280]。同一前向内的native crop mask计算crop GMP，mapped mask仅对应原图GMP；所有proposal数组/分数/ID/顺序与旧缓存逐元素精确才接受。原图raw GMP只读复用旧签名缓存，三seed分别用自身原MLP及34880成员/10569背景单库，balanced K25/top5/search2/center.2明确传参。技术两图新缓存安全复用进入正式池，不能把探针当1449完成；不得额外重复crop encoder。

A严格独立重放当前R3并逐图核历史像素，提交历史PNG原字节；B使用crop严格gate；C保留A原排序，仅增加crop同类/严格gate；D无条件完整保留A，再按.8min(score_o,score_c)+.2proposal排序/candidate_id平局，处理原A未接受且共同前景类的候选，两view score>=.90、margin>=.05、proposal>=.4、novel>=.1。Novel使用A和之前extra之后的实时背景，另记录固定A后的潜在novel。保存FP32值按Python float阈值语义，与原严格R3一致。不用prompt类当预测、不平均分数生成第三类、不改mask/阈值、不重训。

固定链technical1GPU→extract4分片→predict3seed→primary CPU→条件shuffle3seed→finalize CPU。全部12×1449实际PNG与分数/owner/21类分数SHA提交，无GT分数区间/margin/21类agreement诊断先完成，才解码GT。报告三seed均值/样本SD/方向、21IoU、FP/FN/wrongFG、2000共享图片bootstrap（不代替seed方差）；重点原score[.90,.95)新增数量/质量/非可加mIoU反事实。precision_all分母含ignore且ignore不当正确，另报valid-only。最多20有益/20有害独立image/candidate八面板，GT仅最后一栏事后可视化，支持不足如实记录。

只有D三seed均值严格>A才固定shuffle_seed42在共同候选ID顺序上置换crop完整joint rows，三个MLP seed共享同一双射，保留score/margin/class联合分布，普通随机置换允许自映射并记录个数；query和A不动，D-shuffle同规则。全部3×1449 shuffle PNG提交后单独评GT，不重复primary GT/bootstrap。无正均值写skipped_no_positive_mean_signal，固定停止，不自动adaptive/multiscale/score fusion/扫描/部署。

本地16项纯NumPy策略/几何/诊断/Bootstrap测试已通过，另5项torch/真实接口契约须在目标环境完成，不能声称Windows全部21通过。目标technical还须真实两图候选逐元素精确、同前向tokens与原fusion tail精确、少量原图GMP对旧raw数值及三seed×两图A像素回归。首次四extract实际进程与首批缓存SHA/FP32/native几何有效后独立写startup_validation一次；job提交不等于正式1449验收。最终独立验收与controller verification区分，核源码/输入/完整产物SHA、A历史逐图矩阵、12或15组逐图求和和真实Slurm退出。只在必要最终验收流式读新crop tokens，不重扫12031旧token。

每GPU一worker、仅Slurm研究；真实mmlsquota余量先满足40GiB缓存预算+5GiB保留，瞬时quota最多3次，不用df替代。失败保留attempt/具体日志，只取消自有未启动依赖；同版复用须源码/参数/输入/完整产物SHA全同且旧worker退出锁释放，源码修复新版本小probe。用户暂停取消立即停本批和监测，不自动恢复。新结果/小验收下载核SHA，不自动GitHub发布新实验，不打包，不排障邮件。实际manifest/binding/JobID及正式验收后补。

第85节已唯一注册并提交：technical3752041、extract3752042 array0-3%4、predict3752043 array0-2%3、evaluate3752044 debug CPU、条件shuffle3752045 array0-2%3、finalize3752046。六个afterok任务均有JobID，不重复prepare/sbatch。manifest SHA`e35f72452327e2e2f1d7246aec458434465a877a20b12d9a7942d8f1ab69d88d`、binding`96b9d8092d6bb43a8d66e9001050710c6c3c9955f06a7f343f875c5092311d44`，audit/jobs.json为实际命令依据。shuffle/finalize原CPU请求被QOSMaxSubmitJobPerUserLimit明确拒绝且无ID，记录各resource_fallback_001后唯一v100g32配额/CUDA隐藏CPU后备，不是方法失败。

technical3752041已COMPLETED0:0、202秒，21项目标契约无skip通过；真实两图2007_000033/2007_000042历史proposal及score逐元素精确，capture token对原feature fusion tail bitwise/maxabs0，原图GMP首候选对旧raw也bitwise/maxabs0，全部三seed×两图A像素精确。technical_report SHA`9f17b89c6be8b8469186854d26d8fe5c43474075e6ffd86c3d59bba75f214ef1`。这是技术回归，不是1449精度结果。两图cache按同binding复用，没有重复crop encoder。

四个extract实际Slurm/proc命令、各卡唯一UUID和首批FP32[5184,1280]、native/mapped mask、raw_crop/raw_original身份及token内容SHA已通过独立启动验收。startup_validation status=passed、formal_start_reported=true，SHA`412ede3330ba8101b447898832f00abca9a3702cccf9dad9b8fb3e349af55772`，已下载核SHA并向用户通知，不重复。启动审计001曾因最后array child实际JobID等于array root而squeue显示全家多行RUNNING，仅返回NOT_READY、未写验收；002改用scontrol指定ArrayID_TaskID，旧001保留，未修改算法、冻结代码或重新提交任何作业。verify_completion_001.py已准备并上传audit外层，完成后仅运行一次必要独立审核。

后台监测r3已通过工具创建，每15分钟检查本版本，正常阶段不通知；完整结果和独立验收归档后一次通知并删除。所有科学阶段及条件control已Slurm衔接，电脑离线不影响计算；监测不额外启动新实验。冻结正式R3文件和GitHub发布保持此前版本，当前新实验仅本地/HPC独立实现。

本批13个计算child现均COMPLETED0:0，squeue为空，无本批锁或failure。三seed A/B/C/D各1449预测及条件D-shuffle各1449预测、primary和finalize均完成，A历史4347张像素及逐图矩阵精确。controller汇总D均值86.864528%、相对当前R3 A均值86.607375%提升+0.257153pp，三seed均正向；D-shuffle均值86.614286%，D相对shuffle +0.250242pp。此处为已完成计算的汇总，尚不能替代独立验收。已确认无旧audit ledger/guard/accept后唯一提交只读CPU审计3752323（debug2CPU8GiB），audit_job_001.json为依据；不得重复审计或重跑实验，待completion_acceptance_001.json后归档最终凭据。

**最终独立验收已通过。** 审计3752323 COMPLETED0:0、729秒，completion_acceptance_001.json SHA`d02e727537b161665826ff1c5d60e024bd4f2153dc5e67b2c7790a637c4f4002`。13个科学计算child全部COMPLETED0:0；17388张primary PNG和4347张shuffle PNG、15组1449逐图矩阵求和、A全部历史PNG字节/逐图矩阵、CSV/指标/汇总及原模型库SHA均精确。一次流式核完1449份新crop token共35610554821字节，均FP32[5184,1280]，未重扫旧token、未解码GT或模型。正式计算整体Slurm墙钟3055秒（50分55秒，取最早start至最晚end，不累加并行），提取峰值5210.633MiB；新训练0次。

最终三seed（42/123/2026）mIoU(%)：A=`86.933660/86.413496/86.474971`、B=`86.928481/86.414984/86.441859`、C=`86.872904/86.379555/86.478806`、D=`87.104514/86.788614/86.700457`、D-shuffle=`86.942550/86.419541/86.480767`。均值依次`86.607375/86.595108/86.577088/86.864528/86.614286`。D-A分别`+0.170854/+0.375118/+0.225486`pp，3/3正，均值`+0.257153`pp、配对差样本SD`0.105750`pp，D绝对mIoU样本SD`0.212457`pp；D比shuffle均值`+0.250242`pp且3/3同向。2000次共享配对图片bootstrap的三seed均值区间：D-A `[-0.079721,+0.635889]`pp，D-shuffle差反向为Real D−shuffle `[-0.088523,+0.630099]`pp；均跨0，不能把三seed同向或shuffle差当独立测试/统计显著性已证实。

D每seed实际新增49/50/53个候选，平均前景FN减少264894.333像素，同时背景FP增加87031.333、前景错类增加17587.333；valid-only新增像素precision为63.6087%/81.0815%/65.7079%，含ignore分母口径为51.0974%/63.5578%/54.9038%。原score[.90,.95)实际新增46/46/48个，单独保留其实际owner像素的非可加反事实提升分别+0.168597/+0.374601/+0.237595pp。主要类IoU变化：sofa+1.665418、car+1.497007、pottedplant+1.496898、bus+1.082595、bottle+0.562226pp；chair−0.507647、table−0.407957pp，bicycle仅+0.001071、motorbike0。不能据此称已解决椅子/沙发混淆或细结构问题。两view均过匹配宽松置信门槛后，三seed灰区disagreed支持均为0，precision保留null；没有直接证明agreed比同条件disagreed更可靠，也没有单视角宽松门槛的匹配分割对照。固定shuffle支持真实对应有开发集信号，但不能把全部收益单独归因于top1 agreement。

已下载55个小结果/验收/40张八面板图到`experiments/hpc_results/r3_crossview_agreement_20261020_171000/verified_20261020`，下载均核SHA；未下载token、模型或原数据。summary SHA`35848d919513f93f0ba681ec51f96556147da0741c243c012fb0285a3b8c753e`，metrics SHA`33d27d6b31a9b13a8e6896f34ca5f11b5913bc62d8b66f894fe6160f9b5b8b04`，per_class SHA`8238733340acd5413238f6a6a0dd30d4bbaae8d11a95fc8b1197b7c7632fb232`。独立analysis_report.md SHA`8bbaf9f9a2dd4946e532ff78ae696039696a7c25540d9cd01b2f9b4d3fb8fbfa`、download_verification.json SHA`c10ed0834a7688485c2a3a5933da78f974c97747def3848c39a6bdbd871a82d5`、local_archive_001.json SHA`d4401dd5192a1fb72bc912ff1da2aae854a666890926bb55d6adbb26bf1eaf8f`已小文件回传HPC结果目录并核SHA；分析未改正式指标，未重读GT或重做bootstrap。已抽查有益/有害首例八面板。固定批结束，不追加实验、扫描或部署，不自动发布GitHub/打包/排障邮件；正式R3仍按第84节版本，完成通知后删除r3监测。

最终归档状态为`complete_verified_archived`。后台`r3`监测已通过工具删除（deleteStatus=deleted）；本批一次完成通知后不再自动唤醒或追加计算。

## 86. R3 Cross-View Mechanism Control v1：单视角loose与实际预算匹配（2026-10-20）

用户附件明确授权继续实现并运行本批机制对照，目的仅区分第85节D收益与单纯降低原视角门槛的关系。版本`r3_crossview_control_20261020_200000`，计划/原请求/规约在`experiments/plans/r3_crossview_mechanism_control_v1`，独立入口`tools/experimental_r3_crossview_control.py`，新run/log/audit/storage均使用该新版本目录。正式R3三文件、生产src/configs和第85节冻结代码/旧指标只读，不改主线、不自动发布。

A/D引用已验收的6套旧参考；新增E/E_count/E_area各3seed共9套1449张PNG。只读消费原候选mapped masks、各seed自身已保存的original/crop完整21类检索分数、A/D PNG/owner及匹配模型/库签名，不训练、不重建库、不读token或解码模型、不重做SAM3。旧缓存缺失或SHA/身份不一致安全失败，不自动补生成。固定center.2、rho.1、K25/top5/search2、34880成员/10569BG、原HCR和proposal几何。

E只使用原视角前景top1、proposal>=.4、original score>=.90、margin>=.05，排除A实际accepted；按FP32(.8*original score+.2*proposal)降序/image_id及candidate_id稳定平局。保存FP32按Python float阈值语义，依次只填A及之前extra后的实时background，新增>0且novel>=.10。全1449图每seed唯一全局候选顺序，其逐图内相对顺序与普通E相同。

E_count取这个固定顺序中前49/50/53个**实际接受**（对应42/123/2026），不计因重叠而novel失败的eligible候选；耗尽不足直接报错保留attempt，不能静默缩小。E_area在相同实际接受序列的自然前缀中选择累计新增面积最接近318574/444479/540285的停止点，包含空前缀，完全平局选较短前缀；不组合搜索、不切最后mask、不用GT选点，报告实际面积差。D eligible必须为E eligible子集，另区分D/E actual ID集合差异，不能将遮盖引起的actual差异误判为语义子集失败。

先核全部13041张新PNG/NPZ/owner及旧A/D引用SHA，写无GT预算、ID集合和prediction commitment，再解码GT评测。输出5方法15逻辑矩阵、21IoU/误差/三seed均值样本SD和配对方向，7个固定比较使用同一2000次image-ID paired bootstrap；保留其条件性，不代替训练seed波动。extra像素按同旧口径分correct FG/BG FP/wrong FG/ignore，给出效率指标、21×21 transfer及D减两matched control的差矩阵。重点7类及chair/table伤害是否为loose共性；GOOD>=.8、BAD<.5、MID其余，valid=0记unavailable，score/margin/21类分数余弦仅事后诊断，不产生Quality Gate。

本地21项CPU测试通过，覆盖动态novel/真实count/面积prefix与平局/不可覆盖A/FP32 decimal语义等；这不是集群正式结果。计划CPU Slurm technical→predict三seed array→evaluate，technical另做2图×3seed真实缓存A/D精确和D规则回放、原模型库SHA。只有debug明确QOS无JobID拒绝才一次CUDA隐藏GPU资源配额后备，不重复已有提交。真实GPFS余>=5GiB，不df、不扫旧12031 token。失败先日志保留attempt，只取消自有未启动依赖；源码改动新版本小探针，用户暂停立即安全停本批/监测不自动恢复。

完成后一次独立CPU audit核5个计算child退出0、全新PNG/NPZ/矩阵与CSV SHA、A/D旧逐图精确、extra transfer守恒、count精确和area最近前缀；不读GT/token或解码模型。小结果下载核SHA、更新本节并一次通知后停监测；不自动Quality Gate/Adaptive/MultiScale/阈值/新seed/训练/部署/GitHub/打包，邮件不排障。实际manifest/binding/JobID与首次集群验收后补，尚未提交时不得称后台已运行。

本批已唯一注册并提交：technical`3753082`、predict`3753083` array0-2%3、evaluate`3753084`，全afterok衔接。manifest SHA`f216de3c99cbeb912fcbf8db0c2d0e63cc2b13bb3f78038f3a7bb6295f832ec9`、binding`4320873f6bb241346f9a2f3214a370db4890536873a8a09734ff611ee82a9af9`；实际命令以audit/jobs.json为准，已下载核SHA的registered_jobs_verified.json为`9434f0ce3a03be87b61013c5539acca4a95836b42caf7e949d0900c6e3aea2ce`。上传11个新文件/控制器均先对本地SHA精确核对后才注册，旧4个CPU基础模块按第85节冻结code SHA复制，无源码覆盖或token预扫。

technical使用debug2CPU8GiB；predict/evaluate原debug请求均被QOSMaxSubmitJobPerUserLimit明确拒绝且无JobID，分别记录resource_fallback_001后唯一v100g32配额6CPU24GiB后备。后备占GPU资源配额但CUDA始终隐藏，科学计算仍是CPU，不能称零GPU资源分配；不是方法失败，不重提已有JobID。后台新机制监测已创建（automation_id=r3，上一轮同名监测此前已删除，本次为新prompt/新版本），正常推进不通知，完整验收归档后一次通知并删除。首次technical和真实三worker验收尚待完成时，不把提交/探针当正式9套结果。

归因边界：D排序用min(两view score)，E排序用original score，gate也不同；即使D优于matched controls，也首先支持完整跨视角选择策略含有增量信息，不能单独把优势因果归给top1类别一致。全部D/E匹配预算来自无GT实际预测变化，后续GOOD/MID/BAD只作事后诊断。

首次正式启动已验收：technical3753082 COMPLETED0:0、171秒，21集群CPU测试无skip通过，2张真实图×3seed的原A/D PNG/NPZ及D extras独立回放精确，原模型/库SHA不变。technical_report SHA`f623d820a69157a1ff8a69445574d64ea386f971fcdc0547d3950142a8fcbb49`，真实mmlsquota余235009359872字节，已小凭据下载核SHA。三个predict实际child为3753091/3753092/3753083，PID482160/482161/1447671；经scontrol指定ArrayID_TaskID和自有Slurm overlap核实际冻结入口/manifest/seed命令，首份原FP32 21类score缓存身份及SHA通过。startup_validation status=passed、formal_start_reported=true，SHA`30fc2c7db6f17fc30710409af88392f5ac9cdce8f6d9603f8d9e0abb3706563f`，已本地下载核SHA。

首次验收仅处于loading_signed_inputs，当时三个新worker均在GPFS读取（D状态），尚无新PNG；不能把来源首缓存或技术probe当本批9套结果。依赖评测继续等待，正常读等待或排队不重提作业。r3监测prompt已更新为本版本真实JobID/签名、formal_start_reported=true及当前边界，不重复启动验收/技术检查/启动通知。后续完整验收归档及一次结果通知后删除此新监测。

### 86.1 最终独立验收与机制结论

独立审计3753472已COMPLETED0:0、21分49秒。completion_acceptance_001.json SHA`702461204e6cef624c685a4d8b606cc79503380af61858fac26e7cdf8b930040`，status=passed。5个科学计算child均COMPLETED0:0；13041张新PNG及对应NPZ/owner、15组1449逐图矩阵求和、A/D历史引用和逐图矩阵、CSV/SHA、extra transfer守恒、原模型与库SHA、E_count实际接受数、E_area最近自然前缀均核验通过；无本批lock/failure。验收没有解码GT、token或模型。所有新预测SHA提交先于本批GT读取，未重复GT或bootstrap。

本批0次训练、0次SAM3前向、9套新预测和6套旧A/D参考。科学计算Slurm墙钟2782秒（46分22秒，最早start至最晚end，不累加并行）；只读独立审计1309秒另列，不混为科学计算时间。predict峰值CPU RSS为2132.398MiB，evaluate峰值526.332MiB。后备任务占GPU资源配额但CUDA隐藏，科学计算均在CPU。

| 方法 | seed42 mIoU(%) | seed123 | seed2026 | 均值(%) | 相对A(pp) | 绝对mIoU样本SD(pp) |
|---|---:|---:|---:|---:|---:|---:|
| A | 86.933660 | 86.413496 | 86.474971 | 86.607375 | +0.000000 | 0.284237 |
| D | 87.104514 | 86.788614 | 86.700457 | 86.864528 | +0.257153 | 0.212457 |
| E | 87.183731 | 86.780938 | 86.681601 | 86.882090 | +0.274715 | 0.265908 |
| E_count | 87.123516 | 86.641867 | 86.621612 | 86.795665 | +0.188289 | 0.284108 |
| E_area | 87.141796 | 86.637758 | 86.731329 | 86.836961 | +0.229585 | 0.268109 |

| 比较 | seed42差(pp) | seed123差 | seed2026差 | 均值差 | 正向seed | 配对差样本SD(pp) | 95%配对图片区间(pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| D-A | +0.170854 | +0.375118 | +0.225486 | +0.257153 | 3/3 | 0.105750 | [-0.079721, +0.635889] |
| E-A | +0.250071 | +0.367442 | +0.206630 | +0.274715 | 3/3 | 0.083190 | [-0.142747, +0.716378] |
| E_count-A | +0.189856 | +0.228371 | +0.146641 | +0.188289 | 3/3 | 0.040888 | [-0.126032, +0.552377] |
| E_area-A | +0.208136 | +0.224262 | +0.256358 | +0.229585 | 3/3 | 0.024548 | [-0.103085, +0.590571] |
| D-E | -0.079217 | +0.007676 | +0.018855 | -0.017562 | 2/3 | 0.053687 | [-0.150011, +0.103406] |
| D-E_count | -0.019002 | +0.146747 | +0.078845 | +0.068863 | 2/3 | 0.083324 | [-0.097764, +0.263055] |
| D-E_area | -0.037282 | +0.150856 | -0.030872 | +0.027568 | 1/3 | 0.106819 | [-0.102266, +0.150525] |

**机制结论：原视角门槛放宽已足以复现之前整体提升的量级；跨视角策略的稳定额外mIoU优势尚未建立。** E均值高于D 0.017562pp；D相对两个matched controls仅+0.068863/+0.027568pp，分别2/3、1/3 seed正向，区间均跨0。D与E同时改变gate和排序，不能隔离top1 agreement的因果贡献，也不能将mIoU差拆成可加的“阈值贡献比例”。2000次共享image-ID bootstrap只描述固定模型在已查看VOC开发图片下的条件波动，不替代training-seed方差或独立test。

E_count实际接受数严格49/50/53，与D一致，但新增面积306947/427045/429379，对D差−11627/−17434/−110906。E_area实际前缀51/52/58个、面积323296/433165/557076，对D面积差+4722/−11314/+16791；这些是无GT、包含空前缀且平局取较短的最近自然停止点，不能写成完全等面积。完整E实际接受61/66/61个、面积513974/689458/578105。D semantic eligible为E eligible子集；actual集合交并差另外保存在candidate_sets.json，避免将不同排序遮盖误判为语义子集失败。

新增像素valid precision三seed等权均值：D70.1327%、E66.3209%、E_count61.3978%、E_area64.5784%；含ignore分母的precision_all依次56.5197/53.7920/47.0536/50.8601%。D有较保守的选择迹象，但不是每seed都更好。D对E平均BG FP少55703.667、前景错类少1115.667，同时FG FN多73066；相对E_count则BG FP少9816.667、FN少65263、错类少1048.333；相对E_area BG FP少16541.667、FN少22579.667、错类少1048.333。FG FN是前景被判背景的口径，新增错误FG也可减少此FN，须与正确FG及错类分开看。Net valid benefit仅诊断，不替代mIoU。

| 重点类 | D-A IoU差(pp) | E-A | E_count-A | E_area-A |
|---|---:|---:|---:|---:|
| sofa | +1.665418 | +2.319861 | -0.045970 | +0.751848 |
| car | +1.497007 | +1.576820 | +1.576849 | +1.576767 |
| pottedplant | +1.496898 | +0.598590 | +1.859589 | +1.717266 |
| bus | +1.082595 | +1.082595 | +1.082595 | +1.082595 |
| bottle | +0.562226 | +0.562226 | +0.365847 | +0.565194 |
| chair | -0.507647 | -0.642730 | -0.442882 | -0.442882 |
| table | -0.407957 | -0.407957 | -0.389515 | -0.395702 |

D对matched controls的优势集中于sofa：该类D−E_count为+1.711388pp、D−E_area为+0.913570pp。去掉最大正向类sofa后的其余20类IoU差均值为−0.013263/−0.016733pp。这是已保存逐类IoU的事后贡献诊断，不改官方21类指标、不重读GT、不筛候选，不能称普适改善。chair/table在全部loose control均下降；D对matched controls也没有保护这两类。

D新增正确FG最大三seed平均转移为sofa→sofa 79445.333、car→car 56330.333、bus→bus 47899像素。主要错误转移为background→table 28883.333、background→pottedplant 25248.667、background→chair 18331.333、sofa→chair 16945.667。完整21×21及D减matched差矩阵见extra_pixel_transfer_*.csv，ignore单列；不能把此前已有前景的错误说成已被背景填充修正。

D实际接受候选三seed合计152条，GOOD51、MID1、BAD85、无valid支持15。同一候选可跨seed重复，不是152个独立训练/图像重复。GOOD/BAD原score均值0.939442/0.932772、crop score0.947112/0.944862、21类score余弦0.971279/0.973091；只是条件于已接受候选的事后描述，不能据此反向设阈值或声称简单quality gate已验证。

已下载并逐SHA核24份小结果/CSV/bootstrap/独立验收到`experiments/hpc_results/r3_crossview_control_20261020_200000/verified_20261020`；不下载token、权重或大量PNG。summary SHA`cadd919962f42f0e079318c764fa4f25e9fba27d5cb0a7fc7f518c57ebdc66a9`，metrics SHA`3143485c7a73199225ce256cb42a7ad886fa7bf7bda72bccb71156c1c9e07a40`。独立`analysis_report.md`进一步列出类贡献、最大转移和质量分组；本地分析入口`experiments/plans/r3_crossview_mechanism_control_v1/analyze_downloaded.py`，小分析回传入口`archive_results_001.py`，均在冻结科学snapshot外，不重算实验或覆盖正式指标。回传核SHA及监测退役凭据在本节末追加。

第86节最终归档状态为`complete_verified_archived`。analysis_report.md SHA`a9fff66dff7f995b827718c88fff42f76878bf55fd12c138114837268395b068`、download_verification.json SHA`4689edeea6f301e538a9a01b7cce5eb2c088593bf2909049ec021b13541b31c2`、local_archive_001.json SHA`d07a31f51f1b281cc5014c4c510bba7f26043893f88fc9754cadfca6ffe72706`已回传本批HPC结果目录并逐项核SHA。报告只分析已保存的小指标和CSV，没有重读GT、token或模型，没有重做推理或bootstrap。

后台`r3`监测已通过automation_update工具删除（deleteStatus=deleted）。固定机制批在一次最终通知后结束，不自动追加Quality Gate、Adaptive/Object-Centric/MultiScale、扫描、训练、部署、GitHub发布或打包；用户已说明邮件不重要，未做邮件排障。原正式R3配方及旧结果保持原版本。


## 87. Retrieval-Guided Adaptive Re-observation v1（2026-10-21）

用户附件明确授权实现并运行固定RGR-v1批次。版本`rgr_adaptive_reobservation_20261020_215500`；设计标签保留最初实现时间，实际提交时间以Slurm ledger为准。独立入口`tools/experimental_rgr_adaptive.py`，协议/原请求/操作脚本在`experiments/plans/rgr_adaptive_reobservation_v1/`。当前本地38项CPU测试通过；截至本条写入时尚未上传/注册/提交新任务，不能据此声称真实SAM3或精度通过。

E只读复用第86节独立验收结果，三seed42/123/2026分别87.183731/86.780938/86.681601%，均值86.882090%。F=E后按original full-image raw候选检索选最多2个object-centric crop；G逐图逐seed逐rank匹配F的crop数量/宽/高，仅SHA→PCG64位置不同。seed固定FG、retrieval>=.90、margin>=.05、.15<=proposal<.40、相对E novel>=.10，按retrieval-proposal gap、margin、retrieval、原ID排序，source mask IoU>.5去重。bbox扩1.5、最小宽高比例.30、碰边平移保尺寸、整图crop跳过后继续。随机位置函数不读取F中心/bbox/类别/GT；偶然相同位置不重抽。

每crop只运行一次冻结SAM3 encoder和完整20类/五桶，1008 FP32、原context expansion、正常预测presence；原R3实际映射后area/NMS/hash/per-source-class-cap参数不改。crop只生成几何，不用crop token作语义分类。mapped新mask统一用原图签名72×72×1280 FP32 token→原hard-center GMP rho=.1→对应seed原MLP→34880成员/10569BG原单库，balanced K25/top5/search2/显式center=.2。新候选按E式proposal>=.4/retrieval>=.9/margin>=.05、FP32(.8retrieval+.2proposal)/ID，eligible-only hash及semantic-class NMS.7后实时novel>=.1，只填E背景。E已有FG不可改。0训练/0新库/0伪标签/0原图SAM3前向；新增F/G共6套1449 PNG，上界17388次crop forward，实际按无GT计划确定，技术crop完整缓存同binding正式复用，不额外造probe crop。

集群运行目录`experiments/rgr_adaptive_reobservation_v1`；提议/原图查询缓存`storage/proposals/voc/rgr_adaptive_reobservation_v1/版本`；日志`logs/rgr_adaptive_reobservation_v1/版本/%A_%a.out`；冻结snapshot及真实ledger为`experiments/hpc_launch_audits/版本/{code,jobs.json}`。依赖链输入technical单GPU→inventory四GPU按图片分片→实际计划crop technical单GPU→observe四GPU每worker处理独立三seed→CPU evaluate。研究计算仅Slurm，每GPU一worker，真实mmlsquota余>=5GiB。CPU debug仅明确QOS且无JobID拒绝时唯一GPU配额/CUDA隐藏后备；绝不重复prepare/sbatch。原正式R3三个文件记录保护SHA，不修改生产或旧实验。

F/G全部8694新PNG、owner/NPZ、E引用、无GT seed/crop预算/trace核SHA提交后才读取GT；9逻辑矩阵逐图求和，E像素/每图矩阵精确历史。输出21IoU、FP/FN/wrongFG、三seed均值样本SD/配对方向、2000共享image bootstrap F-E/G-E/F-G、yield/像素precision/每forward正确与净收益、21×21transfer及ignore、gap三分位实际owner非可加反事实、F-G去最大正向类敏感性、随机30/有益20/有害20多面板（不足如实记缺；GT最后栏）。GT只事后，不改seed/位置/阈值/预算。F-G只检验在F触发图片/数量/尺度给定条件下的位置选择；gap不是共尺度概率差；图片区间不是seed方差或独立test证据。

结束须唯一独立Slurm审核`verify_completion_001.py`，核源码/模型库字节SHA、6套新PNG、9矩阵/CSV、F/G每对预算、owner/transfer、11个科学child均COMPLETED0:0，不读取GT/token或重算方法。controller verification不冒充independent acceptance。只下载小结果和必要可视化核SHA，更新本节；固定批完成即停，无自动扫描/新crop/新seed/训练/部署/GitHub发布/打包。邮件不排障。


### 87.1 实际注册与调度（2026-10-21 北京时间01:47起）

manifest SHA=`83964b728cb108411316446047ff9837e8993da071234f691642e40aa1a4c847`；binding=`cedf4a09e52bc56cb2bc3d4f53d887d8f5eeab8971765285ad2d9d8266ac2f8c`。真实`audit/jobs.json`已登记：technical_inputs3755242、inventory3755243 array0-3%4、technical3755244、observe3755245 array0-3%4、evaluate3755249（debug8CPU23000M/2h），全afterok。不要再upload/prepare/sbatch。首次CPU4h请求明确time limit拒绝且无JobID，读`MaxTime=02:00:00`后仅外层`submit_evaluate_resource_recovery_001.py`修资源时间并成功提交；`jobs_before_evaluate_recovery_001.json`、`controller_failure_001.json`及`evaluate_resource_recovery_001.json`保留，不属于科研方法失败，冻结code/manifest未改。

上传末尾校验首次因Windows反斜杠拼入Linux路径而失败，所有14个新文件已经实际上传；仅修本地校验路径为POSIX并核SHA，不重复上传。注册前另核发现HPC主目录三个正式R3入口均不存在（此前发布与服务器部署分界），故`pre_registration_recovery_001.json`记录源码注册前修正：保护已有路径SHA并记录原先缺失路径，不在HPC部署生产入口；保留before副本，无JobID/算法规则变更。注册后当前code冻结。

本节最后检查时technical_inputs为RUNNING，真实节点c2v100m2-19，其余依赖等待；这只是技术阶段，不是正式crop验收或1449评测结果。还须核目标38项测试/真实原raw GMP与三seed E回归、inventory后实际计划F/G crop探针、四observe worker实际Slurm/proc/GPU和首产物SHA后，唯一执行startup验收并通知。后续正常推进静默，最终完成/失败/需操作才通知。


### 87.2 当前有效恢复版（2026-10-21 北京时间01:55）

初版3755242在38目标CPU测试通过后，原分类器import阶段因快照遗漏`project_utils`依赖FAILED；未产生新SAM3 forward、crop、预测或GT评测。其余初版3755243/44/45/49均由自有失败处理取消、0秒，旧snapshot/manifest/attempt/failure保留，不是方法质量负结果。禁止重启初版。恢复版从已验收旧source按SHA补齐`project_utils/`和`configs/`依赖（218签名源文件），不改旧snapshot/生产，科学core和固定参数不变；本地canonical方案已指向新版本，旧本地控制文件保存在plan/history/初版目录。

当前有效版本=`rgr_adaptive_runtime_recover_20261021_015500`，run=`experiments/rgr_adaptive_reobservation_v1/该版本`，storage/log/audit均用该新版本。manifest SHA=`583a36ac7ee303adcfcbf1de6b26f7f5b537438bf10961463d47237747181109`，binding=`06dab8108ab6a130c788769f47c0ed15508103e6b4f446e428fbd30a36a0b44f`。audit/jobs.json：technical_inputs3755271、inventory3755272 array0-3%4、technical3755273、observe3755274 array0-3%4、evaluate3755275 debug8CPU23000M/2h，五任务全成功提交afterok，不重复prepare/upload/sbatch。所有控制session已返回；还需等待本恢复版真实技术和四worker正式启动验收，不把旧38项CPU或资源提交当SAM3质量。

本恢复版38项本地测试仍通过，原图raw/GMP/E的针对GPU探针会完整运行，crop technical仅用本批实际已计划crop缓存，F/G预算和0训练等边界均不变。后台rgr监测继续本固定批，正常排队推进静默，正式startup/完成/失败或需行动才通知。最终独立审计与下载脚本已同步指向恢复版，需按其manifest/ledger核11科学child，不套初版签名或JobID。


恢复版输入technical3755271已COMPLETED0:0、56秒；report SHA=`7a30353b48baa89af8908691d793cbdcca3b656e9eb557c24ae29e1768265928`。38目标CPU测试无skip，真实两图2007_000033/2007_000042×三seed E PNG/NPZ及独立规则重放全像素精确；签名原图token上的GMP重算maxabs分别8.940696716308594e-8/0，第一图非bitwise、第二图bitwise，不能声称全体重算逐元素一致。原raw查询FP32有效；无GT/新训练/原图SAM3，峰值90.4995MiB，真实mmlsquota余234102996992字节。报告已下载核SHA，本地launch_receipts/恢复版/input_technical_verified_001.json SHA=`8b2a6806ee0d38f6b3350c203dfcf54e739ceba9771c84a1ee56f9a6e177843f`。inventory已至少三个child真实RUNNING，另一个等资源；这里只是输入探针/计划阶段，尚未完成新crop SAM3 probe、startup正式四worker验收或F/G1449精度。


### 87.3 六套预测完成；评测诊断修复版（2026-10-21 北京时间10:31，当前优先）

恢复版的10个GPU child均COMPLETED0:0：输入probe56秒，inventory四child11分07/15分05/15分26/11分23秒，真实crop technical9分33秒，observe四child44分22/43分25/38分10/45分04秒；新增六套共8694 PNG/NPZ/owner已完整生成。实际每arm crop数seed42/123/2026为611/634/625，F/G逐图逐rank次数与宽高一致，共3740次crop encoder（技术计划crop缓存正式复用，未另加forward），不是上界17388全部运行。源all_predictions_committed SHA=`8153c74a724084d7e5502c00e7da5fdac3baccd21075093dc34c8282943a1532`，在首次GT前完成签名提交。

原evaluate3755275 FAILED1:0、30分20秒：无GT提交核验已完成，进入首图2007_000033、seed42的F gap诊断时，comprehension错误把`if`写成`and`，触发`NameError: name r is not defined`。旧完整metrics/bootstrap尚未产出；之后取消检查对已过保留期squeue ID的二次异常只掩盖stdout尾部，原failure.json保留真正NameError。此为我写的评测诊断实现错误，不是F/G质量结论。四observe已完成，不重跑GPU/SAM3/GMP/检索或生成预测。原source snapshot/manifest、partial attempt/失败记录、原预测及所有crop缓存保持原样。

已按完成worker分支对源技术报告、Slurm历史、单GPU UUID、首份FP32源query/三个21类score、六PNG/NPZ背景保护及首crop原tokenSHA回顾验收；source startup_validation SHA=`0c3048522eb3c43be5aac2c5b76a98ab5f1f73304711adebb07392235e8b37d2`，并不声称本回合观察过存活/proc。真实crop report SHA=`a7d1db1ca139cff671b920fdbd6685944b0d877ffad6e5a9de40406662ad31fd`。

新建仅CPU评测版本`rgr_adaptive_evaluation_recover_20261021_102500`，run=`experiments/rgr_adaptive_reobservation_v1/该版本`，audit/log同版本。manifest SHA=`372888fe7bbbcc7f615abd34eca1ccfd7e810872cc3c8f121b64f6c953f5933f`，binding=`b35de4a1810586f98225a3801907e2c1d13d3ed9337243a42265ccd6e63244be`。唯一evaluate3757199（debug8CPU23000M/2h、CUDA隐藏）已提交并RUNNING，不能重复setup/submit。新snapshot按源code_records逐SHA复制，新增独立`tools/rgr_adaptive_evaluation_recovery.py`及`experimental_rgr_evaluation_recovery.py`；分组修为测试过的逐row selector，明确验证旧prediction binding和新evaluation binding，旧科学源码不改。输入masks/boxes/规则/成员/模型库/图片顺序与source逐项相同，input source_manifest和旧commit/失败/原GT输入metadata及Slurm账本均锚定SHA，不绕过签名。

当前控制入口为绝对Python调用`experiments/plans/rgr_adaptive_reobservation_v1/evaluation_recovery_control.py inspect/audit/audit-status/download/sync-manual`。禁止再调用旧local_control的upload/submit或启动旧GPU链。新CPU作业先通过44目标测试（原38+6个非空/空分组、缺失binding和owner分组回归，无skip）及真实已保存两图×三seed F/G PNG/NPZ/owner/E保护探针，无GT/模型/token解码、新SAM3或新预测为0，probe SHA=`4a86ba091bce824e9dbe7154e3e472f401b75fdcd6e5bd7d97c758df59022221`；真实quota余233826697216字节。之后正在重新核本批已提交产物SHA并完成首次完整GT/指标/bootstrap评测。旧首图GT已部分解码且未保存中间矩阵，完整恢复pass会再解码该一图；已在新manifest/summary明确披露，不能声称全历史只有一次GT decode。未用该图调整任何方法参数/候选/预算，旧bootstrap未运行，不存在复用不完整数字或选择性重算。

本地恢复launch凭据在plan/launch_receipts/评测恢复版/recovery_launch_001.json，SHA=`800efe6c8b8307c9ce291ac36a7039de248dde0574c1d8398314809f67267bea`。最终独立审核脚本`verify_evaluation_completion_001.py`已上传新audit外层为verify_completion_001.py，仅在eval complete且无audit_job/guard/accept时新控制器audit唯一提交。核10个成功源GPU child+1个成功新CPU child、九逻辑矩阵、六套预测/所有原crop/SHA/CSV/owner/transfer，并另列原3755275 FAILED及failure SHA，不将其伪造成成功；不读GT/旧token/模型或重算方法。source_prediction_acceptance_001为读入完成产物证明，不冒充本回合新活worker。下载控制器只取30小文件+必要图到新版本verified_20261021并逐SHA。固定三seed批验收归档/一次完成通知后删除rgr，不额外实验。


### 87.4 完整数值已完成；仅报告与图示恢复（2026-10-21 北京时间11:10，当前优先）

CPU评测3757199已FAILED1:0、21分32秒，但在失败前完整处理1449/1449图，九组逐图矩阵、25项数值/CSV/2000次共享图片bootstrap均已保存。失败发生在后续visualizer调用source_entry时，误将新evaluation输出binding用于原prediction source inventory，触发ValueError: Source inventory identity differs。这是报告路径的实现错误，不是预测/数值失败。旧attempt/failure、全部25项完整统计及原snapshot保留，不能重跑全GT指标、bootstrap、GPU预测或SAM3。

新增独立报告版rgr_adaptive_report_recover_20261021_110000。控制器experiments/plans/rgr_adaptive_reobservation_v1/reporting_recovery_control.py，入口tools/experimental_rgr_reporting_recovery.py。源235个文件按SHA拷贝不改，新增CPU报告/visualizer/summary/针对测试；数值25文件逐字节SHA导入，JSON保留原统计binding并显式标注statistics_source_binding。指标JSON仅由已保存矩阵求和核对CSV点估计，不重新解码GT作指标或重采样。先写独立numeric_report_001保存数值，再画图，避免图示故障损坏数值收尾。

新visualizer严格以原prediction source context加载inventory，没有关闭或绕过签名。只对预先按保存结果固定的30随机/20有益/20有害案例中的唯一图片重新读取GT以绘最后一栏；像素precision注释来自已保存trace计数，不重新评分。须披露原第一次部分首图GT读取、随后完整1449统计pass以及这次少量visual-only读取，不能称全历史仅一次GT decode。47本地目标测试通过；集群须相同47项无skip及真实两份source绑定、九组矩阵/252逐类CSV和原bootstrap字节核对通过，才称报告路径验收。

截至本段写入报告快照正在准备，尚无新JobID；实际manifest/binding/JobID随后补。独立审核将核10成功源GPU与唯一新CPU报告成功，单列原3755275诊断失败、3757199完成数值后图示失败；两失败不能冒充成功。审核不读GT/token/模型、不重算方法或bootstrap。原完整指标的质量结论待独立验收，小结果和必要图归档后一次结果通知并删除rgr；固定批不追加实验。

报告版已唯一提交CPU3757473，debug8CPU23000M/2h、CUDA隐藏，无GPU资源后备。manifest SHA=`48709ac24f57ff23898e93bd509d627f026409ee22f23662d84ca1d9b968c5c0`，binding=`0615b3820f297a152c65d160af2f0f9e6a8adb6640bdab9b265d80f5c46f3eae`；audit/jobs.json为准。setup已核源235文件克隆及6新文件SHA，不重复setup/submit。只待报告完成与独立审计，不重跑数值评测或任何预测。

3757473的真实CPU报告探针现已passed：47目标测试无skip，原prediction source binding两份真实inventory加载通过，九组保存矩阵/252逐类CSV及method/bootstrap点估计精确，25导入文件和bootstrap字节不变；无GT/model/token decode、新预测或SAM3。reporting_recovery_probe_001 SHA=`a226a23a6f369a54c026cd55fed1b9a0c6cbafbe370bd921250fc9ea7716e32b`，真实quota余233756540928字节。作业仍RUNNING、无failure，正在后续图示/报告，尚无独立accept。

报告3757473现已COMPLETED0:0、2分05秒，70张真实12面板图示及完整summary/metrics均已保存，无报告failure。数值25源文件/2000 bootstrap逐字节导入，未重跑GT指标/SAM3/预测。已确认无旧audit/guard/accept后，唯一独立只读CPU审核3757498（debug4CPU10GiB/2h）已提交，audit_job_001.json存在；禁止重复audit/verify，先audit-status等这一作业。独立accept尚未产生时不把controller checks当最终验收。


**最终数值与独立验收（2026-10-21，北京时间11:37核实）：** 3757498 COMPLETED0:0、13分58秒；independent completion_acceptance_001 SHA=`e4475334240660e1d82c15d8a8bf2ef7f5cee481a396f1ba0bb529447bb794da`。10成功源GPU child和新CPU报告3757473均COMPLETED0:0；原3755275、3757199的FAILED状态/失败SHA分别保留，不伪装成功。全部8694新PNG、crop查询/FP32/native映射、owner/E前景保护、F/G预算、模型库字节SHA、九矩阵求和/E历史逐图、25份数值来源字节/CSV/2000bootstrap均通过。独立审核没有读取GT、原token或模型tensor，也未重做推理、GMP、检索、指标或bootstrap。

| 方法 mIoU(%) | seed42 | seed123 | seed2026 | 均值 | 样本SD(pp) |
|---|---:|---:|---:|---:|---:|
| E 原 single-view loose | 87.183731 | 86.780938 | 86.681601 | 86.882090 | 0.265908 |
| F 检索引导位置 | 87.182551 | 86.786650 | 86.720767 | 86.896656 | 0.249774 |
| G 同预算随机位置 | 87.196738 | 86.819238 | 86.749745 | 86.921907 | 0.240533 |

E不是最初严格R3 A，不能混用基线。F−E各seed为−0.001180/+0.005711/+0.039166pp，均值+0.014566pp、2/3正，配对样本SD0.021581pp；G−E均值+0.039817pp、3/3正，配对SD0.027600pp；F−G为−0.014187/−0.032589/−0.028978pp，均值−0.025251pp、0/3正，配对SD0.009751pp。共享2000图片bootstrap三seed均值95%区间：F−E [−0.063013,+0.098539]、G−E [−0.008653,+0.097741]、F−G [−0.091957,+0.045332]pp，均跨0。这不是独立test或seed方差替代，不能说随机G显著优于F；当前固定方案没有验证出guided位置优势，也没有实质稳定提升E。

F−E三seed等权平均BG FP+85406、FG FN−83779、wrongFG+11447.667像素；G−E为+31137/−50507/+1543。F−G多54269背景FP、少33272背景-only FN、但多9904.667前景错类；FN下降包括错误FG，不能全当正确恢复。F平均正确FG新增72331.333，对G48964，但平均net valid benefit为−24522.333，对G+16284（诊断口径不替代mIoU）。新增valid precision三seed等权均值F42.7641%、G60.0994%；包含ignore分母且不当正确的precision_all为32.7339%/47.7599%。F三个seed分别valid precision41.3290/40.4914/46.4720%，G54.6056/56.2797/69.4129%。

F productive crop rate25.3682/23.6593/24.6400%，G10.9656/10.4101/11.5200%；该比率只代表有候选被接受，不是GT正确率。F正确FG/forward114.354/108.737/125.096，高于G60.931/88.052/86.141，但F net valid/forward−47.984/−51.069/−18.994，对G+10.278/+19.650/+48.182。引导位置产生更多接受和正确FG的同时也引入更多错误，不能将高yield独立当成定位机制成功。

F−E主要正向类pottedplant+0.413979、bird+0.327651、table+0.230639pp；chair−0.500922、bottle−0.206417、tvmonitor−0.140006pp。F−G的主要负向为chair−0.393979、sofa−0.338606、car−0.314333pp；最大正向table+0.220635，去掉它后其余20类均值差−0.037546pp，官方21类F−G仍为−0.025251pp，不用20类指标替换。F新增主要错误转移为BG→sofa30220.667、BG→chair17188、BG→table11624.333，sofa→chair6837.333像素；完整21×21转移和ignore另存CSV。它没有解决chair/sofa混淆。

F/G实际各arm每seed611/634/625 crop，合计3740次encoder，逐image/seed/rank次数和宽高完全一致。科学整体Slurm墙钟33622秒（9时20分22秒，最早成功源GPU start至新报告end，含等待及恢复时间，不是纯GPU时长，不累加并行）；源GPU最早01:54:48至最后03:09:54，报告CPU125秒，独立审计838秒另列。两旧CPU失败30分20/21分32秒保留；数值25文件与bootstrap只完成过一份全量统计，报告未重复它们。原首图部分GT读取和完整pass之外，仅45张唯一GT为70张posthoc图最后一栏重新读取，数值注释来自保存trace。

已下载并逐SHA核104个文件（34小结果/凭据含visual index+70张12面板，合计42378129字节）到`experiments/hpc_results/rgr_adaptive_report_recover_20261021_110000/verified_20261021`，不下载token、模型或批量原预测。summary SHA=`14eb3711c647cce90717ed7cfa31df56b134b0950af021170eaf83e9208e357f`，metrics SHA=`a732fcd2bd05f38924e927df7dfe74d39236d3d1d2e164f2c1d64e609ba0b659`，download_verification SHA=`0fd556cad74526d23398533d9dcf00504c31a6a304d3d767840764442cd0ab10`。独立中文analysis_report.md回答附件42节七问，分析器analyze_verified_report.py只读已保存JSON/CSV，无GT/模型/token/重bootstrap。真实抽查有益2011_000112 seed42（F恢复鸟，valid precision .99595）、有害2009_001535 seed42（误填背景角落，precision0）、随机2007_000572 seed2026（零新增且precision unavailable），12栏清楚、GT最后、E前景不改。QA在visual_QA_001.json，分析/QA/下载proof回传SHA后追加归档凭据。

结论：当前证据更符合generic额外候选的微小增量，没有证明固定retrieval-gap guided位置价值；F微小均值提升不能用于继续扫描或扩大预算。G也只是开发集点估计，不直接替换生产。固定批结束，不追加任何实验，保持原正式R3版本；完整回传核SHA及一次完成通知后删除rgr监测。

最终归档状态为`complete_verified_archived`。analysis_report.md SHA=`910a099c8c1c84cead31b0ef116fa17e740c93a5ef19134618aa3d526f65b135`、visual_QA_001.json SHA=`e63bbf1a13f6d678c5ea0ec2a6af1265d617b93b60e3bf738013678003649f26`、download_verification SHA=`0fd556cad74526d23398533d9dcf00504c31a6a304d3d767840764442cd0ab10`及local_archive_001.json SHA=`a76335b66ea818852123f6fd7b1f702a5e9573afe4d3a542d178416d24288ca4`已独立小文件回传HPC当前报告run并逐SHA核对；未覆盖任何正式指标/源数值、GT、预测或bootstrap。当前批研究计算和验收全部结束，不存在待重跑的模型/预测/SAM3/GT评测。后台rgr在本次一次结果通知前退役。

后台`rgr`监测已通过automation_update工具删除（deleteStatus=deleted）。退役小凭据automation_retirement_001.json SHA=`e0e1fd80b552ff76f221d068aea587b4e3407a832792e196ff8ed9718b897927`保存在本地verified目录并回传本报告run。本批一次完成通知后不再自动唤醒或追加；正式R3版本、生产src/configs与所有历史结果保持原样。


## 88. Structured Multi-Crop Proposal Diversification v1（2026-10-21）

用户新附件明确授权执行SMC-v1，版本structured_multicrop_20261021_123600（设计标签、实际提交以Slurm）。原附件/方案/固定参数在experiments/plans/structured_multicrop_v1/{request.md,plan.md,protocol.json}。独立入口tools/experimental_structured_multicrop.py；新run/storage/log/audit均使用structured_multicrop_v1/版本，冻结code位于hpc_launch_audits/版本/code。旧RGR/CrossView源码和结果、正式R3/生产src/configs不改，不部署HPC原不存在的正式入口。

E只读第86节已验收single-view loose（87.183731/86.780938/86.681601%，均值86.882090%），不是严格R3 A。R/H所有1449图片、每图固定额外2个floor(.8W)×floor(.8H)crop，R/H各2898、共5796次encoder；三MLP seed42/123/2026共享geometry和原图GMP，但各自旧MLP/bank独立检索，不复制分数/决策/预测。H只从四角TL/TR/BL/BR依新增center union外覆盖greedy两步选，tie稳定，不硬编码角组合；R按SHA256('structured_multicrop_v1:'+ID+':'+rank)前8little-endian→PCG64均匀合法整数位置，只拒绝与center/前一个random exact同框，最多100重抽，失败不减预算。位置完全不读内容、proposal、检索、类别、GT，不继承RGR的触发/尺度/seed key。

原冻结SAM3/完整20类五桶/context expansion/正常预测presence/1008FP32，每crop一次encoder，仅生成几何，不存crop token或用它分类。mapped mask仅原图签名FP32 72x72x1280→single-phase hard-center GMP rho.1→各自原1280-512-GELU-256 MLP、34880成员/10569BG单库→balanced25/top5/search2/显式center.2。原R3 mapped面积100/max.95、class-NMS.7/hash/cap1保留；跨两view仅对语义eligible池按FP32(.8r+.2q)/candidate ID顺序hash和semantic NMS.7，q>=.4/r>=.9/m>=.05/FG/实时novel>=.1只填E背景。E已有FG绝不改，不跨arm共享候选、不新增全局cap/类阈值。

本地38项测试已通过，含随机/coverage/严格FP32阈值/缓存SHA/GT屏障及合成标签完整CPU评测→2000bootstrap→五组图示尾部；不等于真实GPU/SAM3/精度验收。计划Slurm technical1GPU（两图8个真实计划crop及3seed旧E精确、GMP旧raw数值、原模型库、native映射和E前景保护）→observe四GPU各处理按image分片三seed→CPU evaluate2h；技术8crop及两图完整预测同binding复用，最终仍5796forward。每GPU一计算worker，真实mmlsquota余>=5GiB（最多3瞬时重试、不df），只实际消费1449val token各一次核SHA，不扫12031token或训练库。只有CPUdebug明确QOS且无JobID拒绝才一次GPU配额/CUDA隐藏后备，已有job不重提。

六套8694 PNG/owner/query、5796crop、全部无GTgeometry/diversity/complementarity/budget及内容SHA完成提交后才GT。diversity参考是历史center-R3 post-cap提议池，几何IoU跨类，空center maxIoU0/无候选率null，<.5定义diverse，固定5箱只诊断不gate。评测9逻辑矩阵/E每图历史精确、21IoU、FP/FN/wrongFG/21x21transfer+ignore、每forward效率、三seed均值/样本SD/方向和2000共享image bootstrap（不替代seed方差或独立test）、coverage关系与H-R去最大正类20类敏感性。38测试与全数值矩阵/CSV/metrics/bootstrap先落盘再图示；GT完整pass内保留小palette预览，不为图示重读GT。随机30/最多20H有益/20H有害/20H-R/20R-H图像，效应排序用三seed均值、显示seed42明确标注，GT最后栏，支持不足不伪造。

本段尚未提交：待冻结上传/manifest/JobID后补，不能称正式运行。只有technical通过、四worker真实Slurm/proc/GPU UUID与首批FP32/PNG/模型身份SHA验收后首次通知；正常推进静默。失败具体日志/attempt留存，只取消本ledger未开始依赖；改科学实现新版本和小真实probe，不改运行源码。用户暂停取消即安全停本批与监测，不自动恢复。科学固定批全部完成后唯一独立Slurm审核6个成功child、8694PNG/5796crop/9矩阵/CSV/原模型库字节SHA/geometry/owner/transfer，无GT/token/model decode或重算。小结果与必要图下载核SHA，七问报告/手册同步后一次完成通知并删除新监测。无精度提前gate，不自动Scale/Location/corner/阈值/seed扫描、增加crop、训练、部署、GitHub/打包或邮件排障。

本批已唯一注册提交：technical3758591、observe3758592 array0-3%4、evaluate3758593（debug8CPU23000M2h），全afterok，有JobID不重复submit/prepare/upload。manifest SHA=`429b6851462b6e5ae0dd2908076de90ad36ec1fed6eb3be9c3c94dd47bd3b5c1`、binding=`c67b3df02c897def93d77bd41a8fb60b892afd3f25286d02b421d0befbf4c9a8`；239历史snapshot文件先逐SHA克隆，14本批文件上传逐SHA有效，code_records=247。无CPU QOS后备。集群technical真实38测试已通过无skip，正在真实SAM3/两图crop路径，不能当正式1449结果；四正式worker与startup尚待验收。新后台监测smc已创建，正常阶段静默，固定批验收归档后退役；旧rgr保持删除，不重启旧批。

集群technical3758591现已COMPLETED0:0、3分28秒，真实38测试无skip及两图8计划crop/三seed原E独立像素回归/原GMP对旧raw/native映射/原tokenFP32/每seed独立21类score和E前景保护均通过。report SHA=`fd564cb9ff09f6726a52c17ebba68394091a5c705b5076b5143ef8c60c3e3314`已本地下载核SHA；峰值5208.689453125 MiB，quota=233793486848字节。正式observe3758592四child已RUNNING，仍待startup实际/proc/GPU UUID及首输出审核，技术通过不代表1449质量。

首次正式启动已独立验收：startup_validation status=passed/formal_start_reported=true，SHA=`a17445ceef11e09c551cbe93225f70c789b1e832850a98a50efaabf4dc75cc76`已下载核SHA；四个worker精确ArrayID_TaskID与实际JobID/真实Slurm overlap ps命令/PID/shard、唯一GPU UUID、首FP32 crop与三个21类score、六PNG/owner/E前景保护及内容SHA均通过。实际workers=[{"GPU_UUID": "GPU-0a583fca-d35f-73a1-6092-fa575ce7f912", "actual_job": "3758619", "host": "c2v100m2-4.hpc2", "pid": 860738, "shard": 0, "state": "RUNNING"}, {"GPU_UUID": "GPU-a63f10f0-a1df-941e-6045-b1b2134754b7", "actual_job": "3758620", "host": "c2v100m2-16.hpc2", "pid": 2420930, "shard": 1, "state": "RUNNING"}, {"GPU_UUID": "GPU-97edf60e-5381-8457-7424-e9e5d89cb189", "actual_job": "3758621", "host": "c2v100m2-10.hpc2", "pid": 1954531, "shard": 2, "state": "RUNNING"}, {"GPU_UUID": "GPU-d352379d-ffd7-74c1-c1ea-afc9f9b54ec5", "actual_job": "3758592", "host": "c2v100m2-10.hpc2", "pid": 1954530, "shard": 3, "state": "RUNNING"}]。原center GMP两图首至多2mask探针maxabs均0、bitwise=true，仅针对路径数值回归，不是全token重扫。已正式运行，全1449结果仍待完成；首次启动在本次交互报告后不重复通知或startup执行。自动smc只在完成/失败/必须行动反馈，固定批验收归档后删除。

### 88.1 全量数值完成、独立验收进行中（2026-10-21）

六个科学child均COMPLETED0:0：technical3758591用时3:28；observe3758592四child分别1:00:08、56:29、1:01:09、59:43；evaluate3758593用时1:19:37。5796次crop、六套8694新PNG和9组1449逐图矩阵已完成，无本批failure或活跃科学锁。首GT前全预测提交SHA=`2007cf45fd137399aa83e7db8b9054dd60e9c515c97bb64523ffbcb67c145ac4`；summary SHA=`e2e129873614814da1acc858092b4f87adfeacc8b1b23b32050a496d34016d36`。GT完整评测/2000bootstrap/图示均完成，不再运行推理、检索、GMP或GT数值评测。

保存的数值（独立accept尚待，不能当已验收最高记录）：E均值86.882090%，R均值86.875771%，H均值86.945033%；R-E=-0.006319pp（1/3正），H-E=+0.062943pp（2/3正），H-R=+0.069263pp（3/3正）。H-R配对样本SD=0.046424pp。H-R最大正类为sofa，平均IoU差+1.987510pp；去掉后其余20类均值差-0.026650pp，因此方向信号有明显类别依赖，不能按3/3正单独宣称通用空间覆盖优势。最终需结合已保存的图片区间/错误与diversity表谨慎分析，不新增扫描。

确认audit_job/guard/accept均不存在后，已唯一提交独立只读审核3760520（debug4CPU10G2h）；audit/audit_job_001.json已有JobID，禁止再次audit/verify。审核源码SHA=`8521a84a0f58500b095a244c9f35f8778da0cd0bb778607ec76524c98dd36128`。当前只查audit-status；passed后再下载逐SHA、抽查真实图、外层analyze_verified_report.py/archive_verified_report.py生成七问分析与归档。两个外层脚本只做已保存小文件分析/归档，已语法检查，不在冻结科学snapshot，不重算GT/bootstrap。验收归档及最终通知完成前保留smc监测。

### 88.2 最终独立验收、结果与停止（2026-10-21）

独立审核3760520已COMPLETED0:0、56分46秒，status=passed；accept SHA=`a2baf41d085901be7086cb65a968ef17bfe60fdeb85d27cbe2735f357f64845c`。全部6个科学child成功，8694新PNG、5796crop（R/H各2898）、9组1449逐图矩阵、E历史逐图像素/矩阵、原模型和库字节SHA、FP32三seed独立检索、native映射/owner/transfer/CSV和图示SHA已核，无需重复推理、检索、GMP、GT评测或bootstrap。38目标测试无skip与原8计划crop探针仍为本批正式预算内复用，原GMP bitwise探针只覆盖两图首至多2mask，不外推为全库重算相同。

| 方法 | seed42 mIoU% | seed123 | seed2026 | Mean% | 样本SD pp |
|---|---:|---:|---:|---:|---:|
| E 历史single-view loose | 87.183731 | 86.780938 | 86.681601 | 86.882090 | 0.265908 |
| R 额外随机两crop | 87.165712 | 86.744302 | 86.717299 | 86.875771 | 0.251459 |
| H 额外覆盖优先两crop | 87.182390 | 86.848879 | 86.803830 | 86.945033 | 0.206787 |

| 配对差 | seed42 pp | seed123 | seed2026 | Mean pp | 正向seed | 配对样本SD pp | 2000共享图片95%区间 pp |
|---|---:|---:|---:|---:|---:|---:|---|
| R-E | -0.018019 | -0.036636 | +0.035697 | -0.006319 | 1/3 | 0.037559 | [-0.122242,+0.108409] |
| H-E | -0.001341 | +0.067941 | +0.122229 | +0.062943 | 2/3 | 0.061936 | [-0.095750,+0.225261] |
| H-R | +0.016678 | +0.104578 | +0.086532 | +0.069263 | 3/3 | 0.046424 | [-0.091105,+0.230842] |

H-R三seed方向一致，但三组条件图片区间均跨0；最大正类sofa的H-R平均IoU差+1.987510pp，去掉后20类平均差-0.026650pp。H相对E的sofa为+2.180261pp，chair=-1.051478pp、table=-0.257313pp，不能宣称通用结构化覆盖优势。R未稳定复现此前RGR随机组的收益；旧RGR随机组条件于触发图片/次数/尺度，本批所有图片固定两crop，二者不能混为同一协议。H均值更高不触发部署或更改正式R3配方。

H平均几何union覆盖93.982%，R为77.264%；图片支持平均diverse率H20.624%/R12.541%，按候选池化H25.483%/R15.280%。post-cap候选H5435/R5386，center IoU>=.9池化率H34.480%/R51.857%；H确有更多几何新颖与view-only接受，但新颖不等于有效。coverage与diverse率、逐图mIoU差、正确FG的已保存线性相关均很小，只作描述，不能因果归因为coverage。跨模型seed共享一套random布局，并非三个独立随机布局重复。

实际owner新增像素三seed均值：R正确FG93050、背景FP142828.7、错类8431.7、ignore100509.3，valid precision均值37.530%；H正确FG185401.3、背景FP180031.3、错类6884.3、ignore152644.7，valid precision均值49.360%。precision_all分别26.581%/35.000%，ignore进分母不当正确。H-R正确FG多92351.3，同时背景FP多37202.7、错类少1547.3；H平均net_valid=-1514.3，R=-58210.3。每forward正确FG H63.9756/R32.1084，net_valid H-0.5225/R-20.0864。像素计数与mIoU权重不同；背景型FN下降包含新增错类，不能全部叫正确恢复。H较R新增正确sofa像素平均多105148.3，是明显的类别依赖来源。

数值/CSV/bootstrap先保存后绘图；110张真实图（random30、H helpful/harmful及H-R/R-H各20）均保存于visualizations，GT由同一次完整评测pass的小palette预览生成最后栏，不重读GT。已本地查看五组各一例，visual_QA SHA=`d7f813cce84f408c718082bb9cea3d87e44304f4e553fe8f67c6fbdf081ad8d0`。逐图mIoU排序会因极少像素引入缺席类别而出现大跳变（例如2008_000662背景误报及2010_004042的一像素误报）；图示单图±33pp不代表数据集贡献、不能相加。

本地下载并逐SHA核对143文件（33小统计/验收与索引、110图），目录`experiments/hpc_results/structured_multicrop_20261021_123600/verified_20261021`。中文七问分析`analysis_report.md` SHA=`dbe1171ecb3458d46e797450b5c5568255fb8444d78f1c1741fc179cb1f412d5`；下载proof SHA=`89f42e08f3f3385dfdda5f513b5d48048ee254834a085eef4c7482a03f2c30f7`。分析、下载证明、visual QA已上传run并远端核SHA；`local_archive_001.json` SHA=`a7819a5ce25bf889cee19f5565128437ae51c7090015e53f296c971bab8eee83`，原数值/GT/bootstrap未重算。本地初稿保留为analysis_report_draft_001.md，正式分析补齐coverage相关与单图解释边界，不改科学快照或原指标。

角色耗时：technical3:28，四observe56:29–1:01:09（并行），evaluate1:19:37，独立audit56:46单列；科学整体墙钟为最早科学start至最后科学end=8654秒（2:24:14），不累加并行。真实单worker GPU峰值最大5211.3583984375MiB；技术时真实GPFS余233793486848字节。固定批结论为小幅、类别依赖的方向信号，未验证通用位置优势；建议停止继续增加R3 crop预算，不扫scale/corner/seed救结果。只写后续建议，不执行新实验，不训练、部署或GitHub发布。归档和手册同步核SHA后退役smc监测。


## 89. T4-hard Local Tiling v1（2026-10-21，用户附件并行授权）

附件、注册规则和方案见 `experiments/plans/t4_hard_v1/{request.md,protocol.json,plan.md}`。版本 `t4_hard_20261021_172500`，独立入口 `tools/experimental_t4_hard.py`；独立 run/storage/log 分别为 experiments/t4_hard_v1、storage/proposals/voc/t4_hard_v1、logs/t4_hard_v1 下版本目录；audit/code 在 experiments/hpc_launch_audits/版本。与第90节并行、独立 ledger/manifest/GT屏障，不组合运行。

仅从已验收 E 开始，每张1449图原图坐标按 floor(W/2),floor(H/2) 分 TL/TR/BL/BR 四个无重叠半开象限，奇数余量右/下；固定5796额外 SAM3 encoder，三seed42/123/2026共享几何和原图GMP但独立 MLP/库检索。完整20类5桶、正常presence、1008 FP32，原mapped过滤；仅原图签名72x72x1280 token hard-center GMP rho.1→原1280-512-GELU-256 MLP/34880成员10569BG→balancedK25/top5/search2显式center.2。E式q.4/r.9/m.05、semanticNMS.7/live novel.1，只填E背景。内部边界touch仅诊断，绝不gate。主要比较T4-E/T4-H；T4四forward对H两forward非算力匹配。旧E/H、生产入口与配方不改，不部署。

来源 SMC manifest SHA 429b6851462b6e5ae0dd2908076de90ad36ec1fed6eb3be9c3c94dd47bd3b5c1、accept SHA a2baf41d085901be7086cb65a968ef17bfe60fdeb85d27cbe2735f357f64845c 已通过本轮native SSH实际读取核对；旧批已结束，无旧计算需要重跑。新控制器 `experiments/plans/t4_hard_v1/local_control.py`（委托公共 t4_rli_parallel_v1/control.py）仅处理本 ledger。初次clone已完成247来源源码，冗余sam3-main链接创建FileExistsError发生在上传新文件/注册/sbatch之前，已只修外层布局判断并逐SHA接续，保留layout_recovery_001；无JobID/计算/科研失败。不得重复upload/prepare。

本地36目标CPU检查无skip，包括两批合成预测SHA提交→九矩阵/CSV→2000bootstrap→真实绘图尾部；不代表集群CUDA/SAM3或精度。拟定 technical 两图8实际计划crop正式复用→四GPU observe→CPU evaluate。所有新预测内容与无GT flow/diversity/预算先提交后GT，分tile/内部边界/面积分箱报告actual owner质量。完整数值先checkpoint再图示，GT palette在数值pass内保留，GT最后栏不重读。三seed方向、样本SD、2000共享图片bootstrap（VOC开发、固定模型，不替代seed不确定性/独立test）。当前尚未提交，实际JobID/manifest/技术startup待后补，不能宣称正式运行。

## 90. H Retrieval Leave-One-Patch Refinement v1（2026-10-21，RLI）

附件/参数 `experiments/plans/h_patch_refinement_v1/{request.md,protocol.json,plan.md}`；版本 `h_patch_refinement_20261021_172500`；入口 `tools/experimental_h_patch_refinement.py`。run/storage/log 只在 h_patch_refinement_v1/版本，audit为独立版本目录；公共科学实现 t4_rli_{core,runtime,worker,evaluation,visualize,schedule}.py 固定冻结到各自code，不修改历史snapshot或生产。

零新SAM3/训练/库，固定第88节完整H accepted extra mask、候选排序/类别/最终owner。CPU先完整读取saved H mask/owner和hard-center indices作无GT patch inventory，不读token；之后完整原I baseline重新GMP→各seed原MLP/bank→retrieval：label精确，score/margin/全部21类分数abs<=1e-5，不符停止报错。|I|<=1保持。原I每patch逐一去掉→原精确FP32 uniform_gmp（不近似、不直接patch→MLP）；同原类、原类score和margin均严格增>1e-6才harmful。一次固定rank按score_gain/margin_gain降序index升序；greedy每步重算J去一patch，label不变且score/margin>=current-1e-6才接受，至少留一patch。无隐藏cap/抽样/额外阈值；每worker48h为调度上限，不减科学预算。

被删patch对应原图72x72整数cell与原mask、原owner交集置背景；不重排、回填、改类或动E。RAND每candidate从原I抽RLI实际删的k，key=`image_id:seed:candidate_id:rli_random_matched_v1`，SHA256首8字节little-endian→PCG64无放回，patch数匹配而非面积。所有6套新PNG/NPZ/删owner map和完整noGT trace/SHA先提交后GT。输出RLI-H/RAND-H/RLI-RAND，removed BG FP/正确FG/错类/ignore、retention/removal率、21x21transfer、patch gain与GT多数类型、删比例三分位（linear quantile，同值不拆，空组null）owner非可加counterfactual。图示最多20有益20有害30随机，GT最后栏。

同36本地CPU检查通过，真实baseline/MLP/库/GPU路径尚待集群technical，不把本地测试当方法质量。source与初clone修复依据同89；只控制本ledger。CPUinventory→GPUtechnical（实际首两张有H extra图完整候选，正式复用）→四GPUobserve→CPUevaluate。每seed全1449，不跨seed复制分数或carving。

两批各完成后唯一独立Slurm CPU audit（不GT/token/model decode或重算GMP/检索/bootstrap），核科学child、PNG/owner/crop/CM/CSV/原模型库字节SHA/trace/预算。验收后小结果/必要图下载逐SHA，独立五问分析与归档、手册同步。两批完成/验收后以各自主要mean delta>0且>=2/3正只写JOINT_RECOMMENDED，另披露RLI-RAND；不自动联合、overlap、扫描、训练或部署/GitHub/打包/邮件排障。用户暂停取消立即安全停所指ledger和监测，不自动恢复。当前两批尚未注册/提交，待真实ledger补充。

### 89/90.1 实际注册提交（2026-10-21）

两套源码各247来源+16新文件SHA已核，初clone布局修复只发生注册前，无JobID/计算；外层layout_recovery_001保留，科学实现未因此修改。两条全afterok依赖链均已唯一提交，不能重复upload/prepare/sbatch。T4 technical3761836、observe3761837 array0-3%4、evaluate3761838；manifest SHA `495c305c7b727d322d3acc139034beb7009d53f65fe9f4679fdfca093b1fcb1f`，binding `2f9e8d1d3717b0e8bc11a8698cde350a9a62e754939c1b37d242da64f60f4010`。RLI inventory3761832、technical3761833、observe3761834 array0-3%4、evaluate3761835；manifest SHA `13eed5332fae8e3a8231efa32c40d90286f05bd0350a647c1c8deff93c08b262`，binding `e0f417f7283742213cba7011cd4f61deed912f2196d4b217d0c76ba5daee57e9`。

两evaluate原debug8CPU23000M2h请求均明确QOSMaxSubmitJobPerUserLimit且无JobID，被拒后仅一次v100g32单GPU配额6CPU24G、CUDA隐藏CPU后备，resource_fallback_evaluate_001.json登记；不是科学质量失败，不声称零GPU资源分配。RLI inventory真实debug请求成功。技术与完整worker初产物/正式startup尚待，36本地CPU测试不代表CUDA或1449质量。实际ledger在各audit/jobs.json。后台监测只核本批小progress/log/complete/failure，正常推进静默，首次正式验收/失败/最终收尾再通知。

### 89/90.2 全量数值完成、独立验收中（2026-10-21）

两批所有科学child均COMPLETED0:0且无failure/活跃科学锁：T4 technical3761836 26:05，四observe3761837_0/1/2/3分别1:44:40/1:51:45/1:50:49/1:46:23，evaluate3761838 1:15:53；RLI inventory3761832 43:06、technical3761833 11:12，四observe3761834_0/1/2/3分别24:02/1:03:42/1:07:47/1:14:24，evaluate3761835 51:22。T4完成5796crop、4347新PNG和9组1449矩阵；RLI完成8694新PNG和9组矩阵，零新SAM3。两批2000共享bootstrap与各70张GT最后栏图示已完成，数值checkpoint均GT_images=1449/bootstrap_completed=true，不需要重跑推理、GMP、检索、GT数值评分或bootstrap。

T4 technical_report SHA `57bc389b1bdda11b64c04f17cafc84595edb5866dd592b1a31b73e454907cc9a`；36目标集群测试无skip、两图8计划crop、原E三seed独立像素回归及原GMP两图首至多2mask探针maxabs0/bitwise=true，不能外推全面重算一致。技术GPU峰值5228.16455078125MiB，真实GPFS余233294282752字节。RLI technical_report SHA `204e1fe3924e0ce6366b166e3226d3d1480851b935c346727e83bd648ede5cae`；36目标集群测试无skip，2007_000529/2007_000636两有H extra图真实完整refinement，原E三seed像素回归通过，零SAM3；技术峰值88.1181640625MiB，quota232818974720字节。完整RLI worker最大峰值90.86767578125MiB。

保存数值（尚待独立accept，不称最终已验收记录）：E mean86.882090%、H86.945033%、T4 86.787865%、RLI86.932147%、RAND86.939891%。T4-E=-0.094225pp（0/3正）、T4-H=-0.157168pp（0/3正）；RLI-H=-0.012886pp（1/3正）、RAND-H=-0.005143pp（1/3正）、RLI-RAND=-0.007744pp（0/3正）。T4四forward对H两forward非算力匹配；RLI未优于相同删除patch数随机对照，不能把语义分数单调提升当像素正确性证明。固定批不追加联合/overlap或阈值扫描。T4 summary SHA `3a303685c86649b64f6e43cbad71f703d296c364b6b24bc44b3a9320a5e036de`；RLI summary SHA `a0ae76304057998d26e3e7cfeac26b2f06dd969236408b93cd64eb9bb7fc7100`。

确认两批 audit_job/guard/accept均不存在后，各唯一提交独立只读审核：T4 `3764455`（debug4CPU10G2h），RLI `3764456`（debug明确QOSMaxSubmitJobPerUserLimit无ID后唯一v100g32 1GPU配额、CUDA隐藏CPU后备，4CPU10G2h）。两audit_job_001.json现已存在，绝不再audit/verify提交；只audit-status。audit不读GT/token/modeltensor，不重算方法。完成worker的startup仅回顾核验complete+sacct/保存GPU UUID/产物，不伪造本回合观测过存活/proc，不再重复发启动通知。passed后再下载逐SHA/真实图抽查/独立五问分析归档与joint仅建议。最终验收归档前保留t4-rli监测。

回顾startup核验均passed，并已下载小proof逐SHA：T4 startup SHA `b8821dbaf3b69e756d204054c57f949dc83cfe61dcce53be380718baaf17b003`，RLI startup SHA `e5af7d3ad6104ac3c8b595f96fbc9002ec276aa6c484a286a6f5028f8d9d4701`。全部worker均按COMPLETED+sacct+保存产物/单GPU UUID核，不声称本回合观察过存活进程。两technical/startup/audit ledger小proof均位于各plan/launch_receipts/版本/small_proofs_20261021。

保存的2000共享图片95%区间（pp）T4-E[-0.427682,+0.183490]、T4-H[-0.482462,+0.144537]，RLI-H[-0.052050,+0.023902]、RAND-H[-0.035741,+0.024776]、RLI-RAND[-0.017489,+0.000762]，均跨0，不能宣称统计显著退步。停止依据为预注册正向均值/多数seed条件不满足。T4相对H平均BG FP多144735.7、错误前景类别多8011、背景型FG FN少52551.7；其中正确FG实际多44540.7，不把错类当正确恢复。RLI每seed平均删BG FP42001.7、正确FG34959.3、错类915.3、ignore27447.3；相对RAND多删BG FP3709，同时多删正确FG6272。semantic score单调不等于pixel quality。

报表字段需在独立analysis说明：原summary/extra_pixel_quality的accepted_candidate_count=0来自报表调用默认占位，不代表实际没有accepted；实际T4 count可核t4_candidate_flow，原H extra count可核RLI inventory/refinement trace。RLI原quality字段added/correct描述被删除的原H owner像素，不能叫新增/正确恢复。外层posthoc_details.py只读验收后保存CSV生成posthoc_diagnostics_001.json/derived_quality_counts.csv和五问具体数值分析，不改正式指标或冻结科学代码、不读GT/重bootstrap。当前自动监测在本地原为PAUSED、60分钟（用户/app设置），本次仅更新状态说明并保留PAUSED，未自动恢复；独立Slurm审核不依赖监测在线。

### 90.3 RLI独立验收与归档（2026-10-21，最终）

独立审核3764456已COMPLETED0:0、36分12秒，completion_acceptance_001.json SHA=`7840a0b7af5e6297b4d4bda45f48884f3cdb4c7349cedd01fde6d4387c84abab`。七个科学child全部COMPLETED0:0；8694新PNG/NPZ、完整baseline/single/greedy/RAND/owner记录、原模型库字节SHA、九逻辑矩阵逐图求和/H旧历史、CSV与bootstrap一致性均通过。真实baseline最大差2.384185791015625e-7，低于1e-5。独立审核不读GT/token/modeltensor、不重算GMP/检索/预测/指标/bootstrap。无科学失败；原debug QOS拒绝及唯一CUDA隐藏GPU配额后备保留，不将资源分配称为零GPU。

| 方法 mIoU(%) | seed42 | seed123 | seed2026 | 均值 | 样本SD(pp) |
|---|---:|---:|---:|---:|---:|
| H | 87.182390 | 86.848879 | 86.803830 | 86.945033 | 0.206787 |
| RLI | 87.185671 | 86.828086 | 86.782683 | 86.932147 | 0.220729 |
| RAND | 87.193461 | 86.834402 | 86.791808 | 86.939891 | 0.220629 |

RLI-H均值−0.012886pp，1/3正，配对样本SD0.014003pp；RAND-H−0.005143pp，1/3正，配对SD0.014095pp；RLI-RAND−0.007744pp，0/3正，配对SD0.001405pp。共享2000图片bootstrap的95%区间分别[−0.052050,+0.023902]、[−0.035741,+0.024776]、[−0.017489,+0.000762]pp，均跨0。不能宣称显著退步；当前influence选择未验证出稳定优于H或随机对照的收益，预注册继续条件不满足。RAND仅匹配每candidate的删除patch数，不匹配删除owner像素面积。

RLI每seed平均删除BG FP42001.667、正确FG34959.333、错类915.333、ignore27447.333像素；原H的BG FP平均180031.333。在原H extra的owner像素范围内，RLI平均seed BG去除率23.5052%，正确FG保留80.9871%；RAND为21.4065%/84.4918%。这些比例不代表完整分割图的全部前景，原E前景始终不动。删除valid像素中BG比例的seed均值RLI54.3423%、RAND56.9361%，RLI三个seed分别也都低于RAND。相对RAND，RLI多删BG FP3709，同时多删正确FG6271（以保存retention表逐seed求和为准，纠正前段初步文字约6272的转录误差），多删错类62.333，并增加FG FN6333.333。像素净变化和macro IoU权重不同，不用原始像素差替代mIoU。

H extra候选数seed42/123/2026为154/163/159，总476；原selected patch数33641/45509/41211，总120361，三个seed各一候选支持<=1保持不删。原I单删判定harmful数10528/15145/13049；greedy实际删除6246/8288/7744，共22278patch，发生删除的候选137/149/146。平均candidate删除比例约20.29%/20.22%/21.05%，原类score平均提升0.011542/0.012802/0.012348、margin提升0.065376/0.054635/0.061167；语义分数提升并未转化为更好的像素分割。已删除patch中owner GT多数类为background4130、correctFG4001、ignore3251、mixed32、wrongFG149，另10715个无最终H owner像素、诊断unavailable；不能把所有被删token都当实际删掉图像区域。

RLI-H正向类chair+0.145789、people+0.098492、table+0.089662pp；负向sofa−0.323551、pottedplant−0.141173、cat−0.074157、horse−0.062946、sheep−0.043264pp。RLI-RAND主要损失pottedplant−0.065374、sofa−0.064546、cat−0.033500pp。完整21类、转移/ignore及删比例linear三分位owner-only非可加反事实保存在CSV和posthoc_diagnostics_001；不同seed分组方向不稳定，不反向调阈值。粗72cell删除与语义influence/像素质量不一致是当前方案边界，不能由此宣称任何patch influence方法都无效。

小结果已逐SHA下载103文件（33小结果含visual index、70图，共48467351字节）到`experiments/hpc_results/h_patch_refinement_20261021_172500/verified_20261021`。download_verification SHA=`2ce7ab3ea11527da39e523adef1460f32d37362ff529262eac58c81106f54f52`。只抽查三张真实12面板：random2007_000847为真实无extra/no-op；helpful2007_004189按三seed均值入选，但展示seed42无实际删除，不能拿该面板证明seed42获益；harmful2011_001190展示seed42删除35owner像素，其中25正确FG、0BG FP。GT均为最后栏，排名用三seed均值、展示seed42，单图IoU非数据集可加贡献。没有为抽查重新读取GT或推理。

analysis_report.md SHA=`ddda3152d2e03594be947e818f407d9082e6e8445c79c1278f393ca008dc6ebb`；posthoc_diagnostics SHA=`ab68eecd4005fc90081516d4090a1e5c711e86eca3859ab8825da62050a177c4`；derived_quality_counts.csv SHA=`d9edd653e89152dab1002352219af9086ebbb996e5ab29d2db778cc53654ce38`；visual_QA SHA=`1d1b354ced28b28bc360f9bff9d115b5de47048753c65e5d97f5a8ae977039a0`。这些独立小分析、资源/QA与下载proof已上传当前RLI run并逐远端SHA核；local_archive_001.json SHA=`7cdfbfb4b12f23429d966f61c8397bc3e0dc312bee639c51af4f461960a11ebf`。原正式summary/metrics/CSV及冻结科学源码不改。

RLI科学整体Slurm墙钟12213秒（3:23:33，最早start17:44:14至最晚end21:07:47，北京时间），不累加四worker并行时长；各角色耗时见90.2与resource_snapshot_001，独立审核36:12另列。保存的真实单worker最大torch.cuda.max_memory_allocated=90.86767578125MiB，非整卡资源申请量。RLI固定批已验收归档结束，不追加研究；双批joint建议及监测退役待T4独立验收归档后仅写报告，不执行联合实验。

### 89/90.4 T4最终独立验收与五问诊断（2026-10-21）

T4独立审核3764455已COMPLETED0:0、57分40秒，completion_acceptance_001 SHA=`78f5c1e60d76892b3b5ecdb773c40e2103302077cbf26af3d04ba03cd2bdacc9`。六科学child均COMPLETED0:0；5796真实crop、4347新PNG/NPZ/owner、原模型库字节SHA、九矩阵逐图求和/E与H历史、CSV/2000bootstrap/几何与内容SHA均通过；不读取GT/token/modeltensor、不重做方法或bootstrap。图示与指标均来自原先完整GT pass，当前抽查没有额外GT解码。

| 方法 mIoU(%) | seed42 | seed123 | seed2026 | 均值 | 样本SD(pp) |
|---|---:|---:|---:|---:|---:|
| E | 87.183731 | 86.780938 | 86.681601 | 86.882090 | 0.265908 |
| H | 87.182390 | 86.848879 | 86.803830 | 86.945033 | 0.206787 |
| T4 | 87.027983 | 86.727243 | 86.608369 | 86.787865 | 0.216276 |

1. 四等分是否提升：T4-E均值−0.094225pp，0/3正，配对样本SD0.054169pp；2000共享图片95%区间[−0.427682,+0.183490]pp。
2. 是否超过H：T4-H均值−0.157168pp，0/3正，配对样本SD0.036990pp；图片95%区间[−0.482462,+0.144537]pp。两区间均跨0，不称显著退步；预注册正向均值/多数seed条件不满足。T4每图4次encoder对H的2次，不是算力匹配。
3. 小目标/局部尺度：实际accepted数seed42/123/2026为363/347/358，共1068个模型-candidate实例；原图mask面积<1%有576（53.93%），<5%有928（86.89%）。平均mask/tile面积9.3689%、mask/original2.3429%，数量主要是小mask。但mask面积不是GT物体尺度证据，不能独立归因于局部scale。<1%组每seed平均仅恢复9561.333正确FG、BG FP24248，valid precision seed均值27.1450%；>=10%组恢复98024正确FG、BG FP124090.667，precision43.3788%。有效像素恢复并非主要来自最小mask。
4. hard boundary：接触内部边界组每seed平均accepted234、正确FG200672.333/BG FP253723.667，valid precision均值43.2445%；未接触组accepted122、正确FG29269.667/BG FP71043.333，precision28.0837%。接触组体量较大，不能仅用其错误总量判断失败；其precision并未更低。当前描述性分组不支持认定hard boundary是主要失败来源，也不能证明边界没有因果影响，不据此追加overlap。
5. 下一阶段proposal source：不推荐采用当前T4或继续扩张tile预算。固定四象限在三个seed均未提升，保留H参照，不追加overlap、尺度/位置/seed扫描或联合。

T4新增E背景owner像素每seed平均723971.667：正确FG229942、BG FP324767、wrongFG14895.333、ignore154367.333；valid precision seed均值40.4400%，precision_all31.8460%（ignore在分母但不当正确）。每model每实际forward正确FG39.673、BG FP56.033、wrongFG2.570；这些效率不是mIoU贡献。相对H多恢复正确FG44540.667，同时多BG FP144735.667和wrongFG8011。T4-H的FG FN下降52551.667包括新增错误前景，不全是正确恢复。

TL/TR/BL/BR的新增valid precision seed均值34.6290%/62.7560%/50.8232%/24.8886%，各位置体量/内容不同，仅作事后描述，不改角落规则。T4-H正向pottedplant+1.059914、car+0.520739、people+0.447185、sheep+0.412790、chair+0.379666pp；损失bottle−2.808518、sofa−2.290168、bird−0.959296pp。去最大正类pottedplant后，其余20类均值差−0.218022pp；正式21类差仍−0.157168pp，不以删类指标替代正式结果。

已下载101文件（31小结果含visual index、70图，共33559575字节）到`experiments/hpc_results/t4_hard_20261021_172500/verified_20261021`，逐SHA核通过。download_verification SHA=`776ea799f70cbce2b62c4b96f55f1a2c2733169b07392a89f85434fa286f8810`。真实12面板抽查：random2007_000033飞机是无新增owner/no-op；helpful2011_000112水鸟中E空、H恢复左鸟、T4恢复两鸟；harmful2010_002142在已有鸟前景外把喂食器大片误填前景。GT均最后，展示seed42/按三seed均值选组，单图macro非可加整体贡献；不能用有益一图反驳整体负方向。

analysis_report.md SHA=`d452b8dd3350b28591417f4aea7de65b14f0b78a7c317aa12ddaf210a17985bd`，posthoc_diagnostics SHA=`6038dc6cc874e202d5f07200d1a5bf4b41677b84d824a0cd2c1cf839c4e46e62`，derived_quality_counts SHA=`9e31aecdc27e1a22975ddcd07e07ea2cfaa3281bc7011ae2e9acac89059beb4d`，visual_QA SHA=`720ed90e9cb5cbe67421dbf27f0d8b0400659cbf304768926574be4b2b3c9820`。独立分析只读保存JSON/CSV，原accepted_count默认占位另用真实flow补充，不覆盖正式metrics/summary/CSV或冻结科学代码。

T4科学整体Slurm墙钟13098秒（3:38:18，北京时间17:46:53至21:25:11），不累加四worker并行时长；角色时长见89/90.2，独立审核57:40另列。保存的真实单worker最大torch.cuda.max_memory_allocated=5228.16455078125MiB。技术探针8forward已计入5796固定预算；没有额外crop、训练/新库或原图SAM3。最终小分析远端回传、联合建议只写报告及手册同步凭据在后续收尾段登记。

### 89/90.5 双批最终归档与Recommended Next Step（2026-10-21）

两批状态均为`complete_verified_archived`。T4小分析/下载proof/真实QA/资源及派生计数已回传当前run逐SHA核对，local_archive_001 SHA=`05fe103f7481b8b8e209a87ba3004574fcac6f871513f4419ef217abd7bb2cd3`；RLI同名归档SHA=`7cdfbfb4b12f23429d966f61c8397bc3e0dc312bee639c51af4f461960a11ebf`。两个download_verification和归档引用的本地文件均再次逐SHA核对，未下载模型/token或批量原预测，没有重跑GT数值评分、bootstrap或任何科学计算。

仅在两批独立验收/归档后运行外层`finalize_joint_recommendation.py`，依据T4-H和RLI-H各自mean>0且至少2/3 seed正的固定条件生成新报告。二者均未满足，`JOINT_RECOMMENDED=false`；RLI-RAND三个seed也均为负。该方向条件不是显著性判定，五个共享图片区间均跨0。两run的joint_recommendation_001.json字节一致，SHA=`3fc834277a223a7fd755daec8652d038fd3ba1273d869e897d1bd190f913c6e1`。没有运行联合实验。

各自新summary_final.md保留原summary正文，附已验收分析/字段说明/Recommended Next Step，最后一行均为JOINT_RECOMMENDED=false；原summary/metrics未覆盖。T4 summary_final SHA=`35a0f873514a4f553f48ae54308d1bf2c8ca5c25b47e67c2c4e80529d271228c`，RLI SHA=`2788cec8287e932a3dd40686ecb1c8a5cb861497a77f1c5db9e6b3dcbd8171e9`。这两文件和建议均已上传各自run且远端SHA核对；final_report_archive_001.json分别SHA=`8d9445981181736236e17c940cd698a6d5bc999a2e9be7f4d2ec9ceff810865f`（T4）、`508297ed3a806ed4aea757a1c48889dafa50fd4399982ab7892778be03161e6e`（RLI），其引用文件本地也逐SHA核对。

**Recommended Next Step：保留已验收H参照，停止本固定批，不采用当前T4或RLI规则。若未来另获授权开展研究，应优先诊断新增proposal的precision/像素错误代价，而非继续增加proposal source预算；本次RLI负结果不证明其他precision方法必然有效。** 不自动T4+H/T4+RLI、overlap、尺度/位置/阈值/seed扫描、新训练/库/标签、部署/GitHub/打包或邮件排障。两套36目标集群检查、各自真实技术探针、回顾startup和最终独立审计均保留其证据边界。固定批已无待计算任务，手册同步核SHA后退役t4-rli监测，不恢复其PAUSED状态；最后只发一次完整结果通知。

## 91. Patch Foreground–Background Separability Diagnostic v1（PFBS，2026-10-21附件授权）

这是固定H的独立诊断，不是新分割/mIoU实验。计划、原附件与精确规则在`experiments/plans/patch_fb_separability_v1/{plan.md,request.md,protocol.json}`，版本`patch_fb_separability_20261021_234500`；入口`tools/diagnose_patch_fb_separability.py`，新增core/runtime/analysis均在tools下。run=`experiments/patch_fb_separability_v1/版本`，计算缓存=`storage/diagnostics/patch_fb_separability_v1/版本`，audit/code与log各用独立版本。旧SMC/RLI/T4/R3、生产src/configs、原H/模型库均不改；0 SAM3/训练/新库/新crop/新预测/新mIoU，不carve/raster或筛选patch。不自动跟进refinement、阈值/邻域/分辨率扫描、部署/GitHub/ZIP/邮件工作。旧t4-rli监测已实际删除，不重启旧批。

已native SSH只读核来源：RLI manifest SHA=`13eed5332fae8e3a8231efa32c40d90286f05bd0350a647c1c8deff93c08b262`、independent accept SHA=`7840a0b7af5e6297b4d4bda45f48884f3cdb4c7349cedd01fde6d4387c84abab`。其SMC H accepted source/owner/掩码/类别/顺序固定；476 candidate-model实例、120361 selected-patch行，三seed42/123/2026。原RLI trace_no_GT的original_indices和single gain只读join，不读取旧patch_influence_GT作为新推理输入，不重跑RLI。三套bank meta均21类，id0为background；原retriever确实返回按该metadata排列的完整21分数，能合法计算c-bg，无新背景prototype。

对每实际需要的原图token i，真正调用原冻结`uniform_gmp(F[i:i+1],[0],rho=.1)`，FP32原Cholesky/残差1e-4/最终L2；不跳成direct token→MLP。技术探针还比较原整token输入`uniform_gmp(F,[i])`与slice调用逐元素相同，记录真实finite/norm/Cholesky/residual与full-I H baseline三seed分数回归<=1e-5。同image相同singleton GMP可共享，三个原MLP/bank检索仍各自独立；缓存仅实际需要的patch，不预扫12031旧token或全库。N=1若不支持即SINGLETON_GMP_UNSUPPORTED停止，不补邻居/换pooling。所有seed全量记录raw/projected方差、class-score方差、top1 entropy；明显退化时结论只能INCONCLUSIVE/INCONCLUSIVE_SINGLETON_REPRESENTATION，不由entropy为0单独判退化。

局部证据score_c、candidate-class margin、合法c-bg及production top1/tie语义和独立top1 margin保存；背景不可用分支必须UNAVAILABLE并保留其他分析。每patch的完整21分数与原candidate/seed/patch身份保留。先合并并SHA锁定`patch_evidence_no_gt.csv`与`patch_evidence_commit_001.json`，然后本批才读取GT。GT仅标注整数72cell∩H final owner，分别记correct/background/wrongFG/ignore；NO_OWNER/NO_VALID不进primary。purity_balance=(correct-bg)/valid，PURE>=.80固定，MIXED不进二分类；不把整个cell或完整mask替代owner交集。

主要Spearman，二级AUROC/AUPRC（同阈值grouped AP）、均值/中位数/Cliff delta。逐seed后等权macro/sampleSD。2000次PCG64 seed42、从完整原1449 image IDs共享重采样，整图内全部candidate/patch保留，精确频数权重midrank/ROC/AP与显式重复图样本等价；不使用patch iid CI。原gain按原方向报告，另预注册负gain作为higher=FG的有利RLI基线，不在GT后翻方向。每个local指标都在相同joined行上与两种负gain比较。附件“明显优于”补齐为：对两基线均三个seed配对AUROC差>0，且至少两个seed配对image95%下界>0；同时需该local指标三seed Spearman>0/AUROC>.5、至少两seed绝对AUROC95%下界>.5。每所用区间至少1000/2000有限rep；其余方向不稳/明确不优或不确定按protocol分别NOT_SUPPORTED/INCONCLUSIVE，不强行通过。多个预注册指标和已看VOC开发集的条件区间均不能当独立test或训练seed方差。

20类分解AUROC须两纯组各>=10，否则UNAVAILABLE。candidate score/proposal/margin按seed的candidate级linear三分位、ties不拆/空组null；新颖度仅原center-reference固定5bin，不gate。四组图示各请求20实际patch：高证据正确FG、低证据BG、高证据BG失败、低证据正确FG失败，按保存分数排序的posthoc极端案例，不拟合high/low阈值；不足如实。RGB/Hmask/cell/owner交集/GT最后栏，GT palette由本次composition pass缓存，绘图不重读GT。数值/CSV/bootstrap先numeric_completion再图示，数值全量已完成不能因图示失败重跑。

调度拟为CPU inventory→GPU technical→四GPU evidence→CPU analyze，全部afterok；一GPU一worker，各worker含原三MLP。CPUdebug最多2h，只有明确QOS无JobID拒绝才唯一GPU资源/CUDA隐藏后备，记录真实分配。使用已验证nativeSSH/SCP、固定ED25519/BatchMode/StrictHostKeyChecking，私钥仅本地读，不输出上传。GPFS实际mmlsquota余>=5GiB最多3瞬时尝试，不df。控制session未返回不重复mutation；纯GPFS等待不当退出。

30本地CPU检查无skip，包括合成2000bootstrap→四组80图示尾部、加权ties/空组/背景不可用/退化/决策分支；本地torch缺失，不称真实singleton GMP/CUDA已过。实际JobID/manifest及真实两图技术结果待注册后补；此段不代表已启动。真正四worker首产物/Slurm实际命令/GPU UUID验证后才首次正式startup通知。最终唯一独立Slurm只读审核核7科学child、完整120361行、H/模型库/源代码与产物SHA、score/composition守恒、保存bootstrap draws/replicates区间/decision及80图；不解码GT/原token/modeltensor，不GMP/检索/重采样。controller verification不是独立accept。

验收后仅小结果/必要图下载逐SHA并真实QA/回传proof，三张大patch表留HPC完整签名产物，不下token/模型或大量原预测。手册91同步核SHA，固定批验收归档和一次结果通知后删除本批监测；只输出FEASIBILITY_SUPPORTED/FEASIBILITY_NOT_SUPPORTED/INCONCLUSIVE及Recommended Next Step，不执行推荐新方法。用户暂停/取消立即停本ledger和监测、不自动恢复。失败保留真实attempt/异常，仅取消自有未启动依赖；实现修复新版本，不能改运行快照或绕SHA。

### 91.1 实际冻结与一次提交（2026-10-21 北京时间23:47）

快照257来源文件和12本批文件逐SHA核通过。manifest SHA=`acbfb415cacfe7d96427b49c4c2526785e2f7ef5b40793d4ae6ff834610a72f3`；binding=`63b83d8a0116633ddd7d53f8f43fb5f0be09bce7620534179ef5672f81c156cf`。audit/jobs.json实际inventory3765586（debug8CPU23000M2h）、technical3765587（v100g32单GPU4CPU24G2h）、evidence3765588 array0-3%4（每worker单GPU4CPU24G8h）、analyze3765589，四阶段全afterok成功提交，禁止重复upload/prepare/sbatch。analyze原debug8CPU23000M2h明确QOSMaxSubmitJobPerUserLimit且无JobID后，仅一次v100g32单GPU资源配额6CPU24G、CUDA隐藏CPU后备；resource_fallback_analyze_001保留，不是科学失败，也不称零GPU资源分配。

当前163张有H extra图的源清单核验中，覆盖476候选实例/120361 patch行，未把无H extra图片当GT筛选；bootstrap仍完整1449图总体。首次实际inspect核inventory3765586 RUNNING（c21507），其余依赖等待。无真实技术通过或最终可分性结论，不能将30本地CPU/synthetic检查当N1 CUDA已通过。所有提交session已返回；GPU代码/规则冻结。后台`pfbs`监测15分钟，仅首次正式验收、失败/必需行动、完成时通知；正常推进静默，固定批完整收尾后删除，不追加研究。

### 91.2 无GT清单完成；真实singleton探针等待资源（2026-10-22 北京时间00:03核实）

inventory3765586已COMPLETED0:0、10分38秒，实际所有H accepted mask/index/owner/source trace核验完成，确认163图、476候选实例及120361 selected-patch行；该阶段只读已有H/RLI候选、mask/owner和源PNG内容SHA，并核对正式hard-center索引；未读取真实GT或原图token，未运行GMP/SAM3/MLP检索，也未生成新预测。technical3765587仍PENDING，squeue明确原因Resources，其余afterok等待。这不是技术失败，也尚未验证N1数值路径；不能提前给出可分性或GPU通过结论。监测继续本固定ledger，不改变资源/参数或重复提交。

登记账本与manifest SHA已再次只读核对并保存plan/launch_receipts/版本/registered_001.json；手册上一同步SHA为`3cbb6deab19d72a081869dbb5e46ec6c37cec62fa369c132d026a8ed745cea3d`，本段后另存新同步proof。当前只剩一个有界本地只读watch（不修改任务），所有上传/注册/调度mutation session已返回。后续按work_state_001.json及实际Slurm更新，不套旧T4/RLI状态。

解释补注：MIXED是既非FG_PURE又非BG_PURE的注册分组，可能包含几乎纯其它前景类别的wrong-FG，不可把其全部比例直接当作几何上FG/BG混合程度。AUPRC随机参照取决于纯FG样本比例，可由各seed n_FG/n_BG得到；0.5不是统一AP零假设。两点不改变任何冻结指标、分组或决策规则。


### 91.3 首次技术启动失败与纯调度显式恢复（2026-10-22 北京时间11:03）

用户查询状态后实际SSH核：inventory3765586已成功；technical3765587 FAILED1:0/6:49，异常FileNotFoundError configs/cls_voc21.txt，发生在分类器import。冻结CODE中的该文件实际存在且SHA20b951be30c19d2c558164baee2b524dc8d5047736dd9380f78b4b373617f9fa匹配manifest；原Slurm入口未cd到CODE导致相对路径失效。没有进入真实singleton GMP、全量证据或本批GT，30集群测试即使经过也不能当N1技术通过；没有最终可分性结论。

严格核原worker已退出、无lock、全部冻结code与输入anchor/已完成inventory SHA一致后，取消仅本批原未启动evidence3765588；原analyze3765589已CANCELLED，旧依赖均零计算。保留原technical/attempt_001/failure.json，SHA=a7f99eac18db68888ac9a75daceaf959f0a148b44211e72947c82659571c9d7c；保留原账本jobs_original_attempt_001 SHA=c17e44bda5412573a112194bfab73e7f0adf3e512b21f4f1b8e0175807866581。外层恢复preflight SHA=a8aafdf2e8affde8c979c5b10f3502827911a04ac88f21ef5f3e6f7085a168cd。未修改运行快照、科学参数、manifest或binding，未重跑inventory。

纯调度一次显式同版resume已成功提交：technical3768931（原CLI --resume创建attempt_002）、evidence3768932 array0-3%4、analyze3768933（debug8CPU23000M2h）；技术到证据到分析全afterok。每个新job设置--chdir=冻结CODE；audit外层technical wrapper仅原冻结sbatch末尾追加--resume。当前analyze提交没有QOS后备；初次资源拒绝历史仍保留。不得重复scheduling_recovery/resume/upload/prepare/sbatch。当前ledger保留scheduling_history/原failure和preflight，guard与resume完成proof已存在，具体SHA与新Slurm状态在plan/launch_receipts/版本/scheduling_resume_001.json。

独立外层审计002已准备上传，保留001，最终local_control audit按scheduling_history选择002；核当前7成功科学child，严格单列原FAILED及零秒CANCELLED与精确failure SHA，只允许该已知历史失败，无新failure/锁；不伪造旧成功、不提前审核或重复科学计算。监测pfbs的app状态实际PAUSED，本次保持PAUSED；Slurm链独立继续，不自动恢复通知。用户状态询问不解释为恢复监测授权。


### 91.4 恢复后的真实singleton技术及正式启动验收（2026-10-22 北京时间11:08）

technical3768931 COMPLETED0:0、1分53秒，30目标集群测试无skip；首两真实H图2007_000529/2007_000636×三seed完整H baseline label/21类score/margin回归、全部selected singleton正式GMP、首至多8个整token索引与singleton slice正式GMP bitwise/finite/L2/Cholesky/residual检查通过，背景id0可用，未读GT/未新SAM3/未生成预测，技术输出按同binding正式复用。report SHA=796b35dee1b93769a59a3b0be730d3c407e8b45a7bd5a680cd2d535920725b57。技术torch.cuda.max_memory_allocated峰值88.1279296875MiB，真实GPFS余232037089280字节；这是探针技术证据，不是120361行全量可分性或方法有效性结论。

四evidence worker各progress>=2后独立外层startup_validation唯一执行passed/formal_start_reported=true，SHA=cee1d295b2ec55699f16359970a2b9937745c1ff3ce72383560a04a8009344ee；核实际精确Slurm子任务/PID/proc冻结命令/单卡UUID及末份FP32 GMP与各seed21类score、源token SHA引用/无GT。完成worker走真实complete+sacct分支不伪造存活。两份小proof原字节已下载逐SHA到plan/launch_receipts/版本/exact_startup_proofs_20261022。当前快照进度41+30+30+30=131/163有H extra图，之后以真实Slurm/progress为准；analyze当时依赖等待，未给最终可分性。正式启动仅通知一次，禁止重复startup/注册或恢复提交。监测PAUSED保持，科学afterok链独立继续。


91.4阶段补记（2026-10-22 北京时间11:11）：四evidence child均COMPLETED0:0（3:01/3:49/3:46/3:46），163图全量证据计算已完成；analyze3768933已RUNNING1:28，无新failure。当前尚待GT后验统计/2000共享图片bootstrap/图示及独立验收归档，不能将N1技术支持当最终FEASIBILITY。所有GMP/检索/证据已完成，不再启动该部分。


### 91.5 最终独立验收、七问结论与归档（2026-10-22）

**FEASIBILITY_NOT_SUPPORTED。** 当前注册的singleton正式GMP→原MLP/库局部证据没有满足“higher=正确candidate前景”的三seed正向连续/二分类条件；不执行Foreground–Background Evidence Refinement，不事后翻号或换阈值继续。该结论不是对所有SAM3局部特征或所有precision方法的不可能性证明。

七个当前科学child均COMPLETED0:0：inventory3765586 10:38，恢复technical3768931 1:53，evidence3768932四child 3:01/3:49/3:46/3:46（实际child3768938/3768939/3768940/3768932），analyze3768933 5:13。120361 selected行/476模型候选/163支持图的证据、GT composition、2000共享图片bootstrap和四组80图全部完成。所有本批证据在GT前commit，SHA=`2d843a854625396381639786f50def291e3d939dfe82a18717a37fc4e016e25e`；GT读取163张支持图一次、图示使用保存preview，bootstrap总体仍原1449图片。没有新SAM3、训练/库/crop、预测mask或mIoU。

独立只读审核3769152 COMPLETED0:0、40秒；外层verify_completion_002保留001，严格验证原技术失败与零秒取消历史而不删除或伪装成功。completion_acceptance_001 SHA=`405fa41a899068b5d460999d84ad2658914a6f99f7da7df8c835f912d63a351d`；summary SHA=`5d61f18b04353541828a7c1aeb66a54b989c8ee995a02fac477d39d417544496`。核当前7成功child、完整证据行身份/FP32分数/原模型库字节SHA、score推导/GT composition守恒、保存draws/replicates区间/固定decision和图示内容身份。审核没有GT/旧token/模型tensor解码，没有GMP/检索/评分或bootstrap重算。原technical3765587 FAILED1:0/6:49、failure SHA a7f99eac18db68888ac9a75daceaf959f0a148b44211e72947c82659571c9d7c与旧依赖CANCELLED零秒单独保留。

1–3. 可分性、最佳指标与seed方向：c−BG是预注册主指标，也是三seed描述性AUROC最高的local指标，但三seed rho均负、AUROC均低于0.5。其rho区间均跨0、AUROC区间均跨0.5，不宣称已证明显著反向分离。

| seed | rho | 图片95%区间 | AUROC | 图片95%区间 | AUPRC | FG纯组比例 |
|---|---:|---|---:|---|---:|---:|
| 42 | -0.046233 | [-0.186547,0.122379] | 0.472850 | [0.389246,0.575191] | 0.454170 | 0.469159 |
| 123 | -0.078787 | [-0.213498,0.086709] | 0.454087 | [0.371398,0.550292] | 0.496937 | 0.524826 |
| 2026 | -0.103782 | [-0.210110,0.029326] | 0.438385 | [0.370508,0.517332] | 0.558756 | 0.599398 |

c−BG三seed等权macro rho=-0.076267、样本SD=0.028857、图片95%区间[-0.180227,0.057483]；AUROC=0.455107、样本SD=0.017255、图片区间[0.389995,0.534097]；AUPRC=0.503288。score_c和candidate-class margin的macro AUROC分别0.416310/0.416239，macro rho分别−0.142623/−0.142212，也未满足正向条件。AP随机参照依赖纯FG比例，不是统一0.5；score不是概率。

4. 与RLI比较：保留原single-removal gain原方向，另用预注册negative score/margin gain作为higher=FG的有利对照。严格相同join行中，c−BG相对negative score gain的配对AUROC差42/123/2026为−0.033960/−0.044361/−0.052063，相对negative margin gain为−0.024903/−0.029964/−0.060776。六个主指标配对图片区间全部跨0且点估计全负；没有优于两对照的支持。原RLI未重算，也未用旧patch_influence_GT作推理输入。

5. MIXED分母：Cell72∩候选最终H owner，排除NO_OWNER和NO_VALID后，42/123/2026分别10827/14794/13238有效owner行；其中FG_PURE4944/7526/7761，BG_PURE5594/6814/5187，MIXED289/454/290（2.6693%/3.0688%/2.1907%）。selected总数33641/45509/41211中无owner18645/24999/22660、无valid4169/5716/5313，不把这些当背景。MIXED可含纯wrong-FG，不能全部归因于空间FG/BG混合；低MIXED也不证明72×72足以像素精修。

6. 原因边界：三个seed均未触发singleton数值退化；raw平均pairwise距离²约1.49–1.51、投影总方差0.828–0.861、21score总方差1.289–1.315，top1熵约2.29–2.32nats。真实N1/原H完整I baseline回归通过，background id0和21类分数可用，不能以“数值表示崩溃”或“BG分数缺失”解释。区域训练的MLP/库用于singleton时的粒度/校准不匹配，是相容解释而非已因果定位；不能据此断言feature完全没有局部语义能力、空间分辨率是唯一原因或任一替代方法一定有效。20类/候选confidence三分位/固定novelty5bin的完整输出保留；样本不足的类AUROC/AP为unavailable，不据此调类阈值。

图片bootstrap以1449完整图ID共享重采样，保留图内全部候选/patch、按精确频数处理ties；主表所用区间2000有限rep。这是已查看VOC开发数据上固定模型条件区间，不是独立test或训练seed不确定性；三注册指标的描述性排名不是独立多重显著性证据。没有事后翻号改score、挑seed或扫描阈值。

7. Recommended Next Step：停止当前72×72 singleton-GMP→原MLP/库patch-carving路线，保留已验收H，不执行真正Foreground–Background Patch Refinement；其它precision证据只可另获授权后研究。没有新mIoU，不更新最高记录配方。

小结果已逐SHA下载105文件（25小结果含visual index、80诊断图，共10774397字节）到`experiments/hpc_results/patch_fb_separability_20261021_234500/verified_20261022`。三张大patch表留HPC签名产物，未下载token/model或批量原预测。download_verification SHA=`f9e77be9346ccede37c1bc7bc19b7521404487de241a3e19c3b8025ffabb6a8f`。真实查看四组各第一张五面板，GT均最后：high-correct马seed123 c−BG≈+1.382、low-correct同马候选seed2026≈−1.224；high-background people seed2026≈+1.354且GT纯BG；low-background ship seed42≈−1.281且GT纯BG。cell/owner交集在缩略显示中很小，QA只证明抽查四张实际反例与布局/文字存在，不外推80图或总体频率。

独立七问analysis_report SHA=`43c75fa8b312357853a874c236fe50bccefd878078c30c42b56b0c9390a8df14`；visual_QA SHA=`b90d4a4c3f6cc57f5eb9940e6381f59d1d678910a4ac51914ddb2e378446591f`。分析/QA/downloadproof已回传当前run逐远端SHA核；local_archive_001 SHA=`aae573a567fc4e3829305280285328b33f50e2cd2a5c99172537ef940352d96b`。原正式summary/metrics/CSV、冻结code和旧失败不覆盖，不读取GT/重bootstrap。

科学整体Slurm墙钟41165秒（11:26:05，2026-10-21 23:47:33至2026-10-22 11:13:38北京时间），包含资源等待与隔夜恢复，不累加四worker并行时长。恢复链11:02:43至11:13:38为655秒（10:55）；独立审核40秒另列。最大保存torch峰值为technical88.1279296875MiB，四evidence worker各45.763671875MiB，这是实际allocated tensor峰值，非整GPU资源申请量。原analyze首次QOS拒绝及CUDA隐藏GPU配额历史保留；恢复analyze与审核实际debug CPU。固定批已验收归档完成，暂停中的pfbs监测在本次最终结果通知时删除，不追加实验/训练/部署/GitHub/打包。

FEASIBILITY_NOT_SUPPORTED
Recommended Next Step：停止当前路线，保留H；不执行refinement或新增研究。


## 92. SMCF-v1：SAM3 Spatial Mask Confidence Feasibility Diagnostic（2026-10-22，独立可用性审核阶段）

用户新附件原文保存于experiments/plans/sam3_spatial_confidence_v1/request.md。只诊断已验收SMC H新增candidate最终owner像素中的原生SAM3空间置信度，不做refinement/新预测/mIoU，不恢复T4/RLI/PFBS，不训练或改H/生产。首先审计当前SAM3及H实际代码/缓存，正式实验和GT分析尚未注册。

实际只读来源核：H源manifest429b6851462b6e5ae0dd2908076de90ad36ec1fed6eb3be9c3c94dd47bd3b5c1、独立accepta2baf41d085901be7086cb65a968ef17bfe60fdeb85d27cbe2735f357f64845c。H为476 seed-candidate实例、185唯一candidate ID、163图。仅用已验收no-GT H roster定位原SMC来源，未读RLI influence或PFBS结果。真实H crop查询NPZ抽查字段及冻结producer显示仅保存native/mapped二值mask、scalar score/类别及原区域特征/检索字段，未保存dense logit/probability。

代码语义：冻结tools/experimental_rgr_adaptive.py::decode_crop读取proposer._ground的output[pred_masks]；mask predictor为MLP与pixel embedding点积，未经sigmoid。原H完整20类5桶、1008 FP32、正常presence；按crop原生尺寸F.interpolate bilinear align_corners=False，再sigmoid一次、概率>0.5，整数crop粘贴回原图，原area/NMS/hash/cap。generic Sam3Processor字段masks_logits赋值前已sigmoid，不能被名字误导为原生logit。实际shape/dtype/range和二值重现仍需真实GPU探针。

独立availability版本smcf_availability_20261022_114000，audit根experiments/hpc_launch_audits/该版本；从已验收SMC快照逐SHA复制247文件，两个新增capture/probe文件逐SHA核，代码/输入已冻结。唯一GPU探针3769673（v100g32单GPU4CPU24G30min）已实际提交；固定10唯一H candidate/10图、每view5个，10原crop forward加首crop一次确定性重放，共计划11forward。核完整历史crop pool的ID/score/source class/native/mapped binary和SHA精确，保存真正pred_masks及原resize后的logit/概率；不以IoU近似放行。尚未宣称真实10候选通过或空间可分。状态入口availability_control.py status/download，严禁重复prepare/submit或修改运行快照。通过后先下载spatial_evidence_audit.md/json及technical_reproduction_probe.json逐SHA，再实施固定三seed全owner诊断。


### 92.1 Availability通过、正式固定批注册与启动验收（2026-10-22）

availability3769673已COMPLETED0:0、4:10。10唯一H候选/10原crop、两view各5个、首crop额外一次repeat，共11真实forward。完整历史pool的IDs/score/sourceclass/native和mapped二值/hash精确，repeat的native/crop logit/probability逐值一致。实际pred_masks FP32[4,200,288,288]，所测原生范围[-47.69086837768555,16.4572811126709]，峰值5433.92529296875MiB。spatial_evidence_audit SHA=19e3e90115be1708245c8ad35b424d1e1e670618a04734e5d8e7344223dfa807，technical probe SHA=5cbd76a4aa3b18ca43493ceb3da92f12dadac98c071c799bf6625b2c1d91fd31；已原字节下载逐SHA。只证明合法dense evidence与精确重现，不是FG/BG可分结果。

正式版本smcf_20261022_121000，run=experiments/sam3_spatial_confidence_v1/版本，cache=storage/diagnostics/sam3_spatial_confidence_v1/版本，audit=experiments/hpc_launch_audits/版本。从可用性快照249文件逐SHA复制并核14新文件；manifest SHA=452dd727ed95516c7b2592bd4b6cd704cd618c7c60e20e94cfd56756ef61f59b，binding=8f9b58fa1c7c2cc59abef8ddfc6dfea1566e75b28a8bea29bf48f03a444b08e0。当前audit/jobs.json实际inventory3770037、evidence3770038 array0-3%4、analyze3770039全afterok。analyze原debug QOSMaxSubmitJobPerUserLimit且无ID后唯一v100g32 1GPU资源配额6CPU24G2h、CUDA隐藏CPU后备；不当方法失败，也不说零GPU分配。不要重复prepare/submit/register_recovery/sbatch或修改冻结科学code。

首次外层注册在生成manifest/JobID前因误写历史质量表extra_pixel_quality.csv而停止；0正式计算。原script/registration_001.guard留存；prepare_submit_002仅改指向实际extra_pixel_quality_H.csv（SHA=e1a18f05375bcfb5fb613306eaa5d01317e73bb6d4575de636dcfd631a5d643c），保留registration_path_recovery_001，不改冻结科学code或参数、不重复availability probe。正式分析只在本批evidence commit后解析该历史GT统计用于整数零容差H owner核对。

inventory3770037 COMPLETED0:0、1:53，23目标CPU检查无skip：包含精确weighted-ties/image multiplicity、完整合成commit先于mockGT一次→2000bootstrap→数值checkpoint→80图示尾部，不代表实际可分性。直接H源核出476 seed-candidate实例、185唯一candidate、163图、180所需唯一历史crop和1574885实际owner像素行。区别PFBS selected-cell人口；本批不读原图token/GMP/MLP或bank。只为missing accepted target恢复原crop，probe合法复用，实际新增forward在commit逐一记录，180不是新增forward断言，11 probe含repeat。

正式startup_validation已passed/formal_start_reported=true，SHA=25dbbda78695aa82bb6c8f579e67ff497410bedcf1b43a6dd27f7c203099639f，小proof已原字节下载核SHA；四worker Slurm精确子任务/proc/PID/GPU UUID、FP32 logit/probability/owner无GT证据通过，完成worker走complete+sacct不假造存活。可缓存复用的首图不冒称新SAM3 forward。首次启动已通知一次。最近shard0/1/2 COMPLETED0:0（3:27/3:24/3:26），实际child3770088/3770089/3770090；shard3实际root3770038仍RUNNING（progress2/40，c2v100m2-18，PID2156453），无failure，未称全量完成；以实际新日志为准。

固定证据raw resized logit primary、sigmoid一次probability secondary；完整binary mask8邻域边界中心内部距离/max作几何control，非owner边界。全pixel CSV/NPZ/SHA先commit再GT；H四类owner计数对历史_H.csv逐整数零容差。2000共享原1449图片bootstrap，pixel-pooled与image-macro均报，AP对positive prevalence；class最少1000correct+1000BG，view/size/novelty/GT-quality/boundary固定分层只分析。支持条件与inner-bin替代规则已在protocol/plan/代码冻结，不能事后调threshold/反号/挑view。source prompt class可能不同于冻结H预测class，不能把mask membership当语义类别概率。

数值/composition checkpoint先于80图；后段失败只允许新报告/CPU统计版本复用保存数值与preview，不重跑完整GT/bootstrap/SAM3。最终唯一Slurm独立audit核7科学child（availability1+inventory1+evidence4+analyze1），不GT图/token/modeltensor解码或推理/统计重算；controller verification非independent acceptance。local_control.py inspect/audit/audit-status/download/sync-manual；已有audit job/guard只audit-status，不重复verify。

当前smcf监测ACTIVE每15分钟，仅固定批后续完成/失败/必需行动通知，正常健康推进静默，首次启动不重复。完成独立验收后仅小结果/必要图逐SHA下载到experiments/hpc_results/版本/verified_20261022，大pixel表/NPZ留HPC；外层analyze_verified.py十问具体分析、真实图QA、archive_verified.py回传proof并核SHA，再更新本节/同步，最终删除监测。SUPPORTED也只建议另授权refinement，绝不执行。原PFBS已完整结束且monitor删除，不恢复。


### 92.2 最终独立验收、十问结论与归档（2026-10-22）

**FEASIBILITY_NOT_SUPPORTED。** 冻结higher=SAM3 mask confidence方向未满足三seed稳定正向分离与超出boundary control的注册条件。停止当前H内部按空间置信度减像素的refinement路线，不扫描阈值、不翻号继续，不生成refined mask/新预测/新mIoU；保留既有H及最高记录配方。该结论限于当前H新增owner人口和冻结路径，不能推出所有SAM3 confidence均无信息。

全部7个科学child COMPLETED0:0：availability3769673 4:10；inventory3770037 1:53；evidence3770038的实际child3770088/3770089/3770090/3770038分别3:27/3:24/3:26/5:40；analyze3770039 4:52。inventory目标23CPU测试无skip；真实availability精确历史重放及单次repeat独立通过。完整476 seed-candidate实例、185唯一candidate、163支持图、1574885 owner行、2000共享图片bootstrap和4×20图全部完成。

实际180个需要的历史唯一crop box；availability11forward（10原crop+首crop一次repeat），正式170新增forward，复用probe10个候选，共181实际forward。没有新增位置/尺度、原图SAM3、token/GMP/MLP/检索或训练。只读恢复原H dense map，全pool ID/score/source prompt class/native/mapped binary/hash精确。所有pixel evidence先content-SHA commit再本批GT，commit SHA=05fd6701fbf29f2048894be09489390e0b7710db299f8de9c259656f0d11eb61；163支持图GT各解码一次，图示使用保存preview，visual_GT_rereads=0。四类owner计数逐整数对历史H表完全一致。

独立只读审核3770248 COMPLETED0:0、1:08，completion_acceptance_001 status=passed，SHA=cfb09d2dcad8e1b7630036d3e6a8848e92ff481cf7d4f29ec1c369050ea3164a。核7科学child、来源/冻结code/vendor/checkpoint字节SHA、全部证据/映射/H owner/GT composition守恒、保存draws和replicates/区间/decision/图示。没有GT图/token/modeltensor解码，没有推理、GMP、检索或bootstrap重算；controller completion_verification不替代独立accept。正式summary SHA=02e066547e520070fe577e6ac403f298209b3dedce280135b8294dbba2d75040。

1–2. 合法证据与历史对应：真正output[pred_masks]为sigmoid前FP32 decoder输出，按原H bilinear align_corners=False映射再sigmoid一次；processor内同名masks_logits实际已是概率，未重复sigmoid。10候选两view均覆盖并精确重现，首crop dense repeat逐值一致；正式所有target继续严格对应，未以IoU近似放行。mask membership不是冻结H语义类别正确率，source prompt class与H预测class可不同。

3–5. 完整像素池主指标与图片等权口径同时披露，不能选择更好口径或改变分数方向。

| seed | pooled AUROC | 共享图片95%区间 | image-macro AUROC | image-macro区间 | AUPRC | correct比例 | Cliff delta |
|---|---:|---|---:|---|---:|---:|---:|
| 42 | 0.251412 | [0.092171,0.522566] | 0.573497 | [0.434622,0.718047] | 0.315178 | 0.427748 | -0.497175 |
| 123 | 0.278835 | [0.135547,0.497675] | 0.597182 | [0.459822,0.738047] | 0.385956 | 0.500186 | -0.442329 |
| 2026 | 0.294578 | [0.145566,0.504947] | 0.623838 | [0.488524,0.760048] | 0.459128 | 0.579795 | -0.410845 |

三seed等权macro pooled AUROC=0.274942、样本SD=0.021845、图片95%区间[0.130998,0.490177]；AUPRC=0.386754。image-macro跨seed均值0.598173、样本SD0.025185，但三个seed区间均跨0.5，且低于各自boundary image-macro 0.615545/0.645453/0.667993。pooled与image-macro差异涉及图片权重及跨图分数比较，不能把任一种解释为所有单图都反向/都有效，也不能将pooled反向点估计转成事后翻号删像素策略。AP参照依赖correct比例，不是统一0.5。

6. 几何control为完整历史mapped candidate binary的8邻域边界中心归一化内部距离，不是owner边界。

| seed | boundary pooled AUROC | SAM3减boundary | 配对共享图片95%区间 |
|---|---:|---:|---|
| 42 | 0.299137 | -0.047725 | [-0.155456,0.106895] |
| 123 | 0.373791 | -0.094955 | [-0.233528,0.103175] |
| 2026 | 0.389815 | -0.095237 | [-0.241723,0.081990] |

三seed配对差均负，区间均跨0；未建立超出几何control的额外正向分离，也未证明decoder显著更差。proposal scalar只对跨candidate valid precision做Spearman；42/123/2026分别0.504280/0.519460/0.513361，未广播做candidate内空间ROC。

7. 固定边界四层均未满足同一层三seed正向稳定条件，stable_inner_bins为空、stable_outer_bin=false、scope=LOCALIZATION_UNRESOLVED。不能称仅边界已有效或内部随机。8. correct-vs-wrongFG AUROC为0.144257/0.625405/0.166364，方向不稳；wrongFG和ignore始终独立于BG FP。

9. 类别/尺寸/新颖度只作GT事后描述，不反向gate：table AUROC0.035518/0.046838/0.035518，pottedplant0.022046/0.047947/0.019230，sofa0.260712/0.209258/0.379052；chair0.878297/0.752187/0.785069，但只有1503 correct像素且不是另注册类策略证据；people0.434863/0.455846/0.460282。尺寸与novelty弱组具体见十问analysis和原CSV；不足1000correct+1000BG类保持unavailable，不调类别阈值。两个source view和固定GT质量组均保留。

退化检查未触发：candidate-owner raw-logit std中位数1.021825，degenerate=false。这是可用但未支持当前注册用途，不是由于missing evidence/数值常量而INCONCLUSIVE。2000重采样以原1449 image ID共享multinomial、整图内owner保留、精确频数ties/ROC/AP，无pixel bootstrap/跨seed iid池化。区间限于已查看VOC开发集固定模型，不代替独立test或训练seed不确定性；3个seed改变H owner人口，非3个SAM3训练。

10. Recommended Next Step：停止当前H内部基于SAM3 dense confidence的减像素路线，不执行confidence refinement mIoU实验或阈值扫描。若将来另获授权，可研究proposal接受/组合或其它证据；本批不实施。

验收小结果逐SHA下载115文件（含80图、visual index及小表/proof），共9834545字节，到experiments/hpc_results/smcf_20261022_121000/verified_20261022；全pixel CSV/NPZ留HPC，未下载原token/model或大量预测。真实抽查4组各首张7面板，GT最后栏，fixed0..1 probability旁列raw logit；只检查4/80，不外推总体。high-BG pottedplant raw13.539688仍为BG，low-correct sofa raw约0.0000015仍正确，均为事后极端案例而非阈值建议。

download_verification.json SHA=c5fcdca8be02fdd0657a623e4d8880c9cc73bf8c52360ceb69a68fec481c491e。
analysis_report.md SHA=31f0aa22ae7d5d6ec114c408c48eb8e15d22bb8ce638b7b0939101d3ae783d01。
visual_QA_001.json SHA=45290b09dac937e701cf83415f9a83613eddc888eabd934e339890e1831593df。
local_archive_001.json SHA=1a2cf74a4061d2ebed7f8ba2e7e7da2b5246354428f176b89773cc6680c9ba45。

analysis/downloadproof/QA及local_archive已回传当前run逐远端SHA核，不覆盖原指标/code/失败历史，不GT/统计重算。十问完整解释见上述analysis_report.md；正式JSON/CSV、2000draws和replicates仍按原manifest签名保留。

总体科学Slurm墙钟2693秒（44:53），2026-10-22北京时间11:42:27至12:27:20，包含等待，不累加4worker并行。独立审核1:08另列。availability实际allocated tensor峰值5433.925293MiB；四evidence峰值5428.608398/5432.746094/5429.582031/5434.279297MiB，总最大5434.279297MiB，非GPU申请量。analyze因原debug明确QOS无JobID拒绝唯一GPU资源配额/CUDA隐藏CPU后备，历史保留。首次外层注册路径_H后缀纠正发生manifest/JobID前、零计算；本批无科学计算失败。

固定批已完整验收归档；本节同步HPC核SHA后删除smcf监测，不追加任何实验、GitHub发布、部署或打包。

FEASIBILITY_NOT_SUPPORTED
Recommended Next Step：停止当前H内部空间置信度减像素路线；保留H，不执行refinement。


## 93. Robust Residual GMP / Huber IRLS：RTX6000独立实验准备（2026-10-22，尚未运行）

用户新附件明确授权在RTX6000重跑Robust Residual GMP，原文保存于`experiments/plans/robust_residual_gmp_v1/request.md`，固定机制/停止规则见同目录plan.md和protocol.json。该方法不等同第49节旧robust_huber（覆盖率×余弦残差责任）；不把旧结果当本批基线或本方法实测。

已新增独立`tools/experimental_robust_gmp.py`与`tools/robust_gmp_mechanism.py`、对应两个test模块；未修改src/configs/生产入口/旧库或旧指标。算法沿用原hard-center与nearest fallback，普通GMP未归一化解初始化，最终仅一次L2，rho=.1；固定lambda=.1*原始未加权二阶矩trace，w/N而非w/sum(w)，MAD=1.4826*median(abs(r-median(r)))+1e-6，w_min=.05。只delta1/1.5/2与T3/5，primary1.5/T3。FP32更小primal/dual Cholesky、正规方程残差<=1e-4；原production无solver fallback，故失败直接停止，不加pinv。内部autocast关闭，目标启动还须TF32/BF16关闭。

附件标准化Huber目标与显式IRLS方程存在尺度/1/2系数差异，且MAD每轮重估、权重下限裁剪；本实现严格执行附件显式IRLS更新和固定ridge，不额外乘scale改正则。只称adaptive-scale IRLS估计过程，不声称在最小化一项固定标准化目标或已获得收敛证明。

机制预注册：从真实目标host已验收过滤成员及原fit split中抽500前景区域，每类25、原mask面积三分位9/8/8；SHA排序，保留实际身份。要求同图mask外中心token足够无放回注入30%，不够者预先列明，不足固定类/tier配额即停，不偷偷缩量或截断大N。污染0/5/10/20/30%以ceil(r*Nclean)追加、保留原token，固定PCG64嵌套前缀，同时报最终占比；不称mask外token已知GT背景。duplication固定随机10%原selected行，总重数2/4/8，不引入新方向。全部delta/T原样报告drift分布、class/size/N分组、每token权重/残差/scale、solver、仪器化runtime mean/p50/p90/p99及GMP倍率。

gate：10%或20%至少一个ratio整体mean drift严格改善、至少11/20类非负且至少2类严格正；x4 mean robust漂移<=GMP+.005；零数值失败/零fallback。T3/T5所有成对描述符最小cos>.999才用T3，否则T5；通过的delta按1.5→1→2固定优先顺序，仅内部机制选择。无通过即complete_mechanism_failed、VOC NOT RUN，不扩delta/rho/loss/次数。正式A/B只在gate通过后继续准备：同成员/过滤身份/内部split、重新计算robust raw并训练匹配1280-512-256 MLP三seed42/123/2026，clean/HCR各1449、原GMP配对基线和center等实际设置显式绑定，不能复用普通GMP MLP冒充robust正式结果或按VOC选参。

当前实际只读资源检查：通过已固定nativeSSH/SCP连接hpc.chinahpc.com:1014，sinfo只列v100、v100g32、v100m3与debug，没有RTX6000资源；当前用户queue为空。没有RTX6000的SSH目标/端口或项目位置，已向用户请求连接标识；不能推断旧RTX服务器当前地址、数据与部署，也不会以V100代替。**未上传、未提交JobID、未运行机制/训练/VOC或取得新指标。** 不创建无目标的自动监测。

本地绝对Python语法检查通过，6项非Torch采样/注入/gate/统计协议测试通过；本地缺Torch，不能声称真实Cholesky/CUDA已验证。另10项数值/几何/目标CUDA回归已写入，但必须在实际RTX6000上执行，目标共16项无skip才继续。work_state_001.json记录未启动状态和当前代码SHA；实际目标核验后再冻结新版本/源缓存身份、准备fit-only选中token小缓存和真实小probe，绝不启动其它旧实验。当前文件仍为本地准备版，不是集群已冻结运行快照。

### 93.1 用户明确目标为自有RTX6000服务器；认证阻塞（2026-10-22）

用户回复“rtx6000就是我自己的服务器，不是集群那个”。根据本手册原服务器记录及本机TEMP/moduseg_full_control旧连接器，仅只读查得目标zystu2025@183.175.12.79:9470，历史工程/data/home/zystu2025/nd/1010-main、Python/data/home/zystu2025/.conda/envs/moduseg_sam3/bin/python。不能把旧路径或旧双卡清单冒充当前核验。

当前实际nativeSSH一次StrictHostKeyChecking/BatchMode只读连接返回exit1、Permission denied (publickey,password)，没有执行远端hostname/GPU命令，没有上传/写远端/提交作业。旧连接器使用交互getpass，不保存密码；本地没有它的存活控制进程。没有读取、打印或上传密码/私钥内容，没有自动接受新主机密钥，没有用HPC密钥冒充个人服务器认证，也没有转去V100运行。

已请求恢复本机SSH认证或提供可用私钥本地路径（不要求发送密码/私钥内容）。独立personal_server_control.py inspect [--identity LOCAL_KEY_PATH]已准备，只读取目标hostname、GPU型号/UUID/显存/进程、历史项目目录及环境是否存在；使用nativeSSH已有known_hosts严格核主机，不执行实验或连接HPC。work_state/protocol现为blocked_personal_RTX6000_server_authentication、jobs_submitted为空。认证恢复后继续实际源缓存/显卡占用预检和机制批，不自动恢复任何旧实验。

### 93.2 自有RTX6000连接恢复、真实16项回归及固定机制准备（2026-10-22）

用户本回合明确提供个人服务器当前账号与项目位置并授权连接。认证仅通过本地getpass无回显输入进入SSH客户端内存，未写入脚本、命令行、manifest、自动监测提示或实验产物。复用known_hosts并RejectPolicy核主机；不将个人服务器凭据用于HPC。当前真实host=shuxsu，项目/data/home/zystu2025/nd/1010-main、Python/data/home/zystu2025/.conda/envs/moduseg_sam3/bin/python。两张Quadro RTX6000各24576MiB，检查时GPU0用6MiB/GPU1用16MiB、利用率均0、无GPU计算进程；/data余1864413188096字节。实际使用物理GPU1，UUID GPU-edaba350-ba5c-6883-a48f-8e5e9a0c680a；不使用V100、Slurm或GPU0。

只读确认原responsibility来源manifest SHA=9e1ec16a480fdbc27f3c3d9fe64761fd53608f18a22519b54be3bc7256aa87af；原过滤成员34880、原MLP train_images8466/validation_images2116、50epoch/MLP2 v1。该manifest用于索引原GMP token/伪标签/成员/原区域向量，不引用旧robust_huber权重或其质量作为新方法。机制只从原train_images取前景，不读内部query/VOC GT。

首准备版本robust_gmp_20261022_154000在创建run/数值测试/token读取前发现手录训练报告SHA多一个字符而停止，原audit/code/prepare_001.log/launch guard保留。实际报告与原manifest中已签名source_records完全一致；新的准备实现直接核该记录，未降低校验容差。该错误不是方法质量失败，也未进行GMP/IRLS或GT计算。

当前有效独立版本robust_gmp_20261022_154500，audit=experiments/robust_gmp_launch_audits/版本，code=该audit/code，run=experiments/版本，selected小缓存=storage/robust_gmp_mechanism/版本。279个原fresh不可变源码逐SHA拷贝、8个本批新增/计划文件上传SHA精确核验；不改root生产源码/configs及旧数据。真实prepare PID283276已脱离SSH，GPU1可见；目标16项测试于Quadro RTX6000通过、无skip，3.037秒，包括10项数值/几何/CUDA检查及6协议检查，不再以本地无Torch测试代替。当前正在读取原fit伪标签固定500区域及482支持图片的选中token缓存，原full-image token仅实际消费时核SHA，不全工程或12031token预扫描。此时尚未宣称500区域机制或全VOC完成。

外层唯一continuation supervisor PID283326已脱离SSH，只有prepare进程退出且完整mechanism_manifest存在后，才独立检查目标GPU1/UUID/显存占用并启动固定机制worker；guard/launch ledger存在不重复启动。机制完成后自动运行一次只读CPU独立verify_mechanism_001，不解码原token/模型或读取GT，不重做GMP。若机制FAIL，complete_mechanism_failed/VOC NOT RUN即停；PASS仅授权继续另版匹配raw/三seed新MLP/clean-HCR正式A/B，不准按VOC调delta。所有PID与实际进度以status_001.py和各自小日志/complete/failure为准；前述PID283276是准备而非正式机制worker。

### 93.3 固定500区域正式机制启动验收（2026-10-22）

prepare283276已结束，482支持图选中缓存完成；mechanism_manifest SHA=6aae031bc89e9cf32ed63408cefd3f7720061bd514e85f22380eae68fd3fd9f9，binding=b5709401686d2492ad969ef0388ea7ab07a515976805dde56904cd824e97d49e。正式worker PID283333存活，/proc命令指向本版冻结mechanism入口与manifest，CUDA_VISIBLE_DEVICES=1、UUID为GPU-edaba350-ba5c-6883-a48f-8e5e9a0c680a；GPU0保持空闲。500区域、20类各25、3delta×2T×8condition固定不变。

独立外层startup_validation_001 status=passed/formal_start_reported=true，SHA=f2e68c37ef149058f40145b093c4ecbad0c9576e897ff74e9fd2e86a1bf69574。实际核前两完成区域各48行及144个weights/before/after数组、FP32有限、源产物SHA；数值入口逐区域普通GMP对历史原1280向量allclose1e-6先通过后才执行扰动，不能称全部像素bitwise或全500已验收。目标16测试proof SHA=acef6cd8155fdb4069f0618fd86aba38a3273fb6fd3ea26375bca40ade9d36f6。manifest及4个小proof已下载到plan/launch_receipts/本版本，local_proof_verification_001核SHA。最新只读快照351/500区域，尚未最终gate或VOC结论。

首次正式启动已通知，不重复startup/重提任务。审计尚未启动前审阅发现内存class键为int、JSON磁盘键为str；外层审计统一JSON后比较gate，原审计草稿另存preexecution_draft及audit_preexecution_revision_001，科学快照/结果未修改，不是科学失败。audit guard存在后绝不修改该审计实现或重复启动。

### 93.4 机制完成验收、全量接口测试修正及平台转移（2026-10-23）

第93.3节为历史运行快照。实际固定500前景区域、20类各25、482支持图已全部完成；24000 region/config/condition行、12000个T3/T5配对cosine、4000条扰动记录和500份逐token权重/前后残差数组均保留。目标16项RTX6000回归无skip。独立accept status=passed，SHA=f00baf0c1d75af6833a57853c96cad391188573430d5939036b9d2f2b89215df；机制complete SHA=41412f8be8c91f2c6d80ce8c189d49a4000a5429a9353fbb03f46809d1f32e5a。

Mechanism gate PASS，按原优先序固定delta=1.5、IRLS T=5；所有T3/T5最小cosine=0.9975086451，未超过0.999，故不能选T3。10%污染mean drift由GMP0.017289879降至robust0.003363534；20%由0.034809248降至0.010030748。duplication x4由0.002193673变为0.003376580，excess+0.001182907在注册+0.005界内。区域/condition等权weight mean0.5044767711、p10均值0.3005965988、fraction<0.5均值0.5537055909；完整分布/类别/尺寸/N/成本在独立analysis及原JSON中，不能把上述稳定性当VOC提升。

机制audit001真实失败保留：FP32 mean相对min最大低7.45058e-9，307行，非solver失败或真实权重越界。仅外层verify_mechanism_002将min<=mean改为min-1e-7<=mean，并用独立guard002；没有改科学快照、权重、gate或重跑机制。原001 code/log/guard/exit以及更早preexecution草稿全部保留。

条件full版本robust_gmp_ab_20261022_163000已注册，manifest SHA=f5a68e7cbeda413c37db10a9e662d04b858d331935b35a084cc48bbfd5cd5eae，binding=8c73ce066dc6c5f6e5b01c6fc71f7cfc223bff5a28250d2ba32b770912b7c90b。三目标接口测试2过1error：合成32行两类的生产短训练/导出/严格reload已完成，随后测试误要求旧两层MLP的可选architecture.num_layers字段，KeyError。生产FeatureAdapter为兼容旧签名有意省略该字段，load_adapter默认2层。没有真实全量raw提取、三seed正式训练、预测或GT评分。旧冻结代码/log/manifest不改。

独立恢复版robust_gmp_ab_recover_20261023_023500只修测试默认2及实际Linear1280->512->256断言；科学IRLS和完整A/B核心字节保持不变。旧292文件逐SHA核，288文件原字节复制，4份测试/注册/计划文件版本化更新；两个外层调度/独立验收脚本单独保存。恢复manifest SHA=acce0815f75725690859d68e4c94f47a3a338d96109c593729f96a88cc0cb2ff，binding=c13c200873084098476fae4c3397a1934275d831e7700bacb4a6dedd7212208e5。注册核所有21类原成员34880/含10569BG/顺序/split，native两来源向量FP32外L2舍入<=1e-6并逐类记录，不冒称字节相同。最终center固定0.2；训练内部原选epoch center0.4保留。seed42原pair.historical_metrics为center0.4，禁止误用；本批seed42绑定原fresh center0.2的globalCM（逐图历史不可用如实null），123/2026绑定各center0.2逐图矩阵。

个人服务器两张RTX6000当时均被其它ActiveSAM项目占用约9GiB、util约96–100%。唯一外层等待进程PID293176曾登记，尚未进入任何目标测试/真实提取；不干预其它进程、没有切GPU0或V100。用户随后明确“用集群跑也行”，已在本版audit写stop_requested，实际supervisor退出、/proc/293176不存在、run仅manifest且无scientific launch。queue状态字段stopped_failure_or_user_pause来自通用调度器，本次实际原因是平台转移，非方法质量失败或用户取消整个实验；该个人版本不自动恢复。

当前继续核验HPC原同批缓存、模型/库和资源，以另版Slurm执行授权的固定三seed完整A/B；尚未宣称集群已提交或已有VOC结果。已完成机制不重跑，不跨平台混比，迁移必须核输入/原mask/sampleID/原库/模型/源路径映射的字节身份；新设备小probe须真实通过。个人等待队列已停，禁止两边并发重复运行。

机制15份小结果及恢复版小proof已下载到experiments/hpc_results/robust_gmp_20261022_154500/verified_20261023，download_catalog_002与download_verification_001逐文件SHA通过。三大逐token/原token/模型未下载。初catalog001的队列状态是可变路径，保留该目录记录；catalog002改用独立不可变queue_snapshot_for_download_001后再下载核SHA。没有自动打包/发布/部署。

### 93.5 用户允许集群；只读迁移预检（2026-10-23，尚未提交）

用户明确“用集群跑也行”，撤销本批仅RTX6000的设备限制，准许另版Slurm/V100完成条件full A/B；不因此变更delta1.5/T5、成员、split、配方或VOC后停止规则。个人等待队列已停且不会自动恢复。当前没有HPC新JobID或新full推理/训练；机制RTX6000实测保留其设备边界，不能改写成V100机制测试。

SSH ED25519固定指纹及publickey认证已成功，简单远端命令亦存在分钟级等待，部分只读诊断60秒超时；不能误报认证丢失或计算已失败。首个只读cluster_preflight_001最终正常返回：当前zhengyuan squeue为空，v100/v100g32/v100m3 up。个人原gmp_responsibility_20261013_164316、centroid02_paired_20261013_204924及fresh_tuning路径在HPC同名位置不存在，故禁止仅字符串替换root复用个人manifest后直接启动。真实HPC来源需经已验收r3_paired_20261016_142200、context_relational_20261018_121800和oracle_guided_bridge导入索引定位后做成员/原mask/token/原模型与库身份映射；不扫全工程或12031 token。

首quota探针误用filesystem gpfs1，明确返回“File system gpfs1 is not known”，没有用df冒充通过。正确入口/usr/lpp/mmfs/bin/mmlsquota -g 项目实际group data，后续inventory002读取该配额。现有TEMP旧context_relational source manifest仅供查路径提示，实际源仍须在HPC核SHA；不能把本地缓存当当前证据。第二个只读cluster_source_inventory_002正在读取正式R3/model/pool索引及正确quota，没有注册/提交行为。

已创建robust-gmp延续监测，每30分钟继续固定批迁移/运行/验收收尾，正常等待静默，不重复已存在的提交。个人旧连接session95149仅作小证明/源元数据获取；不重启已停queue或干预ActiveSAM。所有科学实现变更另版冻结并做目标小probe；已有JobID/guard绝不重提。最终无论VOC提升、持平或下降，独立验收/小结果归档/一次完整通知后删除监测，不追加研究。

### 93.6 HPC同源核验、不可变全量版与实际Slurm提交（2026-10-23）

当前有效full版本robust_gmp_cluster_20261023_030000，audit=experiments/hpc_launch_audits/版本，code=该audit/code，run=experiments/版本，raw/cache/projected/weights/predictions均独立版本目录。cluster_plan.md、cluster_protocol.json为本阶段计划；旧个人full_plan/protocol作为历史保留。源oracle_r3_context_recover_20261016_113700的397源码逐SHA复制，11新增科学/测试/计划文件及外层控制/导入共21上传逐SHA通过；核心IRLS和原完整A/B模块字节不变，新增cluster适配器只处理已签名HPC路径/输入布局及Slurm阶段。9个原production核心文件逐字节相同，不修改root src/configs/正式入口。

source manifest链已实际核验：context_relational_20261018_121800 SHA8737afdd7f0a887f03345cb9d0e51a0e089d50469171078d25e6fe7b28a8f579；r3_paired_20261016_142200 SHAfe3b9fd63b10ac5aa9f728fa4bbb76515e8630adb20d5c321012f8ce4e90c100；原pool manifest SHAde989140dac7138124a99c2d127b67da80c55c5bd6e5cb4373dee6f00b4a36bc；models source SHA88d4fafdaf84fb43b779e05f440e375499976a128ef19b417fd1caad35c09b7a。34880成员/类/顺序、train10582/val1449列表和signature完全相同；内部8466/2116 split已对原个人training_report逐ID一致。10582 pseudo PNG（含既有35fallback）SHA全部匹配；12031 token记录的original_reference_token_sha256全部匹配个人原文件引用。该项是来源元数据链接，**不宣称全跨设备token数组bitwise**；既有HPC缓存cluster_mainline_baseline_20261014_175149的实际NPZ/receipt SHA及FP32 geometry在消费时核，真实probe另存canonical数组SHA和GMP对原向量maxabs/bitwise。没有重做token/SAM3/伪标签或扫描全数组。

三seed原adapter实际SHA匹配个人原模型；各seed原projected bank24文件的相对路径集合及SHA元数据也全部一致，额外native_banks_migration_identity_001保留本地。注册在HPC核源模型/库实际字节，未读取model tensor。原raw库源/成员和production签名绑定不变。native HCR历史必须使用r3_paired对应seed的B_commit与seed_B_per_image矩阵/metrics.results[seed].methods.B（center0.2）；不能使用margin原reference中心0.4，更不能将R3结果当GMP baseline。

新manifest SHA=c69c54cd47dac1edffccf3d82f167ec0f29b5351a000dfa5508b6c40c697cbfa，binding=f5b86243563f09ed13e021c5b536a5405437c366bcf4beb7dc17ed1e8772f499。register rc0，真实group zhengyuan/data配额余229811978240字节。上传expected目录SHA=4cb71f0f4eecac9b7c0f79d819aeb71d5b09b713c77f8c0ba2d4e38bf1684382。manifest/注册/配额/实际jobs/upload/source-clone/submission小凭据已下载plan/launch_receipts/本版本，不代表full指标或技术验收已完成。

audit/jobs.json实际afterok链全成功提交：technical3779988（1V100）；extract3779989 array0-3%4；merge3779990；train_predict3779991 array0-2%3；evaluate3779992；独立audit3779993。每GPU一worker；三个seed各自完整新MLP/库/分数/预测，提取只共享geometry与1280D raw。merge优先debug8CPU23000M2h，audit优先debug4CPU10G2h；本次提交均无QOS后备发生。evaluate为1V100资源配额6CPU24G6h且CUDA隐藏，纯CPU评测，不能说零GPU资源分配。独立audit已afterok唯一提交，不能再调用verify或另交审计。预计10科学child及1审计child，按实际array子项核，不把parent压缩行误当单child完成。

当前最后只读快照technical3779988 RUNNING、2:09、c2v100m2-2，其余正常依赖PENDING；真实19目标测试及2train/2query/6 native HCR像素回放尚待proof，未宣称通过或full正式startup。技术通过后复用本版probe cache进入4路提取；全部17388新预测PNG/12组SHA提交后才本批GT。IRLS1.5/T5、原BG/成员/split/50x100 CE+.5SupCon、最终center.2/训练内部center.4全部保持，禁新参数/新seed/R3/H/SMC附加及部署。

cluster_status_remote_001.py只读本ledger/sacct/小log/complete/failure/lock。HPC持久控制session98948已REGISTER_RESULT0/SUBMIT_RESULT0/CONTROL_DONE；先前22510/10918均已返回。另一个只读SFTP诊断79841因冗余且等待被本地关闭，不影响注册、Slurm或主控制器，不当科学失败。用户暂停/取消仅安全停本ledger/监测，不动其它项目；科学失败先日志保留attempt，新代码新版本。完整结果/独立accept/小结果归档/手册同步后一次最终通知并删除robust-gmp。


### 93.7 原GMP数值回放失败、独立两图诊断与新恢复版（2026-10-23）

旧HPC版robust_gmp_cluster_20261023_030000的technical3779988实际FAILED1:0、29分58秒。19目标测试无skip已通过；真实首训练成员2007_000032_000000在原atol=rtol=1e-6比较失败。原code/manifest/日志/failure保留，failure SHA14521eead28553c992058da7fb1d5d6b13778c469e6b3ac2bfe6268786bc9ff1。只取消本ledger尚未运行的extract3779989、merge3779990、train_predict3779991、evaluate3779992、audit3779993，均CANCELLED0秒。没有完整train/query receipt或全量训练、预测、GT结果，不属于Robust质量失败；technical_failure_handling_001.json有实际sacct和字节证据。

独立小输入诊断robust_gmp_input_probe_20261023_041000/3780319 COMPLETED0:0、84秒，冻结旧production求解器，只重放前两张原train图的7个原成员，未运行Robust聚合、训练、预测或GT。原1e-6 allclose有6/7通过，最大分量误差1.6093254089355469e-06、最大L2误差1.0153363291465212e-05，特征近同方向；保存每成员norm/area/support/pairwise cosine、两图canonical token数组SHA和独立向量NPZ。input_probe_001.json SHA4decc9ace64dbab9e016dacb94627c1679f43e7dd018cfe95aa2581425c9f2cb。这与跨平台FP32差异量级一致，但不能据此声称已证明误差原因或全token数组bitwise相同。

新恢复版robust_gmp_cluster_replay_recover_20261023_043000仅修改数值验收和证据记录：联合要求maxabs<=1e-5、L2<=5e-5、float64 cosine>=1-1e-8；保存旧1e-6 allclose/bitwise标志和实际误差。原三seed×两query的6份native center.2 HCR历史PNG仍必须逐像素精确。此变化在全量提取和VOC之前注册，独立审计也核保存数值界及6次精确回放；不改变IRLS/GMP数学solver、hardcenter成员、数据/伪标签/旧模型库、三seed训练与最终推理。IRLS和完整A/B核心字节保持，408旧snapshot文件逐SHA克隆、7新文件补丁逐SHA核后冻结，新模型/缓存/预测均独立路径。第一次恢复草案042000因手抄诊断SHA多一个9在注册前assert停止，无manifest/JobID/科学计算；目录/code/log/guard保留。最终版通过读取真实本地文件SHA生成校验值，不覆盖历史。

当前manifest SHAebf0fc74b2a5195a438b650448b6044f554b8c191154a93944d3a8f6f8bf9cb1，binding8705af4ae2688087b3d2874c1c793ed1a08ec61f56ffcae9b47f7de228458a09；注册rc0，GPFS真实余229286051840字节。实际jobs.json全afterok一次成功：technical3780328、extract3780329 array0-3%4、merge3780330、train_predict3780331 array0-2%3、evaluate3780332、independent audit3780333。未发生QOS后备；eval仍分配1V100配额但CUDA隐藏CPU计算。已排定audit，绝不另交或手动verify。最后技术3780328 RUNNING42秒/c2v100m2-2，其余依赖PENDING；本版19测试/真实probe尚未完成，未正式验收startup，更无mIoU结论。local launch_inputs_verified只证明输入和提交，不是独立完成验收。

当前控制仍用已验证HPC持久session98948且最近请求CONTROL_DONE；只读状态脚本改为plan/cluster_status_remote_002.py。work_state及cluster_replay_recovery_plan_002.json为当前路由。个人等待queue继续停止，禁止重启机制/旧full；固定delta1.5/T5、rho.1、34880含10569BG、8466/2116 split、50x100原MLP训练、最终center.2/训练内部center.4不变。首次四extract真实进展/进程/GPU UUID/新FP32artifact与真实技术probe通过后才一次正式通知；完整12x1449预测SHA先提交后GT，最终accept/小结果归档/手册同步后停，不据VOC追加选择。


### 93.8 HPC正式四路提取启动验收（2026-10-23）

当前版本robust_gmp_cluster_replay_recover_20261023_043000未改manifest/binding或冻结源码。technical3780328现COMPLETED0:0、Slurm3分35秒（阶段execution wall87.0189769025892秒，GPU峰值101.0087890625MiB）。真实19目标测试无skip、2train/2query和三seed×两图native center.2 HCR历史PNG全部逐像素精确。technical_complete SHAe039cd8055bfc36f16391d1dc0a05a5fbac34ef992aee096acd81d30a7c623a5。7个train成员最大abs1.6093254089355469e-6、最大L2=1.0153363286405137e-5、最小float64cos=.9999999999484566，均满足本版联合界；旧1e-6 allclose仍6/7、bitwise0/7，不能称跨平台向量逐位复现。

四路extract均已产生真实非pilot新FP32区域特征及内容SHA，唯一外层validate_startup_001.py已运行并passed/formal_start_reported=true。它核真实target proof、精确scontrol ArrayID_TaskID→actualJobID、各自Slurm overlap读取/proc命令/PID/manifest/phase/shard和真实nvidia-smi PID→GPU UUID；四个UUID不同，每worker一张V100。首两technical cache仅复用，未拿它们冒充正式新图。核验时四进程均真实RUNNING，没有使用事后存活假设。

- shard0: exact 3780329_0 → actual JobID3780378，PID2972081，GPU-5bb108d5-111d-4533-7680-35b2cf6e3fb2，首新图2007_000121、FP32[2, 1280]，核验时progress=408。
- shard1: exact 3780329_1 → actual JobID3780379，PID3998814，GPU-4ff49fbf-7a3a-39ff-e772-799e9c692a79，首新图2007_000170、FP32[7, 1280]，核验时progress=253。
- shard2: exact 3780329_2 → actual JobID3780380，PID3998813，GPU-beff1853-430d-1aa3-0ae6-0844f6963fed，首新图2007_000063、FP32[3, 1280]，核验时progress=242。
- shard3: exact 3780329_3 → actual JobID3780329，PID1262450，GPU-cb0e3b7c-fa64-4ed4-9a1c-fcc45f893c43，首新图2007_000068、FP32[2, 1280]，核验时progress=251。

各首receipt绑定本版，sample IDs/classes/order对原成员精确；own raw NPZ均FP32[区域数,1280]、有限、L2 norm合格，逐文件SHA核验，原token引用及canonical arraySHA保存。未重做GMP/检索/GT。startup_validation_001.json SHA324d69cd2d876aa86b95122108d16b44ef04bcf09135eac618ce8f5659109e39，大小18124，已本地下载并用远端startup_download_catalog_001逐SHA核6个小文件（startup、technical、target tests、execution、两train_probe）。本地startup_download_verification_001.json留证明。

这仅是正式启动验收，不是34880全量提取完成、MLP训练完成或segmentation改善/独立完成验收。固定delta1.5/T5与原参数继续；merge3780330、train_predict3780331、evaluate3780332、independent audit3780333按现有afterok链推进，不重复提交audit或手动verify。第一次正式启动通知本次发出，后续正常推进保持静默；完整预测SHA后本批GT，最后独立accept/小结果分析归档再通知最终结果并删除robust-gmp，不追加实验。


### 93.9 科学计算完成，原独立审计超时（2026-10-23，未独立验收）

本回合通过原持久控制器接收此前只读请求，真实sacct显示Robust恢复版全部10个科学child均COMPLETED0:0；三seed train_predict为57:00/1:04:58/1:04:58，evaluate13:28。原独立audit3780333为TIMEOUT、2:00:03，日志明确time limit。summary已生成，但尚无独立accept，不能称完整验收。保留原审计code/guard/log；只允许确认旧审计终止后另版外层审计恢复，不重跑训练/预测/GT/统计/GMP。原robust-gmp监测已记录此异常；其控制与新HPUD分开，不消费HPUD会话。

### 93.10 独立审计超时后的外层恢复与Slurm过期ID修复（2026-10-23）

科学10 child均已COMPLETED0:0，第93.9节原audit3780333为TIMEOUT/2:00:03，batch CANCELLED0:15、extern COMPLETED0:0，原allocation已终态，无任何科学任务要恢复/取消。原科学snapshot/manifest/参数/预测/CM/GT提交和001审计code/log/guard全部不改，不把已生成summary当独立accept。

新外层cluster_audit_002.py以12h/6CPU24G/一V100资源配额、CUDA隐藏CPU审计，增加逐文件SHA进度以及单进程内按dev/ino/size/mtime/ctime的digest去重；新进程不信任旧checkpoint，不省略实际字节校验。唯一审计002 Job3786930真实FAILED1:0/48秒：其在独立guard/字节审计前调用squeue -j查询已过期的旧3780333，Slurm报Invalid job id specified。原002脚本、audit_job_002/authorization和日志保留，本地log SHA52b01b92644cc332d6a9e96b3792e785c8ee172be821bce3534440a587d9a49b；没有重做科学计算或将其视为质量失败。

新的003外层只将队列查询改为完整本用户squeue -h -r -u zhengyuan再精确匹配旧JobID；真实只读probe成功，sacct确认001 TIMEOUT及002 FAILED且batch/extern全部终止。cluster_audit_003.py与submit_audit_recovery_003.py/audit_recovery_expected_003新文件逐SHA校验，实际audit_job_003.json唯一Job3786984提交成功；12h/6CPU24G/一V100配额且CUDA隐藏。注册manifest仍ebf0fc74b2a5195a438b650448b6044f554b8c191154a93944d3a8f6f8bf9cb1、binding8705af4ae2688087b3d2874c1c793ed1a08ec61f56ffcae9b47f7de228458a09，不改任何科学code/输入/成员/模型/库/统计。原verify_full_completion_001.py SHA cf9dd9569712380e6ca9059e4b16b1393ab80afcbce36f589e32b7c0892c917d，原cluster_audit_001.py SHA0a1559c42ebe417e02704e02136b3073b67480b35800f2b6ea382fd10b558fee均在新audit核字节。

003只调整独立guard/accept输出版本、原既定B历史JSON布局和字节digest仪器化，原全部科学断言保留。各cache receipt/NPZ、模型/库文件字节及保存CM/CSV/IDs核查继续；不解码GT/旧token/model tensor，不重GMP/检索/训练/推理/bootstrap。checkpoint=byte_digests_003.jsonl和audit_progress_003.json，存在Job/guard仅查状态；不能重复submit或手动verify。最终以completion_acceptance_003.json status passed且该独立job COMPLETED0:0为准。随后小结果逐SHA下载、独立分析、原始机制与内部伪标签/VOC开发边界、权重/成本口径、归档与手册最终sync；正式完成通知后删除robust-gmp。HPUD已独立完成归档及删除hpud-v1，不重启、不混合两批结果。


### 93.11 Robust固定三seed完整独立验收、负结果与最终归档（2026-10-23）

有效full robust_gmp_cluster_replay_recover_20261023_043000的10科学child全COMPLETED0:0；最终独立audit003 Job3786984同为COMPLETED0:0，耗时1:18:48，于2026-10-23 18:57:07北京时间（凭据10:57:07.767153Z）通过。completion_acceptance_003.json SHAe85c50dc1283366ebb8be0c7f0add49d8a08ac816c32d3b72a9981ccd6f74444，binding8705af4ae2688087b3d2874c1c793ed1a08ec61f56ffcae9b47f7de228458a09，manifest仍ebf0fc74b2a5195a438b650448b6044f554b8c191154a93944d3a8f6f8bf9cb1。34880成员含10569BG、三seed匹配新MLP、17388新PNG/12逻辑组、12全局CM及17388逐图CM、21类IoU/FPFNwrongFG/chair↔sofa通过保存核验；原native HCR三seed全局与逐图历史CM精确。fresh bytes累计42133唯一文件/1440917854字节，原verifier自身seen30098是另一分母。

正式固定delta1.5/T5结果（mean±sample SD，百分数，差为百分点）：clean GMP85.699896±.141559，Robust85.533168±.093836，差-.166728（paired delta SD.130864），seed42/123/2026差-.315506/-.115240/-.069439，0/3正；HCR GMP86.505997±.274697，Robust86.398415±.117504，差-.107582（paired delta SD.217220），逐seed-.345174/-.058410/+.080836，1/3正。不得只挑seed2026。clean/HCR chair均值IoU分别-1.990289/-1.909936pp、sofa-1.388195/-1.299942pp；HCR虽然BGFP平均-70698、FN-1570.333，wrongFG却+73965.667，sofa→chair+77885（像素数，非mIoU可加贡献）。瓶子等改善不抵消平均下降。

内部fit-only gallery27939/query6941区域来自原8466fit/2116query图、无同图泄漏，伪标签top1三seed均下降，mean98.703357%→98.549681%；top5 mean同99.990395%，mean margin .642976→.630717。其为内部伪标签诊断，原epoch-selection center.4和正式center.2边界保持，不能当独立GT证据或用VOC重选模型。best epochs42:38/123:27/2026:28；原50×100 CE+.5SupCon配方不改。

RTX500区域机制的污染稳定性改善与本HPC下游负结果明确分开。预注册delta1.5/T5的10%/20%污染drift分别.017289879→.003363534、.034809248→.010030748；x4 duplication实际.002193673→.003376580，仍在原+.005容许界。零追加clean权重mean .444727、fraction<.5 macro .638593，均500区域等权而非全量token分布；低weight不证明背景，adaptive MAD/ridge/clamp IRLS不构成固定Huber目标收敛证明。RTX同步仪器化runtime各4500call，按N-bin count加权GMP2.001848ms、Robust16.904912ms（8.4446倍），非纯kernel或V100全量逐region时间。

有效full实际科学墙钟最早Slurm2026-10-23 04:07:21至11:44:50，7:37:29；含最后独立审核至18:57:07为14:49:46，四extract/三seed并行不累加。四extract allocation4:16:02–4:26:03；merge1:49:25；三个train_predict（包括训练/export/检索/两arm预测）57:00/1:04:58/1:04:58；evaluate13:28；audit0031:18:48。execution内时间另存resource_accounting_004，不能把allocation等同纯求解或纯训练。RTX机制小归档缺完整起止墙钟，记未测不以版本名推算。原失败版/探针各自历史保留，不混进有效科学墙钟。

torch peak口径：RTX机制127.716797MiB；HPC各extract105.214844–105.850098MiB、train_predict各45.507324MiB、technical101.008789MiB，是各进程torch allocated峰值，不是驱动/整卡硬件峰值；CPU MaxRSS为Slurm采样task RSS，不能冒称GPU。evaluate/audit各有一V100资源配额、CUDA隐藏，merge为debug CPU。跨平台原GMP probe新注册联合界通过，但旧1e-6仍6/7、0/7bitwise；元数据原token SHA链接不代表全实际数组bitwise。同源成员/伪标签/原adapter/各24库/9核心字节证据保留，不能声称仅aggregation变化且全部逐位相同。

延迟与失败全部保留：原个人num_layers合成接口错误、个人等待queue迁移退出、旧HPC030000真实1e-6回放失败/未启动依赖取消、input probe3780319、042000注册前抄SHA错误见93.4/93.7；本有效科学没有重跑。audit0013780333在2:00:03超时；audit0023786930因过期squeue JobID在48秒失败；audit003只修外层查询/校验仪器化并已通过。18:57完成后状态未及时回传，本次21:21才刷新，是控制/收尾跟进延迟，不能继续说科学任务在运行。

70小结果共16210337字节已按audit/small_archive_catalog_004.json逐SHA下载至experiments/hpc_results/本版/verified_20261023；catalog SHAc706f75b2f2250295de09bc22c8c3c62e8488b6dd01d5bbb0055264b3766eaab，download_verification_004.json通过。包含summary/metrics、12逐图NPZ、6内部检索、3training_reports、phase_execution与必要proof；不下载模型/token/17388原PNG。独立analysis_report.md SHAcbceac1082d8d366f81d502d5a824c3337df1b938399a1556451ca614b717c87；local_archive_manifest_005.json SHA130684e2651946dc30c0ca29c546ed0b9fbe120fa9fbfbb348347f733aeedb38记录小文件及报告清单。analysis_numbers_005只对已保存小指标作报告级差值/mean/SD，不重GT/CM/统计推断或科学compute。报告、下载proof、local archive proof回run以archive_remote_verification_005逐SHA核，后者与manual_sync_005小proof下载保存。

结论：本固定配置的机制鲁棒性未转为内部检索或VOC开发平均精度提升，保留原GMP，不替换主线，不按VOC加扫参数。独立验收及小归档完成，本固定批结束；一次最终结果通知后删除robust-gmp，仅停止本批监测，不影响独立PAF或其它实验，不恢复个人queue，不部署/GitHub/ZIP。


## 94. H Proposal Utility Decomposition v1（HPUD，2026-10-23附件授权并实际提交）

用户附件原文保存于experiments/plans/h_proposal_utility_v1/request.md，并明确“用集群跑”。本批是GT-after诊断，不是分割方法/新精度候选，不恢复T4/RLI/PFBS/SMCF或其它停止分支，不修改E/H或旧代码/旧预测。独立版本h_proposal_utility_20261023_153000，run=experiments/h_proposal_utility_v1/版本，冻结快照=experiments/hpc_launch_audits/版本/code，日志=logs/h_proposal_utility_v1/版本。

固定三seed42/123/2026、全部476个实际写入H的accepted extra实例（154/163/159），163个支持图片；完整1449 image IDs参与基线回归与共享image bootstrap。源SMC manifest SHA429b6851462b6e5ae0dd2908076de90ad36ec1fed6eb3be9c3c94dd47bd3b5c1、独立accept SHAa2baf41d085901be7086cb65a968ef17bfe60fdeb85d27cbe2735f357f64845c均已在HPC真实核；E是第86节accepted single-view loose基线，不是strict R3 A。预检queue为空、quota余227725934592字节。

新模块tools/h_proposal_utility_core.py、tools/analyze_h_proposal_utility.py只用NumPy/Pillow及标准库，不导入Torch/model/production inference，不解码原token或raw/MLP特征。新增SAM3/训练/检索/库/正式prediction均0。先锁定候选metadata、E/H PNG、源prediction/owner与mask池的内容SHA；只对H字段解码，原共享NPZ其它R/feature字段不参与分析。owner实际写入像素求和须逐候选及逐seed与旧TP/BGFP/wrongFG/ignore整数精确，且1449张E/H逐图矩阵精确，否则OWNER_ACCOUNTING_MISMATCH停止。整个GT pass结束、全量账目通过以后才正式计算utility；在RAM保留163张active GT供反事实，不二次读GT，GT-last图示仅复用<=280x210预览。

单proposal DeltaRemove按全局H mIoU的实际owner-only BG移除，数值零仅abs(delta)<=1e-12 pp。逐candidate实际构造counterfactual并核整数CM transfer。两个联合Oracle（all harmful、TP=0且BGFP>0）均用原owner在RAM实际构造并重新算CM，不refill、不重新排序、不raster、不相加独立Delta；不导出新的正式prediction。这是指定GT Oracle潜力诊断，不是可实现方法，也不是已证明的全局最优拒绝子集。

proposal_score/retrieval_score/margin预注册higher=beneficial；其它连续metadata只给原方向Spearman，类别/view/prompt ID不作有序证据，不翻号/拟合threshold或classifier。固定PCG64 seed42，完整1449图2000共享multinomial draws，每次保留图内所有candidate；full-data utility标签固定，区间仅描述固定H/模型及这些标签下的VOC开发图像不确定性，不是seed方差或独立test。每seed独立报告，macro不把三seed候选冒充独立重复。数值checkpoint先于图示；4组各请求20例，不足如实，GT最后栏。完整细则/预注册描述性三路线decision见plan.md/protocol.json。

本地14项科学数值测试无skip通过，另以合成1449图CM/18实例走完2000bootstrap与54张实际可用合成面板尾部；这些不是集群实际476实例验收或质量结果。新快照8 code records、7上传文件逐SHA核后冻结，绝不运行中原地改code/参数。四role均CPU计算、CUDA隐藏，为避免debug两小时上限及GPFS校验延迟，每role申请v100g32一张GPU资源配额/6CPU24G/12h；不能称零GPU分配，但无GPU计算。

实际新manifest SHA376535998d18c51237e3706e22b63ab63b6bbfc732c32ac666fc1a983484d248，binding=751a940288dae78c9bcfe4cb23502380db761417813005590560bdaf8b4c2bfb。唯一afterok链已成功提交：analyze 3785373, audit 3785374, inventory 3785371, technical 3785372。全部returncode0，无重复提交或QOS后备。manifest/ledger/配额/上传证明等7小文件已下载至plan/launch_receipts/版本，local_launch_verification_001核manifest SHA、binding、jobs及upload SHA；它仅是launch_inputs_verified，不是科学完成或独立accept。当前等待真实inventory及technical，未报告HPUD结果。

已排定独立audit3785374，不能再提交audit或手动verify。只读CPU核3科学child成功、源/产物字节、保存CM/owner守恒/CSV/2000draw与replicate区间、Oracle非加性和GT-last图示，不读GT/token/model或重做bootstrap/推理。独立passed后小结果与必要图逐SHA下载、真实view_image抽查、独立九问分析/归档proof回传并远端核SHA，补手册最终结果/角色与整体墙钟（不累加并行）/归档SHA。只建议PROPOSAL_ACCEPTANCE_PROMISING、PROPOSAL_COMPOSITION_PROMISING或NEITHER_CLEAR及Recommended Next Step，绝不执行后续方法。固定批收尾一次结果通知后删除HPUD监测。


### 94.1 HPUD全量完成、独立验收与小结果归档（2026-10-23）

第94节提交状态已成为历史。inventory3785371、technical3785372、analyze3785373及独立audit3785374现均COMPLETED0:0。集群14目标测试无skip、首两真实productive图owner/mask/metadata重复检查通过；technical_report SHA3c3316ed8094014f566a7e7226cd9dead7c893944025175d2b17138bfe376382。完整476实例/163支持图片、1449基线图及2000共享image bootstrap和80例+6统计图全部完成。source_history_linkage_001另外逐SHA对齐旧已验收SMC本地归档中的all_predictions_committed/per-imageCM/metrics/两owner质量CSV，源commit仍2007cf45fd137399aa83e7db8b9054dd60e9c515c97bb64523ffbcb67c145ac4。无SAM3/检索/训练/新库/新正式prediction，无H或mainline修改。

GT仅在本批candidate_metadata_no_gt及owner/mask原内容SHA commit后读取，1449张每seed E/H矩阵和全部candidate/seed owner整数与旧历史精确；本批GT只完整读取一次，active163图GT暂存RAM支持后验actual counterfactual，图示仅小preview缓存、GT最后栏。owner_accounting passed；单DeltaRemove不相加，两个joint Oracle均实际构造原owner-only BG移除raster并算CM，无refill/reorder/reraster、无production PNG。

|seed|total|beneficial|harmful|neutral|harmful BGFP share|harmful TP share|harmful zero-TP fraction|
|---|---:|---:|---:|---:|---:|---:|---:|
|42|154|48|86|20|99.701694%|0.982535%|88.372093%|
|123|163|52|86|25|99.739350%|0.595935%|88.372093%|
|2026|159|53|82|24|99.607279%|0.585473%|87.804878%|

有害candidate并非极少（约52–56%），但GT排序top20% harmful（约全部candidate的11%）承担全部H-extra BGFP的82.04/84.41/85.36%，TP份额仅0.0583/0.0354/0.0348%。harmful约88% TP=0，约12%同时有TP和BGFP；harmful median valid precision均0、precision>.5数量均0，各seed整体拒绝harmful仅损失1263 correct owner pixels。不能把zero-TP视为普适语义垃圾或把GT排序当可部署top-k规则。

|seed|H mIoU %|GT_ORACLE_REJECT_HARMFUL %|gain pp|GT_ORACLE_ZERO_TP_REJECT %|gain pp|
|---|---:|---:|---:|---:|---:|
|42|87.182390018|87.351144454|0.168754436|87.325887883|0.143497865|
|123|86.848879400|87.061261776|0.212382376|87.037123366|0.188243967|
|2026|86.803830185|86.967824124|0.163993939|86.942878265|0.139048081|

原H mean=86.945033201%，指定joint Oracle mean=87.126743451%，mean gain=0.181710250 pp，原共享图片95% macro CI=[0.11460011886831485, 0.2699655382854042]；zero-TP mean gain=0.156929971 pp，CI=[0.09503157693088286, 0.2379572246476382]，均2000有限replicate。是GT ORACLE DIAGNOSTIC，不是新方法精度/全局最优拒绝子集/独立因果贡献，不更新R3最高记录。

proposal score AUROC42/123/2026=0.758236/0.768784/0.760469，retrieval score=0.752422/0.718470/0.721353；margin=0.591570/0.567979/0.564887。三者点方向一致，但margin三seed95%图片区间都跨.5，仅弱信号；proposal score区间下界分别.661243/.679983/.666364，2000有限rep。自然higher=beneficial、不GT翻号；其它连续metadata只原方向Spearman、nominal类/view不作有序支持。AUPRC须对照beneficial prevalence。Harmful的center/H maxIoU均值均低于beneficial，overlap count差异小且有混杂；不支持简单“大重叠冲突集中于harmful”解释。TP主要由sofa贡献，整体AUROC不证明每类内部普适，不做类别/view/size gate。

正式描述性结论=PROPOSAL_ACCEPTANCE_PROMISING；Recommended Next Step: Proposal Acceptance Feasibility v1（仅建议、需另行授权，未执行）。VOC是已看开发数据；shared1449-image draws保留图内全candidate、固定full-data utility labels，模型seed独立报告后macro，不混池、不当training-seed方差或独立test。

Slurm角色耗时inventory937s/technical119s/analyze816s/audit743s。科学整体最早start至最晚scientific end=1872s（31:12），含独立审计至最晚Slurm end=2615s（43:35）；均取时间范围，不累加并行，准备/下载/人工解读归档另记proof时间戳。资源每role6CPU24G/1V100配额、CUDA隐藏、GPU compute0；.batch MaxRSS原记录={"3785371.batch": "71840K", "3785372.batch": "2724K", "3785373.batch": "751436K", "3785374.batch": "58256K"}，为Slurm采样CPU驻留内存，不是GPU峰值。GPU memory peak UNMEASURED。

独立accept SHA8993ff2f122711d218a1dfeeb07cf742e260036dfb85ba26b961d3526b026a1e；summary SHA38a9407f6805650701c7a698f0335b343ade5e3f1bcf7c9e1749027785fabd49；analysis_report SHAe3eb77e037535267f284d21fa64762536c6c25e570a6e635d5de97a699770751；visual_QA SHA2ca3ac91dcc004c46f6bf879c1e2b2a9c7d92892346c68a717397c9e0dff2881。passed后123小结果/矩阵/CSV/replicate/必要图在experiments/hpc_results/h_proposal_utility_20261023_153000/verified_20261023逐SHA通过；真实view_image抽查harmful/beneficial/near-neutral/random各1张，6面板GT最后，tiny owner需结合caption，None precision为UNAVAILABLE。

独立analysis_report.md、download_verification.json及visual_QA_001.json已SCP回run/local_archive_001并逐远端SHA核验，archive proof SHA4d82975656a5f5ab7ff41264210e68bf3afd813cbbec57f9f31c50c7490e8435，remote_archive_verification_001 status remote_bytes_verified。未下载大token/model/原预测，未重GT/bootstrap/推理，不改正式指标。固定批到此停止，只保留建议不执行后续acceptance/composition；最终文档远端核SHA和一次完成通知后删除hpud-v1监测，不删除Robust监测。


## 95. Proposal Acceptance Feasibility v1（PAF，2026-10-23附件授权并实际提交）

用户附件原文保存在experiments/plans/proposal_acceptance_v1/request.md，并明确继续PAF-v1。本批第一次允许无GT proposal acceptance的新正式prediction与完整三seed VOC val评测，独立于已完成HPUD及Robust GMP；不修改E/H历史、source model/bank、原生产入口或已授权最高记录R3。版本proposal_acceptance_20261023_203000，run=experiments/proposal_acceptance_v1/版本，storage=storage/proposal_acceptance_v1/版本，audit=experiments/hpc_launch_audits/版本，冻结code=audit/code，logs=logs/proposal_acceptance_v1/版本/%A_%a.out。

源为HPUD已独立验收的同一476个productive H candidate-model实例：42/123/2026各154/163/159、163支持图片，完整1449原始image IDs；源HPUD manifest SHA376535998d18c51237e3706e22b63ab63b6bbfc732c32ac666fc1a983484d248、独立accept SHA8993ff2f122711d218a1dfeeb07cf742e260036dfb85ba26b961d3526b026a1e在HPC实际核过。只读其no-GT metadata与已签名SMC H mask/score/class/order/owner来源，不将HPUD harmful/TP标签作为selection输入，不重新激活历史skipped/NMS/loose gate未通过候选。

STRICT阈值来源已真实通过注册硬gate：独立验收SMC manifest SHA429b6851462b6e5ae0dd2908076de90ad36ec1fed6eb3be9c3c94dd47bd3b5c1在2026-10-21T04:52:52.690535+00:00创建，早于HPUD的2026-10-23T07:51:03.805633+00:00，绑定tools/oracle_guided_r.py SHA90dc80fe59ee7666be1b2217966fa24d624aa0cbe82000a6a5e9d9fdee974668。其RULES及strict_background_rescue实际q>=.4/r>=.95/m>=.1/novel>=.1，不是按HPUD GT新选cutoff。threshold_provenance.json/provenance_thresholds.json都保存来源与签名。历史H loose gate实际q.4/r.9/m.05/novel.1、FP32 .8r+.2q order、semanticNMS>.7及原hash资格保持，静态strict只恢复既有r/m门槛；saved FP32转Python float比较与历史一致，不加epsilon。

五个预注册arm：STRICT、PROP-MATCH、RETRIEVAL-MATCH、CONSENSUS-MATCH、RANDOM-MATCH。每image+seed先以STRICT静态q/r/m决定K，在runtime novel gate之前冻结。PROP按q降序/ID升序前K；RETRIEVAL按r降序/margin降序/ID升序；CONSENSUS为q/r各自ordinal rank（同score按ID升序），worst rank/mean rank升序，再margin降序/ID升序；RANDOM按完整SHA256 UTF8 image:decimal_seed:candidate_id:paf_random_match_v1升序一次固定布局。所有保留集合统一按历史H accepted顺序从E重新raster，只填live BG且novel>=.1，允许later surviving refill，但不激活历史非productive候选。类别/view/owner/GT均不是accept特征，不改E已有FG。

独立新模块tools/proposal_acceptance_core.py、tools/experimental_proposal_acceptance.py及tests/test_proposal_acceptance.py，使用原HPUD只读来源/IO helper的冻结字节；NumPy/Pillow/标准库，不导入模型/production推理，不解码旧token/raw/projected特征。新增SAM3/encoder/GMP/retrieval/训练/bank rebuild均0。已在本地通过18科学数值测试无skip，并用完全合成1449-image CM完成15比较×2000共享bootstrap与90实际合成六面板图示尾部；这些不是集群H回归、正式PAF预测或真实质量证据。

上传11文件（包含2外层文件及expected JSON）和2个新空init，12条expected upload逐SHA通过；冻结科学code_records为10文件。新manifest SHAc4f314bfe160910e852bdd54e98589302d6b3c2cb12072897500895421bc9500，binding7cd4b1dce134c5cdd9991930539e9d48d853625a62e9ecc2657b87ad741cf560，预检mmlsquota zhengyuan/data实际余227643670528字节，非df。唯一afterok链全部成功提交：inventory3788059 -> technical3788060 -> predict3788061 -> evaluate3788062 -> independent audit3788063。每role为v100g32一GPU资源配额/6CPU24G/12h、CUDA隐藏CPU计算，不能称零GPU分配；无新QOS后备。9份小注册/ledger/阈值来源/配额/上传证明已下载至plan/launch_receipts/版本，local_launch_verification_001逐SHA核manifest、binding、10 code records和实际JobID；它仅launch_inputs_verified，不是完成accept。

technical在集群18目标无skip通过后，先全部1449×3 all-accept H重新raster，逐像素及owner与历史精确才进入PAF正式predict；仅保存163支持图productive mask union的无损packed cache与原owner。新5×3×1449=21735 PNG、全部wide candidate decisions、runtime/owner/refill/count SHA先提交，之后才本批唯一GT pass及HPUD保存GT label事后join。H像素完全相等在相同GT下保证旧mIoU不变，数值阶段还逐图核E/H原矩阵及87.182390018/86.848879400/86.803830185；若不符H_RERASTER_REGRESSION_FAILED停止，绝不改边界绕过。

正式主比较STRICT-H：mean>0且>=2/3 seed正，不额外设CI显著gate。完整报告6方法全部15对比较（含预注册主要/次要以及consensus-vs单score）、逐seed/mean/sampleSD/方向，2000次PCG64 seed42完整1449 image共享multinomial draw与replicate，macro为各seed差值均值，不混池。其VOC开发条件图片区间不代替seed SD或独立test。排名支持采用同mean/方向描述规则同时胜H及RAND，不造新“明显”幅度cutoff，不按最高arm隐藏其它结果。A/B/C/D路由和增量flag规则在plan/protocol/core预先冻结。

GT后拒绝集precision排除neutral分母、beneficial拒绝率/旧GT top20 harmful覆盖、zero-TP覆盖、潜在owner TP/BGFP/WrongFG与正式pixel deltas分开；refill逐class及正确FG/BGFP/wrongFG/ignore保存。21类IoU变化/最大正负类/leave-largest-positive-class-out20类敏感性与q/r rank四象限均只作解释，不加类别/view gate。HPUD原Oracle只读已保存结果，不重建新subset；actual gain>0才给recovery ratio，并披露PAF refill流不同于owner删除Oracle。

GT composition和numeric checkpoint先于图示；请求20 STRICT helpful/20 harmful/20 PROP-over-STRICT/30 no-GT预先commit随机例，不足如实，6面板GT最后且复用一次GT pass的palette小preview，不为图重读GT。独立audit已Job3788063，已有Job/guard仅查状态不能重交/手动verify。其核4科学child C0、所有源/PNG/owner/trace/decision/count/random字节与保存21逻辑CM/CSV/interval/decision/GT-last，不读rawGT/token/model、不reraster/重生成bootstrap。controller proof不是independent acceptance。

独立passed且audit COMPLETED0:0后小summary/CSV/CM/replicate/proof和必要图逐SHA下载，真实view_image抽查4组，再写附件11问独立analysis/归档proof回传远端核SHA，更新本节最终指标/失败/acceptSHA/角色和整体墙钟（最早start到最晚end，不累加并行）/实际峰值口径/归档SHA。固定批无论升平降到此停止，不阈值/权重/rank/类别/view/seed扫描、不训练classifier、不重生成proposal或部署/GitHub/ZIP。后续calibration或mainline adoption只建议不执行。失败保留attempt，只取消本ledger未开始依赖，科学修复新冻结版本+小真实probe，纯报告/审计修复新外层复用数字；用户暂停/取消立即停本批及监测、不自动恢复，不动Robust等其它批。

### 95.1 PAF正式启动验收（2026-10-23 21:01北京时间快照）

inventory3788059实际COMPLETED0:0/2:15；technical3788060实际COMPLETED0:0/11:11。正式h_reraster_regression.json已完整1449图×三seed=4347条pixel与owner精确一致、18目标测试无skip，163无损productive mask cache；未读GT或HPUD utility labels，不是仅300图pilot。PAF predict3788061随后实际运行，已用精确Slurm/scontrol/lock及本job overlap ps核PID4108811、c2v100m2-3、冻结entry/manifest/--phase predict；D状态不当退出。最近progress2026-10-23T13:01:06.071411Z为150/1449图、2250/21735真实新PNG。五arm×三seed首15张新uint8 PNG均[366,500]、合法0..20并保持E已有前景；只检查已保存图，不重raster/GT。

新外层audit/startup_validation_001.json statuspassed，SHAac1f5cd5ac9b8e9ca786bbf28148ac3c5ec9126b1dcabf73b3b436a88133448d，11098字节；startup_download_catalog_001及H完整proof、两阶段complete共4小文件依catalog逐SHA下载核验，local_startup_verification_001保存在plan/launch_receipts/本版。此为正式启动验收，不是全批completion acceptance或新mIoU；formal_start_reported=true，首次启动已通知，正常推进不重复。eval3788062与独立audit3788063既有afterok链保持，不重复提交或手动verify。每CPU role仍占1V100资源配额，CUDA隐藏、GPU峰值UNMEASURED。科学code/signature/参数无改动。


### 95.2 PAF-v1完整独立验收、固定阴性结论与归档（2026-10-23）

本版proposal_acceptance_20261023_203000四科学role及独立audit全COMPLETED0:0：inventory3788059为2:15，technical3788060为11:11，predict3788061为19:28，evaluate3788062为11:02，audit3788063为14:16。独立completion_acceptance_001.json statuspassed，finished2026-10-23T13:37:02.672201Z（北京时间21:37:02），SHA02f940e4b51f9020e41ab853cd2fe8d74fc0fbe096dad57bb24d3c445da2b5be；binding7cd4b1dce134c5cdd9991930539e9d48d853625a62e9ecc2657b87ad741cf560，manifest仍c4f314bfe160910e852bdd54e98589302d6b3c2cb12072897500895421bc9500。完整H1449×3 pixel/owner及E/H历史CM精确，18test无skip，21735新PNG/15逻辑组、21保存CM组/30429逐图CM、2000共享bootstrap、476同预算无GT决策、35881源/结果字节SHA通过。原summary.md/completion_verification保留生成时pending措辞不覆盖，最终状态以单独accept及Slurm C0为准。

最终Decision=ACCEPTANCE_NOT_SUPPORTED；Recommended Next Step=Stop this proposal acceptance implementation; no threshold rescue or additional run。CONSENSUS_ADDS_VALUE=False、RETRIEVAL_ADDS_VALUE=False。严格执行预注册mean>0且>=2/3 seed正，不额外加/减CI gate、不挑最高arm重命名；固定本批到此停止，不救阈值或追加classifier/类别/view/seed/参数/新SAM3/训练/库，不修改主线/部署/GitHub/ZIP。

H三seed mIoU(42/123/2026)=87.182390018/86.848879400/86.803830185，mean±sampleSD=86.945033201±0.206787456%；STRICT=87.214074885/86.785592270/86.680780626，mean±SD=86.893482594±0.282543684%。STRICT−H逐seed差=[0.03168486702494988, -0.06328713005181896, -0.12304955852303578]pp，mean-0.051550607pp、paired sampleSD0.078032013，正seed1/3；共享图macro95%区间[-0.192619010,0.076972847]、finite2000。所有5arm与全部15对比较/逐seed方向/SD/CI均在小报告与原CSV保留，是固定模型VOC开发条件图区间，不是独立test或模型seed方差。

各方法mean mIoU%：H=86.945033201; STRICT=86.893482594; PROP-MATCH=86.892016654; RETRIEVAL-MATCH=86.893482594; CONSENSUS-MATCH=86.893482594; RANDOM-MATCH=86.893341202。STRICT/RETRIEVAL/CONSENSUS全局指标相同，不能仅凭mIoU相同断言全部候选ID或每像素相同。CONS−PROP虽有mean+.001465939pp/3正，但未同时胜H和RAND达ranking support，不能强称Feature Bank增量成立。PROP−RAND mean-.001324548pp/0正，RET/CONS−RAND mean+.000141391pp但仅1正，均不支持排名替换；不额外设“明显”幅度阈值。

STRICT各seed原/保留/拒绝/novelSkip/最终productive：[{"actual_productive_candidates": 81, "added_pixels": 133510, "input_H_extras": 154, "method": "STRICT", "runtime_novel_skip": 0, "seed": 42, "static_accepted": 81, "static_rejected": 73}, {"actual_productive_candidates": 84, "added_pixels": 161947, "input_H_extras": 163, "method": "STRICT", "runtime_novel_skip": 0, "seed": 123, "static_accepted": 84, "static_rejected": 79}, {"actual_productive_candidates": 79, "added_pixels": 162924, "input_H_extras": 159, "method": "STRICT", "runtime_novel_skip": 0, "seed": 2026, "static_accepted": 79, "static_rejected": 80}]。实际BGFP变化/正确FG损失（每seed各1449图，不是mIoU可加贡献）：[{"seed": 42, "BGFP_delta": -123490, "FN_delta": 79429, "WrongFG_delta": -4690, "correct_FG_lost": 74739, "correct_FG_gained": 0, "ignore_changed": 89318}, {"seed": 123, "BGFP_delta": -128530, "FN_delta": 174120, "WrongFG_delta": -10503, "correct_FG_lost": 163617, "correct_FG_gained": 0, "ignore_changed": 148176}, {"seed": 2026, "BGFP_delta": -71035, "FN_delta": 172251, "WrongFG_delta": -4729, "correct_FG_lost": 167522, "correct_FG_gained": 0, "ignore_changed": 130155}]。少BGFP也会丢有益FG，不能把静态拒绝都算收益；E辅助行lost/gained等占位0不用于本结论，五PAF arm为真实逐像素字段。

仅0<K<N图-seed组提供排名可区分空间，已保存预算CSV统计：[{"seed": 42, "active_groups": 138, "zero_K": 62, "full_K": 70, "partial_K_groups": 6, "partial_K_candidate_instances": 12}, {"seed": 123, "active_groups": 146, "zero_K": 67, "full_K": 74, "partial_K_groups": 5, "partial_K_candidate_instances": 10}, {"seed": 2026, "active_groups": 142, "zero_K": 67, "full_K": 67, "partial_K_groups": 8, "partial_K_candidate_instances": 16}]。结论受本476原productive H universe限制，不证明所有proposal acceptance或retrieval普遍无效。later refill保留原full mask/H顺序，只填当前BG且不动E已有FG，不激活旧skipped；STRICT原owner目的地汇总：[{"seed": 42, "historical_owner_pixels": 292237, "remains_background": 292237, "refilled_by_later": 0, "refill_correct_FG": 0, "refill_BGFP": 0, "refill_WrongFG": 0, "refill_Ignore": 0}, {"seed": 123, "historical_owner_pixels": 450826, "remains_background": 450826, "refilled_by_later": 0, "refill_correct_FG": 0, "refill_BGFP": 0, "refill_WrongFG": 0, "refill_Ignore": 0}, {"seed": 2026, "historical_owner_pixels": 373441, "remains_background": 373441, "refilled_by_later": 0, "refill_correct_FG": 0, "refill_BGFP": 0, "refill_WrongFG": 0, "refill_Ignore": 0}]。拒绝潜在TP/BGFP/Wrong与最终重填误差分开保存。HPUD Oracle只读原结果，macro实际gain为负故回收率null；seed42的正增益比率.187757仅该seed，不能宣称整批回收成功。类别21IoU/最大正负类/去最大正类20类敏感性及q/r四象限仅事后解释，不加class/view gate。

所有21735PNG/owner/decisions/runtime/refill先SHA commit，才一次完整1449 rawGT及HPUD标签事后join，GT_reads/authorization/preview proof保存。2000 PCG64seed42共享完整1449 image draw、各seed先加权CM后mIoU再macro，未重生成或重GT。图示实际组数：{"PROP_over_STRICT": {"availability_limited": true, "produced": 0, "requested": 20, "selection": "posthoc per-image mIoU difference extremes; not additive dataset contribution"}, "STRICT_harmful": {"availability_limited": false, "produced": 20, "requested": 20, "selection": "posthoc per-image mIoU difference extremes; not additive dataset contribution"}, "STRICT_helpful": {"availability_limited": false, "produced": 20, "requested": 20, "selection": "posthoc per-image mIoU difference extremes; not additive dataset contribution"}, "random": {"availability_limited": false, "produced": 30, "requested": 30, "selection": "no-GT precommitted SHA image:seed order"}}；PROP_over_STRICT无正例为0/20，按原shortage规则不虚构；最终共70张，已真实view_image抽查三个可用组各一张并核第4组不可用原因，完整QA归档。GT最后，随机30为无GT预commit，极值逐图IoU不可加为数据集贡献。

资源与墙钟：science {"first_start_slurm": "2026-10-23T20:38:50", "last_end_slurm": "2026-10-23T21:22:46", "wall_seconds": 2636.0, "clock": "Slurm display timezone, consistently used; not assumed UTC"}；含独立audit整体 {"first_start_slurm": "2026-10-23T20:38:50", "last_end_slurm": "2026-10-23T21:37:02", "wall_seconds": 3492.0, "clock": "Slurm display timezone, consistently used; not assumed UTC"}。从最早start到最晚end，不叠加角色；Slurm allocation与应用phase wall分别保留。各role6CPU24G12h且1V100资源配额/CUDA隐藏CPU计算，GPU compute0、GPU peak UNMEASURED，不声称零GPU分配；CPU MaxRSS仅Slurm采样task RSS。无新QOS后备、无科学失败或重跑。startup/manual同步控制等待没有改冻结科学code。

按audit/archive_catalog_001.json签名catalog下载115小结果（27025891字节，包括70图）及resource证明；catalog SHAf35cb0be20f841100f392c90aae5fb4cb30538e74a3838757629823698863fbb，download_verification_002逐文件SHA通过，目录experiments/hpc_results/本版/verified_20261023。包含summary/metrics/原21逐图CM/15pairCSV/2000draws+replicates/GT-after/refill/class/oracle/阈值/阶段proof，未下载大token/model或21735原预测PNG。独立analysis_report.md回答11问，SHA4c6bbf400d53b015c42ad99987070280ef439112978dc4ce34f6fe34054314a4；local_archive_manifest_003.json SHA7e222bd773eadb17417b122c89a5569e045b534be32d25eecefd1ebf4726969b。analysis/download/QA/local_archive proof回run以archive_remote_verification_003逐远端SHA核，manual_sync_003保留旧manual并原子同步。本节保留Robust93.11、HPUD94.1及PAF95.1，未改其它实验。

完整独立验收与小归档完成，一次最终Decision/Recommended Next Step通知后删除paf-v1监测，不追加研究、不自动校准、不更新production。本批结论只覆盖固定方法/已看VOC开发数据，无独立泛化主张。

## 96. Retrieval-Aligned GMP v1（RA-GMP / Bank-Feedback GMP，2026-10-24独立实现）

用户附件完整原文保存在`experiments/plans/retrieval_aligned_gmp_v1/request.md`，固定计划与协议为同目录`plan.md`、`protocol.json`，当前状态以`work_state_001.json`为准。版本ID预定为`retrieval_aligned_gmp_20261023_230000`（前一回合创建的版本名称，不代表该时刻已启动）。沿用已授权HPC环境，所有科学计算仅Slurm、每GPU一个worker；个人等待queue不恢复，已结束Robust/HPUD/PAF不重启、不混合结果。

独立实现为`tools/experimental_retrieval_aligned_gmp.py`、`retrieval_aligned_gmp_protocol.py`、`retrieval_aligned_gmp_runtime.py`、`retrieval_aligned_gmp_technical.py`、`retrieval_aligned_gmp_full.py`、`retrieval_aligned_gmp_audit.py`及两个test模块；不改生产入口/src/configs。原Robust的source adapter仅用于已签名token/原mask pool输入IO，不运行其聚合器、不用Robust raw或MLP。teacher为第93节已验收来源链中的原hard-center GMP、各seed原匹配MLP与原投影库，所有旧文件只读。

scorer固定1280→128/GELU→1，末层weight/bias零初始化，a=.5+sigmoid(h)；只输入token feature，无class/bank/prototype/坐标。p=a/sum(a)，mu与S用相同p；lambda=.1×trace(当前weighted S)，即原GMP规则应用于当前统计量，区别于旧Robust固定unweighted ridge。保持原hard-center/nearest fallback、FP32较小primal/dual Cholesky、残差<=1e-4，无精度/solver/pooling fallback。全1在实际平台必须bitwise回到原uniform_gmp；历史跨平台source回放另保存maxabs/L2/float64cos、旧1e-6与bitwise，不能混淆两种断言。技术阶段还要求两原VOC query×三seed native center.2 HCR及零初始化RA均像素精确，绝不读取GT。

冻结原8466fit/2116hold图，gallery仅fit的27939成员，内部query6941成员；不把原MLP已有内部选模的development split称独立test。fit反馈排除全部同图成员及类中心统计，不能可靠核image identity即停。先从实际HPC token重算普通GMP z0；原B0投影向量保持历史字节。固定positive5、production balanced K25/top5/search2/center.2分数最高的三个错误类、各negative5，少于5取实际可用数；只缓存ID/索引/签名，不复制bank向量，训练不动态TopK。训练分数按附件.6local+.4center、tau.07，beta1/gamma.05；AdamWlr1e-4/wd1e-4，三seed42/123/2026各30×100步，batch2=一hard(错或margin<.1)+一均匀non-hard，分两次backward平均loss且全N不截断。MLP0参数冻结，forward保留输入梯度；固定最后epoch30、不内部选epoch或beta搜索。

内部报告top1/macro/top5/margin/NN25、rescued/damaged/both、五margin箱、每region N及weight均值/std/min/max/分位/>1.1/<.9、descriptor coordinate MAE与cosine；rescued/damaged正类和固定baseline hard-negative分数变化。gate固定三seed mean top1正、至少2/3正、至少2/3 net_rescue正，且每seed margin>=.20箱非空且accuracy下降不超.1pp。失败则`complete_no_internal_gain / VOC NOT RUN`并停止。低weight不解释成背景，无效不能证明所有GMP已bank-optimal。

只有gate通过且独立internal audit passed/Slurm COMPLETED0:0，才允许重算各seed全部34880成员含10569BG、无重过滤或最远10%删除，按原CE+.5SupCon50×100/split/sampler/optimizer/center.4内部选模训练匹配1280-512-256 MLP，最终每seed一个RA库、center.2原retrieval/NMS/HCR。固定GMP/RA×三seed×clean/HCR×1449=17388 PNG全SHA提交后才一次VOC GT评测，保存12组逐图CM/21类IoU/错误结构/逐seed及mean/sampleSD/paired方向；VOC已看过，仅开发证据。通过gate后另做两个轻量内部消融：beta0同30×100与未训练all1精确回归，不增加消融VOC arms，不择优替换primary。禁止Round2/query-time feedback/class-specific参数/空间分支/新token/SAM3/伪标签/提议/额外搜索。

大产物分别置于`storage/retrieval_aligned_gmp/版本`、`storage/feedback/版本`、`storage/raw_banks/voc/版本`、`storage/mlp_banks/voc/版本`、`storage/predictions/voc/版本`；scorer在`weights/retrieval_aligned_gmp/版本`，匹配MLP在`weights/sam3_adapter/版本`，清单/指标在`experiments/版本`，冻结code/ledger在`experiments/hpc_launch_audits/版本`，日志在`logs/retrieval_aligned_gmp/版本`。预定technical→4 baseline→3feedback→3scorer→3internal→gate→独立internal audit；条件full/消融/eval/final audit在内部接受前不提交。已有guard/JobID只查不重交；失败保留attempt，科学修复用新版本。真实GPFS group/data余量>=5GiB，不df或预扫旧token。CPU角色占1V100配额但CUDA隐藏，不称零GPU资源；每GPU角色6CPU24G，真实峰值/角色耗时及最早start到最晚end整体墙钟分别报告。

当前仅本地12项协议/身份/排除/gate检查已通过；本地缺torch，不能称26项数值/梯度或真实V100探针通过。尚无本版真实JobID、训练、internal gate或VOC结果，后续提交/技术验收另记。固定批无论升平降结束后停止、独立验收小归档；不更新主线/部署/GitHub/ZIP。用户暂停取消立即安全停本批及监测，不自动恢复、不动其它实验。

### 96.1 首版技术预检提交及反馈同分记录修正（2026-10-24）

230000首版已实际注册，manifest SHA145d91e2609bcd1ff3b614c8884493b89a395c39345048f04aac133d9c1b88b6、binding3f25ce89c78c4021b17898de8a76aa8e7b45919396a4f0f9d9f1cefbba57aad4；408旧文件逐SHA克隆、11新代码/计划文件逐SHA核验后冻结，真实quota余227082264576字节。唯一technical Job3789823已提交，最近快照RUNNING2:40/c2v100m2-7；此为历史运行快照，终态/真实技术proof以ledger与后续检查为准。五份小manifest/实际jobs/注册/clone/expected凭据已下载至plan/launch_receipts/230000并核SHA/binding，local_launch_verification_001只launch_inputs_verified，不是完整accept或方法质量通过。

在首版生成任何feedback或训练scorer之前，检查发现其feedback缓存对class-score同分用了字典序，与原生产检索np.argsort(scores)[::-1]不完全一致。已保留首版code/manifest/JobID/log及本地attempts/230000，首版只作为技术预检，不继续internal/full。新版本retrieval_aligned_gmp_20261024_001500修正同分顺序，添加该边界回归，并对已保存的实际GMP全成员增加finite/nonzero拒绝检查；不改GMP/scorer/loss数学、数据、seed、超参数、检索/NMS/HCR。新版本目标27项测试；本地13项协议检查先行，真实数值/梯度和六份像素proof仍须新Slurm技术验证。必须先确认旧technical真实终态且无活allocation，才唯一提交新technical；不因GPFS/SSH等待重提或假称退出。当前工作路由以plan/work_state_001.json新version为准。

### 96.2 修正版不可变注册与技术路由（2026-10-24）

旧technical3789823现已实际COMPLETED0:0/8:31，无failure；日志26测试无skip通过，technical_report已生成。旧版仅作预检历史保留，不继续feedback/scorer/internal/full；不能用其26测试替代修正版27测试，也不当方法有效性或VOC结果。修正版本地13项协议/身份/排除/同分/gate检查全部通过，当前新版本是retrieval_aligned_gmp_20261024_001500。

修正版注册manifest SHA013c54e4d5f9511c9ec524db2dfc79b174d2ced48f2c17d02fae1c19297db065、binding9e1f539ade8c38133a9c8606beb19f0fdd7d095e7e62c0fa228324a6e49f018c；419条code记录=408旧文件克隆+11新代码/计划补丁，逐SHA核后冻结。注册真实GPFS余226981519360字节。manifest/registration/clone_patch/expected四小凭据已下载至plan/launch_receipts/本版并依实际SHA/binding及所有新补丁核验；local_registration_verification_001只registration_inputs_verified，不是completion acceptance。旧作业终态确认后才向prepare_remote.py technical发出一次修正版提交请求，实际JobID及请求是否CONTROL_DONE以work_state和audit/jobs.json为准，存在guard或JobID不能重提。

后续固定链与内部门槛未变；27无skip、真实gradient/回归及6份native/零初始化RA像素proof且新technical C0后才唯一提交internal链，完整独立internal accept+C0且gate通过才full。正常等待静默，ra-gmp监测仅跟本批并在完整独立接受归档/最终通知后删除。当前没有scorer训练、内部gain、bank重建或新VOC精度结论。

修正版唯一technical实际成功Job3789863，actual jobs.json已下载并核binding与manifestSHA，local_launch_verification_001为launch_inputs_verified而非completion；未排internal/full依赖。旧版technical/target_tests/complete/execution四小文件已下载，依旧technical_complete中的artifact SHA核报告与tests，旧26无skip/3真实梯度且MLP0不变/6native及零初始化RA像素精确均确实通过。旧版只作为数值技术历史，不冒充修正版27目标验收。

### 96.3 修正版技术通过、已完成JobID依赖回收后的外层恢复（2026-10-24）

修正版technical3789863已实际COMPLETED0:0、Slurm1:14；27目标测试无skip，真实2fit/2hold图的全1对当前平台uniform_gmp精确、三seed冻结MLP0的正梯度且参数不变、两原query×三seed的native和zero-init RA历史HCR逐像素一致全部通过，未读GT、未新SAM3。technical_report SHA2c3012a88daf61a97750135969c6a111eb7442fc30ffc50647069dc0ed2deeca；阶段wall28.623480619862676秒，torch allocated峰值326.32763671875MiB（进程分配峰值，不是驱动整卡峰值）。technical/target_tests/complete/execution小文件已下载，依complete签名SHA及binding/27无skip/3gradient/6pixel逐项核过，local_technical_verification_001为技术接受，不是完整completion acceptance。

第一次internal提交的baseline因`--dependency=afterok:3789863`被sbatch拒绝：returncode1、stdout空、无JobID。真实sacct仍COMPLETED0:0，而scontrol已明确Invalid job id、用户队列无该allocation，证明是已完成JobID被调度器回收后的依赖关联问题，不是科学失败或重复跑。原prepare_remote.py/submit_internal.guard/失败命令和ledger留存。新独立外层`submit_continuation_002.py` SHAac511639ed5b193e009cfd6e98c76d5340dda0c17dfa8e1d79e84e7586928c07先核完整技术proof实际字节及parent C0/无活allocation，再只省略已经履行的根afterok；所有新live依赖仍afterok。科学419文件、数据、数学、seed、资源参数均不变，不重新layout/register/上传科学code或重跑technical。

唯一实际内部ledger现为technical3789863、baseline3789972 array0-3%4、feedback3789973 array0-2%3、scorer3789974 array0-2%3、internal3789975 array0-2%3、gate3789976、独立audit_internal3789980，共15科学child及1独立audit。原无JobID的失败行保存在baseline.prior_no_job_attempts与jobs_before_internal_continuation_002.json；新journal/guard/outerSHA及各JobID分别落盘。已有JobID/guards仅查询，不再次submit或手动跑活audit。当前full/消融/VOC尚未提交；必须internal gate和独立accept+C0后才用同一外层002提交条件full，避免给已回收的audit根JobID再挂afterok，新live边仍保持依赖。

条件full前另以已保存整数rescued/damaged复核同一预注册gate，避免浮点相减把三seed零均值误判正收益：6941相同query分母下sum(net_rescue)>0等价mean top1正；高margin下降<=.1pp等价1000×(rescued−damaged)>=−N。两个纯外层边界检查已通过，不计入27科学测试，不重算检索/GT、不改参数、不将原FAIL提升为PASS；如与冻结浮点报告不符，保留两者并停止full，只准新外层数值报告/独立审计修复。内部top5含义为五个最高类别分数的命中，NN25为25邻居同伪类比例。

手册96–96.2已由manual_sync_001保留远端旧SHA30a0dded360d36484c914f8c4dcf7f41c001994483622e9dfd52a1ad74b8145d后原子同步23b63a3b4657e530493561738b9bbde7c9ab28bc6d6dda531689957758966875，proof已下载核验。后续本小节及startup新记录须新manual_sync_002备份当前remote再同步，不覆盖93.11/94.1/95.2。正常进展静默，最终以本版内部或完整独立accept+C0和小归档为准，不能把当前baseline准备或技术通过当scorer/MLP训练完成或新VOC精度。

外层恢复小凭据归档：launch_recovery_catalog_002 SHA783d721ebf4675b908d2c8d54f4385fb8b19d026ef76d275cecda774234d0b56，catalog及actual jobs/原无JobID拒绝ledger备份/新提交完成proof/guard/journal/首次状态snapshot共七小文件已本地下载，其中六个artifact逐远端catalog SHA核验，local_recovery_verification_002为launch_recovery_inputs_verified而非completion。首次snapshot四baseline子项实际RUNNING9:33、节点c2v100m2-8及c2v100m2-9，其余科学与独立audit依赖PENDING，无failure。日志每25图才输出一次，当前日志下界0不能当作零产物或退出；正式startup仍待真实首产物/Slurm/proc/GPU核验，不提前宣称scorer或精度完成。

### 96.4 RA-GMP四路基准特征准备正式启动验收（2026-10-24）

修正版startup_validation_001.json实际passed，SHA295e0d2e984d73088515caf8c5ae783ab33060caeee427e8ac8dacbcbba75cf3、14982字节；startup_download_catalog_001 SHA2f6ab85e064641a5889d7223e3219a16645bb46a8d33005c01aeb732e4f54a96。13份首产物/receipt/技术proof已逐catalog字节SHA及size在本地核验，local_startup_verification_001保存在plan/launch_receipts/本版/startup_001；四份新NPZ均为有限FP32[区域数,1280]、对应原成员ID/class/order。该proof仅正式启动验收，不是internal或full completion acceptance。

四baseline子项在核验快照均真实RUNNING，精确Slurm array/actual JobID、lock/PID/proc命令与四不同V100 UUID均通过：shard0实际3789977/PID3709835，首图2007_000032形状[5,1280]；shard1实际3789978/PID539290，首图2007_000039形状[2,1280]；shard2实际3789979/PID539291，首图2007_000063形状[3,1280]；shard3实际3789972/PID539289，首图2007_000068形状[2,1280]。完整GPU UUID与实际receipt/NPZ SHA见proof。最近启动检查elapsed21:01，日志进度450/400/375/375图，仅为当时下界1600图，不是34880成员完成或实时持续状态。

当前运行的是共享普通GMP基准特征准备；三seed反馈/scorer训练/内部检索/gate及独立internal audit已按原ledger排定，尚无内部gain或VOC精度。27技术测试及真实gradient/像素回归通过不代表方法有效；必须内部gate与独立accept及audit C0后才允许条件full/轻量内部消融/VOC，失败则VOC NOT RUN并固定批停止。未重SAM3、未读GT、未修改冻结419科学文件或任何科学参数。正式启动只通知一次，后续正常推进静默。

### 96.5 RA-GMP内部接受及预注册条件完整链提交（2026-10-24）

固定内部阶段15个科学child全部COMPLETED0:0，独立audit_internal3789980亦COMPLETED0:0/4:08；completion_acceptance_internal.json statuspassed、binding与原manifest一致，SHA8e57dd88c5ef0b97a46c633e96a4b74d1767f4915084266901bdc1be7ab1dde4。独立审计核10602份保存artifact、冻结teacher/split/同图排除、反馈27939/内部6941成员身份与保存统计、三scorer固定30×100及teacher不变；未解码GT/token/model或科学重算。这是内部阶段接受，不是完整VOC接受。

三seed42/123/2026内部top1增益分别0.07203572972194205/0.23051433511022124/0.2593286269989892 pp，net rescue为5/16/18，三seed均正，mean+0.18729289727705084 pp。每seed高margin>=.20组分别6335/6323/6315条，rescued=damaged=0，满足原保护门槛。独立外层从已保存整数核sum(net)=39>0、3/3正及1000×(rescued−damaged)>=−N，与原浮点gate一致；保存gate_arithmetic_review_001，不改变阈值或重检索。内部split复用了原MLP选模development数据和伪标签，不是独立test，更不是VOC mIoU收益。

内部accept/gate/summary/三metrics六小文件依新catalog逐SHA下载核验，local_verification保存在plan/launch_receipts/本版/internal_accept_001。internal_acceptance_catalog_001 SHA0b26fb65b99f609bff0942df45555385ca24284ee15d306adf80c506b35a5a36；条件full前实际mmlsquota组zhengyuan/data余225008582656字节，未df或扫描旧token。原419科学文件与所有数据/数学/seed/参数不变。

满足预注册门槛后仅一次调用local_control.submit(full)，经精确计数guard及submit_continuation_002再核internal audit C0/无活allocation，省略已履行且被回收的audit根依赖；新live边全部afterok。实际唯一新ledger：ablation3790369 array0-2%3、full3790370 array0-2%3、evaluate3790371依赖两array成功、independent audit_final3790372依赖evaluate成功。首快照六GPU子项RUNNING1:02，CPU评测及audit依赖PENDING；此仅Slurm状态，不当完整首产物验收、MLP完成或新VOC精度。已有JobID/guard禁止重复提交或手动verify活audit。

两轻消融仅内部beta0与untrained全1，不择优改primary或增VOC arms；full按既定34880成员重建匹配MLP/单库，GMP/RA×三seed×clean/HCR全部17388PNG先SHA commit才一次GT。最终仍须completion_acceptance_final passed及audit_final C0，再小文件分析归档；当前VOC NOT YET RUN，无最终质量结论。正常后续进展静默，固定批无论升平降结束后停止。

### 96.6 RA-GMP完整独立接受、结果与固定批归档（2026-10-24）

本有效版本22个科学child全部COMPLETED0:0，内部audit3789980为C0/4:08，最终独立audit3790372为C0/4:59（北京时间11:49:30–11:54:29）。`completion_acceptance_final.json` statuspassed、binding完全一致，SHA`4f7591b75973676173aab064866cb554b08990d6a2d81e12b5d0fc50c08bace8`；核28177份保存artifact、12逻辑组/17388张新PNG及保存CM/CSV/成员/模型库字节。审计不解码GT/token/model tensor，不重GMP/检索/训练/推理/统计。本节最终状态覆盖96–96.5中历史“等待/未运行”快照，不改旧记录或冻结419科学文件。

| seed | GMP clean mIoU% | RA clean mIoU% | clean Δ pp | GMP HCR mIoU% | RA HCR mIoU% | HCR Δ pp |
|---|---|---|---|---|---|---|
| 42 | 85.841489153 | 85.828568017 | -0.012921135 | 86.821717076 | 86.823621971 | +0.001904896 |
| 123 | 85.558370535 | 85.567924618 | +0.009554083 | 86.321692168 | 86.347164206 | +0.025472038 |
| 2026 | 85.699829448 | 85.727609215 | +0.027779766 | 86.374583191 | 86.581193555 | +0.206610364 |

HCR三seed mean±sampleSD：GMP86.505997478±0.274697130%，RA86.583993244±0.238241221%；配对meanΔ+0.077995766pp、sampleSD0.112005083pp，3/3正。clean：GMP85.699896379±0.141559321%，RA85.708033950±0.131419704%；配对meanΔ+0.008137571pp、sampleSD0.020387391pp，2/3正。HCR收益主要来自seed2026，seed42接近持平，不能只展示最高seed或将本批均值与历史seed42单值混比。VOC val已查看，仅为固定模型开发证据，不宣称独立test或显著泛化。

冻结teacher内部gate三seednet rescue5/16/18、mean top1+0.187292897pp仍成立，高margin>=.20组三seedrescued/damaged均0。宏类准确率和NN25一致性却三seed均小降，不能只报top1；五margin箱、固定hardnegative及权重统计完整保存在独立分析。beta0的net为5/16/16，untrained为0/0/0且descriptor对当前平台普通GMP bitwise，两个消融仅内部、不另增VOC arms。最终匹配新MLP内部检索与冻结teacher gate分开报告；低weight不能解释为背景，不宣称全局bank optimality。

HCR错误计数三seed均值变化：BG_FP -95189、前景→背景FN +6520.667、前景类间wrong_FG +13212.333；chair→sofa +13396、sofa→chair -25258.667。HCR类均值变化较大包括bottle+1.466566pp/car+1.056427pp，ship-0.668695pp/train-0.431237pp/pottedplant-0.410962pp。完整21类及预指定8类逐seed在analysis，不据此调类别、阈值或参数。

Slurm各角色elapsed：technical1:14；baseline四路1:20:47/1:31:05/1:31:05/1:31:05；feedback三路1:02:25/1:00:53/53:00；scorer1:14:21/1:14:17/1:13:09；internal各1:05:18；gate0:45；内部audit4:08；ablation三路各2:39:37；full4:04:10/4:04:18/4:04:18；evaluate3:04；最终audit4:59。有效批最早start北京时间2026-10-24 02:01:30至最晚end11:54:29，整体墙钟9:52:59；包含旧230000纯技术预检则01:45:39–11:54:29，为10:08:50，不累加并行。逐role execution wall/Slurm MaxRSS及primary/beta0独立训练wall见report。最大已记录Torch allocated峰值326.32763671875MiB，不是驱动整卡峰值；CPU角色仍占V100配额而CUDA隐藏。full execution混合raw重建/MLP/检索/预测，无独立匹配MLP或逐region GMP timer，不能伪造求解倍数/每region耗时。

旧230000同分顺序预检与无JobID根依赖拒绝历史保留，见96.1–96.3；两次技术版本/各自code/manifest/log不覆盖。原GMP查询/MLP/库保留历史字节，RA使用HPC实际FP32 token及匹配新MLP；6native/6zero-init像素精确与当前all1 bitwise回归不等于跨平台全数组逐位相同。旧1e-6及maxabs/L2/cos边界如实保留。

接受后118份小文件、32517779字节逐catalog SHA下载至`experiments/hpc_results/retrieval_aligned_gmp_20261024_001500/verified_20261024`，未下载大token/model/PNG树。catalog SHA`5c376f05b437e58ccad93a64af07b70f9bd03332361eb80227a4712bac6bea14`；`analysis_report.md` SHA`e30b725579821f9069bfb322126082a976a3144639d8d52881621c75b899b8da`；`analysis_tables_001.json` SHA`303b6a358a9b8a16f93cd98f66806d420319f77fa53b5b5c656f935f0b9eb58b`；`download_verification_001.json` SHA`e6cf8740d3e49fe84bd2f1eab9f0739f90511d2de88c3d9dfe43c124737fa755`；`local_archive_proof_001.json` SHA`1566820520a2026b7c0e5e53d54e0373dcbba2fa59be18bc77c41dc65f8d8571`。小analysis/proof回传同run，远端逐SHA归档及manual CAS凭据另存，不冒充新独立科学accept。

本固定v1完整接受并归档，无论升平降到此停止；无Round2、query-time重新聚合、搜索、新proposal/库配方或主线部署。最终通知后仅删除ra-gmp监测，独立Selection×Aggregation四组消融第97节/其监测不动。

## 97. Mask→Token Selection × Aggregation 四组因子消融（2026-10-24用户直接授权）

用户要求A Hard Binary/NN＋Mean、B Hard-Center＋Mean、C Hard Binary/NN＋GMP、D Hard-Center＋GMP。原表列错位，按明确的2×2设计执行。独立计划在`experiments/plans/selection_pooling_factorial_v1/{request.md,plan.md,protocol.json,work_state_001.json}`，版本`selection_pooling_factorial_20261024_142000`；新独立代码`tools/selection_pooling_factorial_{core,runtime,audit}.py`及对应test，不修改生产src/configs或其它运行/历史实验。

Hard Binary/NN固定为原图二值mask直接最近邻缩到72×72的floor索引约定，不经1008中间resize、不用soft/area权重。Hard-Center严格原stride14/center7原坐标取样。两种selection在非空mask空选时共用同一原mask质心nearest-token、row-major tie兜底；记录支持数、fallback、选择索引SHA与交并比。Mean为所选FP32 token算术均值后L2；GMP保持原rho.1、FP32/TF32off/较小Cholesky/残差<=1e-4，无solver替换。

从已独立接受的Robust043000来源链只复用原普通GMP的输入IO：HPC实际1008/72×72×1280 FP32 token、原伪标签与34880成员含10569BG、成员ID/类别/顺序、8466fit/2116hold、原clean/raw候选；不用Robust或RA的特征/MLP/scorer，不重SAM3/标签/提议/过滤。四组全重新计算raw bank，包括D，并各自从头训练42/123/2026匹配1280-512-256两层MLP，共12模型，原CE+.5SupCon50×100和center.4内部选epoch不改。D本批重新训练结果作为因子对照，不能直接拿历史单seed86.821717%填入。历史跨平台GMP保留联合maxabs1e-5/L2 5e-5/cos>=1-1e-8界及旧1e-6/bitwise，当前D计算须精确等于当前生产GMP。

最终固定单库检索center.2/K25/top5/search2、原NMS/score混合及HCR；R3和其它分支关闭。主表为HCR、clean为配对辅助，4×3×2×1449=34776新PNG全SHA提交后才本批一次GT。保存24组逐图CM/21IoU/FPFNwrongFG/chair↔sofa、三seed逐值/mean/sampleSD，以及B-A、D-C、C-A、D-B、interaction(D-B)-(C-A)，不据VOC调参或追加arm。VOC只属已查看开发数据。

预注册链technical17无skip＋真实2train/2query四组探针＋6份历史native HCR像素回归→4extract→CPUmerge→12train_predict最大并行4→CPUevaluate→独立audit。每GPU一个worker，CPU角色仍分配1V100配额但CUDA隐藏；纯CPU计算不能说零GPU资源。现有guard/JobID只查不重交；新科学错误新version，保留attempt，只安全停本ledger未启动依赖。最终passed且audit C0后小结果逐SHA下载、分析归档与手册收尾；固定批无论升平降结束，不部署/发布/ZIP，不动RA等其它实验。当前仅8项本地几何/协议测试通过及语法检查；真实17测试、JobID、启动与精度以work_state及后续实际凭据为准。

首个140000准备版本在注册复制计划时因旧快照已有`plans/request.md`而被“不覆盖”断言拦下；未写manifest、未生成JobID、无科学计算。其code/register.guard/上传记录及`registration_failure_001.json`完整保留，本地attempts也保留当时源码。新142000版本只将新计划放到`code/plans/factorial_v1`独立命名空间，四组数学、输入与参数不变；不续写140000或覆盖旧plan。

### 97.1 独立注册与实际Slurm链（2026-10-24）

142000实际manifest SHA`557744db04d93f5824986faa1a85c129f248b7fcd389cebcd554312fdc2e086f`，binding`6472bcf8b239687603eb7dbc10705117cf198f80cbd3c26b7bf375eec85da344`；408旧代码逐SHA克隆＋8新代码/测试/命名空间/计划文件逐SHA核后冻结，共416记录，未改生产。真实GPFS组zhengyuan/data注册余223108620288字节。manifest、registration、clone、expected、actual jobs、launch complete/guard及catalog已下载至plan/launch_receipts/本版，7份artifact逐catalog SHA核验，local_launch_verification_001为launch_inputs_verified，不是completion acceptance。

唯一实际afterok链：technical3792900→extract3792901 array0-3%4→merge3792902→train_predict3792903 array0-11%4→evaluate3792904→independent audit3792905。已有JobID/guards不得重提或手动verify活审核。首快照technical RUNNING00:00:35/c2v100m2-2，其余依赖等待；后续真实17测试/2train2query/6像素回放及四路新非pilot FP32 artifact启动核验以proof为准，不把JobID或本地8测试当作全量启动/新精度。专用pinned控制session87686，不消费RA17135；所有请求CONTROL_DONE后才下一请求。

### 97.2 四组四路特征提取正式启动验收（2026-10-24）

technical3792900已COMPLETED0:0、Slurm2:09；17目标测试无skip通过，真实2train/2query四组descriptor与7个原源成员数值界通过，当前D对生产GMP bitwise精确，3seed×2query原native HCR历史PNG共6份逐像素精确。阶段execution wall39.23386683315039秒、Torch allocated峰值106.31787109375MiB；历史源7成员仍非bitwise，旧1e-6为6/7通过，最大maxabs1.6167759895324707e-6、L2=1.0153760935619246e-5，不能声称全平台逐位等价。

外层startup_validation_001.json实际passed，SHA`0fe7a67b3c0dfe7f1ddea2bb8f647ff4c548df2cbeafc79d37148f7289c1d9f2`，10121字节；catalog SHA`f2af8ea427def98ebaf752e7050a6ab943bbc3d3572aeb910aa9ddd5747a4f3b`。12份技术/首次新receipt/四组NPZ/proof逐catalog SHA本地核验，四组真实有限FP32[N,1280]及L2范数通过。四路均至少2份非pilot新train产物，Slurm/scontrol实际JobID/lock/PID/命令与四不同V100 GPU_UUID通过；GPFS D等待为真实存活，不当退出。

核验快照：shard0实际3792930/PID3282566，首新图2007_000121/[2,1280]；shard1实际3792931/PID2294668，2007_000170/[7,1280]；shard2实际3792932/PID818559，2007_000063/[3,1280]；shard3实际3792901/PID818558，2007_000068/[2,1280]。GPU UUID全文见proof。本地local_startup_verification_001为startup_proof_verified而非最终completion。四路日志旧进度250/200/250/250仅快照，尚无12匹配MLP完成或新mIoU。后续按既有afterok链继续，不重复startup/提交/独立审核；固定批结果无论升平降均收尾停止。


### 97.3 四组消融完整独立接受与最终归档（2026-10-24）

本版19个科学child全部COMPLETED0:0，独立audit3792905亦C0/11:54。`completion_acceptance_001.json` statuspassed、binding`6472bcf8b239687603eb7dbc10705117cf198f80cbd3c26b7bf375eec85da344`，SHA`4defef154dff0f00794c0a7a64ec912d51ba4630da04be981a68d515ca75ef31`；manifest仍`557744db04d93f5824986faa1a85c129f248b7fcd389cebcd554312fdc2e086f`、冻结416科学记录未变。审计核59839唯一文件、2774289020字节、24预测组/34776真实新PNG及34776保存逐图CM；不解码rawGT/token/modeltensor，不重GMP/检索/训练/推理/bootstrap。本节最终状态覆盖97–97.2历史等待快照，不改旧记录。

| Arm | Selection | Aggregation | HCR mIoU% mean±sampleSD | clean mIoU% mean±sampleSD |
|---|---|---|---|---|
| A | Hard Binary / NN | Mean | 86.091188933±0.462041295 | 85.244631934±0.351834508 |
| B | Hard-Center | Mean | 86.268826663±0.159349042 | 85.386750212±0.131002042 |
| C | Hard Binary / NN | GMP | 86.380753236±0.234753390 | 85.611792115±0.194858442 |
| D | Hard-Center | GMP | 86.505997478±0.274697130 | 85.699896379±0.141559321 |

HCR逐seed42/123/2026：A=86.410642016/85.561402323/86.301522459；B=86.444359713/86.228843280/86.133276996；C=86.646872730/86.292352930/86.203034050；D=86.821717076/86.321692168/86.374583191。全部12匹配MLP均从头训练，D原raw也重新计算，未直接拿旧86.821717%填表；D与历史数值相符不宣称跨平台全数组或模型逐位相同。

HCR配对meanΔ/sampleSD/正seed：B−A +0.177637730/0.436036329pp、2/3；D−C +0.125244242/0.083072510pp、3/3；C−A +0.289564303/0.417283626pp、2/3；D−B +0.237170816/0.142299312pp、3/3；D−A +0.414808545/0.343629768pp、3/3。interaction=(D−B)−(C−A)为−0.052393488pp、sampleSD0.516873135、逐seed+0.141126650/−0.638101718/+0.339794605pp，不能宣称超加性协同或显著性。NN下GMP在seed2026为−0.098488410pp，不能说所有seed/selection均提升。clean各配对及全21类/24组错误计数详见analysis与CSV；D−A clean mean+0.455264445pp、3/3正，interaction−0.054014014pp、1/3正。

Selection诊断分母34880训练region，NN平均支持1569.044094037、Hard-Center1569.061725917，两者fallback均0；逐region选中集合Jaccard均值0.862637881（不是全部token汇池IoU，不是query精度）。34880含10569BG无重过滤。全部17388×2=34776新PNG先SHA commit才一次1449图GT；最终center.2/单匹配库/原NMS/HCR，R3关闭，内部epoch选择center.4不变。VOC仅已查看开发集，无独立test或新增bootstrap/参数/seed/arm。

角色Slurm elapsed：technical2:09；extract四路48:12/49:28/53:27/53:27；merge21:52；train_predict按A42/A123/A2026/B42/B123/B2026/C42/C123/C2026/D42/D123/D2026为00:22:19/00:24:44/00:23:10/00:23:10/00:23:01/00:26:45/00:31:28/00:28:39/00:23:03/00:28:28/00:23:25/00:22:10；evaluate4:11；独立audit11:54。整体从北京时间14:02:54至16:54:51，墙钟2:51:57（10317秒），不累加并行。最大保存Torch allocated峰值111.297363281MiB，不是驱动整卡峰值；角色execution wall和CPU MaxRSS见report。CPUmerge/evaluate/audit仍占1V100配额但CUDA隐藏；train_predict混合训练/导出/预测，无独立逐region求解timer，不能捏造性能倍数。

旧140000注册前同名plan保护失败无manifest/JobID/计算，全部attempt保留；142000仅计划命名空间隔离，没有科学修参救结果。87686监测本地进程失效仅重建pinned98515连接，无任何科学重复提交。17无skip/2train2query/当前D生产bitwise/6native历史像素回归和跨平台非bitwise边界均如实保存。

接受后96份catalog小文件及资源receipt共97份、16913679字节逐SHA下载至`experiments/hpc_results/selection_pooling_factorial_20261024_142000/verified_20261024`。catalog SHA`70ebafe013b81d0b225f1b606e143394d78762151d3bf0fdf18555852269bc08`；analysis报告SHA`35b214b905df2f4d4e3d690c9bf3c7063dacd8cd7712ab405979bd7a1ec66fb7`；analysis_tables SHA`1041dcb2e6aa493da550df98d94ce139c7b05bda9c16bee493a68a1b628cb11e`；download_verification SHA`0ac9453ad4cae1490e351b52815cc1ff98153fbe6f6f99f06a95cca730e225c6`；local_archive_proof SHA`e671ad7899ea7155478586a5f3806878c85adbe6744793d58ae8dfe689d7700b`。小analysis/proof回传本run并核远端SHA，归档receipt与本节CAS同步proof另存；没有下载大token/model或34776PNG树。

固定批完成且停止，无后续搜索/额外研究/主线部署/GitHub/ZIP。最终一次结果通知后仅删除selection-pooling-factorial监测，已完成RA/Robust/HPUD/PAF不重启。

## 98. GMP descriptor的受控MLP/projector内部实验（2026-10-24用户附件授权）

附件原文与三项明确回复保存在experiments/plans/mlp_projector_v1。用户确认先普通GMP，E0–E5×seed42/123/2026共18内部训练；低margin[0,.02)门槛按三seed平均top1差严格>0；真实production balancedK25/top5/search2/center.2宏准确率选epoch。历史v1实际5-NN宏准确率选模，center.4在该分支被忽略，因此E0改变选模规则而非仅center标量；所有训练本身照实际历史report/code不猜参数。原始1280D raw、34880成员含10569BG、8466fit/2116hold图和27939/6941 region不变，不能重新过滤。

新独立tools/mlp_projector_models.py与tests/test_mlp_projector_models.py已实现E0原MLP、E1Linear、E2RMSNorm、E3冻结MLP0零初始化残差alpha.1、E4双head等权融合；主线src/config/历史权重不改。原CE classifier不在旧checkpoint中，保留历史新初始化可训练分类head，E3仅projector残差可训练、MLP0参数冻结且输入梯度图保留。E5严格在E3真实技术验证后才实现，按固定CE+.25SupCon+.25bank+.1preserve、tau.07、positive5/三错误类各5、冻结fit-only teacher与全部同图排除；优先复用已接受RA反馈IDs与排除逻辑，不引入RA特征/scorer。

先代码审查→单元测试→seed42真实smoke→固定三seed内部实验，gate只报告不触发VOC。禁止VOC GT评测/选epoch/选模型、主线改动、历史覆盖/删除、参数或seed搜索、发布/部署/ZIP。当前本地无Torch，尚无真实单元测试或GPU启动证据、JobID或质量结果；后续以不可变probe/full manifest、实际Slurm ledger和独立proof为准。新入口/目录遵守storage数据、weights权重、experiments指标、logs日志；手册同步必须备份旧remote并CAS，不覆盖其它批。

### 98.1 E3真实技术接受、完整快照与固定18组链（2026-10-24）

三项用户回复已全部明确：只普通GMP18组；E0按真实production balanced K25/每类top5/search2/center.2宏准确率选epoch；低margin条件按三seed平均top1差>0。历史5NN/center.4只旁路记录，不能声称E0只改变center标量。历史训练配方实际为AdamW lr.001/wd.0001、CosineAnnealing50epoch、100step/epoch、8类×至多4张不同图片、CE+.5SupCon、温度.1、projector clip5，背景negative-only。保留历史E0初始化，所有arm同seed CE分类head初始化和采样序列SHA强制一致。

本地12项保存指标/gate算术测试无skip通过；本地无Torch，因此Torch数值测试放在真实Slurm。E3独立技术版本mlp_projector_e3_probe_20261024_180000已Job3795563 COMPLETED0:0/3:12，克隆419旧文件+6新文件=425冻结记录。manifest SHAe33fe7cf7a7e76170d99848f761d595588b02402aa1d9bbf70c5bfac09aacab6、bindingeb025bec8157201349a40fbe9634b529c314c6251b109754e3f04cce29a90ddf；原probe源码/日志/manifest保留，6新文件已本地小归档。真实seed42原34880成员/10569BG/27939fit/6941hold身份通过，12模型测试无skip；step0 maxabs0、min float64cos0.9999999999999998，输入梯度norm119.25080108642578，3真实optimizer steps后MLP0张量不变。report SHAc664cce88ee338e238f472d4483263a96207da9e5740b9cd0f3f1ae18a546656及tests/complete小proof逐SHA下载核验。GPU TeslaV100-SXM2-32GB，execution wall130.570652723秒、Torch allocated峰值24.788574219MiB；不是正式18组完成或质量结果。

只有此真实接受后才实现E5，继续E3相同零初始化残差架构，固定CE+.25SupCon+.25bank+.10preserve、tau.07，复用独立接受RA-GMP缓存的positive5/三错误类各5 IDs及原validate_feedback同图排除逻辑。只读取签名teacher B0向量与反馈，不加载RA特征/scorer、token或额外pool/context，不重新挖邻居。原cached teacher mining来自HPC普通GMP replay descriptor，来源边界保留；本批六arm始终使用同一份native普通GMP raw。E3/E5 base为同seed历史MLP0，不是本批新选择的E0。E5和E3之差是注册组合loss差异，不能都归因单独bank项。

完整唯一版本mlp_projector_20261024_183000，run=HPC ROOT/experiments/mlp_ablation/版本，audit=ROOT/experiments/hpc_launch_audits/版本，权重在weights/mlp_projector/版本，投影库storage/mlp_banks/voc/版本，日志logs/mlp_projector/版本。冻结Code419克隆+13独立代码/测试/计划补丁=432记录，未改production src/config；manifest SHA78e193db3127ce083b19e5f9b6c4d10ca2d9ecd66a58e2ee658a8a4b11f688e4、bindingcc18bdaaf6de549a5500c2040bfa2c215a30eeb2fbd53c822ca477313268569b。组quota真实余218952744960字节，按全epoch checkpoint至少保留12GiB，不用df。旧pre-registration draft002本地哈希映射错误在上传前拦下，003重新生成全部SHA后才上传，坏draft保留且无注册/计算。

输入缓存由Slurm technical调用原load_training_bank规范化一次，保存FP32统一transport cache到storage/mlp_projector/版本/normalized_inputs.npz；不是新GMP/encoder/token/伪标签或重新过滤，避免worker重复规范化差异成为变量。全部arm强制同raw数组/成员规范SHA、同seed采样和分类head初始化。最终checkpoint采用独立experimental格式/库签名，普通主线不自动加载。

唯一afterok链：technical3796804（28目标无skip+全部6arm seed42真实smoke，各2epoch×3step）→E0 3796805 array0-2%3→E1–E5 3796806 array0-14%4→CPUsummary3796807→独立audit3796808。20个科学child加1audit，已实际全部提交成功；7小manifest/ledger/注册/clone/launch/expected/guard凭据本地核SHA、binding、432文件及每条afterok边，local_launch_verification_001只launch_inputs_verified，不是最终接受。有JobID/guard绝不重复submit/register/upload科学代码或手动verify活audit。

最近已核2026-10-24T11:58:14Z快照：technical RUNNING12:58/c2v100m2-5，28目标测试0fail/0error/0skip；smoke E0/E1/E2完成、E3进行中；E0三seed及候选15任务仍依赖等待。此为历史快照，不提前称六smoke已通过、18正式训练已启动或最终gain。后续严格以work_state与实际ledger/complete/proof为准。所有compute仅Slurm一GPU一worker；technical1V100/6CPU24G4h，训练各1V100/6CPU24G48h，summary/audit占1V100配额6CPU24G12h但CUDA隐藏，不称零GPU分配。

逐epoch保存loss/top1/macro/top5/margin/NN25、相对最终E0的rescued/damaged与五箱/IDs；E4两头分别诊断并保存全部query cosine，正式固定等权融合。每run每epoch权重与optimizer/scheduler状态、新best.pt、全量bank/export/config/CSV/MD保存；source检查禁止无效成员静默过滤。28测试+全6smoke完整C0后才afterok正式18组；所有18组均固定跑完，gate只报告不启动VOC。完整独立audit接受后逐SHA小结果下载、报告归档/成本/整体最早到最晚wall/手册收尾；无mainline/部署/GitHub/ZIP或自动参数修改，不重启其它已完成实验。

### 98.2 固定18组正式启动的独立外层验收（2026-10-24）

technical3796804已COMPLETED0:0/20:39，28目标测试0fail/0error/0skip，E0–E5六arm均完成真实seed42短程验证（各2epoch×3step）；其参数/权重独立于正式50×100任务。technical execution wall1185.026418209秒，保存Torch allocated峰值70.863769531MiB，不是整卡驱动峰值。旧98.1等待快照保留，本节覆盖当前状态。

已唯一执行外层validate_startup_001.py，startup_validation_001.json statuspassed，SHA998f4adc72cc05a1ac924bcfc59661a67cd33074031578c42390a9a35fea4fb3；catalog SHA12fef5535b84e86cc9c653e801af5a40009370c0c63adc06d89140adf8576f07，8小proof逐SHA本地下载核验。三seed各首轮新checkpoint和6941query数组字节SHA、完整ID顺序/合法预测/有限margin、精确Slurm child→actual JobID、lock/PID/entry/manifest/phase/index以及三不同V100 GPU UUID均通过。检查只读保存数据/进程，不训练推理，不读VOC GT/token/modeltensor；没有下载模型checkpoint或重做检索。

seed42 logical3796805_0→actual3797051/PID1295264/c2v100m2-5/GPU-74bd20a2-39d8-6895-6011-990eed49344a；seed123 logical3796805_1→actual3797052/PID668494/c2v100m2-9/GPU-4c56466d-decf-62f8-61c9-8957076a484f；seed2026 logical3796805_2→actual3796805/PID668493/c2v100m2-9/GPU-0df27264-355c-9d0d-bf50-382898c6e32a。GPU memory448MiB只是当时驱动采样，不是Torch峰值。三worker当时真实RUNNING，未伪造完成worker存活。

最近2026-10-24T12:48:25Z科学快照：E0三seed RUNNING42:30，日志epoch34/22/22（seed42/123/2026），无failure；候选15组/summary/audit仍afterok等待。这些是运行进度，不是50epoch完成、最终best模型、promotion gate或VOC mIoU结果。正式启动验收只首次通知一次，后续不重跑validator、不重复既有科学/审计提交；有JobID/guard的432冻结snapshot保持原字节。

manual98.1已保留旧remote9e41057978d5cf7cec782ec5242cbb7f333fda76106471eebf16f6c42ce77856并CAS同步a69787afef7e905ee100e036178550ef2e8ed6f23fcaf815286f1cb2aa9f4937，manual_sync_001.json SHA b71d608e723ca5ee364119a5c4ef76ef5e3b1d18ad4afab6fd64787da2a30d2e已小下载核。此次同步同样备份实际旧manual并CAS原子替换，不修改冻结科学code。所有18固定训练继续；无论gate正负不自动跑VOC，不改变主线/参数，不追加研究。

### 98.3 内部18组完成及独立接受，另授权VOC follow-up（2026-10-25）

本批20科学child及独立audit3796808全COMPLETED0:0；audit elapsed55:57，核5223唯一文件/7921617779字节，completion_acceptance_001.json passed/exactbinding，SHA6280dcbc173f7d1d6d1dd84b82d7731e94cd53bd1badeab8ad06113344259189。small_result_catalog_001 SHA55cd5e5781e7c44210acf1ceba8a5f434146e18a16ce474c8223506c3f44baea。六小summary/accept/catalog/gate/CSV/analysis已下载，五非catalog文件依catalog逐SHA核，再与远端SHA anchor核accept/catalog；原内部manifest的VOC_authorized=false保持不变。

内部伪标签Top1三seed mean±sampleSD：E0=98.760985449±0.024953909%，E1=98.592902079±0.193471038%，E2=98.564087788±0.064965419%，E3=98.756183067±0.044014558%，E4=98.607309225±0.068085515%，E5=98.837823560±0.029990866%。E5−E0 mean+0.076838112pp、3/3正，macro+0.016925909pp、low-bin+4.936439222pp。E5仅meanTop1>=.10pp未过，8项条件过7项；所有候选原gate均not_supported，不能改阈值声称通过或将内部Top1解释成VOC mIoU。原完整小结果/成本/独立analysis归档仍待完成，不能把六小summary下载冒充完整归档。

用户随后明确要求“现在看看每一个的最终miou能到多少”，因此新独立VOC follow-up按第99节执行全部六arm三seed，并非原gate自动promotion；原内部协议、权重、manifest、结果和失败历史全部保留，不原地修改。

### 98.4 内部18组完整小归档与独立分析完成（2026-10-25）

本批20科学child与audit3796808全C0，原accept SHA6280dcbc173f7d1d6d1dd84b82d7731e94cd53bd1badeab8ad06113344259189、catalog SHA55cd5e5781e7c44210acf1ceba8a5f434146e18a16ce474c8223506c3f44baea不变。157小catalog文件共11851264字节已全部逐SHA下载核，而非此前仅六summary。归档目录experiments/hpc_results/mlp_projector_20261024_183000/verified_20261024，新增独立analysis_report.md（SHAeefac63415450fb91099c6c0242c123d8709b88085ca89f40fcdd6c4aac702e7）、analysis_tables_002及完整每类/五箱CSV，含18组逐seed/mean/sampleSD/paired、完整救损/高箱损伤、E4两头及6941query cosine、参数与optimizer/eval/export/保存forward benchmark分开。E0为基线无candidate gate；五个候选均未过全部条件，E5仍只7/8，不能改旧门槛或归因单独Bank。

整批最早start至最晚end为2026-10-24T19:45:16+08:00至2026-10-25T03:06:52+08:00，墙钟26496秒（7:21:36，不累加并行）。Slurm角色耗时/MaxRSS/AllocTRES全部保存archive_scheduler_resources_002；进程级保存Torch allocated最大71.4755859375MiB，不是驱动整卡峰值。E3前置probe3:12另列，不与full并行时长求和。download_verification SHAbcd0b2b3386e898b24e26e4e4fa9e0b163e8237d3c30adc167b012a6a42e7d9c；local_archive_proof_003 SHA18b6d60b95ceefb1b457bf3790fdc88c24e8f6a43a0feed844730c08ce440e90。小analysis/CSV/download/archiveproof回原run逐远端SHA核，原科学432code、模型、数据、gate、VOC=false历史保持。旧preupload草案与E3技术历史均保留，无新的科学失败。第99节VOC是后续用户明确授权，不反改本内部结论。

## 99. 六种MLP/projector固定VOC mIoU follow-up（2026-10-25新明确授权）

入口tools/mlp_projector_voc.py及独立tools/mlp_projector_voc_audit.py/core，计划experiments/plans/mlp_projector_voc_v1；唯一计划版本mlp_projector_voc_20261025_120000。每种E0–E5各42/123/2026共18组固定已接受best checkpoint及各自匹配34880成员单库；E4固定等权融合，E3/E5保留历史MLP0基座。所有arm均评测，不因原gate失败剔除，不重新训练或按VOC重选epoch。新格式按各自模型factory严格reload，不把所有checkpoint塞进旧两层MLP loader，不改变production src/config或旧432 snapshot。

只读原native普通GMP保存1280D query descriptor，依原mask hash映射冻结proposal pool/order/scores/HCR flags；不重GMP/encoder/SAM3/token/pseudo/proposal。统一balancedK25/top5/search2/center.2/margin0/proposal mixture.2/NMS.4/instance threshold.02和原HCR，R3关闭。主表最终HCR mIoU，clean辅助。36逻辑预测组/52164真实新PNG全SHA commit后才本批GT；1449GT各一次解码，保存36逐图CM/21IoU/BGFP/FN/wrongFG/chair-sofa、逐seedmean/sampleSD和相对E0的配对方向；VOC仍是已查看development，不当独立test。

先6个本地保存CM/配对统计测试（已无skip通过）→Slurm6CPU24G1V100技术阶段6h：18checkpoint前1024同源输入重放原匹配export、18×2图真实clean/HCR形状类别有效、3seed×2原native HCR历史PNG像素精确→predict18array%4各1V100/48h→CPUeval12h→CPU独立audit12h。CPU角色仍分配1V100配额但CUDA隐藏，不称零GPU。新7补丁将逐SHA克隆注册到独立code，已有guard/JobID不可重复提交/手动verify活audit。当前登记/JobID/实时状态以后续work_state与实际ledger为准，未提前声称GPU技术或全量预测完成。

数据与输出storage/predictions/voc/版本，参数指标experiments/mlp_projector_voc/版本，冻结code/manifest/JobID在experiments/hpc_launch_audits/版本，日志logs/mlp_projector_voc/版本；不新增训练权重。失败留全部attempt，只停本ledger未开始依赖；科学修复新版本+针对真实probe，不改运行snapshot/断言。独立accept passed+auditC0后小summary/36CM/CSV/proof逐SHA归档，分析每种模型最终mIoU、seed方向/类与错误成本和资源口径，再backup/CAS原子同步本节。无论升平降固定批结束，不调阈值/seed/权重/结构、不追加研究、不自动部署/主线替换/GitHub/ZIP。用户暂停取消只停本batch和监测，不自动恢复，不动其它实验。

### 99.1 六种模型VOC follow-up已唯一提交（2026-10-25）

第99节独立版本已439冻结记录，manifest SHAb891d7ae92db95e9861b15005b4d86643c9c666a135bf7071f25ef600d6c6756，binding29f06dba06bfecc6d53578225adf4926001caf3ba5396dbd2e8433410f3531db。唯一afterok链technical3801100→predict3801101 array0-17%4→evaluate3801102→独立audit3801103，实际ledger/guard与6小launch凭据本地核SHA/binding/资源/所有依赖边。quota余203942690816字节。最近04:12:19Z快照technical RUNNING3:09/c2v100m2-8，日志空、targettests尚未出，其余PENDING；不当作GPU技术/最终预测/mIoU完成。后续以该plan/work_state最新为准。此批此前用户明确授权，和下述新内部loss拆解互相独立；新附件不向既有VOC链追加模型，不读取其结果调新loss，不修改或取消已有链。

### 99.2 六种模型VOC完整评测、独立接受与归档（2026-10-25）

本批20科学child及独立audit3801103全部COMPLETED0:0，technical9:54、evaluate17:10、audit18:01；18predict分别17:00/26:34/21:13/5:36/9:47/4:03/6:35/5:20/5:12/6:29/4:45/6:56/6:53/4:38/5:27/4:24/8:02/5:58。accept passed/exactbinding29f06dba06bfecc6d53578225adf4926001caf3ba5396dbd2e8433410f3531db，SHA0e9023d270df00c02b55907f9d445135e9d9e9005a07b93a59185223de1b6254；catalog SHAa5659e318db26e9f25baab659688e3b281a6dc2216a417abf20727831e2028c0。审核53138唯一文件/1136782096字节，36组/52164真实新PNG全部SHA commit后才一次1449GT，保存36逐图CM、21IoU与错误成本。无failure/lock，18checkpoint各1024输入reload maxabs0，6native HCR pixel exact和6tests无skip；首4完成worker以C0+execution核真实新16PNG，不伪造存活。startup proof SHAfd768a17355955d9cdade1d5ce342fa983c1af69cf3fc26dfb583a9aefde205a及4小proof/catalog本地逐SHA已核；完成结果覆盖迟到的启动通知，不重复启动/重跑任何JobID。

| Arm | HCR seed42 /123 /2026 mIoU% | HCR mean±sampleSD% | HCR meanΔE0 pp /正seed | clean mean±sampleSD% |
|---|---|---|---|---|
| E0 两层MLP |86.833484 /86.562251 /86.465041|86.620258±0.190948|0 /基线|85.781213±0.144475|
| E1 Linear |86.311658 /86.537694 /86.685031|86.511461±0.188064|−0.108797 /1/3|85.600307±0.163337|
| E2 RMSNorm |85.765844 /85.081838 /86.034236|85.627306±0.491080|−0.992952 /0/3|85.153395±0.446453|
| E3 冻结base残差 |86.870522 /86.174829 /86.496617|86.513989±0.348172|−0.106269 /2/3|85.743498±0.214497|
| E4 双头固定融合 |86.105329 /86.345859 /86.515044|86.322077±0.205891|−0.298181 /1/3|85.403073±0.239844|
| E5 组合loss残差 |86.273346 /86.430361 /86.242704|86.315470±0.100671|−0.304788 /0/3|85.604055±0.053079|

本固定三seed条件下E0平均HCR最好，五候选平均都低于E0；E5内部Top1+0.076838pp未转化为VOC HCR收益（−0.304788pp、三seed均下降），不能将组合loss结果单归因Bank。E3两个seed小正，一个seed较大负，不能挑seed42最高值当总体胜出。完整21类/paired/errors在独立analysis和CSV，FN/BGFP/wrongFG/chair↔sofa口径分开，子项不相加冒充mIoU贡献。普通GMP、原pool/center.2/K25/top5/search2/NMS/HCR、R3关闭；固定内部best checkpoint，无重训/VOC重选epoch/调参。VOC是已查看development而非独立test；不自动替换主线或追加研究。

整批墙钟2026-10-25T12:09:10+08:00至2026-10-25T13:49:58+08:00，6048秒（1:40:48，不累加并行）。保存Torch allocated最大31.134765625MiB，非驱动整卡峰值；CPUevaluate/audit仍有1V100资源配额并隐藏CUDA。66小catalog文件共16066882字节逐SHA下载至experiments/hpc_results/mlp_projector_voc_20261025_120000/verified_20261025，另保存scheduler资源receipt。analysis_report.md SHAad05246ed2f3bc8aae9093a8f053cd3a92c2405ec5de1f5eef61bfd7f710b093；download_verification SHA9eee2425da44c2103d24ac2b91b926e4c23371da387f02891832fd4d78a2692a；local_archive_proof_003 SHAa2288012bf86008409d135cd69c09baddf085ed48428a617ad7402f3943f3485。小analysis/CSV/download/archiveproof回run逐SHA核，不下载大checkpoint/bank/token/52164PNG树。439冻结科学code、所有原内部历史、其它批与主线未改。本固定批完成即停止，最终一次结果通知后只删除mlp-projector-v1监测，不删除独立loss任务。

## 100. 固定残差projector的loss因果拆解（2026-10-25新附件授权）

用户附件原文、固定矩阵与操作状态在experiments/plans/retrieval_aligned_projector_factorial_v1。唯一计划版本retrieval_aligned_projector_factorial_20261025_130000；run=HPC ROOT/experiments/retrieval_aligned_projector_factorial/版本，权重weights/projector_loss_factorial/版本，日志logs/projector_loss_factorial/版本，audit/code独立冻结。只新增R_S025、R_S025_bank、R_S025_preserve×42/123/2026共9完整训练；复用已接受E3为R_S05_existing、E5为R_S025_bank_preserve，另E0作固定分箱/比较基线，共9个复用selected模型。绝不重训E3/E5或扫权重/架构/alpha/tau；旧E5+0.076838pp没有过0.10pp门槛的历史结论不改。

结构直接复用已接受FrozenBaseResidualProjector和同seed历史MLP0：冻结1280-512-256 base，256-64-256/GELU残差，alpha.1/A2全零，保留输入图不detach。分类head、原balanced_batch/seed逻辑与AdamW.001/wd.0001/Cosine50×100/8类×4不同图/CE与SupCon温度.1/BG negative-only/只裁projector梯度5完全沿原训练条件。A=CE+.25SupCon，B=A+.25Bank，C=A+.10Preserve，D复用原E5=A+.25Bank+.10Preserve。Bank类与Preserve表达式均用已接受E5实现，tau.07、positive5/三错误类各5旧缓存、全同图排除、teacher detach，不生成新negative或RA scorer。

数据同一已接受normalization transport cache，34880含10569BG、27939fit/6941hold region与8466/2116图，IDs/class/order/raw/split不变、不重滤。Slurm references阶段核原E0/E3/E5 sampler/init/descriptor/member SHA一致，复算原5000步采样prefix且最终SHA必须等于历史实际报告；新训练逐epoch及末尾核同prefix，classifier init、split与neighbor cache SHA错即technical failure。固定同图排除、fit gallery，不重GMP/SAM3/token/pseudo/proposal/bank成员。

新诊断用进程内只读observer捕获原classify_queries返回的全部21类production class_scores，不改变返回值或分类规则，finally恢复函数。Signed=score(内部伪标签正确类)−最强错误类；correct rank用原np.argsort(scores)[::-1] tie规则；保存signed mean/median/positive/p10/25/50/75/90与rank1/<=2/<=3/<=5，保留旧raw margin。复用E0/E3/E5只在原selected checkpoint对应导出向量上补诊断，预测/Top5/NN25须回归完全一致，replayed raw margin maxabs<=1e-7单列，保存的旧raw margin原样复制且E0五箱逐位固定，不用候选或重评边界改箱。新9训练50epoch全保存分数/权重，仍production balancedK25/top5/search2/center.2宏准确率first strict max选epoch，不按signed或VOC选模。

全部run对同seed E0和A两种reference保存配对指标、五箱N/救损/net/signed变化、完整rescued/damaged IDs与正确类/最强错类分数、全confusion flow及top20 changed flow，不只挑成功。A−R_S05是唯一SupCon权重对照；固定.25下算B−A/D−C、C−A/D−B及Bank/Preserve主效应与两等价interaction式（仅算术一致atol/rtol1e-12）。low-bin准确率和high-bin damaged的factorial效应用E0共同分箱；damage正效应表示更多损伤。Bank方向字段按用户B−A四条件，D−C另外核；Preserve两个环境mean方向和signed方向分别列出，不将固定矩阵解释为普遍因果定律。

14个本地纯NumPy新指标/固定箱/全部错误样本/交互算术测试已无skip通过，本地无Torch；真实seed42 smoke尚待Slurm，不能把本地测试当GPU或质量结果。计划432已接受旧code+8新补丁=440冻结记录，原代码/指标/权重不覆盖。独立chain为CPUreferences1V100quota/CUDAhidden6CPU24G12h→GPUtechnical6h（14无skip/输入梯度/E5组合loss与梯度等价/三新arm各2×3 smoke）→train9array%3各1V10048h→CPUsummary12h→独立audit12h。CPU角色仍占V100配额，不称零GPU。实际manifest/JobID/状态只以work_state/actualledger为准；已有guard/JobID不得重复注册/提交或手动verify活audit。

本批严格禁止VOC GT/segmentation/独立test/跨数据集/新架构/权重搜索/Round2/自动主线部署；此前另授权的第99节VOC任务按原链独立运行，不纳入本研究反馈。失败留attempt、仅停本ledger未启动依赖；科学修复新版本+针对probe，不能改冻结断言；纯审计/报告修复先旧job终态无allocation，再新外层文件不重训练/检索/GT。用户暂停取消只安全停本批和监测，不自动恢复，不动其它实验。

每run保存config、checkpoint或原权重引用、log、epoch CSV、selected完整score/rank NPZ、perclass/bins/signed/flow CSV、全部rescued/damaged JSONL、runtime/manifest/独立acceptance；最终factorial_summary/effects/signed_margin_summary CSV及analysis回答附件8问。独立accept passed+auditC0后小文件逐SHA下载、独立可读分析、小归档proof回远端核SHA，备份旧manual并CAS同步。原复用训练成本与本轮额外内部重评成本分开，不将省略旧旁路5NN/.4诊断带来的总wall差异归因loss速度；整体最早start至最晚end wall不累加并行，Torch allocated不冒充驱动整卡峰值。无论正/平/负固定批到此停止，只在完成独立验收归档后一次最终结果通知，不自动追加VOC或其它研究。

### 100.1 首批接口失败、安全清理及同科学配置的新版本恢复（2026-10-25）

旧版本retrieval_aligned_projector_factorial_20261025_130000的references3801463实际FAILED1:0/2:24；failure SHA0ef138ca30ee40426973e2f34d016f74f08a994ea00afbf9dfc3ebf0c8b3c3f6，日志SHAf7d0942b75e7fdae5a1238e1477c508398de56b353a58c55e8e513730cb80b1d。原因是verified返回NPZ bytes，但运行时把bytes直接交给期望path/file-like的OLD.load_arrays，NumPy报embedded null byte；不是loss方法质量结果。旧snapshot/manifest/guards/log全保留；仅本ledger尚未启动的technical3801464、train3801465_0–8、summary3801466、audit3801467共12child已CANCELLED且Elapsed0，无活allocation/lock。9新训练没有在旧attempt中开始，也未重训E3/E5。

只在新版本runtime中把这一处NPZ bytes包为io.BytesIO，其余runtime字节完全相等（patch SHA5fd932d246c2f1375b1f6bb890d40ed01901eced7f7ef6c787c3ec3c9944d329）；protocol仅改version值。旧原文保存在recovery_sources_002/old_runtime_preserved.py。单独真实Slurm NPZ probe3802284已COMPLETED0:0/59秒，CPU执行wall8.326694250107秒；报告SHA0c069b5b1bf97049f64cfe9042124736325196ed5b4d16927de7443207fc59e1，复现原错误后import实际patch模块，6941行原E0seed42已选query的全部7数组在buffer/path两种读法下逐值完全相同。该probe不检索、不加载modeltensor、不训练、不读GT/token或重GMP；分配1V100配额6CPU24G30min但CUDA隐藏，不能称零GPU资源。report/job/log/guard及原失败3小文件均已本地SHA核，local_probe_acceptance_002只接受接口probe，不是正式研究完成。

唯一当前版本retrieval_aligned_projector_factorial_recover_20261025_150000：run=HPC ROOT/experiments/retrieval_aligned_projector_factorial/版本，audit=ROOT/experiments/hpc_launch_audits/版本，weights/projector_loss_factorial/版本，logs/projector_loss_factorial/版本。manifest SHA153cab3a695c6d39d637c5719ddc4e02de0cdbebdf3061a7212f029a836581e4，bindinga3d7f5b1d25d6e5ced2a15760a0c533b7e683397621148b967b1abedbc38c606，440冻结records=438原字节不变+runtime接口/protocol.version两patch；438包含原接受432及本研究其余新文件，production core、数学loss、架构/alpha/tau/权重/seed/初始化/采样/数据/成员/分箱/选模/所有科学断言均不变。保留原失败manifest/failure及独立probe/job为anchors，注册quota实际余203722620928字节。local_registration_verification_002逐字段比较全部科学设置及440SHA，local_launch_verification_002核实际ledger/resources/依赖；两者只是inputs verified，不是最终独立accept。

实际唯一新链references3802712→technical3802713→train3802714 array0–8%3→summary3802715→independent audit3802716；新live边全afterok，root提交前已核真实probe C0及旧ledger无活allocation。资源仍为CPUreferences/summary/audit各1V100配额6CPU24G12h、CUDA隐藏；technical1V1006CPU24G6h；每个train1V1006CPU24G48h。已有JobID/guard不得重提、重注册、上传修改科学snapshot或手动verify活audit。3新recipe×3seed仍只9完整训练，原E3/E5/E0仅复用selected结果；没有受第99节VOC结果影响而改变任何配方或选择规则，本批仍无VOC GT/分割/独立test/自动promotion。

最近已核历史快照为2026-10-25T08:03:59.705663Z（北京时间16:03）：references RUNNING4:43/c2v100m2-7，其它12child等待依赖。它不表示当前实时状态，也不表示新GPU技术测试、9训练或新质量结果完成；后续只以本plan/work_state与新V实际ledger/proof为准。local_control_recovery_002与quiet validate_startup_002明确新V，原001仍保留；startup须新technical完整C0/14无skip/真实梯度与三新smoke通过，首三A seed实际epoch1 score/checkpoint及Slurm/proc/GPU_UUID验收后只保存小proof，用户要求final-only不重复启动通知。

另有一次本地preupload手抄ANSI断行SHA变65字符被assert拦下，recovery_preparation_failure_003保留；无remote请求/manifest/JobID/科学计算，已改为实际文件程序化SHA及16字符输出校核。原所有历史失败、source旧内部gate及参数边界不改。科学失败先日志、只安全取消本新ledger未启动依赖；科学修复必须新version+真实小probe，不能改运行snapshot或放宽assert。固定批完整独立accept+auditC0后才小SHA下载/8问分析/归档/角色墙钟/本节最终更新；正平负都停止，不自动追加VOC或研究，不改主线/其它批/个人queue。


### 100.2 固定loss矩阵完整独立接受与最终归档（2026-10-25）

本节最终事实覆盖100–100.1历史运行快照。恢复版本retrieval_aligned_projector_factorial_recover_20261025_150000全部12科学child与独立audit3802716实际COMPLETED0:0，9新训练全部完成，9个历史selected模型只作保存向量诊断、不重训。独立completion_acceptance_001.json status passed、binding a3d7f5b1d25d6e5ced2a15760a0c533b7e683397621148b967b1abedbc38c606，SHA 9a08ba8a4afd61b9eefc89d18899fcc7bf5ba5716648a966ee66be70087130ed；审核2576唯一文件/2645813355字节，原科学断言未放宽，未重训练/检索/GT/bootstrap。manifest SHA 153cab3a695c6d39d637c5719ddc4e02de0cdbebdf3061a7212f029a836581e4，440冻结科学记录仍438原字节加两修复，旧130000失败/cancel/probe历史完整保留。

| 组 | Top1 seed42/123/2026% | mean±sampleSD% | signed mean |
|---|---|---|---|
| E0参考（复用） |98.732171/98.775393/98.775393|98.760985±0.024954|0.652263|
| R-S05原E3（复用） |98.717764/98.746578/98.804207|98.756183±0.044015|0.709046|
| A CE+.25SupCon |98.804207/98.717764/98.818614|98.780195±0.054545|0.779080|
| B A+.25Bank |98.861835/98.804207/98.847428|98.837824±0.029991|0.803489|
| C A+.10Preserve |98.847428/98.703357/98.833021|98.794602±0.079348|0.753552|
| D 原E5 Bank+Preserve（复用） |98.861835/98.804207/98.847428|98.837824±0.029991|0.788495|

附件8问结果：①A−R-S05 mean Top1+0.024011910pp、2/3正，signed+0.070033798，macro−0.002384360pp，SupCon减半收益小且seed混合。②Bank B−A+0.057628584pp、D−C+0.043221438pp，均3/3 Top1正及3/3 signed正（分别+0.024409849/+0.034943098），各seed高箱pairwise damaged均0，两个Preserve环境均满足注册四条件；Bank平均主效应+0.050425011pp，不与按B−A定义的direction字段混称。③Preserve C−A+0.014407146pp、2/3正，D−B=0pp，signed分别−0.025527347/−0.014994099、均3/3负，不支持稳定独立收益。④Top1 interaction mean−0.014407146pp、sampleSD0.028814292、1/3正；signed交互+0.010533249，不能称准确率正协同/显著性。⑤Bank最低箱Top1差mean+6.882429740/+6.411049268pp，N=55/45/49；最低箱净救回10/9个query×seed决定，全部query净12/9，其它箱仍有抵消，不是独立20823样本或mIoU贡献。⑥五主配对高箱pairwise damaged均0/0/0，未见高置信伤害，但无法将零损伤归因Preserve额外保护。⑦signed与Bank同步提高，Preserve两环境下降；macro与NN25并非每seed都提高，完整五箱/每类/错误ID/flow与rank/quantile保留。⑧可将较简单Bank-only B列为另行授权固定VOC验证候选，本批不追加执行、采用主线或改旧gate。B/D相同Top1不代表逐预测相同，seed42有1个query类别不同（其它seed0）。

所有结果仅同6941 query/seed的内部伪标签development检索，全部按同holdout生产macro选epoch，不是VOC mIoU、独立test、跨数据集或普遍因果结论。旧E5+0.076838pp未过旧0.10pp门槛结论保持；本矩阵不依据第99节VOC结果修改loss/alpha/tau/seed/选模或分箱。固定结构/同seed基座/采样/初始化/数据/成员/SHA/teacher/全同图排除均保持。没有新架构、参数搜索、Round2、主线部署/GitHub/ZIP。

Slurm elapsed：references8:28，technical13:30；9训练按array0–8为18:05/29:43/33:31/17:01/19:08/19:29/17:49/35:35/18:35；summary20:45，audit38:08。恢复full科学allocation最早北京时间15:59:16至最晚18:44:35，9919秒=2:45:19；包括旧失败/probe/恢复等待间隔的科学allocation时间窗13:09:02至18:44:35，20133秒=5:35:33，不累加并行、不冒充连续计算，也不包含之后本地下载/分析/归档耗时。原references失败2:24，接口probe Slurm59秒/实际执行8.326694250107秒。新9完整训练45000step与technical18smoke step分开；复用9模型new_training_steps=0，历史原训练成本不当本轮重训。最大保存Torch allocated107.3828125MiB，非驱动整卡峰值；CPU MaxRSS为Slurm采样。CPU references/summary/audit/probe仍分配V100配额但CUDA隐藏。optimizer block含采样/传输/前后向/更新，内部评估含投影/检索/保存与SHA，不能拿省略历史5NN/.4旁路后的总wall宣称loss加速；无新GMP/独立forward benchmark。

科学small_result_catalog_001.json SHA 2b3c94f993c57f0d350c0f512c46c0ad63c5cd6e4c87c02387ec5765678c5d22，551小文件33479404字节均逐size/SHA下载到experiments/hpc_results/retrieval_aligned_projector_factorial_recover_20261025_150000/verified_20261025，另有12外层proof记录。未下载大checkpoint/raw/token/全50epoch-score树。独立analysis_report.md SHA 3dbcad92be2130fde74006fa4f77e5b34e0954607dc1941449820573c4234862；download_verification_003.json SHA f5be8be2f745829ab1044f9dc96fc91c9f8b041f5647775a6411ecf492a12eb6；local_archive_proof_004.json SHA 67835285464c36032f381a5b2a9ea905bd09e518718ccffcb344e0bedf1f4203。六analysis文件、下载proof与local archive proof回当前run逐远端SHA核，最终roundtrip与manual_sync_003另存证明；不改冻结科学code/metrics/源历史。当前固定批无论正平负均完成即停，最终一次8问结果报告后仅删除projector-loss-factorial-v1监测，不重启其它已完成批。


## 101. 椅子／沙发局部混淆的检索类分数间隔训练（2026-10-25新授权）

用户在讨论真实检索分差训练后明确“尝试一下你的想法，做做试验”，本批计划experiments/plans/retrieval_score_margin_v1。当前独立版本retrieval_score_margin_recover_20261025_230000；run=HPC ROOT/experiments/retrieval_score_margin/版本，weights/retrieval_score_margin/版本，storage/predictions/voc/版本，logs/retrieval_score_margin/版本，audit/code冻结451records=已接受loss恢复版440逐SHA克隆+9独立代码/测试/计划+2个已接受VOC复用模块。production src/config、旧模型/指标不改。manifest SHA5b6146b2856f6cdadd410b3de52b010229dbfd42ca41ce2f13a1f62555073142，binding3b0becee9718bef0748cb31ac7742e22dcc4dae42883e012b379ba6f74a714e2，真实GPFS group/data余199712686080字节，最低保留12GiB。

A_supcon复用第100节R_S025三seed（CE+.25SupCon）；B_fixed_bank复用R_S025_bank三seed（另+.25Bank）；C_score_margin只新增42/123/2026三个完整训练（CE+.25SupCon+.25Rank）。A是同架构loss控制，不是原两层生产E0；E0仅提供历史固定内部五箱。全部同历史同seed冻结MLP0、256-64-256/GELU残差alpha.1/A2zero、同CEhead初始化/5000步采样/AdamW.001/wd.0001/cosine50×100/8类×4不同图/SupCon温度.1/BGnegativeonly/projectorclip5与FP32/TF32off。原34880含10569BG、27939fit/6941hold、8466/2116图、原raw/ID/order/class/split不变，不重GMP/encoder/SAM3/token/pseudo/proposal。只三新全训练，六原selected模型不重训。

Rank=mean softplus((.05+最高错误类检索分−正确伪标签类检索分)/.07)，系数.25，无Preserve或固定teacher Bank项。score按原global210候选/类内top5mean/.2中心/缺类中心fallback，所有21类可竞争。每epoch开始将全27939fit用当前模型投影并detach为图库，训练query保持梯度；从邻居和类中心中完整排除同图所有成员。明确为按epoch更新的滞后图库训练近似，不是每step实时重投影全库。最终内部评估/推理始终当前模型匹配库；C−B是两种辅助目标及监督方式的整体对照，不归因某一个margin标量。

Ranking可靠性仅用冻结历史MLP0在fit内leave-whole-image-out后top1等于伪标签且signedgap>0，前景anchor限定；不按VOC或chair/sofa特设规则，所有类同规则。不是GT可靠性证明，负样本仍可能有伪标签噪声。此规则不筛CE/SupCon或库成员；同图排除也覆盖中心。保存每seed eligibility/每类覆盖、每epoch图库SHA、实际负类对计数及每步eligible trace。没有heldout/VOC标签参与训练配对。全部50epoch按原production balancedK25/top5/search2/center.2内部macro firststrictmax选epoch，不按chair/sofa、signed或VOC选模；margin.05/tau.07/权重.25固定，不扫描参数/架构/seed。

唯一实际afterok链technical3805064→train3805065 array0–2%3→predict3805066 array0–8%3→evaluate3805067→independent audit3805068。14科学child+1audit已有guard/JobID不可重注册/重上传科学code/重提交或手动verify活audit。technical1V100/6CPU24G6h必须8NumPy+6Torch测试无skip、三seed真实score对原production（完整同图排除）at/rtol2e-6/1e-6且Top1exact、非零input/residualA2梯度/基座冻结、fit可靠性、Cseed42两epoch×三step探针、6native HCR历史PNGpixel exact后C0，才进入3正式训练。当前仅本地8NumPy测试通过，不能把登记/提交当GPU测试通过或正式训练已完成；实时状态以work_state和actualledger为准。

每train/predict各1V100/6CPU24G48h，一卡一worker。CPUevaluate/audit各1V100资源配额/6CPU24G12h并CUDA隐藏，不称零GPU分配。所有compute只Slurm；登记读取源字节/真实quota不解码GT/modeltensor。原Paramiko依赖本地AttributeError在认证前停止，已使用固定ED25519的native SSH/SCP，成功连接不当身份失败；私钥只本地不打印上传。GPFS/SSH等待或D状态不当作失败。

三个固定arm九模型全部做原pool/native1280D query/center.2/单库K25/top5/search2/mixture.2/NMS/HCR的clean+HCR对照，R3关闭；本新批明确VOC授权不改历史VOC=false/gate。严格factory reload须复现6941已保存selected classscores at/rtol1e-6及prediction/top5/NN25exact；source loss批没有34880-row历史导出，不能虚称此类回放基准，本批重新投影匹配导出但不重GMP。18组26082新PNG全SHA commit后才本批一次1449GT解码。内部Top1/signed/pairgap/五E0箱/全错误ID与最终18CM/21IoU/BGFP/FN/wrongFG/chair→sofa/sofa→chair分开，全部三seed逐值/mean/sampleSD/配对，两个混淆方向单列，不用平均中心距离或全局Top1代替分割结果。VOC只为已查看development，无新增bootstrap/独立test声明。

失败保留attempt，只安全停本ledger未开始依赖；科学修复新version+针对小probe，不修改运行snapshot或放宽断言。暂停/取消只停本批和监测，不自动恢复，不动其它实验。完整acceptpassed/exactbinding且auditC0后，smallcatalog小结果逐SHA下载、独立分析和roundtrip归档、角色/实际Torch峰值范围及整体最早start至最晚end墙钟（不叠加并行）更新本节，manual同步前备份实际旧SHA并CAS，冲突下载merge不覆盖98–100等其它章节。当前launch小凭据本地独立核SHA/451source/全部afterok及资源，仅launch_inputs_verified不是completion。无论升平降固定批结束，不追加研究、部署、主线更改/GitHub/ZIP；最终报告归档后只删除retrieval-score-margin-v1监测。


### 101.1 首版导入失败与依赖闭包恢复

旧版本retrieval_score_margin_20261025_224500 technical3804903实际FAILED1:0/32秒，在runtime顶层导入tools.mlp_projector_voc时因源快照未含该模块失败；尚未进入14GPU目标测试、真实分数/梯度探针、任何新全训练或预测，没有方法质量结论。其余train3804904_0–2、predict3804905_0–8、evaluate3804906/audit3804907共14个自有PENDING依赖已全部CANCELLED、Elapsed0，精确Slurm无活allocation/lock；源449code/manifest/jobs/日志/取消记录全保留，不原地改旧版本。

新recover230000快照451记录=旧449除protocol.version外448原字节不变+1协议版本patch+2依赖；从已接受mlp_projector_voc_20261025_120000冻结439源码逐SHA补齐tools/mlp_projector_voc.py及tools/mlp_projector_voc_core.py，不读取其VOC指标反馈训练。loss、margin/tau/weight、teacher可靠性、数据/划分/采样/模型/seed/数值断言全部不变。独立Slurm依赖probe3804959 COMPLETED0:0/53秒，report SHA b50b7ce13861fe241736db11cd142108aae999d09502670a4cbe55094df0ab98，实际4模块都来自新451快照、8NumPy测试无skip，execution wall48.260908842秒；不加载modeltensor/GT/token、不训练/GMP，分配1V100配额但CUDA隐藏。只证明导入闭包和指标接口，不冒充真实GPU数值/质量通过。probe报告/job/451克隆与旧终态小凭据本地SHA核后才允许新full登记，注册逐字段核科学设置与旧完全相同，再唯一新afterok链。当前主实验仍须新technical14测试/真实梯度/native回放完整C0后才进入三新训练，不放宽原断言。


### 101.2 完整独立接受、三seed结果与最终归档（2026-10-26）

本节最终事实覆盖101/101.1初始等待快照：当前retrieval_score_margin_recover_20261025_230000的14科学child与独立audit3805068已全部COMPLETED0:0。technical3805064 C0/9:17，14目标0fail/error/skip、三seed真实检索score/非零input与A2梯度/冻结基座、C42六step smoke和6份native HCR历史PNGpixel exact通过后才进入正式训练；三seed回放maxabs分别1.341104507e-7/2.086162567e-7/1.266598701e-7，Top1exact，未放宽原2e-6/1e-6界。正式3新模型各50轮/5000step、sampler/classifier init/冻结基座签名通过。独立completion_acceptance_001.json status passed、binding3b0becee9718bef0748cb31ac7742e22dcc4dae42883e012b379ba6f74a714e2，SHA7cfbb6680a4671c1da8b09be719e1f148c65e5cfc820a5dc9ce4ce9d2925c9ee；审核27223唯一文件/1134231502字节、18CM/18逻辑预测组/26082真实新PNG全部SHA commit后才本批一次1449GT。manifest SHA5b6146b2856f6cdadd410b3de52b010229dbfd42ca41ce2f13a1f62555073142、451冻结科学records未改。没有新failure/lock，不再运行任何科学任务、startup validator或手动审核；早期未完成的独立startup通知被最终完整接受覆盖，不补造已退出worker存活。

| Arm | HCR seed42/123/2026 mIoU% | HCR mean±sampleSD% | clean mean±sampleSD% | HCR chair IoU mean±SD% | HCR sofa IoU mean±SD% |
|---|---|---|---|---|---|
| A_supcon（复用） |86.155220/86.239065/86.215781|86.203355±0.043281|85.532678±0.070937|61.563803±1.229596|64.084577±2.114792|
| B_fixed_bank（复用） |86.023192/86.414762/86.191888|86.209947±0.196408|85.545247±0.040143|60.624174±1.475280|64.969616±2.342933|
| C_score_margin（三新训练） |86.901832/86.496918/86.218930|86.539226±0.343411|85.721423±0.194397|64.133090±3.074224|64.980536±0.666753|

HCR C−A mean+0.335871040pp、pairedSD0.377821914，逐seed+0.746611434/+0.257853287/+0.003148399pp；C−B mean+0.329279229pp、pairedSD0.476557604，逐seed+0.878639754/+0.082156356/+0.027041577pp，均3/3正，但seed42贡献最大、2026很小。B−A mean+0.006591811pp、仅1/3正。clean C−A+0.188745076pp（2正/1零）、C−B+0.176175628pp（2/3正）。A是同残差架构CE+.25SupCon控制，不是生产两层E0。历史第99节E0生产检索选模对照HCR均值86.620258316%，当前C相差−0.081031862pp、1/3正；只是当前固定批接受后报告的历史参照，没有新增第4组或用其结果调本loss，不能宣布超过E0、刷新主线或采用新方法。

目标互错HCR C−A平均chair→sofa −21366.333、sofa→chair −22323.333像素，合计−12.628142%；C−B −12121/−36499.667像素，合计−13.855924%。但C−A seed2026的sofa→chair由12366增到24553（+12187），chair→sofa不变；相对B的目标互错减少全部来自seed42，123/2026两方向均不变。C−A chair/sofa平均IoU+2.569287/+0.895959pp，均2/3正；C−B chair+3.508917pp、3/3正，sofa仅+0.010920pp、1/3正。此前大难例2011_000661在本批九模型HCR保存逐图CM中的chair→sofa均仍132003像素，未纠正。平均mIoU有开发集收益不代表跨seed稳定解决椅子/沙发混淆，更不能由图像CM推断具体proposal/owner归因。

HCR错误成本C−A均值BG_FP −160801.333、FG→BG FN −5084.333、wrong_FG −44217像素；C−B分别−163521.333/+17853.667/−56984，C对B的FG FN三个seed均增加。C−A wrong_FG在2026增加12189像素。全21类中C−A car平均IoU−0.519565pp，C−B car−0.615001pp；没有隐去类伤害。clean C−A三seed BG_FP完全相同，HCR才下降，完整门控/栅格排序也参与最终差异，不将全部mIoU收益单归因两类类别翻转；像素子项不相加冒充mIoU贡献。

内部伪标签Top1 mean±sampleSD：A98.780195±0.054545%、B98.837824±0.029991%、C98.760985±0.049908%。C−A mean−0.019210pp（1正/1负/1零），C−B −0.076838pp（三seed负）；高margin E0固定箱damaged均0/0/0，但最低箱C−A mean−1.818182pp、C−B−8.700612pp。chair/sofa内部694query/seed，互错合计A92/B89/C94个query×seed决定，C对B pair signed gap三seed均降低，C对A均值也略降。全类signed均值A.779080/B.803489/C.733725。因此没有证据证明两类整体检索空间被稳定拉远，不能由VOC mIoU正向倒推机制成立。固定margin/tau/weight、teacheragreement可靠性与epoch-lag图库是整体干预，未独立消融各组件。

训练确实遇到这组竞争：从保存50epoch负类计数求和，chair anchor事件7355/7345/7513次中negative=sofa 7201/7205/7350次，sofa anchor7688/7643/7553次中negative=chair7062/7170/6925次；是重复采样事件，不是独立region数。可靠chair anchor1913/1897/1909除以1976fit，sofa693/694/698除以711fit；全类覆盖和错误ID完整保留。teacher/pseudo一致不保证真值可靠，也可能排除有用难例；没有GT质量筛库或观察VOC后改参数。C按内部macro firststrictmax选epoch1/23/5，42为100次更新后的第1轮模型，不是zero-init，也没有VOC选模；所有50轮已完整训练。内部query与VOC不同监督、同holdout多轮选模均为development边界，无独立test/跨数据集/新CI/bootstrap声明。

角色Slurm elapsed：technical9:17；train42/123/2026为20:49/31:58/31:58；predict按A42/A123/A2026/B42/B123/B2026/C42/C123/C2026为10:26/6:31/6:37/11:02/1:51/1:46/2:29/2:56/2:17；evaluate5:44，independent audit6:28。当前full科学allocation北京时间2026-10-25 23:06:03至10-26 00:17:05，4262秒=1:11:02，不叠加并行。旧失败/依赖probe/恢复等待间隔计入的allocation窗22:42:45至00:17:05，5660秒=1:34:20；不把之后本地等待、下载、分析、归档时间冒充科学allocation耗时。3新full共15000step，technical另6smoke step；六source模型新训练step=0，旧成本单列。C训练loop1242.290/1571.377/1575.474秒，optimizer block153.117/179.653/179.188秒，internal eval1022.530/1308.921/1313.834秒，gallery refresh62.611/81.464/81.106秒；宽口径区别清楚，不宣称loss-only加速。当前保存Torch allocated最大79.412109375MiB（technical进程），三C训练各79.379394531MiB，非整卡驱动峰值。CPUevaluate/audit/probe仍分配V100资源配额但CUDA隐藏；MaxRSS为Slurm采样。

161个scientific smallcatalog文件45564015字节，catalog SHAeddc339354979661fd5d44cc0a29b41511e2d89b59aa9394cf7ca5e1aa35dd45；补3个epoch_metrics CSV及资源/来源/旧failure小凭据后177唯一小文件45928629字节全部逐size/SHA归档到experiments/hpc_results/retrieval_score_margin_recover_20261025_230000/verified_20261026。没有下载大checkpoint/raw/token/26082PNG或全50epoch-score树。独立analysis_report.md SHA6f742a9a74e0303ad3fe069d61c0a91f47fd1d4d177efe22c11a89684ce06996，download_verification_003 SHA95adc54c8d634f15f2be734c3f02bd67486f694188cc49005d15bf87a3cbea38，local_archive_proof_004 SHAb5c92cb3165645a3c3e4ac721a346822f07b2c40097648203ded7e7a38f58903。十一analysis文件、下载proof与localarchive proof回当前run并逐远端SHA核；archive_roundtrip_verification_004和manual_sync_003另存。分析仅保存指标/计数加减、单位换算与逐图CM定位，不重GT/训练/检索/推理/统计重采样；报告曾只纠正包含零变化类的排序标题并加保存CM个例检查，旧报告草稿保留，没有SCI结果变动。原224500导入失败32秒/14cancelled0秒、依赖probe3804959 C0/53秒、所有签名/快照/日志历史保持，现无新科学失败。

最终结论：当前固定同架构对照有mIoU开发信号，但目标混淆的跨seed稳定修复及全局空间分离机制不成立，且仍低于历史E0对照均值。本固定批完整接受归档后停止，不自动采用主线、搜索参数/seed/架构、追加Round2或其它研究；一次最终结果通知后只删除retrieval-score-margin-v1监测，不动其它实验。


## 102. MLP前后库过滤位置四组三seed（2026-10-26用户新授权）

独立计划experiments/plans/bank_filter_placement_v1。固定A前10%后0、B前0后10%、C前10%后10%、D前5%后5%，各seed42/123/2026，背景不删；每次仅按当前空间内距本类FP32平均中心的欧氏距离，删除floor(N*ratio)最远成员，保留余下原顺序。前后各10%约累计19%，各5%约9.75%，逐类实际floor人数单独保存。只比较用户这四组，不增加邻居过滤/随机删库/新loss或激活组。

普通GMP与两层1280-512-256 GELU/输入输出L2、CE+.5SupCon、原50×100/AdamW.001/wd.0001/cosine/原图片划分，生产balancedK25/top5/search2/center.2 macro firststrictmax选epoch，新增后过滤不进入选模；原单库/提议/NMS/HCR，R3关。C始终共享A同seed模型。只有恢复的前10%输入逐ID及数组bitwise等于已接受E0，A/C才复用旧模型、B/D六新训练；否则A/B/D由同一恢复源重新训练九个，C共享A。这个分支仅由源身份决定，不看内部或VOC成绩，原历史分数不覆盖。

唯一目标full版本bank_filter_placement_20261026_163000，ROOT=/data/home/zhengyuan/25-nd/1010；run=ROOT/experiments/bank_filter_placement/版本，audit=ROOT/experiments/hpc_launch_audits/版本。full 439已接受VOC源码clone+8patch共447记录已准备并逐SHA放置，尚未full注册/提交。唯一有效注册/提交外层为audit/prepare_remote_002.py。来源是既有主线10582张逐图未过滤GMP缓存，不把34880已过滤库冒充未过滤源。

源检查001 Job3812776 FAILED1:0/2:04，在第118张2007_003778发现旧缓存绑定pseudo与当前已接受canonical pseudo PNG字节不同，原图字节一致；未进入full训练/GT。002 Job3813009 FAILED1:0/5:44，导入原src时cwd不对导致configs/cls_voc21.txt不存在，未进入正式MLP训练/GT。001/002源码、pins/guards/jobs/failures/log全部保留，不重跑旧attempt，不当方法质量失败。

新source003仅修执行cwd和独立输出文件名，Job3813054唯一实际提交；最近只读快照RUNNING4:21/c2v100m2-7，不代表源完成或正式训练启动。003逐SHA核缓存和原build完成凭据；伪标签字节不同时先比像素，像素同保存别名证明，否则只用原canonical pseudo和已有FP32 token按原hard-center/rho.1/双L2 GMP恢复该图原区域。无新SAM3/token生成/标签/proposal，无GT解码。前两图真实GMP回放数值界maxabs<=1e-5/L2<=5e-5/float64cos>=1-1e-8固定，不放宽科学参数。仅source003 reportpassed且SlurmC0、小proof SHA核完才能full唯一注册并提交afterok链prepare→technical→train6或9%3→predict12%3→evaluate→independent audit。JobID/guards存在不重复提交，不手动verify活audit。所有科学只Slurm，一GPU一worker；CPU角色如分配V100配额则隐藏CUDA并如实报告。

13本地NumPy目标测试无skip通过，不代替真实GPU验收。full需旧E0三seed×两图×clean/HCR共12PNG像素精确回归、新训练arm各两epoch×三step smoke/非零有限梯度后technicalC0，才正式训练。4×3×2×1449=34776新PNG/24逻辑组全SHAcommit后才一次1449GT，保存24逐图CM/21IoU/BGFP/FN/wrongFG/chair↔sofa；VOC为看过development，不按其结果调比例/epoch/增加arm。完整独立acceptpassed/exactbinding与auditC0后小catalog SHA下载、独立分析/资源/归档和最终结果补记；无论正平负固定批结束，不自动主线/部署/GitHub/ZIP。

专属pinned native controller55817和本plan/local_control.py/native_controller.py；先读work_state.pending，原请求未CONTROL_DONE不发第二mutation，初始write若带DONE须即时清pending。科学afterok自主推进，正常健康等待静默，新失败/必需行动/最终归档才通知。bank-filter-placement-v1每15分钟仅跟进本批，完整归档一次结果后删除；暂停/取消立即仅停本批及监测，不自动恢复，不动其它实验。后续手册同步保留真实remote旧版本并CAS，冲突先merge保留其它章节。


### 102.1 源独立接受与正式四组链唯一提交（2026-10-26）

本节覆盖102初始等待快照。source003 Job3813054已COMPLETED0:0/1:08:52，batch/extern也C0；10582张、37571个未过滤region、10569BG，35张在canonical pseudo实际像素不同后仅用原标签及已有FP32 token按原GMP恢复。两真实原GMP回放maxabs=L2=0、cos=.9999999999999998，原界未放宽；全类前10%保留34880 ID与历史精确，但向量非bitwise，全部类最大分量差2.8893351554870605e-5、最大L2 4.9505967762668364e-5，这是选择不复用历史模型的依据，不冒充两图回放界。report003 SHA6855e17b5470a21a6a14441912d85866abcb9a736234bf37e476943dc96de7c2、catalog003 SHA830c04a8b6bde10973a5e80fd4d0379bed7cf3300d8957c088db4056df983bf31及job/guard/log已逐远端SHA下载，local_source_acceptance_004 status source_accepted_not_full_completion。

因此按预注册源身份分支A/B/D各三seed，共9新训练，C共享本批新A；旧E0分数不填入新A，历史模型/输入/结果不改。full已唯一注册，447冻结code=439已接受VOC源逐SHA克隆+8原科学patch；manifest SHA3b47387946755e6a2ee2e892258db88911afc835a8c9efeecb0569b56fae1f28，binding02c30d21d5900261d7b1676b1154d0137f06484cf080bcb5c738f25dc58c3a4d，真实注册quota reserve 197076549632字节。未执行outer002因其错误检查历史failure002缺失；保留该外层并在注册前新建prepare_remote_003.py，只修指针到003并添加真实source/localaccept/self SHA gate，8科学patch未改。唯一有效full外层为audit/prepare_remote_003.py，不能再register/submit或上传运行code。

全部6个sbatch实际成功：prepare3814050→technical3814051→train3814052 array0-8%3→predict3814053 array0-11%3→evaluate3814054→independent audit3814055，所有live依赖afterok。24科学child+1独立audit，各1V100配额6CPU24G；prepare6h/technical6h、train/predict48h、evaluate/audit12h，prepare/evaluate/audit隐藏CUDA但仍分配GPU资源配额。实际jobs/launch_complete/submit.guard及catalog已小下载逐SHA，local_registration_verification_004和local_launch_verification_004只证明输入/实际提交，不是完整接受。

2026-10-26T11:28:01Z（北京19:28）真实快照prepare3814050 RUNNING43秒/c2v100m2-3，开始北京时间19:27:18；technical、9train、12predict、evaluate、audit均Dependency PENDING，无新failure/complete，尚无新mIoU。科学链afterok自主运行，不依赖监测轮询；完成仍需13目标无skip/12旧E0像素回归/真实smoke，34776新PNG先全SHA提交再一次1449GT，最终completion_acceptance_001 passed且binding精确、audit3814055 C0后才小结果归档/四组报告。现有guards/JobIDs禁止重提或手动verify活audit，异常先真实日志保留attempt，仅处理本ledger。

专属native controller55817当前请求DONE；本地inspect_full_006.py只读本ledger状态/小日志/complete/failure。原监测bank-filter-placement-v1实际设置PAUSED，本轮保留暂停设置、不自动恢复；没有停止已提交科学链。后续以work_state当前pending/真实ledger优先，不能把本初始RUNNING快照当终态。手册同步只备份真实旧remote并CAS原子追加，保留98–101及别批内容。


### 102.2 正式technical实现失败、针对回归及恢复版唯一提交（2026-10-26）

本节覆盖102.1初始RUNNING快照。原full bank_filter_placement_20261026_163000的prepare3814050 COMPLETED0:0/16:13；technical3814051 FAILED1:0/10:24，于北京19:53:55退出。13目标测试0fail/error/skip已过，但真实smoke首轮保存row时dict(epoch,...,**metrics,...,queries=file)与metrics中的queries整数重名，TypeError: dict() got multiple values for keyword argument 'queries'；未完成整体technical，不算技术接受，更不是方法质量失败。9正式训练全部未开始，没有新VOC预测或GT。只精确取消本ledger 9train+12predict+evaluate+audit共23个PENDING，各CANCELLED0秒，核旧prepare/technical及所有batch/extern终态、squeue无本批活allocation、无lock；old failed_full_terminal_007 SHA7ec9587f603bedd3d2854df641a6c2139178b2fb2d3057724b7d344d6679bc6b及failure/log/cancel小文件已下载逐SHA核，旧447/code/manifest/jobs/guards/所有输出全部保留、不重跑旧版。

独立恢复版bank_filter_placement_recover_20261026_201000，仅epoch记录中把原metrics查询数重命名query_count、queries继续保存文件身份；训练损失/梯度/过滤/数据/采样/选模/检索/断言完全不变。protocol只改version。runtime patch SHA2eab111fbdd4d2c54a7b32471ddc5851d56781136bafd1d6957e338527091919。真实Slurm定向probe3814370 COMPLETED0:0/22秒（batch/externC0），report SHA40c6779d6f38e151cf15775b0e6f096569798fe3cf7083f54946f0107157743b；读取原失败smoke实际6941条saved query、计算原指标并eval实际旧/新row AST，先复现旧重复keyword，再验证全部指标、查询数、查询/已保存checkpoint的SHA凭据和JSON roundtrip完全保留。只对metadata使用synthetic zero loss，占1V100配额但CUDA隐藏，未加载模型tensor/GT/token、不训练/检索/GMP；不把它冒充真实新smoke/质量完成。probe6小proof及catalog已本地逐SHA核，local_probe_acceptance_008只接受接口回归。

恢复版已独立登记：447records=445原字节逐SHA克隆+runtime/protocol两patch，科学字段与失败版逐值一致，source003和所有原anchors保持；manifest SHA05fe1452b5f9391d1ace9860140ec602b0721b79d4995a90fa34ee1fc899f1f1、binding24fd3b473c65b3f32b78d35b12a080de46269d188d19ee9ea0094f318a5ce58d、quota reserve 195922378752字节。唯一prepare_recovery_remote_009.py已执行register和submit，registration/clone3小proof与actual jobs/launch/guard3小proof及catalog全SHA核，local_registration_verification_009和local_launch_verification_009是输入/实际提交证明，不是完成接受。原full保存在state.failed_full_attempt，primary已切恢复版，不能误查旧失败链。首次登记下载输出超过工具上限，仅控制台transport截断，随后按7个小chunk接收同一登记文件并SHA核，不改任何科学artifact。

实际唯一新链prepare3814390→technical3814391→train3814392 array0-8%3→predict3814393 array0-11%3→evaluate3814394→independent audit3814395，全部afterok，24科学child+1audit。资源/时限与原矩阵一致：各1V100配额6CPU24G，prepare/technical6h、train/predict48h、evaluate/audit12h；CPUprepare/evaluate/audit隐藏CUDA但仍分配GPU资源配额。北京时间20:08:49快照prepare3814390 RUNNING13秒/c2v100m2-3，其余Dependency PENDING，无新failure，尚无正式9训练或mIoU完成。四组A前10后0/B前0后10/C前10后10/D前5后5、三seed及9新A/B/D、C共享新A完全保持，不追加任何组/比率/参数/seed/训练配方。

现有JobID/guards禁止重新register/upload科学code/submit/audit或手动verify活audit。完整technical真实smoke/12旧E0像素回归仍须C0，34776真实新PNG全SHA提交之后才一次1449GT；最终completion_acceptance_001 passed/binding精确且audit3814395 C0才smallcatalog逐SHA归档、四组分析和一次最终报告。VOC仅development，固定批无论正平负停止，不改主线。专属controller55817按state.pending串行，旧DONE不空轮询；inspect_recovery_010.py只查新V。原监测PAUSED设置保留未恢复，不影响已提交afterok科学链；不动其它实验。后续manual只备份实际旧remote+CAS原子更新，冲突先merge保留其它章节。


### 102.3 全部独立接受、四组最终结果与归档（2026-10-27）

最终覆盖102–102.2所有pending/RUNNING快照：bank_filter_placement_recover_20261026_201000全部24科学child及independent audit3814395实际COMPLETED0:0，全部batch/extern也C0。technical3814391 C0/10:49，13tests无fail/error/skip、旧E0三seed×两图×clean/HCR共12份PNGpixel exact、A/B/D各真实2epoch×3step smoke通过；9正式MLP各50×100完整完成，C始终共享新A。34776新PNG/24组全SHA提交后才一次1449GT，独立保存CM/CSV算术核通过，completion_acceptance_001 statuspassed/binding精确，SHA4aca59dec2b9e583aaf9c68e3ecc11b89654591ac488ae92e947a6e934e34377，audit3814395 C0/37:09，36369唯一文件/3035523027字节实际校验。manifest SHA05fe1452b5f9391d1ace9860140ec602b0721b79d4995a90fa34ee1fc899f1f1、binding24fd3b473c65b3f32b78d35b12a080de46269d188d19ee9ea0094f318a5ce58d、447冻结records保持不变，不再重训练/预测/GT/审核/源GMP。

| arm | HCR seed42/123/2026 mIoU% | HCR mean±sampleSD% | clean mean±sampleSD% | HCR meanΔA pp / 正向seed |
|---|---|---|---|---|
|A_pre10|86.833484/86.562251/86.465041|86.620258±0.190948|85.781213±0.144475|+0.000000 / 0/3|
|B_post10|86.887088/86.996230/86.526122|86.803146±0.246038|85.914177±0.155192|+0.182888 / 3/3|
|C_pre10_post10|86.714651/86.584001/86.228167|86.508940±0.251778|85.672062±0.131662|-0.111319 / 1/3|
|D_pre5_post5|86.454314/86.673343/86.797858|86.641838±0.173926|85.796175±0.080516|+0.021580 / 2/3|

B只MLP后删10%相对A只前删10%的HCR逐seed差+0.053604/+0.433979/+0.061081pp，均正，mean+0.182888pp；clean mean+0.132964pp、2/3正。C前后各10%与A同seed同权重，只追加后过滤，HCR mean−0.111319pp、1/3正；D前后各5% mean+0.021580pp、2/3正，近似持平。B比较同时改变训练成员与最终删库空间，不能把全部增益单归因投影空间删离群向量；不把正向development结果当独立test或自动采用。没有新增阈值/比例/seed/arm/损失/激活搜索。

HCR chair IoU mean A64.297364/B62.327584/C62.806898/D62.077693%；sofa mean A64.133724/B65.167807/C64.059275/D64.070192%。B−A chair−1.969780pp、sofa+1.034083pp；chair→sofa三个seed均增加、平均+37012像素，sofa→chair均值+3811，两方向合计+40823。整体mIoU提高没有改善目标混淆，不称空间被稳定拉远。B−A BG_FP平均−24050.333、FG→BG FN−134868.667、wrong_FG+57669像素；chair/ship/tvmonitor/dog/train meanIoU分别−1.969780/−1.218089/−0.375450/−0.147697/−0.137739pp，完整21类与逐seed成本已保存，像素子项不相加冒充mIoU贡献。C−A BGFP/FN/wrongFG均值+13339/+25721.333/+28397.333；D−A+10244/−85877/+69371.667。

未过滤37571成员，BG10569始终不删。A进MLP34880、后删0、最终34880；B进MLP37571、后删2691、最终34880；C进MLP34880、后删2421、最终32459；D进MLP36231、后删1275、最终34956。完整逐类floor人数/保留与删除ID/SHA均归档。没有保存逐query fallback调用数，不能捏造其频率变化。源恢复与旧34880ID相同但非bitwise触发九新训练分支；接受后只读本地保存CM发现新A六组与历史E0逐图CM及1449IDs完全一致，事后结果一致不等于复用旧模型或输入逐位相同。common-reference hold独立保存，不混合不同training population/hold人数的准确率。

角色Slurm elapsed（数组按index顺序）：prepare: 0:08:47；technical: 0:10:49；train: 0:20:30, 0:32:27, 0:38:03, 0:33:41, 0:24:14, 0:24:46, 0:45:29, 0:21:27, 0:21:37；predict: 0:27:47, 0:28:54, 0:33:49, 0:07:22, 0:35:06, 0:30:52, 0:08:18, 0:10:11, 0:08:03, 0:10:26, 0:37:10, 0:34:12；evaluate: 0:27:17；audit: 0:37:09。当前full UTC2026-10-26T12:08:36至2026-10-26T16:53:28，即北京2026-10-26 20:08:36至2026-10-27 00:53:28，整体17092秒=4:44:52，不叠加并行、不把下载/本地等待/归档时间计为allocation。旧source001/002失败2:04/5:44、source003 C0/1:08:52、原full prepare16:13/technical失败10:24及metadata probe22秒单列保留。新9 full45000step，technical另18smoke step，C新训练step0。24保存execution最大Torch allocated 46.913085938MiB，是Torch分配器而非整卡驱动峰值；CPUprepare/evaluate/audit分配V100配额但CUDA隐藏，MaxRSS为Slurm采样。训练optimizer/评估含保存/导出和选中epoch、UUID见analysis_numbers_013与training_cost CSV，不编造未保存的纯forward/逐region时间。

112个科学smallcatalog artifact75344535字节全部逐size/SHA核到experiments/hpc_results/本版本/verified_20261027，catalogSHAc0a7a6970ba5f37928283bc45d7d2296f953c5de02a5cf0a88cc7def296fff7e；未下载大checkpoint/raw/token/34776PNG。追加metadata/derived analysis等后local_archive_proof014列123文件75629376字节（自proof及之后roundtrip/manualproof不在该计数内），canonical listing SHA6864e3b973bf286b7f84854376980778dc8b9501457c3c33e5768b031b5d2728。独立analysis_report SHA92ec4531baa0383e9a102c5de0a98652cad686d13190ffde783f3d7f36bc0f4a，download_verification011 SHA2a4438571ba970b9fc21e7beb7efaf99d4f9fdaffab9dd24ec4e3022932fbc33，local_archive_proof014 SHA6289248cf401b766405d4489f3e0d70eec0a7fce4648581959e81d14b107d3f5；六analysis文件+downloadproof+archiveproof八文件已回新run并逐远端SHA核，analysis_upload_verification014 SHA2d6bce0a094d347f1448c7f0af319ec1578af5792619e7650d0826409ae4c85d。只做保存数字/CM/计数加减和SHA，不重GT/训练/推理/检索/重采样。原source/full所有失败、guards、冻结code和历史指标保持。

最终结论：B只后过滤10%在本固定VOC development组整体mIoU最好，但chair/sofa互错代价更大；C重复过滤不支持，D近持平。固定批到此停止，不自动改主线/部署/GitHub/ZIP或追加研究。正常startup轮询未及时覆盖的已完成worker不补造存活；完整独立终态接受覆盖startup快照。完成小归档与最终结果报告后删除仅bank-filter-placement-v1监测（原PAUSED），不影响其它任务。手册同步备份真实旧remote并CAS原子追加，保留其它章节。


## 103. B组固定三模型的投影后删除比例开发扫描（2026-10-27用户新授权）

用户在102完整接受后明确要求“现在做B组的调参实验，看删多少更好”。独立experiments/plans/bank_postfilter_sweep_v1，固定后删0/5/10/15/20/25/30%，三seed42/123/2026，MLP前0%、背景10569不删。只复用B_post10三个已选择best checkpoint及37571未后过滤成员的L2投影导出，不重训练或VOC重选epoch；10%复用102已接受六份clean/HCR CM及metrics，其余六档×三seed18worker重新预测。不是重新训练7种MLP，也不把原A/C/D混为B不同删除率。

只改变投影空间每前景类距FP32均值中心最远floor(N*percent/100)的删除数，原np.argsort[::-1]/保留原序，重新建立保留库中心和索引。固定原普通GMP查询、两层GELU、center.2/K25/top5/search2/margin0/mixture.2/原NMS/HCR，R3关。没有新encoder/SAM3/token/GMP/pseudo/proposal/训练/loss/激活/seed/主线或部署。按三seed mean HCR mIoU最大进行本开发网格选择，完全同分删最少；所有七档/逐seedmean/sampleSD/paired10%/clean/21类和chair↔sofa双向错误成本全部报告，不事后加gate或细扫。已看VOC development调参，有选择偏差，不是独立test或全局最优证明，不自动采用主线。

来源bank_filter_placement_recover_20261026_201000 manifest05fe1452b5f9391d1ace9860140ec602b0721b79d4995a90fa34ee1fc899f1f1，binding24fd3b473c65b3f32b78d35b12a080de46269d188d19ee9ea0094f318a5ce58d，accept4aca59dec2b9e583aaf9c68e3ecc11b89654591ac488ae92e947a6e934e34377。102旧所有冻结源码/模型/指标原字节保持且旧monitor已删除，不重启该批。

唯一新版本bank_postfilter_sweep_20261027_020000；ROOT=/data/home/zhengyuan/25-nd/1010，run=ROOT/experiments/bank_postfilter_sweep/版本，audit=ROOT/experiments/hpc_launch_audits/版本，code=audit/code已冻结454records=447已接受源逐SHAclone+7新文件。manifest SHA3e5a00831346c82fb1ca3882d4a98b8f49722baf7a41610a3d0cc4a67df0c0c9，bindinga89a7561516758cf4562a638402336238af28cd929ef4208438b6ccf9ba829c8，真实quota reserve 190903205888字节。source_expected_002/put_manifest_002是实际pin，001上传前草案保留；上传前scope/source labels/background trace断言review加强未触发任何科学计算或JobID。唯一prepare_remote_001已完成register/submit；登记3proof及actualjobs/launch/guard3proof/catalog均逐SHA下载，local_registration_verification_001/local_launch_verification_001仅输入/实际提交证明，不当完成质量。

真实全afterok链technical3816168(6h)→predict3816169 array0-17%3(48h)→evaluate3816170(12h)→independent audit3816171(12h)，20科学child+1audit，各1V100配额6CPU24G；CPUeval/audit CUDA隐藏但仍占V100资源，不能称零GPU分配。2026-10-26T18:07:46Z（北京27日02:07:46）初始快照technical PENDING、18predict/eval/audit等待依赖，无新failure；没有真实GPU技术完成或新mIoU。已有JobID/guards绝不能重复layout/register/upload运行科学code/submit/audit或手动verify活audit。source旧audit C0 gate已实际履行，未使用已完成旧JobID作为易过期afterok根依赖。

14本地NumPy核心测试0fail/error/skip已过，不代替GPU。Slurmtechnical须14无skip、三Bcheckpoint1024原缓存输入导出回放atol=rtol1e-6、10%三seed×两图×clean/HCR共12历史PNGpixel exact及全部7档共84pilot合法后整体C0，才18预测。18×2×1449=52164真实新PNG/36组全SHAcommit后才本批一次1449GT；10%旧6CM按SHA复用，合计42逻辑逐图CM，全部身份/CM/CSV/均值SD/选择算术由独立audit只读核，不读rawGT/modeltensor/token或重推理/检索/训练/重采样。新completion_acceptance_001 passed/exactbinding且audit3816171实际C0才小catalog逐SHA归档、独立analysis/资源/回传proof和最终手册补记。不下载大checkpoint/raw/token/52164PNG。

专属pinned native controller77663、plan/native_controller.py/local_control.py/inspect_remote.py，先work_state.pending，未CONTROL_DONE只空读同一原请求，初始write已DONE须及时清pending，不轮询旧DONE或重发mutation；失联先actualledger/guards再重建，不把GPFS/SSH分钟等待/D当失败。私钥仅本地不打印上传，固定ED25519/BatchMode/StrictHostKeyChecking，不密码。科学异常保留全部attempt，只取消本ledger实际PENDING依赖；修科学错误新version+针对probe，不改运行454/签名/断言。用户暂停取消立即只停本批和监测，不自动恢复、不动其它实验/个人queue。新bank-postfilter-sweep-v1监测只跟进本七档，普通健康进度静默，仅必要新失败/行动/最终结果；完成后删除仅新monitor。固定7档到此停止，不自动第二轮搜索或改变主线。


### 103.1 七档全部独立接受、开发最优及最终归档（2026-10-27）

最终事实覆盖103全部初始PENDING/Resources快照：当前bank_postfilter_sweep_20261027_020000的20科学child及independent audit3816171均实际COMPLETED0:0，batch/extern同C0，无本批科学失败。technical3816168 C0/8:39，14tests无fail/error/skip、三B模型各1024源cache输入导出回放atol=rtol1e-6、10%12历史PNGpixel exact、全7档84pilot合法通过后才18worker预测。52164真实新PNG/36组全SHA commit之后才本批一次1449GT，来源10%六CM按SHA复用，保存合计42逻辑逐图CM及882逐类CSV。completion_acceptance_001 statuspassed/bindingexact，SHA429e20ec08c51f43113b4a3f8d7b741aa9818b89503fbe4f8717da6c43e8b6ce，audit3816171 C0/9:37，52735唯一文件/574367206字节实际校验；manifest SHA3e5a00831346c82fb1ca3882d4a98b8f49722baf7a41610a3d0cc4a67df0c0c9、bindinga89a7561516758cf4562a638402336238af28cd929ef4208438b6ccf9ba829c8、454冻结records未改。无新训练，源B三selected checkpoint/epoch及37571导出固定，无新SAM3/token/GMP/pseudo/proposal。

| 后删% | HCR seed42/123/2026 mIoU% | HCR mean±sampleSD% | meanΔ10% pp / 正向seed | clean mean±sampleSD% |
|---|---|---|---|---|
|0|86.852514/86.968823/86.251439|86.690925±0.385023|-0.112221 / 0/3|85.842763±0.351973|
|5|86.879821/86.983199/86.676811|86.846610±0.155871|+0.043464 / 1/3|85.963786±0.059378|
|10|86.887088/86.996230/86.526122|86.803146±0.246038|+0.000000 / 0/3|85.914177±0.155192|
|15|86.947303/86.849856/86.485250|86.760803±0.243559|-0.042343 / 1/3|85.876849±0.198847|
|20|86.883044/86.872135/86.519568|86.758249±0.206776|-0.044897 / 0/3|85.857966±0.148006|
|25|86.941091/86.811211/86.493846|86.748716±0.230079|-0.054430 / 1/3|85.827793±0.161871|
|30|86.930684/86.811931/86.370414|86.704343±0.295224|-0.098803 / 1/3|85.811449±0.171059|

冻结开发规则选择5%，HCR mean86.846610%，比10%仅+0.043463887pp；逐seed−0.007266389/−0.013030772/+0.150688823pp，paired sampleSD0.092904236pp，仅1/3正，均值优势由2026带来。各seed自身最佳42→15%、123→10%、2026→5%，不能称稳定通用最佳。5%和10%各自相对0%的HCR三seed均正；15%–30%均值都低于10%，不支持越删越好。clean的5%均值85.963786%也最高，但不改变预先选HCR规则。仅当前7档/固定模型/看过VOC development的调参最优，有选择偏差，不是独立test或全局最优；sampleSD非CI，没有新bootstrap/显著性gate。

5%−10%的chair IoU mean+0.192271894pp（三seed均正），sofa−0.370555774pp（1/3正）；chair→sofa均值−5391.333像素，sofa→chair完全不变，双向互错共−5391.333。5%−10% BG_FP mean+12934.333、FG→BG FN+8029.000、wrong_FG-18220.333像素。不能将目标互错轻微减少称空间分离机制成立或解决椅子沙发混淆，完整7档/每seed/21类伤害及相对0%成本全部保存；像素子项不相加冒充mIoU贡献，不从CM推断owner/proposal归因。

每前景类按原FP32 mean欧氏最远floor删除、保留原顺序，BG10569全程不删；0%删0留37571、5%删1340留36231、10%删2691留34880、15%删4042留33529、20%删5393留32178、25%删6742留30829、30%删8090留29481，原始37571含27002FG。百分比不是按整个含BG库直接乘；逐类floor人数/删除与保留ID已归档，三seed各比例保留集合静态核验嵌套。支持下界已保存，未保存逐query fallback次数不编造其频率。10%来源六NPZ的本地保存字节与102源CM逐SHA精确，不能把它说成六新预测组或重复读取GT。无其它模型/训练数据干预，已接受102及旧失败历史全不动。

角色Slurm elapsed按数组index顺序：technical: 0:08:39；predict: 0:25:15, 0:14:38, 0:29:38, 0:29:03, 0:13:00, 0:15:12, 0:27:18, 0:23:59, 0:15:51, 0:09:16, 0:25:44, 0:23:52, 0:11:32, 0:14:16, 0:24:42, 0:23:39, 0:08:30, 0:10:03；evaluate: 0:21:13；audit: 0:09:37。科学UTC2026-10-26T18:32:15至2026-10-26T21:29:19，北京时间2026-10-27 02:32:15至2026-10-27 05:29:19，整体10624秒=2:57:04，不叠加并行；technical Submit至Start排队1497秒另列，不计allocation窗，本地查询/下载/分析/归档等待也不计。保存20角色execution最大Torch allocated 25.502929688MiB，为Torch分配器而非整卡驱动峰值；CPUevaluate/audit分配V100配额但CUDA隐藏，MaxRSS为Slurm采样，不能称零GPU资源。新正式训练step0，无forward独立benchmark，不把整个wall冒充纯推理耗时。

109个科学smallcatalog artifact104692344字节全部逐size/SHA核到experiments/hpc_results/本版本/verified_20261027，catalogSHA432a2c476ebb342f891d5c340b57b63339c7fca74876e16596fd0cb20cbe0492，不下载大checkpoint/raw/token/52164PNG树。追加metadata/6analysis等后local_archive_proof009列121文件105072559字节（不含自proof和之后roundtrip/manualproof），canonical listing SHA96d0c1babf438263872e2f64e5822a2caf3e9cafe2419ba9c65d261061386d16。独立analysis_report SHA47c1ad4c03b98024be6f8c46bc9235d87e16d7346b1c1c4f6172cac3d7ce21e9、download_verification006 SHA85084b695c588538592d5c0d47ac1f1bbab3eeae2da22a6791e5611c95f0da04、local_archive_proof009 SHAe40c44d215b7f605d6445063e8d22b05eb67b9916c3c765a1ca62f8aaac299d8；六analysis文件+downloadproof+archiveproof八文件已回新run并逐远端SHA核，archive_roundtrip_verification009 SHA6a0db4982f4027b0b4e0cb3537110daea1f14a313bd297e26b7ff08e28979b09，冻结science保持。只读取保存CM/数字/IDs和SHA，不重GT/训练/检索/推理/重采样。

最终结论：当前固定网格5%平均最好，但相比10%仅1/3seed正，三seed最优不同，证据支持温和5%–10%而非稳定单点最优。保留类代价与development偏差；本固定批到此停止，不自动细扫2.5/7.5或增加seed/其它研究、不改主线或部署。已完成worker不补造startup存活，完整独立接受覆盖初始排队快照。最终完整归档与结果通知后删除仅bank-postfilter-sweep-v1监测，不恢复102或干扰其它任务。手册同步先备份真实旧remote并CAS原子追加，冲突先merge保留其它章节。


## 104. MLP后过滤5% + single-view Loose E 三seed固定组合（2026-10-27用户新授权）

用户明确要求“那在MLP删除5%的基础上加E来跑一遍，看能不能冲击记录”。独立计划experiments/plans/r3_postfilter5_e_v1，版本r3_postfilter5_e_20261027_154000，run=/data/home/zhengyuan/25-nd/1010/experiments/r3_postfilter5_e/r3_postfilter5_e_20261027_154000，audit=/data/home/zhengyuan/25-nd/1010/experiments/hpc_launch_audits/r3_postfilter5_e_20261027_154000，冻结code=audit/code465records=103已接受454逐SHAclone+7新组合/测试/计划文件+4原E已接受legacy字节。manifestSHA713369531e312a6152331ea6bdfc7e918265ed1b20dcc5b1fcf260ee88bbd884，binding61ff7e4285764df0bd19d3be54706ea9f51bebb7f6c447170dac75b92b4f0d39。source_expected_002为实际登记pin，001/prepare_remote001/首次登记前异常全部保留。

原B三个selected checkpoint/epoch与37571未后删导出保持；MLP前0%，后每FG类按原FP32中心欧氏最远floor(N*.05)删，BG10569全留，库36231。复用103已接受p05 HCR三组PNG/CM；不重训、不VOC重选epoch、不搜索比例。R3原固定80%中心裁剪候选/映射mask/原图1280D hardcenter GMP按旧accepted SHA复用，在新模型/同seed5%库重新投影检索，不能复用旧R3模型的类分数或接受集合。新SAM3/token/GMP/proposal/训练均0；部署仍需中心裁剪编码，不能把本cache组合成本冒充部署成本。

E完全保持旧规则：先q>=.4/r>=.95/m>=.1/novel>=.1严格R3，再q>=.4/r>=.90/m>=.05/novel>=.1单视角loose追加；均只补实时背景，FP32(.8r+.2q)排序/ID平局，保存FP32转Pythonfloat比较无epsilon，不用crop语义分数/类别专用规则。主结果固定完整E，strict仅实现所需中间阶段，不择优换primary。三seed42/123/2026×两阶段×1449=8694新PNG，另3基线复用，共9逻辑CM。8694预测/owner/trace/21classscore全SHA提交后才本批一次1449GT。保存21IoU/FPFNwrongFG/chair↔sofa，三seed逐值/mean/sampleSD/配对方向；已看VOC development及5%先前调参边界保留，不是独立test，不自动推广主线。

首upload13文件逐SHA成功，登记前发现根目录configs/recipes/r3_record_v1.json不存在，在register.guard/code/run/manifest/JobID产生前中止；不是科学质量失败。新outer002仅将本地已发布recipe SHA6089f777caec8d4d364ae04b1df88fac367833de07014f127f99044e2398f07b作为只读凭据放audit/record_references，科学11patch字节完全不变，保留001异常，未修改集群根configs或任何历史源码。

唯一已实际提交链technical3819711(6h)→predict3819712 array0-2%3(48h)→evaluate3819713(12h)→independent audit3819714(12h)，全afterok，5科学child+1独立audit，各1V100配额6CPU24G。CPUevaluate/audit CUDA隐藏但仍占V100资源；真实quota reserve190618402816bytes。登记/实际launch六小proof本地核binding/465字节/资源/边，只是输入和提交证明，不当技术完成/最终mIoU；已有JobID/guards禁止重复登记/上传科学code/提交/手动verify活audit。

本地47tests无fail/error/skip，仍须Slurm技术47无skip、3模型1024真实导出回放1e-6、6p05像素精确、6旧strict/E像素及旧类分数回归、6新组合合法后technicalC0才正式3worker。三worker真实>=25图或C0+execution后只读outerstartup存证，已完成worker不造存活。最终须completion_acceptance_001 passed/exactbinding且audit C0再小catalog逐SHA下载、独立analysis/成本/归档回传、手册最终补记；不能拿summary/launchproof代替独立接受。独立audit只字节/保存owner与gate/CM/CSV算术，无GT/modeltensor/检索/推理/GMP/bootstrap。

报告固定比较新E对p05、strict、旧E及已发布严格R3；旧系列SMC-H均值86.945033%与RGR-G seed42 87.196738%分开作历史参考，不能单seed与mean混用或把两边历史提升相加。正平负均在本固定批停止，不追加阈值/seed/arm/新loss或训练，不部署/GitHub/ZIP，不动其它实验/个人queue。异常留attempt，仅停本ledger未启动依赖；用户暂停取消立即只安全停本批及监测不自动恢复。正常健康推进静默，仅必要失败/行动/最终独立归档报告；结束删除本新monitor。


## 105. 小掩码门槛与HCR救援空档固定三seed对照（2026-10-27用户授权）

用户针对配置/顺序审查，明确要求先做“小掩码＋HCR空档”实验。独立计划experiments/plans/small_mask_hcr_v1，唯一版本small_mask_hcr_20261027_200000，run=/data/home/zhengyuan/25-nd/1010/experiments/small_mask_hcr/small_mask_hcr_20261027_200000，audit=/data/home/zhengyuan/25-nd/1010/experiments/hpc_launch_audits/small_mask_hcr_20261027_200000。冻结472records=source104已接受465原字节clone+7新科学/测试/计划文件；manifest SHA7311538b46f35c40ff79641f57e0eae8f8db4a29c0f91d3302049b92127ddff3，bindingdeac45aa31d178600e55c58c54d56db1fc288f7199fea11bb61ec52363a50200。source_expected_002/prepare_remote_002为最终pin，001是未登记前草案；002仅加强独立审核的实际source mask字节/hash/owner几何检查，无科学参数改变，所有草案保留。

source104独立accept e21467f677a3b619b13381be353e1fc22c49421fda39d48e9e98518a06221177已passed/exactbinding，audit3819714及batch/extern实际C0/51:38，小accept/catalog已本地逐SHA核。source模型/库/权重/候选/指标原字节保持。本批沿用同seed B selected checkpoint、MLP前0/后删5%每FG类floor/背景10569不删的36231单库、native1280D query与原R3中心crop几何/已保存同库score，不重训练/选epoch或生成SAM3/token/GMP/pseudo/proposal。

固定42/123/2026及三组：A_current完整原classified面积300→HCR→strictR3→LooseE；B_gap_hcr在完整原HCR后，仅将原clean且fullmask100–299像素候选按原q>=.4/r>=.95/m>=.1/liveBG>=.1门槛顺序补背景，保护原HCR全部前景，然后原strict/E；C_area100只将classified NMS面积降100，HCR和strict/E其余不变。B不是恢复所有NMS拒绝候选；C可能改变原clean归属，不声称仅补原A背景。固定主比较B−A的最终E，C是直接放宽对照，全部报告不择优改primary。

实际唯一全afterok链technical3822329(6h)→predict3822330array0-2%3(48h)→evaluate3822331(12h)→independent audit3822332(12h)，5科学child+1audit，各1V100配额6CPU24G；CPUevaluate/audit CUDA隐藏但仍占GPU配额。quota实际reserve190589665280bytes。登记3proof及actualjobs/launch/guard3proof本地核SHA/binding/465clone+7patch/资源/边，仅launch输入与实际提交凭据，不能称技术完成或mIoU。

本地21NumPy目标测试无fail/error/skip，包括抽取原生产函数体的A/C合成像素回归；不代替真实GPU。technical须21无skip、三模型1024导出回放1e-6、每seed两图A HCR/E像素exact/C原生产minarea100像素exact/B保护通过且C0，才既有三worker。每seed同次query投影/检索供A/B/C；A1449×3全HCR/E逐像素与source一致才有效。B/C各HCR/E共17388新PNG/12组，另6组A按SHA复用，合计18逐图CM；所有新预测/owner/trace全SHA提交后才一次1449GT，并再次核A每图CM精确。保存21IoU/BGFP/FN/wrongFG/chair↔sofa、三seedmean/sampleSD/配对方向、小掩码分母/eligible/accepted/新增像素和R3/E owner变化。

已有JobID/guard禁止重登记/重提/改冻结code或手动verify活audit。独立audit只字节/候选门槛/owner与保存CM/CSV算术，不读GT/modeltensor/token、不重检索/推理/GMP/训练/bootstrap；audit_progress_001保存真实文件/字节进展。completion_acceptance_001 passed/exactbinding且auditC0后才smallcatalog逐SHA小归档、独立analysis/资源/回传proof与最终结果。当前是已提交，尚无本批真实精度完成。

科学失败保留全部attempt，只安全取消本ledger实际PENDING；修复须新version+针对probe，不放宽断言。用户暂停取消立即只停本批和监测、不自动恢复、不动source104/其它batch/个人queue。VOC已看development且source5%经其调参选择，不是独立test。固定三组升平降都结束，不加面积/阈值/seed/arm/loss/结构搜索，不自动主线/部署/GitHub/ZIP；最终模型入口统一作为另一步集成验收，不与本精度对照混合。后续手册更新备份实际旧remote+CAS，冲突merge保留104其它新增。


## 106. 同训练轨迹KNN-5与production balanced选epoch三seed对照（2026-10-28用户授权）

用户针对旧继承配置审查明确“按照你的想法做实验吧”，固定开展两个独立任务。此项计划experiments/plans/checkpoint_selection_v1，版本checkpoint_selection_20261028_010000，run=ROOT/experiments/checkpoint_selection/版本，audit=ROOT/experiments/hpc_launch_audits/版本，ROOT=/data/home/zhengyuan/25-nd/1010。冻结472records=104已接受465逐SHAclone+7新文件，manifest SHA49133cc324c2f8de4e11f6fe09a61cd58c3d72efa9b07f51ada81411f6b30fe8，binding60389686bd9bd872cb12a5fcf66b0c06ec0a6111d0550d9c312481bfc5559a7f。登记3与实际launch3小凭据已下载逐SHA/clone/资源/afterok独立核；只是输入和实际提交，不是技术完成或新mIoU。

固定当前B三seed42/123/2026、37571未前删成员（10569BG）、30095fit/7476hold、原图划分与50个epoch/checkpoint。不是旧E0的34880/6941。B balanced选43/40/35，本批从同150checkpoint按历史KNN5内部macro firststrictmax选epoch；production balanced仍K25/top5/search2/center.2内部macro firststrictmax，均在后过滤前选模。稳定gallery行序解决近邻同分，投票按count、similaritysum、低数字类；argpartition仅定位阈值，边界ties完全恢复原stable顺序。本地12项无skip过，真实GPU technical还须源150checkpointSHA、三1024导出回放、选中epoch与best投影exact、真实全30095fit/256query原KNN指标exact、六图HCR/E像素回放，不把本地测试当GPU接受。

不重训；KNN所选模型重新投影37571并按原FP32类中心距离删各FG floor5%、BG不删、库36231。native与原80%center crop的既有1280D descriptor须用新epoch模型/匹配库重投影检索，不复用另一模型的类分数。面积300/NMS.4/HCR/strictR3/LooseE全不变，不混入105小掩码或额外NMS组。三seed×HCR/E共8694新PNG/6组全SHA提交后一次1449GT；balanced6组CM/PNG按签名复用，总12CM。主比较knn5-balanced完整E，HCR辅助；三seed逐值/mean/sampleSD/配对方向、21IoU/FPFNwrongFG/chair↔sofa双向，VOC为已看development和先前5%调参边界，不按VOC重选epoch/阈值或bootstrap/主线推广。

实际唯一链technical3823425(6h)→select3823426array0-2%3(24h)→predict3823427array0-2%3(48h)→evaluate3823428(12h)→independent audit3823429(12h)，8科学child+1audit，全afterok，各1V100配额6CPU24G；CPUevaluate/audit隐藏CUDA仍占GPU配额。已有JobID/guards禁止重register/upload科学code/submit/audit或手动verify活audit。首次2026-10-27T16:22:27Z真实快照technicalPENDING Resources，其余Dependency，不能声称已实际GPU测试/新精度。

仅completion_acceptance_001 passed/exactbinding且auditC0后smallcatalog逐SHA小下载/独立analysis/资源/回传归档。审核只保存字节/IDs/50epoch指标/首最大值/owner门槛/CM与CSV算术，不GT/modeltensor解码/推理/检索/GMP/训练。失败保留attempt，仅取消本ledger实际PENDING；科学修复新version+真实小probe，不能改运行snapshot/assert。用户随后明确“不要设置自动化任务，跑完我告诉你”：本两项没有创建、也不得创建自动化/监测；Slurm依赖自主跑完，等用户回来再手动验收收尾。专属pinned native local_control.py按state.pending串行，nativeexit0即DONE；不动其它batch/个人queue。手册同步backup真实旧remote+CAS，冲突merge保留其它章节。

## 107. 训练伪标签跨类别先到先得重叠来源诊断（2026-10-28用户授权）

独立计划experiments/plans/pseudo_overlap_diagnostic_v1，版本pseudo_overlap_diagnostic_20261028_013000，run=ROOT/experiments/pseudo_overlap_diagnostic/版本，audit同规则。冻结470records=104来源465原字节+5新文件，manifest SHA63b313155910e010e14417cdb88e72ea5f7df234aa2d86a43e8337d66762bf64，bindingbd0ec73e2fd853750e603104d54d5526277d3a0b58504e6a8261214471e2ffe3。旧生成阶段complete SHA f7cd5d13980e3d10ecc9f5fcf73ea5456e5872d6848dd05d0dae904a7e78439a锚定10582逐图prompt-choice报告及raw语义PNG；保持原per-class-union概率.4/close3/原迭代/膨胀1后按原mapped_labels顺序合成。原raw/instances不存在、save_instances未启用，报告不含完整实例score或mask，不能从最终PNG伪造被覆盖竞争类与confidence。

诊断只读这些真实历史报告与raw伪标签，不读GT、不生成SAM3/token/GMP/标签/新库、不训练。对每类blocked=after_repair_union-composite_assigned，decode原PNG核各类assigned计数。合计是class-pixel membership excess，不能冒充distinct重叠像素；k个正面积类别时报告ceil(excess/(k-1))至min(excess,union)边界。仅两正面积类才能精确归因later被earlier覆盖方向，>=3不给伪造pair矩阵。完整报告类/顺序负担与chair/sofa共现范围。历史raw在几何清理前，且不等于后来35canonical-source修复图，不能把本计数说成当前B训练库的精确污染量或mIoU损失。缺真实score/mask意味着本批没有confidence-wins质量对照；改compose_instances(score)同时改变per-instance修复，是混杂因素，不能直接开关冒充单因素。

实际唯一diagnose3823435→independent audit3823436，afterok，各6CPU24G12h/1V100配额、CUDA隐藏，GPUcompute0但非零资源分配。本地7项算术测试无skip通过，科学diagnose须重跑7无skip并核10582报告/PNG字节/类直方图。独立audit实际重读来源字节/JSON/保存CSV算术，不模型/GT/重raster。仅acceptpassed/exactbinding与auditC0后smallcatalog小下载、独立分析/资源/归档，正平负均停止，不自动重建标签/训练/VOC或其它研究。

登记与actuallaunch六小proof已本地独立SHA/470clone/资源/依赖核，不能当最终诊断结果。专属controller与106分开、只本ledger，失败留历史，只安全处理自己实际PENDING依赖。用户明确不设自动化，未创建；等用户告知跑完再手动验收和解释证据。两批不改主线/部署/GitHub/ZIP、不消费其它监测连接。后续新增最终结果也须备份真实remote手册并CAS，保留98–106及其它更新。


### 106.1 独立接受、最终结果与小归档（2026-10-28）

所有8科学child及audit3823429实际COMPLETED0:0，batch/extern也C0；accept SHA9b5029b43fdef6ab931ff6cd2d51333aaf67b57f3de29ac3db63e8d2ea96cf1e、manifest49133cc324c2f8de4e11f6fe09a61cd58c3d72efa9b07f51ada81411f6b30fe8/binding60389686bd9bd872cb12a5fcf66b0c06ec0a6111d0550d9c312481bfc5559a7f精确。balanced选43/40/35；KNN选40/40/35，没有新训练。E balanced均值87.095121701±sampleSD.077292526%，KNN87.071293118±.041538921%；KNN-balanced逐seed−.071485749/0/0pp，mean−.023828583pp。HCR mean差−.000796129pp，保留balanced，不把低SD称为更优。全21类/双向椅子沙发/错误成本保存在metrics/per_class及独立analysis；开发集边界，不追加研究/部署/主线改变。194小catalog文件29282598字节逐SHA下载，12CM与metrics矩阵核对；catalogSHA701e3cbb321f0a2e801d54a2a487a5b018a2d894ec0010e92bcdec18b1cd2381。整体科学allocation窗2026-10-27T16:31:51至2026-10-27T18:44:04 TZ=UTC，7933秒，不叠加并行，各角色耗时和实际Torchallocated范围[13.1435546875, 32.5087890625]MiB在cost_summary_014；非整卡峰值，CPU仍占V100配额。完整analysis/downloadproof/localarchiveproof回远端逐SHA。按用户要求无自动化；固定批至此结束。

analysisSHA9af442691c606bbe5f053edebff1b6640208ab778056657d121fbd0249106e72，localarchiveproofSHAcc3df8c36f9f835b9dfec0bf11e0fa03e938995005eb1e06c957c7d546b03dcf，listingSHA1ebc6d158d7b0e2eb887cde06f04f0b4bd462c794dba40df6180a1534813dcbc。


### 107.1 独立接受、重叠诊断结果与小归档（2026-10-28）

diagnose3823435和audit3823436及全部batch/extern均COMPLETED0:0；acceptSHA9014b82fda1647dbff17c9274580f722342cc85ebce56f297c801acd7582d597，manifest63b313155910e010e14417cdb88e72ea5f7df234aa2d86a43e8337d66762bf64/bindingbd0ec73e2fd853750e603104d54d5526277d3a0b58504e6a8261214471e2ffe3精确。10582图中3155（29.814780%）有跨类重叠；累计blocked4679689是class-pixel membership excess，占precomposite类面积和.783832%，非独立像素数；独立重叠像素范围3637242–4679689。sofa候选面积被挡6.203602%/1471078，chair.174619%/47823；可精确归因的双正面积类chair/sofa冲突30图、394004像素归chair。不能断言真实应归sofa、不能由本数据证明confidence更好，也不能从>=3类scalar报告捏造pair分配。raw在几何清理前、35canonical修复边界保留；无GT/新标签/模型/训练。10小catalog文件13413006字节逐SHA核，catalogSHAe0f52c7a500c594528417f30b6daae509133f810e8e99084f5b027785b52bcf9。整体TZ=UTC allocation窗2026-10-27T16:58:10至2026-10-27T18:16:29，4699秒，各角色见cost_summary_014，不叠加并行；CPUcompute但占V100配额，GPU峰值未测。分析及download/localarchiveproof回传核SHA，固定诊断停止，无自动化/自动重标注或新训练。

analysisSHA7883f6dff8062913c6af92b81851fb73e0791333a0276bf37c9349f8f5eb051b，localarchiveproofSHAd75a15ef718e8caa53cfc083f43c32104a7fb16204e7a84f31c6dc2afe313a00，listingSHA148f0299bbdb94a0ad229be4de4cc702e1159bbd46e7b6c4fc6f1ab36b28b01f。

连接记录：2026-10-28联通SSH1014已恢复，exec与交互式shell均可用，实际sacct及验收/归档已读取。客服说明SFTP应使用1018；原1014的SFTP超时不再当异常证据。1018的ED25519指纹为SHA256:/5kA6tA42PbLNUgg5CQXOdb5zF8H7eZe6AR5FeAnwUw，和既有1014不同，未自动信任；本次通过原指纹校验的SSH1014完成传输，不断言此前服务异常根因。


## 108. balanced最终候选之前的训练伪标签重叠归属固定对照（2026-10-28用户授权）

用户明确准备以balanced作为最终模型，要求先实测训练伪标签跨类重叠问题。本批计划experiments/plans/pseudo_overlap_owner_v1，唯一版本pseudo_overlap_owner_20261028_150000，ROOT=/data/home/zhengyuan/25-nd/1010，run=ROOT/experiments/pseudo_overlap_owner/版本，audit=ROOT/experiments/hpc_launch_audits/版本。A_order原类别顺序先到先得；B_score仅重叠处按原实例FP32 SAM分数竞争，同分保持原顺序。两组共享一次恢复的原实例mask/score，先每类union，再完全相同的概率.4/close3/原迭代/膨胀1与diningtable原close2。分数用相同灰度max/min几何扩展到修复边缘，必须support与原binary修复逐像素一致；不是旧compose_instances(score)同时改成实例修复的混杂对照。FG union与非重叠标签不变，后续同一语义清理可能因归属不同产生不同区域/背景数，不虚称新成员数固定或背景特征必然相同。

原始实例/score未保存，故授权在10582训练图按原冻结SAM3/Active/presence1/FP32配置恢复；不生成新token缓存，GMP只读签名原72x72x1280 FP32 token。A必须在同一次恢复输出上与原生成器pixel exact；历史raw差异完整记录，不强称与旧35canonical fallback字节等价。A/B都重新生成匹配raw/clean标签、普通HardCenterGMP rho.1和两层1280-512-256 GELU MLP，各42/123/2026，共6新训练/30000step，原8466fit/2116hold图、CE+.5SupCon/50x100等配方不变。统一production balancedK25/top5/search2/center.2 macro firststrictmax选epoch；前不删/后各FG floor5%/BG不删。不得新阈值/比例/arm/seed/结构/loss或VOC反馈。原accepted balanced作为历史参考，不只把新B与旧A比较；不同内部伪标签/样本量的Top1只作各自选模诊断，不是跨组真实质量。

真实源小probe3828508已唯一提交并实际RUNNING，12本地/集群目标tests无fail/error/skip；最近2026-10-28T08:33:17Z为RUNNING19:12/c2v100m2-2，仍在SAM3源恢复，尚无12图完整接受报告或新mIoU。固定单次Slurm dispatcher3828759已实际提交、afterok3828508、PENDING。它先独立重读12保存mask/FP32score/geometry及probe C0，再唯一注册480code=accepted106的472逐SHAclone+8新文件，然后唯一提交generate0-3%4→prepare→technical→train0-5%3→predict0-5%3→evaluate→独立audit，19full科学child+1audit，全afterok；generate依赖dispatcher C0。当前full尚未确认注册/科学JobID/6训练开始，不预填manifest/binding或结果。probe与dispatcher实际receipt已下载逐SHA/资源/依赖核，均不是completion。source probe1V100/6CPU24G2h，dispatcher CPUcompute但仍1V100配额/6CPU24G12h且CUDA隐藏。健康队列按依赖自主执行，不创建自动化/监测，按用户要求等其手动回来再查。

已上传的outer001在提交前发现继承source new_encoder_GMP=0旧元数据，未执行；另存register_full_002/full_expected_002/submit_dispatcher_002仅纠正训练源恢复/GMP元数据，8科学patch字节不变，001保留。expected002 SHA7eadd3fb3a17689a35040494e73a1e03f23eae55ad59e2d64ed32052d96892e6，实际dispatcher receipt SHA9b64dcb083a0edcdd0a11d8ea3afcfcc86e3ad7cf6eb0bb2af2a62053f1902d9；已有guard/intent/JobID禁止重复提交/登记/上传科学code或手动verify活audit。

完整6模型原native/crop rawquery与proposal固定，面积300/NMS.4/HCR/strictR3/LooseE不变，不新生成VOC SAM3/token/GMP/proposal。17388新PNG/12组全SHA commit后一次1449GT；另复制6历史balanced逐图CM共18CM。主比较B-A完整E、HCR辅助；逐seed/mean/sampleSD/方向/21IoU/BGFP/FN/wrongFG/chair↔sofa双向及两类IoU、重叠归属人数/像素/历史漂移/成本/Torchallocated峰值都报告。meanE差>0且>=2/3正才记方向支持，不自动采用；B只胜freshA却不胜当前参考时不能叫最终模型提升。VOC已看development，无独立test/新bootstrap/CI宣称。

只有completion_acceptance_001 passed/exactbinding且最终audit C0才小catalog逐SHA下载/独立analysis/资源/归档回传和手册最终补记。审计只实际字节、保存mask/score归属、IDs、CM/CSV算术，不重SAM3/训练/检索/GMP/GT。科学失败先具体日志留全部attempt，仅停自有未启动依赖，新科学修复需新版本/针对小probe；不改运行snapshot/断言。无自动主线推广/部署/GitHub/ZIP，不动其它批/个人queue。后续手册backup实际旧remote+CAS，冲突merge保留其它章节。


### 108.1 原准备超时、输入等价检查与独立恢复（2026-10-29）

旧full实际manifestSHA22cac5476d13e4836ba9ec190786b0ac28c2959b5450913f791d5099bd5c94a4/bindingd96d7175b8346d078853ef9c01a15eae9870818c8db74c77900ae71204b7e96b。generate3828811四child均COMPLETED0:0，10582图全部恢复，历史raw changed images均0。prepare3828812在06:00:04 TIMEOUT（batch CANCELLED0:15/externC0）；两组prepared.npz及prepared_report已写出，training_owner_counts.csv为0字节，prepare_complete不存在，不能视为准备通过。六训练未启动；15自有PENDING下游已精确取消且Elapsed0，旧代码/manifest/结果/残留锁/空CSV全部保留不覆盖。超时logSHAae99839a2b5a90ca68c96485bbde62d3367a76c468ea3ce112d358ecb7c6c402，清理proofSHA7b91f3d388d3b0a667d196af2d3a53c22e5e7730880bf237646def60ece7afc4。

旧prepare反复校验84656个生成产物（约626MB）、重复打开10582个receipt多遍。恢复仅改I/O：8线程初次逐SHA读真实字节，进程内缓存依dev/ino/size/mtime_ns/ctime_ns逐次保护，JSON复用解析；进程间不信任缓存，不跳过首遍字节。增加实际文件/字节/耗时进度。Slurm靶向probe3834378根/batch/extern C0/1:13，16真实图两arm的96数组字段exact，receipt完全相同，文件变更/错误SHA正确拒绝；首次9.0847s/三遍重复.00156s只属此probe，不外推全批速度。无GT/model/token/GMP/训练。新prepare必须再次完整重建并逐字段exact核旧两prepared数组后才afterok，不把旧未完成prepare冒充接受。

唯一恢复版本pseudo_overlap_owner_recover_20261029_100000：manifestSHA351413174697440c7a4e3c0a92aa7069d2d7d969090dc06da2b9bdf564842fd5，binding5dfb90fd7e154dea72828a1d01777f004b6bcc3ab47a709738e569acb7dc26d0，481code=477原字节不变+runtime I/O与来源适配/audit来源适配/protocol版本及记录数/helper共4文件。原两arm/三seed/六完整训练/全部超参/选模/后5%/HCR-R3-E/17388PNG及GT后置断言不变。technical/train_one/own_gallery/predict/evaluate/summary/data AST与旧版相同。四旧generation completion与原binding逐SHA锚定复用；不再SAM3/GMP提取，不重写旧生成收据或伪造新generation成功。

唯一actual恢复链：prepare3834436→technical3834437→train3834438 array0-5%3→predict3834439 array0-5%3→evaluate3834440→独立audit3834441，新live边全afterok。prepare/audit24h，technical6h，train/predict48h，eval12h；各1V100配额/6CPU24G，CPU角色CUDA隐藏。首prepare不引用过期afterok，提交前已精确核旧4源child C0/probe C0/旧失败链无活allocation。恢复登记与launch小凭据已本地逐SHA核，仍非完成/精度/独立接受。源恢复与旧超时、probe和新链成本分列，不能累加并行墙钟。

继续只查新ledger/progress/log/failure/complete；已有JobID/guards不重提交/登记/上传科学代码，不手动verify活audit。最终仍以新completion_acceptance_001 passed/bindingexact及auditC0为准；届时小SHAcatalog/18CM/独立analysis/归档，不读新GT或训练调参。用户要求无自动化，未创建，按手动回来检查。当前没有新mIoU/主线采用。


## 109. LSA＋HC-GMP＋PRR正式主线与GitHub发布（2026-10-29）

用户明确选择已推荐的balanced/面积100/MLP后删5%完整配置作为主线，并明确要求上传GitHub及文档，沿用新模块名。本授权覆盖历史实验的“不自动采用/不发布”限制，但不授权改写已冻结实验、重训旧任务、调参或上传数据权重。目标仓库https://github.com/nanding0414-spec/1010的main，原Git历史保留，不强推；工作区无.git，在artifacts/mainline_release_20261029/repo独立clone审查发布。

命名：LSA=Lightweight Semantic Adaptation（轻量语义适配，冻结SAM3与两层MLP）；HC-GMP=Hard-Center Generalized Max Pooling（硬中心广义最大池化）；PRR=Progressive Retrieval Recovery（渐进式检索补全）。实际计算顺序始终是SAM3掩码/token→HC-GMP→LSA的MLP→单库检索，不能为模块命名把MLP移到池化前。特征库是共享存储，不额外包装为创新；这三组模块名不等于三个独立创新已获证明。模型有轻量MLP训练，不称完全training-free。

### 109.1 冻结配方与实测边界

正式配方configs/recipes/lsa_hcgmp_prr_v1.json。原训练图像级类别presence1、概率.4 per-class-union边缘修复、原类别顺序先到先得重叠合成及几何清理均保留；不采用按SAM分数重新分配重叠。冻结SAM3，原1008/FP32/单相位硬中心GMP rho.1、两层1280→512→256 GELU、输入输出L2、CE+.5SupCon/原AdamW/cosine50×100/按图留出。内部选epoch改为生产balanced K25/top5/search2/center.2宏准确率first strict max；临时训练分类器不用于推理。

MLP前删除0%；全成员训练及导出后，只对各FG类在FP32投影空间按类均值Euclidean距离降序删floor(N*.05)，原numpy argsort降序同分语义、保留成员原顺序，BG不删。另存新库，不原地改历史。新库明确记录词典序类别（含背景）及类内保留原成员行序；推理按这个顺序装载，防止旧foreground+background布局改变近邻同分行为。训练区域提取最小面积仍300；**分类后推理面积为100**，不复用同一面积配置去改变训练数据。推理单库、balancedK25/top5/search2/center.2、mixture.2、margin0、NMS.4/instance.02固定。

PRR=原HCR→Strict R3→Loose E。中心80%裁剪生成映射回原图的候选几何，所有候选语义用原图HC-GMP＋同一LSA/同一库，不用crop token冒充原图。Strict q.4/r.95/m.1，Loose q.4/r.9/m.05，live背景比例.1，每步只填背景；Loose E复用相同mapped候选及原图分数，不新增第三个视图。PRR新增一次冻结SAM3裁剪前向；实现中的特征编码器与候选生成器使用同源冻结权重，不声称零额外算力或一定共享同一模型对象。

选定结果来自small_mask_hcr_20261027_200000/C_area100：42/123/2026=87.18510366582714/87.08809049448544/87.03237153523482%，mean87.10185523184913%、sampleSD.07729085925797713pp。manifest7311538b46f35c40ff79641f57e0eae8f8db4a29c0f91d3302049b92127ddff3，bindingdeac45aa31d178600e55c58c54d56db1fc288f7199fea11bb61ec52363a50200；2026-10-29真实核audit3822332根/batch/extern均C0/2:03:48，completion_acceptance_001 passed且bindingexact。相对面积300 Emean87.09512170103622%仅+.006733530812913576pp，三seed均微正，不包装成主要创新或统计显著。5%曾用VOC调参，VOC为已看development，不是独立test。

本次是代码与文档集成，不新增训练/全量预测/VOC评测；新入口尚不能冒称已完整复跑上述精度。新标签、新库、不同环境或排序应记录新版本实际结果。当前重叠对照的评测已完成但audit3834441此前尚RUNNING，其preliminary B均值86.855076%不作为新的独立接受；已接受balanced历史参照不受其审核等待影响。

### 109.2 正式运行入口

README中的安装与命令与本节同步。tools/run_mainline_full.py --version NEW --seed 42 --gpus 0 1为从头完整入口；--preflight-only仅预检，--probe-only只做原GPU/短训练/HCR诊断，不冒充全量PRR验收。raw merge不删除成员，train先导出完整LSA库，postfilter独立阶段输出新post05库，eval默认--rescue --prr。标准资源data/VOC2012、weights/sam3/sam3.pt、训练image-level JSON和分割列表须完整；不自动下载权重或上传数据。

已有native LSA库：tools/postfilter_lsa_bank.py --source-bank UNFILTERED --output-bank NEW_POST05，再tools/run_mainline.py --bank NEW_POST05 --adapter ADAPTER --checkpoint SAM3 --dataset-root VOC --clean-proposals-dir CLEAN --raw-proposals-dir RAW --output-dir NEW_RUN --gpu 1 --seed 42；实际CLI写法见README（参数与值分开）。正式推理默认PRR并核postfilter成员/adapter/SAM3 SHA绑定、原HC-GMP签名和模型尺寸、GPU占用；--no-prr显式HCR-only、--no-hcr显式clean-only，不把消融当主线。低层run_inference.py仍需显式--prr和area100等环境契约；直接低层默认保留旧实验兼容，不等于正式入口。

历史独立checkpoint不是native adapter格式：tools/export_lsa_mainline.py --source-manifest ACCEPTED_SMALLMASK_MANIFEST --source-acceptance ACCEPT --seed 42 --output-dir NEW_EXPORT。只接受真实已签名C_area100来源链，核模型/投影导出SHA/epoch、1024原输入投影1e-6回放；参数原样另存native adapter，完整投影向量按类另存后删5%库。无重训/阈值选择/GT/新SAM3，旧模型库不覆盖。转换后的adapter.pt与bank交正式推理入口。

### 109.3 发布与验证原则

新src/semantic_bank_filter.py提供非破坏后过滤与成员校验；src/progressive_retrieval_recovery.py提供PRR接口，复用原已验收strict/loose纯算子，不重写门槛/排序；src/feature_adapter.py等旧标识作为兼容内部实现，不批量重命名破坏checkpoint或signature。新recipe ID为lsa_hcgmp_prr_v1。旧r3_record_v1及第84节保留历史，不能再称当前默认主线。

本机无torch，NumPy/接口测试不冒充CUDA/真实精度；需PyTorch的回归在隔离集群CPU环境执行，CUDA隐藏但占1V100配额。若执行已验收模型转换/小保存mask-score回放，只是集成一致性，不重训、不GT、不产生新mIoU结论。发布仅白名单源码、必要测试/词表/许可证、固定配方、README/AGENTS/完整手册，不上传数据、权重、token、库、proposal、日志、运行目录、SSH key或凭据；不打ZIP，不创建自动化，不部署覆盖正在运行实验。

### 109.4 发布集成检查与保留的失败记录

发布目录独立执行本地82项NumPy/接口测试全过；集群CPU89项全过、无skip，unit_tests.json SHA4d53c245e6fc67080881f2c6ee042b1756ffa94473d3b7a0316a90fc866fdca3。首轮3838035因独立发布目录漏带测试依赖而主动取消（17:36），不当成功；第二轮3838150的89测试全部成功，但后续外层脚本误把已有acceptance_001写成不存在的_002而FAILED1:0/12:28，未开始模型转换。两个attempt代码、日志和真实Slurm状态保留，不改科学源和断言。

修正仅发布检查外层，独立新Job3838928已COMPLETED0:0；三个已验收checkpoint原参数转换native格式，各1024行投影atol/rtol1e-6通过，三个post5库的36231保留ID与来源逐值相等；六份保存mask/score的PRR回放与已保存C_area100 PNG像素精确。无GT读取、重训、新SAM3/GMP或全量新精度评测。详细资源、各seed最大误差和receipt SHA见docs/lsa_hcgmp_prr_validation_20261029.json。CPU检查仍分配1V100配额而CUDA隐藏，不能称零GPU资源。

GitHub main发布使用旧HEAD634730200007670430d59976cb820bd161632e4d为parent，普通快进更新，不强推；release_manifest.json绑定正式文件逐SHA，旧R3发布状态可沿旧commit复现。源码和完整手册上传，数据/权重/库/缓存/运行目录不上传。GitHub最终commit和远端逐字节核验凭据保存在本地artifacts/mainline_release_20261029；本节不能包含自身commit/hash以免形成循环。
