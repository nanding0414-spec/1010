# LSA + HC-GMP + PRR · SAM3弱监督语义分割

当前主线：**冻结SAM3、两层轻量MLP、MLP后前景库过滤5%、balanced检索、面积100与渐进式补全**。
实际顺序：SAM3掩码与稠密特征 → HC-GMP → LSA的MLP适配 → 单库；推理复用同一表示和库，再执行PRR。
SAM3不训练，MLP和训练用临时分类器需要训练，因此本方法不称为完全training-free。

| 模块名 | 全称 | 实际职责 |
|---|---|---|
| **LSA** | Lightweight Semantic Adaptation，轻量语义适配 | 冻结SAM3＋两层1280→512→256 GELU MLP；CE＋0.5 SupCon；临时分类器推理时丢弃 |
| **HC-GMP** | Hard-Center Generalized Max Pooling，硬中心广义最大池化 | 原图掩码与token硬中心对应，FP32 GMP，ρ=0.1 |
| **PRR** | Progressive Retrieval Recovery，渐进式检索补全 | HCR → Strict R3 → Loose E；仅填当前背景，保护已有前景 |

特征库是共享存储组件，不另包装为创新模块。MLP后过滤属于库维护，不在PRR内。
新名字对应既有运算，不意味着已经证明三个独立创新。

## 已验收结果

VOC2012 val，1449张，seed42/123/2026：**87.185104 / 87.088090 / 87.032372%**。
均值 **87.101855%**、三seed样本标准差 **0.077291个百分点**。
来源 `small_mask_hcr_20261027_200000 / C_area100`，独立audit `3822332` passed且COMPLETED0:0。
这是已看过的VOC开发集结果，含前期5%比例选择；不是独立test或显著性结论。
面积100比面积300完整配置均值仅高0.006734个百分点，不把这个微小差异当作主要贡献。

固定配方与历史凭据：[configs/recipes/lsa_hcgmp_prr_v1.json](configs/recipes/lsa_hcgmp_prr_v1.json)。
本次是正式入口集成，**不冒充新的全量GPU精度验收**；新生成标签、库、权重或环境的运行应记录自己的结果。
旧 `configs/recipes/r3_record_v1.json` 和 `tools/run_r3_record.py` 保留作历史复现，已不是当前主线。

## 安装与资源

复用可用的PyTorch/torchvision CUDA环境，再运行 `pip install -r requirements.txt`。
数据放data，权重放weights，标签/库/提议/token放storage，指标放experiments，日志放logs。
SAM3源码和许可证保留在sam3-main，ActiveSAM说明见docs/activesam_reference.md。
GitHub只提供源码、测试、配方与文档，不提供数据、权重、库、缓存、私有配置或运行产物。

## 从头运行主线

```bash
python tools/run_mainline_full.py --help
python tools/run_mainline_full.py --version lsa_hcgmp_prr_seed42_v1 --seed 42 --gpus 0 1 --preflight-only
python tools/run_mainline_full.py --version lsa_hcgmp_prr_seed42_v1 --seed 42 --gpus 0 1
```

完整入口依次生成原顺序训练伪标签、几何清理、HC-GMP原库、训练LSA、导出未过滤投影库、
另存后删5%的单库、推理与PRR。**原始库不再先删10%**。训练区域面积仍300，分类后推理面积100，两个阶段不混用。
选epoch：内部留出图片上的production balanced K25/top5/search2/center0.2宏准确率，首次严格最大值。
新运行使用新版本和源码快照，不得把新版接在旧实验目录上续跑。

## 使用已有模型推理

普通native格式的LSA未过滤导出库先另存post5库，只做一次：

```bash
python tools/postfilter_lsa_bank.py --source-bank storage/mlp_banks/voc/UNFILTERED --output-bank storage/mlp_banks/voc/POST05
python tools/run_mainline.py --bank storage/mlp_banks/voc/POST05 --adapter weights/LSA/adapter.pt --checkpoint weights/sam3/sam3.pt --dataset-root data/VOC2012 --clean-proposals-dir storage/proposals/voc/CLEAN --raw-proposals-dir storage/proposals/voc/RAW --output-dir experiments/LSA_HCGMP_PRR_NEW --gpu 1 --seed 42
```

推理入口默认包含PRR，不必再追加旧R3脚本。默认物理GPU1；完整入口明确使用GPU0/1。
验证时只给图像和固定20类词表，不给验证图真实图像级类别。
中心80%裁剪只提供映射后候选几何，语义来自**原图特征、同一MLP和同一单库**。
每图增加一次冻结SAM3裁剪前向，不是免费后处理。
`--no-prr` 为HCR-only消融，`--no-hcr` 为clean-only消融，均不是完整主线。

历史独立实验checkpoint格式不同，不能直接填成native adapter。使用明确的已验收源转换：

```bash
python tools/export_lsa_mainline.py --source-manifest /path/to/accepted/small_mask_hcr_manifest.json --source-acceptance /path/to/accepted/completion_acceptance_001.json --seed 42 --output-dir weights/LSA_EXPORT_NEW
```

转换器核独立接受、所选epoch、checkpoint/export SHA及1024行投影回放，不重训参数。
另存的 `LSA_EXPORT_NEW/adapter.pt` 和 `LSA_EXPORT_NEW/bank` 可交给正式推理入口；原实验文件不改。
输入必须是原签名真实HPC文件，不能手写假manifest或重签旧数据。

唯一操作手册：[docs/PROJECT_STORAGE_MANUAL.md](docs/PROJECT_STORAGE_MANUAL.md)，**当前正式操作见第109节**；前文历史与实验记录保留。
