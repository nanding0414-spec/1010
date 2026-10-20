# ModuSeg SAM3 · GMP2、HCR 与 R3 最高记录配置

已验证的最高非 Oracle 单次记录是 **R3 + GMP2 + HCR：86.933660% mIoU**，VOC2012 val 共1,449张、MLP seed42、检索中心权重 **0.2**。对应三 seed 配对结果：

| MLP seed | 原 GMP2 + HCR | 加 R3 | 提升（百分点） |
| --- | ---: | ---: | ---: |
| 42 | 86.821717% | **86.933660%** | +0.111943 |
| 123 | 86.321692% | 86.413496% | +0.091804 |
| 2026 | 86.374583% | 86.474971% | +0.100388 |
| 均值 | 86.505997% | **86.607375%** | **+0.101378** |

这是已查看 VOC 开发集上的真实分割结果，不是 Oracle 上限或独立测试成绩；配对提升的 seed 样本标准差为0.010106个百分点。收益未达到原实验的+0.2点推广门槛。本次按用户明确要求发布配置，不代表重新训练、部署到服务器或通过新一轮GPU精度验证。

唯一项目手册：[docs/PROJECT_STORAGE_MANUAL.md](docs/PROJECT_STORAGE_MANUAL.md)，已同步截至第83节的实验结果，第84节记录本次GitHub发布及复现输入契约。历史失败和负结果仍保留。

## 固定 R3 配置

[configs/recipes/r3_record_v1.json](configs/recipes/r3_record_v1.json) 固定配方、原始权重签名、worker SHA及三seed结果凭据。R3对图片中心的80%宽×80%高裁剪额外运行冻结SAM3，使用全部20个规范VOC类名和正常预测presence；每个生成类别最多一个候选。mask映射回原图后，使用**原图FP32 token**做原GMP（ρ=0.1），进入对应seed的原MLP和单库检索，中心权重0.2。补充候选仅能填原clean+HCR预测的背景，不覆盖任何已有前景；proposal≥0.4、retrieval≥0.95、margin≥0.1、新增面积比例≥0.1。

独立入口复用原实验worker，依次执行四个阶段：

```bash
python tools/run_r3_record.py --help
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/context.json --preflight-only
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/context.json --gpu 1
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/generate.json --gpu 1
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/query.json --gpu 1
python tools/run_r3_record.py --manifest experiments/R3_NEW/manifest.json --job experiments/R3_NEW/rescue.json
```

manifest/job的生成示例和必需字段见手册第84节。新安装必须提供自己的原始签名token、原GMP库、对应MLP、SAM3权重和**center=0.2的clean+HCR预测**；这些运行产物不在GitHub。仅下载代码不具备复现记录所需的全部输入。CLI验证原worker、输入文件SHA及bank/MLP绑定；原worker消费token时核内容和原图几何。完整源码快照、GPU独占、Slurm资源及全量预测提交后再读GT由调用方负责。Slurm中保留分配的GPU可见性，其它环境默认第二张物理GPU。

## 原图主线入口

训练图像级类别 → Active SAM3（训练已知类别presence=1）→ 概率0.4及close/dilate边缘修复 → 逐类别几何清理 → 冻结SAM3 → 硬中心GMP（ρ=0.1）→ MLP2（1280→512→256）→ 单库检索 → 提议清理与HCR只补背景。验证提议保持正常预测presence。

`tools/run_mainline_full.py --version YOUR_NEW_VERSION` 是原图双卡完整入口，默认保存整图FP32 token；`tools/run_mainline.py` 是严格推理入口，默认HCR。两入口及配置的检索中心权重已同步为0.2。R3通过上面的独立入口显式追加，不会在旧完整入口中静默增加SAM3前向。参数在 `configs/config.py` 和 `configs/experiment.py`；正式主线操作与资源路径见手册第46节。

安装：复用可用的CUDA PyTorch/torchvision环境，再运行 `pip install -r requirements.txt`。SAM3源码及许可证保留在 `sam3-main/`。所有新结果使用新版本目录；旧快照和指标不覆盖。只提交源码、配置、测试、文档、必要词表与许可证，不提交数据、权重、缓存、运行日志或凭据。
