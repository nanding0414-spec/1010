# ModuSeg SAM3 · 原图 GMP＋MLP2＋HCR 单库主线

主线：原图 Active 伪标签 → 冻结 SAM3 → 单相位硬中心 GMP（ρ=0.1）→ 原始区域库 → 两层 MLP（1280→512→256）→ 单一投影库 → 已清洗提议检索 → HCR只补背景。无生成多视角图片。

无多视角历史 VOC val 三 seed 平均 **86.4316% mIoU**，42/123/2026 分别为86.2723/86.4403/86.5822%。VOC val已用于开发，不能作为独立测试成绩。

唯一项目交接手册：[docs/PROJECT_STORAGE_MANUAL.md](docs/PROJECT_STORAGE_MANUAL.md)。
参数在 `configs/config.py`（模型、建库、检索和路径）与 `configs/experiment.py`（伪标签、MLP、清理配方），均有中文注释。

具体构建、导入与推理命令见 [主线说明](docs/MAINLINE.md)。正式推理入口为 `tools/run_mainline.py`，默认严格评测并启用HCR；`--no-hcr`用于clean消融。SMFA保留为显式兼容选项，历史实验工具不是正式入口。

安装：复用可用的 CUDA PyTorch/torchvision 环境，再执行 `pip install -r requirements.txt`。SAM3源码及许可证保留在 `sam3-main/`。数据、权重、特征库、提议缓存和历史产物均不提交GitHub。通知密码只从 `MODUSEG_SMTP_PASSWORD` 环境变量读取，参见 `.env.example`。

所有新库、权重与结果使用新版本目录。严格检查算法、参数、精度及权重签名，不覆盖历史数据。

