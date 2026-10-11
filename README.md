# ModuSeg SAM3 · presence=1＋边缘修复＋几何清理＋GMP2＋HCR 主线

主线：训练图像级类别 → Active SAM3（已知类别存在概率固定1）→ 概率阈值0.4、close/dilate边缘修复 → 逐类别语义几何清理 → 冻结SAM3 → 单相位硬中心GMP（ρ=0.1）→ 两层MLP（1280→512→256）→ 单库检索 → clean/raw提议与HCR只补背景。验证提议不使用真值类别，保持正常预测存在概率。

当前已验证版本 `mainline_edge_cleanreuse_20261011_114857`：VOC val **86.7350% mIoU**，seed42，完整50轮训练、1,449张严格评测，内部最佳38轮。比边缘修复但未清理版本提高0.4656个百分点。此分数是单seed开发验证结果，不代表三seed均值或独立测试成绩。此前无多视角三seed均值86.4316%属于历史配方。

唯一项目交接手册：[docs/PROJECT_STORAGE_MANUAL.md](docs/PROJECT_STORAGE_MANUAL.md)。
参数在 `configs/config.py`（模型、建库、检索和路径）与 `configs/experiment.py`（伪标签、MLP、清理配方），均有中文注释。

正式双卡完整入口 `tools/run_mainline_full.py --version YOUR_NEW_VERSION` 默认启用presence=1、边缘修复、语义几何清理及整图FP32缓存；原始标签和清理后标签分开保存。几何清理与验证提议共用去小岛/填小孔函数，语义PNG不伪造实例分数、不做跨类别NMS。

正式推理入口为 `tools/run_mainline.py`，默认严格评测与HCR；服务器可读取本地激活记录 `experiments/mainline_active.json`，新安装需显式提供库、MLP、数据及提议路径。独立标签清理入口为 `tools/clean_pseudo_labels.py`，缓存复用对照入口为 `tools/run_pseudo_cleanup_ablation.py`。所有命令和参数见[唯一操作手册第46节](docs/PROJECT_STORAGE_MANUAL.md#46-当前正式主线与github同步2026-10-11)。

安装：复用可用的CUDA PyTorch/torchvision环境，再执行 `pip install -r requirements.txt`。SAM3源码及许可证保留在 `sam3-main/`。数据、权重、特征库、提议缓存、运行记录和通知凭据不提交GitHub。通知可用 `MODUSEG_SMTP_PASSWORD` 或私有 `MODUSEG_NOTIFY_CONFIG`，参见 `.env.example`。

所有新库、权重与结果使用新版本目录。严格检查算法、参数、精度及权重签名，不覆盖历史数据。
