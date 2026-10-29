# AI 接手入口

执行项目前先读 `docs/PROJECT_STORAGE_MANUAL.md`。它是唯一项目操作与模块说明手册。

- 当前正式主线为 `configs/recipes/lsa_hcgmp_prr_v1.json`：原顺序presence=1标签/概率0.4边缘修复/逐类别几何清理；LSA（冻结SAM3＋两层GELU MLP）、HC-GMP（rho=0.1）、MLP前不删/后各FG删5%且BG不删、balanced选epoch/单库检索、分类面积100/NMS0.4、PRR（HCR→Strict R3→Loose E，只补背景）。完整入口默认保存整图FP32特征；正式操作见手册第109节。训练区域面积仍300。其它实验不自动加入主线。
- 数据、库和提议按 storage 分类；参数、指标放 experiments；权重放 weights；启动日志放 logs。
- 不覆盖历史版本，不修改原始标签、提议、指标；先检查真实 CLI、文件完整性、签名和设备占用。
- 默认第二张物理 GPU；双卡入口明确使用 GPU0/1。没有服务器连接时，不声称已启动或获得真实精度。
- 改入口或目录须同步手册。运行中任务使用原代码；新代码、新参数启用新实验版本。
- 不自动打包；只有用户明确请求交付压缩包时打包纯代码，不含数据权重。
- 用户2026-10-29已明确授权将LSA＋HC-GMP＋PRR配置作为主线并上传GitHub及文档。旧 `r3_record_v1.json` 和第84节保留作历史复现，不是当前默认。发布/CPU接口回归不冒充新的全量GPU精度验收，不修改历史冻结任务或数据。
