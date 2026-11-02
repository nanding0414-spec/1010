# LSA + HC-GMP + PRR：共享编码执行版

推荐入口是 `tools/run_shared_mainline.py`，语义配方仍为 `lsa_hcgmp_prr_v1`，执行配方为 `configs/recipes/lsa_hcgmp_prr_shared_v2.json`。LSA、HC-GMP、PRR 命名保持一致。特征库是共享资源，不单列为创新。

同一冻结 SAM3 对象产生原图候选与原图特征；只有实际预处理输入逐值相同时才复用编码。672 预览与中心80% crop 是不同输入，仍各自编码。PRR 从同一原图特征、同一 MLP、同一后过滤库取得语义证据，通过统一入口按原图补全、严格 crop 补全、宽松 crop 补全的优先级更新实时背景。已有前景始终保持。新入口默认不保存整图 FP32/token 缓存。

## 完整复现路径

先准备 VOC2012 的 JPEGImages、明确的 trainaug/val 图像清单，以及只含训练图像级类别的 JSON（`images` 中包含 `file_name`、`labels`）；SAM3 权重需按原项目许可自行取得。所有输出目录必须是新目录。下面的路径均由使用者替换；GPU 默认第二张物理卡。

```bash
python tools/run_shared_mainline.py build --checkpoint /models/sam3.pt --dataset-root /data/VOC2012 --image-list /data/VOC2012/ImageSets/Segmentation/trainaug.txt --imagelevel-json /data/VOC2012/ImageSets/ImageLevel/trainaug_imagelevel.json --output-dir /runs/voc_shared_build --gpu 1

CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 python tools/train_feature_adapter.py --source-bank /runs/voc_shared_build/raw_bank --train-list /data/VOC2012/ImageSets/Segmentation/trainaug.txt --output-dir /runs/voc_seed42 --export-bank /runs/voc_seed42_projected --recipe v1 --classifier linear --selection-metric balanced --num-layers 2 --hidden-dim 512 --output-dim 256 --epochs 50 --steps-per-epoch 100 --eval-topk 25 --eval-class-topm 5 --eval-search-multiplier 2 --eval-centroid-weight 0.2 --seed 42 --device cuda:0

python tools/postfilter_lsa_bank.py --source-bank /runs/voc_seed42_projected --output-bank /runs/voc_seed42_post05
```

独立机器训练时需在进程启动前明确设置 `CUDA_VISIBLE_DEVICES=1`；进程内部使用 `cuda:0`。Slurm作业则保留调度器分配的可见GPU，不使用示例中的物理卡重映射；共享入口检测到Slurm后也不会覆盖其GPU分配。另两个 seed 为123与2026，使用独立模型/导库目录。训练入口的原始特征 KNN 与 balanced 基线属于额外诊断，其当前 CPU 全库实现可能较慢；每 epoch balanced 选模必须保留。此次发布没有修改这段计算，也没有宣称 COCO 训练已提速。

依据过滤工具的真实 CLI，生成与该模型对应的前景后删5%、背景不删的库后运行：

```bash
python tools/run_shared_mainline.py predict --checkpoint /models/sam3.pt --dataset-root /data/VOC2012 --image-list /data/VOC2012/ImageSets/Segmentation/val.txt --bank /runs/voc_seed42_post05 --adapter /runs/voc_seed42/adapter.pt --output-dir /runs/voc_seed42_prediction --gpu 1 --seed 42
```

`completion.json` 提交 PNG 与 trace 的 SHA；预测过程不读取 GT。像素评测应在全部预测完成之后单独执行，遵循 VOC 的类别映射与 ignore=255。不要把三 seed 的像素混池；按每 seed 数据集 CM 分别算 mIoU，再报告 mean/sampleSD。

## 验证范围与历史结果

已完成的参考实验 `sam3_shared_encoding_full_recover_20261101_163000` 对1449张图、三seed逐像素检查共享/分开原图编码，所有最终预测及逐图CM相同；mean mIoU=87.1018604%，sampleSD=0.07729086pp。其固定顺序测量区间减少10.3285%，包含三seed检索和等价检查，不是单seed部署端到端延迟。参考 accept SHA 为 `29c9a6fe1cbdcc6526444127caf9d1a23d268fabfc792c45aa19b733c3b864b9`。

本次将该已接受算子整理成公开入口，完成语法、CLI与CPU接口回归。它没有新跑一次从头建库/训练/1449图完整GPU验收，不能将发布回归说成新的精度实验。精确历史复现需原匹配模型、库与输入身份；从头生成的新运行需重新测量，不直接填写历史分数。

此便携入口当前限定已验证的 VOC 21 类配方。COCO 的81类独立Slurm实验仍在运行，尚无最终接受的 mIoU，不把 VOC 模型当作 COCO 模型。旧 `tools/run_mainline.py`（已缓存候选输入）及 `tools/run_mainline_full.py`（旧分阶段流程）保留作历史兼容；后者原默认整图缓存行为不能冒充共享入口的无整图落盘行为。

论文的三个技术单元是 LSA、HC-GMP 和 PRR；只有轻量 MLP 参与下游训练，不称完全 training-free。SAM3 的外部预训练监督需单独说明。
