# TCSVT 同方向论文的方法写法：本项目修订依据

本次实际阅读了三篇完整论文的方法及实验设置，而不是只按摘要推测：

|论文|方法部分怎样组织|实现细节放在哪里|
|---|---|---|
|Li等，Vision-Language-Driven Prompt Learning for Weakly Supervised Semantic Segmentation，TCSVT 2026，DOI [10.1109/TCSVT.2026.3652185](https://doi.org/10.1109/TCSVT.2026.3652185)|Sec.III先概述整体结构，再解释各模块如何改善语义定位，配合流程图、变量定义和计算公式|Sec.IV的Implementation Details报告骨干、特征层、阈值、训练和后处理设置|
|Wang等，Class Activation Map Calibration for Weakly Supervised Semantic Segmentation，TCSVT 2024，DOI [10.1109/TCSVT.2024.3427720](https://doi.org/10.1109/TCSVT.2024.3427720)|Sec.III从相关CAM定义过渡到两个校准模块，解释输入输出、相互关系和公式|实验部分给出数据、训练配置和测量；完整方法见[作者机构稿](https://livrepository.liverpool.ac.uk/3183038/1/Wang24_elements.pdf)|
|Cheng等，Cross-Block Sparse Class Token Contrast for Weakly Supervised Semantic Segmentation，TCSVT 2024，DOI [10.1109/TCSVT.2024.3442310](https://doi.org/10.1109/TCSVT.2024.3442310)|Sec.III介绍架构、稀疏注意力、跨块类别token对比及目标；使用模块图和式子说明机制|Sec.IV的Implementation details给出训练分辨率、数据增强、预训练骨干、温度等；[作者机构全文](https://bura.brunel.ac.uk/bitstream/2438/31391/3/FullText.pdf)|

因此，不能概括为“论文不写实现细节”。方法里需要保留定义方法的细节，例如两层MLP、GELU、损失、token选择、检索公式、空邻居处理和背景保护条件；但Python函数、类名、CLI参数名、存储目录、SHA、Slurm JobID、锁文件、失败恢复和下载验收不应充当方法论述。模型维度、学习率、批次、具体阈值、软件及硬件放Implementation Details；研究动机、数学定义、模块关系及性质放Method。

## 本次论文改动

1. 保留LSA、HC-GMP、PRR三个命名，不把共享特征库包装为第四个创新。计算顺序仍是冻结SAM3→HC-GMP→MLP→单库检索→PRR。
2. 重写引言与贡献，明确与ModuSeg的对象发现/语义检索解耦关系，以及本工作增加的区域适配、池化选择与补全机制。GMP已有出处，不宣称首创GMP；轻量MLP需要训练，不称完全training-free。
3. HC-GMP写清token支撑集、nearest-centroid fallback、能量缩放ridge、优化目标及正规方程；不使用函数调用和FP32代码片段代替原理。
4. LSA写清两层映射、临时分类头、CE与SupCon、按图留出及后过滤；检索写清全局候选集、类内top-m、中心融合、缺类fallback和margin。
5. PRR按统一候选处理过程写：互补几何来源→同一原图语义表示→优先级→当前可填背景→门控→更新。保留分级门槛，但不用内部实验名H/R3/E拼接论文段落。
6. 用一个简短推导说明前景不覆盖；明确该性质不保证mIoU单调增加，也不能修正原先的错误前景类。
7. 共享编码写成计算组织，而不是独立学习创新；保留预览和crop仍需单独编码的事实，删除hash、hook、文件身份和控制接口的正文描述。
8. 数字超参数集中在Implementation Details；保留实验结果空表和图片位置。COCO未完成，表中不填写估计值。已接受VOC共享回归的结果与计时范围单独说明，不把其10.33%当单seed端到端部署提升。
9. 删除旧稿大量“代码应该如何检查”“模块命名不足以证明创新”等重复自我审查段，把投前待办放本地说明文档；正文仍保留必要的监督与证据限制。

本次是在表达组织、论证深度和排版方面借鉴，并未复制上述论文的正文、图、公式组合或实验数字。期刊[当前投稿说明](https://ieee-cas.org/tcsvt-submission-transactions-paper)要求IEEE双栏稿总长不超过14页；最终页数应由完整实验和图表决定，不靠重复说明凑页数。

修订稿保存在本机artifacts/tcsvt_paper_revised_v3，源文件可用于Overleaf。未投稿草稿没有默认放入公开仓库。结果表、框架图、COCO、最终同配方消融、对外比较、测试集结果和作者邮箱仍需补齐。以上内容不是“已经可以投稿”的声明。
