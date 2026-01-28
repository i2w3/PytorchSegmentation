# 语义分割
以 [EarthVQA](https://github.com/Junjue-Wang/EarthVQA) 为例的一个简单语义分割示例。数据集可以从 [kaggle](https://www.kaggle.com/datasets/dyiyacao/earthvqa) 下载。

## file structure
```
{ROOT}/
├── datasets/EarthVQA/
│   ├── Train/
│   │   ├── images_png/
│   │   └── masks_png/
│   ├── Val/
│   |   ├── images_png/
│   |   └── masks_png/
│   └── Test/
│       └── images_png/
├── scripts
│   └── build_EarthVQA.py   # 用于下载和预处理 EarthVQA 数据集的脚本
│── train_eomt.py           # 基于 lightly_train 的训练脚本
└── train_smp.py            # 基于 segmentation_models_pytorch 的训练脚本
```