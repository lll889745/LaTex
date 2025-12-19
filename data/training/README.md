# 训练数据目录

本目录用于存放符号识别的训练数据。

## 数据格式

推荐的目录结构：

```
training/
├── 0/          # 数字 0 的样本
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── 1/          # 数字 1 的样本
├── 2/
├── ...
├── a/          # 小写字母 a 的样本
├── b/
├── ...
├── A/          # 大写字母 A 的样本
├── B/
├── ...
├── plus/       # 加号的样本
├── minus/      # 减号的样本
├── times/      # 乘号的样本
├── divide/     # 除号的样本
├── equals/     # 等号的样本
├── sqrt/       # 根号的样本
└── ...
```

## 图像要求

- 格式：PNG 或 JPG
- 建议尺寸：32x32 像素
- 类型：二值图像或灰度图像
- 背景：白色（255）
- 前景：黑色（0）

## 推荐数据集

1. **CROHME** (Competition on Recognition of Online Handwritten Mathematical Expressions)
   - 下载地址: https://www.isical.ac.in/~crohme/

2. **HASYv2** (Handwritten Symbol Classification)
   - 下载地址: https://zenodo.org/record/259444

## 数据准备脚本

```python
import os
import cv2
import numpy as np

def prepare_training_data(data_dir):
    """从目录加载训练数据"""
    images = []
    labels = []
    
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # 标准化大小
                img = cv2.resize(img, (32, 32))
                images.append(img)
                labels.append(label_name)
    
    return images, labels

# 使用示例
images, labels = prepare_training_data('data/training')
```
