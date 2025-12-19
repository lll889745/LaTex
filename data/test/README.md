# 测试数据目录

本目录用于存放测试图像。

## 使用方法

将手写公式图像放入此目录，然后运行：

```bash
python main.py --image data/test/your_formula.png
```

## 图像要求

- 格式：PNG、JPG、BMP
- 建议分辨率：至少 200 像素高度
- 内容：手写或打印的数学公式
- 背景：尽量使用浅色背景

## 测试示例

可以使用以下方式创建测试图像：

1. 手写公式并拍照
2. 使用绘图软件绘制
3. 使用系统提供的演示功能生成

```bash
python main.py --demo
```

这将生成 `demo_input.png` 作为测试用例。
