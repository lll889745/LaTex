# 手写数学公式识别与语义理解系统

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个基于传统图像处理和机器学习方法的手写数学公式识别系统，能够将手写数学公式图像转换为 LaTeX 代码，并进行语义理解（公式求解、化简等）。

## 功能特点

- 📸 **图像预处理**：自适应二值化、去噪、倾斜校正、骨架提取
- ✂️ **符号分割**：连通域分析、组件合并/拆分、特殊结构处理
- 🔍 **符号识别**：多特征提取、SVM/随机森林分类、混淆消解
- 🌳 **结构分析**：空间关系判定、语法树构建、LaTeX 生成
- 🧮 **语义理解**：公式类型识别、符号计算、错误检测

## 系统架构

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│  图像预处理 │ → │  符号分割  │ → │  符号识别  │ → │  结构分析  │ → │  语义理解  │
└────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘
      ↓                ↓                ↓                ↓                ↓
   二值化          连通域分析       特征提取        空间关系判定       公式求解
   去噪            组件合并/拆分    分类识别        语法树构建         符号化简
   倾斜校正        笔画分析        混淆消解        LaTeX生成          错误检测
```

## 安装

### 环境要求

- Python 3.8+
- OpenCV 4.x
- NumPy, SciPy, scikit-learn
- SymPy (用于语义理解)

### 安装步骤

1. 克隆或下载项目：

```bash
cd c:\Users\5555\Desktop\LaTex
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 运行演示

```bash
python main.py --demo
```

### 识别图像中的公式

```bash
python main.py --image path/to/formula.png
```

### 使用代码

```python
from src import (
    ImagePreprocessor,
    SymbolSegmenter,
    SymbolRecognizer,
    StructureAnalyzer,
    SemanticProcessor
)
import cv2

# 加载图像
image = cv2.imread('formula.png', cv2.IMREAD_GRAYSCALE)

# 1. 图像预处理
preprocessor = ImagePreprocessor()
binary = preprocessor.process(image)

# 2. 符号分割
segmenter = SymbolSegmenter()
symbols = segmenter.segment(binary)

# 3. 符号识别（需要先训练模型）
recognizer = SymbolRecognizer()
# recognizer.load_model('path/to/model.pkl')
# symbols = recognizer.recognize_symbols(symbols)

# 4. 结构分析
analyzer = StructureAnalyzer()
syntax_tree, latex = analyzer.analyze(symbols)
print(f"LaTeX: {latex}")

# 5. 语义理解
semantic = SemanticProcessor()
result = semantic.process(latex)
print(f"公式类型: {result.formula_type}")
print(f"解释: {result.interpretation}")
```

## 项目结构

```
LaTex/
├── src/                          # 源代码
│   ├── __init__.py              # 模块初始化
│   ├── config.py                # 配置文件
│   ├── utils.py                 # 工具函数
│   ├── preprocessing.py         # 图像预处理模块
│   ├── segmentation.py          # 符号分割模块
│   ├── recognition.py           # 符号识别模块
│   ├── structure_analysis.py    # 结构分析模块
│   └── semantic.py              # 语义理解模块
├── tests/                        # 测试代码
│   └── test_modules.py          # 单元测试
├── examples/                     # 示例代码
│   └── demo_usage.py            # 使用示例
├── models/                       # 模型存储（训练后生成）
├── data/                         # 数据目录
│   ├── training/                # 训练数据
│   └── test/                    # 测试数据
├── main.py                       # 主程序
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 模块详解

### 1. 图像预处理 (preprocessing.py)

实现自适应二值化（Sauvola 方法）、去噪、倾斜校正和骨架提取。

```python
from src.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()

# 完整预处理
binary = preprocessor.process(image)

# 获取中间结果
result = preprocessor.process(image, return_intermediate=True)
# result['binary']    - 二值化结果
# result['denoised']  - 去噪结果
# result['corrected'] - 倾斜校正结果
# result['skeleton']  - 骨架图像
```

### 2. 符号分割 (segmentation.py)

将预处理后的图像分割为独立符号，处理粘连和分离的符号。

```python
from src.segmentation import SymbolSegmenter

segmenter = SymbolSegmenter()
symbols = segmenter.segment(binary)

# 提取分数结构
for sym in symbols:
    if sym.is_fraction_line:
        num, den = segmenter.extract_fraction_parts(sym, symbols)
```

### 3. 符号识别 (recognition.py)

从符号图像中提取特征并进行分类识别。

```python
from src.recognition import SymbolRecognizer, FeatureExtractor

# 特征提取
extractor = FeatureExtractor()
features = extractor.extract(symbol_image)

# 训练分类器
recognizer = SymbolRecognizer()
report = recognizer.train(images, labels, classifier_type='svm')

# 保存/加载模型
recognizer.save_model('model.pkl')
recognizer.load_model('model.pkl')

# 识别符号
label, confidence, candidates = recognizer.predict(symbol_image)
```

### 4. 结构分析 (structure_analysis.py)

分析符号间的空间关系，构建语法树，生成 LaTeX 代码。

```python
from src.structure_analysis import StructureAnalyzer

analyzer = StructureAnalyzer()
syntax_tree, latex = analyzer.analyze(symbols)
```

### 5. 语义理解 (semantic.py)

对识别出的公式进行语义分析和符号计算。

```python
from src.semantic import SemanticProcessor

processor = SemanticProcessor()

# 完整语义分析
result = processor.process(latex)
print(result.formula_type)  # 公式类型
print(result.variables)     # 变量列表
print(result.solution)      # 求解结果

# 单独功能
solutions = processor.solve_equation("x^2 - 4 = 0", 'x')
simplified = processor.simplify_expression(r"\frac{x^2-1}{x-1}")
derivative = processor.compute_derivative("x^3", 'x')
integral = processor.compute_integral("x^2", 'x')
```

## 支持的符号

- **数字**：0-9
- **拉丁字母**：a-z, A-Z
- **希腊字母**：α, β, γ, δ, θ, λ, μ, π, σ, φ, ω 等
- **运算符**：+, -, ×, ÷, =, ≠, <, >, ≤, ≥, ±
- **特殊符号**：√, ∑, ∏, ∫, ∞, ∂, ∇
- **括号**：(), [], {}

## 运行测试

```bash
cd tests
python -m pytest test_modules.py -v
```

## 运行示例

```bash
cd examples
python demo_usage.py
```

## 配置说明

在 `src/config.py` 中可以调整各模块的参数：

```python
from src.config import SystemConfig, PreprocessingConfig

# 自定义配置
config = SystemConfig()
config.preprocessing.sauvola_window_size = 31
config.preprocessing.sauvola_k = 0.3
config.recognition.svm_c = 100.0
```

## 注意事项

1. **符号识别需要训练**：首次使用需要准备训练数据并训练分类器
2. **数据集推荐**：可使用 CROHME 或 HASYv2 数据集进行训练
3. **SymPy 依赖**：语义理解功能需要安装 SymPy 库

## 扩展开发

### 添加新的符号类别

1. 在 `config.py` 的 `SYMBOL_TO_LATEX` 中添加映射
2. 准备对应的训练样本
3. 重新训练识别模型

### 自定义预处理流程

```python
class CustomPreprocessor(ImagePreprocessor):
    def process(self, image, return_intermediate=False):
        # 自定义预处理逻辑
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = self.custom_binarization(gray)
        return binary
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 参考资料

- [CROHME 数据集](https://www.isical.ac.in/~crohme/)
- [HASYv2 数据集](https://zenodo.org/record/259444)
- [SymPy 文档](https://docs.sympy.org/)
- [OpenCV 文档](https://docs.opencv.org/)
