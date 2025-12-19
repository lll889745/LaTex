# 模型存储目录
# 训练后的模型将保存在此目录

本目录用于存储训练后的符号识别模型文件（.pkl 格式）。

## 使用方法

训练并保存模型：
```python
from src.recognition import SymbolRecognizer

recognizer = SymbolRecognizer()
recognizer.train(images, labels, classifier_type='svm')
recognizer.save_model('models/symbol_classifier.pkl')
```

加载模型：
```python
recognizer.load_model('models/symbol_classifier.pkl')
```
