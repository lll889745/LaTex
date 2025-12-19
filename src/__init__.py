"""
手写数学公式识别与语义理解系统
================================

该系统将手写数学公式图像转换为结构化的 LaTeX 代码，
并进一步实现语义理解功能（如公式求解、化简）。

模块结构：
- preprocessing: 图像预处理
- segmentation: 符号分割
- recognition: 符号识别
- structure_analysis: 结构分析
- semantic: 语义理解
"""

from .preprocessing import ImagePreprocessor
from .segmentation import SymbolSegmenter
from .recognition import SymbolRecognizer
from .structure_analysis import StructureAnalyzer
from .semantic import SemanticProcessor

__version__ = "1.0.0"
__all__ = [
    "ImagePreprocessor",
    "SymbolSegmenter", 
    "SymbolRecognizer",
    "StructureAnalyzer",
    "SemanticProcessor"
]
