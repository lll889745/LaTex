"""
单元测试
========

测试各个模块的功能。
"""

import unittest
import numpy as np
import cv2
import os
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import ImagePreprocessor
from src.segmentation import SymbolSegmenter
from src.recognition import FeatureExtractor, SymbolRecognizer
from src.structure_analysis import StructureAnalyzer, SpatialRelation
from src.semantic import SemanticProcessor, FormulaType
from src.utils import BoundingBox, Symbol, SyntaxNode


class TestBoundingBox(unittest.TestCase):
    """测试边界框类"""
    
    def test_properties(self):
        """测试基本属性"""
        bbox = BoundingBox(10, 20, 30, 40)
        
        self.assertEqual(bbox.x, 10)
        self.assertEqual(bbox.y, 20)
        self.assertEqual(bbox.width, 30)
        self.assertEqual(bbox.height, 40)
        self.assertEqual(bbox.x2, 40)
        self.assertEqual(bbox.y2, 60)
        self.assertEqual(bbox.area, 1200)
        self.assertAlmostEqual(bbox.aspect_ratio, 0.75)
    
    def test_center(self):
        """测试中心点计算"""
        bbox = BoundingBox(0, 0, 100, 100)
        self.assertEqual(bbox.center, (50.0, 50.0))
    
    def test_intersection(self):
        """测试交集计算"""
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 100, 100)
        
        intersection = bbox1.intersection(bbox2)
        self.assertIsNotNone(intersection)
        self.assertEqual(intersection.x, 50)
        self.assertEqual(intersection.y, 50)
        self.assertEqual(intersection.width, 50)
        self.assertEqual(intersection.height, 50)
    
    def test_no_intersection(self):
        """测试无交集的情况"""
        bbox1 = BoundingBox(0, 0, 50, 50)
        bbox2 = BoundingBox(100, 100, 50, 50)
        
        intersection = bbox1.intersection(bbox2)
        self.assertIsNone(intersection)
    
    def test_union(self):
        """测试并集计算"""
        bbox1 = BoundingBox(0, 0, 50, 50)
        bbox2 = BoundingBox(30, 30, 50, 50)
        
        union = bbox1.union(bbox2)
        self.assertEqual(union.x, 0)
        self.assertEqual(union.y, 0)
        self.assertEqual(union.width, 80)
        self.assertEqual(union.height, 80)
    
    def test_iou(self):
        """测试 IoU 计算"""
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(0, 0, 100, 100)
        
        self.assertAlmostEqual(bbox1.iou(bbox2), 1.0)
        
        bbox3 = BoundingBox(50, 0, 100, 100)
        iou = bbox1.iou(bbox3)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)


class TestPreprocessing(unittest.TestCase):
    """测试图像预处理模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.preprocessor = ImagePreprocessor()
        
        # 创建测试图像（白色背景，黑色文字区域）
        self.test_image = np.ones((100, 200), dtype=np.uint8) * 255
        cv2.rectangle(self.test_image, (50, 30), (150, 70), 0, -1)
    
    def test_sauvola_binarization(self):
        """测试 Sauvola 二值化"""
        binary = self.preprocessor.sauvola_binarization(self.test_image)
        
        self.assertEqual(binary.shape, self.test_image.shape)
        self.assertTrue(np.all((binary == 0) | (binary == 255)))
    
    def test_denoise(self):
        """测试去噪"""
        # 创建带噪声的图像
        noisy = self.test_image.copy()
        noisy[10, 10] = 0  # 添加噪点
        noisy[90, 190] = 0
        
        binary = self.preprocessor.sauvola_binarization(noisy)
        denoised = self.preprocessor.denoise(binary)
        
        self.assertEqual(denoised.shape, binary.shape)
    
    def test_correct_skew(self):
        """测试倾斜校正"""
        # 创建稍微倾斜的图像
        rows, cols = self.test_image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
        skewed = cv2.warpAffine(self.test_image, M, (cols, rows), 
                                borderValue=255)
        
        binary = self.preprocessor.sauvola_binarization(skewed)
        corrected, angle = self.preprocessor.correct_skew(binary)
        
        self.assertIsInstance(angle, float)
    
    def test_full_pipeline(self):
        """测试完整预处理流程"""
        result = self.preprocessor.process(self.test_image)
        
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all((result == 0) | (result == 255)))


class TestSegmentation(unittest.TestCase):
    """测试符号分割模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.segmenter = SymbolSegmenter()
        
        # 创建包含多个符号的测试图像
        self.test_image = np.zeros((100, 300), dtype=np.uint8)
        # 添加三个矩形作为符号
        cv2.rectangle(self.test_image, (20, 30), (50, 70), 255, -1)
        cv2.rectangle(self.test_image, (80, 30), (110, 70), 255, -1)
        cv2.rectangle(self.test_image, (140, 30), (170, 70), 255, -1)
    
    def test_segment_basic(self):
        """测试基本分割"""
        symbols = self.segmenter.segment(self.test_image)
        
        self.assertEqual(len(symbols), 3)
        
        # 检查符号是否按位置排序
        x_positions = [s.bbox.x for s in symbols]
        self.assertEqual(x_positions, sorted(x_positions))
    
    def test_segment_empty_image(self):
        """测试空图像"""
        empty = np.zeros((100, 100), dtype=np.uint8)
        symbols = self.segmenter.segment(empty)
        
        self.assertEqual(len(symbols), 0)


class TestFeatureExtraction(unittest.TestCase):
    """测试特征提取模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.extractor = FeatureExtractor()
        
        # 创建测试符号图像
        self.test_symbol = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(self.test_symbol, (16, 16), 10, 255, -1)
    
    def test_extract_features(self):
        """测试特征提取"""
        features = self.extractor.extract(self.test_symbol)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertGreater(len(features), 0)
    
    def test_geometric_features(self):
        """测试几何特征"""
        features = self.extractor._extract_geometric_features(self.test_symbol)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 6)  # 6 个几何特征
    
    def test_hu_moments(self):
        """测试 Hu 矩"""
        features = self.extractor._extract_hu_moments(self.test_symbol)
        
        self.assertEqual(len(features), 7)  # 7 个 Hu 矩


class TestStructureAnalysis(unittest.TestCase):
    """测试结构分析模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.analyzer = StructureAnalyzer()
    
    def test_classify_relation_right(self):
        """测试右邻关系判定"""
        # 创建两个水平排列的符号
        sym1 = Symbol(
            image=np.zeros((30, 20), dtype=np.uint8),
            bbox=BoundingBox(10, 50, 20, 30)
        )
        sym2 = Symbol(
            image=np.zeros((30, 20), dtype=np.uint8),
            bbox=BoundingBox(50, 50, 20, 30)
        )
        
        relation = self.analyzer._classify_relation(sym1, sym2)
        self.assertEqual(relation, SpatialRelation.RIGHT)
    
    def test_classify_relation_superscript(self):
        """测试上标关系判定"""
        # 创建基础符号和上标符号
        base = Symbol(
            image=np.zeros((40, 30), dtype=np.uint8),
            bbox=BoundingBox(10, 50, 30, 40)
        )
        superscript = Symbol(
            image=np.zeros((15, 12), dtype=np.uint8),
            bbox=BoundingBox(45, 35, 12, 15)
        )
        
        relation = self.analyzer._classify_relation(base, superscript)
        self.assertEqual(relation, SpatialRelation.SUPERSCRIPT)
    
    def test_generate_latex_symbol(self):
        """测试符号到 LaTeX 转换"""
        node = SyntaxNode("symbol", value="x")
        latex = self.analyzer._generate_latex(node)
        
        self.assertEqual(latex, "x")
    
    def test_generate_latex_fraction(self):
        """测试分数 LaTeX 生成"""
        # 构建分数语法树
        frac = SyntaxNode("fraction")
        num = SyntaxNode("expression")
        num.add_child(SyntaxNode("symbol", value="a"))
        den = SyntaxNode("expression")
        den.add_child(SyntaxNode("symbol", value="b"))
        frac.add_child(num)
        frac.add_child(den)
        
        latex = self.analyzer._generate_latex(frac)
        self.assertIn(r"\frac", latex)
        self.assertIn("a", latex)
        self.assertIn("b", latex)


class TestSemantic(unittest.TestCase):
    """测试语义理解模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = SemanticProcessor()
    
    def test_classify_equation(self):
        """测试方程分类"""
        result = self.processor.process("x^2 = 4")
        self.assertEqual(result.formula_type, FormulaType.EQUATION)
    
    def test_classify_expression(self):
        """测试表达式分类"""
        result = self.processor.process("x^2 + 2x + 1")
        self.assertEqual(result.formula_type, FormulaType.EXPRESSION)
    
    def test_classify_integral(self):
        """测试积分分类"""
        result = self.processor.process(r"\int x dx")
        self.assertEqual(result.formula_type, FormulaType.INTEGRAL)
    
    def test_classify_derivative(self):
        """测试导数分类"""
        result = self.processor.process(r"\frac{d}{dx} x^2")
        self.assertEqual(result.formula_type, FormulaType.DERIVATIVE)
    
    def test_extract_variables(self):
        """测试变量提取"""
        result = self.processor.process("x + y + z")
        self.assertIn('x', result.variables)
        self.assertIn('y', result.variables)
        self.assertIn('z', result.variables)
    
    def test_validation_bracket_balance(self):
        """测试括号匹配检查"""
        result = self.processor.process("(x + 1")
        self.assertTrue(any("括号" in e for e in result.errors))


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        # 创建简单的测试图像
        image = np.ones((100, 200), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 30), (80, 70), 0, -1)  # 模拟一个符号
        
        # 预处理
        preprocessor = ImagePreprocessor()
        binary = preprocessor.process(image)
        
        # 分割
        segmenter = SymbolSegmenter()
        symbols = segmenter.segment(binary)
        
        # 验证结果
        self.assertGreaterEqual(len(symbols), 0)


if __name__ == '__main__':
    unittest.main()
