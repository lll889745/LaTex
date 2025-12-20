"""
示例代码
========

展示如何使用各个模块。
"""

import numpy as np
import cv2
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import ImagePreprocessor, preprocess_image
from src.segmentation import SymbolSegmenter, segment_symbols
from src.recognition import SymbolRecognizer, FeatureExtractor
from src.structure_analysis import StructureAnalyzer, analyze_structure
from src.semantic import SemanticProcessor, process_semantic


def example_preprocessing():
    """
    示例 1: 图像预处理
    
    展示如何对手写公式图像进行预处理。
    """
    print("=" * 50)
    print("示例 1: 图像预处理")
    print("=" * 50)
    
    # 创建模拟的手写图像
    image = np.ones((200, 400), dtype=np.uint8) * 200  # 灰色背景
    
    # 添加一些"手写"内容
    cv2.putText(image, 'x^2 + y = 5', (50, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 3)
    
    # 添加噪声
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 方法 1: 使用类接口
    preprocessor = ImagePreprocessor()
    
    # 获取完整的预处理结果
    result = preprocessor.process(image, return_intermediate=True)
    
    print(f"原始图像尺寸: {image.shape}")
    print(f"检测到的倾斜角度: {result['skew_angle']:.2f}°")
    print(f"二值化图像中的前景像素: {np.sum(result['binary'] > 0)}")
    
    # 方法 2: 使用便捷函数
    binary = preprocess_image(image)
    print(f"处理后图像尺寸: {binary.shape}")
    
    # 单独使用各个功能
    gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
    
    # Sauvola 二值化
    binary_sauvola = preprocessor.sauvola_binarization(gray)
    print(f"Sauvola 二值化完成")
    
    # 去噪
    denoised = preprocessor.denoise(binary_sauvola)
    print(f"去噪完成")
    
    # 骨架提取
    skeleton = preprocessor.extract_skeleton(denoised)
    print(f"骨架提取完成，骨架像素数: {np.sum(skeleton > 0)}")
    
    print()


def example_segmentation():
    """
    示例 2: 符号分割
    
    展示如何将预处理后的图像分割为独立符号。
    """
    print("=" * 50)
    print("示例 2: 符号分割")
    print("=" * 50)
    
    # 创建包含多个符号的二值图像
    binary = np.zeros((100, 300), dtype=np.uint8)
    
    # 添加三个符号
    cv2.rectangle(binary, (20, 30), (45, 70), 255, -1)   # 第一个符号
    cv2.rectangle(binary, (60, 30), (85, 70), 255, -1)   # 第二个符号
    cv2.circle(binary, (120, 50), 20, 255, -1)            # 第三个符号（圆形）
    
    # 添加一个上标
    cv2.rectangle(binary, (90, 20), (105, 35), 255, -1)
    
    # 添加模拟的 "i" 符号
    cv2.rectangle(binary, (180, 35), (190, 70), 255, -1)  # 主体
    cv2.circle(binary, (185, 25), 5, 255, -1)              # 点
    
    # 分割符号
    segmenter = SymbolSegmenter()
    symbols = segmenter.segment(binary)
    
    print(f"检测到 {len(symbols)} 个符号")
    
    for i, symbol in enumerate(symbols):
        bbox = symbol.bbox
        print(f"符号 {i+1}:")
        print(f"  位置: ({bbox.x}, {bbox.y})")
        print(f"  大小: {bbox.width} x {bbox.height}")
        print(f"  中心: ({bbox.center_x:.1f}, {bbox.center_y:.1f})")
        print(f"  是否为分数线: {symbol.is_fraction_line}")
        print(f"  是否为根号: {symbol.is_sqrt_symbol}")
    
    # 检测特殊结构
    print("\n检测分数结构...")
    # 如果有分数线，可以提取分子分母
    for symbol in symbols:
        if symbol.is_fraction_line:
            numerator, denominator = segmenter.extract_fraction_parts(
                symbol, symbols
            )
            print(f"分子包含 {len(numerator)} 个符号")
            print(f"分母包含 {len(denominator)} 个符号")
    
    print()


def example_feature_extraction():
    """
    示例 3: 特征提取
    
    展示如何从符号图像中提取特征。
    """
    print("=" * 50)
    print("示例 3: 特征提取")
    print("=" * 50)
    
    # 创建不同形状的符号
    symbols = {
        '圆形 (0)': create_circle_image(),
        '矩形 (1)': create_rectangle_image(),
        '十字 (+)': create_cross_image(),
    }
    
    extractor = FeatureExtractor()
    
    for name, image in symbols.items():
        features = extractor.extract(image)
        
        print(f"\n{name}:")
        print(f"  特征维度: {len(features)}")
        
        # 显示部分特征
        print(f"  几何特征 (前6维):")
        print(f"    填充率: {features[0]:.3f}")
        print(f"    重心 X: {features[1]:.3f}")
        print(f"    重心 Y: {features[2]:.3f}")
        print(f"    欧拉数: {features[3]:.3f}")
        print(f"    宽高比: {features[4]:.3f}")
        print(f"    角度: {features[5]:.3f}")
    
    print()


def create_circle_image():
    """创建圆形符号图像"""
    img = np.zeros((32, 32), dtype=np.uint8)
    cv2.circle(img, (16, 16), 12, 255, -1)
    return img


def create_rectangle_image():
    """创建矩形符号图像"""
    img = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(img, (8, 4), (24, 28), 255, -1)
    return img


def create_cross_image():
    """创建十字符号图像"""
    img = np.zeros((32, 32), dtype=np.uint8)
    cv2.line(img, (16, 4), (16, 28), 255, 3)
    cv2.line(img, (4, 16), (28, 16), 255, 3)
    return img


def example_structure_analysis():
    """
    示例 4: 结构分析
    
    展示如何分析符号之间的空间关系并生成 LaTeX。
    """
    print("=" * 50)
    print("示例 4: 结构分析")
    print("=" * 50)
    
    from src.utils import Symbol, BoundingBox
    
    # 创建模拟的符号列表 (x^2 + y)
    symbols = [
        # x
        Symbol(
            image=np.zeros((40, 30), dtype=np.uint8),
            bbox=BoundingBox(10, 50, 30, 40),
            label='x'
        ),
        # 上标 2
        Symbol(
            image=np.zeros((20, 15), dtype=np.uint8),
            bbox=BoundingBox(45, 30, 15, 20),
            label='2'
        ),
        # +
        Symbol(
            image=np.zeros((30, 25), dtype=np.uint8),
            bbox=BoundingBox(80, 55, 25, 30),
            label='+'
        ),
        # y
        Symbol(
            image=np.zeros((40, 30), dtype=np.uint8),
            bbox=BoundingBox(130, 50, 30, 40),
            label='y'
        ),
    ]
    
    # 分析结构
    analyzer = StructureAnalyzer()
    syntax_tree, latex = analyzer.analyze(symbols)
    
    print(f"生成的 LaTeX: {latex}")
    print(f"\n语法树结构:")
    from src.utils import visualize_syntax_tree
    print(visualize_syntax_tree(syntax_tree))
    
    # 使用便捷函数
    _, latex2 = analyze_structure(symbols)
    print(f"便捷函数生成的 LaTeX: {latex2}")
    
    print()


def example_semantic_understanding():
    """
    示例 5: 语义理解
    
    展示如何对 LaTeX 公式进行语义分析和计算。
    """
    print("=" * 50)
    print("示例 5: 语义理解")
    print("=" * 50)
    
    processor = SemanticProcessor()
    
    # 测试不同类型的公式
    test_formulas = [
        # 方程
        ("x^2 - 4 = 0", "一元二次方程"),
        
        # 表达式
        ("x^2 + 2x + 1", "完全平方式"),
        
        # 分数
        (r"\frac{x^2 - 1}{x - 1}", "分式"),
        
        # 三角函数
        (r"\sin^2(x) + \cos^2(x)", "三角恒等式"),
        
        # 导数
        (r"\frac{d}{dx} x^3", "导数"),
        
        # 积分
        (r"\int x^2 dx", "不定积分"),
    ]
    
    for latex, description in test_formulas:
        print(f"\n{'='*40}")
        print(f"公式: {latex}")
        print(f"描述: {description}")
        
        result = processor.process(latex)
        
        print(f"类型: {result.formula_type.value}")
        print(f"变量: {result.variables}")
        print(f"运算: {result.operations}")
        print(f"解释: {result.interpretation}")
        
        if result.solution:
            print(f"计算结果: {result.solution}")
        
        if result.simplified:
            print(f"化简形式: {result.simplified}")
        
        if result.errors:
            print(f"错误: {result.errors}")
        
        if result.warnings:
            print(f"警告: {result.warnings}")
    
    # 单独使用计算功能
    print(f"\n{'='*40}")
    print("单独使用计算功能:")
    
    # 求解方程
    solutions = processor.solve_equation("x^2 - 4 = 0", 'x')
    if solutions:
        print(f"x^2 - 4 = 0 的解: {solutions}")
    
    # 化简表达式
    simplified = processor.simplify_expression(r"\frac{x^2 - 1}{x - 1}")
    if simplified:
        print(f"(x^2 - 1)/(x - 1) 化简为: {simplified}")
    
    # 求导数
    derivative = processor.compute_derivative("x^3", 'x')
    if derivative:
        print(f"d/dx(x^3) = {derivative}")
    
    # 求积分
    integral = processor.compute_integral("x^2", 'x')
    if integral:
        print(f"∫x^2 dx = {integral}")
    
    # 求极限
    limit_result = processor.evaluate_limit(r"\frac{\sin(x)}{x}", 'x', '0')
    if limit_result:
        print(f"lim(x→0) sin(x)/x = {limit_result}")
    
    print()


def example_complete_pipeline():
    """
    示例 6: 完整流程
    
    展示从图像输入到语义理解的完整流程。
    """
    print("=" * 50)
    print("示例 6: 完整流程")
    print("=" * 50)
    
    # 1. 创建模拟的手写公式图像
    print("\n步骤 1: 创建测试图像")
    image = np.ones((150, 400), dtype=np.uint8) * 240
    cv2.putText(image, 'x + 2 = 5', (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, 30, 4)
    
    # 添加轻微噪声
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print(f"图像尺寸: {image.shape}")
    
    # 2. 图像预处理
    print("\n步骤 2: 图像预处理")
    preprocessor = ImagePreprocessor()
    binary = preprocessor.process(image)
    print(f"二值化完成，前景像素: {np.sum(binary > 0)}")
    
    # 3. 符号分割
    print("\n步骤 3: 符号分割")
    segmenter = SymbolSegmenter()
    symbols = segmenter.segment(binary)
    print(f"检测到 {len(symbols)} 个符号")
    
    # 4. 符号识别
    print("\n步骤 4: 符号识别")
    # 由于没有训练模型，手动设置标签
    expected_labels = ['x', '+', '2', '=', '5']
    for i, symbol in enumerate(symbols):
        if i < len(expected_labels):
            symbol.label = expected_labels[i]
            symbol.confidence = 0.9
    
    for i, sym in enumerate(symbols):
        print(f"  符号 {i+1}: '{sym.label}' (置信度: {sym.confidence:.2f})")
    
    # 5. 结构分析
    print("\n步骤 5: 结构分析")
    analyzer = StructureAnalyzer()
    syntax_tree, latex = analyzer.analyze(symbols)
    print(f"生成的 LaTeX: {latex}")
    
    # 6. 语义理解
    print("\n步骤 6: 语义理解")
    semantic = SemanticProcessor()
    result = semantic.process(latex)
    
    print(f"公式类型: {result.formula_type.value}")
    print(f"变量: {result.variables}")
    print(f"解释: {result.interpretation}")
    
    if result.solution:
        print(f"计算结果: {result.solution}")
    
    print("\n完整流程演示完成！")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("手写数学公式识别与语义理解系统 - 示例代码")
    print("=" * 60 + "\n")
    
    # 运行各个示例
    example_preprocessing()
    example_segmentation()
    example_feature_extraction()
    example_structure_analysis()
    example_semantic_understanding()
    example_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
