"""
结构分析模块
============

实现以下功能：
- 空间关系判定
- 语法树构建
- LaTeX 代码生成
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import StructureConfig, DEFAULT_CONFIG, SYMBOL_TO_LATEX, LATEX_TEMPLATES
from .utils import Symbol, BoundingBox, SyntaxNode, sort_symbols_by_position

logger = logging.getLogger(__name__)


class SpatialRelation(Enum):
    """空间关系类型"""
    RIGHT = "right"          # 右邻关系
    SUPERSCRIPT = "superscript"  # 上标关系
    SUBSCRIPT = "subscript"      # 下标关系
    ABOVE = "above"          # 正上方
    BELOW = "below"          # 正下方
    INSIDE = "inside"        # 内部关系
    NUMERATOR = "numerator"      # 分子关系
    DENOMINATOR = "denominator"  # 分母关系
    UNKNOWN = "unknown"      # 未知关系


class NodeType(Enum):
    """语法树节点类型"""
    EXPRESSION = "expression"    # 表达式
    EQUATION = "equation"        # 等式
    FRACTION = "fraction"        # 分数
    SQRT = "sqrt"                # 根号
    SUPERSCRIPT = "superscript"  # 上标
    SUBSCRIPT = "subscript"      # 下标
    SUM = "sum"                  # 求和
    PRODUCT = "product"          # 连乘
    INTEGRAL = "integral"        # 积分
    LIMIT = "limit"              # 极限
    SYMBOL = "symbol"            # 符号
    OPERATOR = "operator"        # 运算符
    GROUP = "group"              # 分组


class StructureAnalyzer:
    """
    结构分析器
    
    分析符号之间的空间关系，构建语法树，生成 LaTeX 代码。
    """
    
    def __init__(self, config: Optional[StructureConfig] = None):
        """
        初始化结构分析器
        
        Args:
            config: 结构分析配置
        """
        self.config = config or DEFAULT_CONFIG.structure
        
        # 运算符优先级
        self.operator_precedence = {
            '=': 1, '<': 1, '>': 1, 'leq': 1, 'geq': 1, 'neq': 1,
            '+': 2, '-': 2, 'pm': 2,
            '*': 3, 'times': 3, 'cdot': 3, '/': 3, 'div': 3,
            '^': 4, '_': 4,
        }
    
    def analyze(self, symbols: List[Symbol]) -> Tuple[SyntaxNode, str]:
        """
        分析符号结构并生成 LaTeX
        
        Args:
            symbols: 识别后的符号列表
            
        Returns:
            (语法树根节点, LaTeX 字符串)
        """
        if not symbols:
            return SyntaxNode(NodeType.EXPRESSION.value), ""
        
        logger.info(f"开始结构分析，符号数: {len(symbols)}")
        
        # 步骤1：计算空间关系矩阵
        relations = self._compute_relations(symbols)
        
        # 步骤2：处理特殊结构（分数、根号等）
        symbols, groups = self._process_special_structures(symbols, relations)
        
        # 步骤3：构建语法树
        syntax_tree = self._build_syntax_tree(symbols, relations, groups)
        
        # 步骤4：生成 LaTeX
        latex = self._generate_latex(syntax_tree)
        
        logger.info(f"生成的 LaTeX: {latex}")
        
        return syntax_tree, latex
    
    def _compute_relations(self, symbols: List[Symbol]) -> Dict[Tuple[int, int], SpatialRelation]:
        """
        计算符号间的空间关系
        
        Args:
            symbols: 符号列表
            
        Returns:
            空间关系字典 {(i, j): relation}
        """
        relations = {}
        
        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                if i == j:
                    continue
                
                relation = self._classify_relation(sym_a, sym_b)
                if relation != SpatialRelation.UNKNOWN:
                    relations[(i, j)] = relation
        
        return relations
    
    def _classify_relation(self, symbol_a: Symbol, symbol_b: Symbol) -> SpatialRelation:
        """
        判断 symbol_b 相对于 symbol_a 的空间关系
        
        Args:
            symbol_a: 基准符号
            symbol_b: 目标符号
            
        Returns:
            空间关系类型
        """
        bbox_a = symbol_a.bbox
        bbox_b = symbol_b.bbox
        
        # 计算相对位置
        dx = bbox_b.center_x - bbox_a.center_x
        dy = bbox_b.center_y - bbox_a.center_y
        
        # 计算尺寸比例
        size_a = max(bbox_a.width, bbox_a.height)
        size_b = max(bbox_b.width, bbox_b.height)
        size_ratio = size_b / max(size_a, 1)
        
        # 计算重叠
        v_overlap = bbox_a.vertical_overlap_ratio(bbox_b)
        h_overlap = bbox_a.horizontal_overlap_ratio(bbox_b)
        
        # 归一化距离
        norm_dx = dx / max(size_a, 1)
        norm_dy = dy / max(size_a, 1)
        
        # 分数线处理
        if symbol_a.is_fraction_line:
            if dy < 0 and abs(norm_dy) < 1.5:
                return SpatialRelation.NUMERATOR
            elif dy > 0 and abs(norm_dy) < 1.5:
                return SpatialRelation.DENOMINATOR
        
        # 上标判定
        if (norm_dx > 0 and norm_dy < -self.config.superscript_y_threshold and
            size_ratio < self.config.script_size_ratio and
            bbox_b.y2 < bbox_a.center_y):
            return SpatialRelation.SUPERSCRIPT
        
        # 下标判定
        if (norm_dx > 0 and norm_dy > self.config.subscript_y_threshold and
            size_ratio < self.config.script_size_ratio and
            bbox_b.y > bbox_a.center_y):
            return SpatialRelation.SUBSCRIPT
        
        # 右邻判定
        if (norm_dx > 0.3 and
            v_overlap > self.config.vertical_overlap_threshold):
            return SpatialRelation.RIGHT
        
        # 正上方
        if (abs(norm_dx) < 0.5 and norm_dy < -0.5 and
            h_overlap > 0.3):
            return SpatialRelation.ABOVE
        
        # 正下方
        if (abs(norm_dx) < 0.5 and norm_dy > 0.5 and
            h_overlap > 0.3):
            return SpatialRelation.BELOW
        
        return SpatialRelation.UNKNOWN
    
    def _process_special_structures(self, symbols: List[Symbol],
                                    relations: Dict) -> Tuple[List[Symbol], List[Dict]]:
        """
        处理特殊结构（分数、根号等）
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            (更新后的符号列表, 结构分组信息)
        """
        groups = []
        
        # 处理分数结构
        fraction_groups = self._find_fraction_groups(symbols, relations)
        groups.extend(fraction_groups)
        
        # 处理根号结构
        sqrt_groups = self._find_sqrt_groups(symbols, relations)
        groups.extend(sqrt_groups)
        
        # 处理求和/积分结构
        operator_groups = self._find_large_operator_groups(symbols, relations)
        groups.extend(operator_groups)
        
        return symbols, groups
    
    def _find_fraction_groups(self, symbols: List[Symbol],
                              relations: Dict) -> List[Dict]:
        """
        查找分数结构
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            分数分组列表
        """
        groups = []
        
        for i, symbol in enumerate(symbols):
            if symbol.is_fraction_line:
                numerator = []
                denominator = []
                
                for j, other in enumerate(symbols):
                    if i == j:
                        continue
                    
                    rel = relations.get((i, j))
                    if rel == SpatialRelation.NUMERATOR:
                        numerator.append(j)
                    elif rel == SpatialRelation.DENOMINATOR:
                        denominator.append(j)
                
                groups.append({
                    'type': 'fraction',
                    'line_idx': i,
                    'numerator_indices': numerator,
                    'denominator_indices': denominator
                })
        
        return groups
    
    def _find_sqrt_groups(self, symbols: List[Symbol],
                          relations: Dict) -> List[Dict]:
        """
        查找根号结构
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            根号分组列表
        """
        groups = []
        
        for i, symbol in enumerate(symbols):
            if symbol.is_sqrt_symbol:
                inner = []
                
                # 找到根号内部的符号
                sqrt_bbox = symbol.bbox
                for j, other in enumerate(symbols):
                    if i == j:
                        continue
                    
                    other_bbox = other.bbox
                    # 检查是否在根号覆盖范围内
                    if (other_bbox.center_x > sqrt_bbox.x + sqrt_bbox.width * 0.3 and
                        other_bbox.center_x < sqrt_bbox.x2 and
                        other_bbox.center_y > sqrt_bbox.y and
                        other_bbox.center_y < sqrt_bbox.y2):
                        inner.append(j)
                
                groups.append({
                    'type': 'sqrt',
                    'sqrt_idx': i,
                    'inner_indices': inner
                })
        
        return groups
    
    def _find_large_operator_groups(self, symbols: List[Symbol],
                                    relations: Dict) -> List[Dict]:
        """
        查找大型运算符结构（求和、积分等）
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            运算符分组列表
        """
        groups = []
        
        for i, symbol in enumerate(symbols):
            if symbol.is_sum_symbol or symbol.is_integral_symbol:
                upper_limit = []
                lower_limit = []
                operand = []
                
                for j, other in enumerate(symbols):
                    if i == j:
                        continue
                    
                    rel = relations.get((i, j))
                    if rel == SpatialRelation.ABOVE:
                        upper_limit.append(j)
                    elif rel == SpatialRelation.BELOW:
                        lower_limit.append(j)
                    elif rel == SpatialRelation.RIGHT:
                        operand.append(j)
                
                op_type = 'sum' if symbol.is_sum_symbol else 'integral'
                groups.append({
                    'type': op_type,
                    'operator_idx': i,
                    'upper_limit_indices': upper_limit,
                    'lower_limit_indices': lower_limit,
                    'operand_indices': operand
                })
        
        return groups
    
    def _build_syntax_tree(self, symbols: List[Symbol],
                           relations: Dict,
                           groups: List[Dict]) -> SyntaxNode:
        """
        构建语法树
        
        Args:
            symbols: 符号列表
            relations: 空间关系字典
            groups: 结构分组信息
            
        Returns:
            语法树根节点
        """
        # 按位置排序符号
        sorted_symbols = sort_symbols_by_position(symbols)
        
        # 标记已处理的符号
        processed = set()
        
        # 处理分组结构
        group_nodes = {}
        for group in groups:
            node = self._create_group_node(group, symbols, relations)
            group_nodes[group.get('line_idx') or group.get('sqrt_idx') or 
                       group.get('operator_idx')] = node
            
            # 标记组内符号为已处理
            for key in ['numerator_indices', 'denominator_indices', 
                       'inner_indices', 'upper_limit_indices',
                       'lower_limit_indices', 'operand_indices']:
                if key in group:
                    processed.update(group[key])
        
        # 构建主表达式
        root = SyntaxNode(NodeType.EXPRESSION.value)
        
        for symbol in sorted_symbols:
            idx = symbols.index(symbol)
            
            if idx in processed:
                continue
            
            if idx in group_nodes:
                root.add_child(group_nodes[idx])
            else:
                # 创建符号节点
                node = self._create_symbol_node(symbol, idx, symbols, relations)
                root.add_child(node)
            
            processed.add(idx)
        
        return root
    
    def _create_group_node(self, group: Dict, symbols: List[Symbol],
                           relations: Dict) -> SyntaxNode:
        """
        创建分组节点（分数、根号等）
        
        Args:
            group: 分组信息
            symbols: 符号列表
            relations: 空间关系字典
            
        Returns:
            语法树节点
        """
        group_type = group['type']
        
        if group_type == 'fraction':
            node = SyntaxNode(NodeType.FRACTION.value)
            
            # 分子
            num_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in group['numerator_indices']:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations)
                num_node.add_child(child)
            node.add_child(num_node)
            
            # 分母
            den_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in group['denominator_indices']:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations)
                den_node.add_child(child)
            node.add_child(den_node)
            
        elif group_type == 'sqrt':
            node = SyntaxNode(NodeType.SQRT.value)
            
            # 内部表达式
            inner_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in group['inner_indices']:
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations)
                inner_node.add_child(child)
            node.add_child(inner_node)
            
        elif group_type in ['sum', 'integral']:
            node_type = NodeType.SUM if group_type == 'sum' else NodeType.INTEGRAL
            node = SyntaxNode(node_type.value)
            
            # 下限
            lower_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in group.get('lower_limit_indices', []):
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations)
                lower_node.add_child(child)
            node.add_child(lower_node)
            
            # 上限
            upper_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in group.get('upper_limit_indices', []):
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations)
                upper_node.add_child(child)
            node.add_child(upper_node)
            
            # 被操作数
            operand_node = SyntaxNode(NodeType.EXPRESSION.value)
            for idx in group.get('operand_indices', []):
                child = self._create_symbol_node(symbols[idx], idx, symbols, relations)
                operand_node.add_child(child)
            node.add_child(operand_node)
        else:
            node = SyntaxNode(NodeType.GROUP.value)
        
        return node
    
    def _create_symbol_node(self, symbol: Symbol, idx: int,
                            symbols: List[Symbol],
                            relations: Dict) -> SyntaxNode:
        """
        创建符号节点
        
        Args:
            symbol: 符号
            idx: 符号索引
            symbols: 所有符号
            relations: 空间关系字典
            
        Returns:
            语法树节点
        """
        # 检查是否有上标或下标
        has_superscript = False
        has_subscript = False
        superscript_indices = []
        subscript_indices = []
        
        for (i, j), rel in relations.items():
            if i == idx:
                if rel == SpatialRelation.SUPERSCRIPT:
                    has_superscript = True
                    superscript_indices.append(j)
                elif rel == SpatialRelation.SUBSCRIPT:
                    has_subscript = True
                    subscript_indices.append(j)
        
        if has_superscript or has_subscript:
            # 创建带上下标的节点
            if has_superscript and has_subscript:
                node = SyntaxNode("subsuperscript")
            elif has_superscript:
                node = SyntaxNode(NodeType.SUPERSCRIPT.value)
            else:
                node = SyntaxNode(NodeType.SUBSCRIPT.value)
            
            # 基础符号
            base_node = SyntaxNode(NodeType.SYMBOL.value, value=symbol.label)
            node.add_child(base_node)
            
            # 下标
            if has_subscript:
                sub_node = SyntaxNode(NodeType.EXPRESSION.value)
                for j in subscript_indices:
                    child = SyntaxNode(NodeType.SYMBOL.value, value=symbols[j].label)
                    sub_node.add_child(child)
                node.add_child(sub_node)
            
            # 上标
            if has_superscript:
                sup_node = SyntaxNode(NodeType.EXPRESSION.value)
                for j in superscript_indices:
                    child = SyntaxNode(NodeType.SYMBOL.value, value=symbols[j].label)
                    sup_node.add_child(child)
                node.add_child(sup_node)
            
            return node
        
        # 简单符号节点
        return SyntaxNode(NodeType.SYMBOL.value, value=symbol.label)
    
    def _generate_latex(self, node: SyntaxNode, level: int = 0) -> str:
        """
        从语法树生成 LaTeX 代码
        
        Args:
            node: 语法树节点
            level: 递归深度
            
        Returns:
            LaTeX 字符串
        """
        node_type = node.node_type
        
        # 符号节点
        if node_type == NodeType.SYMBOL.value:
            return self._symbol_to_latex(node.value)
        
        # 表达式节点
        if node_type == NodeType.EXPRESSION.value:
            parts = [self._generate_latex(child, level + 1) for child in node.children]
            return ' '.join(parts)
        
        # 分数节点
        if node_type == NodeType.FRACTION.value:
            if len(node.children) >= 2:
                num = self._generate_latex(node.children[0], level + 1)
                den = self._generate_latex(node.children[1], level + 1)
                return LATEX_TEMPLATES['frac'].format(num=num, den=den)
            return ''
        
        # 根号节点
        if node_type == NodeType.SQRT.value:
            if node.children:
                inner = self._generate_latex(node.children[0], level + 1)
                return LATEX_TEMPLATES['sqrt'].format(inner=inner)
            return r'\sqrt{}'
        
        # 上标节点
        if node_type == NodeType.SUPERSCRIPT.value:
            if len(node.children) >= 2:
                base = self._generate_latex(node.children[0], level + 1)
                exp = self._generate_latex(node.children[1], level + 1)
                return LATEX_TEMPLATES['superscript'].format(base=base, exp=exp)
            return ''
        
        # 下标节点
        if node_type == NodeType.SUBSCRIPT.value:
            if len(node.children) >= 2:
                base = self._generate_latex(node.children[0], level + 1)
                sub = self._generate_latex(node.children[1], level + 1)
                return LATEX_TEMPLATES['subscript'].format(base=base, sub=sub)
            return ''
        
        # 上下标同时存在
        if node_type == "subsuperscript":
            if len(node.children) >= 3:
                base = self._generate_latex(node.children[0], level + 1)
                sub = self._generate_latex(node.children[1], level + 1)
                exp = self._generate_latex(node.children[2], level + 1)
                return LATEX_TEMPLATES['subsuperscript'].format(base=base, sub=sub, exp=exp)
            return ''
        
        # 求和节点
        if node_type == NodeType.SUM.value:
            if len(node.children) >= 3:
                lower = self._generate_latex(node.children[0], level + 1)
                upper = self._generate_latex(node.children[1], level + 1)
                operand = self._generate_latex(node.children[2], level + 1)
                return LATEX_TEMPLATES['sum'].format(lower=lower, upper=upper) + ' ' + operand
            return r'\sum'
        
        # 积分节点
        if node_type == NodeType.INTEGRAL.value:
            if len(node.children) >= 3:
                lower = self._generate_latex(node.children[0], level + 1)
                upper = self._generate_latex(node.children[1], level + 1)
                operand = self._generate_latex(node.children[2], level + 1)
                return LATEX_TEMPLATES['int'].format(lower=lower, upper=upper) + ' ' + operand
            return r'\int'
        
        # 分组节点
        if node_type == NodeType.GROUP.value:
            parts = [self._generate_latex(child, level + 1) for child in node.children]
            return ' '.join(parts)
        
        # 未知节点类型
        logger.warning(f"未知节点类型: {node_type}")
        return ''
    
    def _symbol_to_latex(self, symbol: str) -> str:
        """
        将符号转换为 LaTeX
        
        Args:
            symbol: 符号标签
            
        Returns:
            LaTeX 表示
        """
        if symbol is None:
            return ''
        
        # 检查是否在映射表中
        if symbol in SYMBOL_TO_LATEX:
            return SYMBOL_TO_LATEX[symbol]
        
        # 单个字符直接返回
        if len(symbol) == 1:
            # 特殊字符需要转义
            if symbol in '#$%&_{}':
                return '\\' + symbol
            return symbol
        
        # 数字直接返回
        if symbol.isdigit():
            return symbol
        
        # 其他情况
        return symbol


class LaTeXFormatter:
    """
    LaTeX 格式化器
    
    对生成的 LaTeX 代码进行格式化和美化。
    """
    
    def format(self, latex: str) -> str:
        """
        格式化 LaTeX 代码
        
        Args:
            latex: 原始 LaTeX 代码
            
        Returns:
            格式化后的 LaTeX 代码
        """
        # 去除多余空格
        latex = ' '.join(latex.split())
        
        # 修复括号
        latex = self._fix_brackets(latex)
        
        # 优化空格
        latex = self._optimize_spacing(latex)
        
        return latex
    
    def _fix_brackets(self, latex: str) -> str:
        """修复括号匹配"""
        # 简单的括号检查和修复
        open_count = latex.count('{') - latex.count('}')
        if open_count > 0:
            latex += '}' * open_count
        elif open_count < 0:
            latex = '{' * (-open_count) + latex
        
        return latex
    
    def _optimize_spacing(self, latex: str) -> str:
        """优化空格"""
        # 运算符周围添加适当空格
        operators = ['+', '-', '=', '<', '>']
        for op in operators:
            latex = latex.replace(f' {op} ', f' {op} ')
            latex = latex.replace(f'{op} ', f' {op} ')
            latex = latex.replace(f' {op}', f' {op} ')
        
        # 去除大括号内的多余空格
        import re
        latex = re.sub(r'\{\s+', '{', latex)
        latex = re.sub(r'\s+\}', '}', latex)
        
        return latex


def analyze_structure(symbols: List[Symbol],
                      config: Optional[StructureConfig] = None) -> Tuple[SyntaxNode, str]:
    """
    分析符号结构的便捷函数
    
    Args:
        symbols: 符号列表
        config: 结构分析配置
        
    Returns:
        (语法树根节点, LaTeX 字符串)
    """
    analyzer = StructureAnalyzer(config)
    return analyzer.analyze(symbols)
