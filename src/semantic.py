"""
语义理解模块
============

实现以下功能：
- 公式类型识别
- 符号计算（使用 SymPy）
- 错误检测
- 公式化简和求解
"""

import re
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from sympy import (
        symbols, Symbol, sympify, solve, simplify, expand, factor,
        diff, integrate, limit, series, summation, product,
        sin, cos, tan, log, exp, sqrt, pi, E, I, oo,
        Eq, Ne, Lt, Le, Gt, Ge, 
        Derivative, Integral, Limit, Sum, Product,
        Matrix, latex as sympy_latex
    )
    from sympy.parsing.latex import parse_latex
    from sympy.core.sympify import SympifyError
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logging.warning("SymPy 未安装，语义理解功能将受限")

from .config import SemanticConfig, DEFAULT_CONFIG
from .utils import SyntaxNode

logger = logging.getLogger(__name__)


class FormulaType(Enum):
    """公式类型枚举"""
    EQUATION = "equation"           # 等式/方程
    INEQUALITY = "inequality"       # 不等式
    EXPRESSION = "expression"       # 表达式
    FUNCTION_DEF = "function_def"   # 函数定义
    DERIVATIVE = "derivative"       # 导数
    INTEGRAL = "integral"           # 积分
    LIMIT = "limit"                 # 极限
    SERIES = "series"               # 级数
    SUM = "sum"                     # 求和
    PRODUCT = "product"             # 连乘
    MATRIX = "matrix"               # 矩阵
    UNKNOWN = "unknown"             # 未知


@dataclass
class SemanticResult:
    """语义理解结果"""
    original_latex: str                     # 原始 LaTeX
    formula_type: FormulaType               # 公式类型
    sympy_expr: Any = None                  # SymPy 表达式对象
    variables: List[str] = field(default_factory=list)      # 包含的变量列表
    constants: List[str] = field(default_factory=list)      # 包含的常数列表
    operations: List[str] = field(default_factory=list)     # 包含的运算类型
    solution: Optional[Dict] = None         # 求解结果
    simplified: Optional[str] = None        # 化简结果
    errors: List[str] = field(default_factory=list)         # 检测到的错误
    warnings: List[str] = field(default_factory=list)       # 警告信息
    interpretation: str = ""                # 解释说明
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'original_latex': self.original_latex,
            'formula_type': self.formula_type.value,
            'variables': self.variables,
            'constants': self.constants,
            'operations': self.operations,
            'solution': self.solution,
            'simplified': self.simplified,
            'errors': self.errors,
            'warnings': self.warnings,
            'interpretation': self.interpretation
        }


class FormulaTypeClassifier:
    """
    公式类型分类器
    
    根据 LaTeX 代码或语法树识别公式类型。
    """
    
    # 特征关键词
    EQUATION_SIGNS = ['=']
    INEQUALITY_SIGNS = ['<', '>', r'\leq', r'\geq', r'\neq', r'\le', r'\ge']
    DERIVATIVE_PATTERNS = [r"\\frac\{d", r"\\frac\{\\partial", r"'", r"''", r"'''"]
    INTEGRAL_PATTERNS = [r'\\int', r'\\iint', r'\\iiint', r'\\oint']
    LIMIT_PATTERNS = [r'\\lim']
    SUM_PATTERNS = [r'\\sum']
    PRODUCT_PATTERNS = [r'\\prod']
    MATRIX_PATTERNS = [r'\\begin\{matrix\}', r'\\begin\{pmatrix\}', 
                       r'\\begin\{bmatrix\}', r'\\begin\{vmatrix\}']
    
    def classify(self, latex: str, syntax_tree: Optional[SyntaxNode] = None) -> FormulaType:
        """
        分类公式类型
        
        Args:
            latex: LaTeX 代码
            syntax_tree: 语法树（可选）
            
        Returns:
            公式类型
        """
        # 检查矩阵
        for pattern in self.MATRIX_PATTERNS:
            if re.search(pattern, latex):
                return FormulaType.MATRIX
        
        # 检查极限
        for pattern in self.LIMIT_PATTERNS:
            if pattern in latex:
                return FormulaType.LIMIT
        
        # 检查积分
        for pattern in self.INTEGRAL_PATTERNS:
            if pattern in latex:
                return FormulaType.INTEGRAL
        
        # 检查导数
        for pattern in self.DERIVATIVE_PATTERNS:
            if re.search(pattern, latex):
                return FormulaType.DERIVATIVE
        
        # 检查求和
        for pattern in self.SUM_PATTERNS:
            if pattern in latex:
                return FormulaType.SUM
        
        # 检查连乘
        for pattern in self.PRODUCT_PATTERNS:
            if pattern in latex:
                return FormulaType.PRODUCT
        
        # 检查函数定义 f(x) = ...
        if re.search(r'[a-zA-Z]\s*\([^)]+\)\s*=', latex):
            return FormulaType.FUNCTION_DEF
        
        # 检查不等式
        for sign in self.INEQUALITY_SIGNS:
            if sign in latex:
                return FormulaType.INEQUALITY
        
        # 检查等式
        for sign in self.EQUATION_SIGNS:
            if sign in latex:
                return FormulaType.EQUATION
        
        # 默认为表达式
        return FormulaType.EXPRESSION


class FormulaValidator:
    """
    公式验证器
    
    检查公式是否存在数学错误。
    """
    
    def __init__(self, config: Optional[SemanticConfig] = None):
        """
        初始化验证器
        
        Args:
            config: 语义配置
        """
        self.config = config or DEFAULT_CONFIG.semantic
    
    def validate(self, latex: str, sympy_expr: Any = None) -> Tuple[List[str], List[str]]:
        """
        验证公式
        
        Args:
            latex: LaTeX 代码
            sympy_expr: SymPy 表达式（可选）
            
        Returns:
            (错误列表, 警告列表)
        """
        errors = []
        warnings = []
        
        # 括号匹配检查
        if self.config.check_bracket_balance:
            bracket_errors = self._check_bracket_balance(latex)
            errors.extend(bracket_errors)
        
        # 运算符使用检查
        operator_errors = self._check_operators(latex)
        errors.extend(operator_errors)
        
        # 除零检查
        if self.config.check_division_by_zero and sympy_expr:
            division_warnings = self._check_division_by_zero(sympy_expr)
            warnings.extend(division_warnings)
        
        # 未定义符号检查
        if self.config.check_undefined_symbols and sympy_expr:
            undefined_warnings = self._check_undefined_symbols(sympy_expr)
            warnings.extend(undefined_warnings)
        
        return errors, warnings
    
    def _check_bracket_balance(self, latex: str) -> List[str]:
        """检查括号是否匹配"""
        errors = []
        
        bracket_pairs = [
            ('(', ')'),
            ('[', ']'),
            ('{', '}'),
            (r'\{', r'\}'),
        ]
        
        for open_b, close_b in bracket_pairs:
            open_count = latex.count(open_b)
            close_count = latex.count(close_b)
            
            if open_count != close_count:
                errors.append(f"括号不匹配: '{open_b}' 出现 {open_count} 次, "
                            f"'{close_b}' 出现 {close_count} 次")
        
        return errors
    
    def _check_operators(self, latex: str) -> List[str]:
        """检查运算符使用"""
        errors = []
        
        # 连续运算符检查
        consecutive_patterns = [
            (r'\+\+', "连续加号 '++'"),
            (r'--', "连续减号 '--'（可能表示负负得正？）"),
            (r'\*\*', "连续乘号 '**'"),
            (r'//', "连续除号 '//'"),
            (r'\+\*', "'+*' 运算符组合错误"),
            (r'\*/','*/' + " 运算符组合错误"),
        ]
        
        for pattern, msg in consecutive_patterns:
            if re.search(pattern, latex):
                errors.append(msg)
        
        # 表达式开头或结尾的运算符
        if re.match(r'^\s*[\+\*\/]', latex):
            errors.append("表达式以运算符开头")
        
        if re.search(r'[\+\-\*\/]\s*$', latex):
            # 允许以减号结尾（可能是负号的一部分）
            if not re.search(r'-\s*$', latex):
                errors.append("表达式以运算符结尾")
        
        return errors
    
    def _check_division_by_zero(self, expr: Any) -> List[str]:
        """检查可能的除零情况"""
        warnings = []
        
        if not SYMPY_AVAILABLE:
            return warnings
        
        try:
            # 查找所有除法操作
            from sympy import Pow, Integer
            
            def find_divisions(e):
                if hasattr(e, 'args'):
                    for arg in e.args:
                        if isinstance(e, Pow) and e.exp == Integer(-1):
                            # 这是一个除法
                            denominator = e.base
                            # 检查分母是否可能为零
                            if denominator.is_zero:
                                warnings.append(f"除数可能为零: {denominator}")
                        find_divisions(arg)
            
            find_divisions(expr)
        except Exception as e:
            logger.debug(f"除零检查出错: {e}")
        
        return warnings
    
    def _check_undefined_symbols(self, expr: Any) -> List[str]:
        """检查未定义的符号"""
        warnings = []
        
        if not SYMPY_AVAILABLE:
            return warnings
        
        try:
            # 获取所有自由符号
            free_symbols = expr.free_symbols
            
            # 常见的数学符号（不需要警告）
            common_symbols = {'x', 'y', 'z', 'a', 'b', 'c', 't', 'n', 'm', 
                            'i', 'j', 'k', 'r', 's', 'theta', 'phi', 'alpha',
                            'beta', 'gamma', 'delta', 'epsilon', 'lambda'}
            
            for sym in free_symbols:
                if str(sym).lower() not in common_symbols:
                    # 不是常见符号，给出提示
                    pass  # 暂不警告，因为可能是用户定义的变量
        
        except Exception as e:
            logger.debug(f"符号检查出错: {e}")
        
        return warnings


class SemanticProcessor:
    """
    语义处理器
    
    对识别出的公式进行语义理解和计算。
    """
    
    def __init__(self, config: Optional[SemanticConfig] = None):
        """
        初始化语义处理器
        
        Args:
            config: 语义配置
        """
        self.config = config or DEFAULT_CONFIG.semantic
        self.classifier = FormulaTypeClassifier()
        self.validator = FormulaValidator(config)
    
    def process(self, latex: str, 
                syntax_tree: Optional[SyntaxNode] = None) -> SemanticResult:
        """
        处理公式并返回语义理解结果
        
        Args:
            latex: LaTeX 代码
            syntax_tree: 语法树（可选）
            
        Returns:
            语义理解结果
        """
        result = SemanticResult(
            original_latex=latex,
            formula_type=FormulaType.UNKNOWN
        )
        
        if not latex.strip():
            result.errors.append("空公式")
            return result
        
        # 1. 分类公式类型
        result.formula_type = self.classifier.classify(latex, syntax_tree)
        
        # 2. 解析为 SymPy 表达式
        if SYMPY_AVAILABLE:
            try:
                result.sympy_expr = self.latex_to_sympy(latex)
            except Exception as e:
                result.warnings.append(f"无法解析公式: {str(e)}")
        
        # 3. 提取变量和常数
        if result.sympy_expr:
            result.variables = self._extract_variables(result.sympy_expr)
            result.operations = self._extract_operations(result.sympy_expr)
        
        # 4. 验证公式
        errors, warnings = self.validator.validate(latex, result.sympy_expr)
        result.errors.extend(errors)
        result.warnings.extend(warnings)
        
        # 5. 尝试计算/化简
        if result.sympy_expr and not result.errors:
            result = self._perform_computation(result)
        
        # 6. 生成解释
        result.interpretation = self._generate_interpretation(result)
        
        return result
    
    def latex_to_sympy(self, latex: str) -> Any:
        """
        将 LaTeX 转换为 SymPy 表达式
        
        Args:
            latex: LaTeX 代码
            
        Returns:
            SymPy 表达式
        """
        if not SYMPY_AVAILABLE:
            raise RuntimeError("SymPy 未安装")
        
        # 预处理 LaTeX
        latex = self._preprocess_latex(latex)
        
        try:
            expr = parse_latex(latex)
            return expr
        except Exception as e:
            logger.warning(f"LaTeX 解析失败: {e}")
            # 尝试使用 sympify
            try:
                # 简单转换
                simple = self._latex_to_simple(latex)
                return sympify(simple)
            except:
                raise
    
    def _preprocess_latex(self, latex: str) -> str:
        """
        预处理 LaTeX 代码以便于解析
        
        Args:
            latex: 原始 LaTeX
            
        Returns:
            处理后的 LaTeX
        """
        # 替换一些常见的变体
        replacements = [
            (r'\cdot', r' \cdot '),
            (r'\times', r' \times '),
            (r'\div', r' / '),
            (r'\pm', r' \pm '),
        ]
        
        for old, new in replacements:
            latex = latex.replace(old, new)
        
        return latex.strip()
    
    def _latex_to_simple(self, latex: str) -> str:
        """
        将 LaTeX 转换为简单的数学表达式
        
        Args:
            latex: LaTeX 代码
            
        Returns:
            简单表达式字符串
        """
        result = latex
        
        # 移除 LaTeX 命令
        result = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', result)
        result = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', result)
        result = re.sub(r'\\sin', 'sin', result)
        result = re.sub(r'\\cos', 'cos', result)
        result = re.sub(r'\\tan', 'tan', result)
        result = re.sub(r'\\log', 'log', result)
        result = re.sub(r'\\ln', 'log', result)
        result = re.sub(r'\\exp', 'exp', result)
        result = re.sub(r'\\pi', 'pi', result)
        result = re.sub(r'\\infty', 'oo', result)
        result = re.sub(r'\^', '**', result)
        result = re.sub(r'\\times', '*', result)
        result = re.sub(r'\\cdot', '*', result)
        result = re.sub(r'\\left', '', result)
        result = re.sub(r'\\right', '', result)
        result = re.sub(r'\\[a-zA-Z]+', '', result)  # 移除其他命令
        result = re.sub(r'[{}]', '', result)  # 移除大括号
        
        return result
    
    def _extract_variables(self, expr: Any) -> List[str]:
        """提取表达式中的变量"""
        if not SYMPY_AVAILABLE:
            return []
        
        try:
            free_symbols = expr.free_symbols
            return sorted([str(s) for s in free_symbols])
        except:
            return []
    
    def _extract_operations(self, expr: Any) -> List[str]:
        """提取表达式中的运算类型"""
        if not SYMPY_AVAILABLE:
            return []
        
        operations = set()
        
        try:
            expr_str = str(expr)
            
            if '+' in expr_str:
                operations.add('addition')
            if '-' in expr_str:
                operations.add('subtraction')
            if '*' in expr_str:
                operations.add('multiplication')
            if '/' in expr_str:
                operations.add('division')
            if '**' in expr_str:
                operations.add('exponentiation')
            if 'sqrt' in expr_str:
                operations.add('square_root')
            if 'sin' in expr_str or 'cos' in expr_str or 'tan' in expr_str:
                operations.add('trigonometric')
            if 'log' in expr_str:
                operations.add('logarithm')
            if 'exp' in expr_str:
                operations.add('exponential')
            if 'Integral' in expr_str:
                operations.add('integration')
            if 'Derivative' in expr_str:
                operations.add('differentiation')
            if 'Limit' in expr_str:
                operations.add('limit')
            if 'Sum' in expr_str:
                operations.add('summation')
        except:
            pass
        
        return list(operations)
    
    def _perform_computation(self, result: SemanticResult) -> SemanticResult:
        """
        执行计算
        
        Args:
            result: 语义结果对象
            
        Returns:
            更新后的结果
        """
        if not SYMPY_AVAILABLE or result.sympy_expr is None:
            return result
        
        try:
            formula_type = result.formula_type
            expr = result.sympy_expr
            
            # 根据公式类型执行不同操作
            if formula_type == FormulaType.EQUATION:
                result = self._solve_equation(result)
            
            elif formula_type == FormulaType.EXPRESSION:
                result = self._simplify_expression(result)
            
            elif formula_type == FormulaType.DERIVATIVE:
                result = self._compute_derivative(result)
            
            elif formula_type == FormulaType.INTEGRAL:
                result = self._compute_integral(result)
            
            elif formula_type == FormulaType.LIMIT:
                result = self._compute_limit(result)
            
            else:
                # 尝试化简
                result = self._simplify_expression(result)
        
        except Exception as e:
            result.warnings.append(f"计算出错: {str(e)}")
        
        return result
    
    def _solve_equation(self, result: SemanticResult) -> SemanticResult:
        """求解方程"""
        try:
            expr = result.sympy_expr
            
            # 对于等式，尝试求解
            if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
                eq = Eq(expr.lhs, expr.rhs)
                
                # 选择求解变量
                if result.variables:
                    solve_var = symbols(result.variables[0])
                    solutions = solve(eq, solve_var)
                    
                    result.solution = {
                        'method': 'solve',
                        'variable': result.variables[0],
                        'results': [str(s) for s in solutions]
                    }
            
            # 也尝试化简
            simplified = simplify(expr)
            result.simplified = sympy_latex(simplified)
        
        except Exception as e:
            logger.debug(f"求解方程出错: {e}")
        
        return result
    
    def _simplify_expression(self, result: SemanticResult) -> SemanticResult:
        """化简表达式"""
        try:
            expr = result.sympy_expr
            
            # 尝试多种化简方法
            simplified = simplify(expr)
            
            # 如果原表达式更简单，保持原样
            if len(str(simplified)) < len(str(expr)):
                result.simplified = sympy_latex(simplified)
            else:
                result.simplified = result.original_latex
            
            # 尝试展开
            expanded = expand(expr)
            
            # 尝试因式分解
            factored = factor(expr)
            
            result.solution = {
                'method': 'simplify',
                'simplified': sympy_latex(simplified),
                'expanded': sympy_latex(expanded),
                'factored': sympy_latex(factored)
            }
        
        except Exception as e:
            logger.debug(f"化简表达式出错: {e}")
        
        return result
    
    def _compute_derivative(self, result: SemanticResult) -> SemanticResult:
        """计算导数"""
        try:
            expr = result.sympy_expr
            
            if result.variables:
                var = symbols(result.variables[0])
                derivative = diff(expr, var)
                
                result.solution = {
                    'method': 'derivative',
                    'variable': result.variables[0],
                    'result': sympy_latex(derivative)
                }
                result.simplified = sympy_latex(simplify(derivative))
        
        except Exception as e:
            logger.debug(f"计算导数出错: {e}")
        
        return result
    
    def _compute_integral(self, result: SemanticResult) -> SemanticResult:
        """计算积分"""
        try:
            expr = result.sympy_expr
            
            if result.variables:
                var = symbols(result.variables[0])
                integral_result = integrate(expr, var)
                
                result.solution = {
                    'method': 'integral',
                    'variable': result.variables[0],
                    'result': sympy_latex(integral_result) + ' + C'
                }
                result.simplified = sympy_latex(simplify(integral_result))
        
        except Exception as e:
            logger.debug(f"计算积分出错: {e}")
        
        return result
    
    def _compute_limit(self, result: SemanticResult) -> SemanticResult:
        """计算极限"""
        try:
            expr = result.sympy_expr
            
            # 极限表达式通常已经包含目标点信息
            # 这里简化处理
            if result.variables:
                var = symbols(result.variables[0])
                # 默认计算 x -> 0 的极限
                limit_result = limit(expr, var, 0)
                
                result.solution = {
                    'method': 'limit',
                    'variable': result.variables[0],
                    'point': '0',
                    'result': sympy_latex(limit_result)
                }
        
        except Exception as e:
            logger.debug(f"计算极限出错: {e}")
        
        return result
    
    def _generate_interpretation(self, result: SemanticResult) -> str:
        """
        生成人类可读的解释
        
        Args:
            result: 语义结果
            
        Returns:
            解释文本
        """
        formula_type = result.formula_type
        
        interpretations = {
            FormulaType.EQUATION: "这是一个方程",
            FormulaType.INEQUALITY: "这是一个不等式",
            FormulaType.EXPRESSION: "这是一个数学表达式",
            FormulaType.FUNCTION_DEF: "这是一个函数定义",
            FormulaType.DERIVATIVE: "这是一个导数表达式",
            FormulaType.INTEGRAL: "这是一个积分表达式",
            FormulaType.LIMIT: "这是一个极限表达式",
            FormulaType.SUM: "这是一个求和表达式",
            FormulaType.PRODUCT: "这是一个连乘表达式",
            FormulaType.MATRIX: "这是一个矩阵",
            FormulaType.UNKNOWN: "无法确定公式类型",
        }
        
        base_interp = interpretations.get(formula_type, "")
        
        # 添加变量信息
        if result.variables:
            var_str = ", ".join(result.variables)
            base_interp += f"，包含变量: {var_str}"
        
        # 添加运算信息
        if result.operations:
            op_names = {
                'addition': '加法',
                'subtraction': '减法',
                'multiplication': '乘法',
                'division': '除法',
                'exponentiation': '幂运算',
                'square_root': '开方',
                'trigonometric': '三角函数',
                'logarithm': '对数',
                'exponential': '指数',
                'integration': '积分',
                'differentiation': '求导',
                'limit': '极限',
                'summation': '求和',
            }
            ops = [op_names.get(op, op) for op in result.operations[:3]]
            base_interp += f"，涉及 {', '.join(ops)} 运算"
        
        # 添加解结果
        if result.solution:
            if 'results' in result.solution and result.solution['results']:
                results = result.solution['results']
                if len(results) == 1:
                    base_interp += f"。解为: {results[0]}"
                else:
                    base_interp += f"。有 {len(results)} 个解"
        
        return base_interp
    
    # 便捷方法
    def solve_equation(self, latex: str, variable: str = 'x') -> Optional[List[str]]:
        """
        求解方程
        
        Args:
            latex: LaTeX 方程
            variable: 求解变量
            
        Returns:
            解的列表
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = self.latex_to_sympy(latex)
            var = symbols(variable)
            
            # 处理等式
            if '=' in latex:
                parts = latex.split('=')
                if len(parts) == 2:
                    left = self.latex_to_sympy(parts[0])
                    right = self.latex_to_sympy(parts[1])
                    eq = Eq(left, right)
                    solutions = solve(eq, var)
                    return [str(s) for s in solutions]
            
            solutions = solve(expr, var)
            return [str(s) for s in solutions]
        
        except Exception as e:
            logger.warning(f"求解方程失败: {e}")
            return None
    
    def simplify_expression(self, latex: str) -> Optional[str]:
        """
        化简表达式
        
        Args:
            latex: LaTeX 表达式
            
        Returns:
            化简后的 LaTeX
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = self.latex_to_sympy(latex)
            simplified = simplify(expr)
            return sympy_latex(simplified)
        except Exception as e:
            logger.warning(f"化简失败: {e}")
            return None
    
    def compute_derivative(self, latex: str, variable: str = 'x') -> Optional[str]:
        """
        计算导数
        
        Args:
            latex: LaTeX 表达式
            variable: 对其求导的变量
            
        Returns:
            导数的 LaTeX 表示
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = self.latex_to_sympy(latex)
            var = symbols(variable)
            derivative = diff(expr, var)
            return sympy_latex(derivative)
        except Exception as e:
            logger.warning(f"求导失败: {e}")
            return None
    
    def compute_integral(self, latex: str, variable: str = 'x',
                         lower: Optional[str] = None,
                         upper: Optional[str] = None) -> Optional[str]:
        """
        计算积分
        
        Args:
            latex: LaTeX 表达式
            variable: 积分变量
            lower: 下限（定积分）
            upper: 上限（定积分）
            
        Returns:
            积分结果的 LaTeX 表示
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = self.latex_to_sympy(latex)
            var = symbols(variable)
            
            if lower is not None and upper is not None:
                # 定积分
                lower_val = sympify(lower)
                upper_val = sympify(upper)
                result = integrate(expr, (var, lower_val, upper_val))
            else:
                # 不定积分
                result = integrate(expr, var)
            
            latex_result = sympy_latex(result)
            if lower is None:
                latex_result += ' + C'
            
            return latex_result
        except Exception as e:
            logger.warning(f"积分失败: {e}")
            return None
    
    def evaluate_limit(self, latex: str, variable: str = 'x',
                       point: str = '0') -> Optional[str]:
        """
        计算极限
        
        Args:
            latex: LaTeX 表达式
            variable: 极限变量
            point: 趋近点
            
        Returns:
            极限值的 LaTeX 表示
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = self.latex_to_sympy(latex)
            var = symbols(variable)
            point_val = sympify(point)
            result = limit(expr, var, point_val)
            return sympy_latex(result)
        except Exception as e:
            logger.warning(f"极限计算失败: {e}")
            return None


def process_semantic(latex: str,
                     config: Optional[SemanticConfig] = None) -> SemanticResult:
    """
    处理公式语义的便捷函数
    
    Args:
        latex: LaTeX 代码
        config: 语义配置
        
    Returns:
        语义理解结果
    """
    processor = SemanticProcessor(config)
    return processor.process(latex)
