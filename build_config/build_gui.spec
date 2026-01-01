# -*- mode: python ; coding: utf-8 -*-
"""
LaTeX公式识别系统 GUI版 打包配置
"""

import os
import sys

block_cipher = None

# 获取项目根目录（build_config的上级目录）
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
PROJECT_ROOT = os.path.dirname(SPEC_DIR)

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'gui.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        # 包含模型文件（所有模型）
        (os.path.join(PROJECT_ROOT, 'models'), 'models'),
        # 包含源代码模块（预处理、分割、识别、结构分析、语义处理等）
        (os.path.join(PROJECT_ROOT, 'src'), 'src'),
    ],
    hiddenimports=[
        # sklearn 相关
        'sklearn',
        'sklearn.svm',
        'sklearn.svm._classes',
        'sklearn.ensemble',
        'sklearn.ensemble._forest',
        'sklearn.ensemble._gb',
        'sklearn.neighbors',
        'sklearn.neighbors._classification',
        'sklearn.neighbors._regression',
        'sklearn.preprocessing',
        'sklearn.preprocessing._label',
        'sklearn.preprocessing._encoders',
        'sklearn.model_selection',
        'sklearn.model_selection._split',
        'sklearn.metrics',
        'sklearn.utils._cython_blas',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree',
        'sklearn.tree._utils',
        # OpenCV
        'cv2',
        # numpy/scipy
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'scipy',
        'scipy.sparse',
        'scipy.sparse.csgraph',
        'scipy.sparse.linalg',
        'scipy.ndimage',
        'scipy.special',
        # sympy 语义模块
        'sympy',
        'sympy.parsing',
        'sympy.parsing.latex',
        'sympy.parsing.latex._parse_latex_antlr',
        'sympy.core',
        'sympy.core.sympify',
        'sympy.solvers',
        'sympy.calculus',
        'sympy.integrals',
        'sympy.series',
        'sympy.simplify',
        'sympy.matrices',
        # antlr4 (sympy.parsing.latex 依赖)
        'antlr4',
        'antlr4.error',
        'antlr4.error.ErrorListener',
        # PIL/Pillow
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        # matplotlib 相关
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'matplotlib.figure',
        'matplotlib.font_manager',
        # joblib (模型加载)
        'joblib',
        # 项目源代码模块
        'src',
        'src.config',
        'src.preprocessing',
        'src.segmentation',
        'src.recognition',
        'src.structure_analysis',
        'src.semantic',
        'src.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='公式识别系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI程序，不显示控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
