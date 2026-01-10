# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller配置文件 - Web UI版本
用于将VRChat翻译器Web UI打包成单个可执行文件

使用方法:
    pyinstaller run_ui.spec

打包后的可执行文件将包含所有必要的资源文件
"""

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files
import importlib.util

# 需要包含的数据文件（资源文件）
datas = [
    ('hot_words', 'hot_words'),  # 公共热词目录
    ('ui/templates', 'ui/templates'),  # UI模板文件
    ('ui/static', 'ui/static'),  # UI静态文件（CSS、JS等）
]

# pykakasi 假名转换依赖包内词典/数据文件；未打包会导致假名功能无效
# 某些环境里 pykakasi 可能以单文件 module 形式存在（不是 package），此时跳过数据收集以避免告警。
_pykakasi_spec = importlib.util.find_spec('pykakasi')
if _pykakasi_spec is not None and getattr(_pykakasi_spec, 'submodule_search_locations', None):
    datas += collect_data_files('pykakasi')

# 需要包含的隐藏导入
hiddenimports = [
    'dashscope',
    'dashscope.audio.asr',
    'dashscope.audio.qwen_omni',
    'dashscope.audio.qwen_omni.omni_realtime',
    'deepl',
    'flask',
    'flask_cors',
    'pythonosc',
    'pyaudio',
    'fast_langdetect',
    'googletrans',
    'aiohttp',
    'asyncio',
    'dotenv',
    'main',  # 确保main模块被包含
    'openai',
    'pykakasi',
    'httpx',
    'websockets',
    'pypinyin',
    'jieba',
]

a = Analysis(
    ['run_ui.py'],  # Web UI入口文件
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='Yakutan',  # 可执行文件名称
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以指定图标文件路径
)
