# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller configuration for Yakutan with local inference support.
"""

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import importlib.util

datas = [
    ('hot_words', 'hot_words'),
    ('ui/templates', 'ui/templates'),
    ('ui/static', 'ui/static'),
    ('YAKUTAN_LOCAL_INFERENCE_BUILD', '.'),
]

_pykakasi_spec = importlib.util.find_spec('pykakasi')
if _pykakasi_spec is not None and getattr(_pykakasi_spec, 'submodule_search_locations', None):
    datas += collect_data_files('pykakasi')

# Bundle Silero VAD; other model weights and Vulkan runtime are downloaded in the UI.
from pathlib import Path
_silero_vad = Path('local_inference/models/silero_vad')
if _silero_vad.is_dir():
    datas += [(str(_silero_vad), 'local_inference/models/silero_vad')]
# Vendor Python modules are collected into PYZ below. Never copy their directory:
# a developer checkout may contain obsolete DLLs, bytecode and model files.

datas += collect_data_files('soundcard')
datas += collect_data_files('fast_langdetect', includes=['resources/lid.176.ftz'])

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
    'main',
    'openai',
    'pykakasi',
    'httpx',
    'websockets',
    'pypinyin',
    'jieba',
    'fancify_text',
    'arabic_reshaper',
    'bidi',
    'bidi.algorithm',
    'panel_app',
    'webview',
    'speech_recognizers.local_speech_recognizer',
    'local_inference',
]

hiddenimports += [
    'webview.platforms.winforms', 'webview.platforms.win32',
    'webview.platforms.edgechromium', 'webview.platforms.mshtml',
]
hiddenimports += collect_submodules('streaming_translation')
hiddenimports += collect_submodules('local_inference')

a = Analysis(
    ['run_ui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_local_inference.py'],
    excludes=[
        'pytest',
        'gguf.scripts',
        'webview.platforms.android', 'webview.platforms.cocoa',
        'webview.platforms.gtk', 'webview.platforms.qt', 'webview.platforms.cef',
    ],
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
    name='Yakutan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
