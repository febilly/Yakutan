# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller configuration for the Local ASR build.
"""

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import importlib.util

datas = [
    ('hot_words', 'hot_words'),
    ('ui/templates', 'ui/templates'),
    ('ui/static', 'ui/static'),
]

_pykakasi_spec = importlib.util.find_spec('pykakasi')
if _pykakasi_spec is not None and getattr(_pykakasi_spec, 'submodule_search_locations', None):
    datas += collect_data_files('pykakasi')

# Silero VAD: CI runs prefetch into local_asr/models/torch before PyInstaller.
from pathlib import Path
_silero_torch = Path('local_asr/models/torch')
if _silero_torch.is_dir():
    datas += [(str(_silero_torch), 'local_asr/models/torch')]

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
    'panel_app',
    'webview',
    'translators.translation_apis.google_web_api',
    'translators.translation_apis.google_dictionary_api',
    'translators.translation_apis.openrouter_api',
    'translators.translation_apis.deepl_api',
    'translators.translation_apis.qwen_mt_api',
    'speech_recognizers.local_speech_recognizer',
    'local_asr',
]

hiddenimports += collect_submodules('webview.platforms')
hiddenimports += collect_submodules('translators.translation_apis')
hiddenimports += collect_submodules('local_asr')

a = Analysis(
    ['run_ui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_local_asr.py'],
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
    name='Yakutan-LocalASR',
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
