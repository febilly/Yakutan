"""打包成 exe 后的模型布局：可写目录必须落在 exe 同级，只读资源才来自 _MEIPASS。

PyInstaller 把程序解压到临时的 `_MEIPASS` 目录，那里是只读的、每次启动还会换路径。所以模型
权重、运行时 DLL、下载中的 .part 都必须写到 `get_user_data_path()`（冻结时 = exe 所在目录）。
这条约束在源码里看不出来——两种模式下同一个函数返回不同的路径——所以在这里钉死。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from local_inference import COMMON_RUNTIME_MODULES, ENGINE_RUNTIME_MODULES  # noqa: E402

# 在子进程里伪装冻结环境：sys.frozen / sys.executable 决定了所有路径，改完必须重新 import。
_FROZEN_PATHS_SCRIPT = """
import json, shutil, sys, tempfile
from pathlib import Path

root = Path(%(root)r)
fake = Path(tempfile.mkdtemp())
exe_dir, meipass = fake / "app", fake / "_MEI123456"
exe_dir.mkdir(); meipass.mkdir()

# 模拟打包布局：包体在 _MEIPASS 下
shutil.copytree(root / "local_inference", meipass / "local_inference",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name in ("resource_path.py", "proxy_detector.py", "config.py"):
    shutil.copy2(root / name, meipass / name)

sys.frozen = True
sys._MEIPASS = str(meipass)
sys.executable = str(exe_dir / "Yakutan-LocalInference.exe")
sys.path.insert(0, str(meipass))
for name in [n for n in sys.modules if n.startswith(("local_inference", "resource_path"))]:
    del sys.modules[name]

import os
import local_inference.model_manager as mm

mm.apply_cache_env()
print(json.dumps({
    "exe_dir": str(exe_dir),
    "meipass": str(meipass),
    "package": str(Path(mm.__file__).parent),
    "writable": {
        "models_dir": str(mm.MODELS_DIR),
        "silero": str(mm._silero_onnx_user_path()),
        "sensevoice": str(mm._sensevoice_onnx_user_dir()),
        "qwen3": str(mm.MODELS_DIR / mm.QWEN3_ASR_DIR_NAME),
        "qwen3_zip": str(mm.MODELS_DIR / "qwen3-asr-1.7b-gguf.zip"),
        "llama_bin": str(mm._qwen_llama_vulkan_user_bin()),
        "llama_zip": str(mm.MODELS_DIR / "llama-cpp-vulkan.zip"),
        "hymt2": str(mm.MODELS_DIR / mm.HYMT2_DIR_NAME / mm.HYMT2_MODEL_FILE),
        "hf_home": os.environ["HF_HOME"],
    },
    "bundled": {
        "silero": str(mm._silero_onnx_bundle_path()),
        "vendor": str(mm._qwen_vendor_package_dir()),
    },
    "vendor_complete": mm._qwen_vendor_files_complete(mm._qwen_vendor_package_dir()),
    "silero_cached": mm.is_silero_cached(),
}))
"""


def _frozen_layout() -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _FROZEN_PATHS_SCRIPT % {"root": str(ROOT)}],
        cwd=str(ROOT), check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_every_writable_model_path_sits_next_to_the_exe():
    layout = _frozen_layout()
    # 包体确实在 _MEIPASS 里，否则这个用例什么都没验证
    assert layout["package"].startswith(layout["meipass"])

    for name, path in layout["writable"].items():
        assert path.startswith(layout["exe_dir"]), f"{name} 写到了 exe 目录之外: {path}"
        assert not path.startswith(layout["meipass"]), f"{name} 写进了只读的 _MEIPASS: {path}"


def test_readonly_assets_come_from_the_bundle():
    layout = _frozen_layout()
    # 打包自带的 Silero 与 Qwen 胶水代码直接从 _MEIPASS 读，不该再联网补一份
    assert layout["bundled"]["silero"].startswith(layout["meipass"])
    assert layout["bundled"]["vendor"].startswith(layout["meipass"])
    assert layout["vendor_complete"] is True
    assert layout["silero_cached"] is True


def test_declared_runtime_modules_are_actually_imported():
    """运行时依赖清单里的库必须真的被 import。

    PyInstaller 只收集源码里 import 过的包。清单里留一个谁都不 import 的名字，打包版就会
    在 find_spec 上判它缺失，于是所有本地引擎都显示“未就绪”，而源码运行一切正常。
    """
    sources = [
        path.read_text(encoding="utf-8", errors="ignore")
        for path in ROOT.rglob("*.py")
        if ".venv" not in path.parts and "tests" not in path.parts
    ]
    declared = set(COMMON_RUNTIME_MODULES)
    for modules in ENGINE_RUNTIME_MODULES.values():
        declared.update(modules)

    for module in declared:
        assert any(f"import {module}" in text for text in sources), (
            f"{module} 在运行时依赖清单里，但仓库里没有任何 import，打包后会被判为缺失"
        )
