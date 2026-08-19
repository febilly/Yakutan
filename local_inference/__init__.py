from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List

# Allow PyTorch and DirectML/ONNX stacks to coexist in one process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

LOCAL_INFERENCE_UI_ENV = "YAKUTAN_LOCAL_INFERENCE_UI"

LOCAL_INFERENCE_ENGINES = ("sensevoice", "qwen3-asr")
LOCAL_MT_ENGINES = ("hymt2",)

LOCAL_INFERENCE_DISPLAY_NAMES: Dict[str, str] = {
    "sensevoice": "SenseVoice Small",
    "qwen3-asr": "Qwen3-ASR 1.7B",
    "hymt2": "Hy-MT2 1.8B",
}

COMMON_RUNTIME_MODULES = (
    "numpy",
    "onnxruntime",
    "huggingface_hub",
    "soundfile",
    "sentencepiece",
)

ENGINE_RUNTIME_MODULES = {
    "sensevoice": ("kaldi_native_fbank",),
    "qwen3-asr": ("gguf",),
}


def _env_to_bool(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _frozen_local_inference_marker_present() -> bool:
    if not getattr(sys, "frozen", False) or not hasattr(sys, "_MEIPASS"):
        return False
    marker = os.path.join(sys._MEIPASS, "YAKUTAN_LOCAL_INFERENCE_BUILD")
    return os.path.isfile(marker)


def is_local_inference_build_enabled() -> bool:
    raw = os.getenv(LOCAL_INFERENCE_UI_ENV)
    if raw is not None:
        return _env_to_bool(raw)
    if getattr(sys, "frozen", False):
        if _frozen_local_inference_marker_present():
            return True
        try:
            return importlib.util.find_spec("local_inference") is not None
        except Exception:
            return False
    return True


def is_local_inference_ui_enabled() -> bool:
    """与构建开关一致：Local Inference 构建始终展示面板；引擎级问题见 engines[].runtime_issues。"""
    return is_local_inference_build_enabled()


def _missing_modules(modules: tuple[str, ...]) -> List[str]:
    missing: List[str] = []
    for module_name in modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def get_common_runtime_issues() -> List[str]:
    return _missing_modules(COMMON_RUNTIME_MODULES)


def get_engine_runtime_issues(engine: str) -> List[str]:
    issues = get_common_runtime_issues()
    issues.extend(_missing_modules(ENGINE_RUNTIME_MODULES.get(engine, ())))
    return sorted(set(issues))


def is_engine_runtime_available(engine: str) -> bool:
    return not get_engine_runtime_issues(engine)


def get_local_inference_features() -> dict:
    # 显存占用是模型的固有属性，不是用户设置，所以走 features 而不是 config：
    # 面板的设置会存进 localStorage 再回灌，未知字段在那一趟里会丢掉。
    import config as _config

    vram = getattr(_config, "QWEN3_ASR_VRAM_MB", {}) or {}
    return {
        "local_inference_build_enabled": is_local_inference_build_enabled(),
        "local_inference_ui_enabled": is_local_inference_ui_enabled(),
        "qwen3_vram_mb": {
            "decoder": int(vram.get("decoder", 0)),
            "encoder": int(vram.get("encoder", 0)),
        },
        "engines": {
            engine: {
                "display_name": LOCAL_INFERENCE_DISPLAY_NAMES.get(engine, engine),
                "runtime_available": is_engine_runtime_available(engine),
                "runtime_issues": get_engine_runtime_issues(engine),
            }
            for engine in LOCAL_INFERENCE_ENGINES
        },
    }

