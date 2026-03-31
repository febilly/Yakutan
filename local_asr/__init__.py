from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List

# Allow PyTorch and DirectML/ONNX stacks to coexist in one process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

LOCAL_ASR_UI_ENV = "YAKUTAN_LOCAL_ASR_UI"

LOCAL_ASR_ENGINES = ("sensevoice", "funasr-nano", "qwen3-asr")

LOCAL_ASR_DISPLAY_NAMES: Dict[str, str] = {
    "sensevoice": "SenseVoice Small",
    "funasr-nano": "Fun-ASR-Nano",
    "qwen3-asr": "Qwen3-ASR 1.7B",
}

COMMON_RUNTIME_MODULES = (
    "numpy",
    "torch",
    "funasr",
    "modelscope",
    "huggingface_hub",
)

ENGINE_RUNTIME_MODULES = {
    "sensevoice": (),
    "funasr-nano": (
        "soundfile",
        "torchaudio",
        "transformers",
    ),
    "qwen3-asr": (
        "onnxruntime",
        "gguf",
    ),
}


def _env_to_bool(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def is_local_asr_build_enabled() -> bool:
    raw = os.getenv(LOCAL_ASR_UI_ENV)
    if raw is not None:
        return _env_to_bool(raw)
    if getattr(sys, "frozen", False):
        return False
    return True


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


def is_local_asr_ui_enabled() -> bool:
    if not is_local_asr_build_enabled():
        return False
    return not get_common_runtime_issues()


def get_local_asr_features() -> dict:
    return {
        "local_asr_build_enabled": is_local_asr_build_enabled(),
        "local_asr_ui_enabled": is_local_asr_ui_enabled(),
        "engines": {
            engine: {
                "display_name": LOCAL_ASR_DISPLAY_NAMES.get(engine, engine),
                "runtime_available": is_engine_runtime_available(engine),
                "runtime_issues": get_engine_runtime_issues(engine),
            }
            for engine in LOCAL_ASR_ENGINES
        },
    }

