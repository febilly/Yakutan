from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

from . import LOCAL_ASR_DISPLAY_NAMES, get_engine_runtime_issues

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
VENDOR_DIR = PACKAGE_DIR / "vendor"


def _default_models_dir() -> Path:
    if getattr(sys, "frozen", False):
        from resource_path import get_user_data_path

        return Path(get_user_data_path("local_asr_models"))
    return PACKAGE_DIR / "models"


MODELS_DIR = _default_models_dir()

ASR_MODEL_IDS = {
    "sensevoice": "iic/SenseVoiceSmall",
    "funasr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "qwen3-asr": "Qwen3-ASR-1.7B",
}

QWEN3_ASR_FILES = [
    "qwen3_asr_encoder_frontend.int4.onnx",
    "qwen3_asr_encoder_backend.int4.onnx",
    "qwen3_asr_llm.q4_k.gguf",
]

QWEN3_ASR_DIR_NAME = "qwen3-asr"

QWEN3_ASR_MODEL_URL = (
    "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/"
    "Qwen3-ASR-1.7B-gguf.zip"
)
LLAMA_CPP_DLL_URL_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/"
    "llama-{tag}-bin-win-vulkan-x64.zip"
)
LLAMA_CPP_LATEST_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

_MODEL_SIZE_BYTES = {
    "silero-vad": 2_000_000,
    "sensevoice": 940_000_000,
    "funasr-nano": 1_050_000_000,
    "qwen3-asr": 770_000_000,
}

_FUNASR_NANO_URLS = {
    "model.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/funasr_nano/model.py",
    "ctc.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/funasr_nano/ctc.py",
    "tools/utils.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/funasr_nano/tools/utils.py",
}

_QWEN3_VENDOR_URLS = {
    "asr_engine.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/asr_engine.py",
    "inference/asr.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/asr.py",
    "inference/schema.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/schema.py",
    "inference/encoder.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/encoder.py",
    "inference/llama.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/llama.py",
    "inference/utils.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/utils.py",
}


def apply_cache_env() -> None:
    resolved = str(MODELS_DIR.resolve())
    os.environ["MODELSCOPE_CACHE"] = os.path.join(resolved, "modelscope")
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    bundle_torch = PACKAGE_DIR / "models" / "torch"
    if getattr(sys, "frozen", False) and bundle_torch.is_dir():
        os.environ["TORCH_HOME"] = str(bundle_torch.resolve())
    else:
        os.environ["TORCH_HOME"] = os.path.join(resolved, "torch")


def silero_torch_home() -> Path:
    """Directory used as TORCH_HOME for Silero (bundled hub in frozen exe or under MODELS_DIR)."""
    bundle_torch = PACKAGE_DIR / "models" / "torch"
    if getattr(sys, "frozen", False) and bundle_torch.is_dir():
        return bundle_torch
    return MODELS_DIR / "torch"


def _write_text_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _download_text_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading vendor source: %s", dest.name)
    request = Request(url, headers={"User-Agent": "Yakutan"})
    with urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def _ensure_qwen_vendor_scaffold(base_dir: Path) -> None:
    _write_text_if_missing(
        base_dir / "__init__.py",
        "import logging\n\nlogger = logging.getLogger('Yakutan.LocalASR.Qwen3')\n",
    )
    _write_text_if_missing(
        base_dir / "inference" / "__init__.py",
        "from .. import logger as logger  # noqa: F401\n",
    )


def ensure_vendor_sources(engine: str) -> Path | None:
    if engine == "sensevoice":
        return None

    if engine == "funasr-nano":
        base_dir = VENDOR_DIR / "funasr_nano"
        for relative_path, url in _FUNASR_NANO_URLS.items():
            dest = base_dir / relative_path
            if not dest.exists():
                _download_text_file(url, dest)
        return base_dir

    if engine == "qwen3-asr":
        base_dir = VENDOR_DIR / "qwen_asr_gguf"
        _ensure_qwen_vendor_scaffold(base_dir)
        for relative_path, url in _QWEN3_VENDOR_URLS.items():
            dest = base_dir / relative_path
            if not dest.exists():
                _download_text_file(url, dest)
        return base_dir

    raise ValueError(f"Unknown local ASR engine: {engine}")


def _vendor_ready(engine: str) -> bool:
    if engine == "sensevoice":
        return True
    if engine == "funasr-nano":
        base_dir = VENDOR_DIR / "funasr_nano"
        return all((base_dir / relative_path).exists() for relative_path in _FUNASR_NANO_URLS)
    if engine == "qwen3-asr":
        base_dir = VENDOR_DIR / "qwen_asr_gguf"
        required = list(_QWEN3_VENDOR_URLS) + ["__init__.py", "inference/__init__.py"]
        return all((base_dir / relative_path).exists() for relative_path in required)
    return False


def is_silero_cached() -> bool:
    torch_hub = silero_torch_home() / "hub"
    return any(torch_hub.glob("snakers4_silero-vad*")) if torch_hub.exists() else False


def _ms_model_path(org: str, name: str) -> Path:
    for sub in (
        MODELS_DIR / "modelscope" / org / name,
        MODELS_DIR / "modelscope" / "hub" / "models" / org / name,
    ):
        if sub.exists():
            return sub
    return MODELS_DIR / "modelscope" / org / name


def get_local_model_path(engine: str, hub: str = "ms") -> str | None:
    if engine == "qwen3-asr":
        model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
        return str(model_dir) if model_dir.exists() else None
    model_id = ASR_MODEL_IDS.get(engine)
    if not model_id:
        return None
    org, name = model_id.split("/")

    def _try_ms() -> str | None:
        local = _ms_model_path(org, name)
        return str(local) if local.exists() else None

    def _try_hf() -> str | None:
        snap_dir = MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
        if snap_dir.exists():
            snapshots = sorted(snap_dir.iterdir())
            if snapshots:
                return str(snapshots[-1])
        return None

    if hub == "ms":
        return _try_ms() or _try_hf()
    return _try_hf() or _try_ms()


def is_qwen3_asr_ready() -> bool:
    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    if not model_dir.exists():
        return False
    for filename in QWEN3_ASR_FILES:
        if not (model_dir / filename).exists():
            return False
    bin_dir = VENDOR_DIR / "qwen_asr_gguf" / "inference" / "bin"
    if not bin_dir.exists():
        return False
    required = ["llama.dll", "ggml.dll", "ggml-base.dll"] if sys.platform == "win32" else [
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
    ]
    return all((bin_dir / filename).exists() for filename in required)


def is_asr_cached(engine: str, hub: str = "ms") -> bool:
    if get_engine_runtime_issues(engine):
        return False
    if not _vendor_ready(engine):
        return False
    if engine == "qwen3-asr":
        return is_qwen3_asr_ready()
    model_id = ASR_MODEL_IDS.get(engine)
    if not model_id:
        return False
    org, name = model_id.split("/")
    if _ms_model_path(org, name).exists():
        return True
    return (MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}").exists()


def get_missing_models(engine: str, hub: str = "ms") -> list[dict]:
    missing: list[dict] = []
    if not is_silero_cached():
        missing.append(
            {
                "name": "Silero VAD",
                "type": "silero-vad",
                "estimated_bytes": _MODEL_SIZE_BYTES["silero-vad"],
            }
        )
    if not is_asr_cached(engine, hub=hub):
        missing.append(
            {
                "name": LOCAL_ASR_DISPLAY_NAMES.get(engine, engine),
                "type": engine,
                "estimated_bytes": _MODEL_SIZE_BYTES.get(engine, 0),
            }
        )
    return missing


def download_silero() -> None:
    apply_cache_env()
    import torch

    logger.info("Downloading Silero VAD...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    del model
    logger.info("Silero VAD downloaded")


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", desc or url)
    urlretrieve(url, str(dest))


def download_qwen3_asr() -> None:
    apply_cache_env()
    ensure_vendor_sources("qwen3-asr")

    import json
    import shutil
    import zipfile

    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = VENDOR_DIR / "qwen_asr_gguf" / "inference" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    if any(not (model_dir / filename).exists() for filename in QWEN3_ASR_FILES):
        zip_path = MODELS_DIR / "qwen3-asr-1.7b-gguf.zip"
        _download_file(QWEN3_ASR_MODEL_URL, zip_path, "Qwen3-ASR model")
        with zipfile.ZipFile(str(zip_path), "r") as archive:
            for member in archive.namelist():
                basename = os.path.basename(member)
                if basename and basename in QWEN3_ASR_FILES:
                    with archive.open(member) as src, open(model_dir / basename, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        zip_path.unlink(missing_ok=True)

    required_dlls = ["llama.dll", "ggml.dll", "ggml-base.dll"] if sys.platform == "win32" else [
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
    ]
    if any(not (bin_dir / filename).exists() for filename in required_dlls):
        try:
            request = Request(LLAMA_CPP_LATEST_API, headers={"User-Agent": "Yakutan"})
            with urlopen(request, timeout=15) as response:
                tag = json.loads(response.read())["tag_name"]
        except Exception:
            tag = "b8391"
        dll_zip = MODELS_DIR / "llama-cpp-vulkan.zip"
        _download_file(LLAMA_CPP_DLL_URL_TEMPLATE.format(tag=tag), dll_zip, f"llama.cpp {tag}")
        with zipfile.ZipFile(str(dll_zip), "r") as archive:
            for member in archive.namelist():
                basename = os.path.basename(member)
                if basename in required_dlls or (
                    sys.platform == "win32" and basename.startswith("ggml-") and basename.endswith(".dll")
                ):
                    with archive.open(member) as src, open(bin_dir / basename, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        dll_zip.unlink(missing_ok=True)


def download_asr(engine: str, hub: str = "ms") -> None:
    apply_cache_env()
    ensure_vendor_sources(engine)

    resolved = str(MODELS_DIR.resolve())
    ms_cache = os.path.join(resolved, "modelscope")
    hf_cache = os.path.join(resolved, "huggingface", "hub")

    if engine == "qwen3-asr":
        download_qwen3_asr()
        return

    model_id = ASR_MODEL_IDS[engine]
    if hub == "ms":
        from modelscope import snapshot_download

        snapshot_download(model_id=model_id, cache_dir=ms_cache)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_id, cache_dir=hf_cache)


def prepare_engine(engine: str, hub: str = "ms") -> None:
    ensure_vendor_sources(engine)
    if not is_silero_cached():
        download_silero()
    if not is_asr_cached(engine, hub=hub):
        download_asr(engine, hub=hub)


def get_engine_status(engine: str, hub: str = "ms") -> dict:
    return {
        "engine": engine,
        "display_name": LOCAL_ASR_DISPLAY_NAMES.get(engine, engine),
        "runtime_issues": get_engine_runtime_issues(engine),
        "vendor_ready": _vendor_ready(engine),
        "model_cached": bool(get_local_model_path(engine, hub=hub)) if engine != "qwen3-asr" else is_qwen3_asr_ready(),
        "ready": is_asr_cached(engine, hub=hub),
        "missing": get_missing_models(engine, hub=hub),
    }

