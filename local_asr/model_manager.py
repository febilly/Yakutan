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

# Qwen3 GGUF 解码依赖的 llama.cpp Vulkan 运行时（Windows DLL / Linux .so），与权重一并放用户可写目录
QWEN_LLAMA_VULKAN_BIN_DIR_NAME = "qwen_llama_vulkan_bin"


def _qwen_llama_vulkan_user_bin() -> Path:
    return MODELS_DIR / QWEN_LLAMA_VULKAN_BIN_DIR_NAME


def _qwen_llama_vulkan_vendor_bin() -> Path:
    return VENDOR_DIR / "qwen_asr_gguf" / "inference" / "bin"


def _llama_vulkan_bin_has_core_dlls(path: Path) -> bool:
    if not path.is_dir():
        return False
    if sys.platform == "win32":
        names = ("llama.dll", "ggml.dll", "ggml-base.dll")
    else:
        names = ("libllama.so", "libggml.so", "libggml-base.so")
    return all((path / n).is_file() for n in names)


def _qwen3_llama_bin_dir() -> Path:
    """解析实际含可用 Vulkan 运行时的目录；优先用户数据，其次兼容旧 vendor/bin。"""
    user = _qwen_llama_vulkan_user_bin()
    if _llama_vulkan_bin_has_core_dlls(user):
        return user
    vendor = _qwen_llama_vulkan_vendor_bin()
    if _llama_vulkan_bin_has_core_dlls(vendor):
        return vendor
    return user


def prepare_qwen_llama_runtime_env() -> None:
    """供 llama.py 通过 YAKUTAN_QWEN_LLAMA_BIN 加载与 exe 同目录下的 Vulkan DLL（先于 import llama 调用）。"""
    os.environ["YAKUTAN_QWEN_LLAMA_BIN"] = str(_qwen_llama_vulkan_user_bin().resolve())


def _default_models_dir() -> Path:
    if getattr(sys, "frozen", False):
        from resource_path import get_user_data_path

        return Path(get_user_data_path("local_asr_models"))
    return PACKAGE_DIR / "models"


MODELS_DIR = _default_models_dir()

SENSEVOICE_ONNX_DIR_NAME = "sensevoice-onnx"
SENSEVOICE_ONNX_REPO = "lovemefan/SenseVoice-onnx"
SENSEVOICE_ENCODER_ONNX = "sense-voice-encoder-int8.onnx"
SENSEVOICE_ONNX_FILES = (
    "embedding.npy",
    SENSEVOICE_ENCODER_ONNX,
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
)

ASR_MODEL_IDS = {
    "sensevoice": SENSEVOICE_ONNX_REPO,
    "qwen3-asr": "Qwen3-ASR-1.7B",
}

QWEN3_ASR_FILES = [
    "qwen3_asr_encoder_frontend.int4.onnx",
    "qwen3_asr_encoder_backend.int4.onnx",
    "qwen3_asr_llm.q4_k.gguf",
]

QWEN3_ASR_DIR_NAME = "qwen3-asr"

# HaujetZhao/Qwen3-ASR-GGUF zip 内仅有 Encoder ONNX×2 + Decoder GGUF；llama.cpp Vulkan DLL 由 download_qwen3_asr() / prefetch_llama_cpp_vulkan_for_pyinstaller_bundle() 从 ggml-org 拉取，不进 Git（手动 workflow「Pack Local ASR model bundles」或本机运行同上函数下载）。
QWEN3_ASR_MODEL_URL = (
    "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/"
    "Qwen3-ASR-1.7B-gguf.zip"
)
LLAMA_CPP_DLL_URL_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/"
    "llama-{tag}-bin-win-vulkan-x64.zip"
)
LLAMA_CPP_LATEST_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

SILERO_VAD_DIR_NAME = "silero_vad"
SILERO_VAD_ONNX_NAME = "silero_vad_16k_op15.onnx"
SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    f"src/silero_vad/data/{SILERO_VAD_ONNX_NAME}"
)

_MODEL_SIZE_BYTES = {
    "silero-vad": 1_300_000,
    "sensevoice": 380_000_000,
    "qwen3-asr": 770_000_000,
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
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")


def _silero_onnx_bundle_path() -> Path:
    return PACKAGE_DIR / "models" / SILERO_VAD_DIR_NAME / SILERO_VAD_ONNX_NAME


def _silero_onnx_user_path() -> Path:
    return MODELS_DIR / SILERO_VAD_DIR_NAME / SILERO_VAD_ONNX_NAME


def silero_onnx_path() -> Path:
    """Bundled ONNX (PyInstaller) first, else user cache under MODELS_DIR."""
    bundle = _silero_onnx_bundle_path()
    if bundle.is_file():
        return bundle
    user = _silero_onnx_user_path()
    if user.is_file():
        return user
    return user


def _sensevoice_onnx_bundle_dir() -> Path:
    return PACKAGE_DIR / "models" / SENSEVOICE_ONNX_DIR_NAME


def _sensevoice_onnx_user_dir() -> Path:
    return MODELS_DIR / SENSEVOICE_ONNX_DIR_NAME


def _sensevoice_onnx_ready(path: Path) -> bool:
    return all((path / name).is_file() for name in SENSEVOICE_ONNX_FILES)


def sensevoice_onnx_model_dir() -> Path | None:
    """PyInstaller bundled tree first, then user cache."""
    bundle = _sensevoice_onnx_bundle_dir()
    if _sensevoice_onnx_ready(bundle):
        return bundle
    user = _sensevoice_onnx_user_dir()
    if _sensevoice_onnx_ready(user):
        return user
    return None


def _ensure_llama_cpp_vulkan_to(bin_dir: Path) -> None:
    """若 bin_dir 中缺少 llama/ggml 主库，则从 ggml-org llama.cpp win-vulkan x64 release 解压写入。"""
    import json
    import shutil
    import zipfile

    if sys.platform != "win32":
        logger.info("Skipping llama.cpp Vulkan bundle download on %s", sys.platform)
        return

    required_dlls = ["llama.dll", "ggml.dll", "ggml-base.dll"]
    if all((bin_dir / filename).is_file() for filename in required_dlls):
        return

    bin_dir.mkdir(parents=True, exist_ok=True)
    try:
        request = Request(LLAMA_CPP_LATEST_API, headers={"User-Agent": "Yakutan"})
        with urlopen(request, timeout=15) as response:
            tag = json.loads(response.read())["tag_name"]
    except Exception:
        tag = "b8391"
    dll_zip = MODELS_DIR / "llama-cpp-vulkan.zip"
    _download_file(LLAMA_CPP_DLL_URL_TEMPLATE.format(tag=tag), dll_zip, f"llama.cpp Vulkan {tag}")
    with zipfile.ZipFile(str(dll_zip), "r") as archive:
        for member in archive.namelist():
            basename = os.path.basename(member)
            if basename in required_dlls or (
                basename.startswith("ggml-") and basename.endswith(".dll")
            ):
                with archive.open(member) as src, open(bin_dir / basename, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    dll_zip.unlink(missing_ok=True)
    logger.info("llama.cpp Vulkan DLLs extracted to %s", bin_dir)


def prefetch_llama_cpp_vulkan_for_pyinstaller_bundle() -> None:
    """将 llama.cpp Vulkan DLL 写入 local_asr/models/qwen_llama_vulkan_bin，供 Qwen3 资源 zip（手动 workflow）或本机缓存。"""
    apply_cache_env()
    ensure_vendor_sources("qwen3-asr")
    _ensure_llama_cpp_vulkan_to(_qwen_llama_vulkan_user_bin())


def prefetch_sensevoice_for_pyinstaller_bundle() -> None:
    """将 SenseVoice INT8 ONNX 下载到 local_asr/models/sensevoice-onnx，供 SenseVoice 资源 zip（手动 workflow）或本机开发打包。"""
    apply_cache_env()
    dest = _sensevoice_onnx_bundle_dir()
    if _sensevoice_onnx_ready(dest):
        logger.info("SenseVoice ONNX already present under %s", dest)
        return
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SenseVoice ONNX (INT8) to %s", dest)
    snapshot_download(
        repo_id=SENSEVOICE_ONNX_REPO,
        local_dir=str(dest),
        allow_patterns=list(SENSEVOICE_ONNX_FILES),
    )


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
    if engine == "qwen3-asr":
        base_dir = VENDOR_DIR / "qwen_asr_gguf"
        required = list(_QWEN3_VENDOR_URLS) + ["__init__.py", "inference/__init__.py"]
        return all((base_dir / relative_path).exists() for relative_path in required)
    return False


def is_silero_cached() -> bool:
    for path in (_silero_onnx_bundle_path(), _silero_onnx_user_path()):
        if path.is_file() and path.stat().st_size > 100_000:
            return True
    return False


def _ms_model_path(org: str, name: str) -> Path:
    for sub in (
        MODELS_DIR / "modelscope" / org / name,
        MODELS_DIR / "modelscope" / "hub" / "models" / org / name,
    ):
        if sub.exists():
            return sub
    return MODELS_DIR / "modelscope" / org / name


def get_local_model_path(engine: str) -> str | None:
    if engine == "qwen3-asr":
        model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
        return str(model_dir) if model_dir.exists() else None
    if engine == "sensevoice":
        resolved = sensevoice_onnx_model_dir()
        return str(resolved) if resolved else None
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

    # 下载始终走 HuggingFace；仍识别旧版 ModelScope 缓存目录
    return _try_hf() or _try_ms()


def is_qwen3_asr_ready() -> bool:
    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    if not model_dir.exists():
        return False
    for filename in QWEN3_ASR_FILES:
        if not (model_dir / filename).exists():
            return False
    bin_dir = _qwen3_llama_bin_dir()
    if not bin_dir.exists():
        return False
    required = ["llama.dll", "ggml.dll", "ggml-base.dll"] if sys.platform == "win32" else [
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
    ]
    return all((bin_dir / filename).exists() for filename in required)


def is_asr_models_ready(engine: str) -> bool:
    """主 ASR 权重与 Qwen 原生库已就绪（不含 Silero、不检查 Python 包 import）。"""
    if not _vendor_ready(engine):
        return False
    if engine == "qwen3-asr":
        return is_qwen3_asr_ready()
    if engine == "sensevoice":
        return sensevoice_onnx_model_dir() is not None
    model_id = ASR_MODEL_IDS.get(engine)
    if not model_id:
        return False
    org, name = model_id.split("/")
    if _ms_model_path(org, name).exists():
        return True
    return (MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}").exists()


def is_asr_cached(engine: str) -> bool:
    if not is_silero_cached():
        return False
    if get_engine_runtime_issues(engine):
        return False
    if not _vendor_ready(engine):
        return False
    if engine == "qwen3-asr":
        return is_qwen3_asr_ready()
    if engine == "sensevoice":
        return sensevoice_onnx_model_dir() is not None
    model_id = ASR_MODEL_IDS.get(engine)
    if not model_id:
        return False
    org, name = model_id.split("/")
    if _ms_model_path(org, name).exists():
        return True
    return (MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}").exists()


def get_missing_models(engine: str) -> list[dict]:
    missing: list[dict] = []
    if not is_silero_cached():
        missing.append(
            {
                "name": "Silero VAD",
                "type": "silero-vad",
                "estimated_bytes": _MODEL_SIZE_BYTES["silero-vad"],
            }
        )
    if not is_asr_cached(engine):
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
    if is_silero_cached():
        logger.info("Silero VAD ONNX already present")
        return
    dest = _silero_onnx_user_path()
    logger.info("Downloading Silero VAD (ONNX)...")
    _download_file(SILERO_VAD_ONNX_URL, dest, "Silero VAD (ONNX)")
    logger.info("Silero VAD downloaded to %s", dest)


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", desc or url)
    urlretrieve(url, str(dest))


def download_qwen3_asr() -> None:
    apply_cache_env()
    ensure_vendor_sources("qwen3-asr")

    import shutil
    import zipfile

    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = _qwen_llama_vulkan_user_bin()
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

    if sys.platform == "win32":
        _ensure_llama_cpp_vulkan_to(bin_dir)
    else:
        required_so = ["libllama.so", "libggml.so", "libggml-base.so"]
        if any(not (bin_dir / filename).exists() for filename in required_so):
            logger.warning(
                "Qwen3 llama.cpp Linux .so not auto-downloaded; place them under %s or use system libs",
                bin_dir,
            )


def download_asr(engine: str) -> None:
    apply_cache_env()
    ensure_vendor_sources(engine)

    if engine == "qwen3-asr":
        download_qwen3_asr()
        return
    if engine == "sensevoice":
        if sensevoice_onnx_model_dir() is not None:
            logger.info("SenseVoice ONNX already available (bundled or cached)")
            return
        from huggingface_hub import snapshot_download

        dest = _sensevoice_onnx_user_dir()
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading SenseVoice ONNX (INT8) from %s", SENSEVOICE_ONNX_REPO)
        snapshot_download(
            repo_id=SENSEVOICE_ONNX_REPO,
            local_dir=str(dest),
            allow_patterns=list(SENSEVOICE_ONNX_FILES),
        )
        return

    raise ValueError(f"Unknown local ASR engine: {engine}")


def prepare_engine(engine: str) -> None:
    ensure_vendor_sources(engine)
    if not is_silero_cached():
        download_silero()
    if not is_asr_cached(engine):
        download_asr(engine)


def get_engine_status(engine: str) -> dict:
    return {
        "engine": engine,
        "display_name": LOCAL_ASR_DISPLAY_NAMES.get(engine, engine),
        "runtime_issues": get_engine_runtime_issues(engine),
        "vendor_ready": _vendor_ready(engine),
        "model_cached": bool(get_local_model_path(engine)) if engine != "qwen3-asr" else is_qwen3_asr_ready(),
        "ready": is_asr_cached(engine),
        "missing": get_missing_models(engine),
    }

