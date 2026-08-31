from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import (
    LOCAL_INFERENCE_DISPLAY_NAMES,
    LOCAL_INFERENCE_ENGINES,
    get_common_runtime_issues,
    get_engine_runtime_issues,
)
from proxy_detector import refresh_system_proxy_env

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
        # Windows: ggml CPU backend depends on OpenMP runtime in this folder.
        names = ("llama.dll", "ggml.dll", "ggml-base.dll", "libomp140.x86_64.dll")
    else:
        # Linux: OpenMP runtime (libgomp/libomp) is expected from system packages.
        # We only require llama/ggml core .so files here for readiness checks.
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


def _gpu_used_mb() -> int | None:
    """当前显卡已用显存（MiB）；拿不到就返回 None。纯诊断用，失败不影响任何流程。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


@contextmanager
def warn_if_weights_not_resident(label: str, weights_path):
    """包住一次模型加载，粗略检查权重是否在加载时进入了独显显存。

    Windows/WDDM 在独显显存紧张时可以把 Vulkan 的 device-local allocation 分页到共享
    GPU 内存。模型仍显示为 Vulkan 模型，kernel 也仍在 GPU 上跑，但每次前向都要经 PCIe
    读取权重：实测 RTX 3080 上单 token 往返会从约 3.3ms 增至约 56ms。

    这里只比较加载前后的全卡占用，因此是辅助诊断，不是驻留性的完整证明；驱动也可能在
    加载完成后才换出 allocation。真正的防护由 ``GGML_VK_ENABLE_MEMORY_PRIORITY`` 提供。
    """
    try:
        size_mb = Path(weights_path).stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    before = _gpu_used_mb()
    try:
        yield
    finally:
        after = _gpu_used_mb()
        if before is None or after is None or size_mb <= 0:
            return
        delta = after - before
        # 留一半的余量：KV 与 compute buffer 也算在 delta 里，权重进了显存就不可能只涨这么点
        if delta < size_mb * 0.5:
            logger.warning(
                "%s 的权重没有进显存：预期 +%.0fMiB，实际只 +%dMiB。"
                "权重可能已被放进共享 GPU 内存，之后每次前向都要经 PCIe 读取，"
                "解码会慢一个数量级。请释放其他程序占用的显存后重新加载模型。",
                label, size_mb, delta,
            )
        else:
            logger.info("%s 权重已驻留显存（+%dMiB）", label, delta)


def prepare_qwen_llama_runtime_env() -> None:
    """在首次 import llama/Vulkan 后端前配置 DLL 路径和显存驻留策略。"""
    os.environ["YAKUTAN_QWEN_LLAMA_BIN"] = str(_qwen_llama_vulkan_user_bin().resolve())
    # llama.cpp 的 Vulkan 后端默认不给 allocation 标记优先级。Windows/WDDM 在显存压力下
    # 会优先把空闲的模型权重换到 shared GPU memory，下一次 forward 才经 PCIe 重新读取。
    # bundled ggml-vulkan.dll 支持该开关；支持 VK_EXT_memory_priority 的驱动会给模型、KV、
    # compute buffer 的 device allocation 设置最高优先级，不支持的驱动会安全忽略。
    os.environ["GGML_VK_ENABLE_MEMORY_PRIORITY"] = "1"
    # llama.cpp 以“变量是否存在”判断该低速回退（即使值是 "0" 也会启用）。Yakutan 的本地
    # 实时模型宁可明确加载失败，也不能把权重直接分配进系统内存后无提示地慢十倍。
    os.environ.pop("GGML_VK_ALLOW_SYSMEM_FALLBACK", None)


def _default_models_dir() -> Path:
    from resource_path import get_user_data_path

    return Path(get_user_data_path("local_asr_models"))


MODELS_DIR = _default_models_dir()

# 单文件运行时 VENDOR_DIR 在 _MEIPASS 下，仅用于读取打包自带的只读副本；可写 vendor 代码见 _qwen_vendor_package_dir()
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

# HaujetZhao/Qwen3-ASR-GGUF zip 内仅有 Encoder ONNX×2 + Decoder GGUF；llama.cpp Vulkan DLL 由 download_qwen3_asr() / prefetch_llama_cpp_vulkan_for_pyinstaller_bundle() 从 ggml-org 拉取，不进 Git（手动 workflow「Pack Local Inference model bundles」或本机运行同上函数下载）。
QWEN3_ASR_MODEL_URL = (
    "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/"
    "Qwen3-ASR-1.7B-gguf.zip"
)
LLAMA_CPP_DLL_URL_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/"
    "llama-{tag}-bin-win-vulkan-x64.zip"
)
LLAMA_CPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
LLAMA_CPP_VULKAN_ASSET_SUFFIX = "-bin-win-vulkan-x64.zip"
# API 查不到时用的兜底版本（该 tag 的 zip 已确认仍可下载）
LLAMA_CPP_FALLBACK_TAG = "b8391"

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
    "hymt2": 1_100_000_000,
}

HYMT2_DIR_NAME = "hymt2"
HYMT2_REPO_ID = "febilly/Hy-MT2-1.8B-StreamRevise-v4-GGUF"
# 仓库里还有 .imatrix.gguf（量化用的校准数据），推理只需要这一个权重
HYMT2_MODEL_FILE = "Hy-MT2-1.8B-StreamRevise-v4-Q4_K_M.gguf"

HF_ENDPOINT_OFFICIAL = "https://huggingface.co"
# huggingface.co 不通时按顺序回退的镜像站（与官方站同构，resolve 路径一致）
HF_ENDPOINT_MIRRORS = ("https://hf-mirror.com",)

_QWEN3_VENDOR_URLS = {
    "asr_engine.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/asr_engine.py",
    "inference/asr.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/asr.py",
    "inference/schema.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/schema.py",
    "inference/encoder.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/encoder.py",
    "inference/llama.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/llama.py",
    "inference/utils.py": "https://raw.githubusercontent.com/TheDeathDragon/LiveTranslate/main/qwen_asr_gguf/inference/utils.py",
}


def _qwen_vendor_files_complete(base_dir: Path) -> bool:
    required = list(_QWEN3_VENDOR_URLS) + ["__init__.py", "inference/__init__.py"]
    return all((base_dir / relative_path).is_file() for relative_path in required)


def apply_cache_env() -> None:
    resolved = str(MODELS_DIR.resolve())
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    # 让 HuggingFace Hub、urllib 下载、websocket/socks 代理等统一复用系统代理。
    refresh_system_proxy_env()


def _qwen_vendor_package_dir() -> Path:
    """Qwen 胶水代码：开发态用仓库 vendor；冻结构建优先用打包进 _MEIPASS 的只读副本，完整则不再依赖 local_asr_models 或联网下载。"""
    if not getattr(sys, "frozen", False):
        return VENDOR_DIR / "qwen_asr_gguf"
    bundle = VENDOR_DIR / "qwen_asr_gguf"
    if bundle.is_dir() and _qwen_vendor_files_complete(bundle):
        return bundle
    return MODELS_DIR / "vendor" / "qwen_asr_gguf"


def _seed_qwen_vendor_from_bundle(dest: Path) -> None:
    """将 exe 内只读 vendor 树合并复制到 dest，避免首次运行向 _MEIPASS 下载且重启后丢失。"""
    if not getattr(sys, "frozen", False):
        return
    bundle_root = VENDOR_DIR / "qwen_asr_gguf"
    if not bundle_root.is_dir():
        return
    binary_suffixes = {".dll", ".so", ".dylib"}
    for path in bundle_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in binary_suffixes:
            continue
        rel = path.relative_to(bundle_root)
        dst = dest / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def _silero_onnx_bundle_path() -> Path:
    return PACKAGE_DIR / "models" / SILERO_VAD_DIR_NAME / SILERO_VAD_ONNX_NAME


def _silero_onnx_user_path() -> Path:
    if getattr(sys, "frozen", False):
        from resource_path import get_user_data_path

        return Path(get_user_data_path(os.path.join("local_inference", "models", SILERO_VAD_DIR_NAME, SILERO_VAD_ONNX_NAME)))
    return PACKAGE_DIR / "models" / SILERO_VAD_DIR_NAME / SILERO_VAD_ONNX_NAME


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


def _resolve_llama_cpp_vulkan_url() -> tuple[str, str]:
    """在最近的 release 里找 Windows Vulkan 包，返回 (下载地址, 版本名)。

    不能再拿 releases/latest 的 tag 去拼文件名：ggml-org/llama.cpp 的 latest 现在是
    v0.3.0 这种只放 nightly-tag.txt 的标记版，拼出来的 zip 是 404；真正的 Windows 构建挂在
    b##### 标签下。所以直接按资产名找，找不到再退回内置版本。
    """
    import json

    try:
        request = Request(LLAMA_CPP_RELEASES_API, headers={"User-Agent": "Yakutan"})
        with urlopen(request, timeout=15) as response:
            releases = json.loads(response.read())
        for release in releases:
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                if name.startswith("llama-") and name.endswith(LLAMA_CPP_VULKAN_ASSET_SUFFIX):
                    return asset["browser_download_url"], release.get("tag_name") or name
    except Exception as exc:
        logger.info("查询 llama.cpp release 失败，改用内置版本 %s: %s", LLAMA_CPP_FALLBACK_TAG, exc)

    return LLAMA_CPP_DLL_URL_TEMPLATE.format(tag=LLAMA_CPP_FALLBACK_TAG), LLAMA_CPP_FALLBACK_TAG


def _ensure_llama_cpp_vulkan_to(bin_dir: Path, progress_callback=None, *,
                                allow_vendor_copy: bool = True) -> None:
    """若 bin_dir 中缺少 llama/ggml 主库，则从 ggml-org llama.cpp win-vulkan x64 release 解压写入。

    ``allow_vendor_copy=False`` 时忽略包内 vendor/bin 那份：打包资源 zip 需要 DLL 实际落在
    bin_dir 里，不能因为本机 vendor 目录里有一份就跳过。
    """
    import zipfile

    if sys.platform != "win32":
        logger.info("Skipping llama.cpp Vulkan bundle download on %s", sys.platform)
        return

    required_dlls = ["llama.dll", "ggml.dll", "ggml-base.dll"]
    runtime_dlls = required_dlls + ["libomp140.x86_64.dll"]
    if all((bin_dir / filename).is_file() for filename in runtime_dlls):
        return
    if allow_vendor_copy and _llama_vulkan_bin_has_core_dlls(_qwen_llama_vulkan_vendor_bin()):
        # 打包版可能把运行时放在包内 vendor/bin：那份 _qwen3_llama_bin_dir() 已经会用，
        # 不必再从 GitHub 拉一份（那边被墙时还会让整个下载算作失败）
        logger.info("llama.cpp Vulkan 运行时已随程序打包，跳过下载")
        return

    bin_dir.mkdir(parents=True, exist_ok=True)
    url, label = _resolve_llama_cpp_vulkan_url()
    dll_zip = MODELS_DIR / "llama-cpp-vulkan.zip"
    _download_file(url, dll_zip, f"llama.cpp Vulkan 运行时 {label}", progress_callback)
    _report(progress_callback, "解压 llama.cpp Vulkan 运行时", 0, None)
    with zipfile.ZipFile(str(dll_zip), "r") as archive:
        for member in archive.namelist():
            basename = os.path.basename(member)
            if basename in required_dlls or (
                basename.startswith("ggml-") and basename.endswith(".dll")
            ) or (
                basename.startswith("libomp") and basename.endswith(".dll")
            ):
                with archive.open(member) as src, open(bin_dir / basename, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    dll_zip.unlink(missing_ok=True)
    logger.info("llama.cpp Vulkan DLLs extracted to %s", bin_dir)


def prefetch_llama_cpp_vulkan_for_pyinstaller_bundle() -> None:
    """将 llama.cpp Vulkan DLL 写入 local_inference/models/qwen_llama_vulkan_bin，供 Qwen3 资源 zip（手动 workflow）或本机缓存。"""
    apply_cache_env()
    ensure_vendor_sources("qwen3-asr")
    # 资源 zip 是按 local_asr_models/qwen_llama_vulkan_bin 打的，必须真的下到那里
    _ensure_llama_cpp_vulkan_to(_qwen_llama_vulkan_user_bin(), allow_vendor_copy=False)


def prefetch_sensevoice_for_pyinstaller_bundle() -> None:
    """将 SenseVoice INT8 ONNX 下载到 local_inference/models/sensevoice-onnx，供 SenseVoice 资源 zip（手动 workflow）或本机开发打包。"""
    apply_cache_env()
    dest = _sensevoice_onnx_bundle_dir()
    if _sensevoice_onnx_ready(dest):
        logger.info("SenseVoice ONNX already present under %s", dest)
        return
    logger.info("Downloading SenseVoice ONNX (INT8) to %s", dest)
    _download_sensevoice_onnx(dest)


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
        "import logging\n\nlogger = logging.getLogger('Yakutan.LocalInference.Qwen3')\n",
    )
    _write_text_if_missing(
        base_dir / "inference" / "__init__.py",
        "from .. import logger as logger  # noqa: F401\n",
    )


def ensure_vendor_sources(engine: str) -> Path | None:
    if engine == "sensevoice":
        return None

    if engine == "qwen3-asr":
        base_dir = _qwen_vendor_package_dir()
        if _qwen_vendor_files_complete(base_dir):
            return base_dir
        if getattr(sys, "frozen", False):
            _seed_qwen_vendor_from_bundle(base_dir)
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
        return _qwen_vendor_files_complete(_qwen_vendor_package_dir())
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
    # Keep platform-specific runtime checks aligned with _llama_vulkan_bin_has_core_dlls().
    required = ["llama.dll", "ggml.dll", "ggml-base.dll", "libomp140.x86_64.dll"] if sys.platform == "win32" else [
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
                "name": LOCAL_INFERENCE_DISPLAY_NAMES.get(engine, engine),
                "type": engine,
                "estimated_bytes": _MODEL_SIZE_BYTES.get(engine, 0),
            }
        )
    return missing


def download_silero(progress_callback=None) -> None:
    apply_cache_env()
    if is_silero_cached():
        logger.info("Silero VAD ONNX already present")
        return
    dest = _silero_onnx_user_path()
    logger.info("Downloading Silero VAD (ONNX)...")
    _download_file(SILERO_VAD_ONNX_URL, dest, "Silero VAD", progress_callback)
    logger.info("Silero VAD downloaded to %s", dest)


def _download_file(url: str, dest: Path, desc: str = "", progress_callback=None) -> None:
    _stream_download(url, dest, desc or dest.name, progress_callback)


_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_PROGRESS_INTERVAL_SEC = 0.5

# 下载完成后按扩展名核对文件头：拿到的若是错误页面或半截文件，宁可当场失败也不要留到加载模型时
_FILE_MAGIC = {".gguf": b"GGUF", ".zip": b"PK"}


def _report(progress_callback, stage: str, done: int, total: int | None) -> None:
    """进度回调统一入口：progress_callback(阶段名, 已下载字节, 总字节或 None)。"""
    if progress_callback is None:
        return
    try:
        progress_callback(stage, done, total)
    except Exception:
        # 进度只是展示，回调里的问题不该中断下载
        logger.debug("进度回调失败", exc_info=True)


def _hf_endpoints() -> list[str]:
    """HuggingFace 下载端点的尝试顺序：环境变量覆盖 > 官方站 > 镜像站。"""
    endpoints: list[str] = []
    for value in (
        os.environ.get("YAKUTAN_HF_ENDPOINT"),
        os.environ.get("HF_ENDPOINT"),
        HF_ENDPOINT_OFFICIAL,
        *HF_ENDPOINT_MIRRORS,
    ):
        endpoint = (value or "").strip().rstrip("/")
        if endpoint and endpoint not in endpoints:
            endpoints.append(endpoint)
    return endpoints


def _hf_file_url(endpoint: str, repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{endpoint}/{repo_id}/resolve/{revision}/{filename}"


def _hf_endpoint_reachable(url: str, timeout: float = 8.0) -> bool:
    """只取首字节探测端点：既验证站点本身，也验证它重定向到的 CDN（国内常见只有后者被墙）。"""
    try:
        request = Request(url, headers={"User-Agent": "Yakutan", "Range": "bytes=0-0"})
        with urlopen(request, timeout=timeout) as response:
            response.read(1)
        return True
    except Exception as exc:
        logger.info("HuggingFace 端点探测失败 %s: %s", url, exc)
        return False


def _ordered_hf_endpoints(repo_id: str, filename: str) -> list[str]:
    """把首个探测可达的端点排到最前；探测失败的端点仍留在末尾兜底（探测本身也可能误判）。"""
    endpoints = _hf_endpoints()
    unreachable: list[str] = []
    for index, endpoint in enumerate(endpoints):
        remaining = endpoints[index + 1:]
        if not remaining:
            # 已经是最后一个候选，没必要再多花一次探测的超时
            return [endpoint, *unreachable]
        if _hf_endpoint_reachable(_hf_file_url(endpoint, repo_id, filename)):
            return [endpoint, *remaining, *unreachable]
        logger.warning("%s 不可达，改用后备端点", endpoint)
        unreachable.append(endpoint)
    return unreachable


def _parse_content_range_total(headers) -> int | None:
    """从 Content-Range 头里取出资源总长度（"bytes 0-3/1069288640" / "bytes */1069288640"）。"""
    value = headers.get("Content-Range") if headers else None
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1].strip()
    return int(total) if total.isdigit() else None


def _verify_download(path: Path, expected_bytes: int | None, magic: bytes | None) -> None:
    """校验下载结果。magic 取自最终文件名的扩展名——path 本身是 .part，看它的后缀没意义。"""
    actual = path.stat().st_size
    if expected_bytes is not None and actual != expected_bytes:
        raise OSError(f"文件大小不符：期望 {expected_bytes} 字节，实际 {actual} 字节")
    if magic:
        with open(path, "rb") as handle:
            if handle.read(len(magic)) != magic:
                raise OSError("文件头不对（可能是错误页面或被网关拦截）")


def _stream_download(url: str, dest: Path, desc: str, progress_callback=None,
                     timeout: float = 30.0) -> None:
    """流式下载到 dest.part（支持断点续传），校验通过后再改名到 dest。

    progress_callback(阶段名, 已下载字节, 总字节或 None) 最多每 0.5s 调一次，另在结束时补一次。
    中断留下的 .part 会在下次调用（含换用镜像端点）时续传；只有校验失败才删掉重来。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.is_file() else 0
    magic = _FILE_MAGIC.get(dest.suffix.lower())

    headers = {"User-Agent": "Yakutan"}
    if done:
        headers["Range"] = f"bytes={done}-"
        logger.info("%s 续传：已有 %.1f MB", desc, done / (1024 * 1024))

    try:
        response = urlopen(Request(url, headers=headers), timeout=timeout)
    except HTTPError as exc:
        if exc.code == 416 and done:
            # 请求范围越界：.part 要么已经是完整文件，要么比远端还长（坏了）。
            # 416 的 Content-Range 形如 "bytes */<total>"，用它判断是哪一种。
            _verify_or_discard(part, _parse_content_range_total(exc.headers), magic)
            os.replace(part, dest)
            _report(progress_callback, desc, done, done)
            return
        raise

    with response:
        status = getattr(response, "status", None) or response.getcode()
        if done and status != 206:
            logger.info("%s 服务端不支持断点续传，从头下载", desc)
            done = 0
        if (response.headers.get_content_type() or "") == "text/html":
            # 二进制权重不会以 HTML 返回：这是门户/网关的拦截页
            raise OSError(f"{desc} 返回的是 HTML 页面（可能被网络网关拦截）")
        length = response.headers.get("Content-Length")
        total = done + int(length) if length and length.isdigit() else None

        _report(progress_callback, desc, done, total)
        last_report = time.monotonic()
        with open(part, "ab" if done else "wb") as handle:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_report >= _PROGRESS_INTERVAL_SEC:
                    last_report = now
                    _report(progress_callback, desc, done, total)

    _report(progress_callback, desc, done, total)
    _verify_or_discard(part, total, magic)
    os.replace(part, dest)


def _verify_or_discard(part: Path, expected_bytes: int | None, magic: bytes | None) -> None:
    try:
        _verify_download(part, expected_bytes, magic)
    except OSError:
        # 内容坏了就没法续传，删掉让下一次从头下
        part.unlink(missing_ok=True)
        raise


def _extract_members(archive, members, dest_dir: Path, total: int | None,
                     progress_callback=None, stage: str = "解压 Qwen3-ASR 模型") -> None:
    """把 zip 里的成员平铺写入 dest_dir，并按写出的字节数报进度。"""
    done = 0
    last_report = 0.0
    for member in members:
        with archive.open(member) as src, open(dest_dir / os.path.basename(member), "wb") as dst:
            while True:
                chunk = src.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_report >= _PROGRESS_INTERVAL_SEC:
                    last_report = now
                    _report(progress_callback, stage, done, total)
    _report(progress_callback, stage, done, total)


def download_qwen3_asr(progress_callback=None) -> None:
    apply_cache_env()
    ensure_vendor_sources("qwen3-asr")

    import zipfile

    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = _qwen_llama_vulkan_user_bin()
    bin_dir.mkdir(parents=True, exist_ok=True)

    if any(not (model_dir / filename).exists() for filename in QWEN3_ASR_FILES):
        zip_path = MODELS_DIR / "qwen3-asr-1.7b-gguf.zip"
        _download_file(QWEN3_ASR_MODEL_URL, zip_path, "Qwen3-ASR 模型", progress_callback)
        with zipfile.ZipFile(str(zip_path), "r") as archive:
            members = [m for m in archive.namelist() if os.path.basename(m) in QWEN3_ASR_FILES]
            # 解压 770MB 要几十秒；按已写出的字节报进度，免得界面卡在 100% 上不动
            total = sum(archive.getinfo(member).file_size for member in members)
            _extract_members(archive, members, model_dir, total, progress_callback)
        zip_path.unlink(missing_ok=True)

    if sys.platform == "win32":
        _ensure_llama_cpp_vulkan_to(bin_dir, progress_callback)
    else:
        # Linux runtime still expects system OpenMP libraries (e.g. libgomp/libomp).
        required_so = ["libllama.so", "libggml.so", "libggml-base.so"]
        if any(not (bin_dir / filename).exists() for filename in required_so):
            logger.warning(
                "Qwen3 llama.cpp Linux .so not auto-downloaded; place them under %s or use system libs",
                bin_dir,
            )


def _download_sensevoice_onnx(dest: Path, progress_callback=None) -> None:
    """逐个文件下载 SenseVoice ONNX：和其他模型走同一套端点回退 / 续传 / 进度。"""
    dest.mkdir(parents=True, exist_ok=True)
    pending = [name for name in SENSEVOICE_ONNX_FILES if not (dest / name).is_file()]
    endpoints = _ordered_hf_endpoints(SENSEVOICE_ONNX_REPO, SENSEVOICE_ENCODER_ONNX)

    for index, name in enumerate(pending, start=1):
        stage = f"SenseVoice {name}（{index}/{len(pending)}）"
        failures: list[str] = []
        for endpoint in endpoints:
            host = urlparse(endpoint).netloc or endpoint
            try:
                _stream_download(
                    _hf_file_url(endpoint, SENSEVOICE_ONNX_REPO, name),
                    dest / name, stage, progress_callback,
                )
                break
            except Exception as exc:
                logger.warning("从 %s 下载 %s 失败: %s", host, name, exc)
                failures.append(f"{host}: {exc}")
        else:
            raise RuntimeError(f"SenseVoice {name} 下载失败（" + "；".join(failures) + "）")


def download_asr(engine: str, progress_callback=None) -> None:
    apply_cache_env()
    ensure_vendor_sources(engine)

    if engine == "qwen3-asr":
        download_qwen3_asr(progress_callback)
        return
    if engine == "sensevoice":
        if sensevoice_onnx_model_dir() is not None:
            logger.info("SenseVoice ONNX already available (bundled or cached)")
            return
        logger.info("Downloading SenseVoice ONNX (INT8) from %s", SENSEVOICE_ONNX_REPO)
        _download_sensevoice_onnx(_sensevoice_onnx_user_dir(), progress_callback)
        return

    raise ValueError(f"Unknown local ASR engine: {engine}")


def prepare_engine(engine: str, progress_callback=None) -> None:
    ensure_vendor_sources(engine)
    if not is_silero_cached():
        download_silero(progress_callback)
    if not is_asr_cached(engine):
        download_asr(engine, progress_callback)


def get_hymt2_model_path() -> Path | None:
    """获取本地 Hy-MT2 GGUF 模型文件路径（如果存在）。"""
    hymt2_dir = MODELS_DIR / HYMT2_DIR_NAME
    if hymt2_dir.is_dir():
        gguf_files = sorted(hymt2_dir.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0]
    return None


def is_hymt2_cached() -> bool:
    """检测本地 Hy-MT2 GGUF 模型是否已就绪。"""
    return get_hymt2_model_path() is not None


def download_hymt2(progress_callback=None) -> None:
    """下载本地 Hy-MT2 GGUF 权重；huggingface.co 不可达时自动改用镜像站。"""
    apply_cache_env()
    hymt2_dir = MODELS_DIR / HYMT2_DIR_NAME
    hymt2_dir.mkdir(parents=True, exist_ok=True)
    if is_hymt2_cached():
        logger.info("Hy-MT2 GGUF model already present: %s", get_hymt2_model_path())
    else:
        _download_hymt2_weights(hymt2_dir / HYMT2_MODEL_FILE, progress_callback)

    # 本地 Hy-MT2 和 Qwen3-ASR 共用同一套 llama.cpp 运行时。只装翻译模型的用户机器上
    # 这些库还不存在，缺了就只能在加载模型时报 ctypes 错误，所以随权重一起备好。
    _ensure_llama_cpp_vulkan_to(_qwen_llama_vulkan_user_bin(), progress_callback)


def _download_hymt2_weights(dest: Path, progress_callback=None) -> None:
    """按端点优先级依次尝试，全部失败才抛错（错误信息里带上每个端点的原因）。"""
    failures: list[str] = []
    for endpoint in _ordered_hf_endpoints(HYMT2_REPO_ID, HYMT2_MODEL_FILE):
        host = urlparse(endpoint).netloc or endpoint
        url = _hf_file_url(endpoint, HYMT2_REPO_ID, HYMT2_MODEL_FILE)
        try:
            logger.info("从 %s 下载 Hy-MT2 模型（%s）", host, HYMT2_MODEL_FILE)
            _stream_download(url, dest, f"Hy-MT2 模型（{host}）", progress_callback)
            logger.info("Hy-MT2 模型已下载到 %s", dest)
            return
        except Exception as exc:
            logger.warning("从 %s 下载 Hy-MT2 模型失败: %s", host, exc)
            failures.append(f"{host}: {exc}")

    raise RuntimeError("Hy-MT2 模型下载失败（" + "；".join(failures) + "）")


def get_hymt2_status() -> dict:
    hymt2_path = get_hymt2_model_path()
    return {
        "engine": "hymt2",
        "display_name": "Hy-MT2 1.8B",
        "model_file": hymt2_path.name if hymt2_path else None,
        "model_path": str(hymt2_path) if hymt2_path else None,
        "ready": hymt2_path is not None,
        "runtime_issues": get_common_runtime_issues(),
    }


def get_engine_status(engine: str) -> dict:
    if engine == "hymt2":
        return get_hymt2_status()
    return {
        "engine": engine,
        "display_name": LOCAL_INFERENCE_DISPLAY_NAMES.get(engine, engine),
        "runtime_issues": get_engine_runtime_issues(engine),
        "vendor_ready": _vendor_ready(engine),
        "model_cached": bool(get_local_model_path(engine)) if engine != "qwen3-asr" else is_qwen3_asr_ready(),
        "ready": is_asr_cached(engine),
        "missing": get_missing_models(engine),
    }


def get_all_local_models_status() -> dict:
    asr_engines = {}
    for engine in LOCAL_INFERENCE_ENGINES:
        try:
            asr_engines[engine] = get_engine_status(engine)
        except Exception as e:
            asr_engines[engine] = {
                "engine": engine,
                "display_name": LOCAL_INFERENCE_DISPLAY_NAMES.get(engine, engine),
                "ready": False,
                "error": str(e),
            }
    translation_engines = {
        "hymt2": get_hymt2_status()
    }
    return {
        "asr": asr_engines,
        "translation": translation_engines,
    }
