from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
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
LLAMA_CPP_TAG = "b11985"
QWEN_LLAMA_VULKAN_BIN_DIR_NAME = f"llama_vulkan_{LLAMA_CPP_TAG}"


def _qwen_llama_vulkan_user_bin() -> Path:
    return MODELS_DIR / QWEN_LLAMA_VULKAN_BIN_DIR_NAME


def _qwen_llama_vulkan_vendor_bin() -> Path:
    # Versioned bundle path; never load the old inference/bin DLLs with new ctypes.
    return PACKAGE_DIR / "models" / QWEN_LLAMA_VULKAN_BIN_DIR_NAME


def _llama_vulkan_bin_has_core_dlls(path: Path) -> bool:
    if not path.is_dir():
        return False
    if sys.platform == "win32" and not any((path / name).is_file() for name in LLAMA_CPP_OPENMP_DLLS):
        return False
    return all((path / name).is_file() for name in _runtime_names())


def _qwen3_llama_bin_dir() -> Path:
    """解析 b11985 运行时目录；优先用户数据，其次为同版本只读包。"""
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
    os.environ["YAKUTAN_QWEN_LLAMA_BIN"] = str(_qwen3_llama_bin_dir().resolve())
    # llama.cpp 的 Vulkan 后端默认不给 allocation 标记优先级。Windows/WDDM 在显存压力下
    # 会优先把空闲的模型权重换到 shared GPU memory，下一次 forward 才经 PCIe 重新读取。
    # bundled ggml-vulkan.dll 支持该开关；支持 VK_EXT_memory_priority 的驱动会给模型、KV、
    # compute buffer 的 device allocation 设置最高优先级，不支持的驱动会安全忽略。
    os.environ["GGML_VK_ENABLE_MEMORY_PRIORITY"] = "1"
    # llama.cpp 以“变量是否存在”判断该低速回退（即使值是 "0" 也会启用）。Yakutan 的本地
    # 实时模型宁可明确加载失败，也不能把权重直接分配进系统内存后无提示地慢十倍。
    os.environ.pop("GGML_VK_ALLOW_SYSMEM_FALLBACK", None)
    # ggml-vulkan 默认优先把权重放进 device-local + host-visible 的内存类型（ReBAR）。
    # Windows 会把这类分配记在 shared GPU memory 里，Qwen3-ASR 与 Hy-MT2 同进程加载时
    # 两个模型的权重都落在那里，前向一次比普通显存慢数倍。关掉后只用普通 device-local 内存。
    os.environ["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] = "1"


def _default_models_dir() -> Path:
    from resource_path import get_user_data_path

    return Path(get_user_data_path("local_asr_models"))


MODELS_DIR = _default_models_dir()

# 单文件运行时 VENDOR_DIR 在 _MEIPASS 下，仅用于读取随程序打包的 Python 绑定。
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

QWEN3_ASR_VARIANTS = {
    "1.7B": {
        "repo": "mradermacher/Qwen3-ASR-1.7B-GGUF",
        "revision": "664b960e358348943aefcdc26b3cf09eff1fb0ac",
        "decoder": "Qwen3-ASR-1.7B.Q6_K.gguf",
        "mmproj": "Qwen3-ASR-1.7B.mmproj-Q8_0.gguf",
        "sha256": {
            "Qwen3-ASR-1.7B.Q6_K.gguf": "534b653ee0074264c80533df9637e9eb556c04457f30cd1ba08fe48dcb608263",
            "Qwen3-ASR-1.7B.mmproj-Q8_0.gguf": "780b5b48dfbc23c219cc9d3da9fdaccc7da567fa661b3724d19bda50fbcfc8c3",
        },
    },
    "0.6B": {
        "repo": "ggml-org/Qwen3-ASR-0.6B-GGUF",
        "revision": "928ab958557df9aa2ef1c93e0e83c7ad0933fae2",
        "decoder": "Qwen3-ASR-0.6B-Q8_0.gguf",
        "mmproj": "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        "sha256": {
            "Qwen3-ASR-0.6B-Q8_0.gguf": "bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971",
            "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf": "41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d",
        },
    },
}
QWEN3_ASR_FILES = tuple(QWEN3_ASR_VARIANTS["1.7B"][key] for key in ("decoder", "mmproj"))

QWEN3_ASR_DIR_NAME = "qwen3-asr-gguf"

LOCAL_RUNTIME_REPO = "febilly/realtime-subtitle-local-runtime"
LOCAL_RUNTIME_REVISION = "dc80402dfe3d6be8d90c93c60bd67a77a4c36115"
LLAMA_RUNTIME_ZIP = f"llama-{LLAMA_CPP_TAG}-bin-win-vulkan-x64-min.zip"
LLAMA_RUNTIME_ZIP_SHA256 = "b007fea086a8a788cc9e266b236cb27b750fa76e20fc45453f995a10d3856f25"
SILERO_VAD_ONNX_SHA256 = "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49"
LEGACY_QWEN3_ASR_DIR_NAME = "qwen3-asr"
LEGACY_QWEN3_ASR_FILES = (
    "qwen3_asr_encoder_frontend.int4.onnx",
    "qwen3_asr_encoder_backend.int4.onnx",
    "qwen3_asr_llm.q4_k.gguf",
)
LEGACY_RUNTIME_DIR_NAME = "qwen_llama_vulkan_bin"
LEGACY_ARCHIVE_PATTERNS = ("qwen3-asr-1.7b-gguf.zip*", "llama-cpp-vulkan-*.zip*", "llama-cpp-vulkan.zip*")

LLAMA_CPP_DLL_URL_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/"
    "llama-{tag}-bin-win-vulkan-x64.zip"
)
# 固定版本，不跟随最新 nightly：vendor 下的 ctypes 绑定是照这一版的 ABI 写的，llama.cpp
# 的 struct 布局隔几百个构建就会变，跟着 latest 走等于把静默传错参数的风险留给实时链路。
# 升级要连同 vendor/qwen_asr_gguf 一起验证后再改这里。
LLAMA_CPP_PINNED_TAG = LLAMA_CPP_TAG
# 新版 llama.cpp 把 MSVC 的 libomp140.x86_64.dll 换成了 LLVM 的 libomp.dll，两种都认
LLAMA_CPP_OPENMP_DLLS = ("libomp140.x86_64.dll", "libomp.dll")

SILERO_VAD_DIR_NAME = "silero_vad"
SILERO_VAD_ONNX_NAME = "silero_vad_16k_op15.onnx"
SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    f"src/silero_vad/data/{SILERO_VAD_ONNX_NAME}"
)

_MODEL_SIZE_BYTES = {
    "silero-vad": 1_300_000,
    "sensevoice": 380_000_000,
    "qwen3-asr": 2_030_000_000,
    "hymt2": 1_100_000_000,
}

HYMT2_DIR_NAME = "hymt2"
HYMT2_REPO_ID = "febilly/Hy-MT2-1.8B-StreamRevise-v4-GGUF"
# 仓库里还有 .imatrix.gguf（量化用的校准数据），推理只需要这一个权重
HYMT2_MODEL_FILE = "Hy-MT2-1.8B-StreamRevise-v4-Q4_K_M.gguf"

HF_ENDPOINT_OFFICIAL = "https://huggingface.co"
# huggingface.co 不通时按顺序回退的镜像站（与官方站同构，resolve 路径一致）
HF_ENDPOINT_MIRRORS = ("https://hf-mirror.com",)


def _qwen_vendor_files_complete(base_dir: Path) -> bool:
    return all((base_dir / name).is_file() for name in
               ("__init__.py", "inference/__init__.py", "inference/llama.py"))


def apply_cache_env() -> None:
    resolved = str(MODELS_DIR.resolve())
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    # 让 HuggingFace Hub、urllib 下载、websocket/socks 代理等统一复用系统代理。
    refresh_system_proxy_env()


def _qwen_vendor_package_dir() -> Path:
    return VENDOR_DIR / "qwen_asr_gguf"


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


def _llama_cpp_vulkan_url() -> str:
    """固定版本的下载地址。

    早先这里查 releases/latest 再拿 tag 拼文件名，但 ggml-org/llama.cpp 的 latest 现在是
    v0.3.0 这种只放 nightly-tag.txt 的标记版，拼出来的 zip 直接 404，而且 API 调用本身是
    成功的，except 兜底根本不会触发。改成跟随最新的 b##### nightly 也不合适：绑定按固定
    ABI 写死，附带文件名还会变（新版就把 libomp140.x86_64.dll 换成了 libomp.dll）。
    """
    return LLAMA_CPP_DLL_URL_TEMPLATE.format(tag=LLAMA_CPP_PINNED_TAG)


def _ensure_llama_cpp_vulkan_to(bin_dir: Path, progress_callback=None, *,
                                allow_vendor_copy: bool = True) -> None:
    if _runtime_ready(bin_dir):
        return
    if allow_vendor_copy and _runtime_ready(_qwen_llama_vulkan_vendor_bin()):
        return
    _download_llama_runtime(bin_dir, progress_callback)
    if not _runtime_ready(bin_dir):
        raise RuntimeError(f"Incomplete llama.cpp {LLAMA_CPP_TAG} runtime: {bin_dir}")


def prefetch_llama_cpp_vulkan_for_pyinstaller_bundle() -> None:
    """将 b11985 Vulkan runtime 写入用户模型目录，供资源包或本机使用。"""
    apply_cache_env()
    ensure_vendor_sources("qwen3-asr")
    # 资源包需要运行时真实落在用户目录，不能跳过到只读包。
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


def ensure_vendor_sources(engine: str) -> Path | None:
    if engine == "sensevoice":
        return None
    if engine != "qwen3-asr":
        raise ValueError(f"Unknown local ASR engine: {engine}")
    package = _qwen_vendor_package_dir()
    if not _qwen_vendor_files_complete(package):
        raise RuntimeError("Qwen3-ASR Python bindings are missing; reinstall Yakutan")
    return package


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
        return str(model_dir) if _qwen_model_ready(model_dir) else None
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
    return get_qwen3_asr_files() is not None and _runtime_ready(_qwen3_llama_bin_dir())


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
    if is_silero_cached():
        return
    dest = _silero_onnx_user_path()
    try:
        _download_hf_file(
            LOCAL_RUNTIME_REPO, SILERO_VAD_ONNX_NAME, dest, "silero", progress_callback,
            revision=LOCAL_RUNTIME_REVISION, sha256=SILERO_VAD_ONNX_SHA256,
        )
    except Exception as exc:
        logger.warning("Silero VAD download from HF failed, trying GitHub: %s", exc)
        _download_file(SILERO_VAD_ONNX_URL, dest, "silero", progress_callback)
    if _sha256_file(dest) != SILERO_VAD_ONNX_SHA256:
        dest.unlink(missing_ok=True)
        raise OSError("Silero VAD sha256 mismatch")


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


def download_qwen3_asr(progress_callback=None) -> None:
    variant = qwen3_asr_variant()
    model_dir = MODELS_DIR / QWEN3_ASR_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    _ensure_llama_cpp_vulkan_to(_qwen_llama_vulkan_user_bin(), progress_callback)
    for key, stage in (("mmproj", "qwen3-asr-encoder"), ("decoder", "qwen3-asr")):
        filename = variant[key]
        dest = model_dir / filename
        if not dest.is_file():
            _download_hf_file(
                variant["repo"], filename, dest, stage, progress_callback,
                revision=variant["revision"], sha256=variant["sha256"].get(filename),
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
        "ready": hymt2_path is not None and _runtime_ready(_qwen3_llama_bin_dir()),
        "runtime_ready": _runtime_ready(_qwen3_llama_bin_dir()),
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
        "legacy": get_legacy_files_status(),
    }


def qwen3_asr_variant() -> dict:
    import config

    size = str(getattr(config, "LOCAL_QWEN_ASR_MODEL", "1.7B") or "1.7B")
    variant = dict(QWEN3_ASR_VARIANTS.get(size) or QWEN3_ASR_VARIANTS["1.7B"])
    # Experiments: other files in the same folder, without touching the table.
    for key, name in (("decoder", "LOCAL_QWEN_ASR_DECODER_FILE"), ("mmproj", "LOCAL_QWEN_ASR_MMPROJ_FILE")):
        override = str(getattr(config, name, "") or "").strip()
        if override:
            variant[key] = override
    return variant


def _qwen_model_ready(path: Path) -> bool:
    variant = qwen3_asr_variant()
    return path.is_dir() and all((path / variant[key]).is_file() for key in ("decoder", "mmproj"))


def _runtime_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-vulkan.dll", "mtmd.dll")
    return ("libllama.so", "libggml.so", "libggml-base.so", "libmtmd.so")


def get_qwen3_asr_files(model_dir: str | None = None) -> tuple[Path, Path] | None:
    """(decoder GGUF, mmproj GGUF) of the configured variant, or None when not downloaded."""
    base = Path(model_dir) if model_dir else None
    if base is None:
        found = get_local_model_path("qwen3-asr")
        if not found:
            return None
        base = Path(found)
    variant = qwen3_asr_variant()
    decoder, mmproj = base / variant["decoder"], base / variant["mmproj"]
    return (decoder, mmproj) if decoder.is_file() and mmproj.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_hf_file(repo_id: str, filename: str, dest: Path, stage: str, progress_callback=None,
                      *, revision: str = "main", sha256: str | None = None) -> None:
    failures: list[str] = []
    for endpoint in _ordered_hf_endpoints(repo_id, filename):
        host = urlparse(endpoint).netloc or endpoint
        try:
            logger.info("Downloading %s from %s", filename, host)
            _stream_download(_hf_file_url(endpoint, repo_id, filename, revision), dest, stage, progress_callback)
            if sha256 and _sha256_file(dest) != sha256:
                dest.unlink(missing_ok=True)
                raise OSError("sha256 mismatch")
            return
        except Exception as exc:
            logger.warning("%s download from %s failed: %s", filename, host, exc)
            failures.append(f"{host}: {exc}")
    raise RuntimeError(f"{filename} download failed (" + "; ".join(failures) + ")")


def _download_llama_runtime(destination: Path, progress_callback=None) -> None:
    if _runtime_ready(destination):
        return
    if sys.platform != "win32":
        raise RuntimeError("Automatic llama.cpp runtime download is currently supported on Windows only")
    destination.mkdir(parents=True, exist_ok=True)
    tag = LLAMA_CPP_TAG
    archive_path = MODELS_DIR / LLAMA_RUNTIME_ZIP
    try:
        _download_hf_file(
            LOCAL_RUNTIME_REPO, LLAMA_RUNTIME_ZIP, archive_path, "llama-runtime", progress_callback,
            revision=LOCAL_RUNTIME_REVISION, sha256=LLAMA_RUNTIME_ZIP_SHA256,
        )
    except Exception as exc:
        logger.warning("llama.cpp runtime download from HF failed, trying GitHub: %s", exc)
        archive_path = MODELS_DIR / f"llama-cpp-vulkan-{tag}.zip"
        _download_file(
            LLAMA_CPP_DLL_URL_TEMPLATE.format(tag=tag), archive_path, "llama-runtime", progress_callback,
        )
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.namelist():
            name = os.path.basename(member)
            if not name:
                continue
            if name in _runtime_names() or (
                name.endswith(".dll") and name.startswith(("ggml-", "libomp"))
            ):
                with archive.open(member) as source, open(destination / name, "wb") as target:
                    shutil.copyfileobj(source, target)
    archive_path.unlink(missing_ok=True)


def _legacy_files() -> list[Path]:
    """Known obsolete files in Yakutan's own model root; never follow links."""
    found: list[Path] = []
    model_dir = MODELS_DIR / LEGACY_QWEN3_ASR_DIR_NAME
    found.extend(model_dir / name for name in LEGACY_QWEN3_ASR_FILES)
    runtime_dir = MODELS_DIR / LEGACY_RUNTIME_DIR_NAME
    runtime_stems = {"llama", "ggml", "ggml-base", "ggml-cpu", "ggml-vulkan",
                     "libllama", "libggml", "libggml-base", "libggml-cpu", "libggml-vulkan",
                     "ggml-rpc", "libomp", "libomp140.x86_64"}
    runtime_stems.update(f"ggml-cpu-{arch}" for arch in (
        "alderlake", "cannonlake", "cascadelake", "cooperlake", "haswell", "icelake",
        "ivybridge", "piledriver", "sandybridge", "sapphirerapids", "skylakex", "sse42", "x64", "zen4",
    ))
    if runtime_dir.is_dir():
        found.extend(
            path for path in runtime_dir.iterdir()
            if path.suffix.lower() in (".dll", ".so")
            and path.stem.lower() in runtime_stems
        )
    if MODELS_DIR.is_dir():
        found.extend(
            path for pattern in LEGACY_ARCHIVE_PATTERNS for path in MODELS_DIR.glob(pattern)
            if LLAMA_CPP_TAG not in path.name
        )
    root = MODELS_DIR.resolve()
    return [path for path in found if path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root)
            and not path.parent.is_symlink()
            and not getattr(path.parent, "is_junction", lambda: False)()]


def qwen3_required_paths(models_root: Path) -> list[Path]:
    """Pure path calculation shared by health checks and the standalone launcher."""
    variant = qwen3_asr_variant()
    model_dir = models_root / QWEN3_ASR_DIR_NAME
    paths = [model_dir / variant[key] for key in ("decoder", "mmproj")]
    return paths + llama_runtime_required_paths(models_root)


def llama_runtime_required_paths(models_root: Path) -> list[Path]:
    runtime_dir = models_root / QWEN_LLAMA_VULKAN_BIN_DIR_NAME
    paths = [runtime_dir / name for name in _runtime_names()]
    if sys.platform == "win32":
        openmp = next((name for name in LLAMA_CPP_OPENMP_DLLS
                       if (runtime_dir / name).is_file()), "libomp.dll")
        paths.append(runtime_dir / openmp)
    return paths


def get_legacy_files_status() -> dict:
    files = _legacy_files()
    return {
        "bytes": sum(path.stat().st_size for path in files),
        "count": len(files),
        "models_dir": str(MODELS_DIR.resolve()),
        "can_cleanup": is_qwen3_asr_ready(),
    }


def remove_legacy_files() -> dict:
    """Delete the old model and runtime; a file still in use (old version running) stays."""
    removed_bytes = 0
    failed: list[str] = []
    for path in _legacy_files():
        try:
            size = path.stat().st_size
            path.unlink()
            removed_bytes += size
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path, exc)
            failed.append(path.name)
    for name in (LEGACY_QWEN3_ASR_DIR_NAME, LEGACY_RUNTIME_DIR_NAME):
        directory = MODELS_DIR / name
        if (directory.is_symlink() or getattr(directory, "is_junction", lambda: False)()
                or not directory.resolve().is_relative_to(MODELS_DIR.resolve())):
            continue
        try:
            directory.rmdir()
        except OSError:
            pass  # missing, or holds files that are not ours
    return {"removed_bytes": removed_bytes, "failed": failed}


def _runtime_ready(path: Path) -> bool:
    return _llama_vulkan_bin_has_core_dlls(path)
