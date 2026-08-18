"""GGML/Vulkan 计算设备枚举与选择。

设备设置在配置里是一个扁平字符串，便于 .env 与网页共用：

* ``'auto'``      —— 自动挑一张 GPU，优先独显；枚举不可用时退回 llama.cpp 默认行为
* ``'cpu'``       —— 纯 CPU（不卸载任何层）
* ``'vulkan:N'``  —— 指定第 N 张 GPU（N 与 llama.cpp 的 ``main_gpu`` 同序）

之所以要显式指定而不是交给 llama.cpp：``llama_model_default_params()`` 的
``split_mode`` 默认是 LAYER 且 ``devices`` 为 NULL，也就是**把层切分到所有可见
GPU**。核显 + 独显的机器上这会把一部分层放到核显，比纯独显更慢。因此这里总是
解析出单一 ``main_gpu``，由 ``load_model`` 配合 ``split_mode = NONE`` 使用。

枚举需要在进程内创建 Vulkan instance，所以对外的 :func:`probe_gpu_devices`
默认放到子进程里跑，避免网页/服务进程长期持有 Vulkan 资源。
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
VULKAN_PREFIX = "vulkan:"

# 全部层卸载到 GPU（llama.cpp 里 99 已远超实际层数，等价于「全部」）
GPU_ALL_LAYERS = 99

# ggml_backend_dev_type
_GGML_DEV_TYPE_GPU = 1

# 独显/核显的名称特征。ggml 的设备 API 只报 CPU/GPU/ACCEL，拿不到 Vulkan 自己的
# DISCRETE_GPU / INTEGRATED_GPU 分类，只能靠描述串猜，因此这里只用于「自动」的
# 排序偏好，判错时用户仍可手动指定。
_DISCRETE_MARKERS = (
    "geforce", "rtx", "gtx", "quadro", "tesla", "nvidia",
    "radeon rx", "radeon pro", "instinct",
    "arc a", "arc b", "arc pro",
)
_INTEGRATED_MARKERS = (
    "uhd graphics", "hd graphics", "iris", "integrated",
    "radeon(tm) graphics", "radeon graphics", "vega", "apu",
    "microsoft basic", "llvmpipe", "software",
)

_probe_cache: Optional[list[dict]] = None


# ── 设置值规整 ─────────────────────────────────────────────────────────

def sanitize_device(value) -> str:
    """把任意输入收敛成 'auto' / 'cpu' / 'vulkan:N'，无法识别时回退 'auto'。"""
    normalized = str(value or "").strip().lower()
    if normalized in ("", "auto", "gpu"):  # 'gpu' 是旧值，等价于自动挑一张 GPU
        return DEVICE_AUTO
    if normalized == DEVICE_CPU:
        return DEVICE_CPU
    if normalized.startswith(VULKAN_PREFIX):
        suffix = normalized[len(VULKAN_PREFIX):]
        if suffix.isdigit():
            return f"{VULKAN_PREFIX}{int(suffix)}"
    return DEVICE_AUTO


def device_gpu_index(setting: str) -> Optional[int]:
    """'vulkan:N' -> N；其它 -> None。"""
    normalized = sanitize_device(setting)
    if normalized.startswith(VULKAN_PREFIX):
        return int(normalized[len(VULKAN_PREFIX):])
    return None


# ── 枚举 ───────────────────────────────────────────────────────────────

def is_discrete_like(description: str) -> bool:
    text = (description or "").lower()
    if any(marker in text for marker in _INTEGRATED_MARKERS):
        return False
    return any(marker in text for marker in _DISCRETE_MARKERS)


def _lib(lib_dir: Path, name: str):
    return ctypes.CDLL(str(lib_dir / name))


def enumerate_gpu_devices() -> list[dict]:
    """在**当前进程**内枚举 GPU 设备（会加载 Vulkan 后端）。

    返回 ``[{'index', 'name', 'description', 'total_bytes', 'free_bytes',
    'discrete'}]``，``index`` 与 llama.cpp 的 ``main_gpu`` 同序（只数 GPU，
    不含 CPU 设备）。失败时返回空列表。
    """
    from .model_manager import prepare_qwen_llama_runtime_env

    prepare_qwen_llama_runtime_env()
    try:
        from .vendor.qwen_asr_gguf.inference import llama as llama_mod
    except Exception as exc:  # pragma: no cover - vendor 源未就绪
        logger.info("GPU 枚举跳过：llama 绑定不可用 (%s)", exc)
        return []

    lib_dir = Path(llama_mod._resolve_llama_lib_dir())
    if not (lib_dir / "ggml.dll").is_file() and not (lib_dir / "libggml.so").is_file():
        logger.info("GPU 枚举跳过：未找到 ggml 运行时 (%s)", lib_dir)
        return []

    original_cwd = Path.cwd()
    # 与 load_model 一致：后端 DLL 是按 cwd/PATH 解析的，不切目录会静默加载失败
    os.chdir(lib_dir)
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.getcwd())
        os.environ["PATH"] = os.getcwd() + os.pathsep + os.environ["PATH"]
        llama_mod.init_llama_lib()

        # dev_* 符号分散在 ggml.dll 与 ggml-base.dll 两个库里
        libs = []
        for name in ("ggml.dll", "ggml-base.dll", "libggml.so", "libggml-base.so"):
            if (lib_dir / name).is_file():
                try:
                    libs.append(_lib(lib_dir, name))
                except OSError:
                    continue

        def pick(symbol: str):
            for lib in libs:
                try:
                    return getattr(lib, symbol)
                except AttributeError:
                    continue
            raise AttributeError(symbol)

        dev_count = pick("ggml_backend_dev_count")
        dev_count.restype = ctypes.c_size_t
        dev_get = pick("ggml_backend_dev_get")
        dev_get.argtypes = [ctypes.c_size_t]
        dev_get.restype = ctypes.c_void_p
        dev_name = pick("ggml_backend_dev_name")
        dev_name.argtypes = [ctypes.c_void_p]
        dev_name.restype = ctypes.c_char_p
        dev_desc = pick("ggml_backend_dev_description")
        dev_desc.argtypes = [ctypes.c_void_p]
        dev_desc.restype = ctypes.c_char_p
        dev_type = pick("ggml_backend_dev_type")
        dev_type.argtypes = [ctypes.c_void_p]
        dev_type.restype = ctypes.c_int
        dev_memory = pick("ggml_backend_dev_memory")
        dev_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        dev_memory.restype = None

        devices: list[dict] = []
        for i in range(dev_count()):
            handle = dev_get(i)
            if not handle or dev_type(handle) != _GGML_DEV_TYPE_GPU:
                continue
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            dev_memory(handle, ctypes.byref(free), ctypes.byref(total))
            description = (dev_desc(handle) or b"").decode("utf-8", "replace")
            devices.append(
                {
                    # 只数 GPU：这个下标才是 llama.cpp 的 main_gpu
                    "index": len(devices),
                    "name": (dev_name(handle) or b"").decode("utf-8", "replace"),
                    "description": description,
                    "total_bytes": int(total.value),
                    "free_bytes": int(free.value),
                    "discrete": is_discrete_like(description),
                }
            )
        return devices
    except Exception as exc:
        logger.info("GPU 枚举失败: %s", exc)
        return []
    finally:
        os.chdir(original_cwd)


def probe_gpu_devices(*, refresh: bool = False, timeout: float = 30.0) -> list[dict]:
    """在子进程里枚举 GPU 设备并缓存结果。

    放子进程的原因：枚举会在进程内建立 Vulkan instance 并加载全部后端 DLL，
    网页进程没必要为了列一次设备长期持有这些资源。
    """
    global _probe_cache
    if _probe_cache is not None and not refresh:
        return _probe_cache

    repo_root = Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "local_inference.gpu_devices"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
        devices = payload if isinstance(payload, list) else []
    except Exception as exc:
        logger.info("GPU 枚举子进程失败: %s", exc)
        devices = []

    _probe_cache = devices
    return devices


def pick_auto_index(devices: list[dict]) -> Optional[int]:
    """「自动」挑一张 GPU：独显优先，其次显存大的，最后索引小的。"""
    if not devices:
        return None
    best = max(
        devices,
        key=lambda d: (
            1 if d.get("discrete") else 0,
            int(d.get("total_bytes") or 0),
            -int(d.get("index") or 0),
        ),
    )
    return int(best["index"])


# ── 解析成 llama.cpp 参数 ──────────────────────────────────────────────

def resolve_device(setting, *, devices: Optional[list[dict]] = None) -> tuple[int, int, str]:
    """把设备设置解析成 ``(n_gpu_layers, main_gpu, effective_setting)``。

    设置里指定的 GPU 已经不存在时回退到「自动」，并把实际生效值一起返回，
    方便调用方记日志或回写界面。
    """
    normalized = sanitize_device(setting)
    if normalized == DEVICE_CPU:
        return 0, 0, DEVICE_CPU

    if devices is None:
        devices = probe_gpu_devices()
    available = {int(d["index"]) for d in devices}

    requested = device_gpu_index(normalized)
    if requested is not None:
        if requested in available:
            return GPU_ALL_LAYERS, requested, normalized
        logger.warning(
            "指定的 GPU vulkan:%s 当前不可用（可用: %s），回退到自动选择",
            requested,
            sorted(available) or "无",
        )
        normalized = DEVICE_AUTO

    auto_index = pick_auto_index(devices)
    if auto_index is None:
        # 枚举不可用（没装 Vulkan 运行时等）：保持 llama.cpp 的默认设备选择，
        # 但仍然只用一张卡（由 split_mode = NONE 保证）
        return GPU_ALL_LAYERS, 0, DEVICE_AUTO
    return GPU_ALL_LAYERS, auto_index, DEVICE_AUTO


def describe_device(setting, *, devices: Optional[list[dict]] = None) -> str:
    """给日志用的一行人类可读描述。"""
    n_gpu_layers, main_gpu, effective = resolve_device(setting, devices=devices)
    if n_gpu_layers == 0:
        return "CPU"
    if devices is None:
        devices = probe_gpu_devices()
    for device in devices:
        if int(device["index"]) == main_gpu:
            return f"GPU {main_gpu} ({device['description']})"
    return f"GPU {main_gpu}"


if __name__ == "__main__":  # 供 probe_gpu_devices 的子进程调用
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    print(json.dumps(enumerate_gpu_devices(), ensure_ascii=False))
