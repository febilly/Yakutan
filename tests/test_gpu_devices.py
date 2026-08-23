"""本地推理设备（GPU/CPU）的选择与回退逻辑。

这些用例不碰真实 GPU：设备列表都是显式传入的，因此在没有显卡的 CI 上也能跑。
"""

import os

import config
from local_inference.gpu_devices import (
    GPU_ALL_LAYERS,
    describe_device,
    is_discrete_like,
    pick_auto_index,
    resolve_device,
    sanitize_device,
)
from local_inference.model_manager import prepare_qwen_llama_runtime_env


def _device(index, description, *, discrete=True, total_gb=8):
    return {
        "index": index,
        "name": f"Vulkan{index}",
        "description": description,
        "total_bytes": int(total_gb * 1024 ** 3),
        "free_bytes": int(total_gb * 1024 ** 3),
        "discrete": discrete,
    }


DISCRETE = _device(0, "NVIDIA GeForce RTX 4060 Laptop GPU", discrete=True, total_gb=8)
INTEGRATED = _device(1, "Intel(R) UHD Graphics 770", discrete=False, total_gb=16)


class TestSanitize:
    def test_accepts_known_forms(self):
        assert sanitize_device("auto") == "auto"
        assert sanitize_device("CPU") == "cpu"
        assert sanitize_device("Vulkan:2") == "vulkan:2"
        assert sanitize_device("vulkan:02") == "vulkan:2"

    def test_falls_back_to_auto(self):
        for value in ("", None, "gpu", "cuda:0", "vulkan:", "vulkan:x", "垃圾"):
            assert sanitize_device(value) == "auto", value

    def test_config_wrapper_matches(self):
        assert config.sanitize_local_device("gpu") == "auto"
        assert config.sanitize_local_device("vulkan:1") == "vulkan:1"
        assert config.sanitize_local_device("cpu") == "cpu"


class TestAutoPick:
    def test_prefers_discrete_over_bigger_integrated(self):
        # 核显常常报告比独显更大的共享内存，只看显存会挑错
        assert pick_auto_index([INTEGRATED, DISCRETE]) == 0

    def test_prefers_larger_memory_among_discrete(self):
        small = _device(0, "NVIDIA GeForce GTX 1650", total_gb=4)
        big = _device(1, "NVIDIA GeForce RTX 4090", total_gb=24)
        assert pick_auto_index([small, big]) == 1

    def test_no_devices(self):
        assert pick_auto_index([]) is None

    def test_name_heuristic(self):
        assert is_discrete_like("NVIDIA GeForce RTX 3080") is True
        assert is_discrete_like("AMD Radeon RX 7900 XTX") is True
        assert is_discrete_like("Intel(R) UHD Graphics 770") is False
        assert is_discrete_like("AMD Radeon(TM) Graphics") is False


class TestResolve:
    def test_cpu_offloads_nothing(self):
        assert resolve_device("cpu", devices=[DISCRETE]) == (0, 0, "cpu")

    def test_auto_picks_discrete(self):
        assert resolve_device("auto", devices=[INTEGRATED, DISCRETE]) == (
            GPU_ALL_LAYERS,
            0,
            "auto",
        )

    def test_explicit_index_is_respected(self):
        # 用户明确选核显（比如独显要留给游戏）就照办
        assert resolve_device("vulkan:1", devices=[DISCRETE, INTEGRATED]) == (
            GPU_ALL_LAYERS,
            1,
            "vulkan:1",
        )

    def test_missing_device_falls_back_to_auto(self):
        assert resolve_device("vulkan:3", devices=[DISCRETE]) == (
            GPU_ALL_LAYERS,
            0,
            "auto",
        )

    def test_without_enumeration_keeps_single_gpu(self):
        # 枚举不可用时仍然只用一张卡（split_mode=NONE + main_gpu=0）
        assert resolve_device("auto", devices=[]) == (GPU_ALL_LAYERS, 0, "auto")
        assert resolve_device("vulkan:1", devices=[]) == (GPU_ALL_LAYERS, 0, "auto")

    def test_describe(self):
        assert describe_device("cpu", devices=[DISCRETE]) == "CPU"
        assert "RTX 4060" in describe_device("auto", devices=[DISCRETE])


class TestLoaderNeverSplits:
    def test_runtime_env_prioritizes_vram_and_disables_silent_sysmem_fallback(
        self, monkeypatch
    ):
        """显存紧张时保住 device-local 权重，不允许 Vulkan 静默分配到系统内存。"""
        monkeypatch.delenv("GGML_VK_ENABLE_MEMORY_PRIORITY", raising=False)
        monkeypatch.setenv("GGML_VK_ALLOW_SYSMEM_FALLBACK", "0")

        prepare_qwen_llama_runtime_env()

        assert os.environ["GGML_VK_ENABLE_MEMORY_PRIORITY"] == "1"
        assert "GGML_VK_ALLOW_SYSMEM_FALLBACK" not in os.environ

    def test_load_model_forces_single_gpu(self):
        """核显+独显的机器上，llama.cpp 默认会把层切分到两张卡，必须显式关掉。"""
        import inspect

        from local_inference.vendor.qwen_asr_gguf.inference import llama as llama_mod

        source = inspect.getsource(llama_mod.load_model)
        assert "split_mode = LLAMA_SPLIT_MODE_NONE" in source
        assert "main_gpu" in source
        assert llama_mod.LLAMA_SPLIT_MODE_NONE == 0
