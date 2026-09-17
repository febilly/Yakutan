"""Windows 远程推理服务端 (winutils / server_windows / launcher) 纯逻辑单测。

不加载任何模型、不绑定网络端口（winutils 端口探测使用临时监听器）。
可直接用 pytest 运行，也可作为普通脚本运行（无 pytest 环境时）。
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RS_DIR = ROOT / "remote_inference_service"
for _p in (str(ROOT), str(RS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import winutils  # noqa: E402
from server_windows import (  # noqa: E402
    ASRSessionState,
    _model_status,
    _normalize_hypothesis,
    _select_warmup_engines,
    _source_text,
    _translation_prompt,
    parse_args,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_get_local_ips_returns_ipv4_list():
    ips = winutils.get_local_ips()
    assert isinstance(ips, list) and ips
    for ip in ips:
        assert ":" not in ip  # 仅 IPv4


def test_check_port_in_use_detects_listener_and_free_port():
    free_port = _free_port()
    assert winutils.check_port_in_use(free_port) is False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = int(srv.getsockname()[1])
        assert winutils.check_port_in_use(port) is True


def test_check_dependencies_returns_installed_and_missing_labels():
    installed, missing = winutils.check_dependencies()
    # 测试环境至少装了 numpy/websockets（server_windows 顶层已导入）
    assert any(label.startswith("numpy") for label in installed)
    assert all(isinstance(label, str) for label in missing)


def test_model_status_reports_missing_and_ready(tmp_dir: Path | None = None):
    with tempfile.TemporaryDirectory() as td:
        models = Path(td)
        # 空目录：全部未就绪
        status = _model_status(models)
        assert status["sensevoice"]["ready"] is False
        assert status["qwen3-asr"]["ready"] is False
        assert status["hymt2"]["ready"] is False

        # SenseVoice 文件齐备
        sv_dir = models / "sensevoice-onnx"
        sv_dir.mkdir(parents=True)
        for name in ("am.mvn", "embedding.npy", "sense-voice-encoder-int8.onnx",
                     "chn_jpn_yue_eng_ko_spectok.bpe.model"):
            (sv_dir / name).write_bytes(b"x")
        status = _model_status(models)
        assert status["sensevoice"]["ready"] is True

        # Hy-MT2：放入任意 gguf 即就绪（进程内引擎）
        hymt2_dir = models / "hymt2"
        hymt2_dir.mkdir(parents=True)
        (hymt2_dir / "Hy-MT2-test.gguf").write_bytes(b"x")
        status = _model_status(models)
        assert status["hymt2"]["ready"] is True

        # Qwen3-ASR：文件齐备 + Windows 下要求 llama.dll
        qwen_dir = models / "qwen3-asr"
        qwen_dir.mkdir(parents=True)
        for name in ("qwen3_asr_encoder_frontend.int4.onnx",
                     "qwen3_asr_encoder_backend.int4.onnx",
                     "qwen3_asr_llm.q4_k.gguf"):
            (qwen_dir / name).write_bytes(b"x")
        vulkan_dir = models / "qwen_llama_vulkan_bin"
        vulkan_dir.mkdir(parents=True)
        dll_name = "llama.dll" if sys.platform == "win32" else "libllama.so"
        (vulkan_dir / dll_name).write_bytes(b"x")
        status = _model_status(models)
        assert status["qwen3-asr"]["ready"] is True


def test_session_state_defaults():
    state = ASRSessionState()
    assert state.context == ""
    assert state.draft_tokens == []
    assert state.qwen_worker is None


def test_parse_args_defaults_are_loopback_and_auto_warmup():
    with patch.object(sys, "argv", ["server_windows.py"]):
        args = parse_args()
    assert args.host == "127.0.0.1"  # 安全默认值：仅本机；局域网需显式 --host 0.0.0.0
    assert args.port == 18775
    assert args.warmup == "auto"
    assert args.microbatch is False


def test_parse_args_warmup_variants():
    with patch.object(sys, "argv", ["server_windows.py", "--warmup", "all"]):
        assert parse_args().warmup == "all"
    with patch.object(sys, "argv", ["server_windows.py", "--no-warmup"]):
        assert parse_args().warmup == "none"


def test_select_warmup_engines_policy():
    """auto：Qwen3-ASR 高频主力必预热，SenseVoice 始终懒加载。"""
    # auto + 双引擎就绪：只预热 Qwen + Hy-MT2，SenseVoice 懒加载
    assert _select_warmup_engines("auto", sv_ready=True, qw_ready=True, mt_ready=True) == (False, True, True)
    # auto + 仅 SenseVoice 就绪：也不预热（等首次请求）
    assert _select_warmup_engines("auto", sv_ready=True, qw_ready=False, mt_ready=True) == (False, False, True)
    # auto + 双双缺失
    assert _select_warmup_engines("auto", sv_ready=False, qw_ready=False, mt_ready=False) == (False, False, False)
    # all：全量预热
    assert _select_warmup_engines("all", sv_ready=True, qw_ready=True, mt_ready=True) == (True, True, True)
    # none：不预热
    assert _select_warmup_engines("none", sv_ready=True, qw_ready=True, mt_ready=True) == (False, False, False)


def test_configure_runtime_clears_linux_abi_profile():
    """T1 回归：无论是否残留非空的 Linux 特化 ABI 配置，Windows 下都应清除。"""
    from server_windows import configure_runtime

    os.environ["YAKUTAN_LLAMA_ABI_PROFILE"] = "cuda-2026-08"
    try:
        with tempfile.TemporaryDirectory() as td:
            configure_runtime(ROOT, Path(td))
        assert "YAKUTAN_LLAMA_ABI_PROFILE" not in os.environ
    finally:
        os.environ.pop("YAKUTAN_LLAMA_ABI_PROFILE", None)


def test_normalize_hypothesis_strips_prefixes_and_wait_markers():
    assert _normalize_hypothesis("Translation: hello") == "hello"
    assert _normalize_hypothesis("翻译：你好") == "你好"
    assert _normalize_hypothesis("<WAIT>") == ""
    assert _normalize_hypothesis("  plain  ") == "plain"


def test_source_text_prefers_explicit_source_and_joins_words():
    assert _source_text({"source": " hello "}) == "hello"
    assert _source_text({"words": [["你", 0.1], ["好", 0.2]]}) == "你好"
    assert _source_text({"tail": {"words": [["a", 0.1]]}, "words": [["b", 0.1]]}) == "ba"


def test_translation_prompt_without_history_is_simple():
    prompt = _translation_prompt({}, "hello world", 10)
    assert "hello world" in prompt
    assert "English" in prompt and "Chinese" in prompt
    assert "[Background Information]" not in prompt


if __name__ == "__main__":
    failed = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {exc!r}")
    sys.exit(1 if failed else 0)
