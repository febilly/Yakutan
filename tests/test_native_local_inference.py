"""Native mtmd migration: ABI parameters, shared resources and old-file boundaries."""
import hashlib
import io
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_inference import model_manager as mm
from local_inference.mtmd_audio import MtmdAudioEncoder, expected_window_tokens
from local_inference.qwen3_asr_mtmd import MtmdQwenASR
from local_inference.qwen_microbatch import QwenMicroBatchEngine
from local_inference.spec_decode import DraftTracker


def test_context_passes_output_limit_and_kv_type_to_native(monkeypatch):
    from local_inference.vendor.qwen_asr_gguf.inference import llama
    params = llama.llama_context_params()
    monkeypatch.setattr(llama, "llama_context_default_params", lambda: params)
    monkeypatch.setattr(llama, "llama_init_from_model", lambda model, p: 123)
    monkeypatch.setattr(llama, "llama_free", lambda ptr: None)
    ctx = llama.LlamaContext(SimpleNamespace(ptr=1), n_outputs_max=64, type_kv=8)
    assert params.n_outputs_max == 64
    assert params.type_k == params.type_v == 8
    del ctx


def test_worker_shares_weights_but_owns_context_and_shutdown(monkeypatch):
    from local_inference.qwen3_asr_mtmd import llama
    contexts = []
    def context(model, **kwargs):
        contexts.append(kwargs)
        return object()
    monkeypatch.setattr(llama, "LlamaContext", context)
    base = object.__new__(MtmdQwenASR)
    base.model, base.encoder, base.embedding_table, base.ctx = object(), object(), object(), object()
    base._n_ctx, base._kv_type = 2048, 8
    worker = base.create_context_worker()
    assert worker.model is base.model and worker.encoder is base.encoder
    assert worker.embedding_table is base.embedding_table
    assert worker.ctx is not base.ctx
    assert contexts == [dict(n_ctx=2048, n_batch=4096, embeddings=False, type_kv=8, n_outputs_max=64)]
    worker.shutdown()
    worker.shutdown()
    assert base.encoder is not None and base.model is not None and base.ctx is not None


def test_encoder_batch_keeps_order_and_copies_one_audio_at_a_time(monkeypatch):
    encoder = object.__new__(MtmdAudioEncoder)
    encoder._lock = threading.RLock()
    encoder._ctx = 123
    monkeypatch.setattr(encoder, "_encode", lambda audio: (audio.reshape(-1, 1).copy(), 0.1))
    audios = [np.array([1, 2], dtype=np.float32), np.array([3], dtype=np.float32)]
    outputs, seconds = encoder.encode_batch(audios)
    assert [x.ravel().tolist() for x in outputs] == [[1, 2], [3]]
    assert seconds >= 0
    encoder._ctx = None
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode(audios[0])


@pytest.mark.parametrize("samples,expected", [(16000, [14]), (128000, [104, 1]), (144000, [104, 14])])
def test_audio_window_padding_is_trimmed(samples, expected):
    assert expected_window_tokens(samples) == expected


def test_multi_session_drafts_are_split_at_64_output_rows():
    engine = object.__new__(QwenMicroBatchEngine)
    engine.core = SimpleNamespace(model=SimpleNamespace(eos_token=-1), ID_IM_END=-2)
    states = [SimpleNamespace(done=False, current=10 + i, emitted=[],
                              tracker=DraftTracker([10 + i] + list(range(100, 163))))
              for i in range(3)]
    groups = []
    engine._decode_active = lambda active: groups.append(active)
    engine._decode_round(states)
    assert [sum(len(tokens) for _, tokens in group) for group in groups] == [64, 64, 64]
    assert [state for group in groups for state, _ in group] == states
    assert all(len(state.emitted) == 1 for state in states)


def test_hf_hash_failure_falls_back_with_pinned_revision(tmp_path, monkeypatch):
    good = b"GGUFgood"
    monkeypatch.setattr(mm, "_ordered_hf_endpoints", lambda *args, **kwargs: ["https://bad", "https://good"])
    urls = []
    def download(url, dest, *args):
        urls.append(url)
        dest.write_bytes(good if url.startswith("https://good") else b"GGUFcorrupt")
    monkeypatch.setattr(mm, "_stream_download", download)
    dest = tmp_path / "weight.gguf"
    mm._download_hf_file("repo/model", "weight.gguf", dest, "test", revision="fixed",
                         sha256=hashlib.sha256(good).hexdigest())
    assert dest.read_bytes() == good
    assert urls == [f"https://{host}/repo/model/resolve/fixed/weight.gguf" for host in ("bad", "good")]


def test_runtime_hf_archive_includes_mtmd_and_never_uses_old_dlls(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(mm, "_ordered_hf_endpoints", lambda *args, **kwargs: ["https://hf-mirror.com"])
    names = (*mm._runtime_names(), "libomp.dll")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        for name in names:
            z.writestr(f"runtime/{name}", b"MZtest")
        z.writestr("../outside.txt", b"not a runtime")
    payload = archive.getvalue()
    monkeypatch.setattr(mm, "LLAMA_RUNTIME_ZIP_SHA256", hashlib.sha256(payload).hexdigest())
    urls = []
    def download(url, dest, *args):
        urls.append(url)
        dest.write_bytes(payload)
    monkeypatch.setattr(mm, "_stream_download", download)
    old = tmp_path / "qwen_llama_vulkan_bin"
    old.mkdir()
    (old / "llama.dll").write_bytes(b"old")
    assert mm._qwen3_llama_bin_dir() != old
    mm._ensure_llama_cpp_vulkan_to(mm._qwen_llama_vulkan_user_bin())
    assert mm._runtime_ready(mm._qwen3_llama_bin_dir())
    assert (old / "llama.dll").read_bytes() == b"old"
    assert not (tmp_path / "outside.txt").exists()
    assert mm.LOCAL_RUNTIME_REVISION in urls[0]


def test_cleanup_only_known_obsolete_files_and_reports_locked_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    obsolete = tmp_path / mm.LEGACY_QWEN3_ASR_DIR_NAME / mm.LEGACY_QWEN3_ASR_FILES[0]
    old_runtime = tmp_path / mm.LEGACY_RUNTIME_DIR_NAME / "llama.dll"
    keep = [tmp_path / mm.QWEN3_ASR_DIR_NAME / "new.gguf", tmp_path / "hymt2/translation.gguf",
            tmp_path / "sensevoice-onnx/encoder.onnx", obsolete.parent / "user.txt",
            old_runtime.parent / "llama-custom.dll"]
    for path in [obsolete, old_runtime, *keep]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"kept")
    unlink = Path.unlink
    def locked(path, *args, **kwargs):
        if path == old_runtime:
            raise PermissionError("in use")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", locked)
    assert mm.remove_legacy_files() == {"removed_bytes": 4, "failed": ["llama.dll"]}
    assert all(path.is_file() for path in keep)
    assert old_runtime.is_file() and not obsolete.exists()


def test_cleanup_api_requires_new_model_and_explicit_post(tmp_path):
    # Keep Flask imports and its configuration out of the main pytest process.
    script = '''
import os, sys
from pathlib import Path
os.environ['YAKUTAN_LOCAL_INFERENCE_UI'] = '1'
from local_inference import model_manager as mm
import ui.app as app
mm.MODELS_DIR = Path(sys.argv[1])
old = mm.MODELS_DIR / mm.LEGACY_QWEN3_ASR_DIR_NAME / mm.LEGACY_QWEN3_ASR_FILES[0]
old.parent.mkdir(parents=True)
old.write_bytes(b'old')
client = app.app.test_client()
assert client.get('/api/local-inference/legacy-cleanup').status_code == 405
assert client.post('/api/local-inference/legacy-cleanup').status_code == 409
assert old.exists()
for path in mm.qwen3_required_paths(mm.MODELS_DIR):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'new')
app.local_inference_download_state['running'] = True
assert client.post('/api/local-inference/legacy-cleanup').status_code == 409
assert old.exists()
app.local_inference_download_state['running'] = False
assert client.post('/api/local-inference/legacy-cleanup').json['removed_bytes'] == 3
assert not old.exists()
assert all(path.exists() for path in mm.qwen3_required_paths(mm.MODELS_DIR))
print('cleanup API passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cleanup_includes_legacy_cpu_backends_and_interrupted_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    paths = [tmp_path / mm.LEGACY_RUNTIME_DIR_NAME / "ggml-cpu-alderlake.dll",
             tmp_path / mm.LEGACY_RUNTIME_DIR_NAME / "ggml-rpc.dll",
             tmp_path / "llama-cpp-vulkan.zip.part"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old")
    assert set(mm._legacy_files()) == set(paths)


def test_linked_legacy_model_directory_is_not_cleaned(tmp_path, monkeypatch):
    root, outside = tmp_path / "models", tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    old = outside / mm.LEGACY_QWEN3_ASR_FILES[0]
    old.write_bytes(b"keep")
    linked = root / mm.LEGACY_QWEN3_ASR_DIR_NAME
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is unavailable on this Windows host")
    monkeypatch.setattr(mm, "MODELS_DIR", root)
    assert mm.remove_legacy_files() == {"removed_bytes": 0, "failed": []}
    assert old.read_bytes() == b"keep" and linked.is_symlink()
