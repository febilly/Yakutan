"""本地模型下载：端点回退、断点续传、进度上报与落盘校验。

真实下载是几百 MB 到 1GB 起步，所以这里用一个支持 Range 的本地 HTTP 服务代替 HuggingFace。
会出问题的恰恰是这几条边界：镜像回退的顺序、被中断后 .part 的续传、面板进度是否真的在动、
以及“把网关返回的错误页面当成模型存下来”这类静默损坏。
"""

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_inference import model_manager as mm  # noqa: E402

PAYLOAD = b"GGUF" + b"\x5a" * 60_000
BAD_PAYLOAD = b"<html>blocked by gateway</html>"


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """只实现测试需要的部分：200 / 206 / 416，以及一个非 GGUF 的错误页面。"""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler 的命名约定)
        self.server.requested_paths.append(self.path)
        self.server.requested_ranges.append(self.headers.get("Range"))
        if self.path.endswith("bad.gguf"):
            self._send(200, BAD_PAYLOAD)
            return
        if self.path.endswith("portal.onnx"):
            # 模拟登录门户/网关：200 + HTML，文件名却是权重
            self._send(200, BAD_PAYLOAD, {"Content-Type": "text/html; charset=utf-8"})
            return

        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[len("bytes="):].split("-")[0])
        if not rng:
            self._send(200, PAYLOAD)
        elif start >= len(PAYLOAD):
            self._send(416, b"", {"Content-Range": f"bytes */{len(PAYLOAD)}"})
        else:
            self._send(
                206,
                PAYLOAD[start:],
                {"Content-Range": f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"},
            )

    def _send(self, code, body, headers=None):
        self.send_response(code)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


@pytest.fixture
def server():
    httpd = socketserver.TCPServer(("127.0.0.1", 0), _RangeHandler)
    httpd.requested_ranges = []
    httpd.requested_paths = []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    httpd.base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield httpd
    httpd.shutdown()
    httpd.server_close()


class TestStreamDownload:
    def test_fresh_download_reports_progress_and_clears_part(self, server, tmp_path):
        dest = tmp_path / "model.gguf"
        seen = []
        mm._stream_download(
            f"{server.base_url}/model.gguf", dest, "阶段名",
            lambda stage, done, total: seen.append((stage, done, total)),
        )
        assert dest.read_bytes() == PAYLOAD
        assert not dest.with_name("model.gguf.part").exists()
        # 开头先报一次（面板立刻有东西显示），结束时再补一次
        assert seen[0] == ("阶段名", 0, len(PAYLOAD))
        assert seen[-1] == ("阶段名", len(PAYLOAD), len(PAYLOAD))

    def test_resume_progress_counts_bytes_already_on_disk(self, server, tmp_path):
        """续传时进度要从已有字节数起算，否则进度条会从 0 重新爬一遍。"""
        dest = tmp_path / "model.gguf"
        half = len(PAYLOAD) // 2
        dest.with_name("model.gguf.part").write_bytes(PAYLOAD[:half])
        seen = []

        mm._stream_download(
            f"{server.base_url}/model.gguf", dest, "阶段名",
            lambda stage, done, total: seen.append((done, total)),
        )

        assert seen[0] == (half, len(PAYLOAD))
        assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))

    def test_progress_callback_failure_does_not_break_download(self, server, tmp_path):
        dest = tmp_path / "model.gguf"

        def _broken(stage, done, total):
            raise RuntimeError("面板挂了")

        mm._stream_download(f"{server.base_url}/model.gguf", dest, "test", _broken)

        assert dest.read_bytes() == PAYLOAD

    def test_resumes_from_partial_file(self, server, tmp_path):
        dest = tmp_path / "model.gguf"
        half = len(PAYLOAD) // 2
        dest.with_name("model.gguf.part").write_bytes(PAYLOAD[:half])

        mm._stream_download(f"{server.base_url}/model.gguf", dest, "test")

        assert dest.read_bytes() == PAYLOAD
        assert server.requested_ranges == [f"bytes={half}-"]

    def test_complete_part_is_promoted_without_refetch(self, server, tmp_path):
        dest = tmp_path / "model.gguf"
        dest.with_name("model.gguf.part").write_bytes(PAYLOAD)

        mm._stream_download(f"{server.base_url}/model.gguf", dest, "test")

        assert dest.read_bytes() == PAYLOAD

    def test_oversized_part_is_discarded(self, server, tmp_path):
        """416 分支不能无脑改名：.part 比远端长说明它已经坏了。"""
        dest = tmp_path / "model.gguf"
        part = dest.with_name("model.gguf.part")
        part.write_bytes(PAYLOAD + b"junk")

        with pytest.raises(OSError):
            mm._stream_download(f"{server.base_url}/model.gguf", dest, "test")

        assert not part.exists()
        assert not dest.exists()

    def test_non_gguf_body_rejected(self, server, tmp_path):
        dest = tmp_path / "model.gguf"

        with pytest.raises(OSError):
            mm._stream_download(f"{server.base_url}/bad.gguf", dest, "test")

        assert not dest.exists()
        assert not dest.with_name("model.gguf.part").exists()

    def test_html_response_rejected_before_writing(self, server, tmp_path):
        """没有文件头可查的格式（.onnx 等）靠 Content-Type 挡住网关拦截页。"""
        dest = tmp_path / "portal.onnx"

        with pytest.raises(OSError, match="HTML"):
            mm._stream_download(f"{server.base_url}/portal.onnx", dest, "test")

        assert not dest.exists()
        assert not dest.with_name("portal.onnx.part").exists()


class TestEndpointSelection:
    @pytest.fixture(autouse=True)
    def _clean_env(self):
        saved = {k: os.environ.get(k) for k in ("HF_ENDPOINT", "YAKUTAN_HF_ENDPOINT")}
        for key in saved:
            os.environ.pop(key, None)
        yield
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_order_is_official_then_mirror(self):
        assert mm._hf_endpoints() == [mm.HF_ENDPOINT_OFFICIAL, *mm.HF_ENDPOINT_MIRRORS]

    def test_env_override_takes_precedence_without_dropping_fallbacks(self):
        os.environ["HF_ENDPOINT"] = "https://example.invalid/"
        endpoints = mm._hf_endpoints()
        assert endpoints[0] == "https://example.invalid"
        assert mm.HF_ENDPOINT_OFFICIAL in endpoints and mm.HF_ENDPOINT_MIRRORS[0] in endpoints

    def test_unreachable_official_falls_back_to_mirror(self, monkeypatch):
        reachable = {mm._hf_file_url(mm.HF_ENDPOINT_MIRRORS[0], "r", "f")}
        monkeypatch.setattr(mm, "_hf_endpoint_reachable", lambda url, **kw: url in reachable)

        assert mm._ordered_hf_endpoints("r", "f")[0] == mm.HF_ENDPOINT_MIRRORS[0]

    def test_all_unreachable_still_returns_candidates(self, monkeypatch):
        monkeypatch.setattr(mm, "_hf_endpoint_reachable", lambda url, **kw: False)

        # 探测本身也可能误判（代理、临时抖动），所以候选一个都不能丢
        assert set(mm._ordered_hf_endpoints("r", "f")) == set(mm._hf_endpoints())

    def test_url_layout_matches_hf_resolve_scheme(self):
        assert mm._hf_file_url("https://x.dev", "org/repo", "a.gguf") == (
            "https://x.dev/org/repo/resolve/main/a.gguf"
        )


class TestDownloadHymt2:
    @pytest.fixture(autouse=True)
    def _stub_runtime(self, monkeypatch):
        """llama.cpp 运行时是几十 MB 的 GitHub 下载，这里只记录它被调用过。"""
        calls = []
        monkeypatch.setattr(
            mm, "_ensure_llama_cpp_vulkan_to",
            lambda bin_dir, progress_callback=None: calls.append(bin_dir),
        )
        return calls

    def test_falls_back_to_next_endpoint_after_failure(self, server, tmp_path, monkeypatch):
        monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(mm, "apply_cache_env", lambda: None)
        monkeypatch.setattr(
            mm, "_ordered_hf_endpoints",
            lambda repo, name: ["http://127.0.0.1:1", server.base_url],
        )
        monkeypatch.setattr(
            mm, "_hf_file_url",
            lambda endpoint, repo, name, revision="main": f"{endpoint}/model.gguf",
        )

        mm.download_hymt2()

        assert (tmp_path / mm.HYMT2_DIR_NAME / mm.HYMT2_MODEL_FILE).read_bytes() == PAYLOAD

    def test_reports_every_endpoint_when_all_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(mm, "apply_cache_env", lambda: None)
        monkeypatch.setattr(
            mm, "_ordered_hf_endpoints",
            lambda repo, name: ["https://a.invalid", "https://b.invalid"],
        )

        def _fail(url, *args, **kwargs):
            raise OSError(f"unreachable: {url}")

        monkeypatch.setattr(mm, "_stream_download", _fail)

        with pytest.raises(RuntimeError) as excinfo:
            mm.download_hymt2()

        assert "a.invalid" in str(excinfo.value) and "b.invalid" in str(excinfo.value)

    def test_existing_model_is_not_redownloaded(self, tmp_path, monkeypatch, _stub_runtime):
        monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(mm, "apply_cache_env", lambda: None)
        model_dir = tmp_path / mm.HYMT2_DIR_NAME
        model_dir.mkdir()
        (model_dir / mm.HYMT2_MODEL_FILE).write_bytes(PAYLOAD)

        def _boom(*args, **kwargs):
            raise AssertionError("已缓存的模型不应再次触发下载")

        monkeypatch.setattr(mm, "_ordered_hf_endpoints", _boom)
        mm.download_hymt2()

        # 权重齐了但运行时可能还缺（只装过翻译模型的机器），所以这一步不能被跳过
        assert _stub_runtime == [mm._qwen_llama_vulkan_user_bin()]

    def test_llama_runtime_fetched_alongside_weights(self, server, tmp_path, monkeypatch,
                                                     _stub_runtime):
        monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(mm, "apply_cache_env", lambda: None)
        monkeypatch.setattr(mm, "_ordered_hf_endpoints", lambda repo, name: [server.base_url])
        monkeypatch.setattr(
            mm, "_hf_file_url",
            lambda endpoint, repo, name, revision="main": f"{endpoint}/model.gguf",
        )

        mm.download_hymt2()

        assert _stub_runtime == [mm._qwen_llama_vulkan_user_bin()]


class TestOtherEnginesReportProgress:
    """面板的进度条对每个模型都要能动，不能只有 Hy-MT2 有。"""

    def test_silero_reports_progress(self, server, tmp_path, monkeypatch):
        dest = tmp_path / "silero_vad.onnx"
        monkeypatch.setattr(mm, "apply_cache_env", lambda: None)
        monkeypatch.setattr(mm, "is_silero_cached", lambda: False)
        monkeypatch.setattr(mm, "_silero_onnx_user_path", lambda: dest)
        monkeypatch.setattr(mm, "SILERO_VAD_ONNX_URL", f"{server.base_url}/silero.onnx")
        seen = []

        mm.download_silero(lambda stage, done, total: seen.append((stage, done, total)))

        assert dest.read_bytes() == PAYLOAD
        assert seen[-1] == ("Silero VAD", len(PAYLOAD), len(PAYLOAD))

    def test_sensevoice_downloads_every_file_with_a_counter(self, server, tmp_path, monkeypatch):
        monkeypatch.setattr(mm, "_ordered_hf_endpoints", lambda repo, name: [server.base_url])
        monkeypatch.setattr(
            mm, "_hf_file_url",
            lambda endpoint, repo, name, revision="main": f"{endpoint}/{name}",
        )
        seen = []

        mm._download_sensevoice_onnx(tmp_path, lambda stage, done, total: seen.append(stage))

        assert mm._sensevoice_onnx_ready(tmp_path)
        # 每个文件一个阶段名，并带上「第几个/共几个」，否则四连下载看起来像卡住了
        stages = {stage for stage in seen}
        assert len(stages) == len(mm.SENSEVOICE_ONNX_FILES)
        assert all(f"/{len(mm.SENSEVOICE_ONNX_FILES)}" in stage for stage in stages)

    def test_sensevoice_skips_files_already_on_disk(self, server, tmp_path, monkeypatch):
        monkeypatch.setattr(mm, "_ordered_hf_endpoints", lambda repo, name: [server.base_url])
        monkeypatch.setattr(
            mm, "_hf_file_url",
            lambda endpoint, repo, name, revision="main": f"{endpoint}/{name}",
        )
        keep = b"already here"
        (tmp_path / mm.SENSEVOICE_ENCODER_ONNX).write_bytes(keep)

        mm._download_sensevoice_onnx(tmp_path)

        assert (tmp_path / mm.SENSEVOICE_ENCODER_ONNX).read_bytes() == keep
        assert len(server.requested_paths) == len(mm.SENSEVOICE_ONNX_FILES) - 1

    def test_qwen3_extraction_reports_bytes(self, tmp_path, monkeypatch):
        """解压 770MB 要几十秒；这一段没有进度的话面板会长时间停在 100%。"""
        import zipfile

        monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(mm, "apply_cache_env", lambda: None)
        monkeypatch.setattr(mm, "ensure_vendor_sources", lambda engine: None)
        monkeypatch.setattr(mm, "_ensure_llama_cpp_vulkan_to", lambda *a, **kw: None)

        contents = {name: bytes([index]) * 1000 for index, name in enumerate(mm.QWEN3_ASR_FILES)}

        def _fake_download(url, dest, desc="", progress_callback=None):
            with zipfile.ZipFile(str(dest), "w") as archive:
                for name, blob in contents.items():
                    archive.writestr(f"Qwen3-ASR-1.7B/{name}", blob)
            mm._report(progress_callback, desc, 1, 1)

        monkeypatch.setattr(mm, "_download_file", _fake_download)
        seen = []

        mm.download_qwen3_asr(lambda stage, done, total: seen.append((stage, done, total)))

        model_dir = tmp_path / mm.QWEN3_ASR_DIR_NAME
        for name, blob in contents.items():
            assert (model_dir / name).read_bytes() == blob
        extract = [entry for entry in seen if entry[0].startswith("解压")]
        assert extract[-1] == ("解压 Qwen3-ASR 模型", 3000, 3000)
        assert not (tmp_path / "qwen3-asr-1.7b-gguf.zip").exists()


# ── 面板进度：worker -> 状态快照 -> HTTP 接口 ────────────────────────────────

# ui.app 会读配置、起后台线程，放进 pytest 进程里会污染其他用例，所以在子进程里跑。
_PANEL_PROGRESS_SCRIPT = """
import json, sys
sys.path.insert(0, %(root)r)
from ui import app as ui_app

steps = [
    ("Silero VAD", 0, 1300000),
    ("Qwen3-ASR 模型", 100000000, 770000000),
    ("解压 Qwen3-ASR 模型", 0, None),
]
seen = []

def fake_download(engine, progress_callback=None):
    for stage, done, total in steps:
        progress_callback(stage, done, total)
        seen.append(ui_app._snapshot_local_inference_download_state())

ui_app.download_local_inference_model = fake_download
ui_app.is_silero_cached = lambda: True
ui_app._download_local_inference_worker("qwen3-asr")

client = ui_app.app.test_client()
done_state = client.get("/api/local-inference/download-progress").get_json()

def boom(engine, progress_callback=None):
    raise RuntimeError("镜像也连不上")

ui_app.download_local_inference_model = boom
ui_app._download_local_inference_worker("qwen3-asr")
failed_state = client.get("/api/local-inference/download-progress").get_json()

print(json.dumps({"seen": seen, "done": done_state, "failed": failed_state}, ensure_ascii=False))
"""


def test_panel_progress_reaches_the_http_endpoint():
    root = str(Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", _PANEL_PROGRESS_SCRIPT % {"root": root}],
        cwd=root, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    percents = [snap["percent"] for snap in payload["seen"]]
    assert percents == [0, 12, None]  # 拿不到总长度的阶段报 None，前端据此画不确定态
    assert payload["seen"][1]["stage"] == "Qwen3-ASR 模型"
    assert "95/734 MB" in payload["seen"][1]["status"]

    assert payload["done"]["running"] is False
    assert payload["done"]["percent"] == 100
    assert payload["done"]["error"] is None

    # 失败时进度条要收起来，而不是停在最后一次的百分比上
    assert payload["failed"]["running"] is False
    assert payload["failed"]["percent"] is None
    assert "镜像也连不上" in payload["failed"]["error"]


# ui.app 同上：放子进程里跑，避免 Flask/配置污染 pytest 进程。
_DOWNLOAD_ENDPOINT_SCRIPT = """
import json, sys, threading, time
sys.path.insert(0, %(root)r)
from ui import app as ui_app

release, started = threading.Event(), threading.Event()

def slow_download(engine, progress_callback=None):
    started.set()
    release.wait(10)

ui_app.download_local_inference_model = slow_download
ui_app.is_silero_cached = lambda: True
client = ui_app.app.test_client()

first = client.post("/api/local-inference/download", json={"engine": "qwen3-asr"})
# 接口返回的那一刻状态就得是 running，不能等 worker 线程被调度到
right_after = client.get("/api/local-inference/download-progress").get_json()
second = client.post("/api/local-inference/download", json={"engine": "qwen3-asr"})

started.wait(10)
release.set()
for _ in range(200):
    final = client.get("/api/local-inference/download-progress").get_json()
    if not final["running"]:
        break
    time.sleep(0.05)

print(json.dumps({
    "first_code": first.status_code,
    "right_after": right_after,
    "second_code": second.status_code,
    "final": final,
}, ensure_ascii=False))
"""


def test_download_endpoint_marks_running_before_it_returns():
    root = str(Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", _DOWNLOAD_ENDPOINT_SCRIPT % {"root": root}],
        cwd=root, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["first_code"] == 200
    # 否则面板要等下一次轮询才显示「下载中」，按钮也还是可点的
    assert payload["right_after"]["running"] is True
    assert payload["right_after"]["engine"] == "qwen3-asr"
    assert payload["second_code"] == 409  # 占位是原子的，第二个请求必须被挡掉
    assert payload["final"]["running"] is False
