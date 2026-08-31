"""本地模型下载：端点回退、断点续传、进度上报与落盘校验。

真实下载是几百 MB 到 1GB 起步，所以这里用一个支持 Range 的本地 HTTP 服务代替 HuggingFace。
会出问题的恰恰是这几条边界：镜像回退的顺序、被中断后 .part 的续传、面板进度是否真的在动、
以及“把网关返回的错误页面当成模型存下来”这类静默损坏。
"""

import http.server
import os
import socketserver
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
