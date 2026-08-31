from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from local_inference.remote_client import (
    RemoteASREngine,
    normalize_server_url,
    probe_remote_server,
)


class FakeWS:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.closed = False

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self, timeout=None):
        return json.dumps(self.replies.pop(0), ensure_ascii=False)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_normalize_server_url_accepts_http_and_trailing_slash():
    assert normalize_server_url(" http://127.0.0.1:18775/ ") == "ws://127.0.0.1:18775"
    assert normalize_server_url("https://example.test/ws/") == "wss://example.test/ws"


def test_health_probe_reports_models():
    ws = FakeWS([{
        "type": "health_ok",
        "ready": True,
        "models": {"qwen3-asr": {"ready": True}},
    }])
    with patch("local_inference.remote_client._sync_ws_connect", return_value=ws):
        result = probe_remote_server("ws://127.0.0.1:18775")
    assert result["ready"] is True
    assert ws.sent == [{"type": "health", "protocol_version": 1}]


def test_remote_asr_handshake_and_transcribe_payload():
    ws = FakeWS([
        {"type": "init_ok", "service": "asr", "engine": "qwen3-asr"},
        {
            "type": "recognition",
            "request_id": 1,
            "result": {"text": "你好", "language": "zh", "language_name": "zh"},
            "context": "你好",
        },
    ])
    with patch("local_inference.remote_client._sync_ws_connect", return_value=ws):
        engine = RemoteASREngine(
            "qwen3-asr", "ws://127.0.0.1:18775", corpus_text="Yakutan"
        )
        engine.set_language("zh")
        engine.reset_draft()
        result = engine.transcribe(np.array([0.0, 0.25, -0.25], dtype=np.float32))

    assert result["text"] == "你好"
    assert ws.sent[0]["service"] == "asr"
    request = ws.sent[1]
    assert request["type"] == "transcribe"
    assert request["audio_format"] == "f32le"
    assert request["language"] == "zh"
    assert request["corpus_text"] == "Yakutan"
    assert request["reset_draft"] is True
    assert request["audio_base64"]
    assert engine._context == "你好"
