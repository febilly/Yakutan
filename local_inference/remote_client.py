from __future__ import annotations

import base64
import json
import threading
from typing import Any

import numpy as np

try:
    from websockets.sync.client import connect as _sync_ws_connect
except ImportError:  # pragma: no cover - surfaced by the local-inference runtime check
    _sync_ws_connect = None


PROTOCOL_VERSION = 1


def normalize_server_url(value: str | None) -> str:
    """Normalize the single endpoint shared by remote ASR and Hy-MT2."""
    url = str(value or "").strip().rstrip("/")
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    return url


def _recv_json(ws, timeout: float) -> dict[str, Any]:
    raw = ws.recv(timeout=timeout)
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"远端推理服务返回了非对象消息: {payload!r}")
    return payload


def probe_remote_server(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    normalized = normalize_server_url(url)
    if not normalized:
        return {"ready": False, "error": "远端推理服务器地址为空"}
    if _sync_ws_connect is None:
        return {"ready": False, "error": "缺少 websockets 依赖"}
    try:
        with _sync_ws_connect(normalized, open_timeout=timeout) as ws:
            ws.send(json.dumps({"type": "health", "protocol_version": PROTOCOL_VERSION}))
            reply = _recv_json(ws, timeout)
        if reply.get("type") != "health_ok":
            raise RuntimeError(f"不兼容的健康检查响应: {reply}")
        return {"ready": bool(reply.get("ready")), **reply}
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


class RemoteASREngine:
    """Drop-in ASR engine that sends complete VAD segments to the shared service."""

    def __init__(
        self,
        engine: str,
        server_url: str,
        *,
        timeout: float = 60.0,
        corpus_text: str | None = None,
    ) -> None:
        if engine not in {"sensevoice", "qwen3-asr"}:
            raise ValueError(f"远端服务不支持 ASR 引擎: {engine}")
        self.engine = engine
        self.server_url = normalize_server_url(server_url)
        if not self.server_url:
            raise RuntimeError("远端推理服务器地址未配置")
        self.timeout = max(1.0, float(timeout))
        self.language: str | None = None
        self._corpus_text = str(corpus_text or "").strip()
        self._ws = None
        self._lock = threading.RLock()
        self._request_id = 0
        self._reset_draft_pending = False
        self._context = ""
        self.device = "remote"

    def set_language(self, language: str) -> None:
        self.language = language if language and language != "auto" else None

    def set_corpus_text(self, text: str | None) -> None:
        self._corpus_text = str(text or "").strip()

    def set_context(self, context: str) -> None:
        self._context = str(context or "")

    def reset_draft(self) -> None:
        self._reset_draft_pending = True

    def to_device(self, device: str) -> bool:
        _ = device
        return False

    def _close_locked(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def unload(self) -> None:
        with self._lock:
            self._close_locked()

    def _connect_locked(self):
        if self._ws is not None:
            return self._ws
        if _sync_ws_connect is None:
            raise RuntimeError("缺少 websockets 依赖，无法连接远端推理服务")
        ws = _sync_ws_connect(self.server_url, open_timeout=self.timeout)
        ws.send(json.dumps({
            "type": "init",
            "service": "asr",
            "protocol_version": PROTOCOL_VERSION,
            "engine": self.engine,
            "sample_rate": 16000,
        }))
        reply = _recv_json(ws, self.timeout)
        if reply.get("type") != "init_ok" or reply.get("service") != "asr":
            ws.close()
            raise RuntimeError(f"远端 ASR 握手失败: {reply}")
        if reply.get("engine") != self.engine:
            ws.close()
            raise RuntimeError(f"远端 ASR 引擎不匹配: {reply}")
        self._ws = ws
        return ws

    def transcribe(self, audio: np.ndarray, *, update_context: bool = True) -> dict | None:
        waveform = np.ascontiguousarray(np.asarray(audio, dtype="<f4").reshape(-1))
        if waveform.size == 0:
            return None
        with self._lock:
            self._request_id += 1
            request = {
                "type": "transcribe",
                "request_id": self._request_id,
                "audio_format": "f32le",
                "sample_rate": 16000,
                "audio_base64": base64.b64encode(waveform.tobytes()).decode("ascii"),
                "language": self.language or "auto",
                "corpus_text": self._corpus_text,
                "context": self._context,
                "update_context": bool(update_context),
                "reset_draft": bool(self._reset_draft_pending),
            }
            self._reset_draft_pending = False
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    ws = self._connect_locked()
                    ws.send(json.dumps(request, ensure_ascii=False))
                    reply = _recv_json(ws, self.timeout)
                    if reply.get("type") == "error":
                        raise RuntimeError(f"远端 ASR 错误: {reply}")
                    if reply.get("type") != "recognition":
                        raise RuntimeError(f"远端 ASR 响应不兼容: {reply}")
                    result = reply.get("result")
                    if isinstance(reply.get("context"), str):
                        self._context = reply["context"]
                    return result if isinstance(result, dict) else None
                except Exception as exc:
                    last_error = exc
                    self._close_locked()
                    if attempt:
                        break
            raise RuntimeError(f"连接远端推理服务失败 {self.server_url!r}: {last_error}")
