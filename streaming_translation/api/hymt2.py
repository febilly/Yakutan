"""
Hy-MT2 (HunYuan-MT 1.8B) real-time streaming translation via WebSocket.

Implements the stateless context-revision protocol documented in
`Hy-MT2-1.8B 实时文本流翻译服务接入文档` (INTEGRATION.md):

    Client -> Server : init / update (UTF-8 JSON text frames, one per frame)
    Server -> Client : init_ok / translation / error

Per the protocol the server keeps **no session state**: every ``update``
message is fully self-contained and carries the client-held revision chain
(``previous_source`` / ``previous_translation`` / ``history``).  This class
owns that chain for the sentence currently being translated, exactly like the
protocol prescribes.

The WebSocket address is intentionally **not** hard-coded: it is injected by
the caller (Web UI setting / ``HYMT2_WEBSOCKET_URL`` env var) and must be
filled in by the user.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Dict, List, Optional

from .base import BaseTranslationAPI

try:
    from websockets.sync.client import connect as _sync_ws_connect
except ImportError:  # pragma: no cover
    raise ImportError(
        "websockets not installed. Run: pip install websockets>=12"
    )

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
MAX_HISTORY = 8        # 服务端接受 <=10 条，取 8 与本应用上下文窗口一致
HISTORY_KEEP = 20      # 本地保留的完成句对上限（截断后再取最新一条发送）


class HyMT2API(BaseTranslationAPI):
    """Hy-MT2 实时流式修订翻译（WebSocket 无状态协议）。

    连接级语义：

    - 第一条消息必须是 ``init``（协商连接级默认方向 / prompt 模式）；
    - 之后每条 ``update`` 都完整自携带修订链，服务端无状态；
    - 连接可随时断开重连（本类会在下次请求时自动重连并重新 init）。

    语言方向：``translate()`` 的 ``source_language``/``target_language``
    每次都可不同（方向随请求切换，无需重连）。``detected_source_language``
    kwarg（本应用识别器已解析出的具体语言码）优先于 ``source_language``。
    """

    SUPPORTS_CONTEXT = True

    DEFAULT_SOURCE_LANG = "en"
    DEFAULT_TARGET_LANG = "zh"

    def __init__(
        self,
        websocket_url: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        proxy_url: Optional[str] = None,
    ):
        self.websocket_url = (websocket_url or "").strip()
        self.timeout = float(timeout or 30.0)
        self.max_retries = int(max_retries if max_retries is not None else 3)
        self._proxy_url = proxy_url  # 透传给 websockets（本地服务通常无需代理）

        self._lock = threading.Lock()
        self._ws = None
        self._boot_ms = time.time() * 1000
        self.reset_session()

    # ── Session / chain state ─────────────────────────────────────────

    def reset_session(self) -> None:
        """重置进行中的句子修订链（识别会话开始 / 应用重配置时调用）。

        已完成句子的 ``history`` 保留（仍可作为上下文），仅清空当前句。
        """
        with self._lock:
            self._seq = 0
            self._utterance_id = -1  # 首次调用自增后从 0 开始
            self._prev_source: Optional[str] = None
            self._prev_translation: Optional[str] = None
            self._last_final = True
            self._sent_any = False
            self._last_source_lang: Optional[str] = None
            self._last_target_lang: Optional[str] = None
            self._utt_start_ms: Optional[float] = None
            self._history: List[List[str]] = []

    # ── Language helpers ───────────────────────────────────────────────

    @staticmethod
    def _normalize_lang(code: Optional[str]) -> str:
        """小写并去掉地区后缀（zh-CN -> zh），'auto'/空 -> ''。"""
        if not code:
            return ""
        norm = str(code).strip().lower()
        if not norm or norm == "auto":
            return ""
        return norm.split("-")[0]

    def _resolve_source_lang(self, source_language: str, detected: Optional[str]) -> str:
        code = self._normalize_lang(detected) or self._normalize_lang(source_language)
        if not code:
            # 兜底：沿用本连接最近一次已知的具体语言码
            code = self._last_source_lang or self.DEFAULT_SOURCE_LANG
        return code

    # ── Connection management ──────────────────────────────────────────

    def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _recv_json(self, ws) -> Dict:
        message = ws.recv(timeout=self.timeout)
        if isinstance(message, (bytes, bytearray)):
            message = bytes(message).decode("utf-8")
        data = json.loads(message)
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected non-object message: {data!r}")
        return data

    def _open_connection(self, source_lang: str, target_lang: str) -> None:
        """Open a WebSocket connection and complete the ``init`` handshake."""
        last_error: Optional[Exception] = None
        connect_kwargs: Dict = {"open_timeout": self.timeout}
        if self._proxy_url:
            connect_kwargs["proxy"] = self._proxy_url

        for attempt in range(self.max_retries + 1):
            ws = None
            try:
                ws = _sync_ws_connect(self.websocket_url, **connect_kwargs)
                init_payload = {
                    "type": "init",
                    "protocol_version": PROTOCOL_VERSION,
                    "source_lang": source_lang or self.DEFAULT_SOURCE_LANG,
                    "target_lang": target_lang or self.DEFAULT_TARGET_LANG,
                    "preset": "standard",
                    "hypothesis_mode": "sentence_revision",
                    "source_token_join_mode": "verbatim",
                }
                ws.send(json.dumps(init_payload, ensure_ascii=False))
                reply = self._recv_json(ws)
                if reply.get("type") != "init_ok":
                    raise RuntimeError(f"init failed: {reply}")
                if reply.get("hypothesis_mode") != "sentence_revision" or (
                    reply.get("source_token_join_mode") != "verbatim"
                ):
                    raise RuntimeError(
                        f"incompatible Hy-MT2 server (protocol check failed): {reply}"
                    )
                self._ws = ws
                if attempt:
                    logger.info(
                        "Hy-MT2 connection established after %d retries", attempt
                    )
                return
            except Exception as exc:  # noqa: BLE001 - surface as [ERROR]
                last_error = exc
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8.0))

        raise RuntimeError(
            f"cannot connect to Hy-MT2 service at {self.websocket_url!r}: "
            f"{last_error}"
        )

    def _request(self, payload: Dict) -> Dict:
        with self._lock:
            ws = self._ws
            if ws is None:
                self._open_connection(
                    payload.get("source_lang") or "",
                    payload.get("target_lang") or "",
                )
                ws = self._ws
            try:
                ws.send(json.dumps(payload, ensure_ascii=False))
                reply = self._recv_json(ws)
            except Exception:
                # 传输层失败：断开，让下一次请求重连（无状态，重连不丢链）
                self._close_ws()
                raise
            if reply.get("type") == "error":
                raise RuntimeError(f"Hy-MT2 server error: {reply}")
            if reply.get("type") != "translation":
                raise RuntimeError(f"unexpected Hy-MT2 message: {reply}")
            return reply

    # ── History helpers ────────────────────────────────────────────────

    @staticmethod
    def _strip_speaker_prefix(source: str) -> str:
        for prefix in ("[Me] ", "[Others] "):
            if source.startswith(prefix):
                return source[len(prefix):]
        return source

    @classmethod
    def _history_from_context_pairs(
        cls,
        context_pairs: Optional[List[Dict[str, str]]],
    ) -> List[List[str]]:
        """把 ContextAwareTranslator 的完成句对转成协议 history 格式。"""
        if not context_pairs:
            return []
        pairs: List[List[str]] = []
        for pair in context_pairs:
            source = cls._strip_speaker_prefix(str(pair.get("source") or "")).strip()
            target = str(pair.get("target") or "").strip()
            if source:
                pairs.append([source, target])
        return pairs[-MAX_HISTORY:]

    # ── Translation entry point ────────────────────────────────────────

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        context_pairs: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        if not text or not text.strip():
            return ""
        if not self.websocket_url:
            return (
                "[ERROR] Hy-MT2: WebSocket 地址未配置，请在网页「翻译API设置」"
                "中填写 Hy-MT2 WebSocket 地址（或设置 HYMT2_WEBSOCKET_URL）。"
            )

        is_partial = bool(kwargs.get("is_partial", False))
        detected_source = kwargs.get("detected_source_language")
        source_lang = self._resolve_source_lang(source_language, detected_source)
        target_lang = self._normalize_lang(target_language) or (
            self._last_target_lang or self.DEFAULT_TARGET_LANG
        )
        history = self._history_from_context_pairs(context_pairs)

        try:
            payload, previous_translation = self._build_update(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                is_partial=is_partial,
                history=history,
            )
            reply = self._request(payload)

            committed = str(reply.get("committed_text") or "")
            buffer_text = str(reply.get("buffer_text") or "")
            display = (committed + buffer_text).strip()
            if not display:
                # stop_reason="wait"：无新内容，保持上一版显示（不返回空串）
                display = previous_translation or ""

            self._record_response(
                text=text,
                display=display,
                is_partial=is_partial,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            return display

        except Exception as e:
            logger.warning("Hy-MT2 translate failed: %s", e)
            return f"[ERROR] Hy-MT2: {e}"

    def _build_update(
        self,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool,
        history: List[List[str]],
    ) -> tuple[Dict, Optional[str]]:
        with self._lock:
            direction_changed = (
                (
                    self._last_source_lang is not None
                    and source_lang != self._last_source_lang
                )
                or (
                    self._last_target_lang is not None
                    and target_lang != self._last_target_lang
                )
            )
            new_utterance = (not self._sent_any) or self._last_final or direction_changed
            if new_utterance:
                self._utterance_id += 1
                self._prev_source = None
                self._prev_translation = None
                self._utt_start_ms = time.time() * 1000

            self._seq += 1
            self._sent_any = True
            previous_source = self._prev_source
            previous_translation = self._prev_translation
            local_history = self._history[-MAX_HISTORY:]

            self._last_source_lang = source_lang
            self._last_target_lang = target_lang
            self._last_final = not is_partial

            payload: Dict = {
                "type": "update",
                "seq": self._seq,
                "utterance_id": self._utterance_id,
                "words": [[text, None, None]],
                "clock_ms": int(
                    time.time() * 1000 - (self._utt_start_ms or self._boot_ms)
                ),
                "is_final": not is_partial,
                "source": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
            if previous_source is not None:
                payload["previous_source"] = previous_source
            if previous_translation is not None:
                payload["previous_translation"] = previous_translation
            history_send = history or local_history
            if history_send:
                payload["history"] = history_send
        return payload, previous_translation

    def _record_response(
        self,
        *,
        text: str,
        display: str,
        is_partial: bool,
        source_lang: str,
        target_lang: str,
    ) -> None:
        with self._lock:
            if not is_partial:
                # 句子已完成：记入历史，清空修订链（下一条 update 自动开新句）
                if text.strip() and display.strip():
                    self._history.append([text, display])
                    if len(self._history) > HISTORY_KEEP:
                        self._history = self._history[-HISTORY_KEEP:]
                self._prev_source = None
                self._prev_translation = None
            else:
                self._prev_source = text
                self._prev_translation = display