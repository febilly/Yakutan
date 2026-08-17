"""Tests for the Hy-MT2 WebSocket streaming translation backend."""

from __future__ import annotations

import json
from unittest.mock import patch

from streaming_translation.api.hymt2 import HyMT2API

INIT_OK = {
    "type": "init_ok",
    "protocol_version": 1,
    "direction": "ja->zh",
    "model": "Hy-MT2-1.8B-context-revision-v3-GGUF-Q4_K_M",
    "hypothesis_mode": "sentence_revision",
    "source_token_join_mode": "verbatim",
    "prompt_mode": "standard",
}

TRANSLATION_OK = {
    "type": "translation",
    "seq": 1,
    "source_lang": "ja",
    "target_lang": "zh",
    "committed_text": "前半句，后半句。",
    "committed_delta": "",
    "buffer_text": "",
    "covered_source_units": 8,
    "stop_reason": "revision",
    "final": False,
}


class FakeWS:
    """A minimal stand-in for ``websockets.sync.client.ClientConnection``."""

    def __init__(self, replies=None, recv_error=None):
        self.sent: list[str] = []
        self.replies = list(replies or [])
        self.recv_error = recv_error  # exception raised once on the 1st recv
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout=None):
        if self.recv_error is not None:
            exc = self.recv_error
            self.recv_error = None
            raise exc
        if not self.replies:
            raise RuntimeError("no more server replies queued")
        reply = self.replies.pop(0)
        return json.dumps(reply, ensure_ascii=False)

    def close(self):
        self.closed = True


def _conn(ws):
    """Return a connect factory that always returns *ws* (ignores args)."""
    return lambda *args, **kwargs: ws


def _decoded(ws: FakeWS, index: int = -1) -> dict:
    return json.loads(ws.sent[index])


def _make_api(url="ws://127.0.0.1:18765", **kwargs):
    return HyMT2API(websocket_url=url, max_retries=0, **kwargs)


# ── Missing / empty address ───────────────────────────────────────────

class TestMissingAddress:
    def test_empty_url_returns_error(self):
        api = HyMT2API(websocket_url="", max_retries=0)
        result = api.translate("こんにちは", source_language="ja", target_language="zh")
        assert result.startswith("[ERROR]")
        assert "WebSocket" in result

    def test_blank_url_returns_error(self):
        api = HyMT2API(websocket_url="   ", max_retries=0)
        result = api.translate("hello", source_language="en", target_language="zh")
        assert result.startswith("[ERROR]")


# ── Init handshake ────────────────────────────────────────────────────

class TestInitHandshake:
    def test_init_sent_before_update(self):
        ws = FakeWS(replies=[INIT_OK, TRANSLATION_OK])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            result = api.translate(
                "こんにちは", source_language="ja", target_language="zh-CN",
            )
        assert result == "前半句，后半句。"
        assert len(ws.sent) == 2
        init = _decoded(ws, 0)
        assert init["type"] == "init"
        assert init["protocol_version"] == 1
        assert init["hypothesis_mode"] == "sentence_revision"
        assert init["source_token_join_mode"] == "verbatim"
        assert init["source_lang"] == "ja"

    def test_incompatible_init_ok_errors(self):
        bad = dict(INIT_OK, hypothesis_mode="append_only")
        ws = FakeWS(replies=[bad])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            result = api.translate("hello", source_language="en", target_language="zh")
        assert result.startswith("[ERROR]")
        assert "incompatible" in result

    def test_init_not_ok_errors(self):
        ws = FakeWS(replies=[{"type": "error", "code": "update_required"}])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            result = api.translate("hello", source_language="en", target_language="zh")
        assert result.startswith("[ERROR]")

    def test_connect_failure_returns_error(self):
        def boom(*args, **kwargs):
            raise ConnectionError("refused")

        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=boom):
            api = _make_api()
            result = api.translate("hello", source_language="en", target_language="zh")
        assert result.startswith("[ERROR]")
        assert "refused" in result


# ── Protocol payload / revision chain ─────────────────────────────────

class TestRevisionChain:
    def test_first_partial_has_no_previous_fields(self):
        ws = FakeWS(replies=[INIT_OK, TRANSLATION_OK])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "前半句，",
                source_language="ja",
                target_language="zh",
                is_partial=True,
                detected_source_language="ja",
            )
        update = _decoded(ws, 1)
        assert update["type"] == "update"
        assert update["seq"] == 1
        assert update["utterance_id"] == 0
        assert update["is_final"] is False
        assert update["source_lang"] == "ja"
        assert update["source"] == "前半句，"
        assert "previous_source" not in update
        assert "previous_translation" not in update

    def test_second_partial_carries_previous_chain_same_utterance(self):
        replies = [
            INIT_OK,
            {**TRANSLATION_OK, "committed_text": "前半句，"},
            {**TRANSLATION_OK, "committed_text": "前半句，后半句。"},
        ]
        ws = FakeWS(replies=replies)
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "前半句，",
                source_language="ja",
                target_language="zh",
                is_partial=True,
                detected_source_language="ja",
            )
            api.translate(
                "前半句，后半句。",
                source_language="ja",
                target_language="zh",
                is_partial=True,
                detected_source_language="ja",
            )
        second = _decoded(ws, 2)
        assert second["utterance_id"] == 0
        assert second["previous_source"] == "前半句，"
        assert second["previous_translation"] == "前半句，"

    def test_final_closes_utterance_and_starts_new_one(self):
        replies = [
            INIT_OK,
            {**TRANSLATION_OK, "committed_text": "前半句，后半句。", "final": True},
            {**TRANSLATION_OK, "committed_text": "新しい文。"},
        ]
        ws = FakeWS(replies=replies)
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            first = api.translate(
                "前半句，后半句。",
                source_language="ja",
                target_language="zh",
                is_partial=False,  # 终译
                detected_source_language="ja",
            )
            assert first == "前半句，后半句。"
            api.translate(
                "新しい文。",
                source_language="ja",
                target_language="zh",
                is_partial=True,
                detected_source_language="ja",
            )
        final_update = _decoded(ws, 1)
        assert final_update["is_final"] is True
        new_update = _decoded(ws, 2)
        assert new_update["utterance_id"] == 1
        assert "previous_source" not in new_update

    def test_history_contains_completed_pairs(self):
        replies = [
            INIT_OK,
            {**TRANSLATION_OK, "committed_text": "完了。", "final": True},
            {**TRANSLATION_OK, "committed_text": "次。", "final": True},
        ]
        ws = FakeWS(replies=replies)
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "完了。", source_language="ja", target_language="zh",
                is_partial=False, detected_source_language="ja",
            )
            api.translate(
                "次。", source_language="ja", target_language="zh",
                is_partial=False, detected_source_language="ja",
            )
        # 第二条 update 上报 history 应包含第一条完成句对
        second = _decoded(ws, 2)
        assert second["history"] == [["完了。", "完了。"]]

    def test_context_pairs_used_as_history(self):
        ws = FakeWS(replies=[INIT_OK, TRANSLATION_OK])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "こんにちは",
                source_language="ja",
                target_language="zh",
                context_pairs=[
                    {"source": "[Me] 昨日の会議", "target": "昨天的会议"},
                    {"source": "[Others] 了解", "target": "明白"},
                ],
            )
        update = _decoded(ws, 1)
        assert update["history"] == [
            ["昨日の会議", "昨天的会议"],
            ["了解", "明白"],
        ]

    def test_wait_keeps_previous_text(self):
        replies = [
            INIT_OK,
            {**TRANSLATION_OK, "committed_text": "前半句，"},
            {
                **TRANSLATION_OK,
                "committed_text": "",
                "buffer_text": "",
                "stop_reason": "wait",
            },
        ]
        ws = FakeWS(replies=replies)
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "前半句，", source_language="ja", target_language="zh",
                is_partial=True, detected_source_language="ja",
            )
            kept = api.translate(
                "前半句，後半。", source_language="ja", target_language="zh",
                is_partial=True, detected_source_language="ja",
            )
        assert kept == "前半句，"


# ── Language resolution ───────────────────────────────────────────────

class TestLanguageResolution:
    def test_detected_source_language_preferred(self):
        ws = FakeWS(replies=[INIT_OK, TRANSLATION_OK])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "hello",
                source_language="auto",
                target_language="zh-cn",
                detected_source_language="en-US",
            )
        init = _decoded(ws, 0)
        update = _decoded(ws, 1)
        assert init["source_lang"] == "en"
        assert init["target_lang"] == "zh"
        assert update["source_lang"] == "en"
        assert update["target_lang"] == "zh"

    def test_auto_source_falls_back_to_previous_known(self):
        replies = [
            INIT_OK,
            {**TRANSLATION_OK, "committed_text": "初回"},
            {**TRANSLATION_OK, "committed_text": "二回目"},
        ]
        ws = FakeWS(replies=replies)
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "初回", source_language="ja", target_language="zh",
                detected_source_language="ja",
            )
            api.translate(
                "二回目", source_language="auto", target_language="zh",
            )
        second = _decoded(ws, 2)
        assert second["source_lang"] == "ja"  # 沿用上一句已知的具体语言码


# ── Failure / reconnect ───────────────────────────────────────────────

class TestFailureRecovery:
    def test_server_error_message(self):
        ws = FakeWS(replies=[INIT_OK, {"type": "error", "code": "update_required"}])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            result = api.translate("hello", source_language="en", target_language="zh")
        assert result.startswith("[ERROR]")
        assert "error" in result.lower()

    def test_recv_timeout_closes_connection_then_reconnects(self):
        ws1 = FakeWS(replies=[INIT_OK], recv_error=TimeoutError("boom"))
        ws2 = FakeWS(replies=[INIT_OK, TRANSLATION_OK])
        connections = iter([ws1, ws2])

        def factory(*args, **kwargs):
            return next(connections)

        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=factory):
            api = _make_api()
            failed = api.translate("hello", source_language="en", target_language="zh")
            assert failed.startswith("[ERROR]")
            assert ws1.closed is True
            # 下一次请求应重新建连并重新 init
            ok = api.translate("hello", source_language="en", target_language="zh")
            assert ok == "前半句，后半句。"
            assert _decoded(ws2, 0)["type"] == "init"

    def test_reset_session_clears_chain(self):
        ws = FakeWS(replies=[
            INIT_OK,
            TRANSLATION_OK,
            {**TRANSLATION_OK, "committed_text": "新句。"},
        ])
        with patch("streaming_translation.api.hymt2._sync_ws_connect", side_effect=_conn(ws)):
            api = _make_api()
            api.translate(
                "前半句，", source_language="ja", target_language="zh",
                is_partial=True, detected_source_language="ja",
            )
            api.reset_session()
            api.translate(
                "新句。", source_language="ja", target_language="zh",
                is_partial=True, detected_source_language="ja",
            )
        # 重置后归属新句：utterance_id 回 0，且不带 previous 字段
        new_update = _decoded(ws, 2)
        assert new_update["utterance_id"] == 0
        assert "previous_source" not in new_update


# ── Pipeline integration ──────────────────────────────────────────────

class TestPipelineIntegration:
    def test_reinit_builds_hymt2_api_with_url(self):
        from streaming_translation import (
            TRANSLATION_API_CLASS_REGISTRY,
            TranslationConfig,
            reinitialize_translator,
        )

        class State:
            pass

        state = State()
        cfg = TranslationConfig(
            target_language="zh",
            translation_api_type="hymt2",
            hymt2_websocket_url="ws://127.0.0.1:18765",
        )
        reinitialize_translator(state, cfg)
        assert isinstance(state.translation_api, HyMT2API)
        assert state.translation_api.websocket_url == "ws://127.0.0.1:18765"
        assert TRANSLATION_API_CLASS_REGISTRY["hymt2"] is HyMT2API

    def test_config_from_module_maps_websocket_url(self):
        from streaming_translation import config_from_module

        class Module:
            HYMT2_WEBSOCKET_URL = "ws://192.168.1.5:18765"
            HYMT2_TIMEOUT_SECONDS = 15
            HYMT2_MAX_RETRIES = 5

        cfg = config_from_module(Module())
        assert cfg.hymt2_websocket_url == "ws://192.168.1.5:18765"
        assert cfg.hymt2_timeout == 15.0
        assert cfg.hymt2_max_retries == 5