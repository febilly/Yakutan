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

    def test_reinit_builds_hymt2_api_with_local_backend(self):
        from streaming_translation import (
            TranslationConfig,
            reinitialize_translator,
        )

        class State:
            pass

        state = State()
        cfg = TranslationConfig(
            target_language="zh",
            translation_api_type="hymt2",
            hymt2_backend="local",
        )
        reinitialize_translator(state, cfg)
        assert isinstance(state.translation_api, HyMT2API)
        assert state.translation_api.backend == "local"

    def test_local_device_reaches_api_and_defaults_to_auto(self):
        from streaming_translation import (
            TranslationConfig,
            config_from_module,
            reinitialize_translator,
        )

        class Module:
            HYMT2_BACKEND = "local"

        assert config_from_module(Module()).hymt2_local_device == "auto"

        for raw, expected in (
            ("CPU", "cpu"),
            ("gpu", "auto"),          # 旧值：等价于自动挑一张 GPU
            ("Vulkan:1", "vulkan:1"),
            ("vulkan:x", "auto"),     # 非法索引回退
            (None, "auto"),
        ):
            class DeviceModule:
                HYMT2_BACKEND = "local"
                HYMT2_LOCAL_DEVICE = raw

            assert config_from_module(DeviceModule()).hymt2_local_device == expected, raw

        class State:
            pass

        state = State()
        reinitialize_translator(
            state,
            TranslationConfig(
                target_language="zh",
                translation_api_type="hymt2",
                hymt2_backend="local",
                hymt2_local_device="cpu",
            ),
        )
        assert state.translation_api.local_device == "cpu"

    def test_switching_local_device_is_seen_as_config_change(self):
        """热重载走 _is_primary_config_changed；漏掉该字段会继续用旧设备上的引擎。"""
        from streaming_translation import TranslationConfig, reinitialize_translator
        from streaming_translation.pipeline import _is_primary_config_changed

        def cfg_for(device):
            return TranslationConfig(
                target_language="zh",
                translation_api_type="hymt2",
                hymt2_backend="local",
                hymt2_local_device=device,
            )

        class State:
            pass

        state = State()
        reinitialize_translator(state, cfg_for("gpu"))

        assert _is_primary_config_changed(state, cfg_for("gpu")) is False
        assert _is_primary_config_changed(state, cfg_for("cpu")) is True

        reinitialize_translator(state, cfg_for("cpu"))
        assert state.translation_api.local_device == "cpu"


# ── Local Backend & Model Manager ──────────────────────────────────────

class TestLocalBackend:
    def test_local_backend_missing_model_returns_error(self):
        api = HyMT2API(backend="local", model_path="nonexistent_model.gguf")
        res = api.translate("Hello", source_language="en", target_language="zh")
        assert res.startswith("[ERROR]")

    def test_local_backend_translates_and_revises(self):
        api = HyMT2API(backend="local")
        # Mock _local_engine to test revision prompt logic
        class FakeLocalEngine:
            def __init__(self):
                self.prompts = []
            def generate(self, prompt, max_tokens=128):
                self.prompts.append(prompt)
                if "Hello" in prompt and "world" not in prompt:
                    return "你好"
                elif "world" in prompt:
                    return "你好世界"
                return "翻译结果"

        fake_engine = FakeLocalEngine()
        api._local_engine = fake_engine

        # Step 1: partial
        out1 = api.translate("Hello", source_language="en", target_language="zh", is_partial=True)
        assert out1 == "你好"
        assert len(fake_engine.prompts) == 1

        # Step 2: revised partial
        out2 = api.translate("Hello world", source_language="en", target_language="zh", is_partial=False)
        assert out2 == "你好世界"
        assert len(fake_engine.prompts) == 2
        assert "Previous version of the current source:\nHello" in fake_engine.prompts[1]
        assert "Previous translation of the current source:\n你好" in fake_engine.prompts[1]

    def test_final_reuses_last_translation_when_source_unchanged(self):
        api = HyMT2API(backend="local")

        class FakeLocalEngine:
            def __init__(self):
                self.calls = 0
            def generate(self, prompt, max_tokens=128):
                self.calls += 1
                return f"译文{self.calls}"

        fake_engine = FakeLocalEngine()
        api._local_engine = fake_engine

        out1 = api.translate("Hi", source_language="en", target_language="zh", is_partial=True)
        assert out1 == "译文1"
        assert fake_engine.calls == 1

        # 终句原文与上次中间一致 → 复用上次译文，不再推理
        out2 = api.translate("Hi", source_language="en", target_language="zh", is_partial=False)
        assert out2 == "译文1"
        assert fake_engine.calls == 1
        # 修订链按终句方式收尾
        assert api._last_final is True
        assert api._prev_source is None
        assert api._history[-1] == ["Hi", "译文1"]

        # 下一句正常触发推理
        out3 = api.translate("Hi there", source_language="en", target_language="zh", is_partial=False)
        assert out3 == "译文2"
        assert fake_engine.calls == 2

    def test_final_not_reused_when_source_or_language_changes(self):
        class FakeLocalEngine:
            def __init__(self):
                self.calls = 0
            def generate(self, prompt, max_tokens=128):
                self.calls += 1
                return "ok"

        # 原文变了 → 不能复用
        api = HyMT2API(backend="local")
        fake_engine = FakeLocalEngine()
        api._local_engine = fake_engine
        api.translate("Hi", source_language="en", target_language="zh", is_partial=True)
        out = api.translate("Hi there", source_language="en", target_language="zh", is_partial=False)
        assert out == "ok"
        assert fake_engine.calls == 2

        # 源语言变化（同原文）→ 也不能复用
        api2 = HyMT2API(backend="local")
        fake_engine2 = FakeLocalEngine()
        api2._local_engine = fake_engine2
        api2.translate("Hi", source_language="en", target_language="zh", is_partial=True)
        out2 = api2.translate("Hi", source_language="ja", target_language="zh", is_partial=False)
        assert out2 == "ok"
        assert fake_engine2.calls == 2

    def test_final_reuse_skips_ws_request_when_source_unchanged(self):
        api = _make_api()
        with patch("streaming_translation.api.hymt2._sync_ws_connect") as connect:
            # 模拟已发送过同一原文的中间结果（链处于 partial 之后状态）
            api._sent_any = True
            api._last_final = False
            api._prev_source = "こんにちは"
            api._prev_translation = "你好"
            api._last_source_lang = "ja"
            api._last_target_lang = "zh"
            out = api.translate("こんにちは", source_language="ja", target_language="zh", is_partial=False)
        assert out == "你好"
        connect.assert_not_called()
        assert api._last_final is True
        assert api._history[-1] == ["こんにちは", "你好"]

    def test_model_manager_hymt2_helpers(self):
        from local_asr.model_manager import (
            get_all_local_models_status,
            get_hymt2_model_path,
            get_hymt2_status,
            is_hymt2_cached,
        )
        status = get_hymt2_status()
        assert status["engine"] == "hymt2"
        assert "ready" in status

        all_status = get_all_local_models_status()
        assert "asr" in all_status
        assert "translation" in all_status
        assert "hymt2" in all_status["translation"]