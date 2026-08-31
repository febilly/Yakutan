"""Tests for the Hy-MT2 WebSocket streaming translation backend."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
        assert init["service"] == "hymt2"
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
                self.drafts = []
            def generate(self, prompt, max_tokens=128, draft=None):
                self.prompts.append(prompt)
                self.drafts.append(draft)
                if "Hello" in prompt and "world" not in prompt:
                    return "你好", [1, 2]
                elif "world" in prompt:
                    return "你好世界", [1, 2, 3]
                return "翻译结果", [9]

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
            def generate(self, prompt, max_tokens=128, draft=None):
                self.calls += 1
                return f"译文{self.calls}", [self.calls]

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
            def generate(self, prompt, max_tokens=128, draft=None):
                self.calls += 1
                return "ok", [1]

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
        from local_inference.model_manager import (
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


# ── Local engine lifecycle (shared instance / refcount / prewarm) ──────────

class TestLocalEngineLifecycle:
    @pytest.fixture(autouse=True)
    def _clean_engine_registry(self):
        from streaming_translation.api.hymt2 import (
            _local_engine_registry,
            _local_engine_registry_lock,
        )
        with _local_engine_registry_lock:
            _local_engine_registry.clear()
        yield
        with _local_engine_registry_lock:
            _local_engine_registry.clear()

    @staticmethod
    def _fake_engine_class():
        instances: list = []

        class FakeLocalEngine:
            def __init__(self, model_path=None, device="auto"):
                self.device = device
                self.model = object()
                self.ctx = object()
                self._engine_lock = threading.Lock()
                instances.append(self)

        return FakeLocalEngine, instances

    def test_acquire_release_refcount_and_dispose(self):
        from streaming_translation.api.hymt2 import (
            acquire_local_engine,
            release_local_engine,
        )
        fake_cls, instances = self._fake_engine_class()
        with patch("streaming_translation.api.hymt2.HyMT2LocalEngine", fake_cls):
            e1 = acquire_local_engine(device="cpu")
            e2 = acquire_local_engine(device="cpu")
            assert e1 is e2
            assert len(instances) == 1

            # 不同运行位置 → 独立实例
            e3 = acquire_local_engine(device="auto")
            assert e3 is not e1
            assert len(instances) == 2

            # 第一个引用释放 → 尚未归零，不卸载
            assert release_local_engine(e1) is False
            assert instances[0].model is not None
            # 最后一个引用释放 → 真正卸载（通过 __del__ 释放模型/上下文）
            assert release_local_engine(e2) is True
            assert instances[0].model is None
            assert instances[0].ctx is None
            assert release_local_engine(e3) is True

    def test_api_load_unload_shares_engine(self):
        fake_cls, instances = self._fake_engine_class()
        with patch("streaming_translation.api.hymt2.HyMT2LocalEngine", fake_cls):
            api = HyMT2API(backend="local", local_device="cpu")
            assert api.load_local_engine() is True
            assert api._local_engine is not None
            assert api.load_local_engine() is True  # 幂等

            api2 = HyMT2API(backend="local", local_device="cpu")
            assert api2.load_local_engine() is True
            assert api2._local_engine is api._local_engine
            assert len(instances) == 1

            assert api.unload_local_engine() is True
            assert api._local_engine is None
            assert api2._local_engine is not None  # 仍被 api2 持有
            assert instances[0].model is not None

            assert api2.unload_local_engine() is True
            assert instances[0].model is None
            assert instances[0].ctx is None
            assert api2.unload_local_engine() is False  # 无引用可释放

            # 非 local 后端不参与加载
            api3 = HyMT2API(backend="api")
            assert api3.load_local_engine() is False
            assert api3.unload_local_engine() is False
            assert len(instances) == 1

    def test_pipeline_reinit_shares_and_frees_engine(self):
        from streaming_translation import (
            TranslationConfig,
            prewarm_local_engines,
            reinitialize_translator,
        )
        fake_cls, instances = self._fake_engine_class()
        cfg_local = dict(
            target_language="zh",
            secondary_target_language="ja",
            translation_api_type="hymt2",
            hymt2_backend="local",
        )
        with patch("streaming_translation.api.hymt2.HyMT2LocalEngine", fake_cls):
            state = SimpleNamespace()
            # 服务启动路径：reinitialize 本身不加载，prewarm 立即加载
            reinitialize_translator(state, TranslationConfig(**cfg_local))
            assert state.translation_api._local_engine is None
            prewarm_local_engines(state)
            assert len(instances) == 1
            assert state.translation_api._local_engine is (
                state.secondary_translation_api._local_engine
            )

            # 切换到 api 后端：旧引擎引用全部释放 → 真正卸载
            reinitialize_translator(
                state,
                TranslationConfig(
                    target_language="zh",
                    translation_api_type="hymt2",
                    hymt2_backend="api",
                ),
            )
            assert instances[0].model is None
            assert instances[0].ctx is None
            assert state.translation_api._local_engine is None

            # 切回 local 并预载：加载新实例
            reinitialize_translator(state, TranslationConfig(**cfg_local))
            prewarm_local_engines(state)
            assert len(instances) == 2
            assert state.translation_api._local_engine is instances[1]
            assert state.secondary_translation_api._local_engine is instances[1]

    def test_runtime_status_reflects_load_state(self):
        from streaming_translation.api.hymt2 import get_local_engine_runtime_status
        fake_cls, instances = self._fake_engine_class()
        with patch("streaming_translation.api.hymt2.HyMT2LocalEngine", fake_cls):
            api = HyMT2API(backend="local", local_device="cpu")
            assert get_local_engine_runtime_status()["loaded"] is False

            api.load_local_engine()
            status = get_local_engine_runtime_status()
            assert status["loaded"] is True
            assert status["loading"] is False
            assert status["engines"][0]["refcount"] == 1

            api.unload_local_engine()
            assert get_local_engine_runtime_status()["loaded"] is False

    def test_failed_load_is_retried_on_next_request(self):
        from streaming_translation.api.hymt2 import (
            acquire_local_engine,
            get_local_engine_runtime_status,
            release_local_engine,
        )
        attempts = {"n": 0}

        class FailingEngine:
            def __init__(self, model_path=None, device="auto"):
                attempts["n"] += 1
                self.model = object()
                self.ctx = object()
                self._engine_lock = threading.Lock()
                if attempts["n"] == 1:
                    raise RuntimeError("load boom")

            def generate(self, prompt, max_tokens=128, draft=None):
                return "ok", [1]

        with patch("streaming_translation.api.hymt2.HyMT2LocalEngine", FailingEngine):
            # 首次加载失败：抛异常，注册表不残留已加载实例
            with pytest.raises(RuntimeError):
                acquire_local_engine(device="cpu")
            assert get_local_engine_runtime_status()["loaded"] is False

            # 之后请求自动重试加载并成功
            api = HyMT2API(backend="local", local_device="cpu")
            assert api.translate("hi", target_language="zh") == "ok"
            assert get_local_engine_runtime_status()["loaded"] is True
            assert api._local_engine is not None
            assert api.unload_local_engine() is True
            assert release_local_engine(api._local_engine) is False
            assert api._local_engine is None
            assert get_local_engine_runtime_status()["loaded"] is False

    def test_config_switch_during_load_does_not_leak(self):
        """加载进行中被热重载替换：旧引用立即释放，最终仅剩 1 个有效引擎。"""
        import time

        from streaming_translation.api.hymt2 import get_local_engine_runtime_status
        from streaming_translation.pipeline import (
            prewarm_local_engines,
            release_local_engines,
        )

        instances: list = []
        started = threading.Event()

        class SlowEngine:
            def __init__(self, model_path=None, device="auto"):
                self.model = object()
                self.ctx = object()
                self._engine_lock = threading.Lock()
                instances.append(self)
                started.set()
                time.sleep(0.2)

        with patch("streaming_translation.api.hymt2.HyMT2LocalEngine", SlowEngine):
            state = SimpleNamespace()
            api1 = HyMT2API(backend="local", local_device="cpu")
            state.translation_api = api1

            prewarm_t = threading.Thread(
                target=prewarm_local_engines, args=(state,)
            )
            prewarm_t.start()
            started.wait(5)
            time.sleep(0.05)

            # 加载中途切换：旧实例失去挂载，新实例接管
            release_local_engines(state)
            api2 = HyMT2API(backend="local", local_device="cpu")
            state.translation_api = api2

            prewarm_t.join()
            # 被替换的旧 API 必须已放弃引用
            assert api1._local_engine is None

            # 切换后再次预载：新 API 必须拿到可用引擎（不得拿到已卸载实例）
            prewarm_local_engines(state)
            assert api2._local_engine is not None
            assert api2._local_engine.ctx is not None
            assert get_local_engine_runtime_status()["loaded"] is True

            # 收尾释放后注册表应为空（无泄漏）
            release_local_engines(state)
            assert get_local_engine_runtime_status()["loaded"] is False
            assert api2._local_engine is None

        assert len(instances) <= 2
        assert all(
            e.ctx is None for e in instances if e is not api2._local_engine
        )

    def test_persistent_load_failure_reports_error(self):
        class AlwaysFailingEngine:
            def __init__(self, model_path=None, device="auto"):
                raise RuntimeError("no gguf available")

        with patch(
            "streaming_translation.api.hymt2.HyMT2LocalEngine",
            AlwaysFailingEngine,
        ):
            api = HyMT2API(backend="local", local_device="cpu")
            for _ in range(2):
                assert api.translate("hi", target_language="zh").startswith(
                    "[ERROR]"
                )
            # 加载不成功后不会持有引用，卸载也应无副作用
            assert api._local_engine is None
            assert api.unload_local_engine() is False
