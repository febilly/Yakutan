"""面向用户的错误汇总：异常描述、归类、去重，以及 Qwen 识别器把真实断连原因报出来。"""
import pytest

from error_reporting import (
    classify_error,
    clear_errors,
    describe_exception,
    describe_websocket_error,
    get_last_error,
    record_error,
)
from speech_recognizers import qwen_speech_recognizer as qwen_module
from speech_recognizers.qwen_speech_recognizer import QwenSpeechRecognizer


@pytest.fixture(autouse=True)
def _reset_errors():
    clear_errors()
    yield
    clear_errors()


class _FakeBadStatus(Exception):
    """模仿 websocket-client 的 WebSocketBadStatusException。"""

    def __init__(self, status_code, resp_body):
        super().__init__(f"Handshake status {status_code}")
        self.status_code = status_code
        self.resp_body = resp_body


class TestDescribe:
    def test_empty_message_still_names_type(self):
        text = describe_exception(TimeoutError())
        assert text.startswith("TimeoutError")
        assert text.strip() != "TimeoutError:"

    def test_includes_explicit_cause(self):
        try:
            try:
                raise OSError("device busy")
            except OSError as inner:
                raise RuntimeError("打开麦克风失败") from inner
        except RuntimeError as outer:
            text = describe_exception(outer)
        assert "打开麦克风失败" in text
        assert "device busy" in text

    def test_websocket_handshake_body_is_parsed(self):
        error = _FakeBadStatus(
            401,
            b'{"code":"InvalidApiKey","message":"Invalid API-key provided."}',
        )
        text = describe_websocket_error(error)
        assert "401" in text
        assert "InvalidApiKey" in text
        assert "Invalid API-key provided." in text


class TestClassify:
    @pytest.mark.parametrize(
        "text, hint",
        [
            ("握手失败 HTTP 401: InvalidApiKey - Invalid API-key provided.", "invalidApiKey"),
            ("code=1007, Model not exist.", "modelNotExist"),
            ("WebSocketProtocolException: Invalid close frame.", "invalidCloseFrame"),
            ("TimeoutError: websocket connection could not established within 5s.", "network"),
            ("RuntimeError: 打开麦克风失败: OSError", "audioDevice"),
            ("something else", None),
        ],
    )
    def test_known_patterns(self, text, hint):
        assert classify_error(text) == hint


class TestRecord:
    def test_record_and_read_back(self):
        record_error("service", TimeoutError())
        last = get_last_error()
        assert last["source"] == "service"
        assert last["message"].startswith("TimeoutError")
        assert last["count"] == 1
        assert "last_at" not in last

    def test_repeated_error_is_deduplicated(self):
        first = record_error("recognizer", ConnectionError("code=1007, Model not exist."))
        second = record_error("recognizer", ConnectionError("code=1007, Model not exist."))
        assert second["id"] == first["id"]
        assert get_last_error()["count"] == 2
        assert get_last_error()["hint"] == "modelNotExist"

    def test_new_error_gets_new_id(self):
        first = record_error("recognizer", RuntimeError("a"))
        second = record_error("recognizer", RuntimeError("b"))
        assert second["id"] != first["id"]

    def test_clear(self):
        record_error("service", RuntimeError("x"))
        clear_errors()
        assert get_last_error() is None


class _RecordingCallback:
    def __init__(self):
        self.errors = []
        self.stopped = 0

    def on_session_started(self):
        pass

    def on_session_stopped(self):
        self.stopped += 1

    def on_result(self, event):
        pass

    def on_error(self, error):
        self.errors.append(error)


class _FakeConversation:
    """最小化的 OmniRealtimeConversation：connect 行为由测试注入。"""

    connect_behavior = None

    def __init__(self, callback, **kwargs):
        self.callback = callback

    def _on_error(self, ws, error):
        pass

    def connect(self):
        type(self).connect_behavior(self)
        self.callback.on_open()

    def update_session(self, **kwargs):
        pass

    def close(self):
        pass


class TestQwenRecognizerErrors:
    def _make(self, monkeypatch):
        monkeypatch.setattr(qwen_module, "OmniRealtimeConversation", _FakeConversation)
        monkeypatch.setattr(qwen_module, "refresh_system_proxy_env", lambda: None)
        callback = _RecordingCallback()
        recognizer = QwenSpeechRecognizer(callback=callback, keepalive_interval=3600)
        return recognizer, callback

    def test_connect_timeout_reports_real_handshake_error(self, monkeypatch):
        def fail_with_401(conversation):
            conversation._on_error(None, _FakeBadStatus(
                401, b'{"code":"InvalidApiKey","message":"Invalid API-key provided."}',
            ))
            raise TimeoutError("websocket connection could not established within 5s.")

        _FakeConversation.connect_behavior = staticmethod(fail_with_401)
        recognizer, _ = self._make(monkeypatch)
        with pytest.raises(ConnectionError) as info:
            recognizer.start()
        assert "InvalidApiKey" in str(info.value)
        assert classify_error(describe_exception(info.value)) == "invalidApiKey"

    def test_server_close_after_open_is_reported(self, monkeypatch):
        _FakeConversation.connect_behavior = staticmethod(lambda conversation: None)
        recognizer, callback = self._make(monkeypatch)
        recognizer.start()
        try:
            adapter = recognizer._adapter
            adapter.on_close(1007, "Model not exist.")
            assert len(callback.errors) == 1
            assert "Model not exist." in str(callback.errors[0])
            assert callback.stopped == 1
        finally:
            recognizer.stop()

    def test_unparseable_close_frame_is_reported(self, monkeypatch):
        _FakeConversation.connect_behavior = staticmethod(lambda conversation: None)
        recognizer, callback = self._make(monkeypatch)
        recognizer.start()
        try:
            adapter = recognizer._adapter
            recognizer._conversation._on_error(None, Exception("Invalid close frame."))
            adapter.on_close(None, None)
            assert len(callback.errors) == 1
            assert classify_error(str(callback.errors[0])) == "invalidCloseFrame"
        finally:
            recognizer.stop()

    def test_normal_close_is_not_an_error(self, monkeypatch):
        _FakeConversation.connect_behavior = staticmethod(lambda conversation: None)
        recognizer, callback = self._make(monkeypatch)
        recognizer.start()
        try:
            recognizer._adapter.on_close(1000, "")
            assert callback.errors == []
        finally:
            recognizer.stop()

    def test_intentional_stop_is_not_an_error(self, monkeypatch):
        _FakeConversation.connect_behavior = staticmethod(lambda conversation: None)
        recognizer, callback = self._make(monkeypatch)
        recognizer.start()
        adapter = recognizer._adapter
        recognizer.stop()
        adapter.on_close(1007, "Model not exist.")
        assert callback.errors == []


class TestKeyAttribution:
    def test_invalid_key_keeps_key(self):
        record = record_error("translation", "翻译失败（DeepL）: DeepL API auth failed", key="deepl")
        assert record["hint"] == "invalidApiKey"
        assert record["key"] == "deepl"

    def test_non_key_error_drops_key(self):
        record = record_error("service", TimeoutError("timed out"), key="dashscope")
        assert record["hint"] == "network"
        assert record["key"] is None

    def test_asr_backend_key(self):
        from error_reporting import asr_backend_key

        assert asr_backend_key("qwen") == "dashscope"
        assert asr_backend_key("soniox") == "soniox"
        assert asr_backend_key("doubao_file") == "doubao"
        assert asr_backend_key("local") is None

    def test_deepseek_named_in_console_hint(self):
        from error_reporting import format_hint_zh

        text = format_hint_zh("invalidApiKey", "llm", "api.deepseek.com")
        assert "DeepSeek API Key" in text

    def test_alternating_errors_each_kept_once(self):
        from error_reporting import get_recent_errors

        a1 = record_error("recognizer", RuntimeError("code=1007, Model not exist."))
        b1 = record_error("translation", "翻译失败（DeepL）: DeepL API auth failed", key="deepl")
        a2 = record_error("recognizer", RuntimeError("code=1007, Model not exist."))
        b2 = record_error("translation", "翻译失败（DeepL）: DeepL API auth failed", key="deepl")
        assert a1["id"] == a2["id"]
        assert b1["id"] == b2["id"]
        assert [record["id"] for record in get_recent_errors()] == [a1["id"], b1["id"]]

    def test_request_ids_do_not_defeat_dedupe(self):
        first = record_error("translation", "翻译失败（LLM）: weird (request_id: 7e34b366-2424-409c-bbd4-706794330111)")
        second = record_error("translation", "翻译失败（LLM）: weird (request_id: 11111111-2424-409c-bbd4-706794330111)")
        assert first["id"] == second["id"]


class _FakeFailingAPI:
    SUPPORTS_CONTEXT = False

    def __init__(self, error_text=None, exc=None):
        self._error_text = error_text
        self._exc = exc

    def translate(self, *args, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._error_text


class TestTranslationErrorReporting:
    def _translator(self, api_cls_name, **kwargs):
        import recognition_handler  # noqa: F401 - 注册翻译错误监听
        from streaming_translation import ContextAwareTranslator

        api_cls = type(api_cls_name, (_FakeFailingAPI,), {})
        return ContextAwareTranslator(api_cls(**kwargs), api_name=api_cls_name, target_language="en")

    def test_deepl_auth_error_is_reported_with_key(self):
        translator = self._translator("DeepLAPI", error_text="[ERROR] DeepL API auth failed")
        assert translator.translate("你好").startswith("[ERROR]")
        last = get_last_error()
        assert last["source"] == "translation"
        assert last["key"] == "deepl"
        assert last["hint"] == "invalidApiKey"

    def test_llm_401_is_reported_with_llm_key(self):
        translator = self._translator(
            "OpenRouterAPI",
            error_text="[ERROR] Error code: 401 - {'error': {'message': 'Authentication Fails, Your api key: ****tkey is invalid'}}",
        )
        translator.translate("你好")
        last = get_last_error()
        assert last["key"] == "llm"
        assert last["hint"] == "invalidApiKey"

    def test_exception_inside_translate_is_reported(self):
        translator = self._translator("QwenMTAPI", exc=RuntimeError("Error code: 401 - invalid_api_key"))
        assert translator.translate("你好").startswith("[ERROR]")
        last = get_last_error()
        assert last["key"] == "dashscope"

    def test_successful_translation_reports_nothing(self):
        translator = self._translator("DeepLAPI", error_text="hello")
        assert translator.translate("你好") == "hello"
        assert get_last_error() is None
