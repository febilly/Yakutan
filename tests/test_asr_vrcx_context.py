from __future__ import annotations

import json

import numpy as np

import vrcx_context_bridge as bridge


class DummyCallback:
    def on_session_started(self):
        pass

    def on_session_stopped(self):
        pass

    def on_result(self, event):
        pass

    def on_error(self, error):
        raise error


def _store_vrcx_context() -> None:
    payload = {
        "sequence": 101,
        "context": {
            "ok": True,
            "self": {"name": "SelfUser"},
            "world": {"name": "Test World", "author": "WorldMaker"},
            "friends": [{"name": "Alice"}],
            "players": [{"name": "Bob"}],
        },
        "contextText": "World: Test World\nKnown players here: Alice; Bob",
    }
    ok, reason = bridge.store_payload(
        bridge.get_token(),
        json.dumps(payload).encode("utf-8"),
    )
    assert ok is True
    assert reason == "ok"


def test_qwen_realtime_transcription_params_include_vrcx_context(monkeypatch):
    import speech_recognizers.qwen_speech_recognizer as qwen_mod

    class DummyTranscriptionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    _store_vrcx_context()
    monkeypatch.setattr(qwen_mod, "TranscriptionParams", DummyTranscriptionParams)

    recognizer = qwen_mod.QwenSpeechRecognizer(
        callback=DummyCallback(),
        corpus_text="HotTerm",
        sample_rate=16000,
    )

    params = recognizer._resolve_transcription_params()
    corpus_text = params.kwargs["corpus_text"]

    assert "HotTerm" in corpus_text
    assert "VRChat ASR hints" in corpus_text
    assert "Test World" in corpus_text
    assert "Alice" in corpus_text


def test_qwen_realtime_refreshes_session_when_vrcx_context_changes(monkeypatch):
    import speech_recognizers.qwen_speech_recognizer as qwen_mod

    class DummyTranscriptionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyConversation:
        def __init__(self):
            self.updates = []

        def update_session(self, **kwargs):
            self.updates.append(kwargs)

    _store_vrcx_context()
    monkeypatch.setattr(qwen_mod, "TranscriptionParams", DummyTranscriptionParams)

    recognizer = qwen_mod.QwenSpeechRecognizer(callback=DummyCallback(), corpus_text="HotTerm")
    conversation = DummyConversation()
    recognizer._conversation = conversation
    recognizer._applied_transcription_corpus_text = "HotTerm"

    recognizer._refresh_dynamic_transcription_context()

    assert len(conversation.updates) == 1
    corpus_text = conversation.updates[0]["transcription_params"].kwargs["corpus_text"]
    assert "VRChat ASR hints" in corpus_text
    assert "Test World" in corpus_text


def test_qwen_realtime_defers_context_refresh_until_next_segment(monkeypatch):
    import speech_recognizers.qwen_speech_recognizer as qwen_mod

    class DummyTranscriptionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyConversation:
        def __init__(self):
            self.updates = []

        def update_session(self, **kwargs):
            self.updates.append(kwargs)

    _store_vrcx_context()
    monkeypatch.setattr(qwen_mod, "TranscriptionParams", DummyTranscriptionParams)

    recognizer = qwen_mod.QwenSpeechRecognizer(callback=DummyCallback(), corpus_text="HotTerm")
    conversation = DummyConversation()
    recognizer._conversation = conversation
    recognizer._applied_transcription_corpus_text = "HotTerm"

    recognizer._mark_input_speech_started("item-1")
    recognizer._refresh_dynamic_transcription_context()

    assert conversation.updates == []
    assert recognizer._pending_transcription_corpus_text is not None

    recognizer._mark_input_speech_stopped("item-1")
    recognizer._mark_transcription_item_done("item-1")

    assert conversation.updates == []

    recognizer._refresh_dynamic_transcription_context()

    assert len(conversation.updates) == 1
    corpus_text = conversation.updates[0]["transcription_params"].kwargs["corpus_text"]
    assert "VRChat ASR hints" in corpus_text
    assert "Test World" in corpus_text


def test_qwen_audio3_start_sends_fresh_vrcx_context_and_hot_words(monkeypatch):
    import speech_recognizers.dashscope_speech_recognizer as dashscope_mod
    import speech_recognizers.qwen_audio3_speech_recognizer as qwen_audio3_mod

    class DummyRecognition:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.starts = []

        def start(self, **kwargs):
            self.starts.append(kwargs)

    _store_vrcx_context()
    monkeypatch.setattr(dashscope_mod, "Recognition", DummyRecognition)
    monkeypatch.setattr(qwen_audio3_mod, "refresh_system_proxy_env", lambda: None)

    recognizer = qwen_audio3_mod.QwenAudio3SpeechRecognizer(
        callback=DummyCallback(),
        model="qwen-audio-3.0-asr-flash-streaming",
        format="pcm",
        sample_rate=16000,
        hot_words=[
            {"text": "HotTerm", "weight": 4},
            {"text": "SuperTerm", "weight": 50},
            {"text": "OutOfRange", "weight": 9},
        ],
    )

    assert recognizer._recognition.kwargs["vocabulary"] == {
        "HotTerm": 4,
        "SuperTerm": 50,
        "OutOfRange": 5,
    }

    recognizer.start()

    context = recognizer._recognition.starts[0]["raw_input"]["context"]
    assert all(message["role"] == "user" for message in context)
    assert len(context) <= qwen_audio3_mod.MAX_CONTEXT_ROUNDS
    joined = "\n".join(message["content"][0]["text"] for message in context)
    assert "VRChat ASR hints" in joined
    assert "Test World" in joined
    assert "Alice" in joined


def test_dashscope_complete_and_close_notify_session_stop_once():
    import speech_recognizers.dashscope_speech_recognizer as dashscope_mod

    class CountingCallback(DummyCallback):
        def __init__(self):
            self.started = 0
            self.stopped = 0

        def on_session_started(self):
            self.started += 1

        def on_session_stopped(self):
            self.stopped += 1

    callback = CountingCallback()
    adapter = dashscope_mod._DashscopeCallbackAdapter(callback)

    adapter.on_open()
    adapter.on_complete()
    adapter.on_close()

    assert callback.started == 1
    assert callback.stopped == 1

    # A subsequent real session gets its own single stop notification.
    adapter.on_open()
    adapter.on_close()
    assert callback.started == 2
    assert callback.stopped == 2


def test_dashscope_error_callback_does_not_stringify_broken_sdk_result():
    import speech_recognizers.dashscope_speech_recognizer as dashscope_mod

    class BrokenError:
        message = 'EmptyAudio'
        request_id = 'request-123'
        code = '44'

        def __str__(self):
            raise AttributeError('headers')

    class CollectingCallback(DummyCallback):
        def __init__(self):
            self.errors = []

        def on_error(self, error):
            self.errors.append(error)

    callback = CollectingCallback()
    adapter = dashscope_mod._DashscopeCallbackAdapter(callback)
    adapter.on_error(BrokenError())

    assert 'EmptyAudio' in str(callback.errors[0])
    assert 'request-123' in str(callback.errors[0])


def test_qwen_audio3_stop_after_pause_is_idempotent(monkeypatch):
    """闭麦（pause 已停掉会话）后关闭服务不应再抛 InvalidParameter。"""
    import speech_recognizers.dashscope_speech_recognizer as dashscope_mod
    import speech_recognizers.qwen_audio3_speech_recognizer as qwen_audio3_mod

    class DummyRecognition:
        def __init__(self, **kwargs):
            self._running = False
            self.stop_calls = 0

        def start(self, **kwargs):
            self._running = True

        def stop(self):
            if not self._running:
                raise RuntimeError("Speech recognition has stopped.")
            self._running = False
            self.stop_calls += 1

        def send_audio_frame(self, data):
            pass

    monkeypatch.setattr(dashscope_mod, "Recognition", DummyRecognition)
    monkeypatch.setattr(qwen_audio3_mod, "refresh_system_proxy_env", lambda: None)

    recognizer = qwen_audio3_mod.QwenAudio3SpeechRecognizer(
        callback=DummyCallback(),
        model="qwen-audio-3.0-asr-flash-streaming",
        format="pcm",
        sample_rate=16000,
    )

    recognizer.start()
    recognizer.pause()
    recognizer.stop()  # 不应抛异常

    assert recognizer._recognition.stop_calls == 1
    assert recognizer._recognition._running is False


def test_qwen_audio3_context_rounds_respect_server_limits():
    from speech_recognizers.qwen_audio3_speech_recognizer import (
        MAX_CONTEXT_CHARS_PER_ROUND,
        MAX_CONTEXT_ROUNDS,
        split_context_rounds,
    )

    assert split_context_rounds("   ") == []

    long_lines = "\n".join("x" * 900 for _ in range(10))
    rounds = split_context_rounds(long_lines)

    assert len(rounds) == MAX_CONTEXT_ROUNDS
    assert all(len(chunk) <= MAX_CONTEXT_CHARS_PER_ROUND for chunk in rounds)

    packed = split_context_rounds("\n".join(["short line"] * 5))
    assert packed == ["\n".join(["short line"] * 5)]


def test_soniox_config_merges_vrcx_context_with_existing_context(monkeypatch):
    import speech_recognizers.soniox_speech_recognizer as soniox_mod

    _store_vrcx_context()
    monkeypatch.setattr(soniox_mod, "WEBSOCKETS_AVAILABLE", True)

    recognizer = soniox_mod.SonioxSpeechRecognizer(
        callback=DummyCallback(),
        api_key="test-key",
        context={"terms": ["ExistingTerm"], "text": "Existing text"},
    )

    config = recognizer._build_config()
    context = config["context"]

    assert "ExistingTerm" in context["terms"]
    assert "Test World" in context["terms"]
    assert "Alice" in context["terms"]
    assert "Existing text" in context["text"]
    assert "VRChat ASR hints" in context["text"]
    assert "Bob" in context["text"]


def test_local_qwen3_asr_receives_fresh_vrcx_context(monkeypatch):
    from speech_recognizers.local_speech_recognizer import LocalSpeechRecognizer

    class DummyEngine:
        def __init__(self):
            self.corpus_text = None
            self.update_context = None

        def set_corpus_text(self, text):
            self.corpus_text = text

        def transcribe(self, audio, *, update_context=True):
            self.update_context = update_context
            return {"text": "hello"}

    _store_vrcx_context()
    engine = DummyEngine()
    recognizer = LocalSpeechRecognizer(callback=DummyCallback(), corpus_text="HotTerm")
    monkeypatch.setattr(recognizer, "_ensure_engine", lambda: engine)

    result = recognizer._transcribe(np.array([0.1], dtype=np.float32), is_final=False)

    assert result == ("hello", {"text": "hello"})
    assert "HotTerm" in engine.corpus_text
    assert "VRChat ASR hints" in engine.corpus_text
    assert "Test World" in engine.corpus_text
    assert engine.update_context is False
