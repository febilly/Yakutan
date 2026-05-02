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
