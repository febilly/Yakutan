import asyncio
import json
from unittest.mock import MagicMock

import pytest

from asr_context import RecentASRContext
from ipc_client import IPCClient
from speech_recognizers.base_speech_recognizer import RecognitionEvent


def test_history_retains_recent_originals_and_expires():
    now = [0.0]
    context = RecentASRContext(clock=lambda: now[0])
    context.add("old")
    now[0] = 100.0
    context.add("Kubernetes", foreign=True)
    context.add("x" * 1000)
    now[0] = 121.0
    assert context.snapshot() == ["对方：Kubernetes", "自己：" + "x" * 397]
    context.clear_foreign()
    assert context.snapshot() == ["自己：" + "x" * 397]
    for i in range(8):
        context.add(str(i))
    assert context.snapshot() == [f"自己：{i}" for i in range(3, 8)]


def test_ipc_original_reaches_asr_without_translator_and_disconnect_clears():
    async def run():
        context = RecentASRContext()
        client = IPCClient(asr_context=context)
        reader = asyncio.StreamReader()
        reader.feed_data((json.dumps({
            "type": "FOREIGN_SPEECH", "source_text": "Kubernetes cluster",
            "translated_text": "must not enter ASR",
        }) + "\n").encode())
        client._reader = reader
        client._on_disconnect = MagicMock(side_effect=lambda: asyncio.sleep(0))
        task = asyncio.create_task(client._read_loop())
        await asyncio.sleep(0)
        assert context.snapshot() == ["对方：Kubernetes cluster"]
        reader.feed_eof()
        await task
        context.add("my reply")
        await client._close_connection()
        assert context.snapshot() == ["自己：my reply"]
    asyncio.run(run())


def test_callback_records_only_accepted_finals(monkeypatch):
    import recognition_handler as handler
    monkeypatch.setattr(handler.config, "ENABLE_TRANSLATION", False)
    monkeypatch.setattr(handler.config, "SHOW_PARTIAL_RESULTS", False)
    state = MagicMock()
    state.asr_context = RecentASRContext()
    callback = handler.VRChatRecognitionCallback(state)
    callback.loop = None
    callback.on_result(RecognitionEvent(text="partial", is_final=False))
    callback.on_result(RecognitionEvent(text="final original", is_final=True))
    callback._discard_results = True
    callback._discard_deadline = float("inf")
    callback.on_result(RecognitionEvent(text="discarded", is_final=True))
    assert state.asr_context.snapshot() == ["自己：final original"]


def test_active_context_updates_before_audio_and_clears_expired(monkeypatch):
    import speech_recognizers.dashscope_speech_recognizer as dashscope_mod
    import speech_recognizers.qwen_audio3_speech_recognizer as qwen_mod
    from dashscope.audio.asr import Recognition

    # Exercise the real SDK queue with network startup mocked out. This catches
    # payload wrappers/ordering mistakes that a mocked update_context cannot.
    sdk = Recognition(model="qwen-audio-3.1-asr-flash-streaming",
                      format="pcm", sample_rate=16000, callback=MagicMock())
    def start(**kwargs):
        sdk._running = True
        sdk._kwargs.update(kwargs)
    monkeypatch.setattr(sdk, "start", start)
    monkeypatch.setattr(dashscope_mod, "Recognition", lambda **kwargs: sdk)
    monkeypatch.setattr(qwen_mod, "refresh_system_proxy_env", lambda: None)
    monkeypatch.setattr(qwen_mod, "get_asr_context_terms", lambda: [])
    now = [0.0]
    history = RecentASRContext(clock=lambda: now[0])
    recognizer = qwen_mod.QwenAudio3SpeechRecognizer(
        callback=MagicMock(), asr_context_provider=history.snapshot,
    )
    try:
        recognizer.start()
        assert sdk._kwargs["raw_input"] is None
        history.add("Kubernetes", foreign=True)
        recognizer.send_audio_frame(b"audio1")
        update = sdk._stream_data.get_nowait()
        assert update == {"input": {"context": [{
            "role": "user", "content": [{"type": "input_text", "text": "对方：Kubernetes"}],
        }]}}
        assert sdk._stream_data.get_nowait() == b"audio1"
        recognizer.send_audio_frame(b"audio2")
        assert sdk._stream_data.get_nowait() == b"audio2"
        assert sdk._stream_data.empty()
        now[0] = 121.0
        recognizer.send_audio_frame(b"audio3")
        assert sdk._stream_data.get_nowait() == {"input": {"context": []}}
        assert sdk._stream_data.get_nowait() == b"audio3"
        history.add("new original")
        sdk._running = False
        recognizer.resume()
        assert sdk._kwargs["raw_input"]["context"][0]["content"][0]["text"] == "自己：new original"
    finally:
        sdk._running = False


def test_vocabulary_limits_and_super_hotwords():
    from speech_recognizers.qwen_audio3_speech_recognizer import build_vocabulary
    entries = [{"text": f"term{i}", "weight": 50} for i in range(2100)]
    vocabulary = build_vocabulary(["字" * 16, "a b c d e f g h", *entries])
    assert len(vocabulary) == 2000
    assert sum(weight == 50 for weight in vocabulary.values()) == 50
    assert vocabulary["term50"] == 5


def test_old_sdk_rejected_for_dynamic_context(monkeypatch):
    import speech_recognizers.dashscope_speech_recognizer as dashscope_mod
    from speech_recognizers.qwen_audio3_speech_recognizer import QwenAudio3SpeechRecognizer
    monkeypatch.setattr(dashscope_mod, "Recognition", lambda **kwargs: object())
    with pytest.raises(RuntimeError, match="1.27.5"):
        QwenAudio3SpeechRecognizer(callback=MagicMock(), asr_context_provider=lambda: [])
