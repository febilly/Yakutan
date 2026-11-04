"""Regressions for the committed realtime-subtitle inference fixes."""

import asyncio
import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import config
from local_inference.asr_qwen3 import Qwen3ASREngine
from local_inference.hymt_prompt import source_was_cut, trained_history
from local_inference.qwen_microbatch import QwenBatchInput, QwenMicroBatchEngine, _Sequence
from local_inference.spec_decode import DraftTracker
from streaming_translation.api.hymt2 import HyMT2API

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote_inference_service"))
from remote_inference_service import server, server_windows


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return "译文", [1, 2, 3]


def test_training_history_filters_missing_translations_before_limiting():
    pairs = [[f"source{i}", f"target{i}"] for i in range(7)]
    assert trained_history(pairs + [["orphan", ""], "source only"]) == pairs[-5:]
    assert trained_history(pairs, 0) == []
    assert trained_history([{"source": " hello ", "translation": " 你好 "}]) == [["hello", "你好"]]


@pytest.mark.parametrize("module", [server, server_windows])
def test_all_backends_render_the_exact_training_template(module):
    pairs = [[f"old{i}", f"译{i}"] for i in range(7)]
    payload = {"source_lang": "en", "target_lang": "zh", "history": pairs}
    expected = (
        "[Background Information]\nRecent utterances and their translations:\n"
        "old2 → 译2\nold3 → 译3\nold4 → 译4\nold5 → 译5\nold6 → 译6\n\n"
        "When the source meaning has not changed, preserve the still-correct prefix "
        "of the previous translation whenever possible. When content is added or "
        "corrected, accuracy and completeness take priority.\n\n"
        "Please translate the following text from English into Chinese, taking the "
        "provided background information into consideration.\n\n[Source Text]\nHello"
    )
    assert module._translation_prompt(payload, "Hello", 10) == expected
    api = HyMT2API(backend="local")
    api._local_engine = engine = RecordingEngine()
    api._history = pairs
    assert api.translate("Hello", "en", "zh", is_partial=True) == "译文"
    assert engine.calls[0][0] == server.CHAT_PREFIX + expected + server.CHAT_SUFFIX


def test_cut_revision_drops_local_translation_and_speculative_draft():
    api = HyMT2API(backend="local")
    api._local_engine = engine = RecordingEngine()
    api.translate("Hello world and another sentence.", "en", "zh", is_partial=True)
    api.translate("Hello", "en", "zh", is_partial=True)
    prompt, options = engine.calls[-1]
    assert "Previous version" not in prompt and "Previous translation" not in prompt
    assert options["draft"] is None
    # A normal extension still uses the newly established revision chain.
    api.translate("Hello again", "en", "zh", is_partial=True)
    assert "Previous version of the current source:\nHello" in engine.calls[-1][0]
    assert engine.calls[-1][1]["draft"] == [1, 2, 3]


def test_cut_remote_revision_cannot_fall_back_to_stale_translation():
    api = HyMT2API(backend="remote", websocket_url="ws://unused")
    replies = iter([{"committed_text": "旧译文"}, {"committed_text": ""}])
    requests = []
    api._request = lambda payload: requests.append(payload) or next(replies)
    assert api.translate("Hello world and another sentence.", "en", "zh", is_partial=True) == "旧译文"
    assert api.translate("Hello", "en", "zh", is_partial=True) == ""
    assert "previous_source" not in requests[-1]
    assert "previous_translation" not in requests[-1]


@pytest.mark.parametrize("module", [server, server_windows])
def test_server_also_guards_cut_revisions_from_older_clients(module):
    payload = {"type": "update", "source": "Hello", "previous_source": "Hello and another sentence.",
               "previous_translation": "stale translation"}
    sent, prompts = [], []

    class WebSocket:
        async def send(self, message):
            sent.append(json.loads(message))

        def __aiter__(self):
            async def messages():
                yield json.dumps(payload)
            return messages()

    class Backend:
        async def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            return ""

    asyncio.run(module.handle_hymt2(WebSocket(), {}, Backend(), 10))
    assert "Previous translation" not in prompts[0]
    assert sent[-1]["committed_text"] == ""


@pytest.mark.parametrize("update_context", [False, True])
def test_local_asr_final_clears_transcript_context_but_keeps_hotwords(monkeypatch, update_context):
    monkeypatch.setattr(config, "LOCAL_QWEN_LOG_PIPELINE_TIMING", False)
    engine = object.__new__(Qwen3ASREngine)
    engine._context, engine._corpus_text, engine._draft_tokens = "previous transcript", "hotword", [9]
    engine.language = "en"
    engine._prompt_context = lambda: "explicit hint"
    engine._engine = SimpleNamespace(
        encoder=SimpleNamespace(encode=lambda audio: (np.zeros((1, 2)), 0.0)),
        _build_prompt_embd=lambda **kwargs: np.zeros((1, 2)),
    )
    engine._decode = lambda embd: dict(text="hello", tokens=[1], t_prefill=0.0, t_generate=0.0,
                                     n_prefill=1, n_passes=1, n_drafted=0, n_accepted=0, n_biased=0)
    assert engine.transcribe(np.zeros(16000), update_context=update_context)["text"] == "hello"
    assert engine.context == ("" if update_context else "previous transcript")
    assert engine._corpus_text == "hotword"
    assert engine._draft_tokens == ([] if update_context else [1])


def test_microbatch_bias_is_per_sequence_and_leaves_the_tail_free():
    engine = object.__new__(QwenMicroBatchEngine)
    engine.ctx = SimpleNamespace(ptr=None)
    engine.core = SimpleNamespace(model=SimpleNamespace(n_vocab=8))
    logits = np.full(8, -20.0, dtype=np.float32)
    logits[1], logits[2], logits[3] = 5.0, 4.5, 2.0
    engine.m = SimpleNamespace(llama_get_logits_ith=lambda ptr, idx: logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    states = [
        _Sequence(None, i, np.zeros((1, 2)), 0.0, 1, None,
                  DraftTracker([token, 1]), draft_bias=1.0, bias_tail=1)
        for i, token in enumerate([2, 3])
    ]
    assert [engine._sample(state, 0, 0) for state in states] == [2, 1]
    assert [state.biased for state in states] == [1, 0]
    assert engine._sample(states[0], 0, 1) == 1


def test_microbatch_final_clears_only_its_own_context_and_uses_greedy_defaults(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_QWEN_ASR_TEMPERATURE", 0.0)
    engine = object.__new__(QwenMicroBatchEngine)
    engine.max_sequences = 2
    engine.core = SimpleNamespace(
        encoder=SimpleNamespace(encode_batch=lambda audios: ([np.zeros((1, 2)) for _ in audios], 0.0)),
        model=SimpleNamespace(detokenize=lambda tokens: "hello"),
    )
    temperatures, logits_indices, removed = [], [], []

    def sampler(**kwargs):
        temperatures.append(kwargs["temperature"])
        return SimpleNamespace(sample=lambda ctx, idx: logits_indices.append(idx) or 1, free=lambda: None)

    engine.m = SimpleNamespace(LlamaSampler=sampler)
    engine.ctx = object()
    engine._remove_sequence = lambda *args: removed.append(args)
    engine._prompt_embeddings = lambda audio, **kwargs: np.zeros((2 if kwargs["context"] == "A" else 3, 2))
    engine._decode_prefill = lambda states: None

    def finish(states):
        for state in states:
            state.emitted = [state.current]
            state.done = True

    engine._decode_round = finish
    requests = [QwenBatchInput(np.zeros(16000), "en", "hotword", hint, [], final)
                for hint, final in [("A", True), ("B", False)]]
    results = engine.transcribe_batch(requests)
    assert temperatures == [0.0, 0.0]
    assert logits_indices == [1, 4]
    assert [result[1] for result in results] == ["", "B"]
    assert [result[2] for result in results] == [[], [1]]
    assert removed == [(0,), (1,), (0,), (1,)]


@pytest.mark.parametrize("previous,current,cut", [("12345", "1234", False), ("12345", "123", True), ("", "a", False)])
def test_cut_threshold_preserves_small_corrections(previous, current, cut):
    assert source_was_cut(previous, current) is cut
