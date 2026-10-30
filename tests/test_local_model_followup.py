import asyncio
import json
import logging
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'remote_inference_service'))

from local_inference.mtmd_audio import _on_mtmd_log
from local_inference.repetition_guard import max_translation_tokens, RETRY_REPEAT_PENALTY
from local_inference.spec_decode import SpeculativeDecoder
from remote_inference_service import server, server_windows
from streaming_translation.api.hymt2 import HyMT2API, HyMT2LocalEngine


def test_mtmd_drops_info_but_keeps_warning_and_error(caplog):
    with caplog.at_level(logging.DEBUG, logger='local_inference.mtmd_audio'):
        _on_mtmd_log(2, b'per-encode INFO\n', None)
        _on_mtmd_log(3, b'encoder warning\n', None)
        _on_mtmd_log(4, b'encoder error\n', None)
    assert [(r.levelname, r.message) for r in caplog.records] == [
        ('WARNING', '[mtmd] encoder warning'), ('ERROR', '[mtmd] encoder error')]


@pytest.mark.parametrize('text,expected', [('', 160), ('short', 160), ('中' * 81, 162), ('长' * 300, 600)])
def test_cap_scales_with_source_length(text, expected):
    assert max_translation_tokens(text) == expected


def test_only_actual_special_tokens_may_end_translation():
    class Model:
        def tokenize(self, text, add_special=False, parse_special=True):
            ordinary = {'$': [3], '</s>': [12, 13, 14], '<|im_end|>': [5, 6, 7]}
            return ordinary.get(text, [120020] if parse_special else [7, 8, 9])
    engine = HyMT2LocalEngine.__new__(HyMT2LocalEngine)
    engine.model = Model()
    assert engine._special_token_id('$') == -1
    assert engine._special_token_id('</s>') == -1
    assert engine._special_token_id('<|im_end|>') == -1
    assert engine._special_token_id('<｜hy_place▁holder▁no▁2｜>') == 120020


def fake_local_engine(raw='Translation: literal <|im_end|> and $5 </s> intact'):
    engine = object.__new__(HyMT2LocalEngine)
    engine._engine_lock = threading.Lock()
    engine._cached_tokens = []
    engine._n_ctx = 10
    engine.stop_ids = {120020}
    engine.model = SimpleNamespace(tokenize=lambda *args, **kwargs: [1, 2, 3],
                                   detokenize=lambda tokens: raw, n_vocab=120030)
    engine._prefill = lambda tokens, cached: len(tokens)
    engine._LlamaSampler = lambda **kwargs: object()
    seen = []
    def generate(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(tokens=[4], n_passes=1, n_accepted=0, n_drafted=0)
    engine._spec = SimpleNamespace(generate=generate)
    return engine, seen


def test_generation_keeps_literal_markers_and_stays_inside_context():
    engine, seen = fake_local_engine()
    text, _ = engine.generate('prompt', max_tokens=600)
    assert text == 'literal <|im_end|> and $5 </s> intact'
    assert seen[0]['max_tokens'] == 7
    engine._n_ctx = 3
    with pytest.raises(ValueError, match='no context space'):
        engine.generate('prompt')
    assert len(seen) == 1


def test_real_fullwidth_special_marker_is_removed():
    engine, _ = fake_local_engine('译文<｜hy_place▁holder▁no▁2｜>ignored')
    assert engine.generate('prompt')[0] == '译文'


def test_draft_verification_obeys_remaining_output_budget():
    decoder = object.__new__(SpeculativeDecoder)
    decoder._ctx = object()
    calls = []
    decoder.decode_tokens = lambda tokens, pos, **kwargs: calls.append((tokens, pos)) or 0
    decoder.seq_rm = lambda *args: None
    next_token = iter(range(1, 100))
    sampler = SimpleNamespace(sample=lambda *args, **kwargs: next(next_token))
    result = decoder.generate(sampler, 100, stop_ids={999}, max_tokens=4, draft=list(range(1, 90)))
    assert result.tokens == [1, 2, 3, 4]
    assert calls == [([1, 2, 3, 4], 100)]
    assert decoder.generate(None, 0, stop_ids=set(), max_tokens=0).tokens == []


class RecordingEngine:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return next(self.outputs), [1]


def test_local_api_adds_bos_scales_cap_and_retries_runaway_once():
    api = HyMT2API(backend='local')
    engine = RecordingEngine(['Run, ' * 12, 'A complete translation.'])
    api._local_engine = engine
    source = '普普通通的较长句子。' * 4
    result = api.translate(source, 'zh', 'en', is_partial=True)
    assert result == 'A complete translation.'
    assert len(engine.calls) == 2
    assert engine.calls[0][0].startswith('<｜hy_begin▁of▁sentence｜><｜hy_User｜>')
    assert engine.calls[0][1]['max_tokens'] == max_translation_tokens(source)
    assert engine.calls[1][1]['repeat_penalty'] == RETRY_REPEAT_PENALTY


def test_source_repetition_does_not_trigger_retry():
    api = HyMT2API(backend='local')
    engine = RecordingEngine(['Ha, ' * 12])
    api._local_engine = engine
    assert api.translate('哈哈哈哈哈哈', 'zh', 'en')
    assert len(engine.calls) == 1 and engine.calls[0][1]['stop_on_runaway'] is False


@pytest.mark.parametrize('module', [server, server_windows])
def test_remote_translation_passes_adaptive_cap_to_both_attempts(module):
    class Backend:
        def __init__(self): self.calls = []
        async def complete(self, prompt, **kwargs):
            self.calls.append(kwargs)
            return 'Run, ' * 12 if len(self.calls) == 1 else '完整译文'
    backend = Backend()
    source = ''.join(str(i) + '词' for i in range(80))
    assert asyncio.run(module._translate(backend, 'prompt', source)) == '完整译文'
    assert backend.calls == [dict(max_new_tokens=2 * len(source)),
                             dict(repeat_penalty=RETRY_REPEAT_PENALTY, max_new_tokens=2 * len(source))]


@pytest.mark.parametrize('module', [server, server_windows])
def test_external_backend_keeps_bos_cap_and_retry_penalty(module, tmp_path, monkeypatch):
    bodies = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"content":"translation"}'
    def open_url(request, **kwargs):
        bodies.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr(module.urllib.request, 'urlopen', open_url)
    backend = (module.HyMT2Backend(tmp_path, 'http://localhost/completion', 160)
               if module is server_windows else module.HyMT2Backend('http://localhost/completion', 160))
    assert asyncio.run(backend.complete('prompt', repeat_penalty=1.15, max_new_tokens=600)) == 'translation'
    assert bodies[0]['n_predict'] == 600 and bodies[0]['repeat_penalty'] == 1.15
    assert bodies[0]['prompt'].startswith('<｜hy_begin▁of▁sentence｜>')
