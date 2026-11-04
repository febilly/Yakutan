import ctypes
from types import SimpleNamespace

import numpy as np

from local_inference.spec_decode import SpeculativeDecoder

VOCAB = 8
STOP = 7


def _decoder(preferred, rivals, margin):
    """A fake model: at generation step k it prefers ``preferred[k]``.

    ``rivals[k]`` (if any) scores ``margin`` below it; everything else far below.
    """
    state = {"pos0": None, "batch_pos": None, "batch_n": 0, "keep": []}

    def logits_for(step):
        values = np.full(VOCAB, -20.0, dtype=np.float32)
        best = preferred[step] if step < len(preferred) else STOP
        values[best] = 5.0
        rival = rivals.get(step)
        if rival is not None:
            values[rival] = 5.0 - margin
        array = (ctypes.c_float * VOCAB)(*values)
        state["keep"].append(array)
        return ctypes.cast(array, ctypes.POINTER(ctypes.c_float))

    def get_logits(_ctx_ptr, idx):
        if state["batch_pos"] is None:
            return logits_for(0)
        if idx < 0:
            idx = state["batch_n"] + idx
        return logits_for(state["batch_pos"] + idx - state["pos0"] + 1)

    decoder = object.__new__(SpeculativeDecoder)
    decoder._m = SimpleNamespace(
        llama_vocab_n_tokens=lambda _vocab: VOCAB, llama_get_logits_ith=get_logits
    )
    decoder._ctx = SimpleNamespace(ptr=None)
    decoder._model = SimpleNamespace(vocab=None)

    def decode_tokens(tokens, pos, **_kwargs):
        state["batch_pos"], state["batch_n"] = pos, len(tokens)
        return 0

    decoder.decode_tokens = decode_tokens
    decoder.decode_one = lambda token, pos, **kwargs: decode_tokens((token,), pos)
    decoder.seq_rm = lambda *_args: None
    return decoder, state


def _generate(preferred, rivals, draft, *, margin, bias, tail=0):
    decoder, state = _decoder(preferred, rivals, margin)
    state["pos0"] = 100
    sampler = SimpleNamespace(sample=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError))
    return decoder.generate(
        sampler, 100, stop_ids={STOP}, max_tokens=50, draft=draft,
        draft_bias=bias, bias_tail=tail,
    )


def test_bias_keeps_a_near_tie_draft_token():
    # The model now slightly prefers 3 at step 2; the previous pass said 4.
    result = _generate([1, 2, 3, 5], {2: 4}, [1, 2, 4, 5], margin=1.0, bias=2.0)
    assert result.tokens == [1, 2, 4, 5]
    assert result.n_biased == 1


def test_bias_yields_when_the_model_is_clearly_surer():
    result = _generate([1, 2, 3, 5], {2: 4}, [1, 2, 4, 5], margin=3.0, bias=2.0)
    assert result.tokens == [1, 2, 3, 5]
    assert result.n_biased == 0


def test_bias_leaves_the_draft_tail_free():
    result = _generate([1, 2, 3, 5], {2: 4}, [1, 2, 4, 5], margin=1.0, bias=2.0, tail=2)
    assert result.tokens == [1, 2, 3, 5]


def test_bias_does_not_stop_early_while_the_draft_continues():
    # The model would stop after two tokens, but the draft's third is close.
    result = _generate([1, 2], {2: 3}, [1, 2, 3], margin=0.5, bias=2.0)
    assert result.tokens == [1, 2, 3]


def test_without_a_draft_bias_decoding_is_plain_greedy():
    result = _generate([1, 2, 3, 5], {2: 4}, None, margin=0.1, bias=2.0)
    assert result.tokens == [1, 2, 3, 5]
