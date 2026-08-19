"""Speculative decoding over the ctypes llama.cpp binding, shared by local ASR and local translation.

Both streaming paths re-run the same model over an input that grew a little: the audio buffer gained a
second, or the source sentence gained a few characters. The previous output is therefore a strong
prediction of the next one. Submitting it alongside the freshly sampled token lets a single forward pass
commit every token the model agrees with, instead of one token per pass.

Verification is exact — each drafted position is sampled normally and kept only if it matches — so
*which* tokens come out never changes, only how many forward passes it takes to produce them.

Measured on this repo's models (RTX 3080, Vulkan): 2.3x on the Qwen3-ASR token loop across the partials
of one utterance, and 1.1x-2.3x on Hy-MT2 depending on utterance length. The win grows with length,
because a long line is re-emitted almost unchanged on every update.

Nothing here touches the vendored glue in ``vendor/qwen_asr_gguf``; ``llama_memory_seq_rm`` is bound
locally because that module does not expose it (it is stable llama.cpp API, present in the shipped DLL).
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# The draft chunk shares one batch with the sampled token. 64 positions is what llama.cpp reserves per
# sequence for speculation, so 63 is the most we can append without growing the batch.
MAX_DRAFT_CHUNK = 63

# Realignment window: look for the emitted token at the expected draft index, then outward. Costs a
# handful of integer compares and lets the draft survive a correction in the middle of an utterance.
# Worth ~0-5% here (streaming revisions are mostly appends, so there is rarely anything to realign);
# kept because it is free on the trajectories where it does not help.
DEFAULT_WINDOW = 8


class DraftTracker:
    """Tracks where the model's output currently sits inside the draft.

    ``chunk_for`` is called with each freshly emitted token. It anchors that token in the draft and
    returns the draft's continuation, which is what the next forward pass will verify. An anchor at the
    expected index is the ordinary case; anchoring nearby is a realignment after a divergence, and covers
    substitution, insertion and deletion alike.
    """

    __slots__ = ("_draft", "_window", "_max_chunk", "_pos", "_gen_idx", "n_drafted", "n_realigned",
                 "_missed")

    def __init__(self, draft: Optional[Sequence[int]], *, window: int = DEFAULT_WINDOW,
                 max_chunk: int = MAX_DRAFT_CHUNK) -> None:
        self._draft = list(draft or ())
        self._window = max(0, int(window))
        self._max_chunk = max(1, min(int(max_chunk), MAX_DRAFT_CHUNK))
        self._pos = -1
        self._gen_idx = -1
        self._missed = False
        self.n_drafted = 0
        self.n_realigned = 0

    def __bool__(self) -> bool:
        return bool(self._draft)

    def chunk_for(self, token: int, gen_idx: int) -> list[int]:
        draft = self._draft
        if not draft:
            return []

        n = len(draft)
        expected = self._pos + (gen_idx - self._gen_idx)
        w_fwd = self._window
        w_back = self._window // 2

        found = -1
        for d in range(max(w_fwd, w_back) + 1):
            ahead = expected + d
            if d <= w_fwd and 0 <= ahead < n and draft[ahead] == token:
                found = ahead
                break
            behind = expected - d
            if 0 < d <= w_back and 0 <= behind < n and draft[behind] == token:
                found = behind
                break

        if found < 0:
            self._missed = True
            return []

        if self._missed:
            self.n_realigned += 1
        self._missed = False
        self._pos = found
        self._gen_idx = gen_idx

        chunk = draft[found + 1: found + 1 + self._max_chunk]
        self.n_drafted += len(chunk)
        return chunk


@dataclass
class GenerationResult:
    tokens: list[int] = field(default_factory=list)
    aborted: bool = False
    n_passes: int = 0
    n_drafted: int = 0
    n_accepted: int = 0
    n_realigned: int = 0

    @property
    def tokens_per_pass(self) -> float:
        return len(self.tokens) / self.n_passes if self.n_passes else 0.0


class SpeculativeDecoder:
    """Thin decode layer over a ``LlamaContext``: batched prefill, draft verification, KV rollback."""

    def __init__(self, llama_mod, ctx, model) -> None:
        self._m = llama_mod
        self._ctx = ctx
        self._model = model

        seq_rm = llama_mod.llama.llama_memory_seq_rm
        seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        seq_rm.restype = ctypes.c_bool
        self._seq_rm = seq_rm
        self._mem = llama_mod.llama_get_memory(ctx.ptr)

    # -- memory -----------------------------------------------------------------

    def seq_rm(self, p0: int, p1: int = -1) -> None:
        """Drop KV entries in [p0, p1); p1 < 0 means "to the end"."""
        self._seq_rm(self._mem, 0, int(p0), int(p1))

    def clear(self) -> None:
        self._ctx.clear_kv_cache()

    # -- decode -----------------------------------------------------------------

    @staticmethod
    def _fill_positions(batch, pos0: int, n: int, planes: int) -> None:
        """Write positions for ``n`` tokens starting at ``pos0``.

        With multi-plane RoPE llama.cpp reads ``n * planes`` positions laid out plane-major. The vendored
        prompt builder writes four planes (three copies of the index, then zeros); mirror that so the
        generated tokens are positioned exactly like the prompt they continue.
        """
        base = np.arange(pos0, pos0 + n, dtype=np.int32)
        if planes <= 1:
            arr = base
        else:
            arr = np.concatenate([base] * (planes - 1) + [np.zeros(n, dtype=np.int32)])
        ctypes.memmove(batch.pos, arr.ctypes.data, arr.nbytes)

    def decode_tokens(self, tokens: Sequence[int], pos0: int, *, planes: int = 1,
                      logits: str = "all") -> int:
        """Decode a token batch. ``logits='all'`` marks every position (needed for verification)."""
        n = len(tokens)
        if n == 0:
            return 0
        batch = self._m.LlamaBatch(max(n * planes, n), 0, 1)
        try:
            want_all = logits == "all"
            for i, tok in enumerate(tokens):
                batch.token[i] = int(tok)
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1 if (want_all or i == n - 1) else 0
            self._fill_positions(batch, pos0, n, planes)
            batch.n_tokens = n
            return self._ctx.decode(batch)
        finally:
            del batch

    def decode_embd(self, embd: np.ndarray, pos0: int = 0, *, planes: int = 4) -> int:
        """Decode an embedding prompt (the audio path); logits only on the last position."""
        n = int(embd.shape[0])
        batch = self._m.LlamaBatch(max(n * planes, 8192), self._model.n_embd, 1)
        try:
            base = np.arange(pos0, pos0 + n, dtype=np.int32)
            if planes <= 1:
                arr = base
            else:
                arr = np.concatenate([base] * (planes - 1) + [np.zeros(n, dtype=np.int32)])
            batch.set_embd(embd, pos=arr)
            return self._ctx.decode(batch)
        finally:
            del batch

    def decode_one(self, token: int, pos: int) -> int:
        return self._ctx.decode_token(int(token), pos=int(pos))

    # -- generation --------------------------------------------------------------

    def generate(
        self,
        sampler,
        pos: int,
        *,
        stop_ids: set,
        max_tokens: int,
        draft: Optional[Sequence[int]] = None,
        planes: int = 1,
        window: int = DEFAULT_WINDOW,
        on_token: Optional[Callable[[list[int]], bool]] = None,
    ) -> GenerationResult:
        """Generate from an already-prefilled context.

        ``pos`` is the first free position. ``on_token`` is called with the emitted tokens after each one
        and may return True to abort (the ASR loop uses it for its repetition circuit breaker).
        """
        res = GenerationResult()
        tracker = DraftTracker(draft, window=window)
        emitted = res.tokens
        pos0 = pos

        sampled = sampler.sample(self._ctx, idx=-1)

        while True:
            if sampled in stop_ids:
                break
            emitted.append(sampled)
            if on_token is not None and on_token(emitted):
                res.aborted = True
                break
            if len(emitted) >= max_tokens:
                break

            chunk = tracker.chunk_for(sampled, len(emitted) - 1)

            if not chunk:
                if self.decode_one(sampled, pos) != 0:
                    break
                pos += 1
                res.n_passes += 1
                sampled = sampler.sample(self._ctx, idx=-1)
                continue

            if self.decode_tokens([sampled, *chunk], pos, planes=planes) != 0:
                break
            res.n_passes += 1

            n_acc = 0
            nxt = None
            finished = False
            for i in range(len(chunk) + 1):
                tok = sampler.sample(self._ctx, idx=i)
                if i == len(chunk) or tok != chunk[i]:
                    nxt = tok
                    break
                # the model agrees with the draft here - commit without another forward pass
                n_acc += 1
                if tok in stop_ids:
                    finished = True
                    break
                emitted.append(tok)
                if on_token is not None and on_token(emitted):
                    res.aborted = True
                    finished = True
                    break
                if len(emitted) >= max_tokens:
                    finished = True
                    break

            res.n_accepted += n_acc
            pos += n_acc + 1
            self.seq_rm(pos, -1)

            if finished or nxt is None:
                break
            sampled = nxt

        # leave the cache holding exactly prompt + emitted, so a caller that reuses the prefix can trust
        # its own bookkeeping (a rejected draft tail or an unemitted stop token would otherwise linger)
        self.seq_rm(pos0 + len(emitted), -1)

        res.n_drafted = tracker.n_drafted
        res.n_realigned = tracker.n_realigned
        return res


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the shared leading run of two token sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
