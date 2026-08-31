"""True dynamic batching for Qwen3-ASR's embedding-input llama.cpp path.

The desktop engine needs one logical KV history per speaker, but a server does
not need one *llama context* per speaker.  This module keeps all active
utterances as sequence IDs in a single context and combines their prefill and
speculative decode passes into one ``llama_decode`` batch.  The model weights
therefore remain one resident allocation and one batched GEMM serves all ready
users, instead of launching one bandwidth-bound GEMM for every WebSocket.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from .asr_qwen3 import Qwen3ASREngine
from .spec_decode import DraftTracker

logger = logging.getLogger(__name__)


@dataclass
class QwenBatchInput:
    audio: np.ndarray
    language: str
    corpus_text: str
    context: str
    draft_tokens: list[int]
    update_context: bool


@dataclass
class _Sequence:
    request: QwenBatchInput
    slot: int
    full_embd: np.ndarray
    encoder_ms: float
    position: int
    sampler: Any
    tracker: DraftTracker
    current: int | None = None
    emitted: list[int] | None = None
    done: bool = False
    aborted: bool = False
    passes: int = 0
    drafted: int = 0
    accepted: int = 0

    def __post_init__(self) -> None:
        self.emitted = []


class QwenMicroBatchEngine:
    """One Qwen model/context, multiple independent llama sequence IDs."""

    def __init__(self, base: Qwen3ASREngine, *, max_sequences: int, n_ctx_per_sequence: int) -> None:
        self.base = base
        self.core = base._engine
        if self.core is None:
            raise RuntimeError("Qwen base engine is unavailable")
        self.m = self.core.llama_mod
        self.max_sequences = max(1, int(max_sequences))
        self.n_ctx_per_sequence = max(256, int(n_ctx_per_sequence))
        # n_ctx is the aggregate KV budget; n_seq_max enables distinct sequence
        # IDs within that one context.  Model weights remain in ``core.model``.
        self.ctx = self.m.LlamaContext(
            self.core.model,
            n_ctx=self.n_ctx_per_sequence * self.max_sequences,
            n_batch=8192,
            n_ubatch=512,
            n_seq_max=self.max_sequences,
            embeddings=False,
        )
        seq_rm = self.m.llama.llama_memory_seq_rm
        seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        seq_rm.restype = ctypes.c_bool
        self._seq_rm = seq_rm
        self._memory = self.m.llama_get_memory(self.ctx.ptr)

    def _remove_sequence(self, slot: int, p0: int = 0) -> None:
        self._seq_rm(self._memory, int(slot), int(p0), -1)

    def _prompt_embeddings(
        self,
        audio_embd: np.ndarray,
        *,
        language: str,
        corpus_text: str,
        context: str,
    ) -> np.ndarray:
        # Reuse the desktop prompt/tokenizer rules without reusing its context.
        self.base.set_language(language)
        self.base.set_corpus_text(corpus_text)
        self.base.set_context(context)
        qwen_language = self.base.language
        from .asr_qwen3 import _LANG_MAP
        return self.core._build_prompt_embd(
            audio_embd=audio_embd,
            prefix_text="",
            context=self.base._prompt_context(),
            language=_LANG_MAP.get(qwen_language) if qwen_language else None,
        )

    def _decode_prefill(self, states: list[_Sequence]) -> None:
        total = sum(int(state.full_embd.shape[0]) for state in states)
        batch = self.m.LlamaBatch(max(total * 4, 8192), self.core.model.n_embd, self.max_sequences)
        try:
            joined = np.ascontiguousarray(np.concatenate([state.full_embd for state in states], axis=0))
            positions = np.concatenate([
                np.concatenate([np.arange(state.full_embd.shape[0], dtype=np.int32) for state in states]),
                np.concatenate([np.arange(state.full_embd.shape[0], dtype=np.int32) for state in states]),
                np.concatenate([np.arange(state.full_embd.shape[0], dtype=np.int32) for state in states]),
                np.zeros(total, dtype=np.int32),
            ])
            batch.set_embd(joined, pos=positions)
            offset = 0
            for state in states:
                length = int(state.full_embd.shape[0])
                for index in range(length):
                    batch.seq_id[offset + index][0] = state.slot
                    batch.logits[offset + index] = 1 if index == length - 1 else 0
                offset += length
            if self.ctx.decode(batch) != 0:
                raise RuntimeError("llama.cpp batched Qwen prefill failed")
        finally:
            del batch

    def _decode_round(self, states: list[_Sequence]) -> None:
        """Run one speculative decode round for every currently active sequence."""
        active: list[tuple[_Sequence, list[int]]] = []
        for state in states:
            if state.done or state.current is None:
                continue
            if state.current in {self.core.model.eos_token, self.core.ID_IM_END}:
                state.done = True
                continue
            state.emitted.append(state.current)
            if len(state.emitted) > 15 and len(set(state.emitted[-15:])) <= 3:
                state.aborted = state.done = True
                continue
            if len(state.emitted) >= 512:
                state.done = True
                continue
            draft = state.tracker.chunk_for(state.current, len(state.emitted) - 1)
            active.append((state, [state.current, *draft]))

        if not active:
            return
        total = sum(len(tokens) for _, tokens in active)
        batch = self.m.LlamaBatch(max(total * 4, 256), 0, self.max_sequences)
        try:
            all_positions: list[int] = []
            offset = 0
            logit_offsets: dict[int, int] = {}
            for state, tokens in active:
                logit_offsets[state.slot] = offset
                for index, token in enumerate(tokens):
                    batch.token[offset + index] = int(token)
                    batch.n_seq_id[offset + index] = 1
                    batch.seq_id[offset + index][0] = state.slot
                    batch.logits[offset + index] = 1
                    all_positions.append(state.position + index)
                offset += len(tokens)
            positions = np.asarray(all_positions * 3 + [0] * total, dtype=np.int32)
            ctypes.memmove(batch.pos, positions.ctypes.data, positions.nbytes)
            batch.n_tokens = total
            if self.ctx.decode(batch) != 0:
                raise RuntimeError("llama.cpp batched Qwen decode failed")

            for state, tokens in active:
                start = logit_offsets[state.slot]
                draft = tokens[1:]
                accepted = 0
                next_token: int | None = None
                for index in range(len(tokens)):
                    sampled = int(state.sampler.sample(self.ctx, idx=start + index))
                    if index == len(draft) or sampled != draft[index]:
                        next_token = sampled
                        break
                    accepted += 1
                    if sampled in {self.core.model.eos_token, self.core.ID_IM_END}:
                        state.done = True
                        break
                    state.emitted.append(sampled)
                    if len(state.emitted) >= 512:
                        state.done = True
                        break
                state.passes += 1
                state.accepted += accepted
                state.position += accepted + 1
                self._remove_sequence(state.slot, state.position)
                state.current = next_token
                if next_token is None:
                    state.done = True
        finally:
            del batch

    def transcribe_batch(self, requests: list[QwenBatchInput]) -> list[tuple[dict[str, Any] | None, str, list[int], dict[str, float | int]]]:
        if not requests:
            return []
        if len(requests) > self.max_sequences:
            raise ValueError("Qwen microbatch exceeds configured sequence count")
        t_batch_start = time.perf_counter()
        # This is a real ONNX batch.  The audio side no longer issues one GPU
        # encoder execution per WebSocket request.
        embeddings, encoder_seconds = self.core.encoder.encode_batch([request.audio for request in requests])
        states: list[_Sequence] = []
        try:
            for slot, (request, audio_embd) in enumerate(zip(requests, embeddings)):
                self._remove_sequence(slot)
                full = self._prompt_embeddings(
                    audio_embd,
                    language=request.language,
                    corpus_text=request.corpus_text,
                    context=request.context,
                )
                sampler = self.m.LlamaSampler(temperature=0.4, seed=int(np.random.randint(0, 2**31 - 1)))
                states.append(_Sequence(
                    request=request,
                    slot=slot,
                    full_embd=full,
                    encoder_ms=encoder_seconds * 1000 / len(requests),
                    position=int(full.shape[0]),
                    sampler=sampler,
                    tracker=DraftTracker(request.draft_tokens if getattr(config, "LOCAL_SPECULATIVE_DECODE", True) else None),
                ))
            t_prefill = time.perf_counter()
            self._decode_prefill(states)
            prefill_ms = (time.perf_counter() - t_prefill) * 1000
            # llama_get_logits_ith indexes the *batch token position*, not the
            # ordinal among tokens whose logits flag is set.
            prefill_offsets = np.cumsum([0] + [int(state.full_embd.shape[0]) for state in states])
            for index, state in enumerate(states):
                state.current = int(state.sampler.sample(self.ctx, idx=int(prefill_offsets[index + 1] - 1)))
            while any(not state.done for state in states):
                self._decode_round(states)

            results: list[tuple[dict[str, Any] | None, str, list[int], dict[str, float | int]]] = []
            batch_ms = (time.perf_counter() - t_batch_start) * 1000
            for state in states:
                text = self.core.model.detokenize(state.emitted).strip() if state.emitted else ""
                context = state.request.context
                if text and state.request.update_context:
                    self.base.set_context(context)
                    context = self.base._truncate_context_to_tokens(context + text)
                draft = [] if state.request.update_context or not text else list(state.emitted)
                result = None if not text else {
                    "text": text,
                    "language": state.request.language if state.request.language != "auto" else self.base._guess_language(text),
                    "language_name": state.request.language if state.request.language != "auto" else self.base._guess_language(text),
                }
                metrics: dict[str, float | int] = {
                    "encoder_ms": round(state.encoder_ms, 3),
                    "prefill_ms": round(prefill_ms, 3),
                    "batch_run_ms": round(batch_ms, 3),
                    "batch_size": len(states),
                    "n_prefill": int(state.full_embd.shape[0]),
                    "n_passes": state.passes,
                    "n_drafted": state.tracker.n_drafted,
                    "n_accepted": state.accepted,
                }
                results.append((result, context, draft, metrics))
            return results
        finally:
            for state in states:
                self._remove_sequence(state.slot)
                try:
                    state.sampler.free()
                except Exception:
                    pass


@dataclass
class _QueuedRequest:
    payload: QwenBatchInput
    future: asyncio.Future
    enqueued_at: float


class QwenDynamicBatchScheduler:
    """Adaptive queue that trades a bounded wait for a larger GPU microbatch."""

    def __init__(self, engine: QwenMicroBatchEngine, *, max_wait_ms: float = 80.0, min_wait_ms: float = 8.0) -> None:
        self.engine = engine
        self.max_wait_ms = max(0.0, float(max_wait_ms))
        self.min_wait_ms = max(0.0, min(float(min_wait_ms), self.max_wait_ms))
        self._queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._serve(), name="qwen-microbatch")

    async def submit(self, payload: QwenBatchInput) -> tuple[dict[str, Any] | None, str, list[int], dict[str, float | int]]:
        self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(_QueuedRequest(payload, future, loop.time()))
        return await future

    def _target(self, depth: int) -> tuple[int, float]:
        """Choose a batch size/deadline from observed backlog.

        More queued work raises the *target* batch first.  The scheduler does
        not blindly sleep longer under load: once that target is already in the
        queue it launches immediately.  Waiting is only used to fill the last
        few positions before the oldest request consumes its latency budget.
        """
        capacity = self.engine.max_sequences
        total_ready = depth + 1  # include the request already dequeued
        if total_ready <= 1:
            return min(2, capacity), self.min_wait_ms / 1000
        if total_ready <= 3:
            return min(4, capacity), min(25.0, self.max_wait_ms) / 1000
        if total_ready <= 7:
            return min(8, capacity), min(50.0, self.max_wait_ms) / 1000
        return capacity, self.max_wait_ms / 1000

    async def _serve(self) -> None:
        while True:
            first = await self._queue.get()
            items = [first]
            target, budget = self._target(self._queue.qsize())
            deadline = asyncio.get_running_loop().time() + budget
            # Consume arrivals until the workload-informed target is full or
            # the oldest request reaches its bounded batching deadline.
            while len(items) < target:
                try:
                    items.append(self._queue.get_nowait())
                    target, _ = self._target(self._queue.qsize() + len(items) - 1)
                    target = max(target, len(items))
                except asyncio.QueueEmpty:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        items.append(await asyncio.wait_for(self._queue.get(), remaining))
                        target, _ = self._target(self._queue.qsize() + len(items) - 1)
                        target = max(target, len(items))
                    except asyncio.TimeoutError:
                        break
            # If a high-load queue reached a larger target during collection,
            # drain it immediately up to the configured sequence capacity.
            while len(items) < self.engine.max_sequences:
                try:
                    items.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                started_at = asyncio.get_running_loop().time()
                values = await asyncio.to_thread(
                    self.engine.transcribe_batch, [item.payload for item in items]
                )
                for item, value in zip(items, values):
                    result, context, draft, metrics = value
                    metrics["queue_ms"] = round((started_at - item.enqueued_at) * 1000, 3)
                    metrics["run_ms"] = float(metrics["batch_run_ms"])
                    metrics["workers"] = self.engine.max_sequences
                    if not item.future.done():
                        item.future.set_result((result, context, draft, metrics))
            except Exception as exc:
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)
