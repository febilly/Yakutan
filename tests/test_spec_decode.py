"""Draft anchoring and encoder-placement rules.

The decode loop itself needs a loaded model, so it is exercised by the benchmarks rather than here.
What is unit-testable — and what would silently cost speed or correctness if it broke — is the anchoring
logic that decides which slice of the previous output to verify, and the settings that decide where each
of the two Qwen3-ASR models runs.
"""

import ctypes
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from local_inference.spec_decode import (  # noqa: E402
    BATCH_CAPACITY,
    DraftTracker,
    MAX_DRAFT_CHUNK,
    SpeculativeDecoder,
    common_prefix_len,
)


class TestCommonPrefixLen:
    def test_identical(self):
        assert common_prefix_len([1, 2, 3], [1, 2, 3]) == 3

    def test_partial(self):
        assert common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2

    def test_no_overlap(self):
        assert common_prefix_len([1, 2], [9, 2]) == 0

    def test_shorter_is_the_bound(self):
        assert common_prefix_len([1, 2], [1, 2, 3, 4]) == 2

    def test_empty(self):
        assert common_prefix_len([], [1, 2]) == 0


class TestDraftTracker:
    def test_no_draft_yields_nothing(self):
        tracker = DraftTracker(None)
        assert not tracker
        assert tracker.chunk_for(1, 0) == []

    def test_first_token_anchors_at_the_start(self):
        tracker = DraftTracker([10, 11, 12, 13])
        assert tracker.chunk_for(10, 0) == [11, 12, 13]

    def test_walks_the_draft_as_generation_advances(self):
        tracker = DraftTracker([10, 11, 12, 13])
        assert tracker.chunk_for(10, 0) == [11, 12, 13]
        assert tracker.chunk_for(11, 1) == [12, 13]
        assert tracker.chunk_for(12, 2) == [13]
        assert tracker.chunk_for(13, 3) == []  # nothing left to verify

    def test_a_token_absent_from_the_draft_yields_no_chunk(self):
        tracker = DraftTracker([10, 11, 12, 13])
        tracker.chunk_for(10, 0)
        assert tracker.chunk_for(999, 1) == []

    def test_realigns_after_a_substitution(self):
        # draft ... 10 11 12 13 14, output substitutes 11 -> 99 and then rejoins at 12
        tracker = DraftTracker([10, 11, 12, 13, 14])
        assert tracker.chunk_for(10, 0) == [11, 12, 13, 14]
        assert tracker.chunk_for(99, 1) == []      # 99 is nowhere in the draft
        assert tracker.chunk_for(12, 2) == [13, 14]  # picked back up
        assert tracker.n_realigned == 1

    def test_realigns_across_a_deletion(self):
        # the new output skips 11 and 12 entirely
        tracker = DraftTracker([10, 11, 12, 13, 14])
        tracker.chunk_for(10, 0)
        assert tracker.chunk_for(13, 1) == [14]

    def test_realigns_across_an_insertion(self):
        # the new output repeats a token, so the draft index lags behind the generation index
        tracker = DraftTracker([10, 11, 12, 13, 14])
        tracker.chunk_for(10, 0)
        tracker.chunk_for(11, 1)
        assert tracker.chunk_for(11, 2) == [12, 13, 14]

    def test_window_zero_only_accepts_the_exact_continuation(self):
        tracker = DraftTracker([10, 11, 12, 13, 14], window=0)
        tracker.chunk_for(10, 0)
        assert tracker.chunk_for(99, 1) == []
        # without a window there is no search, so a token further along is not found
        assert tracker.chunk_for(13, 2) == []

    def test_chunk_is_capped(self):
        draft = list(range(1000, 1000 + 200))
        tracker = DraftTracker(draft)
        chunk = tracker.chunk_for(1000, 0)
        assert len(chunk) == MAX_DRAFT_CHUNK

    def test_long_draft_is_walked_in_chunks(self):
        draft = list(range(1000, 1000 + 200))
        tracker = DraftTracker(draft)
        first = tracker.chunk_for(1000, 0)
        # continuing from the last token of the first chunk yields the next slice
        second = tracker.chunk_for(first[-1], len(first))
        assert second[0] == first[-1] + 1
        assert len(second) == MAX_DRAFT_CHUNK

    def test_counts_drafted_tokens(self):
        tracker = DraftTracker([10, 11, 12])
        tracker.chunk_for(10, 0)
        tracker.chunk_for(11, 1)
        assert tracker.n_drafted == 3  # [11,12] then [12]


class TestLocalEngineDisposalDropsEveryReference:
    """卸载靠的是引用计数归零后 __del__ 里的 llama_free/llama_model_free。

    只要还有任何一个对象攥着 ctx 或 model，显存就要留到进程退出——推测解码器就踩过这个坑。
    这里用 weakref 直接断言"没人再攥着"，比断言某个字段被置 None 更难绕过。
    """

    def test_dispose_releases_ctx_and_model(self):
        import gc
        import threading
        import weakref

        from streaming_translation.api.hymt2 import HyMT2LocalEngine, _dispose_local_engine

        class Ctx:
            ptr = 1

        class Model:
            pass

        class FakeSpec:
            """和真的 SpeculativeDecoder 一样，抓着 ctx 和 model 不放。"""

            def __init__(self, ctx, model):
                self._ctx = ctx
                self._model = model

        engine = HyMT2LocalEngine.__new__(HyMT2LocalEngine)
        ctx, model = Ctx(), Model()
        engine.ctx = ctx
        engine.model = model
        engine._spec = FakeSpec(ctx, model)
        engine._cached_tokens = [1, 2, 3]
        engine._engine_lock = threading.Lock()

        ctx_ref, model_ref = weakref.ref(ctx), weakref.ref(model)
        del ctx, model

        _dispose_local_engine(engine)
        gc.collect()

        assert ctx_ref() is None, "上下文仍被引用，llama_free 不会执行，显存不会归还"
        assert model_ref() is None, "模型仍被引用，llama_model_free 不会执行"
        assert engine._cached_tokens == []


class TestFillPositions:
    """位置缓冲的内存布局。

    多平面 RoPE 下 llama.cpp 按 plane-major 读 ``n * planes`` 个位置，写错了不会报错，
    只会安静地把 RoPE 算歪。窄批走直写、宽批走 memmove，两条分支必须给出同一块内存。
    """

    class _FakeBatch:
        def __init__(self, capacity: int) -> None:
            self.pos = (ctypes.c_int32 * capacity)()

    @staticmethod
    def _reference(pos0: int, n: int, planes: int) -> list[int]:
        base = np.arange(pos0, pos0 + n, dtype=np.int32)
        if planes <= 1:
            return list(base)
        return list(np.concatenate([base] * (planes - 1) + [np.zeros(n, dtype=np.int32)]))

    @pytest.mark.parametrize("planes", [1, 2, 4])
    @pytest.mark.parametrize("n", [1, 2, 24, 63, 64, 65, 200])
    def test_matches_reference_layout(self, n, planes):
        width = n * max(planes, 1)
        batch = self._FakeBatch(width + 8)
        SpeculativeDecoder._fill_positions(batch, 137, n, planes)
        assert list(batch.pos[:width]) == self._reference(137, n, planes)

    def test_both_branches_are_exercised(self):
        """参数里得同时有落在直写和 memmove 两侧的宽度，否则这个测试是摆设。"""
        widths = {n * p for n, p in itertools.product([1, 2, 24, 63, 64, 65, 200], [1, 2, 4])}
        assert any(w <= BATCH_CAPACITY for w in widths)
        assert any(w > BATCH_CAPACITY for w in widths)

    def test_single_token_writes_every_plane(self):
        """无草稿时每个 token 都走这条路；vendor 的单 token 构造只写了 pos[0]。"""
        batch = self._FakeBatch(16)
        for i in range(16):
            batch.pos[i] = -1
        SpeculativeDecoder._fill_positions(batch, 42, 1, 4)
        assert list(batch.pos[:4]) == [42, 42, 42, 0]


class TestEncoderDeviceSetting:
    @pytest.mark.parametrize("value,expected", [
        ("auto", "auto"), ("cpu", "cpu"), ("gpu", "gpu"),
        ("GPU", "gpu"), ("  cpu  ", "cpu"),
        ("", "auto"), (None, "auto"), ("vulkan:0", "auto"), (123, "auto"),
    ])
    def test_sanitize(self, value, expected):
        assert config.sanitize_qwen_encoder_device(value) == expected

    @pytest.mark.parametrize("choice,decoder_device,expected", [
        # 'auto' follows the decoder
        ("auto", "cpu", False),
        ("auto", "auto", True),
        ("auto", "vulkan:0", True),
        # explicit wins over the decoder either way
        ("cpu", "vulkan:0", False),
        ("gpu", "cpu", True),
    ])
    def test_placement(self, choice, decoder_device, expected, monkeypatch):
        from local_inference.asr_qwen3 import Qwen3ASREngine

        monkeypatch.setattr(config, "LOCAL_QWEN_ENCODER_DEVICE", choice, raising=False)
        assert Qwen3ASREngine._want_encoder_on_gpu(decoder_device) is expected
