"""Draft anchoring and encoder-placement rules.

The decode loop itself needs a loaded model, so it is exercised by the benchmarks rather than here.
What is unit-testable — and what would silently cost speed or correctness if it broke — is the anchoring
logic that decides which slice of the previous output to verify, and the settings that decide where each
of the two Qwen3-ASR models runs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from local_inference.spec_decode import (  # noqa: E402
    DraftTracker,
    MAX_DRAFT_CHUNK,
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


class TestEncoderShapeBuckets:
    """DirectML needs a fixed input shape; padding every clip to the largest one wastes most of the win
    on short utterances, and dropping below a clip's real length falls off the fast path entirely."""

    def _engine(self, on_gpu, full_target=390):
        from local_inference.asr_qwen3 import Qwen3ASREngine

        engine = Qwen3ASREngine.__new__(Qwen3ASREngine)
        engine._encoder_on_gpu = on_gpu
        engine._encoder_full_target = full_target

        class _Enc:
            h_target_len = full_target

        class _Inner:
            encoder = _Enc()

        engine._engine = _Inner()
        return engine

    @pytest.mark.parametrize("seconds,expected", [
        (1.0, 5 * 13), (5.0, 5 * 13),
        (5.1, 10 * 13), (10.0, 10 * 13),
        (12.0, 20 * 13), (20.0, 20 * 13),
        (24.0, 30 * 13),
    ])
    def test_picks_the_smallest_bucket_that_fits(self, seconds, expected):
        engine = self._engine(on_gpu=True)
        engine._apply_encoder_shape_bucket(int(seconds * 16000))
        assert engine._engine.encoder.h_target_len == expected

    def test_longer_than_every_bucket_keeps_the_original_target(self):
        engine = self._engine(on_gpu=True)
        engine._apply_encoder_shape_bucket(int(45 * 16000))
        assert engine._engine.encoder.h_target_len == 390

    def test_never_exceeds_the_configured_target(self):
        # a small pad_to must not be inflated by a larger bucket - that would leave the fixed-shape path
        engine = self._engine(on_gpu=True, full_target=104)
        engine._apply_encoder_shape_bucket(int(4 * 16000))
        assert engine._engine.encoder.h_target_len <= 104

    def test_cpu_encoder_is_left_alone(self):
        engine = self._engine(on_gpu=False)
        engine._apply_encoder_shape_bucket(int(2 * 16000))
        assert engine._engine.encoder.h_target_len == 390
