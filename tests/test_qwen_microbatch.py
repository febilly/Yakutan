from local_inference.qwen_microbatch import QwenDynamicBatchScheduler


class _Engine:
    max_sequences = 16


def test_adaptive_batch_target_grows_with_backlog():
    scheduler = QwenDynamicBatchScheduler(_Engine(), max_wait_ms=80, min_wait_ms=8)

    assert scheduler._target(0) == (2, 0.008)
    assert scheduler._target(2) == (4, 0.025)
    assert scheduler._target(6) == (8, 0.05)
    assert scheduler._target(7) == (16, 0.08)
