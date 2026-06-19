"""Tests for OSC-only text processing and VRChat length limits."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from osc_manager import OSCManager
from text_processor import (
    ARABIC_OSC_LINE_MAX_CHARS,
    ARABIC_PDI,
    ARABIC_RLI,
    RTL_OSC_MAX_LINE_BREAKS,
    build_tagged_translation_display,
)


class _FakeIpcClient:
    def __init__(self):
        self.sent = None

    def is_connected(self):
        return True

    def is_delegate_osc_enabled(self):
        return True

    async def send_message(self, text, ongoing):
        self.sent = (text, ongoing)


def test_truncate_preserves_direction_isolate_and_counts_controls():
    manager = OSCManager(truncate_messages=True)
    text = f"{ARABIC_RLI}{'x' * 200}{ARABIC_PDI}"

    result = manager._truncate_text(text, max_length=144)

    assert len(result) == 144
    assert result.startswith(ARABIC_RLI)
    assert result.endswith(ARABIC_PDI)
    assert len(result[1:-1]) == 142


def test_ipc_delegate_path_truncates_after_osc_post_processing():
    manager = OSCManager(truncate_messages=True)
    fake_ipc = _FakeIpcClient()
    wrapped_text = f"{ARABIC_RLI}{'x' * 200}{ARABIC_PDI}"

    manager.set_ipc_client(fake_ipc)
    try:
        with patch("osc_manager.apply_arabic_reshaper_if_needed", return_value=wrapped_text):
            asyncio.run(manager.send_text("raw", ongoing=False))
    finally:
        manager.clear_ipc_client()

    sent_text, ongoing = fake_ipc.sent
    assert ongoing is False
    assert len(sent_text) == 144
    assert sent_text.startswith(ARABIC_RLI)
    assert sent_text.endswith(ARABIC_PDI)


def test_truncate_plain_text_drops_front_within_limit():
    manager = OSCManager(truncate_messages=True)
    text = "前面这段旧内容应当被丢弃。" + "话" * 200

    result = manager._truncate_text(text, max_length=144)

    assert len(result) <= 144
    assert "前面这段旧内容应当被丢弃" not in result


def test_truncate_disabled_returns_untouched_text():
    OSCManager(truncate_messages=False)
    try:
        manager = OSCManager()
        text = "x" * 500
        assert manager._truncate_text(text, max_length=144) == text
    finally:
        # 恢复单例的默认行为，避免影响其它用例（OSCManager 是单例）。
        OSCManager(truncate_messages=True)


def test_prepare_outgoing_clamps_plain_text_to_real_config_limit():
    manager = OSCManager(truncate_messages=True)
    text = "甲" * 400

    result = manager._prepare_outgoing_text_for_osc(text)

    assert len(result) <= 144


def test_tagged_display_through_send_path_stays_within_limit():
    # 端到端：带语言标签的拼接文本经发送预处理后不超过 144，且保留标签前缀。
    manager = OSCManager(truncate_messages=True)
    display = build_tagged_translation_display(
        "en", "zh", "译" * 200, "source" * 30, max_chars=144,
    )

    result = manager._prepare_outgoing_text_for_osc(display)

    assert len(result) <= 144
    assert result.startswith("[en→zh] ")
    assert "(" not in result and ")" not in result


def test_arabic_osc_processing_wraps_each_line_and_keeps_total_limit():
    manager = OSCManager(truncate_messages=True)
    text = " ".join(["\u0645\u0631\u062d\u0628\u0627"] * 60)

    result = manager._prepare_outgoing_text_for_osc(text)

    assert len(result) <= 144
    assert result.count("\n") <= RTL_OSC_MAX_LINE_BREAKS
    for line in result.split("\n"):
        assert line.startswith(ARABIC_RLI)
        assert line.endswith(ARABIC_PDI)
        assert len(line[1:-1]) <= ARABIC_OSC_LINE_MAX_CHARS
        assert ARABIC_RLI not in line[1:-1]
        assert ARABIC_PDI not in line[1:-1]


def test_hebrew_osc_processing_wraps_each_line_and_limits_line_breaks():
    manager = OSCManager(truncate_messages=True)
    text = " ".join(["\u05e9\u05dc\u05d5\u05dd"] * 60)

    with patch("text_processor._bidi_get_display", lambda value: value):
        result = manager._prepare_outgoing_text_for_osc(text)

    assert len(result) <= 144
    assert result.count("\n") <= RTL_OSC_MAX_LINE_BREAKS
    for line in result.split("\n"):
        assert line.startswith(ARABIC_RLI)
        assert line.endswith(ARABIC_PDI)
        assert len(line[1:-1]) <= ARABIC_OSC_LINE_MAX_CHARS
        assert ARABIC_RLI not in line[1:-1]
        assert ARABIC_PDI not in line[1:-1]
