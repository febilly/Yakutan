"""Tests for OSC-only text processing and VRChat length limits."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from osc_manager import OSCManager
from text_processor import ARABIC_OSC_LINE_MAX_CHARS, ARABIC_PDI, ARABIC_RLI


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


def test_arabic_osc_processing_wraps_each_line_and_keeps_total_limit():
    manager = OSCManager(truncate_messages=True)
    text = " ".join(["\u0645\u0631\u062d\u0628\u0627"] * 60)

    result = manager._prepare_outgoing_text_for_osc(text)

    assert len(result) <= 144
    for line in result.split("\n"):
        assert line.startswith(ARABIC_RLI)
        assert line.endswith(ARABIC_PDI)
        assert len(line[1:-1]) <= ARABIC_OSC_LINE_MAX_CHARS
        assert ARABIC_RLI not in line[1:-1]
        assert ARABIC_PDI not in line[1:-1]
