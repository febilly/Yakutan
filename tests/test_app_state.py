"""Tests for app_state."""

from __future__ import annotations

import os
from pathlib import Path

from app_state import AppState, get_state, set_state, get_smart_selector


class TestAppState:
    def test_default_initialization(self):
        state = AppState()
        assert state.translation_api is None
        assert state.translator is None
        assert state.secondary_translator is None
        assert state.backwards_translator is None
        assert state.deepl_fallback_translator is None
        assert state.translation_api_type is None
        assert state.target_language is None
        assert state.secondary_target_language is None
        assert state.recognition_active is False
        assert state.subtitles_state == {
            "original": "",
            "translated": "",
            "reverse_translated": "",
            "ongoing": False,
        }

    def test_update_subtitles(self):
        state = AppState()
        state.update_subtitles("hello", "bonjour", False, "salut")
        assert state.subtitles_state["original"] == "hello"
        assert state.subtitles_state["translated"] == "bonjour"
        assert state.subtitles_state["reverse_translated"] == "salut"
        assert state.subtitles_state["ongoing"] is False

    def test_update_subtitles_different_values(self):
        state = AppState()
        state.update_subtitles("src1", "tgt1", True, "rev1")
        state.update_subtitles("src2", "tgt2", False, "rev2")
        assert state.subtitles_state["original"] == "src2"
        assert state.subtitles_state["translated"] == "tgt2"

    def test_ensure_executor(self):
        state = AppState()
        assert state.executor is not None
        # Shutdown and recreate
        state.executor.shutdown(wait=False)
        state.ensure_executor()
        assert state.executor._shutdown is False

    def test_ensure_audio_executor(self):
        state = AppState()
        assert state.audio_executor is not None
        state.audio_executor.shutdown(wait=False)
        state.ensure_audio_executor()
        assert state.audio_executor._shutdown is False


class TestGetSetState:
    def test_get_state_initially_none(self):
        assert get_state() is None

    def test_set_and_get(self):
        state = AppState()
        set_state(state)
        assert get_state() is state

    def test_set_none(self):
        set_state(None)
        assert get_state() is None


class TestGetSmartSelector:
    def test_singleton(self):
        sel1 = get_smart_selector()
        sel2 = get_smart_selector()
        assert sel1 is sel2

    def test_type(self):
        sel = get_smart_selector()
        assert "SmartTargetLanguageSelector" in type(sel).__name__

    def test_methods_exist(self):
        sel = get_smart_selector()
        assert hasattr(sel, "record_language")
        assert hasattr(sel, "clear_history")
        assert hasattr(sel, "select_target_language")

    def test_select_target_default_disabled(self):
        sel = get_smart_selector()
        result = sel.select_target_language()
        assert result == []
