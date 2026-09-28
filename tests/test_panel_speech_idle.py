"""The panel timer must only reset for speech eligible for transcription."""

import asyncio
import importlib
import time
from types import SimpleNamespace
from unittest.mock import patch

import config
from app_state import AppState
import main


def test_speech_during_mic_pause_does_not_reset_idle_timer():
    state = AppState()
    state.recognition_active = True
    state.last_eligible_speech_at = time.monotonic() - 181

    state.mic_muted_for_recognition = True
    previous = state.last_eligible_speech_at
    state.mark_eligible_speech()
    assert state.last_eligible_speech_at == previous
    assert state.speech_idle_seconds() >= 181

    state.mic_muted_for_recognition = False
    state.mark_eligible_speech()
    assert state.speech_idle_seconds() < 1

    state.recognition_active = False
    previous = state.last_eligible_speech_at
    state.mark_eligible_speech()
    assert state.last_eligible_speech_at == previous


def test_mute_signal_blocks_timer_before_delayed_recognition_pause():
    state = SimpleNamespace(current_asr_backend='qwen', recognition_instance=None)
    with (
        patch.object(main.config, 'ENABLE_DOUBLE_MUTE_CLEAR', False),
        patch.object(main, 'is_effective_mic_control_enabled', return_value=True),
    ):
        asyncio.run(main.handle_mute_change(state, True))
        assert state.mic_muted_for_recognition is True
        asyncio.run(main.handle_mute_change(state, False))
        assert state.mic_muted_for_recognition is False


def test_status_exposes_idle_time_only_while_service_runs():
    ui_app = importlib.import_module('ui.app')
    state = AppState()
    state.last_eligible_speech_at = time.monotonic() - 181
    client = ui_app.app.test_client()

    with (
        patch.object(ui_app, '_live_app_state', return_value=state),
        patch.object(ui_app, '_snapshot_service_status', return_value={
            'lifecycle': 'running', 'running': True,
        }),
    ):
        elapsed = client.get('/api/status').get_json()['speech_idle_seconds']
    assert 181 <= elapsed < 183

    with (
        patch.object(ui_app, '_live_app_state', return_value=state),
        patch.object(ui_app, '_snapshot_service_status', return_value={
            'lifecycle': 'stopped', 'running': False,
        }),
    ):
        assert client.get('/api/status').get_json()['speech_idle_seconds'] is None


def test_panel_timer_settings_round_trip_and_reject_reversed_thresholds():
    ui_app = importlib.import_module('ui.app')
    client = ui_app.app.test_client()
    defaults = client.get('/api/config/defaults').get_json()['panel']
    assert defaults['speech_idle_enabled'] is True
    assert defaults['speech_idle_warning_minutes'] == 3
    assert defaults['speech_idle_critical_minutes'] == 6

    with (
        patch.object(config, 'PANEL_SPEECH_IDLE_ENABLED', True),
        patch.object(config, 'PANEL_SPEECH_IDLE_WARNING_MINUTES', 3),
        patch.object(config, 'PANEL_SPEECH_IDLE_CRITICAL_MINUTES', 6),
    ):
        response = client.post('/api/config', json={'panel': {
            'speech_idle_enabled': False,
            'speech_idle_warning_minutes': 4,
            'speech_idle_critical_minutes': 8,
        }})
        assert response.status_code == 200
        assert response.get_json()['success'] is True
        panel = client.get('/api/config').get_json()['panel']
        assert panel['speech_idle_enabled'] is False
        assert panel['speech_idle_warning_minutes'] == 4
        assert panel['speech_idle_critical_minutes'] == 8
        assert client.get('/api/status').get_json()['panel_speech_idle'] == {
            'enabled': False, 'warning_minutes': 4, 'critical_minutes': 8,
        }

        invalid = client.post('/api/config', json={'panel': {
            'speech_idle_warning_minutes': 8,
            'speech_idle_critical_minutes': 4,
        }})
        assert invalid.get_json()['success'] is False
        assert client.get('/api/config').get_json()['panel'] == panel
