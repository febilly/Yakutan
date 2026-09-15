"""Device selection must not cross PortAudio enumeration lifetimes."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pyaudio
import audio_devices as device_module


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    registry = device_module.AudioDevices()
    monkeypatch.setattr(device_module, 'audio_devices', registry)
    import audio_capture
    import ui.app as web
    monkeypatch.setattr(audio_capture, 'audio_devices', registry)
    monkeypatch.setattr(web, 'audio_devices', registry)
    monkeypatch.setattr(audio_capture.config, 'MIC_DEVICE_ID', None)
    monkeypatch.setattr(audio_capture.config, 'MIC_DEVICE_INDEX', None)
    yield registry
    registry.shutdown()


def fake_audio(names=("USB Mic", "USB Mic")):
    pa = MagicMock()
    infos = [dict(index=i, name=name, hostApi=0, maxInputChannels=1,
                  defaultSampleRate=48000) for i, name in enumerate(names)]
    pa.get_device_count.return_value = len(infos)
    pa.get_device_info_by_index.side_effect = lambda i: infos[i]
    pa.get_host_api_count.return_value = 1
    pa.get_host_api_info_by_index.return_value = dict(index=0, name='Windows WASAPI')
    pa.get_default_input_device_info.return_value = infos[0]
    return pa


def test_input_list_keeps_distinct_devices_with_identical_names(monkeypatch):
    import ui.app as web
    pa = fake_audio()
    monkeypatch.setattr(pyaudio, 'PyAudio', lambda: pa)
    response = web.app.test_client().get('/api/audio/input-devices')
    devices = response.get_json()['devices']
    assert len(devices) == 2
    assert devices[0]['id'] != devices[1]['id']


def test_explicit_device_open_failure_never_uses_default(monkeypatch, isolated_registry):
    import audio_capture
    pa = fake_audio()
    pa.open.side_effect = OSError('device unplugged')
    monkeypatch.setattr(pyaudio, 'PyAudio', lambda: pa)
    token = isolated_registry.list_inputs()['devices'][1]['id']
    monkeypatch.setattr(audio_capture.config, 'MIC_DEVICE_ID', token)
    with ThreadPoolExecutor(max_workers=1) as executor:
        state = SimpleNamespace(audio_executor=executor, ensure_audio_executor=lambda: None)
        with pytest.raises((OSError, ValueError)):
            asyncio.run(audio_capture.init_audio_stream(state))
    assert all(call.kwargs.get('input_device_index') == 1 for call in pa.open.call_args_list)
    assert pa.open.call_count == 2  # 16 kHz and native 48 kHz, same device
    assert isolated_registry.active_captures == 0


def test_list_and_capture_share_instance_even_if_next_enumeration_reorders(monkeypatch, isolated_registry):
    import audio_capture
    import ui.app as web
    first = fake_audio(('Headset', 'USB Mic'))
    reordered = fake_audio(('USB Mic', 'Headset'))
    factory = MagicMock(side_effect=[first, reordered])
    monkeypatch.setattr(pyaudio, 'PyAudio', factory)
    client = web.app.test_client()
    devices = client.get('/api/audio/input-devices').get_json()['devices']
    token = devices[1]['id']
    assert client.get('/api/audio/input-devices').get_json()['devices'] == devices
    ok, *_ = web.update_config({'mic_control': {'mic_device_id': token}})
    assert ok
    with ThreadPoolExecutor(max_workers=1) as executor:
        state = SimpleNamespace(audio_executor=executor, ensure_audio_executor=lambda: None,
                                debug_pre_audio_recorder=None, debug_audio_recorder=None)
        asyncio.run(audio_capture.init_audio_stream(state))
        assert state.mic is first
        assert first.open.call_args.kwargs['input_device_index'] == 1
        assert isolated_registry.active_captures == 1
        assert client.get('/api/audio/input-devices?refresh=1').status_code == 409
        asyncio.run(audio_capture.close_audio_stream(state))
    factory.assert_called_once()
    first.terminate.assert_not_called()
    assert isolated_registry.active_captures == 0
    client.get('/api/audio/input-devices?refresh=1')
    with pytest.raises(device_module.MicrophoneSelectionError):
        isolated_registry.acquire(token)
    reordered.open.assert_not_called()


@pytest.mark.parametrize('token,index', [('prior-process:1', None), (None, 1)])
def test_stale_or_legacy_selection_is_rejected_before_open(monkeypatch, isolated_registry, token, index):
    factory = MagicMock()
    monkeypatch.setattr(pyaudio, 'PyAudio', factory)
    with pytest.raises(device_module.MicrophoneSelectionError):
        isolated_registry.acquire(token, index)
    factory.assert_not_called()


def test_default_selection_remains_explicit(monkeypatch, isolated_registry):
    pa = fake_audio()
    monkeypatch.setattr(pyaudio, 'PyAudio', lambda: pa)
    assert isolated_registry.acquire(None) == (pa, None)
    isolated_registry.release()


def test_duplicate_names_remain_selectable_in_same_snapshot(monkeypatch, isolated_registry):
    pa = fake_audio()
    monkeypatch.setattr(pyaudio, 'PyAudio', lambda: pa)
    devices = isolated_registry.list_inputs()['devices']
    assert isolated_registry.resolve(devices[0]['id']) == 0
    assert isolated_registry.resolve(devices[1]['id']) == 1


def test_start_rejects_stale_selection_with_visible_error(monkeypatch):
    import ui.app as web
    monkeypatch.setattr(web.config, 'MIC_DEVICE_ID', 'old-process:1')
    monkeypatch.setattr(web, '_get_service_lifecycle', lambda: 'stopped')
    start = MagicMock()
    monkeypatch.setattr(web.threading, 'Thread', start)
    result = web.app.test_client().post('/api/service/start', json={})
    assert result.status_code == 400
    assert result.get_json()['message_id'] == 'option.micReselect'
    start.assert_not_called()


def test_default_start_refreshes_old_default_snapshot(monkeypatch, isolated_registry):
    old, new = fake_audio(('Old default',)), fake_audio(('New default',))
    monkeypatch.setattr(pyaudio, 'PyAudio', MagicMock(side_effect=[old, new]))
    isolated_registry.list_inputs()
    assert isolated_registry.acquire(None) == (new, None)
    old.terminate.assert_called_once()
    isolated_registry.release()


def test_unplugged_stream_close_does_not_block_rescan(monkeypatch, isolated_registry):
    import audio_capture
    pa = fake_audio()
    monkeypatch.setattr(pyaudio, 'PyAudio', lambda: pa)
    mic, _ = isolated_registry.acquire(None)
    stream = MagicMock()
    stream.stop_stream.side_effect = OSError('disconnected')
    with ThreadPoolExecutor(max_workers=1) as executor:
        state = SimpleNamespace(audio_executor=executor, stream=stream, mic=mic)
        asyncio.run(audio_capture.close_audio_stream(state))
    stream.close.assert_called_once()
    assert isolated_registry.active_captures == 0
    isolated_registry.refresh()
