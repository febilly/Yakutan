"""Stable endpoint identity and live discovery while a recorder is running."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pytest

import audio_capture
from wasapi_microphones import WasapiAudioDevices, MicrophoneSelectionError
import wasapi_microphones


def microphone(endpoint, name='USB Microphone', channels=2):
    mic = SimpleNamespace(id=endpoint, name=name, channels=channels, recorder=MagicMock())
    mic.recorder.return_value.__enter__.return_value.record.side_effect = (
        lambda numframes: np.tile([-0.5, 0.5][:channels], (numframes, 1))
    )
    return mic


@pytest.fixture
def native(monkeypatch):
    first, second = microphone('endpoint-a'), microphone('endpoint-b')
    sc = MagicMock()
    sc.all_microphones.return_value = [first, second]
    sc.default_microphone.return_value = first
    apartments = []

    @contextmanager
    def apartment():
        thread = threading.get_ident()
        apartments.append(('enter', thread))
        try:
            yield
        finally:
            assert threading.get_ident() == thread
            apartments.append(('exit', thread))

    devices = WasapiAudioDevices(backend=lambda: sc, apartment=apartment)
    monkeypatch.setattr(audio_capture, 'audio_devices', devices)
    monkeypatch.setattr(audio_capture.config, 'MIC_DEVICE_ID', 'wasapi:endpoint-b')
    monkeypatch.setattr(audio_capture.config, 'MIC_DEVICE_INDEX', None)
    monkeypatch.setattr(audio_capture.config, 'SAVE_PRE_RESAMPLE_AUDIO', False)
    monkeypatch.setattr(audio_capture.config, 'SAVE_POST_RESAMPLE_AUDIO', False)
    return devices, sc, first, second, apartments


def test_same_name_devices_have_stable_ids_across_order_and_backend_restart(native):
    devices, sc, first, second, _ = native
    before = devices.list_inputs()['devices']
    assert [d['id'] for d in before] == ['wasapi:endpoint-a', 'wasapi:endpoint-b']
    sc.all_microphones.return_value = [second, first]
    after = devices.list_inputs()['devices']
    assert after == before[::-1]
    restarted = WasapiAudioDevices(backend=lambda: sc, apartment=devices.apartment)
    session, _ = restarted.acquire(before[1]['id'])
    assert session.microphone is second
    restarted.release(session)


def test_live_list_changes_do_not_close_or_reopen_active_capture(native):
    devices, sc, first, second, apartments = native
    session, _ = devices.acquire('wasapi:endpoint-b')
    stream = session.open(rate=16000, channels=2, frames_per_buffer=1600)
    third = microphone('endpoint-c')
    sc.all_microphones.return_value = [second, third]
    devices.refresh()
    assert [d['id'] for d in devices.list_inputs()['devices']] == [
        'wasapi:endpoint-b', 'wasapi:endpoint-c']
    second.recorder.return_value.__exit__.assert_not_called()
    second.recorder.assert_called_once()
    assert session.device_id == 'wasapi:endpoint-b'
    assert devices.active_captures == 1
    stream.close()
    devices.release(session)
    assert sum(event == 'enter' for event, _ in apartments) == sum(
        event == 'exit' for event, _ in apartments)


def test_missing_id_never_fuzzy_matches_same_name_or_falls_back(native):
    devices, sc, first, second, _ = native
    sc.all_microphones.return_value = [first]
    with pytest.raises(MicrophoneSelectionError):
        devices.acquire('wasapi:endpoint-b')
    sc.default_microphone.assert_not_called()
    sc.get_microphone.assert_not_called()
    first.recorder.assert_not_called()
    assert devices.active_captures == 0


@pytest.mark.parametrize('token,index', [('old-snapshot:1', None), (None, 1)])
def test_old_selection_requires_one_time_reselection(native, token, index):
    devices, *_ = native
    with pytest.raises(MicrophoneSelectionError):
        devices.acquire(token, index)


def test_pcm_conversion_and_close_are_safe(native):
    devices, sc, first, second, _ = native
    session, _ = devices.acquire('wasapi:endpoint-b')
    stream = session.open(rate=16000, channels=2, frames_per_buffer=1600)
    recorder = second.recorder.return_value.__enter__.return_value
    recorder.record.side_effect = None
    recorder.record.return_value = np.array([[-1.0, 1.0], [np.nan, np.inf], [-0.5, 0.5]])
    assert np.frombuffer(stream.read(3), dtype=np.int16).tolist() == [
        -32768, 32767, 0, 32767, -16384, 16384]
    stream.stop_stream()
    stream.close()
    second.recorder.return_value.__exit__.assert_called_once()
    devices.release(session)


def test_switch_unplug_and_reconnect_only_reopen_capture(native, monkeypatch):
    devices, sc, first, second, _ = native
    with ThreadPoolExecutor(max_workers=1) as executor:
        state = SimpleNamespace(audio_executor=executor, ensure_audio_executor=lambda: None,
                                debug_pre_audio_recorder=None, debug_audio_recorder=None,
                                recognition_active=True, recognizer=object())
        recognizer = state.recognizer

        async def exercise():
            await audio_capture.init_audio_stream(state)
            assert state.capture_device_id == 'wasapi:endpoint-b'
            assert state.capture_channels == 2  # preserve native WASAPI channels
            pcm = await audio_capture.read_audio_data(state)
            assert np.frombuffer(pcm, dtype=np.int16).size == state.input_block_size
            assert np.all(np.frombuffer(pcm, dtype=np.int16) == 0)  # downmix
            original_stream = state.stream
            assert await audio_capture.maintain_audio_source(state)
            assert state.stream is original_stream

            # Same-name replacement must not steal the selected microphone.
            sc.all_microphones.return_value = [first]
            state._audio_next_device_check = 0
            assert not await audio_capture.maintain_audio_source(state)
            assert state.stream is None
            assert devices.active_captures == 0
            first.recorder.assert_not_called()
            assert audio_capture.config.MIC_DEVICE_ID == 'wasapi:endpoint-b'

            sc.all_microphones.return_value = [first, second]
            state._audio_next_device_check = 0
            assert await audio_capture.maintain_audio_source(state)
            assert state.capture_device_id == 'wasapi:endpoint-b'

            # User selection changes can reopen capture immediately.
            monkeypatch.setattr(audio_capture.config, 'MIC_DEVICE_ID', 'wasapi:endpoint-a')
            assert await audio_capture.maintain_audio_source(state)
            assert state.capture_device_id == 'wasapi:endpoint-a'
            assert state.recognizer is recognizer
            assert state.recognition_active
            await audio_capture.close_audio_stream(state)

        asyncio.run(exercise())
    assert devices.active_captures == 0


def test_default_endpoint_change_only_reopens_capture(native, monkeypatch):
    devices, sc, first, second, _ = native
    monkeypatch.setattr(audio_capture.config, 'MIC_DEVICE_ID', None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        state = SimpleNamespace(audio_executor=executor, ensure_audio_executor=lambda: None,
                                debug_pre_audio_recorder=None, debug_audio_recorder=None)

        async def exercise():
            await audio_capture.init_audio_stream(state)
            assert state.capture_device_id == 'wasapi:endpoint-a'
            sc.default_microphone.return_value = second
            assert await audio_capture.maintain_audio_source(state)
            assert state.capture_device_id == 'wasapi:endpoint-b'
            await audio_capture.close_audio_stream(state)

        asyncio.run(exercise())


def test_api_refresh_works_during_capture(native, monkeypatch):
    import ui.app as web
    devices, sc, first, second, _ = native
    monkeypatch.setattr(web, 'audio_devices', devices)
    session, _ = devices.acquire('wasapi:endpoint-b')
    response = web.app.test_client().get('/api/audio/input-devices?refresh=1')
    assert response.status_code == 200
    assert response.get_json()['supports_hotplug']
    assert response.get_json()['selected_id'] == 'wasapi:endpoint-b'
    devices.release(session)


def test_soundcard_import_balances_its_com_on_fresh_thread(monkeypatch):
    import sys
    current = threading.get_ident()
    unloaded_on = []
    backend = SimpleNamespace(
        _com=SimpleNamespace(com_loaded=True),
        _ole32=SimpleNamespace(CoUninitialize=lambda: unloaded_on.append(threading.get_ident())),
    )
    sc = SimpleNamespace(mediafoundation=backend)
    monkeypatch.setattr(wasapi_microphones, '_backend', None)
    monkeypatch.setitem(sys.modules, 'soundcard', sc)
    monkeypatch.delitem(sys.modules, 'soundcard.mediafoundation', raising=False)
    assert wasapi_microphones.soundcard_backend() is sc
    assert wasapi_microphones.soundcard_backend() is sc
    assert len(unloaded_on) == 1
    assert unloaded_on[0] != current
    assert not backend._com.com_loaded


@pytest.mark.parametrize('hresult', [0, 1])
def test_com_scope_balances_success_and_already_initialized(monkeypatch, hresult):
    ole32 = MagicMock()
    ole32.CoInitializeEx.return_value = hresult
    monkeypatch.setattr(wasapi_microphones.ctypes, 'WinDLL', lambda _: ole32, raising=False)
    with wasapi_microphones.com_apartment():
        ole32.CoUninitialize.assert_not_called()
    ole32.CoUninitialize.assert_called_once()


def test_com_scope_does_not_uninitialize_existing_sta(monkeypatch):
    ole32 = MagicMock()
    ole32.CoInitializeEx.return_value = -2147417850  # RPC_E_CHANGED_MODE
    monkeypatch.setattr(wasapi_microphones.ctypes, 'WinDLL', lambda _: ole32, raising=False)
    with wasapi_microphones.com_apartment():
        pass
    ole32.CoUninitialize.assert_not_called()


def test_capture_loop_survives_read_failure_and_resumes_without_service_restart(monkeypatch):
    state = SimpleNamespace(
        current_asr_backend='soniox', vad_enabled=False, vad_processor=None,
        stop_event=threading.Event(), recognition_active=True, audio_send_generation=0,
        _audio_hotplug_enabled=True, stream=object(),
    )
    checks = []
    reads = []

    async def maintain(_):
        checks.append(True)
        if len(checks) == 2:
            return False  # still unplugged
        state.stream = state.stream or object()
        return True

    async def close(_):
        state.stream = None

    async def read(_):
        reads.append(True)
        if len(reads) == 1:
            return None
        state.stop_event.set()
        return b'\x01\x00' * 512

    send = AsyncMock()
    monkeypatch.setattr(audio_capture, 'maintain_audio_source', maintain)
    monkeypatch.setattr(audio_capture, 'close_audio_stream', close)
    monkeypatch.setattr(audio_capture, 'read_audio_data', read)
    monkeypatch.setattr(audio_capture, 'send_audio_frame_async', send)
    asyncio.run(audio_capture.audio_capture_task(state, object()))
    assert len(checks) == 3
    assert len(reads) == 2
    send.assert_awaited_once()
