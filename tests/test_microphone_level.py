"""Page-owned meters release native resources on disconnect."""
import importlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from microphone_meter import MicrophoneMeter, MicrophoneMeters

@pytest.mark.parametrize('sample, expected', [(0, 0.0), (16384, 0.5), (-32768, 1.0)])
def test_meter_captures_and_closes_on_own_thread(sample, expected):
    owners = []
    meter = None
    def read(*args, **kwargs):
        owners.append(threading.get_ident())
        meter.stop.wait(0.01)
        return np.full(128, sample, dtype=np.int16).tobytes()
    stream = SimpleNamespace(read=read, stop_stream=Mock(), close=Mock())
    session = SimpleNamespace(device_id='wasapi:mic-a', open=Mock(return_value=stream),
        get_device_info_by_index=Mock(return_value={'defaultSampleRate': 48000, 'maxInputChannels': 1}))
    devices = SimpleNamespace(acquire=Mock(return_value=(session, 0)), release=Mock(), supports_hotplug=False)
    meter = MicrophoneMeter('wasapi:mic-a', devices)
    meter.thread.start()
    try:
        deadline = time.monotonic() + 2
        while meter.snapshot[1] == 0:
            assert time.monotonic() < deadline
            threading.Event().wait(0.01)
        assert meter.level() == pytest.approx(expected)
    finally:
        meter.stop.set()
        meter.thread.join(2)
    assert not meter.thread.is_alive()
    assert all(owner == meter.thread.ident for owner in owners)
    devices.acquire.assert_called_once_with('wasapi:mic-a')
    devices.release.assert_called_once_with(session)
    stream.close.assert_called_once()
    assert meter.level() == 0

def fake_meter(device_id):
    return SimpleNamespace(device_id=device_id, clients=0, stop=threading.Event(),
        thread=SimpleNamespace(start=Mock(), join=Mock()), level=lambda: 0.25)

def test_same_device_tabs_share_capture_until_last_disconnect():
    meters = MicrophoneMeters(fake_meter)
    first = meters.subscribe('wasapi:mic-a')
    second = meters.subscribe('wasapi:mic-a')
    assert first is second
    first.thread.start.assert_called_once()
    meters.unsubscribe(first)
    assert not first.stop.is_set()
    meters.unsubscribe(second)
    assert first.stop.is_set()
    assert not meters.meters

def test_different_device_tabs_are_independent():
    meters = MicrophoneMeters(fake_meter)
    first = meters.subscribe(None)
    second = meters.subscribe('wasapi:mic-b')
    assert first is not second
    meters.unsubscribe(first)
    assert not second.stop.is_set()
    meters.close()
    assert second.stop.is_set()

@pytest.mark.parametrize('lifecycle', ['stopped', 'running'])
def test_sse_works_without_asr_and_releases_on_close(monkeypatch, lifecycle):
    web = importlib.import_module('ui.app')
    meters = MicrophoneMeters(fake_meter)
    monkeypatch.setattr(web, 'microphone_meters', meters)
    monkeypatch.setattr(web, '_get_service_lifecycle', lambda: lifecycle)
    response = web.app.test_client().get('/api/audio/level/stream?device_id=wasapi:mic-a', buffered=False)
    assert b'"level": 0.25' in next(response.response)
    assert response.mimetype == 'text/event-stream'
    meter = meters.meters['wasapi:mic-a']
    response.close()
    assert meter.stop.is_set()
    assert not meters.meters

def test_stale_snapshot_returns_zero():
    meter = MicrophoneMeter(None)
    meter.snapshot = (0.5, time.monotonic() - 1)
    assert meter.level() == 0

def test_open_failure_releases_session_without_default_fallback():
    session = SimpleNamespace(get_device_info_by_index=Mock(return_value={'defaultSampleRate': 48000}),
        open=Mock(side_effect=OSError('disconnected')))
    meter = None
    def release(_session):
        meter.stop.set()
    devices = SimpleNamespace(acquire=Mock(return_value=(session, 0)), release=Mock(side_effect=release))
    meter = MicrophoneMeter('wasapi:missing', devices)
    meter.thread.start()
    meter.thread.join(3)
    assert not meter.thread.is_alive()
    devices.acquire.assert_called_once_with('wasapi:missing')
    devices.release.assert_called_once_with(session)
    assert meter.level() == 0
