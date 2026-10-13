"""Page-owned microphone metering, independent of recognition and its mute state."""
import atexit
import logging
import threading
import time

import numpy as np
import pyaudio

from audio_devices import audio_devices
from audio_runtime_guard import hold_portaudio

logger = logging.getLogger(__name__)


class MicrophoneMeter:
    def __init__(self, device_id, devices=audio_devices):
        self.device_id = device_id
        self.devices = devices
        self.clients = 0
        self.stop = threading.Event()
        self.snapshot = (0.0, 0.0)
        self.thread = threading.Thread(target=self._run, name='microphone-meter', daemon=True)

    def level(self):
        level, updated_at = self.snapshot
        return level if time.monotonic() - updated_at < 0.5 else 0.0

    def _run(self):
        last_error = None
        while not self.stop.is_set():
            session = stream = None
            try:
                with hold_portaudio('open_microphone_meter'):
                    session, index = self.devices.acquire(self.device_id)
                    info = (session.get_device_info_by_index(index) if index is not None
                            else session.get_default_input_device_info())
                    rate = int(info.get('defaultSampleRate') or 48000)
                    channels = max(1, int(info.get('maxInputChannels') or 1)) if info.get('capture_all_channels') else 1
                    frames = max(1, rate // 20)
                    kwargs = dict(format=pyaudio.paInt16, channels=channels, rate=rate,
                                  input=True, frames_per_buffer=frames)
                    if index is not None:
                        kwargs['input_device_index'] = index
                    stream = session.open(**kwargs)
                    endpoint = getattr(session, 'device_id', None)
                next_check = time.monotonic() + 1.0
                while not self.stop.is_set():
                    pcm = stream.read(frames, exception_on_overflow=False)
                    if not pcm:
                        raise OSError('Microphone returned no audio')
                    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                    if channels > 1:
                        samples = samples.reshape(-1, channels).mean(axis=1)
                    rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
                    self.snapshot = (min(1.0, rms), time.monotonic())
                    last_error = None
                    # Follow the system default and recover unplug/replug without restarting ASR.
                    if self.devices.supports_hotplug and time.monotonic() >= next_check:
                        next_check = time.monotonic() + 1.0
                        with hold_portaudio('resolve_microphone_meter'):
                            current = self.devices.resolve(self.device_id)
                        if current != endpoint:
                            break
            except Exception as exc:
                self.snapshot = (0.0, 0.0)
                if not self.stop.is_set() and str(exc) != last_error:
                    logger.warning('Microphone meter unavailable: %s', exc)
                    last_error = str(exc)
                self.stop.wait(1.0)
            finally:
                self.snapshot = (0.0, 0.0)
                with hold_portaudio('close_microphone_meter'):
                    if stream is not None:
                        try:
                            stream.stop_stream()
                        except Exception:
                            logger.debug('Failed to stop meter stream', exc_info=True)
                        try:
                            stream.close()
                        except Exception:
                            logger.debug('Failed to close meter stream', exc_info=True)
                    if session is not None:
                        self.devices.release(session)


class MicrophoneMeters:
    def __init__(self, factory=MicrophoneMeter):
        self.factory = factory
        self.lock = threading.Lock()
        self.meters = {}

    def subscribe(self, device_id):
        with self.lock:
            meter = self.meters.get(device_id)
            if meter is None:
                meter = self.factory(device_id)
                self.meters[device_id] = meter
                meter.thread.start()
            meter.clients += 1
            return meter

    def unsubscribe(self, meter):
        with self.lock:
            meter.clients -= 1
            if meter.clients == 0:
                self.meters.pop(meter.device_id, None)
                meter.stop.set()

    def close(self):
        with self.lock:
            meters = list(self.meters.values())
            self.meters.clear()
            for meter in meters:
                meter.stop.set()
        for meter in meters:
            meter.thread.join(timeout=1.0)


microphone_meters = MicrophoneMeters()
atexit.register(microphone_meters.close)
