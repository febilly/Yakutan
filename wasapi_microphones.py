"""Live Windows microphone discovery and recording by exact WASAPI endpoint ID.

Enumeration never reinitializes the audio engine or touches an active recorder.
COM enumeration scopes and recorder leases are created/closed on their owning
threads. Only plain endpoint IDs and metadata leave an enumeration scope.
"""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import sys
import threading

import numpy as np

from audio_runtime_guard import MicrophoneSelectionError

PREFIX = 'wasapi:'
_backend = None
_backend_lock = threading.Lock()


@contextmanager
def com_apartment():
    ole32 = ctypes.WinDLL('ole32')
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    # S_OK/S_FALSE both own an initialization count. An existing STA is usable
    # after RPC_E_CHANGED_MODE, but we must not uninitialize someone else's COM.
    initialized = False
    try:
        result = ole32.CoInitializeEx(None, 0)
        if result >= 0:
            initialized = True
        elif result & 0xffffffff != 0x80010106:
            raise OSError(f'COM initialization failed: 0x{result & 0xffffffff:08x}')
        yield
    finally:
        if initialized:
            ole32.CoUninitialize()


def soundcard_backend():
    global _backend
    with _backend_lock:
        if _backend is None:
            def load():
                already_loaded = 'soundcard.mediafoundation' in sys.modules
                import soundcard
                if not already_loaded:
                    # SoundCard 0.4.6 initializes COM at import and incorrectly
                    # treats S_FALSE (already initialized MTA) as a failure.
                    # Import on a fresh thread, then balance its initialization
                    # there. Our scopes/leases own all subsequent COM lifetime.
                    from soundcard import mediafoundation
                    if mediafoundation._com.com_loaded:
                        mediafoundation._ole32.CoUninitialize()
                        mediafoundation._com.com_loaded = False
                return soundcard

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='wasapi-import') as pool:
                _backend = pool.submit(load).result()
    return _backend


class WasapiStream:
    """Expose the PCM16 read/close interface consumed by audio_capture."""

    def __init__(self, microphone, rate, channels, blocksize):
        self.context = microphone.recorder(samplerate=rate, channels=channels,
                                           blocksize=blocksize)
        self.recorder = self.context.__enter__()

    def read(self, frames, exception_on_overflow=False):
        data = self.recorder.record(numframes=frames)
        samples = np.nan_to_num(np.asarray(data), nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(np.int16).tobytes()

    def stop_stream(self):
        self.close()

    def close(self):
        if self.recorder is not None:
            self.recorder = None
            self.context.__exit__(None, None, None)


class WasapiSession:
    def __init__(self, microphone, apartment):
        self.microphone = microphone
        self.apartment = apartment
        self.device_id = PREFIX + str(microphone.id)
        self.info = dict(index=0, name=microphone.name, maxInputChannels=microphone.channels,
                         defaultSampleRate=48000, capture_all_channels=True)

    def get_device_info_by_index(self, index):
        if index != 0:
            raise MicrophoneSelectionError('Invalid microphone session index')
        return self.info

    def get_default_input_device_info(self):
        return self.info

    def open(self, *, rate, channels, frames_per_buffer, input_device_index=0, **kwargs):
        self.get_device_info_by_index(input_device_index)
        return WasapiStream(self.microphone, rate, channels, frames_per_buffer)

    def close(self):
        if self.apartment is not None:
            apartment, self.apartment = self.apartment, None
            apartment.__exit__(None, None, None)


class WasapiAudioDevices:
    supports_hotplug = True

    def __init__(self, backend=None, apartment=com_apartment):
        self.backend = backend or soundcard_backend
        self.apartment = apartment
        self.active_captures = 0

    def refresh(self):
        # Every list_inputs call enumerates the current OS endpoints.
        pass

    def shutdown(self):
        # Recorder leases belong to the audio executor, which closes them.
        pass

    def list_inputs(self):
        with self.apartment():
            sc = self.backend()
            microphones = sc.all_microphones(include_loopback=False)
            devices = []
            for microphone in microphones:
                endpoint = str(microphone.id)
                try:
                    name, channels = microphone.name, microphone.channels
                except RuntimeError:
                    # An endpoint can disappear between enumeration and lookup.
                    continue
                devices.append(dict(id=PREFIX + endpoint, name=name,
                                    max_input_channels=channels,
                                    label_suffix=hashlib.sha256(endpoint.encode()).hexdigest()[:8]))
            try:
                default = sc.default_microphone()
                default_id, default_name = PREFIX + str(default.id), default.name
            except RuntimeError:
                default_id, default_name = None, None
        return dict(devices=devices, default_id=default_id, default_name=default_name)

    @staticmethod
    def validate(token, legacy_index=None):
        if legacy_index is not None and not token:
            raise MicrophoneSelectionError('旧麦克风序号无法安全恢复，请重新选择麦克风。')
        if token and (not isinstance(token, str) or not token.startswith(PREFIX)):
            raise MicrophoneSelectionError('麦克风选择已失效，请重新选择麦克风。')

    def _microphone(self, sc, token, legacy_index=None):
        self.validate(token, legacy_index)
        if not token:
            return sc.default_microphone()
        # Never use SoundCard's name/fuzzy matching API for a saved ID.
        for microphone in sc.all_microphones(include_loopback=False):
            if PREFIX + str(microphone.id) == token:
                return microphone
        raise MicrophoneSelectionError('所选麦克风未连接，正在等待该设备恢复。')

    def resolve(self, token, legacy_index=None):
        with self.apartment():
            mic = self._microphone(self.backend(), token, legacy_index)
            return PREFIX + str(mic.id)

    def acquire(self, token, legacy_index=None):
        apartment = self.apartment()
        apartment.__enter__()
        try:
            microphone = self._microphone(self.backend(), token, legacy_index)
            session = WasapiSession(microphone, apartment)
        except Exception:
            apartment.__exit__(None, None, None)
            raise
        self.active_captures += 1
        return session, 0

    def release(self, session):
        try:
            session.close()
        finally:
            self.active_captures = max(0, self.active_captures - 1)
