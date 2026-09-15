"""Windows uses live WASAPI endpoint IDs; other platforms use guarded snapshots.

PyAudio exposes enumeration indices, not persistent OS endpoint IDs. Keep the
enumeration alive while its tokens can be used; never resolve a saved index or
a duplicate display name against a new enumeration. Callers serialize native
operations with hold_portaudio (including stream open/close).
"""
import atexit
import sys
import uuid

from audio_runtime_guard import hold_portaudio, _suppress_stderr, MicrophoneSelectionError


class AudioDevices:
    """PortAudio fallback for platforms without the Windows endpoint backend."""
    supports_hotplug = False
    def __init__(self):
        self.pa = None
        self.generation = None
        self.active_captures = 0
        self.tokens = {}

    def instance(self):
        if self.pa is None:
            import pyaudio
            with _suppress_stderr():
                self.pa = pyaudio.PyAudio()
            self.generation = uuid.uuid4().hex
        return self.pa

    def refresh(self):
        if self.active_captures:
            raise MicrophoneSelectionError('请先停止服务，再刷新麦克风设备列表。')
        self.shutdown()

    def shutdown(self):
        with hold_portaudio('audio_devices_shutdown'):
            if self.pa is not None:
                self.pa.terminate()
            self.pa = None
            self.tokens.clear()

    def list_inputs(self):
        pa = self.instance()
        preferred = None
        for i in range(pa.get_host_api_count()):
            host = pa.get_host_api_info_by_index(i)
            if 'wasapi' in str(host.get('name', '')).lower():
                preferred = i
                break
        if preferred is None:
            try:
                preferred = pa.get_default_host_api_info().get('index')
            except OSError:
                preferred = None
        try:
            default = pa.get_default_input_device_info()
        except OSError:
            default = {}
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (preferred is not None and info.get('hostApi') != preferred) or info.get('maxInputChannels', 0) <= 0:
                continue
            token = f'{self.generation}:{i}'
            self.tokens[token] = i
            devices.append(dict(id=token, index=i, name=info['name'],
                                max_input_channels=info['maxInputChannels']))
        return dict(devices=devices, default_index=default.get('index'),
                    default_name=default.get('name'))

    def resolve(self, token, legacy_index=None):
        if not token:
            if legacy_index is not None:
                raise MicrophoneSelectionError('旧麦克风序号无法安全恢复，请重新选择麦克风。')
            return None
        if token not in self.tokens:
            raise MicrophoneSelectionError('麦克风选择已失效，请刷新列表并重新选择麦克风。')
        return self.tokens[token]

    def acquire(self, token, legacy_index=None):
        index = self.resolve(token, legacy_index)
        if index is None and not self.active_captures:
            # The default may have changed since the UI first listed devices.
            # A default selection has no device token to preserve.
            self.refresh()
        pa = self.instance()
        self.active_captures += 1
        return pa, index

    def release(self, session=None):
        self.active_captures = max(0, self.active_captures - 1)


if sys.platform == 'win32':
    from wasapi_microphones import WasapiAudioDevices
    audio_devices = WasapiAudioDevices()
else:
    audio_devices = AudioDevices()
atexit.register(audio_devices.shutdown)
