"""
调试用 PCM/WAV 录制器。

用于把发送给识别器之前的音频落盘，便于人工试听。
"""

from __future__ import annotations

import os
import threading
import wave
from datetime import datetime

from resource_path import ensure_dir, get_user_data_path


class WaveDebugRecorder:
    """把 PCM 数据持续写入本地 WAV 文件。"""

    def __init__(
        self,
        output_dir: str,
        input_rate: int,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        file_prefix: str = 'post_resample',
    ):
        normalized_dir = output_dir.strip() if str(output_dir).strip() else 'debug_audio'
        if not os.path.isabs(normalized_dir):
            normalized_dir = get_user_data_path(normalized_dir)
        ensure_dir(normalized_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{file_prefix}_{timestamp}_{int(input_rate)}to{int(sample_rate)}.wav'
        self.file_path = os.path.join(normalized_dir, filename)
        self._lock = threading.Lock()
        self._closed = False
        self._file = open(self.file_path, 'wb')
        self._wave = wave.open(self._file, 'wb')
        self._wave.setnchannels(int(channels))
        self._wave.setsampwidth(int(sample_width))
        self._wave.setframerate(int(sample_rate))

    def write(self, data: bytes) -> None:
        if not data:
            return

        with self._lock:
            if self._closed:
                return
            self._wave.writeframesraw(data)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            self._closed = True
            try:
                self._wave.close()
            finally:
                self._file.close()