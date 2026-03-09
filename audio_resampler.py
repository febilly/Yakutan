"""
有状态的实时音频重采样器。

基于第三方库 soxr 的流式重采样实现，适用于实时音频输入。
"""

import numpy as np

try:
    import soxr  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover - 运行时依赖保护
    soxr = None
    _SOXR_IMPORT_ERROR = exc
else:
    _SOXR_IMPORT_ERROR = None


class AudioResampler:
    """有状态的 PCM 音频重采样器。

    使用 `soxr.ResampleStream` 保持跨音频块的连续性，适用于实时流式场景。

    Parameters
    ----------
    input_rate : int
        输入采样率 (Hz)，如 48000。
    output_rate : int
        输出采样率 (Hz)，如 16000。
    channels : int
        声道数，默认 1（单声道）。
    sample_width : int
        每个采样的字节数，默认 2（16-bit PCM）。
    """

    _DTYPE_MAP = {1: np.int8, 2: np.int16, 4: np.int32}
    _SOXR_DTYPE_MAP = {2: 'int16', 4: 'int32'}

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int = 1,
        sample_width: int = 2,
    ):
        if sample_width not in self._DTYPE_MAP:
            raise ValueError(f"不支持的 sample_width: {sample_width}，仅支持 1/2/4")
        if sample_width not in self._SOXR_DTYPE_MAP:
            raise ValueError(f"soxr 不支持 sample_width={sample_width}，当前仅支持 2/4")
        if soxr is None:
            raise RuntimeError(
                '缺少 soxr 依赖，请先安装 requirements.txt 中的 soxr'
            ) from _SOXR_IMPORT_ERROR
        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)
        self.channels = int(channels)
        self.sample_width = int(sample_width)
        self._dtype = self._DTYPE_MAP[self.sample_width]
        self._stream = soxr.ResampleStream(
            self.input_rate,
            self.output_rate,
            self.channels,
            dtype=self._SOXR_DTYPE_MAP[self.sample_width],
            quality='QQ',
        )

    @property
    def needs_resample(self) -> bool:
        """输入输出采样率不同时返回 True。"""
        return self.input_rate != self.output_rate

    def reset(self) -> None:
        """重置内部状态（切换音频源/重新初始化时调用）。"""
        self._stream.clear()

    def resample(self, data: bytes) -> bytes:
        """对一段 PCM 数据进行重采样。

        Parameters
        ----------
        data : bytes
            原始 PCM 字节流（input_rate 采样率）。

        Returns
        -------
        bytes
            重采样后的 PCM 字节流（output_rate 采样率）。
        """
        if not self.needs_resample or len(data) == 0:
            return data

        samples = np.frombuffer(data, dtype=self._dtype)
        num_frames = len(samples) // self.channels
        if num_frames == 0:
            return data
        frame_samples = samples[: num_frames * self.channels]
        if self.channels == 1:
            input_chunk = frame_samples
        else:
            input_chunk = frame_samples.reshape(num_frames, self.channels)

        result = self._stream.resample_chunk(input_chunk, last=False)
        return np.ascontiguousarray(result, dtype=self._dtype).tobytes()
