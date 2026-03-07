"""
有状态的实时音频重采样器，替代 Python 3.13 中已移除的 audioop.ratecv。

使用 numpy 线性插值，支持跨 chunk 边界的连续重采样。
"""

import numpy as np


class AudioResampler:
    """有状态的 PCM 音频重采样器。

    保持跨音频块的插值连续性，适用于实时流式场景。

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
    _CLIP_MAP = {1: (-128, 127), 2: (-32768, 32767), 4: (-2147483648, 2147483647)}

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int = 1,
        sample_width: int = 2,
    ):
        if sample_width not in self._DTYPE_MAP:
            raise ValueError(f"不支持的 sample_width: {sample_width}，仅支持 1/2/4")
        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)
        self.channels = int(channels)
        self.sample_width = int(sample_width)
        self._dtype = self._DTYPE_MAP[self.sample_width]
        self._clip_min, self._clip_max = self._CLIP_MAP[self.sample_width]
        self._frac_pos: float = 0.0      # 上一 chunk 结束后的残余分数位置
        self._last_frame: np.ndarray | None = None  # 上一 chunk 最后一帧（用于边界插值）

    @property
    def needs_resample(self) -> bool:
        """输入输出采样率不同时返回 True。"""
        return self.input_rate != self.output_rate

    def reset(self) -> None:
        """重置内部状态（切换音频源/重新初始化时调用）。"""
        self._frac_pos = 0.0
        self._last_frame = None

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

        # --- 解码 PCM ---
        samples = np.frombuffer(data, dtype=self._dtype).astype(np.float64)
        num_frames = len(samples) // self.channels
        if num_frames == 0:
            return data
        samples = samples[: num_frames * self.channels].reshape(num_frames, self.channels)

        # --- 构造带边界上下文的扩展数组 ---
        if self._last_frame is not None:
            extended = np.vstack([self._last_frame, samples])
            pos_offset = 1.0
        else:
            extended = samples
            pos_offset = 0.0

        # --- 计算输出位置 ---
        step = self.input_rate / self.output_rate   # 每个输出采样对应的输入采样步长
        first_pos = self._frac_pos + pos_offset     # 扩展数组中的起始位置
        last_valid = num_frames - 1 + pos_offset    # 扩展数组中最后一个有效输入位置

        num_output = max(0, int((last_valid - first_pos) / step) + 1)
        if num_output == 0:
            self._last_frame = samples[-1:].copy()
            self._frac_pos = self._frac_pos + num_frames  # 未产出但消费了输入
            return b""

        positions = first_pos + np.arange(num_output, dtype=np.float64) * step
        positions = np.clip(positions, 0, len(extended) - 1)

        # --- 线性插值（每个声道独立） ---
        x = np.arange(len(extended), dtype=np.float64)
        output = np.empty((num_output, self.channels), dtype=np.float64)
        for c in range(self.channels):
            output[:, c] = np.interp(positions, x, extended[:, c])

        # --- 裁剪并转回整型 ---
        np.clip(output, self._clip_min, self._clip_max, out=output)
        result = output.astype(self._dtype)

        # --- 更新状态 ---
        self._last_frame = samples[-1:].copy()
        next_pos = positions[-1] + step             # 下一个输出的位置（扩展数组坐标系）
        self._frac_pos = next_pos - pos_offset - num_frames  # 相对于下一 chunk 起始的残余

        return result.tobytes()
