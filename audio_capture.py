"""
音频采集模块 - 负责 PyAudio 初始化、音频流管理、重采样和音频捕获任务
"""
import asyncio
import collections
import logging
import time
from typing import Optional

import numpy as np
import pyaudio

import config
from audio_resampler import AudioResampler
from audio_debug_recorder import WaveDebugRecorder
from audio_runtime_guard import hold_portaudio, _suppress_stderr

logger = logging.getLogger(__name__)

RECOGNIZER_CHANNELS = 1
ASR_SEND_QUEUE_SECONDS = 3.0
ASR_SEND_QUEUE_MIN_FRAMES = 10
ASR_MAX_AUDIO_FRAME_BYTES = 16 * 1024
PCM16_SAMPLE_BYTES = 2
DASHSCOPE_SERVER_VAD_BACKENDS = frozenset({'qwen_audio3', 'dashscope'})


class _SilenceBurst:
    """A synthetic PCM16 silence task expanded by the sender worker."""

    def __init__(self, total_bytes: int) -> None:
        self.total_bytes = max(0, int(total_bytes))


def _vad_gate_tail_seconds(backend: str) -> float:
    """Return the unpaced silence needed after the local VAD falling edge.

    DashScope Recognition stops receiving real microphone silence as soon as
    the local gate closes. Qwen-Audio-3.0 and Fun-ASR therefore need a complete
    copy of their server-side VAD window, plus the configured safety margin.
    Other online backends retain the legacy margin-only behavior.
    """
    margin_seconds = max(
        0.0,
        float(getattr(config, 'ONLINE_VAD_END_BURST_MS', 200) or 0) / 1000.0,
    )
    if backend not in DASHSCOPE_SERVER_VAD_BACKENDS:
        return margin_seconds

    remote_silence_seconds = config.clamp_vad_silence_duration(
        getattr(config, 'LOCAL_VAD_SILENCE_DURATION', 0.8)
    )
    return remote_silence_seconds + margin_seconds


def _make_silence_burst(seconds: float, sample_rate: int) -> _SilenceBurst:
    sample_count = max(0, int(round(float(seconds) * max(1, int(sample_rate)))))
    return _SilenceBurst(sample_count * PCM16_SAMPLE_BYTES * RECOGNIZER_CHANNELS)


def _iter_silence_burst_frames(burst: _SilenceBurst):
    """Split a silence task into API-safe, sample-aligned frames."""
    frame_alignment = PCM16_SAMPLE_BYTES * RECOGNIZER_CHANNELS
    frame_bytes = ASR_MAX_AUDIO_FRAME_BYTES - (ASR_MAX_AUDIO_FRAME_BYTES % frame_alignment)
    remaining = burst.total_bytes - (burst.total_bytes % frame_alignment)
    while remaining > 0:
        chunk_bytes = min(frame_bytes, remaining)
        yield b'\x00' * chunk_bytes
        remaining -= chunk_bytes


async def _send_queue_payload(state, recognizer, generation: int, payload) -> None:
    frames = (
        _iter_silence_burst_frames(payload)
        if isinstance(payload, _SilenceBurst)
        else (payload,)
    )
    for frame in frames:
        if not (
            state.recognition_active
            and generation == getattr(state, 'audio_send_generation', 0)
        ):
            break
        # Synthetic tail frames are intentionally sent back-to-back: server
        # VAD needs audio duration, not wall-clock pacing.
        await send_audio_frame_async(state, recognizer, frame)


async def _run_recognizer_control_async(state, recognizer, method_name: str) -> bool:
    """Run pause/resume in the serialized ASR executor and surface failures."""
    state.ensure_asr_send_executor()
    method = getattr(recognizer, method_name)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(state.asr_send_executor, method)
        return True
    except Exception as exc:
        print(f'[VAD] ASR {method_name} 失败: {exc}')
        return False


async def init_audio_stream(state):
    """异步初始化音频流。

    Args:
        state: AppState 实例，音频设备信息将直接设置到其属性上。
    """
    loop = asyncio.get_event_loop()
    state.ensure_audio_executor()
    state.audio_closing = False

    def _init():
        with hold_portaudio("init_audio_stream"):
            with _suppress_stderr():
                state.mic = pyaudio.PyAudio()
            device_index = getattr(config, 'MIC_DEVICE_INDEX', None)
            target_rate = int(config.SAMPLE_RATE)
            target_channels = RECOGNIZER_CHANNELS

            def _get_device_info(idx: Optional[int]) -> Optional[dict]:
                try:
                    return (
                        state.mic.get_device_info_by_index(int(idx))
                        if idx is not None
                        else state.mic.get_default_input_device_info()
                    )
                except Exception:
                    return None

            def _resolve_capture_channels(idx: Optional[int]) -> int:
                requested_channels = target_channels
                info = _get_device_info(idx)
                if not info:
                    return requested_channels

                max_in = int(info.get('maxInputChannels', 0) or 0)
                device_name = str(info.get('name', '') or '')
                normalized_name = device_name.lower()
                is_virtual_mixer = 'voicemeeter' in normalized_name or 'vb-audio' in normalized_name

                if requested_channels == 1 and is_virtual_mixer and max_in >= 2:
                    print(f'[Audio] 检测到虚拟混音输入 {device_name}，将按 2 声道采集后在程序内转换为单声道')
                    return 2

                if max_in > 0:
                    return max(1, min(requested_channels, max_in))
                return requested_channels

            def _open_with(rate: int, frames_per_buffer: int, idx: Optional[int], channels: int):
                kwargs = dict(
                    format=pyaudio.paInt16,
                    channels=int(channels),
                    rate=int(rate),
                    input=True,
                    frames_per_buffer=int(frames_per_buffer),
                )
                if idx is not None:
                    kwargs['input_device_index'] = int(idx)
                return state.mic.open(**kwargs)

            def _get_device_default_rate(idx: Optional[int]) -> Optional[int]:
                info = _get_device_info(idx)
                if info:
                    r = info.get('defaultSampleRate')
                    if r:
                        return int(round(float(r)))
                return None

            def _init_resampler(in_rate: int, out_rate: int):
                if in_rate != out_rate:
                    state.resampler = AudioResampler(
                        input_rate=in_rate,
                        output_rate=out_rate,
                        channels=target_channels,
                        sample_width=2,
                    )
                else:
                    state.resampler = None

            def _init_debug_audio_recorders(in_rate: int, out_rate: int):
                if state.debug_pre_audio_recorder is not None:
                    state.debug_pre_audio_recorder.close()
                    state.debug_pre_audio_recorder = None

                if state.debug_audio_recorder is not None:
                    state.debug_audio_recorder.close()
                    state.debug_audio_recorder = None

                output_dir = getattr(config, 'DEBUG_AUDIO_OUTPUT_DIR', 'debug_audio')

                if getattr(config, 'SAVE_PRE_RESAMPLE_AUDIO', False):
                    try:
                        state.debug_pre_audio_recorder = WaveDebugRecorder(
                            output_dir=output_dir,
                            input_rate=in_rate,
                            sample_rate=in_rate,
                            channels=int(state.capture_channels),
                            sample_width=2,
                            file_prefix='pre_resample',
                        )
                        print(f'[Audio] 正在录制重采样前的音频: {state.debug_pre_audio_recorder.file_path}')
                    except Exception as e:
                        state.debug_pre_audio_recorder = None
                        print(f'[Audio] 无法初始化重采样前调试录音: {e}')

                if getattr(config, 'SAVE_POST_RESAMPLE_AUDIO', False):
                    try:
                        state.debug_audio_recorder = WaveDebugRecorder(
                            output_dir=output_dir,
                            input_rate=in_rate,
                            sample_rate=out_rate,
                            channels=target_channels,
                            sample_width=2,
                            file_prefix='post_resample',
                        )
                        print(f'[Audio] 正在录制重采样后的音频: {state.debug_audio_recorder.file_path}')
                    except Exception as e:
                        state.debug_audio_recorder = None
                        print(f'[Audio] 无法初始化重采样后调试录音: {e}')

            # 先尝试按 16k（ASR 期望）打开；失败则按设备默认采样率打开并重采样
            dev_idx = int(device_index) if device_index is not None else None
            try:
                state.input_sample_rate = target_rate
                state.input_block_size = int(config.BLOCK_SIZE)
                state.capture_channels = _resolve_capture_channels(dev_idx)
                _init_resampler(target_rate, target_rate)
                state.stream = _open_with(
                    target_rate, state.input_block_size, dev_idx, state.capture_channels,
                )
            except Exception as e_open_16k:
                device_rate = _get_device_default_rate(dev_idx)

                if device_rate is not None and device_rate > 0 and device_rate != target_rate:
                    scaled_block = max(256, int(round(config.BLOCK_SIZE * (device_rate / target_rate))))
                    try:
                        state.input_sample_rate = int(device_rate)
                        state.input_block_size = int(scaled_block)
                        state.capture_channels = _resolve_capture_channels(dev_idx)
                        _init_resampler(state.input_sample_rate, target_rate)
                        state.stream = _open_with(
                            state.input_sample_rate, state.input_block_size,
                            dev_idx, state.capture_channels,
                        )
                        print(
                            f"[Audio] 设备不支持 {target_rate}Hz，"
                            f"已使用 {state.input_sample_rate}Hz / {state.capture_channels}ch 采集并实时重采样"
                        )
                    except Exception as e_open_device_rate:
                        try:
                            state.input_sample_rate = target_rate
                            state.input_block_size = int(config.BLOCK_SIZE)
                            state.capture_channels = _resolve_capture_channels(None)
                            _init_resampler(target_rate, target_rate)
                            state.stream = _open_with(
                                target_rate, state.input_block_size, None, state.capture_channels,
                            )
                            print(f"[Audio] 指定麦克风设备不可用，已回退到系统默认：{e_open_device_rate}")
                        except Exception:
                            raise
                else:
                    try:
                        state.input_sample_rate = target_rate
                        state.input_block_size = int(config.BLOCK_SIZE)
                        state.capture_channels = _resolve_capture_channels(None)
                        _init_resampler(target_rate, target_rate)
                        state.stream = _open_with(
                            target_rate, state.input_block_size, None, state.capture_channels,
                        )
                        print(f"[Audio] 指定麦克风设备不可用，已回退到系统默认：{e_open_16k}")
                    except Exception:
                        raise

            print(f'[Audio] 实际采集格式: {state.input_sample_rate}Hz / {state.capture_channels}ch / 16-bit')
            if state.capture_channels != target_channels:
                print(f'[Audio] 发送给识别器前将转换为: {target_rate}Hz / {target_channels}ch / 16-bit')

            _init_debug_audio_recorders(state.input_sample_rate, target_rate)
            return state.stream

    return await loop.run_in_executor(state.audio_executor, _init)


async def close_audio_stream(state):
    """异步关闭音频流。"""
    loop = asyncio.get_event_loop()
    state.audio_closing = True

    def _close():
        with hold_portaudio("close_audio_stream"):
            if state.stream:
                state.stream.stop_stream()
                state.stream.close()
            if state.mic:
                state.mic.terminate()
            if state.debug_pre_audio_recorder:
                saved_file = state.debug_pre_audio_recorder.file_path
                state.debug_pre_audio_recorder.close()
                print(f'[Audio] 重采样前的音频已保存到: {saved_file}')
                state.debug_pre_audio_recorder = None
            if state.debug_audio_recorder:
                saved_file = state.debug_audio_recorder.file_path
                state.debug_audio_recorder.close()
                print(f'[Audio] 重采样后的音频已保存到: {saved_file}')
                state.debug_audio_recorder = None
            state.stream = None
            state.mic = None

    await loop.run_in_executor(state.audio_executor, _close)


async def read_audio_data(state):
    """异步读取音频数据。"""
    if state.audio_closing or not state.stream:
        return None

    loop = asyncio.get_event_loop()

    def _read():
        try:
            if state.audio_closing or state.stream is None:
                return None
            data = state.stream.read(state.input_block_size, exception_on_overflow=False)
            if not data:
                return None
            if state.debug_pre_audio_recorder is not None:
                state.debug_pre_audio_recorder.write(data)

            capture_data = data
            if state.capture_channels != RECOGNIZER_CHANNELS:
                samples = np.frombuffer(data, dtype=np.int16)
                num_frames = len(samples) // int(state.capture_channels)
                if num_frames <= 0:
                    return b''
                frames = (
                    samples[: num_frames * int(state.capture_channels)]
                    .reshape(num_frames, int(state.capture_channels))
                    .astype(np.int32)
                )
                mono = np.rint(np.mean(frames, axis=1))
                mono = np.clip(mono, -32768, 32767).astype(np.int16)
                capture_data = mono.tobytes()

            processed_data = (
                state.resampler.resample(capture_data)
                if state.resampler is not None
                else capture_data
            )
            if state.debug_audio_recorder is not None and processed_data:
                state.debug_audio_recorder.write(processed_data)
            return processed_data
        except Exception as e:
            print(f'Error reading audio data: {e}')
            return None

    return await loop.run_in_executor(state.audio_executor, _read)


async def send_audio_frame_async(state, recognizer, data: bytes):
    """异步发送音频帧。"""
    loop = asyncio.get_event_loop()
    state.ensure_asr_send_executor()
    try:
        await loop.run_in_executor(state.asr_send_executor, recognizer.send_audio_frame, data)
    except Exception:
        pass


def _asr_send_queue_maxsize() -> int:
    sample_rate = max(1, int(getattr(config, 'SAMPLE_RATE', 16000) or 16000))
    block_size = max(1, int(getattr(config, 'BLOCK_SIZE', 1600) or 1600))
    block_seconds = block_size / sample_rate
    return max(ASR_SEND_QUEUE_MIN_FRAMES, int(ASR_SEND_QUEUE_SECONDS / block_seconds))


def _drop_oldest_queue_item(queue: asyncio.Queue) -> None:
    try:
        queue.get_nowait()
        queue.task_done()
    except asyncio.QueueEmpty:
        pass


class GatePreBuffer:
    """采集侧门控的滚动音频预缓冲。

    门控在静音期会丢弃音频帧，导致开口瞬间的前段语音丢失；本缓冲持续保留
    最近一段时间（字节上限）的音频帧，检测到开口时由调用方整体 flush 补发，
    避免漏掉句首。
    """

    def __init__(self, cap_bytes: int) -> None:
        self._cap_bytes = max(0, int(cap_bytes))
        self._frames: "collections.deque[bytes]" = collections.deque()
        self._total_bytes = 0

    def push(self, frame: bytes) -> None:
        if self._cap_bytes <= 0 or not frame:
            return
        self._frames.append(frame)
        self._total_bytes += len(frame)
        while self._total_bytes > self._cap_bytes and len(self._frames) > 1:
            self._total_bytes -= len(self._frames.popleft())

    def flush(self) -> list:
        if not self._frames:
            return []
        frames = list(self._frames)
        self.clear()
        return frames

    def clear(self) -> None:
        self._frames.clear()
        self._total_bytes = 0

    def __len__(self) -> int:
        return len(self._frames)

    def __bool__(self) -> bool:
        return bool(self._frames)


async def audio_capture_task(state, recognizer):
    """异步音频捕获任务。"""
    print('Starting audio capture...')
    send_queue: asyncio.Queue = asyncio.Queue(maxsize=_asr_send_queue_maxsize())
    last_queue_warning_at = 0.0

    async def _sender_worker():
        while True:
            generation, payload = await send_queue.get()
            try:
                await _send_queue_payload(state, recognizer, generation, payload)
            finally:
                send_queue.task_done()

    sender_task = asyncio.create_task(_sender_worker())

    # VAD 侧路 chunk 大小（Silero 16kHz 固定）
    _vad_chunk_samples = 512
    _vad_chunk_count = 0
    _vad_last_diag_at = 0.0
    _vad_verbose = bool(getattr(config, 'ENABLE_VAD_GATING_VERBOSE', False))
    # 本地门控停止转发真实静音后，补发服务端判停所需的合成静音。
    # DashScope Recognition 后端需要完整判停窗口；其他后端仅保留安全余量。
    _vad_tail_seconds = _vad_gate_tail_seconds(state.current_asr_backend)
    _vad_end_burst = _make_silence_burst(
        _vad_tail_seconds,
        int(getattr(config, 'SAMPLE_RATE', 16000) or 16000),
    )
    # DashScope Recognition 的本地 VAD 断句直接复用游戏内闭麦已验证的
    # pause/stop -> resume/start 生命周期，不再依赖服务端接受突发合成静音。
    _vad_rolls_recognition_session = (
        state.current_asr_backend in DASHSCOPE_SERVER_VAD_BACKENDS
    )
    _vad_recognition_paused = False
    # 防止切换后端或一次短促误触发时，结束一个尚未送入真实音频的
    # Recognition task；DashScope 会对此返回 EmptyAudio。
    _vad_recognition_has_audio = False
    # 门控预缓冲：静音期门控丢弃的音频先滚动保留，开口瞬间整体补发给 ASR，
    # 避免漏掉句首（时长由 config.VAD_PRE_SPEECH_DURATION 统一控制）
    _vad_pre_buffer_seconds = max(
        0.0, float(getattr(config, 'VAD_PRE_SPEECH_DURATION', 0.5) or 0.0)
    )
    _vad_pre_buffer = GatePreBuffer(
        int(_vad_pre_buffer_seconds * int(getattr(config, 'SAMPLE_RATE', 16000) or 16000) * 2)
    )
    # 会话代次（开麦/闭麦/切后端都会 bump）：变化时清空预缓冲，
    # 防止把上一次会话残留的音频补发给新会话
    _vad_pre_generation = getattr(state, 'audio_send_generation', 0)

    # 一次性报告采集侧 VAD 门控状态。
    # 注意：这里的门控只服务于在线 API（静音时暂停发送以省流），本地识别需要连续音频供
    # 识别器内部 VAD 分段，因此本地后端下采集侧本就不门控，不再重复输出，避免被误读成
    # 「VAD 没开」。
    if state.vad_enabled and state.vad_processor is not None:
        print(f'[VAD] capture loop 已激活，等待语音...（起声预缓冲 {_vad_pre_buffer_seconds:.1f}s）')
    elif state.current_asr_backend != 'local':
        print(
            f'[VAD] capture loop — 在线 API 发送门控未启用 '
            f'(enabled={state.vad_enabled}, proc={state.vad_processor is not None})'
        )

    try:
        while not state.stop_event.is_set():
            # 始终读取音频数据,避免缓冲区积压
            data = await read_audio_data(state)
            if data is None:
                break
            if not data:
                await asyncio.sleep(0.001)
                continue

            # 会话代次变化（开麦/闭麦/切后端）：清空预缓冲，避免跨会话补发旧音频
            _generation = getattr(state, 'audio_send_generation', 0)
            if _generation != _vad_pre_generation:
                _vad_pre_generation = _generation
                _vad_pre_buffer.clear()
                _vad_recognition_paused = False
                _vad_recognition_has_audio = False

            # ── VAD 侧路分析（不阻塞主通道，且仅在识别激活时进行） ──
            if state.recognition_active and state.vad_enabled and state.vad_processor is not None:
                try:
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    pending = state._vad_pending_samples
                    if pending is not None and len(pending) > 0:
                        samples = np.concatenate([pending, samples])
                    offset = 0
                    while offset + _vad_chunk_samples <= len(samples):
                        chunk = samples[offset:offset + _vad_chunk_samples]
                        state.vad_processor.process_chunk(chunk)
                        offset += _vad_chunk_samples
                        _vad_chunk_count += 1
                        # 检测 VAD 内部状态变化
                        is_speaking = state.vad_processor.is_speaking
                        if is_speaking != state._vad_was_speaking:
                            state._vad_was_speaking = is_speaking
                            conf = state.vad_processor.last_confidence
                            if is_speaking:
                                if _vad_rolls_recognition_session and _vad_recognition_paused:
                                    resumed = await _run_recognizer_control_async(
                                        state, recognizer, 'resume'
                                    )
                                    if not resumed:
                                        # 保留起声预缓冲，并让下一帧重新触发起声，以便重试。
                                        state.vad_processor.reset()
                                        state._vad_was_speaking = False
                                        break
                                    _vad_recognition_paused = False
                                    _vad_recognition_has_audio = False
                                    print('[VAD] ▶ DashScope Recognition 新会话已开始')
                                # 开口瞬间：把门控期间扣留的预缓冲音频整体补发，
                                # 保证服务端收到句首（当前帧随后由发送块正常入队，不重复）
                                pre_frames = _vad_pre_buffer.flush()
                                if pre_frames:
                                    _vad_recognition_has_audio = True
                                    for pre_frame in pre_frames:
                                        try:
                                            send_queue.put_nowait((_generation, pre_frame))
                                        except asyncio.QueueFull:
                                            _drop_oldest_queue_item(send_queue)
                                            send_queue.put_nowait((_generation, pre_frame))
                                    if _vad_verbose:
                                        print(f'[VAD] 预缓冲补发 {len(pre_frames)} 帧（约 {_vad_pre_buffer_seconds:.1f}s）')
                                print(f'[VAD] ▶ SPEECH 开始 (chunk=#{_vad_chunk_count}, 置信度={conf:.3f})')
                            else:
                                if _vad_rolls_recognition_session:
                                    # pause/stop 会等待 SDK 内部音频队列排空并完成当前 task，
                                    # 行为与游戏内闭麦一致，可确定触发最终识别结果。
                                    if _vad_recognition_has_audio:
                                        await send_queue.join()
                                        _vad_recognition_paused = await _run_recognizer_control_async(
                                            state, recognizer, 'pause'
                                        )
                                        if _vad_recognition_paused:
                                            _vad_recognition_has_audio = False
                                            print('[VAD] ■ DashScope Recognition 当前会话已结束')
                                        elif _vad_end_burst.total_bytes:
                                            # 生命周期切换失败时保留合成静音作为降级路径。
                                            await send_queue.put((_generation, _vad_end_burst))
                                    else:
                                        print('[VAD] 跳过未送入音频的 DashScope task 结束，避免 EmptyAudio')
                                elif _vad_end_burst.total_bytes:
                                    # 其他在线后端维持原有的合成静音余量。
                                    await send_queue.put((_generation, _vad_end_burst))
                                print(f'[VAD] ■ SILENCE (chunk=#{_vad_chunk_count}, 置信度={conf:.3f})')
                        # 定期诊断（仅在 verbose 模式下显示）
                        if _vad_verbose:
                            now = time.monotonic()
                            if now - _vad_last_diag_at > _vad_diag_interval:
                                _vad_last_diag_at = now
                                conf = state.vad_processor.last_confidence
                                label = 'SPEECH' if is_speaking else 'SILENCE'
                                print(f'[VAD] diag: chunks={_vad_chunk_count}, state={label}, conf={conf:.3f}')
                    # 保留不足一个 chunk 的剩余帧
                    state._vad_pending_samples = np.array(
                        samples[offset:], dtype=np.float32, copy=True
                    )
                except Exception:
                    # VAD 错误不应中断音频流
                    import traceback
                    if _vad_verbose:
                        now = time.monotonic()
                        if now - _vad_last_diag_at > _vad_diag_interval:
                            _vad_last_diag_at = now
                            print('[VAD] ⚠ 处理异常（静默）')
                            traceback.print_exc()

            # 只有在识别激活时才发送音频数据,否则丢弃
            # VAD 门控：静音时不发送音频到 ASR（省流），说话时正常发送；
            # 说话结束瞬间的补发静音帧见上方 SPEECH→SILENCE 转换处理
            if state.recognition_active:
                _gating_active = state.vad_enabled and state.vad_processor is not None
                if _gating_active:
                    # 门控激活时每帧都进预缓冲（说话中也持续滚动），
                    # 供下一次开口时补发句首
                    _vad_pre_buffer.push(data)
                _should_send = True
                if _gating_active:
                    _should_send = (
                        state.vad_processor.is_speaking
                        and not _vad_recognition_paused
                    )
                if _should_send:
                    item = (getattr(state, 'audio_send_generation', 0), data)
                    try:
                        send_queue.put_nowait(item)
                    except asyncio.QueueFull:
                        _drop_oldest_queue_item(send_queue)
                        send_queue.put_nowait(item)
                        now = time.monotonic()
                        if now - last_queue_warning_at > 5.0:
                            print('[Audio] ASR发送队列已满，丢弃最旧音频帧以保持实时采集')
                            last_queue_warning_at = now
                    if _vad_rolls_recognition_session:
                        _vad_recognition_has_audio = True
                elif _vad_verbose:
                    state._vad_drop_count += 1
                    if state._vad_drop_count == 1 or state._vad_drop_count % 100 == 0:
                        is_sp = state.vad_processor.is_speaking if state.vad_processor else 'N/A'
                        print(f'[VAD-gate] 丢弃音频帧 (累计={state._vad_drop_count}, is_speaking={is_sp})')
            elif not send_queue.empty():
                while not send_queue.empty():
                    _drop_oldest_queue_item(send_queue)

            await asyncio.sleep(0.001)  # 避免阻塞事件循环
    except asyncio.CancelledError:
        print('Audio capture task cancelled.')
    except Exception as e:
        print(f'Audio capture error: {e}')
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        print('Audio capture stopped.')
