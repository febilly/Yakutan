"""
音频采集模块 - 负责 PyAudio 初始化、音频流管理、重采样和音频捕获任务
"""
import asyncio
import logging
from typing import Optional

import numpy as np
import pyaudio

import config
from audio_resampler import AudioResampler
from audio_debug_recorder import WaveDebugRecorder
from audio_runtime_guard import hold_portaudio, _suppress_stderr

logger = logging.getLogger(__name__)

RECOGNIZER_CHANNELS = 1


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
    try:
        await loop.run_in_executor(state.executor, recognizer.send_audio_frame, data)
    except Exception:
        pass


async def audio_capture_task(state, recognizer):
    """异步音频捕获任务。"""
    print('Starting audio capture...')
    try:
        while not state.stop_event.is_set():
            # 始终读取音频数据,避免缓冲区积压
            data = await read_audio_data(state)
            if data is None:
                break
            if not data:
                await asyncio.sleep(0.001)
                continue

            # 只有在识别激活时才发送音频数据,否则丢弃
            if state.recognition_active:
                await send_audio_frame_async(state, recognizer, data)

            await asyncio.sleep(0.001)  # 避免阻塞事件循环
    except asyncio.CancelledError:
        print('Audio capture task cancelled.')
    except Exception as e:
        print(f'Audio capture error: {e}')
    finally:
        print('Audio capture stopped.')
