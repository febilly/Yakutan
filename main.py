"""
Yakutan 主入口 - 仅负责服务编排和生命周期管理

所有业务逻辑已拆分至：
- app_state.py          : 集中管理运行时状态
- text_processor.py     : 假名/拼音标注、双语裁剪、显示格式化
- translation_pipeline.py : 翻译器初始化、API 注册表、翻译执行
- audio_capture.py      : PyAudio 初始化、音频流管理、音频捕获
- recognition_handler.py : 语音识别回调（VRChatRecognitionCallback）
"""
import os
import logging
import signal
import asyncio
from typing import Callable, Optional

# Allow PyTorch and DirectML/ONNX stacks to coexist in one process.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from dotenv import load_dotenv

from hot_words_manager import HotWordsManager
from proxy_detector import apply_system_proxy, detect_system_proxy, print_proxy_info
from speech_recognizers.recognizer_factory import (
    init_dashscope_api_key,
    create_recognizer,
    select_backend,
)

# 导入配置
import config

# 加载 .env 文件中的环境变量
load_dotenv()

from osc_manager import osc_manager
from ipc_client import IPCClient

# ---- 新模块 ----
from app_state import AppState, get_state, set_state
from streaming_translation import (
    clear_translation_contexts,
    config_from_module,
    reinitialize_translator,
    update_secondary_translator,
)
from streaming_translation.pipeline import (
    _is_primary_config_changed as _is_primary_translator_config_changed,
)
from audio_capture import init_audio_stream, close_audio_stream, audio_capture_task
from recognition_handler import (
    VRChatRecognitionCallback,
    PAUSE_RESUME_BACKENDS,
    is_effective_mic_control_enabled,
    is_doubao_file_backend,
)

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ---- 语言检测器工厂 ----
def _create_language_detector():
    """根据当前配置创建语言检测器实例。"""
    if config.LANGUAGE_DETECTOR_TYPE == 'fasttext':
        from language_detectors.fasttext_detector import FasttextDetector
        return FasttextDetector()
    elif config.LANGUAGE_DETECTOR_TYPE == 'enzh':
        from language_detectors.enzh_detector import EnZhDetector
        return EnZhDetector()
    else:  # 默认使用 cjke
        from language_detectors.cjke_detector import CJKEDetector
        return CJKEDetector()


# ============ 向后兼容：供 ui/app.py 等外部模块访问 ============
# ui/app.py 通过 `import main; main.subtitles_state` 访问字幕状态
# 以及 `main.reinitialize_translator` 触发翻译器热重载
# 以下属性保持向后兼容

@property
def _compat_subtitles_state():
    state = get_state()
    if state:
        return state.subtitles_state
    return {"original": "", "translated": "", "reverse_translated": "", "ongoing": False}

# 使用模块级变量做兼容桥接
subtitles_state = {"original": "", "translated": "", "reverse_translated": "", "ongoing": False}
stop_event = None  # 由 main() 设置


def _sync_subtitles_to_module():
    """将 AppState 中的字幕状态同步到模块级变量（供 ui/app.py 轮询读取）。"""
    state = get_state()
    if state:
        subtitles_state.update(state.subtitles_state)


def update_subtitles(original: str, translated: str, ongoing: bool, reverse_translated: str = ""):
    """兼容旧接口的 update_subtitles。"""
    state = get_state()
    if state:
        state.update_subtitles(original, translated, ongoing, reverse_translated)
        _sync_subtitles_to_module()
    else:
        subtitles_state["original"] = original
        subtitles_state["translated"] = translated
        subtitles_state["reverse_translated"] = reverse_translated
        subtitles_state["ongoing"] = ongoing


def reinitialize_translator_compat():
    state = get_state()
    if state:
        cfg = config_from_module(config)
        if _is_primary_translator_config_changed(state, cfg):
            reinitialize_translator(state, cfg)
        else:
            update_secondary_translator(state, cfg)
        _refresh_ipc_translator_reference(state)
        loop = state.main_loop
        if loop is not None and loop.is_running():
            loop.create_task(osc_manager.apply_runtime_config(app_name="Yakutan"))


def clear_translator_contexts_compat():
    state = get_state()
    if state:
        cleared = clear_translation_contexts(state)
        if cleared:
            logger.info('[Translator] Cleared %s context buffer(s)', cleared)


def _refresh_ipc_translator_reference(state):
    ipc_client = getattr(osc_manager, '_ipc_client', None)
    if ipc_client is None:
        return
    setter = getattr(ipc_client, 'set_translator', None)
    if callable(setter):
        setter(state.translator)


# ============ 识别控制 ============

async def stop_recognition_async(state):
    """异步暂停或停止识别服务"""
    if not state.recognition_active:
        return

    loop = asyncio.get_event_loop()
    state.recognition_active = False

    try:
        await loop.run_in_executor(state.executor, state.recognition_instance.pause)
    except Exception:
        pass


async def start_recognition_async(state):
    """异步开始或恢复识别服务"""
    if state.recognition_active:
        print('Recognition already active.')
        return

    loop = asyncio.get_event_loop()

    try:
        if state.current_asr_backend in PAUSE_RESUME_BACKENDS and state.recognition_started:
            await loop.run_in_executor(state.executor, state.recognition_instance.resume)
        else:
            await loop.run_in_executor(state.executor, state.recognition_instance.start)
            state.recognition_started = True
    except Exception as e:
        state.recognition_active = False
        print(f'[ASR] 启动识别失败: {e}')
        raise

    state.recognition_active = True


async def handle_mute_change(state, is_muted):
    """处理静音状态变化的回调函数"""
    if not is_effective_mic_control_enabled(state.current_asr_backend):
        return

    if state.recognition_instance is None:
        print('[ASR] 识别实例未初始化')
        return

    stop_word = '暂停' if state.current_asr_backend in PAUSE_RESUME_BACKENDS else '停止'
    start_word = (
        '恢复'
        if state.current_asr_backend in PAUSE_RESUME_BACKENDS and state.recognition_started
        else '开始'
    )

    if is_muted:
        if state.recognition_active:
            if state.recognition_callback is not None:
                state.recognition_callback.mark_mute_finalization_requested()
            if state.mute_delay_task and not state.mute_delay_task.done():
                state.mute_delay_task.cancel()

            if config.MUTE_DELAY_SECONDS > 0:
                print(f'[ASR] 检测到静音，将在 {config.MUTE_DELAY_SECONDS} 秒后{stop_word}语音识别...')

                async def delayed_stop():
                    try:
                        await asyncio.sleep(config.MUTE_DELAY_SECONDS)
                        if state.recognition_active:
                            print(f'[ASR] 延迟时间到，{stop_word}语音识别')
                            await stop_recognition_async(state)
                            logger.info('[ASR] 语音识别已%s', stop_word)
                    except asyncio.CancelledError:
                        print('[ASR] 停止识别已取消（取消静音）')

                state.mute_delay_task = asyncio.create_task(delayed_stop())
            else:
                print(f'[ASR] 检测到静音，立即{stop_word}语音识别...')
                await stop_recognition_async(state)
                logger.info('[ASR] 语音识别已%s', stop_word)
    else:
        if state.recognition_callback is not None:
            state.recognition_callback.clear_mute_finalization_requested()
        if state.mute_delay_task and not state.mute_delay_task.done():
            state.mute_delay_task.cancel()
            print('[ASR] 检测到取消静音，已取消延迟停止任务')

        if not state.recognition_active:
            print(f'[ASR] 检测到取消静音，{start_word}语音识别...')
            await start_recognition_async(state)
            logger.info('[ASR] 语音识别已%s', start_word)


def _make_mute_callback(state):
    """创建同步桥接回调：将 OSC 线程中的静音事件安全投递到主事件循环。"""
    def handle_mute_change_sync(is_muted):
        loop = state.main_loop
        if loop is None or not loop.is_running():
            logger.warning('[ASR] 主事件循环不可用，忽略静音状态变化')
            return
        try:
            asyncio.run_coroutine_threadsafe(handle_mute_change(state, is_muted), loop)
        except Exception as e:
            logger.error('[ASR] 投递静音状态变化失败: %s', e)
    return handle_mute_change_sync


def signal_handler(sig, frame):
    print('Ctrl+C pressed, stop recognition ...')
    state = get_state()
    if state and state.stop_event is not None:
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(state.stop_event.set)
        except Exception:
            state.stop_event.set()


# ============ 主入口 ============

async def main(
    keep_oscquery_alive: bool = False,
    lifecycle_callback: Optional[Callable[[str, Optional[bool]], None]] = None,
):
    """主异步函数"""
    global subtitles_state, stop_event

    def emit_lifecycle(lifecycle: str, recognition_active: Optional[bool] = None):
        if lifecycle_callback is not None:
            lifecycle_callback(lifecycle, recognition_active)

    # 创建并注册 AppState
    state = AppState()
    set_state(state)
    emit_lifecycle('starting', False)

    state.update_subtitles("", "", False)
    _sync_subtitles_to_module()

    state.main_loop = asyncio.get_running_loop()
    state.stop_event = asyncio.Event()
    stop_event = state.stop_event  # 向后兼容

    state.ensure_executor()
    state.ensure_audio_executor()

    corpus_text: Optional[str] = None

    # 检测并应用系统代理设置
    system_proxies = apply_system_proxy(detect_system_proxy())
    print_proxy_info(system_proxies)

    # 初始化 DashScope API Key
    init_dashscope_api_key()
    print('Initializing ...')

    # 选择可用的识别后端
    backend = select_backend(config.PREFERRED_ASR_BACKEND, config.VALID_ASR_BACKENDS)
    if backend != config.PREFERRED_ASR_BACKEND:
        print(f'[ASR] 已切换语音识别后端为 {backend}')
    else:
        print(f'[ASR] 目标识别后端: {backend}')

    state.current_asr_backend = backend
    state.recognition_active = False
    state.recognition_started = False

    # 初始化语言检测器
    state.language_detector = _create_language_detector()

    # 初始化翻译器
    cfg = config_from_module(config)
    reinitialize_translator(state, cfg)

    # 初始化热词（在线：qwen 语料 / dashscope 热词表；本地：Qwen3-ASR 走与在线 Qwen 相同的语料注入）
    if config.ENABLE_HOT_WORDS and backend in {'qwen', 'dashscope', 'local'}:
        print('\n[热词] 初始化热词资源...')
        try:
            hot_words_manager = HotWordsManager()
            hot_words_manager.load_all_hot_words()
            if backend == 'qwen':
                words = [
                    entry.get('text')
                    for entry in hot_words_manager.get_hot_words()
                    if entry.get('text')
                ]
                if words:
                    corpus_text = "\n".join(words)
                    print(f'[热词] 已生成 Qwen 语料文本，共 {len(words)} 条\n')
                else:
                    print('[热词] 未加载到热词条目，跳过 Qwen 语料配置\n')
            elif backend == 'local':
                words = [
                    entry.get('text')
                    for entry in hot_words_manager.get_hot_words()
                    if entry.get('text')
                ]
                local_engine = getattr(config, 'LOCAL_ASR_ENGINE', 'sensevoice')
                if words and local_engine == 'qwen3-asr':
                    corpus_text = "\n".join(words)
                    print(f'[热词] 已生成本地 Qwen3-ASR 语料文本，共 {len(words)} 条\n')
                elif words:
                    print(
                        f'[热词] 已加载 {len(words)} 条热词；当前本地引擎为 {local_engine}，'
                        '仅 qwen3-asr 会使用语料注入\n'
                    )
                else:
                    print('[热词] 未加载到热词条目，跳过本地语料配置\n')
            else:
                state.vocabulary_id = hot_words_manager.create_vocabulary(
                    target_model='fun-asr-realtime',
                )
                print(f'[热词] 热词表创建成功，ID: {state.vocabulary_id}\n')
        except Exception as e:
            print(f'[热词] 热词初始化失败: {e}')
            print('[热词] 将继续运行但不使用热词\n')
            state.vocabulary_id = None
            corpus_text = None

    ipc_client = None
    if getattr(config, 'IPC_ENABLED', True):
        ipc_client = IPCClient(translator=state.translator)
        osc_manager.set_ipc_client(ipc_client)
        asyncio.create_task(ipc_client.start())
    else:
        print('[IPC] IPC is disabled in config, using standalone mode')

    # 启动 OSC 服务器
    print('[OSC] 启动OSC服务器...')
    await osc_manager.start_server(app_name="Yakutan")

    # 设置静音状态回调
    osc_manager.set_mute_callback(_make_mute_callback(state))
    print('[OSC] 已设置静音状态回调')

    # 创建识别回调
    callback = VRChatRecognitionCallback(state)
    callback.loop = asyncio.get_event_loop()
    state.recognition_callback = callback

    # 使用工厂创建识别实例
    state.recognition_instance = create_recognizer(
        backend=backend,
        callback=callback,
        sample_rate=config.SAMPLE_RATE,
        audio_format=config.FORMAT_PCM,
        source_language=config.SOURCE_LANGUAGE,
        vocabulary_id=state.vocabulary_id,
        corpus_text=corpus_text,
        enable_vad=config.ENABLE_VAD,
        vad_threshold=config.VAD_THRESHOLD,
        vad_silence_duration_ms=config.VAD_SILENCE_DURATION_MS,
        keepalive_interval=config.KEEPALIVE_INTERVAL,
    )

    if state.vocabulary_id and backend == 'dashscope':
        print(f'[ASR] 使用热词表: {state.vocabulary_id}')

    if backend == 'qwen':
        vad_status = '启用' if config.ENABLE_VAD else '禁用'
        print(f'[ASR] VAD状态: {vad_status}')
        if config.ENABLE_VAD:
            print(f'[ASR] VAD配置: 阈值={config.VAD_THRESHOLD}, 静音时长={config.VAD_SILENCE_DURATION_MS}ms')

        if config.KEEPALIVE_INTERVAL > 0:
            print(f'[ASR] WebSocket心跳已启用: 间隔={config.KEEPALIVE_INTERVAL}秒')
        else:
            print('[ASR] WebSocket心跳已禁用')

    print('[ASR] 识别实例已创建')

    # 初始化音频流
    await init_audio_stream(state)

    # 只在主线程中设置信号处理器
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    # 根据配置决定是否立即启动识别
    effective_mic_control = is_effective_mic_control_enabled(state.current_asr_backend)

    if effective_mic_control:
        if backend == 'doubao_file' and not config.ENABLE_MIC_CONTROL:
            print('[模式] 豆包文件转录已强制启用"游戏静音时暂停转录"（仅运行时生效）')
        stop_hint = '暂停' if backend in PAUSE_RESUME_BACKENDS else '停止'
        resume_hint = '恢复' if backend in PAUSE_RESUME_BACKENDS else '开始'
        print("=" * 60)
        print("[模式] 麦克风控制模式已启用")
        print("等待VRChat静音状态变化...")
        print(f"取消静音(MuteSelf=False)将{resume_hint}语音识别")
        print(f"启用静音(MuteSelf=True)将{stop_hint}语音识别")
        print("按 'Ctrl+C' 退出程序")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[模式] 麦克风控制模式已禁用")
        print("语音识别将立即启动，忽略麦克风开关状态")
        print("按 'Ctrl+C' 退出程序")
        print("=" * 60)
        await start_recognition_async(state)
        print('[ASR] 语音识别已启动')

    # 创建音频捕获任务
    emit_lifecycle('running', state.recognition_active)

    capture_task = asyncio.create_task(
        audio_capture_task(state, state.recognition_instance)
    )

    # ---- 字幕状态同步任务 ----
    async def _subtitles_sync_loop():
        """定期将 AppState.subtitles_state 同步到模块级变量。"""
        try:
            while not state.stop_event.is_set():
                _sync_subtitles_to_module()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    sync_task = asyncio.create_task(_subtitles_sync_loop())

    try:
        await state.stop_event.wait()
        emit_lifecycle('stopping', state.recognition_active)

        capture_task.cancel()
        sync_task.cancel()

        try:
            await asyncio.wait_for(capture_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        if state.recognition_active:
            await stop_recognition_async(state)
            halt_word = 'paused' if state.current_asr_backend in PAUSE_RESUME_BACKENDS else 'stopped'
            print(f'Recognition {halt_word}.')

        if state.recognition_instance:
            loop = asyncio.get_event_loop()
            try:
                request_id = await loop.run_in_executor(
                    state.executor, state.recognition_instance.get_last_request_id,
                )
                first_delay = await loop.run_in_executor(
                    state.executor, state.recognition_instance.get_first_package_delay,
                )
                last_delay = await loop.run_in_executor(
                    state.executor, state.recognition_instance.get_last_package_delay,
                )
                print(
                    '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
                    .format(request_id, first_delay, last_delay)
                )
            except Exception as e:
                print(f'[Metric] 获取统计信息失败: {e}')

    finally:
        emit_lifecycle('stopping', state.recognition_active)
        clear_translator_contexts_compat()
        osc_manager.clear_mute_callback()
        osc_manager.reset_runtime_state()
        if ipc_client is not None:
            await ipc_client.stop()
        osc_manager.clear_ipc_client()

        loop = asyncio.get_event_loop()

        if state.recognition_instance:
            try:
                await loop.run_in_executor(state.executor, state.recognition_instance.stop)
            except Exception:
                pass
            state.recognition_started = False
            state.recognition_active = False

        await close_audio_stream(state)

        if not keep_oscquery_alive:
            await osc_manager.stop_server()

        await loop.run_in_executor(None, state.audio_executor.shutdown, True)
        await loop.run_in_executor(None, state.executor.shutdown, False)
        emit_lifecycle('stopped', False)


# main function
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nProgram terminated by user.')
    finally:
        print('Cleanup completed.')
