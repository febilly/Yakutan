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
import time
import logging
import signal
import asyncio
from typing import Callable, Optional

# Allow PyTorch and DirectML/ONNX stacks to coexist in one process.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# 旧版纯命令行入口继续完全依靠 .env。WebUI 会以 ``import main`` 的方式
# 启动服务，此时绝不加载 .env，也不应用任何用户配置环境变量。
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

# 必须先完成 CLI 的 .env 引导，再导入任何会读取 config 的业务模块。
import config

if __name__ == '__main__':
    config.apply_cli_env()

from hot_words_manager import HotWordsManager
from proxy_detector import refresh_system_proxy_env, print_proxy_info
from speech_recognizers.recognizer_factory import (
    init_dashscope_api_key,
    create_recognizer,
    select_backend,
)

from osc_manager import osc_manager
from ipc_client import IPCClient

# ---- 新模块 ----
from app_state import AppState, get_state, set_state
from streaming_translation import (
    clear_translation_contexts,
    config_from_module,
    prewarm_local_engines,
    release_local_engines,
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
    report_recognition_error,
    get_secondary_translator_lock,
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


# 进行中的本地模型预加载 future（服务停止时先等其结束再释放模型，避免泄漏）
_local_engine_prewarm_futures: set = set()


def _schedule_local_engine_prewarm(state):
    """后台线程预载本地 Hy-MT2 模型（不阻塞主事件循环，音频采集不受影响）。"""

    def _worker():
        try:
            loaded = prewarm_local_engines(state)
            if loaded:
                print(f'[Translator] 本地 Hy-MT2 模型已就绪（供 {loaded} 个翻译器使用）')
        except Exception as e:
            print(f'[Translator] 本地 Hy-MT2 模型预加载失败: {e}')

    loop = state.main_loop
    if loop is None or not loop.is_running():
        _worker()
        return
    try:
        future = loop.run_in_executor(state.executor, _worker)
    except RuntimeError:
        _worker()
        return
    _local_engine_prewarm_futures.add(future)
    future.add_done_callback(_local_engine_prewarm_futures.discard)


def reinitialize_translator_compat():
    state = get_state()
    if state:
        # 翻译关闭时跳过（重）构建翻译器，避免未配置对应 API Key 时报错。
        # 运行时重新开启翻译会再次调用本函数并正常构建；
        # 关闭时同时释放本地模型，不再占用内存/显存。
        if not getattr(config, 'ENABLE_TRANSLATION', True):
            release_local_engines(state)
            return
        cfg = config_from_module(config)
        if _is_primary_translator_config_changed(state, cfg):
            # P2-14: 重建期间持 state 级次翻译器锁，与识别回调路径
            # （on_result / _translate_partial_task）的 ensure_secondary_translator
            # 串行化，避免并发重建导致本地引擎引用计数失衡
            with get_secondary_translator_lock(state):
                reinitialize_translator(state, cfg)
        else:
            with get_secondary_translator_lock(state):
                update_secondary_translator(state, cfg)
        _refresh_ipc_translator_reference(state)
        loop = state.main_loop
        if loop is not None and loop.is_running():
            loop.create_task(osc_manager.apply_runtime_config(app_name="Yakutan"))
        # 切换翻译模式/Hy-MT2 本地开关后，当场（后台）加载新的本地模型
        _schedule_local_engine_prewarm(state)


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

# P2-8: 延迟停止被取消静音撤销时，等待 in-flight pause（含 qwen end_session /
# dashscope stop 的网络 RTT）真正完成的时间上限；超时后交由识别器内部锁兜底。
INFLIGHT_PAUSE_WAIT_SECONDS = 10.0


async def stop_recognition_async(state):
    """异步暂停或停止识别服务"""
    if not state.recognition_active:
        return

    loop = asyncio.get_event_loop()
    state.recognition_active = False

    # pause 段含网络 RTT。用 shield 包裹：外层任务（mute_delay_task）被取消
    # 静音撤销时，CancelledError 不会中止底层 pause 线程；这里在被取消后仍
    # 限时等待其真正完成，避免后台 pause 与 unmute 触发的 resume/start 并发
    # 操作同一识别器（P2-8 取消竞态）。
    pause_future = loop.run_in_executor(state.executor, state.recognition_instance.pause)
    try:
        await asyncio.shield(pause_future)
    except asyncio.CancelledError:
        # 任务被取消（典型：用户取消静音）：先等 in-flight pause 落地，再把
        # 取消向上传播。等待期被再次取消或超时/异常时不再无限阻塞，交由
        # 识别器内部锁串行化兜底。
        try:
            await asyncio.wait_for(
                asyncio.shield(pause_future), timeout=INFLIGHT_PAUSE_WAIT_SECONDS
            )
        except asyncio.CancelledError:
            logger.warning('[ASR] 等待 in-flight 暂停完成时再次被取消，底层 pause 交由识别器内部锁串行化')
        except Exception:
            pass
        raise
    except Exception as e:
        logger.warning('[ASR] 暂停识别失败(已忽略): %s', e)

    # 闭麦时重置本地 VAD 状态与缓存，确保下一次开麦时有干净的历史。
    if state.vad_processor is not None:
        try:
            state.vad_processor.reset()
        except Exception:
            pass
    state._vad_was_speaking = False
    import numpy as np
    state._vad_pending_samples = np.array([], dtype=np.float32)

    state.bump_audio_send_generation()


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

    state.bump_audio_send_generation()
    state.recognition_active = True

    # 识别重新开始/恢复，解除"撤回作废"的结果丢弃状态
    if state.recognition_callback is not None:
        state.recognition_callback.resume_outputs()


async def handle_mute_change(state, is_muted):
    """处理静音状态变化的回调函数"""
    # 快速开关麦克风以清空消息框：短时间内连续两次收到静音消息则清空聊天框。
    # 该逻辑独立于麦克风控制开关，因此放在最前面处理。
    if is_muted and getattr(config, 'ENABLE_DOUBLE_MUTE_CLEAR', True):
        now = time.monotonic()
        window = getattr(config, 'DOUBLE_MUTE_CLEAR_WINDOW_SECONDS', 0.8)
        last = state.last_mute_engaged_time
        if last is not None and (now - last) <= window:
            print('[OSC] 检测到快速开关麦克风，清空聊天框并撤回之前的内容')
            await osc_manager.clear_chatbox()
            # 撤回作废：丢弃之前已发出的识别/翻译请求迟到返回的结果
            if state.recognition_callback is not None:
                state.recognition_callback.discard_pending_outputs()
            state.last_mute_engaged_time = None  # 重置，避免连续误触发
        else:
            state.last_mute_engaged_time = now

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
            # P2-8: 延迟停止可能已进入 in-flight pause（含网络 RTT）。delayed_stop
            # 内部对 pause 段做了 shield，被取消后仍会等 pause 完成才退出；这里
            # 限时等待该任务落地后再触发 start/resume，避免后台 pause 线程与
            # resume/start 并发操作识别器。超时则继续开麦，由识别器内部锁兜底。
            # 注：asyncio.wait 不会因任务以取消收尾而抛出（若 cancel 落在
            # delayed_stop 首步之前，任务会直接以 cancelled 结束）。
            try:
                done, _pending = await asyncio.wait(
                    [state.mute_delay_task], timeout=INFLIGHT_PAUSE_WAIT_SECONDS
                )
            except asyncio.CancelledError:
                raise
            if not done:
                logger.warning('[ASR] 等待 in-flight 停止任务超时，继续开麦（识别器内部锁兜底）')

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
            # P2-9: 保存 run_coroutine_threadsafe 返回的 future 并观察其完成
            # 状态。此前不消费 future，handle_mute_change / start_recognition_async
            # 抛出的异常（resume 失败、网络错误等）在事件层面完全静默。
            future = asyncio.run_coroutine_threadsafe(
                handle_mute_change(state, is_muted), loop,
            )
        except Exception as e:
            logger.error('[ASR] 投递静音状态变化失败: %s', e)
            return

        def _on_mute_future_done(fut, _is_muted=is_muted):
            # 在事件循环线程内执行：回调协程的异常在此落地记录与上报
            if fut.cancelled():
                logger.warning('[ASR] 静音状态处理任务被取消 (is_muted=%s)', _is_muted)
                return
            exc = fut.exception()
            if exc is not None:
                logger.warning(
                    '[ASR] 处理静音状态变化失败 (is_muted=%s): %s', _is_muted, exc,
                    exc_info=exc,
                )
                report_recognition_error(state, exc, 'mute_callback')

        future.add_done_callback(_on_mute_future_done)
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
    hot_word_entries: Optional[list] = None

    # P2-13: 后台任务引用前置为 None，初始化中途失败时 finally 也能安全引用
    ipc_client = None
    ipc_start_task = None
    capture_task = None
    sync_task = None
    mute_probe_task = None

    # P2-13: 初始化段整体纳入 try/finally —— 启动中途失败（无麦克风、后端
    # 不可用、本地模型缺失等导致异常逃逸）时同样执行清理：卸载已加载的本地
    # 翻译模型、停止 OSC、清除 mute 回调、关闭 executor，不再让资源悬挂至
    # 进程退出。正常停机路径语义保持不变（停机步骤仍在 stop_event 置位后执行）。
    try:
        # 检测并应用系统代理设置
        system_proxies = refresh_system_proxy_env()
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

        # ---- 统一 VAD：在线 API 发送门控（本地 ASR 走识别器内部 VAD，此处不处理） ----
        # 门控仅对在线后端生效：本地 ASR 需要连续音频（含静音）供其内部 VAD 分段。
        _online_backend = backend != 'local'
        _vad_gating_enabled = config.VAD_ENABLED and _online_backend

        if _vad_gating_enabled:
            try:
                from local_inference.model_manager import is_silero_cached, download_silero
                from local_inference.vad_processor import VADProcessor

                if not is_silero_cached():
                    print('[VAD] Silero ONNX 模型未下载，正在自动下载...')
                    download_silero()

                print('[VAD] Silero ONNX 模型就绪，正在初始化...')
                state.vad_processor = VADProcessor(
                    sample_rate=config.SAMPLE_RATE,
                    threshold=config.LOCAL_VAD_THRESHOLD,
                    min_speech_duration=config.LOCAL_VAD_MIN_SPEECH_DURATION,
                    chunk_duration=512.0 / config.SAMPLE_RATE,
                    pre_speech_duration=config.VAD_PRE_SPEECH_DURATION,
                )
                vad_silence_duration = config.clamp_vad_silence_duration(
                    config.LOCAL_VAD_SILENCE_DURATION
                )
                state.vad_processor.update_settings({
                    'vad_mode': 'silero',
                    'vad_threshold': config.LOCAL_VAD_THRESHOLD,
                    'min_speech_duration': config.LOCAL_VAD_MIN_SPEECH_DURATION,
                    'silence_duration': vad_silence_duration,
                    'pre_speech_duration': config.VAD_PRE_SPEECH_DURATION,
                })
                state.vad_enabled = True
                import numpy as np
                state._vad_pending_samples = np.array([], dtype=np.float32)
                state._vad_was_speaking = False
                print(f'[VAD] ✓ 在线 API VAD 发送门控已启用')
                print(f'[VAD]   threshold={state.vad_processor.threshold:.2f} '
                      f'min_speech={config.LOCAL_VAD_MIN_SPEECH_DURATION:.1f}s '
                      f'silence={vad_silence_duration:.1f}s '
                      f'(在线服务端断句阈值同步为 {int(round(vad_silence_duration * 1000))}ms) '
                      f'pre={config.VAD_PRE_SPEECH_DURATION:.1f}s '
                      f'mode={state.vad_processor.mode}')
            except Exception as e:
                import traceback
                print(f'[VAD] ✗ 初始化失败，VAD 门控未启用: {e}')
                traceback.print_exc()
                state.vad_enabled = False
        elif not config.VAD_ENABLED:
            print('[VAD] — VAD 未启用（VAD_ENABLED=False）')
        else:
            print('[VAD] — 本地 ASR 后端：门控由识别器内部 VAD 负责，采集侧不做门控')

        # 初始化翻译器（仅在启用翻译时构建；否则识别流程不会用到翻译器，
        # 且此时构建会因未配置对应 API Key 而在启动阶段直接报错）
        cfg = config_from_module(config)
        if config.ENABLE_TRANSLATION:
            reinitialize_translator(state, cfg)
            # 本地 Hy-MT2：服务启动时立即加载模型（此时尚无流量，同步加载不阻塞用户）
            if prewarm_local_engines(state):
                print('[Translator] 本地 Hy-MT2 模型已在服务启动时加载')

        # 初始化热词（在线：qwen 语料 / qwen_audio3 即时热词 / dashscope 热词表；
        # 本地：Qwen3-ASR 走与在线 Qwen 相同的语料注入）
        if config.ENABLE_HOT_WORDS and backend in {'qwen', 'qwen_audio3', 'dashscope', 'local'}:
            print('\n[热词] 初始化热词资源...')
            try:
                hot_words_manager = HotWordsManager(
                    api_key=str(getattr(config, 'DASHSCOPE_API_KEY', '') or '').strip()
                )
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
                elif backend == 'qwen_audio3':
                    # Qwen-Audio-3.1 支持即时热词，直接下发词条与权重，无需创建热词表
                    hot_word_entries = [
                        entry
                        for entry in hot_words_manager.get_hot_words()
                        if entry.get('text')
                    ]
                    if hot_word_entries:
                        print(f'[热词] 已准备 Qwen-Audio-3.1 即时热词，共 {len(hot_word_entries)} 条\n')
                    else:
                        print('[热词] 未加载到热词条目，跳过即时热词配置\n')
                elif backend == 'local':
                    words = [
                        entry.get('text')
                        for entry in hot_words_manager.get_hot_words()
                        if entry.get('text')
                    ]
                    local_engine = getattr(config, 'LOCAL_INFERENCE_ENGINE', 'sensevoice')
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
                hot_word_entries = None

        if getattr(config, 'IPC_ENABLED', True):
            ipc_client = IPCClient(translator=state.translator)
            osc_manager.set_ipc_client(ipc_client)
            # P3-26: 保存任务引用避免被 GC 中途回收；异常经 done callback
            # 记录并上报，不再只以 "never retrieved" 形式丢失
            ipc_start_task = asyncio.create_task(ipc_client.start())

            def _on_ipc_start_done(task):
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    logger.error(
                        '[IPC] IPC 客户端异常退出: %s', exc, exc_info=exc,
                    )
                    report_recognition_error(state, exc, 'ipc')

            ipc_start_task.add_done_callback(_on_ipc_start_done)
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
            hot_words=hot_word_entries,
            enable_vad=config.ENABLE_VAD,
            vad_threshold=config.VAD_THRESHOLD,
            keepalive_interval=config.KEEPALIVE_INTERVAL,
        )

        if state.vocabulary_id and backend == 'dashscope':
            print(f'[ASR] 使用热词表: {state.vocabulary_id}')

        if backend == 'qwen':
            vad_status = '启用' if config.ENABLE_VAD else '禁用'
            print(f'[ASR] VAD状态: {vad_status}')
            if config.ENABLE_VAD:
                print(f'[ASR] VAD配置: 阈值={config.VAD_THRESHOLD}, '
                      f'静音时长={int(round(config.clamp_vad_silence_duration(config.LOCAL_VAD_SILENCE_DURATION) * 1000))}ms '
                      f'(与本地 VAD 对齐)')

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

        # 主动读取一次游戏当前的静音状态，这样开着游戏中途启动也能立刻对齐，
        # 不必等玩家切换一次麦克风。放在识别实例与音频流就绪之后再探测。
        mute_probe_task = None
        if effective_mic_control:
            mute_probe_task = asyncio.create_task(osc_manager.probe_initial_mute_state())

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

        # ---- 运行段：等待停机信号（正常停机路径语义与改造前一致） ----
        await state.stop_event.wait()
        emit_lifecycle('stopping', state.recognition_active)

        capture_task.cancel()
        sync_task.cancel()
        if mute_probe_task is not None and not mute_probe_task.done():
            mute_probe_task.cancel()

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
        # P2-13: 初始化中途失败时，已创建但未进入正常停机流程的后台任务也需
        # 取消，避免悬挂到进程退出；正常停机路径下任务已在上方被取消/等待，
        # 这里的守卫取消是空操作
        for _task in (capture_task, sync_task, mute_probe_task, ipc_start_task):
            if _task is not None and not _task.done():
                _task.cancel()
        # 等待进行中的本地模型预加载结束后再统一释放，避免“加载完才释放”的泄漏
        if _local_engine_prewarm_futures:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            asyncio.wrap_future(fut)
                            for fut in list(_local_engine_prewarm_futures)
                        ),
                        return_exceptions=True,
                    ),
                    timeout=300,
                )
            except Exception as e:
                print(f'[Translator] 等待本地模型预加载结束失败: {e}')
        # 服务关闭：卸载本地翻译模型（释放内存/显存）
        released = release_local_engines(state)
        if released:
            print(f'[Translator] 本地 Hy-MT2 模型已卸载（释放 {released} 个引用）')
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
        await loop.run_in_executor(None, state.asr_send_executor.shutdown, True)
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
