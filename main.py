import logging
import os
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording)
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pyaudio

from dotenv import load_dotenv
from osc_manager import osc_manager
from translators.context_aware_translator import ContextAwareTranslator
from hot_words_manager import HotWordsManager
from proxy_detector import detect_system_proxy, print_proxy_info

from translators.translation_apis.google_dictionary_api import GoogleDictionaryAPI as BackwardsTranslationAPI
from speech_recognizers.base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
    SpeechRecognizer,
)
from speech_recognizers.recognizer_factory import (
    init_dashscope_api_key,
    create_recognizer,
    select_backend,
)

# 导入配置
import config

# 加载 .env 文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ============ 根据配置选择语言检测器 ============
if config.LANGUAGE_DETECTOR_TYPE == 'fasttext':
    from language_detectors.fasttext_detector import FasttextDetector as LanguageDetector
elif config.LANGUAGE_DETECTOR_TYPE == 'enzh':
    from language_detectors.enzh_detector import EnZhDetector as LanguageDetector
else:  # 默认使用 cjke
    from language_detectors.cjke_detector import CJKEDetector as LanguageDetector

# ============ 根据配置选择翻译 API ============
if config.TRANSLATION_API_TYPE == 'google_web':
    from translators.translation_apis.google_web_api import GoogleWebAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'google_dictionary':
    from translators.translation_apis.google_dictionary_api import GoogleDictionaryAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'openrouter':
    from translators.translation_apis.openrouter_api import OpenRouterAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'qwen_mt':
    from translators.translation_apis.qwen_mt_api import QwenMTAPI as TranslationAPI
else:  # 默认使用 deepl
    from translators.translation_apis.deepl_api import DeepLAPI as TranslationAPI

# ============ 全局变量 ============
mic = None
stream = None
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
stop_event = None  # 将在 main() 函数中创建，避免绑定到错误的事件循环
recognition_active = False  # 标记识别是否正在运行
recognition_started = False  # 标记是否已建立识别会话
recognition_instance: Optional[SpeechRecognizer] = None  # 全局识别实例
mute_delay_task = None  # 延迟停止任务
CURRENT_ASR_BACKEND = config.PREFERRED_ASR_BACKEND
vocabulary_id = None  # 热词表 ID

# ============ 初始化服务实例 ============
# 根据 API 类型初始化翻译 API
if config.TRANSLATION_API_TYPE == 'qwen_mt':
    translation_api = TranslationAPI(
        model=config.QWEN_MT_MODEL,
        base_url=config.QWEN_MT_BASE_URL,
        stream=config.QWEN_MT_STREAM,
        terms=config.QWEN_MT_TERMS,
        domains=config.QWEN_MT_DOMAINS,
    )
elif config.TRANSLATION_API_TYPE == 'openrouter':
    translation_api = TranslationAPI(
        model=config.OPENROUTER_TRANSLATION_MODEL,
        temperature=config.OPENROUTER_TRANSLATION_TEMPERATURE,
        timeout=config.OPENROUTER_TRANSLATION_TIMEOUT,
        max_retries=config.OPENROUTER_TRANSLATION_MAX_RETRIES,
    )
else:
    translation_api = TranslationAPI()

translator = ContextAwareTranslator(
    translation_api=translation_api, 
    max_context_size=6,
    target_language=config.TARGET_LANGUAGE,
    context_aware=True
)

backwards_translation_api = BackwardsTranslationAPI()
backwards_translator = ContextAwareTranslator(
    translation_api=backwards_translation_api, 
    max_context_size=6,
    target_language="en",
    context_aware=True
)

language_detector = LanguageDetector()
# ================================


def reverse_translation(translated_text, source_language, target_language):
    """
    对翻译结果进行反向翻译，从目标语言翻译回原始语言
    
    Args:
        translated_text: 已翻译的文本
        source_language: **本方法进行的翻译的** 源语言代码
        target_language: **本方法进行的翻译的** 目标语言代码
    
    Returns:
        反向翻译后的文本
    """
    try:
        backwards_translated = backwards_translator.translate(
            translated_text,
            source_language=source_language,
            target_language=target_language
        )
        print(f'反向翻译：{backwards_translated}')
        return backwards_translated
    except Exception as e:
        print(f'反向翻译失败: {e}')
        return None


# Real-time speech recognition callback
class VRChatRecognitionCallback(SpeechRecognitionCallback):
    def __init__(self):
        self.loop = None  # 将在主线程中设置
    
    def on_session_started(self) -> None:
        logger.info('Speech recognizer session opened.')

    def on_session_stopped(self) -> None:
        logger.info('Speech recognizer session closed.')

    def on_error(self, error: Exception) -> None:
        logger.error('Speech recognizer failed: %s', error)

    def on_result(self, event: RecognitionEvent) -> None:
        text = event.text
        if not text:
            return

        is_translated = False
        display_text = None
        is_ongoing = not event.is_final

        if is_ongoing:
            print(f'部分：{text}', end='\r')
            display_text = text
        else:
            # 如果禁用翻译，直接显示识别结果
            if not config.ENABLE_TRANSLATION:
                print(f'识别：{text}')
                display_text = text
            else:
                # 启用翻译，执行翻译逻辑
                source_lang_info = language_detector.detect(text)
                source_lang = source_lang_info['language']

                def normalize_lang(lang):
                    """标准化语言代码"""
                    lang_lower = lang.lower()
                    if lang_lower in ['zh', 'zh-cn', 'zh-tw', 'zh-hans', 'zh-hant']:
                        return 'zh'
                    if lang_lower in ['en', 'en-us', 'en-gb']:
                        return 'en'
                    return lang_lower

                normalized_source = normalize_lang(source_lang)
                normalized_target = normalize_lang(config.TARGET_LANGUAGE)

                if config.FALLBACK_LANGUAGE and normalized_source == normalized_target:
                    actual_target = config.FALLBACK_LANGUAGE
                    print(f'原文：{text} [{source_lang_info["language"]}]')
                    print(f'检测到源语言与目标语言相同，使用备用语言: {config.FALLBACK_LANGUAGE}')
                else:
                    actual_target = config.TARGET_LANGUAGE
                    print(f'原文：{text} [{source_lang_info["language"]}]')

                translated_text = translator.translate(
                    text,
                    source_language=config.SOURCE_LANGUAGE,
                    target_language=actual_target,
                    context_prefix=config.CONTEXT_PREFIX,
                )
                is_translated = True
                print(f'译文：{translated_text}')

                display_text = f"[{normalized_source}→{actual_target}] {translated_text}"

        if display_text is None:
            return

        should_send = (not is_ongoing) or config.SHOW_PARTIAL_RESULTS

        if self.loop:
            if should_send:
                asyncio.run_coroutine_threadsafe(
                    osc_manager.send_text(display_text, ongoing=is_ongoing),
                    self.loop
                )
            elif is_ongoing:
                asyncio.run_coroutine_threadsafe(
                    osc_manager.set_typing(is_ongoing),
                    self.loop
                )
        else:
            print('[OSC] Warning: Event loop not set, cannot send OSC message.')

        if is_translated and config.ENABLE_REVERSE_TRANSLATION:
            reverse_translation(translated_text, actual_target, normalized_source)


async def init_audio_stream():
    """异步初始化音频流"""
    global mic, stream
    loop = asyncio.get_event_loop()
    
    def _init():
        global mic, stream
        mic = pyaudio.PyAudio()
        stream = mic.open(
            format=pyaudio.paInt16,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.BLOCK_SIZE
        )
        return stream
    
    return await loop.run_in_executor(executor, _init)


async def close_audio_stream():
    """异步关闭音频流"""
    global mic, stream
    loop = asyncio.get_event_loop()
    
    def _close():
        global mic, stream
        if stream:
            stream.stop_stream()
            stream.close()
        if mic:
            mic.terminate()
        stream = None
        mic = None
    
    await loop.run_in_executor(executor, _close)


async def read_audio_data():
    """异步读取音频数据"""
    global stream
    if not stream:
        return None
    
    loop = asyncio.get_event_loop()
    
    def _read():
        try:
            return stream.read(config.BLOCK_SIZE, exception_on_overflow=False)
        except Exception as e:
            print(f'Error reading audio data: {e}')
            return None
    
    return await loop.run_in_executor(executor, _read)


async def send_audio_frame_async(recognizer: SpeechRecognizer, data: bytes):
    """异步发送音频帧"""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(executor, recognizer.send_audio_frame, data)
    except Exception as e:
        pass
    

async def audio_capture_task(recognizer: SpeechRecognizer):
    """异步音频捕获任务"""
    global recognition_active
    print('Starting audio capture...')
    try:
        while not stop_event.is_set():
            # 始终读取音频数据,避免缓冲区积压
            data = await read_audio_data()
            if not data:
                break
            
            # 只有在识别激活时才发送音频数据,否则丢弃
            if recognition_active:
                await send_audio_frame_async(recognizer, data)
            # 静音时数据被读取但不发送,自动丢弃
            
            await asyncio.sleep(0.001)  # 避免阻塞事件循环
    except asyncio.CancelledError:
        print('Audio capture task cancelled.')
    except Exception as e:
        print(f'Audio capture error: {e}')
    finally:
        print('Audio capture stopped.')


def signal_handler(sig, frame):
    print('Ctrl+C pressed, stop recognition ...')
    # 在异步环境中安全地设置停止事件
    if stop_event is not None:
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(stop_event.set)
        except:
            stop_event.set()


async def stop_recognition_async(recognizer: SpeechRecognizer):
    """异步暂停或停止识别服务"""
    global recognition_active, recognition_started
    if not recognition_active:
        return  # 已经暂停

    loop = asyncio.get_event_loop()

    recognition_active = False

    if CURRENT_ASR_BACKEND == 'dashscope':
        # 发送静音音频帧，确保本次识别至少发送了一个音频帧，否则会报错
        silence_frames = config.BLOCK_SIZE
        silence_data = b'\x00' * (config.BITS // 8 * silence_frames)
        await send_audio_frame_async(recognizer, silence_data)
        await asyncio.sleep(0.1)
        try:
            await loop.run_in_executor(executor, recognizer.stop)
        except Exception:
            pass
        recognition_started = False
    else:
        try:
            await loop.run_in_executor(executor, recognizer.pause)
        except Exception:
            pass


async def start_recognition_async(recognizer: SpeechRecognizer):
    """异步开始或恢复识别服务"""
    global recognition_active, recognition_started
    if recognition_active:
        print('Recognition already active.')
        return  # 已经在运行中

    loop = asyncio.get_event_loop()

    try:
        if CURRENT_ASR_BACKEND == 'qwen' and recognition_started:
            await loop.run_in_executor(executor, recognizer.resume)
        else:
            await loop.run_in_executor(executor, recognizer.start)
            recognition_started = True
    except Exception:
        pass

    recognition_active = True


async def handle_mute_change(is_muted):
    """
    处理静音状态变化的回调函数
    
    Args:
        is_muted: True表示静音(停止识别), False表示取消静音(开始识别)
    """
    global recognition_active, recognition_instance, mute_delay_task, recognition_started
    
    # 如果禁用了麦克风控制，则忽略所有麦克风状态变化
    if not config.ENABLE_MIC_CONTROL:
        return
    
    if recognition_instance is None:
        print('[ASR] 识别实例未初始化')
        return
    
    stop_word = '暂停' if CURRENT_ASR_BACKEND == 'qwen' else '停止'
    start_word = '恢复' if CURRENT_ASR_BACKEND == 'qwen' and recognition_started else '开始'

    if is_muted:
        # 静音状态 - 延迟停止识别
        if recognition_active:
            # 如果已有延迟任务在运行，先取消它
            if mute_delay_task and not mute_delay_task.done():
                mute_delay_task.cancel()
            
            if config.MUTE_DELAY_SECONDS > 0:
                print(f'[ASR] 检测到静音，将在 {config.MUTE_DELAY_SECONDS} 秒后{stop_word}语音识别...')
                
                async def delayed_stop():
                    global recognition_active
                    try:
                        await asyncio.sleep(config.MUTE_DELAY_SECONDS)
                        if recognition_active:  # 再次检查，确保期间没有取消静音
                            print(f'[ASR] 延迟时间到，{stop_word}语音识别')
                            await stop_recognition_async(recognition_instance)
                            logger.info(f'[ASR] 语音识别已{stop_word}')
                    except asyncio.CancelledError:
                        print('[ASR] 停止识别已取消（取消静音）')
                
                mute_delay_task = asyncio.create_task(delayed_stop())
            else:
                # 延迟为0，立即停止
                print(f'[ASR] 检测到静音，立即{stop_word}语音识别...')
                await stop_recognition_async(recognition_instance)
                logger.info(f'[ASR] 语音识别已{stop_word}')
    else:
        # 取消静音 - 开始识别
        # 如果有延迟停止任务，取消它
        if mute_delay_task and not mute_delay_task.done():
            mute_delay_task.cancel()
            print('[ASR] 检测到取消静音，已取消延迟停止任务')
        
        if not recognition_active:
            print(f'[ASR] 检测到取消静音，{start_word}语音识别...')
            await start_recognition_async(recognition_instance)
            logger.info(f'[ASR] 语音识别已{start_word}')


async def main():
    """主异步函数"""
    global recognition_instance, recognition_active, vocabulary_id, CURRENT_ASR_BACKEND, recognition_started, executor, stop_event
    
    # 创建当前事件循环的 stop_event
    stop_event = asyncio.Event()
    
    # 重新创建executor（如果已经shutdown）
    if executor._shutdown:
        executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    
    vocabulary_id = None
    corpus_text: Optional[str] = None

    # 检测并应用系统代理设置
    system_proxies = detect_system_proxy()
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

    CURRENT_ASR_BACKEND = backend
    recognition_active = False
    recognition_started = False

    # 初始化热词（如果启用）
    if config.ENABLE_HOT_WORDS:
        print('\n[热词] 初始化热词资源...')
        try:
            hot_words_manager = HotWordsManager()
            hot_words_manager.load_all_hot_words()
            if backend == 'qwen':
                words = [entry.get('text') for entry in hot_words_manager.get_hot_words() if entry.get('text')]
                if words:
                    corpus_text = "\n".join(words)
                    print(f'[热词] 已生成 Qwen 语料文本，共 {len(words)} 条\n')
                else:
                    print('[热词] 未加载到热词条目，跳过 Qwen 语料配置\n')
            else:
                vocabulary_id = hot_words_manager.create_vocabulary(target_model='fun-asr-realtime')
                print(f'[热词] 热词表创建成功，ID: {vocabulary_id}\n')
        except Exception as e:
            print(f'[热词] 热词初始化失败: {e}')
            print('[热词] 将继续运行但不使用热词\n')
            vocabulary_id = None
            corpus_text = None

    # 启动OSC服务器
    print('[OSC] 启动OSC服务器...')
    await osc_manager.start_server()
    
    # 设置静音状态回调
    osc_manager.set_mute_callback(handle_mute_change)
    print('[OSC] 已设置静音状态回调')

    # 创建识别回调
    callback = VRChatRecognitionCallback()
    callback.loop = asyncio.get_event_loop()

    # 使用工厂创建识别实例
    recognition_instance = create_recognizer(
        backend=backend,
        callback=callback,
        sample_rate=config.SAMPLE_RATE,
        audio_format=config.FORMAT_PCM,
        source_language=config.SOURCE_LANGUAGE,
        vocabulary_id=vocabulary_id,
        corpus_text=corpus_text,
        enable_vad=config.ENABLE_VAD,
        vad_threshold=config.VAD_THRESHOLD,
        vad_silence_duration_ms=config.VAD_SILENCE_DURATION_MS,
        keepalive_interval=config.KEEPALIVE_INTERVAL,
    )
    
    if vocabulary_id and backend == 'dashscope':
        print(f'[ASR] 使用热词表: {vocabulary_id}')
    
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
    await init_audio_stream()

    # 只在主线程中设置信号处理器
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        # 在非主线程中运行时，signal.signal会抛出ValueError
        # 这种情况下由Web UI的stop接口处理停止逻辑
        pass
    
    # 根据配置决定是否立即启动识别
    if config.ENABLE_MIC_CONTROL:
        stop_hint = '暂停' if backend == 'qwen' else '停止'
        resume_hint = '恢复' if backend == 'qwen' else '开始'
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
        # 立即启动识别
        await start_recognition_async(recognition_instance)
        print('[ASR] 语音识别已启动')

    # 创建音频捕获任务
    capture_task = asyncio.create_task(audio_capture_task(recognition_instance))
    
    try:
        # 等待停止事件
        await stop_event.wait()
        
        # 取消捕获任务
        capture_task.cancel()
        
        # 等待捕获任务完成(带超时)
        try:
            await asyncio.wait_for(capture_task, timeout=2.0)
        except asyncio.TimeoutError:
            print('Audio capture task timeout, forcing stop.')
        except asyncio.CancelledError:
            pass
        
        # 如果识别正在运行,停止它
        if recognition_active:
            await stop_recognition_async(recognition_instance)
            halt_word = 'paused' if CURRENT_ASR_BACKEND == 'qwen' else 'stopped'
            print(f'Recognition {halt_word}.')
        
        # 获取统计信息(使用异步方式)
        if recognition_instance:
            loop = asyncio.get_event_loop()
            try:
                request_id = await loop.run_in_executor(executor, recognition_instance.get_last_request_id)
                first_delay = await loop.run_in_executor(executor, recognition_instance.get_first_package_delay)
                last_delay = await loop.run_in_executor(executor, recognition_instance.get_last_package_delay)
                
                print(
                    '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
                    .format(request_id, first_delay, last_delay))
            except Exception as e:
                print(f'[Metric] 获取统计信息失败: {e}')
    
    finally:
        # 清除OSC回调
        osc_manager.clear_mute_callback()

        loop = asyncio.get_event_loop()

        if recognition_instance:
            try:
                await loop.run_in_executor(executor, recognition_instance.stop)
            except Exception:
                pass
            recognition_started = False
            recognition_active = False
        
        # 关闭音频流
        await close_audio_stream()
        
        # 停止OSC服务器
        await osc_manager.stop_server()
        
        # 异步关闭线程池
        await loop.run_in_executor(None, executor.shutdown, False)


# main function
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nProgram terminated by user.')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        print('Cleanup completed.')