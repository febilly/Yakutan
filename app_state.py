"""
应用状态模块 - 集中管理所有运行时可变状态，消除全局变量
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any

import config


class AppState:
    """集中管理应用的全部可变运行时状态。

    替代原 main.py 中散落的 19+ 个全局变量，通过依赖注入传递给各模块。
    """

    def __init__(self):
        # ---- 音频采集 ----
        self.mic = None                   # pyaudio.PyAudio 实例
        self.stream = None                # pyaudio.Stream 实例
        self.input_sample_rate: int = config.SAMPLE_RATE
        self.input_block_size: int = config.BLOCK_SIZE
        self.capture_channels: int = config.CHANNELS
        self.resampler = None             # AudioResampler | None
        self.debug_pre_audio_recorder = None   # WaveDebugRecorder | None
        self.debug_audio_recorder = None       # WaveDebugRecorder | None
        self.audio_closing: bool = False

        # ---- 线程 / 异步 ----
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=config.MAX_WORKERS
        )
        self.audio_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="yakutan-audio-io",
        )
        self.stop_event: Optional[asyncio.Event] = None
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None

        # ---- 语音识别 ----
        self.recognition_active: bool = False
        self.recognition_started: bool = False
        self.recognition_instance = None      # SpeechRecognizer | None
        self.recognition_callback = None      # VRChatRecognitionCallback | None
        self.mute_delay_task = None           # asyncio.Task | None
        self.current_asr_backend: str = config.PREFERRED_ASR_BACKEND
        self.vocabulary_id: Optional[str] = None

        # ---- 翻译器实例 ----
        self.translation_api = None
        self.translator = None                # ContextAwareTranslator | None
        self.secondary_translation_api = None
        self.secondary_translator = None
        self.backwards_translation_api = None
        self.backwards_translator = None
        self.deepl_fallback_translation_api = None
        self.deepl_fallback_translator = None
        self.secondary_deepl_fallback_translation_api = None
        self.secondary_deepl_fallback_translator = None

        # ---- 语言检测 ----
        self.language_detector = None

        # ---- 字幕状态（供控制面板轮询） ----
        self.subtitles_state: dict = {
            "original": "",
            "translated": "",
            "reverse_translated": "",
            "ongoing": False,
        }

    def update_subtitles(
        self,
        original: str,
        translated: str,
        ongoing: bool,
        reverse_translated: str = "",
    ):
        self.subtitles_state["original"] = original
        self.subtitles_state["translated"] = translated
        self.subtitles_state["reverse_translated"] = reverse_translated
        self.subtitles_state["ongoing"] = ongoing

    def ensure_executor(self):
        """如果 executor 已关闭，重新创建。"""
        if self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

    def ensure_audio_executor(self):
        """如果 audio_executor 已关闭，重新创建。"""
        if self.audio_executor._shutdown:
            self.audio_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="yakutan-audio-io",
            )


# ---- 模块级单例访问 ----
_current_state: Optional[AppState] = None


def get_state() -> Optional[AppState]:
    """获取当前应用状态（可能为 None，仅在服务启动后可用）。"""
    return _current_state


def set_state(state: Optional[AppState]):
    """设置当前应用状态（由 main() 在启动时调用）。"""
    global _current_state
    _current_state = state
