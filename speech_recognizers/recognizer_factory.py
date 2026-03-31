"""语音识别器工厂模块，负责创建和配置识别器实例"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import dashscope
import config

from .base_speech_recognizer import MonoAudioSpeechRecognizer, SpeechRecognitionCallback, SpeechRecognizer
from .dashscope_speech_recognizer import DashscopeSpeechRecognizer
from .doubao_file_speech_recognizer import DoubaoFileSpeechRecognizer
try:
    from local_asr import is_local_asr_build_enabled
    from local_asr.model_manager import is_asr_cached as is_local_asr_cached
    from .local_speech_recognizer import LocalSpeechRecognizer
except ImportError:  # pragma: no cover
    LocalSpeechRecognizer = None  # type: ignore[assignment]
    is_local_asr_build_enabled = lambda: False  # type: ignore[assignment]
    is_local_asr_cached = lambda *args, **kwargs: False  # type: ignore[assignment]

try:
    from .qwen_speech_recognizer import QwenSpeechRecognizer
except ImportError:  # pragma: no cover
    QwenSpeechRecognizer = None  # type: ignore[assignment]

try:
    from .soniox_speech_recognizer import SonioxSpeechRecognizer, WEBSOCKETS_AVAILABLE
except ImportError:  # pragma: no cover
    SonioxSpeechRecognizer = None  # type: ignore[assignment]
    WEBSOCKETS_AVAILABLE = False


def _normalize_qwen_language(lang: Optional[str]) -> Optional[str]:
    """Normalize language code to Qwen ASR 2-letter hint.
    Returns None when no valid hint (auto or empty)."""
    if not lang:
        return None
    lang_lower = lang.strip().lower()
    if lang_lower in ('auto', 'auto-detect'):
        return None

    # Map common variants to two-letter codes
    if lang_lower in ('zh', 'zh-cn', 'zh-tw', 'zh-hans', 'zh-hant'):
        return 'zh'
    if lang_lower in ('en', 'en-us', 'en-gb'):
        return 'en'
    if lang_lower.startswith('ja'):
        return 'ja'
    if lang_lower.startswith('ko'):
        return 'ko'

    # Fallback: take the first two characters when plausible
    if len(lang_lower) >= 2:
        return lang_lower[:2]
    return None


def _resolve_doubao_credentials() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve Doubao credentials from env vars.

    优先读取 DOUBAO_API_KEY。
    同时兼容 DOUBAO_APP_ID + DOUBAO_ACCESS_KEY 组合方式。
    """
    app_id = os.environ.get('DOUBAO_APP_ID', '').strip()
    access_key = os.environ.get('DOUBAO_ACCESS_KEY', '').strip()
    combined = os.environ.get('DOUBAO_API_KEY', '').strip()
    api_key = ''

    if combined:
        if ':' in combined:
            maybe_app_id, maybe_access_key = combined.split(':', 1)
            if not app_id:
                app_id = maybe_app_id.strip()
            if not access_key:
                access_key = maybe_access_key.strip()
        else:
            api_key = combined

    return (api_key or None, app_id or None, access_key or None)


def init_dashscope_api_key() -> None:
    """
    初始化 DashScope API Key
    从环境变量 DASHSCOPE_API_KEY 加载，如果未设置则使用占位符
    """
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
    else:
        dashscope.api_key = '<your-dashscope-api-key>'


def create_recognizer(
    backend: str,
    callback: SpeechRecognitionCallback,
    sample_rate: int = 16000,
    audio_format: str = 'pcm',
    source_language: str = 'auto',
    vocabulary_id: Optional[str] = None,
    corpus_text: Optional[str] = None,
    enable_vad: bool = True,
    vad_threshold: float = 0.2,
    vad_silence_duration_ms: int = 1300,
    keepalive_interval: int = 30,
    **extra_kwargs: Any
) -> SpeechRecognizer:
    """
    创建语音识别器实例
    
    Args:
        backend: 识别后端，'dashscope'、'qwen'、'soniox'、'doubao_file' 或 'local'
        callback: 识别回调实例
        sample_rate: 音频采样率
        audio_format: 音频格式
        source_language: 源语言，'auto' 表示自动检测
        vocabulary_id: DashScope 热词表 ID（仅 dashscope 后端使用）
        corpus_text: Qwen 语料文本（仅 qwen 后端使用）
        enable_vad: 是否启用VAD（仅 qwen 后端使用）
        vad_threshold: VAD阈值（仅 qwen 后端使用）
        vad_silence_duration_ms: VAD静音持续时间（仅 qwen 后端使用）
        keepalive_interval: WebSocket心跳间隔（秒，仅 qwen 后端使用，0表示禁用）
        **extra_kwargs: 其他自定义参数
    
    Returns:
        SpeechRecognizer: 语音识别器实例
    
    Raises:
        ValueError: 当后端不支持时
        RuntimeError: 当依赖缺失时
    """
    requested_input_channels = (
        extra_kwargs.pop('input_channels', None)
        or extra_kwargs.pop('num_channels', None)
        or extra_kwargs.pop('channels', None)
        or 1
    )
    input_channels = max(1, int(requested_input_channels))

    if backend == 'qwen':
        if QwenSpeechRecognizer is None:
            raise RuntimeError('QwenSpeechRecognizer 不可用，请安装相关依赖')
        
        # 根据全局配置选择 URL（国际版或中国大陆版）
        use_international = getattr(config, 'USE_INTERNATIONAL_ENDPOINT', False)
        if use_international:
            asr_url = config.QWEN_ASR_URL_INTERNATIONAL
        else:
            asr_url = config.QWEN_ASR_URL
        
        recognition_kwargs: Dict[str, Any] = {
            'model': config.QWEN_ASR_MODEL,
            'url': asr_url,
            'sample_rate': sample_rate,
            'input_audio_format': audio_format,
            'enable_turn_detection': enable_vad,
            'turn_detection_threshold': vad_threshold,
            'turn_detection_silence_duration_ms': vad_silence_duration_ms,
            'keepalive_interval': keepalive_interval,
        }
        
        # 语言提示：仅 qwen 支持，且 source_language 不是 auto 时才传递
        language_hint = _normalize_qwen_language(source_language)
        if language_hint:
            recognition_kwargs['language'] = language_hint
        
        # 热词语料
        if corpus_text:
            recognition_kwargs['corpus_text'] = f"{corpus_text}"
        
        # 合并额外参数
        recognition_kwargs.update(extra_kwargs)

        recognizer = QwenSpeechRecognizer(callback=callback, **recognition_kwargs)
        return MonoAudioSpeechRecognizer(recognizer, input_channels=input_channels)
    
    elif backend == 'dashscope':
        recognition_kwargs = {
            'model': config.DASHSCOPE_ASR_MODEL,
            'format': audio_format,
            'sample_rate': sample_rate,
            'semantic_punctuation_enabled': False,
        }
        
        # 热词表
        if vocabulary_id:
            recognition_kwargs['vocabulary_id'] = vocabulary_id
        
        # 合并额外参数
        recognition_kwargs.update(extra_kwargs)

        recognizer = DashscopeSpeechRecognizer(callback=callback, **recognition_kwargs)
        return MonoAudioSpeechRecognizer(recognizer, input_channels=input_channels)
    
    elif backend == 'soniox':
        if SonioxSpeechRecognizer is None or not WEBSOCKETS_AVAILABLE:
            raise RuntimeError('SonioxSpeechRecognizer 不可用，请安装 websockets 库: pip install websockets')
        
        # 从环境变量获取 API Key
        import os
        api_key = os.environ.get('SONIOX_API_KEY', '')
        if not api_key:
            raise RuntimeError('SONIOX_API_KEY 环境变量未设置')
        
        recognition_kwargs = {
            'api_key': api_key,
            'model': getattr(config, 'SONIOX_MODEL', 'stt-rt-v3'),
            'sample_rate': sample_rate,
            'num_channels': 1,
            'audio_format': 'pcm_s16le',
            'language_hints': getattr(config, 'SONIOX_LANGUAGE_HINTS', ['en', 'zh', 'ja', 'ko']),
            'enable_endpoint_detection': getattr(config, 'SONIOX_ENABLE_ENDPOINT_DETECTION', True),
        }
        
        # 合并额外参数
        recognition_kwargs.update(extra_kwargs)

        recognition_kwargs['num_channels'] = 1

        recognizer = SonioxSpeechRecognizer(callback=callback, **recognition_kwargs)
        return MonoAudioSpeechRecognizer(recognizer, input_channels=input_channels)

    elif backend == 'doubao_file':
        doubao_api_key, doubao_app_id, doubao_access_key = _resolve_doubao_credentials()
        if not doubao_api_key and not (doubao_app_id and doubao_access_key):
            raise RuntimeError('豆包录音文件识别缺少凭证，请设置 DOUBAO_API_KEY')

        recognition_kwargs = {
            'api_key': doubao_api_key,
            'api_app_key': doubao_app_id,
            'api_access_key': doubao_access_key,
            'resource_id': getattr(config, 'DOUBAO_ASR_RESOURCE_ID', 'volc.seedasr.auc'),
            'url': getattr(config, 'DOUBAO_ASR_FLASH_URL', 'https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash'),
            'model_name': getattr(config, 'DOUBAO_ASR_MODEL_NAME', 'bigmodel'),
            'sample_rate': sample_rate,
            'channels': getattr(config, 'CHANNELS', 1),
            'bits': getattr(config, 'BITS', 16),
            'timeout_seconds': getattr(config, 'DOUBAO_ASR_TIMEOUT_SECONDS', 60),
            'max_buffer_seconds': getattr(config, 'DOUBAO_ASR_MAX_BUFFER_SECONDS', 60),
        }

        recognition_kwargs.update(extra_kwargs)

        recognition_kwargs['channels'] = 1

        recognizer = DoubaoFileSpeechRecognizer(callback=callback, **recognition_kwargs)
        return MonoAudioSpeechRecognizer(recognizer, input_channels=input_channels)
    
    elif backend == 'local':
        if LocalSpeechRecognizer is None:
            raise RuntimeError('LocalSpeechRecognizer 不可用，请安装本地 ASR 相关依赖')

        recognizer = LocalSpeechRecognizer(
            callback=callback,
            sample_rate=sample_rate,
            source_language=getattr(config, 'LOCAL_ASR_LANGUAGE', source_language),
        )
        return MonoAudioSpeechRecognizer(recognizer, input_channels=input_channels)

    else:
        raise ValueError(f'不支持的识别后端: {backend}')


def is_backend_available(backend: str) -> bool:
    """
    检查指定后端是否可用
    
    Args:
        backend: 后端名称，'dashscope'、'qwen'、'soniox'、'doubao_file' 或 'local'
    
    Returns:
        bool: True 表示可用，False 表示不可用
    """
    if backend == 'qwen':
        return QwenSpeechRecognizer is not None
    elif backend == 'dashscope':
        # dashscope (Fun-ASR) 仅在中国大陆版可用
        use_international = getattr(config, 'USE_INTERNATIONAL_ENDPOINT', False)
        return not use_international
    elif backend == 'soniox':
        # Soniox 需要 websockets 库和 API Key
        import os
        has_lib = SonioxSpeechRecognizer is not None and WEBSOCKETS_AVAILABLE
        has_key = bool(os.environ.get('SONIOX_API_KEY', ''))
        return has_lib and has_key
    elif backend == 'doubao_file':
        api_key, app_id, access_key = _resolve_doubao_credentials()
        return bool(api_key or (app_id and access_key))
    elif backend == 'local':
        if LocalSpeechRecognizer is None or not is_local_asr_build_enabled():
            return False
        engine = getattr(config, 'LOCAL_ASR_ENGINE', 'sensevoice')
        return bool(is_local_asr_cached(engine))
    else:
        return False


def select_backend(preferred_backend: str, valid_backends: set) -> str:
    """
    选择可用的识别后端
    
    Args:
        preferred_backend: 首选后端
        valid_backends: 有效后端集合
    
    Returns:
        str: 最终选定的后端名称
    """
    # 验证首选后端是否有效
    if preferred_backend not in valid_backends:
        preferred_backend = 'qwen'
    
    # 检查首选后端是否可用
    if is_backend_available(preferred_backend):
        return preferred_backend

    if preferred_backend == 'dashscope' and getattr(config, 'USE_INTERNATIONAL_ENDPOINT', False):
        print('[ASR] Fun-ASR 在国际版不可用，正在尝试其他后端...')
    else:
        print(f'[ASR] 首选后端 {preferred_backend} 不可用，正在尝试自动回退...')

    for candidate in ('qwen', 'dashscope', 'doubao_file', 'soniox', 'local'):
        if candidate == preferred_backend:
            continue
        if candidate not in valid_backends:
            continue
        if is_backend_available(candidate):
            print(f'[ASR] 已自动切换到可用后端: {candidate}')
            return candidate

    print(f'[ASR] 未找到可用后端，保留原配置: {preferred_backend}')
    return preferred_backend
