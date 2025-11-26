"""语音识别器工厂模块，负责创建和配置识别器实例"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import dashscope
import config

from .base_speech_recognizer import SpeechRecognitionCallback, SpeechRecognizer
from .dashscope_speech_recognizer import DashscopeSpeechRecognizer

try:
    from .qwen_speech_recognizer import QwenSpeechRecognizer
except ImportError:  # pragma: no cover
    QwenSpeechRecognizer = None  # type: ignore[assignment]


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
        backend: 识别后端，'dashscope', 'qwen' 或 'qwen_international'
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
    if backend in ('qwen', 'qwen_international'):
        if QwenSpeechRecognizer is None:
            raise RuntimeError('QwenSpeechRecognizer 不可用，请安装相关依赖')
        
        # 根据后端类型选择 URL
        if backend == 'qwen_international':
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
        
        # 语言提示
        # language_hint = None if source_language == 'auto' else source_language
        # if language_hint:
        #     recognition_kwargs['language'] = language_hint
        
        # 热词语料
        if corpus_text:
            recognition_kwargs['corpus_text'] = f"这是一段发生在在线多人社交游戏VRChat内的对话。以下是可能出现的关键词:\n{corpus_text}"
        
        # 合并额外参数
        recognition_kwargs.update(extra_kwargs)
        
        return QwenSpeechRecognizer(callback=callback, **recognition_kwargs)
    
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
        
        return DashscopeSpeechRecognizer(callback=callback, **recognition_kwargs)
    
    else:
        raise ValueError(f'不支持的识别后端: {backend}')


def is_backend_available(backend: str) -> bool:
    """
    检查指定后端是否可用
    
    Args:
        backend: 后端名称，'dashscope', 'qwen' 或 'qwen_international'
    
    Returns:
        bool: True 表示可用，False 表示不可用
    """
    if backend in ('qwen', 'qwen_international'):
        return QwenSpeechRecognizer is not None
    elif backend == 'dashscope':
        return True
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
        preferred_backend = 'dashscope'
    
    # 检查首选后端是否可用
    if not is_backend_available(preferred_backend):
        # 回退到 dashscope
        if preferred_backend in ('qwen', 'qwen_international'):
            print('[ASR] Qwen 后端不可用，缺少依赖，已回退到 DashScope.')
        return 'dashscope'
    
    return preferred_backend
