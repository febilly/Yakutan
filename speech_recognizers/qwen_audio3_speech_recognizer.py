from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from proxy_detector import refresh_system_proxy_env

from .base_speech_recognizer import SpeechRecognitionCallback
from .dashscope_speech_recognizer import DashscopeSpeechRecognizer
from vrcx_context_bridge import build_asr_context_text

__all__ = ["QwenAudio3SpeechRecognizer"]


# 服务端约束：上下文增强最多保留最近 5 轮，单轮文本不超过 400 字符。
MAX_CONTEXT_ROUNDS = 5
MAX_CONTEXT_CHARS_PER_ROUND = 400

# 服务端约束：即时热词最多 2000 条；权重取 [1, 5] 或 50（超级热词最多 50 条）。
MAX_VOCABULARY_ENTRIES = 2000
MAX_SUPER_HOT_WORDS = 50
SUPER_HOT_WORD_WEIGHT = 50
DEFAULT_HOT_WORD_WEIGHT = 4


def split_context_rounds(text: str) -> List[str]:
    """把上下文文本切成若干轮，满足单轮字符上限与总轮数上限。

    优先在换行处切分以保留原有结构；超长单行按字符硬切。超出轮数上限的
    尾部内容会被丢弃——`build_asr_context_text` 把热词语料放在最前面，
    因此保留头部即保留信息量更高的部分。
    """
    stripped = (text or "").strip()
    if not stripped:
        return []

    rounds: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            rounds.append("\n".join(current))
            current = []
            current_len = 0

    for raw_line in stripped.splitlines():
        segments = [
            raw_line[index:index + MAX_CONTEXT_CHARS_PER_ROUND]
            for index in range(0, len(raw_line), MAX_CONTEXT_CHARS_PER_ROUND)
        ] or [""]
        for segment in segments:
            extra = len(segment) + (1 if current else 0)
            if current and current_len + extra > MAX_CONTEXT_CHARS_PER_ROUND:
                flush()
                if len(rounds) >= MAX_CONTEXT_ROUNDS:
                    return rounds[:MAX_CONTEXT_ROUNDS]
                extra = len(segment)
            current.append(segment)
            current_len += extra

    flush()
    return rounds[:MAX_CONTEXT_ROUNDS]


def build_vocabulary(hot_words: Optional[Iterable[Any]]) -> Dict[str, int]:
    """把热词条目转换为即时热词映射 {热词: 权重}。"""
    vocabulary: Dict[str, int] = {}
    super_hot_words = 0

    for entry in hot_words or ():
        if isinstance(entry, dict):
            text = str(entry.get("text") or "").strip()
            raw_weight = entry.get("weight", DEFAULT_HOT_WORD_WEIGHT)
        else:
            text = str(entry or "").strip()
            raw_weight = DEFAULT_HOT_WORD_WEIGHT

        if not text or text in vocabulary:
            continue
        if len(vocabulary) >= MAX_VOCABULARY_ENTRIES:
            break

        try:
            weight = int(raw_weight)
        except (TypeError, ValueError):
            weight = DEFAULT_HOT_WORD_WEIGHT

        if weight == SUPER_HOT_WORD_WEIGHT and super_hot_words < MAX_SUPER_HOT_WORDS:
            super_hot_words += 1
        else:
            weight = min(5, max(1, weight))

        vocabulary[text] = weight

    return vocabulary


class QwenAudio3SpeechRecognizer(DashscopeSpeechRecognizer):
    """Qwen-Audio-3.0-ASR-Flash-Streaming 识别器。

    该模型与 Fun-ASR-Realtime 共用 DashScope Recognition（run-task/finish-task）
    协议，因此复用 DashScope 识别器的会话管理，只额外接入两项模型特有能力：

    - 即时热词：构造时以 ``vocabulary`` 参数下发，整个进程生命周期固定；
    - 上下文增强：每次 ``start()``/``resume()`` 时用最新 VRCX 上下文重建
      ``raw_input.context``，因此每个识别分段都能拿到当时的世界/玩家信息。
    """

    def __init__(
        self,
        callback: SpeechRecognitionCallback,
        *,
        corpus_text: Optional[str] = None,
        hot_words: Optional[Iterable[Any]] = None,
        **recognition_kwargs: Any,
    ) -> None:
        self._corpus_text = corpus_text
        vocabulary = build_vocabulary(hot_words)
        if vocabulary:
            recognition_kwargs.setdefault("vocabulary", vocabulary)
        super().__init__(callback, **recognition_kwargs)

    def start(self) -> None:
        refresh_system_proxy_env()
        # Recognition.start(**kwargs) 会覆盖构造时的同名参数，因此每次会话
        # 都会带上重新计算的上下文；传 None 时 SDK 会把该参数剔除。
        self._require_recognition().start(raw_input=self._build_raw_input())

    def stop(self) -> None:
        # pause() 已经把底层会话停掉了，闭麦状态下再关闭服务时 SDK 会抛
        # InvalidParameter，这里直接跳过，避免抛出无意义的异常。
        recognition = self._require_recognition()
        if not getattr(recognition, "_running", False):
            return
        recognition.stop()

    def _build_raw_input(self) -> Optional[Dict[str, Any]]:
        rounds = split_context_rounds(build_asr_context_text(self._corpus_text or ""))
        if not rounds:
            return None
        return {
            "context": [
                {"role": "user", "content": [{"type": "input_text", "text": chunk}]}
                for chunk in rounds
            ]
        }
