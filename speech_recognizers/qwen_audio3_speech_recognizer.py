from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from proxy_detector import refresh_system_proxy_env

from .base_speech_recognizer import SpeechRecognitionCallback
from .dashscope_speech_recognizer import DashscopeSpeechRecognizer
from vrcx_context_bridge import get_asr_context_terms

logger = logging.getLogger(__name__)

__all__ = ["QwenAudio3SpeechRecognizer"]


# 服务端约束：上下文增强最多保留最近 5 轮，单轮文本不超过 400 字符。
MAX_CONTEXT_ROUNDS = 5
MAX_CONTEXT_CHARS_PER_ROUND = 400

# 服务端约束：即时热词最多 2000 条；权重取 [1, 5] 或 50（超级热词最多 50 条）。
MAX_VOCABULARY_ENTRIES = 2000
MAX_SUPER_HOT_WORDS = 50
SUPER_HOT_WORD_WEIGHT = 50
DEFAULT_HOT_WORD_WEIGHT = 4


def build_vocabulary(hot_words: Optional[Iterable[Any]]) -> Dict[str, int]:
    """把热词条目转换为即时热词映射 {热词: 权重}。"""
    vocabulary: Dict[str, int] = {}
    super_hot_words = 0
    invalid_words = 0

    for entry in hot_words or ():
        if isinstance(entry, dict):
            text = str(entry.get("text") or "").strip()
            raw_weight = entry.get("weight", DEFAULT_HOT_WORD_WEIGHT)
        else:
            text = str(entry or "").strip()
            raw_weight = DEFAULT_HOT_WORD_WEIGHT

        if not text or text in vocabulary:
            continue
        if (len(text) > 15 if not text.isascii() else len(text.split()) > 7):
            invalid_words += 1
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

    if invalid_words:
        logger.warning('[QwenAudio3] 跳过 %s 条超出官方长度限制的热词', invalid_words)
    return vocabulary


class QwenAudio3SpeechRecognizer(DashscopeSpeechRecognizer):
    """Qwen-Audio-3.1-ASR-Flash-Streaming 识别器。

    该模型与 Fun-ASR-Realtime 共用 DashScope Recognition（run-task/finish-task）
    协议，因此复用 DashScope 识别器的会话管理，只额外接入两项模型特有能力：

    - 即时热词：以 ``vocabulary`` 参数下发；
    - 上下文增强：一条领域词表 + 最近四条自己/对方的最终识别原文，
      每条不超过 400 字符。无领域词表时可保留五条发言。启动/恢复时通过
      ``raw_input.context`` 下发，运行时在后续音频前通过 ``update_context`` 更新。
    """

    def __init__(
        self,
        callback: SpeechRecognitionCallback,
        *,
        corpus_text: Optional[str] = None,
        hot_words: Optional[Iterable[Any]] = None,
        asr_context_provider: Optional[Callable[[], List[str]]] = None,
        **recognition_kwargs: Any,
    ) -> None:
        self._corpus_text = corpus_text
        self._asr_context_provider = asr_context_provider
        self._applied_raw_input: Optional[Dict[str, Any]] = None
        vocabulary = build_vocabulary(hot_words)
        if vocabulary:
            recognition_kwargs.setdefault("vocabulary", vocabulary)
        super().__init__(callback, **recognition_kwargs)
        if asr_context_provider is not None and not callable(
            getattr(self._require_recognition(), 'update_context', None)
        ):
            raise RuntimeError('Qwen ASR 动态上下文需要 DashScope SDK >= 1.27.5，请更新依赖')

    def start(self) -> None:
        refresh_system_proxy_env()
        # Recognition.start(**kwargs) 会覆盖构造时的同名参数，因此每次会话
        # 都会带上重新计算的上下文；传 None 时 SDK 会把该参数剔除。
        with self._lifecycle_lock:
            raw_input = self._build_raw_input()
            self._require_recognition().start(raw_input=raw_input)
            self._applied_raw_input = raw_input

    def send_audio_frame(self, data: bytes) -> None:
        with self._lifecycle_lock:
            recognition = self._require_recognition()
            raw_input = self._build_raw_input()
            if raw_input != self._applied_raw_input:
                # SDK queues continue-task before this audio in the same FIFO.
                # Only the audio-send thread updates transport; callbacks merely
                # append history and never contend with stop()'s worker join.
                recognition.update_context(payload_input=raw_input or {"context": []})
                self._applied_raw_input = raw_input
            recognition.send_audio_frame(data)

    def stop(self) -> None:
        # pause() 已经把底层会话停掉了，闭麦状态下再关闭服务时 SDK 会抛
        # InvalidParameter，这里直接跳过，避免抛出无意义的异常。
        # _running 是 SDK 私有属性：getattr 兜底，缺失时视为已停止。
        recognition = self._require_recognition()
        if not getattr(recognition, "_running", False):
            return
        # 复用基类的锁与失败告警，与其他生命周期操作串行化（P2-15）。
        super().stop()
        # 后置状态校验：stop 正常返回后若 _running 仍为真，说明底层会话
        # 可能未被真正结束（服务端悬挂/继续计费），留下可观测的告警。
        if getattr(recognition, "_running", False):
            logger.warning(
                '[QwenAudio3] stop() 返回后底层会话仍为运行状态，服务端会话可能未正确结束'
            )

    def _build_raw_input(self) -> Optional[Dict[str, Any]]:
        # Keep reference terms in one message so they cannot consume all five
        # rounds or bury recent speech. Only names are relevant ASR hints; full
        # VRCX metadata contains unrelated IDs/statuses and can exhaust the cap.
        corpus = (self._corpus_text or "").strip()
        terms = get_asr_context_terms()
        hints = "VRChat ASR hints:\n" + "; ".join(terms) if terms else ""
        if corpus and hints:
            domain = corpus[:199] + "\n" + hints[:200]
        else:
            domain = (corpus or hints)[:MAX_CONTEXT_CHARS_PER_ROUND]
        rounds = [domain] if domain else []
        history = self._asr_context_provider() if self._asr_context_provider else []
        history = [text.strip()[:MAX_CONTEXT_CHARS_PER_ROUND]
                   for text in history if isinstance(text, str) and text.strip()]
        rounds.extend(history[-(MAX_CONTEXT_ROUNDS - len(rounds)):])
        if not rounds:
            return None
        return {
            "context": [
                {"role": "user", "content": [{"type": "input_text", "text": chunk}]}
                for chunk in rounds
            ]
        }
