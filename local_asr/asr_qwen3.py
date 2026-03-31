from __future__ import annotations

import logging
import sys

import numpy as np

from .model_manager import (
    MODELS_DIR,
    ensure_vendor_sources,
    get_local_model_path,
    prepare_qwen_llama_runtime_env,
)

logger = logging.getLogger(__name__)
QWEN_SAMPLE_RATE = 16000

_LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


class Qwen3ASREngine:
    """Speech-to-text using Qwen3-ASR (ONNX + GGUF)."""

    def __init__(self, model_dir: str | None = None, use_dml: bool = True, chunk_size: float = 30.0) -> None:
        prepare_qwen_llama_runtime_env()
        vendor_dir = ensure_vendor_sources("qwen3-asr")
        if vendor_dir is None:
            raise RuntimeError("Qwen3-ASR vendor sources are unavailable")
        vendor_parent = str(vendor_dir.parent)
        if vendor_parent not in sys.path:
            sys.path.insert(0, vendor_parent)

        from qwen_asr_gguf.inference.asr import QwenASREngine
        from qwen_asr_gguf.inference.schema import ASREngineConfig

        resolved_model_dir = model_dir or get_local_model_path("qwen3-asr") or str(MODELS_DIR / "qwen3-asr")
        config = ASREngineConfig(
            model_dir=resolved_model_dir,
            use_dml=use_dml,
            n_ctx=2048,
            chunk_size=chunk_size,
            memory_num=1,
            verbose=True,
            enable_aligner=False,
            pad_to=int(chunk_size),
        )
        self._engine = QwenASREngine(config)
        self.language: str | None = None
        self._context = ""
        self.model_dir = resolved_model_dir
        logger.info("Qwen3-ASR loaded: %s (DML=%s)", resolved_model_dir, use_dml)

    def set_language(self, language: str) -> None:
        self.language = language if language != "auto" else None

    def set_context(self, context: str) -> None:
        self._context = context

    def to_device(self, device: str) -> bool:
        return False

    def unload(self) -> None:
        if hasattr(self, "_engine") and self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    def transcribe(self, audio: np.ndarray, *, update_context: bool = True) -> dict | None:
        if self._engine is None:
            return None

        if len(audio) == 0:
            return None

        qwen_language = _LANG_MAP.get(self.language) if self.language else None

        audio_duration = len(audio) / QWEN_SAMPLE_RATE
        context = self._context
        if context:
            context_limit = min(int(audio_duration * 20), 200)
            context = context[-context_limit:] if context_limit > 0 else None

        audio_embd, _enc_time = self._engine.encoder.encode(audio)
        full_embd = self._engine._build_prompt_embd(
            audio_embd=audio_embd,
            prefix_text="",
            context=context,
            language=qwen_language,
        )
        result = self._engine._safe_decode(
            full_embd,
            prefix_text="",
            rollback_num=5,
            is_last_chunk=True,
            temperature=0.4,
        )
        text = result.text.strip()
        if not text:
            return None

        if update_context:
            self._context = (self._context + text)[-200:]
        detected_lang = self.language or self._guess_language(text)
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }

    @staticmethod
    def _guess_language(text: str) -> str:
        cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        jp = sum(1 for char in text if "\u3040" <= char <= "\u30ff" or "\u31f0" <= char <= "\u31ff")
        ko = sum(1 for char in text if "\uac00" <= char <= "\ud7af")
        total = len(text)
        if total == 0:
            return "auto"
        if jp > 0:
            return "ja"
        if ko > total * 0.3:
            return "ko"
        if cjk > total * 0.3:
            return "zh"
        return "en"

