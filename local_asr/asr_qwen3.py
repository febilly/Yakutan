from __future__ import annotations

import logging
import sys

import numpy as np

import config
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

    def __init__(
        self,
        model_dir: str | None = None,
        use_dml: bool | None = None,
        chunk_size: float = 30.0,
        *,
        corpus_text: str | None = None,
    ) -> None:
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
        if use_dml is None:
            use_dml = bool(getattr(config, "LOCAL_QWEN_ENCODER_USE_DML", False))
        n_ctx = int(getattr(config, "LOCAL_QWEN_ASR_N_CTX", 2048))
        engine_cfg = ASREngineConfig(
            model_dir=resolved_model_dir,
            use_dml=use_dml,
            n_ctx=n_ctx,
            chunk_size=chunk_size,
            memory_num=1,
            verbose=True,
            enable_aligner=False,
            pad_to=int(chunk_size),
        )
        self._engine = QwenASREngine(engine_cfg)
        self.language: str | None = None
        self._context = ""
        self._corpus_text = (corpus_text or "").strip()
        self.model_dir = resolved_model_dir
        logger.info(
            "Qwen3-ASR loaded: %s (encoder_DML=%s)",
            resolved_model_dir,
            use_dml,
        )

    def set_language(self, language: str) -> None:
        self.language = language if language != "auto" else None

    def _truncate_context_to_tokens(self, text: str) -> str:
        if not text or self._engine is None:
            return text
        limit = int(getattr(config, "LOCAL_QWEN_CONTEXT_MAX_TOKENS", 1024))
        if limit <= 0:
            return text
        model = self._engine.model
        ids = model.tokenize(text, add_special=False, parse_special=True)
        if len(ids) <= limit:
            return text
        return model.detokenize(ids[-limit:])

    def set_corpus_text(self, text: str | None) -> None:
        """注入热词/参考语料（与滚动识别上下文分开，优先保留在 prompt 前缀）。"""
        self._corpus_text = (text or "").strip()

    def _prompt_context(self) -> str:
        """热词语料 + 识别历史，总长度受 LOCAL_QWEN_CONTEXT_MAX_TOKENS 限制；语料前缀优先保留。"""
        corpus = self._corpus_text
        tail = self._context or ""
        if not corpus:
            return self._truncate_context_to_tokens(tail)
        if self._engine is None:
            return f"{corpus}\n{tail}".strip() if tail else corpus

        limit = int(getattr(config, "LOCAL_QWEN_CONTEXT_MAX_TOKENS", 1024))
        if limit <= 0:
            return f"{corpus}\n{tail}".strip() if tail else corpus

        model = self._engine.model
        c_ids = model.tokenize(corpus, add_special=False, parse_special=True)
        if len(c_ids) >= limit:
            return model.detokenize(c_ids[-limit:])

        if not tail:
            return corpus

        nl_ids = model.tokenize("\n", add_special=False, parse_special=True)
        t_ids = model.tokenize(tail, add_special=False, parse_special=True)
        budget = limit - len(c_ids) - len(nl_ids)
        if budget <= 0:
            return model.detokenize(c_ids[-limit:])
        if len(t_ids) <= budget:
            return f"{corpus}\n{tail}"
        kept_tail = model.detokenize(t_ids[-budget:])
        return f"{corpus}\n{kept_tail}"

    def set_context(self, context: str) -> None:
        """仅设置滚动识别上下文（不含热词语料）。"""
        self._context = context
        self._context = self._truncate_context_to_tokens(self._context)

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

        context = self._prompt_context()

        audio_embd, enc_s = self._engine.encoder.encode(audio)
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
        if getattr(config, "LOCAL_QWEN_LOG_PIPELINE_TIMING", False):
            audio_sec = len(audio) / QWEN_SAMPLE_RATE
            pre_s = float(result.t_prefill)
            gen_s = float(result.t_generate)
            total_s = enc_s + pre_s + gen_s
            logger.info(
                "[qwen3-asr] timing audio=%.2fs onnx_encode=%.3fs llm_prefill=%.3fs llm_generate=%.3fs "
                "sum=%.3fs prefill_positions=%s gen_tokens=%s",
                audio_sec,
                enc_s,
                pre_s,
                gen_s,
                total_s,
                result.n_prefill,
                result.n_generate,
            )
        text = result.text.strip()
        if not text:
            return None

        if update_context:
            # 仅累积「识别原文」；热词在 _corpus_text 中单独维护
            self._context = self._truncate_context_to_tokens(self._context + text)
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

