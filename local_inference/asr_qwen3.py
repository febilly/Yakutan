from __future__ import annotations

import logging
import sys

import numpy as np

import config
from .gpu_devices import describe_device, resolve_device
from .model_manager import (
    MODELS_DIR,
    ensure_vendor_sources,
    get_local_model_path,
    prepare_qwen_llama_runtime_env,
)

logger = logging.getLogger(__name__)
QWEN_SAMPLE_RATE = 16000

# 注：音频编码器后端在 DirectML 下要求固定输入形状，vendor 用单一 pad_to 秒数兜住所有音频，
# 于是 2s 的句子也在算 30s 的注意力。试过在调用前改写 h_target_len 做分档，实测是负优化
# ——会话只认构造时那一个形状，换成别的会永久退化（2s 音频 50ms → 210ms，跑 20 次也不收敛）。
# 想吃到这个收益只能在构造会话时就定小形状，代价是超过它的音频掉出快路径，属于另一个取舍。

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

        # GGUF 解码器的运行位置（SenseVoice 固定 CPU，只有 Qwen3-ASR 用得上 GPU）
        requested_device = getattr(config, "LOCAL_INFERENCE_DEVICE", "auto")
        n_gpu_layers, main_gpu, effective_device = resolve_device(requested_device)
        # ONNX 音频编码器可以和 GGUF 解码器分开放：'auto' 跟随解码器，也可单独钉在 CPU 或显卡上。
        if use_dml is None:
            use_dml = self._want_encoder_on_gpu(effective_device)
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
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,
        )
        self._engine = QwenASREngine(engine_cfg)
        self.language: str | None = None
        self._context = ""
        self._corpus_text = (corpus_text or "").strip()
        self.model_dir = resolved_model_dir
        self.device = effective_device
        self._encoder_on_gpu = bool(getattr(self._engine.encoder, "active_dml", False))
        # 上一次中间结果的 token，作为下一次的推测解码草稿；整句结束后清空。
        self._draft_tokens: list[int] = []
        self._spec = None
        encoder_actual = "DirectML(GPU)" if self._encoder_on_gpu else "CPU"
        logger.info(
            "Qwen3-ASR loaded: %s (device=%s -> %s, GGUF: %s 层卸载到该卡, ONNX 编码器实际=%s)",
            resolved_model_dir,
            requested_device,
            describe_device(effective_device),
            n_gpu_layers if n_gpu_layers > 0 else 0,
            encoder_actual,
        )

    @staticmethod
    def _want_encoder_on_gpu(effective_device: str) -> bool:
        """解析 LOCAL_QWEN_ENCODER_DEVICE：'auto' 跟随解码器，'cpu'/'gpu' 强制。"""
        choice = config.sanitize_qwen_encoder_device(
            getattr(config, "LOCAL_QWEN_ENCODER_DEVICE", "auto")
        )
        if choice == "cpu":
            return False
        if choice == "gpu":
            return True
        return str(effective_device or "").strip().lower() != "cpu"

    def reset_draft(self) -> None:
        """整句边界：丢弃上一句的草稿，避免拿旧句子的译文去推测新句子。"""
        self._draft_tokens = []

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
            # 解码器持有已释放的 ctx，必须跟着一起丢掉
            self._spec = None
            self._draft_tokens = []
            self._engine.shutdown()
            self._engine = None

    def _ensure_spec(self):
        """惰性建立推测解码器；拿引擎自己绑好 DLL 的那个 llama 模块，别再 import 一份没初始化的。"""
        if self._spec is None:
            from .spec_decode import SpeculativeDecoder

            self._spec = SpeculativeDecoder(
                self._engine.llama_mod, self._engine.ctx, self._engine.model
            )
        return self._spec

    @staticmethod
    def _is_repeating(tokens: list[int]) -> bool:
        """vendor 的复读熔断：最近 15 个 token 里只有 <=3 种，判定跑飞。"""
        return len(tokens) > 15 and len(set(tokens[-15:])) <= 3

    def _decode(self, full_embd: np.ndarray) -> dict:
        """整段 prefill + 生成，带草稿推测解码；复读时按 vendor 的做法加温重试。

        对应 vendor 的 _decode/_safe_decode。这里不复刻 rollback_num 那套显示队列：
        yakutan 每次都以 is_last_chunk=True 调用、整段取用文本，队列最终会被全部冲出，
        对结果没有影响。
        """
        import time

        engine = self._engine
        spec = self._ensure_spec()
        stop_ids = {engine.model.eos_token, engine.ID_IM_END}
        use_draft = bool(getattr(config, "LOCAL_SPECULATIVE_DECODE", True))
        draft = self._draft_tokens if use_draft else None

        n_prefill = int(full_embd.shape[0])
        temperature = 0.4
        gen = None
        t_prefill = 0.0
        t_generate = 0.0

        for _ in range(4):
            spec.clear()
            t0 = time.perf_counter()
            spec.decode_embd(full_embd)
            t_prefill = time.perf_counter() - t0

            sampler = engine.llama_mod.LlamaSampler(
                temperature=temperature, seed=int(np.random.randint(0, 2**31 - 1))
            )
            try:
                t0 = time.perf_counter()
                gen = spec.generate(
                    sampler,
                    n_prefill,
                    stop_ids=stop_ids,
                    max_tokens=512,
                    draft=draft,
                    planes=4,
                    on_token=lambda toks: self._is_repeating(toks),
                )
                t_generate = time.perf_counter() - t0
            finally:
                del sampler

            if not gen.aborted:
                break
            temperature += 0.3
            draft = None  # 草稿跟着跑飞了就别再用它
            logger.warning("Decode aborted, retry with temp=%.1f", temperature)

        tokens = gen.tokens if gen else []
        return {
            "text": engine.model.detokenize(tokens) if tokens else "",
            "tokens": tokens,
            "t_prefill": t_prefill,
            "t_generate": t_generate,
            "n_prefill": n_prefill,
            "n_passes": gen.n_passes if gen else 0,
            "n_drafted": gen.n_drafted if gen else 0,
            "n_accepted": gen.n_accepted if gen else 0,
        }

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
        result = self._decode(full_embd)
        if getattr(config, "LOCAL_QWEN_LOG_PIPELINE_TIMING", False):
            audio_sec = len(audio) / QWEN_SAMPLE_RATE
            pre_s = float(result["t_prefill"])
            gen_s = float(result["t_generate"])
            total_s = enc_s + pre_s + gen_s
            logger.info(
                "[qwen3-asr] timing audio=%.2fs onnx_encode=%.3fs llm_prefill=%.3fs llm_generate=%.3fs "
                "sum=%.3fs prefill_positions=%s gen_tokens=%s passes=%s draft=%s/%s",
                audio_sec,
                enc_s,
                pre_s,
                gen_s,
                total_s,
                result["n_prefill"],
                len(result["tokens"]),
                result["n_passes"],
                result["n_accepted"],
                result["n_drafted"],
            )
        text = result["text"].strip()
        if not text:
            self._draft_tokens = []
            return None

        # 本句下一次中间结果的草稿；整句结束后这一句就翻篇了，不该再影响下一句。
        self._draft_tokens = [] if update_context else list(result["tokens"])

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

