from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

import config
from .gpu_devices import describe_device, resolve_device
from .model_manager import (
    MODELS_DIR,
    ensure_vendor_sources,
    get_local_model_path,
    prepare_qwen_llama_runtime_env,
    warn_if_weights_not_resident,
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
        gguf = next(Path(resolved_model_dir).glob("*.gguf"), None)
        if gguf is not None and n_gpu_layers > 0:
            with warn_if_weights_not_resident("Qwen3-ASR 解码器", gguf):
                self._engine = QwenASREngine(engine_cfg)
        else:
            self._engine = QwenASREngine(engine_cfg)
        self.language: str | None = None
        self._context = ""
        self._corpus_text = (corpus_text or "").strip()
        self.model_dir = resolved_model_dir
        self.device = effective_device
        self._encoder_on_gpu = bool(getattr(self._engine.encoder, "active_gpu", False))
        # 上一次中间结果的 token，作为下一次的推测解码草稿；整句结束后清空。
        self._draft_tokens: list[int] = []
        self._spec = None
        provider = getattr(self._engine.encoder, 'provider', '')
        encoder_actual = provider if self._encoder_on_gpu else "CPU"
        logger.info(
            "Qwen3-ASR loaded: %s (device=%s -> %s, GGUF: %s 层卸载到该卡, ONNX 编码器实际=%s)",
            resolved_model_dir,
            requested_device,
            describe_device(effective_device),
            n_gpu_layers if n_gpu_layers > 0 else 0,
            encoder_actual,
        )

    def create_shared_context_worker(self):
        """Create a server worker with its own KV/speculative state, not weights."""
        worker = object.__new__(Qwen3ASREngine)
        worker._engine = self._engine.create_context_worker()
        worker.language = None
        worker._context = ""
        worker._corpus_text = ""
        worker.model_dir = self.model_dir
        worker.device = self.device
        worker._encoder_on_gpu = self._encoder_on_gpu
        worker._draft_tokens = []
        worker._spec = None
        return worker

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
            "t_decode": gen.t_decode if gen else 0.0,
            "t_wait": gen.t_wait if gen else 0.0,
            "t_verify": gen.t_verify if gen else 0.0,
            "n_waits": gen.n_waits if gen else 0,
            "n_verifies": gen.n_verifies if gen else 0,
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
            n_passes = int(result["n_passes"])
            # 每次前向的耗时是这条流水线上最容易出问题的指标：它应该稳定在个位数毫秒。
            # 涨到几十毫秒，先看启动时那条「权重已驻留显存 / 没进显存」的日志——权重被
            # 换到系统内存后每次前向都要经 PCIe 重搬一遍，实测 3.3ms -> 56ms。
            per_pass = f"{gen_s * 1000 / n_passes:.1f}" if n_passes else "-"
            # generate 内部再拆四份，哪一份撑大了指向的原因完全不同：
            #   submit  提交前向（llama_decode 是异步的，只算提交）
            #   gpuwait 提交后第一次取 logits——synchronize 落在这里，即 GPU 往返
            #   verify  草稿块里后续的采样，logits 已就绪，纯 CPU 跑词表
            #   rest    剩下的 Python 开销
            dec_s = float(result["t_decode"])
            wait_s = float(result["t_wait"])
            vfy_s = float(result["t_verify"])
            n_waits = int(result["n_waits"])
            n_vfy = int(result["n_verifies"])
            per_wait = f"{wait_s * 1000 / n_waits:.1f}" if n_waits else "-"
            logger.info(
                "[qwen3-asr] timing audio=%.2fs onnx_encode=%.0fms llm_prefill=%.0fms "
                "llm_generate=%.0fms sum=%.0fms prefill_positions=%s gen_tokens=%s "
                "passes=%s ms_per_pass=%s draft=%s/%s | submit=%.0fms(%s次) "
                "gpuwait=%.0fms(%s次,每次%sms) verify=%.0fms(%s次) rest=%.0fms",
                audio_sec,
                enc_s * 1000,
                pre_s * 1000,
                gen_s * 1000,
                total_s * 1000,
                result["n_prefill"],
                len(result["tokens"]),
                n_passes,
                per_pass,
                result["n_accepted"],
                result["n_drafted"],
                dec_s * 1000,
                n_passes,
                wait_s * 1000,
                n_waits,
                per_wait,
                vfy_s * 1000,
                n_vfy,
                (gen_s - dec_s - wait_s - vfy_s) * 1000,
            )
        text = result["text"].strip()
        # Kept on the worker rather than in the desktop-facing result schema.
        # The remote service exposes this as its optional timing payload so a
        # capacity run can prove that successive partials really reuse drafts.
        self._last_metrics = {
            "encoder_ms": round(enc_s * 1000, 3),
            "prefill_ms": round(float(result["t_prefill"]) * 1000, 3),
            "generate_ms": round(float(result["t_generate"]) * 1000, 3),
            "n_prefill": int(result["n_prefill"]),
            "n_passes": int(result["n_passes"]),
            "n_drafted": int(result["n_drafted"]),
            "n_accepted": int(result["n_accepted"]),
        }
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

