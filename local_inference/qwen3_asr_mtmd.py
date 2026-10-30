"""Qwen3-ASR on llama.cpp's native support: mmproj audio encoder (libmtmd) + GGUF text decoder.

Replaces the vendored HaujetZhao ``QwenASREngine`` (ONNX encoder) and keeps the attributes ``Qwen3ASREngine`` relies on
(``encoder``, ``model``, ``ctx``, ``llama_mod``, ``_build_prompt_embd``, ``ID_IM_END``, ``shutdown``),
so prompt building and the decode loop stay as they were.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .mtmd_audio import MtmdAudioEncoder
from .spec_decode import MAX_DRAFT_CHUNK
from .vendor.qwen_asr_gguf.inference import llama

logger = logging.getLogger(__name__)
_GGML_TYPE_Q8_0 = 8


class MtmdQwenASR:
    def __init__(self, decoder_path: str, mmproj_path: str, *, n_ctx: int = 2048,
                 n_gpu_layers: int = 99, main_gpu: int = 0, encoder_on_gpu: bool = True,
                 kv_q8: bool = False) -> None:
        self.llama_mod = llama
        self._n_ctx = n_ctx
        self._kv_type = _GGML_TYPE_Q8_0 if kv_q8 else None
        self.model = llama.LlamaModel(decoder_path, n_gpu_layers=n_gpu_layers, main_gpu=main_gpu)
        self.embedding_table = llama.get_token_embeddings_gguf(decoder_path)
        self.ctx = llama.LlamaContext(
            self.model, n_ctx=n_ctx, n_batch=4096, embeddings=False,
            type_kv=_GGML_TYPE_Q8_0 if kv_q8 else None,
            # prefill keeps only the last row; a draft pass needs one per drafted token
            n_outputs_max=MAX_DRAFT_CHUNK + 1,
        )
        self.encoder = MtmdAudioEncoder(
            mmproj_path,
            self.model.ptr,
            self.model.n_embd,
            lib_dir=llama._resolve_llama_lib_dir(),
            use_gpu=encoder_on_gpu,
            main_gpu=main_gpu,
        )

        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def create_context_worker(self):
        """Share weights/encoder; each connection owns its KV and decode state."""
        worker = object.__new__(type(self))
        worker.__dict__ = self.__dict__.copy()
        worker.ctx = llama.LlamaContext(
            self.model, n_ctx=self._n_ctx, n_batch=4096, embeddings=False,
            type_kv=self._kv_type, n_outputs_max=64,
        )
        worker._shared_owner = self
        return worker

    def shutdown(self) -> None:
        if getattr(self, "_shared_owner", None) is not None:
            self.ctx = None
            self.model = None
            self.encoder = None
            self.embedding_table = None
            self._shared_owner = None
            return
        # The encoder holds a pointer to the text model; free it first.
        if getattr(self, "encoder", None) is not None:
            self.encoder.close()
            self.encoder = None
        if getattr(self, "ctx", None) is not None:
            del self.ctx
            self.ctx = None
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        self.embedding_table = None
        logger.info("Qwen3-ASR engine shutdown")

    def _build_prompt_embd(self, audio_embd: np.ndarray, prefix_text: str, context: Optional[str],
                           language: Optional[str]) -> np.ndarray:
        """Same prompt layout as the vendored engine, so results stay comparable."""
        def tk(t):
            return self.model.tokenize(t)

        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = [self.ID_IM_START] + tk(prefix_str) + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]

        suffix_head = "assistant\n"
        if language:
            suffix_head += f"language {language}"

        suffix_tokens = [self.ID_AUDIO_END] + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(suffix_head) + [self.ID_ASR_TEXT] + tk(prefix_text)

        n_pre, n_aud, n_suf = len(prefix_tokens), audio_embd.shape[0], len(suffix_tokens)
        total = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32)
        total[:n_pre] = self.embedding_table[prefix_tokens]
        total[n_pre:n_pre + n_aud] = audio_embd
        total[n_pre + n_aud:] = self.embedding_table[suffix_tokens]
        return total
