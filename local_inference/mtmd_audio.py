"""Qwen3-ASR audio encoder on llama.cpp's multimodal library (libmtmd).

Replaces the split int4 ONNX encoder (onnxruntime-directml). The mmproj GGUF runs through the same
ggml backend as the decoder, so Vulkan covers NVIDIA, AMD and Intel and the encoder can be f16/bf16
or q8_0 instead of int4.

Only the encoder half of mtmd is used: audio goes in, one embedding row per audio token comes out.
The prompt is still assembled by ``Qwen3ASREngine`` from token embeddings, which keeps the decode
loop (prefill, speculative drafts, KV rollback) unchanged.

mtmd splits the mel spectrogram into windows of 800 frames (8 s) and encodes each window on its own,
like ``n_window_infer`` in the reference implementation. Each window is padded to a multiple of 100
frames and mtmd emits 13 tokens for every 100 frames, padding included. The reference model drops
the tokens that come from padding, so the last window is trimmed to its real length here.
"""
from __future__ import annotations

import ctypes
import logging
import os
import time
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_HOP = 160
_WINDOW_FRAMES = 800
_GGML_DEV_TYPE_GPU = 1

MTMD_INPUT_CHUNK_TYPE_AUDIO = 2

_ProgressCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)


class mtmd_context_params(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("device", ctypes.c_void_p),
        ("print_timings", ctypes.c_bool),
        ("n_threads", ctypes.c_int),
        ("image_marker", ctypes.c_char_p),
        ("media_marker", ctypes.c_char_p),
        ("flash_attn_type", ctypes.c_int32),
        ("warmup", ctypes.c_bool),
        ("image_min_tokens", ctypes.c_int),
        ("image_max_tokens", ctypes.c_int),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("batch_max_tokens", ctypes.c_int32),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
    ]


class mtmd_input_text(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("text_len", ctypes.c_size_t),
        ("add_special", ctypes.c_bool),
        ("parse_special", ctypes.c_bool),
    ]


_lib = None


def _load(lib_dir: str):
    global _lib
    if _lib is not None:
        return _lib
    name = "mtmd.dll" if os.name == "nt" else "libmtmd.so"
    lib = ctypes.CDLL(os.path.join(lib_dir, name))

    def bind(symbol, argtypes, restype):
        fn = getattr(lib, symbol)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    vp, sz = ctypes.c_void_p, ctypes.c_size_t
    lib.f_params_default = bind("mtmd_context_params_default", [], mtmd_context_params)
    lib.f_default_marker = bind("mtmd_default_marker", [], ctypes.c_char_p)
    lib.f_init = bind("mtmd_init_from_file", [ctypes.c_char_p, vp, mtmd_context_params], vp)
    lib.f_free = bind("mtmd_free", [vp], None)
    lib.f_support_audio = bind("mtmd_support_audio", [vp], ctypes.c_bool)
    lib.f_sample_rate = bind("mtmd_get_audio_sample_rate", [vp], ctypes.c_int)
    lib.f_bitmap_audio = bind("mtmd_bitmap_init_from_audio", [sz, ctypes.POINTER(ctypes.c_float)], vp)
    lib.f_bitmap_free = bind("mtmd_bitmap_free", [vp], None)
    lib.f_chunks_init = bind("mtmd_input_chunks_init", [], vp)
    lib.f_chunks_size = bind("mtmd_input_chunks_size", [vp], sz)
    lib.f_chunks_get = bind("mtmd_input_chunks_get", [vp, sz], vp)
    lib.f_chunks_free = bind("mtmd_input_chunks_free", [vp], None)
    lib.f_chunk_type = bind("mtmd_input_chunk_get_type", [vp], ctypes.c_int)
    lib.f_chunk_n_tokens = bind("mtmd_input_chunk_get_n_tokens", [vp], sz)
    lib.f_tokenize = bind(
        "mtmd_tokenize", [vp, vp, ctypes.POINTER(mtmd_input_text), ctypes.POINTER(vp), sz], ctypes.c_int32
    )
    lib.f_encode_chunk = bind("mtmd_encode_chunk", [vp, vp], ctypes.c_int32)
    lib.f_output_embd = bind("mtmd_get_output_embd", [vp], ctypes.POINTER(ctypes.c_float))
    _lib = lib
    return lib


def _gpu_device_handle(main_gpu: int):
    """The ggml device that llama.cpp calls GPU ``main_gpu`` (GPU devices only, CPU not counted)."""
    from .vendor.qwen_asr_gguf.inference import llama as llama_mod

    libs = [lib for lib in (llama_mod.ggml, llama_mod.ggml_base) if lib is not None]

    def pick(symbol):
        for lib in libs:
            try:
                return getattr(lib, symbol)
            except AttributeError:
                continue
        raise AttributeError(symbol)

    dev_count = pick("ggml_backend_dev_count")
    dev_count.restype = ctypes.c_size_t
    dev_get = pick("ggml_backend_dev_get")
    dev_get.argtypes = [ctypes.c_size_t]
    dev_get.restype = ctypes.c_void_p
    dev_type = pick("ggml_backend_dev_type")
    dev_type.argtypes = [ctypes.c_void_p]
    dev_type.restype = ctypes.c_int
    gpus = [h for h in (dev_get(i) for i in range(dev_count())) if h and dev_type(h) == _GGML_DEV_TYPE_GPU]
    if not gpus:
        return None
    return gpus[min(max(0, int(main_gpu)), len(gpus) - 1)]


def qwen3_audio_token_count(n_frames: int) -> int:
    """Tokens the reference encoder keeps for ``n_frames`` mel frames (``_get_feat_extract_output_lengths``)."""
    leave = n_frames % 100
    feat = (leave - 1) // 2 + 1
    return int(((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n_frames // 100) * 13)


def expected_window_tokens(n_samples: int) -> list[int]:
    """Real token count of each 8 s window mtmd produces for ``n_samples`` of audio."""
    n_frames = n_samples // _HOP + 1  # mtmd: min(stft frames, n_samples / hop + 1)
    return [
        qwen3_audio_token_count(min(_WINDOW_FRAMES, n_frames - off))
        for off in range(0, n_frames, _WINDOW_FRAMES)
    ]


class MtmdAudioEncoder:
    """``encode(audio) -> (embeddings[T, n_embd], seconds)``, the interface of the old ONNX encoder."""

    def __init__(self, mmproj_path: str, model_ptr, n_embd: int, *, lib_dir: str, use_gpu: bool = True,
                 main_gpu: int = 0, n_threads: int | None = None, verbose: bool = True) -> None:
        self.n_embd = int(n_embd)
        # mtmd context/output buffers are mutable, also when weights are shared.
        self._lock = threading.RLock()
        self._lib = _load(lib_dir)
        params = self._lib.f_params_default()
        params.use_gpu = bool(use_gpu)
        params.print_timings = False
        params.n_threads = int(n_threads or max(1, (os.cpu_count() or 4) // 2))
        # mtmd's own warmup sizes the compute buffer for 30 s of mel frames (~550 MB),
        # but a forward pass never sees more than one 8 s window; warm up with that instead.
        params.warmup = False
        if use_gpu:
            params.device = _gpu_device_handle(main_gpu)
        mmproj_path = str(Path(mmproj_path).resolve())
        self._ctx = self._lib.f_init(mmproj_path.encode("utf-8"), model_ptr, params)
        if not self._ctx:
            raise RuntimeError(f"Failed to load audio encoder: {mmproj_path}")
        if not self._lib.f_support_audio(self._ctx):
            self.close()
            raise RuntimeError(f"{mmproj_path} has no audio encoder")
        rate = self._lib.f_sample_rate(self._ctx)
        if rate != SAMPLE_RATE:
            self.close()
            raise RuntimeError(f"audio encoder expects {rate} Hz, not {SAMPLE_RATE}")
        self._marker = self._lib.f_default_marker()
        self.active_gpu = bool(use_gpu)
        self.mmproj_path = mmproj_path
        self.encode(np.zeros(_WINDOW_FRAMES * _HOP, dtype=np.float32))
        if verbose:
            logger.info("mtmd audio encoder loaded: %s (gpu=%s)", mmproj_path, self.active_gpu)

    def encode(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        with self._lock:
            if not getattr(self, "_ctx", None):
                raise RuntimeError("mtmd encoder is closed")
            return self._encode(audio)

    def encode_batch(self, audios: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        # mtmd exposes one audio at a time. Keep the shared context serialized;
        # the decoder still combines independent requests into llama batches.
        t0 = time.perf_counter()
        with self._lock:
            outputs = [self.encode(audio)[0] for audio in audios]
        return outputs, time.perf_counter() - t0

    def _encode(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not samples.size:
            raise ValueError("mtmd expects non-empty mono audio at 16000 Hz")
        lib = self._lib
        bitmap = lib.f_bitmap_audio(samples.size, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if not bitmap:
            raise RuntimeError("mtmd_bitmap_init_from_audio failed")
        chunks = lib.f_chunks_init()
        try:
            text = mtmd_input_text(self._marker, len(self._marker), False, True)
            bitmaps = (ctypes.c_void_p * 1)(bitmap)
            rc = lib.f_tokenize(self._ctx, chunks, ctypes.byref(text), bitmaps, 1)
            if rc != 0:
                raise RuntimeError(f"mtmd_tokenize failed ({rc})")
            expected = expected_window_tokens(samples.size)
            parts: list[np.ndarray] = []
            for i in range(lib.f_chunks_size(chunks)):
                chunk = lib.f_chunks_get(chunks, i)
                if lib.f_chunk_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO:
                    continue  # <|audio_start|>/<|audio_end|>: the prompt builder adds its own
                n_tokens = int(lib.f_chunk_n_tokens(chunk))
                if lib.f_encode_chunk(self._ctx, chunk) != 0:
                    raise RuntimeError("mtmd_encode_chunk failed")
                ptr = lib.f_output_embd(self._ctx)
                embd = np.ctypeslib.as_array(ptr, shape=(n_tokens, self.n_embd)).copy()
                keep = expected[len(parts)] if len(parts) < len(expected) else n_tokens
                parts.append(embd[: min(keep, n_tokens)])
        finally:
            lib.f_chunks_free(chunks)
            lib.f_bitmap_free(bitmap)
        if not parts:
            raise RuntimeError("audio produced no encoder output")
        out = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        return out, time.perf_counter() - t0

    def close(self) -> None:
        if hasattr(self, "_lock"):
            with self._lock:
                if getattr(self, "_ctx", None):
                    self._lib.f_free(self._ctx)
                    self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
