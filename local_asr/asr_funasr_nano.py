from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import wave

import numpy as np

from .asr_sensevoice import _resolve_device
from .model_manager import ASR_MODEL_IDS, ensure_vendor_sources, get_local_model_path

logger = logging.getLogger(__name__)


class FunASRNanoEngine:
    """Speech-to-text using Fun-ASR-Nano-2512."""

    def __init__(self, device: str = "cuda", hub: str = "ms", engine_type: str = "funasr-nano") -> None:
        vendor_dir = ensure_vendor_sources("funasr-nano")
        if vendor_dir is None:
            raise RuntimeError("Fun-ASR-Nano vendor sources are unavailable")
        vendor_dir_str = str(vendor_dir)
        if vendor_dir_str not in sys.path:
            sys.path.insert(0, vendor_dir_str)

        import model as _nano_model  # noqa: F401
        from funasr import AutoModel

        model_name = ASR_MODEL_IDS[engine_type]
        local = get_local_model_path(engine_type, hub=hub)
        model = local or model_name
        if local:
            self._ensure_qwen_weights(local)

        resolved_device = _resolve_device(device)
        prev_cwd = os.getcwd()
        if local:
            os.chdir(local)
        try:
            self._model = AutoModel(
                model=model,
                trust_remote_code=True,
                device=resolved_device,
                hub=hub,
                disable_update=True,
            )
        finally:
            os.chdir(prev_cwd)

        self.language: str | None = None
        self.device = resolved_device
        self.engine_type = engine_type
        logger.info("%s loaded on %s (hub=%s)", engine_type, resolved_device, hub)

    @staticmethod
    def _ensure_qwen_weights(model_dir: str) -> None:
        qwen_dir = os.path.join(model_dir, "Qwen3-0.6B")
        if not os.path.isdir(qwen_dir):
            return
        if any(filename.endswith((".safetensors", ".bin")) for filename in os.listdir(qwen_dir)):
            return
        logger.info("Downloading Qwen3-0.6B weights for Fun-ASR-Nano...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            "Qwen/Qwen3-0.6B",
            local_dir=qwen_dir,
            ignore_patterns=["*.gguf"],
        )

    def set_language(self, language: str) -> None:
        self.language = language if language != "auto" else None

    def to_device(self, device: str) -> bool:
        resolved = _resolve_device(device)
        try:
            self._model.model.to(resolved)
            self.device = resolved
            return True
        except Exception:
            return False

    def unload(self) -> None:
        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.model.to("cpu")
            except Exception:
                pass
            self._model = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        file_descriptor, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(file_descriptor)
        try:
            audio_16bit = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(tmp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_16bit.tobytes())

            kwargs = {"input": [tmp_path], "batch_size": 1, "disable_pbar": True}
            if self.language:
                kwargs["language"] = self.language
            result = self._model.generate(**kwargs)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not result or not result[0].get("text"):
            return None

        raw_text = result[0]["text"]
        text = result[0].get("text_tn", raw_text) or raw_text
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()
        if not text or text == "sil":
            return None
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

