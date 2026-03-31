from __future__ import annotations

import logging
import re

import numpy as np

from .model_manager import get_local_model_path

logger = logging.getLogger(__name__)

LANG_MAP = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|yue|>": "yue",
}


def _resolve_device(device: str) -> str:
    requested = (device or "cpu").strip().lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return requested
    except Exception:
        pass
    logger.warning("Requested local ASR device '%s' is unavailable, falling back to CPU", device)
    return "cpu"


class SenseVoiceEngine:
    """Speech-to-text using FunASR SenseVoice."""

    def __init__(self, model_name: str = "iic/SenseVoiceSmall", device: str = "cuda", hub: str = "ms") -> None:
        from funasr import AutoModel

        local = get_local_model_path("sensevoice", hub=hub)
        model = local or model_name
        resolved_device = _resolve_device(device)
        self._model = AutoModel(
            model=model,
            trust_remote_code=True,
            device=resolved_device,
            hub=hub,
            disable_update=True,
        )
        self.language: str | None = None
        self.device = resolved_device
        logger.info("SenseVoice loaded: %s on %s (hub=%s)", model_name, resolved_device, hub)

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
        result = self._model.generate(
            input=audio,
            cache={},
            language=self.language or "auto",
            use_itn=True,
            batch_size_s=0,
            disable_pbar=True,
        )
        if not result or not result[0].get("text"):
            return None

        raw_text = result[0]["text"]
        detected_lang = "auto"
        text = raw_text
        for tag, lang in LANG_MAP.items():
            if tag in text:
                detected_lang = lang
                text = text.replace(tag, "")
                break
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()
        if not text:
            return None
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }

