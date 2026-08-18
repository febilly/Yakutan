from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .model_manager import SENSEVOICE_ENCODER_ONNX, get_local_model_path
from .vendor.sensevoice_onnx import SenseVoiceInferenceSession, WavFrontend

logger = logging.getLogger(__name__)

LANG_MAP = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|yue|>": "yue",
}

_LANG_IDS = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}


class SenseVoiceEngine:
    """SenseVoice Small INT8 ONNX on CPU (onnxruntime); pipeline from lovemefan/SenseVoice-python."""

    def __init__(self, model_name: str | None = None, *, num_threads: int = 4) -> None:
        _ = model_name
        local = get_local_model_path("sensevoice")
        if not local:
            raise RuntimeError("SenseVoice ONNX 模型目录未找到，请先下载本地模型或使用内置打包资源")
        self._model_dir = Path(local)
        self._num_threads = max(1, int(num_threads))
        self.device = "cpu"
        self.language: str | None = None

        mvn = self._model_dir / "am.mvn"
        embedding = self._model_dir / "embedding.npy"
        encoder = self._model_dir / SENSEVOICE_ENCODER_ONNX
        bpe = self._model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        for path in (mvn, embedding, encoder, bpe):
            if not path.is_file():
                raise FileNotFoundError(f"SenseVoice ONNX 资源缺失: {path}")

        self._frontend = WavFrontend(str(mvn))
        self._session: SenseVoiceInferenceSession | None = None
        self._load_session()
        logger.info("SenseVoice ONNX (INT8) loaded: %s", self._model_dir)

    def _load_session(self) -> None:
        embedding = self._model_dir / "embedding.npy"
        encoder = self._model_dir / SENSEVOICE_ENCODER_ONNX
        bpe = self._model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        self._session = SenseVoiceInferenceSession(
            str(embedding),
            str(encoder),
            str(bpe),
            device_id=-1,
            intra_op_num_threads=self._num_threads,
        )

    def set_language(self, language: str) -> None:
        self.language = language if language != "auto" else None

    def to_device(self, device: str) -> bool:
        _ = device
        return True

    def unload(self) -> None:
        self._session = None
        self._frontend = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        if self._session is None or self._frontend is None:
            return None
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if waveform.size == 0:
            return None

        feats = self._frontend.get_features(waveform)
        lang_key = (self.language or "auto").lower()
        lang_id = _LANG_IDS.get(lang_key, 0)

        raw_text = self._session(feats[None, ...], language=lang_id, use_itn=True)
        if not raw_text or not str(raw_text).strip():
            return None

        raw_text = str(raw_text)
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
