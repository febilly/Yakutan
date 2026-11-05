"""Offline smoke checks run by CI inside the actual frozen executable."""
from __future__ import annotations

import importlib
import importlib.util
import io
import json
import sys
from pathlib import Path


LOCAL_MODULES = ("kaldi_native_fbank", "sentencepiece", "soundfile", "gguf")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main(args: list[str]) -> None:
    require(len(args) == 1 and args[0] in {"standard", "local-inference"},
            "Expected build variant: standard or local-inference")
    require(bool(getattr(sys, "frozen", False)), "Self-test must run inside the EXE")
    local = args[0] == "local-inference"
    root = Path(sys._MEIPASS)

    from local_inference import is_local_inference_build_enabled, get_engine_runtime_issues
    from local_inference.model_manager import silero_onnx_path, ensure_vendor_sources
    from local_inference.vad_processor import _SileroOnnxVAD
    import numpy as np

    require(is_local_inference_build_enabled() == local, "Wrong build feature flag")
    require((root / "YAKUTAN_LOCAL_INFERENCE_BUILD").is_file() == local, "Wrong build marker")
    require(silero_onnx_path().is_relative_to(root), "Silero must be bundled in the EXE")
    vad = _SileroOnnxVAD(str(silero_onnx_path()))
    providers = vad._session.get_providers()
    require(providers == ["CPUExecutionProvider"], f"Unexpected ONNX providers: {providers}")
    probability = vad.probability(np.zeros(512, dtype=np.float32), 16000)
    require(0 <= probability <= 1, "Silero inference returned an invalid probability")

    from fast_langdetect import LangDetector, LangDetectConfig
    language = LangDetector(LangDetectConfig(model="lite")).detect(
        "This is a complete English sentence for language detection."
    )[0]["lang"]
    require(language == "en", f"FastText lite detection failed: {language}")

    # Exercise dictionary data and common native extensions, not just find_spec().
    from pykakasi import kakasi
    from pypinyin import lazy_pinyin
    import jieba
    import soxr
    require(bool(kakasi().convert("日本語")[0]["hira"]), "Missing pykakasi dictionary")
    require(lazy_pinyin("中文") == ["zhong", "wen"], "Missing pypinyin dictionary")
    require(bool(list(jieba.cut("中文测试"))), "Missing jieba dictionary")
    require(soxr.resample(np.zeros(480, dtype=np.float32), 48000, 16000).size == 160,
            "soxr native resampler failed")
    for module in ("pyaudio", "soundcard", "zeroconf", "vrchat_oscquery",
                   "dashscope.audio.asr", "dashscope.audio.qwen_omni.omni_realtime",
                   "googletrans", "deepl", "openai", "httpx", "aiohttp", "websockets",
                   "webview.platforms.winforms", "webview.platforms.edgechromium"):
        importlib.import_module(module)

    if local:
        for module in LOCAL_MODULES:
            importlib.import_module(module)
        for engine in ("sensevoice", "qwen3-asr"):
            require(not get_engine_runtime_issues(engine), f"Missing {engine} runtime dependencies")
        ensure_vendor_sources("qwen3-asr")
        importlib.import_module("local_inference.asr_sensevoice")
        importlib.import_module("local_inference.qwen3_asr_mtmd")
        from local_inference.vendor.sensevoice_onnx.sense_voice_ort_session import OrtInferRuntimeSession
        sensevoice_session = OrtInferRuntimeSession(str(silero_onnx_path()))
        require(sensevoice_session.session.get_providers() == ["CPUExecutionProvider"],
                "SenseVoice ONNX wrapper must select CPU only")
        import kaldi_native_fbank as knf
        import soundfile as sf
        fbank = knf.OnlineFbank(knf.FbankOptions())
        fbank.accept_waveform(16000, [0.0] * 1600)
        require(fbank.num_frames_ready > 0, "Kaldi native frontend failed")
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(160, dtype=np.float32), 16000, format="WAV")
        buffer.seek(0)
        require(sf.read(buffer)[1] == 16000, "libsndfile WAV roundtrip failed")
    else:
        for module in LOCAL_MODULES:
            require(importlib.util.find_spec(module) is None, f"Standard EXE contains {module}")

    import main as app_main  # noqa: F401 - validate the application's import graph
    from ui.app import app
    with app.test_client() as client:
        for route in ("/", "/api/config", "/api/features"):
            require(client.get(route).status_code == 200, f"Web UI route failed: {route}")
    print("PACKAGING_SELF_TEST=" + json.dumps({
        "variant": args[0], "onnx_providers": providers,
        "silero_probability": probability, "fasttext_language": language,
        "local_dependencies": local, "web_ui": "ok",
    }, ensure_ascii=False))
