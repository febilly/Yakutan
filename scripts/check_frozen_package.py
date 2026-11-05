"""Inspect and run a freshly built one-file EXE; used by CI and local validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from importlib.metadata import distributions
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader


def check(executable: Path, variant: str) -> dict:
    executable = executable.resolve()
    archive = CArchiveReader(str(executable))
    files = {name.replace("\\", "/") for name in archive.toc}
    modules = set()
    for name, entry in archive.toc.items():
        if entry[-1] == "z":
            modules.update(archive.open_embedded_archive(name).toc)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    resources = [
        "fast_langdetect/resources/lid.176.ftz",
        "local_inference/models/silero_vad/silero_vad_16k_op15.onnx",
        "soundcard/mediafoundation.py.h",
        "pypinyin/pinyin_dict.json", "pypinyin/phrases_dict.json", "jieba/dict.txt",
    ]
    for name in resources:
        require(name in files, f"Missing bundled resource: {name}")
    require(any("onnxruntime_pybind11_state" in name for name in files), "Missing ONNX native binding")
    for name in files:
        lower = name.lower()
        require(not any(token in lower for token in (
            "directml", "onnxruntime_providers_cuda", "onnxruntime_providers_tensorrt",
        )), f"Unexpected GPU ONNX binary: {name}")
        require(not lower.startswith("local_inference/vendor/"), f"Loose vendor file leaked: {name}")
        require(not lower.endswith(".gguf"), f"Model weights leaked: {name}")
        basename = lower.rsplit("/", 1)[-1]
        require(not (basename in {"llama.dll", "mtmd.dll"} or basename.startswith("ggml")),
                f"External llama.cpp runtime leaked into EXE: {name}")
        if variant == "standard":
            require(not any(token in lower for token in ("kaldi", "sentencepiece", "sndfile", "soundfile")),
                    f"Local-only native dependency leaked into standard EXE: {name}")
    for name in modules:
        require(not name.startswith(("webview.platforms.android", "webview.platforms.cocoa",
                                     "webview.platforms.gtk", "webview.platforms.qt",
                                     "webview.platforms.cef", "gguf.scripts")),
                f"Unused platform or tooling module leaked: {name}")
    local_modules = ("kaldi_native_fbank", "sentencepiece", "soundfile", "gguf")
    for name in local_modules:
        require((name in modules) == (variant == "local-inference"), f"Wrong variant dependency: {name}")
    binding = "local_inference.vendor.qwen_asr_gguf.inference.llama"
    require((binding in modules) == (variant == "local-inference"), "Wrong Qwen binding isolation")
    require(("YAKUTAN_LOCAL_INFERENCE_BUILD" in files) == (variant == "local-inference"),
            "Wrong variant marker")

    # Running outside the checkout catches accidental imports/resources from the source tree.
    with tempfile.TemporaryDirectory(prefix="yakutan-package-check-") as folder:
        copied = Path(folder) / executable.name
        shutil.copy2(executable, copied)
        result = subprocess.run(
            [str(copied), "--packaging-self-test", variant], cwd=folder,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
    print(result.stdout)
    print(result.stderr)
    require(result.returncode == 0, f"EXE self-test exited {result.returncode}")
    summaries = [line.split("=", 1)[1] for line in result.stdout.splitlines()
                 if line.startswith("PACKAGING_SELF_TEST=")]
    require(len(summaries) == 1, "EXE self-test did not return its result")
    with executable.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "variant": variant, "exe": executable.name, "bytes": executable.stat().st_size,
        "sha256": sha256,
        "resources": resources, "file_count": len(files), "module_count": len(modules),
        "self_test": json.loads(summaries[0]),
        "build_dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "files": sorted(files), "modules": sorted(modules),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("variant", choices=("standard", "local-inference"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.executable, args.variant)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {report['exe']}: {report['bytes']} bytes, SHA256 {report['sha256']}")
