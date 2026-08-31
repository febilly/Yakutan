#!/usr/bin/env python3
"""Measure one-GPU Qwen ASR + Hy-MT capacity with live-user-like updates.

Each virtual user keeps one ASR and one translation WebSocket open.  It sends a
partial ASR request at a fixed cadence, retaining Qwen's draft tokens between
partials, and immediately sends the recognized text (or a stable fallback) to
Hy-MT.  This is deliberately a *closed-loop* load: a user never has two speech
updates in flight, which matches the desktop client and makes queue latency
meaningful.

Run this on the inference host, for example::

    python benchmark_concurrent.py --url ws://127.0.0.1:18775 \
      --audio /tmp/normal-speech.wav --users 16 --duration 45 --interval 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.client import connect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18775")
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV containing ordinary speech")
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--duration", type=float, default=45.0, help="measurement time after warm-up")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between one user's ASR+Hy-MT updates")
    parser.add_argument("--partial-seconds", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=1, help="per-user requests excluded from results")
    return parser.parse_args()


def load_audio(path: str, seconds: float) -> str:
    with wave.open(str(Path(path)), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError("audio must be 16 kHz, mono, signed 16-bit PCM WAV")
        frames = wav.readframes(min(wav.getnframes(), int(seconds * 16000)))
    pcm = np.frombuffer(frames, dtype="<i2").astype("<f4") / 32768.0
    if len(pcm) < 16000:
        raise ValueError("audio must include at least one second of speech")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


@dataclass
class Measurements:
    asr_e2e_ms: list[float] = field(default_factory=list)
    asr_queue_ms: list[float] = field(default_factory=list)
    asr_run_ms: list[float] = field(default_factory=list)
    asr_batch_sizes: list[int] = field(default_factory=list)
    hymt_e2e_ms: list[float] = field(default_factory=list)
    drafted: int = 0
    accepted: int = 0
    requests: int = 0
    errors: list[str] = field(default_factory=list)


async def init_asr(url: str):
    ws = await connect(url, max_size=32 * 1024 * 1024)
    await ws.send(json.dumps({"type": "init", "service": "asr", "engine": "qwen3-asr"}))
    reply = json.loads(await ws.recv())
    if reply.get("type") != "init_ok":
        raise RuntimeError(f"ASR init failed: {reply}")
    return ws


async def init_hymt(url: str):
    ws = await connect(url, max_size=32 * 1024 * 1024)
    await ws.send(json.dumps({
        "type": "init", "service": "hymt2", "source_lang": "en", "target_lang": "zh",
    }))
    reply = json.loads(await ws.recv())
    if reply.get("type") != "init_ok":
        raise RuntimeError(f"Hy-MT init failed: {reply}")
    return ws


async def virtual_user(
    user_id: int,
    args: argparse.Namespace,
    audio_base64: str,
    started: float,
    measurements: Measurements,
) -> None:
    asr = await init_asr(args.url)
    hymt = await init_hymt(args.url)
    previous_source = ""
    previous_translation = ""
    request_no = 0
    try:
        # Spread first updates evenly across a cadence interval, then retain a
        # stable speaking rhythm.  The first request is a normal partial too;
        # only its lack of a draft makes it a warm-up sample.
        next_due = started + (user_id % max(1, args.users)) * args.interval / max(1, args.users)
        while True:
            now = time.perf_counter()
            if now >= started + args.duration:
                return
            await asyncio.sleep(max(0.0, next_due - now))
            request_no += 1
            t0 = time.perf_counter()
            await asr.send(json.dumps({
                "type": "transcribe", "request_id": f"{user_id}-{request_no}",
                "audio_format": "f32le", "sample_rate": 16000, "audio_base64": audio_base64,
                "language": "en", "update_context": False, "reset_draft": False,
            }))
            reply = json.loads(await asr.recv())
            asr_ms = (time.perf_counter() - t0) * 1000
            if reply.get("type") != "recognition":
                raise RuntimeError(f"ASR request failed: {reply}")
            text = str((reply.get("result") or {}).get("text") or previous_source or "This is a normal spoken sentence.")
            timing = reply.get("timing") or {}

            t1 = time.perf_counter()
            await hymt.send(json.dumps({
                "type": "update", "seq": request_no, "source": text,
                "source_lang": "en", "target_lang": "zh", "is_final": False,
                "previous_source": previous_source, "previous_translation": previous_translation,
            }))
            translated = json.loads(await hymt.recv())
            hymt_ms = (time.perf_counter() - t1) * 1000
            if translated.get("type") != "translation":
                raise RuntimeError(f"Hy-MT request failed: {translated}")
            previous_source = text
            previous_translation = str(translated.get("committed_text") or previous_translation)

            if request_no > args.warmup:
                measurements.asr_e2e_ms.append(asr_ms)
                measurements.asr_queue_ms.append(float(timing.get("queue_ms", 0.0)))
                measurements.asr_run_ms.append(float(timing.get("run_ms", 0.0)))
                measurements.asr_batch_sizes.append(int(timing.get("batch_size", 1)))
                measurements.hymt_e2e_ms.append(hymt_ms)
                measurements.drafted += int(timing.get("n_drafted", 0))
                measurements.accepted += int(timing.get("n_accepted", 0))
                measurements.requests += 1
            next_due += args.interval
    except Exception as exc:
        measurements.errors.append(f"user {user_id}: {exc}")
    finally:
        await asr.close()
        await hymt.close()


async def warm_service(args: argparse.Namespace, audio_base64: str) -> None:
    """Pay lazy model/context construction before the measurement clock starts."""
    asr = await init_asr(args.url)
    hymt = await init_hymt(args.url)
    try:
        await asr.send(json.dumps({
            "type": "transcribe", "request_id": "warmup", "audio_format": "f32le",
            "sample_rate": 16000, "audio_base64": audio_base64, "language": "en",
            "update_context": False, "reset_draft": True,
        }))
        reply = json.loads(await asr.recv())
        if reply.get("type") != "recognition":
            raise RuntimeError(f"ASR warm-up failed: {reply}")
        text = str((reply.get("result") or {}).get("text") or "This is a normal spoken sentence.")
        await hymt.send(json.dumps({
            "type": "update", "seq": 0, "source": text, "source_lang": "en",
            "target_lang": "zh", "is_final": False,
        }))
        reply = json.loads(await hymt.recv())
        if reply.get("type") != "translation":
            raise RuntimeError(f"Hy-MT warm-up failed: {reply}")
    finally:
        await asr.close()
        await hymt.close()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(percentile(values, 0.50), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "mean_ms": round(statistics.fmean(values), 2) if values else float("nan"),
    }


async def run(args: argparse.Namespace) -> None:
    if args.users < 1 or args.interval <= 0 or args.duration <= 0:
        raise ValueError("users, interval and duration must be positive")
    audio_base64 = load_audio(args.audio, args.partial_seconds)
    await warm_service(args, audio_base64)
    measurements = Measurements()
    started = time.perf_counter() + 0.5
    await asyncio.gather(*[
        virtual_user(index, args, audio_base64, started, measurements)
        for index in range(args.users)
    ])
    measured_seconds = max(0.001, args.duration - args.warmup * args.interval)
    report: dict[str, Any] = {
        "users": args.users,
        "update_interval_s": args.interval,
        "audio_partial_seconds": args.partial_seconds,
        "measured_requests": measurements.requests,
        "asr_updates_per_s": round(measurements.requests / measured_seconds, 3),
        "asr_e2e": summarize(measurements.asr_e2e_ms),
        "asr_server_queue": summarize(measurements.asr_queue_ms),
        "asr_server_run": summarize(measurements.asr_run_ms),
        "asr_microbatch": {
            "p50": percentile([float(value) for value in measurements.asr_batch_sizes], 0.50),
            "p95": percentile([float(value) for value in measurements.asr_batch_sizes], 0.95),
            "max": max(measurements.asr_batch_sizes, default=0),
        },
        "hymt_e2e": summarize(measurements.hymt_e2e_ms),
        "draft_tokens": measurements.drafted,
        "accepted_draft_tokens": measurements.accepted,
        "draft_accept_rate": round(measurements.accepted / measurements.drafted, 4)
        if measurements.drafted else None,
        "errors": measurements.errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
