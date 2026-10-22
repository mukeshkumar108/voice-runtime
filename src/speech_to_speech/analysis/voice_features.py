"""Voice features per utterance, computed off the call's path and kept as numbers only.

The audio is copied once, handed to a separate low-priority process, and dropped; nothing is written to disk.
Everything here is best effort: a full queue, a missing library or a worker crash drops the measurement and never
slows, blocks or fails a turn. Disabled unless VOICE_FEATURES_ENABLED is set.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import select
import subprocess
import sys
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_UTTERANCE_S = 0.6
MAX_UTTERANCE_S = 60.0
MAX_IN_FLIGHT = 4          # more than this and new utterances are skipped, never queued behind
REPLY_TIMEOUT_S = 20.0     # a stuck worker is killed and replaced
KEEP_RESULTS = 256         # newest results kept in memory, keyed by turn id
WORKER_NICE = 15


def enabled() -> bool:
    return os.getenv("VOICE_FEATURES_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


# ── runs in the worker process (python -m speech_to_speech.analysis.voice_features) ──────

def utterance_features(pcm: bytes, sr: int = SAMPLE_RATE) -> dict[str, Any]:
    """Features from 16 kHz int16 mono PCM. Pitch is in semitones re 100 Hz so voices are comparable."""
    import numpy as np
    import parselmouth
    from parselmouth.praat import call

    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    dur = snd.get_total_duration()
    out: dict[str, Any] = {"duration_s": round(dur, 2)}

    f0 = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
    voiced = f0[f0 > 0]
    out["voiced_fraction"] = round(len(voiced) / max(1, len(f0)), 3)
    if len(voiced) >= 10:
        st = 12 * np.log2(voiced / 100.0)
        out.update(f0_mean_hz=round(float(voiced.mean()), 1), f0_sd_st=round(float(st.std()), 2),
                   f0_range_st=round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2))

    inten = snd.to_intensity(minimum_pitch=75, time_step=0.01)
    iv = inten.values[0]
    thr = float(np.percentile(iv, 90)) - 25.0        # "pause" = well below this utterance's own speech level
    speech = iv > thr
    out["pause_ratio"] = round(float(1 - speech.mean()), 3)
    if speech.any():
        out["loudness_mean_db"] = round(float(iv[speech].mean()), 1)
        out["loudness_sd_db"] = round(float(iv[speech].std()), 2)

    t = inten.xs()
    nuclei, last = 0, None
    for i in range(1, len(iv) - 1):
        if not (iv[i] > iv[i - 1] and iv[i] >= iv[i + 1] and iv[i] > thr + 5):
            continue
        if f0[min(int(t[i] / 0.01), len(f0) - 1)] <= 0:
            continue
        if last is None or iv[last:i].min() <= min(iv[last], iv[i]) - 2.0:
            nuclei += 1
            last = i
    out["syllables_per_s"] = round(nuclei / dur, 2)
    out["articulation_rate_syl_s"] = round(nuclei / max(0.01, float(speech.sum()) * 0.01), 2)

    if len(voiced) >= 30:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        out["jitter_local"] = round(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 4)
        out["shimmer_local"] = round(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 4)
        out["hnr_db"] = round(call(snd.to_harmonicity_cc(), "Get mean", 0, 0), 1)
    return out


def _worker_main() -> None:
    """Read `turn_id nbytes\n` + PCM from stdin, answer one JSON line per utterance on stdout."""
    try:
        os.nice(WORKER_NICE)
    except Exception:  # noqa: BLE001
        pass
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        turn_id, nbytes = header.decode().split()
        pcm = stdin.read(int(nbytes))
        try:
            features = utterance_features(pcm)
        except Exception:  # noqa: BLE001 - a bad clip must not matter
            features = None
        stdout.write(json.dumps({"turn_id": turn_id, "features": features}).encode() + b"\n")
        stdout.flush()


# ── runs in the call's process: cheap and never blocks ────────────────────────

_lock = threading.Lock()
_jobs: "queue.Queue[tuple[str, bytes]]" = queue.Queue(maxsize=MAX_IN_FLIGHT)
_pump: threading.Thread | None = None
_results: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def _start_worker() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "speech_to_speech.analysis.voice_features"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _pump_loop() -> None:
    proc: subprocess.Popen | None = None
    failures = 0
    while failures < 5:
        turn_id, pcm = _jobs.get()
        try:
            if proc is None or proc.poll() is not None:
                proc = _start_worker()
            proc.stdin.write(f"{turn_id} {len(pcm)}\n".encode() + pcm)
            proc.stdin.flush()
            ready, _, _ = select.select([proc.stdout], [], [], REPLY_TIMEOUT_S)
            if not ready:
                raise TimeoutError("voice feature worker did not answer")
            reply = json.loads(proc.stdout.readline())
            failures = 0
            if reply.get("features"):
                with _lock:
                    _results[turn_id] = reply["features"]
                    while len(_results) > KEEP_RESULTS:
                        _results.popitem(last=False)
        except Exception:  # noqa: BLE001 - drop this measurement, replace the worker, carry on
            failures += 1
            logger.warning("voice features: worker failed (%d)", failures, exc_info=True)
            if proc is not None:
                proc.kill()
                proc = None
    logger.warning("voice features: giving up after repeated worker failures")
    if proc is not None:
        proc.kill()


def submit(turn_id: str | None, audio: Any) -> None:
    """Queue one utterance for measuring. Returns at once; never raises; drops when busy."""
    global _pump
    try:
        if not turn_id or not enabled():
            return
        n = len(audio)
        if n < MIN_UTTERANCE_S * SAMPLE_RATE or n > MAX_UTTERANCE_S * SAMPLE_RATE:
            return
        with _lock:
            if _pump is None:
                _pump = threading.Thread(target=_pump_loop, name="voice-features", daemon=True)
                _pump.start()
        _jobs.put_nowait((turn_id, _to_pcm16_bytes(audio)))
    except Exception:  # noqa: BLE001 - queue full or anything else: skip
        pass


def _to_pcm16_bytes(audio: Any) -> bytes:
    import numpy as np
    a = np.asarray(audio)
    if a.dtype != np.int16:
        a = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    return a.tobytes()


def take(turn_ids: list[str | None]) -> list[dict[str, Any]]:
    """Results for these turns (in order), removed from memory. Missing ones are skipped."""
    with _lock:
        return [{"turn_id": t, **r} for t in turn_ids if t and (r := _results.pop(t, None))]


def summarize(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """Session-level numbers: duration-weighted means over the utterances that have each feature."""
    keys = ["f0_mean_hz", "f0_sd_st", "f0_range_st", "loudness_mean_db", "loudness_sd_db", "pause_ratio",
            "syllables_per_s", "articulation_rate_syl_s", "jitter_local", "shimmer_local", "hnr_db", "voiced_fraction"]
    out: dict[str, Any] = {"utterances": len(utterances),
                           "speech_seconds": round(sum(u.get("duration_s", 0) for u in utterances), 1)}
    for k in keys:
        pairs = [(u[k], u.get("duration_s", 1.0)) for u in utterances if k in u]
        if pairs:
            w = sum(d for _, d in pairs)
            out[k] = round(sum(v * d for v, d in pairs) / w, 4 if k in {"jitter_local", "shimmer_local"} else 2)
    return out


if __name__ == "__main__":
    _worker_main()
