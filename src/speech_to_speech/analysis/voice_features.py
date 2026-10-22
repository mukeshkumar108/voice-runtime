"""Voice features per utterance, computed off the call's path and kept as numbers only.

The audio is copied once, handed to a separate low-priority process, and dropped; nothing is written to disk.
Everything here is best effort: a full queue, a missing library or a worker crash drops the measurement and never
slows, blocks or fails a turn. Disabled unless VOICE_FEATURES_ENABLED is set.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_UTTERANCE_S = 0.6
MAX_UTTERANCE_S = 60.0
MAX_IN_FLIGHT = 4          # more than this and new utterances are skipped, never queued behind
KEEP_RESULTS = 256         # newest results kept in memory, keyed by turn id
WORKER_NICE = 15


def enabled() -> bool:
    return os.getenv("VOICE_FEATURES_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


# ── runs in the worker process ────────────────────────────────────────────────

def _init_worker() -> None:
    try:
        os.nice(WORKER_NICE)
    except Exception:  # noqa: BLE001
        pass


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


def _work(pcm: bytes) -> dict[str, Any] | None:
    try:
        return utterance_features(pcm)
    except Exception:  # noqa: BLE001 - a bad clip must not matter
        return None


# ── runs in the call's process: cheap and never blocks ────────────────────────

_lock = threading.Lock()
_executor: ProcessPoolExecutor | None = None
_broken = False
_in_flight = 0
_results: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def _get_executor() -> ProcessPoolExecutor | None:
    global _executor, _broken
    if _broken:
        return None
    if _executor is None:
        try:
            import multiprocessing as mp
            _executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=_init_worker)
        except Exception:  # noqa: BLE001
            logger.warning("voice features: could not start worker, disabled", exc_info=True)
            _broken = True
            return None
    return _executor


def submit(turn_id: str | None, audio: Any) -> None:
    """Queue one utterance for measuring. Returns at once; never raises."""
    global _in_flight
    try:
        if not turn_id or not enabled():
            return
        n = len(audio)
        if n < MIN_UTTERANCE_S * SAMPLE_RATE or n > MAX_UTTERANCE_S * SAMPLE_RATE:
            return
        with _lock:
            if _in_flight >= MAX_IN_FLIGHT:
                return
            executor = _get_executor()
            if executor is None:
                return
            _in_flight += 1
        try:
            pcm = _to_pcm16_bytes(audio)
            future = executor.submit(_work, pcm)
        except Exception:  # noqa: BLE001
            with _lock:
                _in_flight -= 1
            return
        future.add_done_callback(lambda f, tid=turn_id: _done(tid, f))
    except Exception:  # noqa: BLE001
        pass


def _to_pcm16_bytes(audio: Any) -> bytes:
    import numpy as np
    a = np.asarray(audio)
    if a.dtype != np.int16:
        a = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    return a.tobytes()


def _done(turn_id: str, future: Future) -> None:
    global _in_flight, _broken
    try:
        result = future.result()
    except Exception:  # noqa: BLE001 - worker died; stop using it rather than retrying every turn
        _broken = True
        result = None
    with _lock:
        _in_flight -= 1
        if result:
            _results[turn_id] = result
            while len(_results) > KEEP_RESULTS:
                _results.popitem(last=False)


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
