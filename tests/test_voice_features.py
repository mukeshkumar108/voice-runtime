import time

import numpy as np
import pytest

from speech_to_speech.analysis import voice_features as vf


def _tone(seconds=1.5, hz=180.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(vf, "_pump", None)
    monkeypatch.setattr(vf, "_jobs", vf.queue.Queue(maxsize=vf.MAX_IN_FLIGHT))
    vf._results.clear()


def test_disabled_by_default_does_nothing(monkeypatch):
    monkeypatch.delenv("VOICE_FEATURES_ENABLED", raising=False)
    vf.submit("t1", _tone())
    assert vf._pump is None and vf.take(["t1"]) == []


def test_submit_returns_immediately_and_never_raises(monkeypatch):
    monkeypatch.setenv("VOICE_FEATURES_ENABLED", "1")
    audio = _tone(5.0)
    start = time.perf_counter()
    for i in range(20):
        vf.submit(f"t{i}", audio)       # far more than MAX_IN_FLIGHT: the rest are dropped, not queued
    assert time.perf_counter() - start < 0.5
    assert vf._jobs.qsize() <= vf.MAX_IN_FLIGHT
    vf.submit(None, audio)
    vf.submit("bad", object())


def test_short_and_empty_clips_are_skipped(monkeypatch):
    monkeypatch.setenv("VOICE_FEATURES_ENABLED", "1")
    vf.submit("short", _tone(0.2))
    assert vf._pump is None


def test_features_arrive_without_audio(monkeypatch):
    pytest.importorskip("parselmouth")
    monkeypatch.setenv("VOICE_FEATURES_ENABLED", "1")
    vf.submit("t1", _tone())
    deadline = time.time() + 30
    got = []
    while time.time() < deadline and not got:
        got = vf.take(["t1"])
        time.sleep(0.1)
    assert got and got[0]["turn_id"] == "t1" and 150 < got[0]["f0_mean_hz"] < 210
    assert not any(isinstance(v, (bytes, bytearray)) for v in got[0].values())


def test_summary_weights_by_duration():
    s = vf.summarize([{"duration_s": 1.0, "pause_ratio": 0.1}, {"duration_s": 3.0, "pause_ratio": 0.3}])
    assert s["utterances"] == 2 and s["pause_ratio"] == 0.25 and s["speech_seconds"] == 4.0


def test_summary_keeps_voice_quality_only_from_clean_clips():
    clips = [{"duration_s": 2.0, "snr_db": 30.0, "jitter_local": 0.01, "pause_ratio": 0.1},
             {"duration_s": 2.0, "snr_db": 5.0, "jitter_local": 0.05, "pause_ratio": 0.3}]
    s = vf.summarize(clips)
    assert s["clean_utterances"] == 1 and s["jitter_local"] == 0.01 and s["pause_ratio"] == 0.2
    assert "jitter_local" not in vf.summarize([clips[1]])


def test_very_short_clips_are_not_measured(monkeypatch):
    monkeypatch.setenv("VOICE_FEATURES_ENABLED", "1")
    vf.submit("short", _tone(0.8))
    assert vf._pump is None
