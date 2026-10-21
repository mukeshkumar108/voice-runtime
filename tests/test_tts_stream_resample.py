"""Providers that stream 44.1 kHz audio (e.g. HeyGen via OpenRouter) must reach the 16 kHz pipeline cleanly."""

import numpy as np
import pytest

from speech_to_speech.TTS.lemonfox_tts_handler import LemonfoxTTSHandler, _LegacyBlocks, _StreamBlocks
from threading import Event

RATE_IN, RATE_OUT, BLOCK = 44100, 16000, 512


def _tone(seconds, hz=440.0, rate=RATE_IN, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * hz * t) * amp * 32767).astype("<i2")


def _run(blocks, pcm: bytes, step: int):
    out = []
    for i in range(0, len(pcm), step):
        out.extend(blocks.feed(pcm[i : i + step]))
    out.extend(blocks.finish())
    return out


def test_stream_resampler_keeps_pitch_length_and_is_continuous():
    pcm = _tone(1.0).tobytes()
    out = _run(_StreamBlocks(RATE_IN, RATE_OUT, BLOCK), pcm, step=1371)  # odd, byte-misaligned-looking sizes
    audio = np.concatenate(out).astype(np.float32) / 32768
    assert all(len(b) == BLOCK for b in out)
    assert abs(len(audio) - RATE_OUT) <= BLOCK  # one second in, about one second out
    spectrum = np.abs(np.fft.rfft(audio[2000:12000] * np.hanning(10000)))
    peak_hz = np.argmax(spectrum) * RATE_OUT / 10000
    assert abs(peak_hz - 440) < 8
    # block-edge clicks would show up as jumps far larger than a 440 Hz sine ever makes
    middle = audio[1000:15000]
    assert np.max(np.abs(np.diff(middle))) < 0.5 * 2 * np.pi * 440 / RATE_OUT * 1.3


def test_odd_byte_boundaries_do_not_corrupt_samples():
    pcm = _tone(0.5).tobytes()
    a = np.concatenate(_run(_StreamBlocks(RATE_IN, RATE_OUT, BLOCK), pcm, step=1001))  # odd byte count per feed
    b = np.concatenate(_run(_StreamBlocks(RATE_IN, RATE_OUT, BLOCK), pcm, step=len(pcm)))
    n = min(len(a), len(b))
    assert np.max(np.abs(a[:n].astype(int) - b[:n].astype(int))) <= 2


def _handler(rate):
    h = object.__new__(LemonfoxTTSHandler)
    h.setup(should_listen=Event(), api_key="k", base_url="http://127.0.0.1:9", input_sample_rate=rate)
    return h


def test_resampler_choice_leaves_the_working_lemonfox_path_alone():
    for rate, kind in ((24000, _LegacyBlocks), (16000, _LegacyBlocks), (44100, _StreamBlocks)):
        h = _handler(rate)
        assert isinstance(h._new_blocks(h._default_provider), kind)


def test_payload_omits_default_speed():
    h = _handler(44100)
    assert "speed" not in h._payload("hi", "aoede", h._default_provider)
    h.speed = 1.2
    assert h._payload("hi", "aoede", h._default_provider)["speed"] == 1.2


def _alt_env(monkeypatch):
    monkeypatch.setenv("VOICE_ALT_TTS_BASE_URL", "https://openrouter.example/api/v1/")
    monkeypatch.setenv("VOICE_ALT_TTS_API_KEY", "alt-key")
    monkeypatch.setenv("VOICE_ALT_TTS_MODEL", "heygen/heygen-voice-1-turbo")
    monkeypatch.setenv("VOICE_ALT_TTS_VOICES", "Noelle, Bram")
    monkeypatch.setenv("VOICE_ALT_TTS_SAMPLE_RATE", "44100")


def _tts_input(voice=None):
    from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
    from speech_to_speech.pipeline.messages import TTSInput

    response = RealtimeResponseCreateParams(audio={"output": {"voice": voice}}) if voice else None
    return TTSInput(text="hi", response=response)


def test_the_voice_a_session_asks_for_picks_the_provider(monkeypatch):
    _alt_env(monkeypatch)
    h = _handler(24000)
    provider, voice = h._resolve(_tts_input("noelle"))
    assert (provider.name, voice, provider.model, provider.sample_rate) == ("alt", "Noelle", "heygen/heygen-voice-1-turbo", 44100)
    assert provider.base_url == "https://openrouter.example/api/v1"
    assert isinstance(h._new_blocks(provider), _StreamBlocks)
    # sessions that do not ask for an alt voice (Sophie) are untouched, whatever their voice
    for requested in (None, "aoede", "not-a-lemonfox-voice"):
        provider, voice = h._resolve(_tts_input(requested))
        assert provider.name == "default" and voice in ("aoede",)


def test_without_alt_settings_everything_stays_on_the_default(monkeypatch):
    for name in ("BASE_URL", "API_KEY", "MODEL", "VOICES", "SAMPLE_RATE"):
        monkeypatch.delenv("VOICE_ALT_TTS_" + name, raising=False)
    h = _handler(24000)
    assert h._alt_provider is None
    assert h._resolve(_tts_input("Noelle"))[0].name == "default"


def test_full_sentence_through_the_alt_provider_at_44k(monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from speech_to_speech.pipeline.cancel_scope import CancelScope

    seen = {}
    pcm = _tone(1.0).tobytes()  # one second at 44.1 kHz

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(0, len(pcm), 4000):
                part = pcm[i : i + 4000]
                self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("VOICE_ALT_TTS_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv("VOICE_ALT_TTS_API_KEY", "alt-key")
        monkeypatch.setenv("VOICE_ALT_TTS_MODEL", "heygen/heygen-voice-1-turbo")
        monkeypatch.setenv("VOICE_ALT_TTS_VOICES", "Noelle")
        monkeypatch.setenv("VOICE_ALT_TTS_SAMPLE_RATE", "44100")
        h = object.__new__(LemonfoxTTSHandler)
        h.setup(should_listen=Event(), api_key="lemonfox-key", base_url="http://127.0.0.1:9", cancel_scope=CancelScope())
        blocks = list(h.process(_tts_input("Noelle")))
    finally:
        server.shutdown()

    assert seen["path"] == "/audio/speech"
    assert seen["auth"] == "Bearer alt-key"  # the alt key, not the Lemonfox key
    assert seen["body"]["model"] == "heygen/heygen-voice-1-turbo" and seen["body"]["voice"] == "Noelle"
    assert seen["body"]["response_format"] == "pcm" and "speed" not in seen["body"]
    assert all(len(b) == BLOCK for b in blocks)
    assert abs(len(blocks) * BLOCK - RATE_OUT) <= BLOCK  # one second of audio at 16 kHz
