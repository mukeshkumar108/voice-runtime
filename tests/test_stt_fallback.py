"""Speech recognition: the primary gets a retry, then an optional fallback provider, so one provider being down, slow or out of
credit never fails the turn."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue

import numpy as np
import pytest

from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.STT import openrouter_stt_handler as module
from speech_to_speech.STT.openrouter_stt_handler import OpenRouterSTTHandler


class _Provider:
    """A fake /audio/transcriptions server; `script` entries: 'ok', '402', '500', 'hang'."""

    def __init__(self, script, text="hello there"):
        self.script, self.text, self.hits, self.auth = list(script), text, 0, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                kind = outer.script[min(outer.hits, len(outer.script) - 1)]
                outer.hits += 1
                outer.auth.append(self.headers.get("Authorization"))
                if kind == "hang":
                    time.sleep(8)
                    return
                if kind != "ok":
                    self.send_response(int(kind))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps({"text": outer.text}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(module, "RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(module, "PRIMARY_ATTEMPT_TIMEOUT_S", 1.0)
    monkeypatch.setattr(module, "FALLBACK_ATTEMPT_TIMEOUT_S", 1.0)


def _run(primary, fallback, monkeypatch):
    queue: Queue = Queue()
    if fallback is not None:
        monkeypatch.setenv("VOICE_STT_FALLBACK_BASE_URL", fallback.url)
        monkeypatch.setenv("VOICE_STT_FALLBACK_API_KEY", "fb-key")
        monkeypatch.setenv("VOICE_STT_FALLBACK_MODEL", "whisper-1")
    else:
        monkeypatch.delenv("VOICE_STT_FALLBACK_BASE_URL", raising=False)
    h = object.__new__(OpenRouterSTTHandler)
    h.setup(api_key="primary-key", base_url=primary.url, text_output_queue=queue)
    audio = VADAudio(audio=np.zeros(16000, dtype=np.int16), mode="final", turn_id="t1", turn_revision=0)
    out = list(h.process(audio))
    return out, queue


def test_healthy_primary_never_touches_the_fallback(monkeypatch):
    p, f = _Provider(["ok"]), _Provider(["ok"], "from fallback")
    try:
        out, _ = _run(p, f, monkeypatch)
        assert [o.text for o in out] == ["hello there"] and p.hits == 1 and f.hits == 0
    finally:
        p.close(); f.close()


def test_primary_gets_a_retry_before_the_fallback(monkeypatch):
    p, f = _Provider(["500", "ok"]), _Provider(["ok"], "from fallback")
    try:
        out, _ = _run(p, f, monkeypatch)
        assert [o.text for o in out] == ["hello there"] and p.hits == 2 and f.hits == 0
    finally:
        p.close(); f.close()


def test_out_of_credit_goes_straight_to_the_fallback_with_its_own_key(monkeypatch):
    p, f = _Provider(["402"]), _Provider(["ok"], "from fallback")
    try:
        out, _ = _run(p, f, monkeypatch)
        assert [o.text for o in out] == ["from fallback"]
        assert p.hits == 1                                   # no pointless retry of a request that cannot work
        assert p.auth == ["Bearer primary-key"] and f.auth == ["Bearer fb-key"]
    finally:
        p.close(); f.close()


def test_a_hung_primary_is_abandoned_quickly(monkeypatch):
    p, f = _Provider(["hang"]), _Provider(["ok"], "from fallback")
    try:
        t = time.monotonic()
        out, _ = _run(p, f, monkeypatch)
        assert [o.text for o in out] == ["from fallback"]
        assert time.monotonic() - t < 5
    finally:
        p.close(); f.close()


def test_when_every_provider_fails_the_turn_is_reported_failed(monkeypatch):
    p, f = _Provider(["500"]), _Provider(["500"])
    try:
        out, queue = _run(p, f, monkeypatch)
        assert out == [] and p.hits == 2 and f.hits == 2
        event = queue.get_nowait()
        assert isinstance(event, ResponseFailedEvent) and event.stage == "transcription"
    finally:
        p.close(); f.close()


def test_without_a_fallback_configured_behaviour_is_the_primary_alone(monkeypatch):
    p = _Provider(["500"])
    try:
        out, queue = _run(p, None, monkeypatch)
        assert out == [] and p.hits == 2 and isinstance(queue.get_nowait(), ResponseFailedEvent)
    finally:
        p.close()
