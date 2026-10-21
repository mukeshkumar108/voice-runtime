"""The TTS thread must never sit blocked on a stalled provider stream.

A fake provider sends a little audio and then hangs, as Lemonfox did on the deployed server. The handler has to
bound the wait, free itself immediately on an interruption, and retry a sentence that produced nothing.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS import lemonfox_tts_handler as module
from speech_to_speech.TTS.lemonfox_tts_handler import LemonfoxTTSHandler

TONE = (np.sin(np.linspace(0, 200, 24000)) * 8000).astype("<i2").tobytes()  # 1 s at 24 kHz


class _Provider:
    """Scripted behaviours, one per request, then the last one repeats."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.requests = 0
        handler = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                kind = handler.behaviours[min(handler.requests, len(handler.behaviours) - 1)]
                handler.requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(data: bytes):
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()

                try:
                    if kind == "ok":
                        chunk(TONE)
                        self.wfile.write(b"0\r\n\r\n")
                    elif kind == "stall_after_audio":
                        chunk(TONE[:9600])
                        time.sleep(30)
                    elif kind == "stall_immediately":
                        time.sleep(30)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


def _handler(url, scope):
    handler = object.__new__(LemonfoxTTSHandler)
    handler.setup(should_listen=Event(), api_key="k", base_url=url, cancel_scope=scope)
    return handler


def _run(handler, scope):
    inp = TTSInput(text="Hello there.", turn_id="t", turn_revision=0, cancel_generation=scope.generation)
    started = time.monotonic()
    blocks = list(handler.process(inp))
    return blocks, time.monotonic() - started


@pytest.fixture(autouse=True)
def short_read_timeout(monkeypatch):
    monkeypatch.setenv(module.READ_TIMEOUT_ENV, "1.0")


def test_healthy_stream_is_unchanged():
    provider = _Provider(["ok"])
    try:
        scope = CancelScope()
        blocks, elapsed = _run(_handler(provider.url, scope), scope)
        assert blocks and elapsed < 5 and provider.requests == 1
    finally:
        provider.close()


def test_midstream_stall_is_bounded_and_keeps_the_audio_it_had():
    provider = _Provider(["stall_after_audio"])
    try:
        scope = CancelScope()
        blocks, elapsed = _run(_handler(provider.url, scope), scope)
        assert blocks, "audio received before the stall must still be played"
        assert elapsed < 5, f"waited {elapsed:.1f}s on a stalled stream"
        assert provider.requests == 1, "a sentence that already produced audio is not repeated"
    finally:
        provider.close()


def test_a_sentence_that_never_starts_is_retried_once():
    provider = _Provider(["stall_immediately", "ok"])
    try:
        scope = CancelScope()
        blocks, elapsed = _run(_handler(provider.url, scope), scope)
        assert blocks and provider.requests == 2 and elapsed < 6
    finally:
        provider.close()


def test_interruption_frees_the_thread_immediately_without_waiting_for_the_read_timeout(monkeypatch):
    monkeypatch.setenv(module.READ_TIMEOUT_ENV, "20")  # prove cancel does not depend on the timeout
    provider = _Provider(["stall_after_audio"])
    try:
        scope = CancelScope()
        handler = _handler(provider.url, scope)
        threading.Timer(0.6, scope.cancel).start()
        blocks, elapsed = _run(handler, scope)
        assert elapsed < 3, f"interrupt took {elapsed:.1f}s to free the TTS thread"
    finally:
        provider.close()
