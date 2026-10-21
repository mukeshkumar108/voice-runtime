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


def test_a_sentence_that_never_starts_is_retried():
    provider = _Provider(["stall_immediately", "stall_immediately", "ok"])
    try:
        scope = CancelScope()
        blocks, elapsed = _run(_handler(provider.url, scope), scope)
        assert blocks and provider.requests == 3 and elapsed < 8
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


# ── sentence look-ahead ─────────────────────────────────────────────────────

class _PacedProvider(_Provider):
    """Streams one second of audio in 100 ms steps after a first-byte delay, and records request times."""

    def __init__(self, first_byte_delay=0.6):
        self.delay = first_byte_delay
        self.started = []
        super().__init__(["paced"])
        handler = self
        base = self.server.RequestHandlerClass
        orig = base.do_POST

        def do_post(req):
            if handler.behaviours[min(handler.requests, len(handler.behaviours) - 1)] == "paced":
                handler.started.append(time.monotonic())
                req.rfile.read(int(req.headers.get("content-length", 0)))
                handler.requests += 1
                req.send_response(200)
                req.send_header("Transfer-Encoding", "chunked")
                req.end_headers()
                time.sleep(handler.delay)
                try:
                    for i in range(10):
                        data = TONE[i * 2400 : (i + 1) * 2400]
                        req.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        req.wfile.flush()
                        time.sleep(0.1)
                    req.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                orig(req)

        base.do_POST = do_post


def _inputs(*texts):
    return [TTSInput(text=t, turn_id="t", turn_revision=0) for t in texts]


def test_next_sentence_is_requested_while_the_current_one_is_still_streaming():
    from queue import Queue

    provider = _PacedProvider(first_byte_delay=0.6)
    try:
        scope = CancelScope()
        handler = _handler(provider.url, scope)
        first, second = _inputs("First sentence.", "Second sentence.")
        handler.queue_in = Queue()
        handler.queue_in.put(second)

        t0 = time.monotonic()
        list(handler.process(first))
        first_done = time.monotonic()
        gen2 = handler.process(second)
        next(gen2)
        gap = time.monotonic() - first_done

        assert len(provider.started) == 2
        # the second request left within moments of the first, not after the first finished
        assert provider.started[1] - provider.started[0] < 0.5
        # so its first audio is ready at once instead of one provider start-up later
        assert gap < 0.3, f"gap between sentences was {gap:.2f}s"
        assert first_done - t0 > 1.4
    finally:
        provider.close()


def test_without_look_ahead_each_sentence_pays_the_provider_start_up_delay():
    provider = _PacedProvider(first_byte_delay=0.6)
    try:
        scope = CancelScope()
        handler = _handler(provider.url, scope)  # no queue_in: nothing to look ahead at
        first, second = _inputs("First sentence.", "Second sentence.")
        list(handler.process(first))
        t = time.monotonic()
        next(handler.process(second))
        assert time.monotonic() - t > 0.5
    finally:
        provider.close()


def test_interrupt_discards_a_look_ahead_in_flight():
    from queue import Queue

    provider = _PacedProvider(first_byte_delay=0.2)
    try:
        scope = CancelScope()
        handler = _handler(provider.url, scope)
        first, second = _inputs("First sentence.", "Second sentence.")
        handler.queue_in = Queue()
        handler.queue_in.put(second)
        gen = handler.process(first)
        next(gen)
        scope.cancel()
        assert list(gen) == []  # the interrupted sentence stops
        # the look-ahead belonged to the old generation: a new response does not reuse it
        assert handler._take_prefetched(("Second sentence.", "aoede"), scope.generation) is None
    finally:
        provider.close()
