from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from math import gcd
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Iterator, Optional

import httpx
import numpy as np
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from rich.console import Console
from scipy.signal import resample_poly

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

# Longest silence tolerated *inside* a synthesis stream. The provider sometimes sends a few bytes and then
# hangs; without a short bound the single TTS thread sat blocked for up to a minute and every later reply
# queued behind it.
READ_TIMEOUT_ENV = "VOICE_TTS_READ_TIMEOUT_S"
DEFAULT_READ_TIMEOUT_S = 4.0
# A sentence that produced no audio at all before stalling is requested again (a slow first byte is usually
# the provider being briefly overloaded, and the next request is normally quick).
EMPTY_STREAM_RETRIES = 2
CANCEL_POLL_S = 0.1
# How many queued sentences are requested ahead of playback. One keeps provider concurrency at two requests.
PREFETCH_AHEAD = 1
SUPPORTED_LEMONFOX_VOICES = {
    "heart",
    "bella",
    "michael",
    "alloy",
    "aoede",
    "kore",
    "jessica",
    "nicole",
    "nova",
    "river",
    "sarah",
    "sky",
    "echo",
    "eric",
    "fenrir",
    "liam",
    "onyx",
    "puck",
    "adam",
    "santa",
    "alice",
    "emma",
    "isabella",
    "lily",
    "daniel",
    "fable",
    "george",
    "lewis",
}


@dataclass(frozen=True)
class _Provider:
    """One OpenAI-compatible /audio/speech endpoint."""

    name: str
    base_url: str
    api_key: str
    model: str
    sample_rate: int
    enforce_catalog: bool


ALT_ENV_PREFIX = "VOICE_ALT_TTS_"


def _alt_provider_from_env() -> tuple[_Provider | None, dict[str, str]]:
    """A second provider chosen by voice name, so one voice server can serve products with different voices.

    VOICE_ALT_TTS_BASE_URL / _API_KEY / _MODEL / _SAMPLE_RATE and VOICE_ALT_TTS_VOICES (comma separated). A session
    that asks for one of those voices uses this provider; every other session keeps the default (Lemonfox).
    """
    base = os.getenv(ALT_ENV_PREFIX + "BASE_URL", "").strip().rstrip("/")
    key = os.getenv(ALT_ENV_PREFIX + "API_KEY", "").strip()
    model = os.getenv(ALT_ENV_PREFIX + "MODEL", "").strip()
    voices = {v.strip().lower(): v.strip() for v in os.getenv(ALT_ENV_PREFIX + "VOICES", "").split(",") if v.strip()}
    if not (base and key and model and voices):
        return None, {}
    try:
        rate = int(os.getenv(ALT_ENV_PREFIX + "SAMPLE_RATE", "24000"))
    except ValueError:
        rate = 24000
    return _Provider("alt", base, key, model, rate, enforce_catalog=False), voices


class _LegacyBlocks:
    """Fixed-ratio block resampling used for Lemonfox's 24 kHz audio (unchanged behaviour)."""

    def __init__(self, handler: "LemonfoxTTSHandler") -> None:
        self._handler = handler
        self._buffer = bytearray()

    def feed(self, raw: bytes) -> list[np.ndarray]:
        self._buffer.extend(raw)
        return list(self._handler._drain_source_buffer(self._buffer))

    def finish(self) -> list[np.ndarray]:
        if not self._buffer:
            return []
        return [self._handler._pcm24k_bytes_to_block(np.frombuffer(bytes(self._buffer), dtype=np.int16))]


class _StreamBlocks:
    """Streaming resampler for providers whose rate has no short filter (e.g. 44.1 kHz -> 16 kHz).

    Resampling each small block on its own puts a filter edge click on every block boundary, so the whole
    sentence goes through one stateful soxr stream instead.
    """

    def __init__(self, in_rate: int, out_rate: int, blocksize: int) -> None:
        import soxr

        self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32", quality="HQ")
        self._blocksize = blocksize
        self._carry = b""
        self._pending = np.zeros(0, dtype=np.float32)

    def _emit(self, flush: bool) -> list[np.ndarray]:
        blocks: list[np.ndarray] = []
        while len(self._pending) >= self._blocksize:
            blocks.append(self._to_int16(self._pending[: self._blocksize]))
            self._pending = self._pending[self._blocksize :]
        if flush and len(self._pending):
            tail = np.zeros(self._blocksize, dtype=np.float32)
            tail[: len(self._pending)] = self._pending
            blocks.append(self._to_int16(tail))
            self._pending = np.zeros(0, dtype=np.float32)
        return blocks

    @staticmethod
    def _to_int16(samples: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip(np.rint(samples * 32768.0), -32768, 32767), dtype=np.int16)

    def feed(self, raw: bytes) -> list[np.ndarray]:
        data = self._carry + raw
        usable = len(data) - (len(data) % 2)
        self._carry = data[usable:]
        if usable:
            samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
            self._pending = np.concatenate([self._pending, self._stream.resample_chunk(samples)])
        return self._emit(flush=False)

    def finish(self) -> list[np.ndarray]:
        self._pending = np.concatenate([self._pending, self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)])
        return self._emit(flush=True)


class LemonfoxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Streaming cloud TTS handler for Lemonfox."""

    def setup(
        self,
        should_listen: Event,
        api_key: str | None = None,
        base_url: str = "https://api.lemonfox.ai/v1",
        model_name: str = "speech-1",
        voice: str = "aoede",
        response_format: str = "pcm",
        speed: float = 1.0,
        blocksize: int = 512,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 16000,
        timeout_s: float = 60.0,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.api_key = api_key or os.getenv("LEMONFOX_API_KEY")
        if not self.api_key:
            raise ValueError("Lemonfox TTS requires lemonfox_tts_api_key or LEMONFOX_API_KEY.")

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.default_voice = voice
        self.response_format = response_format
        self.speed = speed
        self.blocksize = blocksize
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.timeout_s = timeout_s
        self.gen_kwargs = gen_kwargs or {}
        self._default_provider = _Provider(
            "default", self.base_url, self.api_key, self.model_name, self.input_sample_rate, enforce_catalog=True
        )
        self._alt_provider, self._alt_voices = _alt_provider_from_env()
        self._prefetch: dict[tuple[str, str], tuple[int | None, Queue]] = {}
        self._prefetch_lock = Lock()
        self.client = httpx.Client(
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

        self._resample_up, self._resample_down = self._resample_ratio()

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug(
                "Dropping stale Lemonfox TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        generation = self.cancel_scope.generation if self.cancel_scope else None
        provider, voice = self._resolve(tts_input)
        payload = self._payload(tts_input.text, voice, provider)

        console.print(f"[green]ASSISTANT: {tts_input.text}")
        request_started_at = perf_counter()
        first_audio_at: float | None = None
        produced_samples = 0
        blocks = self._new_blocks(provider)

        # This sentence may already be on its way (requested while the previous one was playing); then
        # start the next queued sentence's request so its audio is ready when this one ends.
        raw_chunks = self._sentence_chunks((tts_input.text, voice), payload, provider, generation, tts_input.turn_id)
        self._prefetch_ahead(generation)

        for raw_chunk, prefetched in raw_chunks:
            if self._is_cancelled(generation):
                logger.info("Lemonfox TTS generation cancelled (interruption)")
                return
            if not raw_chunk:
                continue
            if first_audio_at is None:
                first_audio_at = perf_counter()
                logger.info(
                    "Lemonfox TTS TTFA: %.2fs (%s%s%s)",
                    first_audio_at - request_started_at,
                    voice,
                    "" if provider is self._default_provider else f" via {provider.model}",
                    ", prefetched" if prefetched else "",
                )
                if tts_input.speech_stopped_at_s is not None:
                    logger.info(
                        "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
                        first_audio_at - tts_input.speech_stopped_at_s,
                        tts_input.turn_id,
                        tts_input.turn_revision,
                    )
            for audio_block in blocks.feed(raw_chunk):
                produced_samples += len(audio_block)
                yield audio_block
        if self._is_cancelled(generation):
            return

        for audio_block in blocks.finish():
            produced_samples += len(audio_block)
            yield audio_block

        elapsed_s = perf_counter() - request_started_at
        produced_s = produced_samples / self.output_sample_rate
        logger.info(
            "Lemonfox TTS produced %.2fs audio in %.2fs (rate=%.2fx, turn=%s rev=%s)",
            produced_s,
            elapsed_s,
            produced_s / elapsed_s if elapsed_s > 0 else 0.0,
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def _new_blocks(self, provider: _Provider) -> "_LegacyBlocks | _StreamBlocks":
        common = gcd(provider.sample_rate, self.output_sample_rate)
        up, down = self.output_sample_rate // common, provider.sample_rate // common
        if provider.sample_rate != self.input_sample_rate or max(up, down) > 16:
            # e.g. 44.1 kHz -> 16 kHz (160/441): needs one stateful stream, not per-block filters
            return _StreamBlocks(provider.sample_rate, self.output_sample_rate, self.blocksize)
        return _LegacyBlocks(self)

    def _payload(self, text: str, voice: str, provider: _Provider) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": provider.model,
            "input": text,
            "voice": voice,
            "response_format": self.response_format,
        }
        if self.speed != 1.0:  # not every provider accepts the field, and 1.0 is the default anyway
            payload["speed"] = self.speed
        return payload

    def _iter_raw_chunks(
        self, payload: dict[str, Any], provider: _Provider, generation: int | None, turn_id: str | None
    ) -> Iterator[bytes]:
        """Raw audio bytes for one sentence, with every guard: bounded reads, interruption, one retry path."""
        started_at = perf_counter()
        received = 0
        attempts = 0
        while True:
            attempts += 1
            stalled = False
            watcher_done = Event()
            try:
                with self.client.stream(
                    "POST",
                    f"{provider.base_url}/audio/speech",
                    json=payload,
                    headers={"Authorization": f"Bearer {provider.api_key}"},
                    timeout=self._request_timeout(),
                ) as response:
                    response.raise_for_status()
                    # Closing the response is the only way to unblock a read that is waiting on the
                    # provider, so an interruption watches for itself instead of waiting for the next byte.
                    Thread(
                        target=self._close_on_cancel,
                        args=(response, generation, watcher_done),
                        name="lemonfox-cancel-watch",
                        daemon=True,
                    ).start()
                    try:
                        for raw_chunk in response.iter_bytes():
                            if self._is_cancelled(generation):
                                return
                            if raw_chunk:
                                received += len(raw_chunk)
                                yield raw_chunk
                    finally:
                        watcher_done.set()
            except (httpx.TransportError, httpx.StreamError, OSError) as exc:
                if self._is_cancelled(generation):
                    return
                stalled = True
                logger.warning(
                    "Lemonfox TTS stream failed after %.1fs, %d byte(s) so far (attempt %d, turn=%s): %s: %s",
                    perf_counter() - started_at,
                    received,
                    attempts,
                    turn_id,
                    type(exc).__name__,
                    exc,
                )
            if self._is_cancelled(generation):
                return
            if stalled and received == 0 and attempts <= EMPTY_STREAM_RETRIES:
                continue
            return

    # ── look-ahead: request the next queued sentence while this one is still being produced ──

    def _peek_upcoming(self, limit: int) -> list[TTSInput]:
        queue_in = getattr(self, "queue_in", None)
        if queue_in is None:
            return []
        with queue_in.mutex:
            items = list(queue_in.queue)
        upcoming: list[TTSInput] = []
        for item in items:
            if not isinstance(item, TTSInput):
                break  # end of this response, a control message, or anything that is not plain speech
            upcoming.append(item)
            if len(upcoming) >= limit:
                break
        return upcoming

    def _prefetch_ahead(self, generation: int | None) -> None:
        speculative_turns = getattr(self, "speculative_turns", None)
        with self._prefetch_lock:
            # forget look-aheads that belong to an interrupted response
            for key in [k for k, v in self._prefetch.items() if v[0] != generation]:
                del self._prefetch[key]
            for upcoming in self._peek_upcoming(PREFETCH_AHEAD):
                if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                    upcoming.turn_id, upcoming.turn_revision
                ):
                    continue
                provider, voice = self._resolve(upcoming)
                key = (upcoming.text, voice)
                if key in self._prefetch or not upcoming.text.strip():
                    continue
                chunks: Queue = Queue()
                self._prefetch[key] = (generation, chunks)
                payload = self._payload(upcoming.text, voice, provider)
                Thread(
                    target=self._fill_prefetch,
                    args=(payload, provider, generation, upcoming.turn_id, chunks),
                    name="lemonfox-prefetch",
                    daemon=True,
                ).start()

    def _fill_prefetch(
        self, payload: dict[str, Any], provider: _Provider, generation: int | None, turn_id: str | None, out: Queue
    ) -> None:
        try:
            for raw_chunk in self._iter_raw_chunks(payload, provider, generation, turn_id):
                out.put(raw_chunk)
        except Exception as exc:  # the consumer falls back to a direct request if nothing arrived
            logger.warning("Lemonfox TTS prefetch failed: %s: %s", type(exc).__name__, exc)
        finally:
            out.put(None)

    def _sentence_chunks(
        self, key: tuple[str, str], payload: dict[str, Any], provider: _Provider, generation: int | None, turn_id: str | None
    ) -> Iterator[tuple[bytes, bool]]:
        """(chunk, was_prefetched) pairs; a look-ahead that delivered nothing falls back to a direct request."""
        early = self._take_prefetched(key, generation)
        if early is not None:
            got = False
            for chunk in early:
                got = True
                yield chunk, True
            if got or self._is_cancelled(generation):
                return
        for chunk in self._iter_raw_chunks(payload, provider, generation, turn_id):
            yield chunk, False

    def _take_prefetched(self, key: tuple[str, str], generation: int | None) -> Iterator[bytes] | None:
        with self._prefetch_lock:
            entry = self._prefetch.pop(key, None)
        if entry is None or entry[0] != generation:
            return None
        chunks: Queue = entry[1]

        def read() -> Iterator[bytes]:
            while True:
                try:
                    item = chunks.get(timeout=CANCEL_POLL_S)
                except Empty:
                    if self._is_cancelled(generation):
                        return
                    continue
                if item is None:
                    return
                yield item

        return read()

    def cleanup(self) -> None:
        self.client.close()

    def _request_timeout(self) -> httpx.Timeout:
        try:
            read_s = float(os.getenv(READ_TIMEOUT_ENV, DEFAULT_READ_TIMEOUT_S))
        except ValueError:
            read_s = DEFAULT_READ_TIMEOUT_S
        return httpx.Timeout(self.timeout_s, connect=10.0, read=max(0.5, read_s))

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    def _close_on_cancel(self, response: httpx.Response, generation: int | None, done: Event) -> None:
        while not done.wait(CANCEL_POLL_S):
            if self._is_cancelled(generation):
                # Closing a response does not wake a read already blocked in another thread; shutting
                # the socket down does, and the blocked read then raises into the cancel path above.
                try:
                    network_stream = response.extensions.get("network_stream")
                    sock = network_stream.get_extra_info("socket") if network_stream is not None else None
                    if sock is not None:
                        sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    response.close()
                except Exception:  # the read side may already be gone
                    pass
                return

    def _drain_source_buffer(self, source_buffer: bytearray) -> Iterator[np.ndarray]:
        source_bytes_per_block = self.blocksize * 2 * self._resample_down // self._resample_up
        if source_bytes_per_block <= 0:
            source_bytes_per_block = self.blocksize * 2

        while len(source_buffer) >= source_bytes_per_block:
            chunk = bytes(source_buffer[:source_bytes_per_block])
            del source_buffer[:source_bytes_per_block]
            yield self._pcm24k_bytes_to_block(np.frombuffer(chunk, dtype=np.int16))

    def _pcm24k_bytes_to_block(self, chunk: np.ndarray) -> np.ndarray:
        if self.input_sample_rate != self.output_sample_rate:
            resampled = resample_poly(chunk.astype(np.float32), self._resample_up, self._resample_down)
        else:
            resampled = chunk.astype(np.float32)

        audio_int16 = np.asarray(np.clip(np.rint(resampled), -32768, 32767), dtype=np.int16)
        if len(audio_int16) < self.blocksize:
            audio_int16 = np.pad(audio_int16, (0, self.blocksize - len(audio_int16)))
        elif len(audio_int16) > self.blocksize:
            audio_int16 = audio_int16[: self.blocksize]
        return audio_int16

    def _resample_ratio(self) -> tuple[int, int]:
        common = gcd(self.input_sample_rate, self.output_sample_rate)
        return self.output_sample_rate // common, self.input_sample_rate // common

    def _resolve_voice(self, tts_input: TTSInput) -> str:
        return self._resolve(tts_input)[1]

    def _resolve(self, tts_input: TTSInput) -> tuple[_Provider, str]:
        """Which provider and voice this sentence uses (by the voice the session asked for)."""
        requested = self._response_voice(tts_input.response) or self._runtime_voice(tts_input.runtime_config)
        if requested and self._alt_provider is not None and requested.strip().lower() in self._alt_voices:
            return self._alt_provider, self._alt_voices[requested.strip().lower()]
        return self._default_provider, self._normalize_voice(requested or self.default_voice)

    def _normalize_voice(self, voice: str) -> str:
        normalized = voice.strip().lower()
        if normalized in SUPPORTED_LEMONFOX_VOICES:
            return normalized
        logger.info("Lemonfox TTS voice '%s' is unsupported; falling back to %s", voice, self.default_voice)
        return self.default_voice

    def _response_voice(self, response: RealtimeResponseCreateParams | None) -> Optional[str]:
        if response and response.audio and response.audio.output and response.audio.output.voice:
            return str(response.audio.output.voice)
        return None

    def _runtime_voice(self, runtime_config: RuntimeConfig | None) -> Optional[str]:
        if runtime_config and runtime_config.session.audio and runtime_config.session.audio.output:
            voice = runtime_config.session.audio.output.voice
            if voice:
                return str(voice)
        return None
