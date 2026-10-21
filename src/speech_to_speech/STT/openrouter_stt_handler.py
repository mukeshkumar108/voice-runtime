from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from queue import Queue
from threading import Event
from time import perf_counter, sleep
from typing import Any, Iterator

import httpx
import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

console = Console()
logger = logging.getLogger(__name__)

FALLBACK_ENV = "VOICE_STT_FALLBACK_"
PRIMARY_ATTEMPTS = 2
FALLBACK_ATTEMPTS = 2
# Per-attempt time allowed before moving on: a turn waiting on a hung recogniser is worse than a quick switch.
PRIMARY_ATTEMPT_TIMEOUT_S = 6.0
FALLBACK_ATTEMPT_TIMEOUT_S = 10.0
RETRY_BACKOFF_S = 0.3
# These mean "this endpoint will not work for this request" (bad key, no credit, wrong model): retrying it is pointless.
NO_RETRY_STATUS = {400, 401, 402, 403, 404, 422}


@dataclass(frozen=True)
class _Endpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    attempts: int
    timeout_s: float


class OpenRouterSTTHandler(BaseSTTHandler):
    """Cloud STT handler for OpenRouter transcription models.

    This currently transcribes finalized VAD turns only via
    `/api/v1/audio/transcriptions`.
    """

    def setup(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        language: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        text_output_queue: Queue[TextEventItem] | None = None,
        should_listen: Event | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter STT requires openrouter_stt_api_key or OPENROUTER_API_KEY.")

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.language = language
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.text_output_queue = text_output_queue
        self.should_listen = should_listen
        self.gen_kwargs = gen_kwargs or {}
        self.client = httpx.Client(timeout=self.timeout_s)
        # Primary first (with a retry), then an optional OpenAI-compatible fallback (e.g. Lemonfox Whisper), so one
        # provider being down, slow or out of credit does not fail the turn.
        self._endpoints = [_Endpoint("primary", self.base_url, self.api_key, self.model_name, PRIMARY_ATTEMPTS, min(self.timeout_s, PRIMARY_ATTEMPT_TIMEOUT_S))]
        fb_url = os.getenv(FALLBACK_ENV + "BASE_URL", "").strip().rstrip("/")
        fb_key = os.getenv(FALLBACK_ENV + "API_KEY", "").strip()
        fb_model = os.getenv(FALLBACK_ENV + "MODEL", "").strip() or "whisper-1"
        if fb_url and fb_key:
            self._endpoints.append(_Endpoint("fallback", fb_url, fb_key, fb_model, FALLBACK_ATTEMPTS, FALLBACK_ATTEMPT_TIMEOUT_S))

    def _transcribe(self, wav_bytes: bytes, base_data: dict[str, Any], turn_id: str | None) -> tuple[dict[str, Any], _Endpoint, float] | None:
        """(payload, endpoint used, seconds) from the first endpoint that answers, or None when every attempt failed."""
        started_at = perf_counter()
        for endpoint in self._endpoints:
            for attempt in range(endpoint.attempts):
                data = dict(base_data, model=endpoint.model)
                try:
                    response = self.client.post(
                        f"{endpoint.base_url}/audio/transcriptions",
                        data=data,
                        files={"file": ("turn.wav", wav_bytes, "audio/wav")},
                        headers={"Authorization": f"Bearer {endpoint.api_key}"},
                        timeout=endpoint.timeout_s,
                    )
                    response.raise_for_status()
                    return response.json(), endpoint, perf_counter() - started_at
                except Exception as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    logger.warning(
                        "STT %s (%s) attempt %d/%d failed turn=%s after %.2fs: %s%s",
                        endpoint.name, endpoint.model, attempt + 1, endpoint.attempts, turn_id,
                        perf_counter() - started_at, type(exc).__name__, f" HTTP {status}" if status else "",
                    )
                    if status in NO_RETRY_STATUS:
                        break  # no credit / bad key / bad request: move straight to the next provider
                    if attempt + 1 < endpoint.attempts:
                        sleep(RETRY_BACKOFF_S)
        return None

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode != "final":
            return

        wav_bytes = self._wav_bytes_from_audio(vad_audio.audio)
        data: dict[str, Any] = {
            "model": self.model_name,
            "temperature": str(self.temperature),
            "response_format": "json",
        }
        if self.language:
            data["language"] = self.language

        result = self._transcribe(wav_bytes, data, vad_audio.turn_id)
        if result is None:
            logger.error(
                "STT failed on every provider turn=%s rev=%s (%s)",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                ", ".join(f"{e.name}={e.model}" for e in self._endpoints),
            )
            self._notify_failure(
                "Speech recognition is unavailable right now. Please try again.",
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
            return
        payload, endpoint, latency_s = result
        text = str(payload.get("text", "")).strip()
        usage = payload.get("usage") or {}
        if not text:
            logger.debug(
                "OpenRouter STT returned no text for turn=%s rev=%s",
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
            return

        logger.info(
            "OpenRouter STT completed turn=%s rev=%s model=%s (%s) audio=%.3fs latency=%.3fs chars=%d cost=%s",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            endpoint.model,
            endpoint.name,
            len(vad_audio.audio) / 16000.0,
            latency_s,
            len(text),
            usage.get("cost"),
        )
        console.print(f"[yellow]USER: {text}")
        yield Transcription(
            text=text,
            language_code=self.language,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def cleanup(self) -> None:
        self.client.close()

    def _notify_failure(self, message: str, turn_id: str | None, turn_revision: int | None) -> None:
        if self.text_output_queue is not None:
            self.text_output_queue.put(
                ResponseFailedEvent(
                    message=message,
                    stage="transcription",
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                )
            )
        if self.should_listen is not None:
            self.should_listen.set()

    def _wav_bytes_from_audio(self, audio: np.ndarray) -> bytes:
        pcm = self._to_pcm16(audio)
        with io.BytesIO() as buffer:
            import wave

            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm.tobytes())
            return buffer.getvalue()

    def _to_pcm16(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.int16:
            return audio
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
