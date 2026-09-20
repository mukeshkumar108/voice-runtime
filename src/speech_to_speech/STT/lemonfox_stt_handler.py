from __future__ import annotations

import io
import logging
import os
from queue import Queue
from threading import Event
from time import perf_counter
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


class LemonfoxSTTHandler(BaseSTTHandler):
    """Cloud STT handler for Lemonfox.

    This handler transcribes finalized VAD turns via Lemonfox's upload-style
    transcription endpoint. It intentionally ignores progressive inputs because
    Lemonfox is not wired here as a partial-result websocket STT provider.
    """

    def setup(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.lemonfox.ai/v1",
        model_name: str = "whisper-1",
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        text_output_queue: Queue[TextEventItem] | None = None,
        should_listen: Event | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LEMONFOX_API_KEY")
        if not self.api_key:
            raise ValueError("Lemonfox STT requires lemonfox_stt_api_key or LEMONFOX_API_KEY.")

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.language = language
        self.prompt = prompt
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.text_output_queue = text_output_queue
        self.should_listen = should_listen
        self.gen_kwargs = gen_kwargs or {}
        self.client = httpx.Client(
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode != "final":
            return

        wav_bytes = self._wav_bytes_from_audio(vad_audio.audio)
        data: dict[str, Any] = {
            "model": self.model_name,
            "temperature": str(self.temperature),
        }
        if self.language:
            data["language"] = self.language
        if self.prompt:
            data["prompt"] = self.prompt

        started_at = perf_counter()
        try:
            response = self.client.post(
                f"{self.base_url}/audio/transcriptions",
                data=data,
                files={"file": ("turn.wav", wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
            latency_s = perf_counter() - started_at
        except Exception as exc:
            latency_s = perf_counter() - started_at
            logger.error(
                "Lemonfox STT failed turn=%s rev=%s model=%s after %.3fs: %s",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                self.model_name,
                latency_s,
                exc,
                exc_info=True,
            )
            self._notify_failure(
                f"Speech recognition timed out with {self.model_name}. Please try again.",
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
            return

        payload = response.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            logger.debug("Lemonfox STT returned no text for turn=%s rev=%s", vad_audio.turn_id, vad_audio.turn_revision)
            return

        logger.info(
            "Lemonfox STT completed turn=%s rev=%s audio=%.3fs latency=%.3fs chars=%d",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            len(vad_audio.audio) / 16000.0,
            latency_s,
            len(text),
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
