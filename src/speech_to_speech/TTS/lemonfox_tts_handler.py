from __future__ import annotations

import logging
import os
from math import gcd
from threading import Event
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
        voice = self._resolve_voice(tts_input)
        payload = {
            "model": self.model_name,
            "input": tts_input.text,
            "voice": voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }

        console.print(f"[green]ASSISTANT: {tts_input.text}")
        request_started_at = perf_counter()
        first_audio_at: float | None = None
        source_buffer = bytearray()
        produced_samples = 0

        with self.client.stream("POST", f"{self.base_url}/audio/speech", json=payload) as response:
            response.raise_for_status()
            for raw_chunk in response.iter_bytes():
                if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
                    logger.info("Lemonfox TTS generation cancelled (interruption)")
                    return
                if not raw_chunk:
                    continue
                if first_audio_at is None:
                    first_audio_at = perf_counter()
                    logger.info(
                        "Lemonfox TTS TTFA: %.2fs (%s)",
                        first_audio_at - request_started_at,
                        voice,
                    )
                    if tts_input.speech_stopped_at_s is not None:
                        logger.info(
                            "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
                            first_audio_at - tts_input.speech_stopped_at_s,
                            tts_input.turn_id,
                            tts_input.turn_revision,
                        )
                source_buffer.extend(raw_chunk)
                for audio_block in self._drain_source_buffer(source_buffer):
                    produced_samples += len(audio_block)
                    yield audio_block

        if source_buffer:
            audio_block = self._pcm24k_bytes_to_block(np.frombuffer(bytes(source_buffer), dtype=np.int16))
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

    def cleanup(self) -> None:
        self.client.close()

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
        response_voice = self._response_voice(tts_input.response)
        if response_voice:
            return self._normalize_voice(response_voice)
        runtime_voice = self._runtime_voice(tts_input.runtime_config)
        if runtime_voice:
            return self._normalize_voice(runtime_voice)
        return self._normalize_voice(self.default_voice)

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
