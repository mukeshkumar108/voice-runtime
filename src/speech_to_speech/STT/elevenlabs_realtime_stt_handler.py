from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from queue import Queue
from threading import Event
from time import perf_counter
from typing import Any, Iterator, Optional
from urllib.parse import urlencode

import httpx
import numpy as np
from rich.console import Console
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

console = Console()
logger = logging.getLogger(__name__)


class RealtimeSTTConnectionError(RuntimeError):
    """Transient ElevenLabs websocket failure eligible for one replay."""


@dataclass(frozen=True)
class _TranscriptResult:
    text: str
    language_code: str | None = None
    average_logprob: float | None = None
    minimum_logprob: float | None = None
    word_count: int = 0
    payload_keys: tuple[str, ...] = ()


class ElevenLabsRealtimeSTTHandler(BaseSTTHandler):
    """Realtime STT adapter for ElevenLabs Scribe v2 Realtime."""

    def setup(
        self,
        api_key: str | None = None,
        model_id: str = "scribe_v2_realtime",
        audio_format: str = "pcm_16000",
        language_code: str | None = None,
        timeout_s: float = 12.0,
        open_timeout_s: float = 10.0,
        text_output_queue: Queue[TextEventItem] | None = None,
        should_listen: Event | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs realtime STT requires elevenlabs_realtime_stt_api_key or ELEVENLABS_API_KEY."
            )
        self.model_id = model_id
        self.audio_format = audio_format
        self.language_code = language_code
        self.timeout_s = timeout_s
        self.open_timeout_s = open_timeout_s
        self.text_output_queue = text_output_queue
        self.should_listen = should_listen
        self.gen_kwargs = gen_kwargs or {}
        self.minimum_average_logprob = float(os.getenv("SOPHIE_STT_MIN_AVERAGE_LOGPROB", "-0.7"))
        self.minimum_word_logprob = float(os.getenv("SOPHIE_STT_MIN_WORD_LOGPROB", "-1.5"))
        self.sample_rate = 16000
        self.http_client = httpx.Client(
            timeout=20.0,
            headers={"xi-api-key": self.api_key},
        )

        self._ws: ClientConnection | None = None
        self._token: str | None = None
        self._session_id: str | None = None
        self._current_turn_id: str | None = None
        self._current_turn_revision: int | None = None
        self._last_sent_samples = 0

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.turn_id != self._current_turn_id or vad_audio.turn_revision != self._current_turn_revision:
            self._reset_turn_tracking(vad_audio.turn_id, vad_audio.turn_revision)

        audio = vad_audio.audio
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        if audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype=np.int16)
            else:
                audio = audio.astype(np.int16)

        for attempt in range(2):
            delta = self._delta_audio(audio)
            try:
                yield from self._process_connected_audio(vad_audio, audio, delta)
                return
            except (RealtimeSTTConnectionError, ConnectionClosed) as exc:
                self._disconnect()
                self._last_sent_samples = 0
                if attempt == 0:
                    logger.warning(
                        "ElevenLabs realtime STT connection dropped turn=%s rev=%s; replaying %.3fs once: %s",
                        vad_audio.turn_id,
                        vad_audio.turn_revision,
                        len(audio) / self.sample_rate,
                        exc,
                    )
                    continue
                self._report_failure(vad_audio, exc)
                return
            except Exception as exc:
                self._disconnect()
                self._last_sent_samples = 0
                self._report_failure(vad_audio, exc)
                return

    def _process_connected_audio(
        self,
        vad_audio: STTIn,
        audio: np.ndarray,
        delta: np.ndarray,
    ) -> Iterator[STTOut]:
        self._ensure_connection()
        partials = self._send_audio_and_collect_partials(delta)
        if partials:
            yield PartialTranscription(
                text=partials[-1],
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )

        if vad_audio.mode != "final":
            return

        started_at = perf_counter()
        result = self._commit_and_wait_for_transcript()
        latency_s = perf_counter() - started_at
        if result and result.text:
            uncertainty_reason = self._uncertainty_reason(result)
            logger.info(
                "ElevenLabs realtime STT completed turn=%s rev=%s model=%s audio=%.3fs latency=%.3fs "
                "chars=%d words=%d avg_logprob=%s min_logprob=%s uncertain=%s payload_keys=%s",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                self.model_id,
                len(audio) / self.sample_rate,
                latency_s,
                len(result.text),
                result.word_count,
                self._format_score(result.average_logprob),
                self._format_score(result.minimum_logprob),
                bool(uncertainty_reason),
                ",".join(result.payload_keys),
            )
            console.print(f"[yellow]USER: {result.text}")
            yield Transcription(
                text=result.text,
                language_code=result.language_code or self.language_code,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
                average_logprob=result.average_logprob,
                minimum_logprob=result.minimum_logprob,
                uncertainty_reason=uncertainty_reason,
                provider_metadata={
                    "provider": "elevenlabs",
                    "model": self.model_id,
                    "word_count": result.word_count,
                    "payload_keys": list(result.payload_keys),
                },
            )
        else:
            self._notify_failure(
                f"Speech recognition timed out with {self.model_id}. Please try again.",
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
        self._last_sent_samples = 0

    def _report_failure(self, vad_audio: STTIn, exc: Exception) -> None:
        logger.error(
            "ElevenLabs realtime STT failed turn=%s rev=%s model=%s: %s",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            self.model_id,
            exc,
            exc_info=True,
        )
        self._notify_failure(
            f"Speech recognition failed with {self.model_id}. Please try again.",
            vad_audio.turn_id,
            vad_audio.turn_revision,
        )

    def cleanup(self) -> None:
        self._disconnect()
        self.http_client.close()

    def _ensure_connection(self) -> None:
        if self._ws is not None:
            return

        params = {
            "model_id": self.model_id,
            "audio_format": self.audio_format,
            "commit_strategy": "manual",
            "token": self._issue_token(),
            "include_timestamps": "true",
            "timestamps_granularity": "word",
        }
        if self.language_code:
            params["language_code"] = self.language_code

        uri = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{urlencode(params)}"
        self._ws = connect(
            uri,
            open_timeout=self.open_timeout_s,
            close_timeout=2.0,
            max_queue=32,
        )
        session_message = self._recv_json(timeout=3.0)
        if session_message.get("message_type") != "session_started":
            raise RuntimeError(f"Unexpected ElevenLabs session start payload: {session_message}")
        self._session_id = str(session_message.get("session_id"))
        logger.info("ElevenLabs realtime STT connected session=%s model=%s", self._session_id, self.model_id)

    def _issue_token(self) -> str:
        response = self.http_client.post("https://api.elevenlabs.io/v1/single-use-token/realtime_scribe")
        response.raise_for_status()
        token = str(response.json()["token"])
        self._token = token
        return token

    def _send_audio_and_collect_partials(self, delta: np.ndarray) -> list[str]:
        if delta.size == 0:
            return self._drain_partial_messages(timeout=0.01)
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(delta.tobytes()).decode("ascii"),
        }
        assert self._ws is not None
        self._ws.send(json.dumps(payload))
        return self._drain_partial_messages(timeout=0.05)

    def _commit_and_wait_for_transcript(self) -> Optional[_TranscriptResult]:
        assert self._ws is not None
        self._ws.send(json.dumps({"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True}))
        deadline = perf_counter() + self.timeout_s
        latest_final: dict[str, Any] | None = None
        committed: dict[str, Any] | None = None

        while perf_counter() < deadline:
            remaining = max(0.05, deadline - perf_counter())
            message = self._recv_json(timeout=remaining)
            message_type = str(message.get("message_type"))
            if message_type == "partial_transcript":
                continue
            if message_type == "final_transcript":
                if str(message.get("text", "")).strip():
                    latest_final = message
                continue
            if message_type == "committed_transcript":
                committed = message
                deadline = min(deadline, perf_counter() + 0.75)
                continue
            if message_type == "committed_transcript_with_timestamps":
                return self._transcript_result(message)
            if message_type.endswith("error") or message_type in {
                "error",
                "auth_error",
                "quota_exceeded",
                "rate_limited",
                "resource_exhausted",
                "transcriber_error",
                "queue_overflow",
            }:
                raise RuntimeError(message.get("error") or message.get("message") or f"ElevenLabs error: {message_type}")
        payload = committed or latest_final
        return self._transcript_result(payload) if payload else None

    def _transcript_result(self, message: dict[str, Any]) -> _TranscriptResult:
        words = message.get("words")
        word_items = words if isinstance(words, list) else []
        logprobs = [
            float(word["logprob"])
            for word in word_items
            if isinstance(word, dict) and word.get("type") == "word" and isinstance(word.get("logprob"), (int, float))
        ]
        return _TranscriptResult(
            text=str(message.get("text", "")).strip(),
            language_code=str(message["language_code"]) if message.get("language_code") else None,
            average_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
            minimum_logprob=min(logprobs) if logprobs else None,
            word_count=sum(1 for word in word_items if isinstance(word, dict) and word.get("type") == "word"),
            payload_keys=tuple(sorted(str(key) for key in message)),
        )

    def _uncertainty_reason(self, result: _TranscriptResult) -> str | None:
        if result.average_logprob is not None and result.average_logprob < self.minimum_average_logprob:
            return "low_average_word_logprob"
        if result.minimum_logprob is not None and result.minimum_logprob < self.minimum_word_logprob:
            return "very_low_word_logprob"
        return None

    @staticmethod
    def _format_score(value: float | None) -> str:
        return "unavailable" if value is None else f"{value:.3f}"

    def _drain_partial_messages(self, timeout: float) -> list[str]:
        partials: list[str] = []
        end = perf_counter() + timeout
        while perf_counter() < end:
            remaining = end - perf_counter()
            try:
                message = self._recv_json(timeout=max(0.001, remaining))
            except TimeoutError:
                break
            message_type = str(message.get("message_type"))
            if message_type == "partial_transcript":
                text = str(message.get("text", "")).strip()
                if text:
                    partials.append(text)
            elif message_type in {"final_transcript", "committed_transcript"}:
                text = str(message.get("text", "")).strip()
                if text:
                    partials.append(text)
            elif message_type.endswith("error") or message_type in {
                "error",
                "auth_error",
                "quota_exceeded",
                "rate_limited",
                "resource_exhausted",
                "transcriber_error",
                "queue_overflow",
            }:
                raise RuntimeError(message.get("error") or message.get("message") or f"ElevenLabs error: {message_type}")
        return partials

    def _recv_json(self, timeout: float) -> dict[str, Any]:
        assert self._ws is not None
        try:
            raw = self._ws.recv(timeout=timeout)
        except TimeoutError:
            raise
        except ConnectionClosed as exc:
            self._disconnect()
            raise RealtimeSTTConnectionError(f"ElevenLabs websocket closed: {exc}") from exc
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _delta_audio(self, audio: np.ndarray) -> np.ndarray:
        if self._last_sent_samples > len(audio):
            self._last_sent_samples = 0
        delta = audio[self._last_sent_samples :]
        self._last_sent_samples = len(audio)
        return delta

    def _reset_turn_tracking(self, turn_id: str | None, turn_revision: int | None) -> None:
        self._current_turn_id = turn_id
        self._current_turn_revision = turn_revision
        self._last_sent_samples = 0

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

    def _disconnect(self) -> None:
        if self._ws is None:
            self._token = None
            self._session_id = None
            return
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws = None
        self._token = None
        self._session_id = None
