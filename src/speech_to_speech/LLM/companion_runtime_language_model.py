"""LLM backend that delegates every turn to companion-runtime.

The brain owns the system prompt, Honcho/Cortex memory, tool execution and
model selection. Voice only sends the transcript (+ bounded chat history) and
streams back the brain's ``text_delta`` events into the normal TTS path.

``POST /v1/turns/stream`` contract: ``docs/streaming-protocol-v1.md`` in
companion-runtime. Deltas are provisional; the terminal ``completed`` event
carries the canonical ``assistant_message`` which is written back to voice
history so the next turn's ``canonical_history`` stays complete.

Voice owns failure *announcement*, never failure *content*: on transport-level
brain failure (unreachable, timeout, provider stream broke) it speaks a fixed
retry prompt. On semantic failure (denied, cancelled, stale) it stays silent.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from time import monotonic
from typing import Any, Optional

import httpx
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
    Usage,
)
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)

# Spoken only when the brain never produced a usable response. Fixed wording:
# voice must not invent conversational content on the brain's behalf.
FALLBACK_TEXT = "Sorry, I lost that for a second. Could you say it again?"

# Brain error codes that mean "no usable response exists" (transport/system).
# Everything else (CapabilityDenied, TURN_CANCELLED, STALE_ATTEMPT, unknown)
# stays silent: either the brain made a decision or a newer attempt owns it.
TRANSPORT_FAILURE_CODES = frozenset(
    {
        "TRANSPORT",  # voice-side HTTP/connection failure, set in _request
        "TURN_TIMEOUT",
        "STREAM_EXECUTION_ERROR",
        "PROVIDER_STREAM_ERROR",
    }
)


def is_transport_failure(error_code: str | None) -> bool:
    """Whether a brain failure should produce the spoken retry prompt."""
    return error_code in TRANSPORT_FAILURE_CODES


class CompanionRuntimeModelHandler(BaseOpenAICompatibleHandler):
    """Streams answers from companion-runtime instead of calling an LLM directly."""

    # ── setup ─────────────────────────────────────────────────────────────

    def setup(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        companion_id: str = "sophie",
        conversation_id: Optional[str] = None,
        selected_model_id: Optional[str] = None,
        timeout_s: float = 120.0,
        user_id: str = "local-user",
        timezone: str = "Europe/London",
        stream: bool = True,
        stream_batch_sentences: int = 1,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = False
        self.gen_kwargs = dict(gen_kwargs or {})
        self.fallback_model_name = None
        self.base_url = (base_url or "http://127.0.0.1:8080").rstrip("/")
        self.companion_id = companion_id
        # Fallback only: realtime requests carry their connection's
        # conversation_id (stashed per-turn in process()); local/socket modes
        # have no connection concept and share this instance-level id.
        self.conversation_id = conversation_id or f"conv_voice_{uuid.uuid4().hex[:12]}"
        self.selected_model_id = selected_model_id
        self.request_timeout_s = float(timeout_s)
        self.request_timeout = httpx.Timeout(self.request_timeout_s, connect=10.0)
        self.user_id = user_id
        self.timezone = timezone
        headers = {"X-Companion-Runtime-Key": api_key} if api_key else {}
        self.client = httpx.Client(timeout=self.request_timeout, headers=headers)
        self.compactor = None
        # Per-turn request state, written in process() on this handler's
        # pipeline thread and consumed by _serialize/_iter_* for that turn.
        self._req_conversation_id: str = self.conversation_id
        self._req_reliability: dict[str, Any] | None = None
        self._pending_brain_turn: tuple[str, str] | None = None
        self._last_brain_error_code: str | None = None
        self._turn_started_at: float = 0.0
        self._request_sent_at: float = 0.0
        self._first_delta_logged = False
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__} at {self.base_url}")
        try:
            response = self.client.get(f"{self.base_url}/health", timeout=5.0)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"companion-runtime is not reachable at {self.base_url}/health: {exc}. "
                "Start it first (uvicorn runtime_api.main:app --port 8080)."
            ) from exc
        logger.info(f"{self.__class__.__name__}: brain reachable")

    # ── per-turn orchestration ────────────────────────────────────────────

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Stash request-scoped brain fields, then run the shared pipeline.

        Also: propagate barge-in cancellation to the brain, and inject the
        fixed spoken retry prompt when the brain fails at the transport level.
        """
        self._req_conversation_id = request.conversation_id or self.conversation_id
        self._req_reliability = self._reliability_from_uncertainty(request.transcript_uncertainty)
        self._pending_brain_turn = None
        self._last_brain_error_code = None
        self._turn_started_at = monotonic()
        self._first_delta_logged = False
        gen = self.cancel_scope.generation if self.cancel_scope else None
        for out in super().process(request):
            if (
                isinstance(out, EndOfResponse)
                and out.error
                and not self._generation_is_stale(gen)
                and self._turn_output_allowed(request.turn_id, request.turn_revision)
                and is_transport_failure(self._last_brain_error_code)
            ):
                logger.info("Brain transport failure (%s); speaking retry prompt", self._last_brain_error_code)
                yield self._fallback_chunk(request, gen)
            yield out
            self._maybe_cancel_brain_turn(gen)
        self._maybe_cancel_brain_turn(gen)

    # ── prompt ownership ──────────────────────────────────────────────────
    # The brain builds the system prompt server-side. Voice must not inject
    # its own instructions, Sophie compiler output, or lang prompt.

    def _apply_config(self, chat: Chat, runtime_config: Any, instructions: Optional[str], wants_audio: bool = True) -> None:
        return

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        def _disabled(system: str, user: str) -> str:
            raise RuntimeError("History compaction via the brain is not wired yet; Chat evicts synchronously.")

        return _disabled

    # ── TurnInput serialisation ───────────────────────────────────────────

    @staticmethod
    def _reliability_from_uncertainty(uncertainty: str | None) -> dict[str, Any]:
        """Map the STT uncertainty overlay to the brain's reliability contract."""
        if not uncertainty:
            return {"source": "voice_stream", "status": "reliable", "confidence": 1.0}
        return {
            "source": "voice_stream",
            "status": "uncertain",
            "confidence": 0.6,
            "reason": uncertainty[:500],
        }

    @staticmethod
    def _history_from_chat(chat: Chat) -> tuple[list[dict[str, Any]], str]:
        """Split voice chat into (prior history, latest user text).

        The trailing user message is the current turn; everything before it
        becomes the brain's ``canonical_history``.
        """
        messages = [m for m in chat.to_transformers_chat() if m.get("role") in ("user", "assistant")]
        texts = [(m.get("role"), str(m.get("content", ""))) for m in messages]
        latest_user = next((text for role, text in reversed(texts) if role == "user"), "")
        prior = texts[:-1] if texts and texts[-1][0] == "user" else texts
        history = [
            {"id": f"hist_{i}", "role": role, "content": text}
            for i, (role, text) in enumerate(prior)
            if text.strip()
        ]
        return history, latest_user

    def _serialize(self, active_chat: Chat) -> dict[str, Any]:
        history, latest_user = self._history_from_chat(active_chat)
        turn_id = f"voice_{uuid.uuid4().hex[:12]}"
        conversation_id = self._req_conversation_id
        self._pending_brain_turn = (turn_id, conversation_id)
        self._request_sent_at = monotonic()
        logger.info(
            "Brain turn start turn=%s conv=%s history=%d reliability=%s",
            turn_id,
            conversation_id,
            len(history),
            (self._req_reliability or {}).get("status"),
        )
        return {
            "contract_version": "v1",
            "turn_id": turn_id,
            "conversation_id": conversation_id,
            "companion_id": self.companion_id,
            "selected_model_id": self.selected_model_id,
            "current_sanitized_message": latest_user,
            "message_parts": [{"type": "text", "text": latest_user}] if latest_user.strip() else [],
            "canonical_history": history,
            "trusted_user_context": {"user_id": self.user_id, "timezone": self.timezone},
            "transcript_reliability": self._req_reliability,
            "medium": "voice",
        }

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        # Tools execute inside the brain. Voice never forwards client tools.
        return {}

    # ── brain cancellation (barge-in) ─────────────────────────────────────

    def _maybe_cancel_brain_turn(self, gen: int | None) -> None:
        """Cancel the in-flight brain turn if the local generation went stale."""
        pending = self._pending_brain_turn
        if pending is None or not self._generation_is_stale(gen):
            return
        turn_id, conversation_id = pending
        self._pending_brain_turn = None
        self._cancel_brain_turn(turn_id, conversation_id)

    def _cancel_brain_turn(self, turn_id: str, conversation_id: str) -> None:
        try:
            response = self.client.post(
                f"{self.base_url}/v1/turns/{turn_id}/cancel",
                params={"conversation_id": conversation_id},
                timeout=5.0,
            )
            response.raise_for_status()
            logger.info("Cancelled brain turn turn=%s", turn_id)
        except Exception as exc:
            logger.warning("Brain turn cancel failed turn=%s: %s", turn_id, exc)

    # ── spoken fallback (transport failure only) ──────────────────────────

    def _fallback_chunk(self, request: LLMIn, gen: int | None) -> LLMResponseChunk:
        return LLMResponseChunk(
            text=FALLBACK_TEXT,
            language_code=request.language_code,
            runtime_config=request.runtime_config,
            response=request.response,
            turn_id=request.turn_id,
            turn_revision=request.turn_revision,
            speech_stopped_at_s=request.speech_stopped_at_s,
            cancel_generation=gen,
        )

    # ── transport ─────────────────────────────────────────────────────────

    def _request(self, api_input: dict[str, Any], optional_kwargs: dict[str, Any]) -> Any:
        try:
            if self.stream:
                request = self.client.build_request(
                    "POST",
                    f"{self.base_url}/v1/turns/stream",
                    json=api_input,
                    headers={"Accept": "text/event-stream"},
                )
                response = self.client.send(request, stream=True)
                if response.status_code >= 400:
                    body = response.read().decode(errors="replace")
                    response.close()
                    raise RuntimeError(f"companion-runtime stream rejected ({response.status_code}): {body[:500]}")
                return response
            response = self.client.post(f"{self.base_url}/v1/turns", json=api_input)
            response.raise_for_status()
            return response.json()
        except Exception:
            # Unreachable brain, timeout, or HTTP rejection: no usable
            # response exists, so the retry prompt may speak for it.
            self._last_brain_error_code = "TRANSPORT"
            raise

    def _iter_stream_events(self, api_response: httpx.Response) -> Iterator[ProviderEvent]:
        event_name: str | None = None
        try:
            for line in api_response.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    logger.warning("Ignoring undecodable SSE data line")
                    event_name = None
                    continue
                if event_name == "text_delta" and data.get("delta"):
                    if not self._first_delta_logged:
                        self._first_delta_logged = True
                        logger.info(
                            "Brain first delta after %.0fms",
                            (monotonic() - self._request_sent_at) * 1000,
                        )
                    yield TextDelta(text=data["delta"])
                elif event_name == "completed":
                    result = data.get("result", {})
                    text = (result.get("assistant_message") or "").strip()
                    if text:
                        yield AssistantMessage(content=[AssistantContent(type="output_text", text=text)])
                    metadata = result.get("execution_metadata", {}) or {}
                    input_tokens = int(metadata.get("input_tokens", 0) or 0)
                    output_tokens = int(metadata.get("output_tokens", 0) or 0)
                    if input_tokens or output_tokens:
                        yield Usage(input_tokens=input_tokens, output_tokens=output_tokens)
                    self._pending_brain_turn = None
                    logger.info(
                        "Brain turn completed after %.0fms",
                        (monotonic() - self._request_sent_at) * 1000,
                    )
                    return
                elif event_name == "error":
                    error = data.get("error", {})
                    code = str(error.get("error_code", "unknown"))
                    self._last_brain_error_code = code
                    self._pending_brain_turn = None
                    raise RuntimeError(f"companion-runtime turn failed ({code}): {error.get('message', '')}")
                event_name = None
        finally:
            try:
                api_response.close()
            except Exception:
                pass

    def _iter_response_events(self, payload: dict[str, Any]) -> Iterator[ProviderEvent]:
        if payload.get("status") in ("failed", "cancelled") or "error_code" in payload:
            code = str(payload.get("error_code", "unknown"))
            self._last_brain_error_code = code
            raise RuntimeError(f"companion-runtime turn failed ({code}): {payload.get('message', '')}")
        text = (payload.get("assistant_message") or "").strip()
        if text:
            yield TextDelta(text=text)
            yield AssistantMessage(content=[AssistantContent(type="output_text", text=text)])
