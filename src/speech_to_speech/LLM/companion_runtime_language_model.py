"""LLM backend that delegates every turn to companion-runtime.

The brain owns the system prompt, Honcho/Cortex memory, tool execution and
model selection. Voice only sends the transcript (+ bounded chat history) and
streams back the brain's ``text_delta`` events into the normal TTS path.

``POST /v1/turns/stream`` contract: ``docs/streaming-protocol-v1.md`` in
companion-runtime. Deltas are provisional; the terminal ``completed`` event
carries the canonical ``assistant_message`` which is written back to voice
history so the next turn's ``canonical_history`` stays complete.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
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
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)


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
        # One conversation per handler instance. With num_pipelines=1 this is
        # one long-lived voice session; per-connection ids come later.
        self.conversation_id = conversation_id or f"conv_voice_{uuid.uuid4().hex[:12]}"
        self.selected_model_id = selected_model_id
        self.request_timeout_s = float(timeout_s)
        self.request_timeout = httpx.Timeout(self.request_timeout_s, connect=10.0)
        self.user_id = user_id
        self.timezone = timezone
        headers = {"X-Companion-Runtime-Key": api_key} if api_key else {}
        self.client = httpx.Client(timeout=self.request_timeout, headers=headers)
        self.compactor = None
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
        logger.info(f"{self.__class__.__name__}: brain reachable, conversation={self.conversation_id}")

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
        if not latest_user.strip():
            raise ValueError("Cannot generate a response: no user message in the chat.")
        return {
            "contract_version": "v1",
            "turn_id": f"voice_{uuid.uuid4().hex[:12]}",
            "conversation_id": self.conversation_id,
            "companion_id": self.companion_id,
            "selected_model_id": self.selected_model_id,
            "current_sanitized_message": latest_user,
            "message_parts": [{"type": "text", "text": latest_user}],
            "canonical_history": history,
            "trusted_user_context": {"user_id": self.user_id, "timezone": self.timezone},
            "medium": "voice",
        }

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        # Tools execute inside the brain. Voice never forwards client tools.
        return {}

    # ── transport ─────────────────────────────────────────────────────────

    def _request(self, api_input: dict[str, Any], optional_kwargs: dict[str, Any]) -> Any:
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
                    return
                elif event_name == "error":
                    error = data.get("error", {})
                    raise RuntimeError(
                        f"companion-runtime turn failed "
                        f"({error.get('error_code', 'unknown')}): {error.get('message', '')}"
                    )
                event_name = None
        finally:
            try:
                api_response.close()
            except Exception:
                pass

    def _iter_response_events(self, payload: dict[str, Any]) -> Iterator[ProviderEvent]:
        if payload.get("status") in ("failed", "cancelled") or "error_code" in payload:
            raise RuntimeError(
                f"companion-runtime turn failed ({payload.get('error_code', 'unknown')}): {payload.get('message', '')}"
            )
        text = (payload.get("assistant_message") or "").strip()
        if text:
            yield TextDelta(text=text)
            yield AssistantMessage(content=[AssistantContent(type="output_text", text=text)])
