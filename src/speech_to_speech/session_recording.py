"""Passive, channel-neutral recording of accepted realtime session events."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

SESSION_SCHEMA_VERSION = "synapse.session.v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionMessageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["realtime_transcript", "model_response", "text_input"]
    provider: str | None = None
    language: str | None = None
    average_logprob: float | None = None
    minimum_logprob: float | None = None
    uncertainty_reason: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    turn_id: str | None = None
    turn_revision: int | None = None
    response_id: str | None = None
    channel: str
    source: SessionMessageSource


class SessionToolUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any]
    output: str
    latency_ms: float
    timestamp: datetime
    response_id: str | None = None


class SessionCancellation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    reason: str
    response_id: str | None = None


class SessionRecovery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    kind: str
    detail: str | None = None


class SessionEnvelopeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[SessionToolUse] = Field(default_factory=list)
    recoveries: list[SessionRecovery] = Field(default_factory=list)
    cancellations: list[SessionCancellation] = Field(default_factory=list)
    providers: dict[str, str] = Field(default_factory=dict)
    latency_summary: dict[str, float] = Field(default_factory=dict)


class SessionEnvelope(BaseModel):
    """The stable, product-scoped session record emitted by Synapse."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["synapse.session.v1"] = Field(
        default=SESSION_SCHEMA_VERSION,
        alias="_v",
    )
    idempotency_key: str
    product_id: str
    agent_id: str
    user_id: str
    session_id: str
    conversation_id: str | None = None
    channel: str
    started_at: datetime
    ended_at: datetime
    timezone: str
    close_reason: str
    consent_scope: list[str] = Field(default_factory=list)
    messages: list[SessionMessage] = Field(default_factory=list)
    product_state: dict[str, Any] | None = None
    metadata: SessionEnvelopeMetadata = Field(default_factory=SessionEnvelopeMetadata)


class SessionRecorder:
    """Observe accepted events without participating in response execution."""

    def __init__(
        self,
        *,
        session_id: str,
        conversation_id: str | None,
        output_dir: Path | None,
        product_id: str,
        agent_id: str,
        user_id: str,
        channel: str,
        timezone_name: str,
        consent_scope: list[str] | None = None,
        providers: dict[str, str] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.output_dir = output_dir
        self.product_id = product_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.channel = channel
        self.timezone_name = timezone_name
        self.consent_scope = list(consent_scope or [])
        self.providers = dict(providers or {})
        self.started_at = started_at or _utc_now()

        self._messages: list[SessionMessage] = []
        self._user_message_index_by_turn: dict[str, int] = {}
        self._pending_assistant_text: dict[str, list[str]] = {}
        self._pending_assistant_turn: dict[str, tuple[str | None, int | None]] = {}
        self._tools: list[SessionToolUse] = []
        self._recoveries: list[SessionRecovery] = []
        self._cancellations: list[SessionCancellation] = []
        self._finalized = False

    @classmethod
    def from_env(
        cls,
        *,
        session_id: str,
        conversation_id: str | None,
    ) -> SessionRecorder | None:
        output_dir = os.environ.get("SYNAPSE_SESSION_RECORDING_DIR", "").strip()
        if not output_dir:
            return None
        consent_scope = [
            value.strip()
            for value in os.environ.get("SYNAPSE_CONSENT_SCOPE", "conversation_memory").split(",")
            if value.strip()
        ]
        return cls(
            session_id=session_id,
            conversation_id=conversation_id,
            output_dir=Path(output_dir).expanduser(),
            product_id=os.environ.get("SYNAPSE_PRODUCT_ID", "sophie"),
            agent_id=os.environ.get("SYNAPSE_AGENT_ID", "sophie"),
            user_id=os.environ.get("SYNAPSE_USER_ID", "dev-user"),
            channel=os.environ.get("SYNAPSE_CHANNEL", "voice"),
            timezone_name=os.environ.get(
                "SYNAPSE_USER_TIMEZONE",
                os.environ.get("SOPHIE_USER_TIMEZONE", "Europe/London"),
            ),
            consent_scope=consent_scope,
            providers={
                key: value
                for key, value in {
                    "stt": os.environ.get("VOICE_STT_BACKEND", os.environ.get("SOPHIE_STT_BACKEND")),
                    "llm": os.environ.get("SOPHIE_LLM_MODEL"),
                    "tts": os.environ.get("VOICE_TTS_BACKEND", os.environ.get("SOPHIE_TTS_BACKEND")),
                }.items()
                if value
            },
        )

    def record_user_transcript(
        self,
        *,
        transcript: str,
        turn_id: str | None,
        turn_revision: int | None,
        language: str | None,
        average_logprob: float | None,
        minimum_logprob: float | None,
        uncertainty_reason: str | None,
        provider_metadata: dict[str, object],
    ) -> None:
        transcript = transcript.strip()
        if self._finalized:
            return
        if not transcript:
            if turn_id is not None and turn_id in self._user_message_index_by_turn:
                self._messages.pop(self._user_message_index_by_turn[turn_id])
                self._rebuild_user_message_index()
            return
        message_id = turn_id or f"user-{len(self._messages) + 1}"
        message = SessionMessage(
            id=message_id,
            role="user",
            content=transcript,
            timestamp=_utc_now(),
            turn_id=turn_id,
            turn_revision=turn_revision,
            channel=self.channel,
            source=SessionMessageSource(
                type="realtime_transcript",
                language=language,
                average_logprob=average_logprob,
                minimum_logprob=minimum_logprob,
                uncertainty_reason=uncertainty_reason,
                provider_metadata=dict(provider_metadata),
            ),
        )
        if turn_id is not None and turn_id in self._user_message_index_by_turn:
            index = self._user_message_index_by_turn[turn_id]
            message.timestamp = self._messages[index].timestamp
            self._messages[index] = message
            return
        if turn_id is not None:
            self._user_message_index_by_turn[turn_id] = len(self._messages)
        self._messages.append(message)

    def _rebuild_user_message_index(self) -> None:
        self._user_message_index_by_turn = {
            message.turn_id: index
            for index, message in enumerate(self._messages)
            if message.role == "user" and message.turn_id is not None
        }

    def stage_assistant_text(
        self,
        *,
        response_id: str,
        text: str,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> None:
        if self._finalized or not text:
            return
        self._pending_assistant_text.setdefault(response_id, []).append(text)
        self._pending_assistant_turn[response_id] = (turn_id, turn_revision)

    def finish_response(self, *, response_id: str | None, status: str, reason: str | None) -> None:
        if self._finalized or response_id is None:
            return
        parts = self._pending_assistant_text.pop(response_id, [])
        turn_id, turn_revision = self._pending_assistant_turn.pop(response_id, (None, None))
        if status == "completed":
            content = "".join(parts).strip()
            if content:
                self._messages.append(
                    SessionMessage(
                        id=response_id,
                        role="assistant",
                        content=content,
                        timestamp=_utc_now(),
                        turn_id=turn_id,
                        turn_revision=turn_revision,
                        response_id=response_id,
                        channel=self.channel,
                        source=SessionMessageSource(type="model_response"),
                    )
                )
            return
        self._cancellations.append(
            SessionCancellation(
                timestamp=_utc_now(),
                reason=reason or status,
                response_id=response_id,
            )
        )

    def record_tool(
        self,
        *,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        output: str,
        latency_ms: float,
        response_id: str | None,
    ) -> None:
        if self._finalized:
            return
        self._tools.append(
            SessionToolUse(
                call_id=call_id,
                name=name,
                arguments=arguments,
                output=output,
                latency_ms=latency_ms,
                timestamp=_utc_now(),
                response_id=response_id,
            )
        )

    def record_recovery(self, *, kind: str, detail: str | None = None) -> None:
        if not self._finalized:
            self._recoveries.append(SessionRecovery(timestamp=_utc_now(), kind=kind, detail=detail))

    def build_envelope(self, *, close_reason: str, ended_at: datetime | None = None) -> SessionEnvelope:
        return SessionEnvelope(
            idempotency_key=f"session:{self.product_id}:{self.user_id}:{self.session_id}:final",
            product_id=self.product_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            channel=self.channel,
            started_at=self.started_at,
            ended_at=ended_at or _utc_now(),
            timezone=self.timezone_name,
            close_reason=close_reason,
            consent_scope=self.consent_scope,
            messages=list(self._messages),
            metadata=SessionEnvelopeMetadata(
                tools=list(self._tools),
                recoveries=list(self._recoveries),
                cancellations=list(self._cancellations),
                providers=self.providers,
            ),
        )

    def finalize(self, *, close_reason: str = "disconnect") -> SessionEnvelope | None:
        if self._finalized:
            return None
        for response_id in list(self._pending_assistant_text):
            self.finish_response(response_id=response_id, status="cancelled", reason="session_close")
        envelope = self.build_envelope(close_reason=close_reason)
        self._finalized = True
        if self.output_dir is not None:
            self._persist(envelope)
        return envelope

    def _persist(self, envelope: SessionEnvelope) -> None:
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_session_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in self.session_id)
        destination = self.output_dir / f"{safe_session_id}.json"
        payload = envelope.model_dump_json(by_alias=True, indent=2)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{safe_session_id}.", suffix=".tmp", dir=self.output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        logger.info(
            "synapse.session.persisted session=%s messages=%d tools=%d path=%s",
            self.session_id,
            len(envelope.messages),
            len(envelope.metadata.tools),
            destination,
        )
