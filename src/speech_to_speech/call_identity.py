"""Call identity and conversation bridge for the Companion Runtime backend.

A call is a *modality* of a conversation that already exists in the product.
The product host issues a short-lived signed token binding (user, chat,
timezone, companion). This module verifies it, loads the chat's history and
previous Runtime session state from the host, carries that Runtime state across
the call's turns verbatim, and hands each completed spoken turn back to the
host so voice and text are one conversation.

Voice interprets nothing here: it forwards facts and opaque state.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

SECRET_ENV = "VOICE_SESSION_SECRET"
HOST_URL_ENV = "VOICE_HOST_URL"
# Comma separated origins a token may name in its `host` claim (a second product host).
ALLOWED_HOSTS_ENV = "VOICE_ALLOWED_HOSTS"

MAX_CONSTITUTION_CHARS = 40_000


@dataclass
class CallIdentity:
    user_id: str
    conversation_id: str
    timezone: str
    companion_id: str
    token: str
    started_at_iso: str
    started_epoch: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)
    session_routing: dict[str, Any] = field(default_factory=dict)
    turns_sent: int = 0
    # Which session this is (morning, evening, ...): the product names it in the token; the end-of-call report uses it.
    flow: str = ""
    # Product host this call talks to; empty means the default VOICE_HOST_URL.
    host: str = ""
    # Opaque product context from the host, passed to the brain as TurnInput.application_context.
    application_context: Optional[dict[str, Any]] = None


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def secret_configured() -> bool:
    return bool(os.getenv(SECRET_ENV, "").strip())


def allowed_hosts() -> frozenset[str]:
    raw = os.getenv(ALLOWED_HOSTS_ENV, "")
    return frozenset(h.strip().rstrip("/") for h in raw.split(",") if h.strip())


def verify_token(token: Optional[str], now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Verify an HMAC-SHA256 token issued by the product host."""
    secret = os.getenv(SECRET_ENV, "").strip()
    if not secret or not token or "." not in token:
        return None
    payload, _, signature = token.partition(".")
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    try:
        given = _b64url_decode(signature)
    except Exception:
        return None
    if not hmac.compare_digest(given, expected):
        return None
    try:
        claims = json.loads(_b64url_decode(payload))
    except Exception:
        return None
    if not all(isinstance(claims.get(k), str) for k in ("uid", "cid", "tz")):
        return None
    if float(claims.get("exp", 0)) < (now if now is not None else time.time()):
        return None
    # A token naming another product host is only honoured for allow-listed origins; otherwise the
    # call would send its transcript to a host the operator never approved.
    host = claims.get("host")
    if host is not None and (not isinstance(host, str) or host.rstrip("/") not in allowed_hosts()):
        return None
    return claims


class IdentityRegistry:
    """Per-connection call identities keyed by the connection's conversation id."""

    def __init__(self) -> None:
        self._items: dict[str, CallIdentity] = {}
        self._lock = threading.Lock()

    def bind(self, identity: CallIdentity) -> None:
        with self._lock:
            self._items[identity.conversation_id] = identity

    def get(self, conversation_id: Optional[str]) -> Optional[CallIdentity]:
        if not conversation_id:
            return None
        with self._lock:
            return self._items.get(conversation_id)

    def release(self, conversation_id: Optional[str]) -> None:
        if not conversation_id:
            return
        with self._lock:
            self._items.pop(conversation_id, None)


REGISTRY = IdentityRegistry()


def _host_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def host_base_url() -> str:
    return os.getenv(HOST_URL_ENV, "").strip().rstrip("/")


def _valid_application_context(value: Any) -> Optional[dict[str, Any]]:
    """Pass through only the shape the Runtime accepts (product-context-v1 constitution)."""
    if not isinstance(value, dict):
        return None
    constitution = value.get("character_constitution")
    if not isinstance(constitution, str) or not constitution.strip() or len(constitution) > MAX_CONSTITUTION_CHARS:
        return None
    return {"context_type": "product-context-v1", "character_constitution": constitution}


async def load_call_context(claims: dict[str, Any], token: str) -> CallIdentity:
    """Build the call identity; history/state come from the product host."""
    identity = CallIdentity(
        user_id=claims["uid"],
        conversation_id=claims["cid"],
        timezone=claims["tz"],
        companion_id=str(claims.get("companion") or "sophie"),
        token=token,
        started_at_iso=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    )
    identity.host = str(claims.get("host") or "").rstrip("/")
    identity.flow = str(claims.get("flow") or "")
    base = identity.host or host_base_url()
    if not base:
        logger.warning("%s is not set; the call starts with an empty history", HOST_URL_ENV)
        return identity
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base}/api/voice/context", headers=_host_headers(token))
            response.raise_for_status()
            body = response.json()
        identity.history = [
            {"id": str(h["id"]), "role": h["role"], "content": h["content"],
             **({"created_at": h["created_at"]} if h.get("created_at") else {})}
            for h in (body.get("history") or []) if h.get("content")
        ]
        routing = body.get("session_routing")
        identity.session_routing = routing if isinstance(routing, dict) else {}
        identity.application_context = _valid_application_context(body.get("application_context"))
    except Exception as exc:  # the call can still run; Cortex rehydrates on turn one
        logger.warning("Could not load call context from host: %s", exc)
    return identity


def report_completed_turn(
    identity: CallIdentity, *, turn_id: str, user_text: str, assistant_text: str,
    next_session_state: Optional[dict[str, Any]], reliability_status: str,
) -> None:
    """Fire-and-forget (daemon thread): a spoken turn becomes chat history."""
    base = identity.host or host_base_url()
    if not base or not user_text.strip() or not assistant_text.strip():
        return

    def _send() -> None:
        try:
            response = httpx.post(
                f"{base}/api/voice/turn",
                headers=_host_headers(identity.token),
                json={
                    "turn_id": turn_id, "user_text": user_text, "assistant_text": assistant_text,
                    "next_session_state": next_session_state, "reliability_status": reliability_status,
                },
                timeout=15.0,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Could not persist spoken turn %s: %s", turn_id, exc)

    threading.Thread(target=_send, name=f"turn-report-{turn_id}", daemon=True).start()
