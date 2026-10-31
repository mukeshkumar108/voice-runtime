"""Native experiences (breathing, a lesson, a game) between the phone and the Companion Runtime.

The Runtime decides what to offer and remembers what happened; the product renders it. These two routes let the phone fetch the card the companion put on screen and report what the
person did (accepted, declined, completed, exited). They are authorised by the call's own token and only while that call is live, so an expired five-minute token does not end a
ten-minute exercise, and a token cannot be used after the call is over. Voice interprets nothing: it relays facts to the Runtime under the call's user and companion."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from speech_to_speech import call_identity

logger = logging.getLogger(__name__)
TIMEOUT_S = 6.0


class ExperienceEvent(BaseModel):
    status: str = Field(pattern="^(accepted|declined|completed|exited)$")
    duration_s: Optional[int] = Field(default=None, ge=0, le=7200)


def _runtime() -> tuple[str, dict[str, str]]:
    base = os.getenv("VOICE_RUNTIME_URL", "").strip().rstrip("/")
    key = (os.getenv("VOICE_RUNTIME_SECRET") or os.getenv("COMPANION_RUNTIME_SECRET") or "").strip()
    if not base:
        raise HTTPException(status_code=503, detail="runtime_unconfigured")
    return base, ({"X-Companion-Runtime-Key": key} if key else {})


def _live_identity(token: Optional[str]) -> call_identity.CallIdentity:
    identity = call_identity.REGISTRY.find_by_token(token)
    if identity is None or call_identity.verify_token(token, ignore_expiry=True) is None:
        raise HTTPException(status_code=401, detail="no_live_call")
    return identity


def register(app: FastAPI) -> None:
    @app.get("/v1/experiences/open")
    async def open_offer(token: str) -> dict[str, Any]:
        who = _live_identity(token)
        base, headers = _runtime()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
                r = await c.get(f"{base}/v1/users/{who.user_id}/experiences/open", params={"companion_id": who.companion_id}, headers=headers)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="runtime_unreachable")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="runtime_error")
        return r.json()

    @app.post("/v1/experiences/{offer_id}/event")
    async def report(offer_id: str, body: ExperienceEvent, token: str) -> dict[str, Any]:
        who = _live_identity(token)
        base, headers = _runtime()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
                r = await c.post(f"{base}/v1/users/{who.user_id}/experiences/{offer_id}/event", params={"companion_id": who.companion_id}, headers=headers, json=body.model_dump(exclude_none=True))
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="runtime_unreachable")
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="not_found")
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="runtime_error")
        return r.json()
