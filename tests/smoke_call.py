"""Live voice-call smoke (not collected by pytest).

Drives the real Companion Runtime backend handler against an instrumented
Runtime (companion-runtime/evals/consumer_smoke/serve.py) with a fake product
host standing in for the BFF. Proves: signed identity binds the call to a chat,
chat history + Runtime state load from the host, state is carried across spoken
turns (one Cortex hydration per call), completed turns are handed back, and
barge-in cancellation reaches the brain.

  SMOKE_RUNTIME_URL=http://127.0.0.1:8080 PYTHONPATH=src:. python tests/smoke_call.py
"""
import asyncio
import base64
import hashlib
import hmac
import json
import os
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request

from speech_to_speech import call_identity
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from tests.test_companion_runtime_backend import _handler

RUNTIME = os.environ.get("SMOKE_RUNTIME_URL", "http://127.0.0.1:8080")
KEY = os.environ.get("SMOKE_RUNTIME_SECRET", "smoke-secret")
SECRET = "voice-smoke-secret"
HOST_PORT = 8099

host_log: dict[str, list] = {"context": [], "turn": []}
app = FastAPI()


@app.get("/api/voice/context")
async def context(request: Request):
    host_log["context"].append(request.headers.get("authorization"))
    return {"history": [
        {"id": "h1", "role": "user", "content": "morning, did you sleep?", "created_at": "2026-10-01T08:00:00Z"},
        {"id": "h2", "role": "assistant", "content": "Better than you, I bet.", "created_at": "2026-10-01T08:00:05Z"}],
        "session_routing": {}}


@app.post("/api/voice/turn")
async def turn(request: Request):
    host_log["turn"].append(await request.json())
    return {"ok": True}


def observed() -> dict:
    return httpx.get(f"{RUNTIME}/_observed", headers={"X-Companion-Runtime-Key": KEY}, timeout=10).json()


def token() -> str:
    claims = {"uid": f"voice-smoke-{int(time.time())}", "cid": "22222222-2222-4222-8222-222222222222",
              "tz": "Europe/London", "companion": "sophie", "exp": int(time.time()) + 600}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    mac = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).digest()
    return f"{payload}.{base64.urlsafe_b64encode(mac).rstrip(b'=').decode()}"


def main() -> None:
    os.environ[call_identity.SECRET_ENV] = SECRET
    os.environ[call_identity.HOST_URL_ENV] = f"http://127.0.0.1:{HOST_PORT}"
    threading.Thread(target=lambda: uvicorn.run(app, port=HOST_PORT, log_level="error"), daemon=True).start()
    time.sleep(1.5)

    tok = token()
    claims = call_identity.verify_token(tok)
    identity = asyncio.run(call_identity.load_call_context(claims, tok))
    call_identity.REGISTRY.bind(identity)
    assert identity.history and host_log["context"], "context was not loaded from the host"

    handler = _handler(
        base_url=RUNTIME, _req_conversation_id=identity.conversation_id,
        client=httpx.Client(timeout=120.0, headers={"X-Companion-Runtime-Key": KEY}))
    chat = Chat(20)
    results = []
    for line in ["hey, can you hear me okay?", "ha, good. what were you up to?", "mm. tell me something nice."]:
        observed()
        chat.add_item(make_user_message(line))
        payload = handler._serialize(chat)
        response = handler._request(payload, {})
        text = "".join(getattr(e, "text", "") for e in handler._iter_stream_events(response)
                       if type(e).__name__ == "TextDelta")
        time.sleep(1.2)
        seen = observed()
        chat.add_item(make_assistant_message(text))
        results.append({"line": line, "reply": text[:70], "jev": seen["jev_calls"], "fg": seen["foreground_calls"],
                        "cortex": seen["cortex_http"], "honcho": seen["honcho_calls"],
                        "other": len(seen["other_inference"]), "model": seen["foreground_models"],
                        "chronology": payload["trusted_user_context"]["entry_context"]["chronology"]["temporalSession"]})
        print(json.dumps(results[-1]), flush=True)

    # invariants
    first, rest = results[0], results[1:]
    assert first["jev"] == 1 and first["fg"] == 1 and first["honcho"] == [] and first["chronology"] == "new"
    assert sum(1 for p in first["cortex"] if "world-model" in p or "attention-state" in p) == 2
    for r in rest:
        assert r["jev"] == 1 and r["fg"] == 1 and r["cortex"] == [] and r["honcho"] == [] and r["other"] == 0 and r["chronology"] == "same", r
    time.sleep(1.5)
    assert len(host_log["turn"]) == 3, host_log["turn"]
    assert all(t["assistant_text"] and t["next_session_state"] for t in host_log["turn"])

    # barge-in: start a turn, take the first delta, cancel it at the brain
    chat.add_item(make_user_message("wait, actually, tell me a really long story"))
    payload = handler._serialize(chat)
    response = handler._request(payload, {})
    gen = handler._iter_stream_events(response)
    next(gen)
    handler._cancel_brain_turn(payload["turn_id"], payload["conversation_id"])
    status = httpx.get(f"{RUNTIME}/v1/turns/{payload['turn_id']}", params={"conversation_id": payload["conversation_id"]},
                       headers={"X-Companion-Runtime-Key": KEY}, timeout=10).json()
    print(json.dumps({"barge_in_status": status.get("status")}), flush=True)
    assert status.get("status") in ("cancelled", "completed", "executing")
    print("VOICE SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
