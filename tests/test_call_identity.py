"""Call identity: a call joins the product's chat as a modality (no new cognition)."""

import base64
import hashlib
import hmac
import json
import time

import httpx
import pytest

from speech_to_speech import call_identity
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from tests.test_companion_runtime_backend import _handler, _sse_response

SECRET = "voice-test-secret"


def _token(secret=SECRET, **overrides):
    claims = {"uid": "user-1", "cid": "11111111-1111-4111-8111-111111111111",
              "tz": "Europe/London", "companion": "sophie", "exp": int(time.time()) + 600}
    claims.update(overrides)
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    mac = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return f"{payload}.{base64.urlsafe_b64encode(mac).rstrip(b'=').decode()}"


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv(call_identity.SECRET_ENV, SECRET)
    monkeypatch.delenv(call_identity.HOST_URL_ENV, raising=False)
    yield
    call_identity.REGISTRY._items.clear()


def test_token_round_trip_and_rejections():
    claims = call_identity.verify_token(_token())
    assert claims["uid"] == "user-1" and claims["cid"].startswith("1111")
    assert call_identity.verify_token(_token(exp=int(time.time()) - 5)) is None
    assert call_identity.verify_token(_token(secret="other")) is None
    assert call_identity.verify_token("garbage") is None
    assert call_identity.verify_token(None) is None


def _identity(**kwargs):
    identity = call_identity.CallIdentity(
        user_id="user-1", conversation_id="chat-1", timezone="Europe/London", companion_id="sophie",
        token="t", started_at_iso="2026-10-01T10:00:00+00:00",
        history=[{"id": "m1", "role": "user", "content": "earlier text"},
                 {"id": "m2", "role": "assistant", "content": "earlier reply"}], **kwargs)
    call_identity.REGISTRY.bind(identity)
    return identity


def test_first_spoken_turn_is_a_new_sitting_with_chat_history_and_user_identity():
    _identity()
    handler = _handler(_req_conversation_id="chat-1")
    chat = Chat(10)
    chat.add_item(make_user_message("hello from the call"))
    payload = handler._serialize(chat)
    trusted = payload["trusted_user_context"]
    assert trusted["user_id"] == "user-1" and trusted["medium"] == "voice"
    assert trusted["entry_context"]["chronology"]["temporalSession"] == "new"
    assert trusted["session_routing"] == {}
    assert [h["content"] for h in payload["canonical_history"]] == ["earlier text", "earlier reply"]
    assert payload["conversation_id"] == "chat-1"
    assert payload["medium"] == "voice"


def test_runtime_state_is_carried_and_the_spoken_turn_is_handed_back(monkeypatch):
    identity = _identity()
    sent = []
    monkeypatch.setattr(call_identity, "report_completed_turn", lambda ident, **kw: sent.append(kw))
    handler = _handler(_req_conversation_id="chat-1")
    chat = Chat(10)
    chat.add_item(make_user_message("what's up"))
    handler._serialize(chat)
    state = {"residentWorld": {"version": "resident-world-v1"}, "lastJev": {"capability": "reply"}}
    raw = ("event: text_delta\ndata: {\"delta\": \"Not much\"}\n\n"
           "event: completed\ndata: " + json.dumps({"result": {
               "assistant_message": "Not much", "execution_metadata": {"next_session_state": state}}}) + "\n\n").encode()
    list(handler._iter_stream_events(_sse_response(raw)))
    assert identity.session_routing == state
    assert identity.turns_sent == 1
    assert sent and sent[0]["user_text"] == "what's up" and sent[0]["assistant_text"] == "Not much"
    assert sent[0]["next_session_state"] == state

    # Second turn: same sitting, state sent back verbatim (no Cortex rehydration).
    chat.add_item(make_assistant_message("Not much"))
    chat.add_item(make_user_message("cool"))
    payload = handler._serialize(chat)
    trusted = payload["trusted_user_context"]
    assert trusted["session_routing"] == state
    assert trusted["entry_context"]["chronology"]["temporalSession"] == "same"
    assert payload["canonical_history"][-2:] == [
        {"id": "hist_0", "role": "user", "content": "what's up"},
        {"id": "hist_1", "role": "assistant", "content": "Not much"}]


def test_without_identity_the_handler_keeps_local_defaults():
    handler = _handler(_req_conversation_id="unbound")
    chat = Chat(10)
    chat.add_item(make_user_message("hi"))
    trusted = handler._serialize(chat)["trusted_user_context"]
    assert trusted == {"user_id": "local-user", "timezone": "Europe/London"}


def test_unauthenticated_connection_is_refused_when_a_secret_is_configured():
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from threading import Event

    from speech_to_speech.api.openai_realtime.websocket_router import create_app

    client = TestClient(create_app(pool=[], stop_event=Event()))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/realtime"):
            pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/realtime?token=" + _token(secret="wrong")):
            pass


@pytest.mark.asyncio
async def test_context_is_loaded_from_the_product_host(monkeypatch):
    monkeypatch.setenv(call_identity.HOST_URL_ENV, "https://host.test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/voice/context"
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, json={
            "history": [{"id": "a", "role": "user", "content": "hi", "created_at": "2026-10-01T09:00:00Z"}],
            "session_routing": {"residentWorld": {"x": 1}}})

    real = httpx.AsyncClient
    monkeypatch.setattr(call_identity.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    identity = await call_identity.load_call_context(call_identity.verify_token(_token()), "tok")
    assert identity.history[0]["created_at"] == "2026-10-01T09:00:00Z"
    assert identity.session_routing == {"residentWorld": {"x": 1}}


def test_binding_a_connection_to_a_chat_uses_the_services_own_state():
    from speech_to_speech.api.openai_realtime.service import RealtimeService

    service = RealtimeService(runtime_tools=False)
    connection = service.register()
    service.bind_conversation(connection, "11111111-1111-4111-8111-111111111111")
    assert service._state(connection).conversation_id == "11111111-1111-4111-8111-111111111111"
