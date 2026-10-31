import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from speech_to_speech import call_identity, experiences_api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("VOICE_SESSION_SECRET", "s" * 40)
    monkeypatch.setenv("VOICE_RUNTIME_URL", "http://runtime.test")
    monkeypatch.setenv("VOICE_RUNTIME_SECRET", "k")
    app = FastAPI()
    experiences_api.register(app)
    return TestClient(app)


def _token(exp):
    import base64, hashlib, hmac, json
    payload = base64.urlsafe_b64encode(json.dumps({"uid": "u1", "cid": "c1", "tz": "Europe/London", "companion": "bloom", "exp": exp}).encode()).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(hmac.new(b"s" * 40, payload.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    return payload + "." + sig


def _bind(token):
    call_identity.REGISTRY.bind(call_identity.CallIdentity(user_id="u1", conversation_id="c1", timezone="Europe/London", companion_id="bloom", token=token, started_at_iso="x"))


class _Runtime:
    def __init__(self, monkeypatch, replies):
        self.seen = []

        class C:
            def __init__(s, *a, **k): pass
            async def __aenter__(s): return s
            async def __aexit__(s, *a): return False
            async def get(s_, url, params=None, headers=None):
                self.seen.append(("GET", url, params, headers)); return replies.pop(0)
            async def post(s_, url, params=None, headers=None, json=None):
                self.seen.append(("POST", url, params, headers, json)); return replies.pop(0)
        monkeypatch.setattr(experiences_api.httpx, "AsyncClient", C)


def test_a_live_call_fetches_the_open_card_under_its_own_user_and_product(client, monkeypatch):
    tok = _token(9_999_999_999); _bind(tok)
    rt = _Runtime(monkeypatch, [httpx.Response(200, json={"offer": {"offer_id": "o1", "title": "4-7-8 breathing"}})])
    r = client.get("/v1/experiences/open", params={"token": tok})
    assert r.status_code == 200 and r.json()["offer"]["offer_id"] == "o1"
    assert rt.seen[0][1] == "http://runtime.test/v1/users/u1/experiences/open" and rt.seen[0][2] == {"companion_id": "bloom"} and rt.seen[0][3] == {"X-Companion-Runtime-Key": "k"}
    call_identity.REGISTRY.release("c1")


def test_an_expired_connect_token_still_works_during_the_call_but_not_after_it(client, monkeypatch):
    tok = _token(1); _bind(tok)                       # long expired: the call outlived its five minutes
    _Runtime(monkeypatch, [httpx.Response(200, json={"offer_id": "o1", "status": "completed"})])
    ok = client.post("/v1/experiences/o1/event", params={"token": tok}, json={"status": "completed", "duration_s": 120})
    assert ok.status_code == 200
    call_identity.REGISTRY.release("c1")
    assert client.post("/v1/experiences/o1/event", params={"token": tok}, json={"status": "completed"}).status_code == 401


def test_a_forged_token_or_a_bad_status_is_refused_before_the_runtime_is_called(client, monkeypatch):
    rt = _Runtime(monkeypatch, [])
    assert client.get("/v1/experiences/open", params={"token": "x.y"}).status_code == 401
    tok = _token(9_999_999_999); _bind(tok)
    assert client.post("/v1/experiences/o1/event", params={"token": tok}, json={"status": "bogus"}).status_code == 422
    assert rt.seen == []
    call_identity.REGISTRY.release("c1")
