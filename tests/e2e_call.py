"""Deployed voice-call e2e (not collected by pytest).

Signs in to the deployed product as a guest, asks the product host for a call
token joined to an existing chat, synthesizes speech with the TTS provider,
streams it as microphone audio into the deployed Voice Runtime over WSS, and
reports: transcripts, streamed audio back, completion, and what the product host
now holds as chat history.

  HOST=https://project-z963i.vercel.app CHAT_ID=<uuid> LEMONFOX_API_KEY=... \
    PYTHONPATH=src:. python tests/e2e_call.py
"""
import asyncio, base64, json, os, sys, time, uuid
import httpx, numpy as np, websockets

HOST = os.environ["HOST"].rstrip("/")
CHAT_ID = os.environ.get("CHAT_ID") or str(uuid.uuid4())
KEY = os.environ["LEMONFOX_API_KEY"]
LINES = [x for x in os.environ.get("LINES", "Hey, can you hear me okay?|Mm, tell me something nice about today.").split("|")]


def guest(client: httpx.Client) -> dict:
    csrf = client.get("/api/auth/csrf").json()["csrfToken"]
    client.post("/api/auth/callback/guest", data={"csrfToken": csrf, "json": "true", "callbackUrl": HOST + "/"})
    return client.get("/api/auth/session").json()


def synth(text: str) -> bytes:
    r = httpx.post("https://api.lemonfox.ai/v1/audio/speech", headers={"Authorization": f"Bearer {KEY}"},
                   json={"input": text, "voice": "aoede", "response_format": "pcm"}, timeout=60)
    r.raise_for_status()
    pcm24 = np.frombuffer(r.content, dtype="<i2").astype(np.float32)
    n = int(len(pcm24) * 16000 / 24000)
    out = np.interp(np.linspace(0, len(pcm24) - 1, n), np.arange(len(pcm24)), pcm24)
    return out.astype("<i2").tobytes()


async def call(url: str, token: str) -> dict:
    stats = {"transcripts": [], "audio_bytes": 0, "responses_done": 0, "errors": [], "events": {}}
    async with websockets.connect(f"{url}?token={token}", max_size=None, open_timeout=30) as ws:
        async def reader():
            async for raw in ws:
                e = json.loads(raw); t = e.get("type", "")
                stats["events"][t] = stats["events"].get(t, 0) + 1
                if t == "conversation.item.input_audio_transcription.completed":
                    stats["transcripts"].append(e.get("transcript"))
                elif t == "response.output_audio.delta":
                    stats["audio_bytes"] += len(base64.b64decode(e.get("delta") or ""))
                elif t == "response.done":
                    stats["responses_done"] += 1
                elif t == "error":
                    stats["errors"].append(e.get("error"))
        task = asyncio.create_task(reader())
        await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
        for line in LINES:
            pcm = synth(line)
            chunk = 640  # 20ms at 16kHz PCM16
            for i in range(0, len(pcm), chunk * 2):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + chunk * 2]).decode()}))
                await asyncio.sleep(0.04)
            silence = b"\x00\x00" * 640
            for _ in range(60):  # 2.4s of silence so VAD closes the turn
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()}))
                await asyncio.sleep(0.04)
            target = stats["responses_done"] + 1
            deadline = time.time() + 90
            while stats["responses_done"] < target and time.time() < deadline:
                await asyncio.sleep(0.5)
        await asyncio.sleep(2)
        task.cancel()
    return stats


def main():
    client = httpx.Client(base_url=HOST, timeout=60)
    session = guest(client)
    print("guest:", session.get("user", {}).get("id"), flush=True)
    chat_id = CHAT_ID
    # Seed the chat with an ordinary text turn so the call joins existing history.
    seed = client.post("/api/chat", timeout=120, json={
        "id": chat_id, "message": {"id": str(uuid.uuid4()), "role": "user",
                                   "parts": [{"type": "text", "text": "morning. slept badly, ugh."}]},
        "selectedChatModel": "chat-model", "selectedVisibilityType": "private"})
    print("text seed turn:", seed.status_code, flush=True)
    r = client.post("/api/voice/session", json={"chatId": chat_id})
    print("voice session:", r.status_code, {k: (v[:40] + "…" if isinstance(v, str) and len(v) > 40 else v) for k, v in r.json().items()}, flush=True)
    info = r.json()
    stats = asyncio.run(call(info["url"], info["token"]))
    print(json.dumps(stats, indent=1), flush=True)
    time.sleep(3)
    ctx = httpx.get(f"{HOST}/api/voice/context", headers={"Authorization": f"Bearer {info['token']}"}, timeout=30).json()
    print("host chat history after call:", [(h["role"], h["content"][:50]) for h in ctx["history"][-6:]])
    print("host holds runtime state:", bool(ctx.get("session_routing")), list((ctx.get("session_routing") or {}).keys())[:6])


if __name__ == "__main__":
    main()
