# Voice Runtime Local Setup

Realtime speech I/O for Companion Runtime. See `docs/VOICE_RUNTIME_CONTRACT.md`
for the modality boundary.

## 1. Start the brain

Companion Runtime must be reachable at `VOICE_RUNTIME_URL`
(default `http://127.0.0.1:8080`) with a matching `VOICE_RUNTIME_SECRET`.

## 2. Start voice

From this repo:

```bash
uv sync
./scripts/run_voice_realtime.sh
```

The helper script:

- binds to `0.0.0.0:3002`
- uses ElevenLabs Scribe realtime STT by default
- uses Lemonfox TTS with the `aoede` voice by default
- delegates every turn to Companion Runtime, which owns prompt, memory, tools, and model selection
- supports local Parakeet STT and Qwen3 TTS as optional alternatives

Env names are `VOICE_*`. Legacy `SOPHIE_*` / `COMPANION_*` aliases still work with a deprecation warning.

Useful overrides:

```bash
export VOICE_PORT=3002
export VOICE_STT_BACKEND=elevenlabs-realtime
export VOICE_TTS_BACKEND=lemonfox
export VOICE_LEMONFOX_VOICE=aoede
export VOICE_MIN_SILENCE_MS=900
export VOICE_MIN_SPEECH_MS=384
export VOICE_MIN_SPEECH_CONTINUATION_MS=384
export VOICE_SPECULATIVE_REOPEN_MS=3000
export VOICE_UNANSWERED_REOPEN_MS=12000
export VOICE_RUNTIME_URL="http://127.0.0.1:8080"
export VOICE_RUNTIME_SECRET="..."
export LEMONFOX_API_KEY="..."
export ELEVENLABS_API_KEY="..."
./scripts/run_voice_realtime.sh
```

STT/TTS experiments (same script, different backends):

```bash
# Lemonfox STT (final-turn upload; no partials) + Lemonfox TTS
export VOICE_STT_BACKEND=lemonfox
export VOICE_TTS_BACKEND=lemonfox
./scripts/run_voice_realtime.sh
```

```bash
# OpenRouter Parakeet STT + Lemonfox TTS
export VOICE_STT_BACKEND=openrouter-parakeet
export VOICE_TTS_BACKEND=lemonfox
./scripts/run_voice_realtime.sh
```

Current caveat:

- Lemonfox TTS is streamed into the runtime.
- Lemonfox STT is currently wired as final-turn upload STT, not incremental partial-result STT, so this mode is useful for latency comparison but does not yet match the partial-transcript behavior of local Parakeet.

Expected websocket endpoint (replace `<host>` with your machine's address):

```bash
ws://<host>:3002/v1/realtime
```

Health:

```bash
curl http://<host>:3002/healthz
curl http://<host>:3002/readyz
```

## 3. Connect a client

Any OpenAI Realtime-compatible client works:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<host>:3002/v1",
    websocket_base_url="ws://<host>:3002/v1",
    api_key="not-needed",
)
with client.realtime.connect(model="local") as conn:
    ...
```

Assistant playback is 24 kHz PCM.

## First Test Flow

1. Connect a client to `/v1/realtime`
2. Send one short utterance
3. Confirm:
   - user transcript streams
   - assistant transcript streams
   - assistant audio plays
   - session returns to idle after the reply

## Current Scope

Voice owns transport only:

- no model routing, prompts, memory, tools, or response policy (all brain-side)
- no auth handshake yet (single-user local operation)
- session envelopes are recorded locally (see `contracts/`); delivery to memory systems is out of scope for this repo

The runtime path is:

```text
client mic
-> VAD
-> STT
-> OpenAI-Realtime protocol
-> companion-runtime brain (prompt, memory, tools, model)
-> TTS
-> client playback
```
