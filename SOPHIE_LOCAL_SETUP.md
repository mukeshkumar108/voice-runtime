# Sophie Local Setup

This repo is now the Hugging Face `speech-to-speech` runtime used as `sophie-voice`.

## Repos

- `sophie-core`: `/Users/mukeshkumar/play/sophie-core`
- `sophie-voice`: `/Users/mukeshkumar/play/sophie-voice`
- `sophie-mob`: `/Users/mukeshkumar/play/sophie-mob`

## What Changed

- `sophie-mob` now points at the HF/OpenAI-Realtime endpoint: `/v1/realtime`
- the mobile voice client now speaks HF realtime events directly
- assistant playback now expects HF's 24 kHz PCM output

## 1. Start sophie-core

Run whatever command you normally use to start `sophie-core` on port `3000`.

Expected URL:

```bash
http://192.168.0.18:3000
```

## 2. Start sophie-voice

From `/Users/mukeshkumar/play/sophie-voice`:

```bash
uv sync
./scripts/run_sophie_realtime.sh
```

The helper script:

- binds to `0.0.0.0:3002`
- uses ElevenLabs Scribe realtime STT by default
- uses Lemonfox TTS with the `aoede` voice by default
- delegates every turn to companion-runtime (`VOICE_RUNTIME_URL`, default `http://127.0.0.1:8080`), which owns prompt, memory, tools, and model selection
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
./scripts/run_sophie_realtime.sh
```

Lemonfox experiment:

```bash
export VOICE_STT_BACKEND=lemonfox
export VOICE_TTS_BACKEND=lemonfox
export VOICE_LEMONFOX_VOICE=aoede
./scripts/run_sophie_realtime.sh
```

OpenRouter Parakeet + Lemonfox TTS:

```bash
export VOICE_STT_BACKEND=openrouter-parakeet
export VOICE_TTS_BACKEND=lemonfox
export VOICE_LEMONFOX_VOICE=aoede
./scripts/run_sophie_realtime.sh
```

ElevenLabs realtime STT + Lemonfox TTS:

```bash
export VOICE_STT_BACKEND=elevenlabs-realtime
export VOICE_TTS_BACKEND=lemonfox
export VOICE_LEMONFOX_VOICE=aoede
./scripts/run_sophie_realtime.sh
```

Current caveat:

- Lemonfox TTS is streamed into the HF runtime.
- Lemonfox STT is currently wired as final-turn upload STT, not incremental partial-result STT, so this mode is useful for latency comparison but does not yet match the partial-transcript behavior of local Parakeet.

Expected websocket endpoint:

```bash
ws://192.168.0.18:3002/v1/realtime
```

## 3. Start sophie-mob

The local env should include:

```bash
EXPO_PUBLIC_API_URL=http://192.168.0.18:3000
EXPO_PUBLIC_VOICE_URL=ws://192.168.0.18:3002/v1/realtime
```

Then from `/Users/mukeshkumar/play/sophie-mob`:

```bash
pnpm install
pnpm start:mobile
```

Important:

- open the app in your installed Expo development build, not Expo Go
- if the installed dev build is stale, rebuild and reinstall it before testing

## First Test Flow

1. Open `Voice Lab`
2. Tap `Connect`
3. Wait for status `ready`
4. Tap `Start Microphone`
5. Speak one short utterance
6. Confirm:
   - user transcript streams
   - assistant transcript streams
   - assistant audio plays
   - status returns to `ready` after the reply

## Current Scope

This is only the voice runtime slice:

- no Cortex wiring yet
- no Sophie-specific tool calling yet
- no sophie-core auth handshake into voice yet
- no transcript persistence yet

The runtime path being proven is:

```text
sophie-mob mic
-> HF VAD
-> HF STT
-> HF/OpenAI-Realtime protocol
-> companion-runtime brain (prompt, memory, tools, model)
-> HF TTS
-> sophie-mob playback
```
