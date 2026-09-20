# Sophie Voice Integration Notes

This repo is now the new `sophie-voice` base.

- Previous custom TypeScript implementation has been preserved at:
  - `/Users/mukeshkumar/play/sophie-voice-old`
- This repo is a clone of:
  - `https://github.com/huggingface/speech-to-speech`
- Cloned commit:
  - `c766ba1`

## Intended Role

Treat this repo as the realtime voice runtime for Sophie:

- VAD
- STT
- interruption / barge-in
- response lifecycle
- streamed text / audio
- tool-call transport
- OpenAI Realtime-compatible protocol

Do **not** treat this repo as the place to reinvent Sophie cognition.

## Sophie-Specific Layers To Add

These remain Sophie-owned:

- auth/session identity
- Cortex / Synapse memory fetch
- context compiler
- retrieval / tool policy
- persona / instructions
- transcript persistence / analytics
- model-routing policy

## Current Neighbour Repos

- `sophie-mob`: `/Users/mukeshkumar/play/sophie-mob`
- `sophie-core`: `/Users/mukeshkumar/play/sophie-core`
- `synapse-v3`: `/Users/mukeshkumar/play/synapse-v3`

## Lessons From `sophie-voice-old`

Useful lessons from the discarded custom runtime:

- iOS audio lifecycle is fragile.
- Playback queue pressure can destabilize the dev client.
- AppState `inactive` should not be treated like a full disconnect.
- Turn-taking needs provisional end + reopen semantics, not only a silence threshold.
- The mobile client should stay thin; server-side runtime semantics matter more.

## Recommended Next Integration Steps

1. Run HF realtime server locally and confirm the endpoint/protocol expected by `sophie-mob`.
2. Decide whether `sophie-mob` should talk directly to the HF realtime endpoint or through a thin Sophie gateway.
3. Add Sophie auth + context injection in the thinnest possible layer.
4. Port only the minimum mobile-side protocol adaptations needed for compatibility.
5. Reintroduce Cortex and tools on top of the HF runtime, not by recreating the runtime itself.
