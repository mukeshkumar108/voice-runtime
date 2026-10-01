# Voice Runtime Contract (frozen)

Voice Runtime is realtime speech I/O for Companion Runtime. It is a modality
adapter, not a second companion implementation.

## Upward (voice -> brain): one TurnInput per spoken turn

`POST {VOICE_RUNTIME_URL}/v1/turns/stream` with:

| Field | Meaning |
|---|---|
| `turn_id` | Fresh `voice_*` id per generation. Retries re-execute; there is no cross-turn recovery. |
| `conversation_id` | The connection's id, assigned by the realtime service at accept. Stable for the session, never shared across sessions. |
| `companion_id` | Which companion profile the brain should speak as. Voice passes it through; it never interprets it. |
| `current_sanitized_message` + `message_parts` | The final STT transcript for this turn. |
| `canonical_history` | Bounded prior user/assistant turns from this connection's chat. Voice history resets per connection. |
| `trusted_user_context` | Unbound dev mode: `{user_id, timezone}`. Bound call: `{user_id, timezone, medium, session_routing, entry_context}`. See "Call identity". |
| `transcript_reliability` | `{source: voice_stream, status: reliable\|uncertain, confidence, reason?}` from STT signals. |
| `medium` | Always `"voice"`. Modality metadata, not authority: the brain decides what it means. |

Future modality signals ride in the same envelope (`device`, interruption
flags) without changing this shape. Voice reports; the brain interprets.

## Call identity (product-issued)

When `VOICE_SESSION_SECRET` is set the WebSocket requires `?token=` issued by the
product host (HMAC-SHA256 over `{uid, cid, tz, companion, exp}`); anonymous
connections are refused. A bound call is a *modality of an existing chat*:

- `conversation_id` is the product's chat id (not a per-connection id).
- On connect, voice reads `GET {VOICE_HOST_URL}/api/voice/context` (bearer =
  the token): the chat's recent history and the previous Runtime
  `next_session_state`.
- Each turn sends that state back as `trusted_user_context.session_routing`
  and replaces it with the `next_session_state` from the `completed` event, so
  the Runtime hydrates Cortex once (first turn of the call) and not again.
  Voice never reads or edits the state.
- `entry_context.chronology`: the call is its own sitting (`new` on its first
  turn, `same` afterwards).
- After each completed turn, voice posts `{turn_id, user_text, assistant_text,
  next_session_state}` to `{VOICE_HOST_URL}/api/voice/turn` (best effort), so
  spoken turns become ordinary chat history, mirrored exactly like text.

## Downward (brain -> voice): streamed text + terminal state

Server-sent events on the stream: `status` (diagnostic phases), `text_delta`
(provisional, sentence-batched into TTS), then exactly one terminal event:

- `completed` carries the canonical `assistant_message` (the only text ever persisted or spoken in full).
- `error` carries `{error_code, message}`. Transport-level failures
  (`TRANSPORT`, `TURN_TIMEOUT`, `STREAM_EXECUTION_ERROR`,
  `PROVIDER_STREAM_ERROR`) produce the fixed spoken retry prompt.
  Everything else stays silent. Voice never invents reply content.

Barge-in cancels the in-flight brain turn via
`POST /v1/turns/{turn_id}/cancel?conversation_id=...`.

## Session envelopes

Each connection records a `synapse.session.v1` envelope locally
(`contracts/`, `SYNAPSE_SESSION_RECORDING_DIR`). Delivery to memory systems
is out of scope for this repo.

## Explicitly NOT owned here

Identity, memory, prompt compilation, model selection, behaviour, response
policy, tools, overlays, emotional interpretation, companion identity,
Cortex/Synapse reasoning. Changes to any of those belong in Companion
Runtime. Voice changes that would alter what the companion *says or decides*
do not belong here.

## Diagnostic hatch (not a product path)

`--llm_backend responses-api|chat-completions|transformers|mlx-lm` runs the
pipeline against a model directly with a fixed generic prompt
(`LLM/diagnostic_prompt.py`). Purpose: answer "voice transport broken, brain
broken, or provider broken?" Nothing product-specific may live in this path;
it carries no companion prompts, overlays, or product tools.
