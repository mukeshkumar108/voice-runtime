#!/usr/bin/env bash
# Generic voice-runtime entrypoint. All configuration arrives via VOICE_*
# environment variables; nothing product-specific is baked into the image.
set -euo pipefail

compat() {
  local new_var="$1" old_var="$2" default="$3"
  local new_val="${!new_var:-}" old_val="${!old_var:-}"
  if [[ -n "$new_val" ]]; then printf '%s' "$new_val"; return; fi
  if [[ -n "$old_val" ]]; then
    echo "DEPRECATED: ${old_var} is renamed to ${new_var}" >&2
    printf '%s' "$old_val"; return
  fi
  printf '%s' "$default"
}

PORT="$(compat VOICE_PORT SOPHIE_VOICE_PORT 3002)"
BRAIN_URL="$(compat VOICE_RUNTIME_URL COMPANION_RUNTIME_URL http://host.docker.internal:8080)"
BRAIN_SECRET="$(compat VOICE_RUNTIME_SECRET COMPANION_RUNTIME_SECRET "")"
STT_BACKEND="$(compat VOICE_STT_BACKEND SOPHIE_STT_BACKEND elevenlabs-realtime)"
TTS_BACKEND="$(compat VOICE_TTS_BACKEND SOPHIE_TTS_BACKEND lemonfox)"

if [[ -z "$BRAIN_SECRET" ]]; then
  echo "Missing brain key. Set VOICE_RUNTIME_SECRET." >&2
  exit 1
fi

CMD=(
  speech-to-speech
  --mode realtime
  --ws_host 0.0.0.0
  --ws_port "$PORT"
  --stt "$STT_BACKEND"
  --llm_backend companion-runtime
  --tts "$TTS_BACKEND"
  --companion_runtime_base_url "$BRAIN_URL"
  --companion_runtime_api_key "$BRAIN_SECRET"
  --min_silence_ms "$(compat VOICE_MIN_SILENCE_MS SOPHIE_MIN_SILENCE_MS 900)"
  --min_speech_ms "$(compat VOICE_MIN_SPEECH_MS SOPHIE_MIN_SPEECH_MS 384)"
  --min_speech_continuation_ms "$(compat VOICE_MIN_SPEECH_CONTINUATION_MS SOPHIE_MIN_SPEECH_CONTINUATION_MS 384)"
  --speculative_reopen_ms "$(compat VOICE_SPECULATIVE_REOPEN_MS SOPHIE_SPECULATIVE_REOPEN_MS 3000)"
  --unanswered_reopen_ms "$(compat VOICE_UNANSWERED_REOPEN_MS SOPHIE_UNANSWERED_REOPEN_MS 12000)"
  --enable_live_transcription
)

if [[ "$STT_BACKEND" == "lemonfox" ]]; then
  CMD+=(--enable_live_transcription false)
fi
if [[ -n "${LEMONFOX_API_KEY:-}" ]]; then
  CMD+=(--lemonfox_stt_api_key "$LEMONFOX_API_KEY" --lemonfox_tts_api_key "$LEMONFOX_API_KEY")
fi
if [[ -n "${ELEVENLABS_API_KEY:-}" && "$STT_BACKEND" == "elevenlabs-realtime" ]]; then
  CMD+=(--elevenlabs_realtime_stt_api_key "$ELEVENLABS_API_KEY")
fi
if [[ -n "${OPENROUTER_API_KEY:-}" && "$STT_BACKEND" == "openrouter-parakeet" ]]; then
  CMD+=(--openrouter_stt_api_key "$OPENROUTER_API_KEY")
fi
if [[ "$TTS_BACKEND" == "lemonfox" ]]; then
  CMD+=(--lemonfox_tts_voice "$(compat VOICE_LEMONFOX_VOICE SOPHIE_LEMONFOX_VOICE aoede)" --lemonfox_tts_response_format pcm)
fi

exec "${CMD[@]}"
