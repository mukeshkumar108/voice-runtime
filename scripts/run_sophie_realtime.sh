#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

load_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    echo "Loading env from ${env_file}"
    local preserved_vars=(
      VOICE_PORT
      VOICE_RUNTIME_URL
      VOICE_RUNTIME_SECRET
      VOICE_STT_BACKEND
      VOICE_TTS_BACKEND
      VOICE_LEMONFOX_VOICE
      VOICE_MIN_SILENCE_MS
      VOICE_MIN_SPEECH_MS
      VOICE_MIN_SPEECH_CONTINUATION_MS
      VOICE_SPECULATIVE_REOPEN_MS
      VOICE_UNANSWERED_REOPEN_MS
      SOPHIE_VOICE_PORT
      COMPANION_RUNTIME_URL
      COMPANION_RUNTIME_SECRET
      SOPHIE_STT_BACKEND
      SOPHIE_TTS_BACKEND
      SOPHIE_LEMONFOX_VOICE
      SOPHIE_MIN_SILENCE_MS
      SOPHIE_MIN_SPEECH_MS
      SOPHIE_MIN_SPEECH_CONTINUATION_MS
      SOPHIE_SPECULATIVE_REOPEN_MS
      SOPHIE_UNANSWERED_REOPEN_MS
      SYNAPSE_SESSION_RECORDING_DIR
      SYNAPSE_PRODUCT_ID
      SYNAPSE_AGENT_ID
      SYNAPSE_USER_ID
      SYNAPSE_CHANNEL
      SYNAPSE_USER_TIMEZONE
      SYNAPSE_CONSENT_SCOPE
      LEMONFOX_API_KEY
      ELEVENLABS_API_KEY
    )
    local preserved_pairs=()
    local var_name
    for var_name in "${preserved_vars[@]}"; do
      if [[ -n "${!var_name+x}" ]]; then
        preserved_pairs+=("${var_name}=${!var_name}")
      fi
    done
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
    if ((${#preserved_pairs[@]} > 0)); then
      local pair
      for pair in "${preserved_pairs[@]}"; do
        export "${pair}"
      done
    fi
  fi
}

load_env_file "$ROOT_DIR/.env"
load_env_file "$ROOT_DIR/.env.local"
load_env_file "$ROOT_DIR/.env.sophie"

# VOICE_* names are canonical. SOPHIE_*/COMPANION_* are legacy aliases kept
# until the local .env files are migrated; a warning is printed when used.
compat() {
  local new_var="$1" old_var="$2" default="$3"
  local new_val="${!new_var:-}" old_val="${!old_var:-}"
  if [[ -n "$new_val" ]]; then
    printf '%s' "$new_val"
    return
  fi
  if [[ -n "$old_val" ]]; then
    echo "DEPRECATED: ${old_var} is renamed to ${new_var}" >&2
    printf '%s' "$old_val"
    return
  fi
  printf '%s' "$default"
}

PORT="$(compat VOICE_PORT SOPHIE_VOICE_PORT 3002)"
BRAIN_URL="$(compat VOICE_RUNTIME_URL COMPANION_RUNTIME_URL http://127.0.0.1:8080)"
BRAIN_SECRET="$(compat VOICE_RUNTIME_SECRET COMPANION_RUNTIME_SECRET "")"
STT_BACKEND="$(compat VOICE_STT_BACKEND SOPHIE_STT_BACKEND elevenlabs-realtime)"
TTS_BACKEND="$(compat VOICE_TTS_BACKEND SOPHIE_TTS_BACKEND lemonfox)"
LEMONFOX_VOICE="$(compat VOICE_LEMONFOX_VOICE SOPHIE_LEMONFOX_VOICE aoede)"
MIN_SILENCE_MS="$(compat VOICE_MIN_SILENCE_MS SOPHIE_MIN_SILENCE_MS 900)"
MIN_SPEECH_MS="$(compat VOICE_MIN_SPEECH_MS SOPHIE_MIN_SPEECH_MS 384)"
MIN_SPEECH_CONTINUATION_MS="$(compat VOICE_MIN_SPEECH_CONTINUATION_MS SOPHIE_MIN_SPEECH_CONTINUATION_MS 384)"
SPECULATIVE_REOPEN_MS="$(compat VOICE_SPECULATIVE_REOPEN_MS SOPHIE_SPECULATIVE_REOPEN_MS 3000)"
UNANSWERED_REOPEN_MS="$(compat VOICE_UNANSWERED_REOPEN_MS SOPHIE_UNANSWERED_REOPEN_MS 12000)"
export SYNAPSE_SESSION_RECORDING_DIR="${SYNAPSE_SESSION_RECORDING_DIR:-$ROOT_DIR/.synapse/sessions}"
export SYNAPSE_PRODUCT_ID="${SYNAPSE_PRODUCT_ID:-sophie}"
export SYNAPSE_AGENT_ID="${SYNAPSE_AGENT_ID:-sophie}"
export SYNAPSE_USER_ID="${SYNAPSE_USER_ID:-dev-user}"
export SYNAPSE_CHANNEL="${SYNAPSE_CHANNEL:-voice}"
export SYNAPSE_USER_TIMEZONE="${SYNAPSE_USER_TIMEZONE:-${SOPHIE_USER_TIMEZONE:-Europe/London}}"
export SYNAPSE_CONSENT_SCOPE="${SYNAPSE_CONSENT_SCOPE:-conversation_memory}"

if [[ -z "$BRAIN_SECRET" ]]; then
  echo "Missing brain key. Set VOICE_RUNTIME_SECRET (must match the brain's secret)." >&2
  exit 1
fi

CMD=(
  uv run speech-to-speech
  --mode realtime
  --ws_host 0.0.0.0
  --ws_port "$PORT"
  --stt "$STT_BACKEND"
  --llm_backend companion-runtime
  --tts "$TTS_BACKEND"
  --companion_runtime_base_url "$BRAIN_URL"
  --companion_runtime_api_key "$BRAIN_SECRET"
  --min_silence_ms "$MIN_SILENCE_MS"
  --min_speech_ms "$MIN_SPEECH_MS"
  --min_speech_continuation_ms "$MIN_SPEECH_CONTINUATION_MS"
  --speculative_reopen_ms "$SPECULATIVE_REOPEN_MS"
  --unanswered_reopen_ms "$UNANSWERED_REOPEN_MS"
)

if [[ "$STT_BACKEND" == "lemonfox" ]]; then
  CMD+=(--enable_live_transcription false)
else
  CMD+=(--enable_live_transcription)
fi

if [[ "$STT_BACKEND" == "lemonfox" || "$TTS_BACKEND" == "lemonfox" ]]; then
  CMD+=(--lemonfox_stt_api_key "${LEMONFOX_API_KEY:-}")
  CMD+=(--lemonfox_tts_api_key "${LEMONFOX_API_KEY:-}")
fi

if [[ "$STT_BACKEND" == "openrouter-parakeet" ]]; then
  CMD+=(--openrouter_stt_api_key "${OPENROUTER_API_KEY:-}")
  CMD+=(--openrouter_stt_model_name "nvidia/parakeet-tdt-0.6b-v3")
fi

if [[ "$STT_BACKEND" == "elevenlabs-realtime" ]]; then
  CMD+=(--elevenlabs_realtime_stt_api_key "${ELEVENLABS_API_KEY:-}")
  CMD+=(--elevenlabs_realtime_stt_model_id "scribe_v2_realtime")
fi

if [[ "$TTS_BACKEND" == "lemonfox" ]]; then
  CMD+=(--lemonfox_tts_voice "$LEMONFOX_VOICE")
  CMD+=(--lemonfox_tts_response_format pcm)
fi

echo "Starting voice runtime on ws://0.0.0.0:${PORT}/v1/realtime"
echo "Brain: ${BRAIN_URL} (companion-runtime stream)"
echo "STT backend: ${STT_BACKEND}"
echo "TTS backend: ${TTS_BACKEND}"
echo "Session recording: ${SYNAPSE_SESSION_RECORDING_DIR}"
echo "VAD min_silence_ms: ${MIN_SILENCE_MS}"
echo "VAD min_speech_ms: ${MIN_SPEECH_MS}"
echo "VAD min_speech_continuation_ms: ${MIN_SPEECH_CONTINUATION_MS}"
echo "VAD speculative_reopen_ms: ${SPECULATIVE_REOPEN_MS}"
if [[ "$TTS_BACKEND" == "lemonfox" ]]; then
  echo "Lemonfox voice: ${LEMONFOX_VOICE}"
fi

exec "${CMD[@]}"
