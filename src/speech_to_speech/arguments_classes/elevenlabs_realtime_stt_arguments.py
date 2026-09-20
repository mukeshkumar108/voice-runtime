from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ElevenLabsRealtimeSTTHandlerArguments:
    elevenlabs_realtime_stt_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "ElevenLabs API key for realtime STT. Defaults to ELEVENLABS_API_KEY from the environment."},
    )
    elevenlabs_realtime_stt_model_id: str = field(
        default="scribe_v2_realtime",
        metadata={
            "help": "ElevenLabs realtime STT model. Supported examples: scribe_v2_realtime, scribe_v2_realtime_turbo, scribe_v2_realtime_lite."
        },
    )
    elevenlabs_realtime_stt_audio_format: str = field(
        default="pcm_16000",
        metadata={"help": "Audio format query parameter for ElevenLabs realtime STT. Default is pcm_16000."},
    )
    elevenlabs_realtime_stt_language_code: Optional[str] = field(
        default=None,
        metadata={"help": "Optional language code for ElevenLabs realtime STT. Default is automatic language detection."},
    )
    elevenlabs_realtime_stt_timeout_s: float = field(
        default=12.0,
        metadata={"help": "Per-turn timeout waiting for a committed ElevenLabs transcript."},
    )
    elevenlabs_realtime_stt_open_timeout_s: float = field(
        default=10.0,
        metadata={"help": "WebSocket connection timeout for ElevenLabs realtime STT."},
    )
