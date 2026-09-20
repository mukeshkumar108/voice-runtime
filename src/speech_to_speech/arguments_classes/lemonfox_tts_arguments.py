from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LemonfoxTTSHandlerArguments:
    lemonfox_tts_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Lemonfox API key for TTS. Defaults to LEMONFOX_API_KEY from the environment."},
    )
    lemonfox_tts_base_url: str = field(
        default="https://api.lemonfox.ai/v1",
        metadata={"help": "Base URL for the Lemonfox TTS API."},
    )
    lemonfox_tts_model_name: str = field(
        default="speech-1",
        metadata={"help": "Lemonfox TTS model name. Default is 'speech-1'."},
    )
    lemonfox_tts_voice: str = field(
        default="aoede",
        metadata={"help": "Default Lemonfox voice when the realtime session does not override it."},
    )
    lemonfox_tts_response_format: str = field(
        default="pcm",
        metadata={"help": "Audio format to request from Lemonfox. Recommended: 'pcm' for streaming."},
    )
    lemonfox_tts_speed: float = field(
        default=1.0,
        metadata={"help": "Speech speed passed to Lemonfox. Default is 1.0."},
    )
    lemonfox_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in samples for streaming output. Default is 512."},
    )
    lemonfox_tts_input_sample_rate: int = field(
        default=24000,
        metadata={"help": "Expected Lemonfox PCM sample rate before resampling into the 16k pipeline."},
    )
    lemonfox_tts_output_sample_rate: int = field(
        default=16000,
        metadata={"help": "Pipeline audio sample rate. Default is 16000."},
    )
    lemonfox_tts_timeout_s: float = field(
        default=60.0,
        metadata={"help": "HTTP timeout in seconds for each Lemonfox TTS request."},
    )
