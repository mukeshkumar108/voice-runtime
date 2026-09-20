from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LemonfoxSTTHandlerArguments:
    lemonfox_stt_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Lemonfox API key for STT. Defaults to LEMONFOX_API_KEY from the environment."},
    )
    lemonfox_stt_base_url: str = field(
        default="https://api.lemonfox.ai/v1",
        metadata={"help": "Base URL for the Lemonfox STT API."},
    )
    lemonfox_stt_model_name: str = field(
        default="whisper-1",
        metadata={"help": "Lemonfox transcription model name. Default is 'whisper-1'."},
    )
    lemonfox_stt_language: Optional[str] = field(
        default=None,
        metadata={"help": "Optional language code to force for transcription. Default is auto-detect."},
    )
    lemonfox_stt_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Optional transcription prompt forwarded to Lemonfox."},
    )
    lemonfox_stt_temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for transcription. Default is 0.0."},
    )
    lemonfox_stt_timeout_s: float = field(
        default=75.0,
        metadata={"help": "HTTP timeout in seconds for each Lemonfox STT request."},
    )
