from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenRouterSTTHandlerArguments:
    openrouter_stt_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "OpenRouter API key for STT. Defaults to OPENROUTER_API_KEY from the environment."},
    )
    openrouter_stt_base_url: str = field(
        default="https://openrouter.ai/api/v1",
        metadata={"help": "Base URL for the OpenRouter STT API."},
    )
    openrouter_stt_model_name: str = field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        metadata={"help": "OpenRouter STT model slug. Default is 'nvidia/parakeet-tdt-0.6b-v3'."},
    )
    openrouter_stt_language: Optional[str] = field(
        default=None,
        metadata={"help": "Optional language code to force for transcription. Default is auto-detect."},
    )
    openrouter_stt_temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for transcription. Default is 0.0."},
    )
    openrouter_stt_timeout_s: float = field(
        default=12.0,
        metadata={"help": "HTTP timeout in seconds for each OpenRouter STT request."},
    )
