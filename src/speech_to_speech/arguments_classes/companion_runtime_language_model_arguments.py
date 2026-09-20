from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompanionRuntimeLanguageModelHandlerArguments:
    companion_runtime_base_url: str = field(
        default="http://127.0.0.1:8080",
        metadata={"help": "Base URL of the companion-runtime service (owns prompt, memory, tools, model selection)."},
    )
    companion_runtime_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Value for the X-Companion-Runtime-Key header (COMPANION_RUNTIME_SECRET)."},
    )
    companion_runtime_companion_id: str = field(
        default="sophie",
        metadata={"help": "Which companion profile the brain should speak as."},
    )
    companion_runtime_conversation_id: Optional[str] = field(
        default=None,
        metadata={"help": "Stable conversation id for this voice session. Generated per handler when omitted."},
    )
    companion_runtime_selected_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Optional model override. Omitted lets the brain resolve its own model chain."},
    )
    companion_runtime_timeout_s: float = field(
        default=120.0,
        metadata={"help": "HTTP timeout for a brain turn (the brain deadline is 240s; voice should wait, not cut early)."},
    )
    companion_runtime_user_id: str = field(
        default="local-user",
        metadata={"help": "User id sent as trusted context until Core-issued identity exists."},
    )
    companion_runtime_timezone: str = field(
        default="Europe/London",
        metadata={"help": "IANA timezone sent as trusted user context."},
    )
    stream_batch_sentences: int = field(
        default=1,
        metadata={"help": "Sentences per TTS batch. 1 = lowest voice latency."},
    )
