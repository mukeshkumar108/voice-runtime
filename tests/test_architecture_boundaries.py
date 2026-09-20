"""Architecture boundary tests: voice stays a modality adapter.

These pin the frozen contract (docs/VOICE_RUNTIME_CONTRACT.md):
- the brain path never touches product cognition or product tools;
- every connection gets isolated conversation state;
- barge-in cancels the brain turn;
- only transport failures speak; everything else stays silent.
"""

import ast
from pathlib import Path
from threading import Event

import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.companion_runtime_language_model import (
    FALLBACK_TEXT,
    CompanionRuntimeModelHandler,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "speech_to_speech"

# Product-cognition modules the brain path must never import. The brain owns
# all of these; voice only carries the transcript envelope.
FORBIDDEN_BRAIN_PATH_IMPORTS = ("sophie", "runtime_tools", "cortex", "honcho", "synapse")


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_brain_backend_has_no_product_imports():
    handler_path = SRC_ROOT / "LLM" / "companion_runtime_language_model.py"
    assert handler_path.exists()
    violations = [
        name for name in _imported_modules(handler_path) if "sophie" in name or name == "speech_to_speech.runtime_tools"
    ]
    assert violations == []


def test_default_backend_is_the_brain():
    assert ModuleArguments().llm_backend == "companion-runtime"


def test_brain_path_session_advertises_no_product_tools():
    service = RealtimeService(runtime_tools=False)
    session_id = service.register()
    try:
        session = service._state(session_id).runtime_config.session
        assert session.tools is None
    finally:
        service.unregister(session_id)


def test_hatch_session_keeps_diagnostic_tools():
    service = RealtimeService()  # default: direct-LLM diagnostic harness
    session_id = service.register()
    try:
        tools = service._state(session_id).runtime_config.session.tools
        assert tools, "diagnostic harness must still advertise its local tools"
    finally:
        service.unregister(session_id)


def test_connections_get_isolated_conversation_state():
    service = RealtimeService(runtime_tools=False)
    first = service.register()
    second = service.register()
    try:
        a = service._state(first)
        b = service._state(second)
        assert a.conversation_id != b.conversation_id
        assert a.session_id != b.session_id
        assert a.runtime_config.chat is not b.runtime_config.chat
    finally:
        service.unregister(first)
        service.unregister(second)


def _brain_handler(**attrs) -> CompanionRuntimeModelHandler:
    handler = object.__new__(CompanionRuntimeModelHandler)
    handler.cancel_scope = CancelScope()
    handler.speculative_turns = None
    handler.stream = True
    handler.stream_batch_sentences = 1
    handler.enable_lang_prompt = False
    handler.gen_kwargs = {}
    handler.base_url = "http://127.0.0.1:8080"
    handler.companion_id = "sophie"
    handler.selected_model_id = None
    handler.user_id = "local-user"
    handler.timezone = "Europe/London"
    handler.conversation_id = "conv_instance"
    handler._req_conversation_id = "conv_instance"
    handler._req_reliability = None
    handler._pending_brain_turn = None
    handler._last_brain_error_code = None
    handler._turn_started_at = 0.0
    handler._request_sent_at = 0.0
    handler._first_delta_logged = False
    for key, value in attrs.items():
        setattr(handler, key, value)
    return handler


def _request(**kwargs) -> GenerateResponseRequest:
    chat = Chat(10)
    chat.add_item(make_user_message("hello"))
    return GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat),
        language_code="en",
        turn_id="turn_1",
        turn_revision=1,
        **kwargs,
    )


class _CancelSpy:
    def __init__(self):
        self.posts: list[str] = []

    def post(self, url, params=None, timeout=None):
        self.posts.append(url)

        class _Resp:
            def raise_for_status(self):
                pass

        return _Resp()


def test_midstream_barge_in_cancels_brain_turn():
    spy = _CancelSpy()
    handler = _brain_handler(client=spy)

    def _interrupted(*a, **k):
        handler._pending_brain_turn = ("voice_mid", "conv_x")
        handler.cancel_scope.cancel()  # user barges in mid-stream
        return iter([EndOfResponse()])

    handler._generate = _interrupted
    outputs = list(handler.process(_request(conversation_id="conv_x")))

    assert spy.posts == ["http://127.0.0.1:8080/v1/turns/voice_mid/cancel"]
    # Barge-in output stays silent: no retry prompt over the user's speech.
    assert len(outputs) == 1 and isinstance(outputs[0], EndOfResponse)


@pytest.mark.parametrize(
    ("code", "speaks"),
    [
        ("TRANSPORT", True),
        ("TURN_TIMEOUT", True),
        ("STREAM_EXECUTION_ERROR", True),
        ("PROVIDER_STREAM_ERROR", True),
        ("CapabilityDenied", False),
        ("TURN_CANCELLED", False),
        ("STALE_ATTEMPT", False),
        ("SOME_FUTURE_CODE", False),
        (None, False),
    ],
)
def test_only_transport_failures_speak(code, speaks):
    handler = _brain_handler()

    def _failed(*a, **k):
        handler._last_brain_error_code = code
        return iter([EndOfResponse(error="boom")])

    handler._generate = _failed
    outputs = list(handler.process(_request()))

    texts = [o.text for o in outputs if isinstance(o, LLMResponseChunk)]
    assert (FALLBACK_TEXT in texts) is speaks


def test_health_endpoints_exist():
    app = create_app(pool=[], stop_event=Event())
    paths = {route.path for route in app.routes}
    assert "/healthz" in paths
    assert "/readyz" in paths
    assert "/v1/realtime" in paths
