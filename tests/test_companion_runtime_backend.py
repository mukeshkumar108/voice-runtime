"""Unit tests for the companion-runtime LLM backend (no network)."""

import httpx
import pytest

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    AssistantMessage,
    TextDelta,
)
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.companion_runtime_language_model import CompanionRuntimeModelHandler


def _handler() -> CompanionRuntimeModelHandler:
    # Bypass setup() (which would hit the network in warmup).
    return object.__new__(CompanionRuntimeModelHandler)


def _sse_response(payload: bytes) -> httpx.Response:
    return httpx.Response(200, content=payload)


def test_history_split_sends_prior_turns_and_latest_user_text():
    chat = Chat(10)
    chat.add_item(make_user_message("hello"))
    chat.add_item(make_assistant_message("hi there"))
    chat.add_item(make_user_message("how are you?"))

    handler = _handler()
    handler.conversation_id = "conv_test"
    handler.companion_id = "sophie"
    handler.selected_model_id = None
    handler.user_id = "local-user"
    handler.timezone = "Europe/London"

    payload = handler._serialize(chat)

    assert payload["current_sanitized_message"] == "how are you?"
    assert payload["conversation_id"] == "conv_test"
    assert payload["companion_id"] == "sophie"
    assert payload["medium"] == "voice"
    assert payload["message_parts"] == [{"type": "text", "text": "how are you?"}]
    assert [(m["role"], m["content"]) for m in payload["canonical_history"]] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]
    assert payload["turn_id"].startswith("voice_")


def test_serialize_rejects_empty_chat():
    handler = _handler()
    with pytest.raises(ValueError, match="no user message"):
        handler._serialize(Chat(10))


def test_stream_events_yield_deltas_then_canonical_message():
    raw = (
        b'event: status\ndata: {"phase": "accepted"}\n\n'
        b'event: text_delta\ndata: {"delta": "Hello"}\n\n'
        b'event: text_delta\ndata: {"delta": " there"}\n\n'
        b'event: completed\ndata: {"result": {"assistant_message": "Hello there", "execution_metadata": {}}}\n\n'
    )
    events = list(_handler()._iter_stream_events(_sse_response(raw)))

    assert isinstance(events[0], TextDelta) and events[0].text == "Hello"
    assert isinstance(events[1], TextDelta) and events[1].text == " there"
    assert isinstance(events[2], AssistantMessage)
    assert events[2].content[0].text == "Hello there"


def test_stream_error_event_raises():
    raw = b'event: error\ndata: {"error": {"error_code": "TURN_TIMEOUT", "message": "too slow"}}\n\n'
    with pytest.raises(RuntimeError, match="TURN_TIMEOUT"):
        list(_handler()._iter_stream_events(_sse_response(raw)))


def test_nonstreaming_payload_maps_to_events():
    events = list(_handler()._iter_response_events({"assistant_message": "hi"}))
    assert isinstance(events[0], TextDelta) and events[0].text == "hi"
    assert isinstance(events[1], AssistantMessage)


def test_nonstreaming_error_payload_raises():
    with pytest.raises(RuntimeError, match="PROVIDER_STREAM_ERROR"):
        list(
            _handler()._iter_response_events(
                {"status": "failed", "error_code": "PROVIDER_STREAM_ERROR", "message": "boom"}
            )
        )


def test_no_client_tools_forwarded():
    assert _handler()._build_optional_kwargs([{"type": "function", "name": "x"}], "auto") == {}
