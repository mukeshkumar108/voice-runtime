"""Unit tests for the companion-runtime LLM backend (no network)."""

import httpx
import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.base_openai_compatible_language_model import (
    AssistantMessage,
    TextDelta,
)
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.companion_runtime_language_model import (
    FALLBACK_TEXT,
    CompanionRuntimeModelHandler,
    is_transport_failure,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk


def _handler(**attrs) -> CompanionRuntimeModelHandler:
    # Bypass setup() (which would hit the network in warmup).
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


def _sse_response(payload: bytes) -> httpx.Response:
    return httpx.Response(200, content=payload)


def test_history_split_sends_prior_turns_and_latest_user_text():
    chat = Chat(10)
    chat.add_item(make_user_message("hello"))
    chat.add_item(make_assistant_message("hi there"))
    chat.add_item(make_user_message("how are you?"))

    handler = _handler()
    handler._req_conversation_id = "conv_test"
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


def test_serialize_empty_chat_sends_empty_turn_for_brain_to_reject():
    handler = _handler()
    payload = handler._serialize(Chat(10))
    assert payload["current_sanitized_message"] == ""
    assert payload["message_parts"] == []
    assert payload["canonical_history"] == []


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


def test_serialize_prefers_request_conversation_id():
    handler = _handler(conversation_id="conv_instance", _req_conversation_id="conv_conn_9")
    chat = Chat(10)
    chat.add_item(make_user_message("hi"))
    payload = handler._serialize(chat)
    assert payload["conversation_id"] == "conv_conn_9"

    handler2 = _handler(conversation_id="conv_instance", _req_conversation_id="conv_instance")
    payload2 = handler2._serialize(chat)
    assert payload2["conversation_id"] == "conv_instance"


def test_reliability_mapping():
    reliable = CompanionRuntimeModelHandler._reliability_from_uncertainty(None)
    assert reliable == {"source": "voice_stream", "status": "reliable", "confidence": 1.0}

    uncertain = CompanionRuntimeModelHandler._reliability_from_uncertainty("low confidence words")
    assert uncertain["status"] == "uncertain"
    assert uncertain["confidence"] == 0.6
    assert uncertain["reason"] == "low confidence words"

    long_reason = "x" * 600
    assert len(CompanionRuntimeModelHandler._reliability_from_uncertainty(long_reason)["reason"]) == 500


def test_reliability_reaches_payload():
    handler = _handler(_req_reliability={"source": "voice_stream", "status": "uncertain"})
    chat = Chat(10)
    chat.add_item(make_user_message("mumble"))
    payload = handler._serialize(chat)
    assert payload["transcript_reliability"] == {"source": "voice_stream", "status": "uncertain"}


def test_transport_failure_codes():
    for code in ("TRANSPORT", "TURN_TIMEOUT", "STREAM_EXECUTION_ERROR", "PROVIDER_STREAM_ERROR"):
        assert is_transport_failure(code) is True
    for code in (None, "CapabilityDenied", "TURN_CANCELLED", "STALE_ATTEMPT", "whatever"):
        assert is_transport_failure(code) is False


def _process_request(**kwargs) -> GenerateResponseRequest:
    chat = Chat(10)
    chat.add_item(make_user_message("hello"))
    return GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat),
        language_code="en",
        turn_id="turn_1",
        turn_revision=1,
        **kwargs,
    )


def test_process_speaks_fallback_on_transport_failure():
    handler = _handler()

    def _failed(*a, **k):
        handler._last_brain_error_code = "TRANSPORT"
        return iter([LLMResponseChunk(text="partial"), EndOfResponse(error="boom")])

    handler._generate = _failed

    outputs = list(handler.process(_process_request()))

    assert isinstance(outputs[0], LLMResponseChunk) and outputs[0].text == "partial"
    assert isinstance(outputs[1], LLMResponseChunk) and outputs[1].text == FALLBACK_TEXT
    assert isinstance(outputs[2], EndOfResponse) and outputs[2].error == "boom"


def test_process_stays_silent_on_semantic_failure():
    handler = _handler()

    def _denied(*a, **k):
        handler._last_brain_error_code = "CapabilityDenied"
        return iter([EndOfResponse(error="denied")])

    handler._generate = _denied

    outputs = list(handler.process(_process_request()))

    assert len(outputs) == 1
    assert isinstance(outputs[0], EndOfResponse)


def test_process_stays_silent_when_stale():
    handler = _handler()
    handler._last_brain_error_code = "TRANSPORT"
    handler._generate = lambda *a, **k: iter([EndOfResponse(error="boom")])
    request = _process_request()

    gen = handler.cancel_scope.generation
    handler.cancel_scope.cancel()  # barge-in: everything from gen is stale
    assert handler.cancel_scope.generation != gen

    outputs = list(handler.process(request))
    assert len(outputs) == 1
    assert isinstance(outputs[0], EndOfResponse)


def test_process_stashes_connection_id_and_reliability():
    handler = _handler()
    handler._generate = lambda *a, **k: iter([EndOfResponse()])
    request = _process_request(conversation_id="conv_live", transcript_uncertainty="garbled")

    list(handler.process(request))

    assert handler._req_conversation_id == "conv_live"
    assert handler._req_reliability is not None
    assert handler._req_reliability["status"] == "uncertain"


class _FakeCancelClient:
    def __init__(self):
        self.posts: list[str] = []

    def post(self, url, params=None, timeout=None):
        self.posts.append(url)

        class _Resp:
            def raise_for_status(self):
                pass

        return _Resp()


def test_barge_in_cancels_brain_turn():
    client = _FakeCancelClient()
    handler = _handler(client=client)
    handler._pending_brain_turn = ("voice_abc", "conv_live")

    gen = handler.cancel_scope.generation
    handler._maybe_cancel_brain_turn(gen)
    assert client.posts == []  # not stale: no cancel

    handler.cancel_scope.cancel()
    handler._maybe_cancel_brain_turn(gen)
    assert client.posts == ["http://127.0.0.1:8080/v1/turns/voice_abc/cancel"]
    assert handler._pending_brain_turn is None


def test_failed_cancel_is_swallowed():
    class _Boom:
        def post(self, *a, **k):
            raise ConnectionError("down")

    handler = _handler(client=_Boom())
    handler._cancel_brain_turn("voice_x", "conv_y")  # must not raise
