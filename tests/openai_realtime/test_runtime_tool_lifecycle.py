"""Runtime-owned tool continuation tests."""

from queue import Queue

from openai.types.realtime import RealtimeConversationItemFunctionCall
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import AssistantTextEvent


def test_runtime_executes_tool_without_client_round_trip(monkeypatch):
    monkeypatch.setenv("SOPHIE_TOOL_FIXED_TIME", "2026-07-23T12:10:00+01:00")
    prompt_queue = Queue()
    service = RealtimeService(text_prompt_queue=prompt_queue)
    conn_id = service.register()
    state = service._state(conn_id)
    call = ResponseFunctionToolCall(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="get_local_time",
        arguments='{"timezone":"Europe/London"}',
    )
    state.runtime_config.chat.add_item(
        RealtimeConversationItemFunctionCall(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="get_local_time",
            arguments='{"timezone":"Europe/London"}',
        )
    )
    service.response._ensure_response(conn_id)

    public_events = service.dispatch_pipeline_event(
        conn_id,
        AssistantTextEvent(text="", tools=[call]),
    )

    assert not any(event.type == "response.function_call_arguments.done" for event in public_events)
    assert state.pending_runtime_tool_results[0].call_id == "call_1"

    completion_events = service.finish_response(conn_id)

    assert [event.type for event in completion_events][-2:] == ["response.done", "response.created"]
    assert state.in_response
    assert state.pending_runtime_tool_results == []
    assert not prompt_queue.empty()
    serialized_chat = state.runtime_config.chat.to_responses_api_chat()
    outputs = [item for item in serialized_chat if item.get("type") == "function_call_output"]
    assert outputs[0]["call_id"] == "call_1"
    assert '"time_24h":"12:10"' in outputs[0]["output"]
