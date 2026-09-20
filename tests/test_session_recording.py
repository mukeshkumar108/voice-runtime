from __future__ import annotations

import json
from datetime import datetime, timezone
from queue import Queue
from threading import Event

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import AssistantTextEvent, TranscriptionCompletedEvent
from speech_to_speech.session_recording import SessionEnvelope, SessionRecorder


def _recorder(tmp_path) -> SessionRecorder:
    return SessionRecorder(
        session_id="session_test",
        conversation_id="conv_test",
        output_dir=tmp_path,
        product_id="sophie",
        agent_id="sophie",
        user_id="user-1",
        channel="voice",
        timezone_name="Europe/London",
        consent_scope=["conversation_memory"],
        started_at=datetime(2026, 7, 24, 10, 0, tzinfo=timezone.utc),
    )


def test_speculative_transcript_revision_replaces_earlier_text(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record_user_transcript(
        transcript="I was going",
        turn_id="turn_1",
        turn_revision=0,
        language="en",
        average_logprob=-0.2,
        minimum_logprob=-0.6,
        uncertainty_reason=None,
        provider_metadata={},
    )
    recorder.record_user_transcript(
        transcript="I was going to say something else",
        turn_id="turn_1",
        turn_revision=1,
        language="en",
        average_logprob=-0.1,
        minimum_logprob=-0.3,
        uncertainty_reason=None,
        provider_metadata={},
    )

    envelope = recorder.build_envelope(close_reason="disconnect")

    assert len(envelope.messages) == 1
    assert envelope.messages[0].content == "I was going to say something else"
    assert envelope.messages[0].turn_revision == 1


def test_empty_reopened_transcript_removes_earlier_revision(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record_user_transcript(
        transcript="false start",
        turn_id="turn_1",
        turn_revision=0,
        language="en",
        average_logprob=None,
        minimum_logprob=None,
        uncertainty_reason=None,
        provider_metadata={},
    )
    recorder.record_user_transcript(
        transcript="",
        turn_id="turn_1",
        turn_revision=1,
        language="en",
        average_logprob=None,
        minimum_logprob=None,
        uncertainty_reason=None,
        provider_metadata={},
    )

    assert recorder.build_envelope(close_reason="test").messages == []


def test_assistant_text_is_committed_only_after_completed_response(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.stage_assistant_text(response_id="resp_1", text="First sentence. ", turn_id="turn_1", turn_revision=0)
    recorder.stage_assistant_text(response_id="resp_1", text="Second sentence.", turn_id="turn_1", turn_revision=0)

    assert recorder.build_envelope(close_reason="test").messages == []

    recorder.finish_response(response_id="resp_1", status="completed", reason=None)
    envelope = recorder.build_envelope(close_reason="test")

    assert [message.content for message in envelope.messages] == ["First sentence. Second sentence."]


def test_cancelled_response_is_diagnostic_not_conversation_message(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.stage_assistant_text(response_id="resp_1", text="This was interrupted", turn_id="turn_1", turn_revision=0)

    recorder.finish_response(response_id="resp_1", status="cancelled", reason="turn_detected")
    envelope = recorder.build_envelope(close_reason="test")

    assert envelope.messages == []
    assert envelope.metadata.cancellations[0].reason == "turn_detected"


def test_finalize_persists_atomic_contract_json_once(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record_user_transcript(
        transcript="Hello",
        turn_id="turn_1",
        turn_revision=0,
        language="en",
        average_logprob=None,
        minimum_logprob=None,
        uncertainty_reason=None,
        provider_metadata={"provider": "test"},
    )

    envelope = recorder.finalize(close_reason="disconnect")
    second = recorder.finalize(close_reason="disconnect")
    payload = json.loads((tmp_path / "session_test.json").read_text())

    assert envelope is not None
    assert second is None
    assert payload["_v"] == "synapse.session.v1"
    assert payload["messages"][0]["content"] == "Hello"
    assert SessionEnvelope.model_validate(payload).session_id == "session_test"


def test_example_fixture_matches_pydantic_contract():
    with open("contracts/fixtures/synapse.session.v1.example.json", encoding="utf-8") as handle:
        envelope = SessionEnvelope.model_validate(json.load(handle))

    assert envelope.schema_version == "synapse.session.v1"
    assert [message.role for message in envelope.messages] == ["user", "assistant"]


def test_realtime_service_records_only_completed_turns(tmp_path):
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    session_id = service.register()
    recorder = _recorder(tmp_path)
    service._state(session_id).session_recorder = recorder

    service.dispatch_pipeline_event(
        session_id,
        TranscriptionCompletedEvent(
            transcript="Can you hear me?",
            language_code="en",
            turn_id="turn_1",
            turn_revision=0,
        ),
    )
    service.dispatch_pipeline_event(
        session_id,
        AssistantTextEvent(
            text="I'm right here.",
            turn_id="turn_1",
            turn_revision=0,
        ),
    )
    service.finish_response(session_id, status="completed")
    service.unregister(session_id)

    payload = json.loads((tmp_path / "session_test.json").read_text())
    assert [(message["role"], message["content"]) for message in payload["messages"]] == [
        ("user", "Can you hear me?"),
        ("assistant", "I'm right here."),
    ]


def test_realtime_service_does_not_commit_cancelled_assistant_text(tmp_path):
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    session_id = service.register()
    recorder = _recorder(tmp_path)
    service._state(session_id).session_recorder = recorder

    service.dispatch_pipeline_event(
        session_id,
        AssistantTextEvent(text="This response was interrupted.", turn_id="turn_1", turn_revision=0),
    )
    service.finish_response(session_id, status="cancelled", reason="turn_detected")
    envelope = recorder.build_envelope(close_reason="test")

    assert envelope.messages == []
    assert envelope.metadata.cancellations[0].reason == "turn_detected"


def test_realtime_service_records_runtime_tool_result(tmp_path, monkeypatch):
    monkeypatch.setenv("SOPHIE_TOOL_FIXED_TIME", "2026-07-24T12:10:00+01:00")
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    session_id = service.register()
    recorder = _recorder(tmp_path)
    service._state(session_id).session_recorder = recorder

    service.dispatch_pipeline_event(
        session_id,
        AssistantTextEvent(
            text="",
            tools=[
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_time",
                    name="get_local_time",
                    arguments='{"timezone":"Europe/London"}',
                )
            ],
            turn_id="turn_1",
            turn_revision=0,
        ),
    )
    envelope = recorder.build_envelope(close_reason="test")

    assert len(envelope.metadata.tools) == 1
    assert envelope.metadata.tools[0].name == "get_local_time"
    assert '"time_24h":"12:10"' in envelope.metadata.tools[0].output
