"""Tests for runtime-owned Sophie tools."""

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.runtime_tools import execute_runtime_tool, runtime_tool_definitions


def _call(arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="get_local_time",
        arguments=arguments,
    )


def test_runtime_declares_local_time_tool(monkeypatch):
    monkeypatch.setenv("SOPHIE_USER_TIMEZONE", "Europe/London")

    tools = runtime_tool_definitions()

    assert tools[0]["name"] == "get_local_time"
    assert "Europe/London" in tools[0]["description"]


def test_runtime_executes_time_in_requested_timezone(monkeypatch):
    monkeypatch.setenv("SOPHIE_TOOL_FIXED_TIME", "2026-07-23T12:10:00+01:00")

    result = execute_runtime_tool(_call('{"timezone":"Asia/Tokyo"}'))

    assert result.call_id == "call_1"
    assert '"time_24h":"20:10"' in result.output
    assert '"timezone":"Asia/Tokyo"' in result.output


def test_runtime_returns_controlled_error_for_invalid_timezone():
    result = execute_runtime_tool(_call('{"timezone":"Atlantis/Ocean"}'))

    assert '"ok":false' in result.output
    assert "unknown IANA timezone" in result.output
