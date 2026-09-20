"""Unit tests for the synthetic realtime tool-call harness helpers."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "test_realtime_tool_call.py"
SPEC = importlib.util.spec_from_file_location("test_realtime_tool_call_script", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fixed_london_time_is_reproducible():
    result = MODULE.execute_get_local_time(
        {"timezone": "Europe/London"},
        default_timezone="UTC",
        fixed_time="2026-07-23T12:10:00+01:00",
    )

    assert result["time_24h"] == "12:10"
    assert result["day"] == "Thursday"
    assert result["display"] == "12:10 PM on Thursday, 23 July 2026"


def test_fixed_time_converts_to_requested_timezone():
    result = MODULE.execute_get_local_time(
        {"timezone": "Asia/Tokyo"},
        default_timezone="UTC",
        fixed_time="2026-07-23T12:10:00+01:00",
    )

    assert result["time_24h"] == "20:10"
    assert result["day"] == "Thursday"


def test_missing_timezone_uses_default():
    result = MODULE.execute_get_local_time(
        {},
        default_timezone="Europe/London",
        fixed_time="2026-07-23T12:10:00+01:00",
    )

    assert result["timezone"] == "Europe/London"


def test_invalid_timezone_is_controlled_failure():
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        MODULE.execute_get_local_time(
            {"timezone": "Atlantis/Ocean"},
            default_timezone="UTC",
            fixed_time="2026-07-23T12:10:00+01:00",
        )


def test_tool_output_preserves_call_id_and_serializes_result():
    event = MODULE.tool_output_event("call_123", {"time_24h": "12:10"})

    assert event["type"] == "conversation.item.create"
    assert event["item"]["type"] == "function_call_output"
    assert event["item"]["id"].startswith("fco_")
    assert event["item"]["call_id"] == "call_123"
    assert event["item"]["output"] == '{"time_24h":"12:10"}'
