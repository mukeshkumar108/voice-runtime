"""Small runtime-owned tool registry.

These tools execute inside the Sophie runtime. Device clients may provide
context such as timezone, but they do not declare, execute, or continue tools.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from openai.types.responses import ResponseFunctionToolCall

GET_LOCAL_TIME = "get_local_time"


@dataclass(frozen=True)
class RuntimeToolResult:
    call_id: str
    name: str
    arguments: dict[str, Any]
    output: str
    latency_ms: float


def runtime_tool_definitions() -> list[dict[str, Any]]:
    default_timezone = os.environ.get("SOPHIE_USER_TIMEZONE", "Europe/London")
    return [
        {
            "type": "function",
            "name": GET_LOCAL_TIME,
            "description": (
                "Return the current local date and time for an IANA timezone. "
                f"The user's default timezone is {default_timezone}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA timezone such as Europe/London or Asia/Tokyo. "
                            f"Defaults to {default_timezone} for the user's local time."
                        ),
                    }
                },
                "additionalProperties": False,
            },
        }
    ]


def is_runtime_tool(name: str) -> bool:
    return name == GET_LOCAL_TIME


def _parse_arguments(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _now(timezone: ZoneInfo) -> datetime:
    fixed_time = os.environ.get("SOPHIE_TOOL_FIXED_TIME")
    if fixed_time:
        supplied = datetime.fromisoformat(fixed_time)
        if supplied.tzinfo is None:
            raise ValueError("SOPHIE_TOOL_FIXED_TIME must include a UTC offset")
        return supplied.astimezone(timezone)
    return datetime.now(timezone)


def execute_runtime_tool(call: ResponseFunctionToolCall) -> RuntimeToolResult:
    started_at = perf_counter()
    try:
        arguments = _parse_arguments(call.arguments)
        if call.name != GET_LOCAL_TIME:
            raise ValueError(f"unsupported runtime tool: {call.name}")
        timezone_name = arguments.get("timezone") or os.environ.get("SOPHIE_USER_TIMEZONE", "Europe/London")
        if not isinstance(timezone_name, str):
            raise ValueError("timezone must be a string")
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {timezone_name}") from exc
        local_time = _now(timezone)
        payload: dict[str, Any] = {
            "ok": True,
            "timezone": timezone_name,
            "iso_datetime": local_time.isoformat(),
            "date": local_time.strftime("%Y-%m-%d"),
            "day": local_time.strftime("%A"),
            "time_24h": local_time.strftime("%H:%M"),
            "display": local_time.strftime("%-I:%M %p on %A, %-d %B %Y"),
        }
    except (ValueError, json.JSONDecodeError) as exc:
        arguments = {}
        payload = {"ok": False, "error": str(exc)}

    return RuntimeToolResult(
        call_id=call.call_id,
        name=call.name,
        arguments=arguments,
        output=json.dumps(payload, separators=(",", ":")),
        latency_ms=round((perf_counter() - started_at) * 1000, 1),
    )
