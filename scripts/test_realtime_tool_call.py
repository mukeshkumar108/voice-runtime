"""Exercise one complete tool-call loop through the realtime websocket.

This bypasses microphone/STT only. The running runtime owns tool declaration,
execution, conversation write-back, continuation, and TTS output. The client
only injects text and observes the resulting response lifecycle.

Examples:
    uv run python scripts/test_realtime_tool_call.py
    uv run python scripts/test_realtime_tool_call.py --live-clock
    uv run python scripts/test_realtime_tool_call.py \
        --prompt "What time is it in Tokyo?" --default-timezone Asia/Tokyo
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import websockets

DEFAULT_FIXED_TIME = "2026-07-23T12:10:00+01:00"
DEFAULT_PROMPT = (
    "What is the exact current local time in London, including the hour and minutes? "
    "Use the time tool before answering."
)
TOOL_NAME = "get_local_time"


@dataclass
class Milestones:
    connected_ms: float | None = None
    first_response_created_ms: float | None = None
    tool_call_ms: float | None = None
    tool_result_sent_ms: float | None = None
    continuation_created_ms: float | None = None
    first_text_ms: float | None = None
    first_audio_ms: float | None = None
    completed_ms: float | None = None


@dataclass
class Report:
    passed: bool = False
    prompt: str = ""
    session_id: str | None = None
    response_ids: list[str] = field(default_factory=list)
    call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    assistant_text: str = ""
    audio_bytes: int = 0
    audio_duration_ms: float = 0.0
    event_types: list[str] = field(default_factory=list)
    milestones: Milestones = field(default_factory=Milestones)
    assertions: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def elapsed_ms(started_at: float) -> float:
    return round((time.monotonic() - started_at) * 1000, 1)


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def execute_get_local_time(
    arguments: dict[str, Any],
    *,
    default_timezone: str,
    fixed_time: str | None,
) -> dict[str, str]:
    timezone_name = arguments.get("timezone") or default_timezone
    if not isinstance(timezone_name, str):
        raise ValueError("timezone must be a string")
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown IANA timezone: {timezone_name}") from exc

    if fixed_time is None:
        local_time = datetime.now(timezone)
    else:
        supplied = datetime.fromisoformat(fixed_time)
        if supplied.tzinfo is None:
            raise ValueError("--fixed-time must include a UTC offset")
        local_time = supplied.astimezone(timezone)

    return {
        "timezone": timezone_name,
        "iso_datetime": local_time.isoformat(),
        "date": local_time.strftime("%Y-%m-%d"),
        "day": local_time.strftime("%A"),
        "time_24h": local_time.strftime("%H:%M"),
        "display": local_time.strftime("%-I:%M %p on %A, %-d %B %Y"),
    }


def session_update() -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": (
                "Speak naturally and concisely. For current local time questions, "
                "call get_local_time and ground the spoken answer in its result. "
                "When exact time is requested, include the returned hour and minutes."
            ),
        },
    }


def user_text_event(prompt: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        },
    }


def tool_output_event(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {
            "id": f"fco_{uuid.uuid4().hex}",
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result, separators=(",", ":")),
        },
    }


async def receive_event(ws: Any, deadline: float) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("tool-call experiment timed out")
    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
    if isinstance(raw, bytes):
        raise ValueError("expected JSON websocket event, received binary data")
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise ValueError("websocket event must be a JSON object")
    return event


def event_response_id(event: dict[str, Any]) -> str | None:
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    response_id = event.get("response_id")
    return response_id if isinstance(response_id, str) else None


async def run_experiment(args: argparse.Namespace) -> Report:
    report = Report(prompt=args.prompt)
    started_at = time.monotonic()
    deadline = started_at + args.timeout
    fixed_time = None if args.live_clock else args.fixed_time
    first_response_done = False
    continuation_done = False
    session_created = False
    assistant_parts: list[str] = []
    seen_tool_calls = 0
    report.tool_name = TOOL_NAME
    report.tool_arguments = {"timezone": args.default_timezone}
    report.tool_result = execute_get_local_time(
        report.tool_arguments,
        default_timezone=args.default_timezone,
        fixed_time=fixed_time,
    )

    headers = None
    if args.token:
        headers = [("Authorization", f"Bearer {args.token}")]

    try:
        async with websockets.connect(
            args.url,
            max_size=2**24,
            additional_headers=headers,
        ) as ws:
            while True:
                event = await receive_event(ws, deadline)
                event_type = str(event.get("type", ""))
                report.event_types.append(event_type)

                if event_type == "error":
                    report.errors.append(json.dumps(event.get("error", event), sort_keys=True))
                    break

                if event_type == "session.created":
                    session_created = True
                    report.milestones.connected_ms = elapsed_ms(started_at)
                    session = event.get("session") or {}
                    # The local runtime does not currently expose its internal
                    # session ID, so retain the creation event ID for correlation.
                    report.session_id = session.get("id") or event.get("event_id")
                    await ws.send(json.dumps(session_update()))
                    await ws.send(json.dumps(user_text_event(args.prompt)))
                    await ws.send(json.dumps({"type": "response.create"}))
                    continue

                response_id = event_response_id(event)
                if response_id and response_id not in report.response_ids:
                    report.response_ids.append(response_id)

                if event_type == "response.created":
                    if not first_response_done:
                        report.milestones.first_response_created_ms = elapsed_ms(started_at)
                    else:
                        report.milestones.continuation_created_ms = elapsed_ms(started_at)

                elif event_type == "response.function_call_arguments.done":
                    seen_tool_calls += 1
                    report.errors.append("runtime-owned tool call leaked to the websocket client")
                    break

                elif event_type in {"response.output_text.delta", "response.output_audio_transcript.done"}:
                    text = event.get("delta") or event.get("transcript")
                    if text:
                        if report.milestones.first_text_ms is None and first_response_done:
                            report.milestones.first_text_ms = elapsed_ms(started_at)
                        assistant_parts.append(str(text))

                elif event_type == "response.output_audio.delta":
                    audio = base64.b64decode(event.get("delta", ""))
                    if report.milestones.first_audio_ms is None and first_response_done:
                        report.milestones.first_audio_ms = elapsed_ms(started_at)
                    report.audio_bytes += len(audio)

                elif event_type == "response.done":
                    status = (event.get("response") or {}).get("status")
                    if status != "completed":
                        report.errors.append(f"response ended with status {status!r}")
                        break
                    if not first_response_done:
                        first_response_done = True
                    else:
                        continuation_done = True
                        report.milestones.completed_ms = elapsed_ms(started_at)
                        break
    except Exception as exc:  # noqa: BLE001 - diagnostic harness must report failures
        report.errors.append(f"{type(exc).__name__}: {exc}")

    report.assistant_text = " ".join(part.strip() for part in assistant_parts if part.strip()).strip()
    report.audio_duration_ms = round(report.audio_bytes / (16000 * 2) * 1000, 1)
    expected_time = report.tool_result.get("time_24h") if report.tool_result else None
    display_time = report.tool_result.get("display") if report.tool_result else None
    grounding_candidates = {
        value.lower()
        for value in (
            expected_time,
            display_time.split(" on ", 1)[0] if display_time else None,
        )
        if value
    }
    normalized_answer = report.assistant_text.lower()
    report.assertions = {
        "session_created": session_created,
        "runtime_tool_hidden_from_client": seen_tool_calls == 0,
        "correct_tool": report.tool_name == TOOL_NAME,
        "valid_arguments": isinstance(report.tool_arguments, dict),
        "runtime_started_continuation": report.milestones.continuation_created_ms is not None,
        "continuation_completed": continuation_done,
        "spoken_or_text_answer_received": bool(report.assistant_text),
        "answer_grounded_in_tool_result": any(value in normalized_answer for value in grounding_candidates),
        "audio_received": report.audio_bytes > 0,
        "no_protocol_errors": not report.errors,
    }
    report.passed = all(report.assertions.values())
    return report


def write_report(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:3002/v1/realtime")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--default-timezone", default="Europe/London")
    parser.add_argument("--fixed-time", default=DEFAULT_FIXED_TIME)
    parser.add_argument(
        "--live-clock",
        action="store_true",
        help="Use the current time in the tool-requested timezone instead of --fixed-time.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--token", default=None, help="Optional bearer token for a protected endpoint.")
    parser.add_argument("--report", type=Path, default=Path("/tmp/sophie_tool_call_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_experiment(args))
    write_report(report, args.report)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    print(f"\nReport: {args.report}")
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
