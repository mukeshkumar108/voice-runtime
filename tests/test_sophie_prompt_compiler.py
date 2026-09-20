from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.sophie_prompt_compiler import (
    DEFAULT_CLIENT_SESSION_INSTRUCTIONS,
    SophiePromptCompiler,
    add_transcript_uncertainty_overlay,
)


class _FixedNowCompiler(SophiePromptCompiler):
    def __init__(self, fixed_times: list[datetime], **kwargs):
        super().__init__(**kwargs)
        self._fixed_times = fixed_times

    def _now(self) -> datetime:
        return self._fixed_times.pop(0)


def _write_modules(root: Path) -> None:
    files = {
        "core/identity.md": "# Compact Identity\n\nidentity",
        "core/style-guard.md": "# Compact Style\n\nstyle",
        "stable/00_model_kernel.md": "# Model\n\nmodel",
        "stable/10_identity_kernel.md": "# Identity\n\nidentity",
        "stable/20_steering_kernel.md": "# Steering\n\nsteering",
        "stable/30_product_kernel.md": "# Product\n\nproduct",
        "stable/40_style_kernel.md": "# Style\n\nstyle",
        "stable/user-profile.md": "# User\n\nprofile",
        "stable/important-people.md": "# People\n\npeople",
        "stable/important-projects.md": "# Projects\n\nprojects",
        "overlays/contact/first-contact.md": "# First\n\nfirst",
        "overlays/contact/second-contact.md": "# Second\n\nsecond",
        "overlays/contact/third-or-later-contact.md": "# Third\n\nthird",
        "overlays/daypart/morning.md": "# Morning\n\nmorning",
        "overlays/daypart/afternoon.md": "# Afternoon\n\nafternoon",
        "overlays/daypart/evening.md": "# Evening\n\nevening",
        "overlays/continuity/fresh-session.md": "# Fresh\n\nfresh",
        "overlays/continuity/same-day-continuation.md": "# Same Day\n\nsame day",
        "overlays/continuity/long-gap-reconnect.md": "# Gap\n\ngap",
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def _compiler(tmp_path: Path, times: list[datetime], **kwargs) -> _FixedNowCompiler:
    _write_modules(tmp_path)
    return _FixedNowCompiler(
        fixed_times=times,
        modules_dir=tmp_path,
        timezone_name="Europe/London",
        enable_logging=False,
        **kwargs,
    )


def test_compact_prompt_omits_irrelevant_context(tmp_path: Path) -> None:
    compiler = _compiler(tmp_path, [datetime(2026, 7, 22, 8, 0, tzinfo=ZoneInfo("UTC"))])
    prompt = compiler.compile(RuntimeConfig(), DEFAULT_CLIENT_SESSION_INSTRUCTIONS, current_turn="How are you?")

    assert "# Compact Identity" in prompt
    assert "# Compact Style" in prompt
    assert "# User" in prompt
    assert "# Morning" in prompt
    assert "# People" not in prompt
    assert "# Projects" not in prompt
    assert "Additional session guidance" not in prompt


def test_compact_prompt_selects_context_from_recent_window(tmp_path: Path) -> None:
    compiler = _compiler(tmp_path, [datetime(2026, 7, 22, 13, 0, tzinfo=ZoneInfo("UTC"))])
    prompt = compiler.compile(
        RuntimeConfig(),
        None,
        current_turn="What do you think?",
        recent_context="user: I was talking to Jasmine about the Sophie project.",
    )

    assert "# People" in prompt
    assert "# Projects" in prompt


def test_compact_prompt_does_not_select_projects_for_generic_working_language(tmp_path: Path) -> None:
    compiler = _compiler(tmp_path, [datetime(2026, 7, 22, 13, 0, tzinfo=ZoneInfo("UTC"))])
    prompt = compiler.compile(
        RuntimeConfig(),
        None,
        current_turn="Are you back and working now?",
        recent_context="assistant: I am right here.\nuser: Are you back and working now?",
    )

    assert "# Projects" not in prompt


def test_compact_prompt_keeps_projects_for_immediate_referential_continuation(tmp_path: Path) -> None:
    compiler = _compiler(tmp_path, [datetime(2026, 7, 22, 13, 0, tzinfo=ZoneInfo("UTC"))])
    prompt = compiler.compile(
        RuntimeConfig(),
        None,
        current_turn="What should we do next with it?",
        recent_context=(
            "user: The Cortex runtime integration is finally stable.\n"
            "assistant: That is a meaningful milestone.\n"
            "user: What should we do next with it?"
        ),
    )

    assert "# Projects" in prompt


def test_compact_prompt_drops_stale_project_context(tmp_path: Path) -> None:
    compiler = _compiler(tmp_path, [datetime(2026, 7, 22, 13, 0, tzinfo=ZoneInfo("UTC"))])
    prompt = compiler.compile(
        RuntimeConfig(),
        None,
        current_turn="Are you listening?",
        recent_context=(
            "user: The Cortex runtime is stable.\n"
            "assistant: Good.\n"
            "user: I am going to make tea.\n"
            "assistant: Sensible.\n"
            "user: Are you listening?"
        ),
    )

    assert "# Projects" not in prompt


def test_auto_depth_prefers_continuity_over_daypart_for_deep_turn(tmp_path: Path) -> None:
    compiler = _compiler(tmp_path, [datetime(2026, 7, 22, 20, 0, tzinfo=ZoneInfo("UTC"))])
    prompt = compiler.compile(RuntimeConfig(), None, current_turn="I need to talk honestly about my grief.")

    assert "# Fresh" in prompt
    assert "# Evening" not in prompt


def test_contact_ordinal_changes_per_session_not_per_turn(tmp_path: Path) -> None:
    compiler = _compiler(
        tmp_path,
        [
            datetime(2026, 7, 22, 8, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2026, 7, 22, 8, 10, tzinfo=ZoneInfo("UTC")),
            datetime(2026, 7, 22, 9, 0, tzinfo=ZoneInfo("UTC")),
        ],
    )
    first_session = RuntimeConfig()
    second_session = RuntimeConfig()

    first = compiler.compile(first_session, None, current_turn="Hello")
    same_session = compiler.compile(first_session, None, current_turn="Still here")
    second = compiler.compile(second_session, None, current_turn="Hello again")

    assert "contact 1 today" in first
    assert "contact 1 today" in same_session
    assert "contact 2 today" in second
    assert "# Second" in second


def test_full_mode_preserves_legacy_control(tmp_path: Path) -> None:
    compiler = _compiler(
        tmp_path,
        [datetime(2026, 7, 22, 8, 0, tzinfo=ZoneInfo("UTC"))],
        mode="full",
    )
    prompt = compiler.compile(RuntimeConfig(), None, current_turn="Hello")

    assert "# Model" in prompt
    assert "# Identity" in prompt
    assert "# Steering" in prompt
    assert "# Product" in prompt
    assert "# Style" in prompt
    assert "# People" in prompt
    assert "# Projects" in prompt


def test_transcript_uncertainty_overlay_is_small_and_one_turn_scoped() -> None:
    instructions = add_transcript_uncertainty_overlay("Base guidance.", "low_average_word_logprob")

    assert instructions is not None
    assert "Base guidance." in instructions
    assert "# Transcript uncertainty" in instructions
    assert "ask one brief natural clarification" in instructions
    assert add_transcript_uncertainty_overlay("Base guidance.", None) == "Base guidance."
