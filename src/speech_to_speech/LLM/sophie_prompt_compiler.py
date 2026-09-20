from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

DEFAULT_CLIENT_SESSION_INSTRUCTIONS = (
    "You are Sophie, a warm, concise voice companion. Respond conversationally and naturally for spoken dialogue."
)
TRANSCRIPT_UNCERTAINTY_OVERLAY = """# Transcript uncertainty

The speech recognizer marked this latest utterance as uncertain. If its meaning is not obvious and low-risk from
the immediate conversation, ask one brief natural clarification. Do not invent a confident interpretation and do
not mention confidence scores or speech-recognition diagnostics."""


def add_transcript_uncertainty_overlay(instructions: str | None, reason: str | None) -> str | None:
    if not reason:
        return instructions
    logger.info("sophie.transcript_uncertainty reason=%s", reason)
    return "\n\n".join(part for part in ((instructions or "").strip(), TRANSCRIPT_UNCERTAINTY_OVERLAY) if part)


@dataclass
class _ContactState:
    last_contact_at: datetime | None = None
    last_contact_local_date: str | None = None
    contact_count_today: int = 0
    session_started_at: datetime | None = None
    contact_ordinal_today: int = 0
    last_prompt_hash: str | None = None


@dataclass(frozen=True)
class PromptSelection:
    depth: str
    selected_modules: list[str]
    omitted_modules: list[str]
    reasons: list[str]


class SophiePromptCompiler:
    """Build a small, inspectable Sophie prompt from deterministic modules."""

    LEGACY_KERNEL_MODULES = [
        "stable/00_model_kernel.md",
        "stable/10_identity_kernel.md",
        "stable/20_steering_kernel.md",
        "stable/30_product_kernel.md",
        "stable/40_style_kernel.md",
    ]
    COMPACT_CORE_MODULES = ["core/identity.md", "core/style-guard.md"]
    CONTEXT_MODULES = [
        "stable/user-profile.md",
        "stable/important-people.md",
        "stable/important-projects.md",
    ]

    PEOPLE_TERMS = {
        "jasmine",
        "ashley",
        "morgan",
        "daughter",
        "father",
        "dad",
        "fiance",
        "friend",
        "co-founder",
        "cofounder",
        "relationship",
        "estranged",
        "guatemala",
    }
    PROJECT_IDENTIFIERS = {
        "cortex",
        "synapse",
        "bloom",
        "sophie voice",
        "sophie core",
        "sophie mob",
        "rpd2",
    }
    PROJECT_TERMS = {
        "project",
        "product",
        "startup",
        "founder",
        "code",
        "codebase",
        "repo",
        "repository",
        "runtime",
        "compiler",
        "deployment",
        "backend",
        "frontend",
    }
    PROJECT_CONTINUATION_TERMS = {
        "it",
        "that",
        "this",
        "those",
        "changes",
        "working",
        "next",
        "continue",
        "what do you think",
    }
    DEEP_TERMS = {
        "grief",
        "grieving",
        "died",
        "death",
        "estranged",
        "ashamed",
        "guilt",
        "terrified",
        "relationship",
        "breakup",
        "depressed",
        "lonely",
        "overwhelmed",
        "meaning of",
        "deep",
        "honest with me",
        "tell me the truth",
    }
    MEDIUM_TERMS = {
        "think through",
        "help me decide",
        "what should i",
        "why do i",
        "plan",
        "stuck",
        "confused",
        "worried",
        "frustrated",
        "important",
        "serious",
        "challenge me",
    }

    def __init__(
        self,
        *,
        modules_dir: Path | None = None,
        timezone_name: str | None = None,
        long_gap_minutes: int = 240,
        enable_logging: bool = True,
        mode: str | None = None,
        depth: str | None = None,
        log_prompt: bool | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.modules_dir = modules_dir or repo_root / "sophie_prompt"
        self.timezone_name = timezone_name or os.getenv("SOPHIE_USER_TIMEZONE", "Europe/London")
        self.long_gap_minutes = max(1, int(long_gap_minutes))
        self.enable_logging = enable_logging
        self.mode = self._choice(mode or os.getenv("SOPHIE_PROMPT_MODE", "compact"), {"compact", "full"}, "compact")
        self.depth = self._choice(
            depth or os.getenv("SOPHIE_PROMPT_DEPTH", "auto"), {"auto", "short", "medium", "deep"}, "auto"
        )
        self.log_prompt = self._env_bool("SOPHIE_LOG_COMPILED_PROMPT", False) if log_prompt is None else log_prompt
        self._lock = Lock()
        self._state_by_runtime: dict[int, _ContactState] = {}
        self._last_contact_at: datetime | None = None
        self._last_contact_local_date: str | None = None
        self._contact_count_today = 0
        self._module_cache: dict[str, str] = {}

    def compile(
        self,
        runtime_config: RuntimeConfig,
        instructions: str | None,
        *,
        current_turn: str = "",
        recent_context: str = "",
        history_token_estimate: int = 0,
    ) -> str:
        now = self._now()
        context = self._build_context(runtime_config, now)
        selection = self._select_modules(current_turn, recent_context, context)

        sections = [self._read_module(module) for module in selection.selected_modules]
        sections.append(self._build_session_frame(context))

        passthrough = self._normalize_client_instructions(instructions)
        if passthrough:
            sections.append(f"# Session guidance\n\n{passthrough}")

        prompt = "\n\n".join(section.strip() for section in sections if section.strip()).strip()
        prompt_tokens = self._estimate_tokens(prompt)

        if self.enable_logging:
            metadata = {
                "mode": self.mode,
                "depth": selection.depth,
                "timezone": self.timezone_name,
                "local_time": context["local_time_iso"],
                "part_of_day": context["part_of_day"],
                "contact_ordinal_today": context["contact_ordinal_today"],
                "minutes_since_last_contact": context["minutes_since_last_contact"],
                "continuity": context["continuity"],
                "selected_modules": selection.selected_modules,
                "omitted_modules": selection.omitted_modules,
                "selection_reasons": selection.reasons,
                "prompt_token_estimate": prompt_tokens,
                "history_token_estimate": history_token_estimate,
                "combined_token_estimate": prompt_tokens + history_token_estimate,
                "char_count": len(prompt),
            }
            logger.info("sophie.prompt_compiler.selection %s", json.dumps(metadata, sort_keys=True))
            if self.log_prompt:
                logger.info("sophie.prompt_compiler.prompt\n%s", prompt)

        self._commit_context(runtime_config, now, prompt)
        return prompt

    def _select_modules(
        self,
        current_turn: str,
        recent_context: str,
        context: dict[str, object],
    ) -> PromptSelection:
        if self.mode == "full":
            selected = [
                *self.LEGACY_KERNEL_MODULES,
                *self.CONTEXT_MODULES,
                context["contact_module"],
                context["daypart_module"],
                context["continuity_module"],
            ]
            return PromptSelection("full", selected, [], ["legacy control mode"])

        evidence = self._normalize_evidence(f"{recent_context}\n{current_turn}")
        current = self._normalize_evidence(current_turn)
        prior = self._immediate_prior_context(recent_context, current)
        depth = self._select_depth(current)
        selected = [*self.COMPACT_CORE_MODULES, "stable/user-profile.md"]
        reasons = [f"depth={depth}"]

        people_relevant = self._contains_term(evidence, self.PEOPLE_TERMS)
        current_project_evidence = self._contains_term(
            current,
            self.PROJECT_IDENTIFIERS | self.PROJECT_TERMS,
        )
        continuation_project_evidence = self._contains_term(
            current,
            self.PROJECT_CONTINUATION_TERMS,
        ) and self._contains_term(prior, self.PROJECT_IDENTIFIERS | self.PROJECT_TERMS)
        projects_relevant = current_project_evidence or continuation_project_evidence
        if people_relevant:
            selected.append("stable/important-people.md")
            reasons.append("people referenced in current/recent turns")
        if projects_relevant:
            selected.append("stable/important-projects.md")
            reasons.append(
                "project referenced in current turn"
                if current_project_evidence
                else "project referenced by immediate continuation"
            )

        # Short turns need one situational nudge. Deeper turns benefit more from
        # relationship continuity than generic time-of-day flavour.
        if depth == "short":
            selected.append(context["daypart_module"])
        elif context["continuity"] in {"fresh_session", "long_gap_reconnect"}:
            selected.append(context["continuity_module"])

        if context["contact_ordinal_today"] > 1:
            selected.append(context["contact_module"])

        all_optional = [
            *self.CONTEXT_MODULES[1:],
            context["contact_module"],
            context["daypart_module"],
            context["continuity_module"],
        ]
        omitted = [module for module in all_optional if module not in selected]
        return PromptSelection(depth, selected, omitted, reasons)

    def _select_depth(self, current_turn: str) -> str:
        if self.depth != "auto":
            return self.depth
        words = current_turn.split()
        if self._contains_term(current_turn, self.DEEP_TERMS) or len(words) >= 80:
            return "deep"
        if self._contains_term(current_turn, self.MEDIUM_TERMS) or len(words) >= 35:
            return "medium"
        return "short"

    def _build_context(self, runtime_config: RuntimeConfig, now: datetime) -> dict[str, object]:
        local_now = now.astimezone(self._timezone())
        runtime_id = id(runtime_config)
        with self._lock:
            state = self._state_by_runtime.setdefault(runtime_id, _ContactState())
            local_date = local_now.date().isoformat()
            if state.session_started_at is not None:
                contact_ordinal = state.contact_ordinal_today
            else:
                contact_ordinal = 1 if self._last_contact_local_date != local_date else self._contact_count_today + 1
            minutes_since_last_contact = None
            if self._last_contact_at is not None:
                minutes_since_last_contact = max(0, int((now - self._last_contact_at).total_seconds() // 60))
            fresh_session = state.session_started_at is None
            same_day_continuation = self._last_contact_local_date == local_date and not fresh_session
            long_gap = minutes_since_last_contact is not None and minutes_since_last_contact >= self.long_gap_minutes

        part_of_day = self._part_of_day(local_now.hour)
        continuity = "fresh_session" if fresh_session else "same_day_continuation"
        continuity_module = "overlays/continuity/fresh-session.md"
        if not fresh_session and long_gap:
            continuity = "long_gap_reconnect"
            continuity_module = "overlays/continuity/long-gap-reconnect.md"
        elif same_day_continuation:
            continuity_module = "overlays/continuity/same-day-continuation.md"

        contact_module = (
            "overlays/contact/first-contact.md"
            if contact_ordinal <= 1
            else "overlays/contact/second-contact.md"
            if contact_ordinal == 2
            else "overlays/contact/third-or-later-contact.md"
        )
        return {
            "runtime_id": runtime_id,
            "local_time_iso": local_now.isoformat(timespec="minutes"),
            "part_of_day": part_of_day,
            "contact_ordinal_today": contact_ordinal,
            "minutes_since_last_contact": minutes_since_last_contact,
            "fresh_session": fresh_session,
            "continuity": continuity,
            "contact_module": contact_module,
            "continuity_module": continuity_module,
            "daypart_module": f"overlays/daypart/{part_of_day}.md",
        }

    def _commit_context(self, runtime_config: RuntimeConfig, now: datetime, prompt: str) -> None:
        runtime_id = id(runtime_config)
        local_date = now.astimezone(self._timezone()).date().isoformat()
        with self._lock:
            state = self._state_by_runtime.setdefault(runtime_id, _ContactState())
            if state.session_started_at is None:
                self._contact_count_today = (
                    1 if self._last_contact_local_date != local_date else self._contact_count_today + 1
                )
                state.contact_ordinal_today = self._contact_count_today
            state.contact_count_today = state.contact_ordinal_today
            state.last_contact_local_date = local_date
            state.last_contact_at = now
            state.session_started_at = state.session_started_at or now
            state.last_prompt_hash = str(hash(prompt))
            self._last_contact_local_date = local_date
            self._last_contact_at = now

    def _read_module(self, relative_path: str) -> str:
        path = self.modules_dir / relative_path
        text = path.read_text(encoding="utf-8").strip()
        if self._module_cache.get(relative_path) != text:
            self._module_cache[relative_path] = text
        return text

    def _build_session_frame(self, context: dict[str, object]) -> str:
        minutes_since = context["minutes_since_last_contact"]
        elapsed = "first contact" if minutes_since is None else f"{minutes_since}m since contact"
        return (
            f"# Now\n\n{context['part_of_day']}, {context['local_time_iso']}; "
            f"contact {context['contact_ordinal_today']} today; {elapsed}; "
            f"{str(context['continuity']).replace('_', ' ')}."
        )

    def _normalize_client_instructions(self, instructions: str | None) -> str:
        if not instructions:
            return ""
        normalized = instructions.strip()
        return "" if not normalized or normalized == DEFAULT_CLIENT_SESSION_INSTRUCTIONS else normalized

    @staticmethod
    def _normalize_evidence(value: str) -> str:
        return re.sub(r"\s+", " ", value.lower()).strip()

    @classmethod
    def _immediate_prior_context(cls, recent_context: str, current_turn: str) -> str:
        lines = [cls._normalize_evidence(line) for line in recent_context.splitlines() if line.strip()]
        current = cls._normalize_evidence(current_turn)
        if lines and current and lines[-1].removeprefix("user: ").strip() == current:
            lines.pop()
        return " ".join(lines[-2:])

    @staticmethod
    def _contains_term(value: str, terms: set[str]) -> bool:
        return any(re.search(rf"\b{re.escape(term)}\b", value) for term in terms)

    @staticmethod
    def _estimate_tokens(prompt: str) -> int:
        return max(1, round(len(prompt) / 4))

    @staticmethod
    def _choice(value: str, allowed: set[str], fallback: str) -> str:
        normalized = value.strip().lower()
        if normalized not in allowed:
            logger.warning("Invalid Sophie prompt option %r; using %s", value, fallback)
            return fallback
        return normalized

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _part_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        return "evening"

    def _timezone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            logger.warning("Unknown SOPHIE_USER_TIMEZONE=%s; falling back to UTC", self.timezone_name)
            return ZoneInfo("UTC")

    def _now(self) -> datetime:
        return datetime.now(tz=ZoneInfo("UTC"))
