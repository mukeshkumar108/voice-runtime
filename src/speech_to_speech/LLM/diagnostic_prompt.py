"""Generic prompt for the direct-LLM diagnostic harness.

The direct backends (responses-api / chat-completions) exist only to isolate
transport faults from brain faults. They deliberately use this fixed,
product-free prompt instead of any companion prompt compiler: the harness
must never depend on product cognition to do its job.

The canonical path (companion-runtime backend) bypasses this module entirely.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

DIAGNOSTIC_DEFAULT_INSTRUCTIONS = (
    "You are a terse voice assistant used to test the speech pipeline. "
    "Reply conversationally in one or two short sentences."
)

UNCERTAINTY_NOTE = (
    "The speech recognizer marked the latest utterance as uncertain. "
    "If its meaning is not obvious, ask one brief clarification question "
    "instead of guessing."
)


def add_transcript_uncertainty_note(instructions: str | None, reason: str | None) -> str | None:
    """Append the uncertainty note when STT flagged the turn. Same shape as the old overlay."""
    if not reason:
        return instructions
    logger.info("diagnostic.transcript_uncertainty reason=%s", reason)
    return "\n\n".join(part for part in ((instructions or "").strip(), UNCERTAINTY_NOTE) if part)


class DiagnosticPrompt:
    """Fixed prompt with the compiler's call shape, minus product modules."""

    def compile(
        self,
        runtime_config: object,
        instructions: Optional[str],
        current_turn: str = "",
        recent_context: str = "",
        history_token_estimate: int = 0,
    ) -> str:
        del runtime_config, current_turn, recent_context, history_token_estimate
        base = (instructions or "").strip()
        return base or DIAGNOSTIC_DEFAULT_INSTRUCTIONS
