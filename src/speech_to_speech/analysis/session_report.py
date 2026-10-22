"""End-of-session analysis for a finished Bloom call.

Runs once, in a background thread after the call has ended, so it can never touch the live audio path. Two cheap
NanoGPT calls read the transcript: one writes the gratitude card and a descriptive reflection profile, the other
proposes updates to what we know about this person. The finished report (never the transcript) is posted to the
product host, which stores it and merges the memory. Best effort throughout; a failure only means no report.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ENABLED_ENV = "SESSION_REPORT_ENABLED"
NANO_BASE_URL = "https://nano-gpt.com/api/v1"
MODEL_CHAIN = ["deepseek/deepseek-v4-flash", "qwen/qwen3.8-27b", "moonshotai/kimi-k2.6"]
ATTEMPTS_PER_MODEL = 2
CALL_TIMEOUT_S = 75.0
OPENER_MARKERS = ("(i've just opened the session",)
MIN_USER_TURNS = 1
PROMPT_VERSION = "report-v4"
RULE_VERSION = "none"
MAX_ITEMS = 6

CARD_SYSTEM = """You write a private record for a gratitude companion app. You are given a spoken conversation between a person ("user") and Bloom ("assistant"). Reply with one JSON object and nothing else.

{"card": {"headline": "3-6 word title", "moment": "1-2 sentences in the person's own first-person voice", "theme": "one or two words", "tags": ["up to 4 short lowercase tags"]},
 "profile": {"specificity": "general|some|specific", "people_focus": "none|mentioned|central", "feelings": "none|named|explored", "why": "none|implied|stated", "sensory": "none|some|rich", "novelty": "routine|mixed|fresh", "expression": "brief|moderate|expansive", "meaning_signal": "low|medium|high", "entry_mood": "one or two words", "exit_mood": "one or two words", "reading": "one plain sentence describing how this person reflected"}}

Rules for the card: "moment" is one or two clean sentences that condense what they described, in their own words and first person, with the filler and repetition taken out (not a copy of their answer). Use only what the person said; no embellishment, no advice, no therapy language. If no good or grateful moment came up, set "moment" to null and "headline" to a plain description of what they talked about. Keep their wording where you can.
Rules for the profile: this describes a style, it does not grade a person. Short, plain answers are not worse than long, vivid ones: someone who says little can still mean a lot, so judge "meaning_signal" from whether they chose to stay on something, named a person, or said why it mattered, not from how many words or sensory details they used. "expression" is only about how much they said. Never use the words good, bad, poor or weak about the person."""

MEMORY_SYSTEM = """You keep a short memory of what matters to a person, from a spoken conversation between them ("user") and a gratitude companion ("assistant"). You are given what we already know and today's transcript. Reply with one JSON object and nothing else, listing only NEW or CHANGED items:

{"people": [{"ref": "existing id if this is someone already known", "name": "", "relation": "", "meaning": "what they are to this person", "sensitivity": "nominal|sensitive", "basis": "paraphrase, max 20 words"}],
 "themes": [{"ref": "existing theme id if already known", "label": "2-5 words, a noun phrase, e.g. quiet time outdoors", "sensitivity": "nominal|sensitive", "basis": ""}],
 "facts": [{"ref": "existing fact id if this is something already known", "fact": "", "category": "health|work|home|plans|habit|other", "sensitivity": "nominal|sensitive", "expiresInDays": 14, "recurring": false, "basis": ""}],
 "threads": [{"ref": "existing thread id if updating", "topic": "", "status": "open|followed_up|resolved", "notBeforeDays": 0, "expiresInDays": 7, "suggestedCheckin": "a natural, light question", "sensitivity": "nominal|sensitive", "basis": ""}],
 "style": {"notes": ["short phrase about HOW they talk"]}}

Rules: record only what the person actually said; if something they say matches a fact or theme we already know (same thing, different words), return it with that id in "ref" instead of adding a new one. Set "recurring" to true when they say it is regular ("every day", "I always", "most mornings") or it is already known. Style notes are only about how they talk (short or long answers, pace, humour, directness), never about what they enjoy or how they respond to you, and at most one per session; leave it out if unsure. never infer a diagnosis or a motive. Reuse "ref" ids from what we already know instead of creating duplicates. Use "sensitive" only for things that are private or painful: illness or diagnoses, mental health, grief, abuse, family conflict, money trouble, sexuality. Minor injuries, work events, plans and everyday worries are "nominal". A "thread" is something coming up or unresolved that a friend would ask about later; notBeforeDays is the number of days from today until the event has happened (work it out from today's date and weekday, e.g. an interview "on Tuesday" said on a Thursday is 5; use 0 only if it is unknown or already underway), expiresInDays is how many days from today asking would feel stale. Only return an existing thread (with its ref) if the person talked about it in THIS conversation: silence means nothing, never mark something resolved because it was not mentioned. If they said a known thread is finished, return its ref with status "resolved". "basis" is your own paraphrase, never a quote. At most 6 items per list. Use empty lists when there is nothing."""


# What a session's card and memory should capture depends on which session it was. Appended to the two prompts.
CARD_FLOW_NOTES = {
    "morning": 'This was a MORNING session about what they are looking forward to today. "moment" is what they said they look forward to, in first person ("I\'m looking forward to..."); the headline names it. If they named nothing, "moment" is null.',
    "ten_things": 'In this session they named things they like about themselves. "moment" lists up to four of them in first person ("I like that I..."); the headline is "What I like about me".',
    "daytime": 'This was a short drop-in during the day. "moment" is what they caught, if anything; otherwise null.',
    "onboarding": 'This was a first welcome conversation. No good moment is expected: set "moment" to null unless they shared one, and make the headline what brought them here.',
}
MEMORY_FLOW_NOTES = {
    "morning": 'This was a morning session. Record what they said they are looking forward to as ONE thread: topic, "notBeforeHour": 18, "expiresInDays": 1, a light "suggestedCheckin" such as "How did that go?", sensitivity "nominal". If they named nothing, add no thread.',
    "ten_things": 'In this session they named things they like about themselves. Record each one as a fact with category "self_regard", "recurring": true, sensitivity "nominal" and a paraphrased basis. Up to ten. Reuse "ref" for ones we already know.',
    "onboarding": 'This was a welcome conversation. Record why they came and what they hope for as themes. Add no threads.',
    "evening": 'If something we already know, such as what they were looking forward to this morning, came up, return it with its "ref" and status "followed_up" or "resolved".',
}
FLOW_MAX_ITEMS = {"ten_things": 10}


def enabled() -> bool:
    return os.getenv(ENABLED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _clean_transcript(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in messages:
        text = str(m.get("content") or "").strip()
        role = "Person" if m.get("role") == "user" else "Bloom"
        if not text or (m.get("role") == "user" and text.lower().startswith(OPENER_MARKERS)):
            continue
        out.append((role, text))
    return out


def _transcript_text(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for role, text in turns)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model reply, tolerating code fences and chatter around it."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:i + 1])
                        return value if isinstance(value, dict) else None
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def nano_json(system: str, user: str, *, api_key: str, client: httpx.Client | None = None,
              models: list[str] | None = None) -> tuple[dict[str, Any], str, int, int] | None:
    """(parsed object, model used, tokens in, tokens out), trying each model a couple of times; None if all fail."""
    http = client or httpx.Client(timeout=CALL_TIMEOUT_S)
    for model in models or MODEL_CHAIN:
        for attempt in range(ATTEMPTS_PER_MODEL):
            try:
                r = http.post(
                    f"{NANO_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "temperature": 0.2,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                    timeout=CALL_TIMEOUT_S,
                )
                r.raise_for_status()
                body = r.json()
                parsed = parse_json_object(body["choices"][0]["message"]["content"])
                if parsed is not None:
                    usage = body.get("usage") or {}
                    return parsed, model, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                logger.warning("session report: %s returned no JSON (attempt %d)", model, attempt + 1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("session report: %s failed (attempt %d): %s", model, attempt + 1, type(exc).__name__)
                time.sleep(0.5)
    return None


def _str(v: Any, limit: int = 200) -> str:
    return v.strip()[:limit] if isinstance(v, str) else ""


def sanitize_delta(raw: Any, max_items: int = MAX_ITEMS) -> dict[str, Any]:
    """Keep only well-formed items in the shape the host accepts."""
    if not isinstance(raw, dict):
        return {}
    sens = lambda v: "sensitive" if v == "sensitive" else "nominal"  # noqa: E731
    num = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else None  # noqa: E731
    out: dict[str, Any] = {}
    people = []
    for p in (raw.get("people") or [])[:max_items]:
        if isinstance(p, dict) and _str(p.get("name")):
            people.append({k: v for k, v in {"ref": _str(p.get("ref"), 80), "name": _str(p.get("name"), 60),
                           "relation": _str(p.get("relation"), 60), "meaning": _str(p.get("meaning")),
                           "sensitivity": sens(p.get("sensitivity")), "basis": _str(p.get("basis"))}.items() if v})
    themes = [{k: v for k, v in {"ref": _str(t.get("ref"), 80), "label": _str(t.get("label"), 60),
                                 "sensitivity": sens(t.get("sensitivity")), "basis": _str(t.get("basis"))}.items() if v}
              for t in (raw.get("themes") or [])[:max_items] if isinstance(t, dict) and _str(t.get("label"))]
    facts = []
    for f in (raw.get("facts") or [])[:max_items]:
        if isinstance(f, dict) and _str(f.get("fact")):
            item = {"fact": _str(f.get("fact")), "category": _str(f.get("category"), 30) or "other",
                    "sensitivity": sens(f.get("sensitivity")), "basis": _str(f.get("basis"))}
            if _str(f.get("ref"), 80):
                item["ref"] = _str(f.get("ref"), 80)
            if f.get("recurring") is True:
                item["recurring"] = True
            if num(f.get("expiresInDays")):
                item["expiresInDays"] = num(f.get("expiresInDays"))
            facts.append(item)
    threads = []
    for t in (raw.get("threads") or [])[:max_items]:
        if isinstance(t, dict) and (_str(t.get("topic")) or _str(t.get("ref"))):
            item = {"topic": _str(t.get("topic")), "sensitivity": sens(t.get("sensitivity")), "basis": _str(t.get("basis")),
                    "suggestedCheckin": _str(t.get("suggestedCheckin"))}
            if _str(t.get("ref"), 80):
                item["ref"] = _str(t.get("ref"), 80)
            if t.get("status") in {"open", "followed_up", "resolved"}:
                item["status"] = t["status"]
            for k in ("notBeforeDays", "expiresInDays"):
                if num(t.get(k)) is not None:
                    item[k] = num(t.get(k))
            hour = t.get("notBeforeHour")
            if isinstance(hour, (int, float)) and not isinstance(hour, bool) and 0 <= hour <= 23:
                item["notBeforeHour"] = int(hour)
            threads.append(item)
    notes = [_str(n, 160) for n in ((raw.get("style") or {}).get("notes") or [])[:1] if _str(n, 160)] if isinstance(raw.get("style"), dict) else []
    for key, val in (("people", people), ("themes", themes), ("facts", facts), ("threads", threads)):
        if val:
            out[key] = val
    if notes:
        out["style"] = {"notes": notes}
    return out


def _compact_memory(memory: dict[str, Any]) -> str:
    keep = lambda items, keys: [{k: i.get(k) for k in keys if i.get(k)} for i in (items or [])[-30:]]  # noqa: E731
    return json.dumps({
        "people": keep(memory.get("people"), ("id", "name", "relation", "meaning")),
        "themes": keep(memory.get("themes"), ("id", "label")),
        "facts": keep(memory.get("facts"), ("id", "fact", "category", "count", "habit")),
        "openThreads": keep([t for t in (memory.get("openThreads") or []) if t.get("status") in ("open", "followed_up")],
                            ("id", "topic", "status")),
    }, separators=(",", ":"))


def build_report(messages: list[dict[str, Any]], voice: dict[str, Any] | None, existing_memory: dict[str, Any],
                 *, api_key: str, local_date: str, duration_s: int | None, flow: str | None = None) -> dict[str, Any] | None:
    turns = _clean_transcript(messages)
    if sum(1 for role, _ in turns if role == "Person") < MIN_USER_TURNS:
        return None
    transcript = _transcript_text(turns)
    try:
        from datetime import date
        weekday = date.fromisoformat(local_date).strftime("%A")
    except ValueError:
        weekday = "unknown weekday"
    client = httpx.Client(timeout=CALL_TIMEOUT_S)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            card_system = CARD_SYSTEM + (f"\n\n{CARD_FLOW_NOTES[flow]}" if flow in CARD_FLOW_NOTES else "")
            memory_system = MEMORY_SYSTEM + (f"\n\n{MEMORY_FLOW_NOTES[flow]}" if flow in MEMORY_FLOW_NOTES else "")
            card_f = pool.submit(nano_json, card_system, f"Transcript:\n{transcript}", api_key=api_key, client=client)
            mem_f = pool.submit(nano_json, memory_system,
                                f"Today is {local_date} ({weekday}).\nWhat we already know:\n{_compact_memory(existing_memory)}\n\nTranscript:\n{transcript}",
                                api_key=api_key, client=client)
            card_r, mem_r = card_f.result(), mem_f.result()
    finally:
        client.close()
    if card_r is None and mem_r is None:
        return None
    report: dict[str, Any] = {"promptVersion": PROMPT_VERSION, "ruleVersion": RULE_VERSION, "models": {}}
    if flow:
        report["flow"] = flow
    if duration_s is not None:
        report["durationSeconds"] = duration_s
    if voice:
        report["voiceFeatures"] = voice
    tokens = {"in": 0, "out": 0}
    if card_r:
        parsed, model, tin, tout = card_r
        report["card"] = parsed.get("card") if isinstance(parsed.get("card"), dict) else None
        report["profile"] = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else None
        report["models"]["card"] = model
        tokens["in"] += tin; tokens["out"] += tout
    if mem_r:
        parsed, model, tin, tout = mem_r
        report["memoryDelta"] = sanitize_delta(parsed, FLOW_MAX_ITEMS.get(flow or "", MAX_ITEMS))
        known = {i.get("id") for key in ("people", "themes", "facts", "openThreads") for i in (existing_memory.get(key) or []) if isinstance(i, dict)}
        for key in ("people", "themes", "facts", "threads"):
            for item in report["memoryDelta"].get(key, []):
                if item.get("ref") not in known:
                    item.pop("ref", None)       # a ref the model made up points at nothing: treat the item as new
        if flow == "morning":
            # A morning intention is for the evening. The model often leaves the timing out, so it is not left to the model.
            for thread in report["memoryDelta"].get("threads", []):
                if "ref" not in thread:
                    thread["notBeforeHour"] = 18
                    thread["expiresInDays"] = 1.0
                    thread.pop("notBeforeDays", None)
        report["models"]["memory"] = model
        tokens["in"] += tin; tokens["out"] += tout
    report["models"]["tokens"] = tokens
    report["costUsd"] = 0.0  # NanoGPT subscription models
    return report


def run_and_post(*, host: str, token: str, conversation_id: str, messages: list[dict[str, Any]],
                 voice: dict[str, Any] | None, duration_s: int | None, local_date: str, flow: str | None = None) -> None:
    """Fetch what we know, analyse, post. Called in a daemon thread; logs and swallows every failure."""
    try:
        api_key = os.getenv("NANO_API_KEY", "").strip()
        if not api_key or not host:
            return
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        try:
            r = httpx.get(f"{host}/api/voice/memory", headers=headers, timeout=15.0)
            r.raise_for_status()
            existing = r.json().get("memory") or {}
        except Exception as exc:  # noqa: BLE001 - analyse without prior memory rather than lose the session
            logger.warning("session report: could not load memory: %s", exc)
            existing = {}
        report = build_report(messages, voice, existing, api_key=api_key, local_date=local_date, duration_s=duration_s, flow=flow)
        if report is None:
            logger.info("session report: nothing to report conv=%s", conversation_id)
            return
        report["conversationId"] = conversation_id
        resp = httpx.post(f"{host}/api/voice/session-report", headers=headers, json=report, timeout=20.0)
        resp.raise_for_status()
        logger.info("session report posted conv=%s %s", conversation_id, resp.text[:120])
    except Exception:  # noqa: BLE001
        logger.warning("session report failed conv=%s", conversation_id, exc_info=True)


def start(**kwargs: Any) -> None:
    if enabled():
        threading.Thread(target=run_and_post, kwargs=kwargs, name="session-report", daemon=True).start()
