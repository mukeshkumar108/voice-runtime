import json

from speech_to_speech.analysis import session_report as sr


def test_parse_json_object_tolerates_fences_and_chatter():
    assert sr.parse_json_object('Sure!\n```json\n{"a": {"b": "x}"}}\n```') == {"a": {"b": "x}"}}
    assert sr.parse_json_object("no json here") is None
    assert sr.parse_json_object('{"broken": ') is None


def test_sanitize_delta_keeps_only_well_formed_items():
    delta = sr.sanitize_delta({
        "people": [{"name": "Sarah", "relation": "colleague", "sensitivity": "weird"}, {"name": ""}, "junk"],
        "facts": [{"fact": "Hurt knee", "category": "health", "sensitivity": "sensitive", "expiresInDays": 10}],
        "threads": [{"topic": "Interview", "status": "open", "notBeforeDays": 3, "expiresInDays": -1}, {"status": "open"}],
        "themes": [{"label": ""}],
        "style": {"notes": ["short answers", "", "x" * 400]},
    })
    assert delta["people"] == [{"name": "Sarah", "relation": "colleague", "sensitivity": "nominal"}]
    assert delta["facts"][0]["sensitivity"] == "sensitive" and delta["facts"][0]["expiresInDays"] == 10
    assert delta["threads"][0]["notBeforeDays"] == 3 and "expiresInDays" not in delta["threads"][0]
    assert len(delta["threads"]) == 1 and "themes" not in delta
    assert delta["style"]["notes"] == ["short answers"]
    assert sr.sanitize_delta("nonsense") == {}


def test_build_report_skips_the_opener_and_combines_both_calls(monkeypatch):
    seen = {}

    def fake(system, user, *, api_key, client=None, models=None):
        seen[system[:20]] = user
        if "private record" in system:
            return {"card": {"headline": "Sun and coffee", "moment": "I had coffee in the sun."}, "profile": {"expression": "brief"}}, "m1", 10, 5
        return {"people": [{"name": "Dad"}]}, "m2", 20, 7

    monkeypatch.setattr(sr, "nano_json", fake)
    msgs = [{"role": "user", "content": "(I've just opened the session.)"}, {"role": "assistant", "content": "Hey."},
            {"role": "user", "content": "I sat with my dad in the garden."}]
    report = sr.build_report(msgs, {"utterances": 1}, {}, api_key="k", local_date="2026-10-22", duration_s=120)
    assert report["card"]["headline"] == "Sun and coffee" and report["profile"] == {"expression": "brief"}
    assert report["memoryDelta"]["people"][0]["name"] == "Dad"
    assert report["models"]["tokens"] == {"in": 30, "out": 12} and report["voiceFeatures"] == {"utterances": 1}
    assert all("just opened the session" not in u for u in seen.values())


def test_build_report_needs_something_the_person_said(monkeypatch):
    monkeypatch.setattr(sr, "nano_json", lambda *a, **k: ({}, "m", 0, 0))
    assert sr.build_report([{"role": "assistant", "content": "Hey."}], None, {}, api_key="k", local_date="d", duration_s=None) is None


def test_build_report_survives_one_call_failing(monkeypatch):
    monkeypatch.setattr(sr, "nano_json", lambda system, *a, **k: None if "private record" in system else ({"people": []}, "m", 1, 1))
    report = sr.build_report([{"role": "user", "content": "hi there"}], None, {}, api_key="k", local_date="d", duration_s=None)
    assert report and "card" not in report and report["memoryDelta"] == {}
    assert json.dumps(report)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(sr.ENABLED_ENV, raising=False)
    assert sr.enabled() is False


def test_sanitize_keeps_refs_and_recurring_and_one_style_note():
    delta = sr.sanitize_delta({
        "facts": [{"ref": "fact_walk_20261022", "fact": "Walks daily", "recurring": True}, {"fact": "x", "recurring": "yes"}],
        "themes": [{"ref": "theme_nature", "label": "Time outdoors"}],
        "style": {"notes": ["short answers", "likes flattery"]},
    })
    assert delta["facts"][0]["ref"] == "fact_walk_20261022" and delta["facts"][0]["recurring"] is True
    assert "recurring" not in delta["facts"][1]
    assert delta["themes"][0]["ref"] == "theme_nature"
    assert delta["style"]["notes"] == ["short answers"]


def test_known_facts_are_shown_with_their_ids():
    text = sr._compact_memory({"facts": [{"id": "fact_walk_20261022", "fact": "Walks daily", "category": "habit", "count": 3, "habit": True}]})
    assert "fact_walk_20261022" in text and '"count":3' in text
