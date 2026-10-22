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


def test_morning_asks_for_one_evening_thread_and_ten_things_allows_ten_facts(monkeypatch):
    seen = {}

    def fake(system, user, *, api_key, client=None, models=None):
        seen["card" if "private record" in system else "memory"] = system
        if "private record" in system:
            return {"card": {"headline": "x", "moment": "y"}, "profile": {}}, "m", 1, 1
        return {"facts": [{"fact": f"quality {i}", "category": "self_regard", "recurring": True} for i in range(12)],
                "threads": [{"topic": "Dinner with Sam", "notBeforeHour": 18, "expiresInDays": 1}]}, "m", 1, 1

    monkeypatch.setattr(sr, "nano_json", fake)
    msgs = [{"role": "user", "content": "I like that I listen well."}]
    ten = sr.build_report(msgs, None, {}, api_key="k", local_date="2026-10-22", duration_s=60, flow="ten_things")
    assert len(ten["memoryDelta"]["facts"]) == 10 and "self_regard" in seen["memory"] and "like about themselves" in seen["card"]
    morning = sr.build_report(msgs, None, {}, api_key="k", local_date="2026-10-22", duration_s=60, flow="morning")
    assert len(morning["memoryDelta"]["facts"]) == sr.MAX_ITEMS
    assert morning["memoryDelta"]["threads"][0]["notBeforeHour"] == 18 and '"notBeforeHour": 18' in seen["memory"]
    assert morning["flow"] == "morning"


def test_notbeforehour_must_be_a_real_hour():
    d = sr.sanitize_delta({"threads": [{"topic": "a", "notBeforeHour": 25}, {"topic": "b", "notBeforeHour": 18}, {"topic": "c", "notBeforeHour": True}]})
    assert "notBeforeHour" not in d["threads"][0] and d["threads"][1]["notBeforeHour"] == 18 and "notBeforeHour" not in d["threads"][2]


def test_a_morning_thread_is_always_held_until_six_in_the_evening(monkeypatch):
    def fake(system, user, *, api_key, client=None, models=None):
        if "private record" in system:
            return {"card": {"headline": "x", "moment": "y"}, "profile": {}}, "m", 1, 1
        return {"threads": [{"topic": "Dinner with my sister", "notBeforeDays": 0, "expiresInDays": 6}, {"ref": "thread_old", "topic": "", "status": "resolved"}]}, "m", 1, 1

    monkeypatch.setattr(sr, "nano_json", fake)
    known = {"openThreads": [{"id": "thread_old", "topic": "Old thing", "status": "open"}]}
    r = sr.build_report([{"role": "user", "content": "looking forward to dinner"}], None, known, api_key="k", local_date="2026-10-22", duration_s=60, flow="morning")
    new, old = r["memoryDelta"]["threads"]
    assert new["notBeforeHour"] == 18 and new["expiresInDays"] == 1.0 and "notBeforeDays" not in new
    assert "notBeforeHour" not in old and old["status"] == "resolved"


def test_a_ref_the_model_invented_is_dropped_but_a_real_one_is_kept(monkeypatch):
    def fake(system, user, *, api_key, client=None, models=None):
        if "private record" in system:
            return {"card": {"headline": "x", "moment": "y"}, "profile": {}}, "m", 1, 1
        return {"facts": [{"ref": "t1", "fact": "new thing"}, {"ref": "fact_walk_20261022", "fact": "walks daily", "recurring": True}],
                "threads": [{"ref": "t1", "topic": "Dinner"}]}, "m", 1, 1

    monkeypatch.setattr(sr, "nano_json", fake)
    r = sr.build_report([{"role": "user", "content": "hi"}], None, {"facts": [{"id": "fact_walk_20261022", "fact": "walks"}]},
                        api_key="k", local_date="2026-10-22", duration_s=1, flow="morning")
    facts = r["memoryDelta"]["facts"]
    assert "ref" not in facts[0] and facts[1]["ref"] == "fact_walk_20261022"
    assert "ref" not in r["memoryDelta"]["threads"][0] and r["memoryDelta"]["threads"][0]["notBeforeHour"] == 18
