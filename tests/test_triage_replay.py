"""The triage replay set and its runner (`eval/triage_replay/`).

The runner is driven with a fake model: no engine, no backend, no live ledger.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"triage_replay_{name}", ROOT / "eval" / "triage_replay" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("run")


def _case(key: str, label: str, expected: str, *, body="The retry loop double-sends email.") -> dict:
    return {"key": key, "item_id": 9001, "label": label, "expected": expected, "age_days": 3,
            "item": {"name": f"Item {key}", "priority": "medium", "created": "2026-09-01T00:00:00",
                     "tags": ["backlog"], "parent": 77, "worth": "", "size": "", "body": body}}


def _answer(verdict: str) -> str:
    return (f"The premise is checkable.\n\nVERDICT: {verdict}\nSURFACE: code\n"
            "CHECK: replay: text only\nEVIDENCE: Read from the item.\nACCEPTANCE: none\n"
            "ACCEPTANCE_CLAUSES: none\nHUMAN_CLAUSES: none\nSPAWNED: none\n")


# ── the tracked set ─────────────────────────────────────────────────────────

def test_the_tracked_set_has_forty_cases_each_with_a_label_its_evidence_and_as_of_text():
    doc = json.loads((ROOT / "eval" / "triage_replay" / "cases.json").read_text(encoding="utf-8"))
    cases = R.load_cases()
    assert len(cases) >= 40
    assert len({c["key"] for c in cases}) == len(cases)
    for c in cases:
        assert c["label"] in ("good_confirm", "bad_confirm")
        assert c["expected"] == ("confirmed" if c["label"] == "good_confirm" else "not_confirmed")
        assert c["decision"]["verdict"] == "confirmed" and c["decision"]["at"]
        assert c["outcome"]["evidence"].startswith("item_")       # a ledger row, not an opinion
        assert c["text_as_of"] in ("before_verdict", "earliest")
        assert len(c["vault_sha"]) == 40 and len(c["tree_as_of"]) == 40
        assert c["item"]["body"].strip() and c["item"]["name"]
    good = [c for c in cases if c["label"] == "good_confirm"]
    assert len(good) >= 40
    # a good confirm is only taken with the text triage actually saw
    assert all(c["text_as_of"] == "before_verdict" for c in good)
    # the set says how thin its other class is rather than hiding it
    assert doc["available"]["bad_confirm"] == sum(1 for c in cases if c["label"] == "bad_confirm")


def test_a_case_with_no_text_or_an_unknown_expectation_is_refused(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps({"cases": [_case("a", "good_confirm", "confirmed", body="  ")]}))
    with pytest.raises(ValueError, match="no item text"):
        R.load_cases(p)
    p.write_text(json.dumps({"cases": [_case("a", "good_confirm", "retired")]}))
    with pytest.raises(ValueError, match="expected must be"):
        R.load_cases(p)


# ── the prompt ──────────────────────────────────────────────────────────────

def test_the_prompt_is_the_production_one_with_the_as_of_item_and_the_replay_note(tmp_path):
    from workers.sources import autotriage
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("")
    case = _case("a", "good_confirm", "confirmed")
    prompt = R.build_prompt(case, ledger=ledger)
    cand = R.candidate_for(case)
    assert prompt == autotriage.render_prompt(cand, ledger=ledger) + R.REPLAY_NOTE
    assert "It was written 3 days ago" in prompt            # the age it had then, not today's
    assert '<item id="9001" status="draft" priority="medium">' in prompt
    assert "The retry loop double-sends email." in prompt
    assert "no tools are available" in prompt
    assert cand.parent is None                               # no lookup on the live board
    assert 'parent="' not in prompt


# ── the runner, with a fake model ───────────────────────────────────────────

def test_replay_grades_each_case_against_its_outcome_label():
    cases = [_case("g1", "good_confirm", "confirmed"), _case("g2", "good_confirm", "confirmed"),
             _case("b1", "bad_confirm", "not_confirmed"), _case("b2", "bad_confirm", "not_confirmed"),
             _case("x", "good_confirm", "confirmed"), _case("boom", "good_confirm", "confirmed")]
    seen: list[str] = []

    def fake(prompt: str) -> str:
        seen.append(prompt)
        if "Item g1" in prompt or "Item b1" in prompt:
            return _answer("confirmed")
        if "Item g2" in prompt or "Item b2" in prompt:
            return _answer("stale")
        if "Item boom" in prompt:
            raise RuntimeError("engine went away")
        return "I could not decide."

    recorded: list[dict] = []
    rows = R.replay(cases, fake, on_row=recorded.append)
    assert recorded == rows and len(seen) == 6
    by = {r["key"]: r for r in rows}
    assert (by["g1"]["verdict"], by["g1"]["agree"]) == ("confirmed", True)
    assert (by["g2"]["verdict"], by["g2"]["got"], by["g2"]["agree"]) == ("stale", "not_confirmed", False)
    assert by["b1"]["agree"] is False and by["b2"]["agree"] is True
    assert (by["x"]["got"], by["x"]["agree"]) == ("no_verdict", False)
    assert by["boom"]["error"].startswith("RuntimeError") and by["boom"]["agree"] is False

    s = R.summarize(rows, min_n=2)
    assert (s["cases"], s["errors"]) == (6, 1)
    assert s["by_label"]["good_confirm"]["agree"] == 1 and s["by_label"]["good_confirm"]["n"] == 3
    assert s["by_label"]["bad_confirm"]["rate"] == 0.5
    assert s["overall"]["n"] == 5 and s["overall"]["ci"][0] < 0.4 < s["overall"]["ci"][1]
    assert s["by_label"]["good_confirm"]["verdicts"] == {"confirmed": 1, "stale": 1, "no_verdict": 1}
    assert "| bad_confirm | 1/2 | 50%" in R.render(s)


def test_a_label_under_the_minimum_is_unmeasured_and_the_latest_row_per_case_wins():
    rows = [{"key": "g1", "label": "good_confirm", "agree": False, "verdict": "stale", "error": ""},
            {"key": "g1", "label": "good_confirm", "agree": True, "verdict": "confirmed", "error": ""}]
    s = R.summarize(rows)
    assert s["cases"] == 1 and s["by_label"]["good_confirm"]["agree"] == 1
    assert s["by_label"]["good_confirm"]["rate"] is None
    assert f"unmeasured (n < {R.MIN_N})" in R.render(s)


def test_grade_re_reads_recorded_rows_without_a_model(tmp_path, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": [_case("g1", "good_confirm", "confirmed")]}))
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"key": "g1", "label": "good_confirm", "expected": "confirmed",
                                "error": "", "text_tail": _answer("confirmed")}) + "\n"
                    + json.dumps({"key": "gone", "label": "good_confirm"}) + "\n")
    assert R.main(["grade", "--cases", str(cases), "--rows", str(rows)]) == 0
    assert "| good_confirm | 1/1 |" in capsys.readouterr().out


def test_plan_calls_no_model_and_run_has_no_default_endpoint(tmp_path, capsys, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": [_case("g1", "good_confirm", "confirmed")]}))
    monkeypatch.setattr(R, "endpoint_model", lambda *a, **k: pytest.fail("plan built an endpoint"))
    assert R.main(["plan", "--cases", str(cases)]) == 0
    assert "1 cases" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        R.main(["run", "--cases", str(cases)])              # --base-url is required


def test_the_endpoint_model_sends_one_message_and_no_tools(monkeypatch):
    sent: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "VERDICT: stale"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(R.urllib.request, "urlopen", fake_urlopen)
    assert R.endpoint_model("http://engine.invalid:1/")("the prompt") == "VERDICT: stale"
    assert sent["url"] == "http://engine.invalid:1/v1/chat/completions"
    assert "tools" not in sent["body"] and sent["body"]["temperature"] == 0
    assert sent["body"]["messages"] == [{"role": "user", "content": "the prompt"}]
