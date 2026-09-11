"""The scorecard reads what is on disk and says what the loop is worth.

Every rate is None when it has no denominator. "0% failing" for a source
that never ran is the reading `/api/workers/health` was built to prevent, and
the same rule holds here: a loop that has not run is unmeasured, not perfect.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from scripts.automod import scorecard as SC


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


NOW = 1_800_000_000.0
DAY = 86400.0


def _ledger(tmp_path, events):
    p = tmp_path / "promotions.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def _ev(event_name, ts_offset_days, **kw):
    return {"event": event_name, "ts": NOW - ts_offset_days * DAY, **kw}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"; (r / "app").mkdir(parents=True); (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "l@l"); git(r, "config", "user.name", "lloyd")
    (r / "app" / "m.py").write_text("V = 1\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    return r


def _commit(repo, name, files: dict, when_days_ago: float):
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    date = str(int(NOW - when_days_ago * DAY))
    subprocess.run(["git", "-C", str(repo), "-c", f"user.name={name}", "-c", "user.email=x@x",
                    "commit", "-q", "-m", f"by {name}"],
                   env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin"},
                   check=False)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def test_an_empty_ledger_is_unmeasured_not_perfect(tmp_path, repo):
    row = SC.compute(since_days=7, ledger=_ledger(tmp_path, []), backlog_dir=tmp_path / "nope",
                     repo=repo, now=NOW)
    assert row["acceptance"]["hit_rate"] is None
    assert row["review"]["refusal_rate"] is None
    assert row["spawn"]["triage_ratio"] is None and row["spawn"]["implement_ratio"] is None
    assert row["spawn"]["self_spawned_open"] == {"count": 0, "oldest_days": 0, "bound_days": 30,
                                                 "over_bound": 0}
    assert row["human_touch"]["rate"] is None
    assert row["verdict_plumbing"]["regex_rate"] is None
    assert row["throughput"]["items_closed_per_day"] == 0.0
    assert row["rollbacks"]["true_positives"] is None, "a human judgment, never computed"
    text = SC.render(row)
    assert "—" in text and "0%" not in text.split("\n")[4]


def test_the_row_adds_up_a_realistic_week(tmp_path, repo):
    """One landed-and-met round, one landed-not-met, one sent back by review
    and fixed in turn, one escalated, a human commit touching a landed file,
    a regex fallback, a nameless deferral, a bare abort, a rollback."""
    lloyd_sha = _commit(repo, "lloyd", {"app/m.py": "V = 2\n", "tests/test_m.py": "def test_m():\n    assert V == 2 or True\n"}, 3)
    _commit(repo, "alan", {"app/m.py": "V = 3\n"}, 2)          # human touched the same file
    events = [
        # triage: 2 closed, 5 filed; one regex, one structured, one truncated
        _ev("backlog_triage", 6, item_id=1, verdict="stale", closed=True, spawned=[10, 11], verdict_source="structured"),
        _ev("backlog_triage", 6, item_id=2, verdict="confirmed", closed=False, spawned=[12, 13, 14],
            verdict_source="regex", structured_error="finalizer failed: output truncated at 1024 tokens",
            finalizer_tokens=1024),
        _ev("backlog_triage", 5, item_id=3, verdict="already_done", closed=True, spawned=[], verdict_source="structured",
            finalizer_tokens=300),
        # round A: landed, met, grader agreed on 2 of 2; no human touch
        _ev("backlog_implement", 4, item_id=2, phase="finished", round_id="SM_A", num_turns=50, spawned=[20],
            outcome={"acceptance": "met", "clause_outcomes": [{"clause": 1, "outcome": "met"}, {"clause": 2, "outcome": "met"}]}),
        _ev("review", 4, round_id="SM_A", ok=True, blocking=False, clauses=[{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "met"}],
            test_honesty=[], prechecks=[]),
        _ev("gate", 4, round_id="SM_A", rung="review", ok=True, seconds=90),
        _ev("gate", 4, round_id="SM_A", rung="tests", ok=True, seconds=70),
        _ev("promoted", 4, round_id="SM_A", commit="aaaa1111", changed_paths=["app/other.py"]),
        _ev("settled", 3.9, commit="aaaa1111"),
        _ev("item_landed", 3.9, item_id=2, closed=True, acceptance="met"),
        # round B: landed, author said 2 met, grader said 1 met; human touched app/m.py after
        _ev("backlog_implement", 3.5, item_id=4, phase="finished", round_id="SM_B", num_turns=70, spawned=[],
            outcome={"acceptance": "met", "clause_outcomes": [{"clause": 1, "outcome": "met"}, {"clause": 2, "outcome": "met"}]}),
        _ev("review", 3.5, round_id="SM_B", ok=True, blocking=False,
            clauses=[{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "partial"}],
            test_honesty=[{"file": "tests/test_m.py", "line": 2, "problem": "or True"}], prechecks=[]),
        _ev("gate", 3.5, round_id="SM_B", rung="review", ok=True, seconds=120),
        _ev("promoted", 3.1, round_id="SM_B", commit=lloyd_sha, changed_paths=["app/m.py", "tests/test_m.py"]),
        _ev("settled", 3.0, commit=lloyd_sha),
        # round C: review refused once, then passed in the same round (fixed in turn); a nameless deferral
        _ev("gate", 2.5, round_id="SM_C", rung="review", ok=False, review_retry=True, seconds=100),
        _ev("review", 2.5, round_id="SM_C", ok=True, blocking=True, kind="retry", clauses=[{"clause": 1, "verdict": "unmet"}],
            test_honesty=[], prechecks=[]),
        _ev("gate", 2.4, round_id="SM_C", rung="review", ok=True, seconds=95),
        _ev("backlog_implement", 2.3, item_id=5, phase="finished", round_id="SM_C", num_turns=90, spawned=[],
            outcome={"acceptance": "deferred", "deferred_to": [], "clause_outcomes": []}),
        _ev("promoted", 2.3, round_id="SM_C", commit="cccc3333", changed_paths=["app/z.py"]),
        _ev("rollback_succeeded", 2.2, commit="cccc3333", trigger="error_rate"),
        # round D: refused, escalated; aborted with no reason
        _ev("gate", 1.5, round_id="SM_D", rung="review", ok=False, review_retry=True, seconds=80),
        _ev("review", 1.5, round_id="SM_D", ok=True, blocking=True, kind="retry", clauses=[{"clause": 1, "verdict": "unmet"}],
            test_honesty=[], prechecks=[{"file": "t", "line": 1, "problem": "assert True"}]),
        _ev("backlog_implement", 1.4, item_id=6, phase="finished", round_id="SM_D", num_turns=40, spawned=[30]),
        _ev("review_escalated", 1.4, item_id=6, round_id="SM_D"),
        _ev("round_aborted", 1.4, round_id="SM_D"),
        # one grader outage
        _ev("gate", 1.0, round_id="SM_E", rung="review", ok=False, external_blocker=True, seconds=5),
        # outside the window: must not count
        _ev("rollback_succeeded", 20, commit="old", trigger="crash"),
    ]
    backlog = tmp_path / "backlog"; backlog.mkdir()
    (backlog / "4-stranded.md").write_text("---\n" + yaml.dump(
        {"status": "in_progress", "automod_landed": lloyd_sha, "board": "lloyd"}) + "---\n# s\n")
    (backlog / "2-fine.md").write_text("---\n" + yaml.dump(
        {"status": "done", "automod_landed": "aaaa1111", "board": "lloyd"}) + "---\n# f\n")

    row = SC.compute(since_days=7, ledger=_ledger(tmp_path, events), backlog_dir=backlog, repo=repo, now=NOW)

    # 1 — rounds A, B (met) and C (deferred → outcome present) landed; C was reverted so it is not landed.
    assert row["acceptance"] == {"landed": 2, "with_outcome": 2, "met": 2, "hit_rate": 1.0}
    # 2 — author 4 met, grader 3 met
    assert row["audit"] == {"rounds_compared": 2, "author_met": 4, "grader_met": 3, "delta": 0.75}
    # 3 — graded rounds A B C D E(unavailable counts as graded attempt? no: skipped=False so yes) ; refused C D
    assert row["review"]["rounds_refused"] == 2 and row["review"]["fixed_in_turn"] == 1
    assert row["review"]["escalated"] == 1 and row["review"]["grader_unavailable"] == 1
    assert row["review"]["refusal_rate"] == round(2 / 5, 3)
    # 4 — triage filed 5, closed 2; implement filed 2, closed 1
    assert row["spawn"]["triage_filed"] == 5 and row["spawn"]["triage_closed"] == 2
    assert row["spawn"]["triage_ratio"] == 2.5
    assert row["spawn"]["implement_filed"] == 2 and row["spawn"]["implement_closed"] == 1
    # 5 — B's app/m.py was touched by alan a day later
    assert row["human_touch"]["touched_within_7d"] == 1 and row["human_touch"]["rounds"] == ["SM_B"]
    assert row["human_touch"]["landed"] == 3
    # 6 — one grader finding (B) + one precheck (D); B's landed commit adds `or True`
    assert row["test_honesty"]["grader_findings"] == 2
    assert row["test_honesty"]["landed_with_or_true"] == 1
    # 7 — C's nameless deferral, D's bare abort, item 4 stranded in_progress after settle
    assert row["bookkeeping"] == {"nameless_deferrals": 1, "stranded_landings": 1, "bare_aborts": 1}
    # 8 — one regex of three sourced; one truncation; median finalizer tokens
    assert row["verdict_plumbing"]["regex"] == 1 and row["verdict_plumbing"]["regex_rate"] == round(1 / 3, 3)
    assert row["verdict_plumbing"]["truncated"] == 1
    assert row["verdict_plumbing"]["finalizer_tokens_median"] == 662.0
    # 9 — closed: 2 triage + 1 landed
    assert row["throughput"]["items_closed"] == 3
    assert row["throughput"]["rounds_landed"] == 2 and row["throughput"]["rounds_finished"] == 4
    assert row["throughput"]["median_turns_landed"] == 60.0
    # 10 — only the rollback inside the window
    assert row["rollbacks"] == {"count": 1, "triggers": ["error_rate"], "true_positives": None}

    text = SC.render(row)
    assert "100%" in text and "75%" in text and "stranded" in text
    out = SC.record(row, tmp_path / "scorecard.jsonl")
    assert json.loads(out.read_text().splitlines()[-1])["acceptance"]["met"] == 2


def test_since_parses_days_hours_and_weeks():
    assert SC.parse_since("7d") == 7 and SC.parse_since("36h") == 1.5 and SC.parse_since("2w") == 14
    assert SC.parse_since("3") == 3
    with pytest.raises(ValueError):
        SC.parse_since("soon")


def test_the_dashboard_section_is_the_same_row_cached(monkeypatch, tmp_path):
    from app.routers import dashboard as D
    from scripts.automod import state as S

    calls = []
    monkeypatch.setattr(SC, "compute", lambda **kw: calls.append(kw) or {"since_days": 7.0, "events": 0})
    monkeypatch.setattr(S, "read_current", lambda: {"round_id": "SM_X", "state": "observing"})
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: True)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    D._cache.pop("automod", None)
    first = D._automod()
    second = D._automod()
    assert first["current"] == {"round_id": "SM_X", "state": "observing"} and first["enabled"] is True
    assert second is first and len(calls) == 1, "cached: the row moves per round, not per poll"


def test_the_round_cli_has_the_subcommand():
    from scripts.automod import round as R
    import inspect
    src = inspect.getsource(R.main)
    assert 'sub.add_parser("scorecard"' in src and "SC.render" in src


def test_merges_appends_expiries_and_the_open_self_spawned_gauge(tmp_path, repo):
    """The inflow controls report beside the ratio they exist to lower, and
    the gauge says whether expiry is holding the bound: `over_bound` is the
    number of untriaged self-filed drafts older than 30 d, and should be 0."""
    ledger = _ledger(tmp_path, [
        _ev("backlog_triage", 6, item_id=1, verdict="stale", closed=True, spawned=[10], merged=[3]),
        _ev("backlog_implement", 4, item_id=2, phase="finished", round_id="SM_A", num_turns=50,
            spawned=[], merged=[10, 11], findings_appended=3),
        _ev("backlog_expired", 2, item_id=40, age_days=31),
        # An old triage outside the window still counts as judged for the gauge.
        _ev("backlog_triage", 40, item_id=52, verdict="unverifiable"),
    ])
    board = tmp_path / "backlog"; board.mkdir()
    def item(iid, *, days, status="draft", tags=("spawned-by-triage",)):
        created = datetime.fromtimestamp(NOW - days * DAY, tz=timezone.utc).isoformat()
        (board / f"{iid}-x.md").write_text(
            "---\n" + yaml.dump({"status": status, "created": created, "tags": list(tags),
                                  "board": "lloyd"}) + "---\n\n# x\n")
    item(50, days=2)                                   # fresh, held
    item(51, days=40)                                  # over the bound and unjudged: the defect
    item(52, days=40)                                  # over the bound but triaged
    item(53, days=40, tags=("spawned-by-autocode", "grouped"))   # exempt
    item(54, days=40, status="done", tags=("spawned-by-triage", "expired"))  # closed
    item(55, days=40, tags=("backlog",))               # a human's draft: not self-spawned
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=board, repo=repo, now=NOW)
    s = row["spawn"]
    assert s["triage_merged"] == 1 and s["implement_merged"] == 2
    assert s["findings_appended"] == 3 and s["expired"] == 1
    assert s["self_spawned_open"]["count"] == 4
    assert s["self_spawned_open"]["over_bound"] == 1
    assert s["self_spawned_open"]["oldest_days"] == 40.0
    text = SC.render(row)
    assert "merged 1+2, appended 3, expired 1" in text and "over bound 1" in text
