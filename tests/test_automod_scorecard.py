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

from scripts.automod.backlog import spawn_expiry_days
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
    assert row["spawn"]["self_spawned_open"] == {"count": 0, "oldest_days": 0,
                                                 "bound_days": spawn_expiry_days(),
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
    coverage = row["rollbacks"].pop("regression_coverage")
    assert row["rollbacks"] == {"count": 1, "triggers": ["error_rate"], "true_positives": None}
    assert coverage["measured"] <= coverage["promotions"], coverage

    text = SC.render(row)
    assert "100%" in text and "75%" in text and "stranded" in text
    out = SC.record(row, tmp_path / "scorecard.jsonl")
    assert json.loads(out.read_text().splitlines()[-1])["acceptance"]["met"] == 2


def test_row_9_counts_the_two_ways_a_gated_change_used_to_be_lost(tmp_path, repo):
    """A gate that passed after its turn ended (`land_rescued`, the reaper
    lands it) and an item verdict from a turn that was landing
    (`outcome_refused`, no longer taken). Both were found by reading
    transcripts on 2026-09-18; they are numbers now."""
    events = [
        _ev("land_rescued", 1, round_id="SM_A", item_id=1),
        _ev("land_rescued", 2, round_id="SM_B", item_id=2),
        _ev("land_rescued", 30, round_id="SM_OLD", item_id=9),      # outside the window
        _ev("backlog_implement", 1, item_id=3, phase="finished", round_id="SM_C",
            outcome_refused="`rejected` with no measurement in `summary`"),
        _ev("backlog_implement", 1, item_id=4, phase="finished", round_id="SM_D",
            outcome_refused=""),
    ]
    row = SC.compute(since_days=7, ledger=_ledger(tmp_path, events), backlog_dir=tmp_path / "nope",
                     repo=repo, now=NOW)
    assert row["throughput"]["landings_rescued"] == 2
    assert row["throughput"]["item_verdicts_refused"] == 1
    text = SC.render(row)
    assert "2 landed by the reaper after their turn ended" in text
    assert "1 item verdict(s) not taken from a landing round" in text


def test_a_rejection_whose_change_then_landed_is_not_counted_as_one(tmp_path, repo):
    """#1242 and #1053 (2026-09-18): closed `rejected` by a finalizer that
    misreported a landing round, minutes before their changes promoted. The
    rows stay on the ledger; the landing recorded afterwards is what says they
    were never rejections. A real one — nothing landed — still counts."""
    events = [
        _ev("item_closed", 2.0, item_id=1242, by="autocode", acceptance="rejected"),
        _ev("item_landed", 1.9, item_id=1242, commit="e2fc0754", closed=True, acceptance="met"),
        _ev("item_closed", 1.0, item_id=726, by="autocode", acceptance="rejected",
            reason="MRR 0.500 -> 0.497; within noise"),
        # A landing BEFORE the close is an earlier round's, not a contradiction.
        _ev("item_landed", 3.0, item_id=800, commit="aaaa1111", closed=False, acceptance="not_met"),
        _ev("item_closed", 0.5, item_id=800, by="autocode", acceptance="rejected", reason="no gain"),
    ]
    row = SC.compute(since_days=7, ledger=_ledger(tmp_path, events), backlog_dir=tmp_path / "nope",
                     repo=repo, now=NOW)
    assert row["throughput"]["rounds_rejected"] == 2, "726 and 800; 1242 landed"


def test_row_10_says_how_much_of_what_landed_was_measured(tmp_path, repo):
    """A detector that is not running reads exactly like one that finds
    nothing. 2026-09-18: 8 of 17 promotions measured, and the row said only
    how many rollbacks there had been."""
    events = [_ev("promoted", 2.0, round_id=f"SM_{c}", commit=c * 40) for c in "abcd"]
    found = {"doc_hit_rate": {"before": 1.0, "after": 1.0, "delta": 0.0}}
    nothing = {"doc_hit_rate": {"before": 0.0, "after": 0.0, "delta": 0.0}}
    events += [_ev("regression_check", 1.9, commit="a" * 40, regressed=False, detail=found),
               _ev("regression_check", 1.8, commit="b" * 40, regressed=False),      # an old row: believed
               _ev("regression_check", 1.7, commit="z" * 40, regressed=False),      # not a promotion here
               _ev("regression_skipped", 1.6, commit="c" * 40, reason="pinned corpus unavailable"),
               # "no regression", on the ledger, from a daemon that answered
               # neither arm: five of these on 2026-09-18.
               _ev("regression_check", 1.5, commit="d" * 40, regressed=False, detail=nothing)]
    row = SC.compute(since_days=7, ledger=_ledger(tmp_path, events), backlog_dir=tmp_path / "nope",
                     repo=repo, now=NOW)
    assert row["rollbacks"]["regression_coverage"] == {"promotions": 4, "measured": 2,
                                                       "could_not_evaluate": 1, "compared_nothing": 1}
    assert ("regression check measured 2 of 4 promotions (1 could not be evaluated, "
            "1 compared nothing with nothing)") in SC.render(row)


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


def test_the_open_gauge_counts_a_mint_named_after_its_own_session(tmp_path):
    """The gauge asked its own question of the tag set instead of going through
    the loop's one predicate, so it saw only the enumerated mints. Measured on
    the live board on 2026-09-17: 412 open items carried a `spawned-by-*` tag and
    the gauge reported 369, because 43 of them named the session that filed them
    (`spawned-by-task-24`, `spawned-by-data-pipeline`) rather than one of the
    six names somebody thought of in advance. The row is the loop's measure of
    its own inflow, so a 10 % undercount there reads as inflow the loop is not
    producing (#1160).

    Enumerated and un-enumerated mints must land in the same number: the gauge
    bounds the board, and it cannot bound what it does not count.
    """
    board = tmp_path / "backlog"
    board.mkdir()

    def item(iid, *, days=2, tags):
        created = datetime.fromtimestamp(NOW - days * DAY, tz=timezone.utc).isoformat()
        (board / f"{iid}-x.md").write_text(
            "---\n" + yaml.dump({"status": "draft", "created": created, "tags": list(tags),
                                  "board": "lloyd"}) + "---\n\n# x\n")

    item(60, tags=("spawned-by-triage",))                # enumerated: counted before this fix
    item(61, tags=("spawned-by-review",))                # enumerated, and NOT quarantined
    item(62, tags=("spawned-by-anything-42",))           # the hole: counted only via the prefix
    item(63, tags=("spawned-by-data-pipeline",))         # a real minter, 6 open items on 09-17
    item(64, tags=("backlog",))                          # a human's draft: still not loop output
    gauge = SC._self_spawned_gauge([], board, now=NOW)
    assert gauge["count"] == 4, "the four loop filings, the human draft excluded"
    assert gauge["bound_days"] == spawn_expiry_days()


def test_the_grouping_section_adds_up_a_realistic_week(tmp_path, repo):
    ledger = _ledger(tmp_path, [
        _ev("backlog_cluster", 6, clusters=12, items=71),
        _ev("backlog_cluster", 1, clusters=15, items=63),
        _ev("backlog_group_triage", 5, cluster_id="c-1", judged={"1": "duplicate_of", "2": "fold", "3": "fold"},
            duplicates=1, retired=0, folded=2, kept=0, umbrella_id=50),
        _ev("backlog_group_triage", 4, cluster_id="c-2", judged={"4": "keep", "5": "stale"},
            duplicates=0, retired=1, folded=0, kept=1, umbrella_id=None),
        _ev("backlog_group_triage", 3, cluster_id="c-3", verdict="incomplete", judged={}),
        _ev("backlog_triage", 5, item_id=1, verdict="stale", closed=True, spawned=[]),
        _ev("backlog_triage", 5, item_id=2, verdict="folded", closed=False, spawned=[]),
        _ev("backlog_triage", 5, item_id=3, verdict="folded", closed=False, spawned=[]),
        _ev("backlog_triage", 5, item_id=50, verdict="confirmed", umbrella=True, members=[2, 3], spawned=[]),
        _ev("item_landed", 2, item_id=50, closed=True, acceptance="met"),
        _ev("item_closed", 2, item_id=2, by="umbrella", umbrella_id=50),
        _ev("item_closed", 2, item_id=3, by="umbrella", umbrella_id=50),
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "nope", repo=repo, now=NOW)
    g = row["grouping"]
    assert g["cluster_runs"] == 2 and g["clusters_formed"] == 15 and g["items_clustered"] == 63
    assert g["group_triages"] == 2, "an incomplete run judged nothing"
    assert g["duplicates_closed"] == 1 and g["retired_in_group"] == 1 and g["folded"] == 2 and g["kept"] == 1
    assert g["umbrellas_formed"] == 1 and g["umbrellas_landed"] == 1 and g["members_closed"] == 2
    assert g["members_per_landing"] == 2.0
    # Folds and duplicates count as triage closures on row 4.
    assert row["spawn"]["triage_closed"] == 3
    assert "| 11 | grouping | 2 group triages" in SC.render(row)


def test_row_12_adds_up_a_week_of_architecture_reviews(tmp_path, repo):
    """`rejected` is the number worth watching: it counts turns whose doc edit
    was thrown away for breaking a bound, which is the only way this job can
    spend a whole session and produce nothing. A rate that stops being near
    zero means a bound is wrong, not that the model is."""
    ledger = _ledger(tmp_path, [
        _ev("arch_review", 6, unit="doc:memory", kind="doc", verdict="current",
            doc_updated=True, commit="a" * 40, filed=[900, 901], merged=[], appended_to=[],
            stray_writes=[]),
        _ev("arch_review", 5, unit="doc:voice", kind="doc", verdict="stale",
            doc_updated=False, commit="", doc_update_rejected="diff is 900 changed lines, cap 400",
            filed=[902], merged=[500], appended_to=[41],
            stray_writes=[{"repo": "lloyd", "path": "README.md", "action": "checkout"}]),
        _ev("arch_review", 3, unit="group:workers-jobs:Mining", kind="group", verdict="current",
            grouping="split", doc_updated=True, commit="b" * 40, filed=[], merged=[],
            appended_to=[], stray_writes=[]),
        # Outside the window: it must not be counted.
        _ev("arch_review", 30, unit="doc:tools", kind="doc", verdict="current",
            doc_updated=True, commit="c" * 40, filed=[1], merged=[], appended_to=[]),
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "nope",
                     repo=repo, now=NOW)
    ar = row["arch_review"]
    assert ar["reviewed"] == 3
    assert ar["by_kind"] == {"doc": 2, "group": 1}
    assert ar["updated"] == 2 and ar["committed"] == 2 and ar["rejected"] == 1
    assert ar["filed"] == 3 and ar["merged"] == 1 and ar["appended_to"] == 1
    assert ar["stray_writes"] == 1
    assert ar["by_status"] == {"current": 2, "stale": 1}
    text = SC.render(row)
    assert "| 12 | arch review | 3 units |" in text
    assert "2 docs, 1 groups" in text and "2 doc edits committed, 1 rejected" in text


def test_row_12_renders_for_a_row_recorded_before_it_existed():
    """`scorecard.jsonl` is append-only and the trend is the point, so every
    field on this row is read with a default."""
    old = {"since_days": 7, "events": 0, "computed_at": "2026-09-01T00:00:00+00:00",
           "acceptance": {"landed": 0, "with_outcome": 0, "met": 0, "hit_rate": None},
           "audit": {"delta": None, "grader_met": 0, "author_met": 0, "rounds_compared": 0},
           "review": {"refusal_rate": None, "rounds_refused": 0, "rounds_graded": 0,
                      "fixed_in_turn": 0, "premise_unsound": 0, "escalated": 0,
                      "grader_unavailable": 0},
           "spawn": {"triage_ratio": None, "implement_ratio": None, "triage_filed": 0,
                     "triage_closed": 0, "implement_filed": 0, "implement_closed": 0},
           "human_touch": {"rate": None, "touched_within_7d": 0, "landed": 0},
           "test_honesty": {"grader_findings": 0, "per_gated_round": None,
                            "landed_with_or_true": 0},
           "bookkeeping": {"nameless_deferrals": 0, "stranded_landings": 0, "bare_aborts": 0},
           "verdict_plumbing": {"regex_rate": None, "regex": 0, "verdicts_with_source": 0,
                                "truncated": 0, "finalizer_tokens_median": None},
           "throughput": {"items_closed": 0, "items_closed_per_day": 0.0, "rounds_finished": 0,
                          "rounds_landed": 0, "median_turns_landed": None,
                          "median_gate_seconds": None},
           "rollbacks": {"count": 0, "triggers": [], "true_positives": None}}
    text = SC.render(old)
    assert "| 12 | arch review | 0 units |" in text
    assert "no verdicts" in text
    # Row 14 gained a landing-wait breakdown on 2026-09-20; a row older than it
    # carries no `waits` key and must say so rather than render a zero it never
    # measured — the `fail_rate` null rule, one panel over.
    assert "landings waited: unrecorded" in text


def test_triage_appends_are_counted_apart_and_row_13_reads_the_board(tmp_path, repo):
    ledger = _ledger(tmp_path, [
        _ev("backlog_triage", 2, item_id=1, verdict="confirmed", spawned=[], findings_appended=2),
        _ev("backlog_implement", 1, item_id=2, phase="finished", round_id="SM_A",
            spawned=[], findings_appended=3),
    ])
    board = tmp_path / "backlog"; board.mkdir()
    def item(iid, *, hours, status="draft", completed_hours=None):
        fm = {"status": status, "board": "lloyd",
              "created": datetime.fromtimestamp(NOW - hours * 3600, tz=timezone.utc).isoformat()}
        if completed_hours is not None:
            fm["completed"] = datetime.fromtimestamp(NOW - completed_hours * 3600,
                                                     tz=timezone.utc).isoformat()
        (board / f"{iid}-x.md").write_text("---\n" + yaml.dump(fm) + "---\n\n# x\n")
    item(10, hours=2)
    item(11, hours=3)
    item(12, hours=100, status="done", completed_hours=1)
    # `board_flow` skips closed files untouched for a week by mtime; these are
    # written now, and NOW is in the future, so pin the clock the test reads.
    import os
    for f in board.glob("*.md"):
        os.utime(f, (NOW, NOW))
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=board, repo=repo, now=NOW)
    assert row["spawn"]["triage_findings_appended"] == 2
    assert row["spawn"]["findings_appended"] == 3, "implement's key stays implement's"
    assert row["flow"]["24h"] == {"created": 2, "closed": 1, "net": 1}
    text = SC.render(row)
    assert "| 13 | board net flow | +1 / 24 h |" in text
    assert "triage appended 2 findings" in text


def test_row_13_renders_for_a_row_recorded_before_it_existed():
    old = {"since_days": 7, "events": 0, "computed_at": "2026-09-01T00:00:00+00:00",
           "acceptance": {"landed": 0, "with_outcome": 0, "met": 0, "hit_rate": None},
           "audit": {"delta": None, "grader_met": 0, "author_met": 0, "rounds_compared": 0},
           "review": {"refusal_rate": None, "rounds_refused": 0, "rounds_graded": 0,
                      "fixed_in_turn": 0, "premise_unsound": 0, "escalated": 0,
                      "grader_unavailable": 0},
           "spawn": {"triage_ratio": None, "implement_ratio": None, "triage_filed": 0,
                     "triage_closed": 0, "implement_filed": 0, "implement_closed": 0},
           "human_touch": {"rate": None, "touched_within_7d": 0, "landed": 0},
           "test_honesty": {"grader_findings": 0, "per_gated_round": None,
                            "landed_with_or_true": 0},
           "bookkeeping": {"nameless_deferrals": 0, "stranded_landings": 0, "bare_aborts": 0},
           "verdict_plumbing": {"regex_rate": None, "regex": 0, "verdicts_with_source": 0,
                                "truncated": 0, "finalizer_tokens_median": None},
           "throughput": {"items_closed": 0, "items_closed_per_day": 0.0, "rounds_finished": 0,
                          "rounds_landed": 0, "median_turns_landed": None,
                          "median_gate_seconds": None},
           "rollbacks": {"count": 0, "triggers": [], "true_positives": None}}
    assert "| 13 | board net flow | — |" in SC.render(old)
    assert "| 14 | autocode duty cycle | — |" in SC.render(old)


def test_row_14_measures_turn_coverage_and_names_what_each_gap_waited_on(tmp_path, repo):
    """Alan's rule: an autocoder round runs 100% of the time. The row says
    how close the loop is and which kind of gap costs the most."""
    H = 1 / 24
    ledger = _ledger(tmp_path, [
        # turn A: 23 h ago for 2 h, then promoted; 30 min of landing gap
        _ev("backlog_implement", 23 * H, item_id=1, phase="started", round_id="SM_A"),
        _ev("backlog_implement", 21 * H, item_id=1, phase="finished", round_id="SM_A"),
        _ev("promoted", 20.9 * H, round_id="SM_A", commit="a" * 40),
        # turn B: 20.5 h ago for 4 h, aborted; 6 min gap
        _ev("backlog_implement", 20.5 * H, item_id=2, phase="started", round_id="SM_B"),
        _ev("backlog_implement", 16.5 * H, item_id=2, phase="finished", round_id="SM_B"),
        _ev("round_aborted", 16.45 * H, round_id="SM_B", reason="x"),
        # turn C: 16.4 h ago, still open at NOW
        _ev("backlog_implement", 16.4 * H, item_id=3, phase="started", round_id="SM_C"),
        # a turn a landing drain refused: started+skipped, 30 s, no gap of its own
        _ev("backlog_implement", 16.39 * H, item_id=4, phase="started"),
        _ev("backlog_implement", 16.38 * H, item_id=4, phase="skipped"),
    ])
    row = SC.compute(since_days=1, ledger=ledger, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    d = row["duty_cycle"]
    assert d["turns"] == 3 and d["gaps"] == 2
    assert d["idle_minutes"] == {"abort": 6.0, "landing": 30.0} and d["gap_counts"] == {"abort": 1, "landing": 1}
    assert d["largest_gap_minutes"] == 30.0
    # busy: 2 h + 4 h + 16.4 h of 24 h, less the hour before turn A
    assert abs(d["busy_hours"] - 22.4) < 0.05 and d["window_hours"] == 24.0
    assert abs(d["rate"] - 22.4 / 24) < 0.01
    text = SC.render(row)
    assert "| 14 | autocode duty cycle | 93% |" in text
    assert "2 gaps: abort 6 min (1), landing 30 min (1); largest 30 min" in text


def test_row_14_is_unmeasured_with_no_turn_in_the_window(tmp_path, repo):
    ledger = _ledger(tmp_path, [_ev("promoted", 1, round_id="SM_A", commit="a" * 40)])
    row = SC.compute(since_days=1, ledger=ledger, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    assert row["duty_cycle"]["rate"] is None and row["duty_cycle"]["turns"] == 0
    assert "| 14 | autocode duty cycle | — |" in SC.render(row)



# ── row 15: human overrides (#1179) ──────────────────────────────────────
#
# Alan interrupted a running implement round on 2026-09-15 at 22:13:54Z to
# prioritise the backlog sweep, and the loop honoured it: `backlog_retriage`
# plus eight `backlog_confirm_released` rows follow inside the next 60 s. The
# sentence that moved the queue survived only as a `reason` string on line
# 4523 of `promotions.jsonl`. At triage (2026-09-17 04:20Z) the 09-15 daily
# note had 0 hits for the round id across its 44 lines, and
# `scorecard --since 2w --json | grep -c SM_20260915_220118` read 0 over
# 5,798 events. Row 15 is the section that stops that being true.
#
# These tests write through `state.append_event` — the writer the loop uses —
# because the gap is as much the write path's as the reader's. Window
# membership is `ts` alone, which is why the rows below pin a fixture `ts` and
# keep the real `created_at` string for the report to carry verbatim.

_DIRECTIVE = ("Alan 2026-09-15: clear the loop so the backlog sweep runs; "
              "#535 is re-offered")
_DIRECTIVE_ROUND = "SM_20260915_220118"
# The fixture clock is `NOW`, not the real one, so the directive sits two days
# back — inside a 7-day window, which is where the live row sits in a 2-week
# report. Its `created_at` is the real row's stamp, verbatim: the section's whole
# job is to carry that stamp, and window membership never consults it.
_DIRECTIVE_TS = NOW - 2 * DAY
_DIRECTIVE_CREATED = "2026-09-15T22:13:54Z"


def _stamp(ts: float) -> str:
    """The stamp `state.append_event` would have written for `ts`."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _appended(tmp_path, entries):
    """A ledger written by `scripts.automod.state.append_event`, not by hand."""
    from scripts.automod import state as S
    p = tmp_path / "promotions.jsonl"
    for e in entries:
        S.append_event(e, path=p)
    return p


def _directive(ts: float = _DIRECTIVE_TS, created_at: str = _DIRECTIVE_CREATED, **extra):
    """The 09-15 abort row as the ledger holds it. `ts` moves it in or out of a window."""
    return {"event": "round_aborted", "ts": ts, "created_at": created_at,
            "round_id": _DIRECTIVE_ROUND, "reason": _DIRECTIVE, **extra}


def test_an_override_written_through_the_loop_s_writer_lands_in_the_report(tmp_path, repo):
    """#1179 clause 1: the directive survives compute() with its round id,
    timestamp, event type and reason whole — and the machine-shaped release
    rows that followed it do not, because they name nobody."""
    ledger = _appended(tmp_path, [
        _directive(),
        {"event": "backlog_confirm_released", "ts": _DIRECTIVE_TS + 23, "item_id": 535,
         "reason": "the implement pool has room (37 ready < bound 75)"},
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    ov = row["overrides"]
    assert ov["count"] == 1, "a release reason that names no one is not an override"
    hit = ov["rows"][0]
    assert hit["round_id"] == "SM_20260915_220118"
    assert hit["event"] == "round_aborted"
    assert hit["created_at"] == _DIRECTIVE_CREATED
    assert hit["field"] == "reason" and hit["person"] == "Alan"
    assert hit["text"] == _DIRECTIVE, "verbatim — a paraphrase is not a record"
    blob = json.dumps(ov)
    assert "SM_20260915_220118" in blob and "clear the loop so the backlog sweep runs" in blob


def test_an_abort_that_names_nobody_in_the_person_list_is_not_listed(tmp_path, repo):
    """#1179 clause 2: the detector is the person list and not the presence of
    a reason, so a loop-shaped reason, a bare abort and a name nobody in the
    report knows are all excluded, and `count` is the rows the section holds."""
    ledger = _appended(tmp_path, [
        {"event": "round_aborted", "ts": NOW - DAY, "round_id": "SM_GATE",
         "reason": "gate refused twice on the tests rung"},
        {"event": "round_aborted", "ts": NOW - 1.2 * DAY, "round_id": "SM_BARE"},
        {"event": "round_aborted", "ts": NOW - 1.5 * DAY, "round_id": "SM_OTHER",
         "reason": "Priya 2027-01-14: stop this round, it is the wrong item"},
        _directive(),
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    ov = row["overrides"]
    assert [r["round_id"] for r in ov["rows"]] == ["SM_20260915_220118"]
    assert ov["count"] == len(ov["rows"]), "the count is the rows, never a bigger claim"
    assert ov["found"] == 1 == ov["count"]
    assert ov["by_event"] == {"round_aborted": 1}


def test_a_directive_carried_in_note_rather_than_reason_is_surfaced_too(tmp_path, repo):
    """#1179 clause 3: 26 of the 88 person-named strings on the live ledger sit
    in `note` on decision events (`backlog_hand_fold`, `status_moved`), so
    scanning `reason` alone would have caught the 09-15 abort and missed every
    board override. One row per event even when both fields carry the name."""
    ledger = _appended(tmp_path, [
        {"event": "backlog_hand_fold", "ts": NOW - DAY, "item_id": 1202,
         "note": "Alan 2027-01-14: fold #1202 into #1180, keep #1180's clauses"},
        {"event": "item_closed", "ts": NOW - 2 * DAY, "item_id": 1199,
         "reason": _DIRECTIVE, "note": "Alan 2026-09-15: the same directive, quoted again"},
        _directive(),
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    ov = row["overrides"]
    assert ov["count"] == 3 == len(ov["rows"]), "the event carrying both fields counts once"
    assert ov["fields"] == {"reason": 2, "note": 1}
    fold = [r for r in ov["rows"] if r["event"] == "backlog_hand_fold"][0]
    assert fold["field"] == "note" and fold["item_id"] == 1202 and fold["round_id"] is None
    assert "fold #1202 into #1180" in json.dumps(ov)


def test_a_window_with_no_override_reports_zero_rather_than_a_fabricated_row(tmp_path, repo):
    """#1179 clause 4: the section is there with a count of 0 and no rows when
    the window holds nothing override-shaped, and an override that fell out of
    the window is genuinely absent — widening the window brings it back."""
    empty = _appended(tmp_path, [{"event": "round_aborted", "ts": NOW - DAY,
                                  "round_id": "SM_GATE",
                                  "reason": "gate refused twice on the tests rung"}])
    row = SC.compute(since_days=7, ledger=empty, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    ov = row["overrides"]
    assert ov["count"] == 0 and ov["rows"] == [] and ov["found"] == 0
    assert ov["by_event"] == {} and ov["fields"] == {"reason": 0, "note": 0}
    assert ov["cap"] == SC.OVERRIDE_ROW_CAP
    assert "| 15 | human overrides | 0 |" in SC.render(row)

    aged_ts = NOW - 30 * DAY
    directive = _appended(tmp_path / "old", [_directive(ts=aged_ts, created_at=_stamp(aged_ts))])
    out = SC.compute(since_days=7, ledger=directive, backlog_dir=tmp_path / "none",
                     repo=repo, now=NOW)
    assert out["overrides"]["count"] == 0, "30 days old against a 7-day window: not in it"
    widened = SC.compute(since_days=45, ledger=directive, backlog_dir=tmp_path / "none",
                         repo=repo, now=NOW)
    assert widened["overrides"]["count"] == 1, "the same row at 45 days: back"


def test_row_15_names_the_count_and_the_newest_override_and_survives_old_rows(tmp_path, repo):
    """#1179 clause 5: the CLI's text report carries the count and the newest
    override's round id, and a `scorecard.jsonl` row recorded before this
    section existed still renders — `record` appends forever and old rows are
    read back forever."""
    ledger = _appended(tmp_path, [
        _directive(),
        {"event": "item_closed", "ts": NOW - 3 * DAY, "created_at": "2027-01-12T08:00:00Z",
         "item_id": 1199, "reason": "Alan 2027-01-12: close this, the fix is live"},
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    line = next(ln for ln in SC.render(row).splitlines() if "human overrides" in ln)
    assert "| 15 | human overrides | 2 |" in line
    assert "SM_20260915_220118" in line, "the line names the NEWEST override, not the oldest"
    assert "clear the loop so the backlog sweep runs" in line
    assert "#1199" in line, "an override with no round id is named by its item"
    legacy = {k: v for k, v in row.items() if k != "overrides"}
    assert "| 15 | human overrides | — |" in SC.render(legacy)


def test_two_overrides_in_the_same_second_order_by_append_not_by_decode(tmp_path, repo):
    """One loop pass stamps several events with the same `ts`, so `newest` needs a
    tie-break. `sorted` is stable, so reversing the rows first makes the
    last-written row the newest — the order the ledger itself has them in."""
    same = NOW - 0.5 * DAY
    tied = _appended(tmp_path, [
        _directive(ts=same, created_at=_stamp(same)),
        {"event": "item_closed", "ts": same, "created_at": _stamp(same),
         "item_id": 1202, "reason": "Alan 2027-01-15: close #1202, the round landed it"},
    ])
    row = SC.compute(since_days=7, ledger=tied, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    ov = row["overrides"]
    assert ov["count"] == 2
    assert ov["rows"][0]["item_id"] == 1202, "appended last, so listed first"
    assert ov["rows"][1]["round_id"] == _DIRECTIVE_ROUND
    assert ov["rows"][0]["created_at"] == ov["rows"][1]["created_at"], "genuinely tied"


def test_the_tallies_count_the_listed_rows_when_the_cap_bites(tmp_path, repo, monkeypatch):
    """The section claims `count`, `by_event` and `fields` describe the rows it
    holds and `found` describes the window. A careless tally written over the found
    rows breaks that only when the cap bites, so that is the case pinned here —
    with the cap lowered rather than 121 events generated."""
    monkeypatch.setattr(SC, "OVERRIDE_ROW_CAP", 2)
    ledger = _appended(tmp_path, [
        _directive(),
        {"event": "item_closed", "ts": NOW - DAY, "item_id": 1202,
         "reason": "Alan 2027-01-14: close #1202 as duplicate of #1180"},
        {"event": "backlog_hand_fold", "ts": NOW - 1.4 * DAY, "item_id": 1199,
         "note": "Alan 2027-01-14: fold the #1199 finding under the round's clauses"},
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "none", repo=repo, now=NOW)
    ov = row["overrides"]
    assert ov["found"] == 3 and ov["count"] == 2 == len(ov["rows"]) and ov["cap"] == 2
    assert sum(ov["by_event"].values()) == ov["count"] == sum(ov["fields"].values()), (
        "the tallies follow the cap, so the count is the rows and found is the window")


def test_row_14_splits_the_two_waits_a_landing_makes(tmp_path, repo):
    """Landings wait twice and the fixes are different.

    `wait_for_settle` waits out the previous promotion's observation window —
    shortened by changing the window; `wait_for_rounds` waits for a sibling
    round's turn to end so the restart does not kill it — shortened only by
    changing depth. The duty-cycle row's `landing` idle class cannot tell them
    apart: it says a gap had a `promoted` row near it. On 2026-09-20 that put
    7.7 h under one label and the first read of it attributed the whole lot to
    the window.

    `waited` counts rows that actually blocked, `n` counts rows: a landing that
    skipped a wait records ~0 rather than nothing, and averaging those in would
    make a queue of real waits look short.
    """
    ledger = _ledger(tmp_path, [
        _ev("land_wait_settle", 0.5, round_id="SM_A", ok=True, waited_s=600.0, behind=1),
        _ev("land_wait_settle", 0.4, round_id="SM_B", ok=True, waited_s=300.0, behind=1),
        _ev("land_wait_settle", 0.3, round_id="SM_C", ok=True, waited_s=0.1, behind=0),
        _ev("land_wait_rounds", 0.5, round_id="SM_A", ok=True, waited_s=120.0),
        _ev("land_wait_rounds", 0.4, round_id="SM_B", ok=True, waited_s=0.1),
    ])
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "none",
                     repo=repo, now=NOW)
    waits = row["duty_cycle"]["waits"]
    assert waits["settle"] == {"n": 3, "waited": 2, "minutes": 15.0,
                               "median_s": 450.0, "max_s": 600.0}
    assert waits["rounds"] == {"n": 2, "waited": 1, "minutes": 2.0,
                               "median_s": 120.0, "max_s": 120.0}
    assert "settle 15 min (2/3)" in SC.render(row)
    assert "rounds 2 min (1/2)" in SC.render(row)
