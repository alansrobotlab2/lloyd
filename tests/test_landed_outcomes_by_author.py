"""#1667 — landed-change outcomes split by author: the five clauses.

Each node builds a real throwaway git repo (real commits, real diffs, because
`scorecard._undone_by_hand` reads `git show` and a mocked diff would test the
mock) plus a handful of ledger rows, and asks the report one clause's question.
Commit dates are set with `GIT_*_DATE` so the 7-day window is arithmetic the
node can see rather than wall-clock luck.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from eval.stats import wilson_ci
from scripts.automod import landed_outcomes as LO
from scripts.automod import scorecard as SC

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc).timestamp()
DAY = 86400
AGENT_MAIL = "alansrobotlab <alansrobotlab@gmail.com>"
HUMAN_MAIL = "Alan <alan@example.com>"
LOOP_MAIL = "lloyd <lloyd@localhost>"


def _commit(repo: Path, name: str, body: str, *, author: str, days_ago: float,
            message: str | None = None) -> str:
    """One commit `days_ago` days before NOW, replacing `name` with `body`."""
    when = (datetime.fromtimestamp(NOW, tz=timezone.utc)
            - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    env = {"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when,
           "GIT_AUTHOR_NAME": author.split(" <")[0], "GIT_AUTHOR_EMAIL":
           author.split(" <")[1].rstrip(">"), "GIT_COMMITTER_NAME": author.split(" <")[0],
           "GIT_COMMITTER_EMAIL": author.split(" <")[1].rstrip(">"),
           "PATH": "/usr/bin:/bin"}
    (repo / name).write_text(body, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], env=env, check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", message or f"{name} {when}"],
                   env=env, check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   text=True).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "-C", str(r), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(r), "config", "commit.gpgsign", "false"],
                   check=True, capture_output=True)
    return r


def _promoted(sha: str, *, round_id: str = "R1", days_ago: float = 10.0) -> dict:
    return {"event": "promoted", "commit": sha, "round_id": round_id,
            "ts": NOW - days_ago * DAY}


# ── clause 1 ──────────────────────────────────────────────────────────────

def test_a_landing_authored_as_a_person_is_still_the_agent_arm(repo: Path):
    """Clause 1: the arm key is the ledger's landed sha, not `%an`. Since
    `9407ddd1` a landing credits Alan as its author, so the commit under a
    `promoted` row carries the person's mail and must nevertheless be counted as
    the loop's work — otherwise the split reports 25 of the 30 newest landings as
    human and compares the loop with itself."""
    landing = _commit(repo, "a.py", "X = 1\n", author=AGENT_MAIL, days_ago=10)
    loop_named = _commit(repo, "b.py", "Y = 2\n", author=LOOP_MAIL, days_ago=9)
    doc_pass = _commit(repo, "c.md", "# doc\n", author=AGENT_MAIL, days_ago=8)
    person = _commit(repo, "d.py", "Z = 3\n", author=HUMAN_MAIL, days_ago=7)

    events = [_promoted(landing),
              {"event": "arch_review", "commit": doc_pass, "ts": NOW - 8 * DAY}]
    armed = {c["sha"]: c["arm"] for c in LO.assign_arms(SC._git_log(repo, 0),
                                                        LO.landing_shas(events))}
    assert armed[landing] == LO.AGENT, "the post-9407ddd1 shape: Alan's mail, the loop's commit"
    assert armed[loop_named] == LO.AGENT, "a pre-9407ddd1 commit made under a loop identity"
    assert armed[doc_pass] == LO.AGENT, "an arch_review landing is the loop's too"
    assert armed[person] == LO.HUMAN
    assert sorted(armed.values()) == sorted([LO.AGENT] * 3 + [LO.HUMAN] * 1)

    # A sha in the ledger cannot pull an unrelated commit in: prefix matching is
    # exact-or-prefix, and a prefix too short to name a commit matches nothing.
    assert not LO.is_agent({"sha": person, "author": "Alan"}, [landing])
    assert not LO.is_agent({"sha": person, "author": "Alan"}, ["a"])
    assert not LO.is_agent({"sha": person, "author": "Alan"}, [""])


# ── clause 2 ──────────────────────────────────────────────────────────────

def test_every_rate_carries_its_own_numerator_and_denominator_separately():
    """Clause 2: `k` and `n` are fields, and no landings renders as `0/0 (no
    landings)` rather than 0%, a blank or a dash — the zero-denominator rule,
    because `0 %` reads as "checked and clean" over a window that checked
    nothing."""
    empty = LO.rate(0, 0)
    assert (empty["k"], empty["n"], empty["rate"]) == (0, 0, None)
    assert empty["ci_low"] is None and empty["ci_high"] is None
    assert LO.rate_text(empty) == "0/0 (no landings)"
    assert LO.rate_text(LO.rate(0, 5)) == "0/5 (0.0 %) [0.000, 0.434]"
    assert LO.rate_text(LO.rate(3, 12)) == "3/12 (25.0 %) [0.089, 0.532]"


def test_no_rate_in_the_report_ever_renders_without_its_denominator(repo: Path):
    """The same clause over a whole report: every arm row's rate-shaped value
    exposes `k` and `n`, and a rate computed over nothing says so in words."""
    sha = _commit(repo, "a.py", "X = 1\n", author=AGENT_MAIL, days_ago=10)
    row = LO.by_author(now=NOW, repo=repo, events=[_promoted(sha)])
    for arm in (LO.AGENT, LO.HUMAN):
        body = row["by_author"][arm]
        for key in ("later_undo_7d", "revert_commits", "guardian_rollbacks",
                    "rounds_to_land"):
            r = body[key]
            assert {"k", "n", "rate", "ci_low", "ci_high"} <= set(r), (arm, key, r)
            assert isinstance(r["k"], int) and isinstance(r["n"], int)
            # A measured number prints its own denominator; an unmeasured one
            # says which, in words, rather than drawing a dash or a 0 %.
            assert (f"{r['k']}/{r['n']}" in LO.rate_text(r)) if r["n"] else \
                (LO.rate_text(r) == "0/0 (no landings)"), (arm, key, r)
    # The gate-shaped rows are the agent arm's by construction: the one landing
    # in the fixture is the agent's, and the human arm has no promotion at all.
    assert (row["by_author"][LO.AGENT]["guardian_rollbacks"]["n"],
            row["by_author"][LO.HUMAN]["guardian_rollbacks"]["n"]) == (1, 0)
    assert LO.rate_text(row["by_author"][LO.HUMAN]["guardian_rollbacks"]) == \
        "0/0 (no landings)"
    json.dumps(row)   # serialisable: NaN would raise here


# ── clause 3 ──────────────────────────────────────────────────────────────

def test_every_rate_carries_a_wilson_interval_from_the_n_it_prints(repo: Path):
    """Clause 3: the interval is `eval.stats.wilson_ci(k, n)` for the SAME `n`
    on the page — an interval computed against a different denominator than the
    one printed is a decoration, and the two arms are only comparable through
    their intervals."""
    row = LO.rate(7, 21)
    lo, hi = wilson_ci(7, 21)
    assert (row["ci_low"], row["ci_high"]) == (round(lo, 4), round(hi, 4))
    assert f"{row['k']}/{row['n']}" in LO.rate_text(row)

    r = LO.by_author(now=NOW, repo=repo, events=[])
    seen = 0
    for arm in (LO.AGENT, LO.HUMAN):
        for key, val in r["by_author"][arm].items():
            if isinstance(val, dict) and {"k", "n"} <= set(val):
                seen += 1
                if val["n"]:
                    exp = wilson_ci(val["k"], val["n"])
                    assert (val["ci_low"], val["ci_high"]) == (round(exp[0], 4),
                                                              round(exp[1], 4)), key
                else:
                    assert val["ci_low"] is None, key
    assert seen >= 6, "the walk found no rates to check"
    # The CI is not a constant: different n, different interval.
    assert LO.rate(1, 2)["ci_high"] != LO.rate(50, 100)["ci_high"]


# ── clause 4 ──────────────────────────────────────────────────────────────

def _undo_fixture(repo: Path) -> tuple[list[dict], dict[str, str]]:
    """Two arms, each with one of its lines deleted by the other within 7 days.

    Loop landings L1 (day 25), L2 (24), L3 (19); person's commits H1 (23), H2
    (22), H3 (21), and H4 (20) which deletes a line L1 added. L3 (19) deletes a
    line H1 added. Every commit is ≥ 19 days old, so the 7-day window has closed
    on all of them and the denominators are the arm sizes, not a truncation.
    Each of the loop's three commits carries a `promoted` row, because the arm is
    the ledger's landed sha: a commit made under the same mail that is not a
    landing belongs to the person.
    """
    l1 = _commit(repo, "a.py", "AGENT_KEPT = 1\nAGENT_UNDONE = 2\n", author=AGENT_MAIL,
                 days_ago=25)
    l2 = _commit(repo, "b.py", "AGENT_OTHER = 1\n", author=AGENT_MAIL, days_ago=24)
    _commit(repo, "c.py", "HUMAN_KEPT = 1\nHUMAN_UNDONE = 2\n", author=HUMAN_MAIL,
            days_ago=23)
    _commit(repo, "d.py", "HUMAN_OTHER = 1\n", author=HUMAN_MAIL, days_ago=22)
    _commit(repo, "e.py", "HUMAN_THIRD = 1\n", author=HUMAN_MAIL, days_ago=21)
    _commit(repo, "a.py", "AGENT_KEPT = 1\n", author=HUMAN_MAIL, days_ago=20,
            message="clean up after the loop")
    l3 = _commit(repo, "c.py", "HUMAN_KEPT = 1\n", author=AGENT_MAIL, days_ago=19,
                 message="drop the dead constant")
    events = [_promoted(l1, round_id="RL1", days_ago=25),
              _promoted(l2, round_id="RL2", days_ago=24),
              _promoted(l3, round_id="RL3", days_ago=19)]
    return events, {"l1": l1, "l2": l2, "l3": l3}


def test_the_later_undo_measure_is_one_definition_run_on_both_arms(repo: Path):
    """Clause 4: the discriminating measure, computed for both arms from the same
    function, so `agent 1/3` and `human 1/4` are the same question asked twice
    rather than an aggregate on one side. Row 5's semantics are kept: a line the
    other arm added and later removed, within 7 days, not merely edited around."""
    events, shas_ = _undo_fixture(repo)
    commits = LO.assign_arms(SC._git_log(repo, 0), LO.landing_shas(events))
    armed = {c["sha"]: c["arm"] for c in commits}
    assert [armed[s] for s in (shas_["l1"], shas_["l2"], shas_["l3"])] == \
        [LO.AGENT] * 3, "the three landings are the agent arm"
    assert sum(1 for c in commits if c["arm"] == LO.HUMAN) == 4

    agent = LO.later_undo(commits, repo=repo, arm=LO.AGENT, now=NOW)
    human = LO.later_undo(commits, repo=repo, arm=LO.HUMAN, now=NOW)
    assert (agent["k"], agent["n"]) == (1, 3), agent
    assert (human["k"], human["n"]) == (1, 4), human
    assert agent["k"] <= agent["n"] and human["k"] <= human["n"]
    # Comparable, not identical: the arms' denominators differ (4 human commits,
    # 3 landings), so their intervals differ — and each is that arm's own
    # wilson_ci, which is what lets the two be compared at all.
    assert (agent["ci_low"], agent["ci_high"]) == tuple(
        round(x, 4) for x in wilson_ci(1, 3))
    assert (human["ci_low"], human["ci_high"]) == tuple(
        round(x, 4) for x in wilson_ci(1, 4))
    assert (agent["ci_low"], agent["ci_high"]) != (human["ci_low"], human["ci_high"])

    # One definition, not two: both arms reached `_undone_by_hand`, and the
    # commits it was asked about came from both arms.
    asked: list[str] = []
    real = SC._undone_by_hand

    def spy(repo_, landing, later):
        asked.append(str(landing.get("commit")))
        return real(repo_, landing, later)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(SC, "_undone_by_hand", spy)
    try:
        LO.later_undo(commits, repo=repo, arm=LO.AGENT, now=NOW)
        LO.later_undo(commits, repo=repo, arm=LO.HUMAN, now=NOW)
    finally:
        monkey.undo()
    armed = {c["sha"]: c["arm"] for c in commits}
    assert {armed.get(s) for s in asked} == {LO.AGENT, LO.HUMAN}, asked


def test_a_commit_too_young_to_be_undone_is_not_counted_as_clean(repo: Path):
    """The other half of the same denominator: a 2-day-old commit cannot have
    been undone within 7 days yet, so counting it would drag the rate down with
    the age of the window and make the two arms incomparable by age mix."""
    old = _commit(repo, "a.py", "AGENT_KEPT = 1\nAGENT_UNDONE = 2\n", author=AGENT_MAIL,
                  days_ago=20)
    _commit(repo, "a.py", "AGENT_KEPT = 1\n", author=HUMAN_MAIL, days_ago=16)
    young = _commit(repo, "young.py", "RECENT = 1\n", author=AGENT_MAIL, days_ago=2)
    events = [_promoted(old, days_ago=20), _promoted(young, days_ago=2)]
    commits = LO.assign_arms(SC._git_log(repo, 0), LO.landing_shas(events))
    armed = {c["sha"]: c["arm"] for c in commits}
    assert armed[old] == armed[young] == LO.AGENT, "both are landings, so both are the agent arm"
    out = LO.later_undo(commits, repo=repo, arm=LO.AGENT, now=NOW)
    assert out["n"] == 1, "the 2-day-old commit is outside the observable window"
    assert (out["k"], out["n"]) == (1, 1)


# ── clause 5 ──────────────────────────────────────────────────────────────

def _round_events() -> list[dict]:
    """Round RA: refused then passed (2 review rows). Round RB: passed first time
    (1). Round RC: landed with no review row at all. Every round also carries 12
    `gate` rows — 2 of them the review rung — which is the ladder's length, not
    the loop's iteration count, and must not be what is counted."""
    ev: list[dict] = []
    for rid, attempts in (("RA", 2), ("RB", 1), ("RC", 0)):
        ev.append({"event": "promoted", "round_id": rid, "commit": f"{rid}sha",
                   "ts": NOW - 5 * DAY})
        for rung in ("static", "tests", "tests", "canary_boot", "canary_smoke",
                     "vault_sync", "prompt_surface", "review", "drill", "ledger",
                     "boot", "venv"):
            ev.append({"event": "gate", "round_id": rid, "rung": rung,
                       "ts": NOW - 5 * DAY})
        for attempt in range(attempts):
            ev.append({"event": "review", "round_id": rid, "attempt": attempt + 1,
                       "ok": attempt == attempts - 1, "ts": NOW - 5 * DAY})
    return ev


def test_rounds_to_land_counts_review_rows_and_is_agent_only():
    """Clause 5: attempts are `review` rows per `round_id` — 3 reviews over the 2
    rounds the grader saw, mean 1.5, one of the two needing a second pass — not
    `gate` rows, which would say 36 and report the length of the ladder as the
    cost of iterating. A landed round the grader never saw (RC) is left out of the
    denominator rather than booked as one cheap attempt. And the human arm prints
    `n/a (no gate)`: a person's commit goes through no review, so a 0 there would
    claim it was graded once and passed."""
    ev = _round_events()
    row = LO.rounds_to_land(ev, landed_rounds=LO._landed_rounds(ev))
    assert (row["k"], row["n"]) == (1, 2), "one of two graded rounds needed a 2nd attempt"
    assert row["mean"] == 1.5 and row["median"] == 1.5, row
    assert row["total_reviews"] == 3 and row["distribution"] == {"1": 1, "2": 1}, row
    lo, hi = wilson_ci(1, 2)
    assert LO.rounds_text(row) == (
        f"mean 1.5 reviews/round over 2 rounds (median 1.5); "
        f"second attempt 1/2 (50.0 %) [{lo:.3f}, {hi:.3f}]")
    # What a gate-row count would have said, so the difference stays checkable.
    gate_rows = sum(1 for e in ev if e["event"] == "gate")
    assert gate_rows == 36 and row["total_reviews"] != gate_rows, gate_rows
    assert LO.rounds_text(LO.rate(0, 0)) == "n/a (no gate)"

    report = LO.by_author(now=NOW, repo=Path("/nonexistent-repo"), events=ev)
    human = report["by_author"][LO.HUMAN]
    assert human["rounds_to_land"]["n"] == 0 and human["rounds_to_land"]["k"] == 0
    assert LO.rounds_text(human["rounds_to_land"]) == "n/a (no gate)"
    assert "n/a (no gate)" in LO.render(report)
    assert report["by_author"][LO.AGENT]["rounds_to_land"]["n"] == 2


# ── where the row goes ────────────────────────────────────────────────────

def test_the_row_goes_to_the_automod_state_dir_not_into_the_checkout(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The snapshot destination the triage pinned: `scorecard_path()`'s directory.
    `~/lloyd/eval/baselines/` is git-tracked in the code tree (9 files) against
    `.gitignore:127`, and the guardian alerts hourly on runtime data inside the
    checkout, so a measurement that wrote there would page a person."""
    from scripts.automod import state as S

    path = LO.landed_outcomes_path()
    assert path.name == "landed_outcomes.jsonl"
    assert path.parent == S.STATE_DIR == SC.scorecard_path().parent
    assert SC.LIVE_ROOT not in path.parents

    dest = tmp_path / "state" / "landed_outcomes.jsonl"
    out = LO.record({"generated_at": "2026-09-28T12:00:00Z"}, dest)
    assert out == dest and json.loads(dest.read_text())["generated_at"]
    assert SC.LIVE_ROOT not in dest.resolve().parents
