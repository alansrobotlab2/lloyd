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


# ── #1870: the human arm's true reach, and the months no rate covers ──────

def _init_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "commit.gpgsign", "false"],
                   check=True, capture_output=True)
    return path


def _reach_fixture(repo: Path, *, with_old_history: bool = True,
                   loop_old_commit: bool = True) -> list[dict]:
    """The in-window shape fixed: two landings (days 10 and 7, whose `promoted`
    rows open the ledger at day 10), one human commit (day 9) and one human
    cleanup (day 8) that removes an agent line within 7 d, and the day-7 landing
    removing a human line — so by hand `later_undo_7d` is 1/2 on each arm and
    `revert_commits` 0/2, and a k/n equality can never be two empties.

    `with_old_history` prepends commits that predate the ledger's first row: two
    human (days 45 and 30) plus, when `loop_old_commit`, a day-60 commit under
    the loop identity — agent-arm with no ledger row, older than every human
    commit. A reach taken as a min over BOTH arms reports day 60; a reach taken
    from the window-filtered list, the shipped bug, reports day 10; the correct
    one is the human arm's day-45 date from the whole of the history.
    """
    if with_old_history:
        if loop_old_commit:
            _commit(repo, "old_loop.py", "LOOP_OLD = 1\n", author=LOOP_MAIL,
                    days_ago=60)
        _commit(repo, "old_human.py", "OLD_HUMAN = 1\n", author=HUMAN_MAIL,
                days_ago=45)
        _commit(repo, "old_two.py", "OLD_TWO = 2\n", author=HUMAN_MAIL,
                days_ago=30)
    l1 = _commit(repo, "a.py", "AGENT_KEPT = 1\nAGENT_UNDONE = 2\n",
                 author=AGENT_MAIL, days_ago=10)
    _commit(repo, "c.py", "HUMAN_KEPT = 1\nHUMAN_UNDONE = 2\n",
            author=HUMAN_MAIL, days_ago=9)
    _commit(repo, "a.py", "AGENT_KEPT = 1\n", author=HUMAN_MAIL, days_ago=8,
            message="clean up after the loop")
    l2 = _commit(repo, "c.py", "HUMAN_KEPT = 1\n", author=AGENT_MAIL, days_ago=7,
                 message="drop the dead constant")
    return [_promoted(l1, round_id="RL1", days_ago=10),
            _promoted(l2, round_id="RL2", days_ago=7)]


def _epoch(iso: str) -> float:
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def test_human_reach_is_the_oldest_human_commit_of_all_history_not_the_window(
        repo: Path):
    """#1870 clause 1: `human_history_from` is the oldest HUMAN-arm commit of the
    whole of `main`'s history — the day-45 one here — strictly earlier than
    `comparable_from` (day 10, the window start) and not the day-60 loop-identity
    commit either: a min over both arms would credit the person with the loop's
    history."""
    events = _reach_fixture(repo, with_old_history=True)
    w = LO.by_author(now=NOW, repo=repo, events=events)["window"]
    assert _epoch(w["human_history_from"]) == NOW - 45 * DAY
    assert _epoch(w["human_history_from"]) < _epoch(w["comparable_from"])
    assert _epoch(w["human_history_from"]) < _epoch(w["from"])
    assert _epoch(w["comparable_from"]) == NOW - 10 * DAY


def test_render_prints_the_unfiltered_reach_and_not_the_window_start(
        repo: Path):
    """#1870 clause 2: the `human arm history reaches` line prints that
    unfiltered date (day 45), which is neither the window `from` nor
    `comparable_from` (both day 10) while older human history exists."""
    events = _reach_fixture(repo, with_old_history=True)
    row = LO.by_author(now=NOW, repo=repo, events=events)
    reach = row["window"]["human_history_from"]
    lines = [ln for ln in LO.render(row).splitlines()
             if "human arm history reaches" in ln]
    assert len(lines) == 1, LO.render(row)
    assert f"human arm history reaches {reach}" in lines[0]
    assert reach != row["window"]["from"]
    assert reach != row["window"]["comparable_from"]
    assert _epoch(reach) == NOW - 45 * DAY


def test_report_carries_one_line_naming_the_excluded_pre_ledger_count(
        repo: Path):
    """#1870 clause 3: exactly one line names the count of commits older than
    `comparable_from` — the fixture's two human commits (days 45, 30), all of
    them one arm — and says they are excluded from every rate for want of a
    ledger counterpart."""
    events = _reach_fixture(repo, with_old_history=True, loop_old_commit=False)
    row = LO.by_author(now=NOW, repo=repo, events=events)
    assert row["window"]["pre_comparable_excluded"] == 2
    assert row["window"]["pre_comparable_single_arm"] == LO.HUMAN
    text = LO.render(row)
    excl = [ln for ln in text.splitlines() if "excluded from every rate" in ln]
    assert len(excl) == 1, text
    assert "2 commits" in excl[0]
    assert "single-arm (human)" in excl[0]
    assert "no ledger counterpart" in excl[0]


def test_pre_ledger_commits_move_no_rate_k_or_n(tmp_path: Path):
    """#1870 clause 4: the same in-window fixture with three pre-ledger commits
    (two human, one agent) reports the SAME k/n as the repo without them:
    `later_undo_7d` 1/2 and `revert_commits` 0/2 on each arm, and the arms'
    commit counts stay 2/2 — if the excluded commits leaked into a denominator
    they would read 2/5 (agent n) and 2/4 or worse."""
    with_old = _init_repo(tmp_path / "repo-with-old")
    plain = _init_repo(tmp_path / "repo-plain")
    row_old = LO.by_author(
        now=NOW, repo=with_old,
        events=_reach_fixture(with_old, with_old_history=True))
    row_plain = LO.by_author(
        now=NOW, repo=plain,
        events=_reach_fixture(plain, with_old_history=False))
    assert (row_old["window"]["pre_comparable_excluded"],
            row_plain["window"]["pre_comparable_excluded"]) == (3, 0)
    for arm in (LO.AGENT, LO.HUMAN):
        for key in ("later_undo_7d", "revert_commits"):
            a = row_old["by_author"][arm][key]
            b = row_plain["by_author"][arm][key]
            assert (a["k"], a["n"]) == (b["k"], b["n"]), (arm, key, a, b)
    # The equalities are over measured numbers, not two empty rates.
    assert (row_old["by_author"][LO.AGENT]["later_undo_7d"]["k"],
            row_old["by_author"][LO.AGENT]["later_undo_7d"]["n"]) == (1, 2)
    assert (row_old["by_author"][LO.HUMAN]["later_undo_7d"]["k"],
            row_old["by_author"][LO.HUMAN]["later_undo_7d"]["n"]) == (1, 2)
    assert (row_old["by_author"][LO.AGENT]["revert_commits"]["k"],
            row_old["by_author"][LO.AGENT]["revert_commits"]["n"]) == (0, 2)
    assert (row_old["by_author"][LO.HUMAN]["revert_commits"]["k"],
            row_old["by_author"][LO.HUMAN]["revert_commits"]["n"]) == (0, 2)
    assert (row_old["by_author"][LO.AGENT]["commits"],
            row_old["by_author"][LO.HUMAN]["commits"]) == (2, 2)


def test_the_report_leaves_the_repo_it_measured_clean(repo: Path):
    """#1870 clause 5: running the whole report — unfiltered history fetch and
    all — over a fixture repo with pre-ledger history leaves
    `git status --porcelain` empty in that repo."""
    events = _reach_fixture(repo, with_old_history=True)
    LO.render(LO.by_author(now=NOW, repo=repo, events=events))
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    assert dirty == ""


# ── #2378: cost and latency per ACHIEVED LANDING ─────────────────────────────
#
# The economics half reads two stores the quality half never touched: the board
# files behind an `item_id` (the filing stamp no ledger row carries) and the
# queue's `runs` table (the implementer's wall clock, which the ledger's rung
# seconds never covered). Both are real below — real board markdown, a real
# sqlite `runs` table — because the join under test is a regex over the free-text
# `runs.summary`, and a mocked join would test the mock.
#
# One fixture, two rounds, numbers small enough to check by hand: the landing's
# round costs 480 s of ledger time and the never-landed one 120 s. The fields that
# must NOT be counted are in the event list on purpose (`review.rung_wait_s`,
# `review.waited_s`, `error.errors_window_s`, and a `pause_set` row with no
# `round_id`), so a change that sweeps them in fails on the number rather than
# quietly inflating the stated cost of a landing.

ROUND_LAND = "SM_20260927_010203"
ROUND_FAIL = "SM_20260927_020203"
ECON_T0 = datetime(2026, 9, 27, tzinfo=timezone.utc).timestamp()
#: gate 300 + review 100 + review_confirm 20 + land_wait 60 = 480. The same
#: review row also carries `rung_wait_s` 900 (a review graded CONCURRENTLY with
#: the tests rung) and `waited_s` 5 (a retry sleep nested inside its `seconds`);
#: the round also carries `errors_window_s` 7200 (the length of a rate window,
#: not a spend). Count any of the three and this number moves.
LAND_ROUND_SECONDS = 480.0
#: gate 100 + review 20, in a round that produced no `promoted` row.
FAIL_ROUND_SECONDS = 120.0
#: A row with no `round_id`: nothing to charge it to, so it reaches no figure.
UNATTRIBUTED_PAUSE_SECONDS = 5000.0
#: The implementer's own wall clock for the landing's round, as one `runs` row.
LAND_RUN_SECONDS = 1800.0
#: The filing stamps: item 2001 is filed 600 s before its first `round_start`
#: row and 1080 s before its `promoted` row, so every latency is a number this
#: fixture chose. Both predate the first ledger row, which is the cohort split
#: the block has to report rather than truncate away (#1667's arms).
#:
#: Both are after `app.backlog_move.LOCAL_STAMP_CUTOVER` (2026-09-26T04:00Z) on
#: purpose: below it a naive `created:` is legitimately the machine's local
#: clock, so a fixture filed before that instant reads seven hours later than it
#: was written, its filing lands after its own first round, and every latency in
#: the population is excluded as negative — the n drops to 0 and the block
#: correctly prints `n/a` for a fixture that looked like it had numbers.
ITEM_CREATED = {2001: "2026-09-27T00:00:00", 2002: "2026-09-27T01:00:00"}


def _econ_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _econ_events(commit: str) -> list[dict]:
    """Two rounds over one window: one lands, one spends and never lands."""
    start = ECON_T0 + 600
    return [
        {"event": "round_start", "round_id": ROUND_LAND, "item_id": 2001,
         "ts": start, "created_at": _econ_iso(start)},
        {"event": "gate", "round_id": ROUND_LAND, "seconds": 300.0,
         "ts": start + 200, "created_at": _econ_iso(start + 200)},
        {"event": "error", "round_id": ROUND_LAND, "errors_window_s": 7200.0,
         "ts": start + 210, "created_at": _econ_iso(start + 210)},
        {"event": "review", "round_id": ROUND_LAND, "seconds": 100.0,
         "review_confirm_seconds": 20.0, "rung_wait_s": 900.0, "waited_s": 5.0,
         "ts": start + 380, "created_at": _econ_iso(start + 380)},
        {"event": "pause_set", "seconds": UNATTRIBUTED_PAUSE_SECONDS,
         "ts": start + 400, "created_at": _econ_iso(start + 400)},
        {"event": "land_wait_rounds", "round_id": ROUND_LAND, "waited_s": 60.0,
         "ts": start + 460, "created_at": _econ_iso(start + 460)},
        {"event": "promoted", "round_id": ROUND_LAND, "commit": commit,
         "item_id": 2001, "ts": start + LAND_ROUND_SECONDS,
         "created_at": _econ_iso(start + LAND_ROUND_SECONDS)},
        {"event": "round_start", "round_id": ROUND_FAIL, "item_id": 2002,
         "ts": ECON_T0 + 700, "created_at": _econ_iso(ECON_T0 + 700)},
        {"event": "gate", "round_id": ROUND_FAIL, "seconds": 100.0,
         "ts": ECON_T0 + 750, "created_at": _econ_iso(ECON_T0 + 750)},
        {"event": "review", "round_id": ROUND_FAIL, "seconds": 20.0,
         "ts": ECON_T0 + 790, "created_at": _econ_iso(ECON_T0 + 790)},
    ]


def _econ_board(tmp_path: Path) -> Path:
    """The board the two fixture items were filed on, with their `created:`."""
    board = tmp_path / "backlog"
    board.mkdir(parents=True, exist_ok=True)
    for num, stamp in ITEM_CREATED.items():
        (board / f"{num}-fixture.md").write_text(
            f"---\ntype: note\ntimestamp: '{stamp}'\nitem_id: {num}\n"
            f"status: done\ncreated: '{stamp}'\n---\n\n# {num} — a fixture item\n",
            encoding="utf-8")
    return board


def _econ_runs_db(tmp_path: Path, name: str,
                  rows: list[tuple[float, str]]) -> Path:
    """A `runs` table shaped like the queue's, holding (seconds, summary) `rows`."""
    import sqlite3
    db = tmp_path / f"{name}.db"
    con = sqlite3.connect(str(db))
    con.execute("create table runs (run_id integer primary key autoincrement,"
                "source text, task_id integer, duration_seconds real, summary text,"
                "response_json text, meta_json text)")
    con.executemany("insert into runs (source, task_id, duration_seconds, summary) "
                    "values ('autocode', 7, ?, ?)", rows)
    con.commit()
    con.close()
    return db


#: The run rows that make the join strong: the landing's round named once, one
#: row naming TWO rounds (which must be dropped, not split), and one naming none.
STRONG_RUNS = [(LAND_RUN_SECONDS, f"Round {ROUND_LAND}: Implement #2001 — fixture"),
               (500.0, f"Round {ROUND_LAND} and {ROUND_FAIL}: two rounds, one run"),
               (9999.0, "no round named here at all")]


def _econ(tmp_path: Path, *, name: str = "main",
          runs_rows: list[tuple[float, str]] | None = STRONG_RUNS,
          events: list[dict] | None = None):
    """Report row and rendered text over the fixture above.

    `runs_rows=None` names no db at all — the weakest join there is — and any
    list builds a real one. `name` keeps several fixtures in one `tmp_path` apart.
    """
    # The `repo` fixture above cannot be used as a fixture here (one node builds
    # three of these), so the init it does is done here, identically.
    repo = (tmp_path / name / "repo").resolve()
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "commit.gpgsign", "false"],
                   check=True, capture_output=True)
    _commit(repo, "a.txt", "one\n", author=AGENT_MAIL, days_ago=1)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    board = _econ_board(tmp_path)
    db = _econ_runs_db(tmp_path, name, runs_rows) if runs_rows is not None else None
    row = LO.by_author(repo=repo, now=NOW,
                       events=events if events is not None else _econ_events(sha),
                       runs_db=db, backlog_dir=board,
                       ledger=tmp_path / f"{name}-no-ledger.jsonl")
    return row, LO.render(row)


def test_cost_block_divides_ledger_seconds_by_achieved_landing_with_n_window_proxy(tmp_path):
    """Clause 1: the denominator of the cost line is an achieved landing.

    480 s of ledger seconds inside the one promoted round, over the one landing
    that has seconds, is 0.1333 hours — printed under the autonomy page's own
    wording, because `duration_seconds/3600` is what that page calls `gpu_hours`
    (`workers/queue.py:1353`) and the store holds no GPU-seconds term to be
    precise about. `n=1` and the window share the line with the figure, and the
    window IS the arms' window: a cost measured over a different span than the
    quality is not a comparison, it is a second report (#1667's lesson).
    """
    row, text = _econ(tmp_path)
    cost = row["cost"]
    assert cost["n"] == 1 and cost["landings"] == 1
    assert cost["ledger_seconds"] == LAND_ROUND_SECONDS + FAIL_ROUND_SECONDS, (
        "a nested or round-less seconds field leaked into the ledger total: "
        "rung_wait_s, review.waited_s, errors_window_s and the round-less "
        "pause_set row are all in the fixture, and none of them belong here")
    assert cost["landed_seconds"] == LAND_ROUND_SECONDS
    assert cost["gpu_hours_per_landing"] == round(LAND_ROUND_SECONDS / 3600.0, 4)
    assert cost["median_seconds_per_landing"] == LAND_ROUND_SECONDS
    assert cost["window_from"] == row["window"]["from"] == _econ_iso(ECON_T0 + 600)
    assert cost["window_to"] == row["window"]["to"] == LO._iso(NOW)
    assert "wall-clock seconds/3600" in cost["proxy"]
    assert "not measured GPU time" in cost["proxy"]
    lines = [ln for ln in text.splitlines() if ln.startswith("gpu_hours_per_landing")]
    assert len(lines) == 1, "the cost of a landing printed twice, or not at all"
    assert "0.133 h" in lines[0] and "n=1" in lines[0]
    assert cost["window_from"] in lines[0] and cost["window_to"] in lines[0], (
        "a cost figure with no window on its own line is a figure from nowhere")
    assert cost["proxy"] in text


def test_cost_block_prints_the_never_landed_share_beside_the_figure_in_every_state(tmp_path):
    """Clause 2: 120 of the window's 600 ledger seconds sit in a round that never
    landed, and that share is printed in every state the block can be in.

    Measured, join-too-weak, and no-landings: the share line is in all three.
    Cost-per-outcome is gameable by the cheapest move available — attempt less,
    look cheaper — and a guard that can be separated from the figure it guards
    guards nothing.
    """
    row, text = _econ(tmp_path)
    cost = row["cost"]
    assert cost["never_landed"] == {"k": int(FAIL_ROUND_SECONDS),
                                    "n": int(LAND_ROUND_SECONDS + FAIL_ROUND_SECONDS)}
    lines = LO.cost_lines(cost)
    never = [ln for ln in lines if ln.startswith("never_landed_share")]
    assert len(never) == 1
    assert "120/600 s (20.0 %)" in never[0]
    assert "rounds with no `promoted` row" in never[0]
    gpu = next(i for i, ln in enumerate(lines)
               if ln.startswith("gpu_hours_per_landing"))
    assert never[0] in lines[gpu:], "the share printed before the figure it qualifies"
    for name, weak_runs in (("weak", []),
                            ("weaker", [(900.0, f"Round {ROUND_FAIL}: nothing to charge")])):
        weak_row, _ = _econ(tmp_path, name=name, runs_rows=weak_runs)
        assert any(ln.startswith("never_landed_share")
                   for ln in LO.cost_lines(weak_row["cost"])), name
    no_landing_row, _ = _econ(tmp_path, name="empty", runs_rows=None,
                             events=[e for e in _econ_events("deadbeef")
                                     if e.get("event") != "promoted"])
    assert any(ln.startswith("never_landed_share")
               for ln in LO.cost_lines(no_landing_row["cost"]))


def test_latency_block_prints_the_two_waits_as_p50_p90_with_n_on_one_clock(tmp_path):
    """Clause 3: filing→first-round and first-round→landing, each with an n.

    Three stamps from three places — the item's `created:` on the board, the
    round's earliest `round_start` row, its `promoted` row — read as ONE instant.
    Item 2001 was filed 2026-09-27T00:00:00, its round opened 600 s later and
    landed 480 s after that. A block that read the naive board stamp as local
    time would move the filing by this box's offset and report a triage latency
    that never happened; `utc_instant` (#1517) is the one place that decided what
    a naive board stamp means, and the `clock` field names it. Both cohorts are
    printed even when empty, because an item filed before the ledger exists is
    strata, not truncation (#1667's arms).
    """
    row, text = _econ(tmp_path)
    lat = row["latency"]
    assert lat["filing_to_first_round"] == {"n": 1, "p50_s": 600.0, "p90_s": 600.0}
    assert lat["first_round_to_landing"] == {"n": 1, "p50_s": LAND_ROUND_SECONDS,
                                             "p90_s": LAND_ROUND_SECONDS}
    assert lat["filing_to_landing"] == {"n": 1, "p50_s": 1080.0, "p90_s": 1080.0}
    assert lat["missing_item_stamp"] == 0 and lat["missing_round_start"] == 0
    assert lat["negative_latency_excluded"] == 0
    assert "utc_instant" in lat["clock"], "the clock the stamps share is unnamed"
    assert "p50 600 s, p90 600 s  (n=1)" in text
    assert "p50 480 s, p90 480 s  (n=1)" in text
    by_cohort = {st["cohort"]: st for st in lat["strata"]}
    assert by_cohort["filed_before_ledger"]["n"] == 1
    assert by_cohort["filed_in_ledger_window"]["n"] == 0
    assert f"{'filed_before_ledger':30s} n=1:" in text
    assert f"{'filed_in_ledger_window':30s} 0/0 (no landings)" in text, (
        "an empty cohort printed a dash or a bare 0 instead of saying there were none")


def test_run_join_match_rate_prints_and_a_weak_join_withholds_the_cost_figure(tmp_path):
    """Clause 4: `N/M promoted landings attributed to at least one run row`, and
    the floor that replaces the figure below it.

    `round_id` is not a column on a run row — the carrier is the free-text
    `runs.summary` — so the join is measured and printed, not trusted. One
    round-naming row of 1800 s makes the single landing 1/1 and the joined term
    0.5 h, and a row naming TWO rounds is dropped rather than split
    (`ambiguous_runs` says so). Withhold the naming row and the same fixture must
    print the literal `join too weak to interpret` with NO number on the cost
    line: the ledger half alone is gate and review time, so printing it as *the*
    cost of a landing would understate spend by whatever the turns cost, and a
    fraction passed off as a total is worse than no figure at all.
    """
    strong, strong_text = _econ(tmp_path, name="strong", runs_rows=STRONG_RUNS)
    join = strong["run_join"]
    assert (join["k"], join["n"]) == (1, 1) and join["share"] == 1.0
    assert join["strong"] is True and join["floor"] == LO.JOIN_FLOOR
    assert "1/1 promoted landings attributed to at least one run row" in strong_text
    assert strong["cost"]["joined_seconds"] == LAND_RUN_SECONDS, (
        "the row naming two rounds was split or double-counted instead of dropped")
    assert join["ambiguous_runs"] == 1 and join["attributed_runs"] == 1
    assert strong["cost"]["joined_gpu_hours_per_landing"] == round(
        LAND_RUN_SECONDS / 3600.0, 4)
    assert "0.5 h" in strong_text

    weak, weak_text = _econ(tmp_path, name="weak",
                            runs_rows=[(9999.0, "no round named here at all")])
    assert (weak["run_join"]["k"], weak["run_join"]["n"]) == (0, 1)
    assert weak["run_join"]["strong"] is False
    cost_lines = [ln for ln in weak_text.splitlines()
                  if ln.startswith(("gpu_hours_per_landing", "joined_run_hours"))]
    assert len(cost_lines) == 2
    for ln in cost_lines:
        assert LO.JOIN_TOO_WEAK in ln, ln
        assert "0.133 h" not in ln, "a cost figure printed through a weak join"
    assert any(ln.startswith("never_landed_share") for ln in weak_text.splitlines())
    assert "0/1 promoted landings attributed to at least one run row" in weak_text

    no_db, no_db_text = _econ(tmp_path, name="no-db", runs_rows=None)
    assert no_db["run_join"]["available"] is False
    assert no_db["run_join"]["strong"] is False
    assert LO.JOIN_TOO_WEAK in no_db_text


def test_empty_window_prints_no_landings_and_n_a_and_record_keeps_the_new_keys(tmp_path,
                                                                              monkeypatch):
    """Clause 5: an empty window says so in words, and the row survives the terminal.

    A window holding a round that never landed has no cost and no latency: the
    cost lines read `0/0 (no landings)`, the latency lines read `n/a`. Not a `0` —
    which reads as a landing that cost nothing — and not a dash, which reads as no
    opinion. Then `record()` has to carry the new keys through
    `landed_outcomes_path()` into the automod state dir: a figure printed once and
    never journalled is a figure nobody can compare against next month, and the
    state dir is the only destination (#1667's owed-check ruling #3 — the guardian
    alerts hourly on runtime data left inside the checkout).
    """
    events = [e for e in _econ_events("deadbeef") if e.get("event") != "promoted"]
    row, text = _econ(tmp_path, name="empty",
                      runs_rows=[(LAND_RUN_SECONDS, f"Round {ROUND_LAND}: Implement #2001")],
                      events=events)
    cost, lat = row["cost"], row["latency"]
    assert cost["landings"] == 0 and cost["n"] == 0
    assert cost["gpu_hours_per_landing"] is None and cost["joined_seconds"] == 0.0
    assert cost["never_landed"] == {"k": int(LAND_ROUND_SECONDS + FAIL_ROUND_SECONDS),
                                    "n": int(LAND_ROUND_SECONDS + FAIL_ROUND_SECONDS)}
    key_lines = [ln for ln in text.splitlines()
                 if ln.startswith(("gpu_hours_per_landing", "joined_run_hours"))]
    assert len(key_lines) == 2
    for ln in key_lines:
        assert LO.NO_LANDINGS in ln, ln
        assert "None" not in ln and "  -  " not in ln, ln
    for key in ("filing_to_first_round", "first_round_to_landing", "filing_to_landing"):
        assert lat[key] == {"n": 0, "p50_s": None, "p90_s": None}
    pctl_lines = [ln for ln in text.splitlines()
                  if ln.startswith(("filing_to_first_round", "first_round_to_landing",
                                    "filing_to_landing"))]
    assert len(pctl_lines) == 3
    for ln in pctl_lines:
        assert ln.split(None, 1)[1].startswith("n/a") and "None" not in ln, ln
    assert "n/a" in next(ln for ln in text.splitlines()
                         if ln.startswith("queue_wait_share_median"))
    assert LO.share_text({"k": 0, "n": 0}).startswith("0/0 s")

    # `state.STATE_DIR` is bound at import (from `LLOYD_AUTOMOD_STATE`, which
    # `tests/conftest.py` already points at a scratch dir), so the destination is
    # patched on the module — setting the variable here would be read by nobody.
    from scripts.automod import state as S
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    assert str(LO.landed_outcomes_path()).startswith(str(tmp_path / "state"))
    out = tmp_path / "state" / "landed_outcomes.jsonl"
    LO.record(row, path=out)
    written = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert len(written) == 1 and written[0] == row
    assert set(["cost", "latency", "run_join"]) <= set(written[0])
    assert written[0]["cost"]["n"] == 0 and written[0]["latency"]["landings"] == 0
    LO.record(row)                       # no path: the state dir, via landed_outcomes_path
    assert len(out.read_text().strip().splitlines()) == 2
