"""A round whose gate passed and whose turn is over is landed, not closed.

#278's rescue, made deterministic. That round died at its iteration cap with
the change written; two minutes later the Inner Voice observer sent "if the
gate passed, land it" into the session, and the second turn landed it. Cut 1
of senses-not-supervision (2026-09-12) switched the observer off for unattended
turns and gave its stall, budget and context nudges deterministic replacements
— but not that one. From then on a round that was gated and green when its
turn ended was aborted, and its item re-offered for a whole new round.

2026-09-18, by the ledger:

  round    item   gate started   turn ended   gate PASSED   reaper
  070314   #1129  minute 37      00:57:22     00:58:05      aborted 01:02
  083651   #1121  minute 45      02:20:53     02:25:57      aborted 02:41
  121744   #1236  minute 42      06:15:03     06:22:15      aborted 06:34
  122306   #1234  minute 40      06:14:31     06:13:44      aborted 06:14  (saw the pass, 146 s left)

All four landed on their next round — 250 of that day's 994 implement minutes
went to redoing changes that had already passed every rung.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, round as R, state as S, worktree as W
from workers.sources import _common as C
from workers.sources import autocode as I

HEAD = "08a059b3c2d1e4f5a6b7c8d9e0f1a2b3c4d5e6f7"
RID = "SM_20260918_122306"

#: The close under test, captured before `env` swaps in a stub. The reaper runs
#: in the backend and `round.abort` in whichever process calls it, so the
#: reaper-side clause can only be pinned against the real thing.
_REAL_ABORT = R.abort


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """The reaper's world: one open round with a worktree, nothing running in
    its session, no observer, the loop enabled."""
    import app.sessions_io as sio
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: True)
    monkeypatch.setattr(C, "source_inner_voice", lambda source, default=True: False)
    monkeypatch.setattr(sio, "active_sessions_snapshot", lambda: [])
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda rid: wt)
    world = {"head": HEAD, "aborted": [], "spawned": []}
    monkeypatch.setattr(W, "head", lambda path: world["head"])
    monkeypatch.setattr(R, "abort", lambda rid, reason="": world["aborted"].append(rid)
                        or {"aborted": rid})

    def spawn(argv, log, cwd=None):
        world["spawned"].append([str(a) for a in argv])
        return 424242
    monkeypatch.setattr(S, "spawn_detached", spawn)
    monkeypatch.setattr(S, "pid_alive", lambda pid: int(pid or 0) == 424242)
    fm = {"status": "in_progress", "priority": "medium", "board": "lloyd", "tags": [],
          "created": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    (d / "1234-a-thing.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n", encoding="utf-8")
    return world


def _gate(ok=True, head=HEAD):
    (S.ROUNDS_DIR / RID).mkdir(parents=True, exist_ok=True)
    (S.ROUNDS_DIR / RID / "gate.json").write_text(json.dumps(
        {"round_id": RID, "ok": ok, "head": head, "base": "9fab95d", "rungs": [
            {"name": "review", "ok": ok, "detail": "review: 4 met of 4 clause(s); fine"}]}),
        encoding="utf-8")


def _finished(outcome=None, stop_reason="stop"):
    S.append_event({"event": "backlog_implement", "item_id": 1234, "phase": "finished",
                    "round_id": RID, "session_id": "s-1234", "stop_reason": stop_reason,
                    "outcome": outcome}, path=S.LEDGER_PATH)


def _events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


DEFERRED = {"landed": False, "acceptance": "deferred", "deferred_to": [1234], "summary": "",
            "spawned": [], "clause_outcomes": []}


# ── it lands ────────────────────────────────────────────────────────────────

def test_a_passed_gate_at_the_rounds_own_commit_is_landed(env):
    """#1234's first round, to the second: the pass at 06:13:44, the turn's
    end at 06:14:31 with `acceptance: deferred`, the abort in the same second."""
    _gate()
    _finished(DEFERRED)
    out = I.reap_abandoned_rounds()
    assert env["aborted"] == [], "a change that had passed every rung was thrown away"
    assert env["spawned"] and env["spawned"][0][-3:] == ["scripts.automod.round", "land", RID][-3:]
    assert env["spawned"][0][-2:] == ["land", RID]
    assert [(r["round_id"], r["verb"]) for r in out] == [(RID, "landing")]
    marker = S.land_in_progress(RID)
    assert marker["by"] == "reaper" and marker["pid"] == 424242
    row = _events("land_rescued")[0]
    assert row["item_id"] == 1234 and row["head"] == HEAD and "gate passed at 08a059b3" in row["reason"]
    assert not _events("round_abandoned") and not _events("round_aborted")
    fm = B._split_frontmatter(next(B.BACKLOG_DIR.glob("1234-*.md")).read_text())[0]
    assert "the loop started one" in fm["activity_log"][-1]
    assert fm["status"] == "in_progress", "not handed back to the pool under its own landing"


def test_a_turn_that_died_with_no_finalizer_is_landed_too(env):
    """`max_turns` and the wall clock leave no outcome at all. The item then
    closes on the review rung's grading (`backlog.code_review_outcome`)."""
    _gate()
    _finished(None, stop_reason="max_turns")
    I.reap_abandoned_rounds()
    assert env["spawned"] and env["aborted"] == []


def test_the_second_pass_leaves_the_running_landing_alone(env):
    _gate()
    _finished(DEFERRED)
    I.reap_abandoned_rounds()
    assert I.reap_abandoned_rounds() == []
    assert len(env["spawned"]) == 1 and env["aborted"] == []


# ── it does not ─────────────────────────────────────────────────────────────

def _expect_the_old_behaviour(env):
    before = len(_events("land_rescued"))
    out = I.reap_abandoned_rounds()
    assert env["spawned"] == [] and env["aborted"] == [RID]
    assert [(r["round_id"], r.get("verb")) for r in out] == [(RID, None)]
    assert len(_events("land_rescued")) == before, "a second rescue was recorded"


def test_a_commit_made_after_the_gate_was_never_judged(env):
    _gate()
    env["head"] = "f" * 40
    _finished(DEFERRED)
    _expect_the_old_behaviour(env)


def test_a_gate_that_failed_closes_as_it_always_has(env):
    _gate(ok=False)
    _finished(DEFERRED)
    _expect_the_old_behaviour(env)


def test_a_round_that_was_never_gated_closes_as_it_always_has(env):
    # This fixture has no run spec, so `_gate_if_ungated` has no base to judge
    # a change against; with one, a never-gated change is gated instead
    # (tests/test_reaper_gates_ungated.py).
    _finished(DEFERRED)
    _expect_the_old_behaviour(env)


@pytest.mark.parametrize("verdict", ["rejected", "unnecessary"])
def test_a_turn_that_ended_on_an_item_verdict_decided_not_to_land(env, verdict):
    """Built it, gated it, measured no gain: the author's call, and a passed
    gate does not overrule it."""
    _gate()
    _finished({**DEFERRED, "acceptance": verdict, "summary": "p50 412 -> 409 ms; noise"})
    _expect_the_old_behaviour(env)


@pytest.mark.parametrize("prior", ["land_failed", "land_rescued"])
def test_one_rescue_per_round(env, prior):
    """A landing that failed is a ledger verdict the re-offer already reads;
    retrying it from here would loop on whatever refused it."""
    _gate()
    _finished(DEFERRED)
    S.append_event({"event": prior, "round_id": RID}, path=S.LEDGER_PATH)
    _expect_the_old_behaviour(env)


@pytest.mark.parametrize("blocked", ["is_halted", "is_broken", "disabled", "switch"])
def test_nothing_is_started_that_round_land_would_refuse(env, monkeypatch, blocked):
    _gate()
    _finished(DEFERRED)
    if blocked == "disabled":
        monkeypatch.setattr(S, "is_enabled", lambda repo=None: False)
    elif blocked == "switch":
        monkeypatch.setattr(I, "_source_cfg", lambda name: {"land_passed_gates": False})
    else:
        monkeypatch.setattr(S, blocked, lambda: True)
    _expect_the_old_behaviour(env)


def test_a_landing_that_cannot_start_falls_back_to_the_close(env, monkeypatch):
    _gate()
    _finished(DEFERRED)
    monkeypatch.setattr(R, "land_detached", lambda rid, by: {"error": "no worktree"})
    _expect_the_old_behaviour(env)


def test_a_gate_still_running_is_left_for_the_next_pass_and_then_landed(env):
    """The other three rounds: the turn ended first. The reaper must not touch
    a round mid-gate, and must come back for it."""
    _finished(DEFERRED)
    S.write_gate_marker(RID, pid=424242, head=HEAD, by="automod_gate")
    assert I.reap_abandoned_rounds() == [] and env["aborted"] == [] and env["spawned"] == []
    S.clear_gate_marker(RID)
    _gate()
    assert [r["verb"] for r in I.reap_abandoned_rounds()] == ["landing"]


# ── and it comes back within a retry, not an interval ───────────────────────

def test_a_held_loop_runs_the_reaper_at_the_retry_cadence(env, monkeypatch):
    """The reaper ran at turn end and then with housekeeping, up to 900 s
    later — and an open round with a passed gate is what holds the other
    slots (`_rounds_about_to_land`). #1236's gate passed at 06:22:15 and its
    round was closed at 06:34:14."""
    from workers.sources import DECLINED
    _gate()
    _finished(DEFERRED)
    monkeypatch.setattr(I, "_boot_settled", {"done": True})
    monkeypatch.setattr(I, "_housekeeping_due", lambda queue, cfg: False)
    monkeypatch.setattr(I, "_loop_is_free",
                        lambda depth=None: (False, f"{RID} passed its gate and is about to land"))
    out = asyncio.run(I.enqueue_if_due(object(), {"interval_seconds": 900}))
    assert out is DECLINED or out == DECLINED
    assert env["spawned"], "the round sat until housekeeping's next tick"


# ── and it says what the live checkout still held ───────────────────────────

def _scratch_live(tmp_path) -> Path:
    """A checkout standing in for `~/lloyd`. The reaper's whole blind spot is
    that the tree it closes a round against is not its own and not the
    worktree's, so the test needs a real second repo, not a mock."""
    def git(repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=False)
    live = tmp_path / "live"
    (live / "app").mkdir(parents=True)
    git(live.parent, "init", "-q", "-b", "main", str(live))
    git(live, "config", "user.email", "t@e.com")
    git(live, "config", "user.name", "t")
    (live / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(live, "add", "-A")
    git(live, "commit", "-q", "-m", "base")
    return live


def _abandon_note() -> str:
    """The close's own activity line. Not the last one: `_reap_round` notes the
    abandonment and then moves the item, and `set_status` appends its own line
    after it."""
    log = B._split_frontmatter(
        next(B.BACKLOG_DIR.glob("1234-*.md")).read_text(encoding="utf-8"))[0]["activity_log"]
    hits = [line for line in log if "abandoned:" in line]
    assert hits, f"the close left no note; the log is {log}"
    return hits[-1]


def test_the_reapers_close_preserves_and_names_the_live_edits_it_leaves(env, tmp_path, monkeypatch):
    """The reaper tells the item "Its work is on branch `automod/<rid>` in
    ~/lloyd" — and an orphan edit in the live checkout is NOT on that branch,
    so the one sentence meant to preserve a dead round's work points a reader
    at the artifact that cannot contain it. Of 80 `round_abandoned` rows, zero
    named a dirty path.

    With the real `round.abort` behind it, the close now copies the work into
    the round's state dir, puts the paths on its own row, and says them to the
    item beside the branch — so a reader finds the file without running
    `git status` on a tree that has moved on.
    """
    live = _scratch_live(tmp_path)
    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(R, "abort", _REAL_ABORT)
    (live / "app" / "m.py").write_text("V = 2\n", encoding="utf-8")            # tracked edit
    (live / "app" / "orphan.py").write_text("ORPHAN = 1\n", encoding="utf-8")  # untracked source
    _finished(DEFERRED)

    out = I.reap_abandoned_rounds()
    assert [(r["round_id"], r.get("verb")) for r in out] == [(RID, None)]
    row = _events("round_abandoned")[0]
    assert row["live_dirty_paths"] == ["app/m.py", "app/orphan.py"]
    dest = S.ROUNDS_DIR / RID / "live-dirty"
    assert row["live_dirty_patch"] == str(dest / "dirty.patch")
    assert Path(row["live_dirty_patch"]).is_file(), "the row named a patch that is not on disk"
    assert "+V = 2" in Path(row["live_dirty_patch"]).read_text(encoding="utf-8")
    assert (dest / "app" / "orphan.py").read_text(encoding="utf-8") == "ORPHAN = 1\n"

    note = _abandon_note()
    assert "app/m.py" in note and "app/orphan.py" in note, "the note named neither path"
    assert f"branch `automod/{RID}`" in note, "the branch is still named too"
    assert row["live_dirty_patch"] in note, "the note did not say where the bytes went"


def test_a_clean_live_tree_leaves_the_reapers_note_as_it_was(env, tmp_path, monkeypatch):
    """The old sentence, unchanged, when there is nothing extra to say — this
    must not become a note that claims preserved work on every close."""
    live = _scratch_live(tmp_path)
    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(R, "abort", _REAL_ABORT)
    _finished(DEFERRED)
    I.reap_abandoned_rounds()
    row = _events("round_abandoned")[0]
    assert row["live_dirty_paths"] == [] and "live_dirty_patch" not in row
    note = _abandon_note()
    assert "uncommitted edits" not in note
    assert f"branch `automod/{RID}`" in note
    assert not (S.ROUNDS_DIR / RID / "live-dirty").exists()


def test_a_failed_copy_is_reported_as_failed_and_never_as_a_path(env, monkeypatch):
    """The note names a destination only when one was written. When
    `preserve_live_dirt` could not write — a read-only state dir, a failed
    `git diff` — the honest sentence is that the work was NOT copied, with the
    reason; naming a placeholder destination instead sends the reader hunting
    for a file that never existed, which is the failure this feature exists to
    end, restated.

    The stub hands `live_dirty_patch` and `live_dirty_dir` over as empty
    strings rather than leaving them out. `preserve_live_dirt` omits a key it
    could not fill rather than blanking it, so the empty form is not today's
    shape — `abort` spreads `**preserved` straight into this dict, and the
    decoy is what pins the filter that decides what rides on. Left out, the two
    row assertions below would only catch a reaper that invents keys; offered
    empty, they catch one that copies keys unconditionally and then puts an
    empty destination in the row and in the note."""
    monkeypatch.setattr(R, "abort", lambda rid, reason="": {
        "aborted": rid, "live_dirty_paths": ["app/lost.py"], "live_dirty_patch": "",
        "live_dirty_dir": "", "live_dirty_error": "Read-only file system: '/state'"})
    _finished(DEFERRED)
    I.reap_abandoned_rounds()
    row = _events("round_abandoned")[0]
    assert row["live_dirty_paths"] == ["app/lost.py"]
    assert "live_dirty_patch" not in row and "live_dirty_dir" not in row, (
        "an empty string is not a path; the row must not carry one as a destination")
    assert row["live_dirty_error"] == "Read-only file system: '/state'"
    note = _abandon_note()
    assert "`app/lost.py`" in note, "the path still has to be named even unsaved"
    assert "NOT copied" in note and "Read-only file system" in note
    assert "copied to" not in note, "the note named a destination that was never written"
    assert "None" not in note, "the note named a placeholder instead of a path"


def test_a_directory_that_holds_nothing_is_not_reported_as_where_the_work_went(
        env, monkeypatch):
    """`preserve_live_dirt` makes the `live-dirty/` directory before it runs
    `git diff`, so a close can hand over a real directory and no patch: the
    diff read failed. Pointing a person at that directory as where their edits
    "were copied" is the same overstatement as naming a blank, and harder to
    disprove — the path resolves, and it is empty. The directory still rides on
    the row, because it is real and a later reader may well find files in it;
    what is withheld is the note's claim that the work went there."""
    monkeypatch.setattr(R, "abort", lambda rid, reason="": {
        "aborted": rid, "live_dirty_paths": ["app/m.py"],
        "live_dirty_dir": "/state/rounds/x/live-dirty",
        "live_dirty_error": "git diff HEAD rc=128"})
    _finished(DEFERRED)
    I.reap_abandoned_rounds()
    row = _events("round_abandoned")[0]
    assert row["live_dirty_dir"] == "/state/rounds/x/live-dirty"
    assert "live_dirty_patch" not in row
    note = _abandon_note()
    assert "`app/m.py`" in note
    assert "copied to" not in note, "an empty directory was named as a destination"
    assert "NOT copied" in note and "git diff HEAD rc=128" in note
