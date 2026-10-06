"""Rollback mechanics against a real throwaway git repo.

Nothing here touches the live tree or the live supervisord: the repo is built
in `tmp_path` and supervisord is a recording fake. What the tests are really
protecting is a short list of properties that are cheap to get wrong and
expensive to get wrong in production:

  * the tree ends on `main`, not detached — `git checkout <sha>` would satisfy
    "HEAD == target" while quietly breaking the isolation model;
  * `git clean` never reaches the repo root, where usage.db / workers.db /
    .env / .venvs live, all gitignored and none replaceable;
  * a dirty tree is preserved, never silently discarded;
  * a rollback with nothing to roll back to refuses rather than guessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import gstate     # noqa: E402
import policy     # noqa: E402
import rollback as rb  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture()
def repo(tmp_path):
    """Two commits (A then B), a .gitignore, and irreplaceable root state."""
    r = tmp_path / "lloyd"
    (r / "app").mkdir(parents=True)
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")

    (r / ".gitignore").write_text("*.db\n.env\n.venvs/\n", encoding="utf-8")
    (r / "app" / "mod.py").write_text("VALUE = 'A'\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "A")
    sha_a = git(r, "rev-parse", "HEAD").stdout.strip()

    (r / "app" / "mod.py").write_text("VALUE = 'B'\n", encoding="utf-8")
    (r / "app" / "added_by_b.py").write_text("BROKEN = True\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "B")
    sha_b = git(r, "rev-parse", "HEAD").stdout.strip()

    # Gitignored, irreplaceable, at the repo root — must survive a rollback.
    (r / "usage.db").write_bytes(b"precious")
    (r / "workers.db").write_bytes(b"precious")
    (r / ".env").write_text("SECRET=1\n", encoding="utf-8")
    return {"path": r, "a": sha_a, "b": sha_b}


# ---------------------------------------------------------------------------
# restore_tree
# ---------------------------------------------------------------------------

def test_restore_lands_on_the_target_and_stays_on_main(repo):
    r = repo["path"]
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    rb.verify_tree(str(r), repo["a"])
    assert rb.head_commit(str(r)) == repo["a"]
    # The assertion that matters: `git checkout <sha>` would also satisfy the
    # line above while detaching HEAD and breaking the isolation model.
    assert rb.head_branch(str(r)) == "refs/heads/main"
    assert (r / "app" / "mod.py").read_text() == "VALUE = 'A'\n"


def test_verify_refuses_a_detached_head(repo):
    r = repo["path"]
    git(r, "checkout", "-q", "--detach", repo["a"])
    with pytest.raises(rb.RollbackError, match="expected refs/heads/main"):
        rb.verify_tree(str(r), repo["a"])


def test_verify_refuses_when_the_tree_did_not_move(repo):
    with pytest.raises(rb.RollbackError, match="after reset"):
        rb.verify_tree(str(repo["path"]), repo["a"])


def test_untracked_python_under_a_clean_path_is_removed(repo):
    """A file the bad commit left behind can shadow an import after the reset."""
    r = repo["path"]
    stray = r / "app" / "stray.py"
    stray.write_text("BOOM = 1\n", encoding="utf-8")
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    assert not stray.exists()


def test_gitignored_root_state_survives_the_clean(repo):
    """The data-loss pair. A bare `git clean -fdx` here would be an incident."""
    r = repo["path"]
    rb.restore_tree(str(r), repo["a"], policy.CLEAN_PATHS, policy.PYCACHE_PATHS)
    assert (r / "usage.db").read_bytes() == b"precious"
    assert (r / "workers.db").read_bytes() == b"precious"
    assert (r / ".env").exists()


def test_pycache_under_code_dirs_is_dropped(repo):
    r = repo["path"]
    cache = r / "app" / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_bytes(b"stale")
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    assert not cache.exists()


def test_restore_to_a_missing_commit_raises(repo):
    with pytest.raises(rb.RollbackError, match="failed"):
        rb.restore_tree(str(repo["path"]), "0" * 40, ("app",), ("app",))


def test_restore_is_idempotent(repo):
    r = repo["path"]
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    rb.verify_tree(str(r), repo["a"])


# ---------------------------------------------------------------------------
# Evidence preservation
# ---------------------------------------------------------------------------

def test_a_dirty_tree_is_preserved_before_it_is_destroyed(repo, tmp_path):
    """A rollback that erases the bug guarantees you fix it twice."""
    r = repo["path"]
    (r / "app" / "mod.py").write_text("VALUE = 'uncommitted work'\n", encoding="utf-8")
    (r / "app" / "scratch.py").write_text("notes = 1\n", encoding="utf-8")

    ev = rb.preserve_evidence(str(r), tmp_path / "broken" / "ts", "guardian-broken-ts")
    assert ev["tag"] == "guardian-broken-ts"
    assert ev["stash"], "dirty tree was not stashed"
    assert ev["patch"] and Path(ev["patch"]).read_text().strip()
    assert "app/scratch.py" in ev["untracked"]

    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    stashes = git(r, "stash", "list").stdout
    assert "guardian-rollback-guardian-broken-ts" in stashes


def test_the_tag_keeps_the_bad_commit_reachable(repo, tmp_path):
    r = repo["path"]
    rb.preserve_evidence(str(r), tmp_path / "b", "guardian-broken-x")
    rb.restore_tree(str(r), repo["a"], ("app",), ("app",))
    assert rb.commit_exists(str(r), repo["b"])
    assert "guardian-broken-x" in git(r, "tag", "-l").stdout


def test_preserve_on_a_clean_tree_records_no_stash(repo, tmp_path):
    ev = rb.preserve_evidence(str(repo["path"]), tmp_path / "b", "tag-clean")
    assert ev["stash"] is None


# ---------------------------------------------------------------------------
# index.lock
# ---------------------------------------------------------------------------

def test_a_stale_index_lock_is_removed(repo):
    lock = repo["path"] / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (time.time() - 600, time.time() - 600))
    rb._wait_for_index_lock(str(repo["path"]), stale_seconds=60.0, budget=2.0)
    assert not lock.exists()


def test_a_fresh_index_lock_is_waited_on_not_stolen(repo):
    lock = repo["path"] / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    started = time.time()
    rb._wait_for_index_lock(str(repo["path"]), stale_seconds=60.0, budget=1.0)
    assert lock.exists(), "a fresh lock must not be stolen"
    assert time.time() - started >= 0.9


# ---------------------------------------------------------------------------
# Ancestry / floor guards
# ---------------------------------------------------------------------------

def test_ancestry_helpers(repo):
    r = str(repo["path"])
    assert rb.is_ancestor(r, repo["a"], repo["b"])
    assert not rb.is_ancestor(r, repo["b"], repo["a"])
    assert rb.commit_exists(r, repo["a"])
    assert not rb.commit_exists(r, "0" * 40)


# ---------------------------------------------------------------------------
# venv swap
# ---------------------------------------------------------------------------

def test_venv_swap_back_restores_the_previous_clone(repo):
    r = repo["path"]
    venvs = r / ".venvs"
    (venvs / "lloyd").mkdir(parents=True)
    (venvs / "lloyd" / "marker").write_text("candidate", encoding="utf-8")
    (venvs / "lloyd.prev").mkdir()
    (venvs / "lloyd.prev" / "marker").write_text("previous", encoding="utf-8")

    failed = rb.swap_venv_back(str(r))
    assert (venvs / "lloyd" / "marker").read_text() == "previous"
    assert failed and Path(failed).exists(), "the failed clone must be kept for forensics"


def test_venv_swap_back_is_a_noop_without_a_previous(repo):
    (repo["path"] / ".venvs" / "lloyd").mkdir(parents=True)
    assert rb.swap_venv_back(str(repo["path"])) is None


# ---------------------------------------------------------------------------
# Worktree safety
# ---------------------------------------------------------------------------

def test_a_linked_worktree_is_untouched_by_a_rollback(repo, tmp_path):
    """The guardian must never reach into a round's worktree."""
    r = repo["path"]
    wt = tmp_path / "work"
    git(r, "worktree", "add", "-q", "-b", "automod/x", str(wt), repo["b"])
    (wt / "app" / "mod.py").write_text("VALUE = 'worktree'\n", encoding="utf-8")

    rb.restore_tree(str(r), repo["a"], policy.CLEAN_PATHS, policy.PYCACHE_PATHS)
    assert (wt / "app" / "mod.py").read_text() == "VALUE = 'worktree'\n"
    assert rb.head_commit(str(wt)) == repo["b"]


# ---------------------------------------------------------------------------
# State: rollback target resolution and its degradation ladder
# ---------------------------------------------------------------------------

def test_target_comes_from_the_lkg_pointer(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40)
    target, source = st.rollback_target()
    assert target == "a" * 40 and "last_known_good" in source


def test_target_falls_back_to_the_ledger_when_the_pointer_is_gone(tmp_path):
    st = gstate.AutomodState(tmp_path)
    gstate.append_event(st.ledger, {"event": "promoted", "commit": "b" * 40,
                                    "parent": "a" * 40})
    target, source = st.rollback_target()
    assert target == "a" * 40 and "ledger" in source


def test_a_malformed_pointer_falls_through_rather_than_crashing(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.lkg_path.parent.mkdir(parents=True, exist_ok=True)
    st.lkg_path.write_text("{not json", encoding="utf-8")
    gstate.append_event(st.ledger, {"event": "promoted", "commit": "b" * 40,
                                    "parent": "c" * 40})
    target, _ = st.rollback_target()
    assert target == "c" * 40


def test_no_target_anywhere_refuses_rather_than_guessing(tmp_path):
    """There is deliberately no fourth step in the ladder.

    A watchdog that guesses at a commit is worse than one that pages a human.
    """
    st = gstate.AutomodState(tmp_path)
    target, source = st.rollback_target()
    assert target is None and "no usable" in source


def test_a_short_or_non_hex_commit_is_not_accepted(tmp_path):
    st = gstate.AutomodState(tmp_path)
    gstate.write_json_atomic(st.lkg_path, {"schema": 1, "commit": "abc123"})
    assert st.rollback_target()[0] is None


def test_the_floor_is_set_once_and_never_moves(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40)
    st.set_lkg("b" * 40)
    assert st.floor() == "a" * 40


def test_pause_is_capped_by_the_snapshots_own_policy(tmp_path):
    """A forgotten or over-long lease must not disable the watchdog."""
    st = gstate.AutomodState(tmp_path)
    st.pause.parent.mkdir(parents=True, exist_ok=True)
    st.pause.write_text(str(time.time() + 10 * 24 * 3600), encoding="utf-8")
    assert st.pause_remaining(cap=1800.0) == pytest.approx(1800.0, abs=1.0)


def test_an_expired_pause_reads_as_zero(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.pause.parent.mkdir(parents=True, exist_ok=True)
    st.pause.write_text(str(time.time() - 5), encoding="utf-8")
    assert st.pause_remaining(cap=1800.0) == 0.0


def test_unfinished_rollback_is_detected_for_resume(tmp_path):
    st = gstate.AutomodState(tmp_path)
    gstate.append_event(st.ledger, {"event": "rollback_started", "to": "a" * 40})
    assert st.unfinished_rollback() is not None
    gstate.append_event(st.ledger, {"event": "rollback_succeeded", "restored": "a" * 40})
    assert st.unfinished_rollback() is None


def test_recent_rollbacks_counts_only_inside_the_window(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.ledger.parent.mkdir(parents=True, exist_ok=True)
    old = {"event": "rollback_succeeded", "ts": time.time() - 10 * 3600}
    recent = {"event": "rollback_succeeded", "ts": time.time() - 60}
    with open(st.ledger, "w", encoding="utf-8") as f:
        f.write(json.dumps(old) + "\n")
        f.write(json.dumps(recent) + "\n")
    assert st.recent_rollbacks(6 * 3600) == 1, "the 10h-old rollback must age out"
    assert st.recent_rollbacks(24 * 3600) == 2


# ---------------------------------------------------------------------------
# Rollback target: the promotion's own record beats the LKG pointer
#
# LKG advances only when a promotion SETTLES. Two rollbacks in a row leave it
# stranded wherever it last settled while HEAD keeps moving with ordinary
# human commits, so reverting to it discards everything landed in between.
# On 2026-09-06 that turned one false positive into 26 lost commits.
# ---------------------------------------------------------------------------

def test_the_promotions_own_target_beats_a_stranded_lkg(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40)                       # stranded hours ago
    # Deliberately distinct from `parent`, so this pins the preference order
    # rather than passing on either branch: the promoter writes
    # `rollback_target` at landing time and it is the authoritative answer.
    current = {"commit": "c" * 40, "parent": "d" * 40, "rollback_target": "b" * 40}
    target, source = st.rollback_target(current)
    assert target == "b" * 40, "must restore the tree as it was before THIS change"
    assert "rollback_target" in source


def test_the_parent_is_used_when_no_explicit_target_was_recorded(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40)
    target, source = st.rollback_target({"commit": "c" * 40, "parent": "b" * 40})
    assert target == "b" * 40 and "parent" in source


def test_without_a_promotion_under_observation_the_lkg_still_wins(tmp_path):
    """The old ladder is intact for every path that has no `current`."""
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40)
    assert st.rollback_target(None)[0] == "a" * 40
    assert st.rollback_target()[0] == "a" * 40


def test_a_malformed_current_target_falls_through_to_the_lkg(tmp_path):
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40)
    target, source = st.rollback_target({"rollback_target": "nope", "parent": ""})
    assert target == "a" * 40 and "last_known_good" in source


def test_the_26_commit_regression(tmp_path):
    """2026-09-06, verbatim. LKG settled at 14:24 and never moved; a promotion
    landed six hours later on top of a day of human commits. Reverting to the
    LKG took the whole evening with it — the voice work, the alert fan-out, a
    dashboard fix and an IDE fix, none of which the loop had any business
    judging."""
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("9a0a1d84" + "0" * 32)                      # 14:24, stale
    current = {"commit": "5cc8618a" + "0" * 32,            # the promotion
               "parent": "90b6a2d7" + "0" * 32,           # 20:00, evening work
               "rollback_target": "90b6a2d7" + "0" * 32}
    target, _ = st.rollback_target(current)
    assert target == "90b6a2d7" + "0" * 32
    assert target != st.lkg()["commit"]


# ---------------------------------------------------------------------------
# The rollback alert says what was reverted, in words
# ---------------------------------------------------------------------------

def _halt_rows(ledger: Path, event: str) -> list[dict]:
    """Ledger rows whose *subject* is the given halt transition."""
    if not ledger.exists():
        return []
    return [r for r in (json.loads(line) for line in
                        ledger.read_text(encoding="utf-8").splitlines() if line.strip())
            if r.get("event") == event]


# ---------------------------------------------------------------------------
# Halting promotions is a transition, so it belongs in the ledger too.
# #1365 clause 2: gstate.AutomodState.set_halted — the only halt writer with a
# production caller (guardian.py's flap quarantine and vault tripwire) — appends
# a halt-set row carrying its reason, and its clear names who cleared.
# ---------------------------------------------------------------------------

def test_guardian_halt_set_appends_a_row_whose_subject_is_the_halt(tmp_path):
    """The only halt writer that runs in production must leave a trace.

    `guardian.py` halts promotions from the flap quarantine (two thresholds)
    and from the vault tripwire, all through `AutomodState.set_halted`, and
    `is_halted()` gates the liveness predicate and the dashboard's `halted`
    flag on that same file, and before this change no row of the ledger named a
    halt transition at all: the 2026-09-21 21:44Z quarantine, 3 h 36 m with no
    promotions, survived only as prose inside the `alert` row appended at
    21:45:06Z. `escalate` beside it already appends after setting its flag; the
    halt path does the same now.
    """
    st = gstate.AutomodState(tmp_path)
    st.set_halted("4 rollbacks in 6h")
    assert st.is_halted()
    rows = _halt_rows(st.ledger, "promotion_halt_set")
    assert len(rows) == 1
    assert rows[0]["reason"] == "4 rollbacks in 6h"
    assert rows[0]["by"] == "guardian"
    assert rows[0]["path"] == str(st.halted), "so a reader can say what to clear"
    assert rows[0]["already_halted"] is False, "this row is the start of the freeze"


def test_guardian_halt_clear_names_the_clearer_and_skips_a_clear_that_did_nothing(tmp_path):
    """The flap alert tells a human to delete the flag; the clear must be visible.

    Clearing a quarantine is a human act and the guardian never does it, so the
    route that does exist has to record who took it — and a clear against an
    unset halt must not invent a transition that never happened.
    """
    st = gstate.AutomodState(tmp_path)
    assert st.clear_halted(by="alan, after checking the snapshot") is False
    assert not st.ledger.exists(), "no freeze was lifted, so nothing is on the record"
    st.set_halted("vault tripwire: 4,000 files deleted")
    assert st.clear_halted(by="alan, after checking the snapshot") is True
    assert not st.is_halted()
    assert [r["by"] for r in _halt_rows(st.ledger, gstate.HALT_CLEAR_EVENT)] == \
        ["alan, after checking the snapshot"]


def test_the_two_halt_writers_share_one_event_vocabulary(tmp_path, monkeypatch):
    """Two processes, one flag, one ledger — so the event names must not drift.

    `gstate` is deliberately unable to import `scripts.automod.state`: it has
    to write this state while the repo holding that module is mid-rewrite. The
    event names are therefore duplicated by hand, and a drifted spelling would
    file the guardian's halts in a bucket no reader looks at — which is the
    exact defect #1365 is about, re-opened quietly. Both writers are pointed at
    ONE ledger here, so the pairing is checked the way a reader would do it:
    one bucket over `event`.
    """
    from scripts.automod import state as S

    # Literals, not the modules' own constants: renaming both in lockstep must
    # fail here, or a reader greps the names the item named and finds nothing.
    assert (gstate.HALT_SET_EVENT, gstate.HALT_CLEAR_EVENT) == \
        ("promotion_halt_set", "promotion_halt_clear")
    assert (S.HALT_SET_EVENT, S.HALT_CLEAR_EVENT) == \
        ("promotion_halt_set", "promotion_halt_clear")

    st = gstate.AutomodState(tmp_path)
    monkeypatch.setattr(S, "HALTED_PATH", st.halted)
    monkeypatch.setattr(S, "LEDGER_PATH", st.ledger)

    S.set_halted("automod-side freeze", by="automod test")
    assert st.is_halted(), "both writers address the same flag"
    assert S.clear_halted(by="automod test") is True
    st.set_halted("guardian-side freeze")
    assert st.clear_halted(by="guardian test") is True

    events = [json.loads(line)["event"] for line in
              st.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert events == ["promotion_halt_set", "promotion_halt_clear",
                      "promotion_halt_set", "promotion_halt_clear"], \
        "one bucket, both writers, the names the item named"


def test_rollback_title_prefers_the_items_name_and_falls_back_to_hashes():
    """Read aloud, "Rolled back 1a2b3c4d to 5e6f7a8b" is sixteen letters of
    noise. The promoter writes the item's name onto current.json as `title`
    because this process is stdlib-only and cannot look it up; a record from
    before the field existed still gets the hashes, never the round id."""
    import guardian  # noqa: E402  (GUARDIAN_DIR is on sys.path above)
    current = {"round_id": "SM_20260909_160856",
               "title": "Audit deferred tool descriptions for trigger conditions"}
    t = guardian.rollback_title(current, "edc3ce7dae94", "a223a2ed2613")
    assert t == "Rolled back: Audit deferred tool descriptions for trigger conditions"
    assert "SM_" not in t and "edc3ce7d" not in t
    assert guardian.rollback_title({"round_id": "SM_X"}, "edc3ce7dae94", "a223a2ed2613") == "Rolled back edc3ce7d → a223a2ed"
    assert guardian.rollback_title(None, None, "a223a2ed2613") == "Rolled back ? → a223a2ed"


# ---------------------------------------------------------------------------
# Which commit a rollback BLAMES, and which route it takes there (#1358).
#
# 2026-09-21, twice. The detached regression check asked for a rollback of an
# older, already-settled promotion while a NEWER promotion sat in `current.json`
# under observation. `do_rollback` read the requested commit only when
# `current.json` was absent (`if not current and explicit_commit:`), so the
# blamed commit was dropped and the promotion under observation became the bad
# commit: HEAD *was* that promotion, `surgical` came out False, and
# `git reset --hard` to the request's target deleted BOTH promotions while the
# `rollback_succeeded` row named the newer one. Ledger, re-read from
# `~/.local/state/lloyd-automod/promotions.jsonl`:
#   20:35:41Z rollback_requested commit=a802b979 → rollback_succeeded
#     commit=1e219da9 route=reset restored=c1ca704e
#   21:44:52Z rollback_requested commit=dbec85aa → 21:45:05Z rollback_succeeded
#     commit=edc8ec60 route=reset restored=c1ca704e
# Honouring the requested commit makes the route a surgical `revert` of one
# commit, and leaves the promotion nobody blamed standing.
#
# Everything below drives the real `Guardian.do_rollback` against a throwaway
# repo, so the four process boundaries the routing reaches are all in play: the
# git tree, `denied.json`, `last_known_good.json` and `promotions.jsonl`.
# ---------------------------------------------------------------------------

class _RecordingSup:
    """supervisord as a recorder. Which programs a rollback stops and starts,
    in what order, is `policy.RESTART_ORDER` and the staging gate's business
    (`agent-services/bin/guardian-stage.sh`, tested in
    `test_guardian_selftest.py`); what these tests are about is which COMMIT
    the rollback moved."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def stop(self, program, wait=False):
        self.calls.append(("stop", program))
        return True, "stopped"

    def start(self, program, wait=False):
        self.calls.append(("start", program))
        return True, "started"


def _promotion_repo(tmp_path):
    """base → OLDER (settled, and the commit a later check blames) → NEWER
    (under observation, and HEAD): the 2026-09-21 shape exactly. The two
    promotions touch disjoint files, so a revert of one cannot conflict."""
    r = tmp_path / "lloyd"
    (r / "app").mkdir(parents=True)
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / ".gitignore").write_text("*.db\n.env\n.venvs/\n", encoding="utf-8")
    (r / "app" / "base.py").write_text("BASE = 1\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    out: dict = {"path": r}
    for rel, msg in (("older.py", "older item"), ("newer.py", "newer item")):
        (r / "app" / rel).write_text(f"{rel.split('.')[0].upper()} = 'x'\n",
                                     encoding="utf-8")
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", msg)
        out[rel.split(".")[0]] = git(r, "rev-parse", "HEAD").stdout.strip()
    out["base"] = git(r, "rev-list", "--max-parents=0", "HEAD").stdout.strip()
    return out


def _human_commit(r, rel):
    """A nightly job's commit, made straight to live `main` after a landing."""
    (r / "app" / rel).write_text(f"{rel.split('.')[0].upper()} = 1\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", f"nightly: {rel}")
    return git(r, "rev-parse", "HEAD").stdout.strip()


def _fake_guardian(tmp_path, repo_path, monkeypatch):
    """A Guardian with no systemd, no supervisord and no HTTP, in front of a
    real git repo and a real state dir.

    `probes.probe` reports the repo's LIVE HEAD as the running commit, which is
    what `/health` does in production — so the post-restart check that the tree
    actually moved is exercised rather than stubbed away.
    """
    import guardian

    monkeypatch.setattr(guardian.probes, "probe", lambda url, timeout: {
        "status": 200, "url": url,
        "body": {"commit": rb.head_commit(str(repo_path)), "boot_id": "boot-2"}})
    monkeypatch.setattr(guardian.probes, "wait_healthy",
                        lambda url, budget, timeout, on_tick=None: (True, {"status": 200}))

    g = guardian.Guardian.__new__(guardian.Guardian)
    g.repo = str(repo_path)
    g.state = gstate.AutomodState(tmp_path / "state")
    g.sup = _RecordingSup()
    g.programs = list(policy.RESTART_ORDER)
    g.probe_fail, g.probe_timeout, g.start_history = {}, {}, {}
    g.quiet_until = 0.0
    g.backend_url = "http://127.0.0.1:9/health"
    g.mcp_url = "http://127.0.0.1:9/health"
    g._alert_seen = {}
    g.last_alert = ""
    g.started_ts = 0.0
    g.alerts: list[dict] = []
    g.alert = lambda level, title, body, **kw: g.alerts.append(
        {"level": level, "title": title, "body": body, **kw})
    g._beat = lambda: None
    # Enough more of the loop's own state that a test can call `tick` and reach
    # the rollback branch through it, not only through `do_rollback`.
    g.tick_n = 1
    g.sup_down_streak = 0
    g.probe_http = {}
    g.probe_degraded = {}
    g.chronic = set()
    g.last_errors_at = {}
    g.mem = None
    monkeypatch.setattr(rb, "_drain_writers", lambda *a, **kw: [])
    return g


def _settle(st, *commits):
    """Advance LKG the way `maybe_settle` does, in order, so `floor` is the
    first one — the value it really has in production, where it was pinned long
    ago and never moves."""
    for c in commits:
        st.set_lkg(c)


def _observe(st, repo, **over):
    """`current.json` for the NEWER promotion: landed, window still open."""
    rec = {
        "schema": 1, "state": "observing",
        "commit": repo["newer"], "parent": repo["older"],
        "rollback_target": repo["older"],
        "round_id": "SM_NEWER", "title": "Newer promotion still under observation",
        "changed_paths": ["app/newer.py"], "boot_id": "boot-1",
        "errors_until_ts": time.time() + 900,
    }
    rec.update(over)
    gstate.write_json_atomic(st.current_path, rec)


def _regression_request(g, repo, commit):
    """The call the detached regression check actually makes: target is the
    blamed commit's parent, `commit` is the blamed commit, `changed_paths` are
    its own — and it can land minutes after the check chose its subject."""
    return g.do_rollback("regression",
                         f"retrieval-quality regression after {commit[:8]} [floors] ...",
                         explicit_target=git(repo["path"], "rev-parse",
                                             f"{commit}^").stdout.strip(),
                         explicit_commit=commit,
                         explicit_changed=["app/older.py"])


def _row(g, event):
    rows = _halt_rows(g.state.ledger, event)
    assert len(rows) == 1, [r.get("event") for r in
                            _halt_rows(g.state.ledger, "rollback_succeeded")]
    return rows[0]


def test_a_request_blaming_an_older_settled_promotion_reverts_only_it(tmp_path, monkeypatch):
    """Clause 1 of #1358: the request's commit is the one that gets reverted.

    `current.json` names the newer promotion and the request names the older
    settled one — the exact 20:35:41Z and 21:44:52Z shapes. The route must be
    `revert`, the row must name the requested commit, `restored` must be the
    revert commit, and the newer promotion's change must still be in the tree.
    """
    repo = _promotion_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"], repo["older"])
    _observe(g.state, repo)

    assert _regression_request(g, repo, repo["older"]) is True

    row = _row(g, "rollback_succeeded")
    head = rb.head_commit(str(repo["path"]))
    assert row["route"] == "revert", row
    assert row["commit"] == repo["older"], "the ledger names the commit the request named"
    assert row["restored"] == head, "`restored` is the new revert commit"
    assert row["head_before"] == repo["newer"]
    assert head != repo["base"], "the tree was not reset back to the target"
    assert rb.is_ancestor(str(repo["path"]), repo["newer"], head), \
        "the promotion under observation is still in this history"
    assert not (repo["path"] / "app" / "older.py").exists(), "the blamed change is gone"
    assert (repo["path"] / "app" / "newer.py").read_text(encoding="utf-8") == \
        "NEWER = 'x'\n", "the newer promotion's change is still present in the tree"
    assert (repo["path"] / "app" / "base.py").read_text(encoding="utf-8") == "BASE = 1\n"
    assert [c for c, _ in g.sup.calls if c == "stop"] and \
        [c for c, _ in g.sup.calls if c == "start"], "the stack was restarted, not just rewound"


def test_a_rollback_request_the_worker_writes_reaches_the_blamed_commit(
        tmp_path, monkeypatch):
    """The whole seam, end to end: the quality worker writes a request file, the
    guardian's own tick reads it and reverts the commit that file NAMES.

    The routing fix sits two calls in from where the request arrives, so a test
    that calls `do_rollback` itself never crosses the seam the incident crossed:
    `workers/sources/automod_regression.py` calls `state.request_rollback(target
    =baseline_commit, commit=commit)` — the BASELINE as target, the blamed
    promotion as commit — and the target is the one argument that must NOT
    decide the route. Here the request is written by the real writer into a real
    file, read by the real reader from inside the real `tick`, and both rows
    land in the one ledger a reader would grep.

    `tick`'s collection side (log tails, the vault tripwire, memory pressure) is
    stubbed; the request → blame → route → revert chain is not.
    """
    from scripts.automod import state as S

    repo = _promotion_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _observe(g.state, repo)

    state_dir = g.state.dir
    monkeypatch.setattr(S, "ROLLBACK_REQUEST_PATH",
                        state_dir / "rollback_request.json")
    monkeypatch.setattr(S, "LEDGER_PATH", state_dir / "promotions.jsonl")
    # The tick's collection side, all of it about logs/vault/memory, none about
    # routing — and the snapshot shape the guardian's own predicate tests use.
    g.collect = lambda: {"supervisord": "running", "procs": {}, "probes": {},
                         "now": time.time()}
    g.drain_logs = lambda: None
    g.check_vault = lambda: None
    g.check_data = lambda: None
    g.check_memory = lambda: None

    S.request_rollback(
        reason=f"retrieval-quality regression after {repo['older'][:8]} [floors] "
               "recall@5 fell past a defended floor",
        trigger="regression", target=repo["base"], commit=repo["older"],
        changed_paths=["app/older.py"])

    assert g.tick() == "rolling_back"

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "revert", row
    assert row["commit"] == repo["older"], \
        "the ledger names the commit the request named, not the one under observation"
    assert rb.is_ancestor(str(repo["path"]), repo["newer"], row["restored"]), \
        "the promotion under observation is still in history"
    assert (repo["path"] / "app" / "newer.py").exists()
    assert [r["event"] for r in _halt_rows(g.state.ledger, "rollback_requested")
            + [row]] == ["rollback_requested", "rollback_succeeded"], \
        "request and outcome in ONE ledger, which is what a reader greps"
    assert g.state.current() is None


def test_the_denylist_and_the_alert_name_the_blamed_commit(tmp_path, monkeypatch):
    """Clause 2 of #1358: the promotion that was observed but not blamed must
    stay off the denylist, and the alert must not point at it either.

    Denying the promotion under observation left the actual regressing commit
    un-denied — free to be re-derived and re-landed — while a change nobody
    blamed was blocked by SHA *and* by content hash.
    """
    repo = _promotion_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"], repo["older"])
    _observe(g.state, repo)

    _regression_request(g, repo, repo["older"])

    denied = json.loads(g.state.denied.read_text(encoding="utf-8"))
    assert denied["commits"] == [repo["older"]]
    assert repo["newer"] not in denied["commits"], \
        "the promotion under observation was not blamed, so it is not denylisted"
    assert denied["trees"], "the content hash is the half that stops a re-cut"

    fired = [a for a in g.alerts if "Rolled back" in a["title"]]
    assert len(fired) == 1, g.alerts
    alert = fired[0]
    assert alert["commit"] == repo["older"]
    assert repo["older"][:8] in alert["body"]
    assert repo["newer"][:8] not in alert["body"], alert["body"]
    # The item title on `current.json` belongs to the promotion that was NOT
    # reverted, so the headline cannot use it: a request carries no title.
    assert "Newer promotion" not in alert["title"], alert["title"]


def test_lkg_is_repointed_when_the_rollback_removed_the_commit_it_named(
        tmp_path, monkeypatch):
    """Clause 3 of #1358: `last_known_good.json` may not name a commit this
    rollback undid.

    dbec85aa settled at 21:37:32Z, so LKG named it; the detached check reverted
    it at 21:45:05Z and nothing repointed the pointer. `set_lkg` was called only
    from `maybe_settle`, so with `current.json` absent
    `gstate.rollback_target(None)` handed the next rollback that same reverted
    commit as its target — which would have reset `main` onto it and discarded
    every commit since.
    """
    repo = _promotion_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"], repo["older"])
    _observe(g.state, repo)
    assert g.state.rollback_target(None)[0] == repo["older"], "LKG names the blamed commit"

    _regression_request(g, repo, repo["older"])

    row = _row(g, "rollback_succeeded")
    lkg = g.state.lkg()
    assert lkg["commit"] == row["restored"]
    assert not g.state.current_path.exists()
    target, source = g.state.rollback_target(None)
    assert target == row["restored"], source
    assert target != repo["older"], "the reverted commit can no longer be a rollback target"
    assert lkg["floor"] == repo["base"], "the floor is pinned and does not move with a rollback"


def test_a_rollback_that_left_lkg_in_history_does_not_move_it(tmp_path, monkeypatch):
    """The repoint is for a pointer the rollback broke, not for every rollback.

    On a reset back to the promotion's own parent, LKG is an ancestor of what
    was restored and still describes a live, in-effect commit. Advancing it
    there would rewrite a verdict the guardian never earned.
    """
    repo = _promotion_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe(g.state, repo)

    assert g.do_rollback("crash", "backend not answering") is True

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "reset" and row["commit"] == repo["newer"]
    assert rb.head_commit(str(repo["path"])) == repo["older"] == row["restored"]
    assert g.state.lkg()["commit"] == repo["base"], "untouched"
    assert g.state.rollback_target(None)[0] == repo["base"]


def test_rolling_back_the_promotion_under_observation_still_resets_when_head_is_it(
        tmp_path, monkeypatch):
    """Clause 4 of #1358: the route is chosen by whether HEAD *is* the blamed
    commit, not by whether `current.json` exists. A crash rollback of the
    promotion being observed, with the tree not yet moved, still resets."""
    repo = _promotion_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe(g.state, repo)

    assert g.do_rollback("crash", "backend not answering") is True

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "reset"
    assert row["commit"] == repo["newer"]
    assert rb.head_commit(str(repo["path"])) == repo["older"]
    assert not (repo["path"] / "app" / "newer.py").exists()
    assert row["left_unjudged"] is None, "the record closed is the record that was blamed"


def test_a_human_commit_on_top_moves_the_rollback_to_a_surgical_revert(
        tmp_path, monkeypatch):
    """Clause 4 again, other side: the tree moved past the promotion under
    observation, so the same blame reverts instead of resetting past the
    nightly work the loop never promoted."""
    repo = _promotion_repo(tmp_path)
    human = _human_commit(repo["path"], "human.py")
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe(g.state, repo)

    assert g.do_rollback("error_rate", "novel traceback spike") is True

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "revert"
    assert row["commit"] == repo["newer"] and row["head_before"] == human
    assert (repo["path"] / "app" / "human.py").exists(), \
        "a nightly commit the loop never promoted survives"
    assert not (repo["path"] / "app" / "newer.py").exists()


def test_a_request_naming_a_settled_promotion_with_no_current_record_reverts_it(
        tmp_path, monkeypatch):
    """Clause 4's last limb, and the case the settled-request path was built
    for: no `current.json` at all, so the request's own commit decides the
    route — revert while the tree has moved on."""
    repo = _promotion_repo(tmp_path)
    human = _human_commit(repo["path"], "human.py")
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"], repo["older"])
    assert not g.state.current_path.exists(), "the window closed an hour ago"

    assert _regression_request(g, repo, repo["older"]) is True

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "revert" and row["commit"] == repo["older"]
    assert row["head_before"] == human
    assert not (repo["path"] / "app" / "older.py").exists()
    assert (repo["path"] / "app" / "newer.py").exists()
    assert (repo["path"] / "app" / "human.py").exists()


def test_a_request_naming_a_commit_not_in_this_history_does_not_rewind_again(
        tmp_path, monkeypatch):
    """The absence check must judge the blamed commit, not whatever record
    happens to be on disk.

    A stale duplicate of a request whose commit an earlier rollback already
    removed used to be judged against `current.json`'s commit — which *is* in
    the history — so it passed every check and rewound the tree a second time,
    discarding work this loop never touched.
    """
    repo = _promotion_repo(tmp_path)
    git(repo["path"], "checkout", "-q", "-b", "side")
    ghost = _human_commit(repo["path"], "ghost.py")
    git(repo["path"], "checkout", "-q", "main")
    assert not rb.is_ancestor(str(repo["path"]), ghost, repo["newer"])

    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"], repo["older"])
    _observe(g.state, repo)
    before = rb.head_commit(str(repo["path"]))

    assert _regression_request(g, repo, ghost) is False

    assert rb.head_commit(str(repo["path"])) == before, "the tree did not move"
    assert not g.state.denied.exists(), "nothing was denylisted"
    assert g.state.current_path.exists(), \
        "the promotion still under observation keeps its record"
    fired = [a for a in g.alerts if "no longer in this history" in a["title"]]
    assert len(fired) == 1, g.alerts
    assert ghost[:8] in fired[0]["body"], "the alert names the commit that is gone"


def test_a_surgical_revert_closes_the_observation_window_unjudged(
        tmp_path, monkeypatch):
    """Alan's decision on #1358 (2026-09-23): after a surgical revert of an
    older promotion the newer one's window CLOSES UNJUDGED — it is not settled,
    so LKG does not advance to it — and the ledger row says so beside the
    reverted commit. Judging it on the errors this rollback's own restart makes
    would convict it of the rollback; every rollback so far has been a false
    positive, and the detached regression check still measures it."""
    repo = _promotion_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"], repo["older"])
    _observe(g.state, repo)

    _regression_request(g, repo, repo["older"])

    row = _row(g, "rollback_succeeded")
    assert row["commit"] == repo["older"]
    assert row["left_unjudged"] == repo["newer"]
    assert not g.state.current_path.exists(), "the window is closed"
    assert g.state.lkg()["commit"] != repo["newer"], "closing unjudged is not settling"
    assert not g.state.last_settled.exists(), "it never settled, so nothing claims it did"


# ---------------------------------------------------------------------------
# The land train: one record, several landings (`commits`, BATCH_SCHEMA 2)
#
# A flush restarts once for every merged landing the train holds and writes
# ONE record naming them all, oldest first, with the oldest one's parent as the
# rollback target. What a rollback of that record may take off `main` is the
# property Alan cares about most: exactly the batch, never a commit the loop
# did not promote — and 127 of 379 first-parent commits on `main` in the week
# to 2026-09-24 were not promotions, so a batch routinely straddles one.
# ---------------------------------------------------------------------------

def _batch_repo(tmp_path, *, foreign=False, foreign_edits_one=False):
    """base → ONE → [FOREIGN] → TWO. ONE and TWO are the batch; FOREIGN is a
    nightly job's commit the loop never promoted."""
    r = tmp_path / "lloyd"
    (r / "app").mkdir(parents=True)
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / ".gitignore").write_text("*.db\n.env\n.venvs/\n", encoding="utf-8")
    (r / "app" / "base.py").write_text("BASE = 1\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    out: dict = {"path": r, "base": git(r, "rev-parse", "HEAD").stdout.strip()}

    def commit(rel, text, msg):
        (r / "app" / rel).write_text(text, encoding="utf-8")
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", msg)
        return git(r, "rev-parse", "HEAD").stdout.strip()

    out["one"] = commit("one.py", "ONE = 'x'\n", "landing one")
    if foreign:
        out["foreign"] = (commit("one.py", "ONE = 'changed by a human'\n", "nightly: one.py")
                          if foreign_edits_one else commit("human.py", "HUMAN = 1\n",
                                                           "nightly: human.py"))
    out["two"] = commit("two.py", "TWO = 'x'\n", "landing two")
    return out


def _observe_batch(st, repo, **over):
    rec = {
        "schema": 2, "state": "observing",
        "commit": repo["two"], "commits": [repo["one"], repo["two"]],
        "parent": repo["base"], "rollback_target": repo["base"],
        "round_id": "SM_TWO", "title": "2 landings: one; two",
        "changed_paths": ["app/one.py", "app/two.py"],
        "entries": [{"commit": repo["one"], "changed_paths": ["app/one.py"]},
                    {"commit": repo["two"], "changed_paths": ["app/two.py"]}],
        "boot_id": "boot-1", "restart": True,
        "errors_until_ts": time.time() + 300,
    }
    rec.update(over)
    gstate.write_json_atomic(st.current_path, rec)


def test_batch_commits_reads_only_a_real_batch():
    a, b = "a" * 40, "b" * 40
    assert gstate.batch_commits({"commits": [a, b]}) == [a, b]
    assert gstate.batch_commits({"commits": [a]}) == [], "one commit is today's record"
    assert gstate.batch_commits({}) == [] and gstate.batch_commits(None) == []
    assert gstate.batch_commits({"commits": [a, "nope"]}) == [], "a malformed batch is none"
    assert gstate.batch_commits({"commits": [a, a]}) == []
    assert gstate.batch_commits({"commits": "a"}) == []
    assert gstate.BATCH_SCHEMA == 2


def test_range_is_exactly(tmp_path):
    repo = _batch_repo(tmp_path)
    r = str(repo["path"])
    assert rb.range_is_exactly(r, repo["base"], repo["two"], [repo["one"], repo["two"]])
    assert not rb.range_is_exactly(r, repo["base"], repo["two"], [repo["two"]])
    assert not rb.range_is_exactly(r, repo["one"], repo["two"], [repo["one"], repo["two"]])
    assert not rb.range_is_exactly(r, "0" * 40, repo["two"], [repo["one"], repo["two"]])
    foreign = _batch_repo(tmp_path / "f", foreign=True)
    assert not rb.range_is_exactly(str(foreign["path"]), foreign["base"], foreign["two"],
                                   [foreign["one"], foreign["two"]]), "the foreign commit is in range"


def test_a_clean_batch_range_is_reset_and_both_leave_main(tmp_path, monkeypatch):
    repo = _batch_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe_batch(g.state, repo)

    assert g.do_rollback("crash", "backend FATAL") is True

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "reset" and row["commit"] == repo["two"]
    assert row["commits"] == [repo["one"], repo["two"]] and row["batch"] == 2
    assert rb.head_commit(str(repo["path"])) == repo["base"] == row["restored"]
    assert not (repo["path"] / "app" / "one.py").exists()
    assert not (repo["path"] / "app" / "two.py").exists()
    denied = json.loads(g.state.denied.read_text(encoding="utf-8"))
    assert set(denied["commits"]) == {repo["one"], repo["two"]}
    assert len(denied["trees"]) == 2, "each landing is denied by its own content"
    started = _row(g, "rollback_started")
    assert started["commits"] == [repo["one"], repo["two"]]


def test_a_foreign_commit_inside_the_batch_moves_it_to_a_surgical_revert(tmp_path, monkeypatch):
    """The reason for the whole rule: reset to the oldest landing's parent
    would take the nightly commit between them with it."""
    repo = _batch_repo(tmp_path, foreign=True)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe_batch(g.state, repo)

    assert g.do_rollback("crash", "backend FATAL") is True

    row = _row(g, "rollback_succeeded")
    head = rb.head_commit(str(repo["path"]))
    assert row["route"] == "revert" and row["commits"] == [repo["one"], repo["two"]]
    assert row["head_before"] == repo["two"] and row["restored"] == head
    assert rb.is_ancestor(str(repo["path"]), repo["foreign"], head), "foreign commit kept"
    assert (repo["path"] / "app" / "human.py").exists()
    assert not (repo["path"] / "app" / "one.py").exists()
    assert not (repo["path"] / "app" / "two.py").exists()
    assert rb.head_branch(str(repo["path"])) == "refs/heads/main"
    assert git(repo["path"], "status", "--porcelain").stdout.strip() == ""


def test_head_moved_past_the_batch_reverts_both_and_keeps_what_came_after(tmp_path, monkeypatch):
    repo = _batch_repo(tmp_path)
    after = _human_commit(repo["path"], "after.py")
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe_batch(g.state, repo)

    assert g.do_rollback("error_rate", "novel traceback spike") is True

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "revert" and row["head_before"] == after
    assert (repo["path"] / "app" / "after.py").exists()
    assert not (repo["path"] / "app" / "one.py").exists()
    assert not (repo["path"] / "app" / "two.py").exists()


def test_a_request_blaming_one_commit_of_the_batch_reverts_that_one_only(tmp_path, monkeypatch):
    """The detached regression check measures each landing against its own
    parent and names one. It is not a verdict on the batch: that commit goes,
    the rest stays, and the batch's window closes unjudged (#1358's rule)."""
    repo = _batch_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe_batch(g.state, repo)

    assert g.do_rollback("regression", "recall fell", explicit_target=repo["base"],
                         explicit_commit=repo["one"],
                         explicit_changed=["app/one.py"]) is True

    row = _row(g, "rollback_succeeded")
    assert row["route"] == "revert" and row["commit"] == repo["one"]
    assert "commits" not in row, "a one-commit rollback writes the row it always did"
    assert not (repo["path"] / "app" / "one.py").exists()
    assert (repo["path"] / "app" / "two.py").exists()
    assert row["left_unjudged"] == repo["two"]
    denied = json.loads(g.state.denied.read_text(encoding="utf-8"))
    assert denied["commits"] == [repo["one"]]


def test_a_request_naming_a_batch_is_one_batch_rollback(tmp_path, monkeypatch):
    """The promoter hands the guardian a batch it could not undo inline as
    one request (`state.request_rollback(commits=…)`), through the real
    writer, the real file and the real tick."""
    from scripts.automod import state as S
    repo = _batch_repo(tmp_path, foreign=True)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    monkeypatch.setattr(S, "ROLLBACK_REQUEST_PATH", g.state.dir / "rollback_request.json")
    monkeypatch.setattr(S, "LEDGER_PATH", g.state.dir / "promotions.jsonl")
    g.collect = lambda: {"supervisord": "running", "procs": {}, "probes": {}, "now": time.time()}
    g.drain_logs = g.check_vault = g.check_data = g.check_memory = lambda: None

    S.request_rollback(reason="flush_failed: backend never came up", trigger="flush_failed",
                       target=repo["base"], commits=[repo["one"], repo["two"]])
    assert g.tick() == "rolling_back"

    row = _row(g, "rollback_succeeded")
    assert row["commit"] == repo["two"] and row["commits"] == [repo["one"], repo["two"]]
    assert row["route"] == "revert", "the foreign commit is inside the range"
    assert (repo["path"] / "app" / "human.py").exists()
    assert S.reverted_commits(S.read_events(limit=50)) >= {repo["one"], repo["two"]}


def test_a_conflicting_batch_revert_puts_the_tree_back_and_escalates_once(tmp_path, monkeypatch):
    """A foreign commit that rewrote a batch landing's lines: no automatic
    answer is safe. The revert of TWO it already made is taken back off, the
    tree is exactly where it was, and it is not retried."""
    repo = _batch_repo(tmp_path, foreign=True, foreign_edits_one=True)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe_batch(g.state, repo)
    attempts = []
    real = rb.revert_commits
    monkeypatch.setattr(rb, "revert_commits",
                        lambda *a, **k: attempts.append(1) or real(*a, **k))

    assert g.do_rollback("crash", "backend FATAL") is False

    assert attempts == [1], "a conflict is deterministic: one attempt"
    assert rb.head_commit(str(repo["path"])) == repo["two"], "the tree was put back"
    assert git(repo["path"], "status", "--porcelain").stdout.strip() == ""
    assert g.state.is_broken()
    assert _halt_rows(g.state.ledger, "rollback_failed")


def test_a_batch_settles_with_one_row_per_commit(tmp_path, monkeypatch):
    repo = _batch_repo(tmp_path)
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    _observe_batch(g.state, repo, errors_until_ts=time.time() - 1)

    g.maybe_settle(g.state.current())

    rows = _halt_rows(g.state.ledger, "settled")
    assert [r["commit"] for r in rows] == [repo["one"], repo["two"]]
    assert all(r["batch"] == 2 and r["batch_head"] == repo["two"] for r in rows)
    assert g.state.lkg()["commit"] == repo["two"]
    assert json.loads(g.state.last_settled.read_text())["commits"] == [repo["one"], repo["two"]]
    assert not g.state.current_path.exists()


def test_a_record_without_commits_is_judged_exactly_as_before(tmp_path, monkeypatch):
    """Everything a pre-train promoter wrote, and everything written while
    `defer_restart` is off: a record naming the newest commit and an older
    rollback target resets to it (the blunt route it always took), and the
    rows carry no batch fields. A one-element `commits` is the same record."""
    for over in ({"commits": None}, {"commits": ["x"]}):
        sub = tmp_path / str(len(str(over)))
        sub.mkdir()
        repo = _batch_repo(sub)
        g = _fake_guardian(sub, repo["path"], monkeypatch)
        _settle(g.state, repo["base"])
        rec = {"commits": [repo["two"]]} if over["commits"] else {}
        _observe_batch(g.state, repo, entries=None, **rec)
        cur = json.loads(g.state.current_path.read_text())
        if not over["commits"]:
            cur.pop("commits")
            gstate.write_json_atomic(g.state.current_path, cur)

        assert g.do_rollback("crash", "backend FATAL") is True

        row = _row(g, "rollback_succeeded")
        assert row["route"] == "reset" and row["commit"] == repo["two"]
        assert "commits" not in row and "batch" not in row
        assert "commits" not in _row(g, "rollback_started")
        denied = json.loads(g.state.denied.read_text(encoding="utf-8"))
        assert denied["commits"] == [repo["two"]]
        _observe_batch(g.state, repo, errors_until_ts=time.time() - 1, commits=None)
        g.maybe_settle(g.state.current())
        assert [r["commit"] for r in _halt_rows(g.state.ledger, "settled")] == [repo["two"]]
        assert set(_halt_rows(g.state.ledger, "settled")[0]) >= {"event", "commit"}
        assert "batch" not in _halt_rows(g.state.ledger, "settled")[0]


def test_reverted_commits_counts_every_commit_of_a_batch_row():
    from scripts.automod import state as S
    a, b, c = "a" * 40, "b" * 40, "c" * 40
    rows = [{"event": "promoted", "commit": a, "parent": c},
            {"event": "promoted", "commit": b, "parent": a},
            {"event": "rollback_succeeded", "commit": b, "commits": [a, b], "route": "revert",
             "restored": "d" * 40}]
    assert S.reverted_commits(rows) == {a, b}


def test_a_batch_commit_already_gone_is_not_reverted_twice(tmp_path, monkeypatch):
    """ONE left `main` some other way (here, history rewritten under it);
    what remains of the batch is TWO alone, still judged by the batch rule:
    `base..HEAD` is exactly TWO, so it is reset — and ONE, which is no longer
    here, is neither reverted nor named."""
    repo = _batch_repo(tmp_path)
    r = str(repo["path"])
    g = _fake_guardian(tmp_path, repo["path"], monkeypatch)
    _settle(g.state, repo["base"])
    # Rewrite history so ONE is not an ancestor any more: drop it by rebase.
    git(repo["path"], "rebase", "-q", "--onto", repo["base"], repo["one"], "main")
    new_two = rb.head_commit(r)
    _observe_batch(g.state, repo, commit=new_two, commits=[repo["one"], new_two])

    assert g.do_rollback("crash", "backend FATAL") is True

    row = _row(g, "rollback_succeeded")
    assert row["commits"] == [new_two], "ONE is not in this history and is not touched"
    assert row["route"] == "reset", "base..HEAD is exactly what is left"
    assert rb.head_commit(r) == repo["base"]


# ---------------------------------------------------------------------------
# #2024: the other file-gated safety-state edges the guardian owns — BROKEN's
# create edge and the rollback request's remove edge — are on the ledger too.
# ---------------------------------------------------------------------------

def test_setting_broken_appends_a_broken_set_row_from_the_same_call(tmp_path):
    """`set_broken` was a bare `write_text`. The `escalated` row its one caller
    appends afterwards is about the incident; the flag's own transition — what
    `is_broken()` and the promoter's refusal read — had no row."""
    st = gstate.AutomodState(tmp_path)
    st.set_broken("rollback failed twice: probe dead")
    assert st.is_broken()
    rows = _halt_rows(st.ledger, "broken_set")
    assert len(rows) == 1
    assert rows[0]["reason"] == "rollback failed twice: probe dead"
    assert rows[0]["by"] == "guardian" and rows[0]["by"].strip()
    assert rows[0]["path"] == str(st.broken)
    assert rows[0]["already_broken"] is False

    st.set_broken("still dead", by="guardian nag")
    rows = _halt_rows(st.ledger, gstate.BROKEN_SET_EVENT)
    assert [r["already_broken"] for r in rows] == [False, True]
    assert rows[1]["by"] == "guardian nag"


def test_escalate_leaves_the_broken_row_before_the_escalated_row(tmp_path):
    """Through the production caller, in order: the flag's row is written by
    the call that writes the flag, so it precedes everything that reads it."""
    import types

    import guardian as G

    alerts: list[tuple] = []
    g = types.SimpleNamespace(state=gstate.AutomodState(tmp_path),
                              alert=lambda *a, **k: alerts.append(a))
    G.Guardian.escalate(g, "Rollback failed", "could not reset")
    assert alerts and "round recover" in alerts[0][2], (
        "the alert must name the ledgered clear, not the bare file")
    events = [json.loads(line)["event"] for line in
              g.state.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "broken_set" in events and "escalated" in events, events
    assert events.index("broken_set") < events.index("escalated"), events


def test_a_ledger_that_cannot_be_written_never_stops_the_escalation(tmp_path):
    """The flag is the guard's input; the row is the record. A directory where
    the ledger file should be makes every append raise — BROKEN is still set."""
    st = gstate.AutomodState(tmp_path)
    st.ledger.mkdir(parents=True)
    st.set_broken("disk trouble")
    assert st.is_broken()


def test_the_guardian_clearing_a_rollback_request_is_on_the_record(tmp_path):
    st = gstate.AutomodState(tmp_path)
    assert st.clear_rollback_request() is False
    assert not st.ledger.exists(), "nothing was cleared, so nothing is recorded"
    gstate.write_json_atomic(st.rollback_request, {"reason": "regression"})
    assert st.clear_rollback_request() is True
    assert st.read_rollback_request() is None
    rows = _halt_rows(st.ledger, "rollback_request_cleared")
    assert len(rows) == 1 and rows[0]["by"] == "guardian", rows
    assert rows[0]["path"] == str(st.rollback_request)
    assert rows[0]["created_at"].endswith("Z"), "when, as well as who"


def test_the_two_surfaces_share_the_broken_and_rollback_clear_vocabulary(
        tmp_path, monkeypatch):
    """Same arrangement as the halt pair, same reason: `gstate` cannot import
    `scripts.automod.state`, so the names are duplicated and pinned. Both
    writers on ONE ledger, bucketed on `event` as a reader would."""
    from scripts.automod import state as S

    assert (gstate.BROKEN_SET_EVENT, gstate.BROKEN_CLEAR_EVENT,
            gstate.ROLLBACK_CLEAR_EVENT) == \
        ("broken_set", "broken_cleared", "rollback_request_cleared")
    assert (S.BROKEN_SET_EVENT, S.BROKEN_CLEAR_EVENT, S.ROLLBACK_CLEAR_EVENT) == \
        ("broken_set", "broken_cleared", "rollback_request_cleared")

    st = gstate.AutomodState(tmp_path)
    monkeypatch.setattr(S, "LEDGER_PATH", st.ledger)
    monkeypatch.setattr(S, "BROKEN_PATH", st.broken)
    monkeypatch.setattr(S, "ROLLBACK_REQUEST_PATH", st.rollback_request)

    st.set_broken("escalated")                       # guardian sets
    assert S.is_broken()
    assert S.clear_broken(by="alan") is True          # automod side clears
    gstate.write_json_atomic(st.rollback_request, {"reason": "a"})
    assert st.clear_rollback_request() is True        # guardian clears
    gstate.write_json_atomic(st.rollback_request, {"reason": "b"})
    assert S.clear_rollback_request(by="round recover") is True   # automod clears

    rows = [json.loads(line) for line in
            st.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["event"] for r in rows] == [
        "broken_set", "broken_cleared",
        "rollback_request_cleared", "rollback_request_cleared"]
    assert [r["by"] for r in rows] == ["guardian", "alan", "guardian", "round recover"]


def test_the_flap_count_reads_the_whole_window_not_the_last_200_rows(tmp_path):
    """The rows #2024 adds must not push a rollback out of the flap count. The
    default `read_events` window is 200 rows — about five hours of this ledger,
    under the six-hour flap window — so the count names its own bound."""
    st = gstate.AutomodState(tmp_path)
    gstate.append_event(st.ledger, {"event": "rollback_succeeded"})
    for _ in range(300):
        gstate.append_event(st.ledger, {"event": "pause_set"})
    assert len(gstate.read_events(st.ledger)) == 200, "the default still bounds"
    assert st.recent_rollbacks(6 * 3600.0) == 1
    assert gstate.FLAP_SCAN_ROWS >= 10000


# ── #2221: the three alert-state fields answer on THIS construction ─────────
# `_fake_guardian` above (tests/test_guardian_rollback.py:581) is the reason #2221
# needed a class-level default at all: it builds a Guardian with `Guardian.__new__`
# and hand-assigns the ~12 attributes its routing scenario reads. When the two
# service-down fields first landed as `__init__`-only assignments, the two routing
# nodes above died with `'Guardian' object has no attribute
# '_sup_unreachable_open'` inside `tick()` — the failure that refused the first
# commit of this branch. This is the node that pins the contract that fixed it.

#: The three fields #2221 added, spelled as the source spells them.
_ALERT_STATE_FIELDS = ("_down_programs", "_sup_unreachable_open",
                       "liveness_fail_streak")


def test_the_alert_state_fields_answer_on_the_partial_guardian(tmp_path,
                                                               monkeypatch):
    """A Guardian built the way the routing tests build one answers "nothing is
    open" on all three #2221 fields, and answers it from the CLASS.

    Clause 5 of #2221. The distinction that matters is *where* the answer comes
    from: `_fake_guardian` assigns none of the three, so a read resolving to a
    value proves the class-level default at `agent-services/guardian/guardian.py:239-241`
    is doing the work. `__init__` rebinds the same names per instance
    (`guardian.py:309-311`), which is what makes a production Guardian's state
    independent of this default — so the assertion below that none of the three
    is in `g.__dict__` is the half that would go red the day someone moves the
    defaults back into `__init__` and this construction starts raising again.
    """
    g = _fake_guardian(tmp_path, tmp_path / "lloyd", monkeypatch)

    assert g._down_programs == frozenset()
    assert g._sup_unreachable_open is False
    assert g.liveness_fail_streak == 0

    unset = [f for f in _ALERT_STATE_FIELDS if f in g.__dict__]
    assert unset == [], (
        f"the partial build assigned {unset}, so this node is no longer proving "
        "the class-level default answers — it is proving __init__-style "
        "assignment on a hand-built object"
    )


def test_the_shared_alert_state_default_cannot_be_mutated_through_an_instance():
    """The collection default is a `frozenset`, and no write in the source mutates
    it in place — so two Guardians can never share alert state through the class.

    A mutable class-level default (`_down_programs: set = set()`) would have made
    the AttributeError go away and left something worse: `self._down_programs |=
    {...}` reaches the CLASS object, so one Guardian's service-down alert would
    name its programs on every other instance in the process, and the retraction
    at `guardian.py:1672` would clear another instance's set. #2221's whole point
    is that a retraction names the right thing, so a shared default is the same
    defect wearing a different hat.

    Three halves are checked: the object itself refuses in-place mutation, a
    per-instance rebind does not leak to the class or to a sibling instance, and
    the source is read for the one operator that would reach the class anyway. An
    `AugAssign` (`|=`) to `self._down_programs` is that operator — it is the shape
    that would survive a reviewer reading only the diff hunk, and the day the
    default is changed back to a plain `set` it stops raising and starts sharing.
    `liveness_fail_streak` is deliberately NOT in that checked list: its `+= 1` at
    `guardian.py:1656` is an `AugAssign` on an `int`, and integer `+=` rebinds the
    name on the instance, so it cannot reach the class object whatever it is
    assigned. Only a field whose value is a mutable collection can share state
    through the class, and among these three only `_down_programs` is one.
    """
    import ast

    import guardian  # noqa: E402  (GUARDIAN_DIR is on sys.path above)

    assert isinstance(guardian.Guardian._down_programs, frozenset), (
        "the shared default is a mutable "
        f"{type(guardian.Guardian._down_programs).__name__}: every instance in the "
        "process would share one set of down programs"
    )

    g = guardian.Guardian.__new__(guardian.Guardian)
    with pytest.raises(AttributeError):
        g._down_programs.add("lloyd-backend")          # type: ignore[attr-defined]

    # The pattern the source actually uses: build a NEW set, rebind the name.
    g._down_programs = set(g._down_programs) | {"lloyd-backend"}
    assert guardian.Guardian._down_programs == frozenset(), (
        "a per-instance write reached the class default"
    )
    fresh = guardian.Guardian.__new__(guardian.Guardian)
    assert fresh._down_programs == frozenset(), (
        "a second partially built Guardian inherited the first one's down programs"
    )

    src = Path(guardian.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    aug_targets = [
        getattr(node.target, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and getattr(node.target, "attr", "") == "_down_programs"
    ]
    assert aug_targets == [], (
        f"`self._down_programs op=` at {len(aug_targets)} site(s) mutates the shared "
        "class default in place; the write has to rebind a new set"
    )


# ── #2256 clause 3: the dead voice channel gets restarted, by the guardian ─
#
# Not a rollback. `do_rollback("crash", …)` was the only thing a liveness failure used
# to be able to do, and it acts on `policy.RESTART_ORDER` — which `agent-tts` is
# deliberately NOT in — by stopping the writers and rewriting main. On 2026-10-05 the
# program carrying every spoken alert sat STOPPED from 15:08:12 with nothing owning it.
#
# The bound is flap protection's own numbers, not a new invention: `FLAP_HALT_AFTER`
# inside `FLAP_WINDOW_SECONDS`, read off the same ledger (`_recent_recoveries` counts
# `service_recovery` rows for one program where flap protection counts
# `rollback_succeeded` ones). Attempts rather than successes, because a rollback that
# succeeds ends its own incident while the third start of a synthesiser that will not
# serve IS the incident.

class _TtsSup:
    """supervisord as the recovery path sees it: a `stop` that is best-effort, a
    `start` that answers `(ok, msg)`, and a record of both. Its own class rather than
    `_RecordingSup`, because that one is the rollback file's witness for which
    `RESTART_ORDER` members a rollback touched and must stay free of knobs."""

    def __init__(self, restarts_ok=True):
        self.calls: list[tuple[str, str]] = []
        self.restarts_ok = restarts_ok

    def stop(self, program, wait=False):
        self.calls.append(("stop", program))
        return True, "stopped"

    def start(self, program, wait=False):
        self.calls.append(("start", program))
        if not self.restarts_ok:
            # `startsecs=30` in agent-tts.conf means a model that cannot load lands
            # here, not on a silent port.
            return False, "ERRORED (spawn error)"
        return True, "started"

    def snapshot(self):
        return [f"{verb} {program}" for verb, program in self.calls]

    def starts(self):
        return [program for verb, program in self.calls if verb == "start"]


def _tts_recovery_guardian(tmp_path, monkeypatch, *, endpoint_answers, restarts_ok=True):
    """A Guardian whose only real behaviour is `recover_service`, with supervisord and
    the endpoint both recorded. `probes.wait_healthy` keeps the real signature —
    `on_tick=` included — so the confirmation crosses the same seam the backend's
    post-restart check does.

    The speech stage #2282 added AFTER the endpoint is stubbed here, and the node that
    removes that stub is `tests/test_guardian_tts_speech_probe.py`. Without it these three
    endpoint nodes pass only by synthesising 2.56 s of real audio through the configured
    `clone:dave_cullen` against the live :8090 — green on this box, red on a box with the
    model not loaded, and a green that depends on a running model is not a pin. The numbers
    are the ones `tests/fixtures/tts_speech_probe_calibration.json` commits, and they arrive
    as the plain return value `speak.confirm_speech` measures, not as PCM: which half of the
    confirmation the stub is standing in for is exactly what these nodes are NOT about."""
    import types

    import guardian as G

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="agent-tts", interval=5.0,
    )
    g = G.Guardian(args)
    sup = _TtsSup(restarts_ok=restarts_ok)
    g.sup = sup
    g.alerts = []
    g.alert = lambda level, title, body, **kw: g.alerts.append(
        {"level": level, "title": title, "body": body, **kw})
    g._beat = lambda: None
    g.heartbeats: list = []
    g.heartbeat = lambda state, extra=None: g.heartbeats.append((state, extra or {}))
    waited: dict = {}

    monkeypatch.setattr(G.speak, "confirm_speech",
                        lambda cfg, state_dir=None: (2.56, 2250))

    # The real signature (`probes.py:90`: `interval=1.0, on_tick=None`). A fake
    # whose extra keyword is named `pause` accepts the caller's `on_tick=` by
    # accident and would not notice a caller that stopped passing it.
    def fake_wait(url, timeout_total, probe_timeout, interval=1.0, on_tick=None):
        waited.update(url=url, timeout_total=timeout_total,
                      probe_timeout=probe_timeout, on_tick=on_tick)
        return (endpoint_answers, {"status": 200} if endpoint_answers
                else {"error": "[Errno 111] Connection refused"})

    monkeypatch.setattr(G.probes, "wait_healthy", fake_wait)
    return g, sup, waited


def test_a_down_synthesiser_is_restarted_and_confirmed_by_its_endpoint(tmp_path,
                                                                      monkeypatch):
    """Clause 3, the green path: the supervisor call is addressed to the program's own
    name and not to a `RESTART_ORDER` member, and the restart is not called recovered
    until `:8090` answers — the state-does-not-prove-serving rule #1816 settled. Since
    #2282 the endpoint is the FIRST of two confirmations and this node's helper stubs the
    second, so what is pinned here is the endpoint half and the supervisor addressing; the
    speech half is pinned in `tests/test_guardian_tts_speech_probe.py`."""
    g, sup, waited = _tts_recovery_guardian(tmp_path, monkeypatch, endpoint_answers=True)

    ok, detail = g.recover_service("agent-tts", "agent-tts: STOPPED")

    assert ok is True, detail
    assert sup.starts() == ["agent-tts"], sup.snapshot()
    assert waited["url"] == policy.TTS_HEALTH_URL, waited
    assert waited["timeout_total"] >= 240.0, (
        f"a four-minute model compile needs a real budget, got "
        f"{waited['timeout_total']}s")
    assert waited["on_tick"] is g._beat, (
        "the wait has to hand control back to the watchdog beat ITSELF: any "
        "callable would satisfy an `is not None` check while starving the unit")


def test_the_recovery_is_scoped_to_the_voice_channel(tmp_path, monkeypatch):
    """The asymmetry the fix rests on, pinned at the supervisor: recovering the voice
    channel stops and starts `agent-tts` and nothing else. A rollback's staging stops
    `RESTART_ORDER` — the writers — and that is precisely what must not happen because
    the room went quiet."""
    g, sup, _ = _tts_recovery_guardian(tmp_path, monkeypatch, endpoint_answers=True)

    assert g.recover_service("agent-tts", "agent-tts: STOPPED")[0] is True
    touched = {program for _verb, program in sup.calls}
    assert touched == {"agent-tts"}, (
        f"recovery reached outside the voice channel: {sup.snapshot()}")
    assert not (touched & set(policy.RESTART_ORDER)), sup.snapshot()


def test_the_recovery_wait_is_announced_in_the_heartbeat_before_it_begins(tmp_path,
                                                                         monkeypatch):
    """A tick that then blocks for up to seven minutes has to say so in the place a
    reader looks — `heartbeat.json`, in the row before the silence."""
    g, _sup, _waited = _tts_recovery_guardian(tmp_path, monkeypatch, endpoint_answers=True)

    assert g.recover_service("agent-tts", "agent-tts: STOPPED")[0] is True
    assert g.heartbeats, "nothing was written to heartbeat.json before the blocking wait"
    state, extra = g.heartbeats[0]
    assert "agent-tts" in state or "agent-tts" in repr(extra), (state, extra)


def test_a_restart_that_never_serves_is_reported_failed_and_then_refused(tmp_path,
                                                                        monkeypatch):
    """Clause 3's bound, on both halves.

    An endpoint that will not answer is NOT a success: the alert goes out at error
    level with `needs_human=True`, because a guardian whose speech route is dead
    cannot hear its own alert. And the attempt is in the ledger BEFORE it ran, so the
    flap count is real: the third attempt is the one flap protection refuses."""
    g, sup, _ = _tts_recovery_guardian(tmp_path, monkeypatch, endpoint_answers=False)

    ok, detail = g.recover_service("agent-tts", "agent-tts: STOPPED")
    assert ok is False, detail

    events = [ev.get("event") for ev in gstate.read_events(g.state.ledger)]
    assert events.count("service_recovery") == 1, events

    ok2, detail2 = g.recover_service("agent-tts", "agent-tts: STOPPED")
    assert ok2 is False, detail2
    assert sup.starts().count("agent-tts") == 2, (
        f"the second attempt was refused while inside the bound: {sup.snapshot()}")

    ok3, detail3 = g.recover_service("agent-tts", "agent-tts: STOPPED")
    assert ok3 is False, detail3
    assert sup.starts().count("agent-tts") == 2, (
        f"a third restart ran despite the bound: {sup.snapshot()}")
    assert str(policy.FLAP_HALT_AFTER) in detail3, detail3
    assert "rollback" in detail3.lower() or "supervisorctl" in detail3.lower(), (
        "the refusal must say what the guardian will NOT do, or name the human's fix: "
        + detail3)


def test_the_dispatch_announces_both_outcomes_of_a_recovery_attempt(tmp_path, monkeypatch):
    """Recovery is never silent, in either direction: a channel that came back is said
    `Recovered:` (the unattended-recovery visibility this file's house rule asks for),
    and one that could not be brought back is an error with `needs_human=True` — because
    the route that would have said it out loud is the thing that is down."""
    up, _sup_up, _w = _tts_recovery_guardian(tmp_path, monkeypatch, endpoint_answers=True)
    assert up._recover_infra("agent-tts: STOPPED, last stop 900s ago") == "recovered"
    assert any("Recovered: agent-tts" in a["title"] for a in up.alerts), up.alerts

    down, _sup_dn, _w2 = _tts_recovery_guardian(tmp_path / "d", monkeypatch,
                                               endpoint_answers=False)
    assert down._recover_infra("agent-tts: STOPPED, last stop 900s ago") == "needs_human"
    assert any(a["level"] == "error" and "agent-tts" in a["title"] for a in down.alerts), \
        down.alerts
    assert any(a.get("needs_human") is True for a in down.alerts), (
        "a guardian whose speech route is dead cannot hear its own alert: the failure "
        "has to be written down for a person to see")


def test_a_refused_spawn_is_reported_as_a_spawn_error_not_a_dead_port(tmp_path,
                                                                     monkeypatch):
    """supervisord refusing to spawn is a different fact from a port that will not
    answer, and the reason a reader gets has to name the one to go and look at."""
    g, _sup, waited = _tts_recovery_guardian(tmp_path, monkeypatch,
                                            endpoint_answers=False, restarts_ok=False)

    ok, detail = g.recover_service("agent-tts", "agent-tts: STOPPED")
    assert ok is False
    assert waited.get("url") is None, (
        f"supervisord refused the spawn, so :8090 must not be probed at all: a wait "
        f"recorded here would report the port and hide the spawn error ({waited})")
    assert "ERRORED" in detail, detail


def test_a_voice_outage_is_never_evidence_about_the_promoted_tree(tmp_path, monkeypatch):
    """The dispatch, not just the tool: with a recoverable program down, `tick` returns
    from the recovery branch and never reaches the rollback route.

    Before #2256 the two possible outcomes of a dead alert channel were "roll back a
    landing" and "alert that there is nothing to roll back" — both of which assume the
    outage is about the code."""
    import guardian as G

    g, sup, _waited = _tts_recovery_guardian(tmp_path, monkeypatch, endpoint_answers=True)
    monkeypatch.setattr(g.state, "current", lambda: {"commit": "abc", "state": "observing"})
    monkeypatch.setattr(g.state, "lkg", lambda: {"commit": "abc"})
    monkeypatch.setattr(g.state, "rollback_target", lambda: ("abc", "lkg"))
    monkeypatch.setattr(g.state, "is_broken", lambda: False)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(G.gstate, "AutomodState.is_quarantine_budget_available",
                        lambda self: True, raising=False)
    monkeypatch.setattr(g, "collect", lambda: {
        "now": time.time(), "supervisord": "ok", "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "evaluate_liveness",
                        lambda snap: (True, "agent-tts: STOPPED, last stop 900s ago"))
    rolled: list = []
    def _record_rollback(*a):
        rolled.append(a)
        return True

    monkeypatch.setattr(g, "do_rollback", _record_rollback)
    g.notifier = type("N", (), {"alert": lambda *a, **k: None,
                                "resolve": lambda *a, **k: None})()

    out = g.tick()

    assert rolled == [], f"a voice outage was blamed on the tree: {rolled}"
    assert sup.starts() == ["agent-tts"], sup.snapshot()
    assert out == "recovered", (
        f"`_recover_infra` returns `recovered` or `needs_human`, so any other value "
        f"means a restart whose outcome nobody recorded: {out!r}")


# ── #2256 clause 5: the witness bytes behind the premise, re-derived ───────
#
# `tests/fixtures/supervisord.log` is supervisord's own log as of 2026-10-05
# 15:08:12 — the file `~/lloyd-data/logs/services/supervisord.log`, truncated at line
# 2164 so that every line number the item cites resolves HERE to the line it cites,
# and the live file's later growth cannot shift them. The durable copy is the vault's
# `backlog/data/supervisord.log`; this one is byte-identical, and the digest is
# compared rather than quoted whenever the vault is reachable — the gate runs with
# HOME at the round home, where `~/obsidian` does not exist, so a node that only read
# the vault would pin nothing at all.

WITNESS = Path(__file__).resolve().parent / "fixtures" / "supervisord.log"

#: sha256 of the witness BYTES — a content digest, deliberately NOT a git object id,
#: so `git cat-file -t` on this value fails and that is the correct answer: nothing in
#: this repo is identified by it, and nothing should be. It is pinned because the
#: node below compares the vault's durable copy against it, and a comparison whose
#: expected side is only ever computed at run time cannot tell a reader, after the
#: fact, which bytes were checked. Re-derive: `sha256sum tests/fixtures/supervisord.log`.
WITNESS_SHA256 = ("3d76da99700b111bbbf0c4dd95b2dfa9"
                  "f2864cae28faa1e8730286f73f07f084")


def test_the_witness_log_still_shows_the_stop_with_no_spawn_after_it():
    """Clause 5. `wc -l < backlog/data/supervisord.log` answers 2164, and this node
    counts the same thing the same way (`\\n` characters — 2164 of them, the last one
    terminating line 2164) on bytes committed to this repo — then re-derives the three facts the item
    quotes from those bytes rather than restating them."""
    assert WITNESS.is_file(), f"{WITNESS} is missing; the witness has to be committed"
    raw = WITNESS.read_bytes()
    text = raw.decode()
    lines = text.split("\n")
    assert len(lines) - 1 == 2164, (
        f"{WITNESS.name} carries {text.count(chr(10))} newlines; `wc -l` on the "
        f"witness answers 2164")

    # The stop, at the line number the item cites.
    stopped = lines[2163]
    assert "2026-10-05 15:08:12,145 WARN stopped: agent-tts (terminated by SIGTERM)" \
        in stopped, f"line 2164 is not the stop the item cites: {stopped!r}"

    # Nothing spawned it afterwards: the stop is unowned, which is what
    # `autorestart=true` does not cover.
    after = lines[2164:]
    assert not [ln for ln in after
                if "spawned: 'agent-tts'" in ln], (
        "a `spawned:` did follow the 15:08:12 stop, so the premise is false and "
        "supervisord did bring it back")

    # The control the item names: the 2026-10-04 stop at :2089-2092 IS a
    # stop→spawn restart pair, so the absence above is the difference, not the
    # log's general shape.
    pair = "\n".join(lines[2088:2092])
    assert "waiting for agent-tts to stop" in pair and "WARN stopped: agent-tts" in pair \
        and "spawned: 'agent-tts' with pid 1034411" in pair \
        and "entered RUNNING state" in pair, (
        f"lines 2089-2092 are not the 10-04 restart pair: {pair!r}")

    digest = hashlib.sha256(raw).hexdigest()
    assert digest == WITNESS_SHA256, (
        f"the committed witness bytes are no longer the ones the item's figures were "
        f"re-derived from: sha256 {digest} (a content digest, not a commit)")

    vault = Path.home() / "obsidian" / "backlog" / "data" / "supervisord.log"
    if vault.is_file():
        assert hashlib.sha256(vault.read_bytes()).hexdigest() == digest, (
            "the committed witness and the vault's durable copy have diverged")


def test_the_health_vocabulary_is_per_endpoint(monkeypatch):
    """The seam `probes.ok_statuses_for` opened, crossed with the real verdict.

    The synthesiser answers `{"status": "healthy"}`; the backend and the aggregator
    answer `"ok"`. Accepting both words everywhere would have widened the `ok` the
    rollback decision reads from those two, so the accept-set is chosen per URL.
    Asserted through `probes.probe` over sockets this node serves rather than the
    table alone: the table is the design, the verdict is what the guardian consumes.
    """
    import probes

    def serve(body: str) -> str:
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        def loop():
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                conn.recv(65536)
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json"
                             b"\r\nContent-Length: " + str(len(body)).encode()
                             + b"\r\n\r\n" + body.encode())
            srv.close()

        threading.Thread(target=loop, daemon=True).start()
        return f"http://127.0.0.1:{port}/health"

    tts_url = serve('{"status": "healthy"}')
    other_url = serve('{"status": "healthy"}')      # an endpoint speaking TTS's word
    monkeypatch.setattr(probes.policy, "TTS_HEALTH_URL", tts_url)

    assert probes.probe(tts_url, 2.0)["ok"] is True, (
        "the synthesiser's own vocabulary has to count as healthy for the synthesiser")
    assert probes.probe(other_url, 2.0)["ok"] is False, (
        "200 + `healthy` must NOT read healthy to an endpoint whose word is `ok`: its "
        "probe verdict is an input to the rollback decision")
    assert probes.ok_statuses_for(tts_url) == probes.HEALTHY_OK_STATUSES
    assert probes.ok_statuses_for("http://127.0.0.1:8080/health") == probes.OK_STATUSES
