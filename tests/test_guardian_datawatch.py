"""The data-root tripwire, and what it must refuse to alarm on (#3095).

`~/lloyd-data` is watched file by file. That is what makes it the root that can
be lost — the 2026-09-22 data-home move, the stores that live nowhere else — and
it is also why a scratch directory is dangerous there: the guardian counts a
top-level folder going from present to absent as a wiped store, and a trip on the
data root is not a notification. It pauses the worker pool, halts promotions and
refuses snapshots.

At 14:24:14 on 2026-10-20 it fired critical on `top-level folder 'triage-3087'
(3347 files) vanished`. That folder was a `git worktree add` an autotriage
session had run 55 seconds earlier to get a tree the test suite would accept,
and the "loss" was its own clean-up: the before/after measurement differed in
nothing else, `event_logs` +1. The alert was correct about the arithmetic and
wrong about what it meant, and the only thing that made it wrong was that the
measurement had no word for "this folder was code".

So the exemption is a MEASUREMENT, not a name: `measure` records which top-level
folders held a `.git` entry, and a folder the last healthy measurement saw with
one inside it is scratch by what it contained, not by what it was called. Every
node here is one side of that trade — the scratch that must not trip, and the
store that still must.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import datawatch as D  # noqa: E402
import vaultwatch as VW  # noqa: E402

#: Under every threshold the whole-tree rules can fire at (`DROP_MIN` is 200), so
#: a folder going missing can only be caught by the rule under test.
STORE_FILES = 30


def _root(tmp_path: Path, name: str = "data") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / D.ROOT_MARKER).write_text("real data root\n", encoding="utf-8")
    return root


def _fill(dir_: Path, n: int) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dir_ / f"f{i}.txt").write_text(f"{i}\n", encoding="utf-8")
    return dir_


def _watch(root: Path, tmp_path: Path) -> D.DataWatch:
    w = D.DataWatch(root=str(root), state_dir=tmp_path / "state")
    w.clear()          # arms it on the tree as it is now, exactly as a fresh guardian does
    return w


def _worktree(root: Path, rel: str) -> Path:
    """A real `git worktree add` under `root`, which is the incident's own command."""
    main = root.parent / "main-checkout"
    main.mkdir(parents=True, exist_ok=True)

    def git(*a, cwd=main):
        subprocess.run(["git", *a], cwd=str(cwd), check=True, capture_output=True,
                       text=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (main / "app").mkdir()
    # More than TOPDIR_MIN_FILES, or the folder the incident measured would not
    # even be a candidate for the rule under test.
    for i in range(25):
        (main / "app" / f"m{i}.py").write_text("x\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "--detach", str(target), "HEAD")
    return target


# ── the folder that was code, not data ───────────────────────────────────────

def test_measure_records_which_top_level_folders_are_checkouts(tmp_path):
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    (root / "triage-3087" / "check").mkdir(parents=True)
    (root / "triage-3087" / "check" / ".git").write_text("gitdir: /somewhere\n",
                                                         encoding="utf-8")
    _fill(root / "triage-3087" / "check" / "app", 25)

    snap = VW.measure(str(root), skip_top=D.CACHE_DIRS)
    assert snap.checkouts == ("triage-3087",), (
        "the flag is on the measurement: by the time a folder is being judged it is "
        "already gone, and nothing else can say what it held")
    assert "sessions" not in snap.checkouts


def test_a_vanished_folder_the_last_measurement_saw_as_a_checkout_does_not_trip(tmp_path,
                                                                                capsys):
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    _worktree(root, "triage-3087/check")
    w = _watch(root, tmp_path)

    shutil.rmtree(root / "triage-3087")
    reason, _snap = w.tick(now=time_plus(w.history[-1].ts, 60))
    assert reason is None, (
        "the walk-back of a scratch checkout is not data loss, and the incident this "
        "node names paused the worker pool and halted promotions over it")
    out = capsys.readouterr().out
    assert "triage-3087" in out, "not tripping is half of it; it still has to be said"
    rows = (tmp_path / "state" / VW.EXCUSED_LOG).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["watch"] == "data" and "triage-3087" in row["excused"]


def test_a_folder_holding_the_checkout_one_level_down_is_the_same_case(tmp_path):
    """`~/lloyd-data/triage-2918/checkout/.git`, the second leftover of 2026-10-20."""
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    _worktree(root, "triage-2918/checkout")
    w = _watch(root, tmp_path)
    shutil.rmtree(root / "triage-2918")
    assert w.tick(now=time_plus(w.history[-1].ts, 60))[0] is None


def time_plus(base: float, seconds: int) -> float:
    return base + seconds


# ── the 2026-10-20 measurement, replayed ─────────────────────────────────────
# The marker that held the real before/after pair was deleted during recovery and
# the incident directory kept only `cwd-in-vault.txt` and `ps.txt`, so the numbers
# below are transcribed from the alert that fired (`ALERT.md`) and from this
# item's own measurement of `event_logs`. Nobody can re-read the original: this
# fixture IS the replay, and it is the only copy of those bytes that has a
# history.

INCIDENT_BEFORE = {
    "ts": 1_760_988_254.0,          # 2026-10-20 14:24:14 PDT
    "total": 658273,
    "top": {"sessions": 598_000, "event_logs": 41_000, "logs": 9_000,
            "_pipeline": 5_200, "autonomy-runs": 4_000, "triage-3087": 3_347,
            "snapshots": 0},
    "inode": 118_337,
    "checkouts": ["triage-3087"],
}
INCIDENT_AFTER = {
    "ts": 1_760_988_254.0 + 42.0,
    "total": 654927,
    "top": {"sessions": 598_000, "event_logs": 41_001, "logs": 9_000,
            "_pipeline": 5_200, "autonomy-runs": 4_000, "snapshots": 0},
    "inode": 118_337,
    "checkouts": [],
}


def _snap(d: dict) -> VW.Snapshot:
    return VW.Snapshot.from_dict(d)


def test_the_2026_10_20_measurement_replays_as_no_trip_and_says_so(tmp_path, capsys):
    root = _root(tmp_path)
    w = _watch(root, tmp_path)
    w.history = [_snap(INCIDENT_BEFORE)]

    excused: list[str] = []
    why = w.judge(_snap(INCIDENT_AFTER), excused)
    assert why is None, (
        "the alert said 3,347 files vanished from a watched root and it was right "
        "about the arithmetic; what it could not see was that the folder held a "
        "git worktree and that its own session had just cleaned it up")
    assert any("triage-3087" in e and "3347" in e for e in excused), excused

    w.note_excused(excused)
    assert "triage-3087" in capsys.readouterr().out


def test_the_replay_exercises_the_rule_that_actually_fired():
    """The fixture must be a real replay, not a scenario that happens to pass.

    Two things have to be true of the numbers: the whole-tree drop rule could not
    have fired (0.5% of the peak, under `DROP_FRACTION`), and the folder rule is
    the one that did — which is only visible by asking it with the exemption
    switched off.
    """
    before, after = _snap(INCIDENT_BEFORE), _snap(INCIDENT_AFTER)
    lost = INCIDENT_BEFORE["total"] - INCIDENT_AFTER["total"]
    assert lost == 3_346 == 3_347 - 1, "the folder, minus the one event log it added"
    assert lost / INCIDENT_BEFORE["total"] < VW.DROP_FRACTION, (
        "the drop rule was never what fired here")
    raw = VW.evaluate([before], after, what="data")
    assert raw == "top-level folder 'triage-3087' (3347 files) vanished", (
        "this is the reason string the 14:24:14 alert carried, quoted from ALERT.md; "
        "if the fixture cannot reproduce it, the node above is proving nothing")


def test_the_same_measurement_without_the_flag_still_trips(tmp_path):
    """Fail closed: the exemption needs evidence, and evidence is the flag."""
    root = _root(tmp_path)
    w = _watch(root, tmp_path)
    before = _snap({**INCIDENT_BEFORE, "checkouts": []})
    w.history = [before]
    excused: list[str] = []
    why = w.judge(_snap(INCIDENT_AFTER), excused)
    assert why == "top-level folder 'triage-3087' (3347 files) vanished"
    assert excused == []


def test_a_state_file_written_before_this_change_still_loads(tmp_path):
    """`data_watch.json` on the live box has no `checkouts` key.

    The guardian reads that file at boot from the pinned snapshot in
    `~/.local/state/lloyd-guardian/bin`, so a `from_dict` that required the new
    field would not be a schema change, it would be a guardian that starts with no
    baseline at all — which reads as "not armed", which reads as a tripwire that
    is watching nothing.
    """
    legacy = {"root": "/home/alansrobotlab/lloyd-data",
              "last_good": {"ts": 1_760_988_254.0, "total": 658273,
                            "top": INCIDENT_BEFORE["top"], "inode": 118_337}}
    state = tmp_path / "state"
    state.mkdir(parents=True)
    (state / "data_watch.json").write_text(json.dumps(legacy), encoding="utf-8")

    w = D.DataWatch(root=str(_root(tmp_path, "watched")), state_dir=state)
    assert w.armed, "the old file still arms the watcher"
    assert w.history[-1].checkouts == (), (
        "and an unknown folder is treated as not-a-checkout, never as scratch")
    excused: list[str] = []
    assert w.judge(_snap(INCIDENT_AFTER), excused) == (
        "top-level folder 'triage-3087' (3347 files) vanished")


def test_the_serialised_snapshot_carries_the_flag_through_the_state_file(tmp_path):
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    (root / "triage-3087").mkdir()
    (root / "triage-3087" / ".git").write_text("gitdir: /x\n", encoding="utf-8")
    _fill(root / "triage-3087", 25)
    _watch(root, tmp_path)                        # arms and persists the baseline

    stored = json.loads((tmp_path / "state" / "data_watch.json").read_text(
        encoding="utf-8"))["last_good"]
    assert stored["checkouts"] == ["triage-3087"]
    again = D.DataWatch(root=str(root), state_dir=tmp_path / "state")
    assert again.history[-1].checkouts == ("triage-3087",), (
        "the exemption has to survive the round trip: a restart that loses the flag "
        "is a restart that puts the false trip back")


# ── what the widened exemption must not swallow ──────────────────────────────

def test_a_store_that_vanishes_still_trips(tmp_path):
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    _fill(root / "_pipeline", STORE_FILES)
    w = _watch(root, tmp_path)

    shutil.rmtree(root / "sessions")
    reason, _snap = w.tick(now=time_plus(w.history[-1].ts, 60))
    assert reason == "top-level folder 'sessions' (30 files) vanished", (
        "the root's own stores are never excused, whatever else is going on in the "
        "tree — this is the alarm the tripwire exists to raise")


def test_a_store_emptied_at_real_proportions_still_trips():
    """Emptying is not the folder rule's to catch, and this says which rule does.

    A top-level folder that is present with zero files inside is not `vanished`, so
    `evaluate`'s folder loop passes over it by design; the alarm for an emptied
    store is the whole-tree drop rule, and it needs the proportions the real root
    has — `sessions/` is 598,000 of the 658,273 files measured at 14:24:14, so
    emptying it is a 91% fall, not the 45% a 30-file temp folder would be. Pinning
    it here keeps the two rules from being mistaken for one: widening the folder
    rule must not be the reason the drop rule stops mattering.
    """
    before = _snap(INCIDENT_BEFORE)
    emptied = _snap({**INCIDENT_AFTER,
                     "total": INCIDENT_AFTER["total"] - INCIDENT_BEFORE["top"]["sessions"],
                     "top": {**INCIDENT_AFTER["top"], "sessions": 0}})
    w_reason = VW.evaluate([before], emptied, what="data")
    assert w_reason is not None and "disappeared" in w_reason, w_reason
    assert "sessions" not in w_reason, (
        "it is the whole-tree drop that fires, which is the honest attribution: the "
        "folder is still standing, and only its contents are gone")


def test_a_runtime_store_holding_a_cloned_repo_is_still_a_store(tmp_path):
    """The property is a floor, not a ceiling: `RUNTIME_NAMES` outranks the flag.

    `_pipeline/` is exactly where a tool that clones something would land, so a
    `.git` two levels down inside a named store would otherwise make the loss of
    that store un-trippable — the worse failure by far, and the reason a
    measurement is not automatically the whole answer.
    """
    root = _root(tmp_path)
    _fill(root / "_pipeline", STORE_FILES)
    (root / "_pipeline" / "clone").mkdir()
    (root / "_pipeline" / "clone" / ".git").write_text("gitdir: /x\n", encoding="utf-8")
    w = _watch(root, tmp_path)
    assert "_pipeline" in w.history[-1].checkouts, (
        "measure does flag it — the flag is a fact about the folder, not a verdict")

    shutil.rmtree(root / "_pipeline")
    reason, _snap = w.tick(now=time_plus(w.history[-1].ts, 60))
    assert reason is not None and "_pipeline" in reason, (
        "but a folder the root knows as a store is never excused for vanishing")


def test_a_repo_deeper_than_the_bound_does_not_flag_its_folder(tmp_path):
    root = _root(tmp_path)
    _fill(root / "quarantine", STORE_FILES)
    deep = root / "quarantine" / "a" / "b"
    deep.mkdir(parents=True)
    (deep / ".git").write_text("gitdir: /x\n", encoding="utf-8")
    snap = VW.measure(str(root), skip_top=D.CACHE_DIRS)
    assert snap.checkouts == (), (
        f"CHECKOUT_MAX_DEPTH is {VW.CHECKOUT_MAX_DEPTH}: a `.git` that deep is "
        "incidental content, and excusing a whole top-level folder on it would "
        "blank the rule over whatever else that folder holds")


def test_the_root_going_missing_or_being_replaced_still_trips(tmp_path):
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    (root / "triage-3087").mkdir()
    (root / "triage-3087" / ".git").write_text("gitdir: /x\n", encoding="utf-8")
    _fill(root / "triage-3087", 25)
    w = _watch(root, tmp_path)
    before = w.history[-1]

    assert VW.evaluate([before], None, what="data") == "the data root is missing"
    replaced = VW.Snapshot(ts=before.ts + 60, total=before.total, top=dict(before.top),
                           inode=before.inode + 1, checkouts=before.checkouts)
    assert "replaced" in VW.evaluate([before], replaced, what="data"), (
        "the inode rule is the 2026-09-22 data-home move's own shape, and no "
        "checkout exemption touches it")


def test_a_data_root_that_loses_its_marker_is_still_a_trip(tmp_path):
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    (root / "triage-3087").mkdir()
    (root / "triage-3087" / ".git").write_text("gitdir: /x\n", encoding="utf-8")
    w = _watch(root, tmp_path)
    (root / D.ROOT_MARKER).unlink()
    excused: list[str] = []
    assert w.judge(VW.measure(str(root), skip_top=D.CACHE_DIRS), excused) == (
        f"{root} lost its {D.ROOT_MARKER} marker")


# ── the exemption is a property, so no name list can grow to cover it ───────

def test_the_data_watcher_still_exempts_only_its_cache_dir():
    """`skip_top` is the hand-maintained list, and it stays one entry long.

    The tempting fix for the 2026-10-20 trip was `skip_top | {"triage-*"}`, and it
    cannot work: the names are chosen by whoever runs the command, one folder per
    backlog item forever, which is #2160's ruling in this repo — a hand-maintained
    allowlist cannot close a property over an open set. A second shape of the same
    failure is a folder that is a cache today and a store next month, so the
    length of this set is pinned as tightly as its content.
    """
    assert D.CACHE_DIRS == frozenset({"code-graph"})
    assert D.DataWatch.skip_top == D.CACHE_DIRS
    src = (GUARDIAN_DIR / "datawatch.py").read_text(encoding="utf-8")
    for forbidden in ("triage-", "check-"):
        assert f'skip_top = CACHE_DIRS | {{"{forbidden}' not in src
    assert "glob" not in src.split("class DataWatch")[1], (
        "name-pattern matching in the watcher is the allowlist with the corners cut "
        "off; the exemption is allowed to read a measurement and nothing else")


def test_the_ruling_is_not_the_vaults(tmp_path):
    """The vault gets no exemption at all, so its behaviour is byte-identical.

    `~/obsidian` is the watched root that has actually been deleted twice and
    pushed to the cloud in the gap, and it is a git repository by nature: an
    exemption keyed on `.git` there would be keyed on the vault itself.
    """
    vault = VW.VaultWatch(root=str(tmp_path), state_dir=tmp_path / "state")
    assert vault._excuse_vanished("memory", 40,
                                          VW.Snapshot(ts=0, total=1,
                                                      top={"memory": 40},
                                                      checkouts=("memory",))) is None


def test_a_real_worktree_and_app_data_root_agree_on_what_a_checkout_is(tmp_path):
    """One property, two spellings, pinned against the same bytes on disk.

    `app/data_root.tree_is_worktree` is the predicate the item names and the one
    the rest of the product already trusts. The guardian cannot call it — it runs
    from a pinned snapshot under `~/.local/state/lloyd-guardian/bin` on
    `/usr/bin/python3`, where `app` is not importable, which is why
    `policy._data_root_module` loads that file by PATH for the constants it needs.
    So the guardian re-spells the test as a `.git` entry seen by the walk, and
    nothing but running both against a real `git worktree add` catches the day the
    two stop meaning the same thing.
    """
    sys.path.insert(0, str(ROOT))
    from app.data_root import tree_is_worktree

    root = _root(tmp_path)
    scratch = _worktree(root, "triage-3087/check")
    store = _fill(root / "sessions", 5)

    snap = VW.measure(str(root), skip_top=D.CACHE_DIRS)
    assert tree_is_worktree(scratch), (
        "app.data_root's answer about the checkout itself — `.git` is a FILE there")
    assert snap.checkouts == ("triage-3087",), (
        "and the guardian's answer names the top-level folder holding it, which is "
        "the bucket the vanished-folder rule counts; one level apart by design, and "
        "the same fact about the same bytes")
    assert not tree_is_worktree(store)


def test_a_plain_clone_counts_as_a_checkout_too(tmp_path):
    """`tree_is_worktree` answers the FILE form; a `git clone` leaves a DIRECTORY."""
    main = tmp_path / "origin"
    main.mkdir()

    def git(*a, cwd=main):
        subprocess.run(["git", *a], cwd=str(cwd), check=True, capture_output=True,
                       text=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (main / "f.txt").write_text("x\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "base")

    root = _root(tmp_path)
    subprocess.run(["git", "clone", "-q", str(main), str(root / "scratch" / "clone")],
                   check=True, capture_output=True, text=True)
    snap = VW.measure(str(root), skip_top=D.CACHE_DIRS)
    assert snap.checkouts == ("scratch",), (
        "a leftover clone is the same incident with a different verb: 3,200 files in "
        "a watched root, deleted by whoever put them there")


# ── #3153: a scratch folder the watcher saw come into being ──────────────────
#
# #3095 excuses a folder the last measurement saw a `.git` inside. The rest of the
# 2026-10-20 shape holds no `.git`: a job runs `mkdir ~/lloyd-data/<job>-<id>`, writes
# a few thousand files, and deletes them when it succeeds. Nothing about such a
# folder's bytes says scratch, and `VaultWatch.tick` measures on EVERY tick
# (`policy.TICK_SECONDS` is 5.0, and `guardian.check_data` ticks it unconditionally),
# so any job that outlives one gap is in the baseline before it is deleted — the trip
# is the expected outcome of that shape, not a coincidence. What does say scratch is
# `Snapshot.first_seen`: this watcher measured the folder ARRIVE between two of its
# own measurements, and no store this root has ever declared was created between two
# ticks. That is a reading over the open set, which is why it is admissible where
# #2160 rules a name list out, and it leaves `RUNTIME_NAMES` as the floor it is today.

#: The name the #3153 triage repro used, kept so the repro and the node below can be
#: run against the same string.
JOB = "triage-job-1234"


def _job(root: Path, name: str = JOB, files: int = 30) -> Path:
    """A non-checkout scratch folder: files, and no `.git` anywhere in it."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(files):
        (d / f"f{i}.txt").write_text(f"{i}\n", encoding="utf-8")
    return d


def _born(root: Path, w: D.DataWatch, **kw) -> str:
    """Make a scratch folder AFTER the baseline and tick, so the watcher sees it born."""
    name = _job(root, **kw).name
    assert w.tick(now=w.history[-1].ts + 30.0)[0] is None
    assert w.history[-1].first_seen[name] == w.history[-1].ts, (
        "the reading the excuse depends on: the tick that first saw this folder is "
        "recorded on the snapshot, not guessed from its mtime")
    return name


def test_a_scratch_folder_born_after_the_baseline_does_not_trip(tmp_path, capsys):
    """The #3153 acceptance shape: create the folder, work in it, delete it, no `.git`.

    Before this change the verdict is the one the 2026-10-20 critical alert carried,
    and `guardian.check_data` turns it into `set_halted("data tripwire: …")` plus a
    paused worker pool. Now it is None, and the skip is journalled — a rule that
    silently waves through a 3,000-file folder is a rule nobody can audit.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    w = _watch(root, tmp_path)

    _born(root, w)
    shutil.rmtree(root / JOB)
    reason, _snap = w.tick(now=w.history[-1].ts + 30.0)
    assert reason is None, (
        "`top-level folder 'triage-job-1234' (3000 files) vanished` was a job's own "
        "working directory being cleaned up, and it halted promotions")

    out = capsys.readouterr().out
    assert JOB in out, "not tripping is half of it; it still has to be said"
    rows = (tmp_path / "state" / VW.EXCUSED_LOG).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1, rows
    row = json.loads(rows[0])
    assert row["watch"] == "data" and JOB in row["excused"], row
    for said in ("#3153", "(30 files)", "horizon"):
        assert said in row["excused"], (said, row["excused"])


def test_a_scratch_folder_emptied_is_excused_as_well_as_one_deleted(tmp_path):
    """The other clean-up verb: `rm` the files and leave the directory standing.

    `evaluate` judges an emptied top-level folder and a vanished one through the same
    hook, so the excuse has to reach both or it covers half the walk-back and the trip
    comes back for the other half. Both halves are read here: the rule's own verdict on
    this pair of measurements is the `emptied` one, and it is the recorded sighting that
    stops it — which is also why the reason says only "no longer counted" instead of
    borrowing a word from whichever state it happens to be answering for.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    w = _watch(root, tmp_path)
    name = _born(root, w, name="job-88")

    for f in sorted((root / name).iterdir()):
        f.unlink()
    after = VW.measure(str(root), skip_top=D.CACHE_DIRS)
    # The same pair of measurements through the same rule with the exemption hook left
    # out — `evaluate`'s default, which is what the vault runs — so the state being
    # excused is named rather than assumed.
    assert VW.evaluate(w.history, after, what="data") == \
        "top-level folder 'job-88' (30 files) emptied"

    excused: list[str] = []
    assert w.judge(after, excused) is None
    assert any(name in e and "#3153" in e for e in excused), excused


def test_an_aged_folder_that_vanishes_still_trips_with_the_folder_rule(tmp_path):
    """Detection is preserved: the same folder, alive past the horizon, is a store.

    Nothing about the folder changed except how long the watcher has been able to see
    it, and that is the whole of the argument — a folder that has outlasted the
    horizon outlasts any job, so its loss is the event this tripwire exists to raise.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    w = _watch(root, tmp_path)
    name = _born(root, w, name="recovered")

    aged = w.history[-1].first_seen[name] + D.SCRATCH_HORIZON_SECONDS
    assert w.tick(now=aged)[0] is None, "living past the horizon is not itself an event"

    shutil.rmtree(root / name)
    reason, _snap = w.tick(now=aged + 30.0)
    assert reason == "top-level folder 'recovered' (30 files) vanished", reason
    assert not (tmp_path / "state" / VW.EXCUSED_LOG).exists(), (
        "and nothing was waved through on the way there")


def test_the_checkout_exemption_still_wins_whatever_the_folder_s_age(tmp_path):
    """#3095 is not narrowed here: a checkout is excused by what it CONTAINED, at any age.

    The ordering is the audit trail. A folder that both rules could excuse has to be
    journalled under the stronger evidence — a `.git` is a fact about the bytes, an age
    is an inference about the name's history — so the checkout branch answers first and
    its reason keeps naming `.git` and nothing about being newly seen.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    _worktree(root, "triage-3087/check")
    w = _watch(root, tmp_path)

    assert w.tick(now=w.history[-1].ts + D.SCRATCH_HORIZON_SECONDS + 3_600.0)[0] is None
    shutil.rmtree(root / "triage-3087")
    excused: list[str] = []
    assert w.judge(VW.measure(str(root), skip_top=D.CACHE_DIRS), excused) is None
    assert len(excused) == 1 and "#3095" in excused[0], excused
    assert "#3153" not in excused[0], (
        "the age branch answered for a folder the checkout rule already covered")


def test_a_runtime_name_is_never_excused_by_age_however_young_it_is(tmp_path):
    """`RUNTIME_NAMES` stays the floor: the list may only ever make this stricter.

    `logs/` created after the baseline is as young as a folder can be and still cannot
    be waved through. This is the direction of failure #2160 allows — a narrowing
    reading of an exemption, never a widening one — and the reason the age rule sits
    BEHIND the floor rather than in front of it.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    w = _watch(root, tmp_path)

    _fill(root / "logs", 30)                                  # a RUNTIME_NAME, born now
    assert w.tick(now=w.history[-1].ts + 30.0)[0] is None
    assert w.history[-1].first_seen["logs"] == w.history[-1].ts, (
        "the reading says young; the floor says store, and the floor wins")

    shutil.rmtree(root / "logs")
    reason, _snap = w.tick(now=w.history[-1].ts + 30.0)
    assert reason == "top-level folder 'logs' (30 files) vanished", reason
    assert not (tmp_path / "state" / VW.EXCUSED_LOG).exists()


def test_a_folder_the_watcher_never_saw_being_born_is_not_excused(tmp_path):
    """No stamp, or an impossible one: the answer is the trip, both times.

    The live `data_watch.json` was written before `first_seen` existed, so "absent" is
    the state every real root starts in, and a future-dated stamp is a corrupt record
    or a clock jump. Neither may read as scratch: the exemption is granted by evidence
    the watcher measured, and the absence of that evidence is not evidence of anything.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", 40_000)                # the drop rule stays out of the way
    w = _watch(root, tmp_path)
    gone = {"ts": 2_000.0, "total": 40_031, "inode": 7,
            "top": {".": 1, "sessions": 40_000, "job-9": 30}}
    missing = {"ts": 2_030.0, "total": 40_001, "inode": 7,
               "top": {".": 1, "sessions": 40_000}}
    verdict = "top-level folder 'job-9' (30 files) vanished"

    assert _snap(gone).first_seen == {}, "a baseline with no births recorded loads as such"
    w.history = [_snap(gone)]
    excused: list[str] = []
    assert w.judge(_snap(missing), excused) == verdict
    assert excused == []

    w.history = [_snap({**gone, "first_seen": {"job-9": 9_000.0}})]      # born tomorrow
    assert w.judge(_snap(missing), []) == verdict


def test_the_horizon_outlasts_the_history_ring_it_cannot_be_read_from():
    """Why the stamp is persisted at all, and what is NOT this node's to pin.

    `VaultWatch.tick` trims `history` to `WINDOW_SECONDS` plus one older entry, so a
    horizon read off the ring could never exceed 900 s — which is the caveat on the
    item, and why the stamp rides in the persisted baseline instead. The NUMBER,
    though, belongs to a person: it has to be at least the longest legitimate scratch
    job, and only post-landing traffic shows that (the owed ruling may raise it, which
    will need this line changed with it).
    """
    assert D.SCRATCH_HORIZON_SECONDS > VW.WINDOW_SECONDS, (
        "a horizon inside the ring makes the persisted stamp decoration: the excuse "
        "would be capped at the window it was read from")
    assert D.SCRATCH_HORIZON_SECONDS >= 24 * 3600.0, (
        "a job that runs overnight has to be able to clean up after itself without "
        "halting the fleet")




# ── names that newly appear at the data root's top level (#3210) ──────────────
#
# Four detectors look at this directory and none of them asks this question.
# `data_root_shadows` answers "is this 0-byte file SHADOWING a populated store of the same
# name": it drops every name that does not end in a database suffix (`_DB_SUFFIXES`,
# datawatch.py:444, applied at :571) BEFORE its paired-populated-copy test, so a file
# named `false` or `sqlite3` is never even learned by the walk. `_bash_tree_strays` does
# snapshot the root's top level per Bash call, then narrows to new directories holding a
# checkout and new 0-byte files carrying one of `EMPTY_DB_SUFFIXES` (:557, the drop at
# :673) — its docstring rules the rest out on purpose: "a new 0-byte file that does not
# end in a database extension is a placeholder… left alone". `inert_residue` and
# `quarantine_inert` are scoped to the CODE tree (`RUNTIME_NAMES`, no "/" in the name).
# And the tripwire only ever speaks of names that VANISHED. So a mistyped shell argument —
# `sqlite3 <db>` with the argument in the wrong place, a redirect whose target expanded to
# nothing — leaves a permanent entry in the root, and the root's inventory, which the
# snapshot timer and the tripwire's baseline both measure, silently gains names.
#
# This leg asks the one question none of them asks — WHAT IS NEW HERE — and answers it as
# data for a person. Two rulings it inherits rather than invents: #3109's
# `needs_human=True` for this half of the walk (a file in a person's root is theirs;
# `quarantine_inert`'s "provably inert" tests are for git-ignored tree residue, and
# nothing here moves or unlinks), and #2056's raise-instead-of-`[]` (a root that cannot be
# read is not a root that is clean).
#
# The diff reference is this leg's OWN previous scan, which the guardian persists between
# hourly passes; it is deliberately not the tripwire's snapshot. `VaultWatch.tick` runs
# every `TICK_SECONDS = 5.0`, so a snapshot would already contain a file created minutes
# before this leg ran, and "absent from the previous measurement" would then mean "created
# in the last five seconds" — the leg would name almost nothing and read as clean forever.


def _stubs(root: Path, age_seconds: float = 9 * 86400.0) -> dict[str, float]:
    """The live root's two residents, in miniature: 0-byte files whose names carry no
    database suffix, left standing for weeks by mistyped commands — 0 bytes at mtime
    2026-10-12 (`false`) and 2026-10-22 (`sqlite3`) on the live box when this was filed.

    Returns {name: mtime} as it writes them, so a node can compare against the clock it
    seeded rather than against a literal. The age is backdated on purpose: a stub's DATE is
    what distinguishes it in an alert from a file created this turn.
    """
    out: dict[str, float] = {}
    for name in ("false", "sqlite3"):
        f = root / name
        f.write_bytes(b"")
        stamp = time.time() - age_seconds
        os.utime(f, (stamp, stamp))
        out[name] = stamp
    return out


def test_a_new_top_level_file_is_named_whether_or_not_its_name_ends_in_a_database_suffix(
        tmp_path):
    """Clause 1: the question is novelty, so the shape of the name cannot be a gate.

    Both halves come off one scan of one root. A file whose name carries a database suffix
    and one whose name carries none must both come back, because the gate this leg replaces
    is precisely a suffix test — `_DB_SUFFIXES` in `data_root_shadows`,
    `EMPTY_DB_SUFFIXES` in `_bash_tree_strays` — and a rule that named only the suffixed
    one would leave `false` and `sqlite3` exactly as unnamed as they are today. Nor may a
    populated same-basename copy elsewhere be required: that pairing question belongs to
    the shadow leg, and neither stub has a twin.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    (root / "usage.db").write_bytes(b"")                          # suffix, 0 bytes
    (root / "notes.txt").write_text("kept\n", encoding="utf-8")   # no suffix, NOT empty
    (root / "workers.db").write_bytes(b"x" * 4096)                # populated, not a stub
    previous = D.top_level_regular_files(str(root))

    (root / "false").write_bytes(b"")
    (root / "sqlite3").write_bytes(b"")
    (root / "kg.sqlite").write_bytes(b"")                         # suffixed, same shape
    arrivals = {a["name"] for a in D.new_top_level_names(
        D.top_level_regular_files(str(root)), previous)}

    assert arrivals == {"false", "sqlite3", "kg.sqlite"}, (
        "the leg must name every newly-present regular file whatever its name's shape: "
        "`false` and `sqlite3` carry no database suffix and no populated twin, which is "
        f"the whole class it exists for. Got {sorted(arrivals)}")
    held = previous | {"false": 0.0, "sqlite3": 0.0, "kg.sqlite": 0.0}
    assert D.new_top_level_names(D.top_level_regular_files(str(root)), held) == [], (
        "a scan over a baseline that already holds these names must be empty, or the leg "
        "alerts on the world as it stands")


def test_the_arrivals_leg_measures_without_touching_anything(tmp_path):
    """Clause 2: report-only is a property of the code, not of this run's luck.

    The two stubs are still on disk afterwards — a check that resolved the question by
    deleting the evidence would be worse than the silence it replaced, which is why #3109
    ruled the removal of the live pair a human action. And no mutating call is reachable
    from the leg at all: the shadow leg keeps a rail for the same reason, phrased as a
    source scan, because a green suite cannot prove the absence of a side effect it never
    triggered.
    """
    import inspect
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    _stubs(root)
    previous = D.top_level_regular_files(str(root))
    (root / "third-stub").write_bytes(b"")
    files = D.top_level_regular_files(str(root))
    assert [a["name"] for a in D.new_top_level_names(files, previous)] == ["third-stub"]
    assert D.arrivals_alert(D.new_top_level_names(files, previous)), (
        "the report must be built from the same scan the findings come from")

    for name in ("false", "sqlite3", "third-stub"):
        assert (root / name).exists(), f"the leg removed {name!r} instead of reporting it"

    # Whole sections, not just the three functions. A `getsource(fn)` scan alone leaves the
    # reachable surface open: an `unlink` at module level between the functions, or in a
    # helper the leg calls, is invisible to it — and "just add a cleanup helper beside the
    # leg" is exactly the shape a later edit takes. Two scans, because the section DOES
    # legitimately write: `write_arrivals_baseline` puts the baseline in the guardian's own
    # state directory.
    section = Path(D.__file__).read_text(encoding="utf-8")
    leg = section[section.index("def top_level_regular_files"):
                  section.index("class DataWatch")]
    for verb in (".unlink(", ".rename(", "shutil.move", "os.remove", "os.rmdir",
                                     "quarantine_inert", "inert_residue"):
        assert verb not in leg, (
            f"the arrivals leg is report-only and must not reach {verb!r} anywhere in its "
            "section — not in a helper beside it either: it found someone's file, and the "
            "ruling is theirs (#3109)")
    writer = leg[leg.index("def write_arrivals_baseline"):
                 leg.index("def new_top_level_names")]
    for verb in ("os.replace", ".mkdir("):
        assert verb in writer, (
            f"{verb} is allowed only inside `write_arrivals_baseline`, which writes the "
            "guardian's OWN state file; it has vanished from there, so the split below "
            "no longer describes what is permitted where")
    for fn in (D.top_level_regular_files, D.new_top_level_names, D.arrivals_alert):
        for verb in ("os.replace", ".mkdir("):
            assert verb not in inspect.getsource(fn), (
                f"{fn.__name__} must not create or replace anything ({verb!r}): the one "
                "write in this leg is the baseline, and it is its own function")


def test_an_arrival_is_reported_as_a_human_decision_and_never_as_an_action(tmp_path):
    """Clause 3: the routing is data, and it is the #3109 routing.

    `needs_human=True` is what makes the guardian ask and stop. The shadow findings got
    that flag because "the guardian cannot know whether a shadow is the live one", and a
    name that newly appeared is less knowable still: it is a mistyped command, a scratch
    file, or a store being born, and only a person holding the session that made it can
    tell which. The quarantine path is asserted closed from both ends — no action in the
    report, and no quarantine call in the guardian block that consumes it.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    previous = D.top_level_regular_files(str(root))
    (root / "stray-config.json").write_text("{}\n", encoding="utf-8")
    report = D.arrivals_alert(D.new_top_level_names(
        D.top_level_regular_files(str(root)), previous))

    assert report["needs_human"] is True, (
        f"a new name in a person's root is their decision to make: {report}")
    assert not report.get("quarantine") and not report.get("remove"), (
        f"the report must carry no action for the guardian to take: {sorted(report)}")
    assert "stray-config.json" in report["reason"], (
        f"the alert must name the file it is asking about: {report}")

    guardian = (Path(__file__).resolve().parents[1]
                / "agent-services" / "guardian" / "guardian.py").read_text(encoding="utf-8")
    # The arrivals leg's OWN span: from where it measures to where the strays leg's block
    # begins. The whole method is the wrong slice and would make this assertion
    # unsatisfiable — `quarantine_inert` legitimately appears a few lines below, in the
    # stray block the arrivals leg shares this hourly check with. What has to be true is
    # that the arrivals span never reaches it.
    start = guardian.index("arrivals: list[dict] = []")
    block = guardian[start:guardian.index("if strays and self.data.armed:", start)]
    assert block.count("new_top_level_names") == 1, (
        "the span is meant to BE the arrivals leg; if the call moved out of it, the two "
        "assertions below are checking a block that no longer contains the leg")
    # `needs_human=True` has to be on the CALL, not merely somewhere in the block: the
    # block's own prose quotes the literal to explain the routing, and a substring test
    # would read that comment as the flag and pass an alert that sends none.
    calls = [ln for ln in block.splitlines() if "needs_human" in ln
             and not ln.lstrip().startswith("#")]
    assert any("needs_human=True" in ln for ln in calls), (
        "the guardian's arrivals alert must be raised as a human decision on the alert "
        f"call, the way the shadow findings it shares this hourly leg with are: {calls}")
    assert "strays =" not in block, (
        "the arrivals leg must not write to `strays`: that name is the stray scan's, and "
        "the block that consumes it is the one that offers the quarantine")
    assert "quarantine_inert" not in block and "inert_residue" not in block, (
        "an arrival is not a stray: the quarantine offer belongs to the scan that reasons "
        "about strays, and folding a new name into it would hand a person's file to a rule "
        "built for git-ignored tree residue")


def test_an_unreadable_data_root_raises_rather_than_reporting_no_arrivals(tmp_path,
                                                                          monkeypatch):
    """Clause 4: an unreadable root is never a clean root (#2056, and #3109 for this walk).

    The failure this rails is the loud-once-then-silent kind. The scan returns a name set,
    and swallowing a permission error into an empty one makes "I could not look" and
    "nothing is new here" the same value — the second being the value that ENDS an alert.
    The unreadability is provoked by wrapping `scandir` rather than by `chmod`, because this
    suite runs as a user who can read anything it creates: a permission-based premise would
    quietly `return` here and prove nothing.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    (root / "false").write_bytes(b"")
    real = os.scandir

    def barred(path, *a, **k):
        if Path(str(path)).resolve() == root.resolve():
            raise OSError(13, "Permission denied", str(path))
        return real(path, *a, **k)

    # A root that is simply absent is the same class of failure and needs its own
    # assertion: an `if not os.path.isdir(root): return {}` short-circuit answers the
    # permission case correctly and the missing one wrongly, and the wrong answer is the
    # empty set that closes an alert.
    import errno as _errno
    try:
        D.top_level_regular_files(str(root.parent / "no-such-root-at-all"))
    except OSError as exc:
        assert exc.errno in (_errno.ENOENT, _errno.ENOTDIR), (
            f"a missing root raised the wrong thing: {exc!r}")
    else:
        raise AssertionError(
            "a missing data root returned a name set. This leg's own default is "
            "`DATA_ROOT`, so a wrong resolver turns 'no data root' into 'nothing is new'")

    monkeypatch.setattr(os, "scandir", barred)
    try:
        D.top_level_regular_files(str(root))
    except OSError:
        pass
    else:
        raise AssertionError(
            "an unreadable data root returned a name set. An empty one reads to the "
            "guardian as 'nothing new here', which closes the alert the only detector "
            "asking this question was raising")
    # And a failed measure must not be papered over by passing an empty baseline: `{}`
    # means "armed and seen nothing", so every current name becomes an arrival. That is the
    # opposite failure from the raise above — a flood instead of a silence — and it is why
    # the caller persists a baseline only from a pass that actually measured.
    assert D.new_top_level_names({"false": 1.0}, {}) == [{"name": "false",
                                                          "mtime": 1.0}], (
        "an EMPTY baseline is the post-arm case with nothing seen, so a bare `false` is an "
        "arrival; recovering from a failed measure by passing {} would flood the alert")


def test_an_arrival_carries_its_date_and_a_name_is_never_named_twice(tmp_path):
    """Clause 5, first half: a week-old stub and a file made this turn are different
    questions, and a name already asked about is not asked about again next tick.

    The date is what lets the reader tell them apart without opening a shell. `false` has
    been standing since 2026-10-12 and `sqlite3` since 2026-10-22; an alert that said only
    "a new file appeared" about a ten-day-old stub would be dismissed, correctly. So the
    mtime rides on the finding and the report renders it as an ISO date.

    The two stubs here are seeded AFTER the baseline, which is how the live pair will be
    met: an arrival, named once and then held by the persisted baseline. What the leg
    cannot do is name a file that stood at the root before it shipped — clause 5's arm is
    silent, so `false` and `sqlite3` on the live root are owed to whoever deletes them,
    exactly where the item already puts them.
    """
    import datetime as dt
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    # A baseline holding only what the root had before the stubs: the state the live leg is
    # in the hour after its silent arm. `{}` would see the root marker as an arrival too,
    # and `None` would BE the arm, which is silent.
    previous = D.top_level_regular_files(str(root))
    seeded = _stubs(root)
    arrivals = D.new_top_level_names(D.top_level_regular_files(str(root)), previous)

    named = {a["name"]: a for a in arrivals}
    assert set(named) == set(seeded), (
        "arming against the known-good inventory must surface the two residents exactly "
        f"once and add nothing: {sorted(named)}")
    for name, stamp in seeded.items():
        assert abs(named[name]["mtime"] - stamp) < 2, (
            f"{name!r}'s finding carries no usable date: {named[name]}")
        assert dt.date.fromtimestamp(named[name]["mtime"]).isoformat() == \
            dt.date.fromtimestamp(stamp).isoformat(), (
            f"the date carried for {name!r} must be the date the file carries")

    report = D.arrivals_alert(arrivals)
    dates = [dt.date.fromtimestamp(s).isoformat() for s in seeded.values()]
    assert all(d in report["reason"] + report["evidence"] for d in dates), (
        f"the alert text must render each file's date, not merely carry a field: "
        f"{report!r}")

    assert D.new_top_level_names(D.top_level_regular_files(str(root)),
                                 D.top_level_regular_files(str(root))) == [], (
        "a second pass over unchanged state must name nothing: one name per arrival, not "
        "one per hour")


def test_the_persisted_baseline_is_what_makes_the_next_hour_quiet(tmp_path):
    """Clause 5, second half, across the boundary the leg actually crosses.

    The previous scan lives in the guardian's state directory, not in memory: the check
    runs in the hourly leg of a long-lived daemon and its baseline has to survive a
    restart, because a leg that re-armed from an empty baseline at every start would
    re-alert on every file created since the last one. So this is the real hand-off —
    write what was scanned, read it back next hour — and a name alerted in hour one must
    be absent from hour two's findings because the baseline now holds it.
    """
    root = _root(tmp_path)
    _fill(root / "sessions", STORE_FILES)
    state = tmp_path / "state"
    assert D.read_arrivals_baseline(state, str(root)) is None, (
        "no baseline yet is NOT an empty baseline: it is the arm case, and the leg has to "
        "be able to tell the two apart or every restart re-alerts the root's inventory")

    # No baseline on disk at all: the ARM case, and the arm is silent. A first pass that
    # listed the world as it stands would report every store the system owns as news and
    # bury the arrivals the leg exists for.
    files = D.top_level_regular_files(str(root))
    assert D.new_top_level_names(files, None) == [], (
        "no baseline yet must report nothing: a first pass that listed the root as "
        "it stands would call every legitimate resident a new arrival")
    D.write_arrivals_baseline(state, files, now=time.time(), data_root=str(root))
    assert D.read_arrivals_baseline(state, str(root)) == files, (
        "the persisted baseline must round-trip the scan exactly: a truncated one would "
        "make files that were already there look new next hour")

    (root / "sqlite3").write_bytes(b"")
    second = D.top_level_regular_files(str(root))
    found = D.new_top_level_names(second, D.read_arrivals_baseline(state, str(root)))
    assert [a["name"] for a in found] == ["sqlite3"], (
        f"hour two names only what hour one did not see: {found}")
    D.write_arrivals_baseline(state, second, now=time.time(), data_root=str(root))
    assert D.new_top_level_names(D.top_level_regular_files(str(root)),
                                 D.read_arrivals_baseline(state, str(root))) == [], (
        "after the alert was sent and the baseline advanced, the same file must not be "
        "alerted on the following pass")

    # The monotone half: sqlite writes `usage.db-wal` and removes it at checkpoint, so a
    # baseline that is a SNAPSHOT of the last scan would call the returning sidecar a new
    # arrival every time the file came back. The baseline is therefore everything the leg
    # has ever seen, not just what it saw last hour.
    (root / "usage.db-wal").write_bytes(b"w")
    with_wal = D.top_level_regular_files(str(root))
    D.write_arrivals_baseline(state, with_wal, now=time.time(), data_root=str(root))
    (root / "usage.db-wal").unlink()
    assert "usage.db-wal" not in D.top_level_regular_files(str(root)
    ), "the sidecar has to be gone for this to be the case it describes"
    D.write_arrivals_baseline(state, D.top_level_regular_files(str(root)),
                              now=time.time(), data_root=str(root))
    assert "usage.db-wal" in D.read_arrivals_baseline(state, str(root)), (
        "a name the leg has already seen must stay seen: a snapshot baseline would make "
        "every checkpoint-and-return of a sqlite sidecar a fresh human-decision alert")


# ── a sidecar whose store is right there is not a new name (#3229) ────────────
#
# The five unit nodes below are one rule and its falsifier halves; two further nodes at the
# end of this block run the leg itself rather than the function —
# `test_the_hourly_check_is_quiet_for_a_sidecar_and_still_loud_for_the_name_beside_it` over
# three real `_runtime_data_incident` ticks, and
# `test_a_muted_sidecar_files_no_board_item_while_a_real_name_still_files` across the board
# POST. `usage.db-wal` and
# `usage.db-shm` are names SQLite writes itself: opening the store in WAL mode creates the
# pair and the last connection to close cleanly removes it. On 2026-10-23 the arrivals leg
# reported `research.db-wal`/`-shm` as two new names, `Notifier._backlog_task` filed that
# as #3227, and the ALERT.md it wrote displaced a two-day-old open error about vault
# snapshots from the one visible alert slot. The decision it asked for cannot be taken: by
# the time it is read the file is either gone by itself or it is a live store's write-ahead
# log, where deleting it loses committed writes.
#
# The mute is a PAIRING, never a suffix test — #3210 clause 1 rejected suffix filters, and
# `orphan.db-wal` with no `orphan.db` at the root is exactly the mistyped-`sqlite3`-argument
# artefact this leg was built to name. So every node below carries its own positive control:
# a `[]` from the leg is only evidence when the scan demonstrably contained the name.


def _sidecar_pair(root: Path, store: str) -> "list[str]":
    """Create `<store>-wal` and `<store>-shm` beside `store`, as opening it in WAL mode does.

    Real files, not a hand-built dict: the leg's whole subject is names that physically
    stand at the root, and a finding of `[]` over a scan that never held the name would be
    a green node proving nothing (the control assertions below read the same scan back).
    """
    made = []
    for suffix in ("-wal", "-shm"):
        (root / f"{store}{suffix}").write_bytes(b"w" * 8)
        made.append(f"{store}{suffix}")
    return made


def test_a_sidecar_beside_its_store_is_not_reported_as_a_new_name(tmp_path):
    """Clause 1: the pairing explains the name, so it is not an arrival.

    `usage.db` already stands at the root and the baseline already holds it; the pass that
    follows finds `usage.db-wal` and `usage.db-shm` and must report NOTHING. `False` is not
    in the previous scan by accident — it is the state of the live root one hour after the
    leg armed, which is precisely when the sidecar pair shows up.
    """
    root = _root(tmp_path)
    (root / "usage.db").write_bytes(b"x" * 4096)
    previous = D.top_level_regular_files(str(root))
    assert "usage.db" in previous, "the store has to be in the baseline for this to be pairing"

    made = _sidecar_pair(root, "usage.db")
    scan = D.top_level_regular_files(str(root))
    assert set(made) <= set(scan), (
        f"the scan does not hold {made}, so a `[]` below is an absent file, not a mute")
    assert D.new_top_level_names(scan, previous) == [], (
        "a WAL sidecar of a store standing in the same scan is that store's bookkeeping, "
        "not a name a person can decide about — it is gone by the time they read it, or "
        "deleting it loses committed writes (#3227)")

    # `_SQLITE_SIDECARS` carries a third name, and the pairing rule covers the whole
    # constant, not just the two files WAL mode happens to make: a rollback-journal of a
    # store at the root is the same explained name.
    (root / "usage.db-journal").write_bytes(b"w" * 8)
    assert D.new_top_level_names(D.top_level_regular_files(str(root)), previous) == [], (
        "`-journal` is in _SQLITE_SIDECARS and a rollback journal beside its store is "
        f"explained by it: {sorted(D.top_level_regular_files(str(root)))}")


def test_a_sidecar_whose_store_is_absent_is_still_reported(tmp_path):
    """Clause 2: with no base in the scan the name is unexplained, and must still be asked.

    `orphan.db-wal` alone is the artefact the leg exists for — `sqlite3 <db>` with the
    argument in the wrong place creates the pair and the store, and a store-less `-wal` is
    what a partially-cleaned-up one leaves behind. A suffix-only mute would swallow it,
    which is why #3210 clause 1 is stated as a ban on suffix filters.
    """
    root = _root(tmp_path)
    previous = D.top_level_regular_files(str(root))
    assert "orphan.db" not in previous and "orphan.db-wal" not in previous

    (root / "orphan.db-wal").write_bytes(b"w" * 8)
    scan = D.top_level_regular_files(str(root))
    assert "orphan.db-wal" in scan and "orphan.db" not in scan, (
        "the case needs the sidecar present with NO base beside it: that half is the point")
    found = D.new_top_level_names(scan, previous)
    assert [f["name"] for f in found] == ["orphan.db-wal"], (
        "muting the paired case must not mute the unpaired one: "
        f"{found}")


def test_a_muted_sidecar_never_swallows_a_real_arrival_from_the_same_pass(tmp_path):
    """Clause 3: the mute is per-name, and one pass can hold both classes at once.

    The failure this rails is not the sidecar coming back — it is `false` failing to come
    back because a filter aimed at the hour took it with it. Mixed in one scan, the
    findings must be exactly `{false}`, and the alert the guardian would raise must name
    `false` and must not carry the sidecar's name at all: an alert that mentions both has
    not muted anything, and one that mentions neither has lost the arrival.
    """
    root = _root(tmp_path)
    (root / "usage.db").write_bytes(b"x" * 4096)
    previous = D.top_level_regular_files(str(root))

    made = _sidecar_pair(root, "usage.db")
    (root / "false").write_bytes(b"")                              # the #3210 class itself
    scan = D.top_level_regular_files(str(root))
    assert set(made) <= set(scan) and "false" in scan, (
        "one scan has to hold both classes for this to be the mixed pass")

    found = D.new_top_level_names(scan, previous)
    assert {f["name"] for f in found} == {"false"}, (
        f"exactly the unexplained name may survive a pass that also saw a paired sidecar: "
        f"{found}")
    alert = D.arrivals_alert(found)
    assert "false" in alert["reason"], f"the alert must name the arrival: {alert}"
    assert "usage.db-wal" not in alert["reason"] + alert["evidence"], (
        f"the muted sidecar must not reappear in the alert text: {alert}")
    assert "1 new name(s)" in alert["reason"], (
        f"the count must be the findings, not the files that appeared: {alert['reason']}")


def test_the_arrivals_alert_still_routes_a_surviving_name_as_a_human_decision(tmp_path):
    """Clause 4: muting is a filter on WHICH names are reported, never on the routing.

    Same alert, same `needs_human=True`, no `quarantine` or `remove` key — for a name that
    survives the mute. The pair of #3210 nodes this sits beside pins the routing for the
    unfiltered population; this one pins it for the population the mute produces, because
    a change that quietly turned the survivors into an actionable stray would leave both of
    those green (they never see a sidecar) and still break the #3109 ruling.
    """
    root = _root(tmp_path)
    (root / "usage.db").write_bytes(b"x" * 4096)
    previous = D.top_level_regular_files(str(root))
    _sidecar_pair(root, "usage.db")
    (root / "kg.sqlite").write_bytes(b"")                           # unpaired: survives
    alert = D.arrivals_alert(D.new_top_level_names(
        D.top_level_regular_files(str(root)), previous))

    assert alert is not None and alert["needs_human"] is True, (
        f"a surviving arrival is still a person's decision, not the guardian's: {alert}")
    assert not alert.get("quarantine") and not alert.get("remove"), (
        f"the mute must not turn a report into an action: {sorted(alert)}")
    assert "kg.sqlite" in alert["reason"], f"the survivor must be named: {alert}"


def test_a_muted_sidecar_is_still_recorded_in_the_arrivals_baseline(tmp_path):
    """Clause 5: mute the alert, never the record — across the state-file seam.

    This is the boundary the fix could get wrong in the loud-then-silent direction: drop
    the sidecar from the BASELINE as well and the pair re-arrives every hour SQLite cycles
    it, which is the recurring-alert failure `write_arrivals_baseline`'s monotone merge
    exists to prevent. So the muted name must be IN the persisted baseline, and the same
    scan over that baseline must come back empty. The guardian writes the baseline from its
    own `top_level_regular_files` scan (guardian.py:1867), not from the findings, which is
    what makes both halves true at once; `write_arrivals_baseline` is called directly here
    the way `test_the_persisted_baseline_is_what_makes_the_next_hour_quiet` does, because
    the state file is the seam and an in-memory dict is not.
    """
    root = _root(tmp_path)
    (root / "usage.db").write_bytes(b"x" * 4096)
    state = tmp_path / "state"
    D.write_arrivals_baseline(state, D.top_level_regular_files(str(root)),
                              now=time.time(), data_root=str(root))

    made = _sidecar_pair(root, "usage.db")
    scan = D.top_level_regular_files(str(root))
    assert D.new_top_level_names(scan, D.read_arrivals_baseline(state, str(root))) == [], (
        "the leg's own check must stop printing the pair, which is the acceptance")
    D.write_arrivals_baseline(state, scan, now=time.time(), data_root=str(root))

    held = D.read_arrivals_baseline(state, str(root))
    assert set(made) <= set(held), (
        f"a muted name that never reaches the baseline will be reported again next hour: "
        f"{sorted(held)}")
    assert D.new_top_level_names(D.top_level_regular_files(str(root)), held) == [], (
        "and the same root over that baseline must stay quiet")


def test_the_two_stub_names_the_live_root_carries_are_named_where_the_shadow_leg_is_silent(
        tmp_path):
    """The boundary the item measured, kept as a control beside the rule that closes it.

    A synthetic root shaped like the live one: the two 0-byte stubs at top, and a populated
    same-basename copy of a real store deeper down so the shadow leg has something to fire
    on. `data_root_shadows` names the shadow and neither stub — the triage measurement,
    reproduced — and the arrivals leg names both stubs. If a later edit makes the shadow
    leg suffix-free, this node is where the two instruments' division of labour gets
    re-read rather than quietly merged.
    """
    root = _root(tmp_path, "root")
    _fill(root / "_pipeline" / "vault-derived", STORE_FILES)
    (root / "_pipeline" / "vault-derived" / "kg.sqlite").write_bytes(b"y" * 4096)
    (root / "kg.sqlite").write_bytes(b"")
    _stubs(root)

    shadows = D.data_root_shadows(data_root=str(root), repo=str(root), now=10 ** 12)
    assert [s["name"] for s in shadows if s["name"] in ("false", "sqlite3")] == [], (
        f"the shadow leg is a pairing check and is NOT this round's to widen: {shadows}")
    assert any(s["name"] == "kg.sqlite" for s in shadows), (
        "the control has to fire, or this node proves nothing about the pair it names — a "
        f"shadow leg that found nothing here would make the line above vacuous: {shadows}")

    # A baseline that has seen the root's expected residents but not these two: the state
    # the live leg reaches the hour after its silent arm, with the stubs still standing.
    # `None` would BE that arm, and the arm is silent by clause 5.
    armed = {".": 0.0, D.ROOT_MARKER: 0.0}
    arrivals = {a["name"] for a in D.new_top_level_names(
        D.top_level_regular_files(str(root)), armed)}
    assert {"false", "sqlite3"} <= arrivals, (
        f"the leg that exists for these two names does not name them: {sorted(arrivals)}")


# ── the hourly check itself, across the guardian → datawatch → notify seam ───
#
# Everything above this line calls `datawatch` functions directly, which is where the RULE
# lives. The acceptance is about what the guardian does while it is running, and that claim
# crosses a module boundary: the scan measures, the guardian decides whether to speak, the
# notifier decides what lands in the daily note. A node that only calls the scan is green on
# a `DataWatch` leg whose findings the guardian drops on the floor, so these three drive
# `Guardian._runtime_data_incident` itself with a real Notifier over a tmp vault.

_DAY = 86400.0
_NOW = 1_790_000_000.0          # a fixed epoch: the alert's dates are computed from it


def _watcher(tmp_path, monkeypatch, *, armed=True):
    """A `Guardian` over a data root, a code tree and a state dir that are all this test's.

    `policy.GUARDIAN_STATE` and `datawatch.DATA_ROOT` are both per-test for the same reason
    `state_dir=tmp_path` is: the leg persists a baseline, and a baseline written into the
    live guardian's state directory — or into the scratch data root conftest gives a whole
    pytest session — is one test's name set becoming another test's arrival.
    """
    import guardian as G
    import notify

    tree = tmp_path / "lloyd"
    tree.mkdir()
    data = tmp_path / "lloyd-data"
    data.mkdir()
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)
    monkeypatch.setattr(G.policy, "REPO", str(tree))
    monkeypatch.setattr(G.datawatch, "DATA_ROOT", str(data))
    monkeypatch.setattr(G.policy, "GUARDIAN_STATE", str(tmp_path / "guardian-state"))

    quarantined: list = []

    def _never_quarantine(*a, **k):
        quarantined.append(a)
        raise AssertionError("quarantine_inert ran, so a data-root name could be moved")

    monkeypatch.setattr(G.datawatch, "quarantine_inert", _never_quarantine)

    g = G.Guardian.__new__(G.Guardian)
    g.notifier = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                                 vault_root=str(vault),
                                 backend_url="http://127.0.0.1:1")
    g.data = __import__("types").SimpleNamespace(armed=armed)
    g._alert_seen = {}
    g.last_alert = ""
    news: list[tuple[str, str]] = []
    g.notifier.announce = lambda title, body="", level="info": news.append((title, body)) or {}
    alerts: list[tuple[str, str, dict]] = []
    real_alert = g.notifier.alert

    def _alert(level, title, body, **kw):
        alerts.append((title, body, kw))
        return real_alert(level, title, body, **kw)

    g.notifier.alert = _alert
    cleared: list[tuple[str, bool]] = []
    real_resolve = g.notifier.resolve

    def _resolve(title, note, **kw):
        ok = real_resolve(title, note, **kw)
        cleared.append((title, ok))
        return ok

    g.notifier.resolve = _resolve
    return G, g, data, alerts, cleared, quarantined


def test_the_hourly_check_names_a_new_stub_once_with_its_date_and_moves_nothing(
        tmp_path, monkeypatch):
    """Clauses 1, 2, 3 and 5, on the surface the acceptance names: the running guardian.

    Three hourly ticks. The first ARMS and says nothing in either direction; between the
    first and second, two 0-byte files whose names carry no database extension appear in the
    root — `false` and `sqlite3`, the class the item is about, with `false` nine days old and
    `sqlite3` one day old; the second tick alerts; the third is the following hour and must
    not repeat it. `ALERT_REPEAT_SECONDS` is 900 s and the ticks here are an hour apart, so
    the absence of a second alert is NOT the repeat guard doing it — the baseline is.
    """
    import time

    G, g, data, alerts, cleared, quarantined = _watcher(tmp_path, monkeypatch)
    ARR = G.RUNTIME_DATA_ARRIVALS_ALERT_TITLE

    g._runtime_data_incident(_NOW)
    assert alerts == [], f"the pass that only recorded a baseline alerted: {alerts}"
    # Scoped to this leg's own heading. The stray/shadow walk retracts its two headings on
    # every clean pass (#2056, #3124) and that is behaviour this round does not own; what
    # clause 5 rules on is THIS leg's first pass, which compared nothing.
    assert [t for t, _ in cleared if t == ARR] == [], (
        "the pass that compared nothing retracted: an all-clear is a verdict about a "
        f"comparison, and this pass made none: {cleared}")

    for name, age_days in (("false", 9), ("sqlite3", 1)):
        p = data / name
        p.write_bytes(b"")
        stamp = _NOW - age_days * _DAY
        os.utime(p, (stamp, stamp))

    g._runtime_data_incident(_NOW + _DAY)
    hits = [a for a in alerts if a[0] == G.RUNTIME_DATA_ARRIVALS_ALERT_TITLE]
    assert len(hits) == 1, f"the arrivals heading was raised {len(hits)} times: {alerts}"
    title, body, kw = hits[0]
    assert kw.get("needs_human") is True, (
        f"clause 3 is the routing, and the alert went out without it: {kw}")
    assert "quarantine" not in kw and "staged" not in kw, (
        f"an arrival is not a stray; these keys are what the resolve path consumes: {kw}")
    assert "false" in body and "sqlite3" in body, body
    for name, age_days in (("false", 9), ("sqlite3", 1)):
        want = time.strftime("%Y-%m-%d", time.localtime(_NOW - age_days * _DAY))
        assert want in body, (
            f"clause 5: the alert must distinguish a nine-day-old stub from one made this "
            f"turn, and {want} ({name}) is not in it: {body}")
    assert (data / "false").is_file() and (data / "sqlite3").is_file(), (
        "clause 2: the leg is report-only — the two stubs must still be standing after the "
        "measurement that named them")
    assert quarantined == [], "clause 2: quarantine_inert was applied to this root at all"

    before = len(alerts)
    g._runtime_data_incident(_NOW + 2 * _DAY)
    assert len(alerts) == before, (
        "clause 5: a name that was already alerted on alerted again an hour later, and this "
        f"is 24 h after the first alert, well past ALERT_REPEAT_SECONDS: {alerts}")
    # Exactly one retraction of this leg's heading across all three ticks: the arming pass
    # compares nothing, the alerting pass has something open, and only the following
    # measured-and-clean pass closes it.
    assert [t for t, _ in cleared if t == ARR] == [ARR], (
        "the following measured pass must retract this leg's own heading so the note does "
        f"not go on ordering a person about a file they have already dealt with: {cleared}")


def test_a_name_that_arrives_while_disarmed_is_still_reported_after_rearming(
        tmp_path, monkeypatch):
    """A pause is silence in BOTH directions, so it may not absorb what it heard.

    The alert is gated on `armed`. A pass that is disarmed and advances the baseline anyway
    marks the pause's new names as seen without ever reporting them, and the tick that
    re-arms then has nothing left to say: the arrival is swallowed by the pause rather than
    closed by a person. This is #2056 clause 2's partition read on the persist side.
    """
    G, g, data, alerts, cleared, quarantined = _watcher(tmp_path, monkeypatch)
    g._runtime_data_incident(_NOW)                       # arm, armed

    g.data.armed = False
    (data / "made-during-the-pause").write_bytes(b"")
    os.utime(data / "made-during-the-pause", (_NOW + _DAY, _NOW + _DAY))
    g._runtime_data_incident(_NOW + _DAY)
    assert alerts == [], f"a paused guardian alerted: {alerts}"

    g.data.armed = True
    g._runtime_data_incident(_NOW + 2 * _DAY)
    hits = [a for a in alerts if a[0] == G.RUNTIME_DATA_ARRIVALS_ALERT_TITLE]
    assert len(hits) == 1, (
        "the name created while the watcher was disarmed never reached a human: the "
        f"disarmed pass advanced the baseline behind the silent alert: {alerts}")
    assert "made-during-the-pause" in hits[0][1], hits[0][1]


def test_the_hourly_check_is_quiet_for_a_sidecar_and_still_loud_for_the_name_beside_it(
        tmp_path, monkeypatch):
    """#3229 across the seam the harm actually crossed: root → datawatch → guardian → notifier.

    Three hourly ticks over the real `_runtime_data_incident`. Tick one arms and records a
    baseline with `usage.db` in the root. Between one and two SQLite opens that store, which
    is what leaves `usage.db-wal` and `usage.db-shm` standing there; tick two must raise
    NOTHING for this leg, because that alert is the filing that became #3227 and it cost an
    open two-day-old vault-snapshot error the only ALERT.md slot. `false` then appears, and
    tick three must raise exactly one arrivals alert naming `false` alone — which is only
    possible if the muted pair reached the baseline on tick two.
    """
    G, g, data, alerts, cleared, quarantined = _watcher(tmp_path, monkeypatch)
    ARR = G.RUNTIME_DATA_ARRIVALS_ALERT_TITLE
    (data / "usage.db").write_bytes(b"x" * 4096)
    g._runtime_data_incident(_NOW)                        # arms, and records the baseline

    _sidecar_pair(data, "usage.db")                       # SQLite opening the store
    g._runtime_data_incident(_NOW + _DAY)
    assert [a for a in alerts if a[0] == ARR] == [], (
        "the guardian asked a person to decide about SQLite's own write-ahead log, which "
        f"is the filing that buried an open error: {alerts}")
    # Read the baseline HERE, not at the end: tick three's scan still holds the pair, so a
    # later read cannot tell a tick-two persist from a tick-three one, and a leg that muted
    # the pair by dropping it from the baseline write would still look clean at the end.
    held_after_two = D.read_arrivals_baseline(tmp_path / "guardian-state",
                                              os.path.realpath(str(data)))
    assert {"usage.db-wal", "usage.db-shm"} <= set(held_after_two or {}), (
        "the muted names did not reach the persisted baseline ON THIS TICK, so next hour "
        f"they are arrivals again: {sorted(held_after_two or {})}")

    (data / "false").write_bytes(b"")
    os.utime(data / "false", (_NOW + 2 * _DAY, _NOW + 2 * _DAY))
    g._runtime_data_incident(_NOW + 2 * _DAY)
    hits = [a for a in alerts if a[0] == ARR]
    assert len(hits) == 1, (
        f"the real arrival that shared a pass with the muted pair never reached a human: "
        f"{alerts}")
    assert "false" in hits[0][1] and "usage.db-wal" not in hits[0][1], hits[0][1]
    assert hits[0][2].get("needs_human") is True, (
        f"routing is unchanged by the mute: {hits[0][2]}")


def test_a_muted_sidecar_files_no_board_item_while_a_real_name_still_files(
        tmp_path, monkeypatch):
    """#3229 across the last seam: the alert that fires is what becomes a board item.

    The harm was never a log line. `Notifier.alert` routes a `needs_human` finding to
    `_backlog_task` (notify.py:312-313), which POSTs `/api/backlog/task-create` — that is
    how the 2026-10-23 sidecar report became #3227, a `draft` at `high` priority on the
    lloyd board. Every node above this one runs a notifier pointed at a closed port
    (`_watcher`), so none of them can see a filing at all; this one puts the loopback board
    from `tests/test_guardian_predicates.py` behind the same leg and reads its POST log.

    The tick that meets the paired sidecar must POST nothing carrying the arrivals title; the
    tick that meets `false` must POST exactly one, and that item's description must name
    `false` without the muted name in it. The board read answers an empty list in this mode,
    so nothing is suppressed by the exact-name open-duplicate guard and an absent POST is
    evidence of an absent filing rather than of a suppressed one — the question of what that
    guard should key on is a ruling this node deliberately does not touch.
    """
    from tests.test_guardian_predicates import _stub_board

    G, g, data, alerts, cleared, quarantined = _watcher(tmp_path, monkeypatch)
    ARR = G.RUNTIME_DATA_ARRIVALS_ALERT_TITLE
    with _stub_board(tmp_path, {"success": True, "id": 4242}) as (server, seen):
        g.notifier.backend_url = f"http://127.0.0.1:{server.server_address[1]}"
        (data / "usage.db").write_bytes(b"x" * 4096)
        g._runtime_data_incident(_NOW)

        _sidecar_pair(data, "usage.db")
        g._runtime_data_incident(_NOW + _DAY)
        filed = [p for p in seen["posts"] if p.get("name") == f"[guardian] {ARR}"]
        assert seen["posts"] == [] and filed == [], (
            "the muted pair still reached the board, which is the filing that became "
            f"#3227: {filed}")
        assert [a for a in alerts if a[0] == ARR] == [], (
            f"a leg that files nothing but still alerts is the same noise one step later: "
            f"{alerts}")

        (data / "false").write_bytes(b"")
        os.utime(data / "false", (_NOW + 2 * _DAY, _NOW + 2 * _DAY))
        g._runtime_data_incident(_NOW + 2 * _DAY)
        filed = [p for p in seen["posts"] if p.get("name") == f"[guardian] {ARR}"]
        assert len(filed) == 1, (
            "the mute stopped at the findings list and never reached the filing of a real "
            f"arrival: {[p.get('name') for p in seen['posts']]}")
        post = filed[0]
        assert "false" in post["description"] and "usage.db-wal" not in post["description"], (
            f"the filed item must be about the name that needs a decision: "
            f"{post['description'][:200]}")
        assert (post["board"], post["status"], post["priority"]) == (
            "lloyd", "draft", "high"), (
            f"routing to the board is unchanged by the mute: {post}")


def test_a_data_root_the_check_cannot_read_neither_alerts_nor_retracts(
        tmp_path, monkeypatch):
    """Clause 4 at the caller: an unreadable root may not read as a clean one.

    `data_root_shadows` raising is already pinned elsewhere; this is the other half — the
    hourly block must not treat a scan that could not run as a scan that found nothing, in
    EITHER direction. A root removed out from under a baseline that exists is the case that
    would otherwise write a retraction over an open section.
    """
    G2, g, data, alerts, cleared, quarantined = _watcher(tmp_path, monkeypatch)
    g._runtime_data_incident(_NOW)                       # arm, and leaves a baseline behind
    assert [t for t, _ in cleared if t == G2.RUNTIME_DATA_ARRIVALS_ALERT_TITLE] == []
    data.rmdir()                                          # the root is gone; the baseline is not
    monkeypatch.setattr(G2.datawatch, "DATA_ROOT", str(data))

    g._runtime_data_incident(_NOW + _DAY)
    assert alerts == [], f"a root that could not be read alerted: {alerts}"
    assert [t for t, _ in cleared if t == G2.RUNTIME_DATA_ARRIVALS_ALERT_TITLE] == [], (
        "clause 4 / #2056: a measurement that did not happen retracted the heading anyway "
        f"— unreadable read as clean: {cleared}")


# ── #3481 clause 4: the alert line states rows beside bytes, and the heading says
#    "duplicate", because the file it now names is not 0 bytes ─────────────────────────
#
# `_shadow_alert_body` used to print `(0 bytes)` for every finding, because every finding
# was 0 bytes. Once the check measures content, a line reading "0 bytes" over an 81,920-byte
# file is a number the reader will act on — and the shape of the file decides the heading
# too: `RUNTIME_DATA_SHADOW_ALERT_TITLE` literally asserts every finding is 0 bytes, so
# naming a provisioned database under it would order a person to delete a copy that may be
# the half they care about. These nodes pin both halves: the rendered line, and the same
# line reaching the daily note through the hourly leg.


def _tiny_kg(path: Path, rows: int) -> Path:
    """A real SQLite database in WAL mode: `meta` plus one data table, `rows` in it.

    Small and hand-built on purpose. The alert body is rendered from the finding's measured
    fields, so what this has to be is a file the row probe can open and answer for — not a
    copy of the shipped schema, which `tests/test_guardian_data_root_shadow.py` builds from
    `app/kg_store.py` for the measurement nodes.

    WAL mode, because that is the journal mode of every store `KGStore` provisions
    (`app/kg_store.py:228-229`) and therefore the only shape this leg meets in the live
    root: the journal mode decides which read open the probe may use, so a rollback-journal
    fixture would exercise an open the hourly check never performs on a real store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.executescript(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
            "CREATE TABLE things (id INTEGER PRIMARY KEY, note TEXT);")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1')")
        con.executemany("INSERT INTO things(note) VALUES (?)", [("x",) for _ in range(rows)])
        con.commit()
    finally:
        con.close()
    return path


def test_the_shadow_line_states_measured_rows_beside_measured_bytes():
    """Clause 4, on the renderer, over the live incident's own measurements.

    The finding carries the numbers `data_root_shadows` produced for
    `/home/alansrobotlab/lloyd-data/kg.sqlite` on 2026-10-29: 81,920 bytes, 4 data tables,
    0 rows, and a 172,789,760-byte declared store at `_pipeline/vault-derived/kg.sqlite`.
    Those same fields are asserted against a real fixture in
    `test_an_80_kib_schema_only_copy_of_the_declared_store_is_a_shadow`, so this node is
    pinned to the measurement and not merely to itself.
    """
    import guardian as G

    finding = {
        "shadow": "/home/alansrobotlab/lloyd-data/kg.sqlite",
        "name": "kg.sqlite", "depth": 1,
        "trusted": ("/home/alansrobotlab/lloyd-data/_pipeline/vault-derived/kg.sqlite"),
        "trusted_depth": 3, "trusted_size": 172_789_760,
        "declared": True, "declared_relative": "_pipeline/vault-derived/kg.sqlite",
        "shadow_size": 81_920, "data_tables": 4, "data_rows": 0,
    }
    body = G._shadow_alert_body("/home/alansrobotlab/lloyd-data", [finding], _NOW)

    assert ("- empty `/home/alansrobotlab/lloyd-data/kg.sqlite` "
            "(81,920 bytes, 0 rows in 4 tables)") in body, body
    assert "(0 bytes)" not in body, (
        "an 81,920-byte file described as 0 bytes is the sentence a person would act on: "
        + body)
    assert ".tables` must print nothing" not in body, (
        "the verification step has to ask the question the finding was measured on: a "
        "file this check now names prints five tables, so that instruction reads as "
        "'not removable' over a zero-row file: " + body)
    # The trusted half still has to be identified, and at its real size — that is what
    # makes a 4-deep live store recoverable to a human.
    assert "172,789,760 bytes" in body
    assert "_pipeline/vault-derived/kg.sqlite" in body
    assert "moves and deletes NOTHING" in body


def test_the_guardian_source_never_tells_anyone_to_open_a_store_with_the_cli():
    """Clause 1, on the text of the module rather than on one rendered paragraph.

    `git grep -n 'sqlite3 -readonly' -- agent-services/guardian/` is the check the
    item filed with, and it printed `guardian.py:421` for the whole life of #3559.
    Pinning the source string rather than only the rendered body is what keeps the
    wording from returning in a NEW paragraph one refactor away: this node reads the
    file as bytes, so a second renderer added later is caught by it too.

    Scoped to this module on purpose. `sqlite3 -readonly` is the correct re-derive
    command for a rollback-journal store, and other tests here REQUIRE it
    (`tests/test_vllm_doc_claims.py` asserts it for `usage.db`,
    `tests/test_sync_registration_guard.py` cites it). This is a rule about what the
    guardian tells a person to do to the store under inspection, not a rule about
    the shell.
    """
    text = (Path(__file__).resolve().parent.parent
            / "agent-services" / "guardian" / "guardian.py").read_text(encoding="utf-8")

    assert "sqlite3 -readonly" not in text, (
        "the guardian ships a mutating open again: an `sqlite3 -readonly` of a "
        "WAL-mode store re-stamps its -shm index, which is the writer evidence "
        "datawatch reasons from (#3559, #3481)")
    assert "sqlite3 ." not in text and "sqlite3 <path>" not in text, (
        "a bare `sqlite3 <path>` CLI step appeared, and it opens the file "
        "write-capable, which is the same mutation with the flag dropped")


def test_the_confirmation_step_names_the_immutable_read_and_no_cli_open():
    """#3577: the step that CONFIRMS a shadow is empty must not be able to touch it.

    The paragraph is a function of the finding's measured fields, so this node hands
    it the fields of the live incident - the 81,920-byte `kg.sqlite` of 2026-10-29,
    WAL-mode with both sidecars - the same finding
    `test_the_shadow_line_states_measured_rows_beside_measured_bytes` renders one
    paragraph above. That those fields are what `data_root_shadows` answers for a WAL
    store is pinned over real bytes by `tests/test_guardian_data_root_shadow.py`; what
    is pinned HERE is the instruction the renderer builds out of them, which no
    fixture can grade.

    The mutation the paragraph replaced (#3559) was not a write to the store: an
    `sqlite3 -readonly` open of a WAL-mode database re-stamps the store's `-shm`
    index, and that index's mtime is the writer evidence the alert reasons from. The
    step written to check a shadow safe to remove was destroying the evidence of who
    made it, and did - the quarantined witness under
    `~/lloyd-data/quarantine/tree-strays/2026-10-29-kg-sqlite/` carries a `-shm`
    stamp 2 days after its `-wal`.

    The ban covers the `sqlite3` shell WHOLE, not just its `-readonly` flag: a
    `.tables` metacommand opens the same file the same way, so a rewrite that swapped
    the flag and kept the shell step would have changed nothing observable. Scope
    note, because this is NOT a repo-wide rule: `-readonly` is genuinely inert on a
    rollback-journal store and is the required re-derive command elsewhere in this
    tree (`tests/test_vllm_doc_claims.py` asserts it for `usage.db`).
    """
    import guardian as G

    finding = {
        "shadow": "/home/alansrobotlab/lloyd-data/kg.sqlite",
        "name": "kg.sqlite", "depth": 1,
        "trusted": ("/home/alansrobotlab/lloyd-data/_pipeline/vault-derived/kg.sqlite"),
        "trusted_depth": 3, "trusted_size": 172_789_760,
        "declared": True, "declared_relative": "_pipeline/vault-derived/kg.sqlite",
        "shadow_size": 81_920, "data_tables": 4, "data_rows": 0,
        # The two fields this clause is about, and the reason the finding is the
        # WAL case: `KGStore` runs `PRAGMA journal_mode=WAL` on every connection it
        # hands out (`app/kg_store.py:230`, read at it rather than remembered), so a
        # shadow this leg names is WAL by construction.
        "journal_mode": "wal", "sidecars": ["-shm", "-wal"],
    }
    body = G._shadow_alert_body("/home/alansrobotlab/lloyd-data", [finding], _NOW)

    # Clause 2: the read the body names is the one `datawatch._read_open` already
    # picks, spelled as the URI form a person can paste.
    assert "mode=ro&immutable=1" in body, (
        "the paragraph names no immutable read, so there is nothing here that a "
        "reader can run without re-stamping the store: " + body)
    assert "_read_open" in body, (
        "the body does not point at the shipped function that makes this choice, so "
        "the instruction and the code that measures can drift apart: " + body)

    # Clause 3: no CLI open of any shape is offered as the verification step.
    assert "sqlite3" not in body, (
        "the body offers an `sqlite3` CLI step again; any such open of a WAL-mode "
        "store re-stamps its -shm, which is the evidence this check reasons from: "
        + body)
    assert ".tables" not in body, (
        "the shell metacommand stayed, and it opens the file exactly as the flag did: "
        + body)

    # The `why` has to be on the line, or the next person "simplifies" it back.
    assert "-shm" in body and "immutable" in body, (
        "the paragraph names the read but not the reason, and a reason-less "
        "restriction is one that gets relaxed: " + body)
    # The edge case the rewrite must not lose: `immutable=1` ignores an
    # un-checkpointed log, so with bytes in the -wal it reads a POPULATED store as
    # empty. That answer is a lie, and worse than the mutation it replaces
    # (`datawatch.py:699`, pinned by test_a_wal_log_without_its_index_is_declined...).
    assert "-wal" in body and "holds bytes" in body, (
        "the paragraph tells a reader to use an immutable read unconditionally, and "
        "over a populated -wal that reads content as 0 rows: " + body)
    # The rows-not-tables half (#3481) still has to survive the rewrite.
    assert "select count(*) from <table>" in body and "answer 0" in body, (
        "the rewrite dropped the question the finding was measured on: " + body)


def test_the_hourly_leg_alerts_a_zero_row_copy_under_a_heading_that_says_duplicate(
        tmp_path, monkeypatch):
    """Clause 4 end to end: the measured line, the shape-true heading, and the all-clear.

    The heading is the part the byte-blind check got away with. `RUNTIME_DATA_SHADOW_ALERT_TITLE`
    says "0-byte databases"; #3124 established that the title must describe the finding,
    because `notify.resolve` matches the daily-note heading literally and an all-clear that
    does not match the heading the alert was opened under leaves the section standing open
    forever (`tests/test_guardian_alert_retraction.py`). So the rename has to be a pair: the
    constant, and this retired tuple.
    """
    G2, g, data, alerts, cleared, quarantined = _watcher(tmp_path, monkeypatch)
    _tiny_kg(data / "kg.sqlite", rows=0)
    _tiny_kg(data / "_pipeline" / "kg.sqlite", rows=20)
    size = os.path.getsize(data / "kg.sqlite")
    # Aged past `INERT_MIN_AGE_SECONDS` relative to the fixed `_NOW` this file drives the
    # check with: a file stamped at real wall clock is YOUNGER than `_NOW`, and the age
    # floor correctly declines it, which reads as a silent leg rather than a missing one.
    os.utime(data / "kg.sqlite", (_NOW - 7200, _NOW - 7200))
    os.utime(data / "_pipeline" / "kg.sqlite", (_NOW - 7200, _NOW - 7200))

    g._runtime_data_incident(_NOW)

    assert [t for t, _, _ in alerts] == [G2.RUNTIME_DATA_SHADOW_ALERT_TITLE], (
        f"a top-level zero-row duplicate must alert under the heading a person can act on: "
        f"{[t for t, _, _ in alerts]}")
    assert "0-byte" not in G2.RUNTIME_DATA_SHADOW_ALERT_TITLE, (
        "the heading is the condition (#3124), and a heading asserting 0 bytes about an "
        "81,920-byte file is false the moment it prints: "
        + G2.RUNTIME_DATA_SHADOW_ALERT_TITLE)
    # The rename has to be a PAIR: the heading an alert was opened under is the heading
    # `resolve` has to carry, or a pre-#3481 note seals forever (#1536's standing claim).
    assert G2.RUNTIME_DATA_SHADOW_ALERT_TITLES_RETIRED == (
        "A 0-byte database under the data root is shadowing a populated store", ), (
        "the retired wording is the one an open daily-note section may still carry")
    assert set(G2.RUNTIME_DATA_SHADOW_ALERT_TITLES_RETIRED) <= set(
        G2.RUNTIME_DATA_ALERT_TITLES), (
        "a retired heading the hourly all-clear does not walk is how a section survives "
        "its own fix")
    body = alerts[0][1]
    # Singular, and it is a fact about this fixture rather than a tolerance: `_tiny_kg`
    # provisions ONE data table, so the measured count is one and the line has to read
    # "1 table". The first draft of the renderer printed "0 rows in 1 tables" and the
    # review named it; the plural is pinned by
    # `test_the_shadow_line_states_measured_rows_beside_measured_bytes` above, at the
    # live finding's 4 tables.
    assert f"{size:,} bytes, 0 rows in 1 table)" in body, body
    assert "1 tables" not in body, body
    assert "(0 bytes)" not in body, body
    assert "moves and deletes NOTHING" in body, body

    (data / "kg.sqlite").unlink()
    g._runtime_data_incident(_NOW + _DAY)
    assert [ok for t, ok in cleared if t == G2.RUNTIME_DATA_SHADOW_ALERT_TITLE] == [True], (
        "removing the duplicate must seal the section it was opened under, and the "
        f"clearance is title-matched: {cleared}")
    # Membership in the walk list is not a clearance. The review named this as the unverified
    # half: a section opened BEFORE this round under the old heading is closed only if the
    # hourly leg sends `ok=True` under THAT title, because `notify.resolve` matches the
    # heading literally (#1536's standing claim is a section that survived its own fix).
    assert (G2.RUNTIME_DATA_SHADOW_ALERT_TITLES_RETIRED[0], True) in cleared, (
        "the retired heading was walked but not cleared, so a pre-#3481 daily-note section "
        f"stays open over an empty note forever: {sorted(cleared)}")
    assert set(G2.RUNTIME_DATA_ALERT_TITLES) <= {t for t, _ in cleared}, (
        f"the hourly all-clear has to walk every heading this incident could ever have "
        f"been opened under, retired ones included: {sorted(cleared)}")
    assert quarantined == []
