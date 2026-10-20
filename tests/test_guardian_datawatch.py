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
import shutil
import subprocess
import sys
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
