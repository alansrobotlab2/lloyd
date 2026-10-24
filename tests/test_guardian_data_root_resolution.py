"""Which data root the guardian's stray move writes into, and unlinks against (#3208).

`agent-services/guardian/datawatch.py:51` and `policy.py:207` each resolved the runtime
data root by hand — the environment variable over a default naming this machine's account —
which is `app/data_root.py`'s rule 2 with neither the `.lloyd-data-root` marker check nor
rule 3. Until 2026-10-02 a wrong answer was harmless for a reader. `datawatch.quarantine_inert`
(shipped in `5b8d3e8d`) changed that: it CREATES a stamped copy plus a `log.jsonl` row under
`QUARANTINE_DIR_RELATIVE` beneath that root and UNLINKS the copy in the code tree, so the
hand-rolled root stopped being a read and became a write plus a delete.

The divergence is not theoretical and it reproduced read-only at triage. From
`~/lloyd-sandbox` with `LLOYD_DATA` unset, in one process: the hand-rolled formula answered
`/home/alansrobotlab/lloyd-data` (the LIVE root, marker present) while
`app.data_root.resolve_data_root_for_tree(tree)` answered
`/home/alansrobotlab/lloyd-sandbox/.lloyd-data`. The guardian's destination was the first of
those, from a process whose own checkout was the second — an un-witnessed write into another
checkout's data, the class #1415 named for the sweep and this item is about for the watchdog.

Everything here therefore runs in a SUBPROCESS with `LLOYD_DATA` unset, the way
`tests/test_guardian_sink_confinement.py` runs its probes. In-process would prove nothing:
`tests/conftest.py` exports `LLOYD_DATA` for the whole suite, resolver rule 1 then wins over
rules 2 and 3, and the hand-rolled formula and the resolver agree for every node in the suite
— which is how this survived #1415, #1525 and #2110, and the same trap
`tests/test_data_home.py::test_this_suite_never_resolves_the_live_root` walks into. #1415's
own review advisory said exactly this of its equivalent node. A child with the variable
popped is the only process in which rules 2 and 3 can be observed at all.

The interpreter is `/usr/bin/python3` wherever it exists, because that is what
`lloyd-guardian.service` runs the pinned snapshot on: these modules may import nothing beyond
the standard library, and importing cleanly under the project venv would be weaker evidence
than importing cleanly under system python.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
assert GUARDIAN_DIR.is_dir(), f"{GUARDIAN_DIR} is not on the tree, so a probe run proves nothing"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GUARDIAN_DIR))

import datawatch  # noqa: E402  (for INERT_MIN_AGE_SECONDS: the age the residue must be)
from app import data_root as AR  # noqa: E402  (the real three-rule resolver)

PROBE_PY = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable

#: A `now` in 2033, so the quarantined file's stamp is one no real move could produce. The
#: absence asserted in the live data root is then a statement about THIS move, not about a
#: directory that happens to hold nothing — the live quarantine legitimately holds files the
#: deployed watchdog moved for real.
NOW = 2_000_000_000.0
STAMP = datetime.fromtimestamp(NOW, timezone.utc).strftime("%Y%m%dT%H%M%SZ")

#: The root the deployed watchdog resolves today, and the root a wrong answer here writes
#: into. Read off passwd through the same resolver the code loads, never spelled: a test that
#: hardcoded it would fail on another box instead of testing anything.
LIVE_ROOT = Path(AR.production_data_root())
LIVE_QUARANTINE = LIVE_ROOT / AR.QUARANTINE_DIR_RELATIVE

#: argv: guardian dir, the tree to scan, `now`. The child names NO data root: that is the
#: clause. The pre-fix signature took `data_root`, and `guardian.py` passed `policy.DATA_ROOT`
#: — the hand-rolled literal — into it.
_MOVE_PROBE = '''"""One real inert-residue move, by a process that was never told a data root."""
import json
import os
import sys

guardian_dir, tree, now = sys.argv[1], sys.argv[2], float(sys.argv[3])
sys.path.insert(0, guardian_dir)

import datawatch   # noqa: E402  (stdlib-only siblings, as in production)
import policy      # noqa: E402

# The MOVE route's own resolution, not the accessor's: `resolve_move_root` is what
# `quarantine_inert` consults when it was not handed a root, and it keys the resolver on
# THE TREE THE RESIDUE IS IN. The accessor with no repo would load the resolver from this
# process's own checkout instead, which is the right question for a reader and the wrong one
# here — a tree with no `app/data_root.py` of its own has no answer for a move.
root, source, why = datawatch.resolve_move_root(tree)
# The refusal is DATA, not a crash: the probe reports it, because "raised and died" and
# "moved nothing" must stay two different answers in the node's hands.
try:
    moved = datawatch.quarantine_inert(tree, ["workers.db"], now=now)
    error = None
except Exception as exc:                                       # noqa: BLE001
    moved = []
    error = f"{type(exc).__name__}: {exc}"
print(json.dumps({
    "move_root": root,
    "source": source,
    "why": why,
    "moved": [[name, dest] for name, dest in moved],
    "refusal": error,
    "stray_still_in_tree": os.path.exists(os.path.join(tree, "workers.db")),
    "lloyd_data_env": os.environ.get("LLOYD_DATA"),
}))
'''

#: argv: guardian dir, the tree to ask about ("" = the tree this process runs from). Prints
#: the accessor's answer AND the read-only constants computed at import: a watchdog that
#: cannot name its log files is a watchdog that does not start, which is the constraint that
#: makes the move route's refusal a split rather than a blanket raise.
_ROOT_PROBE = '''"""What the guardian's data-root accessors answer for one tree, in a fresh process."""
import json
import os
import sys

guardian_dir, tree = sys.argv[1], sys.argv[2]
sys.path.insert(0, guardian_dir)

import policy   # noqa: E402

root, source, why = policy.data_root_for_tree(tree or None)
print(json.dumps({
    "process_tree": policy.process_data_tree(),
    "resolved": root,
    "source": source,
    "why": why,
    "quarantine_dir": (policy.quarantine_dir(repo=policy.process_data_tree(),
                                             data_root=root) if root else None),
    "policy_data_root": policy.DATA_ROOT,
    "policy_data_root_source": policy.DATA_ROOT_SOURCE,
    "log_files": list(policy.LOG_FILES),
    "kg_db": policy.KG_DB,
    "kg_db_source": policy.KG_DB_SOURCE,
    "lloyd_data_env": os.environ.get("LLOYD_DATA"),
}))
'''


def _child_env(extra: dict | None = None) -> dict:
    """The probe environment: the suite's `LLOYD_DATA` popped, and `LLOYD_TREE` with it.

    Popping rather than leaving them is the node. Rule 1 outrules rules 2 and 3, so an
    inherited `LLOYD_DATA` would make the hand-rolled formula and the resolver agree and the
    clause untestable.
    """
    env = dict(os.environ)
    env.pop("LLOYD_DATA", None)
    env.pop("LLOYD_TREE", None)
    env.update(extra or {})
    return env


def _run_probe(tmp_path: Path, source: str, *args: str,
               guardian_dir: Path = GUARDIAN_DIR, extra_env: dict | None = None) -> dict:
    script = tmp_path / "probe.py"
    script.write_text(source, encoding="utf-8")
    proc = subprocess.run([PROBE_PY, str(script), str(guardian_dir), *args],
                          capture_output=True, text=True, timeout=180,
                          env=_child_env(extra_env))
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _checkout_with_residue(tmp_path: Path, name: str = "checkout") -> Path:
    """A non-production checkout holding one provably inert stray, on its own rule-3 root.

    `.lloyd-data` INSIDE the tree is not a fixture convenience: it is what `app.data_root`'s
    rule 3 says this checkout's data root is, so the trusted store the residue is judged
    against, the quarantine directory the move creates, and the tree the unlink happens in
    are three places that one resolution has to explain. The layout module is copied from the
    tree under test, so the resolver the child loads is this diff's copy.
    """
    tree = tmp_path / name / "lloyd"
    (tree / "app").mkdir(parents=True)
    shutil.copy(ROOT / "app" / "data_root.py", tree / "app" / "data_root.py")
    own_root = AR.data_root_for_tree(tree)
    own_root.mkdir()
    (own_root / "workers.db").write_bytes(b"SQLite format 3\x00" + b"x" * 100)
    stray = tree / "workers.db"
    stray.write_bytes(b"")
    old = NOW - 2 * datawatch.INERT_MIN_AGE_SECONDS
    os.utime(stray, (old, old))
    return tree


def _snapshot_copy(tmp_path: Path, name: str = "lloyd-guardian-snapshot") -> Path:
    """A promoted-snapshot copy of the guardian: `*.py` in a directory that is not a checkout.

    This is the deployed shape as `guardian-stage.sh` builds it and
    `lloyd-guardian.service:29` runs it (`%h/.local/state/lloyd-guardian/bin/guardian.py`).
    It matters here because it is the only shape in which the module-level `DATA_ROOT` can be
    tested against the live root: a copy INSIDE a checkout resolves that checkout's rule-3
    root — which is clause 1 working, not clause 2 failing — and the real
    `~/.local/state/lloyd-guardian/bin` belongs to the running watchdog, which this round does
    not touch.
    """
    dest = tmp_path / name
    shutil.copytree(GUARDIAN_DIR, dest, ignore=shutil.ignore_patterns("__pycache__"))
    assert (dest / "policy.py").is_file(), "the copy is not the module under test"
    assert not (dest.parent / "app" / "data_root.py").exists(), (
        "the snapshot copy landed inside a checkout, so it would resolve that tree and the "
        "deployed shape would not be the thing under test")
    return dest


# ── clause 1: a non-production tree moves residue into its OWN root ──────────

def test_a_non_production_checkout_moves_residue_into_its_own_data_root(tmp_path):
    """The clause as filed: `LLOYD_DATA` unset, the process's tree not the production
    checkout, and the root the move writes into and unlinks from is `<tree>/.lloyd-data` —
    never `<account home>/lloyd-data`.

    Both halves of the write are asserted, because the exposure #3208 narrowed itself to is
    exactly those two: a file CREATED under the root (with its `log.jsonl` witness) and a file
    UNLINKED from the tree. The positive control is the same filename one directory tree over
    — `<tree>/.lloyd-data/quarantine/tree-strays/<STAMP>-workers.db` is on disk — so the
    assertion that the live root's quarantine does NOT hold that name is a fact about which
    root was used, not the silence of a move that never ran. `moved` non-empty and
    `stray_still_in_tree` False pin that the move did run.

    The stamp is 2033 for that reason: `NOW` is a future instant only this node's own `now`
    argument can produce, so a live-root hit on that name could only be this node's bug.
    """
    tree = _checkout_with_residue(tmp_path)
    own_quarantine = AR.data_root_for_tree(tree) / AR.QUARANTINE_DIR_RELATIVE

    out = _run_probe(tmp_path, _MOVE_PROBE, str(tree), str(NOW))

    assert out["lloyd_data_env"] is None, (
        "the child inherited LLOYD_DATA: rule 1 would hide rule 3 and the comparison below "
        "would be between two copies of the same answer")
    assert out["move_root"] == str(AR.data_root_for_tree(tree)), out
    assert out["why"] is None, out
    assert out["refusal"] is None, out
    assert len(out["moved"]) == 1, out
    name, dest = out["moved"][0]
    assert name == "workers.db"
    assert dest == str(own_quarantine / f"{STAMP}-workers.db"), (
        f"the destination named {dest}, which is not this checkout's own root "
        f"({AR.data_root_for_tree(tree)}); the live root is {LIVE_ROOT}")
    assert (own_quarantine / f"{STAMP}-workers.db").is_file(), out
    assert (own_quarantine / "log.jsonl").is_file(), (
        "the move's own audit row is written beside what it moved; a destination that holds "
        "the file but not the line is a relocation nobody can witness")
    assert out["stray_still_in_tree"] is False, out

    assert not (LIVE_QUARANTINE / f"{STAMP}-workers.db").exists(), (
        f"the move created {STAMP}-workers.db under the LIVE quarantine root "
        f"({LIVE_QUARANTINE}) while the same file sits in {own_quarantine}: a process that "
        "is not the watchdog wrote into production data (#3208)")
    assert not (tree / "workers.db").exists(), "the unlink did not happen — the move is the clause"


def test_the_move_refuses_rather_than_guessing_a_root(tmp_path):
    """Clause 1's other half: with no answer from the resolver there is no destination, and a
    move with no destination is a refusal — not the tree's literal, not the account's literal.

    The tree here has no `app/data_root.py` at all, which is the shape a pinned snapshot
    reaches when the checkout it names has moved: the accessor cannot load the module, so it
    answers None, and `quarantine_inert` must raise with the file untouched. This is
    `policy.quarantine_dir`'s None-is-a-refusal rule (#2110) applied one level up, to the root
    instead of the name inside it. The other None branch — a resolver that REFUSES with
    `DataRootMissing`, an unmarked production root — is pinned by
    `tests/test_guardian_inert_stray.py::test_the_move_refuses_a_root_that_cannot_be_resolved_and_leaves_the_stray`.
    """
    tree = _checkout_with_residue(tmp_path)
    shutil.rmtree(tree / "app")

    out = _run_probe(tmp_path, _MOVE_PROBE, str(tree), str(NOW))

    assert out["move_root"] is None, out
    assert "data_root.py" in (out["why"] or ""), (
        f"the refusal names no cause: {out!r} — a reader cannot tell a moved checkout from a "
        "resolver that raised")
    assert "RuntimeError" in (out["refusal"] or ""), out
    assert out["moved"] == [], f"a refusal moved something: {out}"
    assert out["stray_still_in_tree"] is True, out
    assert not (AR.data_root_for_tree(tree) / AR.QUARANTINE_DIR_RELATIVE).exists(), (
        "the destination directory was created by a process that had no answer for where the "
        "destination is")
    assert not (LIVE_QUARANTINE / f"{STAMP}-workers.db").exists(), out


# ── clause 2: the deployed shape is unchanged ────────────────────────────────

def test_the_deployed_shape_still_resolves_the_live_root_and_names_its_paths(tmp_path):
    """The promoted shape — a snapshot outside any checkout, `LLOYD_DATA` unset,
    `.lloyd-data-root` present — still resolves to the account's `lloyd-data` (read off passwd
    through the resolver, not spelled by this file), so the running watchdog's quarantine
    destination does not move when this lands.

    The snapshot copy is the only honest way to pin this from a round's worktree: a copy INSIDE
    a checkout correctly resolves that checkout's rule-3 root — which is clause 1 working, not
    clause 2 failing — and the real promoted directory belongs to the running watchdog.
    `process_data_tree()`'s third answer (`REPO`, the checkout the watchdog guards) is what the
    copy exercises, and the first assertion below prints it so a reader can see which branch
    answered.

    Clause 2 is the no-regression half, and it is the reason the fix cannot be a blanket raise:
    `lloyd-guardian.service` runs the snapshot with no `LLOYD_DATA` exported, so a resolution
    that failed in that shape would leave the watchdog with no log files to read. It is
    asserted on the accessor AND on the two read-only constants computed at import
    (`policy.LOG_FILES`, `policy.KG_DB`), because importing `policy` is where a raise would
    actually land.

    The marker is asserted present before anything else: on a box without it the resolver
    refuses, and every assertion below would be about a refusal instead of about the deploy
    shape.
    """
    live = AR.live_checkout()
    assert not AR.tree_is_worktree(live), (
        f"{live} looks like a linked worktree, so rule 3 would answer here and this node "
        "would be pinning a difference the resolver intends")
    assert (LIVE_ROOT / AR.DATA_ROOT_MARKER).is_file(), (
        f"{LIVE_ROOT} has no {AR.DATA_ROOT_MARKER}: the deploy shape this node pins does not "
        "exist on this box, so nothing below could distinguish a fix from a refusal")

    out = _run_probe(tmp_path, _ROOT_PROBE, "", guardian_dir=_snapshot_copy(tmp_path))

    assert out["lloyd_data_env"] is None, "the child inherited LLOYD_DATA"
    assert out["process_tree"] == str(live), (
        f"the snapshot copy resolved to {out['process_tree']}, not the production checkout "
        f"{live} it was staged to guard")
    assert out["resolved"] == str(LIVE_ROOT), out
    assert out["source"] == "resolver", out
    assert out["quarantine_dir"] == str(LIVE_ROOT / AR.QUARANTINE_DIR_RELATIVE), out
    assert out["policy_data_root"] == str(LIVE_ROOT), (
        f"the module-level root the watchdog imports is {out['policy_data_root']}, not the "
        f"live root {LIVE_ROOT}: every read-only consumer moved with the constant")
    assert out["policy_data_root_source"] == "resolver", out
    assert out["log_files"] == [f"{LIVE_ROOT}/logs/server.err", f"{LIVE_ROOT}/logs/mcp.err"], (
        f"the error-rate check stopped naming the two files it names today "
        f"({out['log_files']}) — a watchdog whose LOG_FILES moved is one that silently greps "
        "nothing")
    assert out["kg_db"] == str(AR.kg_store_for_root(LIVE_ROOT)), out
    assert out["kg_db_source"] == "resolver", out


def test_the_probes_run_on_the_guardian_interpreter():
    """Which interpreter the probes above actually used.

    `PROBE_PY` prefers `/usr/bin/python3`, the one `lloyd-guardian.service` runs the pinned
    snapshot on. On a box without it the nodes still run on pytest's interpreter, and this node
    makes that substitution a printed fact rather than an assumption — the same honesty node
    `tests/test_guardian_sink_confinement.py` carries for the same reason.
    """
    assert Path(PROBE_PY).is_file(), PROBE_PY
    proc = subprocess.run([PROBE_PY, "-c",
                           "import sys; sys.stdout.write(sys.version.split()[0])"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    print(f"\n#3208 probe interpreter: {PROBE_PY} (python {proc.stdout.strip()})")
