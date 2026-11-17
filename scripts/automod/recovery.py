"""Recoverability of the state OUTSIDE the diff (backlog #3930).

Every rung on the gate ladder asks whether the candidate build still *works*.
None of them asks whether the round could be *undone*. `git revert` covers the
repository, and the guardian's drill (`scripts/automod/rehearse.py`) proves the
rescue against a scratch clone of the tree plus a 200 from a health endpoint.
But a round also mutates state that no revert reaches, because the mutation is a
consequence of APPLYING the change rather than a path in the diff: a landed
`agent-services/supervisor/**` edit only becomes the supervisor's loaded
program set when `promote._apply_service_changes` runs `supervisorctl
reread`/`update` (promote.py:661-663); a landed `agent-services/systemd/**` edit
only becomes the user manager's unit set when the same function links it into
`~/.config/systemd/user` and calls `daemon-reload` (promote.py:747-751); a
landed `config.yaml` edit only becomes the running configuration when the
services restart and read the file again.

Backups of that state DO exist — an hourly read-only btrfs snapshot of the data
root (`scripts/backup/snapshot-data.sh`), a daily `kg.sqlite` tarball
(`scripts/backup/backup-graph.sh`), sha-guarded per-turn file pre-images
(`agent_mcp/_change_ledger.py::revert`) — so the gap is not "nothing is saved".
It is that **no rung asks whether a restore of the state this round changed
would actually work**, and none asks at all whether the round is undoable.

That is the EvoUndo loop transplanted onto the landing decision: witness →
apply → recover → verify independently → admit. Recovery is credited because the
measured state came back, not because a revert command ran. The paper's sharpest
negative is honoured deliberately: there is no model call anywhere in this
module, because under the original recovery language, four verifier-guided
repair strategies recovered 0 of the 197 naturally occurring failures, while
machinery (state grounding, richer undo operations) recovered 180 of them.

Three states, and the third is the point (`verify_restore`):

  * ``verified``     — every captured field came back byte-equal.
  * ``failed``       — at least one field differs, vanished, or appeared.
  * ``unevaluated``  — the surface could not be read, or has no restore op.

An empty field list is NEVER read as verified. That is the same lesson the `vet`
rung and `guard_vacuity` already carry: a check whose missing input looks like a
pass is a check that reports OK on its own absence.

Stdlib only and deterministic. The witness readers are read-only, and the
apply/restore pair in :func:`rehearse_restore` is injected by the caller, so this
module can be exercised against a scratch fixture and never touches a live store.
Since #4082 the module DOES write — and only inside a directory the caller supplied
and is expected to delete: that is where :func:`_rehearse_surface` copies a
file-scope surface's affected paths, applies the round's changed bytes to the copy
and runs the surface's scratch undo. It execs nothing: `ScratchCtx.runner` is the
callable a `supervisorctl reread`, a `systemctl --user daemon-reload` or a service
restart would go through, every scratch inverse receives it and none reads it, and
the default is :func:`_refuse_daemon_call`, which raises. Run the table by hand::

    .venvs/lloyd/bin/python -m scripts.automod.recovery --surfaces
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

# ── the three verdicts ──────────────────────────────────────────────────────
#: Every field of the witness came back byte-equal after the restore op ran.
VERIFIED = "verified"
#: At least one field differs (or the restore op itself raised).
FAILED = "failed"
#: Nothing was judged: the surface could not be read, no restore op is
#: registered, or the witness had no fields to compare.
UNEVALUATED = "unevaluated"
VERDICTS = (VERIFIED, FAILED, UNEVALUATED)

#: `RecoveryRecord.contract` — whether the round's diff reaches any non-git
#: state at all. The empty case is the round that must still get a record
#: (#3930's acceptance: a per-round record, not a per-effect one), and it is one
#: of the three buckets the ≥30-round split is published over.
CONTRACT_EMPTY = "empty"
CONTRACT_DECLARED = "declared"

#: How many names one ledger event may carry. Same budget as
#: `Gate.EVENT_ID_CAP`'s smaller siblings: an always-present record that scales
#: with the diff would bloat the ledger that everything else reads.
NAME_CAP = 20
#: One reason string, clipped. The prose is for a human reading the ledger months
#: later; `detail` in the same event already carries the one-line summary.
REASON_CAP = 160


@dataclass(frozen=True)
class Surface:
    """One piece of non-git state a round's change can reach.

    `triggers` are repo-relative path prefixes — the same `startswith` judgement
    `promote._apply_service_changes` and `promote.restart_needed` already make,
    so a surface cannot be triggered by a path the promoter would ignore.
    `digest` says in words what is measured, `op` names the reader that measures
    it, and `inverse` names the operation that puts the state back — or is
    ``None`` for an effect with no true inverse, which is the video's honest
    category and is recorded as `unrecoverable`, never silently admitted.
    """

    name: str
    triggers: tuple[str, ...]
    digest: str
    op: str
    inverse: str | None

    def to_dict(self) -> dict:
        return {"name": self.name, "triggers": list(self.triggers),
                "digest": self.digest, "op": self.op, "inverse": self.inverse}


#: The table. Every trigger is a path a real round has landed, and every inverse
#: is an operation that already exists in this repo — nothing here invents a
#: restore route, and a surface with no route says so.
SURFACES: tuple[Surface, ...] = (
    Surface(
        name="supervisor-program-definitions",
        triggers=("agent-services/supervisor/",),
        digest="sha256 over the sorted (relpath, bytes) of every file under "
               "agent-services/supervisor/ in the live tree",
        op="supervisor_conf",
        inverse="`supervisorctl -c agent-services/supervisord.conf reread && update` "
                "from the reverted tree — the same pair promote.py:661-663 applies",
    ),
    Surface(
        name="systemd-user-unit-set",
        triggers=("agent-services/systemd/",),
        digest="sha256 over the sorted (name, realpath-or-'dangling') of every entry "
               "in ~/.config/systemd/user, plus the file count under "
               "agent-services/systemd/ in the live tree",
        op="systemd_units",
        inverse="re-link the reverted tree's units into ~/.config/systemd/user and "
                "`systemctl --user daemon-reload` — promote.py:747-751 is that op",
    ),
    Surface(
        name="guardian-pinned-snapshot",
        triggers=("agent-services/guardian/", "agent-services/systemd/lloyd-guardian.service"),
        digest="sha256 over the sorted (name, bytes) of the staged guardian snapshot "
               "the unit's ExecStartPre re-stages from",
        op="guardian_snapshot",
        # Re-staging the snapshot is `systemctl --user restart lloyd-guardian`
        # (promote.py:756), and the only thing that runs it is the promoter that
        # has already landed. There is no way to put this state back from inside
        # a gate, so the round says so out loud instead of being admitted.
        inverse=None,
    ),
    Surface(
        name="knowledge-graph-store",
        triggers=("app/kg_store.py",),
        digest="total row count over every table of the production kg.sqlite, "
               "through promote.count_kg_rows (the read-only door)",
        op="kg_rows",
        inverse="daily tarball restore (scripts/backup/backup-graph.sh + its "
                "restore route): day-granular, so it reverts the store but not "
                "one round's rows",
    ),
    Surface(
        name="job-queue-store",
        triggers=("workers/queue.py", "workers/pool.py", "app/autonomy.py"),
        digest="row counts of the `jobs` and `runs` tables of the configured "
               "workers.db, opened read-only (uri mode=ro)",
        op="queue_rows",
        inverse="hourly read-only btrfs snapshot + scripts/backup/restore-data.sh: "
                "hour-granular, so it reverts the root but not one round's rows",
    ),
    Surface(
        name="effective-runtime-config",
        triggers=("config.yaml",),
        digest="sha256 of the live config.yaml bytes, plus the parsed value of "
               "the keys named in CONFIG_KEYS below",
        op="runtime_config",
        inverse="restore the reverted bytes and restart the services that read "
                "them (promote.restart_needed returns True for config.yaml)",
    ),
)

#: The `config.yaml`-consumed keys the config witness fingerprints BY NAME, so the
#: digest is a statement about settings a human can read rather than about opaque
#: bytes. All three are live keys of this file (verified against it, not invented),
#: and each one is a knob that changes what a landing does to the running state:
#: `enabled` is whether this loop runs at all, `reuse_rungs` is whether a rung
#: re-runs on a re-gate, and `skip_restart` is precisely whether promoted service
#: definitions ever reach the running set — the effect the supervisor surface's
#: witness is measuring. Read by the same dotted-path walk the config fence uses.
CONFIG_KEYS: tuple[str, ...] = ("automod.enabled",
                               "automod.gate.reuse_rungs",
                               "automod.landing.skip_restart")

# ── step 3: the file-scope rehearsal in a disposable copy (#4082) ───────────
#: surface `op` → the repo-relative paths whose bytes this surface's witness
#: reads, so a rehearsal can copy exactly those and nothing else. Only these
#: three are copyable: a store's state is not a set of files this process may
#: duplicate, and the guardian snapshot's only undo is a unit restart, which a
#: gate must never perform.
SCRATCH_AFFECTED: dict[str, tuple[str, ...]] = {
    "supervisor_conf": ("agent-services/supervisor",),
    "systemd_units": ("agent-services/systemd",),
    "runtime_config": ("config.yaml",),
}

#: surface `op` → why the available restore is COARSER than one round. These two
#: restores are real and they are bulk: running one would restore the backup
#: rather than undo the round, restoring rows some other process wrote since. That
#: is the distinction #3930 step 3 refuses to blur, so these surfaces are never
#: rehearsed and say what they actually have.
COARSER_RESTORE: dict[str, str] = {
    "kg_rows": "coarser than one round — day-granular: the daily `kg.sqlite` "
               "tarball restores the whole store, not one round's rows",
    "queue_rows": "coarser than one round — hour-granular: the hourly read-only "
                  "btrfs snapshot reverts the whole data root, not one round's rows",
}

#: Where a scratch rehearsal reads the user manager's unit directory from when the
#: caller does not say. Same default `read_systemd_units` uses, so the copy is of
#: the directory the real link set actually lives in.
DEFAULT_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"


def _refuse_daemon_call(*argv: object, **kw: object) -> None:
    """The default `runner`: the rehearsal has no route to a daemon call.

    `build_recovery_record` threads a `runner` down into every scratch inverse so a
    caller can hand it a callable that RAISES on any call and prove, from the
    verdict alone, that nothing invoked it. Left unset, THIS is what a scratch undo
    would call — and it raises, naming what it was asked to run, so an accidental
    `supervisorctl reread`, `systemctl --user daemon-reload` or service restart
    inside the rehearsal is a loud failure rather than a live state change made
    from inside a gate.
    """
    raise RuntimeError(f"the scratch rehearsal runs nothing: {argv!r}")


@dataclass(frozen=True)
class ScratchCtx:
    """The disposable copy one surface's rehearsal is confined to.

    `work` is the copy the candidate's bytes get applied to; `base` is the pristine
    snapshot the inverse re-stages from, which is what a `git revert` + re-stage
    would put back. `unit_dir`/`unit_base` are the same pair for the scratch copy of
    the user manager's unit directory, `candidate` is the round's own tree, `touched`
    the repo-relative paths this round changed under this surface. `runner` is
    carried so the caller can prove nothing execs: the inverse builders below
    receive it and never read it.
    """
    work: Path
    base: Path
    candidate: Path
    touched: tuple[str, ...]
    unit_dir: Path | None = None
    unit_base: Path | None = None
    runner: Callable = _refuse_daemon_call


def _gone(path: Path) -> bool:
    """True when an entry is absent, INCLUDING a dangling symlink."""
    return not os.path.lexists(str(path))


def _remove(path: Path) -> None:
    """Delete one entry and follow nothing: a link goes, its target stays."""
    if os.path.islink(str(path)):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif os.path.lexists(str(path)):
        path.unlink()


def _stage(src: Path, dst: Path) -> None:
    """Copy one file, symlink (raw target, not resolved) or directory tree into
    place, replacing whatever is there. The unit of both the copy and the undo."""
    _remove(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and not src.is_symlink():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)


def _copy_paths(src_root: Path, dst_root: Path, rels: Iterable[str]) -> int:
    """Copy the named repo-relative paths from SRC into DST. How many landed."""
    n = 0
    for rel in rels:
        src = src_root / rel
        if _gone(src):
            continue
        _stage(src, dst_root / rel)
        n += 1
    return n


def _copy_dir(src_dir: Path, dst_dir: Path) -> bool:
    """Copy the CONTENTS of one directory. False when there is no such directory."""
    if not src_dir.is_dir():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_dir.iterdir()):
        _stage(entry, dst_dir / entry.name)
    return True


def _restore_paths(ctx: ScratchCtx) -> None:
    """Re-stage every path this round touched from the pristine snapshot, and
    delete what the round created. The file-level undo a revert would perform."""
    for rel in ctx.touched:
        src = ctx.base / rel
        if _gone(src):
            _remove(ctx.work / rel)      # the round created it: undo is to delete
        else:
            _stage(src, ctx.work / rel)


def _restore_links(src_dir: Path, dst_dir: Path) -> None:
    """Put DST back to exactly the entries SRC holds, links with their raw targets.

    Byte-equality of `resolve()`d targets is what the witness digests, so restoring
    the raw target restores the measured state — and a dangling link the base held
    comes back dangling, which is what a re-link from the tracked tree would make it.
    """
    for entry in sorted(dst_dir.iterdir()):
        _remove(entry)
    for entry in sorted(src_dir.iterdir()):
        _stage(entry, dst_dir / entry.name)


def _inverse_restage_files(ctx: ScratchCtx) -> Callable[[], None]:
    """Undo for a surface whose whole file-level effect is its files' bytes: the
    supervisor confs (the file half of `supervisorctl reread`/`update`,
    promote.py:706-709) and `config.yaml` (the bytes a restart would re-read). The
    commands themselves are not the half a gate may run, so `ctx.runner` is unused.
    """
    def undo() -> None:
        _restore_paths(ctx)
    return undo


def _inverse_relink_units(ctx: ScratchCtx) -> Callable[[], None]:
    """Undo for the systemd surface: re-stage the unit files the candidate wrote,
    then re-link the scratch unit dir to exactly the entries it held before the
    apply. That is the file half of promote.py:768-802 without its
    `systemctl --user daemon-reload`, which a gate must never issue.
    """
    def undo() -> None:
        _restore_paths(ctx)
        if ctx.unit_dir is not None and ctx.unit_base is not None:
            _restore_links(ctx.unit_base, ctx.unit_dir)
    return undo


#: surface `op` → scratch-undo factory. A surface absent here has no file-level
#: undo to rehearse — the two stores' restores are coarser than one round and the
#: guardian's is a unit restart — so it keeps the declare-and-witness verdict.
SCRATCH_INVERSES: dict[str, Callable[[ScratchCtx], Callable[[], None]]] = {
    "supervisor_conf": _inverse_restage_files,
    "systemd_units": _inverse_relink_units,
    "runtime_config": _inverse_restage_files,
}


def _link_unit(unit_file: Path, unit_dir: Path) -> None:
    """One symlink into the unit directory, made the way promote.py:768-782 makes
    it: drop any entry that is there — a plain file or a link to somewhere else —
    then link, never write through it."""
    dst = unit_dir / unit_file.name
    _remove(dst)
    unit_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(unit_file, dst)


def _apply_candidate(ctx: ScratchCtx) -> None:
    """Reproduce, inside the scratch copy ONLY, what landing does to the files.

    The candidate's bytes replace the base's at every touched path, and a unit file
    the round changed gets linked into the scratch unit directory exactly as
    `promote._apply_service_changes` links `agent-services/systemd/<name>` into
    `~/.config/systemd/user` (promote.py:768-782), including its skip of a path
    below that tree's `system/` subdir, which the installer never links. What it
    does NOT do is the rest of that function: no `supervisorctl reread`/`update`, no
    `daemon-reload`, no guardian re-staging, no restart. `ctx.runner` is the
    callable that would run them and this never touches it — the guarantee #4082
    clause 1 pins with a runner that raises.
    """
    for rel in ctx.touched:
        src = ctx.candidate / rel
        if _gone(src):
            _remove(ctx.work / rel)      # the round deleted the path
        else:
            _stage(src, ctx.work / rel)
    if ctx.unit_dir is None:
        return
    unit_tree = ctx.work / "agent-services" / "systemd"
    for rel in ctx.touched:
        if not rel.endswith(".service"):
            continue
        unit_file = ctx.work / rel
        if unit_file.parent != unit_tree or not unit_file.is_file():
            continue                     # below system/, or deleted: not linked
        _link_unit(unit_file, ctx.unit_dir)


def _rehearse_surface(surface: Surface, touched: list[str], *,
                      live_root: Path, candidate_root: Path | None,
                      scratch_root: Path | None, unit_dir: Path | None,
                      runner: Callable,
                      inverses: dict) -> dict | None:
    """Apply-and-unwind one file-scope surface inside a caller-supplied copy.

    Copies the surface's affected paths out of `live_root` TWICE — once as the
    working copy, once as the pristine snapshot the undo re-stages from — applies
    the round's changed bytes to the working copy, runs the surface's
    `scratch_inverse` there, and hands the pair to `rehearse_restore`, which digests
    pre, refuses to credit a match when the apply moved nothing, and compares post
    against pre. Every witness read goes through
    `default_readers(work, unit_dir=scratch unit dir)`, so the measured state is the
    copy's: the live tree is read once, to seed the copy, and written never, and the
    user manager's real unit directory is copied, never re-linked.

    Returns None — so the caller keeps the declare-and-witness verdict rather than
    inventing one — only when there is nothing to rehearse: no copy or no candidate
    tree supplied, or the surface declares no file-level undo. A rehearsal that ran
    and came back vacuous reports that itself.
    """
    builder = inverses.get(surface.op)
    if builder is None or scratch_root is None or candidate_root is None:
        return None
    scratch_dir, live_root = Path(scratch_root), Path(live_root)
    work, base = scratch_dir / "work", scratch_dir / "base"
    affected = SCRATCH_AFFECTED[surface.op]
    if not (_copy_paths(live_root, work, affected)
            and _copy_paths(live_root, base, affected)):
        return {"verdict": UNEVALUATED, "witness_captured": False, "mismatch": [],
                "reason": "no scratch rehearsal: none of the affected paths "
                          f"{list(affected)} was present under {live_root} to copy"}

    s_unit: Path | None = None
    s_unit_base: Path | None = None
    if surface.op == "systemd_units":
        # The witness is the manager's LINK set, so the rehearsal needs a copy of
        # it to move and to put back. Refuse rather than measure the live one:
        # a rehearsal whose inverse cannot run is not a rehearsal.
        src_units = Path(unit_dir) if unit_dir else DEFAULT_UNIT_DIR
        s_unit, s_unit_base = scratch_dir / "units", scratch_dir / "units_base"
        if not (_copy_dir(src_units, s_unit) and _copy_dir(src_units, s_unit_base)):
            return {"verdict": UNEVALUATED, "witness_captured": False,
                    "mismatch": [],
                    "reason": f"no scratch rehearsal: {src_units} could not be "
                              "copied, so the link set has no disposable copy"}
    ctx = ScratchCtx(work=work, base=base, candidate=Path(candidate_root),
                     touched=tuple(touched), unit_dir=s_unit, unit_base=s_unit_base,
                     runner=runner)
    readers = default_readers(work, unit_dir=s_unit)
    rec = rehearse_restore(readers.get(surface.op), lambda: _apply_candidate(ctx),
                           builder(ctx), surface=surface.name)
    if rec["verdict"] == FAILED and rec["mismatch"]:
        # The ladder line and the ledger row carry the reason, so the field that
        # failed to come back has to be IN it, not only in a `mismatch` key that
        # dies with `gate.json`.
        rec["reason"] = f"{rec['reason']}: {', '.join(rec['mismatch'][:4])}"
    return rec


def declare_surfaces(changed_paths: Iterable[str]) -> tuple[Surface, ...]:
    """The non-git surfaces a round's diff reaches. PURE: no git, no filesystem.

    A prefix matches a changed path the way the promoter matches it
    (`path.startswith(prefix)`), except that a trigger with no trailing slash is
    an exact path (`config.yaml`, `app/kg_store.py`) so that `app/kg_store.py`
    cannot be triggered by a file that merely starts with those bytes.

    A pure-code round — `app/harness/*.py`, a test, a skill — declares nothing,
    and that empty answer is the datum #3930 step 6 counts as `empty contract`;
    it is not a pass and it is not a skip.
    """
    paths = [str(p) for p in changed_paths]
    out: list[Surface] = []
    for surface in SURFACES:
        for trig in surface.triggers:
            if trig.endswith("/") or trig.endswith(".service"):
                hit = any(p.startswith(trig) for p in paths)
            else:
                hit = any(p == trig for p in paths)
            if hit:
                out.append(surface)
                break
    return tuple(out)


# ── witness capture ─────────────────────────────────────────────────────────


@dataclass
class Witness:
    """Digests of one surface, plus the names that could not be read.

    `errors` and `fields` are both kept because the reader's failure IS the
    measurement: a witness that silently dropped the unreadable field would make
    the post-restore comparison of two equally-short dicts come back `verified`.
    """

    fields: dict[str, str]
    errors: dict[str, str]

    @property
    def readable(self) -> bool:
        """True only when at least one field exists AND nothing failed."""
        return bool(self.fields) and not self.errors

    def to_dict(self) -> dict:
        return {"fields": dict(self.fields), "errors": dict(self.errors),
                "readable": self.readable}


def capture_witness(read: Callable[[], dict[str, str]] | None) -> Witness:
    """Run one surface's reader once. Never raises.

    A missing reader, a raising reader, a reader that returns something that is
    not a mapping, and a reader that returns NO fields all produce a witness with
    `readable == False`. The empty-mapping case is not pedantry: `{}.startswith`
    comparisons are how a check ends up crediting a restore nobody performed.
    """
    if read is None:
        return Witness({}, {"reader": "no reader registered for this surface"})
    try:
        raw = read()
    except Exception as exc:  # noqa: BLE001 — an unreadable surface is a verdict
        return Witness({}, {"reader": f"{type(exc).__name__}: {str(exc)[:120]}"})
    if not isinstance(raw, dict):
        return Witness({}, {"reader": f"reader returned {type(raw).__name__}, not a mapping"})
    fields = {str(k): str(v) for k, v in raw.items()}
    if not fields:
        return Witness({}, {"reader": "reader returned no fields"})
    return Witness(fields, {})


def restore_mismatch(pre: Witness, post: Witness) -> list[str]:
    """Field names that came back wrong: differing, missing, or newly present."""
    if pre.errors or post.errors or not pre.fields or not post.fields:
        return []
    names = sorted(set(pre.fields) | set(post.fields))
    return [n for n in names if pre.fields.get(n) != post.fields.get(n)]


def verify_restore(pre: Witness, post: Witness) -> str:
    """Exactly one of :data:`VERDICTS`, and only one of them is a pass.

    Three things make this a check that can fail rather than a formality:

      * an empty field list on EITHER side is `unevaluated`, never `verified`;
      * a field that appeared is a mismatch, not "the old state is all there";
      * a read error on either side is `unevaluated` — a comparison that could
        not be made must not be reported as a clean one.
    """
    if pre.errors or post.errors:
        return UNEVALUATED
    if not pre.fields or not post.fields:
        return UNEVALUATED
    return VERIFIED if not restore_mismatch(pre, post) else FAILED


def rehearse_restore(read: Callable[[], dict[str, str]] | None,
                     apply_change: Callable[[], None],
                     restore_op: Callable[[], None] | None,
                     *, surface: str = "") -> dict:
    """Witness → apply → recover → verify, over one surface, with everything injected.

    `apply_change` and `restore_op` are arguments, not module state, so a test can
    register a no-op restore and prove this returns `failed` — the clause that
    keeps this from becoming another guard that reports OK on its own missing
    input (the failure mode #703's drill had to carry a broken-path clause for).

    Two of the exits are vacuity guards rather than verdicts about the state:

      * the witness did not move under `apply_change` → `unevaluated`. Nothing
        was mutated, so nothing was recovered, however green the comparison
        looks;
      * `restore_op` is None → `unevaluated` with `unrecoverable: True`. An effect
        with no inverse is named, not admitted.

    A restore op that RAISES is `failed`, not `unevaluated`: the state cannot be
    shown to have come back, and the safe reading of a failing restore is a
    restore that failed.
    """
    pre = capture_witness(read)
    if not pre.readable:
        return {"surface": surface, "verdict": UNEVALUATED, "witness_captured": False,
                "unrecoverable": False, "reason": f"witness unreadable: "
                                                  f"{_first_error(pre)}",
                "mismatch": []}
    if restore_op is None:
        return {"surface": surface, "verdict": UNEVALUATED, "witness_captured": True,
                "unrecoverable": True,
                "reason": "unrecoverable: no restore op registered for this surface",
                "mismatch": []}
    try:
        apply_change()
    except Exception as exc:  # noqa: BLE001 — see the docstring on the raise path
        return {"surface": surface, "verdict": UNEVALUATED, "witness_captured": True,
                "unrecoverable": False,
                "reason": f"apply failed: {type(exc).__name__}: {str(exc)[:100]}",
                "mismatch": []}
    mid = capture_witness(read)
    if mid.readable and not restore_mismatch(pre, mid):
        return {"surface": surface, "verdict": UNEVALUATED, "witness_captured": True,
                "unrecoverable": False,
                "reason": "the applied change moved no witness field, so nothing "
                          "was there to recover",
                "mismatch": []}
    try:
        restore_op()
    except Exception as exc:  # noqa: BLE001 — a restore that raises did not restore
        return {"surface": surface, "verdict": FAILED, "witness_captured": True,
                "unrecoverable": False,
                "reason": f"restore raised {type(exc).__name__}: {str(exc)[:100]}",
                "mismatch": []}
    post = capture_witness(read)
    verdict = verify_restore(pre, post)
    mismatch = restore_mismatch(pre, post)
    reason = ("all captured fields came back byte-equal" if verdict == VERIFIED
              else "post-restore state differs from the witness" if verdict == FAILED
              else f"witness unreadable after restore: {_first_error(post)}")
    return {"surface": surface, "verdict": verdict, "witness_captured": post.readable,
            "unrecoverable": False, "reason": reason, "mismatch": mismatch[:NAME_CAP]}


def _first_error(w: Witness) -> str:
    for name, msg in w.errors.items():
        return f"{name}: {msg}"
    return "no fields captured"


# ── the live readers (read-only; the gate rung is the only caller) ──────────


def _tree_digest(root: Path) -> dict[str, str]:
    """sha256 over sorted (relpath, bytes) of every file under `root`."""
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"{root} holds no files")
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return {"tree": h.hexdigest(), "files": str(len(files))}


def read_supervisor_conf(live_root: Path) -> dict[str, str]:
    """The program definitions supervisor loads from, as one digest.

    Deliberately the FILES and not `supervisorctl status`: the status command
    asks a live process about itself, which makes the witness's availability a
    function of service liveness, and it reports the *current* set rather than
    the definitions the next `reread`/`update` would load.
    """
    return _tree_digest(Path(live_root) / "agent-services" / "supervisor")


def read_systemd_units(live_root: Path,
                       unit_dir: Path | None = None) -> dict[str, str]:
    """What the user manager has been pointed at, plus what the tree offers it.

    Both halves matter: the landing installs one symlink per unit file, so a
    restore that relinks the reverted tree has to move the manager's directory
    AND come back over the tree's own list.
    """
    unit_dir = Path(unit_dir or (Path.home() / ".config" / "systemd" / "user"))
    if not unit_dir.is_dir():
        raise FileNotFoundError(f"{unit_dir} is not a directory")
    h = hashlib.sha256()
    entries = sorted(unit_dir.iterdir())
    for p in entries:
        h.update(p.name.encode())
        try:
            h.update(("->" + str(p.resolve())).encode())
        except OSError:
            h.update(b"<dangling>")
        h.update(b"\0")
    return {"unit_links": h.hexdigest(), "links": str(len(entries)),
            "tracked_units": str(len(list(
                (Path(live_root) / "agent-services" / "systemd").rglob("*.service"))))}


def read_kg_rows() -> dict[str, str]:
    """Total rows in the production KG store, through the door that already exists.

    `promote.count_kg_rows` is reused rather than re-derived: it is the same
    read-only sum the guardian's `data_damage` tripwire compares against, it
    resolves the layout through `app.data_root.kg_store_for_root` instead of
    spelling the path a fourth time (#1525), and it answers `None` for "unreadable"
    — which here must stay an error, not a zero.
    """
    from scripts.automod.promote import count_kg_rows
    rows = count_kg_rows()
    if rows is None:
        raise RuntimeError("kg.sqlite absent or unreadable")
    return {"kg_rows": str(rows)}


def read_queue_rows() -> dict[str, str]:
    """`jobs` and `runs` row counts of the configured workers.db, read-only.

    The path comes from `workers.queue.configured_db_path`, the one spelling
    (a bare `get_queue()` in a non-backend process raises, and `uri mode=ro`
    means this can never be a second writer of the live queue).
    """
    import sqlite3
    from workers.queue import configured_db_path
    db = configured_db_path()
    if not db.exists():
        raise FileNotFoundError(f"{db} absent")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        out: dict[str, str] = {}
        for table in ("jobs", "runs"):
            out[table] = str(con.execute(f"SELECT count(*) FROM '{table}'").fetchone()[0])
        return out
    finally:
        con.close()


def read_runtime_config(live_root: Path) -> dict[str, str]:
    """The bytes the services read at boot, plus the fenced keys by name."""
    path = Path(live_root) / "config.yaml"
    raw = path.read_bytes()
    out = {"config_sha": hashlib.sha256(raw).hexdigest()}
    try:
        import yaml
        cfg = yaml.safe_load(raw.decode("utf-8", "replace")) or {}
        for key in CONFIG_KEYS:
            node = cfg
            for part in key.split("."):
                node = (node or {}).get(part) if isinstance(node, dict) else None
            out[key] = repr(node)
    except Exception as exc:  # noqa: BLE001 — the bytes above still measured
        out["config_keys_error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    return out


#: surface `op` → live reader. `build_recovery_record` looks names up here, and a
#: test injects its own mapping to exercise the same code paths on a scratch
#: fixture. An op missing from this map yields `unevaluated`, not a crash.
#:
#: `unit_dir` exists for the scratch rehearsal (#4082): it is how the reader is
#: pointed at a disposable copy of the user manager's unit directory instead of
#: `~/.config/systemd/user`, which is the only live path any of these readers
#: would otherwise reach. Every other reader is already root-relative.
def default_readers(live_root: Path,
                    unit_dir: Path | None = None) -> dict[str, Callable[[], dict[str, str]]]:
    live_root = Path(live_root)
    return {
        "supervisor_conf": lambda: read_supervisor_conf(live_root),
        "systemd_units": lambda: read_systemd_units(live_root, unit_dir),
        "guardian_snapshot": None,   # type: ignore[dict-item]  # no reader: see SURFACES
        "kg_rows": read_kg_rows,
        "queue_rows": read_queue_rows,
        "runtime_config": lambda: read_runtime_config(live_root),
    }


# ── the per-round record ────────────────────────────────────────────────────


def _clip(text: str, cap: int) -> str:
    return str(text)[:cap]


def _touched_for(op: str, changed_paths: Iterable[str]) -> list[str]:
    """The changed paths lying under one surface's affected roots — the bytes a
    scratch rehearsal of THAT surface has to move. A path that reaches this
    surface only through a config KEY it consumes (a .py file) is not one of them:
    the rehearsal applies files, so such a round comes back `unevaluated` with the
    vacuity guard's own reason rather than a rehearsal that pretends to apply it.
    """
    roots = SCRATCH_AFFECTED.get(op) or ()
    return [s for s in (str(p) for p in changed_paths)
            if any(s == r or s.startswith(r.rstrip("/") + "/") for r in roots)]


def build_recovery_record(changed_paths: Iterable[str],
                          *, readers: dict | None = None,
                          live_root: Path | None = None,
                          scratch_root: Path | None = None,
                          candidate_root: Path | None = None,
                          unit_dir: Path | None = None,
                          runner: Callable | None = None,
                          scratch_inverses: dict | None = None) -> dict:
    """The one record a round gets for its state outside the diff. Observe-only.

    Three states per declared surface, and the record says which one it got:

      * **rehearsed** (#4082, when the caller supplies `scratch_root` +
        `candidate_root`): for a surface with a `scratch_inverse` — the three
        file-scope ones — the affected paths are copied out of `live_root` into the
        caller's disposable directory, the round's changed bytes are applied THERE,
        the inverse undoes them THERE, and the witness decides. `verified` or
        `failed` with `rehearsed: True`, from `rehearse_restore`, which refuses to
        call a digest match a round-trip when the apply moved nothing.
      * **coarser than one round**: `kg.sqlite` and `workers.db` have real restores
        and they are day- and hour-granular bulk snapshots of the whole store, so
        unwinding one would not undo THIS round — it would restore rows every later
        writer had written. `unevaluated`, saying so.
      * **no inverse at all**: `guardian-pinned-snapshot` — its only undo is a unit
        restart, which a gate must never issue. `unrecoverable: <surface>`.

    `runner` is threaded to every scratch inverse so a caller can hand the builder a
    callable that RAISES on any call and prove from the verdict that the rehearsal
    never execed supervisorctl, systemctl or a restart; unset it is
    `_refuse_daemon_call`, which raises the same way. `scratch_inverses` replaces the
    declared inverses for one call — the seam that proves the rehearsal can FAIL.

    Nothing is read at all for a round that declares no surface, and no scratch copy
    is made for one that declares no rehearseable surface: the cost is bounded to
    the contract, exactly as the item's risk section asks.
    """
    declared = declare_surfaces(changed_paths)
    readers = readers if readers is not None else default_readers(
        live_root or Path(__file__).resolve().parent.parent.parent)
    live_root = Path(live_root) if live_root else \
        Path(__file__).resolve().parent.parent.parent
    inverses = scratch_inverses if scratch_inverses is not None else SCRATCH_INVERSES
    changed = [str(p) for p in changed_paths]
    surfaces: list[dict] = []
    unrecoverable: list[str] = []
    for surface in declared:
        if surface.inverse is None:
            unrecoverable.append(surface.name)
            surfaces.append({"name": surface.name, "has_inverse": False,
                             "witness_captured": False, "verdict": UNEVALUATED,
                             "rehearsed": False,
                             # The reason names the surface and the fact, not a
                             # clipped copy of the digest: what the state IS lives
                             # in the table, and a half-sentence in the ledger is
                             # how a reason stops being checkable.
                             "reason": f"unrecoverable: {surface.name} — this state "
                                       f"has no declared inverse to rehearse"})
            continue
        w = capture_witness(readers.get(surface.op))
        rehearsal = None
        if w.readable and surface.op in inverses:
            rehearsal = _rehearse_surface(
                surface, _touched_for(surface.op, changed),
                live_root=live_root, candidate_root=candidate_root,
                scratch_root=scratch_root, unit_dir=unit_dir,
                runner=runner or _refuse_daemon_call, inverses=inverses)
        if rehearsal is not None:
            entry = {"name": surface.name, "has_inverse": True,
                     "witness_captured": w.readable,
                     "verdict": rehearsal["verdict"],
                     # Rehearsed means the undo actually RAN and was graded. An
                     # apply that moved no witness never reached it, so it is not
                     # credited with a rehearsal it did not perform.
                     "rehearsed": rehearsal["verdict"] in (VERIFIED, FAILED),
                     "reason": rehearsal["reason"],
                     "mismatch": list(rehearsal["mismatch"])}
            surfaces.append(entry)
            continue
        reason = f"witness unreadable: {_first_error(w)}" if not w.readable else (
            f"no scratch rehearsal: the available restore is {COARSER_RESTORE[surface.op]}"
            if surface.op in COARSER_RESTORE else
            "no scratch rehearsal: the caller supplied no scratch copy to apply and "
            "unwind the candidate in")
        surfaces.append({"name": surface.name, "has_inverse": True,
                         "witness_captured": w.readable, "verdict": UNEVALUATED,
                         "rehearsed": False, "reason": reason, "mismatch": []})

    verdict = UNEVALUATED
    if surfaces and all(s["verdict"] == VERIFIED for s in surfaces):
        verdict = VERIFIED
    elif any(s["verdict"] == FAILED for s in surfaces):
        verdict = FAILED
    return {
        "observe_only": True,
        "contract": CONTRACT_EMPTY if not declared else CONTRACT_DECLARED,
        "declared": [s["name"] for s in surfaces][:NAME_CAP],
        "unrecoverable": unrecoverable[:NAME_CAP],
        "verdict": verdict,
        "rehearsed": any(s.get("rehearsed") for s in surfaces),
        "surfaces": [{**s, "reason": _clip(s["reason"], REASON_CAP)}
                     for s in surfaces[:NAME_CAP]],
    }


def record_summary(rec: dict) -> str:
    """One line for the rung detail and the ledger's `detail` field."""
    if rec["contract"] == CONTRACT_EMPTY:
        return "no non-git effect declared by this diff (observe-only)"
    parts = []
    for s in rec["surfaces"]:
        state = ("witness" if s["witness_captured"] else "no-witness")
        parts.append(f"{s['name']}={s['verdict']}/{state}")
    line = f"{rec['contract']} contract: " + ", ".join(parts) if parts else \
        f"{rec['contract']} contract"
    if rec["unrecoverable"]:
        line += " | " + ", ".join(f"unrecoverable: {n}" for n in rec["unrecoverable"])
    return _clip(f"observe-only: {line}", 500)


def to_event(rec: dict) -> dict:
    """The compact form that rides the durable gate ledger event.

    Shaped like `vet`'s and `pyright`'s records for the same reason: `gate.json`
    is copied into the round directory and deleted with the worktree, so a
    non-blocking observation preserved only there has no denominator, and #3930's
    step 6 split (verified restore / unrecoverable / empty contract) is a count
    over landings that has to be answerable months later.

    `surface_verdicts` is the per-SURFACE verdict, which is the only form in which
    #4082's rehearsal result survives the round: the record-level `verdict` is one
    value per round and a round that rehearses one surface and cannot rehearse a
    store still aggregates to `unevaluated`. So the census's verified-restore count
    is a count over `surface_verdicts`, and without this field it would have no
    denominator at all.
    """
    surfaces = rec["surfaces"][:NAME_CAP]
    ev: dict = {"observe_only": True,
                "contract": rec["contract"],
                "verdict": rec["verdict"],
                "rehearsed": bool(rec.get("rehearsed")),
                "declared": list(rec["declared"])[:NAME_CAP],
                "unrecoverable": list(rec["unrecoverable"])[:NAME_CAP],
                "surface_verdicts": {s["name"]: s["verdict"] for s in surfaces},
                "witness_captured": {s["name"]: bool(s["witness_captured"])
                                     for s in surfaces}}
    mismatched = {s["name"]: list(s.get("mismatch") or [])[:NAME_CAP]
                  for s in surfaces if s.get("mismatch")}
    if mismatched:
        ev["mismatch"] = mismatched
    if any(not s["witness_captured"] for s in rec["surfaces"]):
        ev["unevaluated_reasons"] = {s["name"]: _clip(s["reason"], REASON_CAP)
                                     for s in rec["surfaces"][:NAME_CAP]
                                     if not s["witness_captured"]}
    return ev


def main(argv: list[str] | None = None) -> int:
    """`--surfaces`: print the checked-in table. `--changed-paths`: declare it."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m scripts.automod.recovery")
    ap.add_argument("--surfaces", action="store_true")
    ap.add_argument("--changed-paths", nargs="*", default=None)
    ap.add_argument("--live-root", default=None)
    args = ap.parse_args(argv)
    if args.changed_paths is not None:
        rec = build_recovery_record(args.changed_paths,
                                    live_root=Path(args.live_root) if args.live_root
                                    else None)
        print(record_summary(rec))
        print(__import__("json").dumps(rec, indent=2, sort_keys=True))
        return 0
    for s in SURFACES:
        print(f"{s.name}\n  trigger: {', '.join(s.triggers)}"
              f"\n  digest:  {s.digest}"
              f"\n  inverse: {s.inverse or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
