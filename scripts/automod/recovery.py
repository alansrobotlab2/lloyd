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

Stdlib only, deterministic, and nothing here writes: the witness readers are
read-only, and the apply/restore pair in :func:`rehearse_restore` is injected by
the caller, so this module can be exercised against a scratch fixture and never
touches a live store. Run the table by hand::

    .venvs/lloyd/bin/python -m scripts.automod.recovery --surfaces
"""

from __future__ import annotations

import hashlib
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
def default_readers(live_root: Path) -> dict[str, Callable[[], dict[str, str]]]:
    live_root = Path(live_root)
    return {
        "supervisor_conf": lambda: read_supervisor_conf(live_root),
        "systemd_units": lambda: read_systemd_units(live_root),
        "guardian_snapshot": None,   # type: ignore[dict-item]  # no reader: see SURFACES
        "kg_rows": read_kg_rows,
        "queue_rows": read_queue_rows,
        "runtime_config": lambda: read_runtime_config(live_root),
    }


# ── the per-round record ────────────────────────────────────────────────────


def _clip(text: str, cap: int) -> str:
    return str(text)[:cap]


def build_recovery_record(changed_paths: Iterable[str],
                          *, readers: dict | None = None,
                          live_root: Path | None = None) -> dict:
    """The one record a round gets for its state outside the diff. Observe-only.

    At this stage the record is a **declaration plus a witness**, never a
    rehearsal: no rung applies the candidate and unwinds it, because #3930 step 3
    (apply-and-unwind in a disposable copy) was deferred — the drill already
    reuses the canary's fixed ports across two concurrent gates, and a second
    scratch environment is a wall-clock cost to decide later. So every declared
    surface whose witness CAN be read comes back `unevaluated` with that reason
    rather than `verified`, which is why the rung may not block anything yet.

    Nothing is read at all for a round that declares no surface, which is what
    keeps the rung free on the ordinary code round: the cost is bounded to the
    contract, exactly as the item's risk section asks.
    """
    declared = declare_surfaces(changed_paths)
    readers = readers if readers is not None else default_readers(
        live_root or Path(__file__).resolve().parent.parent.parent)
    surfaces: list[dict] = []
    unrecoverable: list[str] = []
    for surface in declared:
        if surface.inverse is None:
            unrecoverable.append(surface.name)
            surfaces.append({"name": surface.name, "has_inverse": False,
                             "witness_captured": False, "verdict": UNEVALUATED,
                             # The reason names the surface and the fact, not a
                             # clipped copy of the digest: what the state IS lives
                             # in the table, and a half-sentence in the ledger is
                             # how a reason stops being checkable.
                             "reason": f"unrecoverable: {surface.name} — this state "
                                       f"has no declared inverse to rehearse"})
            continue
        w = capture_witness(readers.get(surface.op))
        reason = (f"witness unreadable: {_first_error(w)}" if not w.readable
                  else "no rehearsal: the gate does not apply-and-unwind the "
                       "candidate (#3930 step 3 deferred)")
        surfaces.append({"name": surface.name, "has_inverse": True,
                         "witness_captured": w.readable, "verdict": UNEVALUATED,
                         "reason": reason})

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
        "rehearsed": False,
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
    """
    ev: dict = {"observe_only": True,
                "contract": rec["contract"],
                "verdict": rec["verdict"],
                "rehearsed": bool(rec.get("rehearsed")),
                "declared": list(rec["declared"])[:NAME_CAP],
                "unrecoverable": list(rec["unrecoverable"])[:NAME_CAP],
                "witness_captured": {s["name"]: bool(s["witness_captured"])
                                     for s in rec["surfaces"][:NAME_CAP]}}
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
