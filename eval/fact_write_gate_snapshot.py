#!/usr/bin/env python3
"""#2344 — the noop/on fact-tree pair the write gate's arming bar needs.

`config.yaml` reserves `knowledge_graph.write_gate.mode: "on"` and says what
arming it would take: "an explicit decision from Alan plus a measured LloydMemEval
`knowledge_update` gain, never a count of logged rows". The measurement has been
ungettable, for one mechanical reason: every arm of `eval/run_memory_eval.py`
that renders a `<facts>` block resolves its store through `app.paths`
(`LLOYD_FACTS_ROOT` / `LLOYD_KG_DB`), so both sides of the comparison have always
read the SAME tree — the live one, at `noop`, where `gate_write` records an
UPDATE as an ADD and expires nothing. There is no second tree to compare it with,
which is why the 21,174-row decision log cannot produce the CI the bar names.

This builds the two trees. One fixed write sequence
(`WRITE_SEQUENCE`) is replayed twice through `agent_mcp.fact_write_gate.gate_write`
— the production entry point, called exactly the way `agent_mcp/facts.py:662`
calls it, with `mode_` left unset so `mode()` reads `LLOYD_FACT_WRITE_GATE` —
into two isolated snapshot dirs that differ ONLY by that environment variable:

    <out>/noop/   facts/  kg.sqlite  fact-write-gate.jsonl
    <out>/on/     facts/  kg.sqlite  fact-write-gate.jsonl

Where the decision named a prior fact, the `on` tree carries it with
`expired_at` stamped and the `noop` tree carries the same fact still active — the
one field, on one dict, that the whole arming question turns on. `noop` is the
mode that ships; `on` is exercised here, in a snapshot, and nowhere else: the
builder refuses a root that resolves under `production_data_root()`, points every
reader at the snapshot for the duration of the replay and restores the module
after, and never asks to arm anything in a running process.

Each emitted dir is a `--fact-snapshot DIR` for
`eval/run_memory_eval.py --arms prefetch,prefetch_rel`, which renders the same
question twice, once per tree: the paired `knowledge_update` CI the comment
names as the bar. Retrieval and generation are unchanged; the only thing that
moved is which store the `<facts>` block was read from.

**Where the trees go, and why not a tmpfs (#2775).** `--out` has to be on disk and
outside the data root. `refuse_production_root` rules the live data root out on
purpose, and a tmpfs is the wrong answer twice over: a seeded pair is gigabytes of
scratch, and `/tmp` on this host is sized at 1,048,576 inodes total — of which one
seeded PAIR costs 125,626, the ten of them the #2768 round left behind cost
816,670, and 816,670 is 97 % of everything that filesystem held. The documented
place is a checkout's own gitignored data dir, which `git status` never sees and
which is not the live root.

**Seeding (#2775).** `--seed-from` materialises a real facts tree into each leg by
SHARING the seed's file inodes — a leg then costs its directory entries, not a
copy of the corpus — and every write a leg makes unlinks before it writes, so a
replayed fact can never reach through a shared inode back into the seed. A seed
the filesystem cannot link (a corpus on disk, `--out` on another mount) is copied
file by file rather than raising `EXDEV` out of the middle of a run, and a build
the `--out` filesystem cannot hold is refused before it creates the directory.
Only a filesystem that KEEPS AN INODE TABLE can be refused: the btrfs under
`/home`, where the documented `--out` lives, reports no table at all, and `0`
from there means "nothing to price", not "no room".

Run (from a round's worktree, whose `.lloyd-data/` is gitignored):
    cd ~/lloyd-work/<round_id>/home/lloyd
    .venvs/lloyd/bin/python eval/fact_write_gate_snapshot.py \\
        --out .lloyd-data/eval/fact-write-gate/pair
    .venvs/lloyd/bin/python eval/run_memory_eval.py run --arms prefetch,prefetch_rel \\
        --fact-snapshot .lloyd-data/eval/fact-write-gate/pair/noop \\
        --out-dir .lloyd-data/eval/fact-write-gate/pair/runs
    # …and again with --fact-snapshot …/pair/on

    # the seeded pair, over a corpus copied OUT of the live tree first (#2768):
    .venvs/lloyd/bin/python eval/fact_write_gate_snapshot.py \\
        --out .lloyd-data/eval/fact-write-gate/pair \\
        --seed-from .lloyd-data/eval/fact-write-gate/corpus/facts

Pinned by `tests/test_fact_write_gate_snapshot.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import errno
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_mcp import _shared, fact_write_gate as gate, facts, retrieval  # noqa: E402
from app import kg_store, paths  # noqa: E402
from app.data_root import production_data_root  # noqa: E402
from app.fact_ids import assign_ids  # noqa: E402

#: The modes a snapshot pair differs by. `noop` is what ships
#: (`config.yaml`: `knowledge_graph.write_gate.mode`); `on` also supersedes, and
#: is RESERVED (#1487) — this module is the measurement that has to come first.
MODE_PAIR: tuple[str, ...] = ("noop", "on")

#: The layout `eval/run_memory_eval.py --fact-snapshot DIR` expects of a tree.
#: Named here so the runner and its tests take the shape from one place.
SNAPSHOT_FACTS_SUBDIR = "facts"
SNAPSHOT_KG_DB_NAME = "kg.sqlite"
SNAPSHOT_LOG_NAME = "fact-write-gate.jsonl"

ENTITY = "Relay"

#: The one write sequence both trees replay, in order, as
#: `(entity, category, fact_text)`. Fixed on purpose: a paired comparison is
#: only paired if both sides ran the same writes, so this is a fixture and not a
#: parameter — a third mode or a longer sequence is a new measurement, not a
#: rerun of this one.
#:
#:   1. a prior fact, which is the UPDATE candidate's target — the file is empty
#:      when it lands, so `shortlist` finds no candidate and it is an ADD the
#:      gate never asks djev about;
#:   2. a richer restatement of it that satisfies `fact_write_gate.contains`
#:      (every content word and the number 8182 of #1 reappear, and the new text
#:      says more), so the decision over it is the one that can read `update`;
#:   3. a fact in ANOTHER category, which lands in its own file, gets no
#:      candidate, and is an ADD under both modes — the control that keeps "the
#:      two trees differ" from being an artefact of writing less in one of them.
#:
#: #1 ends in a full stop and #2 does not repeat it ("bastion." vs "bastion,").
#: Containment is a word-set test, so #2 must say every word #1 said; the
#: punctuation is what keeps the prior distinguishable as a STRING afterwards,
#: which is how a reader of a rendered `<facts>` block tells the two trees apart.
WRITE_SEQUENCE: tuple[tuple[str, str, str], ...] = (
    (ENTITY, "state", "The Relay relay listens on port 8182 behind the bastion."),
    (ENTITY, "state", "The Relay relay now listens on port 8182 behind the "
                      "bastion, and on the tailnet."),
    (ENTITY, "history", "Relay was first deployed in 2026-05."),
)


class SnapshotRefused(ValueError):
    """A snapshot root this builder will not write."""


def refuse_production_root(root: Path, what: str = "snapshot root") -> Path:
    """Resolve `root`, refusing anything at or under the live data root.

    The builder re-points process-wide module attributes, so a tree that
    resolved under `~/lloyd-data` would be a replay writing into the store that
    live turns read while this was running. An eval artifact is never worth
    that, and nothing about needing the flag should require trusting a typo: a
    missing or misplaced `--out` is caught here, on the resolved path, before a
    byte is written.
    """
    resolved = Path(root).expanduser().resolve()
    prod = Path(production_data_root()).resolve()
    if resolved == prod or prod in resolved.parents:
        raise SnapshotRefused(
            f"{what} {resolved} resolves under production_data_root() {prod}; "
            f"this builder never writes the live tree — point --out somewhere else")
    return resolved


#: The errno set meaning "no link can exist HERE" rather than "this link is being
#: refused": seed and `--out` on different filesystems (`EXDEV` — the ordinary
#: case, a corpus on disk and a scratch `--out` somewhere else), a filesystem with
#: no hardlink support at all (`EOPNOTSUPP`/`ENOSYS`), a filesystem out of link
#: slots (`EMLINK`), and a mount that declines unprivileged links (`EPERM`,
#: `EACCES`, as some overlay/fuse mounts do). Each is answered the same way: copy
#: that one file. Anything else propagates rather than being silently copied.
_LINK_REFUSED: frozenset[int] = frozenset(
    {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK,
     errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS})

#: The one name in a seeded facts root that is COPIED rather than linked, because
#: it is the one a leg is allowed to write: `isolated_tree` points
#: `_shared.ALIASES_PATH` at `<facts_root>/entity-aliases.json`. Every other
#: seeded file is safe to share only because `_write_list` unlinks before it
#: writes; an alias-writing path added later would not inherit that protection.
SEEDED_COPY_NAMES: frozenset[str] = frozenset({"entity-aliases.json"})

#: The documented `--out`, spelled as a constant so `--out`'s help and the refusal
#: messages cannot drift from the module docstring's example (#2775 clause 5). It
#: is relative to the checkout you run from — a round's worktree, whose
#: `.lloyd-data/` `.gitignore:28` already excludes.
DOCUMENTED_OUT = ".lloyd-data/eval/fact-write-gate/pair"

#: Inodes a leg needs beyond the seeded corpus: the leg dir, `kg.sqlite`, the
#: gate's decision log, and room for whatever entity dirs and files the replay's
#: own writes create. Deliberately generous in the expensive direction — a
#: preflight that under-prices is the bug this rail exists to catch, and 32
#: inodes is nothing beside the 62,812 a copied leg really costs.
LEG_INODE_SLACK = 32

#: …and the same for the `--out` root: one dir per leg plus `manifest.json`.
OUT_INODE_SLACK = 8


def tree_footprint(root: Path) -> tuple[int, int]:
    """`(files, dirs)` under `root`.

    Directories are counted because they are the part of a seeded leg that cannot
    be shared: a hardlink may not name a directory, so a corpus of 62,806 files in
    a few thousand entity dirs still bills those dirs once per leg. Sharing the
    files is what took a leg from 62,812 inodes down to its directory count.
    """
    files = dirs = 0
    for _dirpath, dirnames, filenames in os.walk(root):
        dirs += len(dirnames)
        files += len(filenames)
    return files, dirs


def planned_inode_need(*, legs: int, seed: Path | None = None) -> int:
    """The inodes a build of `legs` legs will take out of the `--out` filesystem.

    Priced as a full copy of the seed per leg even though the normal path links it:
    a filesystem refuses a link at the `os.link` that fails, not in advance, so the
    run this has to cover is the one where every link was refused. A seeded `--out`
    on a nearly-full filesystem is therefore refused even where linking would have
    fitted — the safe direction, and the one #2775 clause 4 asks for.
    """
    per_leg = LEG_INODE_SLACK
    if seed is not None:
        files, dirs = tree_footprint(Path(seed))
        per_leg += files + dirs
    return legs * per_leg + OUT_INODE_SLACK


def _statvfs_for(path: Path) -> os.statvfs_result:
    """`statvfs` for the nearest EXISTING ancestor of `path`.

    Existing, because `--out` has not been created at the moment it is being
    priced, and `statvfs` has no opinion about a path that is not there.
    """
    p = Path(path)
    while not p.exists():
        parent = p.parent
        if parent == p:
            raise SnapshotRefused(f"{path} has no existing ancestor to statvfs")
        p = parent
    return os.statvfs(p)


def free_inodes(path: Path) -> int:
    """Free inodes on the filesystem `path` sits on.

    Only meaningful where an inode table exists; the preflight consults
    `inode_capacity`, which is the one that knows a `0` can mean "no table".
    """
    return _statvfs_for(path).f_ffree


def inode_capacity(path: Path) -> tuple[int, int]:
    """`(total, free)` inodes for `path`'s filesystem, from one `statvfs`.

    A filesystem that keeps no inode table reports `(0, 0)` — which is every-
    thing-under-`/home` on this host, btrfs, including the live data root and
    every round worktree, because its inodes are 64-bit numbers rather than slots
    in a fixed table. That is NO BUDGET TO CHECK, not ZERO ROOM: a guard that
    conflates them refuses every seeded build on the round's own filesystem while
    believing it has just prevented a disaster. That conflation is what this
    function exists to prevent; `preflight_inodes` was caught doing it on the
    first real measurement run of this very round.
    """
    st = _statvfs_for(path)
    return st.f_files, st.f_ffree


def preflight_inodes(out: Path, *, legs: int, seed: Path | None = None) -> dict:
    """Refuse a build the `--out` filesystem cannot hold, before a byte is written.

    The #2768 run reached 81 % of a 1,048,576-inode tmpfs and its next pair cost
    251,252 inodes against the 210,276 that were free: one build more and every
    `mktemp` and `socket()` on the host fails, which is the 2026-09-29 failure
    class. The guardian's `tmpwatch` names that state and deletes nothing by
    design ("evidence and an alarm, never a deleter",
    `agent-services/guardian/tmpwatch.py:18`), so the only writer that can stop it
    is the one about to fill it.

    `checked: False` in the return is the no-inode-table case: the run was not
    priced, and a reader of the manifest must be able to tell that from a
    `free` that happened to be large.
    """
    need = planned_inode_need(legs=legs, seed=seed)
    total, free = inode_capacity(out)
    if total == 0:
        return {"free": None, "needed": need, "legs": legs, "checked": False}
    if free < need:
        raise SnapshotRefused(
            f"--out {out} sits on a filesystem with {free} free inodes; this run "
            f"needs {need} ({legs} legs priced at a full copy of the seed plus "
            f"{LEG_INODE_SLACK} slack each). Refusing rather than filling the "
            f"filesystem — point --out at roomier on-disk scratch such as "
            f"{DOCUMENTED_OUT}, never at a tmpfs")
    return {"free": free, "needed": need, "legs": legs, "checked": True}


def resolve_seed(seed: Path) -> Path:
    """Validate `--seed-from` and hand back its resolved facts root.

    The production-root rail applies to the READ side too: a seed that resolves
    under the live root would let the nightly rebuild move the corpus between the
    two legs, so the pair's arms differ by the corpus rather than by the gate.
    Copy the tree out first — the refusal says so.
    """
    src = refuse_production_root(Path(seed), "seed facts root")
    if not src.is_dir():
        raise SnapshotRefused(
            f"seed facts root {src} is not a directory; --seed-from names a facts "
            f"root in the `<corpus>/facts` layout")
    return src


def _link_or_copy_file(src: Path, dest: Path) -> bool:
    """Put `src`'s content at `dest` sharing its inode; copy it where the filesystem
    cannot link. True when the content had to be copied.

    Unlinking first is not housekeeping: `os.link` refuses an existing name, and a
    re-run into a leftover `--out` has to overwrite rather than abort.
    """
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
        return False
    except OSError as exc:
        if exc.errno not in _LINK_REFUSED:
            raise
        shutil.copy2(src, dest)
        return True


def link_or_copy_tree(src: Path, dest: Path,
                      *, copy_only: frozenset[str] = frozenset()) -> dict:
    """Mirror the read-only tree `src` under `dest`, sharing its inodes wherever the
    filesystem allows (#2775).

    This replaces `shutil.copytree`, which bills one inode and one data block PER
    CORPUS FILE PER LEG: the #2768 round's 62,806-file corpus cost 62,812 inodes a
    leg, 125,626 a pair, and ten of them took `/tmp` to 816,670. The seed is only
    ever read — the replay's writes unlink before they write, see `_write_list` —
    so every file of it can be a second NAME for the same inode, and a leg then
    costs its directory entries and nothing else.

    The fallback is per FILE rather than a whole-tree `copytree` retry on the first
    `EXDEV`, because a tree can hold one file on a different mount, and because a
    walk that survives a single refused link is the walk that also survives a
    tmpfs-mounted `--out` over a disk seed.
    """
    counts = {"files": 0, "linked": 0, "copied": 0, "dirs": 0}
    dest.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        (dest / rel).mkdir(parents=True, exist_ok=True)
        counts["dirs"] += len(dirnames)
        for name in sorted(filenames):
            origin = Path(dirpath) / name
            target = dest / rel / name
            if origin.is_symlink():
                # Copy the NAME, not the target: a hardlink to a symlink is not
                # portable, and re-pointing it at the seed's own target would make
                # the leg depend on the seed still being there.
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(os.readlink(origin))
                continue
            counts["files"] += 1
            if name in copy_only:
                if target.exists() or target.is_symlink():
                    target.unlink()
                shutil.copy2(origin, target)
                counts["copied"] += 1
            elif _link_or_copy_file(origin, target):
                counts["copied"] += 1
            else:
                counts["linked"] += 1
    return counts


def seed_leg(tree: Path, seed: Path) -> dict:
    """Materialise the read-only seed into one leg, sharing its inodes instead of
    paying a corpus per arm. Returns `link_or_copy_tree`'s counts for the manifest.
    """
    src = resolve_seed(seed)
    dest_facts = Path(tree) / SNAPSHOT_FACTS_SUBDIR
    counts = link_or_copy_tree(src, dest_facts, copy_only=SEEDED_COPY_NAMES)
    counts["seed"] = str(src)
    return counts


def fact_file(facts_root: Path, entity: str, category: str) -> Path:
    """The one file for (entity, category), as `agent_mcp/facts.py` names it."""
    return facts_root / entity / f"{entity}-{category}.md"


def _read_list(fact_file_path: Path) -> list[dict]:
    """The fact list out of a snapshot's own file, `[]` if it has no file yet."""
    if not fact_file_path.exists():
        return []
    fm = _shared._parse_fact_frontmatter(fact_file_path.read_text(encoding="utf-8"))
    return list(fm.get("facts") or [])


def _write_list(fact_file_path: Path, entity: str, category: str,
                entries: list[dict], now_iso: str) -> None:
    """Write the file back in the shape `agent_mcp/facts.py` writes it.

    Same front matter emitter and same body lines as `_fact_add`'s
    `atomic_write_text` call (`agent_mcp/facts.py:688-690`), because the point of
    a snapshot is that the real read path — `retrieval.get_facts_sync`, and
    through it the prefetch arm — reads it exactly as it reads production.

    The atomic rename is still left out; what was ADDED (#2775) is the unlink. A
    seeded leg SHARES its corpus files with the seed by hardlink, and
    `write_text` on a name with another link open writes through that inode: the
    seed corpus — the frozen artifact both legs are measured against — picks up
    the leg's reconstruction, and the `noop` leg's write leaks through the same
    inode into the `on` leg, erasing the differential the pair exists to measure.
    Unlinking first makes the write copy-on-write: the new file gets a fresh
    inode and every other name for the old one keeps the bytes it was seeded
    with. One writer, this process, is still the only atomicity claimed.
    """
    frontmatter = {"type": "facts", "entity": entity, "category": category,
                   "facts": entries, "last_updated": now_iso}
    body = (f"\n# {entity} - {category}\n\n**Entity:** {entity}\n"
            f"**Category:** {category}\n**Fact Count:** {len(entries)}\n")
    fact_file_path.parent.mkdir(parents=True, exist_ok=True)
    if fact_file_path.exists() or fact_file_path.is_symlink():
        fact_file_path.unlink()
    fact_file_path.write_text(
        _shared._write_fact_frontmatter(frontmatter) + body, encoding="utf-8")


@contextlib.contextmanager
def isolated_tree(tree: Path, mode: str):
    """Point every fact reader at `tree`, arm the gate with `mode`, restore after.

    The seam the whole builder stands on. `app.paths` resolves
    `VAULT_FACTS_ROOT`/`VAULT_KG_DB` from the environment ONCE at import, so a
    child process would be one way to get two trees; this is the other, and it is
    the one a test can use without spawning an interpreter — the same six handles
    `tests/test_fact_write_gate.py::tree` patches, plus the read caches those
    three modules memoise, which have to be dropped or the second tree is read
    through the first one's cache. `LLOYD_FACT_WRITE_GATE` is set here rather
    than passed to `gate_write(mode_=…)`, because that is how the write path
    works: `agent_mcp/facts.py` never passes `mode_`, so `mode()` is the thing
    under test.
    """
    tree = refuse_production_root(tree)
    facts_root = tree / SNAPSHOT_FACTS_SUBDIR
    facts_root.mkdir(parents=True, exist_ok=True)
    saved = {
        "shared_facts_root": _shared.FACTS_ROOT,
        "shared_aliases": _shared.ALIASES_PATH,
        "facts_root": facts.FACTS_ROOT,
        "retrieval_facts_root": retrieval.FACTS_ROOT,
        "log": paths.FACT_WRITE_GATE_LOG,
        "store_path": kg_store._default_path,
        "mode_env": os.environ.get(gate.MODE_ENV),
    }
    try:
        os.environ[gate.MODE_ENV] = mode
        _shared.FACTS_ROOT = facts_root
        _shared.ALIASES_PATH = facts_root / "entity-aliases.json"
        facts.FACTS_ROOT = facts_root
        retrieval.FACTS_ROOT = facts_root
        _shared._invalidate_entity_dirs_cache()
        retrieval._entity_index_cache = None
        retrieval._alias_surface_cache = None
        retrieval._fact_file_cache.clear()
        paths.FACT_WRITE_GATE_LOG = tree / SNAPSHOT_LOG_NAME
        kg_store.configure(tree / SNAPSHOT_KG_DB_NAME)
        yield facts_root
    finally:
        _shared.FACTS_ROOT = saved["shared_facts_root"]
        _shared.ALIASES_PATH = saved["shared_aliases"]
        facts.FACTS_ROOT = saved["facts_root"]
        retrieval.FACTS_ROOT = saved["retrieval_facts_root"]
        paths.FACT_WRITE_GATE_LOG = saved["log"]
        _shared._invalidate_entity_dirs_cache()
        retrieval._entity_index_cache = None
        retrieval._alias_surface_cache = None
        retrieval._fact_file_cache.clear()
        # Restore the pointer by assignment, NOT by `configure()`: that route
        # provisions an absent database, and the path being restored here may be
        # a tmp dir a previous test already removed. `reset()` then closes what
        # this context opened, so the next `store()` reopens the restored path.
        kg_store._default_path = saved["store_path"]
        kg_store.reset()
        if saved["mode_env"] is None:
            os.environ.pop(gate.MODE_ENV, None)
        else:
            os.environ[gate.MODE_ENV] = saved["mode_env"]


def replay_tree(tree: Path, mode: str, *, now_iso: str | None = None,
                seed: Path | None = None) -> dict:
    """Replay `WRITE_SEQUENCE` into `tree` under gate mode `mode`; report it.

    Returns the per-write ledger: what the gate said, what the write TOOK, and
    the fact id the decision named — the reader of this artifact should not have
    to open the markdown to know whether the tree shows anything.

    `seed`, when given, is materialised into the leg FIRST, outside
    `isolated_tree`, so the leg's replay finds a real corpus to differ on and the
    read caches are cleared after the corpus is in place rather than before it.
    """
    now_iso = now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
    writes: list[dict] = []
    seeded = seed_leg(Path(tree), seed) if seed is not None else None
    with isolated_tree(Path(tree), mode) as facts_root:
        for entity, category, text in WRITE_SEQUENCE:
            path = fact_file(facts_root, entity, category)
            entries = _read_list(path)
            before = {e.get("id"): bool(e.get("expired_at"))
                      for e in entries if isinstance(e, dict)}
            took, decision = gate.gate_write(entity, category, text, entries,
                                             now_iso=now_iso)
            if took != "noop":
                entries.append({"fact": text, "confidence": 0.9, "category": category,
                                "id": None, "created_at": now_iso, "valid_at": None,
                                "invalid_at": None, "expired_at": None,
                                "provenance": "STATED", "source_doc": None})
                assign_ids(entries, category)
            _write_list(path, entity, category, entries, now_iso)
            retrieval._fact_file_cache.clear()
            writes.append({
                "category": category, "fact": text, "verdict": decision.verdict
                if decision else None, "took": took,
                "target_fact_id": (decision.target or {}).get("fact_id")
                if decision else None,
                "target_fact": (decision.target or {}).get("fact")
                if decision else None,
                "expired_by_this_write": sorted(
                    k for k, was in before.items()
                    if k is not None and not was
                    and any(e.get("id") == k and e.get("expired_at")
                            for e in entries if isinstance(e, dict))),
            })
        # The whole-tree rebuild, the way `scripts/memory/kg_rebuild.py:932` does
        # it: rows for the tree are replaced, so the index describes exactly the
        # facts on this tree and nothing else.
        stats = kg_store.store().facts_idx.reindex(root=facts_root)
        # Keyed `file::id`, not `id`. `app.fact_ids.next_fact_id` numbers within ONE
        # file, so `fact-001` is a handle inside a category file and an alias across
        # every entity's first fact — keying the report by the bare id would let the
        # `state` file's fact-001 and the `history` file's fact-001 overwrite each
        # other here, and the differential would report a collision as a result.
        facts_by_id = {}
        for entity_dir in sorted(facts_root.iterdir()):
            if not entity_dir.is_dir():
                continue
            for fp in sorted(entity_dir.glob("*.md")):
                rel = str(fp.relative_to(facts_root))
                for e in _read_list(fp):
                    if isinstance(e, dict):
                        facts_by_id[f"{rel}::{e.get('id')}"] = {
                            "fact": e.get("fact"), "expired_at": e.get("expired_at"),
                            "file": rel}
    return {"mode": mode, "tree": str(Path(tree).resolve()), "writes": writes,
            "reindex": stats, "facts": facts_by_id, "seeded": seeded}


def _differential(noop: dict, on: dict) -> dict:
    """The facts whose `expired_at` differs between the two trees, by id."""
    expired_on = {fid for fid, f in on["facts"].items() if f["expired_at"]}
    expired_noop = {fid for fid, f in noop["facts"].items() if f["expired_at"]}
    return {"expired_only_in_on": sorted(expired_on - expired_noop),
            "expired_only_in_noop": sorted(expired_noop - expired_on)}


def build(out: Path, *, modes: tuple[str, ...] = MODE_PAIR,
          now_iso: str | None = None,
          seed: Path | None = None) -> dict:
    """Build one snapshot dir per mode under `out` and report the differential.

    `seed` is a facts root (the `<corpus>/facts` layout) materialised into every
    leg before its replay, so the pair differs on a real corpus instead of on
    three synthetic facts. It is shared by inode, never copied per arm, and the
    run is priced against the `--out` filesystem's free inodes BEFORE the out dir
    is created — both of those are #2775's rails, and both are the builder's, not
    the caller's, because the caller is a shell command line that cannot see
    inodes.
    """
    out = refuse_production_root(Path(out))
    seed_root = None if seed is None else resolve_seed(Path(seed))
    # Before `mkdir`: a refusal that created the directory it was refusing to fill
    # has already spent the inodes it is complaining about.
    capacity = preflight_inodes(out, legs=len(modes), seed=seed_root)
    out.mkdir(parents=True, exist_ok=True)
    reports = {mode: replay_tree(out / mode, mode, now_iso=now_iso, seed=seed_root)
               for mode in modes}
    both = all(m in reports for m in MODE_PAIR)
    diff = _differential(reports["noop"], reports["on"]) if both \
        else {"expired_only_in_on": [], "expired_only_in_noop": []}
    manifest = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "out": str(out), "modes": list(modes),
        "sequence": [{"entity": e, "category": c, "fact": t}
                     for e, c, t in WRITE_SEQUENCE],
        "snapshots": {m: reports[m]["tree"] for m in reports},
        # The #2775 numbers, published rather than left in the caller's stdout:
        # what the seed was, what each leg paid for it, and what the filesystem
        # had to spare. A reader of a leftover `--out` dir should be able to tell
        # a shared build from a copied one without re-running the measurement.
        "seed": None if seed_root is None else str(seed_root),
        "seed_cost": {m: reports[m]["seeded"] for m in reports},
        "inode_budget": capacity,
        "gate_logs": {m: str(out / m / SNAPSHOT_LOG_NAME) for m in reports},
        "writes": {m: reports[m]["writes"] for m in reports},
        "differential": diff,
        "shows_expiry_differential": bool(diff["expired_only_in_on"]),
        "warning": None if diff["expired_only_in_on"] else (
            "no fact is expired in the `on` tree and not in `noop`: the gate "
            "never reached an applied UPDATE in this run. Most often that means "
            "djev did not answer (every failure is ADD), so the pair is a "
            "measurement of nothing and must not be scored."),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                       encoding="utf-8")
    return manifest


def cli_parser() -> argparse.ArgumentParser:
    """The argument surface, in its own function so the documented `--out` example
    is text a test can read back (#2775 clause 5) rather than a line inside `main`.
    """
    ap = argparse.ArgumentParser(
        description="Build the noop/on fact-snapshot pair (#2344).")
    ap.add_argument("--out", required=True,
                    help=f"directory to build <out>/noop and <out>/on under, on "
                         f"on-disk scratch: refused if it resolves under "
                         f"production_data_root(), and refused before anything is "
                         f"written if the filesystem has fewer free inodes than the "
                         f"run needs (#2775). A round's gitignored data dir is the "
                         f"intended place: {DOCUMENTED_OUT}")
    ap.add_argument("--modes", default=",".join(MODE_PAIR),
                    help=f"comma-separated gate modes, default {','.join(MODE_PAIR)}")
    ap.add_argument("--seed-from", default=None,
                    help="a FACTS ROOT (the `<corpus>/facts` layout) to materialise "
                         "into each leg before replaying, sharing the seed's file "
                         "inodes rather than copying a corpus per arm (#2775). "
                         "Refused under production_data_root() — copy the tree out "
                         "first, or the nightly rebuild moves the corpus between "
                         "the legs")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = cli_parser()
    args = ap.parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    bad = [m for m in modes if m not in gate.MODES]
    if bad:
        ap.error(f"unknown gate mode(s) {bad}; MODES is {list(gate.MODES)}")
    try:
        manifest = build(Path(args.out), modes=modes,
                         seed=Path(args.seed_from) if args.seed_from else None)
    except SnapshotRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": manifest["out"], "snapshots": manifest["snapshots"],
                      "differential": manifest["differential"],
                      "shows_expiry_differential":
                          manifest["shows_expiry_differential"]}, indent=1))
    if manifest["warning"]:
        print(f"warning: {manifest['warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
