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

Run:
    .venvs/lloyd/bin/python eval/fact_write_gate_snapshot.py --out /tmp/fwg-2344
    .venvs/lloyd/bin/python eval/run_memory_eval.py run --arms prefetch,prefetch_rel \\
        --fact-snapshot /tmp/fwg-2344/noop --out-dir /tmp/fwg-2344/runs
    # …and again with --fact-snapshot /tmp/fwg-2344/on

Pinned by `tests/test_fact_write_gate_snapshot.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
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
    through it the prefetch arm — reads it exactly as it reads production. The
    atomic rename is left out: this file has one writer, this process.
    """
    frontmatter = {"type": "facts", "entity": entity, "category": category,
                   "facts": entries, "last_updated": now_iso}
    body = (f"\n# {entity} - {category}\n\n**Entity:** {entity}\n"
            f"**Category:** {category}\n**Fact Count:** {len(entries)}\n")
    fact_file_path.parent.mkdir(parents=True, exist_ok=True)
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


def replay_tree(tree: Path, mode: str, *, now_iso: str | None = None) -> dict:
    """Replay `WRITE_SEQUENCE` into `tree` under gate mode `mode`; report it.

    Returns the per-write ledger: what the gate said, what the write TOOK, and
    the fact id the decision named — the reader of this artifact should not have
    to open the markdown to know whether the tree shows anything.
    """
    now_iso = now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
    writes: list[dict] = []
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
            "reindex": stats, "facts": facts_by_id}


def _differential(noop: dict, on: dict) -> dict:
    """The facts whose `expired_at` differs between the two trees, by id."""
    expired_on = {fid for fid, f in on["facts"].items() if f["expired_at"]}
    expired_noop = {fid for fid, f in noop["facts"].items() if f["expired_at"]}
    return {"expired_only_in_on": sorted(expired_on - expired_noop),
            "expired_only_in_noop": sorted(expired_noop - expired_on)}


def build(out: Path, *, modes: tuple[str, ...] = MODE_PAIR,
          now_iso: str | None = None) -> dict:
    """Build one snapshot dir per mode under `out` and report the differential."""
    out = refuse_production_root(Path(out))
    out.mkdir(parents=True, exist_ok=True)
    reports = {mode: replay_tree(out / mode, mode, now_iso=now_iso) for mode in modes}
    both = all(m in reports for m in MODE_PAIR)
    diff = _differential(reports["noop"], reports["on"]) if both \
        else {"expired_only_in_on": [], "expired_only_in_noop": []}
    manifest = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "out": str(out), "modes": list(modes),
        "sequence": [{"entity": e, "category": c, "fact": t}
                     for e, c, t in WRITE_SEQUENCE],
        "snapshots": {m: reports[m]["tree"] for m in reports},
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the noop/on fact-snapshot pair (#2344).")
    ap.add_argument("--out", required=True,
                    help="directory to build <out>/noop and <out>/on under; "
                         "refused if it resolves under production_data_root()")
    ap.add_argument("--modes", default=",".join(MODE_PAIR),
                    help=f"comma-separated gate modes, default {','.join(MODE_PAIR)}")
    args = ap.parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    bad = [m for m in modes if m not in gate.MODES]
    if bad:
        ap.error(f"unknown gate mode(s) {bad}; MODES is {list(gate.MODES)}")
    try:
        manifest = build(Path(args.out), modes=modes)
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
