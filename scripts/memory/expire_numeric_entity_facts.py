#!/usr/bin/env python3
"""Expire the facts filed under a numeric-shaped entity name (#1643).

The store holds, right now, entity rows whose whole name is a tracker id: on the
live store 2026-09-28 that is 112 rows — 87 bare digits (`1051`), 25 with a
leading `#` (`#1051`) — carrying 442 unexpired facts. They were minted by the
extraction loop, which was asked for a `task` and answered a number:
`facts/1051/1051-overview.md` reads "1051 is a design owner…". The name is not an
entity, so a query that wants backlog item #1051 cannot reach the facts through
it, and `agent_mcp/retrieval.py` can only de-score such a candidate
(`_NUMERIC_SEED_MAX_SCORE`, #1025) — de-score is the shipped mitigation and this
sweep does not touch it.

The ruling this implements (#1025's, restated by #1643): **expire, do not evict.**
Eviction would delete the row that makes the id resolvable, and 0 of the 112
numeric rows has a non-numeric alias twin (`SELECT canonical FROM aliases WHERE
surface_lc = ?`), so evicting would orphan all 442 live facts behind them instead
of hiding a scoring artefact. So: `entities` rows stay, `facts/<Name>/` dirs
stay, ids stay resolvable — only `facts_idx.expired_at` and the mirrored
`expired_at` in each fact file's front matter are set. Whether an expired digit
row should also lose its row and its directory, and whether it should hold a
prefetch slot at all, is reserved to a person on #1643 and is NOT done here.

`app/kg_store.py` cannot expire an entity: `entities` has no `expired_at` column
(`app/kg_store.py:89-97`) and the only `expire()` helper is edges-only
(`:714`). So expiry means the fact rows plus the markdown they were indexed
from, which is what this writes, through the same two primitives the `fact_*`
tools use — `locked_file` around a read-modify-write and `atomic_write_text` for
the replace — and then `facts_idx.reindex` for the touched files, so the index
and the tree cannot disagree afterwards.

Dry-run by default; `--apply` writes. The report is the point:

    numeric_entities=112 files_scanned=177 facts_live=442 facts_to_expire=442 \
    facts_expired=0 files_touched=0 facts_still_live_after=442 apply=false

`facts_to_expire` is what the store asks for; `facts_expired` is what this run
wrote, so a dry run reports 0 by definition and cannot be mistaken for a clean
store. Three stores print `facts_expired=0` and only one of them is fine: no
numeric row at all, numeric rows whose facts are already expired, and numeric
rows with live facts this run failed to touch — which is why the line carries
`numeric_entities`, `facts_live` and `facts_still_live_after` beside it.

    expire_numeric_entity_facts.py                      # read-only plan
    expire_numeric_entity_facts.py --apply              # the live store
    expire_numeric_entity_facts.py --apply --json       # for a job to parse

Skip-list: `USER.md` / `MEMORY.md` lines that merely look numeric are out of
scope by #1643's ruling — this file only ever opens `facts/<dir>/*.md` under the
facts root, and the class rule about those lines lives in tests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD = HERE.parent.parent
for _p in (str(LLOYD), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml

from app.atomic_io import atomic_write_text, locked_file
from app.paths import VAULT_FACTS_ROOT, VAULT_KG_DB

#: A whole entity name that is a tracker id: bare digits, or `#` + digits.
#: Deliberately its own copy rather than `entity_naming._TRACKER_ID_RE`: this
#: sweep must still expire what the write-side guard has since started refusing,
#: so the two read the same shape from two directions and each is pinned on its
#: own (`tests/test_fact_extractor.py`, `tests/test_numeric_entity_sweep.py`).
NUMERIC_ENTITY_RE = re.compile(r"#?\d+")

#: What lands in `expired_reason` beside each expired fact. Names the item, not
#: a verdict, because the facts are not wrong — `1051` did record something — the
#: name holding them is not an entity.
DEFAULT_REASON = ("#1643: entity name is a bare tracker id; the fact was filed "
                  "under a pointer rather than an entity")


def is_numeric_entity_name(name: str) -> bool:
    return bool(NUMERIC_ENTITY_RE.fullmatch((name or "").strip()))


# ── reading the store ────────────────────────────────────────────────────────

def numeric_entities(kg) -> list[str]:
    """Entity rows whose WHOLE name is `#?\\d+`, whatever their facts did.

    Read off `entities`, not off `facts_idx`, so a row with no live fact under it
    is still reported: that is the shape a previous run leaves behind, and the
    count is what tells this run whether it is looking at the same store.
    """
    return sorted(r["name"] for r in kg.conn.execute("SELECT name FROM entities")
                  if is_numeric_entity_name(r["name"] or ""))


def live_facts_by_entity(kg) -> dict[str, int]:
    """Unexpired indexed rows per numeric-shaped entity name.

    `expired_at IS NULL AND invalid_at IS NULL`, which is the store's own
    definition of live for a read (`FactsIndex.for_entity`), so a row a
    `fact_invalidate` already took out of service is not counted as work left.
    The pair matters: a `fact_resolve` marks the loser `invalid_at`, and expiring
    an already-invalid row would report a change that is not one.
    """
    out: dict[str, int] = {}
    for r in kg.conn.execute(
            "SELECT entity, COUNT(*) AS n FROM facts_idx "
            "WHERE expired_at IS NULL AND invalid_at IS NULL GROUP BY entity"):
        if is_numeric_entity_name(r["entity"] or ""):
            out[r["entity"]] = int(r["n"])
    return out


def candidate_files(kg, facts_root: Path, names: list[str]
                    ) -> tuple[list[Path], list[str]]:
    """`(files to open, stored paths that escape the root)` for the live rows.

    `file_path` is stored relative to the facts root, so a row whose path resolves
    outside it — a `..`, or an absolute path left by a rebuild run against another
    root — is not opened and is returned instead: this is the only component in
    the item that writes markdown, and index data must never point it above the
    tree it was given. Such a row stays live and is reported, so the run cannot
    read as a clean sweep while leaving it behind.
    """
    if not names:
        return [], []
    placeholders = ",".join("?" * len(names))
    files: list[Path] = []
    escaped: list[str] = []
    seen: set[Path] = set()
    for r in kg.conn.execute(
            f"SELECT DISTINCT file_path FROM facts_idx "
            f"WHERE expired_at IS NULL AND invalid_at IS NULL "
            f"AND entity IN ({placeholders})", tuple(names)):
        rel = (r["file_path"] or "").strip()
        if not rel:
            continue
        p = (facts_root / rel).resolve()
        try:
            p.relative_to(facts_root.resolve())
        except ValueError:
            escaped.append(rel)
            continue
        if p not in seen:
            seen.add(p)
            files.append(p)
    return sorted(files), sorted(set(escaped))


# ── the write side ───────────────────────────────────────────────────────────

def _effective_entity(fm: dict, fact, file_entity: str) -> str:
    """Whose fact this is, by the name the file itself states.

    `app/kg_store._rows_for_file` derives the index's entity the same way — the
    fact's own `entity` key, else the front matter's, else the directory name —
    so this is the same decision, read from the other side. The file is the
    source; the index is derived from it.
    """
    if isinstance(fact, dict):
        own = (fact.get("entity") or "").strip()
        if own:
            return own
    return (fm.get("entity") or file_entity or "").strip()


def expire_file(path: Path, facts_root: Path, reason: str, ended: str,
                apply: bool) -> tuple[int, int]:
    """Return `(matched, left_alone)` for one fact file.

    A file is opened only because some live row in it is attributed to a numeric
    name, so the second number is the guard: a fact in that same file that the
    file attributes to a real entity is left alone and counted, which is how
    "touches nothing else" is checked rather than assumed. `matched` counts both
    directions — what a dry run would do and what an applied run did — and the
    caller decides which of the two it reports.
    """
    with locked_file(path):
        text = path.read_text(encoding="utf-8", errors="replace")
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return (0, 0)
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return (0, 0)
    if not isinstance(fm, dict):
        return (0, 0)
    body = parts[2]
    facts = fm.get("facts")
    if not isinstance(facts, list):
        return (0, 0)

    matched = left = 0
    changed = False
    for fact in facts:
        if is_numeric_entity_name(_effective_entity(fm, fact, path.parent.name)):
            if not isinstance(fact, dict) or fact.get("expired_at") or fact.get("invalid_at"):
                # A bare string under a numeric dir has no `expired_at` field to
                # set, so expiring it would mean rewriting the entry: counted as
                # nothing, and it stays live and shows in `facts_still_live_after`.
                continue
            if apply:
                fact["expired_at"] = ended
                fact["expired_reason"] = reason
                changed = True
            matched += 1
        else:
            left += 1

    if changed:
        atomic_write_text(path, f"---\n{yaml.dump(fm, sort_keys=False)}---\n{body}")
    return (matched, left)


def sweep(kg, facts_root: Path, *, apply: bool = False,
          reason: str = DEFAULT_REASON, ended: str | None = None) -> dict:
    """Expire every live fact under a numeric-shaped entity name. Reports counts.

    `facts_to_expire` is the plan read off the index; `facts_expired` is what this
    run wrote (so 0 on every dry run, by construction); `facts_still_live_after`
    is the store re-measured afterwards. Those three together are what makes a
    clean store, a completed sweep and a silent no-op three different readings.
    """
    ended = ended or dt.datetime.now(dt.timezone.utc).isoformat()
    names = numeric_entities(kg)
    before = live_facts_by_entity(kg)
    files, escaped = candidate_files(kg, facts_root, names)

    planned = expired = untouched = missing = 0
    touched: list[Path] = []
    missing_files: list[str] = []
    for p in files:
        if not p.exists():
            # Indexed but gone from disk: nothing to expire in the tree, and the
            # row stays live. Reported, not swallowed — this is the state a
            # partially-applied rebuild leaves.
            missing += 1
            missing_files.append(str(p.relative_to(facts_root)))
            continue
        e, u = expire_file(p, facts_root, reason, ended, apply)
        planned += e
        untouched += u
        if apply:
            expired += e
            if e:
                touched.append(p)

    if touched:
        # The index is derived from the tree, so the tree is written first and the
        # rows are re-read from it. `register_entities=False`: this sweep expires,
        # it never mints, and registering a directory's name is exactly the entry
        # point that now refuses a bare number.
        kg.facts_idx.reindex(touched, root=facts_root, register_entities=False)

    after = live_facts_by_entity(kg)
    return {
        "apply": apply,
        "numeric_entities": len(names),
        "files_scanned": len(files),
        "facts_live_before": sum(before.values()),
        "facts_to_expire": planned,
        "facts_expired": expired,
        "facts_left_alone_in_scanned_files": untouched,
        "files_touched": len(touched),
        "indexed_files_missing_from_disk": missing,
        "stored_paths_escaping_the_facts_root": len(escaped),
        "escaped_paths": escaped,
        "facts_still_live_after": sum(after.values()),
        "entities_still_live_after": len(after),
        "facts_live_by_entity_before": before,
        "missing_files": missing_files,
        "expired_at": ended,
        "reason": reason,
    }


def print_report(rep: dict) -> None:
    print(f"numeric_entities={rep['numeric_entities']} "
          f"files_scanned={rep['files_scanned']} "
          f"facts_live={rep['facts_live_before']} "
          f"facts_to_expire={rep['facts_to_expire']} "
          f"facts_expired={rep['facts_expired']} "
          f"files_touched={rep['files_touched']} "
          f"facts_still_live_after={rep['facts_still_live_after']} "
          f"apply={'true' if rep['apply'] else 'false'}")
    print(f"facts_left_alone_in_scanned_files={rep['facts_left_alone_in_scanned_files']} "
          f"indexed_files_missing_from_disk={rep['indexed_files_missing_from_disk']} "
          f"stored_paths_escaping_the_facts_root="
          f"{rep['stored_paths_escaping_the_facts_root']} "
          f"entities_still_live_after={rep['entities_still_live_after']}")
    if not rep["numeric_entities"]:
        print("no entity row's whole name is `#?\\d+` — nothing to expire. "
              "This is a measurement of the store, not a success.")
    elif not rep["facts_live_before"]:
        print(f"{rep['numeric_entities']} numeric entity rows hold no live fact — "
              f"already expired, or never extracted into the index")
    if rep["apply"] and rep["facts_live_before"] and not rep["facts_expired"]:
        print(f"{rep['facts_live_before']} live facts sit under numeric names and "
              f"none expired — this run touched nothing")
    if not rep["apply"] and rep["facts_to_expire"]:
        print(f"dry run: {rep['facts_to_expire']} facts would expire. Pass --apply.")
    for rel in rep["missing_files"]:
        print(f"  indexed but absent from disk, left live: {rel}")
    for rel in rep["escaped_paths"]:
        print(f"  stored path escapes the facts root, NOT opened, left live: {rel}")
    if rep["apply"] and rep["files_touched"]:
        print(f"entities rows and facts/<Name>/ dirs are deliberately untouched "
              f"(#1025's no-evict ruling); expired_at={rep['expired_at']!r}")


def writes_disabled_by_rebuild() -> bool:
    """True while a knowledge-graph rebuild holds writes off the tree.

    `kg_rebuild.py:145` sets `knowledge_graph.write_enabled: false` and extracts
    into a parallel tree that can be renamed `facts-quarantine-<ts>` at any moment;
    `agent_mcp/facts.py::_writes_enabled` refuses fact writes in that window, and a
    sweep that rewrites `facts/**/*.md` is the same write. Read, and failed open,
    the same way: a missing config is not a rebuild, and the dry run is the default,
    so this can only ever block an explicit `--apply`.
    """
    from app.config import CONFIG
    try:
        return not bool(CONFIG.get("knowledge_graph", {}).get("write_enabled", True))
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="write expired_at into the fact files and reindex. "
                         "Without it this only reports.")
    ap.add_argument("--facts-root", default=None,
                    help=f"fact tree to sweep (default {VAULT_FACTS_ROOT})")
    ap.add_argument("--kg-db", default=None,
                    help=f"store to reindex (default {VAULT_KG_DB})")
    ap.add_argument("--reason", default=DEFAULT_REASON)
    ap.add_argument("--ended", default=None,
                    help="ISO timestamp for expired_at (default: now, UTC)")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    if args.apply and writes_disabled_by_rebuild():
        print("writes are disabled (knowledge_graph.write_enabled = false), which "
              "kg_rebuild.py sets while a rebuild runs. Re-run --apply after it "
              "finishes; the dry run is always safe.")
        return 2

    from app import kg_store
    if args.kg_db and not Path(args.kg_db).exists():
        # `configure()` is the provisioning route: it CREATES an absent database.
        # Pointed at a typo it would hand this sweep an empty store, whose report
        # is `numeric_entities=0 facts_live=0` — the exact reading of a clean
        # store, from a store that was never measured. `store()` refuses the same
        # mistake on the default path; `--kg-db` gets the same refusal.
        print(f"no knowledge-graph database at {args.kg_db}; refusing to report a "
              f"count from a store that is not there")
        return 2

    kg = kg_store.configure(Path(args.kg_db)) if args.kg_db else kg_store.store()
    rep = sweep(kg, Path(args.facts_root or VAULT_FACTS_ROOT), apply=args.apply,
                reason=args.reason, ended=args.ended)
    if args.as_json:
        print(json.dumps(rep, default=str))
    else:
        print_report(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
