#!/usr/bin/env python3
"""Re-point the 9 fact rows still carrying `source_doc: null` (#1813, entry 1 of #1743).

`facts_idx` attributes every fact to the document it came from, and the provenance
health line (`scripts/memory/kg_hygiene.py::provenance_coverage`) reports what share of
rows can answer that. Nine cannot: `SELECT COUNT(*) FROM facts_idx WHERE source_doc IS
NULL` returned 9 of 119,171 at 2026-09-29T09:18Z, and `provenance_coverage()` said
`source_doc_pct 99.99, both_pct 99.99`. Session-distill extraction wrote all nine with no
`source_doc`, and commit `83b6e0ab` closed that hole at write time
(`agent_mcp/facts.py:574`, `_MACHINE_DERIVED_PROVENANCE`) — a refusal stops new leaks but
repairs none of the existing rows, which is what this script is for. (The item cites
`83b6e60ab`, which resolves in neither repo; the sha is transposed.)

Why an edit and not a reindex: `reindex()` runs `DELETE FROM facts_idx`
(`app/kg_store.py:1282`), so the index is a rendering of the markdown and a repair that
only touched the database would be undone by the next rebuild. The change therefore lands
in the eight fact files under `VAULT_FACTS_ROOT`, and the store is told about exactly
those files through `store().facts_idx.update_file(path, root=…, register_entities=False)`
— the same call the live fact writer makes (`agent_mcp/facts.py:709`) — never a full-tree
reindex,
which would rebuild over 12,560 entity directories to fix nine rows.

The mapping is the one #1743 recorded, and it is the weak part of this script, so it says
what it is: seven facts written at `2026-09-28T00:54Z` are attributed to the 2026-09-27
chat, and the two `Lloyd Chrome Side Panel` facts to the 2026-09-23 chat. That split is
corroborated at topic level only — the 09-27 transcript has no literal "Mission Control
Dashboard" or "automod promotion gate" (it has 16 "Mission Control" and 137 "automod"),
and it is the 2-for-0 pattern on "Chrome Side Panel" in the 09-23 transcript that separates
the two groups. A file existing is not evidence a fact came out of it, so every planned
line prints the FACT'S OWN TEXT beside its new `source_doc`: read those nine pairs once
against the two transcripts before running `--apply`, and if one does not hold, change the
map rather than the fact.

Modes: `--dry-run` is the default and writes nothing — it prints the plan, which is the
eyeball step; `--apply` writes. Exit 0 only when the run can back its claim: every named
entry matched, both transcripts present, and — after an apply — the store itself reporting
zero rows with `source_doc IS NULL` among the files it rewrote. Exit 2 on any of: a named
transcript missing (checked before a single file is opened for writing, so the tree is
byte-identical afterwards), an `(entity, fact_id)` no file carries, zero entries matched
(a 0-row match is not a clean bill of health, it is a script pointed at the wrong tree), or
a post-check that found a row still NULL.

Modes are flags, not a default-to-write: this edits data under `~/lloyd-data` that no test
can undo, and a run whose mode is decided by which flag the operator remembered is not
auditable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.atomic_io import atomic_write_text                                  # noqa: E402
from agent_mcp._shared import _write_fact_frontmatter                        # noqa: E402
from app.kg_store import KGStore, parse_fact_file, store as live_store       # noqa: E402
from app.paths import VAULT_FACTS_ROOT                                       # noqa: E402

#: The two transcripts #1743 names as the originating sessions. Absolute, which is
#: already the prevailing shape for this column: 69 rows carry a `sessions/`-style value
#: and 22 begin with `/`, and the only other reader — the first-path-segment classifier in
#: `app/entity_kind.py:100` — tolerates them.
TRANSCRIPT_2026_09_27 = "/home/alansrobotlab/lloyd-data/sessions/20260927_144519_iv13dc.json"
TRANSCRIPT_2026_09_23 = "/home/alansrobotlab/lloyd-data/sessions/20260923_150805_ivf86b.json"

#: The nine rows to repair, as `(entity, fact_id, new source_doc)`. Keyed by id, not by
#: fact text: a text match could land on a near-duplicate and silently re-point the wrong
#: row, while an id that is not there cannot land on anything at all — it is reported as
#: unmatched and the run exits non-zero.
REPOINTS: tuple[tuple[str, str, str], ...] = (
    ("Mission Control Dashboard", "stat-001", TRANSCRIPT_2026_09_27),
    ("automod promotion gate", "beha-001", TRANSCRIPT_2026_09_27),
    ("automod promotion gate", "beha-002", TRANSCRIPT_2026_09_27),
    ("automod round resume", "beha-001", TRANSCRIPT_2026_09_27),
    ("lloyd web frontend test tier", "conv-001", TRANSCRIPT_2026_09_27),
    ("lloyd backlog", "stat-001", TRANSCRIPT_2026_09_27),
    ("Alan", "pref-089", TRANSCRIPT_2026_09_27),
    ("Lloyd Chrome Side Panel", "stat-001", TRANSCRIPT_2026_09_23),
    ("Lloyd Chrome Side Panel", "buil-001", TRANSCRIPT_2026_09_23),
)

EXIT_OK = 0
EXIT_BLOCKED = 2


def _entity_dir(root: Path, entity: str) -> Path | None:
    """The directory for `entity`, matched case-insensitively like the read path does.

    Entity names on disk differ from the map's spelling in case (`lloyd backlog` sits in
    `lloyd backlog/`, and a re-extraction could title it); a case-sensitive lookup would
    report a present fact as unmatched, which is the failure mode this script must not
    have — a missed row looks like a successful run over a tree that is missing it.
    """
    if (root / entity).is_dir():
        return root / entity
    lowered = {d.name.lower(): d for d in root.iterdir() if d.is_dir()} if root.is_dir() else {}
    return lowered.get(entity.lower())


def plan(facts_root: Path, repoints=None):
    """Read the tree and return `(changes, already, unmatched)`.

    `repoints` defaults to `REPOINTS` resolved at CALL time, not bound as a default
    argument: a default would freeze the map into the signature, and the one thing this
    script needs to survive is an operator editing the map after import (or a test
    substituting a smaller one) and having that choice take effect.

    `changes` is one entry per fact to repair: `(entity, fact_id, path, old, new, text)`.
    `already` is the same shape for entries whose `source_doc` is already what is asked —
    that is what makes a second run a report rather than a rewrite. `unmatched` names every
    `(entity, fact_id)` the map asked for that this tree does not have, with the reason a
    human needs to tell "already repaired" from "wrong tree".
    """
    repoints = REPOINTS if repoints is None else repoints
    changes: list[dict] = []
    already: list[dict] = []
    unmatched: list[dict] = []
    for entity, fact_id, new in repoints:
        entry = {"entity": entity, "fact_id": fact_id, "new": new}
        directory = _entity_dir(facts_root, entity) if facts_root.is_dir() else None
        if directory is None:
            unmatched.append({**entry, "why": f"no entity directory under {facts_root}"})
            continue
        hit = None
        for path in sorted(directory.glob("*.md")):
            fm, facts = parse_fact_file(path)
            for f in facts:
                if f.get("id") == fact_id:
                    hit = (path, f)
                    break
            if hit:
                break
        if hit is None:
            unmatched.append({**entry,
                              "why": f"no fact with id {fact_id!r} in {directory.name}/*.md"})
            continue
        path, fact = hit
        old = fact.get("source_doc")
        item = {**entry, "path": path, "old": old,
                "text": str(fact.get("fact") or "")}
        (already if old == new else changes).append(item)
    return changes, already, unmatched


def _body_after_frontmatter(content: str) -> str:
    """The markdown body, verbatim: everything after the closing `---` LINE.

    Not the splice at `agent_mcp/facts.py:407`, which takes `content[find("---", 3) + 3:]`
    and so keeps the newline that ENDS the closing delimiter line as the body's first
    character. The writer's own output ends `"---\n"`, so that splice emits one blank line
    the file did not have before — every live `fact_add` to a fact file whose body starts
    with a blank line inserts one, once (it converges, so nothing downstream ever noticed).
    Reproducing that here would put a byte change in eight fact files that this repair has
    no reason to make, so this splits on the delimiter line instead. The quirk belongs to
    the live writer and is recorded on #1813 rather than fixed by this round.
    """
    close = content.find("\n---\n", 3)
    return content[close + 5:] if close != -1 else ""


def _rewrite(path: Path, fact_id: str, new: str) -> None:
    """Set `source_doc` on one fact entry and write the file back the way `facts.py` does.

    The same writer and the same atomic write as `agent_mcp/facts.py:409`:
    `atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)`. Using the
    live writer is the point — a locally re-implemented YAML emit would leave this script
    green while the fact files it edits no longer matched what the read path writes, and
    the next `fact_add` would restyle the file as a side effect of an unrelated write. Only
    the body split differs, for the reason in `_body_after_frontmatter`: with it, the
    difference between the file before and after this write is the one `source_doc` line.
    """
    content = path.read_text(encoding="utf-8")
    frontmatter, _facts = parse_fact_file(path)
    for f in frontmatter.get("facts") or []:
        if f.get("id") == fact_id:
            f["source_doc"] = new
    atomic_write_text(path,
                      _write_fact_frontmatter(frontmatter) + _body_after_frontmatter(content))


def run(facts_root: Path, kg: KGStore, *, apply: bool = True, repoints=None,
        out=print) -> dict:
    """Repair the mapped rows over `facts_root` and reindex only the touched files.

    Returns the counts plus `ok`. Refuses to write anything until every named transcript
    exists: a row pointed at a file that is not there is a provenance claim worse than the
    NULL it replaced, because a NULL reads as "unknown" and a path reads as "verified".
    `repoints` is resolved at call time, for the reason in `plan`.
    """
    repoints = REPOINTS if repoints is None else repoints
    transcripts = sorted({new for _, _, new in repoints})
    missing = [t for t in transcripts if not Path(t).is_file()]
    for t in missing:
        out(f"BLOCKED: transcript does not exist: {t}")
    if missing:
        return {"ok": False, "reason": "missing_transcripts", "missing": missing,
                "changed": [], "already": [], "unmatched": [], "files": [], "rows": 0}

    changes, already, unmatched = plan(facts_root, repoints)
    for c in changes:
        out(f"CHANGE entity={c['entity']!r} fact_id={c['fact_id']} "
            f"old={c['old']!r} -> new={c['new']!r}")
        out(f"    fact: {c['text']}")
    for a in already:
        out(f"OK (already correct) entity={a['entity']!r} fact_id={a['fact_id']} "
            f"source_doc={a['new']!r}")
    for u in unmatched:
        out(f"UNMATCHED entity={u['entity']!r} fact_id={u['fact_id']}: {u['why']}")

    # One refusal rule, because a run that matched nothing IS the unmatched case: over an
    # empty or wrong directory all nine entries arrive here saying "no entity directory
    # under …", which is exactly what that mistake looks like. A second gate counting
    # zero-match runs would be a check over the same evidence, and would have to be kept in
    # step with this one; 0 matched can therefore never exit 0 having repaired nothing.
    if unmatched:
        return {"ok": False, "reason": "unmatched",
                "changed": changes, "already": already, "unmatched": unmatched,
                "files": [], "rows": 0}

    touched = sorted({c["path"] for c in changes})
    if not apply:
        out(f"DRY RUN: {len(changes)} to change, {len(already)} already correct, "
            f"nothing written, {len(touched)} file(s) would be reindexed")
        return {"ok": True, "reason": "dry_run", "changed": changes, "already": already,
                "unmatched": [], "files": touched, "rows": 0}

    for path in touched:
        here = [c for c in changes if c["path"] == path]
        # One reindex per file even when it holds two of the nine (automod promotion
        # gate-behavior.md does), and each write is atomic, so the store is never told
        # about a file mid-repair and a reader never sees one row repaired and its
        # neighbour not.
        for c in here:
            _rewrite(path, c["fact_id"], c["new"])
        # register_entities=False, matching the live fact writer: this repair attributes
        # facts inside files whose entity is already registered, and registering here would
        # let a provenance fix touch entities.json/aliases.json as a side effect.
        kg.facts_idx.update_file(path, root=facts_root, register_entities=False)
        out(f"REINDEXED {len(here)} fact(s) from {path}")

    # Re-plan from the bytes just written, BEFORE reading the table. The markdown is the
    # durable artifact — a rebuild reads it and `reindex()` runs `DELETE FROM facts_idx` — so
    # an entry that only LOOKS repaired in the table comes back NULL on the next full rebuild
    # and the repair silently un-happens. That is a file defect, so it gets the file-level
    # message rather than an opaque store miss, and it is the only way to see, inside a single
    # run, a writer whose output does not stick or an escaping form the reader parses
    # differently next time. Disk first also keeps the store check below reachable: a write
    # the index never picked up reads clean here and fails there.
    after_changes, after_already, after_unmatched = plan(facts_root, repoints)
    if after_changes or after_unmatched:
        for c in after_changes:
            out(f"UNSTABLE: {c['entity']}/{c['fact_id']} still reads {c['old']!r} on disk "
                f"after the write — a later rebuild would restore the NULL")
        for u in after_unmatched:
            out(f"UNSTABLE: {u['entity']}/{u['fact_id']} unreadable after the write: "
                f"{u['why']}")
        return {"ok": False, "reason": "unstable", "changed": changes, "already": already,
                "unmatched": [], "files": touched, "rows": 0}


    # The store's own answer, not the writer's: `for_entity` with `include_expired` so a
    # row that is expired rather than missing cannot report a repair that did not happen by
    # simply being filtered out of the default read.
    rows = {c["entity"]: kg.facts_idx.for_entity(c["entity"], include_expired=True)
            for c in changes}
    still = [c for c in changes
             if not any(r.get("fact_id") == c["fact_id"] and r.get("source_doc") == c["new"]
                        for r in rows[c["entity"]])]
    for c in still:
        out(f"BLOCKED after apply: {c['entity']}/{c['fact_id']} is repaired on disk but the "
            f"store still does not attribute it to {c['new']!r}")
    if still:
        return {"ok": False, "reason": "post_check", "changed": changes, "already": already,
                "unmatched": [], "files": touched, "rows": 0}

    out(f"APPLIED: {len(changes)} changed, {len(already)} already correct, "
        f"{len(touched)} file(s) reindexed; re-read clean, a second run would change 0")
    return {"ok": True, "reason": "applied", "changed": changes, "already": already,
            "unmatched": [], "files": touched, "rows": len(changes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--facts-root", default=None,
                    help="facts tree (default: app.paths.VAULT_FACTS_ROOT)")
    ap.add_argument("--db", default=None,
                    help="kg store to reindex (default: app.kg_store.store())")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="print the plan and write nothing (the default)")
    mode.add_argument("--apply", action="store_true",
                      help="write the repaired front matter and reindex the touched files")
    args = ap.parse_args(argv)

    root = Path(args.facts_root) if args.facts_root else Path(VAULT_FACTS_ROOT)
    if not root.is_dir():
        print(f"BLOCKED: no facts tree at {root}")
        return EXIT_BLOCKED
    kg = KGStore(args.db) if args.db else live_store()
    result = run(root, kg, apply=bool(args.apply), out=print)
    return EXIT_OK if result["ok"] else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
