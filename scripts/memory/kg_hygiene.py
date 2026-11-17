"""Knowledge-graph hygiene metrics. READ-ONLY over the facts root.

Four things the 2026-09-03 audit had to compute by hand, now measured on
demand and by kg_health.py:

  contamination   entity directories holding facts whose own `entity:` tag names
                  a DIFFERENT entity — the residue of a wrong merge. Every one of
                  the 63 found on 2026-09-03 came from the sweep's suffix tier
                  fusing distinct things (`Intel Pipeline System` into `Intel`).
  untagged        fact records whose `entity:` tag says NOTHING — the key absent,
                  or a value that strips to empty. Measured 2026-10-09: 3,531 of
                  142,819 records (2.47%) across 531 entity dirs — 2,507 blank
                  strings, 1,024 with no key at all. Empty is a THIRD state,
                  neither "this entity" nor "another entity", so it fell into
                  neither side of the check above and the contamination rail
                  printed 0 while 2.47% of the corpus could not be attributed to
                  anything; `general` is where it collects, 365 untagged records
                  in the sixth-largest fact dir (#2474).
  near_duplicates directories that collapse to the same name after
                  normalisation, by tier — the sweep's input.
  regrowth        directories created since a STORED BASELINE of the entity-dir
                  name set, and how many of them are near-dups of a name that
                  already existed at that baseline — the rate at which
                  extraction re-creates duplicates, i.e. the number
                  extraction-time linking is meant to drive down.

The only file this module writes is that baseline (`write_baseline`, under the
data root, never under the facts root). It exists because dating dirs was not
measurable: every rebuilt `created_at` sits behind the whole tree, so a window
could not tell new from old (#1535, in `regrowth` below).

Usage:
  python scripts/memory/kg_hygiene.py                 # human summary
  python scripts/memory/kg_hygiene.py --json          # raw JSON
  python scripts/memory/kg_hygiene.py --write-baseline  # store today's dir set as the reference
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import PIPELINE_DIR, VAULT_FACTS_ROOT  # noqa: E402
from app.kg_store import StoreUnavailable, store  # noqa: E402

_SWEEP_PATH = Path(__file__).resolve().parent / "entity-resolution-sweep.py"
_sweep_mod = None


def sweep():
    """The sweep script owns the name normalisers; load it once by path."""
    global _sweep_mod
    if _sweep_mod is None:
        spec = importlib.util.spec_from_file_location("entity_resolution_sweep", _SWEEP_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["entity_resolution_sweep"] = mod
        spec.loader.exec_module(mod)
        _sweep_mod = mod
    return _sweep_mod


def iter_entity_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))


def parse_frontmatter(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}
    return fm if isinstance(fm, dict) else {}


def _same_entity(a: str, b: str) -> bool:
    s = sweep()
    return s.normalize_punct(a) == s.normalize_punct(b)


# ── contamination, and the records that carry no entity at all ───────────────

#: Untagged records as a percentage of the records this walk parsed, above which
#: an unattributable corpus is an alarm rather than a line in a table.
#:
#: A share, not an absolute count: an absolute floor loses strictness on its own
#: every time extraction adds records, which is precisely the growth being
#: watched. 0.5% of the 2026-10-09 corpus is ~714 records, against the 3,531
#: measured that day, so the rail goes red today and goes quiet if the historical
#: records are re-tagged — which is the ruling #2474 leaves open, and this floor
#: is one line to change if that ruling re-sets it instead.
UNTAGGED_ALARM_SHARE = 0.5


def _tag_of(x: Any) -> str | None:
    """What a fact record says about who it is about: '' for untagged, None for
    a record that is not a mapping at all.

    `x.get("entity")` — the test this section shipped with — is falsy for BOTH
    "the key is missing" and "the value is empty", and truthy-but-whitespace
    slipped through it as a name. Each is checked separately here so the
    distinction survives: a missing key is `extraction never wrote the field`, an
    empty value is `it wrote the field blank`, and both are untagged.
    """
    if not isinstance(x, dict):
        return None
    if "entity" not in x:
        return ""
    v = x["entity"]
    return "" if v is None else str(v).strip()


def scan_entity_dir(d: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], int]:
    """(foreign, untagged, records) for one entity directory, ONE parse per file.

    `foreign` is `{foreign_entity: {"facts": n, "files": [...]}}`. `untagged` is
    `{"facts": n, "files": [...]}` for records with nothing in their `entity:`
    tag. `records` is every fact record parsed here, the denominator for
    untagged's share.

    They are measured in the same walk deliberately: this walk parses 37,442
    frontmatters holding 142,819 fact records (measured 2026-10-09), and
    `compute_hygiene` already refuses to walk the tree twice for that reason. But
    sharing a walk does not mean sharing a tally — see `foreign_facts_in_dir`,
    which hands out only the first of these.
    """
    foreign: dict[str, dict[str, Any]] = {}
    u_facts = 0
    u_files: list[str] = []
    records = 0
    for f in d.glob("*.md"):
        fm = parse_frontmatter(f)
        if not fm:
            continue
        names: Counter[str] = Counter()
        top = str(fm.get("entity") or "").strip()
        facts = fm.get("facts") or []
        file_untagged = 0
        if isinstance(facts, list) and facts:
            for x in facts:
                records += 1
                tag = _tag_of(x)
                if tag is None or not tag:
                    file_untagged += 1
                    continue
                names[tag] += 1
        elif top:
            names[top] += 1          # overview / factless file: the file-level tag
        for name, n in names.items():
            # Only a real tag can reach here, and only a DIFFERENT one is foreign.
            if _same_entity(name, d.name):
                continue
            slot = foreign.setdefault(name, {"facts": 0, "files": []})
            slot["facts"] += n
            if f.name not in slot["files"]:
                slot["files"].append(f.name)
        if file_untagged:
            u_facts += file_untagged
            u_files.append(f.name)
    return foreign, {"facts": u_facts, "files": u_files}, records


def foreign_facts_in_dir(d: Path) -> dict[str, dict[str, Any]]:
    """{foreign_entity: {"facts": n, "files": [...]}} for one entity directory.

    FOREIGN ONLY, and that exclusion is load-bearing (#2474). Both consumers of
    this read every key as the residue of a wrong merge — `kg_rebuild` gates the
    rebuild on `contamination()["dirs"]`, and the knowledge report's alarm says
    "a merge went wrong" — so a record with a blank tag must never appear in it:
    blank is not "another entity". Untagged records are counted, and alarmed on,
    through `scan()["untagged"]` instead.
    """
    return scan_entity_dir(d)[0]


def scan(root: Path = VAULT_FACTS_ROOT) -> dict[str, Any]:
    """The whole facts tree in one walk: foreign detail plus the untagged tally.

    The untagged share is computed over `records`, the count THIS walk parsed,
    never over a second measurement like `facts_idx.count()`. Two walks over the
    same corpus disagreed by 5 directories on 2026-09-25 (#1535), and a
    percentage whose numerator and denominator come from different passes cannot
    be checked by anyone reading the row.
    """
    s = sweep()
    items = []
    by_tier: Counter[str] = Counter()
    total_facts = 0
    u = {"facts": 0, "dirs": 0, "files": 0}
    records = 0
    for d in iter_entity_dirs(root):
        foreign, u_dir, n = scan_entity_dir(d)
        records += n
        if u_dir["facts"]:
            u["facts"] += u_dir["facts"]
            u["dirs"] += 1
            u["files"] += len(u_dir["files"])
        if not foreign:
            continue
        for name, slot in foreign.items():
            by_tier[s.classify_pair(name, d.name)[0]] += 1
            total_facts += slot["facts"]
        items.append({"dir": d.name, "foreign": foreign})
    share = round(100.0 * u["facts"] / records, 2) if records else 0.0
    untagged = {**u, "records": records, "share_pct": share,
                "floor_pct": UNTAGGED_ALARM_SHARE,
                "over_floor": share > UNTAGGED_ALARM_SHARE}
    return {"dirs": len(items), "foreign_facts": total_facts,
            "by_tier": dict(by_tier), "items": items, "untagged": untagged}


def contamination(root: Path = VAULT_FACTS_ROOT) -> dict[str, Any]:
    """The foreign half of `scan`, with exactly the keys it has always had.

    `untagged` is not a fourth contamination number and stays out of this dict:
    `kg_rebuild` gates on `dirs`, `kg_health` prints it, and the knowledge report
    alarms on it, all reading every one as a wrong merge. Untagged facts have
    their own line and their own alarm condition (#2474).
    """
    return {k: v for k, v in scan(root).items() if k != "untagged"}


# ── near-duplicates and regrowth ──────────────────────────────────────────────

def _clusters(root: Path) -> dict[str, list[Path]]:
    s = sweep()
    by: dict[str, list[Path]] = defaultdict(list)
    for d in iter_entity_dirs(root):
        by[s.normalize_full(d.name)].append(d)
    return {k: v for k, v in by.items() if len(v) > 1}


def near_duplicates(root: Path = VAULT_FACTS_ROOT) -> dict[str, Any]:
    s = sweep()
    cl = _clusters(root)
    tiers = Counter(s.cluster_tier([d.name for d in v]) for v in cl.values())
    return {"clusters": len(cl), "dirs": sum(len(v) for v in cl.values()),
            "by_tier": dict(tiers),
            "samples": [[d.name for d in v] for v in list(cl.values())[:8]]}


# ── the baseline regrowth diffs against (#1535) ───────────────────────────────

#: The stored reference for `regrowth()`: next to the kg_health snapshots it
#: feeds, under the data root. Not the repo — `data/**` is a human-only path —
#: and not the facts root, which this module never writes.
BASELINE_PATH = PIPELINE_DIR / "metrics" / "kg-entity-dirs-baseline.json"

BASELINE_SCHEMA = 1


def write_baseline(root: Path = VAULT_FACTS_ROOT,
                   baseline_path: Path | str | None = None) -> dict[str, Any]:
    """Store the current set of entity-directory NAMES as regrowth's reference.

    The name set, deliberately, and not their timestamps. The 2026-09-23 KG
    rebuild re-derived every fact's `created_at`, so nothing in the tree dated
    older than three days and a 7-day window necessarily covered all of it:
    `new_dirs` read 11,959 of 11,959 and 12,027 of 12,027 on the snapshots
    taken after it (#1535). A rebuild can re-date a directory; it cannot make a
    name that was absent from a stored set have been present in it.

    Atomic replace, because the reader is a health run in the same minute: a
    truncated list here would make every directory look new.
    """
    p = Path(baseline_path) if baseline_path else BASELINE_PATH
    rec = {"schema": BASELINE_SCHEMA,
           "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "facts_root": str(Path(root).resolve()),
           "dirs": sorted(d.name for d in iter_entity_dirs(root))}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(p)
    return {"path": str(p), "dirs": len(rec["dirs"]), "captured_at": rec["captured_at"]}


def _read_baseline(root: Path, path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(baseline, why-not). Never guesses: an unusable reference is reported as
    one by `regrowth`, which is the whole point of having a reference.

    A baseline taken over a DIFFERENT tree is rejected rather than diffed
    against — the report runs over whatever `--facts-dir` names, and subtracting
    some other tree's directory set would print a confident wrong number, which
    is the defect this section was rewritten to stop printing."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"no baseline file at {path} (run `kg_health.py -o` or `kg_hygiene.py --write-baseline`)"
    except (OSError, ValueError) as e:
        return None, f"baseline at {path} is unreadable: {type(e).__name__}: {e}"
    dirs = raw.get("dirs") if isinstance(raw, dict) else None
    if not isinstance(dirs, list) or not dirs:
        return None, f"baseline at {path} names no directories"
    taken_over = str(raw.get("facts_root") or "")
    if taken_over and Path(taken_over).resolve() != Path(root).resolve():
        return None, f"baseline at {path} was taken over {taken_over}, not {Path(root).resolve()}"
    return ({"dirs": [str(x) for x in dirs],
             "captured_at": raw.get("captured_at"),
             "schema": raw.get("schema")}, None)


def regrowth(root: Path = VAULT_FACTS_ROOT, days: int = 7,
             baseline_path: Path | str | None = None) -> dict[str, Any]:
    """Entity directories created since the stored baseline of that name set.

    `new_dirs` is a set difference against the baseline `write_baseline`
    stores, never a count of directories whose `created_at`/mtime falls inside a
    window. A window over mutable timestamps is unfalsifiable: after the
    2026-09-23 rebuild every stamp sat behind the whole tree, so all 12,027
    directories were "born in the last 7 days" and `new_dirs` equalled
    `entities.count` on every snapshot — no amount of contamination regrowth
    could have made it read differently, and a reader took it as "the store grew
    by 12,027 this week" (#1535). With no usable baseline this section reports
    `new_dirs: None` and `no_baseline_reason` rather than the size of the store.

    The number carries its own denominator: `dirs_at_baseline` over
    `baseline_at`, both from the same `iter_entity_dirs` walk that produced the
    live set, so it can never disagree with `entities.count` the way two
    separate walks did (11,812 against 11,807 on 2026-09-25). `days` stays in
    the section for continuity with every snapshot already on disk; it bounds
    nothing now, which is why no printed line quotes it.

    `near_dup_new` counts the baseline-new directories whose normalised name
    equals one that was ALREADY there at the baseline — extraction coining a
    twin next to an existing entity. Two twins that both postdate the baseline
    are not regrowth: neither one is the older. `samples` names those twins,
    which is what the report renders as the regrown pairs.
    """
    s = sweep()
    path = Path(baseline_path) if baseline_path else BASELINE_PATH
    base, why = _read_baseline(root, path)
    live = iter_entity_dirs(root)
    if base is None:
        return {"days": days, "new_dirs": None, "near_dup_new": None,
                "by_tier": {}, "samples": [], "skipped_vanished": 0,
                "dirs_at_baseline": None, "baseline_at": None,
                "baseline_path": str(path), "no_baseline_reason": why}
    at_baseline = set(base["dirs"])
    older_by: dict[str, list[str]] = defaultdict(list)
    for name in at_baseline:
        older_by[s.normalize_full(name)].append(name)

    new: list[str] = []
    vanished = 0
    for d in live:
        if d.name in at_baseline:
            continue
        # Re-check the dirs this is about to call new: the entity sweep renames
        # directories while the pass runs, and a monitor that dies on its
        # subject being edited reports nothing at all (#1404).
        try:
            still_there = d.is_dir()
        except OSError:
            still_there = False
        if not still_there:
            vanished += 1
            continue
        new.append(d.name)

    dups: list[str] = []
    tiers: Counter[str] = Counter()
    for n in new:
        older = sorted(o for o in older_by.get(s.normalize_full(n), []) if o != n)
        if older:
            dups.append(n)
            tiers[s.classify_pair(n, older[0])[0]] += 1

    return {"days": days, "new_dirs": len(new), "near_dup_new": len(dups),
            "by_tier": dict(tiers), "samples": dups[:8],
            # A dir renamed mid-pass is skipped, not fatal; counted so a mass
            # skip reads as one and not as a quietly smaller denominator.
            "skipped_vanished": vanished,
            "dirs_at_baseline": len(at_baseline), "baseline_at": base["captured_at"],
            "baseline_path": str(path), "no_baseline_reason": None}


def describe(r: dict[str, Any]) -> str:
    """The counts and the reference they are a delta against, in one phrase.

    Every human surface that prints regrowth — kg_health's summary, the
    knowledge report's Hygiene row, this module's own summary — goes through
    here, because what #1535 actually leaked was a bare `of 12027 new dirs` on
    two of them with nothing beside it to say what that 12,027 was a count of.
    A snapshot written before the baseline existed renders as well, and says
    its number has no recorded reference instead of presenting it as a delta.
    """
    new = r.get("new_dirs")
    if new is None:
        return f"not measured: {r.get('no_baseline_reason') or 'no baseline was named'}"
    near = r.get("near_dup_new")
    at, dirs_at = r.get("baseline_at"), r.get("dirs_at_baseline")
    if at and isinstance(dirs_at, int):
        ref = f"baseline {at} over {dirs_at:,} dirs"
    else:
        ref = "no baseline recorded — this snapshot predates the entity-dir baseline (#1535)"
    return f"{near if near is not None else '?'} of {new:,} new dirs ({ref})"


def snapshot(root: Path = VAULT_FACTS_ROOT, days: int = 7,
             baseline_path: Path | str | None = None) -> dict[str, Any]:
    s = scan(root)
    out = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "facts_root": str(root),
        "contamination": {k: v for k, v in s.items() if k not in ("items", "untagged")},
        # Its own section, its own alarm: an untagged record is not a contaminated
        # one, and a snapshot that folded the two would put both numbers back
        # under the reader that treats every key as a wrong merge (#2474).
        "untagged": s["untagged"],
        "near_duplicates": near_duplicates(root),
        "regrowth": regrowth(root, days, baseline_path=baseline_path),
    }
    out["provenance"] = provenance_coverage()
    return out


def provenance_coverage(root: Path = VAULT_FACTS_ROOT) -> dict[str, Any]:
    """Share of indexed facts that say where they came from and when.

    99.7% of the 205k facts in the pre-rebuild tree had neither `created_at`
    nor `source_doc`, so nothing could be dated, attributed or selectively
    reverted. The rebuild gate requires 100%.
    """
    try:
        st = store()
        total = st.facts_idx.count()
        if not total:
            return {"facts": 0, "created_at_pct": 0.0, "source_doc_pct": 0.0, "both_pct": 0.0}
        rows = st._query(
            "SELECT SUM(created_at IS NOT NULL) AS c, SUM(source_doc IS NOT NULL) AS s, "
            "SUM(created_at IS NOT NULL AND source_doc IS NOT NULL) AS b FROM facts_idx")[0]
        pct = lambda n: round(100.0 * (n or 0) / total, 2)  # noqa: E731
        return {"facts": total, "created_at_pct": pct(rows["c"]),
                "source_doc_pct": pct(rows["s"]), "both_pct": pct(rows["b"])}
    except StoreUnavailable as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ── the human-only merge queue (#3407) ───────────────────────────────────────

#: Both queues that are left for a person live in the sweep's output dir under
#: the pipeline root, beside the graph baseline the health report already reads.
MERGE_QUEUE_PLANS_GLOB = "entity-merges-*.jsonl"
MERGE_QUEUE_PLANS_PREFIX = "entity-merges-"
#: A run writes its dry plan and then a second plan holding only what it
#: applied; the applied one carries this marker in its name.
MERGE_QUEUE_APPLY_MARKER = "-apply-"
MERGE_QUEUE_PROPOSALS_FILENAME = "semantic-proposals-latest.jsonl"
#: The tier `--apply` never touches on schedule, so every approval in it is by
#: construction waiting on a person.
SUFFIX_TIER = "SUFFIX_SAFE"


def _plan_date(name: str) -> dt.date | None:
    """The run date a plan filename carries, or None when it carries none.

    The date exists ONLY in the filename. The semantic gate's own cache
    (`semantic-verdicts.jsonl`) has keys `a,b,decision,def_a,def_b,judges,key`
    and no timestamp field at all, so an approval carries no date where it was
    written and the age of a waiting queue cannot be derived from the verdict
    that granted it (#3407). A pointer like `entity-merges-latest.jsonl` is not
    a dated plan and returns None: it can be read as the newest document while
    the run that writes it is still going.
    """
    if not name.startswith(MERGE_QUEUE_PLANS_PREFIX) or not name.endswith(".jsonl"):
        return None
    if MERGE_QUEUE_APPLY_MARKER in name:
        return None
    try:
        return dt.date.fromisoformat(name[len(MERGE_QUEUE_PLANS_PREFIX):][:10])
    except ValueError:
        return None


def _jsonl_rows(path: Path) -> list[dict]:
    """Every object row of a JSONL file; an unreadable file is an empty list.

    A torn line is skipped rather than raised: this feeds one report row, and
    the queue's own shape must never be able to take the report down (#3407
    clause 4).
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _parse_seen(value) -> dt.date | None:
    """The UTC date of an ISO timestamp, or None when it has no readable one."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def merge_queue_report(memory_graph_dir: Path,
                       *,
                       today: dt.date | None = None) -> dict[str, Any]:
    """The two proposal queues that are left for a person — counted, and dated.

    Two members of the entity-resolution fleet write *proposals* and apply
    nothing: #48's suffix tier (`SUFFIX_SAFE` is never passed to `--apply` on
    schedule, per `architecture/autonomy-jobs.md` §Canonicalize) and #67, which
    only ever proposes. That boundary is deliberate. What was not deliberate is
    that the waiting is unwatched: until now the number existed only inside a
    per-run record under `autonomy-runs/<id>/`, a retention-bounded store nobody
    opens unless a human goes looking, so there was no surface where "N clusters
    have waited M days" was ever printed, and with no age bound a cluster the
    gate approved in September was re-presented as fresh in November (#3407).

    Two figures, each carrying the age of its oldest entry:

    `suffix_clusters` — clusters approved but never applied: plan rows of tier
        `SUFFIX_SAFE` whose `status` is `SAFE`. The status filter is
        load-bearing. On the 2026-10-26 tree the tier ALONE matches 63 clusters
        because the 40 the gate refused share the tier and differ only in
        `status` (`SAFE` vs `AMBIGUOUS`), so an unfiltered tier walk counts the
        refused set as waiting — and dates it from `python`, first seen
        2026-09-22, which is a REFUSED cluster.
        The queue is read across the plan files of the NEWEST run DATE, never
        from "the newest file": a run also writes a plan of what it applied, and
        newest-of-the-day is that file, whose suffix-approved count is 0
        (measured 2026-10-31: `entity-merges-2026-10-30-20261030T150213Z.jsonl`,
        109 rows, 0 approved suffix clusters, written a minute after the dry
        plan that holds the 23). The applied plan is a subset of its run's dry
        plan, so a union across one date cannot invent a cluster the dry plan
        lacked.
        Age: the earliest DATED plan in which each waiting cluster appears
        approved; the figure carries the oldest of those.
    `merge_proposals` — rows of `semantic-proposals-latest.jsonl` whose `action`
        is `merge`. `alias_only` rows are excluded: they ask for an alias, not a
        merge, and reach the same human at a different cost.
        Age: the earliest `first_seen_at` among the merge rows — #67 stamps its
        own proposals, so this queue needs no filename archaeology.

    An absent or empty store yields a zero with an explicit `*_no_data` reason,
    never a missing figure and never an exception: a drained queue and an
    unreadable queue are different claims, and a silent row is precisely how a
    queue of 23 came to read as a queue of none.

    Nothing here is an alarm. A non-empty queue is the designed steady state of
    a human-only merge step; the defect was the un-reported growth. The caller
    must keep these numbers out of the report's exit-code channel, which is
    reserved for counts a job can repair itself.
    """
    graph = Path(memory_graph_dir)
    today = today or dt.datetime.now(dt.timezone.utc).date()

    plans: dict[dt.date, list[Path]] = {}
    for path in graph.glob(MERGE_QUEUE_PLANS_GLOB):
        day = _plan_date(path.name)
        if day is not None:
            plans.setdefault(day, []).append(path)
    newest = max(plans) if plans else None

    first_approved: dict[str, dt.date] = {}
    waiting: set[str] = set()
    refused: set[str] = set()
    any_plan_row = False
    for day in sorted(plans):                        # oldest first, so the first
        for path in sorted(plans[day]):              # date seen is the earliest
            rows = _jsonl_rows(path)
            any_plan_row = any_plan_row or bool(rows)
            for row in rows:
                if row.get("tier") != SUFFIX_TIER:
                    continue
                key = row.get("norm_key") or row.get("canonical") or row.get("old")
                if not key:
                    continue
                if row.get("status") == "SAFE":
                    first_approved.setdefault(key, day)
                    if day == newest:
                        waiting.add(key)
                elif day == newest:
                    refused.add(key)

    approved_dates = [first_approved[k] for k in waiting if k in first_approved]
    oldest_approved = min(approved_dates) if approved_dates else None
    if not plans:
        suffix_no_data = f"no dated {MERGE_QUEUE_PLANS_PREFIX}*.jsonl plan file"
    elif not any_plan_row:
        suffix_no_data = "no readable row in any plan file"
    else:
        suffix_no_data = None

    proposals_path = graph / MERGE_QUEUE_PROPOSALS_FILENAME
    proposal_rows = _jsonl_rows(proposals_path)
    merge_rows = [r for r in proposal_rows if r.get("action") == "merge"]
    seen = [d for d in (_parse_seen(r.get("first_seen_at")) for r in merge_rows)
            if d is not None]
    oldest_seen = min(seen) if seen else None
    if not proposals_path.exists():
        proposals_no_data = f"{MERGE_QUEUE_PROPOSALS_FILENAME} is absent"
    elif not proposal_rows:
        proposals_no_data = f"{MERGE_QUEUE_PROPOSALS_FILENAME} holds no readable row"
    else:
        proposals_no_data = None

    def _age(day):
        return None if day is None else max(0, (today - day).days)

    return {
        "suffix_clusters": len(waiting),
        "suffix_oldest_date": oldest_approved.isoformat() if oldest_approved else None,
        "suffix_age_days": _age(oldest_approved),
        # The refused set is counted so the printed figure can say what it
        # EXCLUDED: the most likely misreading of this number is the tier total.
        "suffix_refused": len(refused - waiting),
        "suffix_tier_total": len(waiting | refused),
        "suffix_no_data": suffix_no_data,
        "merge_proposals": len(merge_rows),
        "alias_only": sum(1 for r in proposal_rows if r.get("action") == "alias_only"),
        "proposals_oldest_date": oldest_seen.isoformat() if oldest_seen else None,
        "proposals_age_days": _age(oldest_seen),
        "proposals_no_data": proposals_no_data,
        "plans_scanned": sum(len(v) for v in plans.values()),
        "newest_plan_date": newest.isoformat() if newest else None,
        "measured_on": today.isoformat(),
    }


#: The command a person runs to apply one cluster from the review packet (#4228).
#: #4197 (3ead517a) made naming the tier the ONLY route — an `--apply` with no
#: `--tiers` now refuses and returns 4 — so a list of clusters waiting for a human has
#: to state the tier with every row, and the packet cannot be read as a command to run
#: unattended.
SUFFIX_APPLY_FORM = "--apply --tiers SUFFIX_SAFE"

#: A gate-approved SUFFIX cluster that has waited this long with no decision is
#: overdue (#4228). 30 is the item's contract choice, not a measurement: the ageing
#: threshold is Alan's call, so every surface that uses it names the number it used.
SUFFIX_AGE_THRESHOLD_DAYS = 30


def suffix_review_packet(memory_graph_dir: Path,
                         today: dt.date | None = None) -> list[dict]:
    """One record per WAITING gate-approved SUFFIX cluster, oldest first (#4228).

    `merge_queue_report` answers "how big is the queue"; this answers "what is in it",
    because a reviewer handed a count of 24 cannot dispose of one. Each record carries
    what a decision needs: the canonical and variant names, the semantic gate's
    decision and its per-judge reasons, the first date the cluster was approved, how
    long it has waited since, and the apply command that drains it.

    The population is `merge_queue_report`'s, re-derived the same way from the same
    files, so the two can never disagree about who is waiting: rows of tier
    SUFFIX_SAFE with status SAFE in the plan that wins the walk (the last
    `_plan_date`-bearing file, apply plans excluded), keyed by `norm_key` then
    canonical then old, first approved in some earlier plan, and never since refused.
    A REFUSED row is the gate having answered no — 42 of the tier as at 2026-11-17 —
    and a person working this packet must never be handed one.

    `SAFE` is the semantic gate's unanimous SAME, which is why every record's
    `gate_decision` reads `SUFFIX_JUDGED — every judge: SAME`. Opens the same plan
    JSONL files and nothing else: no `kg.sqlite`, no fact directory, no merge.
    """
    today = today or dt.date.today()
    memory_graph_dir = Path(memory_graph_dir)

    dated = []
    for path in sorted(memory_graph_dir.glob(MERGE_QUEUE_PLANS_GLOB)):
        if MERGE_QUEUE_APPLY_MARKER in path.name:
            continue
        date = _plan_date(path.name)
        if date is not None:
            dated.append((date, path))
    if not dated:
        return []
    newest = max(date for date, _ in dated)

    # Every plan file of that date, not just the last one. A sweep writes several
    # plans per run date (measured 2026-11-17: three, at 160037Z/160053Z/160101Z) and
    # the last can be a run that EXCLUDED the tier, which rewrites all 66 statuses to
    # AMBIGUOUS while leaving the keys alone. `merge_queue_report` counts those keys as
    # waiting anyway, so reading only the final file would print a queue of 24 and
    # packet 0 clusters — the two surfaces have to be the same 24.
    newest_safe: dict[str, dict] = {}
    for date, path in dated:
        if date != newest:
            continue
        for row in _jsonl_rows(path):
            if row.get("tier") != SUFFIX_TIER or row.get("status") != "SAFE":
                continue
            key = str(row.get("norm_key") or row.get("canonical")
                      or row.get("old") or "").strip()
            if key:
                newest_safe.setdefault(key, row)

    first_approved: dict[str, dt.date] = {}
    ever_refused: set[str] = set()
    for date, path in sorted(dated):
        for row in _jsonl_rows(path):
            if row.get("tier") != SUFFIX_TIER:
                continue
            key = str(row.get("norm_key") or row.get("canonical")
                      or row.get("old") or "").strip()
            if not key:
                continue
            if row.get("status") == "SAFE" and key not in first_approved:
                first_approved[key] = date
            if row.get("status") == "REFUSED":
                ever_refused.add(key)

    out: list[dict] = []
    for key, row in newest_safe.items():
        if key not in first_approved or key in ever_refused:
            continue
        first = first_approved[key]
        out.append({
            "key": key,
            "canonical": str(row.get("canonical") or row.get("old") or ""),
            "variants": [str(v[0]) for v in (row.get("variants") or [])
                         if isinstance(v, (list, tuple)) and v],
            "gate_decision": str(row.get("decision") or ""),
            # The plan nests a variant's reason at gate[variant]["judges"][judge]
            # ["reason"] (measured on the live 2026-11-17 plans), one judge per
            # cluster here, and gate[variant]["decision"] is that judge's own
            # verdict. Flattened to one string per variant because the reviewer
            # reads a sentence, not a dict.
            "gate_reasons": {
                str(name): "; ".join(
                    str(judge.get("reason") or "")
                    for judge in ((gate or {}).get("judges") or {}).values()
                    if judge.get("reason"))
                for name, gate in (row.get("gate") or {}).items()},
            "first_approved": first.isoformat(),
            "age_days": (today - first).days,
            "overdue": (today - first).days > SUFFIX_AGE_THRESHOLD_DAYS,
            "apply": SUFFIX_APPLY_FORM,
        })
    out.sort(key=lambda r: (r["first_approved"], r["canonical"], r["key"]))
    return out


def suffix_review_summary(memory_graph_dir: Path,
                          today: dt.date | None = None) -> dict:
    """The queue as a reviewer needs it: how many, how long, and where the packet is.

    `over_threshold` counts clusters waiting LONGER than `SUFFIX_AGE_THRESHOLD_DAYS`,
    strictly — a cluster at exactly 30 days is not yet overdue. `oldest_age_days` is
    the age of the longest wait, 0 when nothing waits.
    """
    records = suffix_review_packet(memory_graph_dir, today=today)
    return {
        "waiting": len(records),
        "over_threshold": sum(1 for r in records if r["overdue"]),
        "threshold_days": SUFFIX_AGE_THRESHOLD_DAYS,
        "apply_form": SUFFIX_APPLY_FORM,
        "oldest_age_days": records[0]["age_days"] if records else 0,
        "oldest_first_approved": records[0]["first_approved"] if records else None,
    }


def write_suffix_review_packet(memory_graph_dir: Path,
                               today: dt.date | None = None,
                               out_dir: Path | None = None) -> Path:
    """Write the waiting queue as a dated human-readable packet; return its path.

    The queue is gate-approved and still sitting there: 24 clusters as at 2026-11-17,
    the oldest waiting since 2026-09-24, every one already judged SAFE by the semantic
    gate. `architecture/knowledge-graph.md` rule 2 says a SUFFIX merge "is never
    applied on schedule" and the write-chain ends "a SUFFIX merge waits for a human" —
    true, and the corollary was never scheduled: nothing listed the waiting clusters
    with the command a person would run, and the only printed row went to a machine
    directory under `_pipeline/` that no human reads (#4228).

    Every record states `--apply --tiers SUFFIX_SAFE`, so the packet is executable by
    a person and cannot be mistaken for something to run without the tier (#4197
    refuses that). The writer opens only plan JSONL files and writes one markdown
    file; it never opens `kg.sqlite`, never touches a fact directory, and merges
    nothing — draining the queue stays a human act (#4009).
    """
    today = today or dt.date.today()
    records = suffix_review_packet(memory_graph_dir, today=today)
    dest = Path(out_dir) if out_dir is not None else Path(memory_graph_dir)
    path = dest / f"suffix-review-{today.isoformat()}.md"
    overdue = [r for r in records if r["overdue"]]
    oldest = records[0]["age_days"] if records else 0
    lines = [
        f"# SUFFIX_SAFE review packet - {today.isoformat()}",
        "",
        f"{len(records)} gate-approved SUFFIX cluster(s) are waiting for a person. "
        f"{len(overdue)} of them have waited more than "
        f"{SUFFIX_AGE_THRESHOLD_DAYS} days; the oldest has waited {oldest}. "
        "Oldest first.",
        "",
        "Every cluster below has already been judged SAFE by the semantic gate. The",
        "rule is that a SUFFIX merge is never applied on schedule",
        "(`architecture/knowledge-graph.md` rule 2), so each one waits here until a",
        f"person runs `{SUFFIX_APPLY_FORM}` for it. Writing this packet applied",
        "nothing: it opens only the entity-merge plan files, never the store.",
        "",
    ]
    for i, rec in enumerate(records, start=1):
        variants = ", ".join(f"`{v}`" for v in rec["variants"]) or "(none named)"
        lines.append(f"{i}. `{rec['canonical']}` <- {variants}")
        lines.append(
            f"   - gate decision: {rec['gate_decision'] or '(none recorded)'}; "
            f"first approved {rec['first_approved']}; waiting {rec['age_days']} days"
            + (" - OVERDUE" if rec["overdue"] else ""))
        for name, reason in sorted(rec["gate_reasons"].items()):
            lines.append(f"   - judge reason ({name}): {reason}")
        lines.append(f"   - to apply: `{SUFFIX_APPLY_FORM}`")
        lines.append("")
    dest.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--items", action="store_true", help="list contaminated dirs")
    ap.add_argument("--baseline", type=Path, default=None,
                    help=f"entity-dir baseline to diff against (default: {BASELINE_PATH})")
    ap.add_argument("--write-baseline", action="store_true",
                    help="store today's entity-dir set as the reference regrowth diffs against, and stop")
    args = ap.parse_args()
    if args.items:
        c = contamination()
        for it in c["items"]:
            for name, slot in it["foreign"].items():
                print(f"{it['dir']!r:40} holds {slot['facts']:>3} fact(s) for {name!r}  ({', '.join(slot['files'])})")
        print(f"\n{c['dirs']} dirs, {c['foreign_facts']} foreign facts, by tier {c['by_tier']}")
        return
    if args.write_baseline:
        rec = write_baseline(baseline_path=args.baseline)
        print(f"wrote baseline {rec['path']}: {rec['dirs']:,} entity dirs at {rec['captured_at']}")
        return
    s = snapshot(days=args.days, baseline_path=args.baseline)
    if args.json:
        print(json.dumps(s, indent=2))
        return
    c, n, r = s["contamination"], s["near_duplicates"], s["regrowth"]
    u = s["untagged"]
    print("Knowledge-graph hygiene")
    print(f"  contamination   {c['dirs']:>6} dirs hold {c['foreign_facts']} facts about another entity  {c['by_tier']}")
    print(f"  untagged        {u['facts']:>6} fact records carry no entity tag "
          f"({u['share_pct']}% of {u['records']:,} parsed, {u['dirs']} dirs, "
          f"{u['files']} files; floor {u['floor_pct']}%"
          f"{' — EXCEEDED' if u['over_floor'] else ''})")
    print(f"  near-duplicates {n['clusters']:>6} clusters over {n['dirs']} dirs  {n['by_tier']}")
    print(f"  regrowth        {describe(r)}  {r['by_tier']}")
    pv = s.get("provenance") or {}
    if "facts" in pv:
        print(f"  provenance      {pv['both_pct']:>6}% of {pv['facts']:,} facts carry both created_at and source_doc")


if __name__ == "__main__":
    main()
