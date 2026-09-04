#!/usr/bin/env python3
"""
Seed graph edges from relationship-category fact prose — #380 Phase 2.

Dry-run by default. Nothing is written without --apply.

WHY THIS EXISTS (vs the older `seed-relationships.py`):
that script compiles one regex per entity and searches every entity's fact text
against every pattern — O(n²). At the ~2,700-entity scale it was written for
that is ~7M operations; at today's 63,939 entities it is ~4e9 and does not
finish. This version inverts the loop with a first-token index, so each fact
text is only tested against entities whose leading token actually occurs in it.

SCOPE — deliberately narrow. Sources are only the ~4,451 entities carrying a
`<Entity>-relationship.md`, and only their `relationship`-category facts. That
subset is measurably the dense, recurring part of the corpus (12.1% singletons
vs 40.5% corpus-wide; 57.5% with 6+ facts vs 21.5%), so it avoids dragging the
one-off noun-phrase tail into the graph. See #380 Phase 1 findings.

Edges are emitted as `mentions` with provenance EXTRACTED — the same raw form
`classify-relationships-v4.py` expects as input, which then upgrades them to
typed relations. Seed → classify → apply is the full Phase 2 pipeline; this is
step one and is intentionally dumb about semantics.

NOTE: `mentions` is the edge type that was bulk-expired on 2026-06-27 (313 of
323 expirations) because daily-note filenames were being linked to entities.
This script never sources from note filenames — only from curated relationship
prose — but the volume it produces should still be reviewed before --apply.

Usage:
  python scripts/memory/seed_relationship_edges.py                 # dry-run
  python scripts/memory/seed_relationship_edges.py --sample 40     # show pairs
  python scripts/memory/seed_relationship_edges.py --apply
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.paths import VAULT_FACTS_ROOT as FACTS_DIR  # noqa: E402
from app.atomic_io import atomic_write_text  # noqa: E402

RELATIONSHIPS_FILE = FACTS_DIR / "_relationships.json"
SEED = 380

# Mirrored from seed-relationships.py so behaviour stays consistent.
MIN_ENTITY_NAME_LEN = 5
CONFIDENCE_MENTIONS = 0.8
ENTITY_STOPWORDS = {
    "test", "agent", "agents", "memory", "system", "state", "config",
    "update", "status", "event", "error", "general", "model", "tools",
    "skill", "skills", "plan", "plans", "notes", "data", "pipeline",
    "server", "service", "client", "task", "tasks", "build", "setup",
    "review", "research", "project", "debug", "audit", "queue", "cache",
    "proxy", "bridge", "index", "store", "report", "search", "query",
}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+")

# ── Name-shape rejects (added after the 2026-08-06 df-filter post-mortem) ────
# Document frequency turned out to be a BAD discriminator on its own: the
# high-df tail mixes true hubs (NVIDIA df=90, GR00T df=64, Google DeepMind
# df=44) with junk (`The paper` df=150, `related-to` df=27). Filtering purely
# on df therefore discarded exactly the hub concepts multi-hop traversal needs
# — `Knowledge Graph` was dropped at df=17 against a threshold of 15.
# These shape rejects remove the junk directly, which lets the df threshold be
# raised high enough for real hubs to survive.

# Anaphoric / determiner-led names: `The paper`, `this document`, `The fix`.
_ANAPHORIC_RE = re.compile(r"^(the|this|that|these|those|a|an|其|该)\b", re.IGNORECASE)

# Relationship-vocabulary names that leaked into the entity tree as dirs
# (`related-to` is a real entity dir with df=27).
_EDGE_VOCAB = {
    "related-to", "related_to", "relates-to", "relates_to", "mentions",
    "co_mentioned", "co-mentioned", "part_of", "part-of", "depends_on",
    "depends-on", "created_by", "created-by", "used_by", "uses",
    "implements", "supersedes", "discusses", "competes_with",
}

# All-caps ordinary English words wrongly admitted by the acronym exemption
# (`SPACE`, `FORCE`, `GRASP`, `CONNECT`, `ADEPT`, `BEHAVIOR` are all real
# entity dirs that substring-match almost any prose).
_CAPS_ENGLISH = {
    "SPACE", "FORCE", "GRASP", "CONNECT", "ADEPT", "BEHAVIOR", "SPEED",
    "POWER", "LIGHT", "SOUND", "VALUE", "SCALE", "FOCUS", "IMPACT", "SIGNAL",
    "SOURCE", "TARGET", "RESULT", "ACTION", "MOTION", "VISION", "DESIGN",
    "MEMORY", "ACCESS", "CHANGE", "DEMAND", "EFFORT", "GROWTH", "MARKET",
    "METHOD", "OBJECT", "OPTION", "OUTPUT", "PERIOD", "POLICY", "REASON",
    "SAMPLE", "SEARCH", "SERIES", "SUPPLY", "SYSTEM", "VOLUME", "STATE",
    "PLACE", "ORDER", "LEVEL", "RANGE", "SHARE", "STAGE", "TRACK", "TRUST",
    "AFTER", "BEFORE", "SCORE", "SCOPE", "FIRST", "FINAL", "TOTAL", "LIMIT",
    "PHASE", "POINT", "PRIME", "PROOF", "ROUND", "SHAPE", "SHIFT", "SPLIT",
}

# Generic category nouns — a TYPE, not an instance. `YouTube Video`,
# `GitHub Repository`, `arXiv paper`, `Backlog Item` name a class of thing and
# match indiscriminately.
_CATEGORY_TAIL_RE = re.compile(
    r"\b(video|repository|repo|profile|paper|papers|article|post|release|"
    r"item|thread|channel|account|page|document|link|url|screenshot)$",
    re.IGNORECASE,
)


def junk_target_name(name: str) -> bool:
    """True if `name` should never be an edge target regardless of frequency."""
    s = name.strip()
    if not s or s.lower() in _EDGE_VOCAB:
        return True
    if _ANAPHORIC_RE.match(s):
        return True
    if s.isupper() and s in _CAPS_ENGLISH:
        return True
    if _CATEGORY_TAIL_RE.search(s):
        return True
    return False
# Frontmatter fact entries: `  fact: <text>` possibly wrapped across lines.
FACT_LINE_RE = re.compile(r"^\s{2}fact:\s*(.+?)$", re.M)


def eligible(name: str) -> bool:
    return (
        len(name) >= MIN_ENTITY_NAME_LEN
        and name.lower() not in ENTITY_STOPWORDS
        and not UUID_RE.match(name)
        and not name.isdigit()
    )


def load_relationship_prose(entity_dir: Path) -> str:
    """Return concatenated relationship-category fact text for one entity."""
    f = entity_dir / f"{entity_dir.name}-relationship.md"
    if not f.is_file():
        return ""
    try:
        text = f.read_text(errors="ignore")
    except OSError:
        return ""
    head = text.split("---")[1] if text.startswith("---") else text
    return " ".join(m.group(1).strip() for m in FACT_LINE_RE.finditer(head))


def build_first_token_index(names: list[str]) -> dict[str, list[str]]:
    """Map leading token → entity names starting with it.

    This is the inversion that makes the pass linear-ish: a fact text only gets
    compared against entities whose first token appears in that text, instead of
    against the whole vocabulary.
    """
    idx: dict[str, list[str]] = collections.defaultdict(list)
    for n in names:
        toks = TOKEN_RE.findall(n.lower())
        if toks:
            idx[toks[0]].append(n)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed edges from relationship prose (#380)")
    ap.add_argument("--apply", action="store_true", help="write edges to disk")
    ap.add_argument("--sample", type=int, default=25, help="example pairs to print")
    ap.add_argument("--out", type=Path, help="write proposed edges as JSONL")
    ap.add_argument("--max-target-df", type=int, default=120,
                    help="drop targets matching more than N source proses. "
                         "Raised 15→120 on 2026-08-06: at 15 this discarded real "
                         "hubs (Knowledge Graph df=17, Isaac Lab df=22, Google "
                         "DeepMind df=44) which multi-hop traversal depends on. "
                         "Junk is now removed by junk_target_name() instead. "
                         "0 disables")
    ap.add_argument("--min-target-tokens", type=int, default=2,
                    help="require targets to have at least N tokens; acronyms "
                         "(all-caps, len>=3) are exempt. 1 disables")
    ap.add_argument("--max-per-source", type=int, default=40,
                    help="cap edges emitted from a single source entity "
                         "(hub guard; 0 disables)")
    args = ap.parse_args()

    all_names = [d.name for d in FACTS_DIR.iterdir() if d.is_dir()]
    targets = [n for n in all_names if eligible(n)]
    index = build_first_token_index(targets)
    lower_map = {n.lower(): n for n in targets}

    sources = [
        d for d in FACTS_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}-relationship.md").is_file()
    ]

    data = json.loads(RELATIONSHIPS_FILE.read_text()) if RELATIONSHIPS_FILE.exists() \
        else {"edges": [], "schema_version": 1}
    existing = {
        (e.get("source"), e.get("target"), e.get("type"))
        for e in data.get("edges", [])
    }

    def shape_ok(name: str) -> bool:
        """Reject single generic words; keep acronyms and multi-token names.

        `Quality`, `Knowledge`, `Developer`, `Policy`, `render` are real entity
        dirs whose names are ordinary English words, so they substring-match
        almost any prose. Requiring 2+ tokens removes them while keeping
        `Policy Transfer` / `Google DeepMind`; the acronym exemption keeps
        `HTTPS`-style names that are genuinely specific.
        """
        if args.min_target_tokens <= 1:
            return True
        toks = TOKEN_RE.findall(name.lower())
        if junk_target_name(name):
            return False
        if len(toks) >= args.min_target_tokens:
            return True
        return name.isupper() and len(name) >= 3

    # ── Pass 1: collect raw hits and per-target document frequency ──────────
    raw: list[tuple[str, str]] = []
    target_df: collections.Counter[str] = collections.Counter()
    scanned = 0

    for d in sources:
        prose = load_relationship_prose(d)
        if not prose:
            continue
        scanned += 1
        src = d.name
        src_lower = src.lower()
        prose_lower = prose.lower()
        # Candidate targets: entities whose leading token occurs in this text.
        cand: set[str] = set()
        for tok in set(TOKEN_RE.findall(prose_lower)):
            if tok in index:
                cand.update(index[tok])
        for tgt in cand:
            tl = tgt.lower()
            if tl == src_lower or not shape_ok(tgt):
                continue
            # Word-boundary confirm — the token index only proposed it.
            if not re.search(r"\b" + re.escape(tl) + r"\b", prose_lower):
                continue
            raw.append((src, tgt))
            target_df[tgt] += 1

    # ── Pass 2: drop generic targets, cap hubs, emit ────────────────────────
    generic = {t for t, n in target_df.items()
               if args.max_target_df and n > args.max_target_df}

    proposed: list[dict] = []
    seen: set[tuple[str, str]] = set()
    per_source: collections.Counter[str] = collections.Counter()
    dropped_generic = dropped_cap = 0

    for src, tgt in raw:
        if tgt in generic:
            dropped_generic += 1
            continue
        key = (src, tgt)
        if key in seen:
            continue
        if (src, tgt, "mentions") in existing or (tgt, src, "mentions") in existing:
            continue
        if args.max_per_source and per_source[src] >= args.max_per_source:
            dropped_cap += 1
            continue
        seen.add(key)
        per_source[src] += 1
        proposed.append({
            "source": src,
            "target": tgt,
            "type": "mentions",
            "confidence": CONFIDENCE_MENTIONS,
            "provenance": "EXTRACTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expired_at": None,
            "source_doc": None,
        })

    nodes = {e["source"] for e in proposed} | {e["target"] for e in proposed}
    live_before = sum(1 for e in data.get("edges", []) if not e.get("expired_at"))

    print("Seed relationship edges — #380 Phase 2 (dry-run)" if not args.apply
          else "Seed relationship edges — #380 Phase 2 (APPLYING)")
    print(f"  entity dirs                 {len(all_names):>8,}")
    print(f"  eligible as targets         {len(targets):>8,}")
    print(f"  source entities scanned     {scanned:>8,}  (of {len(sources):,} with -relationship.md)")
    print()
    print(f"  raw hits before filters     {len(raw):>8,}")
    print(f"  dropped: generic target     {dropped_generic:>8,}"
          f"  (df > {args.max_target_df}; {len(generic):,} names)")
    print(f"  dropped: per-source cap     {dropped_cap:>8,}  (cap {args.max_per_source})")
    print()
    print(f"  live edges before           {live_before:>8,}")
    print(f"  NEW edges proposed          {len(proposed):>8,}")
    print(f"  distinct nodes touched      {len(nodes):>8,}")
    if per_source:
        top = per_source.most_common(1)[0]
        print(f"  max edges from one source   {top[1]:>8,}  ({top[0]})")
        print(f"  median edges per source     {sorted(per_source.values())[len(per_source)//2]:>8,}")

    if proposed and args.sample:
        print(f"\n  --- sample of {min(args.sample, len(proposed))} proposed edges ---")
        for e in random.Random(SEED).sample(proposed, min(args.sample, len(proposed))):
            print(f"    {e['source']}  →  {e['target']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(e) for e in proposed) + "\n")
        print(f"\nwrote proposals → {args.out}", file=sys.stderr)

    if args.apply:
        backup = RELATIONSHIPS_FILE.with_suffix(
            f".json.{datetime.now().strftime('%Y%m%dT%H%M%SZ')}.bak")
        backup.write_text(RELATIONSHIPS_FILE.read_text())
        data.setdefault("edges", []).extend(proposed)
        atomic_write_text(RELATIONSHIPS_FILE, json.dumps(data, indent=2) + "\n", fsync=True)
        print(f"\napplied {len(proposed):,} edges; backup → {backup.name}")
    else:
        print("\n(dry-run — pass --apply to write)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
