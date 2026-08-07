#!/usr/bin/env python3
"""
Entity-name shape analysis — Phase 1 measurement for backlog #380.

READ-ONLY. Classifies every entity directory name by "fragment shape" signals
and emits counts plus a stratified random sample for hand review. Nothing is
moved, merged or deleted; this exists so the quarantine heuristic can be
validated against real data BEFORE any sweep runs (#380 Phase 1 gate).

Why a separate predicate from `app.entity_naming.looks_like_junk_entity`:
that guard is deliberately precision-biased and runs at WRITE time, where a
false positive silently drops a real entity. Fragment detection is fuzzier and
is only ever used to build a review queue, so it lives here and is tuned for
recall. Do not wire these signals into the write path.

The extractor mints an entity per noun phrase, so the corpus grew 2,700 →
63,860 in three months. Sentence fragments cannot accumulate graph edges —
they never recur across documents — which is upstream of the #380 sparsity.

Usage:
  python scripts/memory/analyze_entity_names.py
  python scripts/memory/analyze_entity_names.py --sample 40
  python scripts/memory/analyze_entity_names.py --review-out review.tsv
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.entity_naming import looks_like_junk_entity  # noqa: E402
from app.paths import VAULT_FACTS_ROOT  # noqa: E402

# Deterministic sampling — the review set must be reproducible across runs so
# two people (or two sessions) grading it are grading the same rows.
SEED = 380

# ── Fragment signals ─────────────────────────────────────────────────────────
# Each returns True when the name looks clause-shaped rather than name-shaped.
# Tuned for RECALL; expect false positives and grade them by hand.

# Discourse / document-scaffolding openers: "Open question ...", "Part 1 ...".
#
# TIGHTENED after the first review pass: the initial list included bare nouns
# (background, decision, goal, problem, note, summary, overview, conclusion,
# rationale, caveat, takeaway) which are legitimate entity HEADS — it flagged
# `Background Agents` and `decision-logic`. Only unambiguous document
# scaffolding survives, and multi-word forms are required where the single word
# is a plausible entity head.
_OPENER_RE = re.compile(
    r"^(open questions?|part \d|section \d|step \d|appendix|option \d|"
    r"next steps?|key takeaways?|table of contents)\b",
    re.IGNORECASE,
)

# Clause connectives — a real entity name rarely contains these.
_CLAUSE_RE = re.compile(
    r"\b(during|between|when|whenever|while|because|although|though|unless|"
    r"instead of|rather than|in order to|so that|such that|due to|based on|"
    r"according to|as well as|compared to|versus)\b",
    re.IGNORECASE,
)

# Interrogative / instructional shapes: "How to Think", "Why X fails".
_WH_RE = re.compile(r"\b(how to|why|what if|whether|which of)\b", re.IGNORECASE)

# Possessive followed by a common noun: "Peter Groom's proposal for v2".
_POSSESSIVE_RE = re.compile(r"\b\w+['’]s\s+\w", re.IGNORECASE)

# Sentence-like punctuation: an internal period followed by a space.
# Excludes decimals/versions ("v3.18.1") by requiring letters either side, and
# excludes the common abbreviations that produced false positives in review
# ("Velocity vs. Adoption", "Szot et al., NeurIPS 2024").
_ABBREV = (
    "vs", "e.g", "i.e", "etc", "approx", "dr", "mr", "mrs", "ms", "st", "fig",
    "no", "inc", "ltd", "co", "al", "cf", "ca", "est", "min", "max", "avg",
)
_ABBREV_SET = {a.replace(".", "") for a in _ABBREV}
# Candidate: a period + space + letter. Python forbids variable-width
# lookbehind, so the abbreviation exclusion is applied in code below.
_SENTENCE_PUNCT_CAND_RE = re.compile(r"(\w+)\.\s+[a-z]", re.IGNORECASE)


def _has_sentence_punct(s: str) -> bool:
    """True if `s` contains a sentence break not explained by an abbreviation."""
    for m in _SENTENCE_PUNCT_CAND_RE.finditer(s):
        word = m.group(1).lower()
        # `e.g` / `i.e` arrive here as the trailing `g` / `e`; check the char
        # before the token too so those are covered.
        start = m.start(1)
        prefixed = s[max(0, start - 2):m.end(1)].lower().replace(".", "")
        if word in _ABBREV_SET or prefixed in _ABBREV_SET:
            continue
        return True
    return False

# Trailing preposition/article — a truncated phrase: "the impact of the".
_DANGLING_RE = re.compile(
    r"\b(of|in|for|to|with|on|at|by|from|the|a|an|and|or)$", re.IGNORECASE
)

SIGNALS: dict[str, object] = {
    "opener": lambda s: bool(_OPENER_RE.search(s)),
    "clause_connective": lambda s: bool(_CLAUSE_RE.search(s)),
    "wh_phrase": lambda s: bool(_WH_RE.search(s)),
    "possessive": lambda s: bool(_POSSESSIVE_RE.search(s)),
    "sentence_punct": _has_sentence_punct,
    "dangling_function_word": lambda s: bool(_DANGLING_RE.search(s.strip())),
    "very_long_7plus": lambda s: len(s.split()) >= 7,
    "long_5_6": lambda s: 5 <= len(s.split()) <= 6,
    "all_lowercase_multiword": lambda s: (
        " " in s and s == s.lower() and not any(c.isdigit() for c in s)
    ),
}

# Signals strong enough to propose quarantine on their own. Length alone is
# NOT here: "Semantic Routing for RAG Pipelines" is 5 words and legitimate.
STRONG = ("opener", "clause_connective", "wh_phrase", "sentence_punct",
          "dangling_function_word")


def classify(name: str) -> list[str]:
    return [k for k, fn in SIGNALS.items() if fn(name)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Entity-name shape analysis (#380 Phase 1)")
    ap.add_argument("--sample", type=int, default=25,
                    help="rows to sample per bucket for review (default 25)")
    ap.add_argument("--review-out", type=Path,
                    help="write the full stratified review set as TSV")
    ap.add_argument("--json-out", type=Path, help="write counts as JSON")
    args = ap.parse_args()

    root = VAULT_FACTS_ROOT
    names = [d.name for d in root.iterdir() if d.is_dir()]
    total = len(names)

    signal_counts: collections.Counter[str] = collections.Counter()
    strong_hits: list[str] = []
    weak_only: list[str] = []
    clean: list[str] = []
    code_junk: list[str] = []

    for n in names:
        if looks_like_junk_entity(n):
            code_junk.append(n)
            continue
        sigs = classify(n)
        for s in sigs:
            signal_counts[s] += 1
        if any(s in STRONG for s in sigs):
            strong_hits.append(n)
        elif sigs:
            weak_only.append(n)
        else:
            clean.append(n)

    print(f"Entity-name shape analysis — {total:,} entity dirs\n")
    print(f"  existing code/filename guard   {len(code_junk):>7,}"
          f"  ({_pct(len(code_junk), total)}%)")
    print(f"  STRONG fragment signal         {len(strong_hits):>7,}"
          f"  ({_pct(len(strong_hits), total)}%)   <- quarantine candidates")
    print(f"  weak signal only               {len(weak_only):>7,}"
          f"  ({_pct(len(weak_only), total)}%)   <- needs review")
    print(f"  no signal (looks like a name)  {len(clean):>7,}"
          f"  ({_pct(len(clean), total)}%)")

    print("\n  signal breakdown (names may hit several):")
    for k, v in signal_counts.most_common():
        marker = " *" if k in STRONG else "  "
        print(f"   {marker} {v:>7,}  {k}")

    rng = random.Random(SEED)
    buckets = {
        "STRONG": strong_hits,
        "WEAK": weak_only,
        "CLEAN": clean,
    }
    for label, pool in buckets.items():
        if not pool:
            continue
        print(f"\n  --- {label} sample ---")
        for n in rng.sample(pool, min(args.sample, len(pool))):
            print(f"    {n}")

    if args.review_out:
        rng2 = random.Random(SEED)
        rows = ["bucket\tsignals\tentity_name"]
        for label, pool in buckets.items():
            for n in rng2.sample(pool, min(args.sample, len(pool))):
                rows.append(f"{label}\t{','.join(classify(n))}\t{n}")
        args.review_out.write_text("\n".join(rows) + "\n")
        print(f"\nwrote review set → {args.review_out}", file=sys.stderr)

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "total": total,
            "code_junk": len(code_junk),
            "strong": len(strong_hits),
            "weak_only": len(weak_only),
            "clean": len(clean),
            "signals": dict(signal_counts),
        }, indent=2))


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


if __name__ == "__main__":
    main()
