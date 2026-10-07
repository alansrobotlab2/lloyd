#!/usr/bin/env python3
"""Mechanical gold-shape audit for a frozen LloydMemEval set (#2353).

Reads a set's questions read-only and counts, per leg, three gold-shape defects
that a deterministic judge cannot survive. It judges nothing: the
clean/brittle/defective verdicts of a human audit are a reader's call about
whether a gold value is the RIGHT value, and no count below substitutes for one.
What it does establish is which gold values are shaped so that a correct answer
can miss them or a wrong answer can hit them — the classes v1/AUDIT.md recorded
by hand as "brittle" (pr-042's gold "any", pr-044's anti value inside the right
answer, the 5-6 word golds its build validator now caps at 6).

Why this exists separately from #2170's `label_audit` manifest block: that block
is the AUDITED record — what a human judged, and the number the pilot/gold gate
reads through `run_memory_eval.label_quality`. A mechanical pass must not touch
it, and nothing here writes to a manifest or a questions file. `--set v2` prints
counts; `verify --set eval/memory_eval/v2` still prints the same `set_sha` and
the same `audited 32 of 330` afterwards.

    eval/label_audit.py --set v2
    eval/label_audit.py --set v2 --holdout

`--set` takes a version directory name under `eval/memory_eval/` or a path to a
set root. The holdout leg is counted only when `--holdout` is passed, which is
the same flag the runner uses for it, and it is counted as an aggregate: the
reserve rule in the manifest ("only a run with --holdout reads the holdout leg;
it reports aggregates, never per-question rows or ids") is why no holdout item
id, gold string or prompt ever reaches stdout here, even under `--list`.

The printed definitions are part of the output, not decoration. Two independent
passes over the same frozen YAML produced different counts before this script
existed (61/2/24 and 73/0/35 for the dev leg) purely from what an unnamed
implementation meant by "a gold value" and "a common English word". A count with
no rule beside it is a second set of unexplained numbers, so the rule that
produced these figures is printed with them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

EVAL_ROOT = Path(__file__).resolve().parent
SET_ROOT = EVAL_ROOT / "memory_eval"
LEGS = ("dev", "holdout")

#: Words this audit treats as common English. It does double duty, and both roles
#: are printed: an item whose ONLY gold value is one of these is a gold a wrong
#: answer can state by accident (v1's pr-042, gold "any"), and a word NOT in it
#: is a "content word" — so two gold/anti strings sharing one of these is the
#: pr-044 shape, where the anti value is a component of the right answer.
#:
#: Frozen, listed in full beside every count, and deliberately finite: the point
#: is a reproducible figure, not a claim about English. A reviewer who disagrees
#: with one entry can see the whole list, name the entry, and re-run.
COMMON_WORDS: tuple[str, ...] = (
    "a", "about", "above", "after", "again", "all", "also", "always", "any",
    "are", "as", "at", "back", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "did", "do",
    "does", "done", "down", "each", "either", "every", "few", "first", "for",
    "from", "further", "get", "got", "had", "has", "have", "he", "her", "here",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "like", "made", "make", "many", "may", "me", "more", "most", "much", "my",
    "never", "new", "no", "nor", "not", "now", "of", "off", "on", "once",
    "one", "only", "or", "other", "our", "out", "own", "per", "same", "she",
    "should", "since", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "upon", "us", "use", "used", "using", "very",
    "want", "was", "we", "well", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "without", "would", "yet",
    "you", "your",
)

#: The gold-length cap this pass measures. v1/AUDIT.md's build validator caps
#: gold at 6 words and its own reading was that the 5-6 word golds sit at the
#: brittle end; 3 is the tighter bar the #2353 filing named, and it is a NAME of
#: a rule the reader can re-run, not a claim that 3 is correct.
WORD_CAP = 3

_PUNCT = "\"'`.,;:()[]{}?!«»…–—-/*\\|~^@#$%&<>+="


def _tokens(text: str) -> list[str]:
    """Whitespace tokens with surrounding punctuation stripped, lowercased.

    One tokeniser for both jobs on purpose: a count that used one split and a
    word-list lookup that used another is how two runs of the same script stop
    agreeing with each other.
    """
    return [t.strip(_PUNCT).lower() for t in str(text or "").split()]


def _gold_values(q: dict) -> list[str]:
    """Every gold string the item carries: each alias of each `all_of` group.

    All of them, not one per group — which alias "counts" was one of the two
    undefined choices behind the two prior disagreeing counts, so there is no
    choice left to make: a judge matches against any alias, so every alias is a
    gold value and a long alias is a long gold.
    """
    out: list[str] = []
    for group in ((q.get("accept") or {}).get("all_of") or []):
        if isinstance(group, str):          # a bare string where a group was meant
            out.append(group)
        else:
            out.extend(str(v) for v in (group or []))
    return out


def classify(q: dict) -> dict[str, bool]:
    """The three shape defects for one item, by the rules printed with the report."""
    golds = _gold_values(q)
    over = [v for v in golds if len(_tokens(v)) > WORD_CAP]
    distinct = {v.strip().lower() for v in golds}
    sole = next(iter(distinct), "") if len(distinct) == 1 else ""
    sole_common = bool(sole) and sole in COMMON_WORDS and len(_tokens(sole)) == 1
    gold_words = set()
    for v in golds:
        gold_words.update(w for w in _tokens(v) if w and w not in COMMON_WORDS)
    clashes = []
    for anti in ((q.get("accept") or {}).get("none_of") or []):
        shared = {w for w in _tokens(anti) if w and w not in COMMON_WORDS} & gold_words
        if shared:
            clashes.append(sorted(shared))
    return {"gold_over_word_cap": bool(over),
            "sole_gold_is_common_word": sole_common,
            "none_of_shares_content_word": bool(clashes)}


DEFECTS = ("gold_over_word_cap", "sole_gold_is_common_word", "none_of_shares_content_word")

DEFINITIONS = (
    "# definitions this pass applied (the figures above are these rules, not "
    "somebody's impression):",
    "  gold value      : every alias string in every accept.all_of group; an item "
    "is counted once if ANY of its gold values trips the rule",
    "  word count      : len(gold_value.split()) after collapsing whitespace, so "
    "\"arXiv:2501.12345\" is 1 word and \"the SAM decoder patch\" is 4",
    f"  over the cap    : a gold value whose word count exceeds {WORD_CAP}",
    "  common word     : a gold value that is a single token and appears in the "
    "word list below; an item is counted only when EVERY gold string it carries "
    "is that one word",
    "  content word    : a token that is NOT in the word list below; a none_of "
    "value is flagged when it shares at least one with an all_of gold value",
    f"  word list (n={len(COMMON_WORDS)}): {', '.join(COMMON_WORDS)}",
)


def leg_questions(root: Path, leg: str) -> list[dict]:
    path = root / leg / "questions.yaml"
    if not path.exists():
        raise SystemExit(f"label_audit: {path} does not exist")
    return list((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("questions") or [])


def count_leg(qs: list[dict]) -> dict[str, int]:
    """Per-defect item counts over the leg's denominator."""
    flags = [classify(q) for q in qs]
    out = {d: sum(1 for f in flags if f[d]) for d in DEFECTS}
    out["n"] = len(qs)
    return out


def _display(root: Path) -> str:
    """Repo-relative where the set lives in the repo, so the report says the same
    thing from a checkout and from a round's worktree. An absolute path in the
    header would make the same frozen set report different bytes in two trees,
    and clause 3's "two runs are byte-identical" is about the SET, not the cwd."""
    try:
        return str(root.relative_to(EVAL_ROOT.parent))
    except ValueError:
        return str(root)


def report(root: Path, *, holdout: bool, list_items: bool) -> str:
    """The report as text. A pure function of the set's bytes: same set, same
    stdout, and nothing written anywhere."""
    sha = ""
    man = root / "manifest.json"
    if man.exists():
        sha = str(json.loads(man.read_text(encoding="utf-8")).get("set_sha") or "")
    lines = [f"# mechanical gold-shape audit — {_display(root)}",
             f"# set_sha={sha[:16] or 'unrecorded'}  "  # the engine's own short form
             "(read-only: nothing here edits a gold value)",
             ""] + list(DEFINITIONS)
    for leg in LEGS:
        if leg == "holdout" and not holdout:
            lines += ["", "## leg: holdout — not opened",
                      "   counted only with --holdout, and then as counts alone: "
                      "reserve_rule says an aggregate, never per-question rows or ids."]
            continue
        qs = leg_questions(root, leg)
        counts = count_leg(qs)
        lines += ["", f"## leg: {leg} — {counts['n']} items"]
        for defect in DEFECTS:
            lines.append(f"  {defect:<26} {counts[defect]:>4} / {counts['n']}")
        if leg == "dev" and list_items:
            for defect in DEFECTS:
                ids = sorted(q["id"] for q in qs if classify(q)[defect])
                lines.append(f"  ids {defect}: " + (", ".join(ids) if ids else "(none)"))
        elif leg == "holdout" and list_items:
            lines.append("  ids withheld: --list never names a holdout item")
    return "\n".join(lines) + "\n"


def resolve_set(spec: str) -> Path:
    p = Path(spec).expanduser()
    if (p / "manifest.json").exists() or (p / "dev" / "questions.yaml").exists():
        return p
    return SET_ROOT / spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Mechanical gold-shape audit of a frozen LloydMemEval set (#2353). "
                    "Reads only; writes nothing.")
    ap.add_argument("--set", required=True, dest="set_spec",
                    help="version name under eval/memory_eval/ (e.g. v2) or a set root path")
    ap.add_argument("--holdout", action="store_true",
                    help="count the holdout leg too, as counts only")
    ap.add_argument("--list", action="store_true",
                    help="name the dev items behind each count (never the holdout's)")
    args = ap.parse_args(argv)
    sys.stdout.write(report(resolve_set(args.set_spec),
                            holdout=args.holdout, list_items=args.list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
