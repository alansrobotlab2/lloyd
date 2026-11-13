#!/usr/bin/env python3
"""Multi-turn prefetch A/B: a two-turn prefix plus a short follow-up, scored twice.

Why this file exists (#4032). Every other prefetch instrument scores a first turn.
`eval/run_prefetch_eval.py` builds a fresh `SessionFocus` per query and calls
`focus.update(query)` on the query itself — its header says the queries run "as a
*first turn* ... no prior-turn context", its `prefix=` argument is the `[Alan]: `
speaker tag rather than a conversation prefix, and it scores the vault legs only
("Entity expectations are ignored — the facts worker is a separate path").
`eval/run_skill_match_eval.py` reads records that each carry one `turn:` field.
`eval/run_memory_eval.py` calls `prefetch_context(q.prompt)` as a single turn. So
the claim that the facts and skills legs lose a short follow-up could be probed by
hand but never measured — and a backlog item is deployed only on a measured gain.

What it does. For each case in `eval/prefetch_followup_queries.yaml` it builds a
`SessionFocus` from that case's own `prefix` — never from live session state,
because prefetch's tracker is in-memory per session, evicted LRU at 50 sessions,
and lost on restart — then calls the facts leg and the skills leg once per arm and
reports a per-arm, per-leg rate:

    raw    the follow-up alone, which is what those two legs get today
    focus  `focus.enrich_query(follow_up)`, the query the vault legs already get

Two numbers per leg, not one. On a follow-up the facts leg's failure mode is
selecting the WRONG entity (the word "one" fuzzy-matches Checkmate-in-One,
Capital One and Shantanu at 0.25-0.53, and there is no score floor between the
match and the rendered `<facts>` block), not returning nothing — so a set that
only measured hit rate would score that half as a wash. Cases flagged
`facts_must_be_empty` / `skills_must_be_empty` report the rate at which the leg
correctly came back with nothing.

Adoption is not this script's call: `prefetch.followup.focus_legs` ships off and
stays off until the numbers say otherwise. The arms here are chosen by the
harness, not by the key, so one run scores both without editing `config.yaml`;
tests/test_prefetch_followup_eval.py pins that the focus arm equals what
production's chooser returns with the key on, and tests/test_prefetch.py pins the
key's own behaviour.

Usage:
    cd ~/lloyd
    .venvs/lloyd/bin/python eval/run_prefetch_followup_eval.py --arm raw,focus
    .venvs/lloyd/bin/python eval/run_prefetch_followup_eval.py --arm focus \\
        --ids qmd-daemon-other-one --json

The A/B needs the live fact store and the live skill inventory, so the recorded
run (`eval/measurements/prefetch-followup-<date>.md`) is made after the code
lands. Pass `--dry-run` to score the case set with no store access at all: it
prints each arm's query strings and nothing else, which is the mode the schema
test runs under.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLD_SET = Path(__file__).resolve().parent / "prefetch_followup_queries.yaml"

ARMS = ("raw", "focus")
LEGS = ("facts", "skills")

#: Bounds validated on every record. The lower one is `app.prefetch.MIN_MESSAGE_LEN`:
#: `_prefetch_prepare` returns None below it, so a shorter follow-up reaches no leg
#: and a case that short would score a path that never runs. The upper one on the
#: character count keeps every record in the short-follow-up regime this item is
#: about; the content-word bound is `app.prefetch.FOLLOWUP_MAX_CONTENT_WORDS`, and
#: it is applied strictly (<) because that is the regime where enrichment fires — a
#: record at exactly the threshold would score two identical arms and dilute the
#: comparison.
MIN_FOLLOW_UP_CHARS = 10
MAX_FOLLOW_UP_CHARS = 60
MIN_CASES = 10


@dataclass
class Case:
    """One gold record: what was said, what was asked next, what should come back."""

    id: str
    prefix: list[str]
    follow_up: str
    expect_entities: list[str] = field(default_factory=list)
    expect_skills: list[str] = field(default_factory=list)
    facts_must_be_empty: bool = False
    skills_must_be_empty: bool = False
    source: str = ""
    note: str = ""

    @property
    def scores_facts(self) -> bool:
        return bool(self.expect_entities) or self.facts_must_be_empty

    @property
    def scores_skills(self) -> bool:
        return bool(self.expect_skills) or self.skills_must_be_empty


def content_words(text: str) -> int:
    """Content words as prefetch itself counts them (not `len(text.split())`)."""
    from app.prefetch import _extract_focus_keywords

    return len(_extract_focus_keywords(text))


def _clean(value, label: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case {case_id!r}: {label} must be a non-empty string")
    return value


def load_cases(path: Path | str = GOLD_SET) -> list[Case]:
    """Read and validate the gold set. Raises ValueError on any bad record.

    Checked per record: unique id; >=1 prefix turn of non-empty strings; the
    follow-up's character bounds and its content-word bound; and that the case is
    scored by something — an expected target or a must-be-empty flag. Checked over
    the set: at least MIN_CASES records, and at least one must-be-empty record, so
    the false-positive half of the failure mode cannot be dropped by editing the
    file.
    """
    import yaml

    from app.prefetch import FOLLOWUP_MAX_CONTENT_WORDS, MIN_MESSAGE_LEN

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    records = data.get("cases")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty 'cases' list")

    cases: list[Case] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: case {i} is not a mapping")
        case_id = _clean(rec.get("id"), "id", str(i))
        if case_id in seen:
            raise ValueError(f"{path}: duplicate case id {case_id!r}")
        seen.add(case_id)

        prefix = rec.get("prefix")
        if not isinstance(prefix, list) or not prefix:
            raise ValueError(f"case {case_id!r}: prefix must be a non-empty list")
        prefix = [_clean(p, f"prefix[{j}]", case_id) for j, p in enumerate(prefix)]

        follow_up = _clean(rec.get("follow_up"), "follow_up", case_id)
        if len(follow_up) < MIN_FOLLOW_UP_CHARS:
            raise ValueError(
                f"case {case_id!r}: follow_up is {len(follow_up)} chars, under "
                f"MIN_MESSAGE_LEN={MIN_MESSAGE_LEN} — such a turn reaches no leg")
        if len(follow_up) > MAX_FOLLOW_UP_CHARS:
            raise ValueError(
                f"case {case_id!r}: follow_up is {len(follow_up)} chars, over "
                f"{MAX_FOLLOW_UP_CHARS} — not a short follow-up")
        cw = content_words(follow_up)
        if cw >= FOLLOWUP_MAX_CONTENT_WORDS:
            raise ValueError(
                f"case {case_id!r}: follow_up has {cw} content words, at or over "
                f"FOLLOWUP_MAX_CONTENT_WORDS={FOLLOWUP_MAX_CONTENT_WORDS} — the "
                "focus arm would not fire on it")
        if not cw:
            raise ValueError(
                f"case {case_id!r}: follow_up has no content words, so it is a "
                "continuation cue and the continuation guard skips it")

        expect_entities = rec.get("expect_entities") or []
        expect_skills = rec.get("expect_skills") or []
        for label, seq in (("expect_entities", expect_entities),
                           ("expect_skills", expect_skills)):
            if not isinstance(seq, list) or not all(
                    isinstance(s, str) and s.strip() for s in seq):
                raise ValueError(f"case {case_id!r}: {label} must be a list of strings")
        for label in ("facts_must_be_empty", "skills_must_be_empty"):
            if not isinstance(rec.get(label, False), bool):
                raise ValueError(f"case {case_id!r}: {label} must be a boolean")

        case = Case(
            id=case_id, prefix=prefix, follow_up=follow_up,
            expect_entities=list(expect_entities),
            expect_skills=list(expect_skills),
            facts_must_be_empty=bool(rec.get("facts_must_be_empty")),
            skills_must_be_empty=bool(rec.get("skills_must_be_empty")),
            source=str(rec.get("source") or ""),
            note=str(rec.get("note") or "").strip())
        if not case.scores_facts and not case.scores_skills:
            raise ValueError(
                f"case {case_id!r}: no expect_entities, no expect_skills and no "
                "must_be_empty flag — nothing would score it")
        cases.append(case)

    if len(cases) < MIN_CASES:
        raise ValueError(f"{path}: {len(cases)} cases, minimum is {MIN_CASES}")
    if not any(c.facts_must_be_empty or c.skills_must_be_empty for c in cases):
        raise ValueError(
            f"{path}: no must-be-empty case — the wrong-entity half of the failure "
            "mode would go unmeasured")
    return cases


def build_focus(prefix: list[str], focus_factory=None):
    """A SessionFocus carrying exactly this case's prefix, from nothing else.

    Lives here rather than in the caller because the alternative — reading
    prefetch's live per-session tracker — would make the measurement depend on
    whatever the last real turn happened to be, and on whether the process had
    been restarted.
    """
    from app.prefetch import SessionFocus

    focus = (focus_factory or SessionFocus)()
    for turn in prefix:
        focus.update(turn)
    return focus


def arm_query(case: Case, arm: str, focus=None) -> str:
    """The query string one arm hands the facts and skills legs.

    `raw` is the follow-up by itself. `focus` is `focus.enrich_query(follow_up)` —
    the same string `app.prefetch._facts_skills_query` returns for a short turn
    with `prefetch.followup.focus_legs` on, pinned equal by
    tests/test_prefetch_followup_eval.py so the two cannot drift apart silently.
    """
    if arm == "raw":
        return case.follow_up
    if arm != "focus":
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    if focus is None:
        raise ValueError("the focus arm needs the case's focus")
    return focus.enrich_query(case.follow_up)


def default_facts_fn():
    """The facts leg as production calls it."""
    from app.prefetch import _search_facts

    return _search_facts


def default_skills_fn():
    """The skills leg as production calls it: (query tokens, query, session id)."""
    from app.prefetch import _query_tokens, _search_skills_for_turn

    def _skills(query: str, session_id: str) -> list[str]:
        return _search_skills_for_turn(_query_tokens(query), query, session_id)

    return _skills


def run_case(case: Case, arm: str, *, facts_fn=None, skills_fn=None,
             focus=None, session_id: str = "eval-followup-ab") -> dict:
    """Score one case under one arm. Both leg functions are injectable."""
    query = arm_query(case, arm, focus=focus)
    out = {"id": case.id, "arm": arm, "query": query}

    if case.scores_facts:
        blocks = (facts_fn or default_facts_fn())(query) or []
        joined = "\n".join(blocks).lower()
        out["facts_returned"] = bool(blocks)
        if case.facts_must_be_empty:
            out["facts_hit"] = not blocks
        else:
            out["facts_hit"] = any(e.lower() in joined for e in case.expect_entities)
    if case.scores_skills:
        names = (skills_fn or default_skills_fn())(query, session_id) or []
        out["skills_returned"] = list(names)
        if case.skills_must_be_empty:
            out["skills_hit"] = not names
        else:
            out["skills_hit"] = any(s in names for s in case.expect_skills)
    return out


def run_arms(cases: list[Case], arms: tuple[str, ...] = ARMS, *,
             facts_fn=None, skills_fn=None, focus_factory=None) -> list[dict]:
    """Every case under every arm, each with a focus built from its own prefix."""
    rows = []
    for case in cases:
        for arm in arms:
            focus = build_focus(case.prefix, focus_factory) if arm == "focus" else None
            rows.append(run_case(case, arm, facts_fn=facts_fn, skills_fn=skills_fn,
                                 focus=focus))
    return rows


def summarise(cases: list[Case], rows: list[dict]) -> dict:
    """Per-arm, per-leg rates, with the should-be-empty cases kept separate.

    `target` counts cases with an expected target, `empty` counts must-be-empty
    cases; a leg is scored on the union, and the two rates are reported apart
    because a case is in exactly one of the two groups.
    """
    by_id = {c.id: c for c in cases}
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        case = by_id[row["id"]]
        leg_stats = out.setdefault(row["arm"], {})
        for leg in LEGS:
            if leg == "facts":
                if not case.scores_facts:
                    continue
                empty_case = case.facts_must_be_empty
            else:
                if not case.scores_skills:
                    continue
                empty_case = case.skills_must_be_empty
            st = leg_stats.setdefault(leg, {"target_n": 0, "target_hits": 0,
                                            "empty_n": 0, "empty_ok": 0})
            if empty_case:
                st["empty_n"] += 1
                st["empty_ok"] += 1 if row.get(f"{leg}_hit") else 0
            else:
                st["target_n"] += 1
                st["target_hits"] += 1 if row.get(f"{leg}_hit") else 0
    for arm_stats in out.values():
        for st in arm_stats.values():
            st["target_rate"] = (st["target_hits"] / st["target_n"]) if st["target_n"] else None
            st["empty_rate"] = (st["empty_ok"] / st["empty_n"]) if st["empty_n"] else None
    return out


def render_table(summary: dict) -> str:
    lines = ["arm    leg     target n  hits  rate   empty n  ok   rate",
             "-" * 60]
    for arm in ARMS:
        for leg in LEGS:
            st = summary.get(arm, {}).get(leg)
            if not st:
                continue
            t = f"{st['target_rate']:.2f}" if st["target_rate"] is not None else "  -  "
            e = f"{st['empty_rate']:.2f}" if st["empty_rate"] is not None else "  -  "
            lines.append(f"{arm:<6} {leg:<7} {st['target_n']:>6}   {st['target_hits']:>4}  {t}"
                         f"   {st['empty_n']:>5}  {st['empty_ok']:>3}  {e}")
    lines.append("")
    lines.append("target = the leg returned the expected entity/skill; "
                 "empty = the leg returned nothing on a must-be-empty case.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default=str(GOLD_SET))
    # Singular on purpose. `architecture/measurement.md` §Arms that are flag values
    # counts every script that declares `--arms`, and `tests/
    # test_architecture_coverage_doc_claims.py` pins that count as a set of ten with
    # the section's count-word; a twelfth declarer here would move a doc number and a
    # pinned set for no reason this item needs. `dest` keeps the internal name.
    ap.add_argument("--arm", dest="arms", default="raw,focus",
                    help="comma list of raw,focus (default both)")
    ap.add_argument("--ids", default="", help="comma list of case ids to score")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each arm's query string and call no leg (no store access)")
    args = ap.parse_args(argv)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    bad = [a for a in arms if a not in ARMS]
    if bad:
        ap.error(f"unknown arm(s) {bad}; expected {list(ARMS)}")

    cases = load_cases(args.gold)
    if args.ids:
        want = {i.strip() for i in args.ids.split(",") if i.strip()}
        cases = [c for c in cases if c.id in want]

    if args.dry_run:
        rows = []
        for case in cases:
            focus = build_focus(case.prefix)
            for arm in arms:
                rows.append({"id": case.id, "arm": arm,
                             "query": arm_query(case, arm, focus=focus)})
        if args.as_json:
            import json
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"[{row['arm']:<5}] {row['id']}\n          {row['query']!r}")
            identical = sum(1 for c in cases
                            if all(r["query"] == cases[0].follow_up for r in rows
                                   if r["id"] == c.id))
            if identical == len(cases) and "focus" in arms and "raw" in arms:
                print("\nWARN: both arms produced the follow-up alone for every "
                      "case — the focus arm did not enrich anything.")
        return 0

    rows = run_arms(cases, arms)
    summary = summarise(cases, rows)
    if args.as_json:
        import json
        print(json.dumps({"rows": rows, "summary": summary}, indent=2))
    else:
        for row in rows:
            marks = " ".join(
                f"{leg}={'HIT' if row.get(f'{leg}_hit') else '---'}"
                for leg in LEGS if f"{leg}_hit" in row)
            print(f"[{row['arm']:<5}] {row['id']:<40} {marks}")
        print()
        print(render_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
