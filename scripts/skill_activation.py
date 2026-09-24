"""Skill activation: does the matcher inject a skill when it should, and only then (#711).

One definition, three readers: `eval/run_skill_activation_eval.py` (the
per-skill table and its baseline), `scripts/automod/vault_round.py` (the
landing gate for a rewritten `skills/<slug>/SKILL.md`) and the tests.

**Activation is what prefetch injects**, not what the matcher ranks: the top
skill when it clears `SKILL_THRESHOLD_FIRST` (its full body) and the second
when it clears `SKILL_THRESHOLD_SECOND` (an excerpt), nothing for a message
shorter than `MIN_MESSAGE_LEN`. The ranking is `prefetch._search_skills`
itself, called with a substituted skill list when a candidate text is being
judged, so a change to the matcher moves these numbers with no second copy of
it to drift. That makes the whole thing deterministic — no model, no engine —
which is why it can run on every skill landing. It does not see the other
trigger path, dispatch-time delivery (#536), where a task names its skill.

**The corpus is a filter over `eval/skill_match_queries.yaml`** (#557), which
was built for this: for each skill named in `eval/skill_activation_cases.yaml`,
a record labelled with that skill is a `should_trigger: true` case and every
other record is a `should_trigger: false` case carrying the skill that should
have fired instead, or none. `extra_records` in the cases file are authored
supplements in the same schema, marked as such, for a skill the real turns
cover too thinly; they join the corpus for every skill.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import yaml

LLOYD_HOME = Path(__file__).resolve().parent.parent
QUERIES = LLOYD_HOME / "eval" / "skill_match_queries.yaml"
CASES = LLOYD_HOME / "eval" / "skill_activation_cases.yaml"


def load_spec(path: Path = CASES) -> dict:
    spec = yaml.safe_load(path.read_text()) or {}
    spec.setdefault("skills", {})
    spec.setdefault("extra_records", [])
    return spec


def load_records(spec: dict, queries: Path = QUERIES) -> list[dict]:
    records = list(yaml.safe_load(queries.read_text())["records"])
    return records + list(spec.get("extra_records") or [])


def build_cases(records: Iterable[dict], skills: Iterable[str]) -> dict[str, list[dict]]:
    """{skill: [case]}; a case is {id, turn, should_trigger, expected}."""
    records = list(records)
    out: dict[str, list[dict]] = {}
    for slug in skills:
        cases = []
        for rec in records:
            expected = list(rec.get("expected_skills") or [])
            cases.append({"id": rec["id"], "turn": rec["turn"],
                          "should_trigger": slug in expected,
                          # For a negative: who should have fired instead ([] = no one).
                          "expected": [s for s in expected if s != slug]})
        out[slug] = cases
    return out


def injected(turn: str, skills: list[dict] | None = None) -> list[str]:
    """The skills prefetch would inject for this turn, winner first."""
    import prefetch
    if len(turn.strip()) < prefetch.MIN_MESSAGE_LEN:
        return []
    scored = prefetch._search_skills(prefetch._query_tokens(turn), skills=skills)
    names = [s["name"] for _score, s in scored[:1]]
    if len(scored) >= 2 and scored[1][0] >= prefetch.SKILL_THRESHOLD_SECOND:
        names.append(scored[1][1]["name"])
    return names


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def score_skill(slug: str, cases: list[dict], skills: list[dict] | None = None,
                _memo: dict | None = None) -> dict:
    """Counts and rows for one skill. A rate over an empty population is None,
    never 0.0: "no false triggers" and "no negatives to trigger on" are
    different findings."""
    rows = []
    for case in cases:
        if _memo is not None and case["turn"] in _memo:
            got = _memo[case["turn"]]
        else:
            got = injected(case["turn"], skills)
            if _memo is not None:
                _memo[case["turn"]] = got
        fired = slug in got
        rows.append({"id": case["id"], "skill": slug,
                     "label": case["should_trigger"],
                     "expected": case["expected"],
                     "winner": got[0] if got else None,
                     "triggered": fired,
                     "pass": fired == case["should_trigger"]})
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    triggers = sum(r["triggered"] for r in pos)
    false_triggers = sum(r["triggered"] for r in neg)
    return {
        "skill": slug,
        "positives": len(pos), "negatives": len(neg),
        "triggers": triggers, "misses": len(pos) - triggers,
        "false_triggers": false_triggers,
        "recall": _rate(triggers, len(pos)),
        "false_trigger_rate": _rate(false_triggers, len(neg)),
        "rows": rows,
    }


def evaluate(cases_by_skill: dict[str, list[dict]], skills: list[dict] | None = None) -> dict:
    memo: dict = {}
    per = {slug: score_skill(slug, cases, skills, memo)
           for slug, cases in cases_by_skill.items()}
    tot = {k: sum(p[k] for p in per.values())
           for k in ("positives", "negatives", "triggers", "misses", "false_triggers")}
    tot["recall"] = _rate(tot["triggers"], tot["positives"])
    tot["false_trigger_rate"] = _rate(tot["false_triggers"], tot["negatives"])
    return {"skills": per, "corpus": tot}


def contract_errors(cases_by_skill: dict[str, list[dict]], minimum: int = 0) -> list[str]:
    """A skill with positives and no negatives cannot show a false trigger, so
    its table would read clean by construction."""
    errs = []
    for slug, cases in cases_by_skill.items():
        pos = sum(c["should_trigger"] for c in cases)
        neg = len(cases) - pos
        if pos and not neg:
            errs.append(f"{slug}: {pos} positives and no negatives")
        if minimum and (pos < minimum or neg < minimum):
            errs.append(f"{slug}: {pos} positives / {neg} negatives, fewer than {minimum}")
    return errs


# ── the landing gate ─────────────────────────────────────────────────────────

def skill_from_text(slug: str, text: str) -> dict | None:
    """Parse SKILL.md text exactly as the live loader does; None when the loader
    would abstain (a quarantined or retired skill)."""
    from agent_mcp.skills import _load_skill
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / slug
        d.mkdir()
        (d / "SKILL.md").write_text(text, encoding="utf-8")
        skill = _load_skill(d)
    if skill is not None:
        skill.pop("path", None)  # it named the temporary directory
    return skill


def base_skills() -> list[dict]:
    """Every live skill, fresh from disk (never prefetch's cache)."""
    from agent_mcp.skills import _iter_skills
    return list(_iter_skills())


def gate(slug: str, candidate_text: str | None, current_text: str | None, *,
         spec: dict | None = None, records: list[dict] | None = None,
         others: list[dict] | None = None) -> dict:
    """Judge a rewrite of one skill against its current text. Never raises.

    `would_refuse` is True when the candidate triggers falsely more often than
    the current text, or pushes recall under the skill's `recall_floor` (a text
    already under the floor is not refused for staying there — the edit did not
    make it worse). A skill with no entry in the cases file has no eval and is
    never blockable; neither is a retirement, which leaves nothing to score.
    """
    row: dict = {"skill": slug, "has_eval": False, "would_refuse": False, "reason": ""}
    try:
        spec = spec if spec is not None else load_spec()
        entry = (spec.get("skills") or {}).get(slug)
        if entry is None:
            row["reason"] = "no activation eval for this skill"
            return row
        row["has_eval"] = True
        cand = skill_from_text(slug, candidate_text) if candidate_text is not None else None
        if cand is None:
            row["reason"] = "skill removed or retired: nothing to score"
            return row
        records = records if records is not None else load_records(spec)
        cases = build_cases(records, [slug])[slug]
        others = [s for s in (others if others is not None else base_skills())
                  if s.get("name") != slug]
        c = score_skill(slug, cases, others + [cand])
        row["candidate"] = {k: c[k] for k in ("recall", "false_triggers", "misses")}
        cur = skill_from_text(slug, current_text) if current_text is not None else None
        if cur is not None:
            b = score_skill(slug, cases, others + [cur])
            row["current"] = {k: b[k] for k in ("recall", "false_triggers", "misses")}
        floor = entry.get("recall_floor")
        row["recall_floor"] = floor
        reasons = []
        if cur is not None and c["false_triggers"] > b["false_triggers"]:
            reasons.append(f"false triggers {b['false_triggers']} -> {c['false_triggers']}")
        if (floor is not None and c["recall"] is not None and c["recall"] < floor
                and (cur is None or b["recall"] is None or c["recall"] < b["recall"])):
            reasons.append(f"recall {c['recall']} under the floor {floor}")
        row["would_refuse"] = bool(reasons)
        row["reason"] = "; ".join(reasons) or "no regression"
    except Exception as exc:  # noqa: BLE001 — an advisory check never fails a landing itself
        row.update(would_refuse=False, reason=f"gate unavailable: {type(exc).__name__}: {exc}"[:300])
    return row
