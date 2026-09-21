"""Bench validity lint — negative controls, requirement coverage, vacuous layers.

Why this exists (#646)
----------------------
Every bench task votes on prompt promotions through ``mean_composite``, and the
objective layer of that composite is substring and ``re.search`` presence
(``judge.py:170`` substring, ``judge.py:173`` ``re.search``). A task whose objective layer is
satisfied by boilerplate that does no work therefore hands its share of the
promotion decision to whoever types the right keywords, and a bench where a
third of the tasks do that re-ranks models on noise. The measured baseline, in
the round that wrote this file: 9 of the 13 tasks in ``cfg.paths.bench_dir``
score a full 1.00 on their objective layer against a string assembled from the
task's own check values, with no model call.

Three questions, one report per task
------------------------------------
1. ``lazy_pass`` — does a mechanically-derived lazy response pass this task's
   objective layer? The probe is *derived from the task itself*: every
   ``contains`` value plus every top-level ``regex`` alternation branch, joined
   into one string, evaluated through ``judge._score_objective`` with
   ``tool_calls`` empty and ``tool_trace_authoritative`` unset — the
   trace the direct-completion runner produces (``bench_runner.py:83`` hardcodes an
   empty ``tool_calls``, and #416 has the judge report a tool-behaviour check on such a
   trace as ``NOT_MEASURABLE``). No model,
   no LLM, no judgement: same function the round scores with.
2. ``uncovered_requirement`` — two-way prompt↔verifier alignment. A structural
   requirement stated in the prompt or body (a syllable count, a section/sentence
   count, an output format) that no objective check and no named ``rubric_criteria``
   entry covers is an ask the verifier will never grade.
3. ``vacuous_objective`` / ``objective_only_max_tool_calls`` — objective layers
   that cannot fail. ``_score_objective`` returns a free 1.0 for an empty check
   list (``judge.py:218-219``), and a layer whose only check is ``max_tool_calls``
   measures nothing on the arm that runs it and passes on any trace that calls
   nothing on the arm that would.

The lint flags; it does not fix, and it does not retire. Tightening a check or
dropping a task changes what promotes on the live self-mod loop, so that call is
a human's (#646's deferred clause). What the lint *does* decide is validity: a
task carrying any ``error`` finding is not lint-valid, and
``promote.validity_report`` scores the valid subset separately so the
keyword-soup tasks stop carrying the advisory mean.

Why the free 1.0 stays in ``judge.py``
--------------------------------------
Step 4 of #646 asks that an empty objective layer become a lint error "instead
of a free 1.0". It is a lint error here, and the free marks stop mattering
because an invalid task is excluded from the valid-pool mean. ``judge.py``'s
arithmetic is deliberately untouched: the same function re-scores replayed
historical rounds (``replay_promotion_gate.py``), and silently re-scoring stored
rows would rewrite the record of decisions already taken under the old
arithmetic. Neutralising the score where the decision is made, rather than
rewriting history, is the reversible half.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # direct execution: python scripts/autoresearch/bench_lint.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.autoresearch.common import load_bench_tasks, load_config
from scripts.autoresearch.judge import TOOL_BEHAVIOUR_CHECKS, _score_objective

#: The harness shape the probe is evaluated in. ``bench_runner.py:83`` hardcodes
#: ``tool_calls: []`` and never sets ``tool_trace_authoritative``, so this is a
#: trace with no dispatch record: since #416 the judge reports every
#: ``TOOL_BEHAVIOUR_CHECKS`` type on it as ``NOT_MEASURABLE`` instead of guessing
#: from the reply text, and only the ``contains``/``regex`` layer is graded. That
#: is the arm the nightly round actually runs for every task but ``bench_010``.
DIRECT_MODE_TRACE: dict[str, Any] = {"status": "success", "tool_calls": []}

ERROR = "error"
NOTE = "note"


# ---------------------------------------------------------------------------
# 1. lazy-response negative controls
# ---------------------------------------------------------------------------

#: Regex atoms that stand for a character class or an escape, mapped to one
#: concrete character they can match. A lazy reply cannot type ``\s*``, it types
#: a space, so the probe carries the rendered form too — otherwise a branch like
#: ``"status"\s*:\s*"blocked"`` would be probed only as its own backslashes, which
#: that same pattern cannot match, and the probe would under-report.
_REGEX_ATOMS = (
    (re.compile(r"\\[sSwWdD]\*?"), " "),
    (re.compile(r"\\[nrt]\*?"), " "),
    (re.compile(r"\\[bB]\*?"), ""),
    (re.compile(r"\(\?[:<=!][^)]*\)"), ""),
    (re.compile(r"\\(.)"), r"\1"),
)


def _render_branch(branch: str) -> str:
    """A concrete string the branch `branch` can match, for probing.

    Only the escapes above are rendered; ``.`` and ``*`` are left alone, because
    a literal ``.*`` inside the probe still satisfies ``.*`` in the pattern and
    inventing filler for it would make the probe less mechanical, not more.
    """
    out = branch
    for pattern, repl in _REGEX_ATOMS:
        out = pattern.sub(repl, out)
    return out


def _alternation_branches(pattern: str) -> list[str]:
    """Top-level ``|`` branches of a regex, outer capture group stripped.

    Splitting is depth-aware so ``(a|b)c|(d)e`` yields ``['(a|b)c', '(d)e']`` and
    not four fragments — a fragment is not a string the whole pattern matches.
    """
    text = pattern.strip()
    # Strip one layer of enclosing parens when they wrap the whole pattern.
    while text.startswith("(") and text.endswith(")") and _matching_open(text, len(text) - 1) == 0:
        text = text[1:-1]
    branches: list[str] = []
    depth = 0
    in_class = False
    current: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            current.append(ch)
            escaped = True
            continue
        if ch == "[":
            in_class = True
        elif ch == "]" and in_class:
            in_class = False
        elif not in_class and ch == "(":
            depth += 1
        elif not in_class and ch == ")":
            depth = max(0, depth - 1)
        elif not in_class and ch == "|" and depth == 0:
            branches.append("".join(current))
            current = []
            continue
        current.append(ch)
    branches.append("".join(current))
    return [b for b in (br.strip() for br in branches) if b]


def _matching_open(text: str, close_index: int) -> int:
    depth = 0
    for i in range(close_index, -1, -1):
        if text[i] == ")":
            depth += 1
        elif text[i] == "(":
            depth -= 1
            if depth == 0:
                return i
    return -1


def lazy_probe(task: dict[str, Any]) -> list[str]:
    """The tokens of this task's lazy response: its own check values, joined.

    ``contains`` values go in verbatim. ``regex`` values contribute each
    alternation branch twice — as written and as rendered — plus the whole
    pattern, so a pattern with no alternation is still probed.
    ``tool_called`` / ``tool_not_called`` / ``max_tool_calls`` contribute
    nothing: naming a tool is behaviour, not a keyword, and the item's probe is
    defined over the string checks. What those checks *would* do in direct mode
    is reported separately by :func:`mode_notes`.
    """
    tokens: list[str] = []
    for check in task.get("objective_checks") or []:
        ctype = check.get("type")
        value = str(check.get("value", ""))
        if ctype == "contains":
            if value:
                tokens.append(value)
        elif ctype == "regex":
            if not value:
                continue
            tokens.append(value)
            for branch in _alternation_branches(value):
                tokens.append(branch)
                rendered = _render_branch(branch)
                if rendered and rendered != branch:
                    tokens.append(rendered)
    return tokens


def lazy_result(task: dict[str, Any]) -> dict[str, Any]:
    """Score the lazy probe through the real objective scorer. No model call.

    ``objective_score`` is ``None`` when nothing on this task could be measured —
    every check it declares is a tool-behaviour check and the probe carries no
    dispatch record, which is #416's not-rankable state. That is reported as
    ``lazy_pass: false`` with the reason in ``unmeasurable_checks`` rather than as
    a pass: boilerplate that the harness cannot even grade has not satisfied the
    task, and calling it a pass would be the same mistake in the other direction.
    """
    tokens = lazy_probe(task)
    text = ", ".join(tokens)
    trace = {**DIRECT_MODE_TRACE, "final_text": text}
    score, results = _score_objective(task, trace)
    unmeasurable = [f"{r.get('type')}={r.get('value')}"
                    for r in results if r.get("measured") is False]
    return {
        "lazy_pass": bool(tokens) and score is not None and score >= 1.0,
        "objective_score": round(score, 4) if score is not None else None,
        "probe": text,
        "unmeasurable_checks": unmeasurable,
        "checks_passed": [
            f"{r.get('type')}={r.get('value')}" for r in results if r.get("passed")
        ],
        "checks_failed": [
            f"{r.get('type')}={r.get('value')}" for r in results if r.get("passed") is False
        ],
    }


# ---------------------------------------------------------------------------
# 2. requirement coverage
# ---------------------------------------------------------------------------

_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)"

#: (kind, requirement pattern, words that would cover it in a check or criterion)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "syllable_structure",
        r"\d+\s*[-–]\s*\d+(?:\s*[-–]\s*\d+)*\s*(?:syllable|syllables)|\bsyllable\b|\bsyllables\b",
        ("syllable", "syllables", "haiku"),
    ),
    (
        "element_count",
        rf"{_NUM}\s+(?:or\s+{_NUM}\s+)?(?:sections?|headings?|sentences?|paragraphs?|bullets?|bullet points?|lines?|verses?|stanzas?|words)\b",
        # A covering check has to name the unit, the numeral, or the shape.
        ("section", "heading", "sentence", "paragraph", "bullet", "line", "verse", "stanza", "word", "count"),
    ),
    (
        "output_format",
        r"\b(haiku|sonnet|limerick|acrostic|json|json object|markdown table|table|numbered list|bullet list|yaml)\b",
        ("haiku", "sonnet", "limerick", "acrostic", "json", "table", "list", "yaml", "format"),
    ),
    (
        "ordering",
        r"\b(ordered list|in order|alphabetical|alphabetically|chronological|chronologically)\b",
        ("order", "ordered", "alphabet", "chronolog"),
    ),
)


def _requirement_hits(*sources: str | None) -> list[tuple[str, str]]:
    """(kind, matched text) for every structural requirement the sources state."""
    hits: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        if not source:
            continue
        for kind, pattern, _covers in _REQUIREMENTS:
            for m in re.finditer(pattern, source, re.IGNORECASE):
                key = (kind, m.group(0).strip().lower())
                if key not in seen:
                    seen.add(key)
                    hits.append(key)
    return hits


def _covering_terms(task: dict[str, Any]) -> str:
    """Everything the task's own verification layer names, lowercased.

    Only objective check values and named ``rubric_criteria`` count as coverage.
    ``tags`` are excluded on purpose: ``bench_011`` is tagged ``format`` while
    nothing grades format, which is the exact gap this function exists to find.
    """
    parts: list[str] = []
    for check in task.get("objective_checks") or []:
        parts.append(str(check.get("type", "")))
        parts.append(str(check.get("value", "")))
    parts.extend(str(c) for c in (task.get("rubric_criteria") or []))
    return " ".join(parts).lower()


def coverage_findings(task: dict[str, Any]) -> list[dict[str, Any]]:
    stated = _requirement_hits(
        task.get("prompt"), task.get("objective"), task.get("_body"), task.get("body")
    )
    covered_by = _covering_terms(task)
    findings = []
    for kind, text in stated:
        keywords = next(kws for k, _p, kws in _REQUIREMENTS if k == kind)
        # The matched text itself is the strongest covering term: a check that
        # carries `5-7-5` or the word `haiku` covers it whether or not the
        # keyword list predicted that spelling.
        needles = list(keywords) + [text]
        hit = next((n for n in needles if n in covered_by), None)
        if hit is None:
            findings.append({
                "kind": "uncovered_requirement",
                "severity": ERROR,
                "requirement": kind,
                "stated_as": text,
                "message": (
                    f"states a {kind.replace('_', ' ')} ({text!r}) that no objective check "
                    "and no rubric_criteria entry covers"
                ),
            })
    return findings


# ---------------------------------------------------------------------------
# 3. vacuous objective layers
# ---------------------------------------------------------------------------

#: Check types whose verdict is a claim about tool behaviour. Since #416 the judge
#: returns ``NOT_MEASURABLE`` for these on a trace with no dispatch record instead
#: of falling back to whether the tool's name appears in the reply text — so the
#: question about them is no longer "does prose pass it" but "can any harness we
#: actually run read it at all". Imported from the judge rather than restated, so
#: a fifth tool check type cannot drift past this file.
_UNMEASURABLE_ON_DIRECT = TOOL_BEHAVIOUR_CHECKS


def vacuity_findings(task: dict[str, Any]) -> list[dict[str, Any]]:
    checks = task.get("objective_checks") or []
    findings: list[dict[str, Any]] = []
    if not checks:
        findings.append({
            "kind": "vacuous_objective",
            "severity": ERROR,
            "message": (
                "has no objective_checks, so _score_objective hands it a full 1.0 "
                "unconditionally (judge.py:218-219: `if not checks: return 1.0, []`) "
                "and it cannot fail"
            ),
            "free_objective_score": 1.0,
        })
        return findings
    kinds = [str(c.get("type")) for c in checks]
    if set(kinds) == {"max_tool_calls"}:
        findings.append({
            "kind": "objective_only_max_tool_calls",
            "severity": ERROR,
            "message": (
                f"objective layer is only max_tool_calls (value={checks[0].get('value')!r}): "
                "on the direct arm #416 reports it NOT_MEASURABLE, and on a trace with a "
                "dispatch record it passes whenever the run called nothing — no arm can "
                "have it refuse a reply"
            ),
        })
    return findings


def mode_notes(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks no harness this round can actually inform. Advisory, so they do not
    change validity — a ``tool_called`` check is a real check on the arm that
    carries a dispatch record; it is only unreachable for a task that is never
    routed there."""
    notes = []
    checks = task.get("objective_checks") or []
    routed = bool(task.get("requires_runtime"))
    for check in checks:
        ctype = str(check.get("type"))
        if ctype not in _UNMEASURABLE_ON_DIRECT or routed:
            continue
        notes.append({
            "kind": "unmeasurable_on_every_configured_arm",
            "severity": NOTE,
            "check": f"{ctype}={check.get('value')}",
            "message": (
                f"{ctype} is a claim about tool dispatch, so #416 has the judge report it "
                "NOT_MEASURABLE rather than guess from the reply text; the direct runner "
                "produces no dispatch record (bench_runner.py:83 hardcodes an empty "
                "`tool_calls`) and this task sets no `requires_runtime: true`, so nothing "
                "routes it to the arm that has one"
            ),
        })
    if notes and len(notes) == len(checks):
        notes.append({
            "kind": "objective_layer_unmeasurable_in_direct_mode",
            "severity": NOTE,
            "check": "(whole layer)",
            "message": (
                "every check this task declares is a tool-behaviour check and it sets no "
                "`requires_runtime: true`, so on the arm that runs it its objective layer "
                "scores None and the trial is dropped from `mean_composite` as not-rankable "
                "(#416) — the task contributes a rubric score and nothing else"
            ),
        })
    return notes


# ---------------------------------------------------------------------------
# per-task and whole-dir reports
# ---------------------------------------------------------------------------

def lint_task(task: dict[str, Any]) -> dict[str, Any]:
    lazy = lazy_result(task)
    findings = vacuity_findings(task) + coverage_findings(task)
    if lazy["lazy_pass"]:
        findings.append({
            "kind": "lazy_pass",
            "severity": ERROR,
            "message": (
                f"objective layer scores {_fmt_score(lazy['objective_score'])} on a lazy response built "
                f"from its own check values: {lazy['probe'][:160]!r}"
            ),
            "checks_passed": lazy["checks_passed"],
        })
    errors = [f for f in findings if f.get("severity") == ERROR]
    return {
        "id": task.get("id") or task.get("_path") or "?",
        "safety_critical": bool(task.get("safety_critical")),
        "lazy_pass": lazy["lazy_pass"],
        "lazy_objective_score": lazy["objective_score"],
        "lazy_probe": lazy["probe"],
        "checks_failed_by_lazy": lazy["checks_failed"],
        # Named so a `None` score is never read as a zero: these are the checks
        # the probe's trace could not answer, not checks the probe failed.
        "unmeasurable_checks": lazy["unmeasurable_checks"],
        # A safety-critical task that lazy-passes is the loudest row in this
        # report: `safety_passed` is a veto on promotion, and a veto that
        # boilerplate satisfies is not a veto.
        "safety_gate_lazy_pass": bool(task.get("safety_critical")) and lazy["lazy_pass"],
        "coverage_clean": not any(f["kind"] == "uncovered_requirement" for f in findings),
        "findings": findings,
        "notes": mode_notes(task),
        "valid": not errors,
        "error_kinds": sorted({f["kind"] for f in errors}),
    }


def lint_bench_dir(bench_dir: Path | str) -> dict[str, Any]:
    tasks = load_bench_tasks(bench_dir)
    reports = [lint_task(t) for t in tasks]
    lazy_passing = sorted(r["id"] for r in reports if r["lazy_pass"])
    invalid = sorted(r["id"] for r in reports if not r["valid"])
    return {
        "bench_dir": str(bench_dir),
        "task_count": len(reports),
        "tasks": sorted(reports, key=lambda r: r["id"]),
        "lazy_pass_count": len(lazy_passing),
        "lazy_passing": lazy_passing,
        "invalid": invalid,
        "valid": sorted(r["id"] for r in reports if r["valid"]),
        "safety_gate_lazy_pass": sorted(r["id"] for r in reports if r["safety_gate_lazy_pass"]),
        "uncovered": sorted(r["id"] for r in reports if not r["coverage_clean"]),
    }


def valid_task_ids(bench_dir: Path | str) -> set[str]:
    """The lint-valid task ids — the pool the advisory valid-only mean is over."""
    return set(lint_bench_dir(bench_dir)["valid"])


# ---------------------------------------------------------------------------
# rendering / CLI
# ---------------------------------------------------------------------------

def _fmt_score(score: float | None) -> str:
    """`0.00`, or `n/a` — which means no check on this task could be measured at
    all, not that the probe scored zero."""
    return "n/a" if score is None else f"{score:.2f}"


def render(report: dict[str, Any]) -> str:
    lines = [
        f"# Bench validity lint — {report['bench_dir']}",
        f"- tasks: {report['task_count']}",
        f"- lazy_pass: {report['lazy_pass_count']} of {report['task_count']}: {', '.join(report['lazy_passing']) or '(none)'}",
        f"- lint-valid: {len(report['valid'])} of {report['task_count']}: {', '.join(report['valid']) or '(none)'}",
        f"- safety gate satisfied by a lazy response: {', '.join(report['safety_gate_lazy_pass']) or '(none)'}",
        f"- uncovered structural requirements: {', '.join(report['uncovered']) or '(none)'}",
        "",
        "| task | lazy | obj | valid | findings |",
        "|---|---|---|---|---|",
    ]
    for r in report["tasks"]:
        kinds = sorted({f["kind"] for f in r["findings"]})
        lines.append(
            f"| `{r['id']}` | {'YES' if r['lazy_pass'] else 'no'} | "
            f"{_fmt_score(r['lazy_objective_score'])} | "
            f"{'yes' if r['valid'] else 'NO'} | "
            f"{', '.join(kinds) or '—'} |"
        )
    lines.append("")
    for r in report["tasks"]:
        if not r["findings"] and not r["notes"]:
            continue
        lines.append(f"## {r['id']}")
        for f in r["findings"]:
            lines.append(f"- **{f['kind']}** ({f['severity']}): {f['message']}")
        for n in r["notes"]:
            lines.append(f"- note ({n['check']}): {n['message']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench-dir", default=None,
                        help="default: cfg.paths.bench_dir from config.yaml")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any task is lint-invalid")
    args = parser.parse_args(argv)
    bench_dir = Path(args.bench_dir) if args.bench_dir else load_config().paths.bench_dir
    report = lint_bench_dir(bench_dir)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 1 if (args.strict and report["invalid"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
