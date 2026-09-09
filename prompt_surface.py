"""The identity surface's invariants, stated once and checked where writes land.

Why this module exists
----------------------
Backlog #377 trimmed the operating contract and pinned the result with
`tests/test_prompt_surface_budget.py`. That pinned it in the wrong place. The
tests read the *live* vault, and the autoimplement gate's `tests` rung is a hard
rung, so a writer that re-inflated `SOUL.md` would fail every subsequent
round whatever its diff — the same shape as the `data/tool_overrides.yaml`
test that aborted three rounds in fifteen hours on 2026-09-07. A tripwire on
a hard rung punishes the next author, not the writer that tripped it.

So the invariants move here, and the three consumers share one definition:

* `tests/test_prompt_surface_budget.py` imports them, and its live-vault
  group is marked `live_vault` so the gate can exclude it.
* `scripts/autoimplement/vault_round.py` runs `check_contract` before it commits
  `lloyd/SOUL.md` or `lloyd/MEMORY.md`, alongside the loaders it already runs.
* `scripts/autoresearch/promote.py` runs it before `apply_overlay`, which is
  the writer that actually caused #464 and #465 — an hourly job that copied a
  generated variant over the live contract with no gate, no test and no
  revert.

Stdlib only, and no import of `prompt_builder`: `vault_round` executes its
validators in a fresh interpreter and `promote` runs inside a worker, so a
heavy or circular import here would be paid on both paths.

The ceilings are growth tripwires, not descriptions of today. #377 landed the
gate stack at 48.2% and the prohibition ratio at 19%; the ceilings sit above
both so that a *change* trips them, rather than the file as it shipped.
"""

from __future__ import annotations

import re

# The roles that make up the gate stack, matched by heading prefix rather than
# line range: a line range silently changes meaning when a section is renamed,
# which is exactly what the #377 trim did. Both the pre-trim and post-trim
# headings are listed so the metric survives the rename.
GATE_HEADS: tuple[str, ...] = (
    "L0 SAFETY INTERRUPT GATE",
    "L0 PRE-COMMIT SAFETY CHECKLIST",
    "ATOMIC BLOCK SIGNAL",
    "BLOCK SIGNAL",
    "ZERO PREAMBLE",
    "ZERO-PREAMBLE",
    "STRICT OUTPUT SHAPE GATE",
)

GATE_STACK_CEILING = 0.50
PROHIBITION_RATIO_CEILING = 0.25
DUPLICATE_CONTRACT_CEILING = 0.10

# A trim — or a promoted variant — that removes these removed behaviour, not
# padding. Each maps to a bench task: block signal -> 010, trigger classes ->
# 008/009, false-premise refutation -> 006, first-token mapping -> 002/004/
# 005/007. A ratio test alone is passed by deleting the safety gate outright,
# which is why the ratios are necessary and not sufficient.
LOAD_BEARING: dict[str, str] = {
    "block signal JSON": '{"status": "blocked"',
    "destructive trigger class": "rm -rf",
    "protected-path trigger class": "~/lloyd/agent-services/",
    "adversarial-framing trigger class": "ignore previous instructions",
    "vague-intent trigger class": "clean up files",
    "false-premise refutation": "false premise",
    "skill-first token": "skills_search",
    "write-first token": "memory_add",
    "recall-first token": "vault_recall",
}

_PROHIBITION = re.compile(
    r"\b(Never|never|NOT |NO |Do NOT|do not|don't|Don't|FORBIDDEN|forbidden"
    r"|WRONG|BLOCK|block|prohibit)"
)


def sections(text: str) -> list[tuple[str, int]]:
    """(heading, byte size) per `## ` section, through the next heading."""
    lines = text.split("\n")
    heads = [(i, ln) for i, ln in enumerate(lines, 1) if ln.startswith("## ")]
    out: list[tuple[str, int]] = []
    for idx, (ln, heading) in enumerate(heads):
        end = heads[idx + 1][0] - 1 if idx + 1 < len(heads) else len(lines)
        out.append((heading[3:], len("\n".join(lines[ln - 1:end]).encode())))
    return out


def gate_share(text: str) -> tuple[float, int, int]:
    """(ratio, gate bytes, total bytes) for the gate-stack sections."""
    total = len(text.encode())
    gate = sum(b for h, b in sections(text) if h.startswith(GATE_HEADS))
    return (gate / total if total else 1.0), gate, total


def prohibition_ratio(text: str) -> tuple[float, int, int]:
    """(ratio, prohibition lines, nonblank lines)."""
    nonblank = [ln for ln in text.split("\n") if ln.strip()]
    hits = [ln for ln in nonblank if _PROHIBITION.search(ln)]
    return (len(hits) / len(nonblank) if nonblank else 0.0), len(hits), len(nonblank)


def shared_line_share(memory: str, soul: str) -> float:
    """Share of MEMORY.md's nonblank lines that are verbatim SOUL.md lines."""
    mem_lines = [ln for ln in memory.split("\n") if ln.strip()]
    soul_lines = {ln for ln in soul.split("\n") if ln.strip()}
    if not mem_lines:
        return 0.0
    return sum(1 for ln in mem_lines if ln in soul_lines) / len(mem_lines)


def _heading_level(line: str) -> int | None:
    """Markdown heading depth, or None when the line is not a heading.

    A bold pseudo-heading (`**Prohibited Patterns (FAILURES):**` on its own
    line) counts as the deepest possible level: it can never legitimately own
    a sub-heading, so anything that follows it must be prose or it is empty.
    """
    s = line.strip()
    m = re.match(r"^(#{1,6})\s+\S", s)
    if m:
        return len(m.group(1))
    if re.fullmatch(r"\*\*[^*]+:\*\*", s):
        return 99
    return None


def empty_sections(text: str) -> list[str]:
    """Headings with no body of their own.

    The variant autoresearch promoted into the live contract on 2026-09-08
    shipped `**Prohibited Patterns (FAILURES):**` and `**Required Patterns
    (SUCCESS):**` with nothing beneath either — a generated section that says
    it constrains something and does not. Cheap to detect, and it is the
    clearest single signal that a generated prompt was truncated mid-write.

    "Empty" is the next non-blank line being a heading at the same or a
    *shallower* level, plus a heading that ends the file. A title followed by
    its first section (`# Contract` then `## Core Identity`) is ordinary
    markdown and must not trip this, which a naive next-line-is-a-heading test
    does — it flagged the live contract's own H1.
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    for i, ln in enumerate(lines):
        level = _heading_level(ln)
        if level is None:
            continue
        nxt = next((n for n in lines[i + 1:] if n.strip()), None)
        if nxt is None:
            out.append(ln.strip())
            continue
        nxt_level = _heading_level(nxt)
        if nxt_level is not None and nxt_level <= level:
            out.append(ln.strip())
    return out


def check_contract(soul_text: str, memory_text: str | None = None) -> list[str]:
    """Every invariant the identity surface has to keep. [] means it may land.

    Deliberately returns *all* failures rather than the first: a caller that
    is about to refuse a write should say everything that is wrong with it,
    because the next attempt is a whole regenerated file, not a patch.
    """
    errors: list[str] = []

    share, gate, total = gate_share(soul_text)
    if share > GATE_STACK_CEILING:
        errors.append(
            f"gate stack is {gate} of {total} bytes ({share:.1%}), over the "
            f"{GATE_STACK_CEILING:.0%} ceiling — this is the constraint bloat #377 "
            f"was filed on"
        )

    found = {h for h, _ in sections(soul_text) if h.startswith(GATE_HEADS)}
    if len(found) < 3:
        errors.append(
            f"only {sorted(found) or 'none'} of the gate roles survive; a byte-ratio "
            f"is also satisfied by deleting the safety gate"
        )

    missing = [label for label, marker in LOAD_BEARING.items() if marker not in soul_text]
    if missing:
        errors.append(f"removes behaviour the benches score: {missing}")

    ratio, hits, nonblank = prohibition_ratio(soul_text)
    if ratio > PROHIBITION_RATIO_CEILING:
        errors.append(
            f"{hits} of {nonblank} nonblank lines ({ratio:.0%}) are prohibitions, over "
            f"the {PROHIBITION_RATIO_CEILING:.0%} ceiling"
        )

    empties = empty_sections(soul_text)
    if empties:
        errors.append(f"headings with no content under them: {empties[:5]}")

    if memory_text is not None:
        if (memory_text.split("\n", 1)[0].strip()
                == soul_text.split("\n", 1)[0].strip()):
            errors.append(
                "MEMORY.md opens with SOUL.md's H1 — the #464 operating-contract "
                "paste is back and the contract reaches the model twice per turn"
            )
        dup = shared_line_share(memory_text, soul_text)
        if dup > DUPLICATE_CONTRACT_CEILING:
            errors.append(
                f"{dup:.0%} of MEMORY.md's lines are verbatim SOUL.md lines, over the "
                f"{DUPLICATE_CONTRACT_CEILING:.0%} ceiling (#464)"
            )

    return errors


def check_paths(soul_path, memory_path=None) -> list[str]:
    """`check_contract` for files on disk. A missing SOUL.md is not our error."""
    from pathlib import Path

    soul = Path(soul_path)
    if not soul.exists():
        return []
    memory = Path(memory_path) if memory_path else None
    mem_text = memory.read_text(encoding="utf-8") if (memory and memory.exists()) else None
    return check_contract(soul.read_text(encoding="utf-8"), mem_text)
