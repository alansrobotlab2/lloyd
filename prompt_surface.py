"""The identity surface's invariants, stated once and checked where writes land.

Why this module exists
----------------------
Backlog #377 trimmed the operating contract and pinned the result with
`tests/test_prompt_surface_budget.py`. That pinned it in the wrong place. The
tests read the *live* vault, and the automod gate's `tests` rung is a hard
rung, so a writer that re-inflated `SOUL.md` would fail every subsequent
round whatever its diff — the same shape as the `data/tool_overrides.yaml`
test that aborted three rounds in fifteen hours on 2026-09-07. A tripwire on
a hard rung punishes the next author, not the writer that tripped it.

So the invariants move here, and the three consumers share one definition:

* `tests/test_prompt_surface_budget.py` imports them, and its live-vault
  group is marked `live_vault` so the gate can exclude it.
* `scripts/automod/vault_round.py` runs `check_contract` before it commits
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

Everything here reads `body()`, not the raw file. The three loaded prompt files
are Obsidian notes, so they open with a YAML `---` fence that
`vault_round.frontmatter_error` *requires* and this module used to count: line 1
was `---` in all three, which made the #464 guard fire on every healthy pair and
put the fence inside every ratio it reports (#1069).
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


def body(text: str) -> str:
    """The file's content with its leading YAML front-matter block removed.

    Every loaded prompt file is an Obsidian note, so line 1 is `---` in all of
    them — and `scripts/automod/vault_round.py::frontmatter_error` *requires*
    that fence on every `.md` a vault round touches. A reader that starts at raw
    line 1 therefore reads `---` out of every file: it can compare the fence to
    the fence and call that a contract paste (#1069, and the #464 guard fired on
    every healthy pair from 2026-09-11 on), and it counts the fence and
    `type:`/`timestamp:` lines in the byte and line ratios the bloat guards
    report on.

    The closing rule is the *validator*'s, deliberately: first line starting
    `---`, closed by the next line that strips to `---`. The check that demands
    the fence and the check that reads the title have to agree where the body
    starts, or a file can be simultaneously mandatory and unreadable. An
    unclosed fence returns the text whole — `frontmatter_error` is what refuses
    that file; a guard that reported "no body" here would be a second always-fires
    artifact.
    """
    if not text.startswith("---"):
        return text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:])
    return text


def h1(text: str) -> str:
    """The body's first `# ` heading, stripped; '' when the body has none.

    Not "the first line": that is what made the duplicate-contract guard fire on
    front matter (`9ca4fc6` found it in the reporting copy, #1069 in the
    enforcement copy). Heading depth 1 only — a `## ` is a section, not a title,
    so a memory file that opens with a section is not echoing the contract.
    """
    for line in body(text).split("\n"):
        if line.startswith("# "):
            return line.strip()
    return ""


def sections(text: str) -> list[tuple[str, int]]:
    """(heading, byte size) per `## ` section of the `body()`, through the next heading.

    Body-scoped here rather than at each caller so the fence cannot be content in
    one metric and metadata in another: `##` is also YAML's comment marker, so a
    heading-shaped line inside the front matter would otherwise be counted as a
    gate section (#1069).
    """
    lines = body(text).split("\n")
    heads = [(i, ln) for i, ln in enumerate(lines, 1) if ln.startswith("## ")]
    out: list[tuple[str, int]] = []
    for idx, (ln, heading) in enumerate(heads):
        end = heads[idx + 1][0] - 1 if idx + 1 < len(heads) else len(lines)
        out.append((heading[3:], len("\n".join(lines[ln - 1:end]).encode())))
    return out


def gate_share(text: str) -> tuple[float, int, int]:
    """(ratio, gate bytes, total bytes) for the gate-stack sections.

    Both terms over `body()`, not just the denominator: numerator and denominator
    have to be the same document or the ratio is a difference of two texts. The
    fence and its keys are container metadata, and leaving them in diluted the
    bloat guard (#1069) — live SOUL.md reads 45.27% raw against 45.63% as a body,
    under a 50% ceiling.
    """
    total = len(body(text).encode())
    gate = sum(b for h, b in sections(text) if h.startswith(GATE_HEADS))
    return (gate / total if total else 1.0), gate, total


def prohibition_ratio(text: str) -> tuple[float, int, int]:
    """(ratio, prohibition lines, nonblank lines), over `body()`."""
    nonblank = [ln for ln in body(text).split("\n") if ln.strip()]
    hits = [ln for ln in nonblank if _PROHIBITION.search(ln)]
    return (len(hits) / len(nonblank) if nonblank else 0.0), len(hits), len(nonblank)


#: Consecutive recorded rises that make a climb a refusal, counting the
#: candidate itself. Three is the shortest climb the live record contains and the
#: longest run a plateau-tolerant rule can still call a trend: the census over the
#: 65 `_pipeline/research/snapshots/*/SOUL.md` promotion snapshots (2026-09-17)
#: found 28 distinct shape changes across 7 rises, longest consecutive-rise run 3
#: deduped / 2 raw. Two would refuse ordinary round-to-round jitter; four has never
#: occurred, so it would guard nothing that three does not.
CONTRACT_RISE_RUN = 3


def contract_shape(text: str) -> dict[str, float | None]:
    """The contract's two growth ratios and their denominators, computed once.

    `gate_share` and `prohibition_ratio` stay the single definition of each metric;
    this is the aggregate view for the consumers that need the *pair*, so no caller
    does its own arithmetic on the tuple. Three consumers read the pair now and two
    of them disagree about what to do with it: `check_contract` refuses on the
    absolute ceilings, `scripts/autoresearch/post_promotion.py` records it per
    round, and `scripts/autoresearch/promote.py` refuses a *climb* across rounds.
    A second copy of the arithmetic is how #1069 happened — the reporting side read
    one text while the enforcing side read another, so one said healthy while every
    writer-side gate was permanently red.

    Ratios are unrounded, deliberately: rounding before a ceiling comparison lets a
    value inside 5e-5 of the ceiling through. Rounding belongs to the record, which
    is a display artifact.

    A text whose body is empty measures as all-`None`, not as numbers.
    `gate_share` returns 1.0 when its denominator is zero — defensible as a ratio
    ("no contract at all is a maximally bloated contract"), but as a *recorded*
    shape it would log a missing or empty file as a jump to 100% and hand the
    cross-round ratchet a rise that never happened. That is the zero-denominator
    class (instance 7 of the catalogue in
    `knowledge/software/guardian-data-damage-false-trip.md`): a check whose
    denominator can be zero is not a check, and here the zero is not merely vacuous
    but wrong in the alarming direction. `check_contract` still refuses an empty
    contract — through the gate-role and load-bearing markers, which is a true
    statement about it, rather than through a ratio of nothing over nothing.
    """
    if not body(text).strip():
        return {
            "gate_share": None, "gate_bytes": None, "contract_bytes": None,
            "prohibition_ratio": None, "prohibition_lines": None,
            "nonblank_lines": None,
        }
    share, gate, total = gate_share(text)
    ratio, hits, nonblank = prohibition_ratio(text)
    return {
        "gate_share": share, "gate_bytes": gate, "contract_bytes": total,
        "prohibition_ratio": ratio, "prohibition_lines": hits,
        "nonblank_lines": nonblank,
    }


def rising_run(values: list[float], run: int = CONTRACT_RISE_RUN) -> list[float]:
    """The rising run ending at `values[-1]`, deduped, when it spans `run` values.

    `values` is the recorded series oldest first with the candidate appended last,
    so requiring the run to *end* at the last element is what keeps the verdict
    about this candidate: a history that climbed and then fell back must not
    refuse the round that fell back.

    Plateau rule, the part a naive `a < b < c` misses: a value equal to its
    predecessor neither counts as a rise nor resets the streak — it continues the
    run its predecessor belongs to. The census this rule was fitted to counts rises
    over *changed* values (28 distinct shape changes across 65 promotions, most of
    them plateaus), so a repeated measurement has to be inert, or a contract that
    did nothing for two rounds would "rise" twice on its own silence. A drop
    restarts the run at the dropping value.

    Returns the changed values of the run, so a caller can name the series in its
    refusal, and [] when there is no run.
    """
    vals = [float(v) for v in values]
    if run < 2 or len(vals) < run:
        return []
    start, rises = 0, 0
    for i in range(1, len(vals)):
        if vals[i] > vals[i - 1]:
            rises += 1
        elif vals[i] < vals[i - 1]:
            start, rises = i, 0
    if rises < run - 1:
        return []
    window = vals[start:]
    return [v for i, v in enumerate(window) if i == 0 or v != window[i - 1]]


def shared_line_share(memory: str, soul: str) -> float:
    """Share of MEMORY.md's nonblank body lines that are verbatim SOUL.md lines.

    Both sides over `body()`. On the live pair the front matter — `type: note`
    and `timestamp: …` appear in every note — was the *entire* non-zero share:
    0.0606 raw at triage (2026-09-16) and 0.0482 on 2026-09-18, against a 0.10
    ceiling, and 0.0 body-only both times. Counting it left the duplicate guard
    four to six points of real headroom, and it is what made the corrected H1
    read necessary rather than cosmetic (#1069).
    """
    mem_lines = [ln for ln in body(memory).split("\n") if ln.strip()]
    soul_lines = {ln for ln in body(soul).split("\n") if ln.strip()}
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


def duplicate_contract_errors(soul_text: str, memory_text: str) -> list[str]:
    """The two #464 halves, in one place, read over `body()`.

    Split out of `check_contract` because the reporting copy in
    `tests/test_prompt_surface_budget.py` used to carry its own `_h1()` and its
    own line-share loop. That private copy is how the two halves drifted:
    `9ca4fc6` corrected the read *there* and left this module comparing raw line
    1, so the reporting test passed 7/7 while every writer-side enforcement point
    was permanently red (#1069). One definition, asserted through from both sides.
    """
    errors: list[str] = []

    soul_h1 = h1(soul_text)
    memory_h1 = h1(memory_text)
    if memory_h1 and memory_h1 == soul_h1:
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


def check_contract(soul_text: str, memory_text: str | None = None) -> list[str]:
    """Every invariant the identity surface has to keep. [] means it may land.

    Deliberately returns *all* failures rather than the first: a caller that
    is about to refuse a write should say everything that is wrong with it,
    because the next attempt is a whole regenerated file, not a patch.

    Every measurement below — bytes, line counts, the H1 comparison — is over
    `body()`, so a file's YAML fence is neither content nor a title. One body,
    one set of ratios: a check that counted the fence in the denominator while
    reading the title from line 1 was measuring two different documents (#1069).
    Both ratios come from `contract_shape`, the same pair the per-round record and
    the cross-round ratchet read, so the number that refuses a write is the number
    a human later sees in the round report.
    """
    errors: list[str] = []

    shape = contract_shape(soul_text)
    # None means the body was empty, which has no ratio to print; the gate-role
    # and load-bearing checks below refuse that file on what it actually lacks.
    if shape["gate_share"] is not None and shape["gate_share"] > GATE_STACK_CEILING:
        errors.append(
            f"gate stack is {shape['gate_bytes']} of {shape['contract_bytes']} bytes "
            f"({shape['gate_share']:.1%}), over the "
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

    if shape["prohibition_ratio"] is not None and (
        shape["prohibition_ratio"] > PROHIBITION_RATIO_CEILING
    ):
        errors.append(
            f"{shape['prohibition_lines']} of {shape['nonblank_lines']} nonblank lines "
            f"({shape['prohibition_ratio']:.0%}) are prohibitions, over "
            f"the {PROHIBITION_RATIO_CEILING:.0%} ceiling"
        )

    empties = empty_sections(soul_text)
    if empties:
        errors.append(f"headings with no content under them: {empties[:5]}")

    if memory_text is not None:
        errors.extend(duplicate_contract_errors(soul_text, memory_text))

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
