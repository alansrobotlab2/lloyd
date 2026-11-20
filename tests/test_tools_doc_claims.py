"""`architecture/tools.md` must not tell a reader a `Task` batch runs sequentially.

Two sentences on that page described batch overlap for a tree that had already
moved. §5's `May overlap in one iteration` row gave `readOnlyHint` alone as the
batch predicate and stamped the whole row `(**ships off**)`; §9's depth-1 ruling
claimed a turn delegates to at most one live child at a time because a batch
containing `Task` runs sequentially. Neither has been true since `8ac6f4a8`
(2026-09-24) shipped P8: `loop._batch_is_read_only` admits a fresh `Task` whose
`subagent_type` is a `parallel_safe` profile, `loop._is_task_fanout` overlaps a
batch made of nothing but those even with general dispatch off, and `config.yaml`
sets `parallel_safe: true` on the `read-only` profile against
`harness.parallel_tool_calls.enabled: false` with `max_concurrency: 4`. The real
peak is 4 live children per iteration, not one. The §9 sentence is the load-
bearing one: written a day *after* P8 (`fa94f9da`, #1515), it sat inside the
depth-1 rationale and was the half of the KV-pressure argument that bounded the
multiplication at one child per turn. #4434 corrected both sentences in prose;
this file is what stops them coming back.

Same instrument as `tests/test_harness_doc_claims.py` (the #505 class): shipped
states are pinned to the shipped config as the runtime reads it —
`CONFIG['harness']['parallel_tool_calls']` and
`CONFIG['subagents'][<profile>]['parallel_safe']` — never to a literal here. Flip
either switch and a test here fails saying to move the prose, instead of the
prose quietly going stale again. The pre-fix sentences are kept verbatim as
controls: every detector that replaced pre-fix prose must fire on them, or a
green proves nothing — and the one detector guarding a sentence the old page
already got right is proved by mutating that sentence out instead.

Two deliberate judgements, both learned the hard way:

* **No blanket ban on the words `ships off`.** General parallel dispatch genuinely
  does ship off, so true prose saying so must keep passing; what is required is a
  stated state per mechanism, which fails the old single-stamp row for the right
  reason. Likewise a batch that *mixes* `Task` with a `Read` really is sequential:
  the ban targets the claim that a `Task` in a batch serializes everything, not
  every sentence containing `Task` and `sequentially`.
* **The breadth/state detectors read prose positionally, not by exact wording**, so
  a truthful rewording in either order survives: each mechanism's state is the
  nearest `on`/`off` to its own name, and the breadth number is the nearest integer
  to `max_concurrency`. Both fail loudly (naming the cell, or reporting no digit at
  all) rather than passing on a scan that found nothing to read.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import CONFIG

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "tools.md"
DOC_TEXT = DOC.read_text(encoding="utf-8")

#: The §5 `What each property decides` row this file exists for.
OVERLAP_ROW_KEY = "May overlap in one iteration"

#: The §9 section whose depth-1 ruling carried the false breadth claim.
SUBAGENT_HEADING = "### Subagents (`Task`)"

#: The config key the doc cites for the breadth cap. Both spellings are in
#: circulation — `harness.parallel_tool_calls.max_concurrency` in config.yaml,
#: `parallel_tool_calls_max_concurrency` as the `RunOptions` attribute — so the
#: checks below match on the common tail.
MAX_CONCURRENCY_KEY = "max_concurrency"

#: How far from a mechanism's name its own state word may sit, and how far from
#: the breadth key its number. Generous enough for a parenthesised config key,
#: tight enough that prose two sentences away cannot stand in for either.
STATE_WINDOW = 64
BREADTH_WINDOW = 64

#: True prose about the same mechanisms, which the bans below must NOT match —
#: otherwise the fix would be a ban on the topic rather than a correction.
TRUE_PROSE_ABOUT_OFF_DISPATCH = (
    "### Subagents (`Task`)\n\n"
    "General parallel dispatch ships off, so a batch that mixes `Task` with a "
    "Read stays sequential. **No per-level cost has been measured**.")

#: Sentences that re-assert the pre-fix mechanism: a `Task` in a batch excludes
#: it from overlap, so a turn holds one child.
STALE_TASK_SERIALIZATION_RES = [
    re.compile(r"one live child at a time", re.I),
    re.compile(r"at most one live child", re.I),
    re.compile(r"one child per turn", re.I),
    re.compile(r"batch containing[^\n.]{0,40}\bruns sequentially\b", re.I),
    re.compile(r"a batch[^\n.]{0,40}\bruns sequentially\b", re.I),
]

#: The two false sentences, verbatim, as the controls' inputs.
PRE_FIX_OVERLAP_ROW = (
    "| May overlap in one iteration | every call in the batch `readOnlyHint` "
    "| the annotation, as discovered | `loop.py` parallel dispatch "
    "(**ships off**) |")
PRE_FIX_BREADTH_SENTENCE = (
    "At depth 1, and with `Task` not `readOnlyHint` so a batch containing it "
    "runs sequentially, one turn delegates to at most one live child at a time.")
PRE_FIX_SUBAGENT_SECTION = (
    f"{SUBAGENT_HEADING}\n\n{PRE_FIX_BREADTH_SENTENCE}\n")

#: The changelog form that survived the prose fix: a refuted claim is still a
#: claim in anyone's search results, and this is the occurrence #4434's own
#: verification grep counts. The page refers to the deleted sentence as a
#: hyphenated noun phrase instead of quoting it.
PRE_FIX_CHANGELOG_QUOTE = (
    "Two claims the tree refutes outright: §5 and §9's "
    '"at most one live child at a time" — the P8 arm …')

#: §5's own summary of the table, as it read before this round. True of every row
#: until the overlap row started reading a config key as well as the wire hint.
PRE_FIX_SUMMARY_SENTENCE = (
    "The first three read the frozenset in-process. The next two read the hint as\n"
    "it crossed the wire. They agree except where a module set its own annotation.")
PRE_FIX_TABLE_TAIL = (
    f"{PRE_FIX_OVERLAP_ROW}\n"
    "| Effect-ledgered | FX | the sets | `_tool_effects.claim` |\n\n"
    f"{PRE_FIX_SUMMARY_SENTENCE}\n")

#: The claim that no per-level cost is known. §9 may be rewritten around it, but
#: dropping it would leave a structural ruling reading like a measured one.
UNMEASURED_CLAIM = "**No per-level cost has been measured**"


# ── shipped state, read the way the runtime reads it ────────────────────────

def _ptc() -> dict:
    """`harness.parallel_tool_calls` as `app.config` resolves it."""
    ptc = CONFIG["harness"]["parallel_tool_calls"]
    assert isinstance(ptc, dict), f"harness.parallel_tool_calls is {ptc!r}"
    return ptc


def _general_dispatch_ships_on() -> bool:
    return bool(_ptc()["enabled"])


def _shipped_max_concurrency() -> int:
    return int(_ptc()["max_concurrency"])


def _shipped_parallel_safe_profiles() -> list[str]:
    """Profiles with `parallel_safe: true` — the only ones P8 can fan out."""
    subagents = CONFIG.get("subagents", {})
    assert isinstance(subagents, dict) and subagents, "no subagents in config"
    return sorted(name for name, prof in subagents.items()
                  if isinstance(prof, dict) and prof.get("parallel_safe"))


# ── extraction ──────────────────────────────────────────────────────────────

def _row_cells(text: str, key: str) -> list[str]:
    """The cells of the table row whose first cell is exactly `key`.

    Asserts the row exists: a green must never come from a scan that found
    nothing to read.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and cells[0] == key:
            assert len(cells) >= 3, f"row {key!r} has {len(cells)} cells: {line!r}"
            return cells
    raise AssertionError(
        f"no table row keyed {key!r} in {DOC.relative_to(ROOT)}")


def _section(text: str, heading: str) -> str:
    """The body under `heading`, up to the next `##`/`###` heading."""
    out, seen = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("### ") or line.startswith("## "):
            if seen:
                break
            seen = line.startswith(heading)
            continue
        if seen:
            out.append(line)
    assert seen, f"no heading starting {heading!r} in {DOC.relative_to(ROOT)}"
    return "".join(out)


def _nearest_state(text: str, name_pattern: str, window: int = STATE_WINDOW) -> str:
    """`on`/`off` nearest to any mention of one mechanism, within `window`.

    Positional rather than phrase-matched so "the fan-out arm **ships on**" and
    "**off** by default, on for the fan-out" both read correctly, in either
    order. Raises if the mechanism is never named, or its name has no state word
    near it — a cell that mentions only one mechanism cannot pass.
    """
    hits = list(re.finditer(name_pattern, text, re.I))
    assert hits, (
        f"the overlap row never names {name_pattern!r}, so a reader cannot tell "
        f"what that mechanism ships as: {text!r}")
    best: tuple[int, str] | None = None
    for hit in hits:
        lo, hi = max(0, hit.start() - window), min(len(text), hit.end() + window)
        for state in re.finditer(r"\b(off|on)\b", text[lo:hi], re.I):
            at = lo + state.start()
            dist = min(abs(at - hit.start()), abs(at - hit.end()))
            if best is None or dist < best[0]:
                best = (dist, state.group(1).lower())
    assert best is not None, (
        f"no `on`/`off` state word sits within {window} characters of "
        f"{name_pattern!r}, so the row does not say what it ships as: {text!r}")
    return best[1]


def _overlap_states(where_cell: str) -> dict[str, str]:
    """{'dispatch': ..., 'fan-out': ...}, each from its own half of the cell."""
    return {
        "dispatch": _nearest_state(where_cell, r"\bdispatch\b"),
        "fan-out": _nearest_state(where_cell, r"fan[-\s]?out"),
    }


def _breadth_numbers(section: str) -> list[tuple[int, str]]:
    """(number, distance) for the integer nearest each `max_concurrency` mention.

    `§`-references, `#`-item ids and years are stripped first: they are section
    and date markers, not breadth numbers, and the §9 paragraph is full of them.
    """
    out: list[tuple[int, str]] = []
    for hit in re.finditer(MAX_CONCURRENCY_KEY, section):
        lo, hi = max(0, hit.start() - BREADTH_WINDOW), min(
            len(section), hit.end() + BREADTH_WINDOW)
        window = re.sub(r"(§|#|arXiv:)\s?\d[\d.]*", " ", section[lo:hi])
        window = re.sub(r"\b(19|20)\d\d\b", " ", window)
        best: tuple[int, str] | None = None
        for num in re.finditer(r"\b(\d{1,3})\b", window):
            at = lo + num.start()
            dist = min(abs(at - hit.start()), abs(at - hit.end()))
            if best is None or dist < best[0]:
                best = (int(num.group(1)), "near")
        if best is not None:
            out.append(best)
    return out


# ── detectors (each raises AssertionError; the controls feed them old prose) ─

def _check_overlap_rule(doc: str) -> None:
    """§5's row admits the P8 arm and does not call `readOnlyHint` the predicate.

    Clause 1 of #4434.
    """
    cells = _row_cells(doc, OVERLAP_ROW_KEY)
    rule = cells[1]
    assert "`readOnlyHint`" in rule, (
        f"the overlap row no longer mentions `readOnlyHint` at all: {rule!r}")
    assert "Task" in rule, (
        f"the overlap row still reads as annotation-only — no `Task` in the "
        f"predicate: {rule!r}")
    assert "parallel_safe" in rule, (
        f"the overlap row does not name the `parallel_safe` arm (P8), so it "
        f"excludes the one batch shape that overlaps today: {rule!r}")
    assert re.search(r"\bor\b", rule, re.I), (
        f"the overlap row gives one predicate only: {rule!r}")
    assert re.search(r"\bfresh\b|task_id", rule, re.I), (
        "the overlap row does not say a `task_id` resume is excluded, though "
        f"`_is_parallel_safe_task` refuses one: {rule!r}")
    assert "parallel_safe" in doc, (
        f"{DOC.relative_to(ROOT)} never names `parallel_safe` anywhere")


def _check_overlap_shipped_states(doc: str) -> None:
    """The row states both shipped states, matching the shipped config.

    Clause 2 of #4434.
    """
    cells = _row_cells(doc, OVERLAP_ROW_KEY)
    stated = _overlap_states(cells[-1])
    expected = {
        "dispatch": "on" if _general_dispatch_ships_on() else "off",
        "fan-out": "on" if _shipped_parallel_safe_profiles() else "off",
    }
    assert stated == expected, (
        f"the overlap row says {stated} but the shipped config says "
        f"{expected} (harness.parallel_tool_calls.enabled="
        f"{_general_dispatch_ships_on()}, parallel_safe profiles="
        f"{_shipped_parallel_safe_profiles()}). Move the prose to match the "
        f"config; do not revert this test.")


def _check_no_stale_task_serialization(doc: str) -> None:
    """No sentence claims a batch containing `Task` runs sequentially.

    Clause 3 of #4434. Whole-file, because either claim anywhere reads as true to
    the next reader — including in the changelog, where #4434's own verification
    grep counts a quoted copy of it.
    """
    for pattern in STALE_TASK_SERIALIZATION_RES:
        hit = pattern.search(doc)
        if hit is not None:
            raise AssertionError(
                f"{DOC.relative_to(ROOT)} still asserts pre-P8 overlap with "
                f"{pattern.pattern!r}, matched {hit.group(0)!r}")


def _check_breadth_cap(doc: str) -> None:
    """§9 names both caps, and its breadth equals the shipped number.

    Clause 4 of #4434: depth caps nesting, breadth is `max_concurrency`, scoped
    to `parallel_safe` profiles only.
    """
    section = _section(doc, SUBAGENT_HEADING)
    paragraphs = [p for p in section.split("\n\n") if p.strip()]
    both_caps = [p for p in paragraphs
                 if "depth" in p.lower() and MAX_CONCURRENCY_KEY in p]
    assert both_caps, (
        f"{SUBAGENT_HEADING} has no paragraph naming both caps together (the "
        f"depth cap and the {MAX_CONCURRENCY_KEY} breadth cap), so a reader "
        f"cannot see that depth 1 does not bound live children")
    joined = "\n".join(both_caps)
    assert "parallel_safe" in joined, (
        f"the breadth cap is stated without its scope: the fan-out overlaps "
        f"only `parallel_safe` profiles, not every `Task`: {joined!r}")
    for name in _shipped_parallel_safe_profiles():
        assert f"`{name}`" in joined, (
            f"`{name}` ships parallel_safe: true but the §9 breadth paragraph "
            f"never names it, so the scoping prose has not followed the config")
    stated = _breadth_numbers(section)
    assert stated, (
        f"{SUBAGENT_HEADING} never states a number beside "
        f"{MAX_CONCURRENCY_KEY}, so the bound it hands a reader is unmeasurable")
    shipped = _shipped_max_concurrency()
    # §5's row and the changelog state the same cap; every place the page numbers
    # it has to move with the config, or one of them is a stale claim again.
    for number, _ in stated + _breadth_numbers(doc):
        assert number == shipped, (
            f"the doc states a breadth of {number} but the shipped "
            f"harness.parallel_tool_calls.max_concurrency is {shipped}")


def _check_row_reads_more_than_the_wire_hint(doc: str) -> None:
    """§5's summary must not call the overlap row annotation-only.

    The row additionally reads `subagents.<type>.parallel_safe` from the config,
    so the sentence under the table that says the rows after the first three "read
    the hint as it crossed the wire" is false about the one row that decides
    overlap — and a reader who trusts it will look for the fan-out arm in the
    annotations table and conclude it does not exist.
    """
    cells = _row_cells(doc, OVERLAP_ROW_KEY)
    assert "parallel_safe" in cells[2], (
        "the overlap row's Reads cell no longer names the config key it reads, "
        f"which the summary sentence below the table now depends on: {cells[2]!r}")
    tail = doc[doc.index("| Effect-ledgered"):][:600]
    assert "parallel_safe" in tail, (
        "the prose under the §5 table still implies every wire-hint row is "
        "annotation-only; say the overlap row also reads the shipped config")


def _check_unmeasured_cost_claim(doc: str) -> None:
    """§9 still says the per-level cost is unmeasured (clause 4, second half)."""
    section = _section(doc, SUBAGENT_HEADING)
    assert UNMEASURED_CLAIM in section, (
        f"{SUBAGENT_HEADING} dropped {UNMEASURED_CLAIM!r}. The depth-1 ruling "
        f"is structural; without that sentence it reads as a measured cap.")


# ── the live doc ────────────────────────────────────────────────────────────

def test_overlap_row_admits_a_fresh_parallel_safe_task() -> None:
    """§5's predicate is readOnlyHint OR a fresh parallel-safe `Task` (clause 1)."""
    _check_overlap_rule(DOC_TEXT)


def test_overlap_row_states_both_shipped_states() -> None:
    """Dispatch off and fan-out on, each read from the shipped config (clause 2)."""
    _check_overlap_shipped_states(DOC_TEXT)


def test_page_makes_no_sequential_task_or_one_child_claim() -> None:
    """No 'one live child at a time', quoted or not, and no sequential-Task claim."""
    _check_no_stale_task_serialization(DOC_TEXT)


def test_subagents_section_breadth_is_the_shipped_max_concurrency() -> None:
    """Both caps named, breadth = shipped max_concurrency, scoped (clause 4)."""
    _check_breadth_cap(DOC_TEXT)


def test_subagents_section_keeps_the_unmeasured_cost_claim() -> None:
    """The ruling stays visibly structural next to the new breadth number."""
    _check_unmeasured_cost_claim(DOC_TEXT)


def test_overlap_row_reads_more_than_the_wire_hint() -> None:
    """§5's summary of its own table credits the config key the row reads."""
    _check_row_reads_more_than_the_wire_hint(DOC_TEXT)


# ── controls: every detector must fail on the prose this fix replaced ───────

def test_the_overlap_rule_detector_fires_on_the_pre_fix_row() -> None:
    with pytest.raises(AssertionError):
        _check_overlap_rule(PRE_FIX_OVERLAP_ROW)


def test_the_shipped_state_detector_fires_on_the_pre_fix_row() -> None:
    """The single `(**ships off**)` stamp is not a state for the fan-out arm."""
    with pytest.raises(AssertionError):
        _check_overlap_shipped_states(PRE_FIX_OVERLAP_ROW)


def test_the_serialization_detector_fires_on_the_pre_fix_section() -> None:
    with pytest.raises(AssertionError):
        _check_no_stale_task_serialization(PRE_FIX_SUBAGENT_SECTION)


def test_the_serialization_detector_fires_on_a_quoted_changelog_copy() -> None:
    """The occurrence left on the page by the prose fix, and the one #4434 greps."""
    with pytest.raises(AssertionError):
        _check_no_stale_task_serialization(PRE_FIX_CHANGELOG_QUOTE)


def test_the_breadth_detector_fires_on_the_pre_fix_section() -> None:
    with pytest.raises(AssertionError):
        _check_breadth_cap(PRE_FIX_SUBAGENT_SECTION)


def test_the_breadth_detector_fires_on_a_stale_number() -> None:
    """A page that kept the sentence but not the number must fail too."""
    shipped = _shipped_max_concurrency()
    stale = DOC_TEXT.replace(
        f"{MAX_CONCURRENCY_KEY}` ({shipped})",
        f"{MAX_CONCURRENCY_KEY}` ({shipped + 1})")
    stale = stale.replace(
        f"{MAX_CONCURRENCY_KEY}` {shipped}", f"{MAX_CONCURRENCY_KEY}` {shipped + 1}")
    assert stale != DOC_TEXT, (
        "no breadth number on the page is in a form this control can edit, so it "
        "would prove nothing")
    with pytest.raises(AssertionError):
        _check_breadth_cap(stale)


def test_the_unmeasured_claim_detector_fires_on_a_rewrite_that_drops_it() -> None:
    """A mutation control, not a pre-fix one: the pre-fix §9 *did* say this.

    So the detector that keeps the claim can only be proved by taking it out of
    the shipped page and showing the check notices.
    """
    stripped = DOC_TEXT.replace(UNMEASURED_CLAIM, "The ruling is structural.")
    assert stripped != DOC_TEXT, "the claim vanished from the page"
    with pytest.raises(AssertionError):
        _check_unmeasured_cost_claim(stripped)


def test_the_wire_hint_detector_fires_on_the_pre_fix_summary() -> None:
    """The sentence under the table, as it read with the fan-out arm unknown."""
    with pytest.raises(AssertionError):
        _check_row_reads_more_than_the_wire_hint(PRE_FIX_TABLE_TAIL)


def test_the_true_shipped_off_prose_still_passes_the_bans() -> None:
    """The bans correct the claim, they do not forbid saying dispatch is off."""
    _check_no_stale_task_serialization(TRUE_PROSE_ABOUT_OFF_DISPATCH)
