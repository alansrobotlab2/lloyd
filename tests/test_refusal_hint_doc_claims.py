"""Doc-claim test for the refusal paragraph's published figures (#3774).

Two files quote the instrument that measures whether the refusal paragraph in the
system prompt stops a refusal becoming a detour: `CLAUDE.md` in the bullet that
ships the paragraph, and `architecture/harness.md` in the section named for it.
Both quoted the 2026-09-30 baseline — an effect rate printed as `30%` and a raw
count — as the number a reader should re-run and compare against. That baseline
cannot be reproduced: its targets were mined from the refusal's own prose, so every
`tool_sandbox` and `grant` refusal came back with an empty target set and its `0.0`
was an unmeasured pass, and the September corpus it counted is no longer in
readable storage. Nothing pinned any of it — `git grep` for the figures found them
in the two docs and nowhere else — so doc and instrument disagreed with no node
reddening (#3565's fix commit booked that debt at `eval/refusal_detour_baseline.py:38-40`
and it went unpaid).

What this file holds, therefore, is the **shape** of a refusal figure, never its
values. Every rate printed in a refusal passage must have a measurable count on the
same line, because a rate without a denominator is exactly how the retired baseline
passed off 0 measured rows as perfect compliance — and the counts themselves move as
the session store ages (the same command over the same window printed `safety`
393 refusals / 244 measurable, then 394 / 244, then 397 / 247), so a node that pinned
them would go red on a busy day and teach its reader to ignore it.

Run just this file:

    cd ~/lloyd && .venvs/lloyd/bin/python -m pytest -q tests/test_refusal_hint_doc_claims.py

The wording of the prompt paragraph itself is pinned by `tests/test_refusal_hint.py`;
the instrument and its measurable-annotation rules by `tests/test_refusal_detour_baseline.py`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CLAUDE = REPO / "CLAUDE.md"
HARNESS = REPO / "architecture" / "harness.md"

#: The retired effect rate, and the retired count-to-denominator reading, as each
#: file worded them. The whole point of this file is that neither string may appear
#: as a live figure again, so they are kept here rather than in the docs.
RETIRED_STRINGS = (
    "23 of 400",            # CLAUDE.md:128 at HEAD~ of this change
    "30% of the hard-deny",  # CLAUDE.md:129 at HEAD~ of this change
)

#: The sentence each file used to send a reader off to re-run and compare. A round
#: that retires the numbers but leaves these standing has told the reader to do
#: something the corpus can no longer answer.
RERUN_AS_JUDGEMENT = (
    "Re-run it to see whether the paragraph moved that",
    "judged by that number moving on a re-run",
)

# ---------------------------------------------------------------------------
# the rate shape, and the refusal passages it is checked inside
# ---------------------------------------------------------------------------

# A printed compliance rate. Three shapes, and each one is here for a measured
# reason: two-or-more decimals is how the instrument prints `round(rate, 3)` — 0.602,
# 0.071, 0.498; a parenthesised decimal catches its one-decimal and zero readings,
# `(0.0)` and `(1.0)`, which is exactly how a dead arm's artifact reached the page;
# and a percentage is how the retired baseline phrased its effect rate, `same-effect
# (30%)`. Bare integers are not rates — the refusal and measurable COUNTS are integers,
# and a line whose numbers are counts already carries a denominator. A decimal in
# prose that is neither parenthesised nor two-decimal (`the 0.0 printed against a zero
# denominator`) is a sentence about the artifact, not a printed figure, and is out of
# scope by design.
RATE = re.compile(
    r"(?<![\d.])0\.\d{2,}(?![\d.])"          # 0.602, 0.071, 0.498
    r"|\(\s*(?:0|1)\.\d{1,}\s*\)"            # (0.0), (1.0) — the printed reading
    r"|\b\d{1,3}(?:\.\d)?%"                  # 30%, 0.0%
)
# The instrument's own annotation: a count followed by the word measurable.
MEASURABLE = re.compile(r"\b\d[\d,]*\s+measurable\b")
# Or an explicit numerator-over-that-denominator form, which names the denominator
# beside the rate: `147/244 (0.602)`.
OVER = re.compile(r"\b\d[\d,]*\s*/\s*\d[\d,]*\b")


def bare_rate_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Lines that print a compliance rate with no measurable count and no denominator.

    This is the shape that hid the retired baseline's dead arms: `same-effect 0.0`
    printed against 0 measurable rows reads as perfect compliance to anyone who does
    not also read the count.
    """
    out: list[tuple[int, str]] = []
    for n, line in enumerate(lines, 1):
        if not RATE.search(line):
            continue
        if MEASURABLE.search(line) or OVER.search(line):
            continue
        out.append((n, line))
    return out


REFUSAL_CONTEXT = re.compile(r"refusal|refusals|denied|detour", re.I)
NEW_BLOCK = re.compile(r"^\s*(?:#{1,4}\s|[-*]\s|\d+\.\s)")


def claude_passage_from(lines: list[str]) -> list[str]:
    """Every line inside the refusal-paragraph passage of a CLAUDE.md-shaped file.

    CLAUDE.md is a bullet list, so the passage is the wrapped paragraph containing a
    refusal word: the maximal run of non-blank lines around it that does not open a
    new bullet or heading. That keeps the `LLOYD_BENCH` config table two lines below
    it out of scope — its 80%-of-calls numbers are a different claim and belong to
    `tests/test_bench_refusal_sandbox.py`.
    """
    keep: set[int] = set()
    for i, line in enumerate(lines):
        if not REFUSAL_CONTEXT.search(line):
            continue
        if NEW_BLOCK.match(line):
            continue  # a bullet that only mentions a refusal; its own paragraph
        lo = i
        while lo > 0 and lines[lo - 1].strip() and not NEW_BLOCK.match(lines[lo - 1]):
            lo -= 1
        hi = i
        while hi + 1 < len(lines) and lines[hi + 1].strip() and not NEW_BLOCK.match(lines[hi + 1]):
            hi += 1
        keep.update(range(lo, hi + 1))
    assert keep, "no refusal context anywhere in these bytes — did the paragraph move?"
    return [lines[i] for i in sorted(keep)]


def claude_refusal_lines() -> list[str]:
    """CLAUDE.md's refusal passage, as shipped."""
    return claude_passage_from(CLAUDE.read_text().splitlines())


HARNESS_SECTION = re.compile(r"^## The refusal paragraph\b.*$")


def harness_section_from(lines: list[str]) -> list[str]:
    """The ``## The refusal paragraph`` section of a harness.md-shaped file.

    The whole section, heading to next heading: the retired baseline, the instrument's
    rules and the published per-guard block are one argument, and a rate in any of
    them is a refusal figure.
    """
    start = next((i for i, line in enumerate(lines) if HARNESS_SECTION.match(line)), None)
    assert start is not None, "no '## The refusal paragraph' section in these bytes"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def harness_refusal_lines() -> list[str]:
    """harness.md's refusal section, as shipped."""
    return harness_section_from(HARNESS.read_text().splitlines())


def table_rows(lines: list[str]) -> list[list[str]]:
    """Cell contents of each markdown table row, separator row dropped."""
    rows = []
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


# ---------------------------------------------------------------------------
# clause 1 — CLAUDE.md retires the baseline and says why
# ---------------------------------------------------------------------------

def test_claudes_refusal_passage_retires_the_baseline_and_names_its_cause() -> None:
    """CLAUDE.md prints neither retired figure, marks the 2026-09-30 baseline RETIRED
    with both causes, and names the instrument a reader would actually run.

    The corpus cause is a number this node checks because it is checkable and will
    not drift: 38 of the 2026-09 `.json` sessions survive in the session store and
    the oldest is `20260922_201704`. Both are stated in the passage, and the scanner
    globs `*.json` only — 8,324 `.json.gz` sessions are invisible to it — which is
    what makes "re-run it and compare" impossible rather than merely stale.
    """
    passage = "\n".join(claude_refusal_lines())
    for retired in RETIRED_STRINGS:
        assert retired not in passage, f"{retired!r} is still printed in CLAUDE.md:\n{passage}"
    assert "RETIRED" in passage, "the 2026-09-30 baseline is not marked retired"
    assert "prose" in passage, "the retired-with-cause line must name the prose-mined targets"
    assert "eval/refusal_detour_baseline.py" in passage
    assert "38" in passage and "20260922_201704" in passage, (
        "the corpus cause has to be stated as a count and an oldest session id, or a "
        "reader cannot tell aged-out from never-measured"
    )
    assert "gz" in passage, "the passage must say the gzipped sessions are not scanned"


# ---------------------------------------------------------------------------
# clause 2 — harness.md keeps the old numbers inside the retirement, with the defect
# ---------------------------------------------------------------------------

def test_harness_keeps_the_old_numbers_only_inside_the_retirement_and_names_the_defect() -> None:
    """harness.md's refusal section retires the baseline with its measurement defect.

    The defect is what makes the retirement non-negotiable: 306/306 `tool_sandbox`
    refusals and every `grant` refusal came back with an empty target set, so both
    arms were structurally dead for them and the printed `0.0` was an unmeasured
    pass, not compliance. And the old figures may survive only there — a line that
    still prints them outside the retirement sentence is quoting them as live.
    """
    section = harness_refusal_lines()
    joined = "\n".join(section)
    assert "RETIRED" in joined, "the 2026-09-30 baseline is not marked retired"
    assert "306/306" in joined, "the empty-target-set defect must be counted, not gestured at"
    assert "tool_sandbox" in joined and "grant" in joined
    assert "empty target set" in joined
    assert "unmeasured pass" in joined, (
        "the retirement has to say the old 0.0 was unmeasured, not that it was low"
    )
    # Passage-granular, because the section is hard-wrapped and a figure can land on
    # any of its lines: the old numbers may appear ONLY between the retirement marker
    # and the marker that opens the current reading. A figure outside that window is a
    # figure quoted as live.
    retired_at = joined.find("RETIRED")
    current_at = joined.find("What the fixed instrument prints")
    assert retired_at >= 0 and current_at > retired_at, (
        "the retirement and the current reading must be adjacent passages of one section"
    )
    retirement = joined[retired_at:current_at]
    assert "23 same-effect" in retirement and "400" in retirement, (
        "the retired reading is no longer quotable inside the retirement — its numbers "
        "belong on the page as history, they just may not stand alone as figures"
    )
    outside = joined[:retired_at] + joined[current_at:]
    for marker in ("23 same-effect", "3,979", " 400 ", "sudo` 24"):
        assert marker not in outside, (
            f"{marker!r} is printed outside the retirement passage"
        )


# ---------------------------------------------------------------------------
# clause 3 — the published block: every rate beside its measurable count
# ---------------------------------------------------------------------------

def test_the_published_block_prints_every_rate_beside_its_measurable_count() -> None:
    """The section publishes the guard-stratified reading, one row per guard, each rate
    with its denominator, over a named window and dated scans.

    Which rows are published is checked; their values are not, and that is the
    deliberate half of this node. The session store keeps about 30 days uncompressed
    and grows hourly, so `refusals` and `measurable` move on their own — the same
    command over the same window printed `safety` 393 refusals / 244 measurable, then
    394 / 244, then 397 / 247 — and the section says so in the block itself. What must
    never regress is the shape: a guard's rate printed without the count it was
    computed over is the retired baseline's failure mode returning.
    """
    section = harness_refusal_lines()
    joined = "\n".join(section)
    rows = table_rows(section)
    published: dict[str, list[str]] = {}
    for cells in rows:
        for guard in ("grant", "safety", "tool_sandbox", "all"):
            if f"`{guard}`" in cells[0]:
                published.setdefault(guard, []).append(" | ".join(cells))
    for guard in ("all", "grant", "safety", "tool_sandbox"):
        assert guard in published, f"no published row for the {guard} guard"
        for row in published[guard]:
            assert MEASURABLE.search(row), f"row prints no measurable count: {row}"
            assert OVER.search(row) and RATE.search(row), (
                f"row prints no rate over its denominator: {row}"
            )
    assert "--since 2026-10-01" in joined, "the window the block was measured over is not named"
    assert "window 3" in joined or "window of 3" in joined, "the retry window is not named"
    assert "2026-11-04" in joined and "2026-11-05" in joined, (
        "each scan in the block has to be dated, or the drift reads as a contradiction"
    )
    assert "counts move" in joined, "the block must say its own counts move as the store ages"
    assert "not a threshold" in joined, "a dated reading must say it is not a target to hit"
    assert "tests/test_refusal_hint_doc_claims.py" in joined, (
        "the block must name the node that holds its shape"
    )


# ---------------------------------------------------------------------------
# clause 4 — no re-run-as-judgement instruction, and the shape rule has teeth
# ---------------------------------------------------------------------------

def test_neither_file_tells_a_reader_to_judge_the_paragraph_by_the_retired_number() -> None:
    """Neither doc may instruct a reader to compare a fresh run against the retired
    baseline. The corpus for that comparison is gone: the store holds ~30 days
    uncompressed, 38 September `.json` sessions survive, the oldest `20260922_201704`,
    and the scanner never opens the gzipped majority — so the instruction is not
    merely stale, it is unexecutable.
    """
    for name, text in (
        ("CLAUDE.md", "\n".join(claude_refusal_lines())),
        ("architecture/harness.md", "\n".join(harness_refusal_lines())),
    ):
        for phrase in RERUN_AS_JUDGEMENT:
            assert phrase not in text, f"{name} still tells the reader: {phrase!r}"


def test_the_shape_rule_holds_over_both_refusal_passages() -> None:
    """No line in either passage prints a compliance rate with no denominator.

    Counted, not assumed: the rule has to actually inspect rate-bearing lines for
    this to mean anything, so the node asserts it examined at least the eight rows of
    the published block plus their prose.
    """
    harness = harness_refusal_lines()
    claude = claude_refusal_lines()
    checked = sum(1 for line in harness + claude if RATE.search(line))
    assert checked >= 8, (
        f"the rule inspected {checked} rate-bearing refusal line(s); the passages went "
        "vacuous and a bare rate could return unnoticed"
    )
    assert bare_rate_lines(harness) == [], [line for _, line in bare_rate_lines(harness)]
    assert bare_rate_lines(claude) == [], [line for _, line in bare_rate_lines(claude)]


# ---------------------------------------------------------------------------
# Negative control — this is what makes the shape assertion above evidence
# ---------------------------------------------------------------------------

# The pre-fix passage, quoted verbatim from the two files as they stood before this
# change (`git log -S` for either string names `3baf94ed` as the commit that wrote it).
# It is stored here rather than read from `HEAD` on purpose and by measurement: `HEAD`
# advances the moment this change commits, so a control whose red leg is "`HEAD` is the
# old text" passes for exactly one commit and then reddens the suite forever — the first
# run of this file after committing did exactly that. A pinned sha rots the same way on
# the next rebase or squash, and a shallow foreign clone resolves neither. The literals
# ARE the artifact, and the node below proves they are the real retired wording by
# pickaxe-searching the file's own history for them, so they cannot be text invented to
# trip the regex.
PREFIX_CLAUDE_PASSAGE = """\
same rule; a refusal naming a remedy is the way forward. Baseline before the paragraph
(`eval/refusal_detour_baseline.py`, 2026-09-30): 23 of 400 stored refusals were followed
within three calls by a same-effect retry, 30% of the hard-deny refusals. Re-run it to see
whether the paragraph moved that."""
PREFIX_HARNESS_SECTION = """\
the call carries the refused verb or a known alternative for it (same-effect). The
2026-09-30 baseline, before the paragraph: 400 refusals in 3,979 sessions, 306 of them the
bench sandbox (0 retries), 77 the hard-deny table — of which 25 were same-target and **23
same-effect (30%)**; top labels `sudo` 24, `rm -rf on root/home/system path` 13. Retries
were Bash 16, Read 4, Edit 3, Write 2. The paragraph is judged by that number moving on a
re-run, not by its wording. Pins: `tests/test_refusal_hint.py`."""

PREFIX_CONTROL = (
    ("CLAUDE.md", PREFIX_CLAUDE_PASSAGE, "30% of the hard-deny"),
    ("architecture/harness.md", PREFIX_HARNESS_SECTION, "same-effect (30%)"),
)


def test_the_shape_rule_reddens_on_a_bare_rate_restored_to_either_passage() -> None:
    """Negative control: the same rule must redden on the pre-fix wording, in memory.

    Three things have to be true for the green shape node to be evidence rather than a
    vacuous read, and this node checks all three over one shipped rule. (1) Each
    passage, spliced in memory with one bare rate restored to it, is flagged — the
    clause's own construction. (2) Each pre-fix passage as quoted is flagged on the
    retired effect rate: those are the bytes that shipped a dead arm's `0.0` as
    compliance and a `30%` over the wrong denominator. (3) Each quoted passage is
    genuinely the retired text — pickaxe finds it in that file's history — and is not
    present in the shipped file, so restoring the old wording is caught by the node that
    retired it.

    If this node ever passes while the shape node also passes, the rule has gone blind,
    not the documents gone clean.
    """
    splices = (
        (claude_refusal_lines, 2, "within three calls by a same-effect retry, 30% of the "
                                  "hard-deny refusals, judged by that number moving."),
        (harness_refusal_lines, 4, "**23 same-effect (30%)**, judged by that number "
                                   "moving on a re-run"),
    )
    for reader, at, line in splices:
        passage = reader()
        passage.insert(at, line)
        assert any("30%" in text for _, text in bare_rate_lines(passage)), (
            "restoring a bare rate to a refusal passage did not redden the rule, so the "
            "rule is not reading what the clause names it reads"
        )

    for name, prefix, probe in PREFIX_CONTROL:
        shipped = (CLAUDE if name == "CLAUDE.md" else HARNESS).read_text()
        flagged = bare_rate_lines(prefix.splitlines())
        assert any(probe in line for _, line in flagged), (
            f"the rule let the pre-fix {name} passage through: {flagged}"
        )
        hit = subprocess.run(
            ["git", "-C", str(REPO), "log", "-S", probe, "--format=%h", "--", name],
            capture_output=True, text=True, check=True,
        )
        assert hit.stdout.strip(), (
            f"{probe!r} never occurs in {name}'s history, so the control text for {name} "
            "is not the retired wording but something written to trip the regex"
        )
        assert prefix not in shipped, f"the pre-fix {name} passage is back on the page"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
