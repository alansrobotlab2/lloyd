"""Doc-claim test for the refusal paragraph's published figures (#3774).

Two files quote the instrument that measures whether the refusal paragraph in the
system prompt stops a refusal becoming a detour: `CLAUDE.md` in the bullet that
ships the paragraph, and `architecture/harness.md` in the section named for it.
Both quoted the 2026-09-30 baseline — an effect rate printed as `30%` and a raw
count — as the number a reader should re-run and compare against. That baseline
cannot be reproduced: its targets were mined from the refusal's own prose, so every
`tool_sandbox` and `grant` refusal came back with an empty target set and its `0.0`
was an unmeasured pass, and the instrument that measures it now resolves operands the
prose miner could not, so a re-run answers a different question. Nothing pinned any of
it — `git grep` for the figures found them
in the two docs and nowhere else — so doc and instrument disagreed with no node
reddening (#3565's fix commit booked that debt at `eval/refusal_detour_baseline.py:38-40`
and it went unpaid).

That last clause of the reason was itself a false claim, and #4277 is the file that keeps
it from coming back. The two docs said the September corpus "has aged out of readable
storage"; it had not — 4,013 of its sessions are in the store as `.json.gz`, which the
retention sweep never deletes, and what hid them was the scanner's own `*.json` glob, now
crossed by `--include-gz`. A retirement whose second reason is a myth is a retirement a
future round will 'correct' back into a re-run, so `FALSE_CORPUS_CLAIMS` below names the
myth's wording and the nodes below demand the true cause in its place — and demand that
the method-matched before/after pair the corrected cause makes possible is published with
its own labels, because the reason the archive matters is exactly that pair.

What this file holds, therefore, is the **shape** of a refusal figure, never its
values. Every rate printed in a refusal passage must have a measurable count on the
same line, because a rate without a denominator is exactly how the retired baseline
passed off 0 measured rows as perfect compliance — and the counts themselves move as
the session store ages (the same command over the same window printed `safety`
393 refusals / 244 measurable, then 394 / 244, then 397 / 247), so a node that pinned
them would go red on a busy day and teach its reader to ignore it.

One block on that page is the exception, and it is an exception because it is no longer
a reading of a live store: harness.md's `2026-11-04 corpus` block is a scan replayed over
a corpus that no longer moves, whose rows are committed at
`eval/measurements/refusal-detours/2026-11-04.rows.jsonl` under a filename that names its
own scan date. Those bytes cannot age, so the four rows built from them are pinned: the
last nodes here re-aggregate the file with the instrument's own arithmetic and demand the
page print what the bytes say, and the same nodes demand that the block SAY which file it
came from — a table with a command and a scan date but no committed witness is exactly how
#3565's figures became unfalsifiable when the tmpfs reboot deleted the scan behind them.
The live-store block beside it stays unpinned on values, and must: no rows file was ever
written for it, so there is nothing to recompute it from, and the citation that covers the
first block must not be allowed to reach it.

A third block, added by #4277, splits the two cases: it is a reading of the store, but of
the *archived* store over a closed window, and an archived session is never rewritten or
deleted, so the September figures cannot age. Its rates are therefore pinned like the
2026-11-04 block's — by value, not just by shape — and the post-#3774 row printed under it
is pinned like the live block, because that store is still growing. Which of the two a row
belongs to is decided by the label the row itself carries (`method-matched` against `NOT
method-matched`), so a row cannot escape its own doctrine by moving.

Run just this file:

    cd ~/lloyd && .venvs/lloyd/bin/python -m pytest -q tests/test_refusal_hint_doc_claims.py

The wording of the prompt paragraph itself is pinned by `tests/test_refusal_hint.py`;
the instrument and its measurable-annotation rules by `tests/test_refusal_detour_baseline.py`.
"""

from __future__ import annotations

import json
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

#: The false retirement cause, in the exact wording each file carried until #4277. The
#: corpus had not aged out: 4,013 sessions from 2026-09 sit in the store as `.json.gz`
#: (oldest `20260909_153801_autonomy_2cf3`), `scripts/groundskeeper/retention-sweep.py`
#: gzips an aged transcript and never deletes it, and the thing that hid them was the
#: scanner's `*.json` glob — since #4277, `--include-gz`. Both files now say that; these
#: strings are kept out of the docs so a 'helpful' rewrite cannot restore the myth in
#: the present perfect. The corrected sentences say the corpus `had not aged out`, which
#: is why the check is on this wording and not on the words 'aged out'.
FALSE_CORPUS_CLAIMS = (
    "has aged out of readable storage",
    "the corpus it counted has aged out",
    "no longer in readable storage",
)

#: The two anchors of the block #4277 added to harness.md: the method-matched pair across
#: the paragraph's ship commit, and the live-store row printed under it. A node that
#: checks a label has to know which block the label is supposed to be inside, and the
#: section is long enough that searching the whole page would let a row labelled one way
#: answer a check written for the other.
PAIR_BLOCK_START = "**The pair the paragraph was owed"
POST_ROW_BLOCK_START = "**The same instrument on the post-#3774 store"

#: Framings that would turn a pair whose after-side is *worse* into evidence of success.
#: The rate is a detour rate: higher is worse, so the block's own arithmetic is pinned by
#: `test_the_pair_block_says_the_rate_did_not_improve_and_never_calls_it_a_win` and these
#: phrases are the prose that would contradict it.
WIN_FRAMINGS = (
    "the paragraph worked",
    "compliance improved",
    "moved in the right direction",
    "the pair is a win",
    "detours fell",
    "a clear improvement",
)

#: Rate-bearing refusal lines the shape rule inspected before #4277 (measured over the
#: two files at `26814c9b`: 8 rows of the published block plus one prose line). The floor
#: in `test_the_shape_rule_holds_over_both_refusal_passages` is green by count alone, so
#: without this baseline a page could satisfy it by saying less, not more.
SHAPE_CHECKED_BEFORE_4277 = 9

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


def labelled_rows(lines: list[str], label: str, guard: str) -> list[list[str]]:
    """Table rows whose first cell carries both `label` and the quoted `guard`.

    A label belongs to one block: the method-matched pair and the NOT-method-matched row
    sit in the same section, and a check that reads the whole page cannot tell a missing
    label from a label on the wrong row — so callers pass a block, not the section.
    """
    return [cells for cells in table_rows(lines)
            if label in cells[0] and f"`{guard}`" in cells[0]]


def rates_over(row: list[str]) -> list[tuple[int, int]]:
    """Each `(numerator/denominator, 0.rate)` pair a row prints, left to right.

    The same-cell regex keeps the pairing honest: `83/136 (0.61)` yields (83, 136) with
    0.61 beside it, so a node can check the printed rate against its own denominator
    instead of trusting either half.
    """
    out: list[tuple[int, int]] = []
    for cell in row:
        m = re.search(r"\b(\d[\d,]*)\s*/\s*(\d[\d,]*)\b", cell)
        if m:
            out.append((int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))))
    return out


# ---------------------------------------------------------------------------
# clause 1 — CLAUDE.md retires the baseline and says why
# ---------------------------------------------------------------------------

def test_claudes_refusal_passage_retires_the_baseline_and_names_its_cause() -> None:
    """CLAUDE.md prints neither retired figure, marks the 2026-09-30 baseline RETIRED
    with the true causes, and names the instrument a reader would actually run.

    Three of these tokens used to certify the *false* corpus cause — `38`,
    `20260922_201704`, `gz` — and all three survive the correction, now as the dated
    plain-store count that was mistaken for a loss rather than as proof that September
    was gone: 38 plain `.json` sessions from 2026-09 were left when the claim was written
    on 2026-11-04 and 64 by 2026-11-21, oldest plain `20260922_201704`, while 4,013 of
    that month's sessions are `.json.gz` (oldest `20260909_153801_autonomy_2cf3`). The
    month count is a closed window, so it cannot drift; the plain count can, which is why
    the passage dates it. What the node newly demands is the mechanism — the sweep never
    deletes a `.gz`, and `--include-gz` reads it — because a reader who is told only that
    a corpus is thin will go looking for a backup.
    """
    passage = "\n".join(claude_refusal_lines())
    for retired in RETIRED_STRINGS:
        assert retired not in passage, f"{retired!r} is still printed in CLAUDE.md:\n{passage}"
    for claim in FALSE_CORPUS_CLAIMS:
        assert claim not in passage, f"the false corpus cause is back in CLAUDE.md: {claim!r}"
    assert "RETIRED" in passage, "the 2026-09-30 baseline is not marked retired"
    assert "prose" in passage, "the retired-with-cause line must name the prose-mined targets"
    assert "eval/refusal_detour_baseline.py" in passage
    assert "38" in passage and "20260922_201704" in passage, (
        "the corpus cause has to be stated as a count and an oldest session id, or a "
        "reader cannot tell aged-out from never-measured"
    )
    assert "gz" in passage, "the passage must say how the gzipped sessions are read"
    assert "4,013" in passage and "20260909_153801_autonomy_2cf3" in passage, (
        "the correction has to be as countable as the claim it replaces"
    )
    assert "--include-gz" in passage, "the passage must name the flag that reads the archive"
    assert "never deletes" in passage, (
        "without the sweep's non-deletion the archive is a backup, not readable storage"
    )
    assert "287 measurable" in passage and "306" in passage, (
        "the surviving reason has to be the one that is checkable: the same 306 "
        "tool_sandbox refusals now yield 287 measurable rows"
    )


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


# ---------------------------------------------------------------------------
# the committed witness: the first block cites bytes, and those bytes recompute
# ---------------------------------------------------------------------------

MEASUREMENTS_REL = "eval/measurements/refusal-detours"
MEASUREMENTS_DIR = REPO / "eval" / "measurements" / "refusal-detours"

#: A citation of a scan, in the spelling #3775 requires of anything that quotes one: a
#: dated rows file under the measurement dir, never a scratch path. `/tmp` is tmpfs on
#: this box, so the reboot at 2026-11-04T23:06Z deleted the only copy of the scan whose
#: figures #3565 had published, and the numbers outlived their evidence.
CITED_ROWS = re.compile(r"eval/measurements/refusal-detours/(\d{4}-\d{2}-\d{2})\.rows\.jsonl")

#: How the section prints sessions opened: `12,049 scanned`. A witness row is one
#: refusal and carries no session-count field, so this figure can only ever come from
#: the `.json` summary — which is a rendering, not the bytes, and is not committed.
SCANNED_FIGURE = re.compile(r"\d[\d,]{2,}\s+scanned\b")

#: The two words the second block's own row has to carry for a reader to know it has no
#: source to go and check.
LIVE_ROW_LABEL = "live-store reading"
NO_BYTES_LABEL = "no committed bytes"

#: The four first-block rows as the committed witness aggregates them: refusals,
#: measurable, same-target, same-effect. Stated as literals next to the recomputation
#: below on purpose — comparing the page only against the file would let a re-committed
#: scan and a re-written block drift together and stay green. The witness is named by
#: its scan date, so these bytes are frozen and a node may hold them.
EXPECTED_FIRST_BLOCK: dict[str, tuple[int, int, int, int]] = {
    "all": (450, 300, 148, 86),
    "grant": (14, 14, 1, 1),
    "safety": (393, 244, 147, 85),
    "tool_sandbox": (43, 42, 0, 0),
}


def _rows_of(path: Path) -> list[dict]:
    """Every row of a committed rows file, parsed; an unparseable line is a red test."""
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _per_guard(rows: list[dict]) -> dict[str, dict[str, int]]:
    """refusals / measurable / same_target / same_effect per guard, plus the `all` slice.

    The instrument's own `summarize()` arithmetic, transcribed: one row is one refusal,
    the two rates are counted over `measurable` rows only, and `all` is the aggregate
    over every row rather than a guard the file marks. Anything looser here re-derives
    the published figures a different way than the scan wrote them, which is the same
    defect one file over.
    """
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        for guard in ("all", r["guard"]):
            block = out.setdefault(guard, {"refusals": 0, "measurable": 0,
                                           "same_target": 0, "same_effect": 0})
            block["refusals"] += 1
            if r["measurable"]:
                block["measurable"] += 1
                block["same_target"] += int(bool(r["same_target"]))
                block["same_effect"] += int(bool(r["same_effect"]))
    return out


def _rate(n: int, d: int) -> str:
    """One rate cell exactly as the instrument prints it, UNMEASURED included."""
    return "UNMEASURED" if d == 0 else f"{n}/{d} ({round(n / d, 3)})"


def published_blocks(section: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """The published table split into the two scans: (first block, live-store block).

    The split is the live-store row itself, so the boundary is the thing clause 3 pins
    rather than an index someone has to keep correct as rows are added.
    """
    rows = table_rows(section)
    assert len(rows) >= 2, f"the published block has {len(rows)} rows including its header"
    data = rows[1:]
    split = next((i for i, cells in enumerate(data) if LIVE_ROW_LABEL in cells[0]), None)
    assert split is not None, (
        f"no row of the published table carries {LIVE_ROW_LABEL!r}, so nothing marks "
        "where the witnessed scan stops and the reading with no bytes begins"
    )
    assert split > 0, "the live-store row is the first row, so no block is left to witness"
    return data[:split], data[split:]


def _tracked_measurement_rows() -> list[str]:
    """What `git ls-files` lists under the measurement dir, or a red line.

    Read out of git rather than off the filesystem because the two states this clause
    distinguishes are indistinguishable there: an untracked leftover in that directory
    looks exactly like a committed witness to a reader who runs `ls` (#3775).
    """
    proc = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", MEASUREMENTS_REL],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"`git ls-files -- {MEASUREMENTS_REL}` exited {proc.returncode}: "
                    f"{proc.stderr.strip()}. The witness-citation clause is NOT pinned "
                    "by this run.")
    return sorted(proc.stdout.split())


def _wc_l_c(path: Path) -> tuple[int, int]:
    """(lines, bytes) as `wc -l -c` reports them — the figures the citation must print."""
    proc = subprocess.run(["wc", "-l", "-c", str(path)], capture_output=True, text=True,
                          check=True)
    lines, bytes_ = proc.stdout.split()[:2]
    return int(lines), int(bytes_)


def _cited_witness(section: list[str]) -> tuple[Path, str]:
    """The one rows file the section cites, with the sentence that cites it.

    Read out of the page rather than assumed, because the clause is about what the page
    says. Exactly one dated rows file may be named: the second block has no witness, so
    a second citation there would be a citation of bytes that do not exist.
    """
    joined = "\n".join(section)
    hits = list(CITED_ROWS.finditer(joined))
    assert hits, (
        "the section names no dated rows file under eval/measurements/refusal-detours, "
        "so its guard-stratified table has no reachable source: a command and a scan "
        "date are not bytes anyone can re-aggregate"
    )
    dates = {m.group(1) for m in hits}
    assert len(dates) == 1, (
        f"the section cites {sorted(dates)}; the block that has committed bytes is one "
        "scan, and a second citation can only be a claim about the live reading"
    )
    m = hits[0]
    tail = joined[m.end():m.end() + 400]
    stop = tail.find(". ")
    citation = joined[m.start():m.end() + (len(tail) if stop < 0 else stop + 1)]
    return MEASUREMENTS_DIR / f"{m.group(1)}.rows.jsonl", citation


def _check_witness_citation(section: list[str]) -> Path:
    """Clauses 1 and 2 over the given section bytes; returns the witness certified.

    A function of the bytes rather than the body of the node, so the control at the end
    of this file can hand the same rules a spliced section and prove the shipped page is
    what is passing them.
    """
    path, citation = _cited_witness(section)
    assert path.is_file(), (
        f"the section cites {path.name}, which is not in {MEASUREMENTS_REL} — a citation "
        "to bytes nobody can open is worse than none, because it closes the report that "
        "would have said the witness was missing"
    )
    rel = str(path.relative_to(REPO))
    assert rel in _tracked_measurement_rows(), (
        f"{rel} is on disk but not tracked by git: the page cites a witness that only "
        "this box has, which is the #3775 defect in a fresh costume"
    )
    n_lines, n_bytes = _wc_l_c(path)
    assert len(_rows_of(path)) == n_lines, (
        f"{rel} parses to {len(_rows_of(path))} rows but `wc -l` prints {n_lines}; the "
        "citation counts a file it is not reading"
    )
    assert f"{n_lines} rows" in citation or f"{n_lines:,} rows" in citation, (
        f"the citing sentence does not print the {n_lines} rows `wc -l` prints on {rel}: "
        f"{citation[:200]!r}"
    )
    assert f"{n_bytes:,} bytes" in citation or f"{n_bytes} bytes" in citation, (
        f"the citing sentence does not print the {n_bytes:,} bytes `wc -c` prints on "
        f"{rel}: {citation[:200]!r}"
    )
    assert not SCANNED_FIGURE.search(citation), (
        f"the citation carries a sessions-scanned figure: {SCANNED_FIGURE.search(citation).group(0)!r}. "
        f"{rel} has no field for sessions opened, so that number came from the uncommitted "
        "summary and witnesses nothing"
    )
    comma_figures = set(re.findall(r"\d{1,3},\d{3}", citation)) - {f"{n_bytes:,}"}
    assert not comma_figures, (
        f"the citation prints {sorted(comma_figures)}, a five-figure total the witness "
        f"bytes do not yield; only its {n_bytes:,} bytes are recomputable from {rel}"
    )
    # Beside the command, the window and the scan date it documents — clause 1's "beside".
    path_line = next(i for i, line in enumerate(section) if path.name in line)
    table_line = next(i for i, line in enumerate(section) if line.strip().startswith("|"))
    assert path_line < table_line, (
        f"the witness is cited at line {path_line + 1}, at or below the table it is the "
        "source of, so a reader meets the figures before they meet their source"
    )
    lo = path_line
    while lo > 0 and section[lo - 1].strip():
        lo -= 1
    hi = path_line
    while hi + 1 < len(section) and section[hi + 1].strip():
        hi += 1
    paragraph = "\n".join(section[lo:hi + 1])
    for needed in ("--since 2026-10-01", "2026-11-04T23:27Z", "window 3"):
        assert needed in paragraph, (
            f"the paragraph that cites {path.name} no longer names {needed!r}, so the "
            "witness is not attached to the scan it witnesses"
        )
    return path


# ---------------------------------------------------------------------------
# clause 1 + 2 — the block names its witness and states what those bytes are
# ---------------------------------------------------------------------------

def test_the_published_block_cites_a_committed_rows_file_with_its_wc_figures() -> None:
    """Clause 1 + 2: the first block points at committed bytes and prints their size.

    `architecture/harness.md` presents a guard-stratified table as the reading of a
    scan, and until #3775 the only thing a reader could follow was the command — the
    bytes behind the published figures had been written to `/tmp` and deleted by the
    reboot. #3775 committed the witness; this node is what makes the page point at it.
    Three things must hold of the citation, none of them about the figures themselves:
    the named file is tracked by git (an untracked leftover in that directory is the
    exact state #3775 was filed against), the sentence states the row count and byte
    count `wc -l -c` reports rather than a remembered pair, and it states no
    sessions-opened total, because a witness row is one refusal and the file has no
    such field to read it from.
    """
    path = _check_witness_citation(harness_refusal_lines())
    assert path.name == "2026-11-04.rows.jsonl", (
        f"the block's witness is now {path.name}; the published rows are the 2026-11-04 "
        "corpus, so a new scan means a new block and a new expected set, not a silent "
        "re-point of this one"
    )


# ---------------------------------------------------------------------------
# clause 4 — the citation is worth making: the bytes reproduce the block
# ---------------------------------------------------------------------------

def test_the_cited_witness_recomputes_the_four_published_first_block_rows() -> None:
    """Clause 4: aggregating the committed rows reproduces the block, guard for guard.

    All 450/300/148/86, `grant` 14/14/1/1, `safety` 393/244/147/85, `tool_sandbox`
    43/42/0/0 — each of the four published rows is re-derived from the file's own lines
    and demanded of the table's cells: the refusals count, the measurable annotation,
    and both rates rendered the way the instrument renders them. The literals in
    `EXPECTED_FIRST_BLOCK` and the cells are checked against the same aggregation, so
    the page cannot be quietly rewritten to match a re-committed scan.

    The live-store block is deliberately NOT recomputed: 454 refusals over 303
    measurable is a reading of a store that has moved since, and no rows file records
    it. Its counts differing from the witness is the point — the node asserts they
    differ, so a future edit cannot source that block from these bytes either.
    """
    section = harness_refusal_lines()
    path, _ = _cited_witness(section)
    per = _per_guard(_rows_of(path))
    for guard, expected in EXPECTED_FIRST_BLOCK.items():
        block = per[guard]
        got = (block["refusals"], block["measurable"],
               block["same_target"], block["same_effect"])
        assert got == expected, (
            f"{path.name} aggregates {guard} to refusals={got[0]}, measurable={got[1]}, "
            f"same-target={got[2]}, same-effect={got[3]}; the published row says {expected}"
        )

    first, live = published_blocks(section)
    assert len(first) == 4, f"the witnessed block publishes {len(first)} rows, not 4"
    seen: set[str] = set()
    for cells in first:
        guard = next((g for g in EXPECTED_FIRST_BLOCK if f"`{g}`" in cells[0]), None)
        assert guard is not None, f"a published row names no known guard: {cells[0]!r}"
        assert guard not in seen, f"{guard} is published twice inside the witnessed block"
        seen.add(guard)
        refusals, measurable, same_target, same_effect = EXPECTED_FIRST_BLOCK[guard]
        assert cells[1] == str(refusals), f"{guard}: the table says {cells[1]!r} refusals"
        assert cells[2] == f"{measurable} measurable", (
            f"{guard}: the table says {cells[2]!r}, not '{measurable} measurable'"
        )
        assert cells[3] == _rate(same_target, measurable), (
            f"{guard}: same-target prints {cells[3]!r}, the bytes give "
            f"{_rate(same_target, measurable)!r}"
        )
        assert cells[4] == _rate(same_effect, measurable), (
            f"{guard}: same-effect prints {cells[4]!r}, the bytes give "
            f"{_rate(same_effect, measurable)!r}"
        )

    live_all = next(c for c in live if "`all`" in c[0])
    assert live_all[1] != str(EXPECTED_FIRST_BLOCK["all"][0]), (
        f"the live-store row now prints the witness's own {EXPECTED_FIRST_BLOCK['all'][0]} "
        "refusals, so the two blocks have been merged and the second is claiming the "
        "first's committed bytes"
    )


# ---------------------------------------------------------------------------
# clause 3 — the citation stops at the block it covers
# ---------------------------------------------------------------------------

def test_the_live_store_row_is_labelled_a_reading_with_no_committed_bytes() -> None:
    """Clause 3: the second scan says so itself, and no committed path reaches it.

    454 refusals over 303 measurable at 01:03Z on 2026-11-05 was never written to a rows
    file, so the citation that serves the first block cannot be extended over it without
    naming a file that does not exist — a false witness, which is worse than the missing
    one #3965 set out to fix. Pinned at both places a reader looks: the row's own first
    cell and the sentence that introduces the scan.
    """
    section = harness_refusal_lines()
    first, live = published_blocks(section)
    assert LIVE_ROW_LABEL in live[0][0] and NO_BYTES_LABEL in live[0][0], (
        f"the live-store row reads {live[0][0]!r}; it has to carry both {LIVE_ROW_LABEL!r} "
        f"and {NO_BYTES_LABEL!r}, or the table reads as two witnessed scans"
    )
    table_line = next(i for i, line in enumerate(section) if line.strip().startswith("|"))
    below = "\n".join(section[table_line:])
    assert CITED_ROWS.search(below) is None, (
        f"a committed rows path appears at or below the published table "
        f"({CITED_ROWS.search(below).group(0)!r}), where the live-store block lives — "
        "that block has no committed bytes to name"
    )
    intro = "\n".join(section[:table_line])
    at_scan = [line for line in intro.splitlines() if "01:03Z on 2026-11-05" in line]
    assert at_scan, "the second scan is no longer introduced by its own timestamp"
    assert LIVE_ROW_LABEL in intro, "the prose no longer calls the second scan a live-store reading"
    assert NO_BYTES_LABEL in intro, "the prose no longer says the live-store reading has no committed bytes"
    assert len(first) == 4, "the witnessed block changed size without this node being told"


# ---------------------------------------------------------------------------
# clause 5 — the shape rule's neighbour: no scratch path in either passage
# ---------------------------------------------------------------------------

def test_neither_refusal_passage_names_a_scratch_path() -> None:
    """Clause 5's no-regression half: a citation may not become a `/tmp` pointer again.

    The same reboot that made #3565's figures unfalsifiable is why a scratch path in a
    refusal passage is a defect rather than a style choice — the file is gone and the
    claim it supported is left standing. Checked line by line over both passages,
    including the new citation, so a re-runner who documents a fresh scan cannot put its
    output where the last one went.
    """
    for name, lines in (("CLAUDE.md", claude_refusal_lines()),
                        ("architecture/harness.md", harness_refusal_lines())):
        for n, line in enumerate(lines, 1):
            assert "/tmp" not in line, f"{name}:{n} points at scratch storage: {line.strip()[:160]}"
            assert "detour_rows.json" not in line, f"{name}:{n} names #3565's scratch rows file"


# ---------------------------------------------------------------------------
# Negative control for the witness citation — same shape as the shape rule's
# ---------------------------------------------------------------------------

def test_the_witness_citation_reddens_when_the_path_or_its_size_is_dropped() -> None:
    """Negative control: the citation node fails on spliced bytes, so it is reading them.

    Two splices of the shipped section, each one line away from the real page. (1) The
    path itself is replaced by the words a writer would reach for instead of it — and
    the rules must go red, because a section that says "the committed scan bytes"
    without naming them is precisely the state #3965 was filed against. (2) The byte
    figure is replaced with a plausible one, and the rules must go red too, because the
    size is the thing that makes the citation checkable rather than decorative. The
    shipped bytes pass at the end, so the two reds are the splice and nothing else.
    """
    section = harness_refusal_lines()
    path, citation = _cited_witness(section)
    rel = str(path.relative_to(REPO))
    assert rel in citation, "the citation sentence does not hold the path it cites"

    dropped = [line.replace(f"`{rel}`", "the committed scan bytes") for line in section]
    assert sum(1 for line in dropped if rel in line) == 0, "the splice left the path in"
    with pytest.raises(AssertionError) as first:
        _check_witness_citation(dropped)
    assert "no dated rows file" in str(first.value), (
        f"dropping the committed path did not redden the citation rule; it said: {first.value}"
    )

    _n_lines, n_bytes = _wc_l_c(path)
    wrong = [line.replace(f"{n_bytes:,} bytes", "240,000 bytes") for line in section]
    assert sum(1 for line in wrong if "240,000 bytes" in line) == 1, (
        f"the splice did not land: {n_bytes:,} bytes is not the single byte figure the page prints"
    )
    with pytest.raises(AssertionError) as second:
        _check_witness_citation(wrong)
    assert "bytes" in str(second.value), (
        f"falsifying the byte count did not redden the citation rule; it said: {second.value}"
    )

    assert _check_witness_citation(section) == path, (
        "the shipped section does not pass the rules the two splices above fail, so the "
        "control is proving nothing"
    )


# ---------------------------------------------------------------------------
# clause 2 (#4277) — the retirement names the archive as the cause, not the calendar
# ---------------------------------------------------------------------------

def test_harness_names_the_archive_as_the_era_that_survived_and_the_glob_that_hid_it() -> None:
    """harness.md's retirement says the September era survives gzipped, and says it in
    numbers a reader can re-count.

    The node exists because the page asserted the opposite for six weeks: 'the corpus it
    counted has aged out of readable storage — the session store holds about 30 days
    uncompressed'. Measured over the whole store on 2026-11-21, 4,013 sessions from
    2026-09 are `.json.gz` (oldest `20260909_153801_autonomy_2cf3`) against 64 plain
    `.json`, and `scripts/groundskeeper/retention-sweep.py` gzips an aged transcript and
    never deletes it. So the checks here are the count, the oldest archived id, the sweep's
    non-deletion, the flag that reads it, and the thing that actually hid the era — this
    scanner's `*.json` glob — plus the surviving reason stated as a count: those same 306
    `tool_sandbox` refusals now yield 287 measurable rows.

    The plain-store count is checked only as a *dated* pair of readings (38 on 2026-11-04,
    64 on 2026-11-21). It is the one figure in this passage that moves — the sweep runs
    daily and a restore can add files back — and a node that pinned it would go red on an
    unrelated day and teach a reader to ignore the node.
    """
    joined = "\n".join(harness_refusal_lines())
    for claim in FALSE_CORPUS_CLAIMS:
        assert claim not in joined, f"the false corpus cause is back in harness.md: {claim!r}"
    retired_at = joined.find("RETIRED")
    current_at = joined.find("What the fixed instrument prints")
    assert 0 <= retired_at < current_at, "the retirement passage moved"
    retirement = joined[retired_at:current_at]
    for token in ("4,013", "20260909_153801_autonomy_2cf3", "never deletes",
                  "--include-gz", "*.json", "retention-sweep.py", "false"):
        assert token in retirement, f"the corrected cause lost {token!r}: {retirement}"
    assert "287 measurable" in retirement and "306" in retirement, (
        "reason #1 has to stand as a count: the same 306 tool_sandbox refusals now yield "
        "287 measurable rows, which is what makes the old reading unreproducible"
    )
    assert "2026-11-04" in retirement and "2026-11-21" in retirement, (
        "the plain-store count is only honest dated, because it moves"
    )
    assert "38" in retirement and "64" in retirement, \
        "both dated plain-store readings belong to the correction"


# ---------------------------------------------------------------------------
# clause 4 (#4277) — the method-matched pair, published with its labels
# ---------------------------------------------------------------------------

def _blocks_around_the_pair() -> tuple[list[str], list[str]]:
    """(the pair block, the block printed under it) from harness.md's refusal section."""
    section = harness_refusal_lines()
    pair_at = next((i for i, line in enumerate(section) if PAIR_BLOCK_START in line), None)
    post_at = next((i for i, line in enumerate(section) if POST_ROW_BLOCK_START in line), None)
    assert pair_at is not None and post_at is not None and post_at > pair_at, (
        "the pair block and the post-#3774 block must both stand, in that order"
    )
    return section[pair_at:post_at], section[post_at:]


def test_the_method_matched_pair_is_published_across_the_ship_commit() -> None:
    """harness.md prints `safety` on each side of the paragraph's ship commit, both sides
    gz-decoded, each rate beside its measurable count, window and scan date.

    Values are pinned here, against this file's own default, because this block is a
    reading of the *archived* store over a closed window: an archived session is named by
    its date and never rewritten or deleted, so `30/50 (0.6)` is what re-running the named
    command over 2026-09-22..29 will print for as long as the store exists. That is the
    same property the 2026-11-04 block is pinned for; the block under it, a live store
    still growing, stays pinned on shape alone.

    The labels are the load-bearing half. A pair is only evidence if both sides were
    measured the same way, and the only thing saying whether they were is the word
    `method-matched` inside the row — so the node demands the word, the flag that makes it
    true (`--include-gz`), the windows it makes readable, and the ship commit the pair is
    cut at (`3baf94ed`, 2026-09-30). A future round that re-published one side off the
    plain store would have to either keep the label false or redden this node.
    """
    pair, _post = _blocks_around_the_pair()
    head = "\n".join(pair)
    for token in ("3baf94ed", "--include-gz", "window 3", "2026-11-21", "method-matched"):
        assert token in head, f"the pair block lost {token!r}"
    assert "NOT method-matched" not in head, \
        "the NOT label belongs to the block below, not to the pair"

    rows = labelled_rows(pair, "**method-matched**", "safety")
    assert len(rows) == 2, f"expected a before row and an after row, got {len(rows)}: {rows}"
    before, after = (rows[0], rows[1]) if "before" in rows[0][0] else (rows[1], rows[0])
    assert "before" in before[0] and "after" in after[0], [before[0], after[0]]
    assert "2026-09-22..29" in before[0] and "3,513" in before[0], before[0]
    assert "2026-09-30..10-06" in after[0] and "4,346" in after[0], after[0]
    published = {
        "before": ("65", "50 measurable", (30, 50, "0.6"), (16, 50, "0.32")),
        "after": ("66", "54 measurable", (35, 54, "0.648"), (19, 54, "0.352")),
    }
    for name, row in (("before", before), ("after", after)):
        refusals, measurable, *rates = published[name]
        assert row[1] == refusals, f"{name}: refusals printed {row[1]!r}, expected {refusals}"
        assert row[2] == measurable, f"{name}: measurable printed {row[2]!r}"
        assert len(row) == 5, f"{name}: a rate row missing an arm: {row}"
        for (num, den, rate), cell in zip(rates, row[3:]):
            assert cell == f"{num}/{den} ({rate})", f"{name}: printed {cell!r}, expected {num}/{den} ({rate})"
            assert num / den == pytest.approx(float(rate), abs=5e-4), \
                f"{name}: {num}/{den} is not {rate} — a rate disagreeing with its own denominator"
    sandbox = labelled_rows(pair, "same two scans", "tool_sandbox")
    assert len(sandbox) == 2, f"the archived era's measurable rows are not both printed: {sandbox}"
    for row in sandbox:
        joined = " | ".join(row)
        assert MEASURABLE.search(joined) and OVER.search(joined) and RATE.search(joined), row


def test_the_post_3774_row_is_published_and_labelled_not_method_matched() -> None:
    """The block under the pair adds the post-#3774 reading and says plainly that it is
    not a third side of the comparison.

    It cannot be: it reads plain `.json` (all traffic from 2026-11-05 on is plain) over a
    window three times longer than either side of the pair. So the row's label, its
    `--since 2026-11-05`, its scan date and its storage form are all pinned — while its
    values are not, because that store is still being written and its counts move (the
    2026-11-04 triage measured +1 refusal in three hours on the same command). The shape
    rule that holds every other rate to its denominator applies to it in full, and this
    node re-asserts it so the row cannot escape that check by being labelled something else.
    """
    _pair, post = _blocks_around_the_pair()
    head = "\n".join(post)
    assert "NOT method-matched" in head, "the post-#3774 row lost its label"
    assert "--since 2026-11-05" in head, "the row lost the window it was measured over"
    assert "2026-11-18" in head, "the row lost its scan date, and its counts move"
    assert "plain `*.json`" in head, "the row lost the storage form that makes it not-matched"
    rows = labelled_rows(post, "NOT method-matched", "safety")
    assert len(rows) == 1, f"expected exactly one NOT-method-matched safety row: {rows}"
    joined = " | ".join(rows[0])
    assert "**method-matched**" not in joined, "the row carries both labels at once"
    assert MEASURABLE.search(joined), f"row prints no measurable count: {joined}"
    assert OVER.search(joined) and RATE.search(joined), \
        f"row prints no rate over its denominator: {joined}"
    assert "3,639" in rows[0][0], f"the row must name how many sessions it scanned: {rows[0][0]}"


# ---------------------------------------------------------------------------
# clause 5 (#4277) — the pair is not sold as a win, and the owed-rerun sentence is gone
# ---------------------------------------------------------------------------

def test_the_pair_block_says_the_rate_did_not_improve_and_never_calls_it_a_win() -> None:
    """The block states that the detour rate did not improve after the paragraph, and the
    two printed rates have to agree with that sentence.

    This is the arithmetic behind the item's clause, not its prose: same-target went
    30/50 (0.6) before to 35/54 (0.648) after, same-effect 16/50 (0.32) to 19/54 (0.352),
    and a *higher* detour rate is a refused call followed by a look-alike retry more often.
    So the node demands the after-side ratio is not lower on either arm — if someone
    re-publishes this pair off a window where the rate genuinely fell, the sentence 'did
    not improve' becomes false and this node reddens with it, instead of the page quietly
    turning into a claim that the paragraph worked. The block must also say out loud that
    it is not a win, and must not pick any of the framings that would make it one.

    It also holds the deleted instruction down: 'none is full yet: the owed re-run over a
    week of post-paragraph traffic publishes the first figure worth comparing' sent a reader
    off waiting for a number the archive already held. With the pair published, waiting for
    it is a fiction, so the sentence must not return.
    """
    pair, post = _blocks_around_the_pair()
    head = "\n".join(pair)
    assert "did not improve" in head, "the block no longer says the rate did not improve"
    assert "not a win" in head, "the block must say the pair is not a win"
    section = "\n".join(harness_refusal_lines())
    for framing in WIN_FRAMINGS:
        assert framing not in section, f"the section frames the pair as a win: {framing!r}"
    rows = labelled_rows(pair, "**method-matched**", "safety")
    before, after = (rows[0], rows[1]) if "before" in rows[0][0] else (rows[1], rows[0])
    before_rates, after_rates = rates_over(before), rates_over(after)
    assert len(before_rates) == len(after_rates) == 2, (before_rates, after_rates)
    for arm, (bn, bd), (an, ad) in zip(("same-target", "same-effect"), before_rates, after_rates):
        assert an / ad >= bn / bd, (
            f"{arm} improved ({bn}/{bd} -> {an}/{ad}): 'did not improve' is now false, so "
            "the sentence and the block's own framing have to be rewritten together"
        )
    for dead in ("none is full yet", "first figure worth comparing"):
        assert dead not in section, f"the owed-rerun instruction is back: {dead!r}"
    assert post, "the post-#3774 block is part of this argument and has to stand with it"


# ---------------------------------------------------------------------------
# clause 6 (#4277) — the shape rule is shown to inspect the lines this change added
# ---------------------------------------------------------------------------

def test_the_shape_rule_actually_inspects_the_lines_4277_added() -> None:
    """The `checked >= 8` floor is green by count, so it has to be shown to be growing.

    SM_20261105_010319's own advisory measured 0 rate-bearing lines inside the extracted
    CLAUDE.md passage, which means that floor was satisfied in the round that added it
    entirely by harness.md's table rows — a page could satisfy `>= 8` by printing *less*,
    and the node would not notice. So the baseline is recorded at 9 (the figure over the
    two files at `26814c9b`: 8 table rows plus one prose line, counted with the same
    regexes this file ships), and this node demands the count is now above it AND that the
    surplus sits inside the block #4277 added, which did not exist when the floor was
    written. Five is the floor there because the four pair rows and the post-#3774 row
    alone supply five rate-bearing lines; the prose lines in that block are surplus this
    node does not need.
    """
    harness = harness_refusal_lines()
    claude = claude_refusal_lines()
    checked = [line for line in harness + claude if RATE.search(line)]
    assert len(checked) > SHAPE_CHECKED_BEFORE_4277, (
        f"the rule inspected {len(checked)} rate-bearing refusal line(s); the pre-#4277 "
        f"count was {SHAPE_CHECKED_BEFORE_4277}. The new rows are outside what this file "
        "extracts, so the floor is green by absence again"
    )
    pair, post = _blocks_around_the_pair()
    tail_checked = [line for line in pair + post if RATE.search(line)]
    assert len(tail_checked) >= 5, (
        f"only {len(tail_checked)} rate-bearing lines inside the published pair and the row "
        "under it: the shape rule is not reaching the lines it was extended for"
    )
    assert any("method-matched" in line for line in tail_checked), (
        "none of the inspected lines is a labelled row, so the count could be met by "
        "unrelated prose"
    )
    assert bare_rate_lines(harness) == [] and bare_rate_lines(claude) == []
    assert len(checked) >= 8, "the floor this node supersedes as evidence still holds"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
