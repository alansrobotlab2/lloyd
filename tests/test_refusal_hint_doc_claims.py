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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
