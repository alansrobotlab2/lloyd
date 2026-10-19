"""#2983 pins §8 of the published behavioural noise-floor note.

The note (``knowledge/evaluation/behavioural-noise-floor-pair.md``, vault
``4ed6f424``, provenance added by vault ``50492334``) is the authority for the
pair: #2196's behavioural half. What #2983 fixes is that its §8 recorded only
what the op did NOT touch, so the next reader could not tell that both captures
ran while a worker job held the same engine, nor that the spread is therefore a
bound rather than a floor. Every number pinned here is a published measurement
of immutable history — queue row 7984 completed 2026-10-11T00:35:44Z and cannot
move — so unlike a quotation of prose, a node here can only go red if someone
alters or deletes the provenance, which is the thing this item exists to keep.

Why the nodes read the live vault and are NOT marked ``live_vault``
(`tests/test_archived_skill_artifacts.py` argues the inverse case for quote
tests, and this file deliberately has none — see the can-fail node): every
assertion below parses the note's own tables and re-derives the arithmetic they
claim, so the assertion is about a set of numbers, not a sentence. Marking them
would deselect the only check that the provenance is still there. Resolution is
`app.data_root.vault_root()`, which under the gate's round home is the
symlink `ensure_round_home` points at the live vault, and nothing here reads
`LLOYD_DATA`, so the nodes do not skip in gate shape and add no row to
`tests/skip_inventory.py`'s ledger.

The measurements, and the command each came from:

    # the job window, from the extract the note publishes (never the whole
    # store, which is over the 1 MiB witness ceiling):
    python3 -c "import sqlite3;c=sqlite3.connect('file:backlog/data/2026-10-18.2983-lloyd-data-workers.db?mode=ro',uri=True);print(c.execute('SELECT claimed_at,completed_at FROM queue WHERE id=7984').fetchone())"
    # per-trial placement: trial YAMLs carry no timestamp, so each side's
    # sequential per-scenario `seconds` from its capture.yaml are summed from
    # the run-id stamp — the anchor both sides' `generated_at` confirm.

Grader determinism (clause 4) was measured, not assumed: scoring side A twice
through the note's own Reproduce command and `cmp`-ing the cards came out
IDENTICAL — `cmp` exit 0, no output, sha256
`54ced681f63cc8ce17c9ddcf8bd20cf6d0f1789b3f60fafba10c279e279c8e43` for both and
for the card published 2026-10-15. The node pins that the note records the
DIRECTION of that result; it does not re-run the scorer, because the trace dirs
the re-score needs live under `LLOYD_DATA`, which the gate redirects.
"""

from __future__ import annotations

import re
from datetime import datetime

import pytest

from app.data_root import vault_root

NOTE = vault_root() / "knowledge" / "evaluation" / "behavioural-noise-floor-pair.md"

#: Queue job 7984 (`owed-check`, payload `item_id` 2368) — the resident job the
#: pool was NOT drained of. Read from the published extract; both instants are
#: the row's own `claimed_at` / `completed_at`.
JOB_ID = "7984"
WINDOW_START = "2026-10-11T00:11:43.616925+00:00"
WINDOW_END = "2026-10-11T00:35:44.918594+00:00"

#: Each side's t0 is the timestamp inside its run-id, confirmed at the far end
#: by its own `generated_at` (side A lands 0.432 s early, side B 0.819 s).
T0 = {"A": "00:14:13.000", "B": "00:28:12.000"}
GENERATED_AT = {"A": "2026-10-11T00:28:12Z", "B": "2026-10-11T00:43:16Z"}

#: What §8 publishes as each side's overlap, and what it must equal.
INSIDE_A = 838.568
INSIDE_B = 452.919
ELAPSED = {"A": 838.568, "B": 903.181}
ZERO_ROW_TRIAL = "attribution-on-a-recalled-answer"
ZERO_ROW_START = "00:35:35.368"
ZERO_ROW_AFTER_JOB = 295.794  # of the trial's 305.345 s

CARD_SHA256 = "54ced681f63cc8ce17c9ddcf8bd20cf6d0f1789b3f60fafba10c279e279c8e43"

DASH = "\u2013"  # the en dash the tables use between span endpoints


def _note() -> str:
    assert NOTE.is_file(), (
        f"{NOTE} is missing. It is the authority for #2196's behavioural half; "
        "if it has genuinely been retired, delete this file in the same round."
    )
    return NOTE.read_text(encoding="utf-8")


def _section(text: str, number: str) -> str:
    """§<number> of the note, from its `## N.` heading to the next `## ` heading.

    Level-3 headings (`### 8.1`) stay inside the slice: the provenance is §8.1.
    """
    m = re.search(rf"^## {re.escape(number)}\.", text, re.MULTILINE)
    assert m, f"the note has no `## {number}.` heading"
    rest = text[m.end():]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _tables(chunk: str) -> list[list[list[str]]]:
    """Every markdown table in `chunk` as rows of trimmed cells."""
    out: list[list[list[str]]] = []
    rows: list[list[str]] = []
    for line in chunk.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # the |---|---| rule row
            rows.append(cells)
        elif rows:
            out.append(rows)
            rows = []
    if rows:
        out.append(rows)
    return out


def _secs(value: str) -> float:
    return float(re.sub(r"[^\d.]", "", value))


def _clock(stamp: str) -> datetime:
    return datetime.strptime(f"2026-10-11 {stamp}", "%Y-%m-%d %H:%M:%S.%f")


def _placement(text: str) -> dict:
    """Re-derive each side's trial spans from §8's own table and t0.

    Returns per side a list of (scenario, seconds, start, end) plus the total.
    The spans are not trusted: each is recomputed from t0 plus the running sum
    of `seconds`, and a row whose published end disagrees by more than 2 ms is
    a broken claim, not a rounding note.
    """
    side_col = {"A": 1, "B": 3}
    span_col = {"A": 2, "B": 4}
    trial_table = None
    for table in _tables(_section(text, "8")):
        head = table[0]
        if len(head) == 5 and "seconds" in head[1] and "span" in head[2]:
            trial_table = table
            break
    assert trial_table, "§8 lost the per-scenario seconds-and-span table"

    placed: dict = {}
    for side in ("A", "B"):
        t0 = _clock(T0[side])
        rows, cursor = [], t0
        for row in trial_table[1:]:
            if row[0].strip("*").lower().startswith("sum"):
                published = _secs(row[side_col[side]])
                assert abs(published - sum(r[1] for r in rows)) < 0.002, (
                    f"§8's sum row for side {side} says {published} but the "
                    f"{len(rows)} trial rows add to {sum(r[1] for r in rows):.3f}"
                )
                continue
            sec = _secs(row[side_col[side]])
            start_s, end_s = row[span_col[side]].split(DASH)
            start, end = _clock(start_s.strip()), _clock(end_s.strip())
            assert abs((start - cursor).total_seconds()) < 0.002, (
                f"side {side} `{row[0]}` starts {start_s} but the previous trial "
                f"ended {cursor:%H:%M:%S.%f} — the trials are not sequential"
            )
            assert abs((end - start).total_seconds() - sec) < 0.002, (
                f"side {side} `{row[0]}`: span {start_s}{DASH}{end_s} is "
                f"{(end - start).total_seconds():.3f} s, not the {sec} s claimed"
            )
            rows.append((row[0].strip("`"), sec, start, end))
            cursor = end
        placed[side] = rows
    return placed


def _overlap(start: datetime, end: datetime) -> float:
    lo = max(start, _clock(WINDOW_START[11:23]))
    hi = min(end, _clock(WINDOW_END[11:23]))
    return max(0.0, (hi - lo).total_seconds())


def _check_job_window(text: str) -> None:
    """Clause 1: §8 names job 7984, its measured window, and the held-but-not-drained state."""
    sec = _section(text, "8")
    assert JOB_ID in sec, f"§8 does not name queue job {JOB_ID}"
    assert "owed-check" in sec, "§8 names the job but not its source `owed-check`"
    assert "2368" in sec, "§8 omits the payload's `item_id` 2368"
    assert re.search(r"not\s+drained", sec), "§8 must say the pool was NOT drained"
    assert re.search(r"[Nn]o\s+queue row was claimed", sec), (
        "§8 must say new dispatch was held, i.e. that no queue row was claimed "
        "across the captures — that claim is what makes it a hold and not an idle pool"
    )
    job_row = None
    for table in _tables(sec):
        for row in table[1:]:
            if row and JOB_ID in row[0]:
                job_row = row
                break
        if job_row:
            break
    assert job_row, f"§8 lost the table row for job {JOB_ID}"
    assert WINDOW_START in job_row[2], (
        f"job {JOB_ID}'s `claimed_at` cell reads {job_row[2]!r}, not {WINDOW_START}"
    )
    assert WINDOW_END in job_row[3], (
        f"job {JOB_ID}'s `completed_at` cell reads {job_row[3]!r}, not {WINDOW_END}"
    )


def _check_overlap(text: str) -> dict:
    """Clause 2: both sides' overlap as measured, and the arithmetic behind it."""
    sec = _section(text, "8")
    placed = _placement(text)
    inside = {
        side: sum(_overlap(s, e) for _, _, s, e in rows) for side, rows in placed.items()
    }
    elapsed = {side: sum(r[1] for r in rows) for side, rows in placed.items()}

    assert abs(inside["A"] - INSIDE_A) < 0.002, (
        f"side A's overlap re-derives to {inside['A']:.3f} s, not the {INSIDE_A} s §8 states"
    )
    assert abs(inside["B"] - INSIDE_B) < 0.002, (
        f"side B's overlap re-derives to {inside['B']:.3f} s, not the {INSIDE_B} s §8 states"
    )
    for side in ("A", "B"):
        assert abs(elapsed[side] - ELAPSED[side]) < 0.002, (
            f"side {side}'s trials add to {elapsed[side]:.3f} s, not {ELAPSED[side]} s"
        )
    assert inside["A"] == elapsed["A"], (
        "side A is claimed to sit wholly inside the job window; anything less and "
        "the 100% figure is wrong"
    )
    pct_b = 100.0 * inside["B"] / elapsed["B"]
    assert abs(pct_b - 50.1) < 0.05, f"side B's overlap is {pct_b:.1f}%, not the 50.1% published"
    assert f"{INSIDE_B}" in sec and f"{INSIDE_A}" in sec, (
        f"§8 must publish both overlap totals ({INSIDE_A} and {INSIDE_B})"
    )
    assert "50.1" in sec and "100%" in sec, "§8 must publish the per-side proportions"

    # Side A's whole run, and side B's first six trials, inside the window.
    assert placed["A"][0][2] == _clock(T0["A"]) and _overlap(*placed["A"][-1][2:]) > 0, (
        "side A must be placed from 00:14:13 through its last trial inside the window"
    )
    six = placed["B"][:6]
    assert all(_overlap(s, e) == e_s for (_, e_s, s, e) in six), (
        "side B's first six trials are claimed to sit wholly inside the window"
    )
    zero = [r for r in placed["B"] if r[0] == ZERO_ROW_TRIAL]
    assert zero, f"§8 lost side B's zero-row trial `{ZERO_ROW_TRIAL}`"
    _, trial_secs, zstart, zend = zero[0]
    assert f"{ZERO_ROW_START}" in sec, (
        f"§8 must state that side B's zero-row trial starts {ZERO_ROW_START}, "
        "i.e. INSIDE the window — the superseded claim was that it fell outside"
    )
    after = (zend - _clock(WINDOW_END[11:23])).total_seconds()
    assert abs(after - ZERO_ROW_AFTER_JOB) < 0.002, (
        f"the trial runs {(zend - zstart).total_seconds():.3f} s from {zstart:%H:%M:%S.%f}; "
        f"the part after the job is {after:.3f} s, not {ZERO_ROW_AFTER_JOB} s"
    )
    assert abs(trial_secs - 305.345) < 0.002, "the zero-row trial is 305.345 s"
    assert f"{ZERO_ROW_AFTER_JOB}" in sec and "305.345" in sec, (
        f"§8 must publish both {ZERO_ROW_AFTER_JOB} s after the job and the trial's 305.345 s"
    )
    assert re.search(r"[Nn]either side is a clean idle-surface run", sec), (
        "§8 must say in words that neither side is a clean idle-surface run"
    )
    return placed


def _check_upper_bound(text: str) -> None:
    """Clause 3: the consequence, in plain words."""
    sec = _section(text, "8")
    hits = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", sec)) if "upper bound" in s]
    assert hits, "§8 never states an upper bound"
    joined = " ".join(hits)
    assert "0.507143" in joined and "0.5" in joined, (
        "the upper-bound sentence must name the spread it bounds: " + joined[:160]
    )
    assert re.search(r"not a clean idle-surface floor", joined), (
        "the upper-bound sentence must say what the spread is NOT: " + joined[:160]
    )


def _check_grader_determinism(text: str) -> None:
    """Clause 4: §8 records which way the `cmp` came out, and on what artefacts."""
    sec = _section(text, "8")
    assert re.search(r"\bcmp\b", sec), "§8 does not mention `cmp`"
    assert "/tmp/rescore_A_1.json" in sec and "/tmp/rescore_A_2.json" in sec, (
        "§8 must name the two re-scored cards it compared, so the run is repeatable"
    )
    assert re.search(r"byte-identical", sec, re.IGNORECASE), (
        "§8 must record the outcome of the double scoring; the word `byte-identical` "
        "is the item's check for it"
    )
    assert re.search(r"exited 0 with no output", sec), (
        "§8 must record WHICH WAY the cmp came out, not that it was run"
    )
    assert not re.search(r"cmp[^\n]{0,80}?\b(differ|differed|mismatch)", sec, re.IGNORECASE), (
        "§8 must not claim the two cards differed — they are identical"
    )
    assert CARD_SHA256 in sec, f"§8 must publish the digest both re-scores produced ({CARD_SHA256})"
    assert re.fullmatch(r"[0-9a-f]{64}", CARD_SHA256), "the published digest must be a sha256"


def _check_zero_row_is_a_timeout(text: str) -> None:
    """Clause 5: side B's `ran: 0` trial is a timeout boundary, not a contention signature."""
    sec = _section(text, "8")
    assert "run_timeout" in sec, "§8 must record `kind: run_timeout` on side B's zero-row trial"
    assert re.search(r"duration_seconds: 300\.0\b", sec), (
        "§8 must record `duration_seconds: 300.0` — the timeout boundary itself"
    )
    assert re.search(r"timeout-boundary", sec), "§8 must name it a timeout-boundary artefact"
    # The sentence is wrapped in the file, so the pattern matches on \s+ and not
    # [^\n]: a checker that only passes on one line-break layout is testing prose.
    assert re.search(
        r"not\s+by\s+itself\s+evidence\s+that\s+the\s+queue\s+job", sec, re.IGNORECASE
    ), (
        "§8 must say the timeout is not by itself evidence of queue contention, "
        "or the next reader blames the worker job"
    )


def _check_untouched_sections(text: str) -> None:
    """The parts of the note this op had to leave alone, pinned structurally."""
    lines = text.splitlines()
    assert len(lines) >= 284, f"the note is {len(lines)} lines; the published note is 284+"

    axes = None
    for table in _tables(_section(text, "2")):
        if table[0][:2] == ["axis", "value_a"]:
            axes = table
            break
    assert axes, "§2's axis table is gone"
    published = {}
    for row in axes[1:]:
        axis, value_a, value_b, delta = row[0].strip("`"), float(row[1]), float(row[2]), row[3]
        assert axis not in published, f"§2 lists axis {axis} twice"
        published[axis] = (value_a, value_b)
        assert abs(abs(value_a - value_b) - _secs(delta)) < 1e-6, (
            f"§2's `{axis}` row is internally inconsistent: |{value_a} - {value_b}| "
            f"is not {_secs(delta)}"
        )
    assert published["action_consistency"] == (0.857143, 0.35), (
        "§2's `action_consistency` row is the 0.507143 spread this note exists to "
        "price; it is quoted from the artifact and must not be edited here"
    )
    assert published["stale_fact_action"] == (1.0, 0.5), "§2's 0.5 row must stand"

    sec7 = _section(text, "7")
    for figure in ("838.568", "903.18", "1200"):
        assert figure in sec7, f"§7 no longer carries its timing figure {figure}"
    # §7's two totals are the same measurements §8 now splits per trial.
    placed = _placement(text)
    assert abs(sum(r[1] for r in placed["A"]) - 838.568) < 0.002
    assert round(sum(r[1] for r in placed["B"]), 2) == 903.18


# ── the graded nodes ────────────────────────────────────────────────────────


def test_section_8_names_job_7984_and_its_measured_window():
    """Clause 1: the hold, the job, the window, and that the pool was not drained."""
    _check_job_window(_note())


def test_section_8_gives_both_sides_overlap_as_measured():
    """Clause 2: A 100% inside, B 50.1%, and the zero-row trial's 295.794 s after."""
    _check_overlap(_note())


def test_section_8_states_the_spread_is_an_upper_bound_not_an_idle_surface_floor():
    """Clause 3: 0.507143 / 0.5 is a bound on noise, and the note says so."""
    _check_upper_bound(_note())


def test_section_8_records_which_way_the_double_scoring_cmp_came_out():
    """Clause 4: scoring one side twice came out byte-identical, and that is written down."""
    _check_grader_determinism(_note())


def test_section_8_records_side_b_zero_row_trial_as_a_run_timeout():
    """Clause 5: `kind: run_timeout` at `duration_seconds: 300.0`, not queue contention."""
    _check_zero_row_is_a_timeout(_note())


def test_the_sections_this_op_had_to_leave_alone_are_intact():
    """§2's 0.507143 and 0.5 rows, §7's timing totals, and the note's length."""
    _check_untouched_sections(_note())


def test_the_pins_fail_on_a_note_without_the_provenance():
    """The can-fail proof, and it must fail on the REAL pre-#2983 shape.

    §8 before this item ran was the single heading "Not touched by this op" plus
    one paragraph — exactly what `_strip_8_1` reproduces from the live file, so
    the fixture is the note as published by vault `4ed6f424`, not an invented
    string. Each checker is pointed at it and must refuse.
    """
    stripped = _strip_8_1(_note())
    for checker in (
        _check_job_window,
        _check_overlap,
        _check_upper_bound,
        _check_grader_determinism,
        _check_zero_row_is_a_timeout,
    ):
        with pytest.raises(AssertionError):
            checker(stripped)


def _strip_8_1(text: str) -> str:
    """Drop §8.1 from the note, leaving the §8 heading and §8.2 as they were."""
    out = re.sub(
        r"### 8\.1 .*?(?=### 8\.2 )", "### 8.1 \n\n", text, flags=re.DOTALL
    )
    assert out != text, "the strip found no §8.1 to remove — the fixture is not the note"
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
