"""#2452 — skill_lint's published stamp is aware UTC, not the box's wall clock.

`lint()` stamped the run with `dt.datetime.now().isoformat(timespec="seconds")`:
naive local time, no `tzinfo`. On this box (UTC-7) the 2026-10-08 run wrote
`"generated_at": "2026-10-08T18:44:36"` into `skill-lint-report.json` while the
file's own mtime was `2026-10-09T01:44:41Z`, and `skill-lint-report.md` repeated
that same value three times — `summary:`, `timestamp:` and the `# Skill Lint
Report — …` heading. A machine-facing payload carrying a naive local time is read
as UTC by every later reader (the class recorded 2026-09-21), so "when did lint
last run" was wrong by the box's offset — 7 h today, 8 h after the DST flip — and
silently disagreed with any UTC-stamped neighbour (a git commit time, another
report's `generated_at`, `prefetch.skill_match` rows). Every sibling note in
`~/obsidian/autonomy/` is already stamped `+00:00`; this report was the only
naive-local one in its own directory.

One stamp site feeds all three renderings: `render_report` takes the heading from
`result["generated_at"]` and `render_frontmatter` takes the same key and says so in
its own docstring ("not a fresh `now()`"), so the fix is one line, and this file
tests the value where it is born plus the artifact it is written into.

Two seams are crossed for real:

  * `lint()` → the dict a caller reads: clauses 1 and 2 run on the returned value,
    from an empty-record scan (instant, so the equality to the clock is a
    *second*-level claim and not a scan-duration fudge) and from the live library
    walk (so the pin is not a fixture-only path).
  * `main()` → bytes on disk → `json.load`: the nightly runs
    `python scripts/skill_lint.py` under its own `$HOME`, and the artifact the
    acceptance check names is `skill-lint-report.json`, so a subprocess writes it
    for real and this file re-reads it from disk. An in-memory dict cannot prove
    what `json.dumps` emitted.

The STALE-day arithmetic is deliberately NOT touched here — it compares naive-local
against naive-local and belongs to
`tests/test_skill_lint_report_trust.py::test_the_stale_day_count_is_naive_local_and_does_not_shift_by_the_box_offset`.
"""

from __future__ import annotations

import calendar
import datetime as dt
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_stamp_utc", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()

#: An explicit UTC marker: the `+00:00` `isoformat()` emits for an aware UTC
#: datetime, or the `Z` spelling a reader meets in other notes' stamps.
UTC_OFFSET_RE = re.compile(r"(Z|[+-]\d{1,2}:?\d{2})$")

#: How far the stamp may sit from the real epoch. This is a second-level pin: the
#: defect it is about put the stamp 25 200 s (7 h) from real UTC, so any bound
#: under an hour separates them, and this one still says the stamp matches
#: `date -u` to the second rather than to the day.
MAX_STAMP_SKEW_S = 2.0


def _utc_epoch(stamp: str) -> float:
    """The epoch the stamp string claims, read through its own offset.

    `calendar.timegm` over the parsed tuple rather than `.timestamp()`, because
    `.timestamp()` on a datetime whose `tzinfo` had somehow been dropped silently
    interprets it in the *box's* zone — which is the bug, so it must not be
    available to the test of the bug.
    """
    return float(calendar.timegm(dt.datetime.fromisoformat(stamp).utctimetuple()))


# ── clauses 1 and 2: the stamp as `lint()` returns it ───────────────────────────

def test_lint_publishes_an_aware_utc_stamp():
    """Clause 1: the string carries an offset, and parses back tz-aware at zero.

    `lint(skill_records=[])` is the fastest honest way to call the real function:
    the parameter is compared with `is None`, so an empty list means "these are the
    records" and the scan is instant — no live library read, and no stamp typed by
    hand into a fixture.

    The naive form is checked first as the control. `tzinfo is None` is also what a
    parse of the *broken* value gives, so if this interpreter's `fromisoformat`
    behaved differently the awareness assertion could pass on a naive stamp;
    proving it distinguishes the two shapes here makes the assertion below a
    verdict rather than a property of the parser.
    """
    naive = "2026-10-08T19:47:57"
    assert dt.datetime.fromisoformat(naive).tzinfo is None, (
        "the control is not holding: a naive stamp parsed as aware here, so the "
        "awareness assertion below could pass on the broken value")

    stamp = sl.lint(skill_records=[])["generated_at"]
    assert isinstance(stamp, str), repr(stamp)
    assert UTC_OFFSET_RE.search(stamp), (
        f"{stamp!r} carries no explicit UTC offset — a naive local stamp is the "
        "thing that reads 7 h stale to every later reader")
    parsed = dt.datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None, f"{stamp!r} parsed naive"
    assert parsed.utcoffset() == dt.timedelta(0), (
        f"{stamp!r} is aware but not at UTC zero: {parsed.utcoffset()}")


def test_the_stamp_equals_real_utc_at_call_time_not_the_box_clock_relabelled():
    """Clause 2: the instant is real UTC, so an offset cannot be a sticker on local time.

    Awareness alone would not be enough:
    `dt.datetime.now().replace(tzinfo=timezone.utc)` satisfies clause 1 and still
    publishes the box's wall clock, 7 h behind real UTC on this box. So the pin is
    against the epoch — the wall-clock fields inside the string have to name an
    instant `time.time()` agrees with, which is exactly what `date -u` prints — and
    a second assertion pins the *wall clock* against `time.gmtime()`'s fields to
    the second, which is the acceptance check's own units.

    Independent of `$TZ`: today a relabelled-local stamp is 25 200 s out on this
    box; on a UTC box the two are indistinguishable and there is nothing to catch.
    """
    before = time.time()
    stamp = sl.lint(skill_records=[])["generated_at"]
    after = time.time()
    epoch = _utc_epoch(stamp)
    # `timespec="seconds"` truncates microseconds, so the stamp names an instant up
    # to one second *earlier* than the call — never later, which is the half of this
    # window that would say the stamp came from a different clock.
    assert epoch >= before - 1.0 and epoch <= after, (
        f"{stamp!r} claims epoch {epoch}, but real UTC across the call was "
        f"[{before}, {after}] — {epoch - before:+.1f}s off")
    assert abs(epoch - time.time()) <= MAX_STAMP_SKEW_S, (
        f"{stamp!r} is more than {MAX_STAMP_SKEW_S}s from `date -u` now")
    # Same claim, wall-clock fields: `date -u` at run time, to the second. A
    # relabelled local stamp differs here by the box's whole offset (25200 s).
    gmtime_now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    assert stamp[:19] in (gmtime_now, _one_second_earlier(gmtime_now)), (
        f"stamp wall clock {stamp[:19]!r} is not `date -u` ({gmtime_now!r}) or the "
        "second before it — the offset is not carrying real UTC")


def _one_second_earlier(stamp: str) -> str:
    """`stamp` minus one second, for the wall-clock comparison's second boundary."""
    parsed = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
    return (parsed - dt.timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")


def test_the_live_walk_stamps_the_same_aware_instant():
    """Clause 1 again over the path the weekly job actually takes: `lint()` with no argument.

    `tests/test_skills_single_walk.py` pins that this is the nightly's shape. The
    empty-record scan above cannot prove the live branch of `lint()` reaches the
    same stamp line — it takes a different path to the same `return`, and a stamp
    taken before the walk would still satisfy the node above.
    """
    before = time.time()
    result = sl.lint()
    stamp = result["generated_at"]
    assert result["total"] > 0, (
        f"the live walk saw {result['total']} skills, so the stamp below proves "
        "nothing about the real path")
    assert UTC_OFFSET_RE.search(stamp), f"live stamp {stamp!r} has no offset"
    assert dt.datetime.fromisoformat(stamp).utcoffset() == dt.timedelta(0), stamp
    # The walk takes seconds, so the bound is that the stamp is not ahead of the
    # clock and not older than the walk that produced it.
    epoch = _utc_epoch(stamp)
    assert before - MAX_STAMP_SKEW_S <= epoch <= time.time() + 1.0, (
        f"live stamp {stamp!r} claims epoch {epoch} outside the walk window "
        f"[{before}, {time.time()}]")


# ── the artifact the acceptance check greps: bytes on disk ─────────────────────

def test_the_json_the_nightly_writes_carries_the_aware_stamp(tmp_path):
    """`skill-lint-report.json` as written, re-read and parsed.

    `main()` writes `REPORT_PATH.with_suffix(".json")` through
    `json.dumps(result, indent=2, default=str)`, and `default=str` is exactly the
    setting under which an aware value could quietly lose its offset if the stamp
    ever stopped being a string. So the nightly's own invocation runs in a
    subprocess under a redirected `$HOME` (the shape
    `tests/test_skill_lint_report_frontmatter.py` uses to keep the live vault out
    of reach), and this node reads the artifact back off disk rather than trusting
    the dict that preceded it.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "skill_lint.py")],
        cwd=str(tmp_path), env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]

    json_path = tmp_path / "obsidian" / "autonomy" / "skill-lint-report.json"
    assert json_path.is_file(), "the JSON artifact moved"
    stamp = json.loads(json_path.read_text(encoding="utf-8"))["generated_at"]
    assert UTC_OFFSET_RE.search(stamp), f"on-disk stamp {stamp!r} has no offset"
    parsed = dt.datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0), stamp
    mtime_epoch = json_path.stat().st_mtime
    assert abs(_utc_epoch(stamp) - mtime_epoch) <= 5.0, (
        f"the file's own mtime is {mtime_epoch - _utc_epoch(stamp):+.0f}s from the "
        f"instant its stamp claims ({stamp!r}) — the stale-by-the-offset report again")


def test_the_stamp_site_no_longer_reads_the_naive_clock():
    """The acceptance grep as an executable check: no `generated_at` line uses naive `now()`.

    `grep -n 'dt.datetime.now()' scripts/skill_lint.py | grep generated_at` is the
    contract's command; this is the same predicate over the same text, so it runs
    wherever the suite runs. The positive control is in the same read: the STALE-day
    path still calls naive `dt.datetime.now()` on purpose (naive-local on both sides
    of that subtraction, so `max_age_days` is unaffected by this item), which is
    what proves the pattern and the path both resolved. A test that only asserted
    emptiness would pass forever on a file it failed to read.
    """
    source = (ROOT / "scripts" / "skill_lint.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    naive_sites = [i for i, line in enumerate(lines, 1) if "dt.datetime.now()" in line]
    assert len(naive_sites) >= 2, (
        f"found {naive_sites}: the two STALE-day sites are meant to keep a naive "
        "`dt.datetime.now()`, so finding none means the read or the pattern failed "
        "and the assertion below proves nothing")
    stamp_sites = [i for i, line in enumerate(lines, 1)
                   if "generated_at" in line and "datetime.now()" in line
                   and "timezone" not in line]
    if stamp_sites:
        first = lines[stamp_sites[0] - 1].strip()
        raise AssertionError(
            f"line(s) {stamp_sites} still stamp generated_at from the naive local "
            f"clock, e.g. {first!r}")
