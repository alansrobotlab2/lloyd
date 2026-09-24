"""The shape of one Activity Log entry (backlog #845).

`autonomy._append_activity_log` is the scheduler's writer into
`~/obsidian/autonomy/NN-*.md`; `autonomy.append_activity_line` is the same
implementation the MCP tool and the HTTP route call with a body in memory. The
retention sweep bounds those logs by counting *entry bullets* — lines beginning
`- ` — so a note that carried its own newlines wrote continuation lines the cap
could neither count nor remove: `autonomy/24-data-pipeline.md` stood at 359 such
lines through every weekly sweep of it, its prune marker meanwhile counting 6,841
entries pruned. The other half of the same defect is why a loop was unreadable: a
task failing in a loop wrote one entry per attempt, all byte-distinct only by
stamp and run id, until the retained window held nothing but that error string.

Two rules, both pinned here at the append site:

1. one physical line per note, whatever the note contains;
2. a back-to-back repeat of a failure — same text once the stamp and the run id
   are normalised away — is counted on the line it already occupies, not appended.
"""
import datetime
import re

import pytest

import autonomy

#: The instant the pinned clock stamps its first append with. `_append_activity_log`
#: formats `datetime.datetime.now(timezone.utc)` into every entry, and the
#: property under test is about notes that differ ONLY in that stamp and in the
#: run id, which two real calls inside one second cannot produce.
T0 = datetime.datetime(2026, 9, 24, 9, 0, tzinfo=datetime.timezone.utc)

# A run summary is markdown prose and the failure note truncates it by
# characters (`f"Run … FAILED ({kind}): {summary[:280]}"`, autonomy.py), so a
# note routinely arrives carrying line breaks. This is that note.
MULTI_LINE_NOTE = (
    "Traceback (most recent call last):\n"
    '  File "/srv/jobs/vault_write.py", line 118, in main\n'
    "    raise StoreUnavailable(path)\n"
    "ValueError: boom"
)


#: The real module, under a name that survives the class body below: inside
#: `_Clock`, the bare name `datetime` is the inner class, not the module.
_dt = datetime  # the module itself, not the class of the same name below


class _Clock:
    """Stand-in for the `datetime` module with a settable instant.

    Attribute names matter: the writer spells the call
    `datetime.datetime.now(datetime.timezone.utc)`, so this exposes a `datetime`
    with `now` and a `timezone` with `utc`, and nothing else. Patching the instant
    is what lets a test append two runs of one failure loop a minute apart, which
    two real calls inside one second cannot produce.
    """

    instant = T0

    class datetime:  # noqa: N801 — the name the call site spells
        @staticmethod
        def now(tz=None):
            return _Clock.instant

    timezone = _dt.timezone


@pytest.fixture
def task_file(tmp_path, monkeypatch):
    """One task file at `<tmp_path>/autonomy/991-activity-log.md`, clock pinned.

    Returns `(path, stamps)`: `stamps` is an object whose `advance()` moves the
    next append's entry stamp forward a minute, so a test can write two runs of
    one failure loop a minute apart the way the scheduler really does.
    """

    class _Stamps:
        def __init__(self):
            _Clock.instant = T0

        @staticmethod
        def advance(minutes=1):
            _Clock.instant = _Clock.instant + datetime.timedelta(minutes=minutes)

        @staticmethod
        def now_str():
            return _Clock.instant.strftime("%Y-%m-%dT%H:%M:%SZ")

    task_dir = tmp_path / "autonomy"
    task_dir.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", task_dir)
    monkeypatch.setattr(autonomy, "datetime", _Clock)
    path = task_dir / "991-activity-log.md"
    path.write_text(
        "---\nid: 991\nname: Activity log\nstatus: up_next\n---\n\n"
        "# Activity log\n\n## Activity Log\n",
        encoding="utf-8")
    return path, _Stamps()


def _under_heading(path):
    """Every non-blank line after `## Activity Log`, verbatim.

    Deliberately the sweep's own reading rule — a bullet is a line that begins
    `- ` at column 0, not one that begins with whitespace and then `- ` — because
    that is the rule which decides what the cap can see.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip().lower() == "## activity log")
    return [ln for ln in lines[start + 1:] if ln.strip()]


def _entries(under):
    return [ln for ln in under if ln.startswith("- ")]


def _failures(under):
    return [ln for ln in under if "FAILED" in ln]


def _failure_note(run_id, summary="Error output: Check stderr output for details"):
    """The note `_record_failure` writes, spelled the same way: a stamp-bearing
    run id twice, once as the run and once inside the run-record path."""
    return (f"Run {run_id} — FAILED (task): {summary[:280]} "
            f"[full: autonomy-runs/991/{run_id}.md]")


# ── Clause 3: one physical line per note ─────────────────────────────────────


def test_a_note_carrying_newlines_writes_one_physical_line(task_file):
    path, _stamps = task_file

    autonomy._append_activity_log(991, MULTI_LINE_NOTE)

    under = _under_heading(path)
    assert len(under) == 1, (
        f"a 4-line note wrote {len(under)} lines under the heading: {under}")
    assert _entries(under) == under, (
        f"the line written is not an entry bullet, so no cap can count it: {under}")
    assert under[0].startswith(f"- {T0.strftime('%Y-%m-%dT%H:%M:%SZ')}: "), under[0]
    # Folding may not lose text: the value of the note is the traceback.
    for fragment in ("Traceback (most recent call last):",
                     'File "/srv/jobs/vault_write.py", line 118, in main',
                     "raise StoreUnavailable(path)",
                     "ValueError: boom"):
        assert fragment in under[0], (
            f"the fold dropped {fragment!r} from the entry: {under[0]}")


def test_a_multiline_note_on_a_clean_log_leaves_no_line_the_cap_cannot_see(task_file):
    """The acceptance shape, stated the way the sweep reads the file.

    Counting the lines that are neither bullets nor blank is what the weekly
    sweep's own bound depends on, so this is the check the fix has to move from
    non-zero to zero — the same expression the fleet check runs against
    `~/obsidian/autonomy/*.md`, applied to a seeded file.
    """
    path, _stamps = task_file

    for note in (MULTI_LINE_NOTE, "plain note\nsecond line\nthird line\n",
                 "trailing newline\n"):
        autonomy._append_activity_log(991, note)

    unbulleted = [ln for ln in _under_heading(path) if not ln.startswith("- ")]
    assert unbulleted == [], (
        f"{len(unbulleted)} non-entry lines under the heading, which the "
        f"ACTIVITY_LOG_MAX_ENTRIES cap cannot count: {unbulleted}")
    assert len(_entries(_under_heading(path))) == 3, (
        "one line per call is the rule; three calls, three entries")


def test_the_fold_holds_for_the_in_memory_writer_too(task_file):
    """The MCP tool and the HTTP route append through `append_activity_line`, and
    its own docstring claims the scheduler's shape. A fold on one surface and not
    the other leaves two of the three writers able to leak the same lines, so the
    claim is pinned on the function that makes it."""
    path, stamps = task_file

    body = path.read_text(encoding="utf-8")
    body = autonomy.append_activity_line(body, MULTI_LINE_NOTE, stamps.now_str())
    path.write_text(body, encoding="utf-8")

    under = _under_heading(path)
    assert len(under) == 1, f"the in-memory writer still spills: {under}"
    assert under[0].startswith("- "), under[0]


# ── Clause 4: a repeated failure is one counted entry ────────────────────────


def test_repeated_failures_become_one_counted_entry(task_file):
    path, stamps = task_file

    # Two attempts of one loop: distinct stamps, distinct run ids, same failure.
    autonomy._append_activity_log(991, _failure_note("run_991_20260924_090000"))
    stamps.advance()
    autonomy._append_activity_log(991, _failure_note("run_991_20260924_090100"))
    stamps.advance()
    autonomy._append_activity_log(
        991, _failure_note("run_991_20260924_090200",
                           summary="Error output: Check stderr output for details"))

    failures = _failures(_under_heading(path))
    assert len(failures) == 1, (
        f"three attempts of one failure filled {len(failures)} lines, which is "
        f"how a task failing in a loop filled the whole retained window: {failures}")
    m = re.search(r"×(\d+)$", failures[0])
    assert m and m.group(1) == "3", (
        f"the entry must carry a count of what it stands for, got: {failures[0]}")
    # The count is the whole point only if the line still says when and where.
    assert stamps.now_str() in failures[0], (
        f"a collapsed entry keeps the latest stamp, not the first: {failures[0]}")
    assert "run_991_20260924_090200" in failures[0], failures[0]


def test_a_note_that_differs_beyond_stamp_and_run_id_still_appends(task_file):
    """The counter must not swallow a change of state: a different failure, and a
    run that finally worked, are each a fact worth a line of their own."""
    path, stamps = task_file

    autonomy._append_activity_log(991, _failure_note("run_991_20260924_090000"))
    stamps.advance()
    autonomy._append_activity_log(
        991, _failure_note("run_991_20260924_090100",
                           summary="StoreUnavailable: kg.sqlite is locked"))
    stamps.advance()
    autonomy._append_activity_log(991, "Run run_991_20260924_090200 — success (61s)")

    under = _under_heading(path)
    assert len(under) == 3, f"a changed note was absorbed by the counter: {under}"
    assert all(ln.startswith("- ") for ln in under), under
    assert "kg.sqlite is locked" in under[1], under[1]
    assert "success (61s)" in under[2], under[2]


def test_two_consecutive_successes_keep_their_own_lines(task_file):
    """The counter is gated on a failure, and that gate is a rule rather than an
    accident. Two successes are two facts about the schedule; collapsing a pair
    would delete the earlier run's stamp from the only per-task history anyone
    reads, and a loop of identical-looking successes is the silent-failure signal
    the log exists to surface."""
    path, stamps = task_file

    for run in ("run_991_20260924_090000", "run_991_20260924_090100"):
        autonomy._append_activity_log(991, f"Run {run} — success (61s)")
        stamps.advance()

    entries = _entries(_under_heading(path))
    assert len(entries) == 2, (
        f"identical successes collapsed into one counted entry: {entries}")
    assert not any("×" in ln for ln in entries), entries


def test_the_collapse_survives_the_stamp_inside_the_run_id(task_file):
    """The key must normalise the run id, not merely the entry's own stamp.

    `run_<task>_<YYYYMMDD>_<HHMMSS>` embeds a second timestamp, and a failure
    note spells the id twice — once as the run, once inside
    `[full: autonomy-runs/991/<run>.md]` — so a key that only flattened the
    leading stamp would still see two different lines and collapse nothing.
    """
    path, stamps = task_file

    first = _failure_note("run_991_20260924_090000")
    second = _failure_note("run_991_20260924_093000")
    assert first != second
    # Two real entries from those two runs: the leading stamp differs, the id
    # differs, twice over, and the failure text does not.
    line_a = f"- 2026-09-24T09:00:00Z: {first}"
    line_b = f"- 2026-09-24T09:30:00Z: {second}"
    assert line_a != line_b
    assert autonomy._activity_entry_key(line_a) == autonomy._activity_entry_key(line_b), (
        "the two notes from one failure loop must key identically or the "
        "×N branch can never fire")

    autonomy._append_activity_log(991, first)
    stamps.advance(minutes=30)
    autonomy._append_activity_log(991, second)
    assert len(_failures(_under_heading(path))) == 1
