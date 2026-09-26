"""A run record is a claim about a file, so the writer has to read the file back.

#1567 is the read side of a gap the 2026-09-22 data-root cutover made visible:
`autonomy-runs/77/` is gone, yet `~/obsidian/autonomy/77-weekly-backlog-hygiene.md`
cites `autonomy-runs/77/run_77_20260922_150638.md`, and `~/lloyd-data/workers.db`
has no `runs` row for `task_id='77'` — a job invisible to both stores the health
route reads. Triage proved the cause was the cutover and that no program prunes
`autonomy-runs/<id>/` whole, and that every one of the 83 citations dated after
the cutover resolves: this is not an active leak. What is live is that the write
is 100% trusted and 0% verified. `_write_run_record` returned a `Path`, all three
call sites discarded it, and `workers/sources/scheduled_task.py` built
`artifact_path` as a relative string from `run_id` without touching disk. So the
next root move, mount failure or permissions change strands a run's record and
the Activity Log still prints a path a reader cannot open.

These tests pin the writer's side: `_write_run_record` answers `None` — never a
path — when `AUTONOMY_RUNS_DIR/<task_id>/<run_id>.md` is not a readable non-empty
file after the call, and its answer is a measurement of the bytes on disk rather
than the absence of an exception. The read side (what a run does with a `None`) is
`tests/test_autonomy_activity_log.py`; the queue-row side is in
`tests/test_workers_sources.py`.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import autonomy  # noqa: E402


@pytest.fixture
def aut(tmp_path, monkeypatch):
    """Runs directory pointed at scratch. The real one is production evidence."""
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    return tmp_path


def _write(task_id: int = 77, run_id: str = "run_77_20260101_000000", **over):
    kwargs = dict(status="success", started_at="2026-01-01T00:00:00Z",
                  completed_at="2026-01-01T00:00:05Z", duration_seconds=5.0,
                  summary="done", body="## Response\n\nworked\n")
    kwargs.update(over)
    return autonomy._write_run_record(task_id=task_id, run_id=run_id, **kwargs)


# ── the clean case: an answer that matches disk ───────────────────────────────

def test_a_record_that_lands_on_disk_returns_its_path_and_it_is_readable(aut):
    """The positive control for the whole file. A verifier that only ever fires
    cannot tell a real write from a `None`, and neither can the reader."""
    path = _write()

    assert path is not None, "an ordinary write must not be reported as lost"
    assert path == aut / "runs" / "77" / "run_77_20260101_000000.md"
    assert path.is_file(), path
    assert path.stat().st_size > 0, path
    assert "worked" in path.read_text(encoding="utf-8")


def test_the_verdict_is_measured_from_disk_not_from_the_write_call(aut, monkeypatch):
    """The write must not be trusted because it did not raise. Here it succeeds
    and then a reader is shown the bytes: the returned path is checked as a
    reader would check it, and agrees with the file."""
    written: list[Path] = []
    real = Path.write_text

    def spy(self, content, *a, **kw):
        out = real(self, content, *a, **kw)
        written.append(self)
        return out

    monkeypatch.setattr(Path, "write_text", spy)
    path = _write(task_id=700, run_id="run_700_20260101_000000")

    assert written == [path], written
    assert path is not None
    on_disk = path.read_text(encoding="utf-8")
    assert on_disk.startswith("---") and "worked" in on_disk


# ── the failure half: a write that leaves nothing to read ─────────────────────

def test_a_write_that_leaves_no_file_answers_none_and_logs_it(aut, monkeypatch, caplog):
    """The bug in one line: `path.write_text` raised, the exception was caught
    upstream, and the returned `Path` was still printed as the run's record. The
    answer must be `None` — a path is what a reader is told to open."""
    def boom(self, content, *a, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", boom)
    with caplog.at_level("ERROR"):
        path = _write(task_id=77, run_id="run_77_20260101_000001")

    assert path is None, path
    assert "run_77_20260101_000001" in caplog.text
    assert not (aut / "runs" / "77" / "run_77_20260101_000001.md").exists()


def test_a_directory_that_cannot_be_created_answers_none(aut, monkeypatch):
    """`mkdir` sat outside the try/except that caught the write, so a read-only
    or quota-blocked runs root escaped the verification entirely."""
    def boom(self, *a, **kw):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(Path, "mkdir", boom)
    assert _write(task_id=78, run_id="run_78_20260101_000000") is None


def test_a_file_written_zero_bytes_is_not_a_record(aut, monkeypatch):
    """A truncated write — the classic symptom of a full mount — leaves a 0-byte
    file that `exists()` answers true for. It is the failure this item is about,
    so the check has to be about size, not existence."""
    monkeypatch.setattr(Path, "write_text", lambda self, content, *a, **kw: None)
    assert _write(task_id=79, run_id="run_79_20260101_000000") is None


def test_a_file_that_cannot_be_read_back_is_not_a_record(aut, monkeypatch):
    """`mode=000` covers a wrong umask or an ACL on a mounted root: the bytes are
    there and still nobody can open the record, which is the claim being made."""
    real = Path.write_text

    def unreadable(self, content, *a, **kw):
        out = real(self, content, *a, **kw)
        if os.geteuid() != 0:      # root reads mode 000 anyway; skip on a root test run
            os.chmod(self, 0)
        return out

    monkeypatch.setattr(Path, "write_text", unreadable)
    if os.geteuid() == 0:
        pytest.skip("running as root: mode 000 is still readable")
    assert _write(task_id=80, run_id="run_80_20260101_000000") is None


def test_readable_distinguishes_the_three_kinds_of_absence(aut):
    """Three ways a reader is wrong, one of which is `exists()` answering false
    about a path that was never meant to be opened (a directory) and one of which
    is `exists()` answering TRUE about nothing (0 bytes). The positive control at
    the end is what makes the 0s above a measurement and not a broken probe."""
    base = aut / "runs" / "81"
    missing = base / "run_81_absent.md"
    empty = base / "run_81_empty.md"
    a_directory = base
    good = base / "run_81_good.md"
    base.mkdir(parents=True)
    empty.write_text("", encoding="utf-8")
    good.write_text("---\nstatus: success\n---\n\nbody\n", encoding="utf-8")

    assert autonomy._run_record_readable(missing) is False
    assert autonomy._run_record_readable(empty) is False
    assert autonomy._run_record_readable(a_directory) is False
    assert autonomy._run_record_readable(good) is True, "positive control"


# ── the reader half: queue-health-check Step 2 (#1567 clause 4) ───────────────
#
# The watchdog's Step 2 used to be one pipeline — `ls -t … | head -1 | xargs sed`
# — which fails twice on this box. For task #77, whose `autonomy-runs/77/` does not
# exist, it errors with `No such file or directory` and the report says nothing; for
# a task whose directory exists but holds only newer records, it silently quotes a
# DIFFERENT run than the one under investigation and presents it as the diagnosis.
# These read the live vault, deliberately unmarked: the gate runner hardcodes
# `-m "not live_vault"`, so a marked check pins nothing (same trade
# `tests/test_brief_triage_clock_skill.py` documents). A missing vault fails.

SKILL = Path.home() / "obsidian" / "skills" / "queue-health-check" / "SKILL.md"


def _step2() -> str:
    if not SKILL.exists():
        pytest.fail(f"the live vault is not at {SKILL.parent.parent.parent}")
    text = SKILL.read_text(encoding="utf-8")
    start = text.index("## Step 2")
    end = text.index("## Step 3")
    return text[start:end]


def test_step2_no_longer_reads_whichever_record_happens_to_be_newest():
    """`ls -t | head -1` is the defect: it answers a question about one run with a
    record from another. The step must name the run under investigation and take
    the path from the queue's own row."""
    step = _step2()

    assert "ls -t" not in step, step
    assert "run_id" in step and "artifact_path" in step, step


def test_step2_reports_something_when_the_task_has_no_runs_directory():
    """The clause: a missing `autonomy-runs/<id>/` must produce a finding, not a
    shell error. Requires the fallback to name what it will report AND where, so a
    reader gets a location rather than a stack of `No such file or directory`s."""
    step = _step2()

    assert "No such file or directory" in step, \
        "the step has to acknowledge the error it used to emit"
    assert "no run record on disk" in step, step
    assert "fall back" in step.lower(), step
    for fallback in ("meta_json", "summary", "attempts"):
        assert fallback in step, f"fallback source missing: {fallback}"


def test_step2_says_the_record_path_is_relative_to_the_data_root():
    """`artifact_path` is DATA_ROOT-relative. A check of it against the process cwd
    reports nearly every real record missing — triage measured 96 of 97 — which is
    a false alarm bad enough to get this step switched off. The step now states the
    prefix."""
    step = _step2()

    assert "~/lloyd-data/" in step, step
    assert "DATA_ROOT-relative" in step, step
