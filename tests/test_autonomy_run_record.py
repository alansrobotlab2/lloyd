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


# ── #2619: a monitoring script's alarm must reach the run row ────────────────
#
# The knowledge-health report exits 2 when an alarm rail fires
# (`EXIT_ALARM = 2`). On this box that happened on the nights of 2026-10-10,
# 2026-10-11 and 2026-10-12 — and `workers.db` recorded `status: success` beside
# every one of them, because the runner's `status` is the verdict on the AGENT
# TURN and the script's exit code dies inside one of that turn's Bash calls. So
# the fleet tally (`failed = status != "success" or …`) saw nothing, and the only
# surviving signal was line 291 of a markdown file nothing reads on schedule.
#
# The fix is a sentinel the script prints and the runner reads out of the TOOL
# RESULT, recorded on the run as `script_alarm`. These nodes drive the real
# `run_task` over a stubbed event stream, then the real adapter and the real
# `normalize_result`, because the marker has to survive every hop to be worth
# anything: script stdout → tool result → turn loop → result dict → queue row.

#: What an alarm night's Bash tool result actually ends with, transcribed from
#: this box's 2026-10-12 run: the report's own hygiene row
#: (`~/lloyd-data/_pipeline/reflection/knowledge-health-2026-10-12.md:291`), the
#: stderr `ALARM:` block `_alarms()` builds for the untagged rail (its
#: `u.get("over_floor")` branch), then the sentinel. No line number: this very
#: diff inserts lines above that function, and a citation that moves with the
#: diff is the #2605 trap.
ALARMED_TAIL = (
    "| Untagged fact records (`entity:` blank or absent) | 3,206 records in 544 "
    "dirs / 830 files (2.13% of 150,868 records parsed by this walk; "
    "floor 0.5% — ABOVE FLOOR) |\n"
    "\nALARM:\n"
    "  - 3,206 fact records carry no entity tag — 2.13% of 150,868 records "
    "parsed, in 544 dirs, above the 0.5% floor: the contamination rail cannot "
    "see them and neither can a merge audit\n"
    "LLOYD_SCRIPT_EXIT: knowledge-health-report.py exit=2\n"
)

#: The same invocation on a clean night.
CLEAN_TAIL = (
    "Report written to knowledge-health-2026-10-13.md\n"
    "LLOYD_SCRIPT_EXIT: knowledge-health-report.py exit=0\n"
)


def _task(id: int = 60, **over) -> dict:
    task = {
        "id": id, "name": "Knowledge health report", "status": "up_next",
        "description": "Run the knowledge-health report.",
        "frequency": "daily", "model": "primary", "scheduled_at": None,
        "skill_name": "knowledge-health-report", "skill_path": None,
        "timeout_seconds": 600,
        "agent": "primary", "output_section": None, "notify_on_complete": True,
        "auto_advance": False, "preemptible": False, "last_run": None,
        "next_run": None, "output_format": None, "scheduled_by": None,
        "depends_on": None, "pipeline": None, "watching": None,
        "priority": "low", "activity": [],
    }
    task.update(over)
    return task


def _stream(tool_content: str, final_text: str) -> list[dict]:
    """One Bash call whose RESULT is `tool_content`, then one closing block.

    The final text is deliberately separate from the tool content: what the
    marker must NOT depend on is the agent's account of the run, so every node
    here chooses the two independently.
    """
    return [
        {"type": "tool_call", "call_id": "c1", "name": "Bash",
         "args_json": "{}", "summary": "Running the knowledge-health report"},
        {"type": "tool_result", "call_id": "c1", "name": "Bash",
         "content": tool_content, "is_error": False},
        {"type": "text_delta", "text": final_text},
        {"type": "assistant_message", "text": final_text, "tool_calls": [],
         "iteration": 1, "finish_reason": "stop"},
        {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}},
    ]


@pytest.fixture
def real_run(monkeypatch, tmp_path):
    """Drive the REAL `run_task` over a scripted stream, record written for real.

    Everything AROUND the turn is stubbed — the turn loop itself is the code
    under test, and the run record is written by the real writer because the
    claim is about what lands on disk and, one hop later, in `runs.meta_json`.
    """
    def _stub(task: dict, events: list[dict]):
        monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "t.md")
        monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
        monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
        monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
        monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
        monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
        monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
        monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False)

        async def _run_query(messages, options):
            for evt in events:
                yield evt

        class Opts:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        import app.harness as harness_mod
        import app.harness.mcp_pool as mcp_pool
        monkeypatch.setattr(harness_mod, "run_query", _run_query)
        monkeypatch.setattr(harness_mod, "RunOptions", Opts)
        monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
        monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "SYS")
        monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")

    return _stub


def _record_text(aut, task_id: int = 60) -> str:
    """The run record the real writer just produced, read back off disk."""
    files = sorted((aut / "runs" / str(task_id)).glob("*.md"))
    assert files, f"no run record under {aut / 'runs'} — the writer did not run"
    return files[-1].read_text(encoding="utf-8")


def test_an_alarm_exit_marks_the_run_record_without_touching_status(
        real_run, aut):
    """clause 1: the alarm becomes a field on the run, beside `status`.

    Asserted on the record FILE, not a captured kwarg, because the thing a
    downstream reader can see is the file: `status: success` must still be there
    (the turn really did work), and the alarm must be legible from the same
    front matter WITHOUT opening the report or reading the agent's prose.
    """
    import asyncio
    real_run(_task(), _stream(ALARMED_TAIL, "Report written for 2026-10-12.\n"))
    result = asyncio.run(autonomy.run_task(60))

    assert result["status"] == "success", "the turn's verdict must not be rewritten"
    assert result["meta"]["script_alarm"] == {"knowledge-health-report.py": 2}

    text = _record_text(aut)
    assert "script_alarm" in text, (
        f"the record front matter carries no alarm marker:\n{text[:400]}")
    front = text.split("---", 2)[1]
    assert "knowledge-health-report.py" in front, (
        f"the front-matter marker does not name the script:\n{front}")
    assert "exit" not in front or ": 2" in front, (
        f"the front-matter marker does not carry the exit code:\n{front}")
    assert result["script_alarm"] == {"knowledge-health-report.py": 2}, (
        "the result dict must carry the marker TOP LEVEL too: the adapter "
        "`workers/sources/scheduled_task.py` is a hand-written whitelist that "
        "lifts named keys from the result, and `claims` was severed exactly "
        "that way (#945)")


def test_a_clean_run_carries_no_marker_anywhere(real_run, aut):
    """clause 1's other half: no alarm, no marker — including a clean EXIT=0.

    The zero exit is the interesting input: `emit_exit_sentinel` prints on EVERY
    run, alarm or not. Recording it would make `script_alarm` mean "a script
    ran", which is not a signal; the marker means only "a script is red".
    """
    import asyncio
    real_run(_task(), _stream(CLEAN_TAIL, "Report written, all rails clear.\n"))
    result = asyncio.run(autonomy.run_task(60))

    assert result["script_alarm"] == {}, (
        "the handoff key is always present so the adapter's `in result` test "
        "sees a producer that honours the marker; its EMPTY value is the claim")
    text = _record_text(aut)
    assert "script_alarm" not in text, (
        "a clean run grew a marker, so absence can no longer be read as \"no "
        f"alarm was seen\":\n{text[:400]}")


def test_the_marker_comes_from_the_scripts_own_line_and_not_from_the_prose(real_run):
    """clause 2: prose that MENTIONS the alarm records nothing; silence records it.

    Both directions in one node because they are one claim — the marker is the
    script's, not the model's account:

    * an agent that writes "exit code 2" and "ALARM" in its closing block while
      the tool result holds no sentinel records NO `script_alarm`. (Its
      `silent_failure_indicators` DOES fire, which is the point: that prose
      channel exists, and it is not this one — it missed all three real alarm
      nights because the model did not quote the exit code.)
    * the alarm night's real shape — sentinel in the result, closing block that
      never mentions it — DOES record the marker.
    """
    import asyncio

    real_run(_task(), _stream(
        "report done\n", "ALARM fired, script exited with exit code 2.\n"))
    prose_only = asyncio.run(autonomy.run_task(60))
    assert prose_only["script_alarm"] == {}, (
        "the marker was set from the agent's PROSE, which is exactly the "
        "channel that reported 0 on all three alarm nights")
    assert prose_only["meta"]["silent_failure_indicators"] >= 1, (
        "the prose channel is the one this node refuses to trust; if it no "
        "longer fires at all, the pair below proves nothing")

    real_run(_task(), _stream(ALARMED_TAIL, "Report written for 2026-10-12.\n"))
    alarm = asyncio.run(autonomy.run_task(60))
    assert alarm["script_alarm"] == {"knowledge-health-report.py": 2}


def test_the_marker_survives_the_adapter_into_the_runs_row(real_run, tmp_path,
                                                           monkeypatch):
    """The chain, not the pieces: `run_task`'s real result → the real adapter → meta_json.

    `workers/sources/scheduled_task.py`'s `execute` ends in a hand-written
    whitelist of the keys it copies out of `run_task`'s result. `claims` was
    severed one hop upstream of the verifier that way (#945), so a marker that
    only ever lived in `run_task`'s own dict would die here — and this is the
    seam the graph is blind across (runner → pool → queue). `normalize_result`
    is the real pool function, and `meta` is the only field it writes to
    `runs.meta_json`, which is the half the health endpoint parses.
    """
    import asyncio
    from types import SimpleNamespace

    from workers.pool import normalize_result
    from workers.sources import scheduled_task

    real_run(_task(), _stream(ALARMED_TAIL, "Report written for 2026-10-12.\n"))
    run_task_result = asyncio.run(autonomy.run_task(60))
    assert run_task_result["script_alarm"], (
        "the marker was not produced at all — this node would otherwise pass "
        "on an empty meta and prove nothing")

    async def _run_task(task_id, max_duration=None):
        return run_task_result

    monkeypatch.setattr(autonomy, "run_task", _run_task)
    monkeypatch.setattr(scheduled_task, "_vllm_healthy",
                        lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(scheduled_task, "run_trigger",
                        lambda _s: SimpleNamespace(__enter__=lambda _s: None,
                                                   __exit__=lambda *a: False),
                        raising=False)
    async def _notify(*_a, **_k):
        return None
    monkeypatch.setattr(scheduled_task, "_discord_notify_task_complete", _notify,
                        raising=False)
    monkeypatch.setattr(scheduled_task, "_find_task_file",
                        lambda _tid: tmp_path / "60-knowledge-health.md", raising=False)
    monkeypatch.setattr(scheduled_task, "_parse_task_file",
                        lambda _p: {"id": "60", "name": "Knowledge health report",
                                    "status": "up_next", "frequency": "daily",
                                    "notify_on_complete": False}, raising=False)
    from workers import sources
    monkeypatch.setattr(sources, "get_sources_config", lambda: {})

    item = SimpleNamespace(
        id=7001, source="scheduled-task",
        payload={"task_id": "60", "attempt": 1, "max_attempts": 2},
        run_id="run_scheduled-task_20261012_110011_a391f9")
    out = asyncio.run(scheduled_task.execute(item))
    norm = normalize_result(item, out)

    assert norm["status"] == "success"
    assert norm["meta"].get("script_alarm") == {"knowledge-health-report.py": 2}, (
        f"the marker did not survive the adapter into what becomes "
        f"`runs.meta_json`; meta was {norm['meta']}")


def test_the_producer_s_real_key_list_is_the_one_the_adapter_test_pins(real_run, aut,
                                                                       monkeypatch):
    """#3515 clause 4: the list `scheduled_task`'s contract test checks against is
    checked back against the producer that emits it, in both directions.

    The whitelist rotted three times (#945, #623, #2619) because nothing asked
    `run_task` what it actually returns. So this drives the REAL success path and
    the REAL `_record_failure`, and the two results' keys together must equal
    `WS.PRODUCER_KEYS` exactly — a key the producer adds fails with "not on the
    list", and a name the list keeps after the producer stopped sending it fails
    too, because a stale exclusion entry is a hole waiting for the next rename.

    `_evidence_pilot` and `_acceptance_grade` are forced because both keys are
    conditional on the success path (pilot task ids; a grade that exists), and the
    check below is exact — leaving them out would put two names on the list that no
    real run ever produced.
    """
    import asyncio
    import datetime

    import test_workers_sources as WS  # tests/ is on sys.path, as in board_presence

    monkeypatch.setattr(autonomy, "_evidence_pilot", lambda _tid: True, raising=False)
    monkeypatch.setattr(autonomy, "_acceptance_grade",
                        lambda *_a, **_k: {"grade": "pass", "checked": ["record"]},
                        raising=False)
    real_run(_task(), _stream(CLEAN_TAIL, "Report written, nothing above the floor.\n"))
    ok = asyncio.run(autonomy.run_task(60))
    assert ok["success"] is True, f"the scripted run did not succeed: {ok.get('error')}"

    failed = asyncio.run(autonomy._record_failure(
        _task(), 60, "run_60_20261030_000000", "",
        datetime.datetime.now(datetime.timezone.utc),
        summary="probe failure", body="", kind="task", extra=None, alert=False))

    emitted = set(ok) | set(failed)
    assert emitted - WS.PRODUCER_KEYS == set(), (
        f"run_task now returns {sorted(emitted - WS.PRODUCER_KEYS)}, which is on "
        f"neither the carried list nor the exclusion list in "
        f"tests/test_workers_sources.py — the adapter drops an unnamed key silently, "
        f"which is exactly how #945, #623 and #2619 each happened")
    assert WS.PRODUCER_KEYS - emitted == set(), (
        f"the named list still carries {sorted(WS.PRODUCER_KEYS - emitted)} that "
        f"neither producer exit emits: a name nothing sends is how an exclusion "
        f"list hides a rename")
    assert set(WS.PRODUCER_KEYS) - set(ok) == {"error", "failure_kind", "disabled"}, (
        f"the keys a SUCCESS run does not send changed from the three the "
        f"`_record_failure` dict adds ({sorted(set(WS.PRODUCER_KEYS) - set(ok))}); "
        f"the failure exit is the only thing guarding them")

    # The three keys this whole seam has been about are on the PRODUCER side too,
    # and on the side of the list that says they must cross.
    for name in ("claims", "acceptance_grade", "script_alarm"):
        assert name in ok, f"the producer stopped sending {name!r} at all"
        assert name in WS.PRODUCER_KEYS_CARRIED, (
            f"{name!r} moved to the exclusion list — that is a severance signed off "
            f"by the test instead of caught by it")
    assert set(ok) | set(failed) == WS.PRODUCER_KEYS


# ── #2619: the script side of the seam ───────────────────────────────────────
#
# Everything above fixes the shape of the string; none of it fails if
# `knowledge-health-report.py` stops printing it. The emitter and the scanner are
# one contract across a process boundary — a separate script, its own interpreter
# — which is exactly the seam the code graph is blind across, so it gets a test
# that runs BOTH sides: the real `emit_exit_sentinel` writing real bytes, and the
# real `EXIT_SENTINEL_RE` reading them back.

#: The emitter's module, loaded the way `tests/test_knowledge_health_stale_facts.py`
#: loads it (the filename has hyphens, so it is not importable by path).
def _khr():
    import importlib.util
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "khr_2619", root / "scripts" / "memory" / "knowledge-health-report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["khr_2619"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_emitter_s_bytes_are_the_scanner_s_input(capsys):
    """One contract, two processes: what the script prints is what the runner parses.

    Both streams, because the invocation that hid the exit code
    (`... 2>&1 | tail -80`) merges them, and a marker on one stream only is a
    marker that depends on how the script happened to be called. And the alarm
    code is the only one the scanner answers for: `emit_exit_sentinel(…, 0)` runs
    on every clean night and must scan to nothing, or the marker would mean "a
    script ran".
    """
    khr = _khr()

    khr.emit_exit_sentinel("knowledge-health-report.py", 2)
    cap = capsys.readouterr()
    line = "LLOYD_SCRIPT_EXIT: knowledge-health-report.py exit=2"
    assert line in cap.out, f"not on stdout: {cap.out[-200:]!r}"
    assert line in cap.err, f"not on stderr: {cap.err[-200:]!r}"
    assert autonomy._scan_exit_sentinels(cap.out + cap.err) == \
        {"knowledge-health-report.py": 2}, (
        "the runner's regex no longer reads the emitter's line — the two ends "
        "of this contract drift independently and only a test like this one "
        "catches it")

    khr.emit_exit_sentinel("knowledge-health-report.py", 0)
    cap = capsys.readouterr()
    assert "exit=0" in cap.out, "the emitter stopped marking clean nights"
    assert autonomy._scan_exit_sentinels(cap.out + cap.err) == {}, (
        "a zero exit marked the run, so the field would mean 'a script ran'")


def test_the_report_still_exits_two_and_still_prints_the_marker():
    """#2619 changes how the alarm is READ, never whether it is RAISED.

    `tests/test_knowledge_health_stale_facts.py` already pins
    `(EXIT_OK, EXIT_ALARM) == (0, 2)`, so the pair asserted here is the other
    half of that: `main` must not gain a second `return EXIT_ALARM` that leaves
    the process without emitting — that is the exact shape of the bug, an alarm
    that raises but cannot be seen. The sentinel call is the last thing on the
    alarm path, and every exit goes through `rc`.
    """
    import inspect

    khr = _khr()
    src = inspect.getsource(khr.main)
    assert "emit_exit_sentinel(" in src, (
        "`main` no longer emits the marker, so the run row goes blind again "
        "while the report itself still looks right")
    assert "return EXIT_ALARM" not in src, (
        "an early `return EXIT_ALARM` bypasses the emitter — the alarm raises "
        "and nothing downstream can see it")
    assert src.index("alarms = _alarms(") < src.index("emit_exit_sentinel("), (
        "the marker is emitted before the alarms are computed, so it can only "
        "ever report a code the rails did not set")
