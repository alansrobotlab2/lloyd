"""The fast-failure alert (#1209): three sub-30s failures say so, once, in the daily note.

A run that reaches the engine and fails at its work takes minutes. A run that
comes back in 0.3 s was refused on the way in — task #85's three
`vLLM returned 404: The model `eco` does not exist` refusals on 2026-09-16 were
0.3 s / 5.6 s / 0.3 s apart 38 minutes, and the only trace on the box was the
task row reading `status: failed` with `next_run` nulled, which both stall
alarms skip by construction, plus a `discord_alert` that this box cannot deliver
because `discord.home_channel` is null. These tests pin the two properties that
were missing: a duration-shaped signal, and a surface that works here.
"""
import asyncio
import datetime as dt
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autonomy  # noqa: E402


@pytest.fixture
def aut(tmp_path, monkeypatch):
    """Isolated task dir, runs dir, skill file — and a redirected daily note.

    `LLOYD_DAILY_NOTE_DIR` is what keeps a test that fails a task three times
    from writing "Autonomy #1 failed 3 times in a row" into the live
    `~/obsidian/memory/<today>.md`; the conftest default already points it at
    scratch, and pinning it to this test's own `tmp_path` is what lets the
    assertions below read the note back.
    """
    autonomy_dir = tmp_path / "autonomy"
    autonomy_dir.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", autonomy_dir)
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LLOYD_DAILY_NOTE_DIR", str(tmp_path / "notes"))
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n")
    monkeypatch.setattr(autonomy, "_SKILL_FOR_TESTS", str(skill), raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "sys",
                        raising=False)
    return autonomy


def write_task(aut, task_id, **fm):
    base = {
        "id": task_id, "name": f"task{task_id}", "type": "autonomy",
        "status": "up_next", "frequency": "daily", "priority": "medium",
        "skill_name": aut._SKILL_FOR_TESTS, "timeout_seconds": 2,
        "max_retries": 5, "failure_count": 0,
    }
    base.update(fm)
    base = {k: v for k, v in base.items() if v is not None}
    path = aut.AUTONOMY_DIR / f"{task_id}-task{task_id}.md"
    path.write_text(f"---\n{yaml.dump(base)}---\n\nbody\n\n## Activity Log\n")
    return path


def read_task(aut, task_id):
    return aut._parse_task_file(aut._find_task_file(task_id))


async def _fail(aut, task, *, run_id, seconds, kind="task"):
    """Record one real failure through the production path, `seconds` long.

    `started_dt` is back-dated rather than slept, because the quantity under
    test is the duration the record carries and a 45-second negative case
    should not cost 45 seconds of suite time. Everything else — the record, the
    streak read off it, the cooldown, the note — is production code.

    `run_id` is supplied per call rather than generated, because a run id has
    second resolution and a back-dated batch would otherwise collide.
    """
    started = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)
    return await aut._record_failure(
        task, task["id"], run_id, started.isoformat(), started,
        summary="vLLM returned 404: The model `eco` does not exist.",
        body="## Prompt\n\n(test)\n", kind=kind)


def _note_file(aut):
    """Today's daily note, by the same LA-date filename the writer uses."""
    today = dt.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    return autonomy._daily_note_dir() / f"{today}.md"


def _alert_lines():
    path = _note_file(autonomy)
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if "Autonomy #" in line]


# ── clause 4: three fast failures write one line; three slow ones write none ──

async def test_three_fast_failures_write_one_line_naming_the_task_and_durations(aut, monkeypatch):
    """Clause 4, end to end: three instant failures of a daily task, via run_task.

    Driven through `run_task` rather than the recorder so the assertion crosses
    the same seam `scheduled_task.execute` uses: the exception from the engine,
    the run record it leaves, the streak read off those records, the note.
    `max_retries` is 5 so the task is NOT disabled at the third failure — the
    line is the alert's own doing, not a side effect of the disable, and a task
    with a larger budget still says so at three.

    The sleeps keep the three run ids distinct: a run id has second resolution,
    and three instant failures inside one second would be one file.
    """
    async def _boom(messages, options):
        raise RuntimeError("vLLM returned 404: The model `eco` does not exist.")
        yield  # pragma: no cover - makes this an async generator

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _boom)
    write_task(aut, 1, name="Nightly Secondary Routing Eval", max_retries=5)

    for _ in range(3):
        await aut.run_task(1)
        await asyncio.sleep(1.02)

    lines = _alert_lines()
    assert len(lines) == 1, f"exactly one alert line, got {len(lines)}: {lines}"
    line = lines[0]
    assert "#1" in line and "Nightly Secondary Routing Eval" in line, (
        f"the line must name the task: {line}")
    assert "under 30s" in line, f"the line must name the bound: {line}"
    # The durations, not just the count: "failed 3 times in under 30s each"
    # without the numbers is a claim, "(0.2s, 0.2s, 0.2s)" is evidence, and the
    # whole point of the line is that the SHAPE of the failure is the finding.
    assert re.search(r"\(\d+\.\ds, \d+\.\ds, \d+\.\ds\)", line), (
        f"the line must carry each duration: {line}")
    # Still retrying, so the line cannot be the disable's doing.
    assert read_task(aut, 1)["status"] == "up_next"


async def test_three_failures_each_over_30s_write_no_line(aut):
    """Clause 4's other half: a task failing at real work is not this alert.

    45 s each, three in a row — long enough that the run reached the engine and
    did something, which is the case whose remedy is reading the run record, not
    a line in the daily note. Without the duration bound this alert would fire
    on every ordinary nightly timeout and be unread within a week.
    """
    write_task(aut, 1, name="Nightly Knowledge Write", max_retries=5)
    task = read_task(aut, 1)

    for idx in range(3):
        result = await _fail(aut, task, run_id=f"run_1_20260921_00000{idx}", seconds=45)
        assert result["success"] is False

    assert _alert_lines() == []


async def test_a_slow_failure_resets_the_fast_streak(aut):
    """The streak is CONSECUTIVE and duration-shaped, not a count of failures.

    0.3, 0.3, 45, 0.3, 0.3 is two fast failures at the end of a mixed history —
    a run that took 45 s in the middle proves the engine was answering then, so
    the three-file window that follows is not the same evidence. And the sixth
    failure, which makes three fast in a row, must be the one that speaks.
    """
    write_task(aut, 1, name="Nightly Vault Maintenance", max_retries=9)
    task = read_task(aut, 1)

    for idx, seconds in ((0, 0.3), (1, 0.3), (2, 45.0), (3, 0.3), (4, 0.3)):
        await _fail(aut, task, run_id=f"run_1_20260921_0000{idx}", seconds=seconds)
    assert _alert_lines() == [], "two fast failures must not alert"

    await _fail(aut, task, run_id="run_1_20260921_00005", seconds=0.3)
    lines = _alert_lines()
    assert len(lines) == 1, f"the third consecutive fast failure alerts: {lines}"
    assert "0.3s" in lines[0]


async def test_a_successful_run_between_two_fast_failures_does_not_alert(aut):
    """CONSECUTIVE means across every run, not across the failures only.

    The first cut of `_fast_failure_streak` collected the records that qualified
    and measured the tail of that filtered list, so `0.3s, 0.3s, SUCCESS, 0.3s`
    read as a streak of three and the daily note said "3 consecutive failures"
    about a task that had succeeded in the middle of it. A filter cannot express
    consecutiveness; stopping can. This is the case the filter got wrong and the
    walk now refuses: a success is the strongest counter-evidence there is — the
    task reached the engine and did its work — so the two failures before it are
    history, not a streak in progress, and the reader must restart from zero.
    """
    write_task(aut, 1, name="Nightly Secondary Routing Eval", max_retries=9)
    task = read_task(aut, 1)

    await _fail(aut, task, run_id="run_1_20260921_400000", seconds=0.3)
    await _fail(aut, task, run_id="run_1_20260921_400001", seconds=0.3)
    assert _alert_lines() == [], "two fast failures must not alert on their own"

    # A real success, written by the same writer the success path uses, so the
    # reader is shown production's own shape and not a fixture's guess at it.
    now = dt.datetime.now(dt.timezone.utc)
    aut._write_run_record(
        task_id=1, run_id="run_1_20260921_400002", status="success",
        started_at=(now - dt.timedelta(seconds=210)).isoformat(),
        completed_at=now.isoformat(), duration_seconds=210.0,
        summary="did the thing", body="## Prompt\n\n(work)\n")

    await _fail(aut, task, run_id="run_1_20260921_400003", seconds=0.3)
    await _fail(aut, task, run_id="run_1_20260921_400004", seconds=0.3)
    assert _alert_lines() == [], (
        "the success resets the count: four fast failures in the directory are "
        "two fast failures in the streak, and two is not three")

    await _fail(aut, task, run_id="run_1_20260921_400005", seconds=0.3)
    lines = _alert_lines()
    assert len(lines) == 1, f"the third fast run AFTER the success alerts: {lines}"


async def test_an_infra_failure_between_fast_failures_breaks_the_streak(aut):
    """Not-counting and breaking are different properties, and both are pinned.

    `test_infra_failures_do_not_write_the_line` shows an outage record does not
    ADD to a streak; this shows it ENDS one, which is the half a filter got
    silently wrong in the same way. The order matters for what the line then
    claims: after the eleven-hour empty-server window of 2026-09-01, a task that
    goes on to refuse three times in 0.3 s is describing something the outage did
    not do for it — so the alert speaks only of the post-outage sequence, counted
    from zero, and it is the third failure after recovery that says so.
    """
    write_task(aut, 1, name="Nightly Knowledge Write", max_retries=9)
    task = read_task(aut, 1)

    await _fail(aut, task, run_id="run_1_20260921_500000", seconds=0.3)
    await _fail(aut, task, run_id="run_1_20260921_500001", seconds=0.3)

    started = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=0.3)
    await aut._record_failure(task, 1, "run_1_20260921_500002",
                              started.isoformat(), started,
                              summary="ConnectError", body="## Prompt\n\nx\n",
                              kind="infra")

    await _fail(aut, task, run_id="run_1_20260921_500003", seconds=0.3)
    await _fail(aut, task, run_id="run_1_20260921_500004", seconds=0.3)
    assert _alert_lines() == [], "the outage record broke the earlier pair"

    await _fail(aut, task, run_id="run_1_20260921_500005", seconds=0.3)
    lines = _alert_lines()
    assert len(lines) == 1, f"three fast failures after the outage alert: {lines}"
    assert "0.3s" in lines[0]


async def test_a_fourth_fast_failure_does_not_write_a_second_line(aut):
    """`Exactly one alert line`: one per streak, not one per retry.

    The streak reader cannot distinguish "third" from "fourth" by counting up to
    three and firing — it has to compare the length, so this is the test that
    keeps four fast failures from printing four identical lines into the note a
    human reads.
    """
    write_task(aut, 1, name="Nightly Reflection Signals", max_retries=9)
    task = read_task(aut, 1)

    for idx in range(4):
        await _fail(aut, task, run_id=f"run_1_20260921_1000{idx}", seconds=0.3)

    assert len(_alert_lines()) == 1


async def test_infra_failures_do_not_write_the_line(aut):
    """The filter is the failure KIND as well as the duration.

    A 0.3 s `failure_kind: infra` failure is the outage signature — on
    2026-09-01 every task returned empty for eleven hours — and its remedy is
    patience, not a look at the task file. If those counted, the worst night of
    the year would fill the daily note with thirty-one lines that all say
    "the server was down", and the one line that means "this task is
    misconfigured" would be among them.
    """
    write_task(aut, 1, name="Nightly Knowledge Write", max_retries=9)
    task = read_task(aut, 1)

    for idx in range(3):
        result = await _fail(aut, task, run_id=f"run_1_20260921_2000{idx}",
                             seconds=0.3)
        assert "infra" not in str(result["failure_kind"])

    assert len(_alert_lines()) == 1  # the three above are kind=task

    # Now the same durations booked as an outage: no new line.
    started = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=0.3)
    await aut._record_failure(task, 1, "run_1_20260921_20009",
                              started.isoformat(), started,
                              summary="ConnectError", body="## Prompt\n\nx\n",
                              kind="infra")
    assert len(_alert_lines()) == 1


# ── clause 5: the line is written on a box where Discord is not configured ────

_ROOT = Path(__file__).resolve().parent.parent


def _discord_block_on_disk() -> dict:
    """The `discord:` block of the `config.yaml` this box boots with.

    Read from the file, not from a CONFIG the test could have written: clause 5
    is a claim about this machine's transport, so its premise has to be
    re-measurable against the artifact that decides it.
    """
    return (yaml.safe_load((_ROOT / "config.yaml").read_text()) or {}).get("discord") or {}


async def test_the_line_is_written_with_discord_home_channel_null(aut):
    """Clause 5: the note lands on a box whose Discord transport is dead.

    `app/discord_notify.discord_alert` returns after a warning when
    `discord.home_channel` or the token is unset (app/discord_notify.py:49-56) —
    which is precisely why #85's disable alert fired three times in 38 minutes
    and reached nobody. The premise is read out of `config.yaml`, the file that
    decides it, and is NOT installed by this test: the first cut asserted
    `CONFIG["discord"]["home_channel"] is None` two statements after
    `monkeypatch.setitem(discord, "home_channel", None)`, so it graded the
    fixture and could not fail. A premise the test itself writes is not a premise.
    If a person takes the decision #1209 leaves open and configures Discord, this
    assertion fails and says which premise went stale — that is the tripwire
    working, because the clause is written in terms of that null.

    The run is the 09-16 shape exactly: three sub-30s failures, `max_retries: 3`,
    so the THIRD failure also disables the row and drives the existing disable
    alert down the dead transport while the note has to land beside it.
    """
    from app.discord_notify import _discord_token

    disk = _discord_block_on_disk()
    assert disk.get("home_channel") is None, (
        "clause 5's premise, read from config.yaml: discord.home_channel is null "
        f"here; it reads {disk.get('home_channel')!r} now, so the human decision "
        "#1209 recorded has been taken and this test's premise is stale")
    assert not _discord_token(), (
        "and no token resolves, the other half of the dead end")

    write_task(aut, 1, name="Nightly Secondary Routing Eval", max_retries=3)

    for idx in range(3):
        # Re-read each time: `_record_failure` derives the new failure_count from
        # the dict it is handed, so a stale one would pin the streak's first
        # value and the disable below would never fire.
        task = read_task(aut, 1)
        await _fail(aut, task, run_id=f"run_1_20260921_3000{idx}", seconds=0.3)

    lines = _alert_lines()
    assert len(lines) == 1, f"the daily note line is the working surface: {lines}"
    assert "#1" in lines[0] and "0.3s" in lines[0]
    assert read_task(aut, 1)["status"] == "failed", (
        "the task did self-disable — the note is not standing in for the disable")


async def test_the_fast_failure_line_has_no_discord_code_path_at_all(aut, monkeypatch):
    """The clause's real claim, stated so it cannot go stale on a config edit.

    Box-level premises age the day a human configures the transport; this one is
    about the writer, not the box. `discord_alert` becomes a recorder and the task
    gets a retry budget the streak does not exhaust (`max_retries: 9` against
    three failures), so the ONLY event that could put anything on Discord is the
    new alert itself. It puts nothing there, and the note still gets its line —
    which is what "must not depend on the Discord route" has to mean before
    anyone routes the next alert the same way.
    """
    posted: list[str] = []

    async def _recorder(msg, *a, **k):
        posted.append(str(msg))

    monkeypatch.setattr("app.discord_notify.discord_alert", _recorder, raising=False)
    write_task(aut, 1, name="Nightly Reflection Signals", max_retries=9)
    task = read_task(aut, 1)

    for idx in range(3):
        await _fail(aut, task, run_id=f"run_1_20260921_6000{idx}", seconds=0.3)

    assert len(_alert_lines()) == 1, "the note landed"
    assert posted == [], f"the fast-failure alert must not use Discord: {posted}"

