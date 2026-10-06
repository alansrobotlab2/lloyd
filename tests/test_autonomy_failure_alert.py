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
import inspect
import json
import logging
import os
import re
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import autonomy  # noqa: E402


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
    monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "sys",
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
    """Clause 5: the note lands on a box whose Discord transport is unconfigured.

    History, not live behaviour: before #1592 an unconfigured
    `app/discord_notify.discord_alert` returned after a warning, which is when #85's
    disable alert fired three times in 38 minutes and reached nobody — and that was
    the reason this alert was written straight to the daily note instead of routed
    through it. Since #1592 the same call appends the refused alarm to the note
    itself (app/discord_notify.py:131-134), so the null below is a premise about the
    transport this box ships with, not a claim that an alarm goes unseen.

    The premise is read out of `config.yaml`, the file that decides it, and is NOT
    installed by this test: the first cut asserted
    `CONFIG["discord"]["home_channel"] is None` two statements after
    `monkeypatch.setitem(discord, "home_channel", None)`, so it graded the
    fixture and could not fail. A premise the test itself writes is not a premise.
    If a person takes the decision #1209 leaves open and configures Discord, this
    assertion fails and says which premise went stale — that is the tripwire
    working, because the clause is written in terms of that null.

    The run is the 09-16 shape exactly: three sub-30s failures, `max_retries: 3`,
    so the THIRD failure also disables the row and drives the existing disable
    alert through the unconfigured transport while the fast-failure note has to
    land beside it.
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


# ── #2316: the prose beside this alert describes the transport as it is now ─────
#
# Two docstrings — `_append_fast_failure_alert`'s in `app/autonomy.py` and this
# file's `test_the_line_is_written_with_discord_home_channel_null` — told the
# pre-#1592 drop story as live behaviour (`discord_alert` "logs a warning and
# returns", so #85's disable alert "reached nobody") and cited
# `app/discord_notify.py:49-56`/`:52-56` for it. Those ranges are not in
# `discord_alert` at all: they span the tail of `_missing_transport_halves` and
# the opening of `_discord_notify_task_complete`, whose own unconfigured return
# emits no warning whatsoever. The nodes below pin the corrected shape: the story
# may be told only as marked history, and a line citation of `app/discord_notify.py`
# inside those two docstrings has to land on a line of the branch it describes,
# with that branch located from the file rather than restated.
#
# The seam is a real one. `app/discord_notify.py` is the alert route of the worker
# process (`workers/sources/scheduled_task.py`, `workers/fleet_watchdog.py` await
# `discord_alert`) and `app/autonomy.py` is imported by this test process, so the
# sentences under test are claims about code that does not run here — reading the
# docstring string and locating the branch in the other module's source is what
# crosses it. `tests/test_autonomy_failure_alert.py:1260`'s citation of
# `app/discord_notify.py:115-117` is CORRECT (the `if not appended:` warning inside
# `_survive_the_dropped_alert`) and is deliberately outside this scope.

# Phrases that describe `discord_alert` as if it still dropped an alarm on the
# floor. True of the function as it stood before #1592 and of nothing since, so
# each is legal in a docstring only inside a sentence that says which state it
# describes. Matched in both tenses because the corrected prose tells them past.
# Only the two mechanisms are here. "down the dead transport" is NOT: it is a
# verdict rather than a mechanism, so it sits in `_CLAIMS_1592_REFUTES` below and
# is banned outright — a phrase cannot be both conditionally legal and forbidden,
# which is what the two tables disagreed on when they both carried it.
_STALE_DROP_CLAIMS = (
    ("`discord_alert` only logs a warning and returns",
     re.compile(r"warning and return|returns? after a warning|returned after a warning",
                re.IGNORECASE)),
    ("the alert reached nobody", re.compile(r"reached nobody", re.IGNORECASE)),
)

# The mark that makes one of those phrases legal: the sentence names #1592 as the
# point the described behaviour stops at.
_PRE_1592_MARK = re.compile(r"pre-#1592|before #1592|prior to #1592|until #1592",
                            re.IGNORECASE)

# Rhetoric for the same refuted claim, banned outright instead of marked: the
# history needs the mechanism (a warning and a `return`), not the verdict word.
_CLAIMS_1592_REFUTES = (
    re.compile(r"dead end", re.IGNORECASE),
    re.compile(r"dead transport", re.IGNORECASE),
    re.compile(r"only transport", re.IGNORECASE),
    re.compile(r"(?:would|will|could) reach nobody", re.IGNORECASE),
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_DISCORD_NOTIFY_CITATION = re.compile(r"app/discord_notify\.py:(\d+)(?:-(\d+))?")


def _sentences_of(doc: str) -> list[str]:
    """The docstring's sentences, whitespace-flattened, so a claim is judged whole.

    The flattening is the point, not tidiness: a docstring is wrapped at ~79
    columns, so the old prose carried "logged a warning\n    and returned" across
    a line break and a pattern written against one line matched nothing — the node
    would have waved through exactly the sentence it exists to catch. Rejoin the
    whitespace first and the wrap is irrelevant, which is what makes the same
    pattern hold whatever column the author happened to break at.
    """
    return [s.strip() for s in _SENTENCE_END.split(" ".join(doc.split())) if s.strip()]


def _discord_alert_unconfigured_branch():
    """`discord_alert`'s unconfigured branch as (first, last) source line numbers.

    Located from `app/discord_notify.py` at test time: the branch opens on the
    `if not home_channel or not token:` inside `discord_alert` and closes on that
    branch's own `return`. Nothing here is hard-coded, so a docstring citation that
    drifts from the code goes red instead of ageing quietly — and the branch has to
    be the one that hands the alarm to `_survive_the_dropped_alert`, because a
    locator that matched some other early `return` would certify a stale citation.
    """
    from app import discord_notify

    src_lines, start = inspect.getsourcelines(discord_notify.discord_alert)
    cond_idx = next(
        (i for i, line in enumerate(src_lines)
         if line.strip().startswith("if not home_channel or not token:")), None)
    assert cond_idx is not None, (
        "`discord_alert` no longer opens its unconfigured path with "
        "`if not home_channel or not token:`, so the citations this node grades "
        "have nothing fixed to be measured against")
    cond_indent = len(src_lines[cond_idx]) - len(src_lines[cond_idx].lstrip())
    end_idx = next(
        (i for i in range(cond_idx + 1, len(src_lines))
         if src_lines[i].strip() and not src_lines[i].strip().startswith("#")
         and (len(src_lines[i]) - len(src_lines[i].lstrip())) > cond_indent
         and src_lines[i].strip() == "return"), None)
    assert end_idx is not None, "the unconfigured branch never returns — find its end"
    body = "".join(src_lines[cond_idx:end_idx + 1])
    assert "_survive_the_dropped_alert(" in body, (
        "the branch this node calls the fallback branch no longer calls "
        f"`_survive_the_dropped_alert`: {body}")
    return start + cond_idx, start + end_idx


def _assert_drop_story_is_marked_history(doc: str, where: str) -> None:
    """Shared body of clauses 1 and 3, applied to a docstring read off the object."""
    assert doc, f"{where} has no docstring to check"
    sentences = _sentences_of(doc)
    marked = 0
    for label, pattern in _STALE_DROP_CLAIMS:
        for sentence in sentences:
            if pattern.search(sentence):
                assert _PRE_1592_MARK.search(sentence), (
                    f"{label}, stated of live code in {where}: {sentence!r} — after "
                    "#1592 the unconfigured branch also appends the alarm to the "
                    "daily note (see `_survive_the_dropped_alert`), so this may "
                    "only be said inside a sentence marked `before #1592`")
                marked += 1
    assert marked, (
        f"{where} tells no drop story at all. The item asks for the pre-#1592 "
        "narrative rewritten as past-tense history, not deleted: the reason the "
        "line was written directly is that this used to be the only branch")


def test_the_fast_failure_docstring_only_tells_the_drop_story_as_pre_1592_history():
    """Clause 1: `app/autonomy.py` may describe the drop only as marked history.

    Reads the function's own `inspect.getdoc`, not a copy of the text, so the node
    fails when the sentence is edited in place and stays green when it is moved —
    it is the sentence the next reader of `_append_fast_failure_alert` is told.
    """
    _assert_drop_story_is_marked_history(
        inspect.getdoc(autonomy._append_fast_failure_alert) or "",
        "app/autonomy.py `_append_fast_failure_alert`")


def test_the_home_channel_null_docstring_describes_the_transport_as_it_is():
    """Clause 3: the tripwire's own docstring states live behaviour correctly.

    Same shared body, second docstring. This is the node whose premise outlived the
    code it described: `config.yaml` really does still carry `home_channel: null`,
    and the docstring beside it claimed a `return`-after-a-warning that
    `discord_alert` has not done since #1592.
    """
    _assert_drop_story_is_marked_history(
        inspect.getdoc(test_the_line_is_written_with_discord_home_channel_null) or "",
        "tests/test_autonomy_failure_alert.py "
        "`test_the_line_is_written_with_discord_home_channel_null`")


def test_the_discord_notify_citations_in_those_docstrings_land_in_the_drop_branch():
    """Clause 2: every line citation points at a line of the branch it names.

    Both docstrings, every `app/discord_notify.py:N` or `:N-M` citation, measured
    against the span `_discord_alert_unconfigured_branch` locates from the source
    file. The old citations (`:49-56`, `:52-56`) were inside neither `discord_alert`
    nor its fallback, which is exactly the failure this node cannot be talked out
    of: it reads the file, not the prose.
    """
    first, last = _discord_alert_unconfigured_branch()
    docs = {
        "app/autonomy.py `_append_fast_failure_alert`":
            inspect.getdoc(autonomy._append_fast_failure_alert) or "",
        "tests/test_autonomy_failure_alert.py "
        "`test_the_line_is_written_with_discord_home_channel_null`":
            inspect.getdoc(test_the_line_is_written_with_discord_home_channel_null) or "",
    }
    for where, doc in docs.items():
        found = 0
        for match in _DISCORD_NOTIFY_CITATION.finditer(doc):
            low, high = int(match.group(1)), int(match.group(2) or match.group(1))
            assert first <= low <= high <= last, (
                f"{where} cites `app/discord_notify.py:{low}-{high}`, but "
                f"`discord_alert`'s unconfigured branch — the code the sentence is "
                f"about — is app/discord_notify.py:{first}-{last}")
            found += 1
        assert found, (
            f"{where} cites `app/discord_notify.py` at no line at all; the item "
            "wants the citation repointed into the branch, not dropped")


def test_the_fast_failure_docstring_still_says_why_the_discord_route_is_bypassed():
    """Clause 4: the surviving reason survives, and nothing refuted replaces it.

    `discord_alert` reaching the note now means "Discord is dead" is no longer a
    reason for anything, so the bypass has to be justified by what it still buys:
    this line keeps its own shape instead of arriving as a "Scheduler alert not
    delivered" alarm line, and `append_daily_alert_line` is the shared writer
    either way. Both are asserted by the phrases the prose has to carry, and the
    refuted verdict words are banned outright rather than merely marked.
    """
    doc = " ".join(
        (inspect.getdoc(autonomy._append_fast_failure_alert) or "").split())
    assert "Scheduler alert not delivered" in doc, (
        "the bypass has to say what the routed line would have looked like — an "
        f"alarm line owned by `discord_alert`: {doc}")
    assert "append_daily_alert_line" in doc and "shared writer" in doc, (
        f"and that the writer is shared either way: {doc}")
    assert "durations" in doc, (
        f"the shape this function owns is the streak's durations: {doc}")
    for where, text in (
            ("`_append_fast_failure_alert`", doc),
            ("`test_the_line_is_written_with_discord_home_channel_null`",
             " ".join((inspect.getdoc(
                 test_the_line_is_written_with_discord_home_channel_null)
                 or "").split()))):
        for pattern in _CLAIMS_1592_REFUTES:
            assert not pattern.search(text), (
                f"{where} still says {pattern.pattern!r} of the transport; since "
                "#1592 an unconfigured `discord_alert` appends the alarm to the "
                "daily note, so that verdict is refuted even as history")


def test_the_home_channel_null_tripwire_still_reads_its_premise_off_disk():
    """Clause 5's invisible half: the premise is still a fact about the artifact.

    The assertions of `test_the_line_is_written_with_discord_home_channel_null`
    must survive a prose-only round untouched, and the one that makes it a tripwire
    is that `home_channel is None` is read out of `config.yaml` rather than
    installed by the test. That property lives in the source text, so this pins it
    there: a later edit that monkeypatches the config instead would otherwise leave
    the tripwire green while checking nothing.
    """
    src = inspect.getsource(test_the_line_is_written_with_discord_home_channel_null)
    assert "_discord_block_on_disk()" in src, "the premise is still read from disk"
    assert 'disk.get("home_channel") is None' in src, (
        "and still asserted as a null, not merely observed")
    helper = inspect.getsource(_discord_block_on_disk)
    assert "config.yaml" in helper and "read_text" in helper, (
        f"the reader itself must open the file: {helper}")
    assert "write_text" not in helper, "and the reader may not author the premise"


# ── #1592: an alarm the Discord transport refuses still has to reach a person ──

def _bullet_lines():
    """Every bullet in today's note — the surface a dropped alarm now lands on.

    `_alert_lines()` filters on `Autonomy #`, which is the fast-failure line's own
    wording; a dropped Discord alarm quotes whatever message it was handed and does
    not necessarily name a task, so the drop has to be read off the note itself.

    The front matter comes off first: a fresh note's `tags:` block is itself a list of
    `- ` lines, and counting those as entries would make every note look like it
    already carried an alarm.
    """
    path = _note_file(autonomy)
    if not path.exists():
        return []
    text = path.read_text()
    if text.startswith("---\n"):
        _, _, text = text.partition("\n---\n")
    return [ln for ln in text.splitlines() if ln.startswith("- ")]


async def test_an_undeliverable_discord_alert_lands_on_todays_note(aut):
    """Clauses 1 and 2, over the transport's real drop branch.

    `discord_alert` is the terminus of all five scheduler alarms (model-server
    outage, unparseable task files, their recovery, due-ness stall, next_run stall),
    and until now its no-channel branch was a `logger.warning` and a `return`: the
    2026-09-27 alarm about #68 sitting 258 h past its own `next_run` reached a
    rotating log and no person. Nothing is mocked here — the transport's two halves
    are read from the boot-merged `CONFIG`, whose `home_channel` the test above pins
    off `config.yaml` on disk and whose empty token `_discord_token()` resolves at
    call time — so the branch under test is the box's own, not a fixture's.

    Both properties in one read of one line, because they are one line's job: the
    alert text has to be there at all (clause 1), and it has to say that Discord
    refused it and which half of the transport is missing (clause 2). A note that
    only quotes the alarm would be indistinguishable from an alarm someone chose to
    ignore, which is the difference between setting up a transport and dismissing a
    report.
    """
    from app import discord_notify

    assert not discord_notify._discord_token(), (
        "the token resolved to something, so this would be exercising the delivery "
        "branch while claiming to test the drop")

    await discord_notify.discord_alert(
        "autonomy scheduler may be stalled: task #42 next_run 40h past due",
        title="Scheduler stall")

    lines = _bullet_lines()
    assert len(lines) == 1, f"expected the dropped alarm on the note, once: {lines}"
    line = lines[0]
    assert "task #42 next_run 40h past due" in line, (
        f"the alarm text itself never reached the note: {line!r}")
    assert "Scheduler stall" in line, (
        f"the alert's own title is how a reader sorts it: {line!r}")
    assert "not delivered" in line, (
        "a reader must be able to tell a dropped alarm from one someone ignored: "
        f"{line!r}")
    assert "discord.home_channel" in line and "unset" in line, (
        "the line must name the half of the transport that is missing, and not as a "
        f"generic 'not configured': {line!r}")


async def test_the_drop_reason_names_only_the_half_that_is_actually_missing(aut,
                                                                           monkeypatch):
    """The reason a reader gets must be true of this box, not of the branch.

    Both halves being absent is the live state, but a channel configured with no
    token is the state a person is most likely to create by half-doing option (a),
    and a line telling them `home_channel` is unconfigured then sends them to the
    wrong key. The title and message are also capped the way the Discord path caps
    them, so the note can never carry a longer account of an alarm than the refused
    channel would have.
    """
    from app import discord_notify

    monkeypatch.setattr("app.discord_notify.CONFIG",
                        {"discord": {"home_channel": "1234567890", "token": ""}},
                        raising=False)
    await discord_notify.discord_alert("probe alarm text", title="Probe")

    lines = _bullet_lines()
    assert len(lines) == 1, f"one dropped alarm, one note line: {lines}"
    line = lines[0]
    assert "bot token" in line, (
        f"a missing token has to be named as the missing half: {line!r}")
    assert "discord.home_channel" not in line, (
        f"the reason blames a key that is configured on this call: {line!r}")


async def test_the_dropped_alert_never_raises_even_when_the_note_cannot_be_written(
        aut, monkeypatch):
    """Clause 1's second word, which is the half that can only fail at runtime.

    Every caller is a scheduler tick or a run's failure path — `_alert`, the
    infra-ceiling crossing, the disable — and none of them is written to survive an
    exception from the alert. So a vault on a read-only mount has to cost a log line
    and nothing else: the same non-propagation rule `_append_fast_failure_alert`
    already follows for its own note write, extended to the route that now shares
    that writer.
    """
    from app import discord_notify

    # `/proc` exists and is not writable by this user, so the mkdir inside the
    # writer fails for a real reason instead of a mocked one.
    monkeypatch.setenv("LLOYD_DAILY_NOTE_DIR", "/proc/definitely-not-writable/notes")

    await discord_notify.discord_alert("alarm that has nowhere to go")
    assert not _bullet_lines(), "nothing landed, and that is the passing case"



# ── #1736: a returned True has to mean the line is in the note ────────────────
#
# `2026-09-27 21:14:20,838 [ERROR] lloyd-workers.fleet_watchdog` and its paired
# `discord_alert (no channel/token configured)` WARNING are both in
# `~/lloyd-data/logs/server.err`; `grep -c "21:14 PDT" ~/obsidian/memory/2026-09-27.md`
# returns 0, `git -C ~/obsidian log -S '21:14 PDT' -- memory/2026-09-27.md` is empty,
# and `grep -c "refused the alert too\|daily-note fallback failed" ~/lloyd-data/logs/server.err`
# returns 0 — so neither fallback warning the contract promises was emitted either.
# All three facts are one return value: `append_daily_alert_line` wrote, believed the
# syscall, said True, and its caller had no reason left to speak.

ALERT_SHAPE = re.compile(r"^- \d{2}:\d{2} [A-Z]{2,5} — ")

NOTE_BODY_BEFORE = (
    "---\nsegment: memory\ntags: [memory, daily-notes]\ntype: note\n"
    "timestamp: '2026-09-27T20:00:00'\n---\n\n"
    "# 2026-09-27 Daily Notes\n\n"
    "## Decisions\n\n- decided the round order for tomorrow\n\n"
    "## Session\n\n- pre-existing session capture, the note was not empty\n")


def _seed_note_with_body() -> Path:
    """A note that already has prose in it, written before the alert is appended.

    Every case below needs this because the read-back must never be proved against an
    empty file: a note whose entire content is the one appended line would come back
    correct after a clobber that happened to keep the header, and the pre-existing
    text is what makes "the line is in the note" a claim about the note rather than
    about a scratch file we just made.
    """
    path = _note_file(autonomy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NOTE_BODY_BEFORE, encoding="utf-8")
    return path


def _error_lines(caplog) -> list[str]:
    """ERROR records from the autonomy logger only, so the assertion is about this writer."""
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-autonomy" and r.levelno == logging.ERROR]


def test_a_returned_true_means_the_line_is_readable_back_from_the_note(aut):
    """Clause 1, the positive control: True now implies a readable-back line.

    The exact text asserted is what `append_daily_alert_line` formats — `- HH:MM %Z —
    <body>`, America/Los_Angeles — matched through the timestamp wildcard rather than
    a literal the test hand-built, so a change to the line format has to be a change
    this test notices. The pre-existing body is asserted twice, before and after:
    before, because an empty scratch file proves nothing; after, because the alert
    that lands by erasing the note's own prose is not an alert that landed.
    """
    path = _seed_note_with_body()
    assert "pre-existing session capture" in path.read_text(encoding="utf-8"), (
        "the note has no body, so a read-back proved nothing about a real note")

    assert autonomy.append_daily_alert_line("fleet watchdog says the fleet is silent")

    after = path.read_text(encoding="utf-8")
    assert "pre-existing session capture" in after, "the append cost the note its own body"
    hit = [ln for ln in after.splitlines()
           if ALERT_SHAPE.match(ln) and "fleet watchdog says the fleet is silent" in ln]
    assert len(hit) == 1, f"the appended line is not readable back: {after!r}"


def test_a_note_clobbered_after_the_write_returns_false_and_says_so_at_error(
        aut, monkeypatch, caplog):
    """Clause 2, in the incident's own shape: the write lands, then something rewrites it.

    Mechanism (b) of the two the item names is `a later whole-file rewrite composed
    from a pre-21:14 snapshot` — the note's mtime was 21:24, ten minutes after the
    alert, with content byte-identical to HEAD, which is exactly what a no-net-diff
    snapshot write looks like. Reproduced here by letting the append happen for real
    and restoring the pre-call bytes when the file handle closes, so the sequence the
    function sees is: write succeeded, verify, line is not there. The assertion that
    the file on disk is the snapshot again is what keeps this from being a test of a
    failed write — the write did not fail, and under the old body that was invisible.
    """
    path = _seed_note_with_body()
    snapshot = path.read_text(encoding="utf-8")
    real_open = open

    class _Clobbered:
        """The real handle, with someone else's snapshot rewrite at close."""

        def __init__(self, handle):
            self._handle = handle

        def write(self, text):
            return self._handle.write(text)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            path.write_text(snapshot, encoding="utf-8")
            return False

    def open_that_clobbers(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        return _Clobbered(handle) if "a" in mode else handle

    monkeypatch.setattr(autonomy, "open", open_that_clobbers, raising=False)
    caplog.set_level(logging.ERROR, logger="lloyd-autonomy")

    assert autonomy.append_daily_alert_line("the alarm that vanished on 2026-09-27") is False
    errors = _error_lines(caplog)
    assert len(errors) == 1, f"the mismatch must be reported at ERROR exactly once: {errors}"
    assert str(path) in errors[0], f"the ERROR has to name the note it checked: {errors[0]}"
    assert path.read_text(encoding="utf-8") == snapshot, (
        "the write was not clobbered, so this stopped being the case it claims")


def test_a_note_whose_write_is_discarded_returns_false_without_mocking_anything(
        aut, caplog):
    """Clause 2 again, with no seam intercepted: a note that swallows what it is given.

    `memory/<today>.md` as a symlink to `/dev/null` makes every part of the contract
    real — `exists()` is true so the append branch runs, `open(..., "a")` succeeds, the
    `write()` returns a count, and the re-read finds nothing. It is the same observable
    state as the 09-27 instance from the writer's side, which is the only side this
    function can see, and it proves the check does not depend on the interception
    above being wired correctly.
    """
    path = _note_file(autonomy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.symlink_to("/dev/null")
    caplog.set_level(logging.ERROR, logger="lloyd-autonomy")

    assert autonomy.append_daily_alert_line("an alarm written somewhere that keeps nothing") is False
    errors = _error_lines(caplog)
    assert len(errors) == 1, f"expected one ERROR naming the discard: {errors}"


def test_a_note_that_cannot_be_read_back_costs_a_log_and_no_exception(
        aut, monkeypatch, caplog):
    """Clause 3: the verification is never allowed to become the outage it reports.

    The mode goes on the note file from the moment the append handle closes, so the
    write itself is unharmed and it is the re-open that raises `PermissionError`. The
    note *directory* is the other way to break a read, but taking write permission off
    a directory does not stop an append to a file already in it — file permission does
    — so a directory mode would be an unfalsifiable stage on the append that runs
    first. Running as root would make the mode inert rather than the check inert, which
    is why this skips instead of passing.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores file modes, so the re-open would not raise")
    path = _seed_note_with_body()
    real_open = open

    class _UnreadableAfter:
        def __init__(self, handle):
            self._handle = handle

        def write(self, text):
            return self._handle.write(text)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            path.chmod(0o000)
            return False

    monkeypatch.setattr(autonomy, "open",
                        lambda file, mode="r", *a, **kw: _UnreadableAfter(real_open(file, mode, *a, **kw))
                        if "a" in mode else real_open(file, mode, *a, **kw),
                        raising=False)
    caplog.set_level(logging.ERROR, logger="lloyd-autonomy")
    try:
        assert autonomy.append_daily_alert_line("an alarm whose note went unreadable") is False
    finally:
        path.chmod(0o644)
    errors = _error_lines(caplog)
    assert len(errors) == 1, (
        f"the failed re-read has to be reported, not swallowed: {errors}")
    assert path.exists(), "the note is still there; only the read failed"


def test_two_alerts_the_same_day_both_stay_readable_because_this_is_no_dedupe(aut):
    """Clause 4: the read-back must not quietly become a marker check.

    Both alerts carry the SAME body, which is the only text that could collide with a
    marker: `~/obsidian/memory/2026-09-27.md` carries 13 same-shaped `Scheduler alert
    not delivered` lines from one day, and #1727's triage established that every
    producer already paces itself — so a presence test used as a dedupe key would have
    dropped 12 legitimate re-fires. The read-back asks only whether THIS line is in the
    file, so the second call has to append, not agree.
    """
    _seed_note_with_body()
    body = "fleet watchdog says the fleet is silent (twice)"

    assert autonomy.append_daily_alert_line(body)
    assert autonomy.append_daily_alert_line(body)

    hits = [ln for ln in _note_file(autonomy).read_text(encoding="utf-8").splitlines()
            if body in ln and ALERT_SHAPE.match(ln)]
    assert len(hits) == 2, (
        f"two alerts, two lines — the read-back dropped or merged one: {hits}")


# ── #1798: a mismatch files one coalesced backlog item; the drop branch gets a witness
#
# The block above (#1736) made a dropped alert line return False and say so at ERROR,
# and stopped there: the False went to a rotating log and nowhere a person opens,
# because the other candidate transport is the one Alan left deliberately unconfigured
# (`discord.home_channel: null`, empty token) and `config.yaml` is off-limits to a
# round. What follows is the second surface — one `[alerts] daily-note drop` item on
# the `lloyd` board, refreshed rather than re-filed while it stays open.
#
# Two fixtures make that testable with no live server and no live board.
# `refused_backend` is module-wide and refuses every outbound call, so nothing in this
# file can file on `~/obsidian/backlog` by accident; `board_server` replaces it with a
# dispatcher into the REAL route functions over a `_BACKLOG_DIR` in tmp. That is the
# shape #1703 settled for the guardian's poster in `tests/test_guardian_predicates.py`
# — capture what the client sent, replay it into the one loader — for the reason it
# stands: a hand-written stand-in for a route drifts from the route, and the drift is
# the interesting part.

#: The one name-prefix this feature files, and the coalescing key. Asserted here as
#: a literal rather than imported from `autonomy.DAILY_NOTE_DROP_PREFIX` so a rename
#: in the code has to be a decision in two places: coalescing is a prefix match, so a
#: name the code changed silently would stop coalescing and start growing a file a day.
ALERT_ITEM_PREFIX = "[alerts] daily-note drop"
CREATE_PATH = "/api/backlog/task-create"
UPDATE_PATH = "/api/backlog/task-update"
TASKS_PATH = "/api/backlog/tasks"

_FM_BLOCK = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class _Resp:
    """One `urlopen` reply: the status attribute and the bytes, as a context manager."""

    def __init__(self, payload, status: int = 200):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeRequest:
    """Just the `json()` the two write routes await; they read no other request face."""

    def __init__(self, payload):
        self._payload = payload or {}

    async def json(self):
        return self._payload


class _Board:
    """The live backlog routes, writing to a directory in tmp, behind `urlopen`.

    Deliberately thin: it maps a path onto the real coroutine and hands back a reply,
    so the assertions land on what the route did to a file — front matter, board,
    body — rather than on what a stand-in would have done. `_BACKLOG_DIR` is the
    module's single directory constant and every helper here reads it as an attribute
    at call time, which is what makes one `monkeypatch.setattr` enough; the board map
    is positional over the corpus (`app/routers/backlog.py:244`), so with one board on
    tmp the positional id and the name agree and neither needs the real filesystem.
    """

    def __init__(self, backlog_dir: Path, monkeypatch):
        from app.routers import backlog as routes
        from fastapi import HTTPException

        self.routes = routes
        self._HTTPException = HTTPException
        self.dir = backlog_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        #: [(method, path, payload)] in call order — the coalescing assertions read it
        self.requests: list = []
        #: Set to an exception class to make the create blow up in the transport.
        self.fail_create = None
        #: Set to a payload to answer a create with it verbatim, route unbuilt.
        self.reply_create = None
        #: Set True to make the item's own read-back 404 while the file stays put.
        self.hide_detail = False

        monkeypatch.setattr(routes, "_BACKLOG_DIR", self.dir)
        monkeypatch.setattr(routes, "_board_index", lambda: ({"lloyd": 0}, {}))
        monkeypatch.setattr(routes, "_backlog_board_map", lambda: {"lloyd": 0})
        monkeypatch.setattr(urllib.request, "urlopen", self._dispatch)

    def _dispatch(self, req, *args, **kwargs):
        parsed = urlparse(req.full_url)
        payload = json.loads(req.data.decode("utf-8")) if req.data else None
        method = req.get_method()
        self.requests.append((method, parsed.path, payload))
        try:
            result = self._route(method, parsed, payload)
        except self._HTTPException as exc:
            return _Resp({"detail": exc.detail}, exc.status_code)
        if hasattr(result, "body"):        # a JSONResponse: already serialised
            return _Resp(json.loads(result.body) if result.body else None,
                         result.status_code)
        return _Resp(result)

    def _call(self, coro):
        """Finish one route call, from whatever stack the caller is on.

        Two shapes, because the router itself is two shapes: the corpus routes are
        plain `def` so FastAPI threadpools them (`app/routers/backlog.py:426-433`
        explains that a coroutine parsing 11 MB of YAML blocks the loop exclusively),
        and only the two write routes are `async def`, because they await
        `request.json()`. A non-coroutine is therefore already the answer.

        For the coroutines: the real backend serves them on its own loop in its own
        process. From a sync test `asyncio.run` is the whole job; from an async test the
        caller already has a running loop and a nested `asyncio.run` is refused, so the
        coroutine goes to a one-shot loop in a thread that is joined before returning.
        Either way the reply is complete before `urlopen` returns, which is the property
        the production caller depends on: the filing is a blocking call, not a
        fire-and-forget.
        """
        if not asyncio.iscoroutine(coro):
            return coro
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)

    def _route(self, method, parsed, payload):
        path = parsed.path
        if path == CREATE_PATH:
            if self.fail_create:
                raise self.fail_create("transport died mid-create")
            if self.reply_create is not None:
                return self.reply_create
            return self._call(self.routes.backlog_task_create(_FakeRequest(payload)))
        if path == UPDATE_PATH:
            return self._call(self.routes.backlog_task_update(_FakeRequest(payload)))
        if path == TASKS_PATH:
            qs = parse_qs(parsed.query)
            return self._call(self.routes.backlog_tasks(
                board_id=qs.get("board_id", [None])[0],
                q=qs.get("q", [None])[0]))
        hit = re.fullmatch(r"/api/backlog/task/(\d+)", path)
        if hit:
            if self.hide_detail:
                raise self._HTTPException(404, "Item not found")
            return self._call(self.routes.backlog_task_detail(int(hit.group(1))))
        raise AssertionError(f"unexpected backlog path {path!r} from the alert path")

    def paths_of(self, path: str) -> list:
        return [entry[1] for entry in self.requests if entry[1] == path]

    def payload_of(self, path: str, index: int = 0):
        same = [entry for entry in self.requests if entry[1] == path]
        assert same, f"no call to {path}; the calls were {self.requests}"
        return same[index][2]

    def items(self) -> list:
        """[(path, front_matter_with_identity, body)] for each file on this board.

        The two identity fields are DERIVED here rather than read from the front matter,
        because the route does not store them: `backlog_task_create` keeps the name in
        the file's `# ` heading and the id in the filename (`{task_id}-slug.md`, the
        pair `_backlog_parse_fm` reconstructs for every reader on the board). A test that
        looked for `name:` in the front matter would be asserting a shape the board has
        never written, and a filing that coalesced on the wrong name would still satisfy
        it.
        """
        out = []
        for path in sorted(self.dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            match = _FM_BLOCK.match(raw)
            assert match, f"{path} has no front matter block at all: {raw[:120]!r}"
            fm = yaml.safe_load(match.group(1))
            body = raw[match.end():].strip()
            heading = body.splitlines()[0] if body else ""
            assert heading.startswith("# "), f"{path.name} has no H1: {heading!r}"
            fm["name"] = heading[2:].strip()
            fm["id"] = int(path.name.split("-", 1)[0])
            out.append((path, fm, body))
        return out


@pytest.fixture(autouse=True)
def refused_backend(monkeypatch):
    """Refuse every outbound call this file makes, and record that it tried.

    Not a convenience: without it, a test that drives the mismatch path with the
    production default URL posts to the backend that is running right now, and a suite
    run would file real items on the real `lloyd` board. Connection-refused is also
    the honest default for a test that never asked for a backend at all.
    """
    attempts: list = []

    def _refuse(req, *args, **kwargs):
        attempts.append(getattr(req, "full_url", str(req)))
        raise urllib.error.URLError(
            ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    return attempts


@pytest.fixture
def board_server(tmp_path, monkeypatch, refused_backend):
    """The real routes on a tmp board, replacing the module-wide refusal."""
    return _Board(tmp_path / "backlog", monkeypatch)


def _clobber_on_write(monkeypatch, snapshot: str) -> None:
    """Clobber the note with `snapshot` the moment the writer's append handle closes.

    The 09-27 incident's own shape, reproduced the way the case above does it: the
    writer reaches the note through the module-global `open`, so the seam is that name,
    and the append itself runs for real. What the function then sees is write succeeded,
    verify, line is not there — which is the state a False-returning mismatch exists to
    report, and the reason the filing tests need the write to have actually happened.
    """
    real_open = open

    class _SnapshotRewrite:
        """The real handle, with someone else's stale-snapshot write at close."""

        def __init__(self, handle, target):
            self._handle, self._target = handle, target

        def write(self, text):
            return self._handle.write(text)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            self._target.write_text(snapshot, encoding="utf-8")
            return False

    monkeypatch.setattr(
        autonomy, "open",
        lambda file, mode="r", *a, **kw: _SnapshotRewrite(real_open(file, mode, *a, **kw), file)
        if "a" in mode else real_open(file, mode, *a, **kw),
        raising=False)


def _note_keeping_nothing() -> Path:
    """Today's note as a symlink to `/dev/null`: a write that is kept nowhere.

    The mock-free mismatch the case above establishes — `exists()` is true, the append
    succeeds, the read-back is `""` — used wherever a node is about what the mismatch
    DOES rather than how it was produced.
    """
    path = _note_file(autonomy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.symlink_to(Path("/dev/null"))
    return path


def _field(body: str, label: str):
    """The value of a `Label: value` line of the filed body — the lines the refresh reads."""
    hit = re.search(rf"^{re.escape(label)}: (.*)$", body, re.MULTILINE)
    assert hit, f"{label!r} is not a line of the filed body:\n{body}"
    return hit.group(1).strip()


def test_a_dropped_daily_note_line_files_one_backlog_item_on_the_lloyd_board(
        aut, board_server, monkeypatch, caplog):
    """Clause 1: the False now reaches a file a person opens, not only a log line.

    The payload assertions are the contract the owed ruling is about — board `lloyd`
    and priority `high`, which follow the guardian's own alert filing so an auto-filed
    alert sits where the other auto-filed alerts sit — plus the one field #1893
    deliberately does NOT follow it on: `status`, filed `draft` because the autotriage
    pool reads `draft` and nothing else, so at any other status the alarm is filed
    where no reader looks. They are asserted on what the code SENT as well as on the
    file the route wrote, because a board that arrives defaulted and a board that
    arrives named are the same file and a different decision.
    """
    path = _seed_note_with_body()
    _clobber_on_write(monkeypatch, path.read_text(encoding="utf-8"))
    with caplog.at_level(logging.ERROR, logger="lloyd-autonomy"):
        assert autonomy.append_daily_alert_line(
            "Autonomy #7 failed 3 times in a row (run #42)") is False

    assert board_server.paths_of(CREATE_PATH) == [CREATE_PATH], board_server.requests
    assert board_server.paths_of(UPDATE_PATH) == [], "a first drop has nothing to refresh"
    sent = board_server.payload_of(CREATE_PATH)
    assert sent["name"].startswith(ALERT_ITEM_PREFIX), sent
    assert sent["board"] == "lloyd", sent
    assert (sent["status"], sent["priority"]) == ("draft", "high"), sent

    items = board_server.items()
    assert len(items) == 1, [str(p) for p, _, _ in items]
    _, fm, body = items[0]
    assert (fm["board"], fm["status"], fm["priority"]) == ("lloyd", "draft", "high"), fm
    assert fm["name"] == sent["name"], (fm, sent)
    assert _field(body, "Mismatches observed") == "1", body
    assert _note_file(autonomy).name in body, body
    # The ERROR line is the record that already existed; the item is added to it, not
    # swapped for it.
    assert len(_error_lines(caplog)) == 1, _error_lines(caplog)


def test_the_second_mismatch_refreshes_that_one_item_instead_of_filing_a_second(
        aut, board_server, monkeypatch):
    """Clause 2: five drops in a week are one item that says five.

    Both fields the contract names advance: the count, and the `Last seen` stamp — read
    back as a real datetime rather than compared as text, because two stamps that
    differ only in their UTC-offset spelling would pass a string comparison while
    telling the same instant twice, which is the whole purpose of the field.
    """
    path = _seed_note_with_body()
    _clobber_on_write(monkeypatch, path.read_text(encoding="utf-8"))

    assert autonomy.append_daily_alert_line("Autonomy #7 failed 3 times (run #42)") is False
    first_body = board_server.items()[0][2]
    first_seen = _field(first_body, "Last seen")

    assert autonomy.append_daily_alert_line("Autonomy #7 failed 3 times (run #43)") is False

    items = board_server.items()
    assert len(items) == 1, [str(p) for p, _, _ in items]
    _, fm, body = items[0]
    assert body.count("Mismatches observed:") == 1, "the count is now in two places"
    assert _field(body, "Mismatches observed") == "2", body
    second_seen = _field(body, "Last seen")
    assert (dt.datetime.fromisoformat(second_seen)
            > dt.datetime.fromisoformat(first_seen)), (first_seen, second_seen)
    # The refresh posts a body and no status, so this is the field the ROUTE must
    # carry across its own front-matter rewrite — and it is the field that decides
    # whether the item a person is watching is still sitting in the pool autotriage
    # reads (#1893). A refresh that dropped it to some other status would leave the
    # count advancing on an item nobody polls.
    assert fm["status"] == "draft", fm

    assert board_server.paths_of(CREATE_PATH) == [CREATE_PATH], board_server.requests
    assert board_server.paths_of(UPDATE_PATH) == [UPDATE_PATH], board_server.requests
    assert board_server.payload_of(UPDATE_PATH)["force_body_replace"] is True, \
        "the refresh replaces a body this function authored whole, and the route " \
        "ignores a shorter body without that flag"


#: The three functions that ARE the drop-alert filing path, named by symbol rather than
#: as a line range: #1893 retired a status literal from this path, and a range citation
#: drifts while a symbol boundary does not. The region is exact because the module keeps
#: `up_next` elsewhere legitimately — `_record_failure` re-arms an autonomy TASK to
#: `up_next` (`app/autonomy.py`, the `fields` dict) and `RUNNABLE_STATUSES` lists it —
#: which is the scheduler's task-file state, a different store from a backlog item's
#: front matter. These three functions are what decides what the filed item reads.
DROP_FILING_FUNCS = ("_daily_note_drop_body", "_open_daily_note_drop_item",
                     "_file_daily_note_mismatch")


def test_the_drop_alert_is_filed_where_the_triage_loop_reads():
    """#1893 clause 4: the dead status is gone from the filing path, and so is its reason.

    `up_next` was the value that made this an alarm nobody received: autotriage's pool
    filter is `i.status == TRIAGE_POOL_STATUS` with that constant set to `draft`, and
    `ready_confirmed` takes only the implement status plus a confirmed triage verdict an
    auto-filed item has never had — so an item filed at any other status was in neither
    pool, and the reconciler pushed it back as never triaged. Pinned here: the absence of
    that literal over the source of the functions that file the alert, and the presence,
    in both places that used to argue for it, of the replacement rationale — board and
    priority DO follow the guardian's alert filing, the status deliberately does not. The
    test's own docstring is read for the same reason as the code's: a stale rationale in
    prose is a second copy of the bug.
    """
    src = "".join(
        inspect.getsource(getattr(autonomy, name)) for name in DROP_FILING_FUNCS)
    # Not a vacuous sweep: each function contributed, and the payload it files is in
    # what was read. A test over an empty string passes every absent-literal check
    # forever, which is the failure mode this whole item is about.
    for name in DROP_FILING_FUNCS:
        assert f"def {name}(" in src, f"{name} contributed nothing to the region"
    assert '"status"' in src and '"priority"' in src and '"board"' in src, src
    # What the status must BE is pinned by the two tests above, which drive the route
    # and read back the file it wrote. Requiring the word here as source text would only
    # reject a later refactor that names the value through a constant, so this sweep
    # checks the one thing that must never return: the retired literal.
    assert "up_next" not in src, [
        f"{name}:{n}: {line.strip()}" for name in DROP_FILING_FUNCS
        for n, line in enumerate(
            inspect.getsource(getattr(autonomy, name)).splitlines(), 1)
        if "up_next" in line]

    filed = " ".join((autonomy._file_daily_note_mismatch.__doc__ or "").split())
    assert "Board and priority follow" in filed, filed
    assert "does not" in filed, "the divergence needs stating, not implying"
    assert "draft" in filed and "triage" in filed, filed
    assert "up_next" not in filed, filed
    assert "notify.py:514-519" not in filed, (
        "the stale range citation is back: the guardian's filing payload moved, so a "
        "rationale must not hang on a line range")

    test_prose = " ".join(
        (test_a_dropped_daily_note_line_files_one_backlog_item_on_the_lloyd_board
         .__doc__ or "").split())
    assert "up_next" not in test_prose, test_prose
    assert "draft" in test_prose, test_prose


def test_a_closed_drop_item_does_not_absorb_the_next_mismatch(aut, board_server,
        monkeypatch):
    """The coalescing rule's other half: closing the item closes that incident.

    Left out of the graded clauses by the triage and pinned anyway, because a refresh
    that found a `done` item would silently reopen something a person closed — the loop
    overruling the one reader who engaged with it. `status: done` here goes through the
    route rather than the front matter, so it is the same close the board performs.
    """
    path = _seed_note_with_body()
    _clobber_on_write(monkeypatch, path.read_text(encoding="utf-8"))
    autonomy.append_daily_alert_line("Autonomy #7 failed 3 times (run #42)")
    _, fm, closed_body = board_server.items()[0]
    asyncio.run(board_server.routes.backlog_task_update(
        _FakeRequest({"id": fm["id"], "status": "done"})))
    before = board_server.items()
    assert before[0][1]["status"] == "done", before[0][1]

    assert autonomy.append_daily_alert_line("Autonomy #7 failed 3 times (run #44)") is False

    items = board_server.items()
    assert len(items) == 2, [str(p) for p, _, _ in items]
    fresh = [entry for entry in items if entry[1]["status"] != "done"]
    assert len(fresh) == 1, [entry[1] for entry in items]
    assert _field(fresh[0][2], "Mismatches observed") == "1", fresh[0][2]
    assert _field(_closed_body(board_server, fm["id"]), "Mismatches observed") == "1", \
        "the closed incident keeps the tally it was closed with"


def _closed_body(board: _Board, task_id: int) -> str:
    for path, fm, body in board.items():
        if fm["id"] == task_id:
            return body
    raise AssertionError(f"backlog #{task_id} vanished from the board")


def test_an_alert_that_lands_files_nothing_and_opens_no_socket(aut, board_server):
    """Clause 3: the filing hangs off the mismatch, so a delivered alert is silent.

    Asserted against the board's directory AND against the transport: `requests` empty
    is the stronger half, because an implementation that posted a create on every
    append and let the route decide would leave this directory empty on a happy day
    anyway, and only the call list shows it never reached for the backend at all.
    """
    assert autonomy.append_daily_alert_line(
        "Autonomy #8 disabled after 3 failures (run #43)") is True
    assert board_server.requests == [], board_server.requests
    assert board_server.items() == [], [str(p) for p, _, _ in board_server.items()]
    assert _bullet_lines(), "the line itself still landed, which is why nothing filed"


def test_a_backlog_create_that_raises_still_returns_false_and_still_logs_the_error(
        aut, board_server, caplog):
    """Clause 4, mode 1: a bug in the new dependency may not cost the alarm.

    A bare `RuntimeError` is the mode that matters most: `_backend_json` swallows the
    transport's own exceptions by contract, so what reaches the caller uncaught in
    production is a failure from this code's own logic, which is exactly what a
    defensive `try` around the filing exists for.
    """
    _note_keeping_nothing()
    board_server.fail_create = RuntimeError
    with caplog.at_level(logging.ERROR, logger="lloyd-autonomy"):
        assert autonomy.append_daily_alert_line(
            "Autonomy #9 failed 3 times (run #44)") is False
    # The attempt itself is the half a writer that never files would otherwise satisfy:
    # the code reached for the board, the board blew up in the transport, and neither
    # the exception nor a missing item cost the caller its False.
    assert board_server.paths_of(CREATE_PATH) == [CREATE_PATH], board_server.requests
    assert len(_error_lines(caplog)) == 1, _error_lines(caplog)
    assert board_server.items() == [], [str(p) for p, _, _ in board_server.items()]


def test_a_refused_backlog_connection_still_returns_false_and_still_logs_the_error(
        aut, refused_backend, caplog):
    """Clause 4, mode 2: the backend down, which is also the note's own bad day.

    The one mode that is not hypothetical: a vault-backed backend that is unreachable
    and a daily note that is being rewritten underneath the writer are frequently the
    same outage, and this is the state in which the alert still has to get out.
    `refused_backend` is the module-wide refusal, so the assertion here is that the
    attempt happened and was absorbed.
    """
    _note_keeping_nothing()
    with caplog.at_level(logging.ERROR, logger="lloyd-autonomy"):
        assert autonomy.append_daily_alert_line(
            "Autonomy #10 failed 3 times (run #45)") is False
    assert refused_backend, "the alert path never reached for the backlog at all"
    assert any(TASKS_PATH in url for url in refused_backend), refused_backend
    assert len(_error_lines(caplog)) == 1, _error_lines(caplog)


def test_a_backlog_create_answered_unsuccessful_still_returns_false_and_still_logs(
        aut, board_server, caplog):
    """Clause 4, mode 3: a 200 whose body says no is still a failure, and a quiet one.

    `{"success": false}` arrives with HTTP 200 — every validation refusal in
    `app/routers/backlog.py:704` does — so an implementation that looked only at the
    status would log a filing that never happened, and the count would then be the
    number of times the code believed it had filed.
    """
    _note_keeping_nothing()
    board_server.reply_create = {"success": False, "error": "boom"}
    with caplog.at_level(logging.ERROR, logger="lloyd-autonomy"):
        assert autonomy.append_daily_alert_line(
            "Autonomy #11 failed 3 times (run #46)") is False
    assert board_server.paths_of(CREATE_PATH) == [CREATE_PATH], board_server.requests
    assert len(_error_lines(caplog)) == 1, _error_lines(caplog)
    assert board_server.items() == [], [str(p) for p, _, _ in board_server.items()]


def test_a_drop_item_the_backend_cannot_be_read_back_is_left_as_it_stands(
        aut, board_server, caplog):
    """The refresh's own failure mode, which the never-raises clause covers in practice.

    `force_body_replace` means a refresh is not additive: it overwrites the body with
    whatever the caller assembled, so assembling one from an unread item would post a
    body containing only this mismatch and move a tally that says 4 back to 1. Refusing
    to refresh is the safe half — the ERROR line and the item's own last-known count
    both still stand — and it must cost nothing else, including the append's False.
    """
    _note_keeping_nothing()
    assert autonomy.append_daily_alert_line("Autonomy #12 failed (run #47)") is False
    board_server.hide_detail = True
    caplog.clear()
    # WARNING, not ERROR: the half being asserted here is the refusal's own log line,
    # and a level of ERROR would drop the record the assertion is about.
    with caplog.at_level(logging.WARNING, logger="lloyd-autonomy"):
        assert autonomy.append_daily_alert_line("Autonomy #12 failed (run #48)") is False
    items = board_server.items()
    assert len(items) == 1, [str(p) for p, _, _ in items]
    assert _field(items[0][2], "Mismatches observed") == "1", items[0][2]
    assert board_server.paths_of(UPDATE_PATH) == [], \
        "a body assembled from nothing must not be posted over a tally it cannot see"
    assert board_server.paths_of(TASKS_PATH) == [TASKS_PATH, TASKS_PATH], \
        board_server.requests
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("advance its count" in msg for msg in warnings), warnings
    assert len(_error_lines(caplog)) == 1, _error_lines(caplog)


async def test_the_drop_branch_says_so_when_the_note_refuses_the_alert(aut, caplog):
    """Clause 5: the witness `app/discord_notify.py:115-117` has never had.

    Before #1736 the drop path appended and returned unconditionally, so the only
    branch that ever KNEW an alarm had been lost kept its knowledge to itself; #1736
    added the warning and left it unpinned, which is why the triage for this item could
    say that deleting those three lines turned no test red. The assertion is on the
    warning text, so deleting the `if not appended:` block — or downgrading it to a
    debug line, which `at_level(WARNING)` would then drop — turns this node red.
    """
    from app import discord_notify

    _seed_note_with_body()
    _note_keeping_nothing()
    with caplog.at_level(logging.WARNING, logger="lloyd-server"):
        await discord_notify.discord_alert("watchdog says the fleet is silent")

    warnings = [r.getMessage() for r in caplog.records
                if r.name == "lloyd-server" and r.levelno == logging.WARNING]
    assert any("the daily note refused the alert too" in msg for msg in warnings), \
        f"the drop branch kept its knowledge to itself: {warnings}"
    assert any("exists only in this log line" in msg for msg in warnings), warnings
    assert not _bullet_lines(), "and the alarm is still not in the note"


async def test_a_drop_files_its_item_from_inside_the_worker_s_running_loop(
        aut, board_server):
    """The process boundary this feature crosses, walked from the caller that has a loop.

    The caller that drops an alert is an `async def` running in `lloyd-agent-worker`:
    `workers/sources/scheduled_task.py:164` and `workers/fleet_watchdog.py:498` both
    `await discord_alert`, which reaches `_survive_the_dropped_alert` and this writer — a
    separate supervisor program from `lloyd-backend`, which is why the filing is a
    cross-process POST. The filing is a blocking `urlopen` on that stack, bounded at
    `autonomy._BACKEND_ALERT_TIMEOUT_SECONDS` = 3.0 rather than the guardian's 5.0, so
    a wedged backend costs a stalled tick and not a stalled worker. This node is what
    proves the whole async path completes and files: a version that nested
    `asyncio.run` inside the running loop, or moved the post onto a thread nobody
    joined, would fail here while every sync node above stayed green.
    """
    from app import discord_notify

    assert autonomy._BACKEND_ALERT_TIMEOUT_SECONDS <= 3.0, \
        "the client bound on a blocking call made from a running event loop"
    _seed_note_with_body()
    _note_keeping_nothing()
    await discord_notify.discord_alert("watchdog says the fleet is silent")

    items = board_server.items()
    assert len(items) == 1, [str(p) for p, _, _ in items]
    assert _field(items[0][2], "Mismatches observed") == "1", items[0][2]


# ── #2037 clause 5: a death charged from the pool reaches THIS seam, not a new one ──


@pytest.mark.asyncio
async def test_four_charged_pool_deaths_append_exactly_one_fast_failure_line(
        aut, monkeypatch):
    """A task that dies on import every minute now says so once, in the daily note.

    Task #74 died 41 times in 27 minutes on 2026-09-29 and nothing on the box
    said so: every alert that could name a failure lives inside `_record_failure`
    (the disable line, this streak line, the infra-ceiling line), and a death in
    `scheduled_task.execute`'s imports never reached it. Both stall alarms skip a
    task with a live queue row, and the churn keeps one live, so the silence was
    structural rather than incidental.

    The change under test is the CHARGE, not a new alert: the deaths are driven
    through `charge_death_without_verdict`, and the seam they have to reach is
    `_append_fast_failure_alert` — spied on here to prove the line is that
    writer's, since a second transport would fill the note and still leave two
    places for the alert to be wrong. Four deaths because the clause is about the
    fourth: `len(streak) == 3` is one line per streak, not one per failure.

    An import death is sub-second, so it is a fast failure by this file's own
    definition, and `max_retries: 5` keeps the task un-retired at the third
    death: the line is the streak's doing, not a side effect of a disable.
    """
    import app.discord_notify as discord_notify

    async def _no_alert(*_a, **_k):
        return None

    monkeypatch.setattr(discord_notify, "discord_alert", _no_alert)
    calls: list = []
    real_writer = aut._append_fast_failure_alert

    def spy(task, task_id, durations):
        calls.append((task_id, len(durations)))
        return real_writer(task, task_id, durations)

    monkeypatch.setattr(aut, "_append_fast_failure_alert", spy)
    write_task(aut, 91, skill_name="some-skill")

    for i in range(4):
        started = dt.datetime.now(dt.timezone.utc)
        await aut.charge_death_without_verdict(
            91, ModuleNotFoundError("No module named 'app.discord_notify'"),
            run_id=f"run_91_20261001_00000{i}_a{i}", started_at=started.isoformat())

    assert calls == [(91, 3)], (
        f"the seam was called {calls}: a fourth line for one streak, or none at all")
    lines = _alert_lines()
    assert len(lines) == 1, f"the note carries {len(lines)} lines: {lines}"
    assert "#91" in lines[0], lines[0]
    assert read_task(aut, 91)["failure_count"] == 4, (
        "the alert fired over deaths that never reached the retry budget, which "
        "means the loop is still running underneath the sentence about it")
