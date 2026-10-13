"""One fallback-field list, three readers — backlog #1014.

`agent_mcp._shared.parse_frontmatter_text` recovers a YAML-broken frontmatter
block by regex over the `fallback_fields` it is handed, and *only* over those:
a field missing from the list is silently absent from a degraded record. The
doc's invariant ("The parser can never drop a task", architecture/autonomy.md)
therefore lives in the field list, not in the parser — and until now there were
three lists. At HEAD 551e904 the scheduler passed 25 fields
(`autonomy._parse_task_file`), `agent_mcp/autonomy._parse_task_file` passed 13,
and `app/routers/autonomy._autonomy_parse` passed none at all and returned
`None` on the failure instead.

That asymmetry is not cosmetic because the MCP reader is a read-modify-write
path: `agent_mcp/autonomy._write_task_file` starts from the file's prior parse,
so a 13-field recovery followed by a write rebuilds the file from 13 fields. A
degraded task read by the MCP reader comes back window-less, chain-less and
model-less (`preferred_hours: []`, `model: ''`, `depends_on` gone) while the
scheduler is still dispatching it on the values it read for itself.

So the field list is now ONE constant, `AUTONOMY_TASK_FIELDS`, defined beside
the parser that consumes it and imported by all three readers. This file pins
three things: the two file readers return identical values for the twelve
fields that were diverging; each reader hands the parser the one shared
constant (identity, not a copy); and `board_id` is not in it — that name
appears in the MCP list only, in 0 of the live task files, and in no scheduler
code path.

The assertion deliberately does NOT claim that every frontmatter key in use is
in the list. The live fleet carries keys outside it (`tags`, `segment`,
`pipeline`, `cron_id`, `max_turns`-adjacent extras among ~16 names); the claim
that holds, and the only one worth pinning, is that the readers agree.
"""

from __future__ import annotations

import datetime
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import autonomy as SCHED
from agent_mcp._shared import AUTONOMY_TASK_FIELDS
from app.routers import autonomy as ROUTER
import agent_mcp.autonomy as MCP

# The twelve fields the item measured as diverging between the scheduler's list
# and the MCP reader's, at the HEAD this round was cut from.
DIVERGING_FIELDS = (
    "preferred_hours", "depends_on", "model", "stale_bypass_hours",
    "inner_voice", "auto_advance", "preemptible", "max_retries",
    "failure_count", "runs_per_day", "last_attempt", "expected_error_patterns",
)

# `name: ...: ...` is the classic agent-written breakage (an unquoted colon):
# strict YAML fails, the orphaned-tags repair cannot help, so the regex
# fallback is the only layer left standing. The values are chosen so every one
# of the twelve is recoverable by regex and type-typed by it too.
BROKEN_TASK_FILE = """---
id: 7
name: nightly: reflection: knowledge write
status: up_next
frequency: daily
preferred_hours: [1, 2, 3, 4]
depends_on: 42
model: eco
stale_broken_canary: yes
stale_bypass_hours: 24
inner_voice: false
auto_advance: true
preemptible: false
max_retries: 5
failure_count: 2
runs_per_day: 1
last_attempt: '2026-09-19T22:11:42Z'
expected_error_patterns: [TimeoutError, OOM]
skill_name: nightly-reflection-knowledge-write
tags: [autonomy, nightly]
---

# body

## Activity Log

- 2026-09-19T22:11:42Z: run completed
"""


@pytest.fixture
def broken_file(tmp_path) -> Path:
    path = tmp_path / "7-nightly-reflection-knowledge-write.md"
    path.write_text(BROKEN_TASK_FILE, encoding="utf-8")
    return path


def _spy_captured_lists(monkeypatch, module) -> list:
    """Record every `fallback_fields` value `module` hands the shared parser.

    Identity is what is being pinned, so the spy delegates to the real parser
    rather than stubbing it: the reader still has to work.
    """
    captured: list = []
    real = module.parse_frontmatter_text

    def spy(fm_text, *, fallback_fields=(), log_label="frontmatter"):
        captured.append(fallback_fields)
        return real(fm_text, fallback_fields=fallback_fields, log_label=log_label)

    monkeypatch.setattr(module, "parse_frontmatter_text", spy)
    return captured


# ── clause 1: the two file readers agree field-for-field ─────────────────────


def test_both_file_readers_recover_the_twelve_diverging_fields_identically(broken_file):
    """Same bytes in, same values out — the acceptance check for #1014.

    Before the fix the MCP reader's 13-field list recovered none of these, so
    `preferred_hours` came back `[]`, `model` `''`, `depends_on` `None` and
    `inner_voice` was not even a key of its result, while the scheduler
    recovered all twelve from the same file.
    """
    scheduler_task = SCHED._parse_task_file(broken_file)
    mcp_task = MCP._parse_task_file(broken_file)

    assert scheduler_task is not None and mcp_task is not None
    # Positive control: the fixture really does exercise the fallback layer,
    # otherwise "the two readers agree" would be agreement between two YAML
    # parses and prove nothing about the field list.
    assert scheduler_task.get("_yaml_broken") is True, "fixture must degrade to regex"
    assert mcp_task.get("_yaml_broken") is True, "fixture must degrade to regex"

    for field in DIVERGING_FIELDS:
        assert field in mcp_task, f"MCP reader dropped {field!r} from its result"
        assert mcp_task[field] == scheduler_task[field], (
            f"{field}: MCP read {mcp_task[field]!r}, scheduler read "
            f"{scheduler_task[field]!r}")

    # And not merely equal to each other — equal to what the file says. Two
    # readers agreeing on `None` for all twelve would satisfy the pairwise
    # check above and still be the bug.
    assert scheduler_task["preferred_hours"] == [1, 2, 3, 4]
    assert scheduler_task["depends_on"] == 42
    assert scheduler_task["model"] == "eco"
    assert scheduler_task["stale_bypass_hours"] == 24
    assert scheduler_task["inner_voice"] is False
    assert scheduler_task["last_attempt"] == "2026-09-19T22:11:42Z"
    assert scheduler_task["expected_error_patterns"] == ["TimeoutError", "OOM"]


# ── clause 2: one shared constant, imported, no board_id ─────────────────────


def test_the_scheduler_hands_the_parser_the_shared_constant(broken_file, monkeypatch):
    captured = _spy_captured_lists(monkeypatch, SCHED)
    SCHED._parse_task_file(broken_file)
    assert any(fields is AUTONOMY_TASK_FIELDS for fields in captured), (
        f"scheduler passed {captured!r}, not the shared constant")


def test_the_mcp_reader_hands_the_parser_the_shared_constant(broken_file, monkeypatch):
    captured = _spy_captured_lists(monkeypatch, MCP)
    MCP._parse_task_file(broken_file)
    assert any(fields is AUTONOMY_TASK_FIELDS for fields in captured), (
        f"MCP reader passed {captured!r}, not the shared constant")


def test_the_board_reader_hands_the_parser_the_shared_constant(broken_file, monkeypatch):
    captured = _spy_captured_lists(monkeypatch, ROUTER)
    parsed = ROUTER._autonomy_parse(broken_file)
    assert any(fields is AUTONOMY_TASK_FIELDS for fields in captured), (
        f"board reader passed {captured!r}, not the shared constant")
    assert parsed is not None and parsed.get("_yaml_broken") is True


def test_the_shared_constant_has_no_board_id():
    """`board_id` is in the MCP list only: 0 of the live task files carry it and
    no scheduler code path reads it. It is the tell that the lists were never
    one thing, so it does not get to define the union."""
    assert "board_id" not in AUTONOMY_TASK_FIELDS


def test_the_mcp_reader_stops_inventing_board_id(broken_file):
    """The reader used to emit `board_id: 4` for every task, defaulted from
    nothing — a field the board it claims to describe never reads."""
    assert "board_id" not in MCP._parse_task_file(broken_file)


def test_the_shared_constant_covers_every_field_the_scheduler_used_to_list():
    """Moving the list into one constant must not silently lose a name. The
    scheduler's list is the dispatch-critical one, so its 25 names are the
    floor; `board_id` is the only name the union used to have that is gone."""
    scheduler_only = (
        "id", "name", "description", "status", "priority", "frequency",
        "scheduled_at", "next_run", "last_run", "last_attempt", "agent_id",
        "skill_name", "timeout_seconds", "max_turns", "preemptible",
        "auto_advance", "depends_on", "max_retries", "failure_count",
        "runs_per_day", "preferred_hours", "model", "stale_bypass_hours",
        "expected_error_patterns", "inner_voice",
    )
    assert len(scheduler_only) == 25
    assert set(scheduler_only) <= set(AUTONOMY_TASK_FIELDS)
    assert len(AUTONOMY_TASK_FIELDS) == len(set(AUTONOMY_TASK_FIELDS))


def test_a_healthy_file_is_unaffected_by_the_shared_list(broken_file, tmp_path):
    """The fix must not read as a parser that now fails healthy files: clean
    YAML still parses clean, with no `_yaml_broken` and no regex involvement."""
    healthy = tmp_path / "8-healthy.md"
    healthy.write_text(
        "---\nid: 8\nname: healthy task\nstatus: up_next\n"
        "preferred_hours: [5, 6]\ndepends_on: 7\nmodel: primary\n---\n\n# body\n",
        encoding="utf-8")
    scheduler_task = SCHED._parse_task_file(healthy)
    mcp_task = MCP._parse_task_file(healthy)
    board_task = ROUTER._autonomy_parse(healthy)
    assert "_yaml_broken" not in scheduler_task
    assert not mcp_task.get("_yaml_broken")
    assert not board_task.get("_yaml_broken")
    assert scheduler_task["preferred_hours"] == [5, 6]
    assert mcp_task["preferred_hours"] == [5, 6]
    assert mcp_task["depends_on"] == 7
    assert mcp_task["model"] == "primary"


def test_a_file_with_no_frontmatter_block_is_still_dropped_by_every_reader(tmp_path):
    """The degraded-file recovery is about broken YAML, not about missing
    frontmatter: a file with no block at all is not a task, and #1014 must not
    turn the board into a list of every markdown file in the directory."""
    note = tmp_path / "9-release-note.md"
    note.write_text("# just prose, no frontmatter\n", encoding="utf-8")
    assert SCHED._parse_task_file(note) is None
    assert MCP._parse_task_file(note) is None
    assert ROUTER._autonomy_parse(note) is None


def test_the_infra_ceiling_hold_survives_a_degraded_task_file(tmp_path):
    """#1085's two fields are in the shared list because they are dispatch-
    critical in the same sense `failure_count` is, and this is the test that says
    which of them is the dangerous one.

    `infra_rest_until` IS the hold: `_in_infra_rest` is a due-gate, so a task file
    that recovered without it comes back with no rest recorded and dispatches on
    the very outage that just bounded it. `infra_failure_count` is the softer
    half — losing it only restarts the count from 1, so the ceiling is reached
    one failure late. Both belong in the fallback layer; the last thing this list
    should be asked to carry is state that, when dropped, makes a hold invisible.

    The positive control is the same `_yaml_broken` flag the twelve-diverging
    fields use: without it this assertion could be satisfied by a plain YAML parse
    and would prove nothing about the field list.
    """
    until = "2026-09-23T09:00:00+00:00"
    broken = tmp_path / "12-broken-infra.md"
    broken.write_text(
        "---\nid: 12\nname: nightly: infra ceiling holder\nstatus: up_next\n"
        f"frequency: daily\ninfra_failure_count: 5\ninfra_rest_until: '{until}'\n"
        "skill_name: some-skill\n---\n\n# body\n\n## Activity Log\n",
        encoding="utf-8")

    task = SCHED._parse_task_file(broken)
    assert task is not None and task.get("_yaml_broken") is True, (
        "the fixture must degrade to the regex fallback, or nothing here tests "
        "the field list")
    assert task.get("infra_rest_until") == until
    assert int(task["infra_failure_count"]) == 5
    # The recovered record still holds, six minutes before it ends.
    assert SCHED._in_infra_rest(
        task, datetime.datetime.fromisoformat("2026-09-23T08:54:00+00:00"))


# ── #1085 review finding: the two PROJECTIONS dropped the ceiling ────────────
#
# The clause above is about the fallback field list, which only runs on a
# YAML-broken file. Both other readers then project a HAND-WRITTEN key subset on
# the healthy path, and that subset is what every consumer actually sees:
# `agent_mcp/autonomy._parse_task_file` is the MCP tool's answer in another
# process, and `app/routers/autonomy._autonomy_parse` is the row behind
# `GET /api/autonomy/tasks` — whose list handler calls `autonomy.hold_reason` on
# ITS OWN projection two statements later. A subset without `infra_rest_until`
# therefore answered "nothing is holding this task" for a task the scheduler was
# refusing for a whole declared period, on the one surface a human reads to find
# out why a task stopped running. Same shape as the #1014 bug this file exists
# for, one layer up: readers disagreeing in the direction that looks healthy.


# The rest must still be running when the board answers, so the stamp is
# generated from the clock rather than hard-coded: a literal date here would
# expire, and an expired rest makes `hold_reason` correctly answer `None` and the
# test would then be pinning nothing.
REST_UNTIL = (datetime.datetime.now(datetime.timezone.utc)
              + datetime.timedelta(hours=3)).replace(microsecond=0)


def _ceilinged_file_text() -> str:
    crossed = REST_UNTIL - datetime.timedelta(hours=1)
    return (
        "---\n"
        "id: 12\n"
        "name: Infra ceiling holder\n"
        "status: up_next\n"
        "frequency: hourly\n"
        f"last_run: '{(crossed - datetime.timedelta(hours=1)).isoformat()}'\n"
        f"last_attempt: '{crossed.isoformat()}'\n"
        "failure_count: 0\n"
        # `hold_reason`'s FIRST gate is `no skill`, which pre-empts every hold
        # below it — a skill-less fixture would report "no skill" and prove
        # nothing about the ceiling.
        "skill_name: some-skill\n"
        "infra_failure_count: 5\n"
        f"infra_rest_until: '{REST_UNTIL.isoformat()}'\n"
        "---\n\n# body\n")


@pytest.fixture
def ceilinged_dir(tmp_path, monkeypatch):
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    (dirn / "12-ceiling-holder.md").write_text(_ceilinged_file_text(), encoding="utf-8")
    monkeypatch.setattr(SCHED, "AUTONOMY_DIR", dirn)
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", dirn)
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", dirn)
    return dirn


def test_all_three_readers_answer_the_hold_on_a_healthy_file(ceilinged_dir):
    """The scheduler, the MCP tool and the board must give the same answer for
    the same bytes. Compared through the scheduler's own `_parse_iso` because the
    two projections serialise a YAML timestamp their own way; what has to agree is
    WHEN THE REST ENDS, not the string spelling of it."""
    path = ceilinged_dir / "12-ceiling-holder.md"
    until = REST_UNTIL
    readers = {
        "scheduler": SCHED._parse_task_file(path),
        "mcp": MCP._parse_task_file(path),
        "board": ROUTER._autonomy_parse(path),
    }
    for label, task in readers.items():
        assert task is not None, f"{label} dropped the task entirely"
        assert SCHED._parse_iso(task.get("infra_rest_until")) == until, (
            f"{label} cannot see the infra rest, so it reports a ceilinged task "
            f"as free to run: {task.get('infra_rest_until')!r}")
        assert int(task.get("infra_failure_count") or 0) == 5, (
            f"{label} cannot see the ceiling counter")


def test_the_board_row_reports_the_infra_ceiling_as_the_hold(ceilinged_dir):
    """The seam in one assertion: `GET /api/autonomy/tasks` must not present a
    ceilinged task as unheld. Before the projection carried the field this route
    called `hold_reason` on a dict with no `infra_rest_until`, so the board's
    `blocked` was the answer to a question about an incomplete record."""
    import asyncio
    import json

    response = asyncio.run(ROUTER.autonomy_tasks())
    rows = json.loads(response.body)["tasks"]
    row = next(r for r in rows if r["id"] == 12)
    assert row["blocked"] is not None, (
        "the board reported a task resting on the infra ceiling as not held at "
        "all — the field that sets the hold never reached `hold_reason`")
    assert "infra ceiling" in row["blocked"], (
        f"wrong hold on the board: {row['blocked']!r}")


def test_the_board_hold_assertion_bites_on_an_incomplete_projection(ceilinged_dir,
                                                                   monkeypatch):
    """Positive control for the test above, and the reason it is not enough to
    assert a non-`None` hold: a projection that drops the field must make THIS
    fail, or the pair proves nothing about which keys are load-bearing."""
    real = ROUTER._autonomy_parse

    def without_ceiling(path):
        task = real(path)
        if task is not None:
            task.pop("infra_rest_until", None)
        return task

    monkeypatch.setattr(ROUTER, "_autonomy_parse", without_ceiling)
    import asyncio
    import json

    rows = json.loads(asyncio.run(ROUTER.autonomy_tasks()).body)["tasks"]
    row = next(r for r in rows if r["id"] == 12)
    assert "infra ceiling" not in (row["blocked"] or ""), (
        "the hold survived an incomplete projection, so the board test above "
        "was not sensitive to the field at all")


# ── #2640: one DEFAULT for `max_retries`, read from the scheduler's constant ──
#
# Everything above is about which FIELDS a reader recovers. This is about the
# VALUE a reader prints for one of them — the same failure shape one layer up.
# `app/autonomy.py` computed a task's retirement budget as
# `int(task.get("max_retries") or _DEFAULT_MAX_RETRIES)` with
# `_DEFAULT_MAX_RETRIES = 5`, while each of the two projections defaulted the
# field to a literal 3 of its own. For a task file that declares no
# `max_retries:` key — #82, #83, #86, #91, #92 as measured on 2026-10-13, four
# of them with a `last_run` the previous day — the board row, the MCP row and
# Mission Control printed **3** while `_record_failure` was going to retire the
# task at **5**: the budget a person read was not the budget that acted, wrong
# in the direction that handed the task two extra failures. Nothing about
# retirement moved (that half is pinned in
# `tests/test_autonomy_failure_alert.py`); what moved is that both fallbacks are
# now `autonomy.max_retries_budget`, the scheduler's own function over its own
# constant, which is what these tests pin and what makes a re-typed literal
# impossible to reintroduce quietly.

KEYLESS_TASK_FILE = """---
id: 86
name: nightly iv metrics series
status: up_next
frequency: daily
skill_name: iv-metrics-series
---

# body

## Activity Log
"""


@pytest.fixture
def keyless_file(tmp_path) -> Path:
    """A task file declaring no `max_retries:` at all — the five live files' shape."""
    path = tmp_path / "86-nightly-iv-metrics-series.md"
    path.write_text(KEYLESS_TASK_FILE, encoding="utf-8")
    return path


def test_both_projections_answer_with_the_budget_the_scheduler_will_apply(keyless_file):
    """One key-less file, three readers, one number — #2640's acceptance check.

    The comparison is not `5`: it is `SCHED.max_retries_budget(...)`, the value
    `_record_failure` charges the streak against, so the assertion holds what it
    claims (the surfaces match the scheduler) whatever the constant is today. The
    scheduler's own parse is the positive control: it must NOT carry the key, or
    the number below is being read off the file and the fallback is unobserved —
    which is exactly how the old bug stayed invisible for so long.
    """
    declared = SCHED._parse_task_file(keyless_file)
    assert declared is not None
    assert "max_retries" not in declared, (
        "the fixture grew a `max_retries:` key, so every assertion below is "
        "reading the file rather than a fallback")
    budget = SCHED.max_retries_budget(declared)

    board = ROUTER._autonomy_parse(keyless_file)
    mcp = MCP._parse_task_file(keyless_file)

    assert board["max_retries"] == budget, (
        f"GET /api/autonomy/tasks prints {board['max_retries']!r} while the "
        f"scheduler retires this file at {budget}")
    assert mcp["max_retries"] == budget, (
        f"the MCP row prints {mcp['max_retries']!r} while the scheduler retires "
        f"this file at {budget}")
    assert board["max_retries"] == mcp["max_retries"], (
        "the two surfaces print different budgets for one file")
    # The number that used to be printed. Asserting against the constant here
    # keeps the check honest if the constant ever moves: what is banned is a
    # second, reader-local default, not the digit 5.
    assert SCHED._DEFAULT_MAX_RETRIES == budget


def test_a_change_to_the_constant_moves_both_projections_with_no_further_edit(
        keyless_file, monkeypatch):
    """Clause 3's actual property: the fallback is a reference, not a copy.

    A projection that read `_DEFAULT_MAX_RETRIES` once at import time, or that
    re-typed the literal, would keep printing the old number after the constant
    moved — the drift that made this bug possible in the first place. Nine is
    chosen so it cannot collide with the constant's real value or with the 3 the
    readers used to print.
    """
    assert SCHED._DEFAULT_MAX_RETRIES != 9, (
        "the fixture's discriminator is now the constant's real value")
    monkeypatch.setattr(SCHED, "_DEFAULT_MAX_RETRIES", 9)

    assert ROUTER._autonomy_parse(keyless_file)["max_retries"] == 9, (
        "the board's fallback did not follow the constant, so it is a copy")
    assert MCP._parse_task_file(keyless_file)["max_retries"] == 9, (
        "the MCP row's fallback did not follow the constant, so it is a copy")


def test_a_declared_value_still_prints_itself_on_both_projections(tmp_path):
    """The fallback is a fallback: a file that names its own budget wins.

    Most of the fleet declares its own budget — 33 of the 39 task files declared
    `max_retries: 3` when #2640 was triaged on 2026-10-13, and the HTTP create
    route still writes that value into every new file. If the shared function
    overrode a declared value, the "truthful board" fix would have silently
    raised most of the fleet's budget to 5 instead.
    """
    path = tmp_path / "84-declared-two.md"
    path.write_text(
        "---\nid: 84\nname: declared budget\nstatus: up_next\n"
        "frequency: daily\nmax_retries: 2\n---\n\n# body\n", encoding="utf-8")

    assert ROUTER._autonomy_parse(path)["max_retries"] == 2
    assert MCP._parse_task_file(path)["max_retries"] == 2
    assert SCHED._parse_task_file(path)["max_retries"] == 2


def test_neither_projection_re_types_a_default_anymore():
    """The shape, not just the value: both projections must CALL the scheduler.

    A value test can be satisfied by someone who re-types `5` after the constant
    moves. This one cannot: the shared call has to be in the projection, and a
    reader-local default has to be gone — including in a comment, because the
    acceptance grep is over text, and the sentence that quotes the old spelling is
    indistinguishable from having reintroduced it.

    The two banned spellings are ASSEMBLED, not written out: #2640's acceptance
    check is a `git grep` for the reader-default spelling coming back with no
    hits, and a literal written here would fail that check from the very test
    that forbids it.
    """
    banned = ('max_retries' + '", 3', "max_retries" + "', 3")
    for label, fn in (("board", ROUTER._autonomy_parse),
                      ("MCP", MCP._parse_task_file)):
        src = inspect.getsource(fn)
        assert "max_retries_budget(" in src, (
            f"the {label} projection no longer calls the scheduler's "
            "`max_retries_budget`, so its default is its own again")
        for needle in banned:
            assert needle not in src, (
                f"the {label} projection re-types a default: {needle!r}")
