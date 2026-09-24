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
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autonomy as SCHED
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
