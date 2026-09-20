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
