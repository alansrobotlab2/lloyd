"""A degraded autonomy task file must not be round-tripped — backlog #1014 clause 3.

`agent_mcp/autonomy._write_task_file` deliberately starts from the file's
EXISTING frontmatter so keys the module does not model survive an update (that
was #1064's fix). On a file whose YAML is broken the prior parse comes back as
the regex fallback's output — over an empty field list, because that call
passes no `fallback_fields` at all — so the "existing frontmatter" it starts
from is `{}`, and the write rebuilds the file from the `updates` map alone.

Measured on a temp dir before this round, with an
`autonomy_write_task(id=…, activity_note=…)` on a file whose only defect was an
unquoted colon in `name:`: after the write the file had `preferred_hours: []`
(value destroyed), `model: ''` (destroyed), and `depends_on`,
`stale_bypass_hours`, `inner_voice` and `tags` removed outright. The scheduler
kept dispatching that task — so the loss was invisible until the window or the
chain edge was missed. `autonomy_write_task` appears 208× in the retained
transcripts, so the writer is not hypothetical.

The fix shape is already in the tree: `agent_mcp/backlog.py::save_task` refuses
to write a `_yaml_broken` record ("A degraded record is fine to *show*; it is
not fine to round-trip"). Autonomy writers had no such guard. These tests pin
that a write is refused, the file is left byte-for-byte alone, and — the part
that keeps the guard honest — a healthy file still writes normally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_mcp.autonomy as MCP
# The scheduler is the reader whose loss this item is about, so the proof that a
# refusal preserved a task's schedule is that the scheduler still reads it.
import autonomy as SCHED

# Broken for the same reason real ones are: an agent wrote `name: a: b`. The
# four load-bearing lines below are the ones the acceptance clause names —
# window, chain edge, model, staleness override — plus a `tags` key the module
# does not model at all, which is what the previous round's fix (#1064) was for.
BROKEN = """---
id: 7
name: nightly: reflection: knowledge write
status: up_next
frequency: daily
preferred_hours: [1, 2, 3, 4]
depends_on: 42
model: eco
stale_bypass_hours: 24
tags: [autonomy, nightly]
---

# body

## Activity Log

- 2026-09-19T22:11:42Z: run completed
"""

HEALTHY = """---
id: 7
name: Nightly reflection knowledge write
status: up_next
frequency: daily
preferred_hours: [1, 2, 3, 4]
depends_on: 42
model: eco
stale_bypass_hours: 24
---

# body

## Activity Log

- 2026-09-19T22:11:42Z: run completed
"""


@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", dirn)
    return dirn


def _write(dirn: Path, text: str) -> Path:
    path = dirn / "7-nightly-reflection-knowledge-write.md"
    path.write_text(text, encoding="utf-8")
    return path


def _add_activity(dirn: Path) -> dict:
    """The exact call the item names: an activity note, nothing else."""
    return json.loads(MCP._handle_write({"id": 7, "activity_note": "checked in"}))


# ── clause 3 ─────────────────────────────────────────────────────────────────


def test_the_fixture_file_only_parses_by_regex(autonomy_dir):
    """Positive control. Every assertion below about "refused" is only about
    degraded files if this file really does reach the regex layer — a fixture
    that quietly parses as clean YAML would make the whole suite pass while the
    guard never ran."""
    _write(autonomy_dir, BROKEN)
    parsed = MCP._parse_task_file(autonomy_dir / "7-nightly-reflection-knowledge-write.md")
    assert parsed is not None
    assert parsed.get("_yaml_broken") is True
    with pytest.raises(Exception):
        yaml.safe_load(BROKEN.split("---\n")[1])


def test_an_activity_note_on_a_degraded_file_is_refused(autonomy_dir):
    _write(autonomy_dir, BROKEN)
    result = _add_activity(autonomy_dir)
    assert "error" in result, f"write was accepted: {result!r}"
    assert "7" in result["error"], f"the error must name the task: {result['error']!r}"


def test_the_refused_write_leaves_the_file_byte_identical(autonomy_dir):
    """Not "still has most of it" — untouched. The failure this guards produced
    a file that still LOOKED like a task, which is why it went unnoticed."""
    path = _write(autonomy_dir, BROKEN)
    before = path.read_text(encoding="utf-8")
    _add_activity(autonomy_dir)
    assert path.read_text(encoding="utf-8") == before


def test_the_load_bearing_lines_are_still_on_disk_after_the_refusal(autonomy_dir):
    """The four the acceptance clause names, checked as the literal lines the
    scheduler's own regex fallback would read off them."""
    path = _write(autonomy_dir, BROKEN)
    _add_activity(autonomy_dir)
    text = path.read_text(encoding="utf-8")
    assert "preferred_hours: [1, 2, 3, 4]" in text
    assert "depends_on: 42" in text
    assert "model: eco" in text
    assert "stale_bypass_hours: 24" in text
    assert "tags: [autonomy, nightly]" in text


def test_the_scheduler_still_reads_the_window_and_the_chain_after_the_refusal(autonomy_dir):
    """Crosses the reader boundary the loss actually hurt: what dispatch sees.

    Before the fix this read `preferred_hours=[]`, `depends_on=None`,
    `model=''` — i.e. a task dispatched outside its window and off its chain.
    """
    import autonomy as SCHED
    path = _write(autonomy_dir, BROKEN)
    _add_activity(autonomy_dir)
    task = SCHED._parse_task_file(path)
    assert task["preferred_hours"] == [1, 2, 3, 4]
    assert task["depends_on"] == 42
    assert task["model"] == "eco"
    assert task["stale_bypass_hours"] == 24


def test_a_status_change_on_a_degraded_file_is_refused_too(autonomy_dir):
    """The other update branch of the same handler: `status` is a dispatch kill
    switch (#1127), so a write that parks a task AND flattens its frontmatter
    is the worst case, not an edge case."""
    _write(autonomy_dir, BROKEN)
    result = json.loads(MCP._handle_write({"id": 7, "status": "draft"}))
    assert "error" in result, f"status write was accepted: {result!r}"


def test_archiving_a_degraded_file_is_refused_too(autonomy_dir):
    """`autonomy_delete_task(archive=True)` is the third writer through the same
    `_write_task_file`. Refusing to archive costs a person one manual edit;
    round-tripping a degraded file costs the schedule its window."""
    path = _write(autonomy_dir, BROKEN)
    before = path.read_text(encoding="utf-8")
    result = json.loads(MCP._handle_delete({"id": 7, "archive": True}))
    assert "error" in result, f"archive was accepted: {result!r}"
    assert path.read_text(encoding="utf-8") == before


# ── the guard is a guard, not a ban ─────────────────────────────────────────


def test_a_healthy_file_still_writes_an_activity_note(autonomy_dir):
    """If this fails, the guard is refusing every write and the autonomy loop is
    dead — which is a worse bug than the one being fixed."""
    path = _write(autonomy_dir, HEALTHY)
    result = _add_activity(autonomy_dir)
    assert "error" not in result, f"healthy write refused: {result!r}"
    text = path.read_text(encoding="utf-8")
    assert "checked in" in text
    # The window survives as the values, not as a literal flow-style line —
    # `yaml.dump` block-styles a list, so the assertion is on the parse the
    # scheduler does, which is the only one that dispatches anything.
    assert SCHED._parse_task_file(path)["preferred_hours"] == [1, 2, 3, 4]
    assert SCHED._parse_task_file(path)["depends_on"] == 42
    assert SCHED._parse_task_file(path)["model"] == "eco"
    assert "depends_on: 42" in text
    assert "model: eco" in text
    assert "stale_bypass_hours: 24" in text


def test_a_healthy_file_still_archives(autonomy_dir):
    path = _write(autonomy_dir, HEALTHY)
    result = json.loads(MCP._handle_delete({"id": 7, "archive": True}))
    assert result.get("success") is True, f"healthy archive refused: {result!r}"
    assert "status: draft" in path.read_text(encoding="utf-8")


def test_writing_a_degraded_file_directly_still_raises(autonomy_dir):
    """The guard sits in `_write_task_file`, not only in the two handlers, so a
    future caller cannot reach around it by calling the writer itself."""
    _write(autonomy_dir, BROKEN)
    task = MCP._parse_task_file(autonomy_dir / "7-nightly-reflection-knowledge-write.md")
    with pytest.raises(Exception) as excinfo:
        MCP._write_task_file(task)
    msg = str(excinfo.value)
    assert "regex fallback" in msg, f"the refusal must name why, got: {msg}"
    assert "fix" in msg.lower(), f"the refusal must say what to do, got: {msg}"
