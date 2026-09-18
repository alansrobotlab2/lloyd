"""Shared resilient frontmatter parser (agent_mcp._shared.parse_frontmatter_text).

Pins the three-layer recovery contract: plain YAML → orphaned-tags repair →
regex field extraction. A record may come back degraded (_yaml_broken) but
never None — the silent-task-drop failure mode (2026-05-28, 34/40 tasks)
must stay dead in every reader of agent-written frontmatter.

The second half is the write side of the same corruption: a parser that
returns an empty dict instead of raising is safe to read and fatal to write
from, so the backlog writers that dump the parsed dict back over the file
refuse that shape (#1020). `scripts/automod/backlog.py` reads with
`_split_frontmatter`, which defaults to `{}` exactly as this file's reader
does — the reader's resilience is the writer's danger.
"""
import logging
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp._shared import parse_frontmatter_text
from scripts.automod import backlog as B


def test_clean_yaml_parses_normally():
    fm = parse_frontmatter_text("id: 38\nname: nightly\ntags: [a, b]\n")
    assert fm == {"id": 38, "name": "nightly", "tags": ["a", "b"]}
    assert "_yaml_broken" not in fm


def test_orphaned_tags_corruption_recovers():
    # The exact corruption shape from project_autonomy_silent_task_drop:
    # inline tags followed by orphaned block-list items.
    fm_text = (
        "id: 38\n"
        "name: nightly reflection\n"
        "tags: [38-foo, autonomy, pipeline]\n"
        "- nightly\n"
        "- reflection\n"
        "status: active\n"
    )
    fm = parse_frontmatter_text(fm_text, log_label="test")
    assert fm["id"] == 38
    assert fm["status"] == "active"
    assert set(fm["tags"]) == {"38-foo", "autonomy", "pipeline", "nightly", "reflection"}
    assert "_yaml_broken" not in fm


def test_unparseable_yaml_falls_back_to_regex():
    # Unquoted colon in a value — classic agent-written YAML breakage that
    # the tags repair can't fix.
    fm_text = (
        "id: 12\n"
        "name: fix: the thing: again\n"
        "status: active\n"
        "priority: high\n"
    )
    fm = parse_frontmatter_text(
        fm_text, fallback_fields=("id", "name", "status", "priority"), log_label="test"
    )
    assert fm["_yaml_broken"] is True
    assert fm["id"] == 12  # scalar re-parse recovers the int
    assert fm["status"] == "active"
    assert fm["priority"] == "high"
    assert "fix" in fm["name"]


def test_non_mapping_frontmatter_degrades_not_none():
    fm = parse_frontmatter_text("- just\n- a\n- list\n", fallback_fields=("id",))
    assert isinstance(fm, dict)
    assert fm["_yaml_broken"] is True


def test_never_returns_none_on_garbage():
    fm = parse_frontmatter_text("{{{{:::not yaml at all\x00", fallback_fields=("id",))
    assert isinstance(fm, dict)


# ===========================================================================
# The write side: a backlog writer must not rewrite what it could not parse
# (#1020)
# ===========================================================================

GOOD_FM = {
    "status": "draft",
    "priority": "high",
    "board": "lloyd",
    "tags": ["backlog"],
    "activity_log": ["**2026-09-01T00:00:00.000000** — created"],
}


def item_file(tmp_path: Path, *, corrupt: bool) -> Path:
    """One backlog item, optionally carrying the corruption that empties the parse.

    The corruption is the shape that reaches this module in the wild — a value
    with an unterminated quote, the family #918 and #866 document — inserted
    inside the fenced block. The parse then raises inside `_split_frontmatter`,
    which swallows it and returns an empty dict: readable, and exactly what a
    writer that re-dumps the dict turns into a rewritten file.
    """
    text = (f"---\n{yaml.dump(dict(GOOD_FM), default_flow_style=False, sort_keys=False)}"
            "---\n\n# The item\n\nBody of the item.\n")
    if corrupt:
        lines = text.split("\n")
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                lines.insert(i, 'note: "unterminated quote')
                break
        else:
            raise AssertionError("fixture has no closing fence to corrupt against")
        text = "\n".join(lines)
    path = tmp_path / "1020-the-item.md"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def scratch_board(tmp_path, monkeypatch):
    """Point the module's board walk at a scratch dir, as the unattended tests do."""
    monkeypatch.setattr(B, "BACKLOG_DIR", tmp_path)
    return tmp_path


# --- the two writers the item names -------------------------------------------------

def test_apply_status_refuses_a_file_whose_front_matter_parsed_to_no_keys(tmp_path):
    path = item_file(tmp_path, corrupt=True)
    before = path.read_text(encoding="utf-8")
    assert B._split_frontmatter(before)[0] == {}, "fixture must parse to no keys"
    assert B._apply_status(path, "up_next", "test move") is False
    assert path.read_text(encoding="utf-8") == before


def test_note_item_refuses_a_file_whose_front_matter_parsed_to_no_keys(scratch_board):
    path = item_file(scratch_board, corrupt=True)
    before = path.read_text(encoding="utf-8")
    # Reachable: `load_item` defaults what it cannot parse, so the file is on
    # the board that `note_item` walks by id.
    assert {i.id for i in B.open_items(None)} == {1020}
    assert B.note_item(1020, "an activity note") is False
    assert path.read_text(encoding="utf-8") == before


# --- the three the item found beside them ------------------------------------------

@pytest.mark.parametrize("writer", ["tag_item", "close_landed", "record_verdict"])
def test_the_other_parse_then_write_writers_refuse_the_same_shape(scratch_board, writer):
    path = item_file(scratch_board, corrupt=True)
    before = path.read_text(encoding="utf-8")
    item = B.load_item(path)
    if writer == "tag_item":
        refused = B.tag_item(1020, add=("probe",)) is False
    elif writer == "close_landed":
        refused = B.close_landed(item, commit="a" * 40, round_id="SM_TEST",
                                settled_at="2026-09-18T00:00:00Z", close=True,
                                why="test") is None
    else:
        refused = B.record_verdict(item, "confirmed", "evidence",
                                   acceptance="it passes") is None
    assert refused, f"{writer} did not refuse an unparseable file"
    assert path.read_text(encoding="utf-8") == before


def test_update_frontmatter_still_refuses_it_after_the_guard_was_shared(scratch_board):
    """The writer the guard came from must not lose it in the refactor."""
    path = item_file(scratch_board, corrupt=True)
    before = path.read_text(encoding="utf-8")
    assert B.update_frontmatter(path, {"priority": "low"}) is False
    assert path.read_text(encoding="utf-8") == before


# --- clause 4: a refusal has to be diagnosable --------------------------------------

def test_every_refusal_is_logged_with_the_writer_and_the_file_path(scratch_board, caplog):
    corrupt = item_file(scratch_board, corrupt=True)
    item = B.load_item(corrupt)
    with caplog.at_level(logging.WARNING, logger="scripts.automod.backlog"):
        B._apply_status(corrupt, "up_next", "test move")
        B.note_item(1020, "an activity note")
        B.tag_item(1020, add=("probe",))
        B.close_landed(item, commit="a" * 40, round_id="SM_TEST",
                       settled_at="2026-09-18T00:00:00Z", close=True, why="test")
        B.record_verdict(item, "confirmed", "evidence", acceptance="it passes")
        B.update_frontmatter(corrupt, {"priority": "low"})
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 6, f"expected one warning per writer, got {messages}"
    for writer in ("_apply_status", "note_item", "tag_item", "close_landed",
                   "record_verdict", "update_frontmatter"):
        named = [m for m in messages if m.startswith(f"{writer} refused")]
        assert len(named) == 1, f"{writer} logged {named}"
        assert "1020-the-item.md" in named[0]


def test_a_refusal_is_not_producible_for_a_file_that_parsed(scratch_board, caplog):
    """Positive control for the log assertion above.

    Without it the six warnings in that test would pass on a guard that fired
    for every file, which is a different — and board-paralysing — bug.
    """
    path = item_file(scratch_board, corrupt=False)
    with caplog.at_level(logging.WARNING, logger="scripts.automod.backlog"):
        assert B._apply_status(path, "up_next", "test move") is True
        assert B.note_item(1020, "an activity note") is True
        assert B.tag_item(1020, add=("probe",)) is True
    assert [r for r in caplog.records
            if r.name.startswith("scripts.automod.backlog")] == []


# --- clause 5's positive controls, per writer --------------------------------------

def test_a_file_whose_front_matter_parses_is_still_written_by_every_guarded_writer(scratch_board):
    """The guard must cost a healthy item nothing, whoever the writer is."""
    path = item_file(scratch_board, corrupt=False)
    item = B.load_item(path)
    lines = lambda: len(B._split_frontmatter(path.read_text(encoding="utf-8"))[0]["activity_log"])
    assert lines() == len(GOOD_FM["activity_log"])
    # The two writers that touch no log line: `update_frontmatter` called with no
    # `activity`, and `tag_item`, which has no activity parameter at all.
    assert B.update_frontmatter(path, {"priority": "medium"}) is True
    assert lines() == len(GOOD_FM["activity_log"])
    assert B._apply_status(path, "up_next", "test move") is True
    assert B.note_item(1020, "an activity note") is True
    assert B.tag_item(1020, add=("probe",)) is True
    assert B.close_landed(item, commit="b" * 40, round_id="SM_TEST",
                          settled_at="2026-09-18T00:00:00Z", close=False,
                          why="test") == path
    assert B.record_verdict(B.load_item(path), "confirmed", "evidence",
                            acceptance="it passes") == path
    fm, _ = B._split_frontmatter(path.read_text(encoding="utf-8"))
    assert fm["priority"] == "medium"
    assert fm["board"] == "lloyd"          # survived the round of six writes
    assert fm["automod_landed"] == "b" * 40
    assert "probe" in fm["tags"]
    # `_apply_status`, `note_item`, `close_landed` and `record_verdict` each append
    # exactly one line, so the log is the fixture's plus those four — measured off
    # the file, against the two writers pinned above as adding none.
    assert len(fm["activity_log"]) == len(GOOD_FM["activity_log"]) + 4
