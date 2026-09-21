"""Shared resilient frontmatter parser (agent_mcp._shared.parse_frontmatter_text).

Pins the three-layer recovery contract: plain YAML → orphaned-tags repair →
regex field extraction. A record may come back degraded (_yaml_broken) but
never None — the silent-task-drop failure mode (2026-05-28, 34/40 tasks)
must stay dead in every reader of agent-written frontmatter.

The second half is the write side of the same corruption: a parser that
returns an empty dict instead of raising is safe to read and fatal to write
from, so the backlog writers that dump the parsed dict back over the file
refuse that shape (#1020). `scripts.automod.backlog` reads with
`_split_frontmatter`, which defaults to `{}` exactly as this file's reader
does — the reader's resilience is the writer's danger.

The third section is where those two layers meet: *where the block ends*. All
three backlog readers used to find that end by splitting on the substring
`---` (or on a line beginning with it), so an item whose own text quotes that
very expression — an activity-log entry recording the split, which is how #1146
found it — cut the block inside a quoted scalar, failed the parse, and was
reported as `_yaml_broken` and refused by every writer. The rule is now one
rule, in `app.frontmatter`: the block ends at the first line that is exactly
`---`. These tests pin that boundary in all three readers, pin that the two
readers agree with each other, and pin that a genuinely broken file is still
reported as broken — the fix must not read as a parser that never fails.
"""
import contextlib
import json
import logging
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp._shared import parse_frontmatter_text
from app import frontmatter as FM
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


# ===========================================================================
# Where the front-matter block ends: the closing fence is a whole line, not a
# substring (#1146)
# ===========================================================================
#
# Every recovery layer above operates on the text *between* the fences, so what
# they promise depends on finding that span. All three backlog readers used to
# locate it by searching for the fence as text — `content.split("---", 2)` in
# `agent_mcp/backlog.py` and in `app/routers/backlog.py`, and
# `text.split("---\n", 2)` in this module's own `_split_frontmatter`. An item
# that quotes that expression in its activity log is therefore self-referentially
# unparseable: the entry recording the defect cuts the block inside its own
# quoted scalar, the parse dies with "while scanning a quoted scalar", the record
# comes back `_yaml_broken` (or, in this module, with its later keys silently
# relocated into the body), and every writer refuses it. That is how #1146 was
# written at all — `backlog_write_task(task_id=1146)` refused its own front
# matter, so the item's triage section went in as a byte-preserving body edit.
#
# The rule is now one rule, in `app.frontmatter.split_frontmatter`: the block
# ends at the first line that is exactly `---`. The item's other half — a close
# taking `needs-human` off — is pinned in `tests/test_backlog_unattended.py`,
# beside the closes that depend on the tag surviving.
from agent_mcp import backlog as MCP

#: The two characters backslash and `n`, as they sit on disk inside #460's
#: activity entry. YAML single quotes do not interpret backslashes, so a record
#: quoting `text.split('---\n', 2)` holds a literal backslash — and the three
#: hyphens beside it are enough for an unanchored split to cut on.
QUOTED_RULE = "text.split('---\\n', 2)"

#: Keys every item in this section carries, so a block truncated mid-way shows up
#: as a missing key rather than as a merely shorter file.
FENCE_KEYS = {
    "board": "lloyd",
    "status": "up_next",
    "priority": "medium",
    "name": "The item that quotes the fence",
}

FENCE_BODY = "# The item that quotes the fence\n\nBody of the item.\n"

PLAIN_ITEM = "---\nboard: lloyd\nstatus: draft\n---\n\n# Plain\n\nBody.\n"


def fence_item_text() -> str:
    """One backlog item whose front matter holds the fence in two more ways.

    Built with `yaml.dump` rather than typed out, because that is how the writers
    emit these files — and because the dump is what makes the case subtle. A
    hand-written fixture over-states it: `yaml.dump` will not put `---` at column
    0 inside a value, and nor will YAML accept one (the scanner rejects an
    unindented continuation of a quoted scalar, and a column-0 `---` is a document
    separator, which `safe_load` refuses as "expected a single document"). So the
    closing fence really is the only column-0 fence a *valid* block can have,
    which is what makes the line-anchored rule safe rather than merely different.
    What the dump does emit — and what the board is full of — is:

    - entry 2, the retired rule quoted verbatim: a `---` substring mid-line, the
      shape `split("---", 2)` cut on. 31 items read as `_yaml_broken` this way on
      2026-09-20, none of them broken.
    - entry 3, a value holding a real line break, which the dump renders as a
      multi-line scalar with an *indented* `  ---` line: a `---\n` substring that
      is not at a line start, the shape this module's `split("---\n", 2)` cut on.
      The loop's own writers are what put this shape on the board.

    The whole block is valid YAML — `yaml.safe_load` reads every key, which
    `test_the_fixture_breaks_both_old_splits_and_neither_yaml` pins.
    """
    fm = dict(FENCE_KEYS)
    fm["activity_log"] = [
        "**2026-09-13T00:00:00** — created",
        "**2026-09-14T00:00:00** — autotriage: **confirmed**. The parser at "
        "`agent_mcp/backlog.py:30` does " + QUOTED_RULE + ", which cuts anywhere",
        "**2026-09-15T00:00:00** — a rule line of its own: before\n---\nafter",
    ]
    block = yaml.dump(fm, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)
    return f"---\n{block}---\n\n{FENCE_BODY}"


def _old_unanchored(text: str) -> str:
    """The front-matter text as `agent_mcp` and the API cut it: `split("---", 2)`."""
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise AssertionError("fixture has no closing fence to cut at")
    return parts[1]


def _old_line_prefixed(text: str) -> str:
    """The same cut as this module made it: `split("---\\n", 2)`."""
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        raise AssertionError("fixture has no closing fence to cut at")
    return parts[1]


def _unparsable(block: str) -> bool:
    try:
        yaml.safe_load(block)
    except yaml.YAMLError:
        return True
    return False


def _with_an_unterminated_quote(text: str) -> str:
    """Corrupt `text` the way #918/#866 document it: an unterminated quote in the block.

    The line goes immediately before the closing fence — a line that is *exactly*
    `---`, not merely one that strips to it. The distinction matters only for this
    fixture, which also carries an indented `  ---` inside a multi-line scalar:
    corrupting there would put the bad line inside a quoted scalar, where it is
    just text and nothing breaks. Inserted before the fence, the quote runs off the
    end of the block and the parse dies the way the real corruption does.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines[1:], 1):
        if line == "---":                       # the closing fence, not an indented one
            lines.insert(i, 'note: "unterminated quote')
            break
    else:
        raise AssertionError("fixture has no closing fence to corrupt against")
    return "\n".join(lines)


@contextlib.contextmanager
def _scratch_board(text: str):
    """A throwaway backlog holding one item, with all three readers aimed at it.

    Each module keeps the board path in its own global, so a test that crosses the
    reader/writer boundary has to redirect every one of them at once — otherwise
    the API reader would read the real board while the writer wrote here, and the
    agreement this section is about would not be the agreement being tested.
    """
    from app.routers import backlog as BR

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "460-the-item.md").write_text(text, encoding="utf-8")
        saved = (MCP.BACKLOG_DIR, BR._BACKLOG_DIR, B.BACKLOG_DIR)
        MCP.BACKLOG_DIR = root
        BR._BACKLOG_DIR = root
        B.BACKLOG_DIR = root
        try:
            yield root
        finally:
            MCP.BACKLOG_DIR, BR._BACKLOG_DIR, B.BACKLOG_DIR = saved


def test_the_fixture_breaks_both_old_splits_and_neither_yaml():
    """Positive control for this section, and the reason the section exists at all.

    Without it the tests below could be pinning a shape that cannot occur — which
    is exactly how the split shipped unpinned once: this file already covered
    `parse_frontmatter_text`, the layer *below* the split, so a green suite said
    nothing about where the block was cut, and #1146 was filed, confirmed, and
    then sent a round that wrote nothing. Both retired rules are read here, on the
    fixture, separately — so an edit that quietly makes either case vacuous fails
    loudly instead of passing.
    """
    text = fence_item_text()
    assert _unparsable(_old_unanchored(text)), "the unanchored split parsed cleanly"
    assert _unparsable(_old_line_prefixed(text)), "the line-prefixed split parsed cleanly"
    # And the block the anchored rule returns is valid and whole.
    assert yaml.safe_load(FM.split_frontmatter(text)[0])["status"] == "up_next"


def test_the_shared_rule_ends_the_block_at_a_line_that_is_exactly_the_fence():
    """`app.frontmatter.split_frontmatter`, the boundary all three readers share.

    A `---` inside a value cannot close the block, in either shape a writer emits
    it; a `---` with trailing blanks still can, because hand-edited items on the
    board carry them and such a line is still a fence; a file whose fence never
    closes has no front matter rather than an unterminated one. The split is
    byte-exact, which is what lets a writer rewrite a block and leave the body
    alone.

    The opener is deliberately as loose as the `content.startswith("---")` it
    replaced, down to a first line of `----`. Such a file has no valid front
    matter either way, and an empty parse is not a harmless answer: the board's
    readers skip what they cannot parse, so a stricter opener would take a
    degraded-but-visible item off the listing and off the board's count — the
    silent-task-drop failure mode this whole file exists to keep dead. Only the
    closing rule was the defect, so only the closing rule got strict.
    """
    assert FM.split_frontmatter("---\na: 1\n---\nbody\n") == ("a: 1\n", "body\n")
    assert FM.split_frontmatter("---\na: 1\nnope ---\n---\nbody\n") == (
        "a: 1\nnope ---\n", "body\n")
    assert FM.split_frontmatter("---\na: 1\n--- \nbody\n") == ("a: 1\n", "body\n")
    assert FM.split_frontmatter("---\na: 1\ndivider: ---\n---\nbody\n") == (
        "a: 1\ndivider: ---\n", "body\n")
    # Loose opener: the whole first line is consumed, whatever it says.
    assert FM.split_frontmatter("----\na: 1\n---\nbody\n") == ("a: 1\n", "body\n")
    assert FM.split_frontmatter("---\na: 1\n") is None
    assert FM.split_frontmatter("# no front matter at all\n") is None
    for text in (fence_item_text(), PLAIN_ITEM):
        block, body = FM.split_frontmatter(text)
        assert f"---\n{block}---\n{body}" == text, "the split is not byte-exact"


def test_the_mcp_reader_ends_the_block_at_the_fence_line():
    """Clause 1. An item quoting the retired rule parses with every key intact.

    Under the unanchored split this file came back `_yaml_broken` — the flag that
    makes `save_task` and `_handle_write` refuse it — for YAML that
    `yaml.safe_load` reads without complaint.
    """
    fm, body = MCP.parse_frontmatter(fence_item_text())
    assert not fm.get("_yaml_broken"), "a fence quoted inside the block is not broken YAML"
    for key, want in FENCE_KEYS.items():
        assert fm.get(key) == want, f"{key} is in the file but not in the parse"
    log = fm.get("activity_log") or []
    assert len(log) == 3, "all three activity entries are inside the block"
    assert QUOTED_RULE in log[1], "the quoting itself did not survive intact"
    assert body == FENCE_BODY.strip()


def test_the_mcp_writer_updates_the_item_the_unanchored_split_locked():
    """Clause 1, the writable half: `backlog_write_task` accepts the item.

    The refusal was the user-visible cost — "malformed YAML frontmatter … fix the
    file by hand", answered for valid YAML on 31 items at the last count. Through
    `_handle_write` rather than `save_task`, because the tool is the boundary the
    clause names and the refusal is its own check. The writer re-dumps the block,
    so it must also prove it keeps what it did not touch: the quoting entries and
    the body.
    """
    with _scratch_board(fence_item_text()) as board:
        out = json.loads(MCP._handle_write({
            "task_id": 460, "priority": "high",
            "activity": "updated through the MCP writer",
        }))
        assert out.get("success") is True, out
        fm, body = MCP.parse_frontmatter(
            (board / "460-the-item.md").read_text(encoding="utf-8"))
        assert not fm.get("_yaml_broken")
        assert fm["priority"] == "high", "the update did not land"
        for key, want in FENCE_KEYS.items():
            if key == "priority":
                continue
            assert fm.get(key) == want, f"{key} dropped by the re-dump"
        log = fm.get("activity_log") or []
        assert len(log) == 4, "the new entry was substituted for the old ones"
        assert QUOTED_RULE in log[1], "the quoting was destroyed by the rewrite"
        assert body == FENCE_BODY.strip(), "the body moved"


def test_the_api_reader_agrees_with_the_mcp_reader():
    """Clause 2. Whatever parses under the writer parses under the API reader too.

    `app/routers/backlog.py::_backlog_parse_fm` carried the same unanchored split,
    so it flagged the same items `_yaml_broken` and `_reject_broken_fm` answered
    HTTP 409 to every board edit of them. The item as filed asserted that the
    dashboard already bounded the block on a line-anchored `^---$`; it never had —
    a `git log -S` on the split leaves it unanchored since `533f69f`, the module
    refactor — so the agreement between the two readers is pinned here rather than
    assumed. `_reject_broken_fm` is called unguarded at the end: it raises HTTP 409
    on a `_yaml_broken` record, so merely reaching the line is the assertion.
    """
    from app.routers import backlog as BR

    with _scratch_board(fence_item_text()) as board:
        path = board / "460-the-item.md"
        fm, body = BR._backlog_parse_fm(path)
        assert not fm.get("_yaml_broken"), "the API reader invented a broken item"
        for key, want in FENCE_KEYS.items():
            assert fm.get(key) == want
        assert len(fm.get("activity_log") or []) == 3
        assert body == FENCE_BODY.strip()
        BR._reject_broken_fm(fm, path)


def test_yaml_that_really_is_broken_is_still_broken_to_both_readers():
    """The control this fix could fake: never failing is not the same as being fixed.

    An unterminated quote is the corruption #918 and #866 document, and it is
    unparseable for real. Both readers must still mark it, and the API must still
    refuse to rewrite it, or `_yaml_broken` has become a flag nobody can trust and
    the 409 has stopped meaning anything.
    """
    from app.routers import backlog as BR
    from fastapi import HTTPException

    broken = _with_an_unterminated_quote(fence_item_text())
    assert _unparsable(FM.split_frontmatter(broken)[0]), \
        "fixture no longer reproduces an unterminated scalar"
    assert MCP.parse_frontmatter(broken)[0].get("_yaml_broken"), \
        "the MCP reader stopped reporting it"
    with _scratch_board(broken) as board:
        path = board / "460-the-item.md"
        fm, _ = BR._backlog_parse_fm(path)
        assert fm.get("_yaml_broken"), "the API reader stopped reporting it"
        with pytest.raises(HTTPException) as exc:
            BR._reject_broken_fm(fm, path)
        assert exc.value.status_code == 409


def test_the_automod_split_uses_the_same_line_anchored_rule():
    """Clause 5, the reader half: `B._split_frontmatter` cuts where the others cut.

    Here the exposure was latent rather than live — #1146's triage measured 0
    mis-parses across the board — because this split needed `---\n` at a line start
    and `yaml.dump` renders an embedded fence indented, never at column 0. It was
    still the odd one out among the three readers, and unlike the other two it
    feeds a writer: a mis-cut relocates the keys after the cut into the body, and
    `update_frontmatter`'s guard refuses only an *empty* parse, which a partial one
    slips through with its keys missing.
    """
    fm, body = B._split_frontmatter(fence_item_text())
    for key, want in FENCE_KEYS.items():
        assert fm.get(key) == want, f"{key} fell into the body at the cut"
    assert len(fm.get("activity_log") or []) == 3
    assert QUOTED_RULE in fm["activity_log"][1]
    assert body == "\n" + FENCE_BODY, "the body lost its leading blank line"
    assert B._split_frontmatter(PLAIN_ITEM)[0] == {"board": "lloyd", "status": "draft"}


def test_update_frontmatter_round_trips_an_item_whose_block_holds_a_fence():
    """Clause 5, the writer half: the round trip loses nothing.

    `update_frontmatter` re-dumps the block, so the guarantee it can offer is over
    the *body* — which is exactly what the old split put at risk, since a mis-cut
    moved the keys after the fence into the body and this writer then buried them
    under the re-dump. Every front-matter key and the entire body must still be
    there afterwards.
    """
    with _scratch_board(fence_item_text()) as board:
        path = board / "460-the-item.md"
        before_fm, before_body = B._split_frontmatter(
            path.read_text(encoding="utf-8"))

        assert B.update_frontmatter(path, {"priority": "high"},
                                    activity="round-tripped by the loop") is True

        after_fm, after_body = B._split_frontmatter(
            path.read_text(encoding="utf-8"))
        assert after_body == before_body, "the body moved"
        assert after_body.strip() == FENCE_BODY.strip()
        assert after_fm["priority"] == "high", "the update did not land"
        for key, want in before_fm.items():
            if key in ("priority", "activity_log", "updated"):
                continue
            assert after_fm.get(key) == want, f"{key} lost in the round trip"
        assert len(after_fm["activity_log"]) == len(before_fm["activity_log"]) + 1
        assert QUOTED_RULE in after_fm["activity_log"][1], \
            "the quoting was destroyed by the re-dump"


def test_a_file_with_no_front_matter_at_all_has_no_block_to_find():
    """The control the other tests in this section cannot give each other.

    Every fence test here asserts a block *was* found, at the right line. Asserted
    alone, a rule that returned the whole file as the block would pass some of them
    and fail others in ways that read as fixture trouble. So pin the other end:
    front matter has to start at byte 0 — the opening fence is `^---` at the top of
    the file, not the first fence anywhere — and a file with none gets the empty
    parse every guarded writer refuses to write back, which is the shape #1020
    filed, not a block.
    """
    body_only = "# No front matter\n\nBody text mentioning --- a horizontal rule.\n"
    assert FM.split_frontmatter(body_only) is None
    assert B._split_frontmatter(body_only) == ({}, body_only)
    # Neither MCP reader has a sentinel to return here — `parse_frontmatter` gives
    # `({}, content)`, and `_handle_write`'s own `_split_fm` yields an empty dict,
    # which is the empty parse every guarded writer in this file refuses to write
    # back. An honest "no block" is what makes that refusal reach the right files;
    # a rule that called the whole file a block would instead hand those writers a
    # garbage dict and let them rewrite the item from it.
    assert MCP.parse_frontmatter(body_only) == ({}, body_only)

    # The opener stays as loose as the `startswith("---")` it replaced, on purpose
    # (see `_OPENING_FENCE`): tightening it would turn "parsed, however oddly" into
    # a None and drop an item off the board listing altogether. So a first line of
    # `--- not a fence` still yields a block — with the *documented* content, which
    # is the YAML between it and the first line that is exactly `---`.
    odd = FM.split_frontmatter("--- not a fence\nstatus: draft\n---\nbody\n")
    assert odd is not None and odd[0] == "status: draft\n" and odd[1] == "body\n"
    assert FM.split_frontmatter(PLAIN_ITEM) is not None, "positive control: the real thing is found"


# ===========================================================================
# The twenty real items the substring split locked (#1221 clause 2), and the
# two writers' refusal to round-trip a file that is broken for real (clause 4).
# ===========================================================================
#
# Everything above is built from synthetic items. The section below cannot be:
# `tests/fixtures/frontmatter_locked_items/` holds the front matter of the twenty
# board items that stamped `_yaml_broken` under the retired split, captured from
# disk exactly as they stood on 2026-09-17 — the set #1146 counted as 15 on
# 2026-09-14, re-counted as 20 three days later, and grown to 21 that evening when
# #1221 locked *itself* by recording a finding about the defect into its own
# activity log. Only the front matter is captured, because the front matter is the
# whole defect: the body was never the ambiguous half, and the files on the board
# have moved since (several of the twenty have been written to since the fix
# landed), so the bodies are supplied here as plain prose and the block under test
# is byte-exactly the one that locked each item.
#
# Re-measured on 2026-09-21 against the live board: 1,263 files, 0 stamping
# `_yaml_broken` through the anchored rule. That is the acceptance check, and these
# fixtures are its frozen half — the live count can only show that the board is
# clean *today*, whereas these twenty show that the specific texts which used to
# break it now do not, on a reader that cannot be edited to fit.
LOCKED_ITEMS = Path(__file__).resolve().parent / "fixtures" / "frontmatter_locked_items"

#: The twenty ids #1221's sweep named, spelled out rather than derived from the
#: directory: a deleted fixture must fail here, naming the id, instead of quietly
#: shrinking the parametrised set to whatever files happen to be left.
LOCKED_IDS = (
    460, 478, 519, 520, 525, 575, 601, 642, 787, 866,
    918, 933, 971, 981, 992, 1068, 1069, 1146, 1167, 1190,
)


def locked_item_doc(id_: int) -> str:
    """One captured item laid down as a backlog file: fences around its real block.

    The block is the fixture verbatim, so a fence-inside-the-block case is exactly
    as hard as the file that was on the board.
    """
    block = (LOCKED_ITEMS / f"{id_}-frontmatter.txt").read_text(encoding="utf-8")
    return f"---\n{block}---\n\n# Captured item {id_}\n\nBody prose of item {id_}.\n"


def test_the_captures_are_the_twenty_items_the_sweep_locked():
    """The fixture set itself, before any reader gets to interpret it.

    Two things are pinned so the parametrised tests below cannot pass by
    attrition. The twenty named ids are all present — and each capture really
    holds a `---` inside its block, which is the property that made the item
    unwritable. A fixture that lost that substring would stop testing anything the
    anchored rule handles better than the retired one did, and no test that reads
    it would notice.
    """
    present = {int(p.name.split("-")[0]) for p in LOCKED_ITEMS.glob("*.txt")}
    assert set(LOCKED_IDS) <= present, f"missing captures: {sorted(set(LOCKED_IDS) - present)}"
    for id_ in LOCKED_IDS:
        block = (LOCKED_ITEMS / f"{id_}-frontmatter.txt").read_text(encoding="utf-8")
        assert "---" in block, f"{id_}: capture holds no fence-looking text"
        assert "activity_log" in block, f"{id_}: capture lost its activity log"
        # A capture truncated mid-scalar would parse for the wrong reason — nothing
        # to do with where the block ends.
        assert yaml.safe_load(FM.split_frontmatter(locked_item_doc(id_))[0]), \
            f"{id_}: captured block is not valid YAML on its own"


@pytest.mark.parametrize("id_", LOCKED_IDS)
def test_the_retired_split_still_locks_every_captured_item(id_):
    """The control that makes the two reader tests below worth reading.

    Each capture is fed to the retired rule as well as the live one, and must still
    fail there — the block the substring split returns is not valid YAML. Without
    this, "the anchored rule parses all twenty" could be satisfied by swapping the
    captures for ordinary items, which is the same vacuous green that let the split
    ship unpinned the first time.
    """
    assert _unparsable(_old_unanchored(locked_item_doc(id_))), (
        f"{id_}: the substring split parses this capture cleanly, so it no "
        "longer reproduces the lock it was captured for")


@pytest.mark.parametrize("id_", LOCKED_IDS)
def test_every_item_the_substring_split_locked_parses_in_the_mcp_reader(id_):
    """Clause 2, the reader the tool path uses.

    `parse_frontmatter` is what `backlog_get_task` and `backlog_write_task` read
    through, and `_yaml_broken` from here is what made `save_task` answer
    "malformed YAML frontmatter … fix the file by hand". `status` and
    `activity_log` are the two keys the clause names because they are the two the
    regex fallback could not always reach — the flag absent and both keys present
    is the whole of the acceptance for these files.
    """
    fm, body = MCP.parse_frontmatter(locked_item_doc(id_))
    assert not fm.get("_yaml_broken"), f"{id_}: a valid block still reported broken"
    assert fm.get("status"), f"{id_}: no status recovered"
    assert isinstance(fm.get("activity_log"), list) and fm["activity_log"], \
        f"{id_}: no activity log recovered"
    assert fm.get("board") == "lloyd", f"{id_}: the item fell off its board"
    assert body.startswith(f"# Captured item {id_}"), f"{id_}: the body was cut"


@pytest.mark.parametrize("id_", LOCKED_IDS)
def test_every_item_the_substring_split_locked_parses_in_the_api_reader(id_, tmp_path):
    """Clause 2, the reader the Mission Control board uses.

    The same twenty files through `app/routers/backlog.py`'s own reader, which is
    the one whose `_reject_broken_fm` answered HTTP 409 to every board edit of
    them. That guard is called unguarded at the end of the test: it raises on a
    `_yaml_broken` record, so simply reaching the line is the assertion, and a
    regression here fails as a 409 rather than as a passing check.
    """
    from app.routers import backlog as BR

    path = tmp_path / f"{id_}-captured-item.md"
    path.write_text(locked_item_doc(id_), encoding="utf-8")
    fm, body = BR._backlog_parse_fm(path)
    assert not fm.get("_yaml_broken"), f"{id_}: the API reader invented a broken item"
    assert fm.get("status"), f"{id_}: no status recovered"
    assert isinstance(fm.get("activity_log"), list) and fm["activity_log"], \
        f"{id_}: no activity log recovered"
    assert body.startswith(f"# Captured item {id_}"), f"{id_}: the body was cut"
    BR._reject_broken_fm(fm, path)


# --- clause 4, the writers: the fix must not have disarmed the guard ---------------

def test_the_mcp_writer_refuses_to_round_trip_a_genuinely_broken_record():
    """Clause 4, half one: `backlog_write_task` still refuses what is really broken.

    The refusal is the point of `_yaml_broken`, and an over-eager fix that made
    every block parse would clear it for files whose YAML is genuinely
    unparseable — where the regex fallback recovers a handful of fields and a
    re-dump from that dict deletes the rest. `save_task` is the writer, so the test
    runs the tool handler and then reads the file back: the answer must be the
    refusal, and the bytes on disk must be untouched.
    """
    broken = _with_an_unterminated_quote(fence_item_text())
    with _scratch_board(broken) as board:
        path = board / "460-the-item.md"
        before = path.read_text(encoding="utf-8")

        out = json.loads(MCP._handle_write({
            "task_id": 460, "priority": "high", "activity": "must not be written",
        }))

        assert out.get("success") is False, f"a broken record was written: {out}"
        assert "malformed YAML frontmatter" in out["error"], out
        assert path.read_text(encoding="utf-8") == before, (
            "the refusal still rewrote the file")

        # `save_task` is the writer underneath that handler, and it guards on the
        # flag itself rather than trusting its caller — the guard that matters for
        # every caller that is not `_handle_write`. Pinned directly for the same
        # reason: the fallback dict is short, and dumping it back is what would
        # delete `activity_log` and the timestamps from a real item.
        record = MCP.load_task(460)
        assert record.get("_yaml_broken"), "the reader stopped degrading this file"
        assert MCP.save_task(record) is False, "save_task round-tripped a broken record"
        assert path.read_text(encoding="utf-8") == before, (
            "save_task refused and wrote anyway")


async def test_the_board_route_refuses_to_round_trip_a_genuinely_broken_record():
    """Clause 4, half two: the HTTP route still answers 409 for the same file.

    `app/routers/backlog.py` is the second writer of the pair, and its guard is the
    one that had stopped meaning anything — on 2026-09-17 it fired for twenty-one
    items whose YAML was fine. So the assertion is deliberately split: a *quoted
    fence* passes (pinned by every test above) while an unterminated quote is
    refused, and both directions are checked against the route, not only against
    `_reject_broken_fm` standing alone.
    """
    from fastapi import HTTPException

    from app.routers import backlog as BR

    class _Req:
        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

    broken = _with_an_unterminated_quote(fence_item_text())
    with _scratch_board(broken) as board:
        path = board / "460-the-item.md"
        before = path.read_text(encoding="utf-8")

        with pytest.raises(HTTPException) as exc:
            await BR.backlog_task_update(_Req({"id": 460, "priority": "high"}))
        assert exc.value.status_code == 409
        assert "malformed YAML frontmatter" in exc.value.detail
        assert path.read_text(encoding="utf-8") == before, (
            "the 409 still rewrote the file")


def test_only_a_line_that_is_nothing_but_the_fence_closes_the_block():
    """The other half of the closing rule, pinned against the fixtures' own shape.

    "Line-anchored" is two claims, and the test above only pins one of them: the
    fence has to start the line (so `divider: ---` stays inside the block), and
    nothing but blanks may follow it. The second claim is what a body's
    `--- Some Heading` or a markdown horizontal rule with text after it would
    otherwise trip: a closing rule that tolerated trailing text would end the block
    at that line, take whatever followed for the body, and hand the writer a
    truncated front matter to re-dump — the same class of damage as the substring
    split, one line further down the file. And a block whose fence never closes at
    all has to answer `None`, which every guarded reader in this file already
    refuses to write back, rather than reaching forward for the next line that
    merely starts like a fence.
    """
    # A line that only *starts* like the fence is content, not the boundary.
    assert FM.split_frontmatter("---\nstatus: draft\n--- not a fence\n---\nbody\n") == (
        "status: draft\n--- not a fence\n", "body\n")
    # With no exact fence anywhere after the opener there is no block to find —
    # even though a line beginning `---` is present, and would be a plausible-
    # looking boundary to a rule that let text follow it.
    assert FM.split_frontmatter("---\nstatus: draft\n--- still not a fence\nbody\n") is None
    # Trailing blanks stay legal: a hand-edited item on the board can carry them,
    # and such a line is still nothing but the fence.
    assert FM.split_frontmatter("---\nstatus: draft\n---   \nbody\n") == (
        "status: draft\n", "body\n")
