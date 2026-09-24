"""`backlog.all_items` parses a file only when it changed (plan C4).

Housekeeping walked the ~1,400-file board about thirteen times a pass, each
walk a YAML parse per file. The per-path cache is keyed on
`(mtime_ns, size, inode)` like `state.ledger_rows`; these tests pin that a
write is seen by the next walk, an unchanged file is parsed once, and a caller
that mutates what it was handed cannot change what the next walk sees.
"""
from __future__ import annotations

import os

import pytest

from scripts.automod import backlog as B, state as S
from tests.test_backlog_unattended import write_item


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def _parses(path) -> int:
    return B._items_parses.get(str(path), 0)


def test_two_walks_with_no_writes_parse_once(isolated):
    p = write_item(isolated, 11)
    q = write_item(isolated, 12, name="Other")
    B.all_items(None)
    first = (_parses(p), _parses(q))
    for _ in range(3):
        assert {i.id for i in B.all_items(None)} == {11, 12}
    assert (_parses(p), _parses(q)) == first


def test_a_walk_after_set_status_sees_the_new_status(isolated):
    p = write_item(isolated, 21, status="draft")
    assert B.all_items(None)[0].status == "draft"
    before = _parses(p)
    assert B.set_status(21, "up_next", "test") is True
    [item] = B.all_items(None)
    assert item.status == "up_next"
    assert _parses(p) == before + 1
    assert [i.id for i in B.open_items(None)] == [21]


def test_a_same_size_rewrite_is_seen_through_mtime(isolated):
    p = write_item(isolated, 22, status="draft", name="Same")
    B.all_items(None)
    text = p.read_text()
    p.write_text(text.replace("# Same", "# Emas"))
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert B.all_items(None)[0].name == "Emas"


def test_mutating_a_returned_item_does_not_poison_the_cache(isolated):
    write_item(isolated, 31)
    item = B.all_items(None)[0]
    item.tags.append("umbrella")
    item.members.append(99)
    item.status = "done"
    again = B.all_items(None)[0]
    assert "umbrella" not in again.tags and again.members == [] and again.status == "draft"


def test_a_deleted_file_leaves_the_walk_and_the_cache(isolated):
    p = write_item(isolated, 41)
    write_item(isolated, 42, name="Keep")
    B.all_items(None)
    p.unlink()
    assert [i.id for i in B.all_items(None)] == [42]
    assert str(p) not in B._items_cache


def test_the_board_filter_is_applied_after_the_cache(isolated):
    write_item(isolated, 51, board="lloyd")
    write_item(isolated, 52, name="Alfie", board="alfie")
    assert [i.id for i in B.all_items(("lloyd",))] == [51]
    assert [i.id for i in B.all_items(("alfie",))] == [52]
    assert {i.id for i in B.all_items(None)} == {51, 52}


def test_reopen_reverted_landings_skips_when_its_inputs_have_not_moved(isolated, monkeypatch):
    """The optional watermark: with no new rollback or landing row since a
    pass that left nothing pending, the board is not walked at all; a new
    rollback row re-arms it."""
    S.append_event({"event": "promoted", "commit": "a" * 40, "parent": "b" * 40},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "rollback_succeeded", "commit": "a" * 40}, path=S.LEDGER_PATH)
    walks = []
    real = B.all_items
    monkeypatch.setattr(B, "all_items", lambda *a, **k: walks.append(1) or real(*a, **k))
    assert B.reopen_reverted_landings(S.LEDGER_PATH) == []
    assert B.reopen_reverted_landings(S.LEDGER_PATH) == []
    assert len(walks) == 1
    S.append_event({"event": "rollback_succeeded", "commit": "c" * 40}, path=S.LEDGER_PATH)
    B.reopen_reverted_landings(S.LEDGER_PATH)
    assert len(walks) == 2
