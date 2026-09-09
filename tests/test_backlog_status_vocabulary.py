"""Five lists name a backlog task's `status`, and none had a case for a
value outside them.

The four Python readers and the React board all agree on the four words, so
drift between them was never the bug. The bug is the *gap*: an item whose
status is none of the four is counted **open** by
`dashboard._BACKLOG_CLOSED` and is invisible to `backlog.OPEN_STATUSES`, so a
human sees work in progress and every machine that would move it — including
`reconcile_statuses`, which forms opinions only about items already in
`open_items` — cannot reach it. #287 (`review`) and #304 (`closed`) sat in
that gap from April 2026 until 2026-09-09.

These pin the shared definition, the rescue, and the direction of the
mapping: a word wrongly called terminal buries a live item in `done` forever,
while a word wrongly called live costs one triage run that closes it.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.backlog_status import (  # noqa: E402
    CLOSED_ALIASES,
    CLOSED_STATUSES,
    PIPELINE_STATUSES,
    canonical_status,
    is_off_vocabulary,
)

WEB = ROOT / "web" / "src"


def _board_page_statuses() -> set[str]:
    """The statuses the React board renders — it cannot import Python."""
    src = (WEB / "components" / "pages" / "BacklogPage.tsx").read_text(encoding="utf-8")
    m = re.search(r"const STATUSES\s*=\s*\[(.*?)\]\s*as const", src, re.DOTALL)
    assert m, "no `const STATUSES = [...] as const` in BacklogPage.tsx"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


# ── the five lists ─────────────────────────────────────────────────────────

def test_every_python_reader_shares_one_vocabulary():
    from agent_mcp import backlog as mcp_backlog
    from app.routers import backlog as board_router
    from scripts.automod import backlog as loop

    assert set(mcp_backlog.VALID_STATUSES) == set(PIPELINE_STATUSES)
    assert set(board_router._VALID_STATUSES) == set(PIPELINE_STATUSES)
    assert set(loop.PIPELINE_STATUSES) == set(PIPELINE_STATUSES)
    assert set(loop.OPEN_STATUSES) == set(PIPELINE_STATUSES) - {"done"}


def test_react_board_renders_every_status_the_writers_accept():
    missing = set(PIPELINE_STATUSES) - _board_page_statuses()
    assert not missing, (
        f"statuses the API accepts that the board cannot render: {sorted(missing)}. "
        "An item written with one lands in no column."
    )


def test_dashboard_closed_set_is_the_shared_one():
    from app.routers import dashboard

    assert dashboard._BACKLOG_CLOSED is CLOSED_STATUSES
    # The legacy half is what makes the gap asymmetric: the dashboard knows
    # these words take a task off the board and the pipeline has never heard
    # of them.
    assert CLOSED_STATUSES - {"done"} == CLOSED_ALIASES
    assert "done" in CLOSED_STATUSES


# ── the mapping ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("review", "draft"),      # #287 — the case that started this
    ("closed", "done"),       # #304
    ("cancelled", "done"),
    ("wontfix", "done"),
    ("draft", "draft"),
    ("up_next", "up_next"),
    ("Done", "done"),         # case is a human writing the same word
    ("  In_Progress ", "in_progress"),
    ("blocked", "draft"),     # unknown → the recoverable direction
    ("", "draft"),
    (None, "draft"),
])
def test_canonical_status(raw, want):
    assert canonical_status(raw) == want


def test_unknown_words_go_to_draft_not_done():
    """The lopsided direction, stated as a property.

    `done` is terminal for every writer here, so a live item sent there is
    never read again. A finished item sent to `draft` costs one triage run,
    which closes it. Only words already known to be terminal may map to
    `done`.
    """
    for word in ("review", "blocked", "waiting", "needs-review", "parked"):
        assert canonical_status(word) == "draft", word


def test_is_off_vocabulary_is_case_sensitive_and_ignores_absent():
    # `Done` strands exactly as `review` does — neither reader lowercases.
    assert is_off_vocabulary("Done")
    assert is_off_vocabulary("review")
    assert is_off_vocabulary("closed")
    # Absent: `load_item` and the dashboard both already default it to draft,
    # so the halves agree and there is nothing on disk to correct.
    assert not is_off_vocabulary("")
    assert not is_off_vocabulary(None)
    for good in PIPELINE_STATUSES:
        assert not is_off_vocabulary(good)


# ── the rescue ─────────────────────────────────────────────────────────────

def _write_item(d: Path, num: int, status: str, board: str = "lloyd") -> Path:
    p = d / f"{num}-item.md"
    p.write_text(
        f"---\nstatus: {status}\npriority: medium\nboard: {board}\n"
        f"created: '2026-04-13T07:09:17'\n---\n\n# Item {num}\n\nbody\n",
        encoding="utf-8")
    return p


@pytest.fixture()
def board(tmp_path, monkeypatch):
    from scripts.automod import backlog as loop
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(loop, "BACKLOG_DIR", d)
    return d


def test_an_off_vocabulary_item_is_stranded_between_the_two_halves(board):
    """The regression, stated end to end — this is what the rescue undoes."""
    from app.routers import dashboard
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")

    # The loop cannot see it...
    assert [i.id for i in loop.open_items()] == []
    # ...but the walk that ignores status can.
    assert [(i.id, i.status) for i in loop.all_items()] == [(287, "review")]
    # ...and the dashboard counts it as open work.
    assert "review" not in dashboard._BACKLOG_CLOSED


def test_rescue_moves_review_to_draft_and_closed_to_done(board, tmp_path):
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    _write_item(board, 304, "closed")
    ledger = tmp_path / "ledger.jsonl"

    moved = loop.rescue_off_vocabulary(ledger)

    assert {m["item_id"]: m["to"] for m in moved} == {287: "draft", 304: "done"}
    assert {i.id: i.status for i in loop.all_items()} == {287: "draft", 304: "done"}
    # #287 is now reachable by the pass that could never see it.
    assert [i.id for i in loop.open_items()] == [287]
    # #304 is off the board, not back in the triage pool.
    assert 304 not in {i.id for i in loop.open_items()}


def test_rescue_records_each_move_on_the_ledger(board, tmp_path):
    import json

    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    ledger = tmp_path / "ledger.jsonl"
    loop.rescue_off_vocabulary(ledger)

    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["event"] == "status_moved"
    assert (rows[0]["item_id"], rows[0]["from"], rows[0]["to"]) == (287, "review", "draft")
    assert rows[0]["off_vocabulary"] is True


def test_rescue_writes_the_reason_into_the_activity_log(board, tmp_path):
    from scripts.automod import backlog as loop

    p = _write_item(board, 287, "review")
    loop.rescue_off_vocabulary(tmp_path / "ledger.jsonl")

    text = p.read_text(encoding="utf-8")
    assert "review" in text and "→ draft" in text


def test_rescue_is_idempotent(board, tmp_path):
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    ledger = tmp_path / "ledger.jsonl"

    assert len(loop.rescue_off_vocabulary(ledger)) == 1
    assert loop.rescue_off_vocabulary(ledger) == []
    assert loop.rescue_off_vocabulary(ledger) == []


def test_rescue_respects_the_board_filter(board, tmp_path):
    """The backlog is shared. An Alfie item with an unusual status is not
    this loop's to rewrite."""
    from scripts.automod import backlog as loop

    _write_item(board, 500, "review", board="alfie")
    ledger = tmp_path / "ledger.jsonl"

    assert loop.rescue_off_vocabulary(ledger, ("lloyd",)) == []
    assert {i.id: i.status for i in loop.all_items(None)} == {500: "review"}
    # ...but None means every board, as it does everywhere else here.
    assert len(loop.rescue_off_vocabulary(ledger, None)) == 1


def test_rescue_leaves_the_four_good_statuses_alone(board, tmp_path):
    from scripts.automod import backlog as loop

    for n, st in enumerate(PIPELINE_STATUSES, start=100):
        _write_item(board, n, st)
    before = {i.id: i.path.read_text(encoding="utf-8") for i in loop.all_items()}

    assert loop.rescue_off_vocabulary(tmp_path / "ledger.jsonl") == []
    assert {i.id: i.path.read_text(encoding="utf-8") for i in loop.all_items()} == before


def test_reconcile_runs_the_rescue_first(board, tmp_path):
    """The rescued item must be judged in the same pass, not the next one."""
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    ledger = tmp_path / "ledger.jsonl"

    moved = loop.reconcile_statuses(ledger, ("lloyd",))

    assert {m["item_id"]: m["to"] for m in moved} == {287: "draft"}
    assert [i.status for i in loop.all_items()] == ["draft"]


def test_reconcile_kill_switch_covers_the_rescue(board, tmp_path):
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    assert loop.reconcile_statuses(tmp_path / "ledger.jsonl", ("lloyd",), enabled=False) == []
    assert [i.status for i in loop.all_items()] == ["review"]
