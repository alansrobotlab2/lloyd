"""The backlog list route ships a snippet, the detail route ships the body, and
`?q=` still matches text in the middle of a long item.

Item #1199. Measured on the live box 2026-09-16: `GET /api/backlog/tasks`
returned 1,137 rows and 9,202,414 bytes, of which 8,409,812 (91.4 %) were the
`description` field — each task's entire markdown body. `TaskCard` renders that
field through `line-clamp-2`, two visible lines, and the modal is the only
thing that needs the whole thing. So the list was paying 9 MB, 1,122 full-text
layout nodes, and a per-keystroke `toLowerCase()` over 8.3 MB to show two
lines of prose.

The fix has three parts, and these tests pin all three:

* the list row carries `description_snippet` (300 characters) and no full body;
* `GET /api/backlog/task/{id}` returns that item's complete body, so the modal
  has somewhere to read the whole thing from;
* the list accepts `?q=` and searches the **cached full body** server-side.
  That third one is what keeps search honest: `filteredTasks` in
  `BacklogPage.tsx` greps `t.description`, and once the row carries only 300
  characters, a client-side search silently stops matching mid-body. Server-side
  `?q=` over the cached bodies is the option the item chose (option a), and the
  test below searches for a needle planted ~5 KB into a 10 KB+ body — a place
  the snippet provably cannot reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.routers import backlog as BR


def write_item(
    d: Path, item_id: int, *, board: str = "lloyd", status: str = "draft",
    name: str | None = None, body: str = "Do the thing.",
) -> Path:
    """One conformant backlog file, the shape `backlog_task_create` writes."""
    name = name or f"Item {item_id}"
    p = d / f"{item_id}-{name.lower().replace(' ', '-')[:30]}.md"
    fm = {
        "type": "backlog", "segment": "backlog", "status": status,
        "priority": "medium", "board": board, "blocked": False,
        "assigned": False, "position": item_id * 1000,
        "created": "2026-09-01T00:00:00", "updated": "2026-09-01T00:00:00",
    }
    p.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def backlog_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_FM_CACHE", {})
    return d


def rows_of(resp) -> list[dict]:
    return json.loads(bytes(resp.body))


# ── clause 1: snippet on the list, whole body on the detail route ────────────

def test_list_row_carries_a_snippet_and_not_the_body(backlog_dir):
    body = "HANDOFF. " + ("filler text that the board never renders. " * 400)
    write_item(backlog_dir, 7, body=body)

    rows = rows_of(BR.backlog_tasks())
    assert len(rows) == 1, "fixture expected exactly one item"
    row = rows[0]
    snippet = row["description_snippet"]

    # Clause: "a description_snippet of at least 200 characters per row".
    assert len(snippet) >= 200, f"snippet was only {len(snippet)} chars"
    assert len(snippet) <= BR.DESC_SNIPPET_CHARS + 1, "snippet longer than the cap"
    assert body.startswith(snippet.rstrip("…")), "snippet is not the head of the body"
    # Clause: "and no full body" — 7.2 KB of body must not be in the row.
    assert len(body) > 10_000
    assert len(snippet) <= BR.DESC_SNIPPET_CHARS + 1
    # And the row carries no `description` key at all: that name belongs to the
    # whole body, which lives on `/api/backlog/task/{id}`. An alias here is what
    # lets a caller post a snippet where a body was expected — `task-update`
    # replaces the entire body with whatever `description` it is sent.
    assert "description" not in row, (
        "the list row grew a `description` again; the modal's guard in "
        "BacklogPage.tsx assumes the row cannot be an editing source"
    )


def test_unfiltered_payload_is_bytes_of_rows_not_bytes_of_bodies(backlog_dir):
    """9.2 MB -> < 1 MB. Asserted as a ratio so it fails on the old behaviour."""
    n, per_body = 200, 5_000
    for i in range(1, n + 1):
        write_item(backlog_dir, i, body="x" * per_body)
    total_body = n * per_body  # 1,000,000 B of bodies on disk

    payload = bytes(BR.backlog_tasks().body)
    assert len(payload) < 1_000_000, f"payload {len(payload)} B is still MB-scale"
    assert len(payload) < total_body / 2, (
        f"payload {len(payload)} B is not smaller than the {total_body} B of bodies"
    )
    assert len(rows_of(BR.backlog_tasks())) == n, "payload shrank by dropping rows"


def test_short_body_is_returned_whole_as_the_snippet(backlog_dir):
    write_item(backlog_dir, 3, body="Two lines only.")
    row = rows_of(BR.backlog_tasks())[0]
    assert row["description_snippet"] == "Two lines only."


def test_detail_route_returns_the_complete_body_unchanged(backlog_dir):
    body = "## Handoff\n\n" + ("middle of the body " * 900) + "\n\n## Tail\n"
    write_item(backlog_dir, 42, name="Long item", body=body)

    detail = json.loads(bytes(BR.backlog_task_detail(42).body))
    assert detail["id"] == 42
    assert detail["name"] == "Long item"
    # "that item's complete body unchanged": byte-for-byte the text after the
    # heading, which is exactly the string `task-update` writes back.
    assert detail["description"] == body.strip()
    assert len(detail["description"]) > 10_000


def test_detail_route_404s_for_an_unknown_id(backlog_dir):
    write_item(backlog_dir, 1)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        BR.backlog_task_detail(999)
    assert ei.value.status_code == 404


# ── clause 2: search still matches the middle of a long body ────────────────

MID = "UNAMBIGUOUSMIDBODYNEEDLE"


def test_query_matches_text_past_where_the_snippet_ends(backlog_dir):
    write_item(backlog_dir, 1, body="# nope\n" + ("pad " * 1_500) + MID + (" pad " * 1_500))
    write_item(backlog_dir, 2, body="Nothing to do with the needle here.")
    long_body = next(backlog_dir.glob("1-*.md")).read_text()
    assert len(long_body) > 10_000, "fixture must exceed the snippet cap by miles"
    assert long_body.index(MID) > BR.DESC_SNIPPET_CHARS, (
        "the needle sits inside the snippet, so this test would prove nothing"
    )

    hits = rows_of(BR.backlog_tasks(q=MID))
    assert [r["id"] for r in hits] == [1]
    # The row that matched still ships only the snippet, not the body it matched in.
    assert len(hits[0]["description_snippet"]) <= BR.DESC_SNIPPET_CHARS + 1


def test_query_is_case_insensitive_and_combines_with_the_board_filter(backlog_dir):
    write_item(backlog_dir, 1, board="lloyd", body="alpha " + "pad " * 400 + "ZebraCross")
    write_item(backlog_dir, 2, board="alfie", body="beta " + "pad " * 400 + "ZebraCross")

    board_map = BR._backlog_board_map()
    hits = rows_of(BR.backlog_tasks(q="zebracross", board_id=str(board_map["alfie"])))
    assert [r["id"] for r in hits] == [2], "board filter must still apply under ?q="


def test_query_matches_names_and_tags_too(backlog_dir):
    write_item(backlog_dir, 1, name="Deploy the kestrel relay")
    write_item(backlog_dir, 2, name="Unrelated")
    assert [r["id"] for r in rows_of(BR.backlog_tasks(q="kestrel"))] == [1]


def test_query_with_no_match_returns_an_empty_list(backlog_dir):
    write_item(backlog_dir, 1, body="plain body")
    assert rows_of(BR.backlog_tasks(q="nothinglikesthis")) == []


# ── The live board ───────────────────────────────────────────────────────────
# The clause this pins is a byte budget on the *real* corpus: 9,202,414 B before
# #1199, and the acceptance line is under 1 MB. A tmp_path fixture can prove the
# shape of a row; only the board itself can prove the total, because the total is
# what 1,137 rows add up to. Marked `live_vault`, so the gate's `-m "not
# live_vault"` skips it — it is run and cited separately (see the round report),
# and `scripts/maintenance/backlog_route_probe.py` prints the same number.


@pytest.mark.live_vault
def test_the_live_board_ships_under_a_megabyte_of_rows():
    root = Path.home() / "obsidian" / "backlog"
    if not root.exists():
        pytest.skip("no vault")
    body_bytes = BR.backlog_tasks().body
    rows = json.loads(bytes(body_bytes))
    assert len(rows) > 1000, f"only {len(rows)} rows: not the corpus this measures"
    assert len(body_bytes) < 1_000_000, (
        f"{len(body_bytes):,} B for {len(rows)} rows: the list route is carrying "
        "bodies again (was 9,202,414 B before #1199)"
    )
    assert all("description" not in r for r in rows), (
        "a live row still carries `description`, which is the whole body's name"
    )
    assert all(len(r["description_snippet"]) <= BR.DESC_SNIPPET_CHARS + 1 for r in rows)


@pytest.mark.live_vault
def test_the_live_detail_route_returns_a_body_the_list_refused_to_send():
    """One real item, both routes: the row's snippet and the detail's full body."""
    root = Path.home() / "obsidian" / "backlog"
    if not root.exists():
        pytest.skip("no vault")
    rows = json.loads(bytes(BR.backlog_tasks().body))
    row = max(rows, key=lambda r: len(r["description_snippet"]))
    detail = json.loads(bytes(BR.backlog_task_detail(row["id"]).body))
    assert len(detail["description"]) > len(row["description_snippet"]), (
        f"item {row['id']}: detail is no bigger than the snippet"
    )
    # The row's snippet is the body's head plus the marker `_snippet` adds when it
    # had to cut, so the card preview is literally the first thing in the body.
    assert row["description_snippet"] == (
        detail["description"][:BR.DESC_SNIPPET_CHARS].rstrip() + "…"
    )
