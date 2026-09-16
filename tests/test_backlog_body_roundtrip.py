"""A `task-update` POST can never shrink an item's body.

Item #1199's addendum. The cheapest version of the payload fix — cap
`description` on the list route — silently destroys data if it ships alone,
because both halves of the write path are unguarded:

* `backlog_task_update` does `body = body[:heading.end()].rstrip() + "\\n\\n" +
  data["description"]` — it **replaces the entire body** with whatever string
  arrives. No merge, no length guard.
* `TaskModal` seeds `description` from the list row and `handleSave` sends it
  unconditionally in `updates`.

So once the row carries a 300-character snippet, "open a card, change its
priority, Save" overwrites a median 4,607-byte body with those 300 characters —
on the 409 `draft` items the nightly loop reads, and on the item that asked for
the change. The median is why this is not theoretical: the routine, one-click
edit is the one that loses ~4 KB of handoff history every time.

The server is where this gets fixed, because the server is the half that can be
tested. A description shorter than the body on disk is applied only with
`force_body_replace: true`; otherwise it is ignored and the request still
succeeds, so a priority change stays a priority change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from app.routers import backlog as BR


class _Req:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


def write_item(d: Path, item_id: int, *, body: str, name: str = "Long item",
               status: str = "draft", priority: str = "medium") -> Path:
    p = d / f"{item_id}-{name.lower().replace(' ', '-')[:30]}.md"
    fm = {
        "type": "backlog", "segment": "backlog", "status": status,
        "priority": priority, "board": "lloyd", "blocked": False,
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


def body_on_disk(path: Path) -> str:
    """Everything after the `# heading`, stripped — the string the route writes."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    after = parts[2]
    heading = re.search(r"^#\s+.+$", after, re.MULTILINE)
    return after[heading.end():].strip()


def strip_frontmatter(path: Path) -> str:
    return path.read_text(encoding="utf-8").split("---", 2)[2]


# ── the hazard: a stale row saved back must not truncate the file ────────────

@pytest.mark.asyncio
async def test_a_snippet_shaped_description_cannot_replace_the_body(backlog_dir):
    body = "## Handoff\n\n" + ("measured evidence that must survive. " * 300)
    path = write_item(backlog_dir, 11, body=body)
    snippet = body[:300]  # exactly what the list route now sends

    resp = await BR.backlog_task_update(_Req({
        "id": 11, "name": "Long item", "description": snippet,
        "status": "draft", "priority": "high", "blocked": False,
    }))
    assert resp.status_code == 200
    assert body_on_disk(path) == body.strip(), (
        "saving a card whose description field is a snippet truncated the body"
    )
    # The priority the same click asked for still lands.
    fm = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
    assert fm["priority"] == "high"


@pytest.mark.asyncio
async def test_an_empty_description_is_ignored_not_truncating(backlog_dir):
    """The modal seeds from the row; an unloaded row must not erase the body."""
    body = "## Why\n\n" + ("context the next run needs. " * 200)
    path = write_item(backlog_dir, 12, body=body)

    await BR.backlog_task_update(_Req({"id": 12, "description": ""}))
    assert body_on_disk(path) == body.strip()


@pytest.mark.asyncio
async def test_force_body_replace_is_the_explicit_way_through(backlog_dir):
    body = "## Handoff\n\n" + ("long enough that a snippet fits inside it. " * 200)
    path = write_item(backlog_dir, 13, body=body)

    await BR.backlog_task_update(_Req({
        "id": 13, "description": "deliberately rewritten", "force_body_replace": True,
    }))
    assert body_on_disk(path) == "deliberately rewritten"


@pytest.mark.asyncio
async def test_a_full_body_edit_that_adds_text_is_applied(backlog_dir):
    """The guard is about *shrinking*, not about blocking real edits."""
    body = "## Handoff\n\n" + ("existing content. " * 300)
    path = write_item(backlog_dir, 14, body=body)
    edited = body.strip() + "\n\n## Follow-up\n\nNew finding from the round."

    await BR.backlog_task_update(_Req({"id": 14, "description": edited}))
    assert body_on_disk(path) == edited


@pytest.mark.asyncio
async def test_the_response_says_when_the_description_was_ignored(backlog_dir):
    """Silence here is how the hazard would have stayed invisible."""
    body = "## Handoff\n\n" + ("content. " * 400)
    write_item(backlog_dir, 15, body=body)

    ignored = await BR.backlog_task_update(_Req({"id": 15, "description": "short"}))
    assert yaml.safe_load(ignored.body.decode())["description_ignored"] is True
    applied = await BR.backlog_task_update(_Req({"id": 15, "description": "x" * 4_000}))
    assert yaml.safe_load(applied.body.decode())["description_ignored"] is False


# ── clause 3's second half: a priority-only save rewrites nothing else ───────

def _changed_lines(path: Path, before: str) -> list[tuple[str, str]]:
    """Lines that differ, as (before, after), from the whole-file text."""
    after = path.read_text(encoding="utf-8")
    b, a = before.splitlines(), after.splitlines()
    assert len(b) == len(a), f"line count moved: {len(b)} -> {len(a)}"
    return [(x, y) for x, y in zip(b, a) if x != y]


def _key_of(line: str) -> str:
    """The frontmatter key a line carries, quotes and indentation stripped."""
    return line.strip().partition(":")[0].strip("'\" ")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [4_000, 120_000], ids=["4KB", "120KB"])
async def test_priority_only_update_changes_only_priority_and_updated(
    backlog_dir, size
):
    # Ends without trailing whitespace on purpose: `_backlog_parse_fm` returns
    # `parts[2].strip()`, so a body ending in a space would lose it and the
    # clause's "byte-identical" would be about that old normalisation instead of
    # about the body. Vault bodies end with a newline after prose.
    unit = "payload that must survive a priority click."
    body = "## Handoff\n\n" + " ".join([unit] * (size // len(unit) + 1))
    path = write_item(backlog_dir, 21, body=body)
    assert len(body) >= size
    before = path.read_text(encoding="utf-8")

    await BR.backlog_task_update(_Req({"id": 21, "priority": "high"}))

    changed = _changed_lines(path, before)
    assert changed, "expected the priority and updated lines to move"
    for before_line, after_line in changed:
        assert _key_of(before_line) in {"priority", "updated"}, (
            f"a line outside priority:/updated: changed: {before_line!r}"
        )
        assert _key_of(after_line) == _key_of(before_line), (
            f"a key was renamed rather than edited: {before_line!r} -> {after_line!r}"
        )
    assert body_on_disk(path) == body, "the body moved at all"


@pytest.mark.asyncio
async def test_name_and_description_is_a_valid_full_rewrite_pair(backlog_dir):
    """Renaming through the modal keeps working (the heading path of the guard)."""
    path = write_item(backlog_dir, 22, body="## Notes\n\n" + "keep me. " * 500)

    await BR.backlog_task_update(_Req({"id": 22, "name": "Renamed item"}))
    assert strip_frontmatter(path).lstrip().startswith("# Renamed item")
    assert "keep me." in path.read_text(encoding="utf-8")
