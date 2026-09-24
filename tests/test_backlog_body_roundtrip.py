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

import json
import re
from pathlib import Path

import pytest
import yaml

from app import frontmatter as FM
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
    # `parts[2].strip()`, so a body ending in a space would lose it and this check
    # would be about that pre-existing normalisation instead of about the body.
    # Vault bodies end with a newline after prose.
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
async def test_a_priority_only_update_survives_frontmatter_the_writer_would_reflow(
    backlog_dir,
):
    """The half of the clause that survives its amendment: the body never moves.

    `_write_task_file` re-dumps the whole frontmatter block with `yaml.dump`, so a
    file whose block is not already in that canonical shape changes more than
    `priority:`/`updated:` when it is saved: a hand-written `'board': lloyd`
    becomes `board: lloyd`, and a value past the 80-column wrap is folded onto
    continuation lines. Measured on the live board 2026-09-17, 266 of the 1,129
    files carrying a `created:` key store it unquoted (`grep -lE '^created: [0-9]'
    | wc -l` in `~/obsidian/backlog`) and each re-quotes on the next save. Alan
    dropped the byte-identity half of clause 3 on 2026-09-17 for exactly this
    reason (item #1199 `activity`): it is pre-existing route behaviour, and fixing
    it means preserving the block's original text through a write — a write-path
    change, recorded as a finding on #1199, not a payload fix.

    So the guarantee that must never break, and the one this test pins for *any*
    frontmatter shape, is that the **body** is untouched. The line-level identity
    above holds for files in the shape the route itself writes.
    """
    p = backlog_dir / "23-reflowed-item.md"
    p.write_text(
        "---\ntype: 'backlog'\nsegment: backlog\nstatus: 'draft'\npriority: medium\n"
        "board: \"lloyd\"\nblocked: False\nassigned: False\nposition: 23000\n"
        "title: A backlog item whose hand written summary of the change is long "
        "enough that yaml dot dump will want to fold it onto a continuation line\n"
        "updated: '2026-09-01T00:00:00'\n---\n\n"
        "# Reflowed item\n\n## Handoff\n\n" + "keep every byte. " * 400 + "\n",
        encoding="utf-8",
    )
    before = p.read_text(encoding="utf-8")
    body_before = body_on_disk(p)
    assert "'backlog'" in before, "fixture must start non-canonical or it proves nothing"

    await BR.backlog_task_update(_Req({"id": 23, "priority": "high"}))

    assert body_on_disk(p) == body_before, (
        "a priority click changed the body of a file whose frontmatter the writer "
        "also re-flowed — the body is the part that must never move"
    )
    after = p.read_text(encoding="utf-8")
    # Positive control: the block really was re-dumped, so this is the scenario and
    # not a canonical file passing by accident. `yaml.dump` drops quotes it does not
    # need and lowercases Python bools.
    assert "'backlog'" not in after and "board: lloyd" in after, (
        "frontmatter came back un-re-quoted: the fixture is already canonical and "
        "the scenario this test exists for is untested"
    )
    # The body's text survives exactly; the one thing that does not survive is the
    # file's trailing whitespace, because `_backlog_parse_fm` hands the writer a
    # `.strip()`ed body. That byte is the writer's pre-existing normalisation, and
    # it is the whole of what a hand-written file's `git diff` can show besides the
    # re-quoted frontmatter lines — so it is named here rather than glossed.
    tail_of = lambda t: t.split("---", 2)[2].rstrip()
    assert tail_of(after) == tail_of(before), (
        "everything after the frontmatter moved apart from trailing whitespace"
    )
    assert "keep every byte." in after


@pytest.mark.asyncio
async def test_name_and_description_is_a_valid_full_rewrite_pair(backlog_dir):
    """Renaming through the modal keeps working (the heading path of the guard)."""
    path = write_item(backlog_dir, 22, body="## Notes\n\n" + "keep me. " * 500)

    await BR.backlog_task_update(_Req({"id": 22, "name": "Renamed item"}))
    assert strip_frontmatter(path).lstrip().startswith("# Renamed item")
    assert "keep me." in path.read_text(encoding="utf-8")


# ── the identity the modal depends on ────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("size", [200, 120_000], ids=["short-body", "120KB-body"])
async def test_the_detail_routes_description_is_a_fixed_point_of_the_write(
    backlog_dir, size
):
    """`GET /api/backlog/task/{id}`'s `description`, posted straight back with
    `force_body_replace`, must reproduce the same `description`.

    This is the seam the editor's guard stands on: `TaskModal` reads the body from
    the detail route, withholds `description` until it arrives, then posts it. If
    the detail route's string and the write's string were not the same value, a
    faithful read-edit-save would still lose bytes — the loss would just be
    smaller than the snippet's and nobody would notice. 200 chars covers a body
    under the snippet cap (so the modal is posting what the list would have shown
    anyway); 120 KB covers the largest real body (118,494 B on the live board).
    """
    unit = "sentence that has to come back unchanged. "
    body = "## Handoff\n\n" + " ".join([unit] * (size // len(unit) + 1))
    write_item(backlog_dir, 31, body=body)

    sent = BR.backlog_task_detail(31)
    description = json.loads(bytes(sent.body))["description"]
    assert description, "detail route returned an empty body to round-trip"

    await BR.backlog_task_update(_Req({
        "id": 31, "description": description, "force_body_replace": True,
    }))

    again = json.loads(bytes(BR.backlog_task_detail(31).body))["description"]
    assert again == description, (
        f"the write did not reproduce the read: {len(description)} B in, "
        f"{len(again)} B out"
    )
    assert again == body.strip(), "the round-trip is not the body that was written"


# ── #1221 clause 3: an item whose front matter *quotes the fence* is writable ─────

#: The captured front matter of the twenty items the retired split locked — the
#: same fixtures `tests/test_frontmatter_parsing.py` reads, reused here because this
#: file is where the body-preservation guarantee lives, and a fenced block is
#: exactly the case that guarantee had never been held against.
LOCKED_CAPTURES = Path(__file__).resolve().parent / "fixtures" / "frontmatter_locked_items"

CAPTURE_BODY = "## Handoff\n\nProse that has to survive an edit to a locked item.\n"


def write_captured_item(d: Path, item_id: int, capture_id: int) -> Path:
    """Item `item_id` carrying item `capture_id`'s real captured front matter.

    The block comes off disk verbatim: it is the text that locked the real item,
    including the activity-log scalar that quotes the retired split. Only the body
    is written fresh, so this file can assert about it.
    """
    block = (LOCKED_CAPTURES / f"{capture_id}-frontmatter.txt").read_text(encoding="utf-8")
    p = d / f"{item_id}-captured-item.md"
    p.write_text(f"---\n{block}---\n\n# Captured item {capture_id}\n\n{CAPTURE_BODY}",
                 encoding="utf-8")
    return p


def anchored_fm(path: Path) -> dict:
    """The file's front matter, read the way the route now reads it.

    Deliberately not the two helpers above, which slice on the bare fence because
    no file *they* build has one inside its block. On this fixture the unanchored
    slice returns the text up to the fence quoted inside the activity log — the
    defect itself, not a way of measuring it.
    """
    block = FM.split_frontmatter(path.read_text(encoding="utf-8"))[0]
    return yaml.safe_load(block)


def anchored_body(path: Path) -> str:
    """Everything after the closing fence line, heading and all."""
    return FM.split_frontmatter(path.read_text(encoding="utf-8"))[1]


@pytest.mark.asyncio
async def test_an_update_to_an_item_whose_front_matter_quotes_the_fence_succeeds(
    backlog_dir
):
    """Clause 3: the malformed-frontmatter refusal must stop firing on valid YAML.

    This is the user-visible half of #1146 and #1221: `backlog_write_task` answered
    "has malformed YAML frontmatter … fix the file by hand" and the board route
    answered HTTP 409, on items whose YAML `yaml.safe_load` reads without complaint.
    #460 is the capture used here because it is the item #1146 names as the trigger
    — its activity log quotes the retired split, and that quotation is what ended
    its front-matter block.

    The assertion is the update itself plus everything it may not touch: the
    request succeeds, the recorded status note is the only front-matter text that
    grew, every key the capture carried is still there after the re-dump, the
    quoted fence is still quoted, and not one byte of the body moved.
    """
    path = write_captured_item(backlog_dir, 40, 460)

    # Control: the retired rule still cannot read this file. It takes the text up to
    # the `---` inside the activity-log scalar for the whole block, and that
    # truncated YAML does not parse — which is what produced the `_yaml_broken` the
    # 409 was built on.
    truncated = path.read_text(encoding="utf-8").split("---", 2)[1]
    try:
        yaml.safe_load(truncated)
        raise AssertionError("capture no longer breaks the retired split")
    except yaml.YAMLError:
        pass

    before = anchored_fm(path)
    assert isinstance(before.get("activity_log"), list) and before["activity_log"]
    before_body = anchored_body(path)
    assert before_body.strip() == f"# Captured item 460\n\n{CAPTURE_BODY}".strip()

    resp = await BR.backlog_task_update(_Req({
        "id": 40, "status": "up_next", "priority": "high",
    }))

    assert resp.status_code == 200, "a valid item was still refused as malformed"
    after = anchored_fm(path)
    assert "_yaml_broken" not in after, "the write re-broke the record it just read"
    assert after["status"] == "up_next" and after["priority"] == "high"
    # `segment: backlog` is the one key a save may add: the store's invariant,
    # restored on legacy files by #1167.
    assert set(after) == set(before) | {"segment"}, (
        f"the re-dump changed the key set: {sorted(set(before) ^ set(after))}")
    # The one thing the update is allowed to add: the note recording the move.
    assert len(after["activity_log"]) == len(before["activity_log"]) + 1, \
        "the status move did not append exactly one note"
    assert any("---" in entry for entry in after["activity_log"]), \
        "the quoted fence did not survive the re-dump"
    assert anchored_body(path).strip() == before_body.strip(), "the body moved"
