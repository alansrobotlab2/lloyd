"""`tags` is a list of strings at every reader, whatever the file says.

Clicking the **lloyd** board in Mission Control threw
``TypeError: task.tags.map is not a function`` and blanked the page. Four items
on that board carried

    tags: '[youtube-eval, ai-engineer, eval, retrieval, memory]'

— a *string* that looks like a list, written by ``youtube_digest``: the prompt
asks for "tags ``{eval_tag}``, ``{channel_key}``, plus the area tags", the tool
schema says ``array of string``, the model answered with one bracketed scalar,
and nothing between the model and the file disagreed. ``save_task`` dumped it
back as a quoted scalar and the shape became durable.

The interesting part is that every reader degraded *differently* and none of
them said so:

  * ``BacklogPage`` called ``.map`` and took the whole board down with it — one
    malformed row out of 600, and the other 599 are unreachable;
  * ``scripts/selfmod/backlog.py`` ran ``[str(t) for t in fm["tags"]]``, which
    iterates a string *by character*, so those items carried 47 one-character
    tags and ``is_quarantined`` — the gate that keeps the triage queue from
    doubling — quietly stopped matching them;
  * ``agent_mcp/backlog.py`` wrapped the string in a list, which reads exactly
    like the fix and means ``backlog_tasks(tag="youtube-eval")`` returns none
    of the four items tagged ``youtube-eval``.

Three private coercions, three different wrong answers. These tests pin the one
definition (``app/backlog_tags.normalize_tags``), that each reader uses it, and
that the writers coerce so the shape cannot reach disk again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
from app.backlog_tags import normalize_tags
from app.routers import backlog as BR
from scripts.selfmod import backlog as SB


# The exact scalar that was on disk, and the list it was meant to be.
BAD_SCALAR = "[youtube-eval, ai-engineer, eval, retrieval, memory]"
WANT = ["youtube-eval", "ai-engineer", "eval", "retrieval", "memory"]


# ── The normalizer ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None, []),                                    # `tags:` with nothing after it
    ([], []),
    (["a", "b"], ["a", "b"]),                      # the well-formed case is identity
    ("a", ["a"]),                                  # a bare word is one tag
    ("a, b, c", ["a", "b", "c"]),
    (BAD_SCALAR, WANT),                            # the shape that broke the board
    ("'" + BAD_SCALAR + "'", WANT),                # ... read from the raw frontmatter line
    ("['a', 'b']", ["a", "b"]),
    ("[]", []),
    (["a", "a", "b"], ["a", "b"]),                 # `key={tag}` — a repeat is a key collision
    ([" spaced ", ""], ["spaced"]),
    (["a", 1, True], ["a", "1", "True"]),          # yaml gives ints for numeric-looking tags
    (42, ["42"]),
])
def test_normalize_tags(value, expected):
    assert normalize_tags(value) == expected


def test_normalize_tags_never_iterates_a_string_by_character():
    """The selfmod failure specifically: `[str(t) for t in "abc"]` is 3 tags."""
    out = normalize_tags(BAD_SCALAR)
    assert all(len(t) > 1 for t in out), out


# ── Every reader ─────────────────────────────────────────────────────────────

def _write(dirpath: Path, task_id: int, tags_block: str, *, board: str = "lloyd",
           status: str = "draft") -> Path:
    p = dirpath / f"{task_id}-a-task.md"
    p.write_text(
        f"---\ntype: backlog\nsegment: backlog\nstatus: {status}\n"
        f"priority: medium\nboard: {board}\nblocked: false\nassigned: false\n"
        f"position: {task_id * 1000}\n{tags_block}\n---\n\n# A task\n\nBody.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def boards(tmp_path, monkeypatch):
    """One directory, wired into all three readers at once."""
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(SB, "BACKLOG_DIR", d)
    return d


@pytest.mark.asyncio
async def test_http_listing_hands_the_frontend_a_list(boards):
    """The reported crash: the board must render with a malformed row on it."""
    _write(boards, 1, f"tags: '{BAD_SCALAR}'")
    _write(boards, 2, "tags:\n- clean")
    _write(boards, 3, "tags:")  # `tags:` with no value parses to None
    resp = await BR.backlog_tasks()
    tasks = {t["id"]: t for t in json.loads(bytes(resp.body))}
    assert tasks[1]["tags"] == WANT
    assert tasks[2]["tags"] == ["clean"]
    assert tasks[3]["tags"] == []
    for t in tasks.values():
        assert isinstance(t["tags"], list)
        assert all(isinstance(x, str) for x in t["tags"])


def test_mcp_listing_and_tag_filter_see_the_real_tags(boards):
    """`_handle_tasks` wrapped the scalar in a list, so the filter matched
    nothing — the failure that reads like a fix."""
    _write(boards, 1, f"tags: '{BAD_SCALAR}'")
    _write(boards, 2, "tags:\n- clean")
    out = json.loads(BL._handle_tasks({}))
    assert {t["id"]: t["tags"] for t in out["tasks"]} == {1: WANT, 2: ["clean"]}
    filtered = json.loads(BL._handle_tasks({"tag": "youtube-eval"}))
    assert [t["id"] for t in filtered["tasks"]] == [1]


def test_mcp_get_task_returns_a_list(boards):
    _write(boards, 1, f"tags: '{BAD_SCALAR}'")
    task = json.loads(BL._handle_get({"task_id": 1}))["task"]
    assert task["tags"] == WANT


def test_selfmod_item_tags_are_whole_tags(boards):
    """`is_quarantined` keys on these; character-tags match nothing."""
    _write(boards, 1, "tags:\n- spawned-by-triage")
    _write(boards, 2, f"tags: '{BAD_SCALAR}'")
    items = {i.id: i for i in SB.open_items()}
    assert items[1].tags == ["spawned-by-triage"]
    assert items[2].tags == WANT
    assert SB.is_quarantined is not None  # imported, not renamed out from under us


# ── Every writer ─────────────────────────────────────────────────────────────

def test_mcp_writer_coerces_a_scalar_before_it_reaches_disk(boards):
    """The origin of all four bad rows: the model answered `array` with a
    string, and `save_task` made it durable."""
    result = json.loads(BL._handle_write({
        "name": "Filed by a digest run", "description": "Body.",
        "board": "lloyd", "tags": BAD_SCALAR,
    }))
    assert result.get("success"), result
    written = sorted(boards.glob("*.md"))[0]
    fm = yaml.safe_load(written.read_text(encoding="utf-8").split("---", 2)[1])
    assert fm["tags"] == WANT, "a scalar reached disk again"


@pytest.mark.asyncio
async def test_http_writers_coerce_too(boards, monkeypatch):
    monkeypatch.setattr(BR, "_backlog_board_map", lambda: {})

    class _Req:
        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

    resp = await BR.backlog_task_create(_Req(
        {"name": "From the UI", "description": "Body.", "tags": BAD_SCALAR}))
    assert resp.status_code == 200, resp.body
    created = sorted(boards.glob("*.md"))[0]
    task_id = int(re.match(r"^(\d+)-", created.name).group(1))
    fm = yaml.safe_load(created.read_text(encoding="utf-8").split("---", 2)[1])
    assert fm["tags"] == WANT

    await BR.backlog_task_update(_Req({"id": task_id, "tags": "one, two"}))
    fm = yaml.safe_load(created.read_text(encoding="utf-8").split("---", 2)[1])
    assert fm["tags"] == ["one", "two"]


# ── Files whose YAML does not parse at all ───────────────────────────────────

# Three items on the lloyd board carry an activity_log entry with an
# unbalanced quote. The router used to `yaml.safe_load` strictly inside a
# blanket `except Exception: continue`, so those three vanished from the
# listing *and* from the board's task count, silently. They are now read by
# the same graduated recovery `agent_mcp/backlog.py` already used.
# The real corruption, reduced: a triage note quoted a code snippet containing
# `\n`-bearing text, and the single-quoted scalar never closes — pyyaml stops
# with "while scanning a quoted scalar ... found unexpected end of stream",
# which is verbatim what #519, #520 and #525 raise today.
BROKEN_FM = (
    "---\ntype: backlog\nstatus: up_next\nboard: lloyd\nposition: 7000\n"
    "tags: [youtube-eval, inference]\n"
    "activity_log:\n- '**2026-09-09** — the writer runs write_text(f\"---\n"
    "segment: agents\n"
    "---\n\n# A task whose activity log broke the YAML\n\nBody.\n"
)


@pytest.mark.asyncio
async def test_a_yaml_broken_file_is_listed_not_dropped(boards):
    (boards / "7-broken.md").write_text(BROKEN_FM, encoding="utf-8")
    tasks = json.loads(bytes((await BR.backlog_tasks()).body))
    assert [t["id"] for t in tasks] == [7]
    assert tasks[0]["status"] == "up_next"
    assert tasks[0]["tags"] == ["youtube-eval", "inference"]


@pytest.mark.asyncio
async def test_a_yaml_broken_file_is_counted_on_its_board(boards):
    (boards / "7-broken.md").write_text(BROKEN_FM, encoding="utf-8")
    boards_out = json.loads(bytes((await BR.backlog_boards()).body))
    assert [(b["name"], b["tasks_count"]) for b in boards_out] == [("lloyd", 1)]


@pytest.mark.asyncio
async def test_the_http_writer_refuses_a_yaml_broken_file(boards):
    """Visible is not the same as writable. The regex fallback recovers a short
    list of fields; rewriting the file from it would drop activity_log and the
    timestamps. Refuse by name rather than round-trip a partial parse."""
    from fastapi import HTTPException

    (boards / "7-broken.md").write_text(BROKEN_FM, encoding="utf-8")
    before = (boards / "7-broken.md").read_text(encoding="utf-8")

    class _Req:
        async def json(self):
            return {"id": 7, "status": "done"}

    with pytest.raises(HTTPException) as exc:
        await BR.backlog_task_update(_Req())
    assert exc.value.status_code == 409
    assert "7-broken.md" in exc.value.detail
    assert (boards / "7-broken.md").read_text(encoding="utf-8") == before


def test_the_mcp_writer_refuses_a_yaml_broken_file(boards):
    (boards / "7-broken.md").write_text(BROKEN_FM, encoding="utf-8")
    before = (boards / "7-broken.md").read_text(encoding="utf-8")
    result = json.loads(BL._handle_write({"task_id": 7, "status": "done"}))
    assert result.get("success") is False
    assert "malformed YAML" in result["error"]
    assert (boards / "7-broken.md").read_text(encoding="utf-8") == before


# ── The live board ───────────────────────────────────────────────────────────

@pytest.mark.live_vault
def test_no_task_on_the_live_board_has_a_scalar_tags_field():
    """The four repaired files stay repaired. Carries `live_vault` so the
    selfmod gate (`-m "not live_vault"`) does not fail a round on a vault the
    round never touched."""
    root = Path.home() / "obsidian" / "backlog"
    if not root.exists():
        pytest.skip("no vault")
    offenders = []
    for f in sorted(root.glob("*.md")):
        if not BR._BACKLOG_PATTERN.match(f.name):
            continue
        fm, _ = BR._backlog_parse_fm(f.read_text(encoding="utf-8"))
        tags = fm.get("tags")
        if tags is not None and not isinstance(tags, list):
            offenders.append((f.name, repr(tags)[:60]))
    assert not offenders, offenders
