"""Every backlog task must be OKF-conformant *at birth*, not eventually.

OKF v0.1 requires exactly one thing of a concept document: parseable frontmatter
with a non-empty ``type`` (``scripts/vault/validate_okf.py``). Two live writers
create task files and neither declared one:

  * ``agent_mcp/backlog.py`` ``_handle_write`` — the new-task dict is
    ``{id, filename, created, status, priority, blocked, assigned, position}``
    and ``save_task`` dumps exactly that dict, so a task created through
    ``backlog_write_task`` has no ``type`` the moment it hits disk;
  * ``app/routers/backlog.py`` ``backlog_task_create`` — the frontend/guardian
    create route builds its own ``fm`` dict with the same keys and the same gap
    (``agent-services/guardian/notify.py`` POSTs to it, so this is not only the
    browser).

Measured at file time: 535 backlog files scanned, 275 violations, and the
newest files written *by these two paths* carry
``created/status/priority/blocked/assigned/position/board/tags`` and no
``type``. So the conformance gate does not decay slowly from old notes — the
writer emits a violation continuously, and every run of the nightly OKF check
counts another one. The fix belongs on the writer; re-running ``okf_migrate.py
--apply`` would stamp these files ``type: note`` instead (no ``backlog`` branch
in ``infer_type``), which is a different item.

These tests pin both writers to the convention the 260 conformant files already
follow — ``type: backlog`` plus ``segment: backlog`` (e.g.
``backlog/100-speaker-id-train-per-speaker-voice-profiles-and-improve-differentiation.md``)
— and re-check the written file through the validator's own rule rather than
only through the spelling of a key.

#1804 added one more key to the same rule on the *create* paths: ``tags`` must
be a non-empty list. ``scripts/vault/segment_scan.py`` now counts a file whose
parsed front matter carries no ``tags``, an empty ``tags: []``, or a scalar, and
12 board files were in that state — 11 with no key (both writers set ``tags``
only ``if`` the caller named one) and one ``tags: []`` — newest written
2026-09-29T02:23:47. Both writers now fall back to the one shared
``app.backlog_tags.DEFAULT_NEW_TASK_TAGS``, for the reason ``type`` was fixed
twice: an MCP-only guard leaves the HTTP route, which is what
``agent-services/guardian/notify.py`` posts to, emitting the next one. The
fallback is create-only — an update neither invents tags on a legacy file nor
replaces tags a loop already stamped — and it is presence, not vocabulary: #868
retired tag-vocabulary maintenance because no query-time consumer reads these
strings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
from app import backlog_tags as BT
from app.routers import backlog as BR

# The strict frontmatter form validate_okf.py requires — the same object the
# nightly gate imports, so a writer that produces a block *this* regex rejects
# fails here even though a lenient parser would have forgiven it.
from scripts.vault.validate_okf import STRICT_FM_RE


def okf_type_of(path: Path) -> str:
    """``validate_okf.main``'s predicate, applied to one file.

    Mirrored (not imported) because the validator's rule is inline in ``main()``
    and its ``VAULT_ROOT`` is fixed at ``Path.home()/obsidian`` — this is the
    4 lines that decide whether a file is a violation, run against a tmp file.
    """
    m = STRICT_FM_RE.match(path.read_text(encoding="utf-8"))
    assert m, f"{path.name}: no parseable frontmatter block"
    fm = yaml.safe_load(m.group(1))
    assert isinstance(fm, dict), f"{path.name}: frontmatter is not a mapping"
    return str(fm.get("type", "") or "").strip()


@pytest.fixture
def mcp_board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    return d


def _mcp_create(board: Path, **extra) -> Path:
    args = {"name": "A newly written task", "description": "Body text.",
            "board": "lloyd", **extra}
    result = json.loads(BL._handle_write(args))
    assert result.get("success"), result
    files = sorted(board.glob("*.md"))
    assert len(files) == 1, files
    return files[0]


# ── MCP writer: backlog_write_task ───────────────────────────────────────────

def test_new_task_frontmatter_declares_type_and_segment(mcp_board):
    path = _mcp_create(mcp_board)
    fm, _ = BL.parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm.get("type") == "backlog", f"got {fm.get('type')!r} in {path.name}"
    assert fm.get("segment") == "backlog", f"got {fm.get('segment')!r} in {path.name}"


def test_new_task_file_passes_the_okf_gate(mcp_board):
    """The acceptance check: the nightly gate must not count the new file."""
    path = _mcp_create(mcp_board, status="up_next", priority="high",
                       tags=["spawned-by-autocode"])
    assert okf_type_of(path) == "backlog"


def test_update_path_does_not_lose_type(mcp_board):
    """``save_task`` round-trips unknown keys verbatim — the fix is not lossy,
    and an update must not be the thing that drops the key back off."""
    path = _mcp_create(mcp_board)
    task_id = int(re.match(r"^(\d+)-", path.name).group(1))
    result = json.loads(BL._handle_write({"task_id": task_id, "status": "in_progress"}))
    assert result.get("success"), result
    assert okf_type_of(path) == "backlog"


def test_update_on_a_legacy_file_restores_segment_but_does_not_invent_type(mcp_board):
    """#1167 changed half of this contract on purpose. `segment: backlog` is the
    store's invariant for every file here, and "update never invents a key" left
    128 pre-#518 files editable for weeks without ever healing — so a save now
    restores it. Which OKF `type` a legacy file is remains a migration decision
    (#585), not a side effect of an unrelated status edit, so type stays absent."""
    legacy = mcp_board / "9-legacy-task.md"
    legacy.write_text("---\nstatus: draft\nboard: lloyd\n---\n\n# Legacy task\n",
                      encoding="utf-8")
    result = json.loads(BL._handle_write({"task_id": 9, "status": "in_progress"}))
    assert result.get("success"), result
    fm, _ = BL.parse_frontmatter(legacy.read_text(encoding="utf-8"))
    assert fm.get("segment") == "backlog"
    assert "type" not in fm
    assert fm.get("status") == "in_progress"


def test_update_keeps_a_segment_the_file_already_has(mcp_board):
    """Restoring is `setdefault`, never an overwrite of a value somebody chose."""
    odd = mcp_board / "10-odd-task.md"
    odd.write_text("---\nstatus: draft\nsegment: projects\n---\n\n# Odd task\n",
                   encoding="utf-8")
    assert json.loads(BL._handle_write({"task_id": 10, "priority": "high"}))["success"]
    fm, _ = BL.parse_frontmatter(odd.read_text(encoding="utf-8"))
    assert fm.get("segment") == "projects"


# ── HTTP writer: POST /api/backlog/task-create ───────────────────────────────

class _FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


@pytest.fixture
def http_board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_backlog_board_map", lambda: {})
    return d


@pytest.mark.asyncio
async def test_frontend_create_route_emits_type_and_segment(http_board):
    resp = await BR.backlog_task_create(_FakeRequest(
        {"name": "Created from the UI", "description": "Body.", "board_id": "lloyd"}))
    assert resp.status_code == 200, resp.body
    files = sorted(http_board.glob("*.md"))
    assert len(files) == 1, files
    assert okf_type_of(files[0]) == "backlog"
    fm = yaml.safe_load(STRICT_FM_RE.match(files[0].read_text()).group(1))
    assert fm.get("segment") == "backlog"


@pytest.mark.asyncio
async def test_frontend_update_route_restores_segment_on_a_legacy_file(http_board):
    """The Mission Control save path is the other editor of these files (#1167)."""
    http_board.mkdir()
    legacy = http_board / "9-legacy-task.md"
    legacy.write_text("---\nstatus: draft\nboard: lloyd\n---\n\n# Legacy task\n",
                      encoding="utf-8")
    resp = await BR.backlog_task_update(_FakeRequest({"id": 9, "priority": "high"}))
    assert resp.status_code == 200, resp.body
    fm = yaml.safe_load(STRICT_FM_RE.match(legacy.read_text()).group(1))
    assert fm.get("segment") == "backlog"
    assert "type" not in fm
    assert fm.get("priority") == "high"


# ── #1804: neither create path may emit an absent or empty `tags` ────────────

def _tags_of(path: Path) -> list:
    """The written file's `tags`, required to be a non-empty list.

    Asserted on the parsed shape, not on the spelling of a line, because the
    three shapes `scripts/vault/segment_scan.py` now counts as missing — no key,
    `tags: []`, a scalar — are three different ways of failing this one check.
    """
    m = STRICT_FM_RE.match(path.read_text(encoding="utf-8"))
    assert m, f"{path.name}: no parseable frontmatter block"
    fm = yaml.safe_load(m.group(1))
    tags = fm.get("tags")
    assert isinstance(tags, list), f"{path.name}: tags is {type(tags).__name__} {tags!r}"
    assert tags, f"{path.name}: tags is an empty list"
    return tags


def test_mcp_create_with_no_tags_writes_a_non_empty_tags_list(mcp_board):
    """Clause 4: `backlog_write_task` with no `tags` argument still declares one."""
    path = _mcp_create(mcp_board)
    assert _tags_of(path) == list(BT.DEFAULT_NEW_TASK_TAGS)


def test_mcp_create_answering_the_array_with_nothing_still_writes_a_list(mcp_board):
    """`tags: []` from a caller is the same hole as no key, and gets the fallback.

    The measured product of the old guard was exactly this file shape: 12 of
    them under `~/obsidian/backlog`, 11 with no `tags` key and one `tags: []`,
    newest born 2026-09-29T02:23:47 — after `segment_scan.py` first shipped.
    """
    path = _mcp_create(mcp_board, tags=[])
    assert _tags_of(path) == list(BT.DEFAULT_NEW_TASK_TAGS)


def test_mcp_create_keeps_the_tags_the_caller_named(mcp_board):
    """The fallback fills a gap, it never overwrites provenance.

    Losing a loop's `spawned-by-*` tag would un-count its own filings from
    expiry and the scorecard's self-spawned gauge (`loop_spawn_tag`).
    """
    path = _mcp_create(mcp_board, tags=["spawned-by-autocode", "blocker"])
    assert _tags_of(path) == ["spawned-by-autocode", "blocker"]


def test_mcp_update_does_not_replace_tags_with_the_default(mcp_board):
    """The guard is create-only, so the create/update branch has to be right.

    Were `creating` ever true on an update, every status change would rewrite a
    loop item's tags to the default and quietly erase who filed it.
    """
    path = _mcp_create(mcp_board, tags=["spawned-by-autocode"])
    task_id = int(re.match(r"^(\d+)-", path.name).group(1))
    result = json.loads(BL._handle_write({"task_id": task_id, "status": "in_progress"}))
    assert result.get("success"), result
    fm, _ = BL.parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm.get("tags") == ["spawned-by-autocode"]


def test_mcp_update_of_a_legacy_file_does_not_invent_tags(mcp_board):
    """The other edge of create-only: an edit does not stamp a tag the file never had.

    Same stance as `type` (#585) — healing the 12 legacy files was a seeded vault
    change, not a side effect of whoever next moves their status.
    """
    legacy = mcp_board / "9-legacy-task.md"
    legacy.write_text("---\nstatus: draft\nboard: lloyd\n---\n\n# Legacy task\n",
                      encoding="utf-8")
    assert json.loads(BL._handle_write({"task_id": 9, "status": "in_progress"}))["success"]
    fm, _ = BL.parse_frontmatter(legacy.read_text(encoding="utf-8"))
    assert "tags" not in fm
    assert fm.get("segment") == "backlog"


@pytest.mark.asyncio
async def test_frontend_create_route_emits_a_non_empty_tags_list(http_board):
    """Clause 5, coroutine side: the route builds its own `fm` dict, so it needs its own guard."""
    resp = await BR.backlog_task_create(_FakeRequest(
        {"name": "Created from the UI", "description": "Body.", "board_id": "lloyd"}))
    assert resp.status_code == 200, resp.body
    files = sorted(http_board.glob("*.md"))
    assert len(files) == 1, files
    assert okf_type_of(files[0]) == "backlog"
    assert _tags_of(files[0]) == list(BT.DEFAULT_NEW_TASK_TAGS)


@pytest.mark.asyncio
async def test_frontend_create_route_keeps_the_tags_the_post_named(http_board):
    """A POST that names tags gets those, not the fallback."""
    resp = await BR.backlog_task_create(_FakeRequest(
        {"name": "Alert with provenance", "description": "Body.", "board_id": "lloyd",
         "tags": ["observability"]}))
    assert resp.status_code == 200, resp.body
    files = sorted(http_board.glob("*.md"))
    assert len(files) == 1, files
    assert _tags_of(files[0]) == ["observability"]


def test_task_create_post_over_http_writes_a_non_empty_tags_list(http_board):
    """The real seam: a POST down a TestClient-mounted router, as the UI and guardian send it.

    `agent-services/guardian/notify.py` posts with no `tags` field at all, which
    is the request shape that used to produce a tagless file, so the assertion
    runs on the file the route wrote through FastAPI's own request parsing
    rather than on a hand-built dict.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(BR.router)
    with TestClient(app) as client:
        resp = client.post("/api/backlog/task-create",
                           json={"name": "Guardian alert", "description": "Body."})
    assert resp.status_code == 200, resp.text
    files = sorted(http_board.glob("*.md"))
    assert len(files) == 1, files
    assert _tags_of(files[0]) == list(BT.DEFAULT_NEW_TASK_TAGS)
