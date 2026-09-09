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
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
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
                       tags=["spawned-by-selfmod"])
    assert okf_type_of(path) == "backlog"


def test_update_path_does_not_lose_type(mcp_board):
    """``save_task`` round-trips unknown keys verbatim — the fix is not lossy,
    and an update must not be the thing that drops the key back off."""
    path = _mcp_create(mcp_board)
    task_id = int(re.match(r"^(\d+)-", path.name).group(1))
    result = json.loads(BL._handle_write({"task_id": task_id, "status": "in_progress"}))
    assert result.get("success"), result
    assert okf_type_of(path) == "backlog"


def test_update_on_a_legacy_file_without_type_does_not_invent_one(mcp_board):
    """Scope guard for the fix above: backfilling the 275 existing violations is
    a migration decision (#585), not a side effect of an unrelated status edit.
    An update to a file that has no type must leave it absent."""
    legacy = mcp_board / "9-legacy-task.md"
    legacy.write_text("---\nstatus: draft\nboard: lloyd\n---\n\n# Legacy task\n",
                      encoding="utf-8")
    result = json.loads(BL._handle_write({"task_id": 9, "status": "in_progress"}))
    assert result.get("success"), result
    fm, _ = BL.parse_frontmatter(legacy.read_text(encoding="utf-8"))
    assert "type" not in fm


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
