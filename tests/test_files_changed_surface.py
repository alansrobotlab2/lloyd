"""The chat's side of the change ledger: the footer's data and the revert route.

The ledger lives in the aggregator (it owns the filesystem tools); the chat
lives in the backend. These pin the seam — that the backend reads the ledger
over loopback through `service_url` rather than a hardcoded port, that a
failure there costs a footer and never a turn, and that a revert stamps the
persisted message so a reload does not offer the button again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.routers import messages as M


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class _Client:
    """httpx.AsyncClient stand-in that records requests."""
    calls: list = []
    response = None
    raises = None

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        _Client.calls.append(("GET", url, params))
        if _Client.raises:
            raise _Client.raises
        return _Client.response

    async def post(self, url, json=None):
        _Client.calls.append(("POST", url, json))
        if _Client.raises:
            raise _Client.raises
        return _Client.response


@pytest.fixture(autouse=True)
def _client(monkeypatch):
    import httpx
    _Client.calls = []
    _Client.response = _Resp(200, {"files": []})
    _Client.raises = None
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return _Client


LEDGER_FILES = [
    {"path": "/home/x/a.py", "real": "/home/x/a.py", "op": "edit",
     "reverted_at": None, "call_id": "c1"},
    {"path": "/home/x/b.tsx", "real": "/home/x/b.tsx", "op": "create",
     "reverted_at": None, "call_id": "c2"},
]


# ── reading the ledger ──────────────────────────────────────────────────────

async def test_files_changed_is_fetched_over_loopback(_client):
    _client.response = _Resp(200, {"files": LEDGER_FILES})
    out = await M._fetch_files_changed("sid", "turn-1")
    assert out["turn_id"] == "turn-1"
    assert [f["path"] for f in out["files"]] == ["/home/x/a.py", "/home/x/b.tsx"]
    assert [f["op"] for f in out["files"]] == ["edit", "create"]
    method, url, params = _client.calls[0]
    assert method == "GET" and url.endswith("/changes")
    assert params == {"session": "sid", "turn": "turn-1"}


async def test_the_aggregator_url_comes_from_the_service_registry(_client, monkeypatch):
    """`dashboard.py` hardcodes 8500; this route must not inherit that."""
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "services",
                        {**(CONFIG.get("services") or {}),
                         "lloyd_mcp": "http://127.0.0.1:9999/mcp"})
    _client.response = _Resp(200, {"files": LEDGER_FILES})
    await M._fetch_files_changed("sid", "turn-1")
    assert _client.calls[0][1] == "http://127.0.0.1:9999/changes"


async def test_no_files_means_no_footer(_client):
    _client.response = _Resp(200, {"files": []})
    assert await M._fetch_files_changed("sid", "turn-1") is None


async def test_a_missing_turn_id_never_asks(_client):
    assert await M._fetch_files_changed("sid", "") is None
    assert await M._fetch_files_changed("", "turn-1") is None
    assert _client.calls == []


async def test_an_unreachable_aggregator_costs_a_footer_not_a_turn(_client):
    _client.raises = RuntimeError("connection refused")
    assert await M._fetch_files_changed("sid", "turn-1") is None


async def test_a_non_200_is_not_a_footer(_client):
    _client.response = _Resp(503, {}, "degraded")
    assert await M._fetch_files_changed("sid", "turn-1") is None


def test_files_changed_rides_on_the_turn_stats():
    """Same dict object as `structured`: persisted on the message AND sent in
    every `done` branch, so a reload keeps the footer."""
    src = Path(M.__file__).read_text()
    assert 'stream_stats["files_changed"] = files_changed' in src
    assert "options.turn_id = turn.turn_id" in src


# ── the revert route ────────────────────────────────────────────────────────

async def test_revert_proxies_and_stamps_the_persisted_message(_client, tmp_path,
                                                               monkeypatch):
    from app.routers import sessions as S

    _client.response = _Resp(200, {"results": [
        {"path": "/home/x/a.py", "op": "edit", "status": "restored"},
        {"path": "/home/x/b.tsx", "op": "create", "status": "refused",
         "reason": "file changed since this turn wrote it"},
    ]})

    session = {"messages": [{
        "id": "m1", "role": "assistant",
        "stats": {"files_changed": {"turn_id": "turn-1", "files": [
            {"path": "/home/x/a.py", "op": "edit", "reverted_at": None},
            {"path": "/home/x/b.tsx", "op": "create", "reverted_at": None},
        ]}},
    }]}

    async def fake_mutate(session_id, fn):
        fn(session)
        return True
    monkeypatch.setattr(S, "mutate_session", fake_mutate)

    class _Req:
        async def json(self):
            return {}

    resp = await S.revert_turn("sid", "turn-1", _Req())
    body = json.loads(resp.body)
    assert [r["status"] for r in body["results"]] == ["restored", "refused"]

    files = session["messages"][0]["stats"]["files_changed"]["files"]
    assert files[0]["reverted_at"] is not None, "the restored file was not stamped"
    assert files[1]["reverted_at"] is None, "a refused file must NOT read as reverted"


async def test_revert_does_not_stamp_another_turns_footer(_client, monkeypatch):
    from app.routers import sessions as S

    _client.response = _Resp(200, {"results": [
        {"path": "/home/x/a.py", "op": "edit", "status": "restored"}]})
    session = {"messages": [{
        "id": "m1", "role": "assistant",
        "stats": {"files_changed": {"turn_id": "turn-OTHER", "files": [
            {"path": "/home/x/a.py", "op": "edit", "reverted_at": None}]}},
    }]}

    async def fake_mutate(session_id, fn):
        fn(session)
        return True
    monkeypatch.setattr(S, "mutate_session", fake_mutate)

    class _Req:
        async def json(self):
            return {}

    await S.revert_turn("sid", "turn-1", _Req())
    assert session["messages"][0]["stats"]["files_changed"]["files"][0][
        "reverted_at"] is None


async def test_an_unreachable_aggregator_is_a_503(_client):
    from fastapi import HTTPException
    from app.routers import sessions as S

    _client.raises = RuntimeError("connection refused")

    class _Req:
        async def json(self):
            return {}

    with pytest.raises(HTTPException) as exc:
        await S.revert_turn("sid", "turn-1", _Req())
    assert exc.value.status_code == 503


async def test_named_paths_are_forwarded(_client, monkeypatch):
    from app.routers import sessions as S

    _client.response = _Resp(200, {"results": []})

    class _Req:
        async def json(self):
            return {"paths": ["/home/x/a.py"]}

    await S.revert_turn("sid", "turn-1", _Req())
    _method, url, body = _client.calls[0]
    assert url.endswith("/changes/revert")
    assert body == {"session": "sid", "turn": "turn-1", "paths": ["/home/x/a.py"]}


# ── the web surface ─────────────────────────────────────────────────────────

WEB = Path(M.__file__).resolve().parent.parent.parent / "web" / "src"


def test_api_ts_exposes_the_types_and_the_call():
    src = (WEB / "api.ts").read_text()
    assert "export interface ChangedFile" in src
    assert "export interface FilesChanged" in src
    assert "files_changed?: FilesChanged" in src
    assert "async revertTurn(" in src
    assert "/turns/${encodeURIComponent(turnId)}/revert" in src


def test_the_footer_confirms_and_renders_refusals_per_file():
    src = (WEB / "components" / "ChatPanel.tsx").read_text()
    assert "ChangedFilesFooter" in src
    assert "window.confirm" in src, "an undo that touches production asks first"
    assert "refused:" in src, "a refused file must be visible, not silently absent"
    assert "(reverted)" in src


def test_node_modules_symlink_is_ignored():
    """gate.py's frontend rung symlinks node_modules into every worktree it
    type-checks, and a trailing-slash pattern is directory-only."""
    import subprocess
    root = Path(M.__file__).resolve().parent.parent.parent
    r = subprocess.run(["git", "-C", str(root), "check-ignore", "-v",
                        "web/node_modules"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, "a node_modules symlink would show as untracked"
    assert r.stdout.split(":")[0].endswith(".gitignore")
