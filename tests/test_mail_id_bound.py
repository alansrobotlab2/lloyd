"""Backlog #614: the aggregator bounds a bulk-id mail call by count.

The Thunderbird extension declares `messageIds` with no `maxItems`
(`extension/mcp_server/api.js:1624`) and its `deleteMessages` handler checks
only non-empty — and the whole extension tree is gitignored, so no round can
change it. The one choke point every mail tool passes through is
`agent_mcp.thunderbird.call_tool`, so the count bound lives there: a constant,
a comparison and a rendered refusal, applied before the call is forwarded and
therefore before any grant decision, for every scope.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from agent_mcp import thunderbird as tb
from app.harness.policy import GrantStore, check_grants

CAP = 50


class StubPool:
    """Stands in for the bridge and records exactly what was forwarded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, bridge_name: str, arguments: dict) -> dict:
        self.calls.append((bridge_name, arguments))
        return {"content": "forwarded", "is_error": False}


@pytest.fixture
def stub(monkeypatch):
    pool = StubPool()

    async def _fake_pool():
        return pool

    monkeypatch.setattr(tb, "_get_pool", _fake_pool)
    return pool


def _ids(n: int) -> list[str]:
    return [f"msg-{i}" for i in range(n)]


def _text(result) -> str:
    return "".join(part.text for part in result.content)


# ── Clause 1: email_delete is count-bound and never forwarded over the cap ──

async def test_email_delete_over_the_cap_is_refused_and_not_forwarded(stub):
    result = await tb.call_tool("email_delete", {"messageIds": _ids(1200)})

    assert result.is_error is True
    text = _text(result)
    assert "1200" in text, f"refusal must name the actual count, got: {text}"
    assert "50" in text, f"refusal must name the cap, got: {text}"
    assert stub.calls == [], f"over-cap ids must never reach the bridge: {stub.calls}"


async def test_email_delete_at_the_cap_is_forwarded(stub):
    result = await tb.call_tool("email_delete", {"messageIds": _ids(CAP)})

    assert result.is_error is False
    assert stub.calls == [("deleteMessages", {"messageIds": _ids(CAP)})]


# ── Clause 2: email_update's array is bound identically; other shapes pass ──

async def test_email_update_array_over_the_cap_is_refused(stub):
    result = await tb.call_tool(
        "email_update", {"folderPath": "Inbox", "messageIds": _ids(1200)}
    )

    assert result.is_error is True
    text = _text(result)
    assert "1200" in text and "50" in text, f"got: {text}"
    assert stub.calls == []


async def test_email_update_singular_message_id_forwards_unchanged(stub):
    args = {"folderPath": "Inbox", "messageId": "msg-1", "read": True}
    result = await tb.call_tool("email_update", args)

    assert result.is_error is False
    assert stub.calls == [("updateMessage", args)], "arguments must pass through as given"


async def test_email_update_array_at_the_cap_forwards_unchanged(stub):
    ids = _ids(CAP)
    args = {"folderPath": "Inbox", "messageIds": ids}
    result = await tb.call_tool("email_update", args)

    assert result.is_error is False
    assert stub.calls == [("updateMessage", args)]


# ── Clause 3: a constant, liftable only by an env var read at call time ─────

def test_the_cap_is_a_module_constant_of_fifty():
    assert tb.MAIL_ID_ARRAY_CAP == 50


async def test_the_env_var_lifting_the_cap_is_read_at_call_time(stub, monkeypatch):
    """Raising the cap needs no code edit and no config.yaml change."""
    monkeypatch.delenv("LLOYD_MAIL_ID_CAP", raising=False)
    assert (await tb.call_tool("email_delete", {"messageIds": _ids(60)})).is_error is True
    assert stub.calls == []

    monkeypatch.setenv("LLOYD_MAIL_ID_CAP", "2000")
    assert (await tb.call_tool("email_delete", {"messageIds": _ids(60)})).is_error is False
    assert len(stub.calls) == 1


async def test_an_unparseable_env_value_leaves_the_bound_at_fifty(stub, monkeypatch):
    """A bad override must not become no bound."""
    monkeypatch.setenv("LLOYD_MAIL_ID_CAP", "unlimited")
    assert (await tb.call_tool("email_delete", {"messageIds": _ids(60)})).is_error is True
    assert "50" in _text(await tb.call_tool("email_delete", {"messageIds": _ids(60)}))
    assert stub.calls == []


# ── Clause 4: enforced in the aggregator, before the grant decision ─────────

async def test_the_bound_holds_where_the_grant_gate_does_not_fire(tmp_path, stub):
    """Attended chat and a predicate-less grant both let 1200 ids through the
    policy gate (#614's two live residuals). The aggregator bound does not
    depend on either: it refuses first, for every scope."""
    store = GrantStore(tmp_path / "grants.sqlite")
    store.ensure_schema()
    ids = _ids(1200)

    interactive = check_grants(
        store, scope="interactive", tool_name="email_delete",
        tool_input={"messageIds": ids}, record=False,
    )
    assert interactive.allowed is True, "attended scope is ungated by policy"

    store.mint(scope="worker:autotriage", tool_pattern="email_delete",
               arg_predicate="", issued_by="alan",
               expires_at=_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(minutes=10),
               note="predicate-less: authorizes any array size")
    granted = check_grants(
        store, scope="worker:autotriage", tool_name="email_delete",
        tool_input={"messageIds": ids}, record=False,
    )
    assert granted.allowed is True, "a predicate-less grant bounds nothing"

    result = await tb.call_tool("email_delete", {"messageIds": ids})
    assert result.is_error is True
    assert "1200" in _text(result)
    assert stub.calls == []

    # The bound was added at the aggregator, not by re-tiering the gate.
    from app.harness import policy
    assert "email_delete" in policy.TIER3_TOOLS
    assert "email_update" in policy.TIER2_TOOLS


# ── Clause 5: tools with no id array are untouched by the new refusal path ──

@pytest.mark.parametrize("tool_name,bridge_name,args", [
    ("email_empty_trash", "emptyTrash", {"accountId": "acct1"}),
    ("email_empty_junk", "emptyJunk", {"accountId": "acct1"}),
    ("email_delete_folder", "deleteFolder", {"folderPath": "Inbox/Junk"}),
    ("contacts_delete", "deleteContact", {"contactId": "uid-1"}),
    ("calendar_delete_event", "deleteEvent", {"eventId": "e-1", "calendarId": "c-1"}),
])
async def test_tools_taking_no_id_array_still_forward(stub, tool_name, bridge_name, args):
    result = await tb.call_tool(tool_name, args)

    assert result.is_error is False
    assert stub.calls == [(bridge_name, args)]
