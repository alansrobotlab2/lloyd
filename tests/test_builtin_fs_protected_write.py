"""The Write/Edit lane refuses the write deny-set, and nothing else does.

Backlog #1049. The failure this pins is the shortest route to the identity
file: `Read ~/obsidian/lloyd/SOUL.md`, then `Write` it back with different
bytes. Two tool calls, both ordinary, both allowed at HEAD `617f708b` — the
Bash route to the same file had a checker (`app/harness/safety.py`, and the
read-only sandbox for bench sessions) and the file route had none. Measured
there before the change:

    Write isError=False text=File written: …/lloyd/SOUL.md (9 chars)
    after:  'CLOBBERED'

and after it:

    Write isError=True text={"error": "Write refused: … is protected (…
    codes:  ['PROTECTED_PATH']
    after:  'ORIGINAL IDENTITY FILE'

Every test here runs against a scratch `$HOME`, so the live vault is never a
target; the deny-set is home-relative and moves with `HOME`, which is also why
`agent_mcp.builtin_fs` has to resolve it per call rather than cache it.

The allow half is as load-bearing as the deny half. An automod round edits its
own worktree, which contains directories named like deny entries; the nightly
knowledge-write job edits the loaded-memory files that sit *beside* the denied
identity file; `vault_write` is a separate lane with its own root check. A
predicate that refused any of those would not be a fix, it would be an outage.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import builtin_fs as FS  # noqa: E402
from agent_mcp import main as M  # noqa: E402
from app.harness import protected_paths as PP  # noqa: E402

SID = "20260921_1049_test"
ORIGINAL = "ORIGINAL IDENTITY FILE"

#: One existing file per deny-set entry, vault-relative to the scratch home.
DENIED = {
    "identity file": "obsidian/lloyd/SOUL.md",
    "credential tree": ".openclaw/config.json",
    "service unit": "lloyd/agent-services/supervisor/conf.d/agent-backend.conf",
    "interpreter": "lloyd/.venvs/lloyd/bin/python",
}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """Off the change ledger: these tests assert on the bytes on disk, not on
    who is recorded as having changed them."""
    from agent_mcp import _change_ledger
    monkeypatch.setattr(_change_ledger, "enabled", lambda: False)
    FS.reset_read_records()
    yield
    FS.reset_read_records()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch `$HOME` with every deny entry present and readable."""
    h = tmp_path / "home"
    for rel in DENIED.values():
        p = h / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(ORIGINAL)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    return h


def _text(res) -> str:
    return res.content[0].text


def _json(res) -> dict:
    try:
        return json.loads(_text(res))
    except (ValueError, TypeError):
        return {}


def _code(res) -> str:
    return _json(res).get("code", "")


def _refused(res, target: Path, expected: str = ORIGINAL) -> None:
    """A refusal that demonstrably wrote nothing."""
    assert res.is_error is True, _text(res)
    assert _code(res) == "PROTECTED_PATH", _text(res)
    assert target.read_text() == expected, (
        "the refusal must leave the bytes exactly as they were")


# ── clause 1: a denied target is refused, even after a Read ─────────────────

@pytest.mark.parametrize("rel", sorted(DENIED.values()), ids=sorted(DENIED))
async def test_a_write_is_refused_even_after_reading_that_same_file(home, rel):
    target = home / rel
    assert not (await FS.call_tool("Read", {"file_path": str(target)})).is_error
    _refused(await FS.call_tool("Write", {"file_path": str(target),
                                          "content": "OVERWRITTEN"}), target)


@pytest.mark.parametrize("rel", sorted(DENIED.values()), ids=sorted(DENIED))
async def test_an_edit_is_refused_even_after_reading_that_same_file(home, rel):
    target = home / rel
    assert not (await FS.call_tool("Read", {"file_path": str(target)})).is_error
    _refused(await FS.call_tool("Edit", {"file_path": str(target),
                                         "old_string": "ORIGINAL",
                                         "new_string": "REPLACED"}), target)


async def test_a_tilde_spelling_of_a_denied_path_is_refused_too(home):
    """`_expand` runs before the check, or `~/obsidian/...` walks past it."""
    target = home / DENIED["identity file"]
    res = await FS.call_tool("Write", {"file_path": "~/obsidian/lloyd/SOUL.md",
                                       "content": "OVERWRITTEN"})
    _refused(res, target)
    assert "~" in _json(res)["error"] or str(target) in _json(res)["error"]


async def test_a_symlink_pointing_into_the_set_is_refused_from_outside(home, tmp_path):
    """Realpath-first cuts both ways: the link lives outside, the bytes land
    inside, and the answer is the same as for the file itself."""
    target = home / DENIED["identity file"]
    link = tmp_path / "elsewhere.md"
    link.symlink_to(target)
    await FS.call_tool("Read", {"file_path": str(link)})
    _refused(await FS.call_tool("Write", {"file_path": str(link),
                                          "content": "OVERWRITTEN"}), target)


async def test_reading_a_denied_file_is_still_allowed(home):
    """The deny-set is about writes. This file is in the system prompt already;
    refusing to read it would only break the turn that is trying to fix it."""
    target = home / DENIED["identity file"]
    res = await FS.call_tool("Read", {"file_path": str(target)})
    assert res.is_error is False
    assert ORIGINAL in _text(res)


# ── clause 2: a create inside a denied directory is refused too ─────────────

async def test_creating_a_file_inside_a_denied_directory_is_refused(home):
    """`_gate_check` used to return early for a target that does not exist —
    there is nothing to clobber. Location is not a clobber question, so the
    deny-set is consulted before that early return."""
    new = home / "lloyd" / "agent-services" / "supervisor" / "conf.d" / "evil.conf"
    assert not new.exists()
    res = await FS.call_tool("Write", {"file_path": str(new), "content": "x"})
    assert res.is_error is True and _code(res) == "PROTECTED_PATH", _text(res)
    assert not new.exists(), "the refusal must not have created the file"


async def test_removing_a_protected_file_and_writing_it_back_is_not_a_route(home):
    """The two-step version of clause 2: delete out of band, then `Write` the
    replacement. At the moment of the write the target is a create."""
    target = home / DENIED["identity file"]
    target.unlink()
    res = await FS.call_tool("Write", {"file_path": str(target), "content": "NEW IDENTITY"})
    assert res.is_error is True and _code(res) == "PROTECTED_PATH", _text(res)
    assert not target.exists(), "the refusal must not have recreated it"


async def test_the_deny_set_holds_with_no_session_bound(home, monkeypatch):
    """`gate_on` is false without a session, and that switch belongs to the
    clobber gate. A caller that arrives with no aggregator context still may
    not write here."""
    monkeypatch.setattr(FS, "get_bound_session", lambda: "")
    target = home / DENIED["identity file"]
    _refused(await FS.call_tool("Write", {"file_path": str(target),
                                          "content": "OVERWRITTEN"}), target)


async def test_the_deny_set_holds_when_the_clobber_gate_is_disabled(home, monkeypatch):
    """`harness.edit_gates.enabled` turns the Read-before-write gate off. It
    must not turn this one off with it."""
    monkeypatch.setattr(FS, "_gates_enabled", lambda: False)
    target = home / DENIED["identity file"]
    _refused(await FS.call_tool("Write", {"file_path": str(target),
                                          "content": "OVERWRITTEN"}), target)


# ── clause 4: only code can lift it ─────────────────────────────────────────

async def test_the_same_arguments_are_refused_until_code_lifts_them(home):
    """The one pair the clause asks for: identical arguments, different
    outcome, and the difference is a call to `allow_protected_writes`."""
    target = home / DENIED["identity file"]
    args = {"file_path": str(target), "content": "GRANTED WRITE"}
    # Read first: the lift is of the deny-set only, so a granted Write still
    # owes the Read-before-overwrite clobber gate like any other write.
    assert not (await FS.call_tool("Read", {"file_path": str(target)})).is_error
    _refused(await FS.call_tool("Write", args), target)
    with PP.allow_protected_writes("test: the nightly knowledge-write lane"):
        assert (await FS.call_tool("Write", args)).is_error is False
    assert target.read_text() == "GRANTED WRITE"
    # And it cannot outlive its scope: the same call is refused again outside.
    target.write_text(ORIGINAL)
    _refused(await FS.call_tool("Write", args), target)


async def test_a_grant_taken_on_the_event_loop_reaches_the_worker_thread(home):
    """The seam: `call_tool` hands the handler to `asyncio.to_thread`, so the
    write happens on another thread while the grant lives in this task's
    context. `to_thread` copies the context, which is what makes that work —
    a plain module global read on the loop and written on the executor would
    not have."""
    target = home / DENIED["credential tree"]
    await FS.call_tool("Read", {"file_path": str(target)})
    token = PP.grant_protected_writes("test: crossing into to_thread")
    try:
        res = await FS.call_tool("Write", {"file_path": str(target), "content": "ok"})
        assert res.is_error is False, _text(res)
        assert target.read_text() == "ok"
    finally:
        PP.release_protected_writes(token)
    assert PP.protected_writes_granted() is None


async def test_a_narrowed_grant_lifts_only_what_it_names(home):
    """A job handed one location must not also get the rest of the set."""
    soul = home / DENIED["identity file"]
    venv = home / DENIED["interpreter"]
    await FS.call_tool("Read", {"file_path": str(venv)})
    with PP.allow_protected_writes("test: one location only", paths=["~/lloyd/.venvs"]):
        _refused(await FS.call_tool("Write", {"file_path": str(soul),
                                              "content": "OVERWRITTEN"}), soul)
        assert (await FS.call_tool("Write", {"file_path": str(venv),
                                             "content": "ok"})).is_error is False


@pytest.mark.parametrize("extra", [
    {"_protected": False},
    {"protected": False},
    {"authorized": True},
    {"bypass_protected_path": True},
    {"grant_protected_writes": "nightly"},
    {"force": True},
    {"code": "PROTECTED_PATH"},
    {"_session_id": "20260921_060000_autocode_nightly"},
], ids=lambda d: next(iter(d)))
async def test_no_tool_argument_lifts_the_denial(home, extra):
    """No top-level schema sets `additionalProperties: false` —
    `test_mcp_layer.py::test_no_toplevel_additional_properties_false` keeps
    that true because the harness injects keys beside the model's arguments.
    So a model *can* send any key it likes: none of them is read here, and the
    outcome never depends on one."""
    target = home / DENIED["identity file"]
    args = {"file_path": str(target), "content": "OVERWRITTEN", **extra}
    _refused(await FS.call_tool("Write", args), target)


async def test_a_spoofed_session_argument_changes_nothing_at_the_aggregator(home):
    """`_session_id` in the arguments is stripped and replaced by the harness
    `_meta` value (`main._bound_session_id`), so neither half of that pair is
    a lever — the identity a write is attributed to is not a permission."""
    target = home / DENIED["identity file"]
    res = await M.call_tool("Write", {"file_path": str(target), "content": "OVERWRITTEN",
                                      "_session_id": "20260921_060000_autocode"},
                            {"lloyd/session_id": SID})
    assert res.is_error is True, _text(res)
    assert "PROTECTED_PATH" in _text(res)
    assert target.read_text() == ORIGINAL


# ── clause 5: everything outside the set behaves exactly as today ───────────

async def test_a_new_vault_note_still_lands(home):
    p = home / "obsidian" / "projects" / "new-note.md"
    res = await FS.call_tool("Write", {"file_path": str(p), "content": "# note\n"})
    assert res.is_error is False, _text(res)
    assert p.read_text() == "# note\n"


async def test_the_loaded_memory_files_beside_the_denied_one_still_land(home):
    """The 06:00 knowledge-write job's lane, pinned as it stands today:
    `lloyd/USER.md` and `lloyd/MEMORY.md` are NOT in the set (that membership
    is the open human decision #1049 leaves behind), and they sit in the same
    directory as the file that is."""
    for name in ("USER.md", "MEMORY.md"):
        p = home / "obsidian" / "lloyd" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        res = await FS.call_tool("Write", {"file_path": str(p), "content": "entry\n"})
        assert res.is_error is False, f"{name}: {_text(res)}"
        assert p.read_text() == "entry\n"


async def test_a_rounds_own_worktree_copy_of_a_denied_directory_name_still_lands(home):
    """The realpath-first rule's whole reason for existing. A round's tree
    holds `agent-services/…`; a suffix match would refuse the round the file
    it was opened to change."""
    f = (home / ".local" / "state" / "lloyd-automod" / "rounds" / "SM_20260921_000000"
         / "agent-services" / "supervisor" / "conf.d" / "agent-backend.conf")
    res = await FS.call_tool("Write", {"file_path": str(f), "content": "unit\n"})
    assert res.is_error is False, _text(res)
    assert f.read_text() == "unit\n"


async def test_a_sibling_named_like_an_entry_still_lands(home):
    """`/…/agent-services-extra` is not under `/…/agent-services`: matching is
    by path component, not by string prefix."""
    f = home / "lloyd" / "agent-services-extra" / "readme.md"
    res = await FS.call_tool("Write", {"file_path": str(f), "content": "hi\n"})
    assert res.is_error is False, _text(res)
    assert f.read_text() == "hi\n"


async def test_the_vault_write_lane_still_writes_the_file_the_fs_lane_refuses(
        home, monkeypatch, tmp_path):
    """`vault_write` is a separate lane with its own root check
    (`agent_mcp/vault.py`), and the nightly job that lands prompt surfaces
    goes through it. The deny-set must not reach it."""
    from agent_mcp import vault as V
    # The OKF taxonomy guard is orthogonal to path protection and imports
    # `scripts.vault`, which demands the vault-derived store — absent in a
    # round worktree. Pass it through: this test is about which lane a write
    # travels, not about note front matter.
    monkeypatch.setattr(V, "_guard_knowledge_type",
                        lambda path, content: (content, None, None))
    vault_root = home / "obsidian"
    monkeypatch.setattr(V, "VAULT", vault_root)
    audit_dir = tmp_path / "audit"
    monkeypatch.setattr(V, "AUDIT_LOG_DIR", audit_dir)
    monkeypatch.setattr(V, "AUDIT_LOG_FILE", audit_dir / "writes.jsonl")
    res = V._vault_write({"path": "lloyd/SOUL.md", "content": "SANCTIONED WRITE"})
    assert res.get("success") is True, res
    assert (vault_root / "lloyd" / "SOUL.md").read_text() == "SANCTIONED WRITE"


# ── the seam: the refusal arrives at the top of the aggregator ───────────────

async def test_the_refusal_comes_back_over_the_aggregator_with_is_error(home):
    """`agent_mcp.main.call_tool` is what every real caller crosses: the
    session id arrives in `_meta`, the contextvar is bound there, and the
    module handler is dispatched from a table. The refusal has to survive that
    trip as an error result, not as a success whose text happens to be JSON."""
    target = home / DENIED["identity file"]
    await M.list_tools()
    res = await M.call_tool("Write", {"file_path": str(target), "content": "OVERWRITTEN"},
                            {"lloyd/session_id": SID})
    assert type(res).__name__ == "CallToolResult"
    assert res.is_error is True, _text(res)
    assert _code(res) == "PROTECTED_PATH", _text(res)
    assert target.read_text() == ORIGINAL


async def test_the_write_still_lands_over_the_aggregator_once_the_grant_is_given(home):
    """The other side of the seam: a grant taken by in-process code around a
    real `main.call_tool` trip applies, so the property is a permission model
    and not just a hard refusal."""
    target = home / DENIED["service unit"]
    await M.list_tools()
    await M.call_tool("Read", {"file_path": str(target)}, {"lloyd/session_id": SID})
    with PP.allow_protected_writes("test: aggregator lane"):
        res = await M.call_tool("Write", {"file_path": str(target), "content": "unit\n"},
                                {"lloyd/session_id": SID})
    assert res.is_error is False, _text(res)
    assert target.read_text() == "unit\n"


def test_the_probe_from_the_triage_record_reproduces_only_the_refusal(home):
    """The acceptance check, in-process: Read, Write, Edit, read the bytes.
    Written so it *fails* on the pre-fix tree — running the same three calls
    at HEAD `617f708b` left 'CLOBBERED' in the file."""
    target = home / DENIED["identity file"]

    async def run():
        await FS.call_tool("Read", {"file_path": str(target)})
        w = await FS.call_tool("Write", {"file_path": str(target), "content": "CLOBBERED"})
        e = await FS.call_tool("Edit", {"file_path": str(target), "old_string": "ORIGINAL",
                                        "new_string": "REPLACED"})
        return w, e

    w, e = asyncio.run(run())
    assert _code(w) == "PROTECTED_PATH"
    assert _code(e) == "PROTECTED_PATH"
    assert target.read_text() == ORIGINAL
