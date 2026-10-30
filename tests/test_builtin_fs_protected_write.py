"""The Write/Edit lane refuses the write deny-set, and now so does every other
lane that reaches the same files.

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
identity file; `vault_write` asks the same predicate since #1757 and lands those
neighbours while refusing the one file. A predicate that refused any of those
would not be a fix, it would be an outage.

Since #1620 the Bash lane asks the same predicate of what a command writes, so
the last two tests here run `check_bash_command` as well as `Write`: one grant
contextvar, three lanes (fs, Bash, vault), and the allow half above holds on every
one of them.
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
from app.harness.safety import check_bash_command  # noqa: E402

SID = "20260921_1049_test"
ORIGINAL = "ORIGINAL IDENTITY FILE"

#: Where the interpreter the venvs name actually lives: the uv store, spelled
#: relative to a scratch `$HOME`. Every `.venvs/*/pyvenv.cfg` on this box names this
#: layout (`home = …/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin`) and
#: `.venvs/lloyd/bin/python` is a symlink into it, which is the hole #1741 closes —
#: the `~/lloyd/.venvs` entry named "the interpreter the lloyd services run on" while
#: realpath-first judged a write to that interpreter by the store it points at,
#: outside the set. The fifth entry is the store, so this is the path it covers.
UV_STORE_BIN = ".local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin"

#: One existing file per deny-set entry, vault-relative to the scratch home — five
#: since #1741. The `interpreter` key is the name a person types, and on this box it
#: is not a file but a link into the fifth entry, which the `home` fixture below
#: writes as a plain file: `linked_home` restores the link, and the difference is
#: what that node is for.
DENIED = {
    "identity file": "obsidian/lloyd/SOUL.md",
    "credential tree": ".openclaw/config.json",
    "service unit": "lloyd/agent-services/supervisor/conf.d/agent-backend.conf",
    "interpreter": "lloyd/.venvs/lloyd/bin/python",
    "uv store binary": UV_STORE_BIN + "/python3.12",
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


# ── clause 4: nothing can lift it, in-process or from the turn ──────────────
#
# #3563 inverts the two allow-half proofs this file was written with, and the
# inversion is the whole of clause 2:
#   `test_the_same_arguments_are_refused_until_code_lifts_them`
#     -> `test_the_identity_file_is_refused_and_there_is_no_call_that_changes_it`
#   `test_the_vault_write_lane_lands_that_file_under_a_grant`
#     -> `test_the_vault_write_lane_refuses_that_file_for_want_of_a_grant`
# Both kept their fixture, their path and their byte assertions, so the diff shows
# the expected outcome flipping rather than a node disappearing. The nodes that
# exercised the `paths=[…]` narrowing are rewritten the same way — for the Bash
# lane they live in `tests/test_bash_write_guard.py`.

async def test_the_identity_file_is_refused_and_there_is_no_call_that_changes_it(home):
    """#3563 inverts this node. It used to read: identical arguments, different
    outcome, the difference being one in-process lift call — which is how a test
    suite can hold an allow-half proof while the mechanism it proves is dead code:
    the only caller of that call was the test itself.

    Now the identical-arguments pair is across the two things that exist: the
    turn's own call, and the same call again after asking whether the module has
    any way to change the answer. Both refuse, the bytes never move, and the
    module exposes no name ending in `protected_writes` — asked as a suffix
    because a renamed re-add would satisfy a check written against the four names
    this item deleted and would not satisfy this one."""
    target = home / DENIED["identity file"]
    args = {"file_path": str(target), "content": "GRANTED WRITE"}
    _refused(await FS.call_tool("Write", args), target)
    lifts = sorted(n for n in dir(PP) if n.endswith("protected_writes"))
    assert not lifts, f"an in-process write lift came back on the module: {lifts}"
    _refused(await FS.call_tool("Write", args), target)
    assert target.read_text() == ORIGINAL, "the bytes moved on a refused call"


async def test_the_refusal_survives_the_thread_boundary_the_handler_runs_on(home):
    """The seam this node used to prove in the other direction.

    `call_tool` hands the handler to `asyncio.to_thread`, so the write decision is
    made on a worker thread while the caller's context lives on the loop — which
    is exactly why the deleted mechanism was a `ContextVar` and not a module
    global: `to_thread` copies the context, so a lift taken on the loop reached the
    executor and the write went through there. That boundary is now the thing worth
    pinning on its own: the answer cannot depend on which thread asks, because
    `write_deny_reason` takes one argument and there is no ambient state for the
    copy to carry. Both deny-set locations are refused on the worker thread and the
    bytes are untouched on the loop."""
    for key in ("credential tree", "identity file"):
        target = home / DENIED[key]
        res = await FS.call_tool("Write", {"file_path": str(target), "content": "ok"})
        assert res.is_error is True, f"{key} was written on the worker thread"
        assert target.read_text() == ORIGINAL, f"{key}'s bytes changed anyway"


async def test_two_deny_set_locations_are_refused_in_the_same_call_sequence(home):
    """What the removed `paths=[…]` argument used to be tested against: one
    location granted and its sibling refused, so a job handed one location could
    not also get the rest of the set. With no location-grantable state, the same
    sequence refuses both — which is the narrowing's property with its useful half
    kept: no request, in any order, ever opens one entry and not another."""
    soul = home / DENIED["identity file"]
    venv = home / DENIED["interpreter"]
    for path in (soul, venv):
        _refused(await FS.call_tool("Write", {"file_path": str(path),
                                              "content": "OVERWRITTEN"}), path)
    assert soul.read_text() == ORIGINAL and venv.read_text() == ORIGINAL


# ── #1741: the fifth entry, on the lane where it is not a file ──────────────


@pytest.fixture
def linked_home(tmp_path, monkeypatch):
    """`home`, with the venv interpreter restored to what it is on the box: a link.

    The distinction is the whole of #1741. As a plain file, `.venvs/lloyd/bin/python`
    is refused by `~/lloyd/.venvs` and a grant naming the uv store changes nothing,
    so every assertion below would hold at the base commit. As a link, the write is
    judged by the store — refused by the fifth entry alone, and liftable by a grant
    naming the fifth entry alone.
    """
    h = tmp_path / "home"
    for rel in DENIED.values():
        p = h / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(ORIGINAL)
    binary = h / UV_STORE_BIN / "python3.12"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(ORIGINAL)
    link = h / "lloyd" / ".venvs" / "lloyd" / "bin" / "python"
    link.unlink()
    link.symlink_to(binary)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    return h


async def test_the_store_link_stays_refused_on_both_lanes_with_nothing_to_narrow(linked_home):
    """#3563: the fifth entry's own refusal, where only the store covers it.

    The distinction #1741 set up still decides this node. As a link,
    `.venvs/lloyd/bin/python` is outside `~/lloyd/.venvs` and inside
    `~/.local/share/uv/python` by realpath alone — so the fifth entry is the only
    thing refusing it, and the grant that used to be able to open it by naming the
    store is gone. Both lanes, and the Bash lane's credential write beside it,
    stay refused, which is what the narrowing used to be differentiated against."""
    link = linked_home / DENIED["interpreter"]
    binary = linked_home / UV_STORE_BIN / "python3.12"
    assert link.is_symlink() and link.resolve() == binary.resolve()
    redirect = "echo x > ~/.openclaw/x"
    args = {"file_path": str(link), "content": "GRANTED WRITE"}
    _refused(await FS.call_tool("Write", args), binary)
    assert "uv-managed" in _json(await FS.call_tool("Write", args))["error"]
    assert check_bash_command(redirect, str(linked_home / "lloyd"),
                              at_dispatch=True, session_id=SID) is not None, redirect
    _refused(await FS.call_tool("Write", args), binary)
    assert binary.read_text() == ORIGINAL, "a refused write into the store changed the interpreter"
    assert not (linked_home / ".openclaw" / "x").exists()
def test_the_bash_lane_and_the_file_lane_refuse_the_same_two_writes(home):
    """One predicate, two lanes, and now only one state to be in (#1620, #3563).

    This node proved that a lift naming `~/lloyd/agent-services` opened the `tee`
    and left `> ~/.openclaw/x` refused — the lanes agreeing because they shared a
    switch. With the switch deleted the agreement is stronger and cheaper to
    state: both spellings are refused here and on `Write`, so there is no pair of
    commands on which the two lanes could report differently."""
    tee = ("echo unit | tee -a ~/lloyd/agent-services/supervisor/conf.d"
           "/agent-backend.conf")
    redirect = "echo x > ~/.openclaw/credentials.json"
    for cmd in (tee, redirect):
        match = check_bash_command(cmd, str(home / "lloyd"), at_dispatch=True,
                                   session_id=SID)
        assert match is not None, cmd
        assert match[0].startswith("protected write: "), match


# The sixth case is spelled after the mechanism #3563 deleted, on purpose and not
# as residue: a turn that reads the old doc text and sends the old name as an
# argument must get the same refusal as one that invents a name. It is the only
# remaining mention of that name anywhere outside the sweep that keeps it out of
# the shipped tree, and `tests/test_protected_write_exemption_docs.py` is what
# stops the name returning to `app/`, `agent_mcp/` or `architecture/`.
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


def _vault_lane(home, monkeypatch, tmp_path):
    """Point `agent_mcp.vault` at the scratch vault, off the change ledger.

    Nothing else is stubbed. In particular the two OKF guards are left wired:
    both short-circuit on `path.startswith("knowledge/")`, so a `lloyd/` path
    never reaches them, and stubbing them here (as this lane's test used to)
    implied the type guard would otherwise interfere — it would not, and a
    reader could take that as "the real lane is blocked anyway".
    """
    from agent_mcp import vault as V
    vault_root = home / "obsidian"
    monkeypatch.setattr(V, "VAULT", vault_root)
    audit_dir = tmp_path / "audit"
    monkeypatch.setattr(V, "AUDIT_LOG_DIR", audit_dir)
    monkeypatch.setattr(V, "AUDIT_LOG_FILE", audit_dir / "writes.jsonl")
    return V, vault_root


async def test_the_vault_write_lane_refuses_the_file_the_fs_lane_refuses(
        home, monkeypatch, tmp_path):
    """#1757, clause 1: the third lane onto the identity file asks the deny-set.

    `vault_write` used to be the one lane that did not: `_vault_write` gated a
    path on normalisation, two `knowledge/`-only OKF guards, containment under
    `VAULT` and `memory_write_error` (which maps only MEMORY.md/USER.md/
    `topics/<slug>`), so `vault_write(path="lloyd/SOUL.md")` landed the identity
    file from an ordinary turn — the write the fs lane refuses with
    `PROTECTED_PATH` and whose refusal text pointed here.

    The blanket exemption was pinned on the ground that the nightly prompt-landing
    job walks this lane. It does not: `scripts/autoresearch/promote.py::apply_overlay`
    writes SOUL/MEMORY/USER with `shutil.copy2` after `check_contract`, and
    `automod_vault_land` is its own validated lane. No scheduled job is changed by
    this refusal.
    """
    V, vault_root = _vault_lane(home, monkeypatch, tmp_path)
    target = vault_root / "lloyd" / "SOUL.md"
    res = V._vault_write({"path": "lloyd/SOUL.md", "content": "UNSANCTIONED WRITE"})
    assert res.get("success") is not True, (
        f"the vault lane landed the identity file with no grant: {res}")
    assert res.get("code") == "PROTECTED_PATH", res
    assert "protected" in res["error"] and "automod_vault_land" in res["error"], res
    assert target.read_text() == ORIGINAL, (
        "the refusal must leave the identity file's bytes exactly as they were")


async def test_the_vault_write_lane_refuses_that_file_for_want_of_a_grant(
        home, monkeypatch, tmp_path):
    """#1757's clause-2 allow half, inverted by #3563: the lane refuses, always.

    Same call, same path. What used to make it land was an in-process grant
    narrowed to that path — the mechanism this item deleted after reading all
    eight commits that ever named it and finding the test itself to be its only
    caller. So the lane is now a pure denial, and the route the refusal text
    offers (`automod_vault_land`) is exempt by never coming through here:
    `scripts/automod/vault_round.py::land` validates the named path and commits it
    with a `git` subprocess. The node keeps its shape so the inversion is visible
    in the diff rather than resting on an absent function."""
    V, vault_root = _vault_lane(home, monkeypatch, tmp_path)
    target = vault_root / "lloyd" / "SOUL.md"
    res = V._vault_write({"path": "lloyd/SOUL.md", "content": "SANCTIONED WRITE"})
    assert res.get("code") == "PROTECTED_PATH", res
    assert "automod_vault_land" in res["error"], res
    assert target.read_text() == ORIGINAL, "the identity file was written by a refusal"


async def test_the_vault_write_denial_survives_the_aggregator_trip(
        home, monkeypatch, tmp_path):
    """The seam, for this lane: `main.call_tool` dispatches `vault_write` by
    module table, so the refusal has to arrive as an error result there and not
    as a success whose text happens to be JSON."""
    V, vault_root = _vault_lane(home, monkeypatch, tmp_path)
    await M.list_tools()
    res = await M.call_tool("vault_write",
                            {"path": "lloyd/SOUL.md", "content": "UNSANCTIONED WRITE"},
                            {"lloyd/session_id": SID})
    assert res.is_error is True, _text(res)
    assert _code(res) == "PROTECTED_PATH", _text(res)
    assert (vault_root / "lloyd" / "SOUL.md").read_text() == ORIGINAL


async def test_the_vault_lane_refuses_the_identity_file_and_lands_its_sibling(
        home, monkeypatch, tmp_path):
    """The pair the deleted narrowing used to be tested with, now in one node.

    A grant lifted for `~/lloyd/agent-services` was the old way to show that a
    permission on one location is not a permission on the identity file. Nothing
    is grantable now, so the pair that matters is the one a real round sees: the
    deny-set path in the same `lloyd/` folder is refused while `lloyd/MEMORY.md`,
    outside the set, lands. The second half is the item's own measurement — the
    loaded-memory writers never needed the lift, which is why deleting it broke
    nothing."""
    V, vault_root = _vault_lane(home, monkeypatch, tmp_path)
    res = V._vault_write({"path": "lloyd/SOUL.md", "content": "WRONG ROUTE"})
    assert res.get("code") == "PROTECTED_PATH", res
    assert (vault_root / "lloyd" / "SOUL.md").read_text() == ORIGINAL
    ok = V._vault_write({"path": "lloyd/MEMORY.md", "content": "# memory\n"})
    assert ok.get("success") is True, ok
    assert (vault_root / "lloyd" / "MEMORY.md").read_text() == "# memory\n"


async def test_the_vault_lane_allow_half_lands_the_file_beside_the_denied_one(
        home, monkeypatch, tmp_path):
    """The exemption narrowed to a grant must not become a freeze. `lloyd/MEMORY.md`
    sits in the same directory as the denied identity file and is the file the
    exemption was originally justified by (#1049: the 06:00 knowledge-write job
    lands USER.md/MEMORY.md, and that item closed with them staying *out* of the
    deny-set) — so it is the sharpest possible test that the guard refuses one file
    and not the directory. Clause 3's twin here; `tests/test_mcp_layer.py` pins the
    same thing over the aggregator."""
    V, vault_root = _vault_lane(home, monkeypatch, tmp_path)
    (vault_root / "lloyd" / "MEMORY.md").write_text("# old\n", encoding="utf-8")
    res = V._vault_write({"path": "lloyd/MEMORY.md", "content": "# new\n"})
    assert res.get("success") is True, res
    assert (vault_root / "lloyd" / "MEMORY.md").read_text(encoding="utf-8") == "# new\n"


# ── clause 4: the refusal text must not route a writer to a lane that refuses ─

async def test_the_refusal_no_longer_offers_the_vault_lane_as_the_way_through(home):
    """The protected lane's own text used to read "Land the change through the route
    that validates it — `vault_write` or `automod_vault_land` …", which sent a writer
    refused here straight to the lane that refused nothing. #1757 closes that lane,
    so the sentence may only offer routes that will land the write.

    Asserted positionally rather than by absence: naming `vault_write` *as also
    refusing* is worth the tokens, because a writer that has just been refused and
    told nothing else goes and tries it. What may not happen is the offer — and the
    offered routes are the ones after "…validates it".
    """
    target = home / DENIED["identity file"]
    res = await FS.call_tool("Write", {"file_path": str(target), "content": "x"})
    err = _json(res)["error"]
    offered = err.split("validates it", 1)[1]
    assert "vault_write" not in offered, (
        f"the refusal still offers the vault lane as a route through: {offered}")
    assert "automod_vault_land" in offered, (
        f"the refusal must still name a route that validates the write: {offered}")
    assert err.count("vault_write") == 1, (
        f"`vault_write` may be named once, as also refusing: {err}")
    assert err.index("vault_write") < err.index("validates it"), err


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


async def test_the_refusal_arrives_over_the_aggregator_with_no_grant_to_wait_for(home):
    """The seam, deny half — and the allow half it used to be paired with is gone.

    `main.call_tool` is the boundary every real tool call crosses. The node here
    before #3563 proved that a grant taken by in-process code around the trip
    applied, "so the property is a permission model and not just a hard refusal";
    the honest statement now is that on this lane the property *is* a hard
    refusal, and the only thing that can vary across the boundary is whether the
    refusal text routes the caller somewhere. It does, and the bytes do not
    move."""
    target = home / DENIED["service unit"]
    await M.list_tools()
    res = await M.call_tool("Write", {"file_path": str(target), "content": "unit\n"},
                            {"lloyd/session_id": SID})
    assert res.is_error is True, _text(res)
    assert _code(res) == "PROTECTED_PATH", _text(res)
    assert "automod_vault_land" in _text(res), _text(res)
    assert target.read_text() == ORIGINAL


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


# ── #2173 clause 2: the topic-slug bound reaches the generic writers ─────────
# `app/memory_ceiling.py` is the one entry point every writer asks, and since #2173
# it refuses a topic file whose stem is outside `TOPIC_SLUG_RE`. These nodes are the
# seam test for that: the guard's own answer is pinned in
# `tests/test_memory_index_cap.py`, and what is pinned *here* is that the refusal
# survives the trip out through `Write`, `Edit` and `vault_write` as a refusal, and
# that the two writes which must still land do land — a legal slug, and a shrinking
# repair to a topic file that is already over a bound. Without the second half this
# is a freeze with three doors.

#: The incident's stem: 52 characters, four over `TOPIC_SLUG_MAX_CHARS`.
SLUG_52 = "a-done-closure-over-a-refuted-premise-is-refuted-not"


@pytest.fixture
def topics_dir(home, monkeypatch):
    """The scratch vault's topic directory, with the guard pointed at it.

    `memory_ceiling.MEMORIES_DIR` is a module constant resolved at import from the
    real `$HOME`, so it does NOT move with the `home` fixture the way the deny-set
    does. Pointing it at the scratch vault is what lets these nodes exercise the
    real lanes against a real path; without it every scratch path answers "not a
    memory file" and the node passes for the wrong reason.
    """
    from app import memory_ceiling as MC
    monkeypatch.setattr(MC, "MEMORIES_DIR", home / "obsidian" / "lloyd")
    tdir = home / "obsidian" / "lloyd" / "memory"
    tdir.mkdir(parents=True, exist_ok=True)
    return tdir


async def test_the_write_and_edit_lanes_refuse_an_over_long_topic_slug(topics_dir):
    """`Write`/`Edit` are the lanes the knowledge-write skill routes a topic file
    through, and they take the ceiling from `memory_write_error` verbatim — so the
    slug bound arrives with no change to either handler."""
    target = topics_dir / f"{SLUG_52}.md"
    res = await FS.call_tool("Write", {"file_path": str(target),
                                       "content": "# detail\n"})
    assert res.is_error is True, _text(res)
    err = _json(res)["error"]
    assert "48-character topic-slug cap" in err, err
    assert not target.exists(), "a refused create must leave no file behind"

    target.write_text("- original detail, in full\n", encoding="utf-8")
    await FS.call_tool("Read", {"file_path": str(target)})
    res = await FS.call_tool("Edit", {"file_path": str(target),
                                      "old_string": "original", "new_string": "GROWN" * 40})
    assert res.is_error is True, _text(res)
    assert "48-character topic-slug cap" in _json(res)["error"], _text(res)
    assert target.read_text() == "- original detail, in full\n", \
        "a refused Edit must leave the bytes exactly as they were"


async def test_the_write_lane_still_lands_a_slug_at_the_cap_and_a_shrinking_repair(
        topics_dir):
    """The allow half, both shapes. A 48-character slug is legal and must land
    (a guard that overshoots by one is a guard the next writer routes around);
    a *shrinking* write to a file already over a bound — here the 32,768-byte
    topic ceiling — is still the repair route, which is the rule that keeps a
    ceiling from becoming a freeze, and it has to survive a name that is
    simultaneously over the slug cap."""
    ok = topics_dir / f"{'a' * 48}.md"
    res = await FS.call_tool("Write", {"file_path": str(ok), "content": "# detail\n"})
    assert res.is_error is False, _text(res)
    assert ok.read_text() == "# detail\n"

    big = topics_dir / "over-the-byte-ceiling.md"
    big.write_text("x" * (32_768 + 4_000), encoding="utf-8")
    await FS.call_tool("Read", {"file_path": str(big)})
    res = await FS.call_tool("Write", {"file_path": str(big), "content": "- trimmed\n"})
    assert res.is_error is False, _text(res)
    assert big.read_text() == "- trimmed\n"

    both = topics_dir / f"{SLUG_52}.md"
    both.write_text("- detail " + "x" * 200 + "\n", encoding="utf-8")
    await FS.call_tool("Read", {"file_path": str(both)})
    res = await FS.call_tool("Write", {"file_path": str(both), "content": "- trimmed\n"})
    assert res.is_error is False, _text(res)
    assert both.read_text() == "- trimmed\n"


async def test_the_vault_write_lane_refuses_the_same_slug_it_refuses_nothing_for(
        home, monkeypatch, tmp_path, topics_dir):
    """`vault_write(path="lloyd/memory/<slug>.md")` is the third lane onto the same
    directory, and #2173 closes its gap the same way #1757 closed the deny-set one:
    by asking the one predicate. A legal topic file still lands through it — the
    nightly job writes topic files on this lane every run."""
    V, vault_root = _vault_lane(home, monkeypatch, tmp_path)
    res = V._vault_write({"path": f"lloyd/memory/{SLUG_52}.md", "content": "# detail\n"})
    assert res.get("success") is not True, f"the vault lane landed it: {res}"
    assert "48-character topic-slug cap" in res.get("error", ""), res
    assert not (vault_root / "lloyd" / "memory" / f"{SLUG_52}.md").exists(), res

    res = V._vault_write({"path": "lloyd/memory/voice-loop.md", "content": "# detail\n"})
    assert res.get("success") is True, res
    assert (vault_root / "lloyd" / "memory" / "voice-loop.md").read_text() == "# detail\n"


async def test_the_slug_refusal_comes_back_over_the_aggregator_as_an_error(topics_dir):
    """The seam every real caller crosses: `main.call_tool` dispatches `Write` and
    `vault_write` by module table and wraps the handler's text, so the refusal has
    to arrive as an error result there rather than as a success whose text happens
    to be JSON — the failure shape that let #1049 and #1757 through."""
    await M.list_tools()
    target = topics_dir / f"{SLUG_52}.md"
    res = await M.call_tool("Write", {"file_path": str(target), "content": "# detail\n"},
                            {"lloyd/session_id": SID})
    assert res.is_error is True, _text(res)
    assert "48-character topic-slug cap" in _text(res), _text(res)
    assert not target.exists()
