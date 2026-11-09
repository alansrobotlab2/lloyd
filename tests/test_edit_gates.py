"""Read-before-edit and stale-file gates on Write/Edit.

The failure these exist for is not "the model edited a file it hadn't read"
— that one announces itself as `old_string not found`. It is the edit that
*succeeds*: the model Reads a file, something else rewrites it, the model's
old_string still matches, and the edit silently reverts the other writer.
Nothing in the transcript, the tool result or the logs says so.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from agent_mcp import _shared as SHARED
from agent_mcp import builtin_fs as FS

SID = "20260908_120000_test"


@pytest.fixture(autouse=True)
def _clean():
    FS.reset_read_records()
    yield
    FS.reset_read_records()


@pytest.fixture
def bound(monkeypatch):
    """Bind a session id, as main.call_tool does for a real tool call."""
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    return SID


@pytest.fixture
def f(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("alpha\nbeta\n")
    return p


def _text(res):
    return res.content[0].text


def _err(res):
    return json.loads(_text(res)).get("error", "")


def _touch_later(p: Path, content: str):
    """Rewrite `p` with a stat that is unambiguously different.

    mtime_ns alone can collide inside one filesystem tick, and the content
    here is the same length in some cases, so the test would pass or fail on
    timer granularity.
    """
    p.write_text(content)
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


# ── the gate ────────────────────────────────────────────────────────────────

async def test_edit_without_a_read_is_refused(bound, f):
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error
    assert "has not been Read in this session" in _err(res)
    assert "Read it first" in _err(res)
    assert f.read_text() == "alpha\nbeta\n", "the file must not have been touched"


async def test_read_then_edit_succeeds(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False
    assert f.read_text() == "ALPHA\nbeta\n"


async def test_a_partial_read_is_enough(bound, f):
    await FS.call_tool("Read", {"file_path": str(f), "offset": 2, "limit": 1})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "beta",
                                      "new_string": "BETA"})
    assert res.is_error is False


async def test_an_out_of_band_write_makes_the_edit_stale(bound, f):
    """This is the silent-revert case: old_string still matches."""
    await FS.call_tool("Read", {"file_path": str(f)})
    _touch_later(f, "alpha\nbeta\ngamma\n")     # e.g. `sed -i` from Bash
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error
    assert "changed on disk since you last Read it" in _err(res)
    assert f.read_text() == "alpha\nbeta\ngamma\n", "the other writer survived"


async def test_re_reading_clears_the_staleness(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    _touch_later(f, "alpha\nbeta\ngamma\n")
    assert (await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                        "new_string": "ALPHA"})).is_error
    await FS.call_tool("Read", {"file_path": str(f)})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False


async def test_consecutive_edits_need_only_one_read(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    assert not (await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                            "new_string": "ALPHA"})).is_error
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "beta",
                                      "new_string": "BETA"})
    assert res.is_error is False
    assert f.read_text() == "ALPHA\nBETA\n"


async def test_write_then_edit_needs_no_read(bound, tmp_path):
    p = tmp_path / "new.py"
    assert not (await FS.call_tool("Write", {"file_path": str(p),
                                             "content": "one\n"})).is_error
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "one",
                                      "new_string": "two"})
    assert res.is_error is False


# ── Write ───────────────────────────────────────────────────────────────────

async def test_write_creates_a_new_file_with_no_read(bound, tmp_path):
    p = tmp_path / "sub" / "new.txt"
    res = await FS.call_tool("Write", {"file_path": str(p), "content": "hi"})
    assert res.is_error is False
    assert p.read_text() == "hi"


async def test_write_over_an_unread_existing_file_is_refused(bound, f):
    res = await FS.call_tool("Write", {"file_path": str(f), "content": "clobbered"})
    assert res.is_error
    assert "already exists and has not been Read" in _err(res)
    assert f.read_text() == "alpha\nbeta\n"


async def test_write_over_a_stale_file_is_refused(bound, f):
    await FS.call_tool("Read", {"file_path": str(f)})
    _touch_later(f, "somebody else\n")
    res = await FS.call_tool("Write", {"file_path": str(f), "content": "clobbered"})
    assert res.is_error
    assert "changed on disk since you last Read it" in _err(res)


async def test_write_through_a_dangling_symlink_is_a_create(bound, tmp_path):
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert not target.exists()
    res = await FS.call_tool("Write", {"file_path": str(link), "content": "made"})
    assert res.is_error is False
    assert target.read_text() == "made"


# ── the governed §2e pattern pointers (#3626) ────────────────────────────────
#
# Both `/home/alansrobotlab/lloyd-data/_pipeline/reflection/{tool,conversation}
# -patterns-latest.md` were symlinks for three days after 2026-10-28 08:09 UTC.
# `_write_target` above is built to follow a link, and §2e writes the pointer in
# place, so every `Write` that week landed its bytes in the dated archive copy the
# link named — the copy whose whole job is to preserve the previous night — and left
# the link standing. Nothing could see it: `find … ! -type l` skips the entry,
# `date -u -r` prints the target's date, and `Path.stat()` in the health report
# follows the link too. The refusal pinned below is scoped to those names under that
# root, because the same lane's symlink-following is deliberate everywhere else
# (`test_write_through_a_dangling_symlink_is_a_create` above) and a blanket rule
# would undo it.

@pytest.fixture
def governed(tmp_path, monkeypatch):
    """A reflection root the write lane treats as the governed one.

    `app.paths.PIPELINE_DIR` is patched rather than the guard's own function, so the
    derivation from the data root is part of what runs: #3626's other finding is a §2e
    cycle (2026-10-21, vault commit `7c0fd930`) that wrote its pointers into
    `~/obsidian` instead, and a refusal hard-wired to a literal spelling would guard a
    root nothing writes to while leaving that one open.
    """
    root = tmp_path / "_pipeline" / "reflection"
    root.mkdir(parents=True)
    monkeypatch.setattr("app.paths.PIPELINE_DIR", tmp_path / "_pipeline")
    return root


def _pointer(root: Path, stem: str = "tool-patterns") -> Path:
    """The governed pointer path for `stem` — `<stem>-latest.md`, the real shape."""
    return root / f"{stem}-latest.md"


def _linked_pointer(root: Path, stem: str, *, missing: bool = False) -> tuple[Path, Path]:
    """A governed pointer replaced by a symlink to its dated archive copy.

    `missing=True` leaves the archive uncreated, which is the dangling case: the write
    lane's own docstring calls a write through such a link a create, and on a governed
    pointer a create is exactly the hole — the link stands, a new file appears behind
    it, and the pointer is still not a regular file for the next cycle.
    """
    archive = root / f"{stem}-latest-2026-10-28-0809.md"
    if not missing:
        archive.write_text("the previous night\n", encoding="utf-8")
    pointer = _pointer(root, stem)
    pointer.symlink_to(archive.name)
    return pointer, archive


async def test_a_write_to_a_symlinked_governed_pointer_is_refused(bound, governed):
    """Clause 1: the refusal names the type, and the link and its target survive it.

    The `Read` comes first because the real §2e run reads the pointer before writing
    it: being allowed to overwrite a file you have looked at is not permission to
    write through it into something else. The `iterdir()` line is the failure mode the
    guard exists for — this lane replaces by `os.replace`, which renames over the NAME
    and would leave a regular file where the link stood while the archive behind it
    kept the old bytes, so the pointer would read "repaired" and the cycle's bytes
    would still be nowhere.
    """
    pointer, archive = _linked_pointer(governed, "tool-patterns")
    await FS.call_tool("Read", {"file_path": str(pointer)})
    res = await FS.call_tool("Write", {"file_path": str(pointer),
                                       "content": "tonight's patterns\n"})
    assert res.is_error, _text(res)
    err = _err(res)
    assert "symbolic link" in err, err
    assert "regular file" in err, err
    assert "#3626" in err, err
    assert os.path.islink(str(pointer)), "the refusal left the link standing as something else"
    assert os.readlink(str(pointer)) == archive.name
    assert archive.read_bytes() == b"the previous night\n", "the refusal wrote through the link"
    assert sorted(p.name for p in governed.iterdir()) == [
        "tool-patterns-latest-2026-10-28-0809.md", "tool-patterns-latest.md"], (
        "the refusal created a file beside the pointer")


async def test_a_regular_governed_pointer_still_writes_exactly_as_before(bound, governed):
    """Clause 1's other half: the guard must not stall the nightly write it governs.

    A §2e cycle that repaired its pointer and came back to write it is the normal
    path; if the refusal also fired here, the nightly pattern corpus stops being
    written and the guard is the outage.
    """
    pointer = _pointer(governed)
    pointer.write_text("the previous night\n", encoding="utf-8")
    await FS.call_tool("Read", {"file_path": str(pointer)})
    res = await FS.call_tool("Write", {"file_path": str(pointer),
                                       "content": "tonight's patterns\n"})
    assert res.is_error is False, _text(res)
    assert pointer.read_text() == "tonight's patterns\n"
    assert not os.path.islink(str(pointer))


async def test_a_dangling_link_on_a_governed_pointer_is_refused_not_a_create(bound, governed):
    """Clause 1 again, on the branch that skips the clobber gate: the create branch.

    `Write` treats a dangling link as a file that does not exist, so on this path the
    old code never even reached a refusal — it wrote the link's target into existence
    and left the pointer a link, which is how the archive history gets eaten a second
    time. `existed=False` is also why the guard runs ahead of the create early-return.
    """
    pointer, archive = _linked_pointer(governed, "conversation-patterns", missing=True)
    res = await FS.call_tool("Write", {"file_path": str(pointer),
                                      "content": "tonight's patterns\n"})
    assert res.is_error, _text(res)
    assert "symbolic link" in _err(res)
    assert os.path.islink(str(pointer))
    assert not archive.exists(), "the write created the link's target and left the link"


async def test_the_pointer_type_refusal_does_not_wait_for_the_session_gate(governed):
    """The rule is about where bytes may land, so an unbound caller meets it too.

    No `bound` fixture: `gate_on` is off, the Read-before-Write gate never runs, and a
    refusal written after that switch would be a rule only some writers follow — the
    same shape as the deny-set's placement at the top of `_gate_check`.
    """
    pointer, archive = _linked_pointer(governed, "tool-patterns")
    res = await FS.call_tool("Write", {"file_path": str(pointer),
                                      "content": "tonight's patterns\n"})
    assert res.is_error, _text(res)
    assert "symbolic link" in _err(res)
    assert archive.read_bytes() == b"the previous night\n"


async def test_a_directory_wearing_a_governed_pointer_name_is_refused_by_its_type(bound,
                                                                                 governed):
    """The refusal names the type it found, not a hard-coded guess at one.

    A directory cannot be `Read`, so without a real type read here the answer would be
    the clobber gate's "already exists and has not been Read" — an error either way,
    and only this one says what is wrong with the path.
    """
    pointer = _pointer(governed)
    pointer.mkdir()
    res = await FS.call_tool("Write", {"file_path": str(pointer),
                                      "content": "tonight's patterns\n"})
    assert res.is_error, _text(res)
    err = _err(res)
    assert "directory" in err, err
    assert "symbolic link" not in err, err
    assert "has not been Read" not in err, err


async def test_an_edit_of_a_symlinked_governed_pointer_is_refused_too(bound, governed):
    """The same rule on the second write surface, which is where a partial guard ends.

    §2e's own instruction is to add a dated section to the pattern files, and the house
    rule for an existing file is `Edit`; `_edit` resolves through `_write_target` just as
    `Write` does, so a refusal wired only into `_write` would let a targeted change
    replace the archive copy's body and leave the pointer a link.
    """
    pointer, archive = _linked_pointer(governed, "tool-patterns")
    await FS.call_tool("Read", {"file_path": str(pointer)})
    res = await FS.call_tool("Edit", {"file_path": str(pointer),
                                      "old_string": "previous", "new_string": "tonight"})
    assert res.is_error, _text(res)
    err = _err(res)
    assert err.startswith("Edit refused"), err
    assert "symbolic link" in err and "#3626" in err, err
    assert archive.read_bytes() == b"the previous night\n"
    assert os.path.islink(str(pointer))


async def test_a_governed_pointer_name_outside_the_governed_root_keeps_its_behaviour(bound,
                                                                                     tmp_path):
    """Clause 2: the guard is scoped to the governed root, not to the name.

    This is the 2026-10-21 shape — a `conversation-patterns-latest.md` sitting under
    the vault, written by the §2e cycle that mis-rooted its whole output there (vault
    commit `7c0fd930`). Deleting that file is #3626 clause 4's business; refusing to
    WRITE a governed-looking name anywhere on the box is not, and would make this lane
    refuse ordinary files. The bytes still travel through the link, as
    `test_write_through_a_dangling_symlink_is_a_create` above pins for every other
    path.
    """
    stray_dir = tmp_path / "obsidian" / "projects" / "research"
    stray_dir.mkdir(parents=True)
    archive = stray_dir / "conversation-patterns-latest-2026-10-21-0820.md"
    archive.write_text("the 2026-10-21 cycle\n", encoding="utf-8")
    pointer = stray_dir / "conversation-patterns-latest.md"
    pointer.symlink_to(archive.name)
    await FS.call_tool("Read", {"file_path": str(pointer)})
    res = await FS.call_tool("Write", {"file_path": str(pointer), "content": "rewritten\n"})
    assert res.is_error is False, _text(res)
    assert archive.read_text() == "rewritten\n"
    assert os.path.islink(str(pointer)), "the link must stay a link on an ordinary path"


async def test_a_dated_archive_copy_inside_the_governed_root_is_not_a_pointer(bound,
                                                                              governed):
    """Clause 2 again, on the other axis: shape, not directory.

    §2e's archive copies are `<stem>-latest-<YYYY-MM-DD>-<HHMM>.md` and live in the same
    directory as the pointers. A guard keyed on "any file under the reflection root"
    would refuse a hand restore of an archive the 10-29 lesson tells the run to write
    back by hand (`cp -L --remove-destination`), so the shape carries as much of the
    scope as the root does.
    """
    older = governed / "tool-patterns-latest-2026-10-27-0800.md"
    older.write_text("two nights back\n", encoding="utf-8")
    copy = governed / "tool-patterns-latest-2026-10-28-0809.md"
    copy.symlink_to(older.name)
    await FS.call_tool("Read", {"file_path": str(copy)})
    res = await FS.call_tool("Write", {"file_path": str(copy), "content": "restored\n"})
    assert res.is_error is False, _text(res)
    assert older.read_text() == "restored\n"
    assert os.path.islink(str(copy))


# ── a governed §2e stem aimed at the VAULT instead of the data root (#4004) ────
#
# On 2026-11-08 the nightly knowledge-write cycle read `steps-2b-2f.md`, which mandates
# `Write(file_path="/home/alansrobotlab/lloyd-data/_pipeline/reflection/
# tool-patterns-latest.md", …)`, and wrote BOTH governed pointers to
# `~/obsidian/projects/ai/patterns/` instead — creating that directory, with no refusal
# anywhere in the session. The pointer at the governed path did not advance that night
# (`…-2026-11-07-0907.md` jumps to `…-2026-11-09-0905.md`, no 11-08 archive for either
# stem), and the vault's own maintenance pass then filed the strays as knowledge notes
# (`370986f20`, `e9345ab48`). Fifth stray root for these stems after `memory/`
# (2026-10-15), `knowledge/conversation/` (2026-10-09) and `projects/ai/research/`
# (2026-10-21).

@pytest.fixture
def vault_and_data_root(monkeypatch, tmp_path):
    """A vault root and a data root, both under `tmp_path`, patched at the resolver.

    `app.paths.VAULT_ROOT` and `app.paths.PIPELINE_DIR` rather than anything in the guard,
    so the test crosses the same resolver the live write lane resolves through.
    """
    vault = tmp_path / "obsidian"
    reflection = tmp_path / "_pipeline" / "reflection"
    reflection.mkdir(parents=True)
    monkeypatch.setattr("app.paths.VAULT_ROOT", vault)
    monkeypatch.setattr("app.paths.PIPELINE_DIR", tmp_path / "_pipeline")
    return vault, reflection


async def test_a_governed_stem_written_under_the_vault_is_refused_and_named(bound,
                                                                           vault_and_data_root):
    """Clause 3, the create that the 2026-11-08 cycle actually attempted.

    The file does not exist and its directory does not either: this is exactly the shape
    `_gate_check` returns early on, which is why the guard sits ahead of it. Three things
    are asserted, because a refusal that refuses the right write for the wrong reason is
    how a rail becomes a mystery: the refusal, nothing on disk, and the ABSOLUTE reflection
    target it tells the caller to re-issue against.
    """
    vault, reflection = vault_and_data_root
    stray = vault / "projects" / "ai" / "patterns" / "tool-patterns-latest.md"
    res = await FS.call_tool("Write", {"file_path": str(stray), "content": "# patterns\n"})
    assert res.is_error, _text(res)
    msg = _err(res)
    assert "governed §2e pattern stem" in msg, msg
    assert str(reflection / "tool-patterns-latest.md") in msg, (
        f"the refusal must name the absolute target to re-issue: {msg}")
    assert not stray.exists() and not stray.parent.exists(), (
        "a refused write created the file or its directory")
    assert not (reflection / "tool-patterns-latest.md").exists(), (
        "the refusal is not a redirect: it writes nothing anywhere")


async def test_a_governed_stem_edited_under_the_vault_is_refused_too(bound,
                                                                     vault_and_data_root):
    """Both verbs, because a stray a previous cycle created is edited, not created.

    The file is on disk and Read first, so every other pre-write gate is satisfied — the
    only thing left standing between these bytes and the wrong root is this guard.
    """
    vault, reflection = vault_and_data_root
    stray = vault / "memory" / "conversation-patterns-latest.md"
    stray.parent.mkdir(parents=True)
    stray.write_text("the 2026-10-15 stray\n", encoding="utf-8")
    await FS.call_tool("Read", {"file_path": str(stray)})
    res = await FS.call_tool("Edit", {"file_path": str(stray),
                                      "old_string": "2026-10-15", "new_string": "2026-11-08"})
    assert res.is_error, _text(res)
    assert "governed §2e pattern stem" in _err(res)
    assert str(reflection / "conversation-patterns-latest.md") in _err(res)
    assert stray.read_text() == "the 2026-10-15 stray\n"


async def test_the_two_shielded_knowledge_notes_still_write_under_the_vault(bound,
                                                                           vault_and_data_root):
    """The scope trap: this guard is keyed on `PATTERN_STEMS`, not on the pointer SHAPE.

    `autonomous-systems-{failure,success}-patterns-latest.md` end in `-patterns-latest.md`
    like the governed pointers do, and live under the vault ON PURPOSE — the 2026-11-07
    owed-check ruling on #3751 keeps them as hand-maintained knowledge notes, and §2e names
    them as never a run's deletion targets. A guard keyed on the `_POINTER_LATEST` shape
    would have broken both, which is what this node is here to make expensive.
    """
    from scripts.reflection_archive import PATTERN_STEMS

    vault, _ = vault_and_data_root
    notes = vault / "projects" / "ai" / "tools"
    notes.mkdir(parents=True)
    for name in ("autonomous-systems-failure-patterns-latest.md",
                 "autonomous-systems-success-patterns-latest.md"):
        assert name[: -len(".md")] not in PATTERN_STEMS, (
            f"{name} would be governed by the guard's own derivation")
        p = notes / name
        res = await FS.call_tool("Write", {"file_path": str(p), "content": "note\n"})
        assert res.is_error is False, _text(res)
        assert p.read_text() == "note\n", f"{name} did not land"
        assert "governed §2e pattern stem" not in _text(res)


async def test_a_governed_stem_at_its_reflection_root_is_still_written(bound,
                                                                       vault_and_data_root):
    """The other side of the directory axis: inside the data root the same name is normal.

    Without this the guard's scope is unproven — a rule that refused the name everywhere
    would pass the two refusal nodes above and would then refuse the only write that is
    allowed, which is the §2e step itself.
    """
    _, reflection = vault_and_data_root
    pointer = reflection / "tool-patterns-latest.md"
    res = await FS.call_tool("Write", {"file_path": str(pointer), "content": "nightly\n"})
    assert res.is_error is False, _text(res)
    assert pointer.read_text() == "nightly\n"


async def test_a_governed_stem_outside_both_roots_is_left_alone(bound, vault_and_data_root):
    """Neither the vault nor the data root: none of this guard's business.

    A scratch checkout, another project's tree — the 2026-10-21 node above already covers a
    symlinked pointer there, and this covers the create, so the guard cannot have become a
    global ban on the two names.
    """
    vault, reflection = vault_and_data_root
    elsewhere = vault.parent / "elsewhere" / "notes" / "tool-patterns-latest.md"
    res = await FS.call_tool("Write", {"file_path": str(elsewhere), "content": "scratch\n"})
    assert res.is_error is False, _text(res)
    assert elsewhere.read_text() == "scratch\n"


# ── keys, sessions, kill switch ─────────────────────────────────────────────

async def test_a_symlink_and_its_target_share_one_record(bound, tmp_path):
    target = tmp_path / "real.py"
    target.write_text("x = 1\n")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    await FS.call_tool("Read", {"file_path": str(link)})
    res = await FS.call_tool("Edit", {"file_path": str(target), "old_string": "x = 1",
                                      "new_string": "x = 2"})
    assert res.is_error is False, _text(res)


async def test_two_sessions_do_not_share_reads(monkeypatch, f):
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-a")
    await FS.call_tool("Read", {"file_path": str(f)})
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-b")
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error and "has not been Read" in _err(res)


async def test_the_second_writer_of_two_sessions_is_told(monkeypatch, f):
    for sid in ("session-a", "session-b"):
        monkeypatch.setattr(FS, "get_bound_session", lambda s=sid: s)
        await FS.call_tool("Read", {"file_path": str(f)})
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-a")
    assert not (await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                            "new_string": "ALPHA"})).is_error
    monkeypatch.setattr(FS, "get_bound_session", lambda: "session-b")
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "beta",
                                      "new_string": "BETA"})
    assert res.is_error and "changed on disk" in _err(res)


async def test_no_bound_session_skips_the_gate(monkeypatch, f):
    """Unit tests and legacy callers dispatch with no aggregator context."""
    monkeypatch.setattr(FS, "get_bound_session", lambda: "")
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False


async def test_kill_switch_keeps_recording_but_stops_refusing(bound, f, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness", {**CONFIG.get("harness", {}),
                                            "edit_gates": {"enabled": False}})
    res = await FS.call_tool("Edit", {"file_path": str(f), "old_string": "alpha",
                                      "new_string": "ALPHA"})
    assert res.is_error is False
    assert FS._seen(SID, os.path.realpath(f)) is not None, \
        "records must keep accruing so flipping the switch back works at once"


async def test_records_are_bounded_per_session_and_overall(bound, tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_READ_PATHS_MAX", 3)
    monkeypatch.setattr(FS, "_READ_SESSIONS_MAX", 2)
    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.txt"
        p.write_text(str(i))
        paths.append(p)
        await FS.call_tool("Read", {"file_path": str(p)})
    assert len(FS._read_records[SID]) == 3
    assert FS._seen(SID, os.path.realpath(paths[0])) is None
    assert FS._seen(SID, os.path.realpath(paths[4])) is not None

    for sid in ("s1", "s2", "s3"):
        monkeypatch.setattr(FS, "get_bound_session", lambda s=sid: s)
        await FS.call_tool("Read", {"file_path": str(paths[0])})
    assert len(FS._read_records) == 2


# ── errors that used to escape ──────────────────────────────────────────────

async def test_editing_a_binary_file_is_a_normal_error(bound, tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\xff\xfe\x00\x01not utf8")
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "a",
                                      "new_string": "b"})
    assert res.is_error
    assert "not valid UTF-8" in _err(res)
    assert str(p) in _err(res)


async def test_a_missing_file_records_nothing(bound, tmp_path):
    res = await FS.call_tool("Read", {"file_path": str(tmp_path / "nope.txt")})
    assert res.is_error
    assert FS._read_records.get(SID) in (None, {})


async def test_reading_a_directory_records_nothing(bound, tmp_path):
    res = await FS.call_tool("Read", {"file_path": str(tmp_path)})
    assert res.is_error
    assert FS._read_records.get(SID) in (None, {})


# ── the not-found payload names a line ──────────────────────────────────────
#
# `old_string not found in file (must match exactly)` is 49 characters that say
# only "the bytes differ". Measured over the 285 occurrences of that one body in
# the 21 days before backlog #849 was filed, the model's next tool call after one
# was Bash 143 times, a Read or Grep of the same file 92 times, and the same
# unchanged `old_string` 44 times: 82% of these failures paid for a probe call
# that a line number in the payload already answers. So the line number goes in
# the payload — one line of it, bounded, not a dump of the file.

BASE_NOT_FOUND_BODY = json.dumps(
    {"error": "old_string not found in file (must match exactly)"})


def _named_line(body: str) -> int:
    """The 1-based line number the payload names, or a failure that quotes it."""
    m = re.search(r"\bnearest lines? (\d+)", body)
    assert m, f"the payload names no candidate line: {body}"
    return int(m.group(1))


async def test_a_near_match_names_the_candidate_line(bound, tmp_path):
    """Clause 1: a near-matching line is reported by its 1-based number.

    The `old_string` differs from line 2 both by one character and by its
    indent — the two shapes an `Edit` written from memory actually gets wrong,
    and the reason the comparison is against the *stripped* line.
    """
    p = tmp_path / "f.py"
    p.write_text("alpha\n    beta = compute(1)\ngamma\n")
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "  beta = compute(2)",
                                      "new_string": "beta = compute(2)"})
    assert res.is_error, _text(res)
    body = _text(res)
    assert _named_line(body) == 2
    assert "beta = compute(1)" in body, "the candidate is quoted so the diff is visible"
    assert p.read_text() == "alpha\n    beta = compute(1)\ngamma\n"


async def test_no_near_match_reports_the_line_count(bound, tmp_path):
    """Clause 2: nothing close → the total line count and the probe that missed."""
    p = tmp_path / "f.py"
    p.write_text("alpha\nbeta\ngamma\n")
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "quantum_entanglement_transducer",
                                      "new_string": "x"})
    assert res.is_error
    body = _text(res)
    assert "file has 3 lines" in body
    assert "none containing the first 40 chars of old_string" in body
    assert "nearest line" not in body


async def test_the_hint_is_bounded_to_200_extra_chars(bound, tmp_path, monkeypatch):
    """Clause 3: a long, quote-heavy candidate still lands inside the budget.

    The cap is measured on the encoded body, not the raw hint: an excerpt of `"`
    and `\\` is longer again once `json.dumps` escapes it, and 200 characters is the
    ceiling the clause sets on the body the model reads. The shipped excerpt is
    already short enough not to bind here, so the second half of this node widens
    it and asks the same question where the budget *is* the constraint: the trim
    must give up the quote, not the number.
    """
    p = tmp_path / "f.py"
    nasty = 'x = "' + 'quote"delim\\ ' * 40 + '"'
    p.write_text("alpha\n" + nasty + "\ngamma\n")
    await FS.call_tool("Read", {"file_path": str(p)})
    args = {"file_path": str(p), "old_string": nasty.replace("delim", "delim2", 1),
            "new_string": "x"}
    body = _text(await FS.call_tool("Edit", args))
    assert _named_line(body) == 2, body
    assert len(body) <= len(BASE_NOT_FOUND_BODY) + 200, f"{len(body)} chars: {body}"

    monkeypatch.setattr(SHARED, "_NEAR_EXCERPT_CHARS", 200)
    widened = _text(await FS.call_tool("Edit", args))
    assert "nearest line 2" in widened, f"the trim ate the line number: {widened}"
    assert len(widened) <= len(BASE_NOT_FOUND_BODY) + 200, f"{len(widened)} chars: {widened}"


async def test_a_capped_scan_never_claims_a_none_it_did_not_read(bound, tmp_path,
                                                                monkeypatch):
    """The fallback says "none" only about the lines the scan actually read.

    The similarity scan is bounded, so on a file longer than the bound the
    unqualified sentence would be a claim about bytes nobody looked at. Shrinking
    the bound is how this is checked without shipping a 20,000-line fixture.
    """
    monkeypatch.setattr(SHARED, "_NEAR_SCAN_MAX_LINES", 2)
    p = tmp_path / "f.py"
    p.write_text("alpha\nbeta\nthe needle line\n")
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "quantum_entanglement_transducer",
                                      "new_string": "x"})
    assert res.is_error
    body = _text(res)
    assert "file has 3 lines, none of its first 2 containing the first 40 chars" in body


# ── the contract is in the schema, not only the error ───────────────────────

async def test_the_tool_descriptions_state_the_contract():
    tools = {t.name: t for t in await FS.list_tools()}
    assert "Read the file in this session first" in tools["Edit"].description
    assert "OVERWRITING an existing one requires" in tools["Write"].description


# ── the seam the guard is actually reached through ───────────────────────────

async def test_the_pointer_refusal_survives_the_aggregator_trip(governed):
    """The refusal arrives as an error result over `main.call_tool`, not as a success.

    Every node above calls `FS.call_tool` in-process. A real session crosses
    `agent_mcp.main.call_tool` instead, where the tool is looked up in a dispatch
    table, the session id arrives in `_meta` and binds a contextvar, and the handler's
    JSON comes back inside a `CallToolResult` whose `isError` the harness reads — so a
    refusal that only works in-process is a guard the tests see and the nightly run
    does not. `#3626`'s run is exactly such a caller: §2e writes the pointer over MCP.

    No `bound` fixture here on purpose: `main` binds the session itself from `_meta`,
    which is how the guard ends up running for a caller this lane never gated before.
    """
    from agent_mcp import main as M

    pointer, archive = _linked_pointer(governed, "conversation-patterns")
    await M.list_tools()
    res = await M.call_tool("Write", {"file_path": str(pointer),
                                      "content": "tonight's patterns\n"},
                            {"lloyd/session_id": SID})
    assert type(res).__name__ == "CallToolResult", type(res).__name__
    assert res.is_error is True, _text(res)
    assert "symbolic link" in _err(res), _err(res)
    assert "#3626" in _err(res), _err(res)
    assert os.path.islink(str(pointer)), "the trip through the aggregator wrote anyway"
    assert archive.read_bytes() == b"the previous night\n", (
        "the refusal came back as text but its bytes still landed in the archive copy")
