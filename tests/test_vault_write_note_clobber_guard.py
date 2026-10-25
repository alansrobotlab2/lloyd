"""The read-before-overwrite gate on the `vault_write` lane (#3291).

On 2026-10-24 an interactive session digesting arXiv 2610.14224 replaced, with no
warning and no refusal, a `digest: full` paper note the `youtube-digest` worker had
authored nine days earlier. `vault_write` returned `{"success": true, …}`, an audit
row appeared with a non-empty `pre_sha256`, and the caller had no idea a survivor was
there to replace. Recoverability was already fixed (#962 landed `_read_pre_bytes` and
`_ledger_record`, which snapshot the pre-image), so the overwrite was undoable —
and still silent. Nothing on this lane warns *before* the bytes go.

The asymmetry is the finding: the sibling lane already refuses exactly this.
`Write`/`Edit` build a `_Mutation`, consult `_seen(session_id, realpath)` and answer
"already exists and has not been Read in this session" (`agent_mcp/builtin_fs.py`
`_gate_check`). `vault_write` never consulted that registry, and `_vault_read` never
recorded into it — only the `Read` handler does — so the lane the research and digest
skills are *told* to use was the one lane with no clobber gate on it.

Scoped to `knowledge/`, and measured rather than assumed. Measured 2026-10-25 (that journal grows, so these are counts as of a date, not
invariants): of 5418 rows in
`~/obsidian/memory/audit/writes.jsonl`, 2883 carry a non-empty `pre_sha256` (an
overwrite of an existing file); by first path segment those are `lloyd` 2714,
`knowledge` 52, `memory` 50, `skills` 26, `backlog` 17, `autonomy` 9. The `lloyd/`
population is the loaded-memory lane the nightly memory writer rewrites whole, so a
gate on every overwrite would refuse the memory lane; on `knowledge/` it covers the
incident class at ~2% of overwrite traffic. Every node below that says "not gated" is
that boundary being held, not a gap.

These tests drive the real handler against a scratch vault — never `~/obsidian` —
with a session bound the way `main.call_tool` binds one.
"""
from __future__ import annotations

import json
import types

import pytest

from agent_mcp import _change_ledger as CL
from agent_mcp import _task_registry as TR
from agent_mcp import builtin_fs as FS

SID = "20261025_041224_chat"
TURN = "turn-clobber"

#: The note that was clobbered, in the shape the real one had: front matter a worker
#: wrote, and `digest: full` — the field that says a finished document lives here.
#: `type: notes`, not `type: paper`: this lane rewrites the retired spelling to its
#: canonical value (`okf_taxonomy.TYPE_ALIASES`, the #370 vocabulary #872 consolidated),
#: so a fixture asking for `paper` would be comparing the guard's rewrite to the bytes it
#: asked for. The `digest: full` field is the one the item names as the tell.
PAPER = "knowledge/papers/2610.14224-harness2-recursive-agent-harnessing.md"
PAPER_ORIGINAL = """---
type: notes
arxiv_id: 2610.14224
digest: full
---

# Harness2 — authored by the youtube-digest worker on 2026-10-15

The worker's own findings, 4 KB of them.
"""
PAPER_OVERWRITE = """---
type: notes
arxiv_id: 2610.14224
digest: full
---

# Harness2 — written over by an interactive session on 2026-10-24
"""

#: The guard's own name, as it reaches the denial journal. Asserted rather than
#: incidental: `~/lloyd-data/safety/denials.jsonl` holds 620 rows, 4 naming
#: `vault_write` (all dispatch-field refusals) and 0 naming an overwrite, while 2883
#: overwrites went through. A refusal nobody can find is a refusal nobody can audit.
GUARD = "vault_note_clobber"


@pytest.fixture(autouse=True)
def _clean():
    FS.reset_read_records()
    CL.reset()
    yield
    FS.reset_read_records()
    CL.reset()


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """`vault_write`/`vault_read` aimed at a scratch tree with its own audit and ledger.

    `VAULT` is imported into `agent_mcp.vault`'s namespace, so that is the name to
    patch — patching `agent_mcp._shared` would leave the handler writing into
    ~/obsidian, which is the tree this item exists to protect.
    """
    import agent_mcp.vault as V
    from app.harness import denial_journal

    monkeypatch.setattr(V, "VAULT", tmp_path)
    monkeypatch.setattr(V, "AUDIT_LOG_DIR", tmp_path / "audit")
    monkeypatch.setattr(V, "AUDIT_LOG_FILE", tmp_path / "audit" / "writes.jsonl")
    monkeypatch.setattr(CL, "CHANGES_ROOT", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    journal: list[dict] = []
    # Plain append, no `or True`: a spy that returns a truthy value reads like an
    # assertion that cannot fail, and this one's job is to be believed.
    monkeypatch.setattr(denial_journal, "record", lambda **kw: journal.append(kw))
    return types.SimpleNamespace(root=tmp_path, mod=V, journal=journal)


@pytest.fixture
def bound(monkeypatch):
    """Bind a session and a turn, as `main.call_tool` does for a real tool call.

    Patched on `agent_mcp.vault`'s own imported name: `_ledger_scope` there calls
    `get_bound_session()` resolved in vault's namespace, so patching `FS`'s copy would
    test a lane the handler never takes.
    """
    import agent_mcp.vault as V

    monkeypatch.setattr(V, "get_bound_session", lambda: SID)
    tok = TR.current_turn_id.set(TURN)
    ctok = TR.current_call_id.set("call_clobber")
    yield SID
    TR.current_turn_id.reset(tok)
    TR.current_call_id.reset(ctok)


def _write(scratch, rel: str, content: str) -> dict:
    return scratch.mod._vault_write({"path": rel, "content": content})


def _read(scratch, rel: str) -> dict:
    return scratch.mod._vault_read({"path": rel})


def _audit_rows(scratch) -> list[dict]:
    log = scratch.root / "audit" / "writes.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _seed(scratch, content: str = PAPER_ORIGINAL, rel: str = PAPER) -> None:
    """Land the note through the lane itself, with NO session bound.

    Two reasons, and the second is why this is a helper rather than one line: the real
    survivor's own commit is by `lloyd-autonomy-data-pipeline` (2026-10-15,
    `4a76d7ca`) — the item records the youtube-digest worker as the author, and the
    commit author is a different identity again, which is why this gate makes no
    authorship claim at all, and a seed written *by the session under test* would record the note as
    seen — a successful `vault_write` records its own post-write stat, as `Write`'s handler
    does — which would leave every refusal node below reading as a gate that never fires.
    Seeding through the handler rather than `write_text` also proves the guard does not
    fire on a create.
    """
    mod = scratch.mod
    saved = mod.get_bound_session
    mod.get_bound_session = lambda: ""
    try:
        res = _write(scratch, rel, content)
    finally:
        mod.get_bound_session = saved
    assert res.get("success") is True, f"seed write refused: {res}"
    assert (scratch.root / rel).read_text(encoding="utf-8") == content


# ── the repro: an unread overwrite of a knowledge note ───────────────────────


def test_an_unread_overwrite_of_a_knowledge_note_is_refused(scratch, bound):
    """Clause 1's repro, run against a fixture vault, never ~/obsidian.

    Before this file's change this node is the RED one: the handler reached
    `write_text_durable` and returned `{"success": True, "bytes": …}`.
    """
    _seed(scratch)
    res = _write(scratch, PAPER, PAPER_OVERWRITE)

    assert res.get("success") is not True, (
        f"the overwrite landed with no prior read: {res!r} — this is the silent "
        "clobber #3291 was filed for")
    assert "error" in res and res["error"], res
    assert (scratch.root / PAPER).read_text(encoding="utf-8") == PAPER_ORIGINAL, (
        "the refusal wrote bytes anyway, which makes it a warning and not a refusal")


def test_the_refusal_writes_no_audit_row_and_no_ledger_entry(scratch, bound):
    """The refusal must return before `_ledger_record` and `_audit_write`.

    Both halves are checked because they answer different later readers: the audit row
    is what makes an overwrite countable in `writes.jsonl` (the 2883 rows that made
    this item's blast radius measurable), and the ledger entry is what a human would
    use to undo it. A refusal that recorded either would be an overwrite that merely
    announced itself — and the undo snapshot would imply bytes were replaced.
    """
    _seed(scratch)
    assert len(_audit_rows(scratch)) == 1, "the seed should have left exactly one row"
    before = len(CL.list_changes(SID, TURN))

    res = _write(scratch, PAPER, PAPER_OVERWRITE)

    assert res.get("success") is not True, res
    assert len(_audit_rows(scratch)) == 1, (
        "a refused write appended an audit row, so the refusal is indistinguishable "
        "from the overwrite it refused")
    assert len(CL.list_changes(SID, TURN)) == before, (
        "a refused write opened a change-ledger entry for bytes it did not write")


def test_the_refusal_names_the_read_that_clears_it(scratch, bound):
    """A refusal a caller cannot act on is a wall, not a gate.

    The sibling lane's message says "Read it first, then Write it"; this one has to
    name `vault_read`, because `Read` is not the tool that reads a vault path — and a
    caller told to "read it first" who reaches for `Read` gets a protected-path refusal
    and learns nothing.
    """
    _seed(scratch)
    res = _write(scratch, PAPER, PAPER_OVERWRITE)
    msg = res.get("error", "")

    assert "vault_read" in msg, f"the message never names the tool that clears it: {msg!r}"
    assert PAPER in msg, f"the message does not name the path it refused: {msg!r}"
    # Case-insensitively: the claim is that the message states the bytes are untouched,
    # and it opens that sentence with a capital because it follows a full stop.
    assert "nothing was written" in msg.lower(), (
        f"the message does not say the bytes are untouched: {msg!r}")


def test_the_refusal_is_journalled_where_denials_are_found(scratch, bound):
    """The refusal reaches `denial_journal`, or nobody can ask later whether it fired.

    This is how the question "has the clobber gate ever stopped anything?" gets answered
    at all: `~/lloyd-data/safety/denials.jsonl` could only ever report the refusals that
    were recorded, which is why it reported 0 overwrites across 2883 of them.
    """
    _seed(scratch)
    _write(scratch, PAPER, PAPER_OVERWRITE)

    rows = [r for r in scratch.journal if r.get("guard") == GUARD]
    assert len(rows) == 1, (
        f"the refusal recorded no denial row (journal holds {scratch.journal!r}), so "
        "the gate's own firing rate would be unmeasurable")
    assert PAPER in json.dumps(rows[0]), f"the denial row omits the path: {rows[0]!r}"


# ── the lane still works: the gate is not a ban ──────────────────────────────


def test_reading_the_note_first_lets_the_write_through(scratch, bound):
    """The whole cost of the gate is one read. If it cost more, it would be refused
    by its own users and the lane would drift back to silence."""
    _seed(scratch)
    read = _read(scratch, PAPER)
    assert read.get("text", "").startswith("---"), read

    res = _write(scratch, PAPER, PAPER_OVERWRITE)

    assert res.get("success") is True, f"a read-then-write was refused: {res!r}"
    assert (scratch.root / PAPER).read_text(encoding="utf-8") == PAPER_OVERWRITE
    assert len(_audit_rows(scratch)) == 2, (
        "the accepted write left no audit row, so the accepted half of this guard is "
        "an absence rather than an observation")


def test_a_partial_read_is_enough(scratch, bound):
    """`vault_read` with `start_line`/`num_lines` records too.

    The sibling gate accepts a partial Read for the same reason: the caller needed to
    see what is there, not to have memorised the file. A gate that demanded a whole-file
    read would be a gate callers route around.
    """
    _seed(scratch)
    _read_partial = scratch.mod._vault_read(
        {"path": PAPER, "start_line": 1, "num_lines": 2})
    assert _read_partial.get("text"), _read_partial

    assert _write(scratch, PAPER, PAPER_OVERWRITE).get("success") is True


def test_a_read_the_nightly_moved_under_the_caller_is_refused(scratch, bound):
    """The record's second half: it carries `(mtime_ns, size)`, not just "was read".

    The real-life order is the bad one: a session reads the stub note, a digest worker
    lands the full digest behind it, the session writes its draft over the finished
    document. A bare "has this session ever opened this path" flag cannot catch that; the
    recorded stat can, and it is the same tuple `Write`'s gate compares, so the two lanes
    cannot disagree about whether the caller is looking at current bytes.
    """
    _seed(scratch)
    assert _read(scratch, PAPER).get("text"), "the read should have returned the note"

    # The worker lands, off this session: an external writer, exactly the shape of a
    # nightly digest job committing behind an interactive turn.
    (scratch.root / PAPER).write_text(
        PAPER_ORIGINAL + "\nThe worker added 3 KB more this morning.\n", encoding="utf-8")

    res = _write(scratch, PAPER, PAPER_OVERWRITE)

    assert res.get("success") is not True, (
        "a draft composed against a stub was written over the worker's finished note "
        f"because the session had read the stub: {res!r}")
    assert "changed on disk" in res.get("error", ""), res
    assert "The worker added 3 KB more this morning." in (
        scratch.root / PAPER).read_text(encoding="utf-8"), (
        "the refusal wrote over the worker's bytes anyway")


def test_a_note_this_session_wrote_may_be_written_again(scratch, bound):
    """A create-then-replace in one session is the digest worker's own shape.

    `Write`'s handler records its own post-write stat (`_record_seen` on `post_stat`)
    so a file the session itself authored stays writable by it; without the same
    recording, this guard would refuse a worker to update the note it created minutes
    ago, which is a new failure mode invented to fix an old one.
    """
    created = _write(scratch, "knowledge/papers/fresh-note.md",
                     "---\ntype: notes\n---\n\n# Draft\n")
    assert created.get("success") is True, created

    again = _write(scratch, "knowledge/papers/fresh-note.md",
                   "---\ntype: notes\n---\n\n# Draft, revised the same session\n")

    assert again.get("success") is True, (
        f"a session was refused the note it had just written: {again!r}")


def test_creating_a_knowledge_note_needs_no_prior_read(scratch, bound):
    """There is nothing to clobber on a create, so there is nothing to refuse.

    The sibling gate has the identical exemption (`mut.kind == "write" and not
    mut.existed`). A gate that fired on creates would stop the research and digest lanes
    at their first write — and a create is far more common than an overwrite, so the
    52 `knowledge/` overwrite rows are only the tail of what it would also refuse.
    """
    res = _write(scratch, "knowledge/research/2026-10-25-brand-new-card.md",
                 "---\ntype: research\n---\n\n# New\n")

    assert res.get("success") is True, f"a create was gated: {res!r}"
    assert (scratch.root / "knowledge/research/2026-10-25-brand-new-card.md").is_file()


# ── the boundary of the gate: what it deliberately does NOT touch ────────────


@pytest.mark.parametrize("rel", [
    "lloyd/memory/memory-md-ledger.md",
    "memory/topics/some-topic.md",
    "backlog/3291-an-item.md",
    "skills/some-skill/SKILL.md",
])
def test_the_gate_does_not_reach_outside_knowledge(scratch, bound, rel):
    """2718 of the 2883 overwrite rows (measured 2026-10-25) are `lloyd/`, and they
    stay untouched.

    That population is the nightly loaded-memory lane, which rewrites `lloyd/MEMORY.md`
    and the ledger whole without reading them first — a gate extended to it would refuse
    the memory writer every night. `memory/`, `backlog/` and `skills/` are here for the
    same reason, at their own measured counts (50, 17, 26). The item's fix shape scopes
    this to `knowledge/`, and a node that only proves the fire would not tell anyone the
    fence has a gate in it.
    """
    _seed(scratch, content="---\ntype: note\n---\n\n# Before\n", rel=rel)
    res = _write(scratch, rel, "---\ntype: note\n---\n\n# After, unread\n")

    assert res.get("success") is True, (
        f"the clobber gate refused an overwrite outside knowledge/, which is the "
        f"nightly-memory-lane breakage the scoping exists to avoid: {res!r}")
    assert (scratch.root / rel).read_text(encoding="utf-8") == (
        "---\ntype: note\n---\n\n# After, unread\n")


def test_an_unattended_call_is_not_gated(scratch, monkeypatch):
    """No bound session means no caller to instruct, so the lane is open.

    This is the sibling gate's rule verbatim (`gate_on = bool(session_id) and
    _gates_enabled()`), and it is what keeps the 2703 turn-less overwrite rows — the
    in-process nightly writers — moving. A gate that fired here would fail the nightly
    chain, not protect a note.
    """
    import agent_mcp.vault as V

    monkeypatch.setattr(V, "get_bound_session", lambda: "")
    _seed(scratch)

    res = _write(scratch, PAPER, PAPER_OVERWRITE)

    assert res.get("success") is True, f"an unattended write was gated: {res!r}"
    assert not [r for r in scratch.journal if r.get("guard") == GUARD]


def test_the_edit_gates_kill_switch_opens_this_lane_too(scratch, bound, monkeypatch):
    """One switch for both lanes, or the vault lane becomes the way around the switch.

    `harness.edit_gates.enabled` is how an operator turns the Write/Edit clobber gates
    off; if this guard ignored it, `vault_write` would be the lane that keeps enforcing
    after an operator asked it to stop — and the lane the skills use is the one they
    would notice.
    """
    from app.config import CONFIG

    _seed(scratch)
    monkeypatch.setitem(CONFIG, "harness", {**CONFIG.get("harness", {}),
                                            "edit_gates": {"enabled": False}})

    assert _write(scratch, PAPER, PAPER_OVERWRITE).get("success") is True, (
        "the clobber guard kept enforcing with the gates switched off")


# ── the registry: one record, two lanes ──────────────────────────────────────


def test_vault_read_records_into_the_registry_the_write_lane_uses(scratch, bound):
    """The two lanes must share one read-registry, or they disagree about who looked.

    Asserted through the sibling's own lookup rather than through a successful write:
    a shared file is the seam, and a second registry inside `vault.py` would satisfy
    every behavioural node above while letting `Write` and `vault_write` answer the same
    question with different answers.
    """
    from pathlib import Path

    _seed(scratch)
    target = scratch.root / PAPER
    assert FS._seen(SID, str(Path(target).resolve())) is None, (
        "the session already 'has read' the note it never read, which would make the "
        "gate's own firing unprovable")

    _read(scratch, PAPER)

    assert FS._seen(SID, str(target.resolve())) is not None, (
        "vault_read did not record into the registry Write's gate consults, so the two "
        "lanes are tracking two different ideas of what the session has seen")


def test_the_clobber_rule_lives_on_the_lane_that_was_missing_it(scratch):
    """Placement, not behaviour: the refusal is returned before the bytes go.

    The node reads the handler's source because the ordering is the claim: a guard that
    sits after `write_text_durable` would report a refusal over bytes already replaced,
    which is #3291's second complaint about the same pair of tools. The refusal call must
    precede the write in the same function that contains it.
    """
    import inspect

    src = inspect.getsource(scratch.mod._vault_write)
    # Matched on the call sites, not the bare names: the guard's own comment above the
    # call mentions `write_text_durable` to explain the ordering, and a `find` on the name
    # alone reads that comment as the write — the node would then be comparing a comment
    # to a call and reporting an ordering that it had invented.
    guard_at = src.find("_note_clobber_refusal(path, target, pre_bytes")
    write_at = src.find("write_text_durable(target, content)")

    assert guard_at != -1, "_vault_write contains no call to the clobber guard at all"
    assert write_at != -1, "the handler's write call was renamed; this node is stale"
    assert guard_at < write_at, (
        "the clobber guard runs after the write, so its refusal reports over bytes "
        "that are already gone")
