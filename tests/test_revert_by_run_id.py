"""Putting a dead run's writes back, from its run record alone (#963).

`agent_mcp/_change_ledger.py` has always been able to restore a turn's files,
and the only way to ask it was `POST /changes/revert {session, turn}` — a body
somebody had to compose by hand with three facts they had to go and find first.
Across 359 turn indexes and 1,051 entries at triage, `reverted_at` was set on
zero: the undo was not unused so much as unreachable.

`autonomy.revert_run_writes` is the in-process call that takes what the record
already prints — the session id and the run id, which `run_task` passes as the
ledger's turn id — and answers per path. Per-path is the point: a file another
writer has touched since the snapshot comes back REFUSED with its name, is left
exactly as that later writer left it, and is not quietly skipped. Restoring over
a later write is the damage the ledger exists to prevent; skipping in silence
would make the undo untrustworthy.

Nothing calls this on a failure path. A run that died with 6 of its 9 notes
written has 6 notes of real progress: reports, decisions, and restores are three
different acts, and this module does the first.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
import yaml

import autonomy
from agent_mcp import _change_ledger as CL

WRITES = ["knowledge/note-a.md", "knowledge/note-b.md", "lloyd/MEMORY.md"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Sessions, run records and the change ledger, all under tmp_path.

    The ledger root is the sessions tree, as in production: pre-images are a
    sibling directory of the transcripts, named `<session_id>.changes`.
    """
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "autonomy-runs")
    monkeypatch.setattr(CL, "CHANGES_ROOT", tmp_path / "sessions")
    CL.reset()
    yield tmp_path
    CL.reset()


def _stub_run(monkeypatch, tmp_path, task):
    """Neutralise everything `run_task` touches except the record and ledger.

    Deliberately does NOT capture the `RunOptions` handed to the harness: every
    id these tests use comes off the run record on disk, which is the claim
    under test. A test that read the ids out of memory would still pass if the
    record printed nothing.
    """
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "x.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})

    class Opts:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    import app.harness as harness
    import app.harness.mcp_pool as mcp_pool
    monkeypatch.setattr(harness, "RunOptions", Opts)
    monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")


def _dead_run_that_wrote(monkeypatch, tmp_path, names):
    """Run a real `run_task` that records `names` and then dies mid-write.

    Returns the run record's parsed front matter, its body, and the files. The
    ids the record prints are what the tests hand to the undo — nothing here
    reaches for the in-memory `RunOptions`, because the whole claim is that the
    record is a complete address.
    """
    task = {"id": 58, "name": "Nightly vault maintenance", "skill_name": "s",
            "status": "up_next", "timeout_seconds": 300}
    _stub_run(monkeypatch, tmp_path, task)

    async def _crash(messages, options):
        scope = CL.scope(options.session_id, options.turn_id)
        for i, name in enumerate(names):
            p = tmp_path / "vault" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            pre = f"the note as it stood before run {options.turn_id}\n"
            p.write_text(pre)
            entry = CL.begin(scope, real=os.path.realpath(p), path=os.fspath(p),
                             op="edit", call_id=f"call-{i}")
            CL.snapshot_pre(scope, entry, pre.encode())
            p.write_text(pre + f"rewritten by run {options.turn_id}\n")
            CL.commit(scope, entry, CL.sha256_bytes(p.read_bytes()))
        # Forget the cache: the writes belong to the aggregator process, and the
        # backend doing the restoring holds nothing but the index on disk.
        CL.reset()
        yield {"type": "text_delta", "text": "rewriting the notes…"}
        raise RuntimeError("boom mid-write")

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _crash)
    out = asyncio.run(autonomy.run_task(58))
    assert out["success"] is False

    records = list((tmp_path / "autonomy-runs" / "58").glob("run_58_*.md"))
    assert len(records) == 1, f"expected one failure record, got {records}"
    _, fm_text, body = records[0].read_text().split("---\n", 2)
    files = [tmp_path / "vault" / name for name in names]
    return yaml.safe_load(fm_text), body, files


def _by_path(result):
    return {row["path"]: row for row in result["results"]}


# ── clause 3: the record's two ids are the whole address ───────────────

def test_the_ids_a_record_prints_are_enough_to_put_every_write_back(world,
                                                                    monkeypatch):
    """No hand-composed HTTP body: the run id IS the ledger's turn id.

    `run_task` passes `turn_id=run_id`, so the `session_id` and `run_id` in a
    run record's front matter are the complete address. This test takes them off
    the record and hands them to the undo, which must restore all three files to
    the bytes they held before the run touched them.
    """
    fm, body, files = _dead_run_that_wrote(monkeypatch, world, WRITES)
    session_id, run_id = fm["session_id"], fm["run_id"]
    assert run_id.startswith("run_58_"), "a run id is the ledger turn id"

    # The scope the record printed and the scope the undo uses are one string,
    # so nothing has to be translated between reading and acting.
    recorded_scope = fm["changes"].split(" (")[0]
    assert recorded_scope == f"sessions/{session_id}.changes/{run_id}/"
    out = autonomy.revert_run_writes(session_id, run_id)
    assert out["scope"] == recorded_scope
    assert out["recorded"] == len(WRITES)
    assert out["counts"] == {"restored": len(WRITES)}, out["results"]
    for p in files:
        assert "rewritten by run" not in p.read_text(), (
            f"{p} still holds the run's write")
        assert "as it stood before" in p.read_text()


def test_the_undo_reports_per_path_rather_than_one_verdict_for_the_turn(
        world, monkeypatch):
    """A single success flag cannot be honest when one of nine files was left.

    Every row names the path it speaks about, so a caller can act on the half
    that did not happen.
    """
    fm, _, files = _dead_run_that_wrote(monkeypatch, world, WRITES)
    out = autonomy.revert_run_writes(fm["session_id"], fm["run_id"])
    rows = _by_path(out)
    assert set(rows) == {os.fspath(p) for p in files}
    assert all(r["status"] == "restored" for r in rows.values())
    assert out["dir"] == str(world / "sessions" / f"{fm['session_id']}.changes"
                             / fm["run_id"])


# ── clause 4: a file somebody else moved is refused by name ────────────

def test_a_file_another_writer_moved_is_refused_by_name_and_left_alone(
        world, monkeypatch):
    """The one status that keeps the undo safe to run at 03:00 about a 02:00 run.

    Another job wrote `lloyd/MEMORY.md` after this run died. Putting the
    pre-image back would delete that later work, which is the exact destruction
    this ledger was built to stop, so that one file comes back `refused` with
    its name and a reason — and the other two still get restored. Neither
    outcome may be inferred from a fleet-wide status.
    """
    fm, _, files = _dead_run_that_wrote(monkeypatch, world, WRITES)
    moved = files[2]
    later = "written by the job that came after this run\n"
    moved.write_text(later)

    out = autonomy.revert_run_writes(fm["session_id"], fm["run_id"])
    rows = _by_path(out)

    refused = rows[os.fspath(moved)]
    assert refused["status"] == "refused"
    assert "changed since" in refused["reason"]
    assert os.fspath(moved) in json.dumps(out["results"]), (
        "the refusal has to name the file, or it cannot be acted on")
    assert moved.read_text() == later, (
        "the later write survives: refusing is the point, restoring is the damage")

    assert out["counts"] == {"restored": 2, "refused": 1}
    for p in files[:2]:
        assert "as it stood before" in p.read_text()


def test_a_refused_file_is_still_refused_on_the_second_call(world, monkeypatch):
    """A retry must not turn a refusal into a silent success.

    The second run of the undo reports the same file refused for the same
    reason; nothing about it improves by asking twice, and a caller reading
    `restored` the second time would conclude the file had been put back.
    """
    fm, _, files = _dead_run_that_wrote(monkeypatch, world, WRITES)
    moved = files[2]
    moved.write_text("written after this run\n")

    first = autonomy.revert_run_writes(fm["session_id"], fm["run_id"])
    second = autonomy.revert_run_writes(fm["session_id"], fm["run_id"])
    rows = _by_path(second)
    assert rows[os.fspath(moved)]["status"] == "refused"
    assert rows[os.fspath(files[0])]["status"] == "skipped"
    assert "already reverted" in rows[os.fspath(files[0])]["reason"]
    assert first["counts"]["refused"] == second["counts"]["refused"] == 1


# ── clause 3's other half: the failure path does not call this ─────────

def test_autonomy_reverts_only_from_the_explicit_entry_point():
    """The property the failure-path tripwire above cannot see from one run.

    One test can show that a timeout did not revert; only a read of the module
    can show that NOTHING on the failure side can. Before #963 the only revert
    caller in the tree was the aggregator's HTTP route, and the gap the item
    names was that no in-process caller existed; after it, `autonomy.py` holds
    exactly one call to `_change_ledger.revert(`, and it sits inside
    `revert_run_writes` — nowhere near `_record_failure`.

    Scoped to this module rather than grepped across the repository on purpose:
    a tree-wide assertion would fail the next time any other surface legitimately
    learns to revert, and a test that fails for somebody else's correct change
    teaches people to delete tests.
    """
    from pathlib import Path

    src = (Path(autonomy.__file__)).read_text()
    assert src.count("_change_ledger.revert(") == 1, (
        "exactly one revert call site in autonomy.py, and it is the entry point")
    entry_point = src.split("def revert_run_writes", 1)[1].split("\ndef ", 1)[0]
    assert "_change_ledger.revert(" in entry_point
    failure_path = src.split("async def _record_failure", 1)[1].split(
        "\nasync def ", 1)[0]
    assert "revert_run_writes(" not in failure_path, (
        "_record_failure reports and never restores")
