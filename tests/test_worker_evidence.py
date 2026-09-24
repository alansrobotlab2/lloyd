"""Verifier-bound evidence claims on worker runs — backlog #525.

The most-repeated defect in the corrections log is a run record whose prose
summary disagrees with the disk. Three seeded instances, all real, all in
`~/lloyd/_pipeline/reflection/`:

* 2026-08-24 — handoff: entity graph "restored to 12,131 relationships".
  The same night's health report: `| Total relationships | 0 |`.
* 2026-09-03 — handoff: "96 files total" of `hypothesis_fail_*.txt`, 09-02 peak
  21. Disk held 121 and 64.
* 2026-09-04 — handoff: `signals-latest.md` is "307KB / 383,722 chars". Disk:
  13,503 bytes.

Until now the only defence was procedural — a skill telling the model to verify
on disk first, which binds only a willing model. These tests pin the structural
version: claims arrive as `{claim, check}` pairs, a stdlib verifier re-runs each
check when the ledger row is written, the verified/gap partition lands in the run
record, the rate is per task with a claim-level denominator, and the gap list
goes into the next run of that task's prompt.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import autonomy
from workers import evidence
from workers.evidence import (CHECK_KINDS, gaps_key, gaps_prompt,
                              parse_claims_block, parse_gap_list,
                              verify_bundle, verify_claim)
from workers.pool import WorkerPool, normalize_result
from workers.queue import QueueItem, WorkQueue

CHECKOUT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evidence"


def _seed_root_for(rel_path: str) -> Path | None:
    """The root that actually holds this artifact — live tree first, then the
    checked-in mirror of the same relative layout.

    Resolved per artifact rather than once per tree, because `_pipeline/` is
    created lazily by the pipeline code that runs under it: a round's worktree
    can hold an empty `_pipeline/reflection/` skeleton — the gate's full-suite
    run creates one — and an empty directory is not the artifact store. The
    mirror keeps the assertion alive where `_pipeline` is absent entirely (it is
    gitignored); on the live checkout the real files win, which is what makes
    the replay an assertion about the artifacts the corrections log argued
    about. Every caller asserts the artifact exists, so neither branch is
    vacuous and neither skips.
    """
    for root in (CHECKOUT, FIXTURES / "seeded"):
        if (root / rel_path).exists():
            return root
    return None


def _item(payload: dict | None = None, source: str = "s", **kw) -> QueueItem:
    base = dict(id=1, source=source, kind="k", priority=50, payload=payload or {},
                dedup_key=None, state="running", attempts=1,
                enqueued_at="", claimed_at=None, claimed_by=None,
                completed_at=None, error=None)
    base.update(kw)
    return QueueItem(**base)


# ── The schema ─────────────────────────────────────────────────────────────

def test_the_check_kinds_are_the_four_the_item_named():
    assert set(CHECK_KINDS) == {"file_exists", "count_eq", "json_key", "regex"}
    assert set(evidence.CLAIM_STATUSES) == {"verified", "refuted", "insufficient"}


def test_an_unknown_kind_is_insufficient_rather_than_a_pass(tmp_path):
    v = verify_claim({"claim": "it worked", "check": {"kind": "vibes"}},
                     root=tmp_path)
    assert v["status"] == "insufficient"
    assert "no verifier ran" in v["detail"]


def test_a_claim_that_is_not_an_object_is_insufficient(tmp_path):
    assert verify_claim("25 rows written", root=tmp_path)["status"] == "insufficient"


# ── file_exists ────────────────────────────────────────────────────────────

def test_file_exists_verifies_and_refutes(tmp_path):
    (tmp_path / "handoff.md").write_text("done")
    ok = verify_claim({"claim": "handoff written",
                       "check": {"kind": "file_exists", "path": "handoff.md"}},
                      root=tmp_path)
    assert ok["status"] == "verified" and ok["observed"] is True

    bad = verify_claim({"claim": "the second artifact exists",
                        "check": {"kind": "file_exists", "path": "missing.md"}},
                       root=tmp_path)
    assert bad["status"] == "refuted"
    assert bad["observed"] is False and "exists=False" in bad["detail"]


def test_a_path_outside_the_tree_is_unevaluable_not_refuted(tmp_path):
    v = verify_claim({"claim": "etc is readable",
                      "check": {"kind": "file_exists", "path": "/etc/passwd"}},
                     root=tmp_path)
    assert v["status"] == "insufficient"
    assert "escapes the evidence root" in v["detail"]


# ── count_eq ───────────────────────────────────────────────────────────────

def test_count_eq_counts_matching_files(tmp_path):
    d = tmp_path / "debug"
    d.mkdir()
    for i in range(3):
        (d / f"fail_{i}.txt").write_text("x")
    ok = verify_claim({"claim": "3 dumps", "check": {
        "kind": "count_eq", "path": "debug", "glob": "fail_*.txt",
        "measure": "files", "expected": 3}}, root=tmp_path)
    assert ok["status"] == "verified" and ok["observed"] == 3

    bad = verify_claim({"claim": "96 dumps", "check": {
        "kind": "count_eq", "path": "debug", "glob": "fail_*.txt",
        "measure": "files", "expected": 96}}, root=tmp_path)
    assert bad["status"] == "refuted" and bad["observed"] == 3


def test_count_eq_measures_bytes_lines_and_matches(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("a\nb\nc\n")
    assert verify_claim({"claim": "size", "check": {
        "kind": "count_eq", "path": "notes.md", "measure": "bytes",
        "expected": 6}}, root=tmp_path)["status"] == "verified"
    assert verify_claim({"claim": "three lines", "check": {
        "kind": "count_eq", "path": "notes.md", "measure": "lines",
        "expected": 3}}, root=tmp_path)["observed"] == 3
    assert verify_claim({"claim": "no 'zzz'", "check": {
        "kind": "count_eq", "path": "notes.md", "measure": "matches",
        "pattern": "zzz", "expected": 0}}, root=tmp_path)["status"] == "verified"


def test_count_eq_on_a_missing_directory_is_refuted_with_a_zero(tmp_path):
    """A missing output directory does not make "25 rows were written"
    unevaluable — it makes it false, and the ledger has to say so."""
    v = verify_claim({"claim": "25 rows written", "check": {
        "kind": "count_eq", "path": "nope", "measure": "files",
        "expected": 25}}, root=tmp_path)
    assert v["status"] == "refuted" and v["observed"] == 0


def test_count_eq_needs_a_number_and_a_known_measure(tmp_path):
    assert verify_claim({"claim": "size", "check": {
        "kind": "count_eq", "path": "x", "measure": "bytes",
        "expected": "big"}}, root=tmp_path)["status"] == "insufficient"
    assert verify_claim({"claim": "size", "check": {
        "kind": "count_eq", "path": "x", "measure": "words",
        "expected": 3}}, root=tmp_path)["status"] == "insufficient"


def test_count_eq_tolerates_an_explicit_slack(tmp_path):
    (tmp_path / "n.txt").write_text("aaaa")
    assert verify_claim({"claim": "~4 bytes", "check": {
        "kind": "count_eq", "path": "n.txt", "measure": "bytes",
        "expected": 5, "tolerance": 1}}, root=tmp_path)["status"] == "verified"


# ── json_key / regex ───────────────────────────────────────────────────────

def test_json_key_verifies_compares_and_refutes(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps(
        {"counters": {"written": 25, "runs": [1, 2, 3]}}))
    assert verify_claim({"claim": "25 written", "check": {
        "kind": "json_key", "path": "state.json", "key": "counters.written",
        "expected": 25}}, root=tmp_path)["status"] == "verified"

    wrong = verify_claim({"claim": "96 written", "check": {
        "kind": "json_key", "path": "state.json", "key": "counters.written",
        "expected": 96}}, root=tmp_path)
    assert wrong["status"] == "refuted" and wrong["observed"] == 25

    assert verify_claim({"claim": "third run is 3", "check": {
        "kind": "json_key", "path": "state.json", "key": "counters.runs.2",
        "expected": 3}}, root=tmp_path)["status"] == "verified"

    assert verify_claim({"claim": "absent key", "check": {
        "kind": "json_key", "path": "state.json", "key": "counters.missing"}},
        root=tmp_path)["status"] == "refuted"


def test_unparseable_json_is_insufficient_not_refuted(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    v = verify_claim({"claim": "written=25", "check": {
        "kind": "json_key", "path": "state.json", "key": "written",
        "expected": 25}}, root=tmp_path)
    assert v["status"] == "insufficient"
    assert "not valid JSON" in v["detail"]


def test_regex_match_and_absent_both_bite(tmp_path):
    (tmp_path / "health.md").write_text("| Total relationships | 0 |\n")
    assert verify_claim({"claim": "zero relationships", "check": {
        "kind": "regex", "path": "health.md",
        "pattern": r"^\| Total relationships \| 0 \|$"}},
        root=tmp_path)["status"] == "verified"

    bad = verify_claim({"claim": "12,131 relationships", "check": {
        "kind": "regex", "path": "health.md",
        "pattern": r"^\| Total relationships \| 12,?131 \|$"}},
        root=tmp_path)
    assert bad["status"] == "refuted" and bad["observed"] == 0

    assert verify_claim({"claim": "no failures mentioned", "check": {
        "kind": "regex", "path": "health.md", "pattern": "Traceback",
        "expect": "absent"}}, root=tmp_path)["status"] == "verified"


def test_a_broken_pattern_is_the_verifiers_problem_not_the_models(tmp_path):
    (tmp_path / "a.md").write_text("x")
    v = verify_claim({"claim": "bad regex", "check": {
        "kind": "regex", "path": "a.md", "pattern": "(["}}, root=tmp_path)
    assert v["status"] == "insufficient" and "invalid regex" in v["detail"]


# ── Bundle partition, rate, and the empty case ─────────────────────────────

def test_the_bundle_partitions_verified_against_gap(tmp_path):
    (tmp_path / "a.md").write_text("hi")
    b = verify_bundle([
        {"claim": "a.md exists", "check": {"kind": "file_exists", "path": "a.md"}},
        {"claim": "b.md exists", "check": {"kind": "file_exists", "path": "b.md"}},
        {"claim": "uncheckable thing", "check": {"kind": "trust_me"}},
    ], root=tmp_path)

    assert b["verified"] == ["a.md exists"]
    assert len(b["gap"]) == 2
    assert b["gap"][0].startswith("[refuted] b.md exists")
    assert b["counts"] == {"total": 3, "verified": 1, "refuted": 1,
                           "insufficient": 1}
    assert b["refuted_or_insufficient_rate"] == pytest.approx(0.667, abs=0.01)


def test_a_run_that_asserted_nothing_reports_a_gap_and_no_rate(tmp_path):
    """Zero claims must not read as zero refutations. This is the class of bug
    MEMORY.md records three times: a gate that reads its own missing input as a
    pass (`graph-baseline.json`, `_is_dependency_met`, the dream lock)."""
    b = verify_bundle([], root=tmp_path)
    assert b["counts"]["total"] == 0
    assert b["refuted_or_insufficient_rate"] is None
    assert b["gap"] and "nothing this run asserted" in b["gap"][0]


def test_gap_entries_are_bounded(tmp_path):
    claims = [{"claim": f"claim {i} " + "x" * 600,
               "check": {"kind": "file_exists", "path": "nope.md"}}
              for i in range(30)]
    b = verify_bundle(claims, root=tmp_path)
    assert len(b["gap"]) <= evidence.MAX_GAP_ITEMS
    assert all(len(g) <= evidence.MAX_GAP_CHARS for g in b["gap"])


# ── Parsing the model's claims out of its answer ───────────────────────────

def test_the_evidence_fence_is_parsed_out_of_a_long_answer():
    response = ("Scanned 3 messages.\n\n"
                "```evidence\n"
                '{"claims": [{"claim": "3 scanned", "check": '
                '{"kind": "count_eq", "path": "x", "measure": "bytes", '
                '"expected": 1}}]}\n'
                "```\n\nDone.\n")
    claims = parse_claims_block(response)
    assert len(claims) == 1
    assert claims[0]["check"]["kind"] == "count_eq"


def test_a_claims_block_that_does_not_parse_becomes_a_gap_not_silence(tmp_path):
    claims = parse_claims_block("```evidence\n{\"claims\": [oops\n```")
    assert len(claims) == 1
    assert verify_claim(claims[0], root=tmp_path)["status"] == "insufficient"


def test_a_bare_json_object_and_a_plain_list_both_parse():
    assert len(parse_claims_block(
        '```evidence\n{"claims": {"claim": "x", "check": {"kind": "regex"}}}\n```'
    )) == 1
    assert len(parse_claims_block(
        '```evidence\n[{"claim": "x", "check": {"kind": "regex"}}]\n```'
    )) == 1


def test_no_claims_block_is_an_empty_list():
    assert parse_claims_block("Everything went well, 25 rows written.") == []


def test_the_gap_block_is_appended_only_when_there_are_gaps():
    assert gaps_prompt([]) == ""
    assert gaps_prompt(None) == ""
    block = gaps_prompt(["[refuted] 96 files — disk says 192"])
    assert "Evidence gaps carried from your previous run" in block
    assert "[refuted] 96 files" in block


def test_a_corrupt_stored_gap_watermark_is_no_gaps_not_a_crash():
    assert parse_gap_list(None) == []
    assert parse_gap_list("{not json") == []
    assert parse_gap_list('["a", "b"]') == ["a", "b"]


# ── The ledger row ─────────────────────────────────────────────────────────

def test_normalize_result_carries_claims_without_touching_the_summary():
    """The prose summary stays for humans (acceptance: bundle *alongside* it)."""
    norm = normalize_result(_item(payload={"task_id": 38}), {
        "status": "success", "summary": "25 scanned, 25 written",
        "claims": [{"claim": "25 written", "check": {"kind": "regex"}}]})
    assert norm["summary"] == "25 scanned, 25 written"
    assert len(norm["claims"]) == 1

    assert normalize_result(_item(), {"status": "success",
                                      "summary": "x"})["claims"] is None


def test_the_runs_table_has_a_claims_column(tmp_path):
    q = WorkQueue(tmp_path / "w.db")
    cols = {r[1] for r in q._connect().execute("PRAGMA table_info(runs)")}
    assert "claims_json" in cols


def test_record_run_stores_the_bundle(tmp_path):
    q = WorkQueue(tmp_path / "w.db")
    (tmp_path / "x").write_text("here")
    bundle = verify_bundle([{"claim": "x exists",
                             "check": {"kind": "file_exists", "path": "x"}}],
                           root=tmp_path)
    q.record_run(run_id="r1", queue_id=None, source="scheduled-task",
                 status="success", started_at="now", completed_at="now",
                 duration_seconds=1.0, summary="did the thing",
                 task_id="38", claims_json=json.dumps(bundle))
    row = q.list_runs(source="scheduled-task")[0]
    stored = json.loads(row["claims_json"])
    assert row["summary"] == "did the thing"
    assert stored["verified"] == ["x exists"]


async def test_the_pool_verifies_at_ledger_write_and_carries_the_gap(tmp_path,
                                                                    monkeypatch):
    """The end-to-end contract: bundle in the run record next to the summary,
    and the gap list stored under the task id for its successor's prompt."""
    import workers.sources as sources

    q = WorkQueue(tmp_path / "w.db")
    (tmp_path / "artifact.md").write_text("written")
    monkeypatch.setattr(evidence, "default_root", lambda: tmp_path)

    class PilotSource:
        NAME = "scheduled-task"

        @staticmethod
        async def enqueue_if_due(queue, cfg):
            return None

        @staticmethod
        async def execute(item):
            return {
                "status": "success",
                "summary": "25 scanned, 25 written to the vault",
                "task_id": "38",
                "claims": [
                    {"claim": "the artifact was written",
                     "check": {"kind": "file_exists", "path": "artifact.md"}},
                    {"claim": "25 files landed in the outbox",
                     "check": {"kind": "count_eq", "path": "outbox",
                               "measure": "files", "expected": 25}},
                ],
            }

    monkeypatch.setitem(sources.SOURCE_REGISTRY, "scheduled-task", PilotSource)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"scheduled-task": {"max_duration_seconds": 60}})
    q.enqueue(source="scheduled-task", kind="run", payload={"task_id": 38})

    pool = WorkerPool(q, slots=1)
    pool._running = True
    worker = asyncio.create_task(pool._worker_loop("worker-0"))
    for _ in range(60):
        await asyncio.sleep(0.1)
        if q.list_runs(source="scheduled-task"):
            break
    pool._running = False
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    row = q.list_runs(source="scheduled-task")[0]
    assert row["summary"].startswith("25 scanned")          # prose survives
    bundle = json.loads(row["claims_json"])
    assert bundle["verified"] == ["the artifact was written"]
    assert bundle["counts"] == {"total": 2, "verified": 1, "refuted": 1,
                                "insufficient": 0}
    assert bundle["refuted_or_insufficient_rate"] == 0.5

    carried = parse_gap_list(q.wm_get("scheduled-task", gaps_key(38)))
    assert len(carried) == 1
    assert carried[0].startswith("[refuted] 25 files landed in the outbox")


async def test_a_non_pilot_run_gets_no_bundle_rather_than_an_empty_one(
        tmp_path, monkeypatch):
    """Sources outside the pilot emit no claims; writing an empty bundle for
    them would turn "not yet piloted" into "checked and clean"."""
    import workers.sources as sources

    q = WorkQueue(tmp_path / "w.db")
    monkeypatch.setattr(evidence, "default_root", lambda: tmp_path)

    class PlainSource:
        NAME = "session-distill"

        @staticmethod
        async def enqueue_if_due(queue, cfg):
            return None

        @staticmethod
        async def execute(item):
            return {"status": "success", "summary": "distilled 3 sessions"}

    monkeypatch.setitem(sources.SOURCE_REGISTRY, "session-distill", PlainSource)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"session-distill": {"max_duration_seconds": 60}})
    q.enqueue(source="session-distill", kind="run", payload={})

    pool = WorkerPool(q, slots=1)
    pool._running = True
    worker = asyncio.create_task(pool._worker_loop("worker-0"))
    for _ in range(60):
        await asyncio.sleep(0.1)
        if q.list_runs(source="session-distill"):
            break
    pool._running = False
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    assert q.list_runs(source="session-distill")[0]["claims_json"] is None


# ── The source adapter: autonomy run → pool (backlog #945) ─────────────────
#
# Everything above this line builds the source's return dict by hand and hands
# it straight to `normalize_result`, which is why the whole suite stayed green
# while `scheduled_task.execute()` — the adapter that actually produces that
# dict for every autonomy run — dropped `claims` on the floor: a hand-built
# dict cannot demonstrate that a key survives a function it never passed
# through. The tests below call the real `execute()`, and one of them the real
# pool, over a real pilot artifact.

# Byte-identical mirror of `autonomy-runs/38/run_38_20260916_050201.md`, which
# is gitignored (`/autonomy-runs/`) and so is not part of the tree. Its fenced
# ```evidence block carries the 7 claims that run emitted; every count asserted
# below is measured from this file through the real parser, not quoted from a
# report.
PILOT_ARTIFACT = FIXTURES / "pilot-run-38-20260916_050201.md"


def _pilot_artifact_text() -> str:
    text = PILOT_ARTIFACT.read_text(encoding="utf-8")
    # Positive control, because a 0-length parse reads the same two ways: "the
    # block is gone" and "this is not the file this section thinks it is". If
    # the fixture is ever swapped or truncated, the assertions below would
    # quietly assert about zero claims instead of failing on the fixture.
    assert f"```{evidence.FENCE_TAG}" in text
    return text


def _pilot_claims() -> list[dict]:
    """The claims the fixture's run emitted, parsed by the real parser."""
    return autonomy._evidence_claims(_pilot_artifact_text())


def _fake_run_task(final_response: str, *, pilot: bool):
    """Stand in for `autonomy.run_task` with its real success shape.

    `claims` is attached the way `run_task` attaches it — by calling the real
    `_evidence_claims` on the run's final text, and only when the task is in
    `EVIDENCE_PILOT_TASK_IDS` — so `pilot=False` reproduces the non-pilot
    return exactly: no `claims` key at all.
    """
    async def _run(task_id, max_duration=None):
        result = {
            "success": True, "status": "success", "task_id": int(task_id),
            "run_id": f"run_{task_id}_20260916_050201",
            "duration_seconds": 296.1,
            "response_preview": final_response[:300],
            "meta": {},
        }
        if pilot:
            assert int(task_id) in autonomy.EVIDENCE_PILOT_TASK_IDS
            result["claims"] = autonomy._evidence_claims(final_response)
        else:
            assert int(task_id) not in autonomy.EVIDENCE_PILOT_TASK_IDS
        return result

    return _run


def _stub_adapter(monkeypatch, *, final_response: str, pilot: bool,
                  task_id: int = 38) -> None:
    """Replace everything `execute()` does around the dict it returns.

    The vLLM health probe, the vault task-file read and Discord notify on the
    success path, and the pool's per-source config. Nothing here patches the
    return dict itself — that is the code under test. Leaves the real
    `scheduled-task` entry in `SOURCE_REGISTRY`, so the pool dispatches to the
    real adapter too.
    """
    import app.discord_notify as discord_notify
    import workers.sources as sources
    import workers.sources.scheduled_task as scheduled_task

    monkeypatch.setattr(scheduled_task, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: None)
    monkeypatch.setattr(autonomy, "run_task",
                        _fake_run_task(final_response, pilot=pilot))
    monkeypatch.setattr(sources, "get_sources_config", lambda: {})

    async def _no_notify(*a, **k):
        return None

    monkeypatch.setattr(discord_notify, "_discord_notify_task_complete", _no_notify)


async def _drive_real_execute(monkeypatch, *, final_response: str, pilot: bool,
                             task_id: int = 38) -> dict:
    """Call the real `scheduled_task.execute()` and return what the pool receives."""
    _stub_adapter(monkeypatch, final_response=final_response, pilot=pilot,
                  task_id=task_id)
    import workers.sources.scheduled_task as scheduled_task
    return await scheduled_task.execute(
        _item(payload={"task_id": task_id}, source="scheduled-task"))


async def test_execute_forwards_the_claims_a_pilot_run_emitted(monkeypatch):
    """#945 clause 1: `out["claims"]` is the list `run_task` attached."""
    claims = _pilot_claims()
    assert len(claims) == 7
    out = await _drive_real_execute(monkeypatch,
                                    final_response=_pilot_artifact_text(),
                                    pilot=True)
    assert out["claims"] == claims
    # Same dict, not a replacement: the rest of the whitelist still behaves.
    assert out["status"] == "success"
    assert out["task_id"] == "38"
    assert out["artifact_path"] == "autonomy-runs/38/run_38_20260916_050201.md"


async def test_execute_adds_no_claims_key_for_a_task_outside_the_pilot(monkeypatch):
    """#945 clause 1's other half. `run_task` omits the key for a non-pilot
    task, and the adapter must not invent one: presence is the pool's scope
    switch, so inventing it would score an un-piloted task as checked."""
    out = await _drive_real_execute(monkeypatch,
                                    final_response=_pilot_artifact_text(),
                                    pilot=False, task_id=24)
    assert "claims" not in out
    assert out["task_id"] == "24"


async def test_the_pilot_claims_survive_from_the_source_into_the_pool(monkeypatch):
    """#945 clause 2: the key is still present, and intact, one hop later."""
    claims = _pilot_claims()
    out = await _drive_real_execute(monkeypatch,
                                    final_response=_pilot_artifact_text(),
                                    pilot=True)
    norm = normalize_result(_item(payload={"task_id": 38}, source="scheduled-task"),
                            out)
    assert norm["claims"] == claims
    assert len(norm["claims"]) == 7
    assert {k for c in norm["claims"] for k in c} == {"claim", "check"}


async def test_a_pilot_that_emitted_nothing_normalizes_to_a_list_not_none(monkeypatch):
    """#945 clause 3. `[]` and `None` are different facts: an empty list is a
    piloted run whose model asserted nothing — a visible gap, recorded as a
    bundle — while `None` is a run outside the pilot, which must not be scored
    as a clean check."""
    report_without_block = _pilot_artifact_text().split(
        f"\n```{evidence.FENCE_TAG}")[0]
    assert autonomy._evidence_claims(report_without_block) == []   # the split bit

    pilot_norm = normalize_result(
        _item(payload={"task_id": 38}, source="scheduled-task"),
        await _drive_real_execute(monkeypatch,
                                  final_response=report_without_block, pilot=True))
    assert pilot_norm["claims"] == []

    other_norm = normalize_result(
        _item(payload={"task_id": 24}, source="scheduled-task"),
        await _drive_real_execute(monkeypatch,
                                  final_response=report_without_block,
                                  pilot=False, task_id=24))
    assert other_norm["claims"] is None


async def test_a_pilot_run_reaches_the_ledger_with_a_bundle(tmp_path, monkeypatch):
    """The acceptance check for #945, read off the row the health view counts.

    `verify_bundle` runs against `tmp_path`, which holds none of the artifacts
    the fixture claims, so all 7 checks refute — the point is not the verdict
    but that 7 claims reached the stdlib verifier and were written to the run
    record at all, which no scheduled-task row had done in 4,900+ runs.
    """
    q = WorkQueue(tmp_path / "w.db")
    monkeypatch.setattr(evidence, "default_root", lambda: tmp_path)
    # Stubs only the adapter's I/O and `autonomy.run_task`; the pool looks
    # `scheduled-task` up in the real registry, so `execute()` runs for real.
    _stub_adapter(monkeypatch, final_response=_pilot_artifact_text(), pilot=True)
    q.enqueue(source="scheduled-task", kind="run", payload={"task_id": 38})

    pool = WorkerPool(q, slots=1)
    pool._running = True
    worker = asyncio.create_task(pool._worker_loop("worker-0"))
    for _ in range(60):
        await asyncio.sleep(0.1)
        if q.list_runs(source="scheduled-task"):
            break
    pool._running = False
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    row = q.list_runs(source="scheduled-task")[0]
    bundle = json.loads(row["claims_json"])
    assert bundle["counts"] == {"total": 7, "verified": 0, "refuted": 7,
                                "insufficient": 0}
    assert bundle["refuted_or_insufficient_rate"] == 1.0
    assert row["summary"]                                   # prose survives too
    # The side-signal #945 records: `_carry_gaps` only fires when a bundle is
    # built, which is why the watermark table held only `last_enqueue_check`.
    carried = parse_gap_list(q.wm_get("scheduled-task", gaps_key(38)))
    assert len(carried) == 7


# ── Pilot scoping and prompt carry-forward ─────────────────────────────────

def test_the_pilot_is_the_reflection_chain_and_nothing_else():
    assert autonomy.EVIDENCE_PILOT_TASK_IDS == frozenset({38, 42, 39, 40})
    for tid in (38, 42, 39, 40):
        block = autonomy._evidence_prompt(tid, gaps=[])
        assert "Evidence claims" in block
        assert evidence.FENCE_TAG in block
    assert autonomy._evidence_prompt(24, gaps=[]) == ""
    assert autonomy._evidence_prompt(None, gaps=[]) == ""


def test_the_gap_list_reaches_the_next_run_of_the_same_task():
    block = autonomy._evidence_prompt(
        39, gaps=["[refuted] 96 files — disk says 192"])
    assert "Evidence claims" in block                  # static half first
    assert block.index("Evidence claims") < block.index("Evidence gaps")
    assert "[refuted] 96 files" in block
    assert autonomy._evidence_prompt(39, gaps=[]) == evidence.CLAIMS_INSTRUCTION


def test_run_task_appends_the_evidence_section_to_the_pilot_prompt(monkeypatch,
                                                                   tmp_path):
    """The prompt the model actually receives carries the block, appended — the
    system prompt is untouched, so the cached prefix is untouched (#520)."""
    captured: dict = {}

    task = {"id": 38, "name": "Signals", "skill_name": "nightly-reflection-signals",
            "status": "up_next", "timeout_seconds": 300}
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "38.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_evidence_gap_list", lambda tid: ["[refuted] x"])
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})

    async def _boom(messages, options):
        captured["prompt"] = messages[0]["content"]
        captured["anchor"] = options.state_anchor
        yield {"type": "text_delta", "text": "all good\n\n"
               "```evidence\n{\"claims\": [{\"claim\": \"the artifact exists\", "
               "\"check\": {\"kind\": \"file_exists\", \"path\": \"a.md\"}}]}\n```"}
        yield {"type": "result", "stop_reason": "stop", "usage": None,
               "num_turns": 3}
        return

    class Opts:
        def __init__(self, **kw):
            captured["model"] = kw.get("model")
            self.state_anchor = kw.get("state_anchor")

    import app.harness as harness
    import app.harness.mcp_pool as mcp_pool
    monkeypatch.setattr(harness, "run_query", _boom)
    monkeypatch.setattr(harness, "RunOptions", Opts)
    monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")
    monkeypatch.setattr(autonomy, "_write_run_record", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)

    out = asyncio.run(autonomy.run_task(38))
    assert "Evidence claims" in captured["prompt"]
    assert "Evidence gaps carried from your previous run" in captured["prompt"]
    assert captured["prompt"].index("SKILL BODY") < captured["prompt"].index(
        "Evidence claims")
    # The run's own claims travel with the result for the pool to verify.
    assert out["claims"][0]["check"]["kind"] == "file_exists"
    assert out["response_preview"].startswith("all good")


def test_a_pilot_run_that_emitted_no_claims_still_carries_a_bundle(monkeypatch,
                                                                   tmp_path):
    """"Every completed run in that set carries a claims bundle" — including the
    one where the model ignored the instruction. That is a gap, not an absence."""
    task = {"id": 40, "name": "Config", "skill_name": "nightly-reflection-config",
            "status": "up_next", "timeout_seconds": 300}
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "40.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_evidence_gap_list", lambda tid: [])
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})

    async def _quiet(messages, options):
        yield {"type": "text_delta", "text": "[SILENT]"}
        yield {"type": "result", "stop_reason": "stop", "usage": None,
               "num_turns": 2}
        return

    class Opts:
        def __init__(self, **kw):
            self.state_anchor = kw.get("state_anchor")

    import app.harness as harness
    import app.harness.mcp_pool as mcp_pool
    monkeypatch.setattr(harness, "run_query", _quiet)
    monkeypatch.setattr(harness, "RunOptions", Opts)
    monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")
    monkeypatch.setattr(autonomy, "_write_run_record", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)

    out = asyncio.run(autonomy.run_task(40))
    assert out["claims"] == []


# ── Per-task reporting: [SILENT] runs cannot dilute the rate ───────────────

def _claims_row(status_counts: dict | None, **row_kw) -> dict:
    if status_counts is None:
        claims_json = None
    else:
        claims = [{"claim": f"c{i}", "check": {}, "status": s}
                  for i, s in enumerate(status_counts)]
        claims_json = json.dumps({"claims": claims, "verified": [],
                                  "gap": ["[refuted] c0"] if "refuted" in status_counts else [],
                                  "counts": {
                                      "total": len(status_counts),
                                      "verified": status_counts.count("verified"),
                                      "refuted": status_counts.count("refuted"),
                                      "insufficient": status_counts.count("insufficient")}})
    base = {"task_id": "38", "status": "success", "duration_seconds": 20.0,
            "summary": "done", "response_json": "all good", "meta_json": None,
            "claims_json": claims_json,
            "completed_at": "2026-09-09T06:00:00+00:00"}
    base.update(row_kw)
    return base


def test_health_reports_the_refuted_rate_per_task_id():
    rows = [_claims_row(["verified", "refuted"]), _claims_row(["insufficient"])]
    tasks = [{"id": 38, "name": "Signals", "status": "up_next"}]
    t = autonomy.compute_health(rows, tasks, 7)["tasks"][0]

    assert t["claims_checked"] == 3
    assert t["claims_verified"] == 1 and t["claims_refuted"] == 1
    assert t["claims_insufficient"] == 1
    assert t["refuted_or_insufficient_rate"] == pytest.approx(0.667, abs=0.01)
    assert t["runs_with_bundle"] == 2 and t["runs_without_bundle"] == 0


def test_a_silent_run_cannot_dilute_its_tasks_rate():
    """Denominator is claims checked, not runs. Two clean claim-bearing runs
    plus a [SILENT] run that asserted nothing stays at 0.0, and the silent run
    is visible as a run without a bundle rather than as a clean check."""
    rows = [
        _claims_row(["verified"]), _claims_row(["verified"]),
        _claims_row(None, summary="[SILENT]", response_json="[SILENT]",
                    meta_json=json.dumps({"silent": True})),
    ]
    tasks = [{"id": 38, "name": "Signals", "status": "up_next"}]
    t = autonomy.compute_health(rows, tasks, 7)["tasks"][0]

    assert t["silent"] == 1
    assert t["runs_without_bundle"] == 1
    assert t["claims_checked"] == 2
    assert t["refuted_or_insufficient_rate"] == 0.0


def test_a_task_with_no_claims_at_all_reads_as_unevaluable_not_clean():
    rows = [_claims_row(None)]
    tasks = [{"id": 39, "name": "Knowledge Write", "status": "up_next"}]
    t = autonomy.compute_health(rows, tasks, 7)["tasks"][0]
    assert t["claims_checked"] == 0
    assert t["refuted_or_insufficient_rate"] is None


def test_the_fleet_block_totals_the_evidence_counts():
    rows = [_claims_row(["verified", "refuted"]), _claims_row(None)]
    tasks = [{"id": 38, "name": "Signals", "status": "up_next"}]
    h = autonomy.compute_health(rows, tasks, 7)
    assert h["fleet"]["claims_checked"] == 2
    assert h["fleet"]["claims_refuted"] == 1
    assert h["fleet"]["runs_without_bundle"] == 1
    assert h["fleet"]["refuted_or_insufficient_rate"] == 0.5


def test_health_survives_a_malformed_claims_column():
    rows = [_claims_row(None)]
    rows[0]["claims_json"] = "{this is not json"
    tasks = [{"id": 38, "name": "Signals", "status": "up_next"}]
    t = autonomy.compute_health(rows, tasks, 7)["tasks"][0]
    assert t["runs_without_bundle"] == 1


# ── The reproducing check: seeded report-vs-disk cases ─────────────────────

SEED_CLAIMS = {
    # handoff 2026-08-24: "graph restored to 12,131 relationships"
    "relationships_12131": {
        "claim": "the entity graph was restored to 12,131 relationships",
        "check": {"kind": "regex",
                  "path": "_pipeline/reflection/knowledge-health-2026-08-24.md",
                  "pattern": r"^\| (?:Total|Active) relationships \| 12,?131 \|$",
                  "expect": "match"},
    },
    # handoff 2026-09-03: "96 files total" of hypothesis_fail_*.txt (disk: 121)
    "hypothesis_fail_96": {
        "claim": "there are 96 hypothesis-failure dumps on disk",
        "check": {"kind": "count_eq", "path": "_pipeline/research/_debug",
                  "glob": "hypothesis_fail_*.txt", "measure": "files",
                  "expected": 96},
    },
    # handoff 2026-09-04: signals-latest.md "307KB / 383,722 chars" (disk: 13,503 B)
    "signals_latest_307kb": {
        "claim": "signals-latest.md is 307 KB",
        "check": {"kind": "count_eq",
                  "path": "_pipeline/reflection/signals-latest.md",
                  "measure": "bytes", "expected": 307_000, "tolerance": 10_000},
    },
}


def test_a_seeded_report_versus_disk_case_replays_as_refuted():
    """The acceptance check. Each seed is a claim a nightly handoff really made;
    the verifier binds it to the artifact the claim was about and re-runs it
    against whatever is on disk now. Zero refutations would mean the claims are
    too weak, not that the system is clean — so all three must come back
    refuted."""
    statuses = {}
    for name, seed in SEED_CLAIMS.items():
        rel = seed["check"]["path"]
        root = _seed_root_for(rel)
        assert root is not None, (
            f"seed {name}: no copy of {rel} in the tree or in the fixture "
            f"mirror — the replay would be vacuous")
        statuses[name] = verify_claim(seed, root=root)["status"]

    assert statuses == {"relationships_12131": "refuted",
                        "hypothesis_fail_96": "refuted",
                        "signals_latest_307kb": "refuted"}


def test_the_seeds_come_from_real_handoff_text():
    """If a seed stops matching the handoff it was quoted from, the replay above
    is testing a claim nobody made — so pin the provenance too."""
    rel = "_pipeline/reflection/knowledge-handoff-2026-08-24.md"
    root = _seed_root_for(rel)
    assert root is not None, f"no copy of {rel} in the tree or the fixture mirror"
    assert "12,131" in (root / rel).read_text(errors="replace")


# ── Cross-task / cross-artifact rollup (backlog #713) ────────────────────────
#
# `compute_health` reports evidence per task id, which cannot answer the
# corrections-log question: *which artifact has this fleet repeatedly
# misreported about?* The refutations about one path are spread over every task
# that claimed something about it — #39's claim about
# `_pipeline/reflection/signals-latest.md` and #38's disproval of it are two
# rows in two tasks, and each refutation only ever reached its own task's next
# prompt through the `evidence_gaps:<id>` watermark. These tests pin the rollup
# that puts them in one entry, over the same rows the per-task sums read.

def _artifact_row(task_id: str, specs, **row_kw) -> dict:
    """One run row whose bundle asserts claims about artifact paths.

    `specs` is a list of `(status, path)` pairs. Three distinct inputs, three
    distinct meanings, which is the whole distinction #525 drew and #713 has to
    keep: a list of claims is a run that asserted things, `[]` is a piloted run
    whose model asserted nothing (a bundle, and an empty one), and `None` is a
    run that carried no bundle at all — the case `runs_without_bundle` counts
    and the unevaluable rule exists for.
    """
    if specs is None:
        claims_json = None
    else:
        claims = [{"claim": f"{status} claim about {path}",
                   "check": {"kind": "file_exists", "path": path},
                   "status": status} for status, path in specs]
        claims_json = json.dumps({
            "claims": claims,
            "verified": [c["claim"] for c in claims
                         if c["status"] == "verified"],
            "gap": [f"[{c['status']}] {c['claim']}" for c in claims
                    if c["status"] in ("refuted", "insufficient")],
            "counts": {"total": len(claims),
                       "verified": sum(1 for c in claims
                                       if c["status"] == "verified"),
                       "refuted": sum(1 for c in claims
                                      if c["status"] == "refuted"),
                       "insufficient": sum(1 for c in claims
                                           if c["status"] == "insufficient")},
            "refuted_or_insufficient_rate": None,
        })
    base = {"task_id": task_id, "status": "success", "duration_seconds": 20.0,
            "summary": "done", "response_json": "all good", "meta_json": None,
            "claims_json": claims_json,
            "completed_at": "2026-09-24T06:00:00+00:00"}
    base.update(row_kw)
    return base


def test_the_rollup_groups_refuted_claims_by_artifact_worst_first():
    """#713 clause 3, first half: group by `check.path`, most-misreported first."""
    rows = [
        _artifact_row("39", [("refuted", "_pipeline/reflection/signals-latest.md"),
                             ("insufficient", "_pipeline/reflection/signals-latest.md"),
                             ("verified", "_pipeline/reflection/signals-latest.md")]),
        _artifact_row("38", [("refuted", "_pipeline/research/_debug/hypothesis_fail.txt"),
                             ("verified", "_pipeline/research/_debug/hypothesis_fail.txt"),
                             ("verified", "_pipeline/research/_debug/hypothesis_fail.txt")]),
        # A clean artifact: checked, never wrong. It belongs in the denominator
        # of "how much did we look at" and not in the list of findings.
        _artifact_row("40", [("verified", "_pipeline/reflection/config-latest.md")]),
    ]
    r = autonomy.claim_artifact_rollup(rows)

    assert r["artifacts_checked"] == 3, "all three paths were claimed about"
    assert r["artifacts_with_refutations"] == 2
    assert [e["path"] for e in r["entries"]] == [
        "_pipeline/reflection/signals-latest.md",
        "_pipeline/research/_debug/hypothesis_fail.txt"]

    worst = r["entries"][0]
    assert worst["claims_refuted"] == 1
    assert worst["claims_insufficient"] == 1
    assert worst["refuted_or_insufficient"] == 2
    assert worst["claims_checked"] == 3
    assert worst["refuted_or_insufficient_rate"] == pytest.approx(0.667, abs=0.01)


def test_the_rollup_orders_by_refutation_count_not_by_ratio():
    """Two refutations out of five claims outranks one out of one.

    The list a reader came for is the artifact the fleet gets wrong most often,
    not the one with the prettiest ratio — a 1-of-1 is usually a brand-new
    claim, and a 2-of-5 is a pattern.
    """
    rows = [
        _artifact_row("39", [("refuted", "a.md"), ("refuted", "a.md"),
                             ("verified", "a.md"), ("verified", "a.md"),
                             ("verified", "a.md")]),
        _artifact_row("40", [("refuted", "b.md")]),
    ]
    entries = autonomy.claim_artifact_rollup(rows)["entries"]
    assert [e["path"] for e in entries] == ["a.md", "b.md"]
    assert entries[0]["refuted_or_insufficient_rate"] == pytest.approx(0.4, abs=0.001)
    assert entries[1]["refuted_or_insufficient_rate"] == 1.0


def test_an_artifact_two_tasks_claimed_about_names_both_task_ids():
    """#713 clause 3, cross-task half — the reason the rollup exists.

    #39 asserted something about `signals-latest.md`; #38's run is what would
    later prove it false. Under the per-task watermark each refutation reaches
    only its own task's next prompt, so the pair never meets. In the rollup one
    entry carries both, and its `per_task` map says which half is whose.
    """
    rows = [
        _artifact_row("39", [("refuted", "_pipeline/reflection/signals-latest.md")]),
        _artifact_row("38", [("verified", "_pipeline/reflection/signals-latest.md")]),
    ]
    entry = autonomy.claim_artifact_rollup(rows)["entries"][0]

    assert entry["path"] == "_pipeline/reflection/signals-latest.md"
    assert entry["task_ids"] == ["38", "39"]
    assert set(entry["per_task"]) == {"38", "39"}
    assert entry["per_task"]["39"]["claims_refuted"] == 1
    assert entry["per_task"]["38"]["claims_refuted"] == 0
    assert entry["claims_checked"] == 2
    assert entry["refuted_or_insufficient"] == 1


def test_the_rollup_rate_divides_by_claims_and_never_by_runs():
    """#713 clause 4, first half — #525's denominator rule, re-held one layer up.

    Three piloted runs that asserted nothing sit beside the one run that made
    two claims. All four carry a bundle, so none of them is excluded: the rate
    is 1-of-2 claims (0.5), where a run-level denominator would report 1-of-4
    (0.25) — which is exactly the dilution that lets a fleet look clean by
    saying less.
    """
    rows = [_artifact_row("39", [("refuted", "p.md"), ("verified", "p.md")])] + [
        _artifact_row("39", []) for _ in range(3)]
    r = autonomy.claim_artifact_rollup(rows)

    assert r["runs_with_bundle"] == 4 and r["runs_without_bundle"] == 0
    assert r["claims_checked"] == 2
    assert r["refuted_or_insufficient_rate"] == 0.5
    assert r["unevaluable"] is False


def test_a_window_mostly_without_bundles_reports_unevaluable_not_clean():
    """#713 clause 4, second half: coverage is printed, or the rate is a lie.

    Same two claims as the test above, but now three of the four runs carry no
    bundle at all. The rate would still arithmetic out to 0.5 — over a slice
    nobody chose — so the window reports `None` with the count that says why.
    The artifact entry keeps its own rate: those two claims really were made and
    really were checked, and that is a measurement even when the window-wide one
    is not.
    """
    rows = [_artifact_row("39", [("refuted", "p.md"), ("verified", "p.md")])] + [
        _artifact_row("39", None) for _ in range(3)]
    r = autonomy.claim_artifact_rollup(rows)

    assert r["unevaluable"] is True
    assert r["refuted_or_insufficient_rate"] is None
    assert r["runs_without_bundle"] == 3
    assert r["unevaluable_reason"].startswith("3 of 4 runs")
    assert r["entries"][0]["refuted_or_insufficient_rate"] == 0.5

    # And `compute_health`'s fleet block says the same thing over the same rows.
    # It used to sum the claims itself and divide whenever any existed, so the
    # live window on 2026-09-24 — 3 bundles against 46 bare rows — printed a
    # clean `fleet.refuted_or_insufficient_rate: 0.0` thirty characters away from
    # `artifacts.unevaluable: true`. Two answers to one question is the shape
    # this file keeps re-recording; the fleet block quotes the rollup now.
    h = autonomy.compute_health(rows, [{"id": 39, "name": "Knowledge Write",
                                        "status": "up_next"}], 7)
    assert h["fleet"]["refuted_or_insufficient_rate"] is None
    assert h["fleet"]["evidence_unevaluable_reason"] == r["unevaluable_reason"]


def test_the_rollup_per_task_totals_equal_compute_healths():
    """#713 clause 4, third half: one set of rows, one answer, two doors.

    The rows are deliberately awkward — a `skipped` row, a row with no
    `task_id`, a `claims_json` that is not JSON, and a claim whose `check` names
    no path — because every one of those is a place where a second loop over
    `rows` could have quietly disagreed with the first. A `skipped` row counted
    in one of them, or a malformed bundle treated as a bundle in one and not the
    other, is a fleet report whose two halves disagree about the same run.
    """
    rows = [
        _artifact_row("39", [("refuted", "p.md"), ("verified", "q.md")]),
        _artifact_row("38", [("verified", "p.md")]),
        _artifact_row("38", []),
        _artifact_row("39", None),
        # A claim whose check names a whitespace-only path: counted, unattributable.
        _artifact_row("39", [("refuted", "   "), ("insufficient", "p.md")]),
        _artifact_row("38", [("verified", "p.md")], status="skipped"),
        # No `task_id` anywhere on the row: both consumers must file it under the
        # same synthetic id, or the unattributed refutation lands in one half and
        # not the other.
        _artifact_row(None, [("refuted", "p.md")]),
    ]
    # ... and a `claims_json` that is not JSON: an unreadable bundle is "no
    # bundle", which both halves must count as `runs_without_bundle`.
    broken = _artifact_row("38", None)
    broken["claims_json"] = "{this is not json"
    rows.append(broken)
    tasks = [{"id": 38, "name": "Signals", "status": "up_next"},
             {"id": 39, "name": "Knowledge Write", "status": "up_next"}]

    h = autonomy.compute_health(rows, tasks, 7)
    r = h["artifacts"]

    assert r == autonomy.claim_artifact_rollup(rows), (
        "`compute_health`'s `artifacts` and the standalone query are two "
        "answers about one set of rows")
    for t in h["tasks"]:
        pt = r["per_task"][t["task_id"]]
        for k in ("claims_checked", "claims_verified", "claims_refuted",
                  "claims_insufficient", "runs_with_bundle",
                  "runs_without_bundle"):
            assert pt[k] == t[k], f"task {t['task_id']} field {k}"
    assert (r["claims_checked"], r["claims_refuted"], r["claims_insufficient"],
            r["runs_without_bundle"]) == (
        h["fleet"]["claims_checked"], h["fleet"]["claims_refuted"],
        h["fleet"]["claims_insufficient"], h["fleet"]["runs_without_bundle"])
    # The blank-path claim is counted but unattributable: in the totals, in the
    # per-task sums, named as `claims_without_path`, and in no entry. `q.md` was
    # claimed once and verified, so it is a checked artifact with nothing to
    # report — in the denominator, out of `entries`.
    assert r["claims_without_path"] == 1
    assert {e["path"] for e in r["entries"]} == {"p.md"}
    assert r["artifacts_checked"] == 2 and r["artifacts_with_refutations"] == 1
    # The `skipped` row is in neither half; the unattributed and the unparseable
    # ones are in both, the same way.
    assert r["per_task"]["unattributed"]["claims_refuted"] == 1
    assert r["per_task"]["38"]["runs_with_bundle"] == 2
    assert r["per_task"]["38"]["runs_without_bundle"] == 1
    assert r["per_task"]["39"]["claims_checked"] == 4
    assert r["claims_checked"] == 6 and r["claims_verified"] == 2
    assert r["refuted_or_insufficient_rate"] == pytest.approx(0.667, abs=0.001)
    # `p.md` is the artifact two tasks and one unattributed run all got wrong:
    # the row the per-task views could never have assembled.
    worst = r["entries"][0]
    assert worst["task_ids"] == ["38", "39", "unattributed"]
    assert worst["refuted_or_insufficient"] == 3 and worst["claims_checked"] == 4


async def test_a_piloted_run_that_asserted_nothing_still_lands_a_bundle_row(
        tmp_path, monkeypatch):
    """#713 clause 2, re-pinned at the row the rollup reads.

    #945 pinned `execute()` forwarding the key and `run_task` emitting `[]`, and
    #945's own ledger test drove seven claims end to end. What nothing pinned is
    the case this clause names: an EMPTY list through the real adapter, the real
    `normalize_result` and the real `record_run`. `[]` must survive as a bundle
    whose total is 0 — "this piloted run checked nothing" — and not collapse
    into NULL, which is the different fact "this run was never checked". The
    rollup treats those two as `runs_with_bundle` and `runs_without_bundle`
    respectively, so the collapse would move a gap into the clean column.
    """
    q = WorkQueue(tmp_path / "w.db")
    monkeypatch.setattr(evidence, "default_root", lambda: tmp_path)
    quiet = _pilot_artifact_text().split(f"\n```{evidence.FENCE_TAG}")[0]
    assert autonomy._evidence_claims(quiet) == []   # the split really did remove it
    _stub_adapter(monkeypatch, final_response=quiet, pilot=True)
    q.enqueue(source="scheduled-task", kind="run", payload={"task_id": 38})

    pool = WorkerPool(q, slots=1)
    pool._running = True
    worker = asyncio.create_task(pool._worker_loop("worker-0"))
    for _ in range(60):
        await asyncio.sleep(0.1)
        if q.list_runs(source="scheduled-task"):
            break
    pool._running = False
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

    row = q.list_runs(source="scheduled-task")[0]
    assert row["claims_json"] is not None, (
        "an empty claim list must not be flattened into a missing bundle")
    bundle = json.loads(row["claims_json"])
    assert bundle["counts"] == {"total": 0, "verified": 0, "refuted": 0,
                                "insufficient": 0}
    assert bundle["refuted_or_insufficient_rate"] is None
    # Read back the way the rollup reads it: a bundle, and a bundle with no
    # claims in it — so it lands in `runs_with_bundle`, never in the
    # `runs_without_bundle` count that can make a window unevaluable.
    assert autonomy._row_claims(row) is not None
    assert autonomy.claim_artifact_rollup([row])["runs_with_bundle"] == 1
    assert autonomy.claim_artifact_rollup([row])["runs_without_bundle"] == 0
