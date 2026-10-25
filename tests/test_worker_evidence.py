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

Since #2817 that re-run has one shape it refuses to grade: a `count_eq` over the whole
run-record corpus (`*/run_*.md` or `**/run_*.md` bound at `autonomy-runs`) is
`insufficient`, never `verified` and never `refuted`, because the graded run's own
record is written into that corpus before its claims are graded and the corpus grows at
roughly 37 records a day. What IS graded is the closed set — one task's directory with
a glob that crosses no separator — verified and refuted exactly as before, `tolerance`
included. The rule, its discriminator and its consequences are pinned in the `#2817`
section at the end of this file.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app import autonomy
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
    switch, so inventing it would score an un-piloted task as checked.

    Task #81 (QMD Index Maintenance), deliberately: #24 and #56 joined the pilot
    on 2026-10-14, so they can no longer stand for "outside it".
    """
    out = await _drive_real_execute(monkeypatch,
                                    final_response=_pilot_artifact_text(),
                                    pilot=False, task_id=81)
    assert "claims" not in out
    assert out["task_id"] == "81"


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
        _item(payload={"task_id": 81}, source="scheduled-task"),
        await _drive_real_execute(monkeypatch,
                                  final_response=report_without_block,
                                  pilot=False, task_id=81))
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

def test_the_pilot_is_the_reflection_chain_plus_the_digester_and_two_writers_and_nothing_else():
    """The set, and the instruction each member's run is actually handed.

    #53 joined on 2026-10-09 (#2482): its run `run_53_20261009_113528.md` filed a
    `priority: high` backlog item (#2478) quoting an audit database, a `device_events`
    table and "1804 rows, 27 positives" that exist nowhere on this box, cross-checked
    against an autonomy task archived two months earlier. Nothing in that run could
    have been refuted, because nothing was bound — `grep -c '```evidence'` on the run
    is 0. It is now a pilot member, so the same stdlib verifier that grades the
    reflection chain grades its numbers too.

    #24 and #56 joined on 2026-10-14 (#2675), the two writers #902 named as the same
    shape: a 6x-daily job whose activity line is `files_processed=133 facts=4969
    failed=1048`, and a daily job that reports how many sessions it wrote into one
    dated JSONL. Both had asserted a count every run with 0 blocks on disk.

    Asserted as an exact set, not a subset: after three widenings a subset-only check
    could no longer tell that a member had been dropped. And #81 stays out — its
    state is the qmd embedding index's own, which none of the four check kinds reads,
    so it is the negative control now that #24 is inside.
    """
    assert autonomy.EVIDENCE_PILOT_TASK_IDS == frozenset(
        {24, 38, 39, 40, 42, 53, 56})
    for tid in (24, 38, 39, 40, 42, 53, 56):
        assert autonomy._evidence_pilot(tid)
        block = autonomy._evidence_prompt(tid, gaps=[])
        assert "Evidence claims" in block
        assert evidence.FENCE_TAG in block
    # The writer still outside is #81, and a job that asserts nothing verifiable
    # must not be scored as checked-and-clean (#525).
    assert autonomy._evidence_prompt(81, gaps=[]) == ""
    assert autonomy._evidence_prompt(None, gaps=[]) == ""


def test_the_gap_list_reaches_the_next_run_of_the_same_task():
    block = autonomy._evidence_prompt(
        39, gaps=["[refuted] 96 files — disk says 192"])
    assert "Evidence claims" in block                  # static half first
    assert block.index("Evidence claims") < block.index("Evidence gaps")
    assert "[refuted] 96 files" in block
    assert autonomy._evidence_prompt(39, gaps=[]) == evidence.CLAIMS_INSTRUCTION


#: The #2478 filing, in the one shape the verifier can grade: a `file_exists` and a
#: `count_eq` over the wake-word audit database that has never existed on this box
#: (`ls -la ~/lloyd-data/wakeword-audit/` → `No such file or directory`). The row
#: count is the one that run's prose asserted.
FABRICATED_AUDIT_REPORT = (
    "Wake-word coverage is dead: the audit table holds 1804 rows and 27 positives, "
    "all 11 below threshold since the retrain.\n\n"
    f"```{evidence.FENCE_TAG}\n"
    '{"claims": ['
    '{"claim": "the wake-word audit database exists", '
    '"check": {"kind": "file_exists", "path": "wakeword-audit/audit.sqlite"}}, '
    '{"claim": "the audit table holds 1804 rows", '
    '"check": {"kind": "count_eq", "path": "wakeword-audit/audit.sqlite", '
    '"measure": "lines", "expected": 1804}}]}\n'
    "```\n"
)


async def test_task_53s_fabricated_audit_store_refutes_and_carries_forward(tmp_path,
                                                                          monkeypatch):
    """#2482 clause 3, across the seam: prompt → run → ledger write → next prompt.

    The digester's only structural gap was that nothing re-ran what it asserted. As a
    pilot member, a task #53 run is handed the `## Evidence claims` instruction, and
    the bundle it emits is re-run by `workers/evidence.py` against the data root at
    the moment `workers/pool.py` writes its row — so the invented store lands
    `refuted` in the record instead of reaching the board as a `priority: high` item,
    and the refutation is carried into the next run of #53 by name.
    """
    q = WorkQueue(tmp_path / "w.db")
    monkeypatch.setattr(evidence, "default_root", lambda: tmp_path)
    _stub_adapter(monkeypatch, final_response=FABRICATED_AUDIT_REPORT,
                  pilot=True, task_id=53)
    q.enqueue(source="scheduled-task", kind="run", payload={"task_id": 53})

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
    assert bundle["counts"] == {"total": 2, "verified": 0, "refuted": 2,
                                "insufficient": 0}
    assert bundle["refuted_or_insufficient_rate"] == 1.0
    carried = parse_gap_list(q.wm_get("scheduled-task", gaps_key(53)))
    assert len(carried) == 2
    assert any("wakeword-audit/audit.sqlite" in g for g in carried)
    # The half that makes a refutation load-bearing rather than merely recorded.
    nxt = autonomy._evidence_prompt(53, gaps=carried)
    assert "Evidence gaps carried from your previous run" in nxt
    assert "1804 rows" in nxt


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
    monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "SYS")
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
    monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "SYS")
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


# ─────────────────────────────────────────────────────────────────────────────
#  The JSON claim shape that faked a refutation (#2675)
#
#  #2675 widens the pilot to #24 and #56 and fixes the one claim shape that both
#  of those writers will hit on their first run: a check over a JSON body. The
#  process boundary here is the instruction the model reads and the stdlib
#  verifier that grades what it wrote, and it is asserted across both sides: the
#  guidance is only real if `verify_claim` — the shipped verifier, not a
#  re-implementation — returns `refuted` for the pattern the instruction tells the
#  model not to write and `verified` for the one it tells the model to write.
#  The roster half of the change crosses a second boundary (the docs cite the
#  frozenset by path:line and `tests/test_autonomy_jobs_doc_claims.py` reads both
#  sides); it is pinned there, not here.
# ─────────────────────────────────────────────────────────────────────────────

#: A doc-digester state file at fixture size, in the SHAPE of
#: `~/lloyd-data/_pipeline/doc-digester-state.json` — the state file the
#: doc-digester worker keeps for itself, and where the key paths below come from:
#: a nested `deployed.<tool>.checked_at`, a `gateway_probe` block, and a `sources`
#: block of per-collection doc counts. Every property stated next is a property of
#: THIS fixture, never of that file. The fixture holds `CHECKED_AT_COUNT = 6` of
#: the `checked_at` keys, is dumped with `indent=2` for the tests below, and has
#: `last_pass_utc` last in the fixture so that line ends at the value with no
#: trailing comma. The live file gets no size, no indentation width and no key
#: order from this note: nothing in this repo writes it — the digester's own tool
#: call does — so its size and its shape are whatever that run's model wrote, and
#: they differ next run. #2675 put a line count and a byte count for the live file in
#: this note, and nine days falsified the size before anyone re-read it.
#: `test_the_digester_fixture_note_carries_no_live_file_size` is what keeps a new one
#: out; `test_the_digester_note_numbers_are_measured_from_the_fixture` reads the
#: fixture's bytes and fails if the counts stated above are no longer what they are.
DIGESTER_STATE = {
    "deployed": {
        "claude_code_cli": {"checked_at": "2026-10-13T00:38:10Z",
                            "installed": "2.1.296"},
        "llama_cpp_8091": {"checked_at": "2026-10-13T00:38:10Z"},
        "vllm_djev": {"checked_at": "2026-10-13T00:38:10Z"},
        "vllm_omni_8093": {"checked_at": "2026-10-13T00:38:10Z"},
        "vllm_primary_8096": {"checked_at": "2026-10-13T00:38:10Z"},
    },
    "gateway_probe": {"checked_at": "2026-10-13T00:38:10Z",
                      "surface_used": "autonomy-runs/**/*.md (rglob)"},
    "sources": {"knowledge": {"docs": 12}, "plans": {"docs": 3}},
    "last_pass_utc": "2026-10-13T00:38:10Z",
}
CHECKED_AT_COUNT = 6


@pytest.fixture()
def digester_state(tmp_path):
    p = tmp_path / "_pipeline" / "doc-digester-state.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(DIGESTER_STATE, indent=2) + "\n")
    return tmp_path


def _count(pattern: str, expected: int) -> dict:
    return {"claim": f"the state file has {expected} matches",
            "check": {"kind": "count_eq",
                      "path": "_pipeline/doc-digester-state.json",
                      "measure": "matches", "pattern": pattern,
                      "expected": expected}}


def test_an_anchored_pattern_over_the_indented_json_body_is_refuted_and_the_unanchored_one_verifies(
        digester_state):
    """Both halves of the rule, measured over the shape that faked it.

    `count_eq` (`measure: matches`) and `regex` run `re.findall(pattern, text,
    re.MULTILINE)`, so `^` is a line start and every key in an indented JSON body
    sits behind at least two spaces. The anchored claim is therefore refuted by a
    file that says exactly what was claimed — `observed 0` on six timestamps — and
    a run record that read that verdict as "the number is wrong" would have filed
    an item against a correct count. The unanchored form, and the form that matches
    the indentation instead of pretending it is not there, both verify against the
    same bytes.
    """
    anchored = verify_claim(_count('^"checked_at":', CHECKED_AT_COUNT),
                            root=digester_state)
    assert anchored["status"] == "refuted"
    assert anchored["observed"] == 0, (
        "an anchored pattern over an indented key line has to be the one that "
        "fails, or the instruction below is warning about nothing")

    unanchored = verify_claim(_count('"checked_at":', CHECKED_AT_COUNT),
                              root=digester_state)
    assert unanchored["status"] == "verified"
    assert unanchored["observed"] == CHECKED_AT_COUNT

    indent_aware = verify_claim(_count(r'^\s+"checked_at":', CHECKED_AT_COUNT),
                                root=digester_state)
    assert indent_aware["status"] == "verified"

    # `$` is not the problem: it matches before the line's newline, so an
    # end-anchored claim over the same file's last key verifies.
    end_anchored = verify_claim(
        _count('"last_pass_utc": "2026-10-13T00:38:10Z"$', 1),
        root=digester_state)
    assert end_anchored["status"] == "verified"


def test_a_dotted_json_key_names_the_nested_value_the_digester_actually_writes(
        digester_state):
    """The other half of the same instruction: a nested value is a dotted key.

    `deployed.claude_code_cli.checked_at` is the live path. It verifies with the
    value, verifies on presence with no `expected`, and refutes only when the value
    really differs — which is what makes it the form the model should reach for
    instead of a pattern that has to guess indentation.
    """
    def claim(key, expected=None):
        check = {"kind": "json_key",
                 "path": "_pipeline/doc-digester-state.json", "key": key}
        if expected is not None:
            check["expected"] = expected
        return {"claim": f"state[{key}]", "check": check}

    v = verify_claim(claim("deployed.claude_code_cli.checked_at",
                           "2026-10-13T00:38:10Z"), root=digester_state)
    assert v["status"] == "verified" and v["observed"] == "2026-10-13T00:38:10Z"
    assert verify_claim(claim("gateway_probe.checked_at"),
                        root=digester_state)["status"] == "verified"
    assert verify_claim(claim("sources.knowledge.docs", 12),
                        root=digester_state)["status"] == "verified"
    assert verify_claim(claim("deployed.nope.checked_at"),
                        root=digester_state)["status"] == "refuted"


def _note_above(module_text: str, name: str) -> str:
    """The contiguous `#:` comment block immediately above a module-level name.

    `re` is imported here and in the two nodes below rather than at the top of the
    file: prose citations into this file name line 686 (`architecture/autonomy-jobs.md`
    twice and `tests/test_autonomy_jobs_doc_claims.py` twice), and a one-line module-level
    import would move that anchor off the assertion it points at.
    """
    import re

    m = re.search(rf"(?:^#:[^\n]*\n)+(?={name} = \{{)", module_text, re.M)
    assert m, (f"there is no `#:` note block directly above {name}, so a pin about "
               f"the note above it has nothing to read")
    return m.group(0)


def test_the_digester_fixture_note_carries_no_live_file_size():
    """Clause 1-3: the note states the fixture's shape and quotes nothing about
    the live file's size, nesting width or key order.

    Three groups, each with the control that makes its zero meaningful. (1) The note
    must still say where the shape came from — the live path is named — because the
    alternative to quoting a live file is not saying nothing about it, it is naming it
    as the origin and describing only what was copied. (2) The two stale figures the
    note used to carry are gone and so is the pair that is true of the file today: the
    check is that a live byte or line count cannot be written here at all, not that
    those particular digits are absent, which is why the forbidden tokens are built by
    concatenation and why the general prose form is checked too — a digit glued to the
    word for a line count or a byte count, which is what a future edit writes when it
    re-measures the file rather than copying the stale numbers. This file consequently
    holds no instance of any string it forbids; a pin that spelled one out would fail
    the repo-wide grep this item's acceptance is measured by, which is precisely how the
    note it replaces got there. (3) A positive control: the count pattern is handed a
    count of its own to find, so a zero hit in the note is a checked absence and not a
    pattern that never matched anything.

    The third clause is the part #2675's note got wrong beyond the two dead numbers.
    Its `indent=2` and `last_pass_utc` last were written as properties of the live
    file, and on 2026-10-23 the live file is indented one space per level and its
    top-level order ends with `sources` — `last_pass_utc` is third of five. Both are
    true of the fixture, which is dumped with `indent=2` and does have that key last,
    so the fix is to bind the claims to the fixture rather than delete them. The note's
    guard is then mechanical: a sentence making an indent or key-order claim, or holding
    a digit (an issue citation like `#2675` excepted), has to name the fixture as its
    subject. That admits every true statement about these bytes and refuses every
    measurement of the live file, which no sentence here can truthfully call a fixture.

    Two controls, because both halves of the check are greps over a slice. The count
    pattern is handed a count of its own to prove it can find one — a synthetic one, so
    the pin never touches the runtime data root. And both shape properties are asserted
    to still be stated somewhere in the note, because a loop over sentences is as green
    for a note that dropped the claims as for one whose claims are correctly bounded.
    """
    import re

    src = Path(__file__).read_text(encoding="utf-8")
    note = _note_above(src, "DIGESTER_STATE")

    assert "doc-digester-state.json" in note, (
        "the note no longer names the live file the fixture's shape came from")

    stale = ("288 " + "lines", "92," + "466", "99," + "719", "268 " + "lines")
    for token in stale:
        assert token not in note, (
            f"a live-file size ({token}) is quoted in the note again — that file is "
            "rewritten by the doc-digester worker's own tool call, so the number is "
            "true only for the run that wrote it")
    digits = re.compile(r"\d[\d,]* ?" + "(lines|bytes)")
    m_digits = digits.search(note)
    assert m_digits is None, (
        "the note quotes a byte or line count"
        + (f" ({m_digits.group(0)!r})" if m_digits else ""))
    control = digits.search("the real state file is 288 " + "lines")
    assert control and control.group(0).startswith("288 "), (
        "the pattern itself cannot match a digit glued to `lines` or `bytes`, so the "
        "assertion above could never have caught a count — fix the pattern before "
        "believing the zero hits")

    shape = re.compile("(indent=" + "2" + "|last_" + "pass_utc)")
    sentences = re.split(r"(?<=[.;])\s+", note.replace("#: ", ""))
    for token in ("indent=" + "2", "last_" + "pass_utc"):
        assert token in note, (
            f"the note no longer states the fixture's {token} property at all, so the "
            "loop below would pass on a note that had quietly dropped the claim")
    for sentence in sentences:
        bare = re.sub("#" + r"\d+", "", sentence)     # issue citations are not counts
        if (shape.search(sentence) or re.search(r"\d", bare)) \
                and "fixture" not in sentence.lower():
            raise AssertionError(
                "a note sentence makes a size or shape claim a reader will take about "
                f"the live file without naming the fixture as its subject: {sentence!r}")


def test_the_digester_note_numbers_are_measured_from_the_fixture():
    """The note's three remaining numbers are re-derived from the committed fixture.

    This is the check the old note was written without: it quoted 6 as the fixture's
    own count and that half happened to be right, but nothing read the bytes to find
    out, and the same sentence carried two live-file claims that were already wrong.
    So `CHECKED_AT_COUNT`, the `indent=2` and the `last_pass_utc`-last ordering are
    each read off `DIGESTER_STATE` here, and the note is required to state what the
    fixture actually is. `CHECKED_AT_COUNT` and the fixture itself are untouched by
    this change — clause 4's "behaviourally unchanged" is this node plus
    `test_an_anchored_pattern_over_the_indented_json_body_is_refuted_and_the_unanchored_one_verifies`,
    which is the only reader of the constant.
    """
    import re

    src = Path(__file__).read_text(encoding="utf-8")
    note = _note_above(src, "DIGESTER_STATE")
    assert re.search("(288" + "|" + "268) +(lines|bytes)", note) is None, (
        "the note already carries a live-file size, which the sibling pin exists to "
        "refuse; nothing stated here can be true of it")
    body = json.dumps(DIGESTER_STATE, indent=2)

    assert body.count('"checked_at":') == CHECKED_AT_COUNT == 6, (
        f"the fixture holds a different number of checked_at keys than the "
        f"{CHECKED_AT_COUNT} the note states and the tests count")
    def stated(text, pattern):
        m = re.search(pattern, text)
        return None if m is None else m.group(1)

    m_count = stated(note, r"`CHECKED_AT_COUNT = (" + r"\d+" + r")`")
    assert m_count is not None and int(m_count) == CHECKED_AT_COUNT, (
        "the note no longer states the fixture's own count, so the number the tests "
        "rely on is not readable from the fixture's description of itself")

    m_indent = stated(note, "(?:" + "indent=" + r")(\d+)")
    assert m_indent is not None, "the note no longer says how the fixture is dumped"
    real_indent = len(body.splitlines()[1]) - len(body.splitlines()[1].lstrip())
    assert int(m_indent) == real_indent == 2, (
        f"the note says indent={m_indent} and the fixture is dumped at "
        f"{real_indent}: the whole point of `^` failing here is that gap")

    doctored = note.replace("CHECKED_AT_COUNT = 6", "CHECKED_AT_COUNT = 7")
    assert stated(doctored, "`CHECKED_AT_COUNT = (" + r"\d+" + ")`") == "7", (
        "the extract-and-read helper above cannot pull a number out of a note, so the "
        "assertion over it is reading a None-shaped nothing rather than the bytes")

    assert list(DIGESTER_STATE)[-1] == "last_pass_utc", (
        "the note says this key is last so its line has no trailing comma; the "
        f"fixture's last key is {list(DIGESTER_STATE)[-1]!r}")
    assert body.splitlines()[-1] == "}"
    assert f'"last_pass_utc": "{DIGESTER_STATE["last_pass_utc"]}"' in \
        body.removesuffix("}").rstrip().splitlines()[-1], (
        "`last_pass_utc` is last but its line no longer ends at the value, so the "
        "note's reason for putting it there is false")


def test_a_json_key_expected_of_the_wrong_json_type_is_refuted_where_the_value_agrees(
        tmp_path):
    """The type rule, measured. `_check_json_key` compares with plain `==`.

    The state file holds an int, the model quotes it as a string — the file and the
    claim then disagree by JSON type alone and the verdict is `refuted`, which a
    model reading only "refuted" will report as a wrong count. `7` and `"7"` are the
    same fact to a model and different facts to `==`.
    """
    (tmp_path / "pipeline.json").write_text(json.dumps({"rows_written": 41}))
    quoted = verify_claim(
        {"claim": "41 rows written",
         "check": {"kind": "json_key", "path": "pipeline.json",
                   "key": "rows_written", "expected": "41"}}, root=tmp_path)
    assert quoted["status"] == "refuted" and quoted["observed"] == 41
    typed = verify_claim(
        {"claim": "41 rows written",
         "check": {"kind": "json_key", "path": "pipeline.json",
                   "key": "rows_written", "expected": 41}}, root=tmp_path)
    assert typed["status"] == "verified"


def test_the_claims_instruction_works_the_json_shape_that_broke_and_states_why():
    """#2675 clause 3: the instruction the model is handed carries the case, not
    just a rule about it.

    Parsed rather than grepped where it matters — the worked examples have to stay
    valid JSON, because a model copies this block literally and an example that does
    not parse is an instruction that cannot be followed. Then the two claims the
    guidance rests on are named by their load-bearing tokens: `re.MULTILINE` (why
    `^` fails, and why `$` does not) and `plain equality` (why the JSON type of
    `expected` is part of the claim).
    """
    block = evidence.CLAIMS_INSTRUCTION.split("```evidence\n")[1].split("\n```")[0]
    claims = json.loads(block)["claims"]
    assert len(claims) == 6, (
        "the worked-example block is what a model imitates; it parses today into 6 "
        "claims, and a broken brace here ships an instruction nobody can follow")
    json_body = [c["check"] for c in claims
                 if str(c["check"].get("path", "")).endswith(".json")]
    dotted = [c for c in json_body
              if c["kind"] == "json_key" and "." in str(c.get("key", ""))]
    assert dotted, "no dotted json_key example over a JSON body"
    assert dotted[0]["key"] == "deployed.claude_code_cli.checked_at"
    unanchored = [c for c in json_body
                  if c["kind"] in ("count_eq", "regex")
                  and not str(c.get("pattern", "")).startswith("^")]
    assert unanchored, "no unanchored pattern example over a JSON body"
    assert unanchored[0]["measure"] == "matches"
    # The rule text, by its load-bearing tokens.
    assert "re.MULTILINE" in evidence.CLAIMS_INSTRUCTION
    assert "`$` end-anchor" in evidence.CLAIMS_INSTRUCTION
    assert "plain equality" in evidence.CLAIMS_INSTRUCTION
    # And every member of the pilot is handed exactly this.
    for tid in (24, 38, 39, 40, 42, 53, 56):
        assert autonomy._evidence_prompt(tid, gaps=[]) == \
            evidence.CLAIMS_INSTRUCTION


# ── #2817 — a count of the whole run-record corpus cannot hold still ────────

def _seed_run_records(root: Path, per_task: dict[str, int]) -> int:
    """Seed `autonomy-runs/<task>/run_<task>_<stamp>.md` the way `autonomy.py` does.

    Returns the total written, so every caller states the on-disk number it is
    claiming against instead of trusting a literal it did not create.
    """
    total = 0
    for task_id, count in per_task.items():
        d = root / evidence.RUN_RECORDS_DIR / task_id
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (d / f"run_{task_id}_2026101{i}_050000.md").write_text("x\n")
        total += count
    return total


def test_a_whole_corpus_run_record_count_is_insufficient_not_refuted(tmp_path):
    """#2817 clause 1: the corpus the graded run writes into is a check failure.

    `autonomy.run_task` writes `autonomy-runs/<id>/<run_id>.md` and only then
    returns, and `workers/pool.py` verifies the bundle after the source returns, so a
    glob that crosses a directory boundary at `autonomy-runs` always counts the graded
    run's own record. The live instance behind this item:
    `run_scheduled-task_20261015_050046_db3930` (task 38) bound
    `{path: autonomy-runs, glob: "*/run_*.md", measure: files, expected: 709}` and was
    recorded `refuted — claimed files=709, disk says 710`, 9 ms after its own record
    landed. The corpus also moves on its own: ~37 records a day.

    Claimed one below the disk count, which is exactly the self-membership shape: the
    number a run can measure while it runs is short of the number the same check sees
    when it grades.
    """
    total = _seed_run_records(tmp_path, {"38": 3, "86": 3})
    assert total == 6
    for pattern in ("*/run_*.md", "**/run_*.md"):
        v = verify_claim({"claim": f"the fleet has written {total - 1} run records",
                          "check": {"kind": "count_eq", "path": "autonomy-runs",
                                    "glob": pattern, "measure": "files",
                                    "expected": total - 1}}, root=tmp_path)
        assert v["status"] == "insufficient", (
            f"{pattern!r} over the run-record root was graded {v['status']} at "
            f"{total - 1} claimed against {total} on disk — the off-by-one a run cannot "
            "avoid, because its own record is in the corpus it counts")
        assert v["observed"] is None, (
            "an unevaluable check reports no observation; reporting the moving count "
            "would let a later reader weigh it against the claim as if it were settled")
        assert f"{evidence.RUN_RECORDS_DIR}/<task_id>" in v["detail"], v["detail"]
        assert evidence.RUN_RECORDS_CLOSED_GLOB in v["detail"], v["detail"]


def test_a_fabricated_whole_corpus_count_reaches_the_gap_and_not_verified(tmp_path):
    """#2817 clause 2: refusing to refute a whole-corpus count is not a free pass.

    The rule keys on the claim's shape, before the number is read, so a run claiming
    1 record against 40 on disk lands in `gap` — which is what the next run of that
    task is handed — instead of joining `verified`, the set the next run must preserve.
    """
    total = _seed_run_records(tmp_path, {str(t): 5 for t in range(40, 48)})
    assert total == 40
    out = verify_bundle([{"claim": "the fleet has one run record", "check": {
        "kind": "count_eq", "path": "autonomy-runs", "glob": "**/run_*.md",
        "measure": "files", "expected": 1}}], root=tmp_path)
    assert out["verified"] == [], (
        "a whole-corpus count of 1 against 40 records was accepted as something the "
        "next run must preserve")
    assert len(out["gap"]) == 1, out["gap"]
    assert out["gap"][0].startswith("[insufficient]"), out["gap"][0]
    assert out["counts"]["refuted"] == 0 and out["counts"]["insufficient"] == 1
    # The repair has to survive the gap cap, so it leads the detail: `gap` items are
    # truncated at MAX_GAP_CHARS with the claim in front of them.
    assert out["gap"][0].index(f"{evidence.RUN_RECORDS_DIR}/<task_id>") < 90, out["gap"][0]


def test_a_per_task_run_record_count_is_still_graded_both_ways(tmp_path):
    """#2817 clause 3: the closed-set form is the one that verifies, and it still does.

    Both pilot runs of 2026-10-16 bind `path: autonomy-runs/86`, `glob: run_*.md` and
    came back `verified` (`run_scheduled-task_20261016_050026_218f69`,
    `run_scheduled-task_20261016_090013_93f62b`). One task's directory has no sibling
    to sweep in, so the separator is the discriminator — and a recursive glob bound
    INSIDE one task's directory is not the whole corpus either, which is why the rule
    tests the bound path and not just the glob.
    """
    total = _seed_run_records(tmp_path, {"86": 22, "38": 4})
    assert total == 26

    ok = verify_claim({"claim": "task 86 has 22 run records", "check": {
        "kind": "count_eq", "path": "autonomy-runs/86",
        "glob": evidence.RUN_RECORDS_CLOSED_GLOB, "measure": "files",
        "expected": 22}}, root=tmp_path)
    assert ok["status"] == "verified" and ok["observed"] == 22, ok

    bad = verify_claim({"claim": "task 86 has 96 run records", "check": {
        "kind": "count_eq", "path": "autonomy-runs/86",
        "glob": evidence.RUN_RECORDS_CLOSED_GLOB, "measure": "files",
        "expected": 96}}, root=tmp_path)
    assert bad["status"] == "refuted" and bad["observed"] == 22, bad
    assert f"claimed files=96, disk says 22 ({tmp_path / 'autonomy-runs' / '86'})" \
        in bad["detail"], bad["detail"]

    recursive = verify_claim({"claim": "task 86 has 22 run records", "check": {
        "kind": "count_eq", "path": "autonomy-runs/86", "glob": "**/run_*.md",
        "measure": "files", "expected": 22}}, root=tmp_path)
    assert recursive["status"] == "verified" and recursive["observed"] == 22, (
        "the rule reached a glob that crosses no task boundary, and would have refused "
        "the one count form that verifies today")


def test_tolerance_still_bridges_a_count_the_new_rule_does_not_reach(tmp_path):
    """#2817 clause 4: slack keeps working everywhere the rule does not apply.

    `run_scheduled-task_20261013_050047_2ed1ce` found this workaround on its own
    (637 claimed, tolerance 1, 638 observed, verified). What #2817 restricts is the
    whole-corpus shape; a count over a directory that is not the run-record root — and
    a closed set that is — keeps the semantics it has always had, including the
    refutation when the slack is not there.

    The one consequence the contract deliberately does not decide — whether a
    whole-corpus claim carrying `tolerance: 1`, the workaround that real run used, stays
    `verified` or becomes `insufficient` — is owed to owed-check as a RULING, so it is
    pinned here as the shape the code has: the refusal reads the bound path and the
    glob before the number, so slack does not buy back a claim about a corpus that
    includes the claimant. If owed-check rules the other way, this assertion is the one
    that has to change with it.
    """
    debug = tmp_path / "_pipeline/research/_debug"
    debug.mkdir(parents=True)
    for i in range(7):
        (debug / f"hypothesis_fail_{i}.txt").write_text("x")
    assert verify_claim({"claim": "~6 dumps", "check": {
        "kind": "count_eq", "path": "_pipeline/research/_debug",
        "glob": "hypothesis_fail_*.txt", "measure": "files",
        "expected": 6, "tolerance": 1}}, root=tmp_path)["status"] == "verified"

    nested = tmp_path / "_pipeline/runs"
    (nested / "86").mkdir(parents=True)
    for i in range(5):
        (nested / "86" / f"run_86_{i}.md").write_text("x")
    nested_claim = {"kind": "count_eq", "path": "_pipeline/runs",
                    "glob": "*/run_*.md", "measure": "files", "expected": 3}
    off_root = verify_claim({"claim": "3 nested records", "check": nested_claim},
                            root=tmp_path)
    assert off_root["status"] == "refuted" and off_root["observed"] == 5, off_root
    assert "claimed files=3, disk says 5" in off_root["detail"], off_root["detail"]
    # The same claim with slack verifies. Both halves matter: the first shows the rule
    # never reached a crossing glob bound OUTSIDE the run-record root (it would have
    # said `insufficient` with no observation at all), the second shows the slack that
    # found it still doing its job.
    assert verify_claim({"claim": "~4 nested records", "check": {
        **nested_claim, "expected": 4, "tolerance": 1}},
        root=tmp_path)["status"] == "verified", (
        "the rule reached a corpus outside the run-record root; the separator alone is "
        "not what refuses a claim")

    _seed_run_records(tmp_path, {"86": 22})
    assert verify_claim({"claim": "~21 records for task 86", "check": {
        "kind": "count_eq", "path": "autonomy-runs/86",
        "glob": evidence.RUN_RECORDS_CLOSED_GLOB, "measure": "files",
        "expected": 21, "tolerance": 1}}, root=tmp_path)["status"] == "verified"
    assert verify_claim({"claim": "21 records for task 86", "check": {
        "kind": "count_eq", "path": "autonomy-runs/86",
        "glob": evidence.RUN_RECORDS_CLOSED_GLOB, "measure": "files",
        "expected": 21}}, root=tmp_path)["status"] == "refuted", (
        "removing the slack did not change the verdict, so tolerance stopped meaning "
        "anything on a closed set")

    refused = verify_claim({"claim": "21 run records fleet-wide", "check": {
        "kind": "count_eq", "path": "autonomy-runs", "glob": "*/run_*.md",
        "measure": "files", "expected": 21, "tolerance": 1}}, root=tmp_path)
    assert refused["status"] == "insufficient", (
        "slack bought back a claim about a corpus that includes the claimant: the "
        "refusal reads the bound path and glob before the number, so `tolerance` "
        f"cannot rescue it — it came back {refused['status']}")
    assert f"{evidence.RUN_RECORDS_DIR}/<task_id>" in refused["detail"], refused["detail"]


def test_the_claims_instruction_tells_a_pilot_run_to_bind_a_closed_set():
    """#2817 clause 5: the rule is in the prompt every pilot job is handed.

    The closed-set discipline existed only as prose in the 2026-10-16 reflection
    report, and skill prose binds only the jobs that read it. The vault skill
    `skills/nightly-reflection-config/SKILL.md` — a vault path, not a file in this
    checkout — was itself prescribing the recursive glob until vault commit `5cc7a93a`
    (#2997) re-pointed it to the closed-set form. This block reaches every member of
    the pilot whether or not its skill says anything, so it is asserted through the
    seam that ships it (`autonomy._evidence_prompt`), not just against the constant.

    Asserted as a rule about the shape plus the outcome the run is told to expect, and
    that it stayed prose: a seventh worked example would change what a model imitates,
    and this file pins the example block at six claims elsewhere.
    """
    text = evidence.CLAIMS_INSTRUCTION
    assert f"{evidence.RUN_RECORDS_DIR}/<task_id>" in text
    assert evidence.RUN_RECORDS_CLOSED_GLOB in text
    assert "whole corpus" in text
    assert "`insufficient`" in text
    assert text.count(f"```{evidence.FENCE_TAG}") == 1, (
        "the rule was added as a second worked example; the example block is what a "
        "model copies verbatim and this file pins its shape elsewhere")
    for tid in sorted(autonomy.EVIDENCE_PILOT_TASK_IDS):
        block = autonomy._evidence_prompt(tid, gaps=[])
        assert f"{evidence.RUN_RECORDS_DIR}/<task_id>" in block, (
            f"pilot task {tid} is not being handed the closed-set rule")


# ── A directory where the check needs a file (#2993) ────────────────────────
#
# Every check below is handed a REAL directory seeded with real files, and every
# one of them used to grade the directory as if it were a missing file:
# `refuted observed=0` with the false detail `does not exist` for `count_eq` and
# `json_key`, and — the only false PASS of the set — `verified` for
# `regex expect=absent`, because a path that "does not exist" cannot hold a
# match. Four such misgrades sat in the ledger (runs 20261013_090004_1712aa,
# 20261012_050010_a0f87f, 20261004_050044_c1b854, 20260928_080049_141f03, on
# `_pipeline/reflection`, `autonomy-runs` and `safety`). A check that never ran
# is the check's failure, so all three sites now return `insufficient` with a
# detail naming the directory and the check — never a count, never a pass.

def _seed_dir(tmp_path, name="reflection"):
    """A directory holding files, so a `count_eq` over it is a check that could
    have produced a number and did not — not a check against empty space."""
    d = tmp_path / name
    d.mkdir()
    (d / "a.md").write_text("handoff one\n")
    (d / "b.md").write_text("handoff two\n")
    return d


def test_count_eq_matches_over_a_directory_is_insufficient_not_refuted(tmp_path):
    d = _seed_dir(tmp_path)
    v = verify_claim({"claim": "11 checked_at entries", "check": {
        "kind": "count_eq", "path": "reflection", "measure": "matches",
        "pattern": "checked_at", "expected": 11}}, root=tmp_path)
    assert v["status"] == "insufficient"
    assert v["observed"] is None, "a directory is not a count of zero"
    assert str(d) in v["detail"] and "matches" in v["detail"]
    assert "does not exist" not in v["detail"]


def test_count_eq_bytes_and_lines_over_a_directory_are_insufficient_too(tmp_path):
    d = _seed_dir(tmp_path)
    for measure in ("bytes", "lines"):
        v = verify_claim({"claim": f"the store is 26 {measure}", "check": {
            "kind": "count_eq", "path": "reflection", "measure": measure,
            "expected": 26}}, root=tmp_path)
        assert v["status"] == "insufficient", measure
        assert v["observed"] is None, measure
        assert str(d) in v["detail"] and measure in v["detail"], v["detail"]
        assert "does not exist" not in v["detail"], v["detail"]


def test_json_key_over_a_directory_is_insufficient_not_refuted(tmp_path):
    d = _seed_dir(tmp_path, name="state")
    (d / "pipeline.json").write_text(json.dumps({"rows_written": 25}))
    v = verify_claim({"claim": "state records 25 rows", "check": {
        "kind": "json_key", "path": "state", "key": "rows_written",
        "expected": 25}}, root=tmp_path)
    assert v["status"] == "insufficient" and v["observed"] is None
    assert str(d) in v["detail"] and "json_key" in v["detail"]
    assert "does not exist" not in v["detail"]


def test_regex_expect_absent_over_a_directory_cannot_verify(tmp_path):
    """The false pass: an absence claim used to be satisfied by pointing at a
    directory, and the bundle then recorded it as a fact the next run has to
    PRESERVE rather than a gap it has to address."""
    d = _seed_dir(tmp_path)
    for want in ("absent", "match"):
        v = verify_claim({"claim": f"no Traceback in the store ({want})", "check": {
            "kind": "regex", "path": "reflection", "pattern": "Traceback",
            "expect": want}}, root=tmp_path)
        assert v["status"] == "insufficient", want
        assert v["observed"] is None, want
        assert str(d) in v["detail"] and "regex" in v["detail"], v["detail"]

    b = verify_bundle([{"claim": "no Traceback in the store",
                        "check": {"kind": "regex", "path": "reflection",
                                  "pattern": "Traceback", "expect": "absent"}}],
                      root=tmp_path)
    assert b["verified"] == []
    assert b["counts"] == {"total": 1, "verified": 0, "refuted": 0,
                           "insufficient": 1}
    assert b["gap"] == [f"[insufficient] no Traceback in the store — {d} is a "
                        f"directory, not a file — the regex check never ran, so "
                        f"bind the file inside it"]


# ── The controls: a genuinely missing file keeps today's grades ─────────────

def test_a_missing_file_keeps_every_grade_it_has_today(tmp_path):
    """Clause 5's control. `run_scheduled-task_20261018_080011_14b93a` is the
    live positive case: a `does not exist` refutation is honest ONLY when the
    path really is absent, so the new branch may not widen it. The directory
    probes above and these lines are the two halves of one distinction — the
    gap text is what separates them, and it says `is a directory` for one and
    `does not exist` for the other."""
    (tmp_path / "here.md").write_text("| Total relationships | 0 |\n")
    missing = "no-such-file-xyz.md"

    assert verify_claim({"claim": "a match", "check": {
        "kind": "regex", "path": missing, "pattern": "Traceback",
        "expect": "match"}}, root=tmp_path)["status"] == "refuted"
    absent = verify_claim({"claim": "no match", "check": {
        "kind": "regex", "path": missing, "pattern": "Traceback",
        "expect": "absent"}}, root=tmp_path)
    assert absent["status"] == "verified" and absent["observed"] == 0
    assert "does not exist" in absent["detail"]

    json_v = verify_claim({"claim": "25 rows", "check": {
        "kind": "json_key", "path": missing, "key": "rows_written",
        "expected": 25}}, root=tmp_path)
    assert json_v["status"] == "refuted" and json_v["observed"] is None
    assert "does not exist" in json_v["detail"]

    for measure, check in (
            ("bytes", {"measure": "bytes", "expected": 100}),
            ("lines", {"measure": "lines", "expected": 3}),
            ("matches", {"measure": "matches", "pattern": "Traceback",
                         "expected": 2})):
        v = verify_claim({"claim": f"the file has {measure}",
                          "check": dict(kind="count_eq", path=missing, **check)},
                         root=tmp_path)
        assert v["status"] == "refuted", measure
        assert v["observed"] == 0, measure
        assert "does not exist" in v["detail"], v["detail"]

    assert verify_claim({"claim": "a match on a real file", "check": {
        "kind": "regex", "path": "here.md",
        "pattern": r"^\| Total relationships \| 0 \|$"}},
        root=tmp_path)["status"] == "verified"


def test_count_eq_measure_files_over_a_path_that_is_a_file_stays_refuted(tmp_path):
    """The `files` measure needs a directory and says so: a FILE is not one, and
    that is a false count, not an unevaluable check. The missing-directory half
    of this is pinned by
    `test_count_eq_on_a_missing_directory_is_refuted_with_a_zero`; the new
    directory branch sits in the OTHER branch and may not reach either."""
    f = tmp_path / "notes.md"
    f.write_text("a\nb\n")
    v = verify_claim({"claim": "3 files", "check": {
        "kind": "count_eq", "path": "notes.md", "measure": "files",
        "expected": 3}}, root=tmp_path)
    assert v["status"] == "refuted" and v["observed"] == 0
    assert "is not a directory" in v["detail"]


# ── A body past the read cap is the check's own failure (#3295) ───────────────
#
# `MAX_CHECK_BYTES` bounds what `_read_text` hands a check, and until #3295 every
# body-measuring kind graded whatever came back — a prefix — as if it were the
# file. Four such grades were measured live on 2026-10-25 over
# `_pipeline/relations-index.json` (180,645,844 bytes, 22.6x the cap, the artifact
# autonomy task #24 rebuilds on every run) and not one of them was true:
#
#   json_key total_relationships       → insufficient `… is not valid JSON`, of a
#                                        file that parses whole in 0.7 s holding
#                                        548,124 relationships and 4,235 documents
#   regex pattern=last_updated absent  → VERIFIED observed=0, for a key sitting in
#                                        the file's tail
#   count_eq measure=matches  exp=1    → refuted observed=0
#   count_eq measure=lines    exp=1    → refuted observed=0
#
# The first names a broken artifact when the artifact is fine and reads to the run
# as a lie it told; the second is worse, because a verified claim is what the next
# run is told to PRESERVE; the last two refute true claims. The cap is a sane cost
# bound and stays at 8,000,000 — the label was the defect, so `_oversized_body`
# now returns `insufficient` with the size, the cap and a repair, and no body
# measure grades from a prefix at all.
#
# Every node below pins this with `MAX_CHECK_BYTES` monkeypatched to 64 rather
# than with a 180 MB fixture: the property under test is "the bytes this check
# needs are past the cap", and that property is identical at 64 bytes.

_SMALL_CAP = 64


def _seed_beyond_cap(tmp_path, monkeypatch, body: str, name: str):
    """Write `body`, then drop the read cap to 64 bytes so every body-measuring
    check is looking at a prefix of it. Returns the file."""
    monkeypatch.setattr(evidence, "MAX_CHECK_BYTES", _SMALL_CAP)
    f = tmp_path / name
    f.write_text(body)
    assert f.stat().st_size > _SMALL_CAP, "the fixture must be over the cap it pins"
    return f


def test_json_key_over_a_body_past_the_cap_names_the_cap_not_broken_json(
        tmp_path, monkeypatch):
    """#3295 clause 1. The gap line is the whole instruction the next run of task
    #24 gets about its rebuild's two counts, and until now it read
    `relations-index.json is not valid JSON` about a file that parses clean — an
    accusation about the run's honesty for a limit of the instrument."""
    assert evidence.MAX_CHECK_BYTES == 8_000_000, "#3295 moved the label, not the bound"

    body = json.dumps({"relationships": {f"rel:{i}": i for i in range(40)},
                       "total_relationships": 548124})
    f = _seed_beyond_cap(tmp_path, monkeypatch, body, name="index.json")
    size = f.stat().st_size
    assert body.index("total_relationships") > _SMALL_CAP, \
        "the requested key has to live past the cap, as it does in the real index"
    claim = {"claim": "the rebuild stored 548124 relationships",
             "check": {"kind": "json_key", "path": "index.json",
                       "key": "total_relationships", "expected": 548124}}

    v = verify_claim(claim, root=tmp_path)
    assert v["status"] == "insufficient"
    assert v["observed"] is None, "a file the check could not read is not a None value"
    assert "not valid JSON" not in v["detail"], v["detail"]
    assert f"{size:,}" in v["detail"], f"the detail must name the size: {v['detail']}"
    assert "64-byte" in v["detail"], f"and the cap: {v['detail']}"
    assert "could not read" in v["detail"], v["detail"]
    assert v["detail"].startswith("bind a smaller artefact carrying the same numbers"), (
        "the repair leads, as `_whole_run_record_corpus` does: a gap line is cut to "
        f"{evidence.MAX_GAP_CHARS} chars with the claim in front of it")

    b = verify_bundle([claim], root=tmp_path)
    assert b["verified"] == [], "an unreadable claim is never a fact to preserve"
    assert b["counts"] == {"total": 1, "verified": 0, "refuted": 0, "insufficient": 1}
    line = b["gap"][0]
    assert line == ("[insufficient] the rebuild stored 548124 relationships — "
                    + v["detail"]), line
    assert len(line) <= evidence.MAX_GAP_CHARS, (
        f"the repair, the size and the cap all have to survive the truncation: {line}")

    # The seam this item is about. That line is stored under `evidence_gaps:<id>`
    # and replayed by `gaps_prompt` into the NEXT run of task #24, which is where
    # "the index rebuild stored 547,931 relationships — … is not valid JSON" read
    # as a suspicion about the run. Here it has to arrive naming the size and the
    # cap; the carry plumbing itself is pinned by
    # `test_the_gap_list_reaches_the_next_run_of_the_same_task`.
    prompt = gaps_prompt(b["gap"])
    assert "bind a smaller artefact carrying the same numbers" in prompt, prompt
    assert f"{size:,}" in prompt, prompt
    assert "64-byte" in prompt, prompt
    assert "is not valid JSON" not in prompt, prompt

    # Control: the same claim over the same bytes verifies as soon as the cap can
    # reach the key, so the gate is the cap and not the file's JSON.
    monkeypatch.setattr(evidence, "MAX_CHECK_BYTES", 8_000_000)
    back = verify_claim(claim, root=tmp_path)
    assert back["status"] == "verified" and back["observed"] == 548124


def test_regex_over_a_body_past_the_cap_cannot_grade_an_absence_as_verified(
        tmp_path, monkeypatch):
    """#3295 clause 2, the false pass. `pattern=last_updated expect=absent` over
    the 180 MB index returned `verified observed=0` for a key the file holds in
    its tail, and `verify_bundle` puts a verified claim in the set the next run of
    the task is told to preserve."""
    body = json.dumps({"relationships": {f"rel:{i}": i for i in range(40)},
                       "last_updated": "2026-10-25T00:09:52"})
    f = _seed_beyond_cap(tmp_path, monkeypatch, body, name="index.json")
    size = f.stat().st_size
    assert body.index("last_updated") > _SMALL_CAP, \
        "the pattern must live past the cap, as `last_updated` does in the real index"

    for want in ("absent", "match"):
        v = verify_claim({"claim": f"last_updated count ({want})", "check": {
            "kind": "regex", "path": "index.json", "pattern": "last_updated",
            "expect": want}}, root=tmp_path)
        assert v["status"] == "insufficient", (want, v["status"])
        assert v["observed"] is None, want
        assert f"{size:,}" in v["detail"], (want, v["detail"])
        assert "64-byte" in v["detail"], (want, v["detail"])

    # Control: under a cap that reaches the tail, both arms grade as they always
    # did — the key is there, so `absent` is genuinely refuted.
    monkeypatch.setattr(evidence, "MAX_CHECK_BYTES", 8_000_000)
    hit = verify_claim({"claim": "the index stamped last_updated", "check": {
        "kind": "regex", "path": "index.json", "pattern": "last_updated",
        "expect": "match"}}, root=tmp_path)
    assert hit["status"] == "verified" and hit["observed"] == 1
    miss = verify_claim({"claim": "the index has no last_updated", "check": {
        "kind": "regex", "path": "index.json", "pattern": "last_updated",
        "expect": "absent"}}, root=tmp_path)
    assert miss["status"] == "refuted" and miss["observed"] == 1


def test_count_eq_lines_and_matches_over_a_body_past_the_cap_are_not_refuted(
        tmp_path, monkeypatch):
    """#3295 clause 2, the false refutation. Both measures counted the prefix: on
    2026-10-25 `measure=matches pattern=total_relationships expected=1` over the
    180 MB index came back `refuted observed=0` for a key the file holds. Here the
    pattern occurs twice — once inside the cap, once past it — so the truncated
    count is 1 against a claimed 2, and the grade a run gets today is a refutation
    of a true claim."""
    body = '{"first":"needle",\n"pad":"' + "p" * 200 + '",\n"last":"needle"}'
    assert body.count("needle") == 2
    f = _seed_beyond_cap(tmp_path, monkeypatch, body, name="index.json")
    size = f.stat().st_size
    assert body.rindex("needle") > _SMALL_CAP, \
        "the second hit has to live past the cap for the count to be truncated"

    arms = (("lines", {"measure": "lines", "pattern": "needle", "expected": 2}),
            ("matches", {"measure": "matches", "pattern": "needle", "expected": 2}))
    for measure, extra in arms:
        v = verify_claim({"claim": "the index names needle twice", "check": dict(
            kind="count_eq", path="index.json", **extra)}, root=tmp_path)
        assert v["status"] == "insufficient", (measure, v["status"])
        assert v["observed"] is None, f"{measure}: a prefix is not a count of one"
        assert measure in v["detail"], v["detail"]
        assert f"{size:,}" in v["detail"], (measure, v["detail"])
        assert "64-byte" in v["detail"], (measure, v["detail"])

    monkeypatch.setattr(evidence, "MAX_CHECK_BYTES", 8_000_000)
    for measure, extra in arms:
        v = verify_claim({"claim": "the index names needle twice", "check": dict(
            kind="count_eq", path="index.json", **extra)}, root=tmp_path)
        assert v["status"] == "verified", (measure, v["detail"])
        assert v["observed"] == 2, measure


def test_body_free_checks_still_grade_a_target_past_the_read_cap(tmp_path,
                                                                monkeypatch):
    """#3295 clause 3. The gate is "reads a body", not "is big". `measure=bytes`
    reads `stat().st_size` and reported all 180,645,844 bytes of the relations
    index correctly on 2026-10-25 — it is the binding task #24 used to bound the
    two counts this item is about, and a fix keyed on size would have taken the
    only proof that worked away from the run."""
    monkeypatch.setattr(evidence, "MAX_CHECK_BYTES", _SMALL_CAP)
    store = tmp_path / "store"
    store.mkdir()
    f = store / "index.json"
    f.write_text(json.dumps({"total_relationships": 548124}) + "x" * 200)
    size = f.stat().st_size
    assert size > _SMALL_CAP

    present = verify_claim({"claim": "the index exists", "check": {
        "kind": "file_exists", "path": "store/index.json"}}, root=tmp_path)
    assert present["status"] == "verified" and present["observed"] is True
    gone = verify_claim({"claim": "the index was replaced", "check": {
        "kind": "file_exists", "path": "store/index.json",
        "expected": False}}, root=tmp_path)
    assert gone["status"] == "refuted" and gone["observed"] is True
    assert "body-read cap" not in gone["detail"], gone["detail"]

    exact = verify_claim({"claim": f"the index is {size} bytes", "check": {
        "kind": "count_eq", "path": "store/index.json", "measure": "bytes",
        "expected": size}}, root=tmp_path)
    assert exact["status"] == "verified" and exact["observed"] == size
    assert exact["detail"] == "", "a byte count needs no apology: the check ran"
    wrong = verify_claim({"claim": f"the index is {size + 1} bytes", "check": {
        "kind": "count_eq", "path": "store/index.json", "measure": "bytes",
        "expected": size + 1}}, root=tmp_path)
    assert wrong["status"] == "refuted" and wrong["observed"] == size
    assert "body-read cap" not in wrong["detail"], wrong["detail"]

    files = verify_claim({"claim": "one index file", "check": {
        "kind": "count_eq", "path": "store", "measure": "files",
        "glob": "*.json", "expected": 1}}, root=tmp_path)
    assert files["status"] == "verified" and files["observed"] == 1
    two = verify_claim({"claim": "two index files", "check": {
        "kind": "count_eq", "path": "store", "measure": "files",
        "glob": "*.json", "expected": 2}}, root=tmp_path)
    assert two["status"] == "refuted" and two["observed"] == 1
