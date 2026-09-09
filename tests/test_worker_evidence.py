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


def _seed_root() -> Path:
    """Root the seeded claims resolve against: the live tree when it has a
    `_pipeline`, else a checked-in mirror of the same relative layout.

    `_pipeline/**` is gitignored, so a round's worktree has none and the mirror
    keeps the assertion alive there (its excerpt copies of the same two dated
    artifacts, and three failure dumps instead of a hundred). On the live
    checkout the real files are used, which is what makes the replay an
    assertion about the artifacts the corrections log argued about. Both
    branches assert; neither skips.
    """
    return CHECKOUT if (CHECKOUT / "_pipeline").is_dir() else FIXTURES / "seeded"


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
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda: "SYS")
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
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda: "SYS")
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
    root = _seed_root()
    statuses = {}
    for name, seed in SEED_CLAIMS.items():
        artifact = root / seed["check"]["path"]
        assert artifact.exists(), (
            f"seed {name}: the artifact its claim is about is missing from "
            f"{artifact} — the replay would be vacuous")
        statuses[name] = verify_claim(seed, root=root)["status"]

    assert statuses == {"relationships_12131": "refuted",
                        "hypothesis_fail_96": "refuted",
                        "signals_latest_307kb": "refuted"}


def test_the_seeds_come_from_real_handoff_text():
    """If a seed stops matching the handoff it was quoted from, the replay above
    is testing a claim nobody made — so pin the provenance too."""
    handoff = _seed_root() / "_pipeline/reflection/knowledge-handoff-2026-08-24.md"
    assert handoff.is_file()
    assert "12,131" in handoff.read_text(errors="replace")
