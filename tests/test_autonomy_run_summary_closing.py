"""The run ledger stores a run's outcome, not the sentence it opened with.

`run_task` accumulates one string — `final_response`, the join of every
`text_delta` the agent loop emitted — and used to cut its LEDGER SUMMARY from the
front of it: `summary=final_response[:200]` for the run record,
`"response_preview": final_response[:300]` for the dict handed to the worker pool.
The whole response was already persisted to `autonomy-runs/<task_id>/<run_id>.md`;
only the summary existed in `workers.db.runs`, and it was the opening.

Measured on this box, 2026-09-14 (item #642, triage) and re-measured 2026-09-20
against `workers.db` before this round: 92 `scheduled-task` rows since 09-18, 82 of
them at the 300-character cap, `MAX(length(summary))` exactly 300, 13 holding a
markdown table rule row, and 10/10 of the capped rows joined to their artifact
matched the HEAD of that artifact's `## Response` and none its tail. So the ledger
line a person or a health report reads was `Now let me classify the kept items and
compose the signal.` followed by 260 characters of a table's header, and the
verdict — `| fleet verdict | HEALTHY |`, `No deletions proposed, per guardrails.` —
was on disk and nowhere in the ledger. The same string is what Discord sends a
human, and the same string is what `compute_health` string-matches to decide
whether a run timed out.

The fix is a suffix slice through one constant (`RUN_SUMMARY_CAP`), plus a guard on
the classifier: a `success` row states its own verdict, so narrative text can no
longer overturn it. The harness here is the same one
`tests/test_autonomy_silence.py` uses — the real `run_task` over a scripted
`run_query`, the real `_write_run_record` to disk.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
import yaml

import autonomy
from workers.pool import normalize_result
from workers.queue import QueueItem, WorkQueue

# The tree whose source the text assertions below read — the checkout that
# `autonomy` was imported from, never the process cwd.
_REPO = Path(autonomy.__file__).resolve().parent

CAP = autonomy.RUN_SUMMARY_CAP
assert CAP == 300, "one constant, and it is the cap the live ledger was pinned to"

# ── fixtures ─────────────────────────────────────────────────────────────────

# Three parts, each longer than one window except the last. `VERDICT` is the
# sentence a report signs off with — short enough to fit inside the cap, so it is
# exactly what a suffix slice preserves and a prefix slice drops. `OPENING` and
# `REPORT` are both longer than the cap on their own, so the ledger line can hold
# either the narration or the verdict, never both.
OPENING = ("Now let me classify the kept items and compose the signal. "
           "Loading the last 24 hours of email first, then the calendar. ") * 6
REPORT = ("Kept 4 of 12 hits and 0 duplicates. The two newsletter threads were "
          "folded into the digest rather than surfaced, and the one calendar invite "
          "with no reply yet is left open for the next pass; the reminders the user "
          "asked about are queued against their own times. ") * 4
VERDICT = "Injected 1 notable email and 0 events; signal handed to the surface."
RESPONSE = OPENING + "\n\n" + REPORT + "\n\n" + VERDICT
# The harness terminates an assistant turn with a newline and `final_response` is
# the bare concatenation of the deltas, so the string the slicer receives — and the
# string the `## Response` body holds — is `RESPONSE` plus that newline.
FINAL_RESPONSE = RESPONSE + "\n"
# The exact slice the ledger must hold, computed independently of the code under
# test so a helper that returned the head could not satisfy it as well.
EXPECTED_CLOSING = FINAL_RESPONSE[-CAP:]
assert len(OPENING) > CAP and len(REPORT) > CAP
assert RESPONSE[:CAP] != EXPECTED_CLOSING
assert VERDICT in EXPECTED_CLOSING                   # the outcome survives truncation
assert "Now let me" not in EXPECTED_CLOSING          # the narration does not
# Positive control: the head slice this replaces held the narration and dropped
# the verdict, which is the whole defect in two lines.
assert "Now let me" in RESPONSE[:CAP]
assert VERDICT not in RESPONSE[:CAP]

# A report ending in a markdown table longer than the cap — the shape the item
# cites from live data, where the stored summary ended
# `| Check | Result |\n|---|---|\n| Bro`: the scaffolding survived, the verdict did
# not. The header and its `|---|` rule row sit at the far end of the table from the
# cap window, so a suffix slice holds the closing rows and none of the scaffolding.
TABLE_ROWS = [
    "| Check | Result |",
    "|---|---|",
    "| lloyd-backend supervisor state | RUNNING, uptime 6h12m |",
    "| lloyd-mcp supervisor state | RUNNING, uptime 6h12m |",
    "| primary engine /health | 200 in 0.31s |",
    "| frontend :5173 over TLS | 200, cert valid to 2026-11-02 |",
    "| email probe, unread last 24h | 3 messages, 0 errors |",
    "| calendar probe, events today | 2 events, non-empty control held |",
    "| wake-word session with non-zero rms_peak | 1 session at 18:40:58 |",
    "| fleet verdict | HEALTHY, 0 regressions, 0 new alerts |",
]
TABLE_RESPONSE = ("Health probe finished. Every check below ran against the live "
                  "box, not a cached reading; one row per probe.\n"
                  + "\n".join(TABLE_ROWS) + "\n")
TABLE_LAST_LINE = TABLE_ROWS[-1]
assert len(TABLE_RESPONSE) > CAP
assert TABLE_LAST_LINE in TABLE_RESPONSE[-CAP:]
assert "|---|" in TABLE_RESPONSE[:CAP]               # the head slice held the scaffolding…
assert TABLE_LAST_LINE not in TABLE_RESPONSE[:CAP]   # …and dropped the verdict


def _text_stream(text: str) -> list[dict]:
    """One assistant turn carrying `text`, in the shape the harness emits.

    `run_task` concatenates `evt["text"]` over every `text_delta` into
    `final_response`, and remembers the text of the LAST `assistant_message` as the
    run's terminal block — which is what decides `[SILENT]`. One iteration therefore
    needs both events, each carrying the block's own text, or the fixture does not
    look like a record it is meant to reproduce.
    """
    block = text + "\n"      # a real assistant turn is newline-terminated
    return [{"type": "text_delta", "text": block},
            {"type": "assistant_message", "text": block}]


def _item(payload: dict | None = None, source: str = "scheduled-task", **kw) -> QueueItem:
    base = dict(id=1, source=source, kind="run", priority=50, payload=payload or {},
                dedup_key=None, state="running", attempts=1,
                enqueued_at="", claimed_at=None, claimed_by=None,
                completed_at=None, error=None)
    base.update(kw)
    return QueueItem(**base)


def _task(task_id: int = 24, **extra) -> dict:
    """A task record for `run_task` (through `_parse_task_file`) and for
    `compute_health`'s task list."""
    task = {"id": task_id, "name": "Nightly Signal", "skill_name": "triage",
            "status": "up_next", "timeout_seconds": 300,
            "notify_on_complete": True}
    task.update(extra)
    return task


@pytest.fixture
def run_driver(monkeypatch, tmp_path):
    """Drive the real `run_task` over a scripted `run_query`.

    Same seam `tests/test_autonomy_silence.py` uses: the engine and the task-file
    lookups are stubbed, the run record is the real `_write_run_record` writing to a
    temp `AUTONOMY_RUNS_DIR` — wrapped, not replaced, so the artifact on disk is
    what clause 4 reads back. `captured["record"]` is the kwargs it was handed.
    """
    captured: dict = {}
    runs_dir = tmp_path / "autonomy-runs"
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", runs_dir)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)

    real_write = autonomy._write_run_record   # grabbed before the patch: no recursion

    def _record(*args, **kwargs):
        captured["record"] = kwargs
        return real_write(*args, **kwargs)

    monkeypatch.setattr(autonomy, "_write_run_record", _record)

    def _stub(events: list[dict], task: dict | None = None):
        captured["record"] = None

        async def _run_query(messages, options):
            captured["prompt"] = messages[0]["content"]
            for evt in events:
                yield evt

        class Opts:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        import app.harness as harness_mod
        import app.harness.mcp_pool as mcp_pool
        monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "t.md")
        monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task or _task())
        monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
        monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
        monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
        monkeypatch.setattr(harness_mod, "run_query", _run_query)
        monkeypatch.setattr(harness_mod, "RunOptions", Opts)
        monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
        monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")
        monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False)
        monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
        return captured

    _stub.runs_dir = runs_dir
    return _stub


# ── Clauses 1 & 2: the stored summary is the response's closing ───────────────

def test_a_long_response_is_stored_by_its_closing_not_its_opening(run_driver):
    """Clause 1. One constant; the last N characters, not the first N."""
    captured = run_driver(_text_stream(RESPONSE))
    out = asyncio.run(autonomy.run_task(24))

    assert out["response_preview"] == EXPECTED_CLOSING
    assert len(out["response_preview"]) == CAP
    assert VERDICT in out["response_preview"]
    assert not out["response_preview"].startswith("Now let me")
    # The run record and the dict handed to the worker pool are ONE slice, not the
    # 200/300 pair that used to disagree with each other.
    assert captured["record"]["summary"] == out["response_preview"]


def test_a_short_response_is_stored_whole(run_driver):
    """A suffix slice of a string shorter than the cap is the string itself — the
    whole answer, so a short report is not gutted by a rule written for long ones."""
    short = "Injected 1 email, 0 calendar events, nothing else actionable."
    captured = run_driver(_text_stream(short))
    out = asyncio.run(autonomy.run_task(24))
    # Whole, not cut: only the turn's terminating newline is present, and no
    # character of the report is missing.
    assert out["response_preview"] == short + "\n"
    assert out["response_preview"].strip() == short
    assert captured["record"]["summary"] == out["response_preview"]


def test_a_report_ending_in_a_table_keeps_its_verdict_row(run_driver):
    """Clause 2. A table's outcome is its last row; its scaffolding is worth
    nothing. The positive control below pins that the head slice this replaces
    really did hold the `|---|` rule row and really did drop the verdict."""
    captured = run_driver(_text_stream(TABLE_RESPONSE))
    out = asyncio.run(autonomy.run_task(24))
    summary = out["response_preview"]

    assert TABLE_LAST_LINE in summary
    assert captured["record"]["summary"] == summary
    assert "|---|" not in summary
    assert "| Check | Result |" not in summary
    head = TABLE_RESPONSE[:CAP]
    assert "|---|" in head
    assert TABLE_LAST_LINE not in head


# ── Clause 3: one constant, one helper, no literal slice ─────────────────────

def test_the_one_slice_of_final_response_goes_through_the_named_constant():
    r"""Clause 3, asserted with the clause's own grep. `grep -n "final_response\["
    autonomy.py` must show a slice, and every line it shows must
    reach the cap through `RUN_SUMMARY_CAP`. No `[:200]` / `[:300]` survives, and
    300 is still the one length every consumer of the ledger field trims to."""
    src = (_REPO / "autonomy.py").read_text(encoding="utf-8")
    lines = [ln for ln in src.splitlines()
             if re.search(r"final_response\[[^\]]*\]", ln)]
    assert len(lines) == 1, f"expected exactly one slice of final_response, got {lines}"
    assert "RUN_SUMMARY_CAP" in lines[0], \
        "the surviving slice must use the named constant, not a literal"
    for literal in ("[:200]", "[:300]"):
        assert literal not in src, f"the old {literal} cap is still in autonomy.py"
    assert src.count("RUN_SUMMARY_CAP") >= 2   # defined once, used by both helpers
    assert autonomy.RUN_SUMMARY_CAP == 300


def test_every_summary_writer_in_autonomy_calls_a_helper_not_a_slice():
    """Both writers are reached through `_outcome_summary` / `_failure_summary`; the
    counts are exact so a future writer cannot re-add a literal slice unnoticed. The
    failure path head-slices an error string through the SAME constant."""
    src = (_REPO / "autonomy.py").read_text(encoding="utf-8")
    assert src.count("_outcome_summary(final_response)") == 2   # record + preview
    # Four since #1085, and the fourth is a new CALLER of the existing helper,
    # not a new way to build a summary: the infra-ceiling alert. What makes this
    # test bite is that the number is EXACT, not that it only ever falls — it went
    # UP when the ceiling alert arrived, and goes up or down whenever a call site
    # appears or disappears, so a new writer has to be named here rather than slip
    # past a `>=`. The anti-slice half is the assertion below: a writer that went
    # back to a literal slice still fails on `"summary[:RUN_SUMMARY_CAP]"`.
    assert src.count("_failure_summary(summary)") == 4  # record, disable alert, log, ceiling alert
    assert "summary[:RUN_SUMMARY_CAP]" in src


# ── Clause 4: artifact front matter and the DB row are one slice ─────────────

async def _drive_execute(monkeypatch, adapter_env, *, response: str, meta: dict,
                         status: str = "success"):
    """Run the REAL `scheduled_task.execute` against a stubbed `run_task`, and
    return the dict `normalize_result` receives. The notify branch of `execute`
    reads `result["success"]`, so the stub carries it, as `run_task` does."""
    run_task_stub = adapter_env

    async def _fake_run_task(tid, task=None, **pool_kw):
        return {"status": status, "success": status == "success",
                # The real run_task returns the SLICED closing here, not the whole
                # response — the adapter must see what it actually sees.
                "run_id": "run_x",
                "response_preview": autonomy._outcome_summary(response),
                "meta": meta, "duration_seconds": 61.0,
                "summary": autonomy._outcome_summary(response)}

    run_task_stub["run_task"] = _fake_run_task
    import workers.sources.scheduled_task as scheduled_task
    return await scheduled_task.execute(
        _item(payload={"task_id": 24}, source="scheduled-task"))


@pytest.fixture
def adapter_env(monkeypatch, tmp_path):
    """Point the scheduled-task adapter at a stubbed `run_task` and capture the
    completion notification a person would receive.

    `execute` imports `run_task`, `_find_task_file` and `_parse_task_file` from the
    `autonomy` module INSIDE its body, so patching the module attributes is the
    seam — and the task file must resolve with `notify_on_complete`, or the notify
    assertion passes vacuously with zero calls.
    """
    import app.discord_notify as discord_notify
    import workers.sources as sources
    import workers.sources.scheduled_task as scheduled_task

    notifications: list[tuple] = []

    async def _notify(task_id, name, preview):
        notifications.append((task_id, name, preview))

    seen: dict = {}

    def _find_task_file(tid):
        seen["tid"] = tid
        return tmp_path / f"{tid}-nightly-signal.md"

    def _parse_task_file(path):
        return {"id": 24, "name": "Nightly Signal", "status": "up_next",
                "notify_on_complete": True, "timeout_seconds": 300}

    monkeypatch.setattr(scheduled_task, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {})
    monkeypatch.setattr(autonomy, "_find_task_file", _find_task_file)
    monkeypatch.setattr(autonomy, "_parse_task_file", _parse_task_file)
    monkeypatch.setattr(discord_notify, "_discord_notify_task_complete", _notify)
    holder: dict = {"notifications": notifications, "seen": seen}
    monkeypatch.setattr(autonomy, "run_task",
                        lambda *a, **k: holder["run_task"](*a, **k))
    return holder


def test_the_run_record_and_the_db_row_carry_one_slice(tmp_path, run_driver):
    """Clause 4. `autonomy-runs/<id>/<run>.md`'s `summary:` and `runs.summary` are
    two stores of one fact, written by code 100 lines apart that used to disagree
    (200 vs 300, both heads). Walked end to end — the real `_write_run_record` to
    disk, the returned dict, `normalize_result`, a real `runs` row — so the
    assertion is the JOIN between the two stores, not a tautology about one."""
    captured = run_driver(_text_stream(RESPONSE))
    out = asyncio.run(autonomy.run_task(24))

    record = next((run_driver.runs_dir / "24").glob("run_*.md"))
    text = record.read_text(encoding="utf-8")
    fm = yaml.safe_load(text.split("---")[1])
    assert text.split("## Response\n\n", 1)[1] == FINAL_RESPONSE  # holds it all

    # The same hand-off `scheduled_task.execute` makes to the pool, then to the DB.
    norm = normalize_result(_item(payload={"task_id": 24}), {
        "status": "success", "summary": out["response_preview"],
        "task_id": "24", "run_id": out["run_id"],
        "artifact_path": f"autonomy-runs/24/{out['run_id']}.md",
        "response": out["response_preview"], "meta": out["meta"],
    })
    q = WorkQueue(tmp_path / "closing-seam.db")
    q.record_run("run_scheduled-task_20260920_120000", 1, "scheduled-task",
                 norm["status"], "", "", 61.0, summary=norm["summary"],
                 artifact_path=norm["artifact_path"],
                 response_json=json.dumps(norm.get("response") or ""),
                 task_id=norm["task_id"], meta_json=json.dumps(norm["meta"]))
    row = q.list_runs(limit=1)[0]

    assert fm["summary"] == EXPECTED_CLOSING
    assert row["summary"] == fm["summary"] == captured["record"]["summary"]
    assert len(row["summary"]) == CAP
    assert not row["summary"].startswith("Now let me")


def test_the_two_stores_of_the_summary_are_written_by_one_expression(run_driver):
    """The 200/300 disagreement was two literals 100 lines apart. Both now call the
    same helper, so no future edit can put the record back to the head while the
    ledger stays at the tail."""
    captured = run_driver(_text_stream(RESPONSE))
    out = asyncio.run(autonomy.run_task(24))
    assert captured["record"]["summary"] == autonomy._outcome_summary(FINAL_RESPONSE)
    assert out["response_preview"] == autonomy._outcome_summary(FINAL_RESPONSE)


# ── Clause 5: the health classifier reads meta/status, not narrative ─────────

def _row(run_id: str, status: str, *, summary: str = "", response_json: str = "",
         meta_json: str = "") -> dict:
    return {"run_id": run_id, "task_id": "24", "source": "scheduled-task",
            "status": status,
            "started_at": "", "completed_at": "2026-09-20T12:00:00+00:00",
            "duration_seconds": 60.0, "summary": summary,
            "artifact_path": f"autonomy-runs/24/{run_id}.md",
            "response_json": response_json, "meta_json": meta_json,
            "claims_json": None, "queue_id": 1, "meta": None}


def _timeouts(rows) -> int:
    """Timeouts charged to the fleet — the figure the autonomy tab renders
    (`compute_health`'s `fleet.timeout_runs`)."""
    return autonomy.compute_health(rows, [_task()], days=7)["fleet"]["timeout_runs"]


def test_a_success_row_whose_report_mentions_a_timeout_is_not_counted_as_one():
    """The live misclassification. `run_scheduled-task_20260909_084821_30159b` is
    `status='success'`, its meta carries `stop_reason`/`usage`/`num_turns` and no
    timeout flag, and `compute_health` counted it as a timeout because its summary
    says "already retrying the CV files that timed out last run". Below is that row
    in the shape this round creates — the sentence inside the LAST 300 characters,
    which is now where such prose lands. Report prose cannot overturn a verdict."""
    recovering = ("Step 1 is extracting; the earlier attempt was cut off, and this "
                  "pass retried the CV files that timed out last run. 1,109 files "
                  "processed, 0 failed, handoff written to the vault.")
    assert "timed out" in recovering[-CAP:]
    rows = [_row("run_scheduled-task_20260920_084821_30159b", "success",
                 summary=autonomy._outcome_summary(recovering),
                 response_json=autonomy._outcome_summary(recovering),
                 meta_json=json.dumps({"stop_reason": "stop", "num_turns": 19,
                                       "tool_errors": 1, "silent": False}))]
    assert _timeouts(rows) == 0


def test_a_failed_row_with_no_meta_is_still_counted_as_a_timeout():
    """The regression the clause demands: 143 of the 163 `TimeoutError`-prefixed
    rows in `workers.db` are `status='failed'` with no `meta_json` at all — for
    them the error string IS the evidence, and it must keep counting."""
    rows = [_row("run_scheduled-task_20260914_000001_aaaaaa", "failed",
                 summary="TimeoutError: autonomy run exceeded 1200s",
                 response_json="")]
    assert _timeouts(rows) == 1
    rows = [_row("run_scheduled-task_20260914_000002_bbbbbb", "failed",
                 summary="TimeoutError: autonomy run exceeded 1200s",
                 meta_json=json.dumps({"failure_kind": "infra"}))]
    assert _timeouts(rows) == 1


def test_a_run_with_timeout_meta_or_an_interrupted_status_still_counts():
    """The signal that does not depend on prose stays sufficient."""
    meta = json.dumps({"timeout": True, "timeout_seconds": 1200})
    rows = [_row("run_scheduled-task_20260920_000003_cccccc", "failed",
                 summary="engine stalled; partial output below", meta_json=meta)]
    assert _timeouts(rows) == 1
    rows = [_row("run_scheduled-task_20260920_000004_dddddd", "failed",
                 summary="cancelled by the pool",
                 meta_json=json.dumps({"pool_timeout": True}))]
    assert _timeouts(rows) == 1
    rows = [_row("run_scheduled-task_20260920_000005_eeeeee", "interrupted",
                 summary="recovered by the boot sweep")]
    assert _timeouts(rows) == 1


def test_the_silent_sentinel_now_reaches_the_ledger_line_it_is_read_from(run_driver):
    """The other classifier in the same loop: `[SILENT]` is what a run with nothing
    to report writes, and it writes it at the END. Under the head slice the marker
    could not be in the ledger line at all once a run had narrated, so any consumer
    reading that line saw prose with no verdict. The run is still judged silent
    through `meta`, which is what `compute_health` reads."""
    narration = ("Now let me look at the inbox, then the calendar, then the reminders. "
                 * 10 + "\n")
    # Two assistant turns, as a run that worked before it answered actually emits
    # them: the sentinel is the TERMINAL block, which is what `run_task` reads.
    stream = (_text_stream(narration.rstrip()) + _text_stream("[SILENT]"))
    text = narration + "[SILENT]\n"
    captured = run_driver(stream)
    out = asyncio.run(autonomy.run_task(24))
    assert captured["record"]["summary"].rstrip().endswith("[SILENT]")
    assert "[SILENT]" not in text[:CAP]        # the head slice could never reach it
    assert out["meta"]["silent"] is True       # judged silent through meta
    assert out["meta"]["silent"] is True
    row = _row("run_scheduled-task_20260920_000006_ffffff", "success",
               summary=captured["record"]["summary"], meta_json=json.dumps(out["meta"]))
    health = autonomy.compute_health([row], [_task()], days=7)
    assert sum(t["silent"] for t in health["tasks"]) == 1   # judged through meta, not prose
    assert health["fleet"]["timeout_runs"] == 0


# ── Clause 6: the completion notification still behaves ─────────────────────

async def test_a_notified_run_announces_the_reports_closing_once(monkeypatch,
                                                                 adapter_env):
    """The person-facing surface. `preview` reaches `_discord_notify_task_complete`
    straight from `response_preview`, so the same slice that cut the ledger in half
    was also what a human was sent."""
    out = await _drive_execute(monkeypatch, adapter_env, response=FINAL_RESPONSE,
                               meta={"silent": False})
    assert out["status"] == "success"
    assert len(adapter_env["notifications"]) == 1
    _tid, _name, preview = adapter_env["notifications"][0]
    assert VERDICT in preview
    assert preview == EXPECTED_CLOSING
    assert not preview.startswith("Now let me")


async def test_a_silent_run_does_not_notify(monkeypatch, adapter_env):
    """`"[SILENT]" not in preview` must keep suppressing the message. With the tail
    slice the marker is now inside the window, so the guard is strictly better than
    it was — the head slice could not see a closing `[SILENT]` at all."""
    text = "Scanned the inbox and the calendar; nothing needed you. " * 4 + "\n[SILENT]"
    await _drive_execute(monkeypatch, adapter_env, response=text, meta={"silent": True})
    assert adapter_env["notifications"] == []


async def test_the_notify_preflight_reads_the_task_file_via_run_task(monkeypatch,
                                                                     adapter_env):
    """The `seen` record: proves the adapter really entered the notify branch —
    `_find_task_file` was consulted with the queue item's task id — so the
    silent-run test above is a real refusal and not an early return."""
    out = await _drive_execute(monkeypatch, adapter_env, response=RESPONSE,
                               meta={"silent": False})
    assert out["status"] == "success"
    assert adapter_env["seen"]["tid"] == 24   # `execute` coerces with int(task_id)


# ── #1507: one [SILENT] predicate, and a MENTION is not a decline ────────────
#
# The run record and the Discord helper read the sentinel by exact match; the
# adapter and the health rollup read it by substring. A report that mentions
# the token mid-sentence — the 2026-09-03 case — was recorded as a real run,
# had its notification dropped, and counted as declined in `silent_rate`.

MENTION = ("Checked the feeds. Nothing is urgent, but I did not answer [SILENT] "
           "because two items need you: the renewal and the calendar clash.")


def test_the_predicate_is_exact_match():
    from app.silent_sentinel import is_silent_response, run_is_silent
    assert is_silent_response("[SILENT]")
    assert is_silent_response("  [SILENT]\n")
    assert not is_silent_response(MENTION)
    assert not is_silent_response("[SILENT] nothing new")
    assert not is_silent_response("")
    assert not is_silent_response(None)
    # A recorded flag wins either way; a row without one falls back to exact.
    assert run_is_silent({"silent": True}, MENTION)
    assert not run_is_silent({"silent": False}, "[SILENT]")
    assert not run_is_silent({}, MENTION)
    assert run_is_silent(None, "[SILENT]")


def test_a_mention_is_not_silent_in_the_run_record(run_driver):
    run_driver(_text_stream(MENTION))
    out = asyncio.run(autonomy.run_task(24))
    assert out["meta"]["silent"] is False


def test_a_mention_is_a_reporting_run_in_the_health_rollup():
    """The assertion none of the four sites pinned: the metric agrees with the
    record. One row with the recorded flag, one legacy row without meta."""
    rows = [
        _row("run_scheduled-task_20260920_000010_aaaaaa", "success",
             summary=MENTION, response_json=MENTION,
             meta_json=json.dumps({"silent": False})),
        _row("run_scheduled-task_20260920_000011_bbbbbb", "success",
             summary=MENTION, response_json=MENTION),
        _row("run_scheduled-task_20260920_000012_cccccc", "success",
             summary="[SILENT]", response_json="[SILENT]"),
    ]
    health = autonomy.compute_health(rows, [_task()], days=7)
    (task,) = [t for t in health["tasks"] if str(t["task_id"]) == "24"]
    assert task["runs"] == 3
    assert task["silent"] == 1, task
    assert task["silent_rate"] == round(1 / 3, 3)


async def test_a_mention_still_notifies_through_the_adapter(monkeypatch, adapter_env):
    await _drive_execute(monkeypatch, adapter_env, response=MENTION,
                         meta={"silent": False})
    assert len(adapter_env["notifications"]) == 1
    await _drive_execute(monkeypatch, adapter_env, response=MENTION, meta={})
    assert len(adapter_env["notifications"]) == 2


async def test_the_discord_helper_posts_a_mention_and_drops_the_sentinel(monkeypatch):
    import app.discord_notify as discord_notify
    import httpx

    posts: list = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            posts.append(kw["json"])

    monkeypatch.setitem(discord_notify.CONFIG, "discord",
                        {"home_channel": "123", "token": "t"})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    await discord_notify._discord_notify_task_complete(24, "Nightly Signal", MENTION)
    await discord_notify._discord_notify_task_complete(24, "Nightly Signal", " [SILENT]\n")
    assert len(posts) == 1
    assert posts[0]["embeds"][0]["description"] == MENTION


def test_no_reader_of_the_sentinel_uses_a_substring():
    """Four sites, one predicate: none may test the token by containment."""
    for rel in ("autonomy.py", "app/discord_notify.py",
                "workers/sources/scheduled_task.py"):
        src = (_REPO / rel).read_text(encoding="utf-8")
        assert not re.search(r'"\[SILENT\]"\s+(not\s+)?in\b', src), rel
        assert '.strip() == "[SILENT]"' not in src, rel


def test_the_autonomy_router_does_not_import_the_notify_helper():
    src = (_REPO / "app/routers/autonomy.py").read_text(encoding="utf-8")
    assert "_discord_notify_task_complete" not in src
