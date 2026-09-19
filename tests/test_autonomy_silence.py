"""The `[SILENT]` contract is about the run's LAST block, not everything it said.

Why this file exists
--------------------
`run_task` accumulates one string — the join of every `text_delta` of every
agent-loop iteration — and used to read four different verdicts off it,
including the two that decide whether the run asked to be left alone:

    is_silent      = final_response.strip() == "[SILENT]"
    silent_failures = _detect_silent_failures(final_response, ...)

Equality against a join means a run that narrated while it worked — "Now let me
classify the hits" — can never equal the sentinel, whatever it signed off with.
The harness prints that contract into every autonomy prompt, so it was
unreachable for exactly the runs that did work. Measured on this machine over
task #68's records for 2026-09-14/16: 173 records, 104 of which end their
`## Response` on a line that is exactly `[SILENT]`, and 65 of those 104 were
written `silent: false`. The sentinel landed in under 40 % of the runs that
emitted it.

The fix is a second accumulator, not new plumbing: the harness already flushes
one `assistant_message` event per iteration carrying that block's own text
(`app/harness/events.py::assistant_message`, emitted at the end of an iteration
in `app/harness/loop.py`), so `last_block` is a sibling assignment beside the
`text_delta` one.

What stays on the join, deliberately, because other items own it: the `##
Response` body and the transcript (clause 3 below — deciding silence on the last
block must lose nothing), and the `summary` / `response_preview` slicing, which
is #642's, and the empty-response status decision, whose terminal-block half is
#832's. The delivery-side substring test in
`workers/sources/scheduled_task.py` runs against the 300-char head preview and
is #642's to move as well.
"""
from __future__ import annotations

import asyncio

import pytest
import yaml

import autonomy

NARRATION = ("Now let me classify the hits against prefs and dedup state.\n\n"
             "**Calendar:** nothing in the window.")
PROSE = "1 new actionable email: invoice from the vet. Injected it."
# A phrase `_SILENT_FAILURE_PATTERNS` matches, sitting in the narration where
# the old scan read it and the new one must not.
INDICATOR = "The probe failed because the media port is unbound."


def _assistant_blocks(*blocks: str, tool_after: int | None = None) -> list[dict]:
    """Build a harness event stream from a sequence of assistant blocks.

    Each block becomes what the real loop emits for one iteration: a
    `text_delta` chunk carrying the block's text plus an `assistant_message`
    flush carrying that block's OWN text — not the accumulation, which is why
    the last one identifies how the run ended. `tool_after` drops a
    tool_call/tool_result pair after that 1-based iteration, because the runs
    this pins are runs that worked before they answered.
    """
    events: list[dict] = []
    for i, text in enumerate(blocks, start=1):
        # Newline-terminated like a real assistant turn. `final_response` is a
        # bare concatenation with no separator inserted, so without this the
        # join would run the blocks together into one line and the fixture
        # would not look like the records it is meant to reproduce.
        text = text + "\n"
        events.append({"type": "text_delta", "text": text})
        events.append({"type": "assistant_message", "text": text,
                       "tool_calls": [], "iteration": i,
                       "finish_reason": "tool_calls" if tool_after == i else "stop"})
        if tool_after == i:
            events.append({"type": "tool_call", "call_id": f"c{i}", "name": "Bash",
                           "args_json": "{}", "summary": "Classifying"})
            events.append({"type": "tool_result", "call_id": f"c{i}",
                           "content": "classified 4 hits"})
    events.append({"type": "result", "stop_reason": "stop",
                   "num_turns": len(blocks), "usage": {}})
    return events


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Real `_write_run_record`, pointed off the real runs directory.

    The assertion that matters is what a person greps — `^silent:` in the
    front matter of `autonomy-runs/<id>/<run>.md` — so the record is written
    for real rather than captured as kwargs.
    """
    d = tmp_path / "autonomy-runs"
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", d)
    return d


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Stub everything around `run_task` except the event loop itself."""
    captured: dict = {}

    def _stub(task: dict, events: list[dict]):
        monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "t.md")
        monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
        monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
        monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
        monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
        monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
        # Watching is a separate mechanism with its own tests; a round of
        # Inner Voice here would only make the fixture slower.
        monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
        # Transcript recording has its own file; here it would only add
        # directories to keep alive.
        monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False)

        async def _run_query(messages, options):
            captured["prompt"] = messages[0]["content"]
            for evt in events:
                yield evt

        class Opts:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        import app.harness as harness_mod
        import app.harness.mcp_pool as mcp_pool
        monkeypatch.setattr(harness_mod, "run_query", _run_query)
        monkeypatch.setattr(harness_mod, "RunOptions", Opts)
        monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
        monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")
        monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
        return captured

    return _stub


def _task(task_id: int = 68, **extra) -> dict:
    task = {"id": task_id, "name": "Email & Calendar Triage", "skill_name": "triage",
            "status": "up_next", "timeout_seconds": 300,
            "notify_on_complete": True}
    task.update(extra)
    return task


def _record(runs_dir, task_id: int) -> tuple[dict, str]:
    """(front matter, whole file) for the single run record this test wrote."""
    files = list((runs_dir / str(task_id)).glob("run_*.md"))
    assert len(files) == 1, [p.name for p in files]
    text = files[0].read_text(encoding="utf-8")
    head, fm, body = text.split("---\n", 2)
    assert head == ""
    return yaml.safe_load(fm), body


def _front_silent(runs_dir, task_id: int) -> bool:
    fm, _ = _record(runs_dir, task_id)
    return fm["silent"]


# ── clause 1: narrated, then [SILENT] ────────────────────────────────────────

def test_a_run_that_narrated_and_signed_off_silent_is_recorded_silent(
        harness, runs_dir):
    """The defect, pinned. `silent: false` here is 65 of the 104 sentinel-terminated
    task-#68 records from Sep 14-16 saying the opposite of what the run asked.

    Under the join this fails: `NARRATION + "[SILENT]"` has a non-empty prefix,
    so equality against the sentinel is false and the record says `false`.
    """
    harness(_task(), _assistant_blocks(NARRATION, "[SILENT]", tool_after=1))

    out = asyncio.run(autonomy.run_task(68))

    assert out["success"] is True
    assert out["meta"]["silent"] is True
    assert _front_silent(runs_dir, 68) is True


def test_the_sentinel_still_honours_the_run_when_no_block_event_arrives(
        harness, runs_dir):
    """The fallback, pinned. `last_block` is None until an
    `assistant_message` is seen, so a stream that carries only `text_delta`
    (which is what every other autonomy test in this suite feeds) keeps the
    pre-existing behaviour instead of reading an empty terminal block.
    """
    harness(_task(), [
        {"type": "text_delta", "text": "[SILENT]"},
        {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}},
    ])

    out = asyncio.run(autonomy.run_task(68))

    assert out["meta"]["silent"] is True
    assert _front_silent(runs_dir, 68) is True


# ── clause 2: a mid-run [SILENT] is not a verdict ────────────────────────────

def test_a_silence_sentinel_earlier_than_the_last_block_does_not_suppress(
        harness, runs_dir):
    """The sentinel counts as the WHOLE terminal block, never as a substring.

    The accumulation here contains `[SILENT]` — asserted below, because that is
    what makes this test fail for the wrong fix as well as the old one: a
    containment test on the join (`"[SILENT]" in final_response`, the form
    `workers/sources/scheduled_task.py` still uses) records this run as silent
    and buries a run that reported an invoice.
    """
    harness(_task(), _assistant_blocks(
        "[SILENT]", "Checked the inbox; nothing yet.", PROSE))

    out = asyncio.run(autonomy.run_task(68))

    fm, body = _record(runs_dir, 68)
    assert "[SILENT]" in fm["summary"] or "[SILENT]" in body
    assert fm["silent"] is False
    assert out["meta"]["silent"] is False


def test_a_run_that_ends_in_ordinary_prose_is_not_silent(harness, runs_dir):
    harness(_task(), _assistant_blocks(NARRATION, PROSE, tool_after=1))

    out = asyncio.run(autonomy.run_task(68))

    assert _front_silent(runs_dir, 68) is False
    assert out["meta"]["silent"] is False


# ── clause 3: judging silence on the last block loses no transcript ──────────

def test_the_response_body_still_holds_every_turns_text(harness, runs_dir):
    """Silence is a verdict about the last block; the record is still the whole
    run. Dropping the intermediate narration to make the first assertion pass
    would destroy the only record of what an unattended run did.
    """
    harness(_task(), _assistant_blocks(
        NARRATION, "Reading prefs.", "[SILENT]", tool_after=2))

    asyncio.run(autonomy.run_task(68))

    _, body = _record(runs_dir, 68)
    response = body.split("## Response", 1)[1]
    assert NARRATION in response
    assert "Reading prefs." in response
    assert "[SILENT]" in response
    # The sentinel is the last line a person greps for, exactly as it was
    # before the verdict moved off the join.
    assert [l for l in response.splitlines() if l.strip()][-1].strip() == "[SILENT]"


# ── clause 4: the indicator scan reads the terminal block ────────────────────

def test_an_indicator_only_in_narration_does_not_flag_a_silent_run(harness, runs_dir):
    """`failed because` is a `_SILENT_FAILURE_PATTERNS` hit. Spoken mid-run by
    a run that then declined to be surfaced, it used to brand the run a silent
    failure and write the indicator section into its record.
    """
    harness(_task(), _assistant_blocks(INDICATOR, "[SILENT]"))

    out = asyncio.run(autonomy.run_task(68))

    fm, body = _record(runs_dir, 68)
    assert fm["silent_failure_indicators"] == 0
    assert out["meta"]["silent_failure_indicators"] == 0
    assert "Silent failure indicators detected" not in body
    assert fm["silent"] is True


def test_an_indicator_in_the_final_block_still_flags_the_run(harness, runs_dir):
    """Moving the scan must not neuter it: the same phrase, said last, is still
    the thing that stops a clean-looking failure from reading as a success.
    """
    harness(_task(), _assistant_blocks(NARRATION, INDICATOR))

    out = asyncio.run(autonomy.run_task(68))

    fm, body = _record(runs_dir, 68)
    assert fm["silent_failure_indicators"] == 1
    assert out["meta"]["silent_failure_indicators"] == 1
    assert "Silent failure indicators detected" in body
    assert fm["silent"] is False


# ── clause 5: the empty-response failure is untouched ────────────────────────

def test_a_run_with_no_assistant_text_is_still_a_failure(harness, runs_dir):
    """#832 owns the terminal-block half of run STATUS; this round changes none
    of it. A run that dispatched a tool and said nothing is still recorded
    failed, its summary still begins "empty response after", and `empty` is
    still in the front matter — the guard that ended a week of phantom
    "successes" from an empty window on 2026-09-01.
    """
    harness(_task(), [
        {"type": "tool_call", "call_id": "c1", "name": "Bash",
         "args_json": "{}", "summary": "Triaging"},
        {"type": "tool_result", "call_id": "c1", "content": "did something"},
        {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}},
    ])

    out = asyncio.run(autonomy.run_task(68))

    assert out["success"] is False
    assert out["status"] == "failed"
    assert out["error"].startswith("empty response after")
    fm, _ = _record(runs_dir, 68)
    assert fm["status"] == "failed"
    assert fm["empty"] is True


def test_empty_assistant_blocks_do_not_make_an_empty_run_look_silent(
        harness, runs_dir):
    """`assistant_message` events with no text leave `last_block` as None, so an
    empty run cannot reach the success path and be judged on a terminal block
    that never existed.
    """
    harness(_task(), [
        {"type": "assistant_message", "text": "", "tool_calls": [], "iteration": 1},
        {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}},
    ])

    out = asyncio.run(autonomy.run_task(68))

    assert out["success"] is False
    assert out["error"].startswith("empty response after")
    fm, _ = _record(runs_dir, 68)
    assert fm["empty"] is True
    # `silent` is never written on the failure path at all: no verdict was made.
    assert "silent" not in fm
