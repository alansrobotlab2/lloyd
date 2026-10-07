"""The structured-verdict finalizer: when it runs, what it sends, what it returns.

The load-bearing assertion in this file is
`test_the_finalizer_sends_the_identical_tools_array`. Qwen's chat template
renders `tools` inside the system message, so a finalizer that drops the
array changes the rendered prompt from the first token and vLLM re-prefills
the entire conversation — on a 100k-token turn that is the most expensive
thing this feature could do, to transcribe one object.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.harness import finalizer as F

SCHEMA = {
    "type": "object",
    "title": "verdict",
    "properties": {"verdict": {"type": "string", "enum": ["confirmed", "rejected"]}},
    "required": ["verdict"],
    "additionalProperties": False,
}

# A schema with an open-ended string, which is what autotriage actually sends:
# `TRIAGE_VERDICT_SCHEMA`'s `check`, `evidence` and `acceptance` carry no
# maxLength (tests/test_structured_verdict.py::test_no_maxlength_in_the_schema
# pins that). Only such a schema can legitimately need more room, so it is the
# one whose truncation still gets the budget advice (#1706).
UNBOUNDED = {
    "type": "object",
    "title": "verdict",
    "properties": {"verdict": {"type": "string", "enum": ["confirmed", "rejected"]},
                   "evidence": {"type": "string"}},
    "required": ["verdict", "evidence"],
    "additionalProperties": False,
}

# Every string field capped: an enum, or a finite maxLength. This is
# `deep_research.RESULT_SCHEMA` after #1706 — its grammar cannot ask for more
# than the caps add up to, so a completion that runs past them is the model
# writing junk inside a field, not the budget being too small.
BOUNDED = {
    "type": "object",
    "title": "deep_research_result",
    "properties": {
        "result": {"type": "string", "enum": ["written", "nothing_found", "duplicate"]},
        "note": {"type": "string", "maxLength": 200},
        "duplicate_of": {"type": "string", "maxLength": 200},
        "facts": {"type": "string", "maxLength": 8},
        "sources": {"type": "string", "maxLength": 8},
    },
    "required": ["result", "note", "duplicate_of", "facts", "sources"],
    "additionalProperties": False,
}

# The 200 characters of run_deep-research_20260928_032533_df61c0's completion
# that its own `structured_error` captured: a valid object through
# `"duplicate_of": ""`, then `"facts": ` followed by newlines, spaces and junk
# that never closes. It ran to the 8192-token cap.
DEGENERATE = ('{"result": "written", "note": "/home/alansrobotlab/obsidian/knowledge/'
              'research/2026-09-28-answer-option-order-sensitivity-in-constrained-'
              'llm-decision-.md", "duplicate_of": "", "facts": \n\n       "}: 8,')

# Item #2226's `vault_review` grading, cut where the engine cut it: clause 1
# whole, then `"te` — the key `test_honesty` opens with. Copied verbatim from the
# `findings` field of that row in ~/.local/state/lloyd-automod/promotions.jsonl
# (`grep '"item_id": 2226'`), where it rode inside the message
# `output truncated at 8192 tokens — raise harness.finalizer.max_tokens`. Note the
# first clause is COMPLETE and its `note` is the empty string: what this fragment
# can prove is that the object ran past a grammar that admitted unbounded strings,
# and it cannot prove which field spent the budget — the ledger stores only the
# first 200 characters (`finalizer.py`, `content[:200]!r`).
REVIEW_CUT = ('{"premise":"sound","clauses":[{"clause":1,"verdict":"met",'
              '"evidence_path":"skills/nightly-reflection-signals/SKILL.md:419-426",'
              '"evidence_line":0,"test_node_id":"","how_verified":"read",'
              '"note":""}],"te')

MESSAGES = [
    {"role": "system", "content": "you are lloyd"},
    {"role": "user", "content": "triage item 42"},
    {"role": "assistant", "content": "VERDICT: confirmed"},
]

TOOLS = [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.text = text or json.dumps(self._body)

    def json(self):
        return self._body


def _ok(content, usage=None, finish_reason="stop"):
    return _Resp(200, {"choices": [{"message": {"content": content},
                                    "finish_reason": finish_reason}],
                       "usage": usage or {}})


class _Client:
    """Stands in for httpx.AsyncClient, recording every request body."""

    posted: list[dict] = []
    responses: list = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        _Client.posted.append(json)
        if _Client.responses:
            r = _Client.responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        return _ok('{"verdict": "confirmed"}')


@pytest.fixture(autouse=True)
def _client(monkeypatch):
    _Client.posted = []
    _Client.responses = []
    monkeypatch.setattr(F.httpx, "AsyncClient", _Client)
    return _Client


async def _run(**kw):
    args = dict(base_url="http://x:8096", model="primary", chat_messages=MESSAGES,
                tools=TOOLS, schema=SCHEMA)
    args.update(kw)
    return await F.run_finalizer(**args)


# ── the skip rule ───────────────────────────────────────────────────────────

def test_no_schema_means_no_finalizer():
    assert F.should_finalize("stop", None) == (False, "")


def test_a_turn_that_died_at_its_budget_has_no_verdict_to_restate():
    """Forcing one recreates the failure INCOMPLETE was added to fix."""
    run, reason = F.should_finalize("max_turns", SCHEMA)
    assert run is False
    assert reason == "skipped: stop_reason=max_turns"


def test_a_cancelled_turn_is_skipped():
    assert F.should_finalize("cancelled", SCHEMA)[0] is False


def test_a_broken_stream_has_no_verdict_to_restate():
    """D7: a turn that ended on a stream it could not retry holds only the
    text that streamed before the break — restating it would invent a verdict
    the model never reached."""
    assert "stream_error" not in F.FINALIZABLE_STOP_REASONS
    assert F.should_finalize("stream_error", SCHEMA) == (
        False, "skipped: stop_reason=stream_error")


def test_a_finished_turn_runs():
    assert F.should_finalize("stop", SCHEMA) == (True, "")
    assert F.should_finalize("end_turn", SCHEMA) == (True, "")


# ── the request ─────────────────────────────────────────────────────────────

async def test_the_finalizer_sends_the_identical_tools_array(_client):
    await _run()
    body = _client.posted[0]
    assert body["tools"] == TOOLS, "dropping tools re-prefills the conversation"
    assert body["tool_choice"] == "none"


async def test_it_sends_the_vllm_spelling_first(_client):
    await _run()
    body = _client.posted[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "json_schema" not in body or body.get("json_schema") is None


async def test_a_400_falls_through_to_the_llamacpp_spelling(_client):
    _client.responses = [_Resp(400, {}, "unknown field response_format"),
                         _ok('{"verdict": "rejected"}')]
    parsed, error, _usage = await _run()
    assert error == ""
    assert parsed == {"verdict": "rejected"}
    assert len(_client.posted) == 2
    assert "response_format" not in _client.posted[1]
    assert _client.posted[1]["json_schema"] == SCHEMA
    # And still the identical tools, on the retry too.
    assert _client.posted[1]["tools"] == TOOLS


async def test_the_restate_prompt_is_appended_to_a_copy(_client):
    before = list(MESSAGES)
    await _run()
    assert MESSAGES == before, "the loop's live buffer must not be mutated"
    sent = _client.posted[0]["messages"]
    assert len(sent) == len(MESSAGES) + 1
    assert sent[-1]["role"] == "user"
    assert "single JSON object" in sent[-1]["content"]


async def test_a_custom_prompt_replaces_the_default(_client):
    await _run(prompt="restate it as the schema, nothing else")
    assert _client.posted[0]["messages"][-1]["content"] == \
        "restate it as the schema, nothing else"


async def test_budgets_reach_the_request(_client):
    await _run(max_tokens=64, priority=1)
    assert _client.posted[0]["max_tokens"] == 64
    assert _client.posted[0]["priority"] == 1
    assert _client.posted[0]["stream"] is False


async def test_no_tools_means_no_tool_choice(_client):
    await _run(tools=None)
    assert "tools" not in _client.posted[0]
    assert "tool_choice" not in _client.posted[0]


# ── the response ────────────────────────────────────────────────────────────

async def test_a_valid_object_comes_back_parsed(_client):
    parsed, error, _usage = await _run()
    assert parsed == {"verdict": "confirmed"} and error == ""


async def test_non_json_output_is_an_error_not_an_exception(_client):
    _client.responses = [_ok("I think it is confirmed.")]
    parsed, error, _ = await _run()
    assert parsed is None and "not JSON" in error


async def test_a_json_array_is_refused(_client):
    _client.responses = [_ok('["confirmed"]')]
    parsed, error, _ = await _run()
    assert parsed is None and "not an object" in error


async def test_a_500_is_reported_without_retrying_the_other_spelling(_client):
    _client.responses = [_Resp(500, {}, "boom")]
    parsed, error, _ = await _run()
    assert parsed is None and "HTTP 500" in error
    assert len(_client.posted) == 1


async def test_a_transport_failure_is_reported(_client):
    _client.responses = [RuntimeError("connection reset")]
    parsed, error, _ = await _run()
    assert parsed is None and "connection reset" in error


async def test_both_spellings_rejected_is_reported(_client):
    _client.responses = [_Resp(400, {}, "no"), _Resp(400, {}, "also no")]
    parsed, error, _ = await _run()
    assert parsed is None and "no accepted schema spelling" in error


async def test_a_cancelled_event_short_circuits(_client):
    ev = asyncio.Event()
    ev.set()
    parsed, error, _ = await _run(cancel_event=ev)
    assert parsed is None and "cancelled" in error
    assert _client.posted == []


async def test_usage_is_translated_into_harness_key_names(_client):
    _client.responses = [_ok('{"verdict": "confirmed"}', usage={
        "prompt_tokens": 100_000, "completion_tokens": 40, "total_tokens": 100_040,
        "prompt_tokens_details": {"cached_tokens": 99_900}})]
    _parsed, _error, usage = await _run()
    assert usage == {"input_tokens": 100_000, "output_tokens": 40,
                     "total_tokens": 100_040, "cached_tokens": 99_900}


# ── the loop's side ─────────────────────────────────────────────────────────

async def test_the_loop_skips_and_records_the_reason(monkeypatch):
    from app.harness import loop as L
    from app.harness.options import RunOptions

    opts = RunOptions(model="primary", final_schema=SCHEMA)
    parsed, error = await L._maybe_finalize(
        options=opts, stop_reason="max_turns", chat_messages=[], tools=None,
        total_usage={})
    assert parsed is None and error == "skipped: stop_reason=max_turns"


async def test_the_loop_folds_finalizer_tokens_into_the_turns_usage(monkeypatch):
    from app.harness import loop as L
    from app.harness.options import RunOptions

    async def fake(**kw):
        return {"verdict": "confirmed"}, "", {"output_tokens": 40, "total_tokens": 90}
    monkeypatch.setattr("app.harness.finalizer.run_finalizer", fake)

    total = {"output_tokens": 1000, "total_tokens": 5000}
    opts = RunOptions(model="primary", final_schema=SCHEMA)
    parsed, error = await L._maybe_finalize(
        options=opts, stop_reason="stop", chat_messages=[], tools=None,
        total_usage=total)
    assert parsed == {"verdict": "confirmed"} and error == ""
    assert total == {"output_tokens": 1040, "total_tokens": 5090,
                     "finalizer_output_tokens": 40}


async def test_nothing_raises_out_of_the_loops_finalizer(monkeypatch):
    from app.harness import loop as L
    from app.harness.options import RunOptions

    async def boom(**kw):
        raise RuntimeError("engine gone")
    monkeypatch.setattr("app.harness.finalizer.run_finalizer", boom)
    parsed, error = await L._maybe_finalize(
        options=RunOptions(model="primary", final_schema=SCHEMA),
        stop_reason="stop", chat_messages=[], tools=None, total_usage={})
    assert parsed is None and "engine gone" in error


def test_the_result_event_carries_both_fields():
    from app.harness import events
    evt = events.result(stop_reason="stop", structured={"verdict": "confirmed"})
    assert evt["structured"] == {"verdict": "confirmed"}
    assert evt["structured_error"] == ""
    plain = events.result(stop_reason="stop")
    assert plain["structured"] is None


def test_config_maps_the_finalizer_budgets(monkeypatch):
    from app import mcp_discovery as D
    from app.config import CONFIG

    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "finalizer": {"max_tokens": 256, "timeout_seconds": 30}})
    kw = D._get_harness_kwargs()
    assert kw["finalizer_max_tokens"] == 256
    assert kw["finalizer_timeout_s"] == 30.0


# ── the budget, and naming a truncation as one ──────────────────────────────

async def test_a_cut_off_object_is_reported_as_truncated_not_as_not_json():
    """14 of the first 34 triage verdicts came back as well-formed JSON cut
    mid-string and were recorded "output is not JSON" — the same message a
    model that wrote prose would get, which hid a budget problem behind a
    model-behaviour one for two days. The engine says `finish_reason: length`;
    when it does not, an unclosed `{` is the tell.

    Run against `UNBOUNDED`, the schema this failure actually happened to:
    triage's open-ended `evidence` field can need room, so the budget is the
    right advice there. Under a schema that caps every field it is not, and
    #1706 split those two cases apart — see the two nodes below."""
    _Client.responses = [_ok('{"verdict":"confirmed","check":"ls _pipe',
                             usage={"completion_tokens": 1024},
                             finish_reason="length")]
    parsed, error, usage = await _run(schema=UNBOUNDED)
    assert parsed is None
    assert "truncated at 1024 tokens" in error and "max_tokens" in error
    assert usage["output_tokens"] == 1024

    _Client.responses = [_ok('{"verdict":"confirmed","evidence":"…',
                             finish_reason="")]
    parsed, error, _ = await _run(schema=UNBOUNDED)
    assert parsed is None and "truncated" in error

    _Client.responses = [_ok("I think the verdict is confirmed.")]
    parsed, error, _ = await _run(schema=UNBOUNDED)
    assert parsed is None and "not JSON" in error and "truncated" not in error


async def test_a_cut_under_a_schema_that_caps_every_field_is_a_divergence():
    """#1706: deep-research sent a 5-field object and 6 of 12 post-#710 runs
    died at `output truncated at 8192 tokens — raise harness.finalizer.max_tokens`,
    which sent every reader to a config knob that cannot help. With `maxLength`
    on every string the grammar has a ceiling far below 8192, so a completion
    that reaches the cap anyway is the model writing whitespace and junk inside
    a field — a generation failure, and naming a budget for it is a lie."""
    _Client.responses = [_ok(DEGENERATE, usage={"completion_tokens": 8192},
                             finish_reason="length")]
    parsed, error, usage = await _run(schema=BOUNDED)
    assert parsed is None
    assert "generation diverged" in error, error
    assert "raise harness.finalizer.max_tokens" not in error, (
        "the reason still sends the operator to the budget: " + error)
    assert "8192 tokens" in error, "the reason must still say how far it ran"
    assert usage["output_tokens"] == 8192


async def test_an_unclosed_bounded_object_without_the_length_reason_diverges_too():
    """The engine does not always say `finish_reason: length`, so the unclosed
    `{` is the only tell — and under a bounded grammar it means the same
    divergence, not a budget."""
    _Client.responses = [_ok(DEGENERATE, finish_reason="")]
    parsed, error, _ = await _run(schema=BOUNDED)
    assert parsed is None
    assert "generation diverged" in error and "max_tokens" not in error, error


async def test_a_cut_under_the_automod_review_schema_is_a_divergence_too():
    """#2240: the review grader's own answer shape carried eight open strings, so
    a grading that ran to the budget was reported AS a budget, and the reader was
    sent to `harness.finalizer.max_tokens` — nine `vault_review` rows of the
    promotion ledger are that message (items 868, 1233, 1563, 1618, 1885, 1886,
    2126, 2230, 2226), each of them `kind: skipped` with `clauses: []`, the change
    landed and no clause graded. The two stand-in schemas above pin the split in
    the abstract; this one pins the review schema to it, because "capped" is a
    property of a schema somebody has to actually go and cap.

    The control underneath it is the same completion under the grammar as it
    stood before #2240, every `maxLength` stripped: still the budget advice. Which
    of the two messages a truncation becomes is decided by the schema alone."""
    import copy

    from scripts.automod.review import REVIEW_SCHEMA

    def _strip(node):
        if isinstance(node, dict):
            node.pop("maxLength", None)
            for child in node.values():
                _strip(child)
        elif isinstance(node, list):
            for child in node:
                _strip(child)

    _Client.responses = [_ok(REVIEW_CUT, usage={"completion_tokens": 8192},
                             finish_reason="length")]
    parsed, error, usage = await _run(schema=REVIEW_SCHEMA)
    assert parsed is None
    assert "generation diverged" in error, error
    assert "raise harness.finalizer.max_tokens" not in error, (
        "REVIEW_SCHEMA reads as unbounded again and its reader is back at the "
        "config knob: " + error)
    assert "8192 tokens" in error, error
    assert usage["output_tokens"] == 8192

    uncapped = copy.deepcopy(REVIEW_SCHEMA)
    _strip(uncapped)
    assert F._schema_is_bounded(uncapped) is False, (
        "stripping caps left the schema bounded — the control below proves nothing")
    _Client.responses = [_ok(REVIEW_CUT, usage={"completion_tokens": 8192},
                             finish_reason="length")]
    parsed, error, _ = await _run(schema=uncapped)
    assert parsed is None
    assert "raise harness.finalizer.max_tokens" in error, error


async def test_a_completion_that_never_left_thinking_is_still_the_budget():
    """The control on the two nodes above, and a shape that is real: thinking
    is on for the finalizer (#1431) and its tokens come out of the same
    `max_tokens`, so a 200-token probe on the live engine returned
    `finish_reason: length` with no content at all, every token of it
    reasoning. That IS the budget even under a bounded grammar, and the advice
    to raise it is correct — so the divergence message may not swallow it."""
    _Client.responses = [_ok(None, usage={
        "completion_tokens": 200,
        "completion_tokens_details": {"reasoning_tokens": 200}},
        finish_reason="length")]
    parsed, error, usage = await _run(schema=BOUNDED)
    assert parsed is None
    assert "truncated at 200 tokens" in error and "max_tokens" in error, error
    assert "diverged" not in error
    assert usage["reasoning_tokens"] == 200


def test_the_default_budget_is_8192_and_config_agrees():
    """The budget is the whole completion, thinking included, and the grammar
    only applies after </think>. 1024 truncated 41% of triage verdicts; 4096
    truncated the review grader's object on its second calibration case; the
    config value is what production runs, so the two must not drift — this
    test blocked round SM_20260911_031441 at the `tests` rung when config
    moved to 8192 and the code default did not."""
    import re
    from pathlib import Path
    from app.harness.options import RunOptions

    assert RunOptions(model="primary").finalizer_max_tokens == 8192
    cfg = Path(F.__file__).parents[2].joinpath("config.yaml").read_text()
    block = re.search(r"\n  finalizer:\n((?:    .*\n)+)", cfg)
    assert block, "harness.finalizer block missing from config.yaml"
    assert re.search(r"^\s+max_tokens:\s+8192\s*$", block.group(1), re.M)


async def test_the_finalizers_own_output_tokens_get_their_own_key():
    """So the ledger can carry the cost per verdict beside the truncation."""
    from app.harness import loop as L
    from app.harness.options import RunOptions

    async def fake(**kw):
        return {"verdict": "confirmed"}, "", {"output_tokens": 300, "total_tokens": 900,
                                              "input_tokens": 600}
    import app.harness.finalizer as FF
    orig = FF.run_finalizer
    FF.run_finalizer = fake
    try:
        total: dict = {"output_tokens": 10}
        parsed, error = await L._maybe_finalize(
            options=RunOptions(model="primary", final_schema=SCHEMA),
            stop_reason="stop", chat_messages=[], tools=None, total_usage=total)
    finally:
        FF.run_finalizer = orig
    assert parsed == {"verdict": "confirmed"} and error == ""
    assert total["output_tokens"] == 310
    assert total["finalizer_output_tokens"] == 300
    assert total["finalizer_input_tokens"] == 600


# ── #1431: the reasoning tax is measured, and no thinking knob is sent ──────

_THINKING_KEYS = ("reasoning_effort", "chat_template_kwargs", "enable_thinking",
                  "thinking_token_budget")


async def test_the_engines_reasoning_tokens_survive_the_usage_mapping(_client):
    _client.responses = [_ok('{"verdict": "confirmed"}', usage={
        "prompt_tokens": 3791, "completion_tokens": 431, "total_tokens": 4222,
        "completion_tokens_details": {"reasoning_tokens": 252}})]
    parsed, error, usage = await _run()
    assert parsed == {"verdict": "confirmed"} and error == ""
    assert usage["reasoning_tokens"] == 252
    assert usage["output_tokens"] == 431


async def test_an_engine_that_reports_no_details_gets_no_reasoning_key(_client):
    """Absent is not zero: llama.cpp and older vLLM builds omit the block."""
    _client.responses = [_ok('{"verdict": "confirmed"}', usage={
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})]
    _, _, usage = await _run()
    assert "reasoning_tokens" not in usage


async def test_the_loop_reports_finalizer_reasoning_beside_its_output(monkeypatch):
    """Its own key, and never folded into output_tokens a second time: the
    reasoning tokens are already inside completion_tokens."""
    from app.harness import loop as L
    from app.harness.options import RunOptions

    async def fake(**kw):
        return {"verdict": "confirmed"}, "", {
            "output_tokens": 431, "total_tokens": 4222, "input_tokens": 3791,
            "reasoning_tokens": 252}
    monkeypatch.setattr("app.harness.finalizer.run_finalizer", fake)

    total = {"output_tokens": 1000, "total_tokens": 5000}
    await L._maybe_finalize(
        options=RunOptions(model="primary", final_schema=SCHEMA),
        stop_reason="stop", chat_messages=[], tools=None, total_usage=total)
    assert total["finalizer_reasoning_tokens"] == 252
    assert total["finalizer_output_tokens"] == 431
    assert total["output_tokens"] == 1431


def test_the_worker_run_carries_the_finalizer_reasoning_tokens():
    """Loop usage -> done frame -> the worker's per-run dict, so a source can
    report its own reasoning tax. Source files, not live attributes: see
    tests/test_structured_verdict.py on `_run_turn` being replaced by a stub."""
    from pathlib import Path
    from app.routers import messages as M
    from workers.sources import _common

    router = Path(M.__file__).read_text()
    assert '.get("finalizer_reasoning_tokens"))' in router
    assert "done_payload['finalizer_reasoning_tokens']" in router
    worker = Path(_common.__file__).read_text()
    assert '"finalizer_reasoning_tokens": None' in worker
    assert 'out["finalizer_reasoning_tokens"] = data.get(' in worker


async def test_the_finalizer_sends_no_thinking_knob_in_either_spelling(_client):
    """The cache pin behind #1431's trade-off. On the primary's template both
    `reasoning_effort: "none"` and `chat_template_kwargs.enable_thinking:
    false` remove the system message's opening "Reasoning effort is set to
    xhigh" sentence, so the rendered prompt diverges at character 19 and the
    whole turn re-prefills (finalizer.py docstring). Changing this needs a
    knob measured against the live engine not to touch the prefix."""
    _client.responses = [_Resp(400, text="unknown response_format"),
                         _ok('{"verdict": "confirmed"}')]
    await _run()
    assert len(_client.posted) == 2
    for body in _client.posted:
        for key in _THINKING_KEYS:
            assert key not in body, key


# ── #2370: one re-sample when a capped schema diverges ────────────────────────
#
# The 10 board-steward ticks that died naming "not a budget" between
# 2026-10-04T23:51Z and 2026-10-07T15:12Z each spent the full 8192 output tokens
# on `{"moves":[],"next_pick":` followed by CR and nothing else — whitespace the
# grammar admits, so nothing about the REQUEST was wrong. 6 of those 10 were
# followed by a parseable tick at the very next scheduled turn (~15 min later),
# which is what a re-sample buys: one further draw of the same prefix, for free
# of the 8192-token budget advice that cannot help. These nodes drive
# `loop._maybe_finalize` over the fake engine, so what is counted is requests on
# the wire — the same unit the ledger counts.

BOUNDED_OK = ('{"result": "written", "note": "n", "duplicate_of": "", '
              '"facts": "f", "sources": "s"}')


def _diverged(draws: int = 2):
    """`draws` copies of the captured divergence: junk after `"facts": ` at the
    8192-token cap, the shape of `run_deep-research_20260928_032533_df61c0`."""
    return [_ok(DEGENERATE, usage={"completion_tokens": 8192,
                                   "prompt_tokens": 5_000,
                                   "total_tokens": 13_192},
                finish_reason="length") for _ in range(draws)]


async def _via_loop(client, responses, *, schema=BOUNDED, **opt_kw):
    """One turn's `_close_turn` leg, over the real finalizer and the fake engine."""
    from app.harness import loop as L
    from app.harness.options import RunOptions

    client.posted = []
    client.responses = list(responses)
    total: dict[str, int] = {}
    parsed, error = await L._maybe_finalize(
        options=RunOptions(model="primary", final_schema=schema, **opt_kw),
        stop_reason="stop", chat_messages=MESSAGES, tools=TOOLS, total_usage=total)
    return parsed, error, total


async def test_a_divergence_under_a_capped_schema_is_drawn_once_more(_client):
    """Clause 1. A divergence is a bad draw, not a bad request, so the loop
    draws again before it reports — over the identical message prefix and the
    identical `tools` array, because a tools array that differs re-prefills the
    whole conversation (the load-bearing assertion at the top of this file).
    And a second failure is still a failure: the caller gets the SECOND error,
    never a silent pass and never an empty verdict."""
    parsed, error, _total = await _via_loop(_client, _diverged(2))

    assert len(_client.posted) == 2, (
        f"expected exactly one re-sample, got {len(_client.posted)} requests")
    first, second = _client.posted
    assert second["messages"] == first["messages"], (
        "the re-sample changed the prefix, which re-prefills the turn")
    assert second["tools"] == first["tools"] == TOOLS, (
        "a re-sample with a different tools array is the most expensive thing "
        "this feature could do")
    assert second["tool_choice"] == first["tool_choice"] == "none"
    assert second["response_format"] == first["response_format"]

    assert parsed is None, "a second divergence must not read as a verdict"
    assert "generation diverged" in error and "not a budget" in error, error
    assert error.startswith("[attempts=2] "), (
        "the published failure has to distinguish a tick that was re-drawn "
        "from one that was not, and the marker is the only place it can say so: " + error)


async def test_the_re_sample_s_verdict_is_the_one_the_caller_gets(_client):
    """Clause 1's recovery half, and the whole point of the change: the second
    draw parses, and the turn reports success — the 6-in-10 case in the
    ledger, in which the next attempt was fine."""
    parsed, error, _total = await _via_loop(
        _client, [_diverged()[0], _ok(BOUNDED_OK)])
    assert len(_client.posted) == 2
    assert error == "", error
    assert parsed == json.loads(BOUNDED_OK)


async def test_the_re_sample_skips_the_budget_branch_and_an_uncapped_schema(_client):
    """Clause 2. Two exclusions, counted in requests.

    (a) an uncapped schema whose completion ran out of room: the budget IS the
    answer there, and a re-draw spends another 8192 tokens buying the same
    truncation. (b) the empty-content case under a BOUNDED grammar — thinking
    ate the budget (#1431), which is the budget whatever the grammar says, so
    boundedness alone must not license the retry. (c) prose instead of JSON
    under a bounded grammar: not a divergence either."""
    parsed, error, _ = await _via_loop(
        _client, [_ok('{"result": "written", "note": "a very long note that ra',
                      usage={"completion_tokens": 8192}, finish_reason="length")],
        schema=UNBOUNDED)
    assert len(_client.posted) == 1, "a budget truncation is not re-drawn"
    assert "raise harness.finalizer.max_tokens" in error, error
    assert "attempts" not in error

    parsed, error, _ = await _via_loop(
        _client, [_ok(None, usage={"completion_tokens": 200,
                                   "completion_tokens_details":
                                       {"reasoning_tokens": 200}},
                      finish_reason="length")], schema=BOUNDED)
    assert len(_client.posted) == 1, (
        "content that never left thinking is the budget under a capped schema "
        "too — see test_a_completion_that_never_left_thinking_is_still_the_budget")
    assert "raise harness.finalizer.max_tokens" in error and "attempts" not in error

    parsed, error, _ = await _via_loop(_client, [_ok("I think it is written.")],
                                       schema=BOUNDED)
    assert len(_client.posted) == 1, "prose is a model-behaviour failure, not a draw"
    assert "not JSON" in error and "attempts" not in error

    # Positive control, so the three counts above cannot mean "nothing in this
    # file ever re-samples": one bounded divergence, same fake, two requests.
    await _via_loop(_client, _diverged(2))
    assert len(_client.posted) == 2, "the counting fake has stopped counting"



def test_the_resample_predicate_reads_the_message_and_the_grammar():
    """Clause 2's branch, directly: the two phrases decide it, and where an
    error somehow carries both the budget wins — re-drawing a request that was
    legitimately out of room is the one thing this must never do."""
    assert F.should_resample_divergence(
        "finalizer failed: generation diverged at 8192 tokens — malformed "
        "output, not a budget ('…')", BOUNDED) is True
    assert F.should_resample_divergence(
        "finalizer failed: generation diverged at 8192 tokens — malformed "
        "output, not a budget ('…')", UNBOUNDED) is False
    assert F.should_resample_divergence(
        "finalizer failed: output truncated at 8192 tokens — raise "
        "harness.finalizer.max_tokens ('…')", BOUNDED) is False
    assert F.should_resample_divergence(
        "not a budget … and raise harness.finalizer.max_tokens", BOUNDED) is False
    assert F.should_resample_divergence("", BOUNDED) is False
    assert F.should_resample_divergence(
        "finalizer failed: HTTP 500 boom", BOUNDED) is False


async def test_two_diverged_draws_is_the_most_one_turn_spends(_client):
    """Clause 3. Five junk answers queued, two requests made: the retry is a
    rung, not a loop. The usage figures are the second half of the pin — both
    draws' 8192 tokens are folded, and a third draw would have made it 24576."""
    parsed, error, total = await _via_loop(_client, _diverged(5))
    assert len(_client.posted) == 2, "the retry became a loop"
    assert parsed is None and error.startswith("[attempts=2] "), error
    assert total["finalizer_output_tokens"] == 16_384, total
    assert total["finalizer_input_tokens"] == 10_000, total


async def test_the_re_sample_asks_for_the_max_tokens_it_was_given(_client):
    """Clause 4. The retry re-sends the budget it was handed, unchanged: 8192 on
    both requests, the shipped `harness.finalizer.max_tokens`. A re-sample that
    quietly asked for more would reintroduce exactly the advice #1706 removed —
    and would make the recovery depend on a cap this item forbids touching."""
    parsed, error, _total = await _via_loop(_client, _diverged(2),
                                            finalizer_max_tokens=8192)
    assert len(_client.posted) == 2
    assert [b["max_tokens"] for b in _client.posted] == [8192, 8192], (
        "the re-sample changed the budget: "
        + repr([b["max_tokens"] for b in _client.posted]))
    assert "raise harness.finalizer.max_tokens" not in error, error

