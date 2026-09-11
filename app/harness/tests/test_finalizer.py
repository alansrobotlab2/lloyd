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
    when it does not, an unclosed `{` is the tell."""
    _Client.responses = [_ok('{"verdict":"confirmed","check":"ls _pipe',
                             usage={"completion_tokens": 1024},
                             finish_reason="length")]
    parsed, error, usage = await _run()
    assert parsed is None
    assert "truncated at 1024 tokens" in error and "max_tokens" in error
    assert usage["output_tokens"] == 1024

    _Client.responses = [_ok('{"verdict":"confirmed","evidence":"…',
                             finish_reason="")]
    parsed, error, _ = await _run()
    assert parsed is None and "truncated" in error

    _Client.responses = [_ok("I think the verdict is confirmed.")]
    parsed, error, _ = await _run()
    assert parsed is None and "not JSON" in error and "truncated" not in error


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
