"""The tool-choice eval's cost capture and its safety citation — backlog #875
(clauses 5, 6, 7, 8, 9), members #748 and #818.

Two things are pinned here that a reader cannot see by eyeballing the runner.

The first is a comment. `eval/tool_choice_queries.yaml` explains why the set is
safe to run — the harness yields the first `tool_call` before dispatching it —
and until #748 pointed it at `loop.py:378-388`, which was the context-overflow
recovery block. A file that documents why an eval never executes a tool was
citing code that has nothing to do with tool calls, so the next reader to check
the claim would have found nothing to check. The test reads the range out of the
comment and requires the named symbols to be IN that range, so the comment
cannot rot again without failing.

The second is that the cost figures are real. `injected_tokens` must come from
the same estimator the prompt-budget path uses (#875 clause 7 — the duplicate
`CHARS_PER_TOKEN = 4.0` sitting in `eval/run_skill_dispatch_probe.py` is what
let the two drift), and `usage` must come from the engine's own events for that
query rather than from `usage.db`, which has no eval rows because the eval has no
session id (#818). A number in a cost column that was never measured is worse
than an empty column, because it closes the question.
"""
import asyncio
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prompt_builder  # noqa: E402

EVAL = ROOT / "eval"
QUERIES = EVAL / "tool_choice_queries.yaml"


def _load_runner():
    for name in ("run_tool_choice_eval", "compare_tool_choice"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("run_tool_choice_eval",
                                                  EVAL / "run_tool_choice_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_tool_choice_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── clause 5: the safety citation names code that contains the yield ─────────

def _cited_loop_ranges() -> list[tuple[int, int]]:
    """Every `app/harness/loop.py:A-B` range the query set's header cites."""
    header = "\n".join(QUERIES.read_text(encoding="utf-8").splitlines()[:40])
    return [(int(a), int(b)) for a, b in
            re.findall(r"app/harness/loop\.py[:\s]*(\d+)-(\d+)", header)]


def test_the_query_set_cites_a_loop_range():
    assert _cited_loop_ranges(), (
        "the header must name a loop.py line range for its yield-before-dispatch "
        "claim; #748 exists because it named one that had moved")


def test_the_cited_range_contains_the_yield_that_precedes_dispatch():
    """sed over the cited range prints `events.tool_call` — clause 5 verbatim."""
    lines = (ROOT / "app" / "harness" / "loop.py").read_text().splitlines()
    for start, end in _cited_loop_ranges():
        cited = "\n".join(lines[start - 1:end])
        assert "events.tool_call" in cited, (
            f"loop.py:{start}-{end} is cited as the yield-before-dispatch site but "
            f"contains no `events.tool_call` — it starts with: "
            f"{lines[start - 1].strip()[:70]!r}")
        assert "_execute_tool_call(" in cited, (
            f"loop.py:{start}-{end} must contain the dispatch the yield precedes, "
            "or 'yields before dispatch' is not what the range shows")


def test_the_yield_really_does_precede_the_dispatch_in_that_range():
    """Not merely present: yielded first, dispatched after, in that order."""
    lines = (ROOT / "app" / "harness" / "loop.py").read_text().splitlines()
    for start, end in _cited_loop_ranges():
        cited = lines[start - 1:end]
        yield_at = _first_line_containing(cited, "yield tc_evt",
                                          f"loop.py:{start}-{end}")
        dispatch_at = _first_line_containing(cited, "_execute_tool_call(",
                                             f"loop.py:{start}-{end}")
        assert yield_at < dispatch_at, (
            f"in the cited range the dispatch at +{dispatch_at} comes before the "
            f"yield at +{yield_at}: the eval would execute tools")


def _first_line_containing(lines: list[str], needle: str, where: str) -> int:
    """Index of `needle`, or a failure that names the range — not a bare
    StopIteration from `next()`, which tells a reader nothing about why."""
    for i, line in enumerate(lines):
        if needle in line:
            return i
    pytest.fail(f"{where} contains no {needle!r}; the cited lines start with "
                f"{lines[0].strip()[:60]!r}")
    raise AssertionError  # unreachable; keeps the return type honest


# ── clause 5, behavioural half: the CLAIM, not the numbers ──────────────────
#
# The cited range is a pointer and pointers rot. This file's own history is the
# evidence: it cited 378-388 (the context-overflow recovery block, the defect
# #748 fixed), then after #748 it cited 737-748 — correct when written — and a
# later loop.py edit slid those nine lines down without touching the mechanism.
# So the range test above pins the comment to today's tree; the tests below pin
# the mechanism on BOTH tool-call emit paths (the single-call loop and Phase 1 of
# the parallel read-only batch) by driving the real `run_query` on the replay
# seams (`app/harness/tests/_replay.py`) and breaking at the first `tool_call`,
# exactly as the eval does. They used to scan `run_query`'s source for the order
# of `yield tc_evt` and the old `_dispatch_one_tool_call(` (P13.0 replaced that): a
# source scan certifies text, and the eval's safety is a behaviour.

_EMIT_PATHS = {
    # (tools in the first completion, parallel flag)
    "single_call": ([("c1", "Bash")], False),
    "sequential_batch": ([("c1", "Read"), ("c2", "Bash")], True),
    "parallel_batch": ([("c1", "Read"), ("c2", "Grep")], True),
}


def _emit_turn(monkeypatch, path, *, stop_at_first_call):
    from app.harness.options import RunOptions
    from app.harness.tests import _replay as R

    calls, parallel = _EMIT_PATHS[path]
    pool = R.ReplayPool()
    R.install(monkeypatch, R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call(cid, name, "s") for cid, name in calls]),
        R.Step(text="done"),
    ]), pool)
    opts = RunOptions(model="m", max_turns=4, tool_search_enabled=False,
                      parallel_tool_calls_enabled=parallel)
    stop = (lambda e: e["type"] == "tool_call") if stop_at_first_call else None
    out = asyncio.run(R.drive(opts, on_event=stop))
    return out, pool


@pytest.mark.parametrize("path", sorted(_EMIT_PATHS))
def test_every_tool_call_emit_site_yields_before_its_dispatch(monkeypatch, path):
    """No path may hand out a `tool_call` and then have already run it.

    The eval breaks at the first `tool_call` and closes the generator; on every
    dispatch path the pool must not have been asked for anything by then, and
    closing there must not dispatch it afterwards either.
    """
    out, pool = _emit_turn(monkeypatch, path, stop_at_first_call=True)
    assert out[-1]["type"] == "tool_call", [e["type"] for e in out]
    assert pool.started == [], (
        f"{path}: the pool was called ({pool.started}) before the first tool_call "
        "was handed out — the tool-choice eval would execute the tool")


@pytest.mark.parametrize("path", sorted(_EMIT_PATHS))
def test_a_dispatch_follows_each_yield_somewhere_so_the_break_is_what_stops_it(
        monkeypatch, path):
    """The other direction: if no dispatch ever followed the yield, the comment
    would be describing a generator that returns before running tools, and the
    eval's safety would be an accident of the caller rather than of the loop."""
    out, pool = _emit_turn(monkeypatch, path, stop_at_first_call=False)
    announced = [e["call_id"] for e in out if e["type"] == "tool_call"]
    assert announced and sorted(pool.started) == sorted(announced), (
        f"{path}: announced {announced}, dispatched {pool.started}")


# ── clause 6: per-query usage, from this query's own request ────────────────

FAKE_USAGE = {"input_tokens": 41200, "output_tokens": 118,
              "cache_read": 38000, "cache_create": 2000, "reasoning_tokens": 40}


def _run_first_tool_call(events, monkeypatch):
    import app.harness as harness

    async def fake(messages, options):
        for e in events:
            yield e

    monkeypatch.setattr(harness, "run_query", fake)
    mod = _load_runner()

    class Opts:
        pass

    return asyncio.run(mod._first_tool_call("hello", Opts(), timeout=5))


def test_first_tool_call_returns_its_own_usage(monkeypatch):
    """The usage dict on the record is the one this query's stream carried."""
    # Real stream order: `loop.py:524` yields `assistant_message` (carrying
    # `usage`) and only `loop.py:770` yields the `tool_call`, so the usage of the
    # turn that produced the tool call is already in hand when we stop. (These two
    # numbers moved with #800's relief latch, like the header range in
    # `tool_choice_queries.yaml` did; the claim they name did not.)
    out = _run_first_tool_call([
        {"type": "assistant_message", "usage": dict(FAKE_USAGE)},
        {"type": "tool_call", "name": "http_search", "args_dict": {"query": "q"}},
    ], monkeypatch)
    assert out["first_tool"] == {"name": "http_search", "args": {"query": "q"}}
    assert out["usage"] == FAKE_USAGE
    assert out["uncached_prompt_tokens"] == 41200 - 38000
    assert out["error"] is None


def test_the_record_is_written_before_dispatch_and_the_generator_is_closed():
    """No tool runs: we break at `tool_call`, so the stream must end there.

    This is the mechanism `tool_choice_queries.yaml` documents. If a refactor
    moved the yield after the dispatch, this fake would raise, because the event
    after `tool_call` in a real stream is a `tool_result` the runner would then
    be waiting on — and the eval would be fetching URLs.
    """
    seen = []

    async def stream(messages, options):
        yield {"type": "tool_call", "name": "http_fetch", "args_dict": {"url": "x"}}
        seen.append("PAST THE YIELD")
        yield {"type": "tool_result", "call_id": "1", "content": "should never arrive"}

    import app.harness as harness
    orig = harness.run_query
    harness.run_query = stream
    try:
        mod = _load_runner()

        class Opts:
            pass
        out = asyncio.run(mod._first_tool_call("p", Opts(), timeout=5))
    finally:
        harness.run_query = orig
    assert out["first_tool"]["name"] == "http_fetch"
    assert seen == [], "the runner consumed events past the first tool_call"


def test_missing_usage_records_as_none_not_zero(monkeypatch):
    """A stream that reported nothing yields no number, not a free-looking 0."""
    out = _run_first_tool_call([
        {"type": "tool_call", "name": "http_search", "args_dict": {}},
        {"type": "assistant_message"},
    ], monkeypatch)
    assert out["usage"] == {}
    assert out["uncached_prompt_tokens"] is None


def test_uncached_arithmetic_and_its_missing_key_rule():
    mod = _load_runner()
    assert mod._uncached_prompt_tokens({"input_tokens": 100, "cache_read": 90}) == 10
    # No prompt size at all -> None (absent from the average), never 0.
    assert mod._uncached_prompt_tokens({"output_tokens": 5}) is None
    assert mod._uncached_prompt_tokens({}) is None
    # cache_read alone cannot tell us anything: with no input figure the
    # difference would read as a negative prefill.
    assert mod._uncached_prompt_tokens({"cache_read": 90}) is None


# ── clause 6, across the normalization seam ─────────────────────────────────
#
# The tests above hand `_first_tool_call` a usage dict that ALREADY has the keys
# it reads (`input_tokens`, `cache_read`). That is the shape the harness produces,
# not the shape the engine sends, and the eval reads across that boundary: vLLM
# answers with `prompt_tokens`/`completion_tokens` and puts the prefix-cache hit
# in a NESTED `prompt_tokens_details.cached_tokens`, which `loop._merge_usage`
# (app/harness/loop.py) renames to `input_tokens`/`output_tokens`/`cache_read`.
# A stub cannot see that rename going wrong, and it went wrong once already: the
# nested dict is skipped by `_merge_usage`'s int-only loop, so `cache_read` was
# absent from every usage block the harness ever produced and every session on
# 2026-09-08 reported `cache_read: 0` (tests/test_caption_ratchet_and_cache.py).
# Had the eval existed then, its `uncached_prompt_tokens` would have equalled the
# FULL prompt every time — a number that looks measured and is not.
#
# So: drive the REAL loop over a scripted vLLM-shaped stream and let the eval's
# own `_first_tool_call` read the result. Nothing here asserts about `_merge_usage`
# in isolation; it asserts what the eval records when the engine's bytes come in
# from one end.

def _vllm_stream(monkeypatch, *, prompt_tokens, cached_tokens, completion_tokens):
    """Patch the loop's transport so one turn streams `script`, and hand back the
    eval module. `choices=[]` on the usage chunk is what vLLM does when
    `stream_options.include_usage` is on."""
    from app.harness import loop as loop_mod

    async def fake_stream_chat(**_kwargs):
        yield {"choices": [{"delta": {"content": "looking it up"},
                            "finish_reason": None}]}
        yield {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "http_search",
                         "arguments": '{"query":"qwen release"}'}}]},
            "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        yield {"choices": [], "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens}}}

    class _FakePool:
        discovered = [("lloyd-mcp", [{"name": "http_search",
                                      "description": "search the web",
                                      "inputSchema": {"type": "object",
                                                      "properties": {}}}])]

    async def fake_build_pool(_options):
        return _FakePool()

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(loop_mod, "_build_pool", fake_build_pool)
    return _load_runner()


def test_the_usage_the_engine_reports_survives_the_harness_rename(monkeypatch):
    """Raw vLLM keys in, eval's recorded cost out — through the real loop."""
    from app.harness.options import RunOptions

    mod = _vllm_stream(monkeypatch, prompt_tokens=41200, cached_tokens=38000,
                       completion_tokens=118)
    out = asyncio.run(mod._first_tool_call(
        "search the web for the latest Qwen release",
        RunOptions(model="primary", tool_call_summaries=False), timeout=10))

    assert out["error"] is None, out["error"]
    assert out["first_tool"] == {"name": "http_search",
                                 "args": {"query": "qwen release"}}
    # The rename happened: the eval reads Anthropic-style keys, the engine sent
    # OpenAI-style ones.
    assert out["usage"]["input_tokens"] == 41200
    assert out["usage"]["output_tokens"] == 118
    # The nested field arrived. This is the assertion the stub could not make.
    assert out["usage"]["cache_read"] == 38000, (
        "`cache_read` is nested under `prompt_tokens_details.cached_tokens`; if "
        "the harness stops reading it, uncached_prompt_tokens silently becomes the "
        "whole prompt and every cost figure in the artifact is wrong by ~90%")
    assert out["uncached_prompt_tokens"] == 41200 - 38000


def test_a_cold_prefix_records_the_whole_prompt_not_zero(monkeypatch):
    """cached_tokens=0 is a real answer (cold prefix), and it must not be
    confused with the absent field, which is `None`. Both arrive through the
    same seam and they mean opposite things."""
    from app.harness.options import RunOptions

    mod = _vllm_stream(monkeypatch, prompt_tokens=41200, cached_tokens=0,
                       completion_tokens=118)
    out = asyncio.run(mod._first_tool_call(
        "p", RunOptions(model="primary", tool_call_summaries=False), timeout=10))
    assert out["usage"]["cache_read"] == 0
    assert out["uncached_prompt_tokens"] == 41200


def test_an_engine_that_reports_nothing_costs_the_eval_no_number(monkeypatch):
    """A stream with no usage chunk — the loopback engine mid-restart, or a
    build without `include_usage` — must leave the column empty. A 0 here would
    read as 'this query cost nothing to prefill', which is the false reassurance
    the whole null-vs-zero rule exists to prevent."""
    from app.harness import loop as loop_mod
    from app.harness.options import RunOptions

    async def fake_stream_chat(**_kwargs):
        yield {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "c1", "type": "function",
            "function": {"name": "http_search", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}]}

    class _FakePool:
        discovered = [("lloyd-mcp", [{"name": "http_search", "description": "s",
                                      "inputSchema": {"type": "object"}}])]

    async def fake_build_pool(_options):
        return _FakePool()

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(loop_mod, "_build_pool", fake_build_pool)
    mod = _load_runner()
    out = asyncio.run(mod._first_tool_call(
        "p", RunOptions(model="primary", tool_call_summaries=False), timeout=10))
    assert out["first_tool"]["name"] == "http_search"
    assert out["uncached_prompt_tokens"] is None


# ── clause 7: one estimator, shared with the prompt-budget path ─────────────

def test_injected_tokens_uses_the_shared_estimator_not_a_local_one():
    """The eval must not own its own chars/token arithmetic (#875 clause 7).

    `eval/run_skill_dispatch_probe.py` carries a second `CHARS_PER_TOKEN = 4`; the
    reason this clause exists is that two copies of a constant drift, and here
    they would drift into a cost column that no longer matched the
    `PROMPT_BUDGET` line printed for the same prompt.
    """
    mod = _load_runner()
    prompt = "x" * 100
    prefetched = prompt + ("<context>" + "y" * 4000 + "</context>")
    cost = mod._cost(prompt, prefetched, {"usage": {}, "uncached_prompt_tokens": None})
    assert cost["injected_chars"] == len(prefetched) - len(prompt)
    assert cost["injected_tokens"] == prompt_builder.prompt_token_estimate(
        cost["injected_chars"])


def test_injected_tokens_is_the_context_block_and_not_the_whole_prompt():
    mod = _load_runner()
    prompt, prefetched = "hello", "hello<context>a bunch of injected vault text</context>"
    cost = mod._cost(prompt, prefetched, {})
    whole = prompt_builder.prompt_token_estimate(len(prefetched))
    assert cost["prompt_tokens_est"] == whole          # the total, labelled as such
    assert cost["injected_tokens"] < whole             # the injection, smaller
    assert cost["injected_tokens"] == prompt_builder.prompt_token_estimate(
        len(prefetched) - len(prompt))


def test_the_runner_does_not_define_its_own_chars_per_token():
    """A second copy is the drift this clause forbids — grep it, don't trust it."""
    src = (EVAL / "run_tool_choice_eval.py").read_text(encoding="utf-8")
    assert not re.search(r"^CHARS_PER_TOKEN\s*=", src, re.M), (
        "the eval re-declared the chars/token constant; import "
        "prompt_builder.prompt_token_estimate instead (#875 clause 7)")
    assert "from prompt_builder import prompt_token_estimate" in src


def test_estimator_and_budget_share_the_same_number():
    """The eval's figure and the PROMPT_BUDGET line cannot drift: one function."""
    assert prompt_builder.prompt_token_estimate(8000) == math.ceil(8000 / 4)
    mod = _load_runner()
    assert mod.prompt_token_estimate is prompt_builder.prompt_token_estimate


# ── clauses 8 and 9: one artifact answers correct AND cost, or says why not ──

def _load_prefetch_runner():
    """Import `eval/run_prefetch_eval.py` as a module, so its `run`, `summarize`
    and `main` can be EXECUTED rather than described. It is a script, not an
    importable package, so it goes in by path like the tool-choice runner."""
    sys.modules.pop("run_prefetch_eval", None)
    spec = importlib.util.spec_from_file_location(
        "run_prefetch_eval", EVAL / "run_prefetch_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_prefetch_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_search_legs(monkeypatch, hits):
    """Stub the three search legs `run()` calls. The legs hit qmd and the vector
    index; the fields the eval reads off their return value are `file`, so one
    canned hit is enough to make the script run end to end without a vault."""
    import prefetch

    monkeypatch.setattr(prefetch, "_search_vault_lex",
                        lambda q, focus, deadline=None: list(hits))
    monkeypatch.setattr(prefetch, "_search_vault_hybrid_and_stash",
                        lambda q, focus: list(hits))
    monkeypatch.setattr(prefetch, "_merge_vault_results",
                        lambda lex, hybrid: list(lex))


def _run_prefetch_end_to_end(tmp_path, monkeypatch):
    """Run the prefetch eval's own `main()` over one query and hand back the
    artifact it wrote, read off disk. Everything below this line is the script's
    real code path: record building, summarize, the header, the JSON dump."""
    mod = _load_prefetch_runner()
    _fake_search_legs(monkeypatch, [{"file": "qmd://knowledge/example-note.md"}])
    queries = tmp_path / "queries.yaml"
    queries.write_text(
        "queries:\n"
        "  - id: q-example\n"
        "    query: the example note\n"
        "    category: knowledge\n"
        "    expect_docs: [knowledge/example-note.md]\n",
        encoding="utf-8")
    (tmp_path / "baselines").mkdir()
    monkeypatch.setattr("app.paths.EVAL_BASELINES_DIR", tmp_path / "baselines")  # where main() writes
    monkeypatch.setattr(sys, "argv", [
        "run_prefetch_eval.py", "--queries", str(queries),
        "--label", "pinned", "--skip-hybrid"])
    assert mod.main() == 0
    written = sorted((tmp_path / "baselines").glob("*.json"))
    assert len(written) == 1, f"expected one artifact, got {[p.name for p in written]}"
    return json.loads(written[0].read_text(encoding="utf-8"))


def test_run_prefetch_eval_states_why_it_reports_no_tokens(tmp_path, monkeypatch):
    """Clause 8: `run_prefetch_eval.py` builds no `<context>` block and spends no
    model turn, so it has nothing to count. Clause 8 accepts that — but only if
    the artifact SAYS so. Run the script and read the file it wrote: the header
    carries a `cost_note` that names both the reason and where the real per-query
    token cost lives. An absent column reads as 'free' to whoever joins it."""
    art = _run_prefetch_end_to_end(tmp_path, monkeypatch)
    note = art["cost_note"]
    assert note, "the artifact header must state why it reports no token cost"
    assert "no token cost" in note.lower()
    assert "run_tool_choice_eval" in note or "tool-choice" in note, (
        "and name the artifact that DOES carry per-query tokens, so a cost arm "
        "knows which file to join instead of assuming this one is cheap")


def test_the_prefetch_artifact_carries_an_explicit_null_cost_per_query(tmp_path,
                                                                      monkeypatch):
    """Read it back: every record says `injected_tokens: null`, not missing. The
    field name is the tool-choice eval's, deliberately — a join across the two
    files must see 'no figure recorded' (None), never a KeyError and never a 0
    that looks like a retrieval that injected nothing."""
    art = _run_prefetch_end_to_end(tmp_path, monkeypatch)
    records = art["records"]
    assert len(records) == 1
    rec = records[0]
    assert "injected_tokens" in rec, "the key must be present, not omitted"
    assert rec["injected_tokens"] is None, "and null, not 0"
    assert art["summary"]["cost_note"] == art["cost_note"], (
        "the summary and the header carry one note, so a reader cannot get the "
        "explanation from one copy and the number from the other")


def test_the_prefetch_header_line_names_the_field_it_leaves_null():
    """The one header line clause 8 allows must name the field it is nulling; a
    reader of the file alone has to learn the same thing as a reader of the JSON.
    Read from the imported module's docstring, so a comment above the imports or
    a stray string cannot satisfy it."""
    doc = _load_prefetch_runner().__doc__ or ""
    assert "injected_tokens" in doc, (
        "the header must name the field it reports none for")
    assert "_search_vault_lex" in doc, (
        "and the reason: it calls the search legs directly and renders no block")


def test_tool_choice_record_answers_both_questions_in_one_file():
    """#562's cost arm is a join on `records[]`, not a second probe (#875 clause 9)."""
    mod = _load_runner()
    prompt = "search the web for the latest Qwen model release"
    prefetched = prompt + "<context>" + "vault text " * 500 + "</context>"
    outcome = {"usage": dict(FAKE_USAGE), "uncached_prompt_tokens": 3200}
    cost = mod._cost(prompt, prefetched, outcome)
    scoring = mod._score({"category": "public-search", "id": "x",
                          "expect_tools": ["http_search"]},
                         {"name": "http_search", "args": {}})
    assert scoring["correct"] is True
    assert cost["injected_tokens"] > 0
    assert cost["usage"]["cache_read"] == 38000
    assert cost["uncached_prompt_tokens"] == 3200
    # And the summary reports it, averaged over the queries that HAVE a measured
    # figure: a zero would be counted as a perfect cache hit.
    rec = {"scoring": scoring, "cost": cost, "latency_ms": 1.0, "error": None}
    s = mod.summarize([rec])["overall"]
    assert s["injected_tokens_avg"] == cost["injected_tokens"]
    assert s["uncached_prompt_tokens_avg"] == 3200
    assert s["queries_with_usage"] == 1


def test_the_end_to_end_run_writes_the_join_into_its_own_artifact(tmp_path, monkeypatch,
                                                                  capsys):
    """Clause 9, executed: the file `#562`'s cost arm joins on must be produced by
    the code path that produces it.

    Everything else in this file asserts about a `cost` block handed to
    `summarize`/`print_table` as a literal. That leaves the one line that matters
    unpinned: `"cost": _cost(...)` inside `run_eval`'s loop. Delete it and every
    hand-built-record test stays green while the artifact a cost arm reads has no
    cost half -- the same silent loss, in the opposite direction, that clause 6
    exists to prevent.

    So this drives the real `run_eval` over two real query specs with only the
    engine and the prefetcher replaced, writes the artifact through the same
    lines `main()` does, re-reads the file from disk, and asks it both questions
    #562 asks of one query: was it right, and what did injection cost. The
    prompt and the real system prompt come from the real code path; only the
    model turn and the search are stood in for, because neither may be reached
    from a unit test.
    """
    mod = _load_runner()

    specs = [
        {"id": "pub-search-1", "category": "public-search",
         "prompt": "What is the latest release of vllm?",
         "expect_tools": ["http_search"]},
        {"id": "local-health", "category": "localhost",
         "prompt": "Check whether the Lloyd backend on localhost:8080 is healthy.",
         "expect_tools": ["Bash"]},
    ]

    # The engine answer per query, keyed by the prompt the runner was handed, so
    # the test also proves the query text reaches the model intact.
    answers = {specs[0]["prompt"]: "http_search", specs[1]["prompt"]: "Bash"}

    class Opts:
        model = "primary"
        extra_body = None

    # `prefetch_context` is the retrieval stack behind a tokenizer; the harness
    # seam above covers the engine. Only the search is stood in for, and it
    # returns a real <context> block prefixed to the prompt -- which is the
    # shape the real prefetcher returns -- so the injected-skill and
    # injected-token halves are computed by the runner, not asserted.
    context_block = ('<context>\n<skill name="web-search" score="9.9">b</skill>'
                     '\n</context>\n')
    monkeypatch.setattr("prefetch.prefetch_context",
                        lambda prompt, **kw: context_block + prompt)

    async def fake_first_tool_call(prefetched, options, timeout):
        # `run_eval` hands the PREFETCHED text to the harness, so the query is
        # found inside it. The engine is answered on the text it would receive.
        spec = next(s for s in specs if s["prompt"] in prefetched)
        name = answers[spec["prompt"]]
        args = ({"command": "curl -s http://localhost:8080/health"}
                if name == "Bash" else {"query": "vllm release"})
        assert options.max_turns == 1, (
            "one turn per query is what makes the set cheap enough to re-run; a "
            "multi-turn eval measures the harness, not tool choice")
        return {
            "first_tool": {"name": name, "args": args},
            "text": "", "error": None,
            "usage": {"input_tokens": 41200, "output_tokens": 20,
                      "cache_read": 38000},
            "uncached_prompt_tokens": 3200,
        }

    monkeypatch.setattr(mod, "_first_tool_call", fake_first_tool_call)

    records, cfg = asyncio.run(mod.run_eval(specs, timeout=5.0, model="primary"))

    out = {
        "label": "join-test", "notes": "", "ran_at": "2026-09-19T00:00:00+00:00",
        "n_queries": len(records), "config": cfg,
        "summary": mod.summarize(records), "records": records,
    }
    path = tmp_path / "join.json"
    path.write_text(json.dumps(out, indent=2, default=str))

    art = json.loads(path.read_text())
    assert art["n_queries"] == 2
    for rec in art["records"]:
        # question 1: was it correct
        assert rec["scoring"]["correct"] is True, (
            f"{rec['id']}: the real run_eval must score the query it ran")
        # question 2: how many tokens did it inject
        cost = rec["cost"]
        assert cost["injected_chars"] > 0, "the <context> block is what got injected"
        assert cost["injected_tokens"] == prompt_builder.prompt_token_estimate(
            cost["injected_chars"]), (
            "the token figure in the artifact must be the shared estimator's over "
            "the injected chars -- not a local chars/4, and not the whole prompt")
        # `context_chars` is the injected block (`len(prefetched) - len(prompt)`),
        # and `prompt_tokens_est` is the whole turn — so the two must differ by
        # exactly the query's own length. Pinned, because a cost arm that reads
        # `prompt_tokens_est` as "injection cost" would overstate it.
        assert rec["context_chars"] == cost["injected_chars"]
        assert cost["prompt_tokens_est"] == prompt_builder.prompt_token_estimate(
            rec["context_chars"] + len(rec["prompt"])), (
            "the whole-turn estimate must be the injected block plus the query")
        assert cost["uncached_prompt_tokens"] == 3200
        assert cost["usage"]["cache_read"] == 38000
    assert art["summary"]["overall"]["queries_with_usage"] == 2
    # The table prints both halves on the row a human reads.
    printed = capsys.readouterr().out
    assert "pub-search-1" in printed and "local-health" in printed


def test_deleting_the_cost_line_from_run_eval_breaks_the_artifact(tmp_path, monkeypatch):
    """The mutation this whole test exists to catch, run as an assertion.

    `run_eval` is the only place a record's `cost` is attached. This rewrites
    that one key out of the source in memory and shows the artifact it then
    writes has correctness but no cost -- which is exactly the state a green
    suite would have shipped. It is a mutation test in the literal sense: the
    file on disk is untouched, and `mod.__dict__` is discarded with the test.
    """
    mod = _load_runner()
    src = (EVAL / "run_tool_choice_eval.py").read_text(encoding="utf-8")
    needle = '"cost": _cost(prompt, prefetched, outcome),'
    assert needle in src, (
        "run_eval no longer attaches cost on this line; update this test AND "
        "check clause 9 still holds -- if cost moved, the join may have moved "
        "somewhere no test looks")

    spec_src = src.replace(needle, "")
    assert spec_src != src
    import types as _t
    broken = _t.ModuleType("run_tool_choice_eval_broken")
    broken.__dict__["__file__"] = str(EVAL / "run_tool_choice_eval.py")
    broken.__dict__["__name__"] = "run_tool_choice_eval_broken"
    exec(compile(spec_src, "run_eval_without_cost.py", "exec"), broken.__dict__)

    async def fake_first_tool_call(prefetched, options, timeout):
        return {"first_tool": {"name": "http_search", "args": {}},
                "text": "", "error": None,
                "usage": {"input_tokens": 100, "cache_read": 0},
                "uncached_prompt_tokens": 100}

    monkeypatch.setattr(broken, "_first_tool_call", fake_first_tool_call)
    monkeypatch.setattr("prefetch.prefetch_context",
                        lambda prompt, **kw: "<context>x</context>" + prompt)
    records, _ = asyncio.run(broken.run_eval(
        [{"id": "x", "category": "public-search", "prompt": "p",
          "expect_tools": ["http_search"]}], timeout=5.0, model="primary"))
    assert "cost" not in records[0], (
        "the mutation did not take — the line is now load-bearing somewhere "
        "else and the guard above is pointing at dead code")
    assert records[0]["scoring"]["correct"] is True, (
        "correctness survives, which is the point: a suite that only checks "
        "correctness cannot see the cost half go missing")
    mod = _load_runner()
    scoring = mod._score({"category": "public-search", "id": "x",
                          "expect_tools": ["http_search"]},
                         {"name": "http_search", "args": {}})
    with_u = {"scoring": scoring, "latency_ms": 1.0, "error": None,
              "cost": {"injected_tokens": 10, "uncached_prompt_tokens": 500}}
    without = {"scoring": scoring, "latency_ms": 1.0, "error": None,
               "cost": {"injected_tokens": 12, "uncached_prompt_tokens": None}}
    legacy = {"scoring": scoring, "latency_ms": 1.0, "error": None}   # pre-cost artifact
    s = mod.summarize([with_u, without, legacy])["overall"]
    assert s["uncached_prompt_tokens_avg"] == 500.0, (
        "the average is over the one query that measured something, not over three")
    assert s["queries_with_usage"] == 1
    assert s["injected_tokens_avg"] == pytest.approx(11.0)   # all three have the estimate
    assert s["n_queries"] == 3


def test_a_record_with_no_usage_is_absent_from_the_cost_average():
    mod = _load_runner()
    scoring = mod._score({"category": "public-search", "id": "x",
                          "expect_tools": ["http_search"]},
                         {"name": "http_search", "args": {}})
    with_u = {"scoring": scoring, "latency_ms": 1.0, "error": None,
              "cost": {"injected_tokens": 10, "uncached_prompt_tokens": 500}}
    without = {"scoring": scoring, "latency_ms": 1.0, "error": None,
               "cost": {"injected_tokens": 12, "uncached_prompt_tokens": None}}
    legacy = {"scoring": scoring, "latency_ms": 1.0, "error": None}   # pre-cost artifact
    s = mod.summarize([with_u, without, legacy])["overall"]
    assert s["uncached_prompt_tokens_avg"] == 500.0, (
        "the average is over the one query that measured something, not over three")
    assert s["queries_with_usage"] == 1
    assert s["injected_tokens_avg"] == pytest.approx(11.0)   # all three have the estimate
    assert s["n_queries"] == 3


def test_a_scoring_key_absent_on_some_queries_is_absent_not_a_zero():
    """Finding appended to #875 by round SM_20260912_035714: `summarize.rate()`
    divided by the number of RECORDS, not the number that carry the key.

    Every key is written today, so no number is wrong right now. The reason to
    pin it anyway is that a diluted rate errs silently toward 0.0 -- and 0.0 is a
    plausible-looking score, so the silent direction is a fabricated regression
    in the very metric the per-metric noise floor in `compare_tool_choice.py` is
    computed against. Denominators that can silently disagree with numerators are
    the defect class this whole item is about.
    """
    mod = _load_runner()
    scored = {"scoring": {"correct": True, "used_http_tool": True,
                          "is_web_category": True},
              "latency_ms": 1.0, "error": None}
    unscored = {"scoring": {"correct": True, "is_web_category": True},   # no `used_http_tool`
                "latency_ms": 1.0, "error": None}

    s = mod.summarize([scored, unscored])["overall"]
    assert s["correct_rate"] == 1.0, "both records carry `correct`"
    assert s["http_tool_first_rate"] == 1.0, (
        "1 of 1 queries SCORED, not 1 of 2: a missing key is 'not measured', and "
        "dividing by the record count would report 0.5, which reads as a real "
        "half-and-half measurement")
    assert s["n_queries"] == 2, "the n is still every query run"


def test_the_printed_table_shows_right_and_its_price_on_one_row():
    """The row is where a human reads the join; it must not be artifact-only."""
    import io
    import contextlib
    mod = _load_runner()
    scoring = mod._score({"category": "public-search", "id": "x",
                          "expect_tools": ["http_search"]},
                         {"name": "http_search", "args": {}})
    rec = {"id": "search-qwen", "category": "public-search",
           "injected_skills": [], "scoring": scoring, "latency_ms": 1.0, "error": None,
           "cost": {"injected_tokens": 812, "uncached_prompt_tokens": 3200}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.print_table([rec], mod.summarize([rec]))
    out = buf.getvalue()
    assert "search-qwen" in out and "812" in out and "3,200" in out
    assert "OK" in out
