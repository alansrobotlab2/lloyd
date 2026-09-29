"""#1632: the paired scratchpad A/B driver (`eval/run_scratchpad_ab.py`).

Offline, like the memory A/B tests: the arms are built by the real worker
options builder and checked against the real aggregator catalog (listed
in-process, no :8500), and trials run through a stub turn whose "model" calls
`Scratchpad` through `agent_mcp.main.call_tool` — the production dispatcher —
whenever the arm's own advertised catalog offers it. No engine, no pool, no live
data root: scratchpad files land under a `tmp_path` data root.
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import run_scratchpad_ab as ab  # noqa: E402

SOURCE = "session-distill"


@pytest.fixture(scope="module")
def catalog():
    return asyncio.run(ab.discover_catalog())


@pytest.fixture(scope="module")
def shared(catalog):
    return ab.read_only_deny(catalog["discovered"])


def _arms(shared, **kw):
    return {arm: ab.build_arm_options(arm, source=SOURCE, max_turns=15, shared_deny=shared,
                                      session_id="scratchpad-ab-test", **kw)
            for arm in ab.ARMS}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    import app.scratchpad as sp

    root = tmp_path / "data"
    monkeypatch.setattr(sp, "DATA_ROOT", root)
    return root


# ── clause 1: both arms built, differing only by the affordance ─────────────


def test_the_arms_differ_by_the_scratchpad_and_nothing_else(catalog, shared):
    arms = _arms(shared)
    on, off = arms["scratchpad"], arms["control"]
    sizes = ab.check_arms(on, off, catalog["discovered"])

    assert set(off.disallowed_tools) - set(on.disallowed_tools) == set(ab.SCRATCHPAD_DENY)
    assert set(on.disallowed_tools) <= set(off.disallowed_tools)
    assert sizes["tools"]["scratchpad"] == sizes["tools"]["control"] + 1
    # The control's system message is not the scratchpad's minus nothing: it names
    # the tool in its denied block and drops the tool's catalog line.
    sys_on = ab.assemble_position_zero(on, ab.advertised_catalog(on, catalog["discovered"]))
    sys_off = ab.assemble_position_zero(off, ab.advertised_catalog(off, catalog["discovered"]))
    assert sys_on != sys_off
    assert sizes["catalog_line"], "tool search is on for a worker turn, so the reminder lists it"


def test_the_source_is_named_and_runs_its_production_prompt():
    from workers.sources import session_distill

    build, turns = ab.SOURCES[SOURCE]()
    assert build is session_distill.distill_prompt
    assert turns == session_distill.DEFAULT_MAX_TURNS
    assert "/x/long.json" in build("/x/long.json")


def test_extra_prompt_text_in_one_arm_is_refused(catalog, shared):
    from workers.sources._common import build_denied_tools_block

    arms = _arms(shared)
    off = arms["control"]
    block = build_denied_tools_block(off.disallowed_tools)
    tampered = dataclasses.replace(
        off, system_prompt=off.system_prompt[:-len(block)] + "\nOne more rule.\n" + block)
    with pytest.raises(ab.ArmMismatch, match="outside the denied-tools block"):
        ab.check_arms(arms["scratchpad"], tampered, catalog["discovered"])


def test_a_second_denied_tool_is_refused(catalog, shared):
    arms = _arms(shared)
    off = arms["control"]
    extra = dataclasses.replace(off, disallowed_tools=list(off.disallowed_tools) + ["Grep"])
    with pytest.raises(ab.ArmMismatch, match="deny lists differ"):
        ab.check_arms(arms["scratchpad"], extra, catalog["discovered"])


def test_any_other_option_difference_is_refused(catalog, shared):
    arms = _arms(shared)
    other = dataclasses.replace(arms["control"], max_turns=arms["control"].max_turns + 1)
    with pytest.raises(ab.ArmMismatch, match="max_turns"):
        ab.check_arms(arms["scratchpad"], other, catalog["discovered"])


def test_a_scratchpad_arm_that_cannot_see_the_tool_is_refused(catalog, shared):
    """Scratchpad switched off from the Tools page leaves both arms without it."""
    on = _arms(shared + list(ab.SCRATCHPAD_DENY))["scratchpad"]
    off = dataclasses.replace(on, disallowed_tools=list(on.disallowed_tools))
    with pytest.raises(ab.ArmMismatch, match="does not advertise"):
        ab.check_arms(on, off, catalog["discovered"])


def test_the_production_turn_carries_the_arms_deny_list(monkeypatch):
    """`run_prompt_on_primary` hands `extra_disallowed` to the options builder."""
    import workers.sources._common as C

    seen = {}

    class Stop(Exception):
        pass

    def fake_build(max_turns, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(C, "_worker_run_options", fake_build)
    with pytest.raises(Stop):
        asyncio.run(C.run_prompt_on_primary(
            "p", max_turns=3, source=SOURCE,
            extra_disallowed=ab.arm_extra_disallowed("control", ["Write"])))
    assert list(seen["extra_disallowed"]) == ["Write", *ab.SCRATCHPAD_DENY]
    assert seen["session_id"], "the scratchpad anchor still gets the run's session id"


def test_both_arms_are_read_only_except_the_tool_under_test(catalog, shared):
    from agent_mcp.annotations import read_only_tool_names

    on = _arms(shared)["scratchpad"]
    advertised = {t["function"]["name"] for t in ab.advertised_catalog(on, catalog["discovered"])}
    assert advertised - read_only_tool_names() == {ab.SCRATCHPAD_TOOL}


# ── clause 2: the tool resolves through production's registration ───────────


def test_the_tool_resolves_through_the_aggregator_modules(catalog):
    import agent_mcp.builtin_scratchpad as B
    import agent_mcp.main as M

    assert catalog["module"] == ab.SCRATCHPAD_MODULE == B.__name__
    assert B in M.MODULES
    names = [t["name"] for _srv, tools in catalog["discovered"] for t in tools]
    assert ab.SCRATCHPAD_TOOL in names


def test_an_unlisted_module_is_refused(monkeypatch):
    import agent_mcp.builtin_scratchpad as B
    import agent_mcp.main as M

    monkeypatch.setattr(M, "MODULES", [m for m in M.MODULES if m is not B])
    with pytest.raises(ab.ToolNotReachable, match="not in agent_mcp.main.MODULES"):
        asyncio.run(ab.discover_catalog())


def test_a_name_routed_to_another_module_is_refused(monkeypatch):
    """A shadowing module listed first wins the name in the dispatch table."""
    import types

    import agent_mcp.builtin_scratchpad as B
    import agent_mcp.main as M

    impostor = types.ModuleType("agent_mcp.impostor")

    async def list_tools():
        return await B.list_tools()

    async def call_tool(name, arguments):
        return None

    impostor.list_tools, impostor.call_tool = list_tools, call_tool
    monkeypatch.setattr(M, "MODULES", [impostor, *M.MODULES])
    try:
        with pytest.raises(ab.ToolNotReachable, match="dispatches to"):
            asyncio.run(ab.discover_catalog())
    finally:
        monkeypatch.undo()
        asyncio.run(M.list_tools())   # leave the dispatch table as production builds it


# ── clause 3: the trial record carries the run-row tally and the cache ──────


def _stub_turn(catalog):
    """A worker turn whose model appends one note iff its catalog offers the tool."""
    import agent_mcp.main as M
    from app.sessions_io import new_background_session_id, note_run_session
    from workers.sources._common import TurnResult

    async def runner(prompt, *, max_turns, source, title, extra_disallowed):
        opts = ab.build_arm_options("scratchpad", source=source, max_turns=max_turns,
                                    shared_deny=[], session_id="x")
        opts = dataclasses.replace(opts, disallowed_tools=list(opts.disallowed_tools)
                                   + list(extra_disallowed))
        sid = new_background_session_id(source)
        note_run_session(sid)
        offered = {t["function"]["name"]
                   for t in ab.advertised_catalog(opts, catalog["discovered"])}
        if ab.SCRATCHPAD_TOOL in offered:
            res = await M.call_tool(ab.SCRATCHPAD_TOOL,
                                    {"action": "append", "content": "ruled out the cache"},
                                    meta={M.META_SESSION_ID: sid})
            assert not res.is_error, res.content[0].text
        return TurnResult(text="## Struggles\n- none", stop_reason="max_turns", num_turns=15,
                          usage={"input_tokens": 9000, "cache_read": 8000,
                                 "prompt_tokens_sum": 40000, "cache_read_sum": 30000},
                          session_id=sid)
    return runner


def _engine():
    n = {"i": 0}

    def read():
        n["i"] += 1
        return {k: float(n["i"]) * v for k, v in
                (("prompt_tokens_total", 10000), ("prefix_cache_hits_total", 7000),
                 ("prefix_cache_queries_total", 10000), ("prefill_time_sum", 2.5),
                 ("prefill_time_count", 15), ("ttft_sum", 1), ("request_count", 15))}
    return read


def _trial(arm, catalog, pair=0):
    return asyncio.run(ab.run_trial(
        arm, pair, "distill this", source=SOURCE, max_turns=15, shared_deny=[],
        runner=_stub_turn(catalog), engine=_engine(),
        usage_lookup=lambda sid, since: {"session_id": sid, "prefix_misses": 1,
                                         "reprefill_tokens": 12000}))


def test_a_scratchpad_trial_carries_the_tally_the_run_row_gets(catalog, data_root):
    rec = _trial("scratchpad", catalog)
    assert rec["status"] == "ok"
    assert rec["scratchpad"] == {"writes": 1, "bytes": len("ruled out the cache"), "sessions": 1}
    assert rec["session_ids"] == [rec["session_id"]]
    assert list(data_root.rglob("*.md")), "the note landed under the data root"
    pc = rec["prefix_cache"]
    assert (pc["hit_tokens"], pc["miss_tokens"], pc["hit_rate"]) == (30000, 10000, 0.75)
    assert (pc["miss_events"], pc["reprefill_tokens"]) == (1, 12000)
    assert rec["prefill_seconds"] == 2.5
    assert rec["prefill_seconds_estimated"] == round(10000 / ab.prefill_tokens_per_second(), 3)


def test_a_control_trial_reports_zero_writes_not_an_absent_field(catalog, data_root):
    rec = _trial("control", catalog)
    assert rec["scratchpad"] == ab.ZERO_TALLY
    assert rec["scratchpad"]["writes"] == 0
    assert not data_root.exists() or not list(data_root.rglob("*.md"))


def test_a_run_meta_without_a_tally_is_zero_filled():
    rec = ab.trial_record("control", 0, turn=None, run_meta={}, usage_row=None,
                          engine={"error": "engine down"}, wall_seconds=1.0, error="boom")
    assert rec["scratchpad"] == {"writes": 0, "bytes": 0, "sessions": 0}
    assert rec["prefix_cache"]["hit_rate"] is None
    assert rec["prefix_cache"]["miss_events"] is None, "unmeasured stays None, as the store means it"
    assert rec["prefill_seconds"] is None and rec["status"] == "error"


def test_pairs_alternate_which_arm_runs_first():
    assert [ab.pair_order(i) for i in range(4)] == [
        ("scratchpad", "control"), ("control", "scratchpad"),
        ("scratchpad", "control"), ("control", "scratchpad")]


# ── clause 4: the dated artifact and the same table on stdout ───────────────


def test_the_report_is_written_and_printed_side_by_side(catalog, data_root, tmp_path, capsys):
    records = asyncio.run(ab.run(
        "distill this", source=SOURCE, max_turns=15, shared_deny=[], pairs=2,
        out_jsonl=tmp_path / "trials.jsonl", runner=_stub_turn(catalog), engine=_engine(),
        usage_lookup=lambda sid, since: {"prefix_misses": 0, "reprefill_tokens": 0}))
    assert [r["arm"] for r in records] == ["scratchpad", "control", "control", "scratchpad"]
    capsys.readouterr()

    out_dir = tmp_path / "measurements"
    path = ab.publish(out_dir, records, {"source": SOURCE, "session": "`long.json`"},
                      day="2026-09-28")
    printed = capsys.readouterr().out

    assert path == out_dir / "scratchpad-ab-2026-09-28.md"
    body = path.read_text(encoding="utf-8")
    table = ab.render_table(ab.summarize(records))
    assert table in body and table in printed
    header = table.splitlines()[0]
    assert "scratchpad" in header and "control" in header
    hit = next(ln for ln in table.splitlines() if ln.startswith("| prefix hit rate"))
    prefill = next(ln for ln in table.splitlines() if ln.startswith("| prefill s/run (vLLM"))
    assert hit.split("|")[2].strip() == hit.split("|")[3].strip() == "0.7500"
    assert prefill.split("|")[2].strip() == "2.500"
    writes = next(ln for ln in table.splitlines() if ln.startswith("| scratchpad writes"))
    assert [c.strip() for c in writes.split("|")[2:4]] == ["1.00", "0.00"]


def test_the_default_artifact_name_is_dated():
    from datetime import datetime

    p = ab.report_path(ROOT / "eval" / "measurements")
    assert p.name == f"scratchpad-ab-{datetime.now().strftime('%Y-%m-%d')}.md"
