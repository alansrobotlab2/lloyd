#!/usr/bin/env python3
"""Paired A/B of the #1554 scratchpad on one long worker source (#1632).

`Scratchpad` shipped live in 720b4357 before any measurement of what it costs,
so this is a retroactive non-inferiority check on a deployed tool: does a long
worker turn that can keep its own notes pay more prefill, or hit the prefix
cache less, than the same turn without it? The keep-vs-revert ruling is a
human's, read off the artifact this writes; nothing here decides it.

Design, and the reason for each part (the shape of `eval/run_memory_trim_ab.py`
and `eval/run_memory_index_ab.py`, on a worker turn instead of a bench turn):

* **One named long source, run the way production runs it.** `--source
  session-distill` (#731's replay subject, and 173 `max_turns` deaths in 30
  days) with `--session` naming one long transcript. Every trial is a real
  `workers.sources._common.run_prompt_on_primary` turn on the job's own prompt
  (`session_distill.distill_prompt`): the same session mint, recorder, usage
  row and scratchpad anchor a queued run gets.
* **The arms differ by the affordance and nothing else.** `scratchpad` is the
  production turn; `control` is the same turn with `Scratchpad` on its deny
  list. How the affordance is switched today, verified before writing this: it
  is not a flag. The tool is on for every turn whose deny list does not name it
  (`agent_mcp/main.py` MODULES), the injection is `_worker_state_anchor`'s
  third clock keyed on the session id and emits nothing for a session that
  never wrote, and the tally is read off the session's file. So denying the
  tool is the whole switch: both arms keep the same anchor, and the control's
  stays silent because its file never exists. `check_arms` builds both arms'
  `RunOptions` through the production builder and refuses to run unless every
  option matches, the deny lists differ by exactly `SCRATCHPAD_DENY`, the
  advertised catalogs differ by exactly the one tool, and the assembled
  position-0 system message (builder prompt + denied-tools block + the
  deferred-tool catalog reminder the loop appends) differs by exactly the
  scratchpad's name in the denied block and its one catalog line.
* **The tool is reached through production's registration.** `discover_catalog`
  lists the aggregator in-process (`agent_mcp.main.list_tools`, which builds the
  dispatch table `tools/list` is served from) and refuses unless `Scratchpad`
  dispatches to `builtin_scratchpad` from `MODULES`; the catalogs `check_arms`
  compares are built from that discovery with the loop's own
  `build_tool_list`. A trial's `writes` can only be non-zero if a model call
  crossed that dispatcher, because the handler is the only writer of the file.
* **Both arms are read-only, alike.** Every tool that is not `readOnlyHint`
  (`agent_mcp.annotations.READ_ONLY`) is denied in both arms, except
  `Scratchpad` itself on the arm under test. `scripts/replay_run_state.py` made
  the same choice for #731 with a hand list; deriving it from the annotations
  is the sandbox's own rule, and a replay of session-distill must not write
  facts. The session id is a normal worker id, not a sandboxed `bench_` one,
  because the aggregator sandbox refuses every non-read-only tool — the
  scratchpad included — and a sandboxed scratchpad arm would measure dormant
  machinery.
* **Each trial record carries the run-row tally.** The trial binds
  `sessions_io.current_run_sessions` exactly as `workers/pool.py` does around a
  job and records `workers.pool._scratchpad_meta()` — the object the pool writes
  as `meta_json.scratchpad` — so `scratchpad.writes` is the number a run row
  would carry, zero-filled, never absent. Beside it: prefix-cache hit and miss
  tokens from the turn's own usage sums (`prompt_tokens_sum`/`cache_read_sum`,
  attributable), the run's usage row `prefix_misses`/`reprefill_tokens`
  (`app/prefix_miss.py`), and prefill seconds from vLLM's own
  `request_prefill_time_seconds` delta around the trial (authoritative but
  engine-global — hence the paused pool), with the token-rate estimate kept
  beside it.
* **Paired, order-balanced.** Pairs run ABBA (`pair_order`): a warm prefix left
  by the previous trial favours whichever arm runs second, so the order flips
  every pair rather than handing one arm the warm cache every time.

Usage — needs the primary to itself: the worker pool paused (the driver pauses
it and resumes only a pause it took), nothing else on the engine, and the
primary lock held. Run from the live tree: the aggregator writes scratchpads
under the production data root, and a tally read anywhere else is a zero.

    flock <scratch>/primary.lock .venvs/lloyd/bin/python eval/run_scratchpad_ab.py \\
        --session <long session stem> --pairs 4
    .venvs/lloyd/bin/python eval/run_scratchpad_ab.py --session <stem> --check  # arms only
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ITEM = 1632
ARMS = ("scratchpad", "control")
SCRATCHPAD_TOOL = "Scratchpad"
SERVER = "lloyd-mcp"
SCRATCHPAD_MODULE = "agent_mcp.builtin_scratchpad"
#: The one thing the control arm is denied that the scratchpad arm is not. Both
#: spellings, the way `_worker_run_options` denies the automod tools: the bare
#: name is what the harness advertises, the prefixed one what legacy config uses.
SCRATCHPAD_DENY = (SCRATCHPAD_TOOL, f"mcp__{SERVER}__{SCRATCHPAD_TOOL}")

#: The long sources this can replay: name -> (prompt builder, default turn cap).
#: One today. A source joins by exposing its production prompt as a function —
#: a copy of the prose here would drift, as `scripts/replay_run_state.py`'s has.
def _session_distill() -> tuple[Callable[[str], str], int]:
    from workers.sources.session_distill import DEFAULT_MAX_TURNS, distill_prompt
    return distill_prompt, DEFAULT_MAX_TURNS


SOURCES: dict[str, Callable[[], tuple[Callable[[str], str], int]]] = {
    "session-distill": _session_distill,
}

#: Scratchpad totals as `workers/pool.py` writes them when a tally fails: the
#: shape an absent field is filled to, so no arm ever reports a missing count.
ZERO_TALLY = {"writes": 0, "bytes": 0, "sessions": 0}

#: `RunOptions` fields allowed to differ between the arms, and why:
#: `system_prompt` and `disallowed_tools` carry the affordance (checked
#: separately, to the byte); the rest are per-run identity or live objects
#: compared by type (see `_same_option`).
_OPTION_FIELDS_CHECKED_ELSEWHERE = frozenset({"system_prompt", "disallowed_tools"})


class ArmMismatch(RuntimeError):
    """The two arms differ by more than the scratchpad affordance."""


class ToolNotReachable(RuntimeError):
    """The scratchpad is not dispatchable the way production dispatches it."""


class NotReady(RuntimeError):
    """The machine is not in a state a trial may be measured in."""


# --------------------------------------------------------------------------
# The tool, through production's registration (clause 2)
# --------------------------------------------------------------------------


def _tool_dict(tool: Any) -> dict[str, Any]:
    """An MCP `Tool` in the shape `MCPPool.discovered` holds (`build_tool_list`'s input)."""
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "input_schema", None)
    return {"name": tool.name, "description": tool.description or "",
            "inputSchema": schema or {"type": "object", "properties": {}}}


async def discover_catalog() -> dict[str, Any]:
    """The aggregator's catalog, listed in-process, with the scratchpad's route.

    `agent_mcp.main.list_tools` is what `tools/list` serves and what rebuilds the
    dispatch table `call_tool` routes through, so a tool it lists and routes to
    `builtin_scratchpad` is one a model's call reaches. Raises
    `ToolNotReachable` for a module un-listed from MODULES, a name nobody
    exports, or a name that routes anywhere else.
    """
    import agent_mcp.main as M

    if not any(getattr(m, "__name__", "") == SCRATCHPAD_MODULE for m in M.MODULES):
        raise ToolNotReachable(f"{SCRATCHPAD_MODULE} is not in agent_mcp.main.MODULES, "
                               "so no turn can call Scratchpad")
    tools = await M.list_tools()
    if SCRATCHPAD_TOOL not in {t.name for t in tools}:
        raise ToolNotReachable(f"the aggregator does not list {SCRATCHPAD_TOOL}")
    routed = getattr(M._dispatch.get(SCRATCHPAD_TOOL), "__name__", None)
    if routed != SCRATCHPAD_MODULE:
        raise ToolNotReachable(f"{SCRATCHPAD_TOOL} dispatches to {routed!r}, "
                               f"not {SCRATCHPAD_MODULE}")
    return {"discovered": [(SERVER, [_tool_dict(t) for t in tools])],
            "module": routed, "tools": len(tools)}


def read_only_deny(discovered: list[tuple[str, list[dict[str, Any]]]]) -> list[str]:
    """Every discovered tool that can change something, except the one under test.

    Shared by both arms, so it is not a difference between them.
    """
    from agent_mcp.annotations import read_only_tool_names

    ro = read_only_tool_names()
    return sorted({t["name"] for _srv, tools in discovered for t in tools
                   if t["name"] not in ro and t["name"] != SCRATCHPAD_TOOL})


def arm_extra_disallowed(arm: str, shared: list[str]) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return list(shared) + (list(SCRATCHPAD_DENY) if arm == "control" else [])


# --------------------------------------------------------------------------
# Arms (clause 1)
# --------------------------------------------------------------------------


def build_arm_options(arm: str, *, source: str, max_turns: int, shared_deny: list[str],
                      session_id: str, build=None):
    """The `RunOptions` `run_prompt_on_primary` gives this arm's turn.

    Same builder, same arguments, as the trial's own call: `session_id` is a
    stand-in (the trial mints its own), and it is the same for both arms here so
    the only argument that differs is `extra_disallowed`.
    """
    if build is None:
        from workers.sources._common import _worker_run_options as build
    return build(max_turns, source=source, session_id=session_id,
                 extra_disallowed=arm_extra_disallowed(arm, shared_deny))


def advertised_catalog(options, discovered) -> list[dict[str, Any]]:
    """The tools array the loop builds for this turn (`loop._open_turn`)."""
    from app.harness import loop as L
    from app.harness.tool_schema import build_tool_list

    hidden = L._surface_hidden(options) | L._allow_list_hidden(options, discovered)
    if not getattr(options, "intra_turn_microcompact_observation_stubs", False):
        hidden.add(L.RECALL_OBSERVATION_TOOL)
    return build_tool_list(list(discovered), set(options.disallowed_tools) | hidden)


def assemble_position_zero(options, catalog: list[dict[str, Any]]) -> str:
    """The system message a turn's first request carries, as the loop builds it.

    The builder's prompt (denied-tools block included), plus the deferred-tool
    catalog reminder `loop._inject_catalog_reminder` appends when tool search is
    on — the activation rule of `loop._resolve_loaded_tool_set`, on a fresh
    session (nothing loaded yet).
    """
    from app.harness import loop as L
    from app.harness.tool_search import LoadedToolSet

    disallowed = set(options.disallowed_tools or [])
    enabled = (options.tool_search_enabled
               and L.TOOLSEARCH_TOOL_NAME not in disallowed
               and len(catalog) >= options.tool_search_threshold_tools)
    messages = [{"role": "system", "content": options.system_prompt}]
    if enabled:
        L._inject_catalog_reminder(messages, LoadedToolSet(catalog=catalog, enabled=True))
    return messages[0]["content"]


def _same_option(a: Any, b: Any) -> bool:
    """Values equal; live objects (hooks, anchors, callbacks) by type."""
    if a is None or b is None:
        return a is b
    if callable(a) or callable(b) or not _plain(a):
        return type(a) is type(b)
    return a == b


def _plain(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, list, tuple, dict, set, frozenset))


def _catalog_line(catalog: list[dict[str, Any]], name: str) -> str | None:
    from app.harness.tool_search import format_catalog_reminder

    one = [t for t in catalog if t["function"]["name"] == name]
    if not one:
        return None
    body = format_catalog_reminder(one)
    return next((ln for ln in body.splitlines() if ln.startswith(f"- {name}")), None)


def check_arms(on, off, discovered) -> dict[str, Any]:
    """Raise `ArmMismatch` unless the arms differ by the scratchpad and nothing else."""
    from workers.sources._common import build_denied_tools_block

    fields = [f.name for f in dataclasses.fields(on)]
    differ = [n for n in fields if n not in _OPTION_FIELDS_CHECKED_ELSEWHERE
              and not _same_option(getattr(on, n), getattr(off, n))]
    if differ:
        raise ArmMismatch(f"arms' RunOptions differ in {differ}")

    cat_on, cat_off = advertised_catalog(on, discovered), advertised_catalog(off, discovered)
    names_on = [t["function"]["name"] for t in cat_on]
    names_off = [t["function"]["name"] for t in cat_off]
    if SCRATCHPAD_TOOL not in names_on:
        raise ArmMismatch(f"the scratchpad arm does not advertise {SCRATCHPAD_TOOL} "
                          "(disabled in config or hidden on this surface?)")

    on_deny, off_deny = set(on.disallowed_tools), set(off.disallowed_tools)
    if not on_deny <= off_deny or off_deny - on_deny != set(SCRATCHPAD_DENY):
        raise ArmMismatch(f"deny lists differ by {sorted(off_deny ^ on_deny)}, "
                          f"not exactly {list(SCRATCHPAD_DENY)}")
    if [t for t in cat_on if t["function"]["name"] != SCRATCHPAD_TOOL] != cat_off:
        raise ArmMismatch("advertised catalogs differ by more than the scratchpad tool")

    # Position 0, to the byte: take the scratchpad arm's system message, swap its
    # denied block for the control's and drop the tool's one catalog line — what
    # is left must be the control's system message exactly.
    sys_on = assemble_position_zero(on, cat_on)
    sys_off = assemble_position_zero(off, cat_off)
    block_on = build_denied_tools_block(on.disallowed_tools)
    block_off = build_denied_tools_block(off.disallowed_tools)
    if not block_on or not on.system_prompt.endswith(block_on) \
            or not off.system_prompt.endswith(block_off):
        raise ArmMismatch("a builder prompt does not end with its denied-tools block")
    base = on.system_prompt[:-len(block_on)]
    if off.system_prompt[:-len(block_off)] != base:
        raise ArmMismatch("builder prompts differ outside the denied-tools block")
    expected = sys_on.replace(on.system_prompt, base + block_off, 1)
    line = _catalog_line(cat_on, SCRATCHPAD_TOOL)
    if line and sys_on.count(line) == 1:
        expected = expected.replace("\n" + line, "", 1)
    if expected != sys_off:
        raise ArmMismatch("assembled system prompts differ by more than the scratchpad "
                          "in the denied-tools block and the catalog reminder")
    return {"system_chars": {"scratchpad": len(sys_on), "control": len(sys_off)},
            "tools": {"scratchpad": len(names_on), "control": len(names_off)},
            "shared_deny": len(on_deny), "catalog_line": bool(line)}


# --------------------------------------------------------------------------
# One trial (clause 3)
# --------------------------------------------------------------------------


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def prefill_tokens_per_second() -> float:
    from app.harness.run_state import PREFILL_TOKENS_PER_SECOND
    return PREFILL_TOKENS_PER_SECOND


def trial_record(arm: str, pair: int, *, turn: Any, run_meta: dict[str, Any],
                 usage_row: dict[str, Any] | None, engine: dict[str, Any] | None,
                 wall_seconds: float, error: str = "") -> dict[str, Any]:
    """One trial, as the report reads it.

    `scratchpad` is the run row's `meta_json.scratchpad`, zero-filled: an arm that
    never touched the tool reads `writes: 0`, never a missing key. Hit/miss tokens
    are the turn's own sums (attributable to this run); `miss_events` and
    `reprefill_tokens` its usage row's prefix-miss count (None = unmeasured, as
    the store means it); `prefill_seconds` vLLM's own delta around the trial,
    `prefill_seconds_estimated` the miss tokens at the engine's measured rate.
    """
    tally = {**ZERO_TALLY, **{k: _int(v) for k, v in
                              ((run_meta or {}).get("scratchpad") or {}).items()
                              if k in ZERO_TALLY}}
    usage = dict(getattr(turn, "usage", None) or {})
    prompt = _int(usage.get("prompt_tokens_sum")) or _int(usage.get("input_tokens"))
    hits = min(prompt, _int(usage.get("cache_read_sum")) or _int(usage.get("cache_read")))
    eng = engine or {}
    measured = eng.get("prefill_seconds_measured") if "error" not in eng else None
    row = usage_row or {}
    return {
        "arm": arm, "pair": pair,
        "status": "error" if error else "ok", "error": error,
        "session_id": getattr(turn, "session_id", "") or "",
        "session_ids": list((run_meta or {}).get("session_ids") or []),
        "stop_reason": getattr(turn, "stop_reason", None),
        "num_turns": getattr(turn, "num_turns", None),
        "wall_seconds": round(wall_seconds, 2),
        "scratchpad": tally,
        "prefix_cache": {
            "prompt_tokens": prompt, "hit_tokens": hits, "miss_tokens": prompt - hits,
            "hit_rate": round(hits / prompt, 4) if prompt else None,
            "miss_events": row.get("prefix_misses"),
            "reprefill_tokens": row.get("reprefill_tokens"),
        },
        "prefill_seconds": None if measured is None else round(float(measured), 3),
        "prefill_seconds_estimated": round((prompt - hits) / prefill_tokens_per_second(), 3),
        "engine": eng,
    }


_USAGE_COLUMNS = ["session_id", "ts", "prefix_misses", "reprefill_tokens",
                  "input_tokens", "cache_read", "num_turns", "stop_reason"]


def usage_row_for(session_id: str, since_ts: str) -> dict[str, Any] | None:
    """The run's usage row, read-only (the recorder books one per run)."""
    from app.paths import USAGE_DB
    from app.usage_store import read_rows_readonly

    if not session_id or not Path(USAGE_DB).exists():
        return None
    rows = [r for r in read_rows_readonly(USAGE_DB, _USAGE_COLUMNS, since_ts)
            if r.get("session_id") == session_id]
    return rows[-1] if rows else None


async def run_trial(arm: str, pair: int, prompt: str, *, source: str, max_turns: int,
                    shared_deny: list[str], runner=None,
                    engine: Callable[[], dict[str, float]] | None = None,
                    usage_lookup: Callable[[str, str], dict | None] = usage_row_for,
                    ) -> dict[str, Any]:
    """One production worker turn, with the pool's run-row bookkeeping around it."""
    from app.sessions_io import current_run_sessions
    from workers.pool import _scratchpad_meta

    if runner is None:
        from workers.sources._common import run_prompt_on_primary as runner
    if engine is None:
        from scripts.replay_run_state import read_engine as engine
    from scripts.replay_run_state import engine_delta

    since = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
    token = current_run_sessions.set([])     # what workers/pool.py binds per job
    turn, error = None, ""
    started = time.perf_counter()
    try:
        before = engine()
        try:
            turn = await runner(prompt, max_turns=max_turns, source=source,
                                title=f"scratchpad-ab {arm} pair {pair}",
                                extra_disallowed=arm_extra_disallowed(arm, shared_deny))
        except Exception as exc:  # noqa: BLE001 — a failed trial is a record, not a crash
            error = f"{type(exc).__name__}: {exc}"
        delta = engine_delta(before, engine())
        run_meta = {"session_ids": list(current_run_sessions.get() or []),
                    "scratchpad": _scratchpad_meta()}
    finally:
        current_run_sessions.reset(token)
    sid = getattr(turn, "session_id", "") or ""
    return trial_record(arm, pair, turn=turn, run_meta=run_meta,
                        usage_row=usage_lookup(sid, since), engine=delta,
                        wall_seconds=time.perf_counter() - started, error=error)


def pair_order(pair: int) -> tuple[str, str]:
    """ABBA: the arm that runs first alternates, so neither owns the warm cache."""
    return ARMS if pair % 2 == 0 else ARMS[::-1]


async def run(prompt: str, *, source: str, max_turns: int, shared_deny: list[str],
              pairs: int, out_jsonl: Path | None = None, **trial_kw) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in range(pairs):
        for arm in pair_order(pair):
            rec = await run_trial(arm, pair, prompt, source=source, max_turns=max_turns,
                                  shared_deny=shared_deny, **trial_kw)
            records.append(rec)
            if out_jsonl is not None:
                with out_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            pc = rec["prefix_cache"]
            print(f"pair {pair} {arm:10s} {rec['status']} stop={rec['stop_reason']} "
                  f"writes={rec['scratchpad']['writes']} hit={pc['hit_rate']} "
                  f"prefill={rec['prefill_seconds']}s", flush=True)
    return records


# --------------------------------------------------------------------------
# Report (clause 4)
# --------------------------------------------------------------------------


def _mean(xs: list[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm means over the trials that ran; errored trials are counted, not averaged."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        ok = [r for r in records if r["arm"] == arm and r["status"] == "ok"]
        out[arm] = {
            "runs": len(ok),
            "errors": sum(1 for r in records if r["arm"] == arm and r["status"] != "ok"),
            "hit_rate_mean": _mean([r["prefix_cache"]["hit_rate"] for r in ok]),
            "prefill_s_per_run_mean": _mean([r["prefill_seconds"] for r in ok]),
            "prefill_s_per_run_estimated_mean": _mean(
                [r["prefill_seconds_estimated"] for r in ok]),
            "miss_events_mean": _mean([r["prefix_cache"]["miss_events"] for r in ok]),
            "reprefill_tokens_mean": _mean([r["prefix_cache"]["reprefill_tokens"] for r in ok]),
            "prompt_tokens_mean": _mean([r["prefix_cache"]["prompt_tokens"] for r in ok]),
            "scratchpad_writes_mean": _mean([r["scratchpad"]["writes"] for r in ok]),
            "max_turns_endings": sum(1 for r in ok if r["stop_reason"] == "max_turns"),
        }
    return out


_ROWS = (
    ("runs", "runs", "{:.0f}"),
    ("prefix hit rate (mean/run)", "hit_rate_mean", "{:.4f}"),
    ("prefill s/run (vLLM, mean)", "prefill_s_per_run_mean", "{:.3f}"),
    ("prefill s/run (estimated, mean)", "prefill_s_per_run_estimated_mean", "{:.3f}"),
    ("prompt tokens/run (sum over iterations)", "prompt_tokens_mean", "{:,.0f}"),
    ("prefix miss events/run", "miss_events_mean", "{:.2f}"),
    ("re-prefilled tokens/run", "reprefill_tokens_mean", "{:,.0f}"),
    ("scratchpad writes/run", "scratchpad_writes_mean", "{:.2f}"),
    ("ended on max_turns", "max_turns_endings", "{:.0f}"),
    ("errored trials", "errors", "{:.0f}"),
)


def _fmt(v: Any, spec: str) -> str:
    return "—" if v is None else spec.format(v)


def render_table(summary: dict[str, Any]) -> str:
    """Both arms side by side, with scratchpad − control."""
    a, b = summary["scratchpad"], summary["control"]
    lines = ["| metric | scratchpad | control | Δ (scratchpad − control) |",
             "|---|---:|---:|---:|"]
    for label, key, spec in _ROWS:
        d = (None if a.get(key) is None or b.get(key) is None
             else round(a[key] - b[key], 4))
        lines.append(f"| {label} | {_fmt(a.get(key), spec)} | {_fmt(b.get(key), spec)} "
                     f"| {_fmt(d, spec.replace('{:', '{:+'))} |")
    return "\n".join(lines)


def _trial_table(records: list[dict[str, Any]]) -> str:
    lines = ["| pair | arm | status | stop | writes | hit rate | miss events | prefill s | est. s |",
             "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        pc = r["prefix_cache"]
        lines.append(f"| {r['pair']} | {r['arm']} | {r['status']} | {r['stop_reason']} "
                     f"| {r['scratchpad']['writes']} | {_fmt(pc['hit_rate'], '{:.4f}')} "
                     f"| {_fmt(pc['miss_events'], '{}')} | {_fmt(r['prefill_seconds'], '{:.3f}')} "
                     f"| {r['prefill_seconds_estimated']:.3f} |")
    return "\n".join(lines)


def report_path(out_dir: Path, day: str | None = None) -> Path:
    return out_dir / f"scratchpad-ab-{day or datetime.now().strftime('%Y-%m-%d')}.md"


def write_report(out_dir: Path, records: list[dict[str, Any]], context: dict[str, Any],
                 *, day: str | None = None) -> tuple[Path, str]:
    """Write `scratchpad-ab-<date>.md` and return (path, the side-by-side table)."""
    summary = summarize(records)
    table = render_table(summary)
    ctx = "\n".join(f"- **{k}**: {v}" for k, v in context.items())
    body = f"""# Scratchpad A/B — {context.get('source', '?')} (#{ITEM})

Paired A/B of the #1554 `Scratchpad` affordance on one long worker source,
written by `eval/run_scratchpad_ab.py`. The `scratchpad` arm is the production
worker turn; `control` is the same turn with `Scratchpad` denied, and the driver
refused to run unless the two assembled system prompts differed by that and
nothing else. The keep-vs-revert ruling on the live tool is a human's, from this
table.

{ctx}

## Both arms

{table}

Hit rate is the turn's own `cache_read_sum / prompt_tokens_sum` (attributable to
the run). Prefill s is vLLM's `request_prefill_time_seconds` delta around the
trial, which is engine-global: it is only this run's with the pool paused and
nothing else on the primary. The estimate is the miss tokens at
{prefill_tokens_per_second():,.0f} tok/s.

## Trials

{_trial_table(records)}

```json
{json.dumps(summary, indent=2)}
```
"""
    path = report_path(out_dir, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path, table


def publish(out_dir: Path, records: list[dict[str, Any]], context: dict[str, Any],
            *, day: str | None = None) -> Path:
    """Write the artifact and print the same side-by-side table to stdout."""
    path, table = write_report(out_dir, records, context, day=day)
    print("\n" + table)
    print(f"\nreport: {path}")
    return path


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


async def _pool_state() -> dict[str, Any]:
    import httpx

    from scripts.replay_run_state import BACKEND
    async with httpx.AsyncClient(timeout=10.0) as c:
        return (await c.get(f"{BACKEND}/api/workers/pause")).json()


async def prepare(source: str, session: str, max_turns: int | None) -> dict[str, Any]:
    """Resolve the source and session, discover the tool, build and check both arms."""
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}; known: {sorted(SOURCES)}")
    from scripts.replay_run_state import resolve_session

    build_prompt, default_turns = SOURCES[source]()
    path = resolve_session(session)
    turns = max_turns or default_turns
    cat = await discover_catalog()
    shared = read_only_deny(cat["discovered"])
    opts = {arm: build_arm_options(arm, source=source, max_turns=turns, shared_deny=shared,
                                   session_id="scratchpad-ab-check")
            for arm in ARMS}
    sizes = check_arms(opts["scratchpad"], opts["control"], cat["discovered"])
    return {"prompt": build_prompt(str(path)), "session_path": path, "max_turns": turns,
            "shared_deny": shared, "catalog": cat, "arm_check": sizes}


async def amain(args) -> int:
    prep = await prepare(args.source, args.session, args.max_turns)
    print(f"{args.source} on {prep['session_path'].name} "
          f"({prep['session_path'].stat().st_size:,} B); "
          f"{SCRATCHPAD_TOOL} -> {prep['catalog']['module']}; "
          f"arms verified: {json.dumps(prep['arm_check'])}")
    if args.check:
        return 0

    from app.paths import DATA_ROOT, IS_PRODUCTION_DATA
    if not IS_PRODUCTION_DATA:
        raise NotReady(f"data root {DATA_ROOT} is not production's: the aggregator writes "
                       "scratchpads under the production root, so every tally here would "
                       "read zero. Run from the live tree.")

    took_pause = False
    if not args.no_pause:
        from scripts.replay_run_state import set_pool_paused
        state = await _pool_state()
        if not state.get("paused"):
            print(await set_pool_paused(True))
            took_pause = True
        if not (await _pool_state()).get("paused"):
            raise NotReady("the worker pool did not pause; a trial measured beside live "
                           "worker turns reads their prefill as its own")
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        raw = args.out / f"scratchpad-ab-{day}.jsonl"
        raw.unlink(missing_ok=True)
        started = time.time()
        records = await run(prep["prompt"], source=args.source, max_turns=prep["max_turns"],
                            shared_deny=prep["shared_deny"], pairs=args.pairs, out_jsonl=raw)
    finally:
        if took_pause:
            from scripts.replay_run_state import set_pool_paused
            print(await set_pool_paused(False))

    context = {
        "source": args.source,
        "session": f"`{prep['session_path'].name}` ({prep['session_path'].stat().st_size:,} B)",
        "pairs": f"{args.pairs} (ABBA order)", "max_turns": prep["max_turns"],
        "ran": datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip(),
        "wall": f"{time.time() - started:,.0f} s", "commit": _git_head(),
        "tool route": f"{SCRATCHPAD_TOOL} -> {prep['catalog']['module']} "
                      f"({prep['catalog']['tools']} tools listed)",
        "shared deny list": f"{len(prep['shared_deny'])} non-read-only tools, both arms",
        "pool": "paused by this run" if took_pause else
                ("not paused (--no-pause)" if args.no_pause else "already paused"),
        "raw trials": f"`{raw.name}`",
    }
    publish(args.out, records, context, day=day)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="session-distill", choices=sorted(SOURCES))
    ap.add_argument("--session", required=True,
                    help="the long transcript to replay: a session stem or an absolute path")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="turn cap for both arms (default: the source's own)")
    ap.add_argument("--out", type=Path, default=HERE / "measurements")
    ap.add_argument("--check", action="store_true",
                    help="discover the tool, build and verify both arms, run nothing")
    ap.add_argument("--no-pause", action="store_true",
                    help="do not touch the worker pool (the prefill numbers are then "
                         "not this run's alone)")
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
