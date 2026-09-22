#!/usr/bin/env python3
"""A/B the `harness.preserve_thinking_iterations` knob on a real task.

Qwen3.8-Flash-Next's model card says preserved thinking reduces redundant
reasoning in agent loops. That is a claim about the model, not a
measurement of this harness, so it gets measured before it ships on by
default.

WHAT THE KNOB ACTUALLY DOES, AND WHAT THIS SCRIPT MEASURED BEFORE
------------------------------------------------------------------
`preserve_thinking_iterations` is enforced at exactly ONE place in the
shipped harness: turn entry. `app/harness/loop.py:236` calls
`_cap_history_reasoning` (`app/harness/loop.py:1561`), which windows the
reasoning ARRIVING FROM EARLIER TURNS down to the most recent `keep`
reasoning-bearing assistant messages. Inside a turn the same knob is a
bare boolean (`app/harness/loop.py:532`,
`reasoning=thinking_text if keep_reasoning > 0 else ""`): a positive value
attaches this iteration's reasoning to the assistant message, and nothing
already appended is ever edited again — because taking tokens out of a
message vLLM has already prefilled collapses the prefix cache (backlog
#520). A turn that stays inside one turn is therefore NEVER windowed,
whatever `keep` says. The only intra-turn trim left is `_relieve_context`
rung 2 (`app/harness/loop.py:1138`), which fires only when the context
meter is already over target — emergency relief near the wall, not a bound.

Every number this script published BEFORE commit `2677dea` (2026-09-09,
"item #520: apply the preserved-thinking window at turn entry, not every
iteration") describes the per-iteration mechanism this script no longer
measures, and must not be quoted as a measurement of shipped behaviour
(backlog #617). Back then `preserve_thinking_iterations=N` meant "carry at
most N iterations of reasoning, re-trimmed inside the loop every
iteration", so a single-turn A/B priced an intra-turn token bound. Under
the shipped mechanism a single turn is not windowed at all: a one-turn A/B
compares "attach reasoning" against "attach nothing", and the `keep` value
makes no difference to it — the gap it reports is a turn-boundary effect
wearing an intra-turn label, which is what made the old numbers wrong
rather than merely stale.

THE SESSION IS THEREFORE TWO TURNS
----------------------------------
Both turns share one `chat_messages` buffer (`RunOptions
.chat_messages_handle`), which is the state the turn-entry cap sees in
production: prior-turn assistant messages that still carry their
`reasoning`.

  * turn 1 (the warm-up) runs TASK below — a multi-step code investigation
    that accumulates a real reasoning mass. Its numbers are reported but
    are not the A/B.
  * turn 2 (the measured turn) runs FOLLOWUP — a short answer-from-memory
    query. The cap fires at its entry, so its peak input tokens and its
    `carried_reasoning_chars` ARE the thing the knob bounds.

`carried_reasoning_chars` is read off the first request of turn 2 by a
transparent wrapper around `app.harness.loop.stream_chat`, so it is what
went on the wire, not what the buffer looks like afterwards.

ARMS — what each one varies now
-------------------------------
  off         keep=0. No reasoning is attached to any assistant message, in
              either turn. The control: pre-2026-09-05 behaviour.
  window=N    keep=N. Reasoning is carried, and the turn-entry window lets
              the N most recent reasoning-bearing assistant messages
              through into the next turn. The shipped default shape
              (`harness.preserve_thinking_iterations: 6`).
  all         keep=999 (or whatever `--carry-all` says). Reasoning is
              carried and the cap is so wide it never binds.

The decisive comparison is `window=N` against `all`: that is what the
turn-entry window costs or saves. `off` against either prices preserved
thinking itself. Arms are interleaved within each trial so drift in server
state (KV-cache occupancy, other Lloyd traffic) lands on all of them; the
warm-up turn of every arm is the same scripted task, and one arm per boot
is still the discipline when the primary engine is being shared.

    .venvs/lloyd/bin/python eval/run_preserve_thinking_eval.py --trials 3
    .venvs/lloyd/bin/python eval/run_preserve_thinking_eval.py \\
        --trials 1 --keep 2 --carry-all 999 --out /tmp/pt.json

A live re-run needs `agent-llm-primary` up (95.37 GiB host-RAM n-gram
table, `MemAvailable` gate before boot), which is an operator scheduling
decision. `tests/test_preserve_thinking_eval.py` pins the
session shape and the window's effect on a scripted engine, so the
mechanism is checked without a GPU.

The last live run under the shipped mechanism (2026-09-21, 3 trials, folded
into a primary restart as #617 asked) is in
`eval/measurements/yarn-2026-09-21.md`: against `off`, `window=6` cut output
tokens 34% and tool calls 11%, `all` 34% and 18%, with the same answers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

LLOYD_HOME = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LLOYD_HOME))

TASK = (
    "In this repo (/home/alansrobotlab/lloyd), trace end to end how a tool "
    "that has been disabled in config actually stops reaching the model. "
    "Name every file and function on the path, from the config key to the "
    "point the model can no longer call it, with file:line references. "
    "Read the code — do not guess. Finish with a numbered list of the steps."
)

# Turn 2 has to be answerable from what the model already thought about, and
# must not be able to dodge the question by reading files again: it is the
# turn that spends the reasoning the window let through.
FOLLOWUP = (
    "Do not call any tools and do not read any more files. From the "
    "investigation you just did, write the final numbered list of steps, "
    "condensed to one line per step with its file:line, and name the one "
    "step you were least sure about and why."
)

DEFAULT_KEEP = 6
DEFAULT_CARRY_ALL = 999

# Headline metrics, in the order the summary table prints them.
METRICS = (
    "seconds", "output_tokens", "iterations", "tool_calls",
    "prior_reasoning_chars", "carried_reasoning_blocks",
    "carried_reasoning_chars", "window_removed_chars",
    "measured_seconds", "measured_peak_input", "answer_chars", "completed",
)


def _arm_label(keep: int, carry_all: int) -> str:
    """Arm name that says what the value varies under the shipped mechanism."""
    if keep <= 0:
        return "off"
    if carry_all and keep >= carry_all:
        return "all"
    return f"window={keep}"


def _install_request_probe(state: dict, requests: list[dict], *,
                           capture: bool) -> Callable[[], None]:
    """Wrap `app.harness.loop.stream_chat` to record what each request carried.

    Transparent: it tallies the `reasoning` still on the wire and hands the
    call straight to the real client, so it observes the request the loop
    assembled without changing it. Each entry is tagged with the turn the
    loop was in, because the quantity that matters — what the turn-entry
    window let through — is the FIRST request of turn 2.

    The snapshot is per-message `dict(m)`, so a later in-place `pop` by the
    cap or by microcompaction does not retroactively appear in an earlier
    entry. That is what makes this a record of what was sent rather than of
    what the buffer looks like afterwards.

    Returns the restore function; the caller must run it.
    """
    import app.harness.loop as loop_mod

    real = loop_mod.stream_chat

    def _probe(**kwargs):
        msgs = kwargs.get("messages") or []
        blocks = [m for m in msgs
                  if m.get("role") == "assistant" and m.get("reasoning")]
        entry: dict[str, Any] = {
            "turn": state["turn"],
            "n_messages": len(msgs),
            "reasoning_blocks": len(blocks),
            "reasoning_chars": sum(len(m.get("reasoning") or "") for m in blocks),
        }
        if capture:
            entry["bodies"] = [dict(m) for m in msgs]
        requests.append(entry)
        return real(**kwargs)

    loop_mod.stream_chat = _probe
    return lambda: setattr(loop_mod, "stream_chat", real)


async def _run_turn(run_query, options, *, expect_user_message: str) -> dict:
    """Drain one turn (one `run_query`) and collect its numbers.

    The two guards here are what make a two-turn A/B measurable at all. If
    the loop ignored `chat_messages_handle`, each turn would start from an
    empty buffer, no prior-turn assistant message would ever reach the
    turn-entry cap, and both arms would report the same numbers while
    looking like a clean result — the failure mode that left #617's premise
    standing for a week. So: the user turn we just appended must be the last
    message going in, and the loop must have written its own messages back to
    the same list coming out.
    """
    handle = options.chat_messages_handle
    if not handle or handle[-1].get("content") != expect_user_message:
        raise RuntimeError(
            "the shared chat_messages buffer does not end on this turn's "
            f"user message (wanted {expect_user_message[:40]!r}...), so the "
            "script and the loop are not sharing one history."
        )
    n_before = len(handle)

    out_tokens = 0
    peak_in = 0
    iterations = 0
    tools = 0
    answer_chars = 0
    stop = None
    t0 = time.perf_counter()
    async for evt in run_query([], options):
        # `[]`, not the user turn: this turn's user message is already in
        # options.chat_messages_handle, and the loop only splices the
        # `messages` argument in when the handle is still empty
        # (app/harness/loop.py:121-124). Passing both would double it.
        if evt["type"] == "tool_call":
            tools += 1
        elif evt["type"] == "assistant_message":
            iterations += 1
            u = evt.get("usage") or {}
            out_tokens += int(u.get("output_tokens", 0) or 0)
            peak_in = max(peak_in, int(u.get("input_tokens", 0) or 0))
            answer_chars = len(evt.get("text") or "") or answer_chars
        elif evt["type"] == "result":
            stop = evt.get("stop_reason")

    if len(handle) <= n_before:
        raise RuntimeError(
            "the loop appended nothing to the shared chat_messages buffer, "
            "so the next turn has no prior-turn assistant message to window "
            "and this arm would measure nothing. Check "
            "RunOptions.chat_messages_handle semantics."
        )
    return {
        "seconds": round(time.perf_counter() - t0, 1),
        "output_tokens": out_tokens,
        "iterations": iterations,
        "tool_calls": tools,
        "peak_input_tokens": peak_in,
        "answer_chars": answer_chars,
        "stop_reason": stop,
    }


#: The one place this driver's session-id shape is written. A trial replays
#: real recorded prompts with the live toolbox, and `app/harness/mcp_pool.py`
#: stamps `RunOptions.session_id` into the `lloyd/session_id` of every tool
#: call's `_meta`, where the aggregator decides sandboxing from the prefix alone
#: (`agent_mcp/main.py:476`, `agent_mcp/_tool_sandbox.py:54`). An id minted
#: without this prefix is therefore not a cheaper trial — it is a measurement run
#: holding the full write surface. `tests/test_tool_sandbox.py` pins it through
#: `new_trial_session_id` (#1333).
SANDBOXED_TRIAL_ID_PREFIX = "pt-eval-"


def new_trial_session_id(keep: int, *, now: float | None = None) -> str:
    """The session id one trial runs under: `pt-eval-<keep>-<epoch seconds>`.

    A named module-level mint rather than an inline f-string so the id a test
    asserts the aggregator sandboxes is the id a trial actually uses. `now` is
    injectable for callers that need a deterministic id; production callers omit
    it and get the wall clock.
    """
    ts = int(time.time() if now is None else now)
    return f"{SANDBOXED_TRIAL_ID_PREFIX}{keep}-{ts}"


def require_sandboxed_trial_session(session_id: str) -> str:
    """Refuse to run a trial the aggregator would not sandbox.

    The local backstop to the prefix pin in `tests/test_tool_sandbox.py`: it
    asks the aggregator's own predicate, so the driver cannot start a trial the
    matcher has since stopped matching — which is the exact failure this driver
    had, where the id could move out of the list and nothing said so.

    Deliberately local. `is_sandboxed_session` is a pure function over the id, so
    this reaches no aggregator and no state URL: unlike the bench runner's
    `require_tool_sandbox` (`scripts/autoresearch/bench_runner_sdk.py:247`), which
    asks the running aggregator whether the sandbox is enforced and so cannot run
    where the aggregator is down, this one fails closed on its own.
    """
    from agent_mcp._tool_sandbox import SANDBOXED_ID_PREFIXES, is_sandboxed_session

    if not is_sandboxed_session(session_id):
        raise RuntimeError(
            f"refusing to start a trial with session id {session_id!r}: the "
            f"aggregator sandboxes only ids matching the prefix list "
            f"{list(SANDBOXED_ID_PREFIXES)} (or a recorded `bench` background "
            f"slug), and this driver replays real recorded prompts with the live "
            f"toolbox. Fix SANDBOXED_TRIAL_ID_PREFIX here, or the prefix list in "
            f"agent_mcp/_tool_sandbox.py — but do not run the trial unsandboxed."
        )
    return session_id


async def _one_run(*, keep: int, max_turns: int,
                   followup_max_turns: int = 6, capture: bool = False) -> dict:
    """One trial: a two-turn session with the knob set to `keep`.

    `capture=True` additionally stores each request's message bodies, which
    is what `tests/test_preserve_thinking_eval.py` asserts on.
    """
    import yaml as _yaml
    from app.harness import HookRegistry, RunOptions, install_default_safety_hook
    from app.harness.loop import run_query
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from prompt_builder import build_system_prompt

    config = _yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
    models = config.get("models") or {}
    alias = (config.get("model") or {}).get("default", "primary")
    model_env = (models.get(alias) or {}).get("env") or {}
    base_url = model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096")

    kwargs = _get_harness_kwargs()
    kwargs["preserve_thinking_iterations"] = keep

    chat: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    state = {"turn": 1}
    # Minted once per trial, through the module-level helper, so the id the
    # sandboxing test pins is the id `_options` hands to `RunOptions` and
    # therefore the id `mcp_pool` stamps into `_meta` (#1333).
    session_id = require_sandboxed_trial_session(new_trial_session_id(keep))

    # #1136: this eval registers the real Lloyd MCP servers, so its trial can
    # call a tier-2 sender exactly as a production turn can, and it ran with
    # `hooks=None` until now. One registry for both turns of the trial — the two
    # turns share one chat buffer, so they are one job.
    hooks = HookRegistry()
    install_default_safety_hook(hooks)

    def _options(turn_max: int) -> RunOptions:
        return RunOptions(
            model=alias,
            base_url=base_url,
            system_prompt=build_system_prompt(),
            max_turns=turn_max,
            mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
            disallowed_tools=_get_disallowed_tools(),
            session_id=session_id,
            priority=1,                    # yield to real user traffic
            chat_messages_handle=chat,     # one buffer, two turns
            hooks=hooks,
            **kwargs,
        )

    restore = _install_request_probe(state, requests, capture=capture)
    try:
        # ── turn 1: builds the reasoning mass the window will bound ──
        chat.append({"role": "user", "content": TASK})
        warm = await _run_turn(run_query, _options(max_turns),
                               expect_user_message=TASK)
        # Read BEFORE turn 2: `_cap_history_reasoning` runs inside
        # `run_query` and pops from these same dicts in place.
        prior_chars = sum(len(m.get("reasoning") or "") for m in chat
                          if m.get("role") == "assistant")

        # ── turn 2: the turn-entry cap fires here, and only here ──
        state["turn"] = 2
        chat.append({"role": "user", "content": FOLLOWUP})
        measured = await _run_turn(run_query, _options(followup_max_turns),
                                   expect_user_message=FOLLOWUP)
    finally:
        restore()

    first = next((r for r in requests if r["turn"] == 2),
                 {"reasoning_blocks": 0, "reasoning_chars": 0})
    return {
        "keep": keep,
        "session_id": session_id,
        "warm": warm,
        "measured": measured,
        "seconds": round(warm["seconds"] + measured["seconds"], 1),
        "output_tokens": warm["output_tokens"] + measured["output_tokens"],
        "iterations": warm["iterations"] + measured["iterations"],
        "tool_calls": warm["tool_calls"] + measured["tool_calls"],
        # What turn 1 left behind, what the cap let into turn 2's first
        # request, and therefore what the window took away.
        "prior_reasoning_chars": prior_chars,
        "carried_reasoning_blocks": first["reasoning_blocks"],
        "carried_reasoning_chars": first["reasoning_chars"],
        "window_removed_chars": max(prior_chars - first["reasoning_chars"], 0),
        "measured_seconds": measured["seconds"],
        "measured_peak_input": measured["peak_input_tokens"],
        "answer_chars": measured["answer_chars"],
        "stop_reason": measured["stop_reason"],
        # An arm can look cheap simply by never arriving at an answer, so
        # the completion flag stays next to the cost figures.
        "completed": int(measured["stop_reason"] == "stop"
                         and measured["answer_chars"] > 0),
        "requests": requests,
    }


def _summarize(rows: list[dict]) -> dict:
    def med(k):
        return round(statistics.median([r[k] for r in rows]), 1)
    out = {"runs": len(rows)}
    out.update({k: med(k) for k in METRICS})
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP, help=(
        "ARM window=N: how many of the MOST RECENT reasoning-bearing "
        "assistant messages the TURN-ENTRY window lets through into the NEXT "
        "TURN (`_cap_history_reasoning`, app/harness/loop.py:1561). It is "
        "NOT 'carry at most N iterations of reasoning' — since 2677dea "
        "(#520) nothing is trimmed inside a turn, so this value changes only "
        "what crosses a turn boundary."))
    ap.add_argument("--carry-all", type=int, default=DEFAULT_CARRY_ALL, help=(
        "ARM all: a window so large the turn-entry cap never binds, i.e. "
        "every prior turn's reasoning is carried. This is the arm the "
        "--keep window has to be compared against. Pass 0 to run only the "
        "off/window pair."))
    ap.add_argument("--max-turns", type=int, default=25,
                    help="iteration cap for the warm-up turn (turn 1)")
    ap.add_argument("--followup-max-turns", type=int, default=6,
                    help="iteration cap for the measured turn (turn 2)")
    ap.add_argument("--out", default="")
    return ap


def _print_trial(trial: int, label: str, row: dict) -> None:
    print(f"[trial {trial + 1}] {label:>10} {row['seconds']:>7.1f}s  "
          f"iters={row['iterations']:>3}  tools={row['tool_calls']:>3}  "
          f"out={row['output_tokens']:>6}  "
          f"prior={row['prior_reasoning_chars']:>7}c  "
          f"carried={row['carried_reasoning_blocks']:>2}blk/"
          f"{row['carried_reasoning_chars']:>7}c  "
          f"turn2_peak={row['measured_peak_input']:>6}  "
          f"stop={row['stop_reason']:<10} "
          f"answered={'yes' if row['completed'] else 'NO'}",
          flush=True)


def _print_summary(labels: list[str], summaries: dict[str, dict]) -> None:
    print()
    print("medians over trials; 'carried_*' is what the turn-entry window "
          "let into turn 2's first request")
    print(f"{'metric':24}" + "".join(f"{n:>14}" for n in labels))
    for k in METRICS:
        print(f"{k:24}" + "".join(f"{summaries[n][k]:>14}" for n in labels))
    if len(labels) < 2:
        return
    base_label = labels[0]
    base = summaries[base_label]
    print()
    print(f"{'delta vs ' + base_label:24}" + "".join(f"{n:>14}" for n in labels))
    for k in METRICS:
        print(f"{'  ' + k:24}" + "".join(
            f"{'-' if n == base_label else _delta_cell(summaries[n][k], base[k]):>14}"
            for n in labels))
    print("\n`prior_reasoning_chars` should be flat across the arms — it is "
          "what turn 1 left behind, and turn 1 is the same scripted task. A "
          "gap there is model variance, not an arm effect: the arms then "
          "differ in what they had to carry, and `carried_*` is not "
          "comparable until that is straight.")


def _delta_cell(value: float, base: float) -> str:
    """Percentage change against the same metric in the baseline arm.

    A zero denominator is `n/a`, not `+inf%` or a crash: `answer_chars` and
    `completed` are both legitimately 0 for an arm that never answered, and
    an arm that never answered is exactly the case the table exists to show.
    """
    if not base:
        return "n/a"
    return f"{(value - base) / base * 100:+.1f}%"


async def main() -> int:
    args = build_argparser().parse_args()

    keeps = [0, args.keep]
    if args.carry_all and args.carry_all not in keeps:
        keeps.append(args.carry_all)
    arms = {k: _arm_label(k, args.carry_all) for k in keeps}
    rows: dict[str, list[dict]] = {arms[k]: [] for k in keeps}

    print("arms: " + " | ".join(
        f"{arms[k]} = preserve_thinking_iterations={k}" for k in keeps))
    print(f"session: turn 1 (warm-up, {args.max_turns} iterations max) -> "
          f"turn 2 (measured, {args.followup_max_turns} iterations max) over "
          f"one shared chat_messages buffer")

    for trial in range(args.trials):
        for keep in keeps:                      # interleaved
            row = await _one_run(keep=keep, max_turns=args.max_turns,
                                 followup_max_turns=args.followup_max_turns)
            rows[arms[keep]].append(row)
            _print_trial(trial, arms[keep], row)

    labels = [arms[k] for k in keeps]
    _print_summary(labels, {n: _summarize(rows[n]) for n in labels})

    if args.out:
        payload = {
            "mechanism": ("turn-entry window (post-2677dea / #520); numbers "
                          "published before that commit describe the "
                          "removed per-iteration mechanism"),
            "task": TASK,
            "followup": FOLLOWUP,
            "arms": {arms[k]: [dict(r, requests="<omitted>") for r in rows[arms[k]]]
                     for k in keeps},
            "summary": {n: _summarize(rows[n]) for n in labels},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
