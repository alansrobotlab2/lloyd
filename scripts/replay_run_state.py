"""Replay a real worker run twice — transcript-shaped and state-shaped (#529).

Backlog #529's acceptance is a measurement, not a code shape: cumulative prompt
tokens per run must drop >=3x, and total prefill seconds per run must be
reported rather than assumed, because the prefix-cache hit rate is expected to
move the WRONG way under this design (rewriting `Σ` every step guarantees a
fresh suffix, and #520 is separately trying to keep the prefix stable). Neither
number is available from history: `usage.input_tokens` on a persisted turn is
the PEAK single prompt, and nothing in the tree records the sum across a turn.
So the comparison has to be run, and it has to be run against the real engine.

What it does
------------
1. Pauses the worker pool over the backend API, so nothing else starts a turn
   while the two arms run. (It cannot pause THIS process's own traffic — see
   the contamination note below.)
2. Reads engine counters around each arm: `prompt_tokens_total` (prefill tokens
   actually processed), `prefix_cache_{hits,queries}_total`, and
   `request_prefill_time_seconds_{sum,count}`.
3. Runs the baseline exactly the way the job class does today: one prompt in,
   `run_query` accumulating one transcript until the model stops (the same
   options builder `run_prompt_on_primary` uses, so the two arms differ in the
   substrate and in nothing else).
4. Runs the same prompt through `app/harness.run_state.run_state_turn`.
5. Prints and dumps both, including the per-iteration prompt-token series — the
   series is the whole argument (flat vs growing), the ratio alone is not.

Contamination, stated because it bounds what the engine numbers mean
-------------------------------------------------------------------
The engine counters are process-global. The agent driving this script is itself
generating on the primary while the replay runs, so the DELTAS are an upper
bound attributable to the arm, not a clean measurement. The per-iteration
prompt tokens come from the harness's own usage events and ARE attributable.
Read the deltas as corroboration; read the per-iteration sums as the result.
`prefill_seconds_estimated` is derived from uncached prompt tokens
(`input_tokens - cache_read`, per iteration) at the engine's measured prefill
throughput, for the same reason.

Both arms run READ-ONLY. `REPLAY_DISALLOWED` strips every mutating tool from
both arms alike, so replaying a job that would normally write a staging note or
a fact cannot leave a side effect behind — and so neither arm gets an
advantage from having tools the other lacks.

Usage (from the worktree, with the live venv):

    ~/lloyd/.venvs/lloyd/bin/python -m scripts.replay_run_state \
        --session 20260909_153812_autonomy_2649 --steps 5 --iterations-per-step 2

    --arm baseline|state|both   --no-pause   --max-turns 15   --keep-going
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.paths import SESSIONS_DIR
from app.harness import run_query
from app.harness import run_state as RS
from app.harness.run_state import RunState, run_state_turn
from workers.sources._common import _worker_run_options

BACKEND = "http://127.0.0.1:8080"
METRICS = "http://127.0.0.1:8096/metrics"

#: Every arm runs without the ability to change anything. Identical for both
#: arms on purpose: a comparison where one side has more tools is not a
#: comparison, and a replay that writes staging notes is not a replay.
REPLAY_DISALLOWED = [
    "Write", "Edit", "NotebookEdit", "Bash",
    "mcp__lloyd-mcp__fact_add", "mcp__lloyd-mcp__remember",
    "mcp__lloyd-mcp__fact_relate", "mcp__lloyd-mcp__memory_add",
    "mcp__lloyd-mcp__vault_write", "mcp__lloyd-mcp__forget",
    "mcp__lloyd-mcp__backlog_write_task", "mcp__lloyd-mcp__research_propose",
    "mcp__lloyd-mcp__autonomy_write_task", "mcp__lloyd-mcp__tasks_create",
    "mcp__lloyd-mcp__calendar_create", "mcp__lloyd-mcp__email_send",
]

# The job class under replay is `session-distill`, whose prompt is copied from
# `workers/sources/session_distill.py::execute` (see --session). The state
# schema below is that job's `Σ`: five things a distill turn actually needs to
# carry between steps, sized to a few hundred tokens. It lives here and not in
# the source module because the production cutover is the decision this
# measurement is supposed to inform, not a decision already taken.
DISTILL_STATE_SCHEMA = {
    "title": "session_distill_run_state",
    "type": "object",
    "properties": {
        "session_path": {"type": "string"},
        "read_progress": {"type": "string"},
        "next_action": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "sections_written": {"type": "boolean"},
    },
    "additionalProperties": False,
}

DISTILL_SKILL = """WORKER JOB: session-distill.

You are analyzing a saved Lloyd session transcript to distill what can be
learned from it. Read the session file with the Read tool, then produce:

## Struggles
## Gaps
## Skill Candidates
## Durable Facts
## Confidence

GUARDRAIL: if the session is trivial (health check, empty, routine maintenance,
fewer than 5 messages, no substantive content), put '- none' in each section and
invent nothing about the session itself. Sessions are never entities.

You are working through a long file in steps. Record in the state what you have
read so far (`read_progress`) and what remains (`next_action`) so the next step
can continue without re-reading. The deliverable is your visible reply on the
step where you set done=true.
"""


# ---------------------------------------------------------------------------
# Engine counters
# ---------------------------------------------------------------------------

_WANTED = {
    "prompt_tokens_total": "vllm:prompt_tokens_total",
    "prefix_cache_hits_total": "vllm:prefix_cache_hits_total",
    "prefix_cache_queries_total": "vllm:prefix_cache_queries_total",
    "prefill_time_sum": "vllm:request_prefill_time_seconds_sum",
    "prefill_time_count": "vllm:request_prefill_time_seconds_count",
    "ttft_sum": "vllm:time_to_first_token_seconds_sum",
    "request_count": "vllm:request_success_total",
}


def read_engine() -> dict[str, float]:
    try:
        body = httpx.get(METRICS, timeout=8.0).text
    except Exception as exc:  # engine not answering is a result, not a crash
        return {"error": str(exc)}
    out: dict[str, float] = {}
    for name, metric in _WANTED.items():
        for line in body.splitlines():
            if line.startswith(metric + " ") or line.startswith(metric + "{"):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
                break
    return out


def engine_delta(before: dict, after: dict) -> dict:
    if "error" in before or "error" in after:
        return {"error": before.get("error") or after.get("error")}
    delta = {k: round(after.get(k, 0.0) - before.get(k, 0.0), 3)
             for k in _WANTED}
    queries = delta.get("prefix_cache_queries_total") or 0.0
    delta["prefix_cache_hit_rate"] = (
        round(delta.get("prefix_cache_hits_total", 0.0) / queries, 4)
        if queries else None)
    n = delta.get("prefill_time_count") or 0.0
    delta["prefill_seconds_measured"] = round(
        delta.get("prefill_time_sum", 0.0), 3)
    delta["requests_observed"] = int(n)
    return delta


async def set_pool_paused(paused: bool) -> str:
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(f"{BACKEND}/api/workers/pause",
                             json={"paused": paused})
            return f"pool paused={r.json().get('paused')} (http {r.status_code})"
    except Exception as exc:
        return f"pause({paused}) failed: {exc}"


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


async def arm_transcript(prompt: str, *, max_turns: int) -> dict:
    """The job class as it runs today: one prompt, one accumulating transcript."""
    options = _worker_run_options(max_turns, extra_disallowed=REPLAY_DISALLOWED)
    messages = [{"role": "user", "content": prompt}]
    series: list[int] = []
    cached = 0
    text = ""
    started = time.perf_counter()
    async for evt in run_query(messages, options):
        if evt["type"] == "assistant_message":
            usage = evt.get("usage") or {}
            series.append(int(usage.get("input_tokens") or 0))
            cached += int(usage.get("cache_read") or 0)
        elif evt["type"] == "text_delta":
            text += evt.get("text", "")
        elif evt["type"] == "result":
            if not text and evt.get("response_text"):
                text = str(evt["response_text"])
            stop = evt.get("stop_reason")
            final_usage = evt.get("usage") or {}
    prompt_tokens = sum(series)
    finalizer_prompt = int(final_usage.get("finalizer_input_tokens") or 0)
    return {
        "arm": "transcript",
        "iterations": len(series),
        "prompt_token_series": series,
        "cumulative_prompt_tokens": prompt_tokens + finalizer_prompt,
        "iteration_prompt_tokens": prompt_tokens,
        "finalizer_prompt_tokens": finalizer_prompt,
        "cumulative_cached_tokens": cached,
        "uncached_prompt_tokens": max(0, prompt_tokens - cached),
        "peak_prompt_tokens": max(series) if series else 0,
        "stop_reason": stop,
        "text_chars": len(text),
        "text": text,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }


async def arm_state(prompt: str, *, session_path: str, run_dir: Path,
                    steps: int, iterations_per_step: int) -> dict:
    state = RunState(
        job="session-distill",
        schema=DISTILL_STATE_SCHEMA,
        max_state_chars=2_500,
        run_dir=run_dir,
        values={"session_path": session_path,
                "next_action": "Read the session file from the beginning."},
    )
    result = await run_state_turn(
        job="session-distill",
        skill_text=DISTILL_SKILL,
        task_block=prompt,
        state=state,
        run_dir=run_dir,
        template=_worker_run_options(
            iterations_per_step, extra_disallowed=REPLAY_DISALLOWED),
        max_steps=steps,
        iterations_per_step=iterations_per_step,
    )
    return {
        "arm": "state",
        "steps": [{"step": s.step, "prompt_tokens": s.prompt_tokens,
                   "finalizer_prompt_tokens": s.finalizer_prompt_tokens,
                   "iterations": s.iterations, "state_chars": s.state_chars,
                   "action": s.action[:120], "done": s.done, "attempts": s.attempts}
                  for s in result.steps],
        "iterations": result.iterations,
        "prompt_token_series": [s.prompt_tokens for s in result.steps],
        "cumulative_prompt_tokens": result.prompt_tokens,
        "cumulative_cached_tokens": result.cached_tokens,
        "uncached_prompt_tokens": result.uncached_prompt_tokens,
        "finalizer_prompt_tokens": result.finalizer_prompt_tokens,
        "prefill_seconds_estimated": result.prefill_seconds,
        "peak_prompt_tokens": max((s.prompt_tokens for s in result.steps), default=0),
        "state_rejections": state.rejections,
        "state_chars_final": state.size_chars(),
        "done": result.done,
        "stop_reason": "stop" if result.done else "max_steps",
        "text_chars": len(result.text),
        "text": result.text,
        "trace": str(run_dir / RS.TRACE_FILENAME),
        "state_file": str(run_dir / RS.STATE_FILENAME),
    }


# ---------------------------------------------------------------------------
# Drift probe
# ---------------------------------------------------------------------------

PROBE_SCHEMA = {
    "title": "drift_probe_run_state",
    "type": "object",
    "properties": {
        "source_path": {"type": "string"},
        "value": {"type": "string"},
        "next_action": {"type": "string"},
    },
    "additionalProperties": False,
}

PROBE_SKILL = """WORKER JOB: drift-probe.

A file at {path} holds one line of the form `counter: <integer>`. On EVERY step,
call Read on that file before you do anything else — even if the state already
carries a value. A value carried in the state is a claim about the past: the
file is the authority, and it may have changed while you worked. Record the
exact current integer in `value`, then say what you are doing next.

Finish (done=true) once you have read the file {steps} times.
"""


async def _wait_for_step(run_dir: Path, want: int, limit: float = 300.0) -> bool:
    """Poll the run's NDJSON until `want` patches have been applied."""
    trace = run_dir / RS.TRACE_FILENAME
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if trace.exists():
            got = sum(1 for line in trace.read_text().splitlines()
                      if '"patch_applied"' in line)
            if got >= want:
                return True
        await asyncio.sleep(0.25)
    return False


async def arm_drift_probe(*, run_dir: Path, steps: int,
                          iterations_per_step: int) -> dict:
    """Mutate a source file mid-run; count steps until the run follows it.

    The paper's number is that a state-carried agent recovers immediately while
    transcript baselines keep acting on the stale in-context fact. The honest
    version of that claim for Lloyd is bounded by what this design actually
    does: the previous step's observation IS replayed as `O_t`, so a run only
    corrects if it re-reads. Which is exactly what the state's staleness rule
    tells it to do, and exactly what this probe measures.
    """
    source = run_dir / "drift_source.txt"
    original = "counter: 7\n# unchanged otherwise\n"
    mutated = "counter: 42\n# changed by the probe between steps 1 and 2\n"
    source.write_text(original)

    probe_run = run_dir / "probe"
    prompt = (f"Read {source} and report the counter. Take {steps} steps.")

    async def mutate_after(first: int) -> None:
        ok = await _wait_for_step(probe_run, first)
        if ok:
            source.write_text(mutated)

    m_target = re.search(r"counter: (\d+)", mutated)
    assert m_target
    wanted = m_target.group(1)
    mutator = asyncio.create_task(mutate_after(1))
    try:
        result = await run_state_turn(
            job="drift-probe",
            skill_text=PROBE_SKILL.format(path=source, steps=steps),
            task_block=prompt,
            state=RunState(job="drift-probe", schema=PROBE_SCHEMA,
                           max_state_chars=1_200, run_dir=probe_run,
                           values={"source_path": str(source)}),
            run_dir=probe_run,
            template=_worker_run_options(
                iterations_per_step, extra_disallowed=REPLAY_DISALLOWED),
            max_steps=steps,
            iterations_per_step=iterations_per_step,
        )
    finally:
        mutator.cancel()

    trace = [json.loads(l) for l in (probe_run / RS.TRACE_FILENAME).read_text().splitlines()
             if l.strip()]
    applied = [r for r in trace if r.get("kind") == "patch_applied"]
    mutated_at = 1  # by construction: after the 1st applied patch
    corrected_at = None
    for r in applied:
        if str((r.get("state_patch") or {}).get("value")) == wanted:
            corrected_at = r["step"]
            break
    return {
        "arm": "drift_probe",
        "mutated_after_step": mutated_at,
        "corrected_at_step": corrected_at,
        "steps_to_correct": (None if corrected_at is None
                             else corrected_at - mutated_at),
        "wanted_value": wanted,
        "values_seen": [str((r.get("state_patch") or {}).get("value"))
                        for r in applied],
        "trace_rejections": sum(1 for r in trace
                                if r.get("kind") == "patch_rejected"),
        "cumulative_prompt_tokens": result.prompt_tokens,
        "steps_run": len(result.steps),
        "done": result.done,
    }


# ---------------------------------------------------------------------------


def resolve_session(name: str) -> Path:
    """A stem in the sessions dir, or an absolute path.

    The absolute form exists because `app.paths` anchors `LLOYD_HOME` to the
    tree it was imported from, so a replay run out of a worktree sees the
    worktree's (empty) sessions dir while the transcripts it is meant to replay
    live under the live checkout.
    """
    direct = Path(name)
    if direct.is_absolute():
        if direct.exists():
            return direct
        raise SystemExit(f"no session file at {direct}")
    for p in sorted(SESSIONS_DIR.glob(f"{name}*.json")):
        return p
    raise SystemExit(f"no session file matching {name!r} in {SESSIONS_DIR}")


def distill_prompt(path: Path) -> str:
    """The production prompt, copied from session_distill.execute()."""
    return (
        f"You are analyzing a saved Lloyd session transcript to distill what we can "
        f"learn from it. The session file is at: {path}\n\n"
        f"Read the file using the Read tool, then identify:\n"
        f"1. Any repeated user struggles or failed assistant attempts\n"
        f"2. Knowledge gaps — questions Lloyd couldn't confidently answer\n"
        f"3. Patterns that could become a skill\n"
        f"4. Durable facts about the user that should be captured\n\n"
        f"Return in this structure:\n"
        f"## Struggles\n- ...\n\n## Gaps\n- ...\n\n## Skill Candidates\n- ...\n\n"
        f"## Durable Facts\n- ...\n\n## Confidence\n<0.0-1.0>: <justification>\n"
    )


def _ratio(before, after):
    if not before or not after:
        return None
    return round(before / after, 2)


def _stats(series: list[int]) -> dict:
    if not series:
        return {}
    return {"min": min(series), "median": statistics.median(series),
            "max": max(series), "n": len(series)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="", help="session file stem to replay")
    ap.add_argument("--arm", default="both", choices=["baseline", "state", "both", "probe"])
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--iterations-per-step", type=int, default=2)
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--out", default="", help="where to write the JSON report")
    args = ap.parse_args()

    session_path = resolve_session(args.session) if args.session else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Under /tmp, not the tree: a replay's transcripts are measurement output,
    # and `.gitignore` is not a path a self-modification round may edit to make
    # them stop showing up in `git status`.
    run_root = Path("/tmp/lloyd-replay-run-state") / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "at": stamp,
        "session": str(session_path),
        "session_bytes": session_path.stat().st_size if session_path else 0,
        "max_turns": args.max_turns,
        "steps": args.steps,
        "iterations_per_step": args.iterations_per_step,
        "disallowed_tools": REPLAY_DISALLOWED,
        "arms": {},
    }

    if args.arm in ("baseline", "state", "both") and session_path is None:
        ap.error("--session is required for the token arms")

    pause_note = ""
    if not args.no_pause:
        pause_note = await set_pool_paused(True)
        print(pause_note)
    try:
        if args.arm in ("baseline", "both"):
            before = read_engine()
            res = await arm_transcript(
                distill_prompt(session_path), max_turns=args.max_turns)
            res["engine"] = engine_delta(before, read_engine())
            report["arms"]["baseline"] = res
            _print_arm(res)
            _dump(run_root / "baseline.json", res)

        if args.arm in ("state", "both"):
            before = read_engine()
            res = await arm_state(
                distill_prompt(session_path), session_path=str(session_path),
                run_dir=run_root / "state", steps=args.steps,
                iterations_per_step=args.iterations_per_step)
            res["engine"] = engine_delta(before, read_engine())
            report["arms"]["state"] = res
            _print_arm(res)
            _dump(run_root / "state.json", res)

        if args.arm in ("probe", "both"):
            before = read_engine()
            res = await arm_drift_probe(
                run_dir=run_root, steps=max(3, args.steps),
                iterations_per_step=args.iterations_per_step)
            res["engine"] = engine_delta(before, read_engine())
            report["arms"]["drift_probe"] = res
            print(json.dumps(res, indent=2)[:2000])
            _dump(run_root / "probe.json", res)
    finally:
        if not args.no_pause:
            print(await set_pool_paused(False))

    b = report["arms"].get("baseline") or {}
    s = report["arms"].get("state") or {}
    verdict = {
        "cumulative_prompt_tokens": {
            "baseline": b.get("cumulative_prompt_tokens"),
            "state": s.get("cumulative_prompt_tokens"),
            "ratio": _ratio(b.get("cumulative_prompt_tokens"),
                            s.get("cumulative_prompt_tokens")),
        },
        "uncached_prompt_tokens": {
            "baseline": b.get("uncached_prompt_tokens"),
            "state": s.get("uncached_prompt_tokens"),
            "ratio": _ratio(b.get("uncached_prompt_tokens"),
                            s.get("uncached_prompt_tokens")),
        },
        "prefill_seconds_measured_engine": {
            "baseline": (b.get("engine") or {}).get("prefill_seconds_measured"),
            "state": (s.get("engine") or {}).get("prefill_seconds_measured"),
        },
        "prefill_seconds_estimated": {
            "baseline": None,
            "state": s.get("prefill_seconds_estimated"),
        },
        "prefix_cache_hit_rate": {
            "baseline": (b.get("engine") or {}).get("prefix_cache_hit_rate"),
            "state": (s.get("engine") or {}).get("prefix_cache_hit_rate"),
        },
        "iterations": {"baseline": b.get("iterations"), "state": s.get("iterations")},
        "meets_3x_on_cumulative_prompt_tokens": (
            (_ratio(b.get("cumulative_prompt_tokens"),
                    s.get("cumulative_prompt_tokens")) or 0) >= 3.0),
    }
    if "drift_probe" in report["arms"]:
        verdict["drift_probe"] = {
            "steps_to_correct": report["arms"]["drift_probe"]["steps_to_correct"],
            "values_seen": report["arms"]["drift_probe"]["values_seen"],
            "meets_1_step": (
                report["arms"]["drift_probe"]["steps_to_correct"] == 1),
        }
    report["verdict"] = verdict
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    out = Path(args.out) if args.out else (run_root / "report.json")
    _dump(out, report)
    print(f"\nreport: {out}")
    return 0


def _print_arm(res: dict) -> None:
    print(f"\n--- {res['arm']} ---")
    print(f"  cumulative prompt tokens : {res['cumulative_prompt_tokens']:,}")
    print(f"  of which cached          : {res['cumulative_cached_tokens']:,}")
    print(f"  finalizer prompt tokens  : {res['finalizer_prompt_tokens']:,}")
    print(f"  iterations/steps         : {res.get('iterations')} / "
          f"{len(res.get('steps', [])) or 'n/a'}")
    print(f"  per-step prompt series   : {res['prompt_token_series']}")
    print(f"  peak prompt              : {res.get('peak_prompt_tokens'):,}")
    print(f"  stop_reason              : {res.get('stop_reason')}")
    print(f"  reply chars              : {res.get('text_chars')}")
    if res.get("steps"):
        for st in res["steps"]:
            print(f"    step {st['step']}: {st['iterations']} it, "
                  f"{st['prompt_tokens']:,} prompt, "
                  f"{st['finalizer_prompt_tokens']:,} finalizer, "
                  f"state {st['state_chars']}, done={st['done']}")
    eng = res.get("engine") or {}
    if eng:
        print(f"  engine: prefill={eng.get('prefill_seconds_measured')}s "
              f"prompt_tokens={eng.get('prompt_tokens_total'):,.0f} "
              f"hit_rate={eng.get('prefix_cache_hit_rate')} "
              f"requests={eng.get('requests_observed')}")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
