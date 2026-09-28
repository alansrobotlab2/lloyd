"""#1659 — produce the whole-run traces the behavioural suite scores.

`scripts/autoresearch/behavioural.py` could read traces since #1549 but nothing
in the repository could write one, which made the whole report-only rung a
tautology: the four rounds run on 2026-09-27 each scored the shipped reference
capture against a baseline pinned from that same capture, so all four artifacts
read `trace_source: "reference"`, `delta: 0.0` on all four axes and
`guardrail_hit: false`. An instrument that cannot move cannot catch a regression
or clear a promotion, so the 10-10 decision on wiring it into promotion had no
record to be made on. This module is the missing writer.

Why it is a separate module that a round never calls: #1546. An autoresearch
round body dies at the pool cap, and a scenario suite inside it dies with it —
`round_scorecard` only ever *scores* a capture somebody produced out of band.
Producing traces means real engine runs, so this entry point is a deliberate act
on an idle primary (`--yes` is required), which is exactly what the N=5 retro
re-score is (#1659's owed step 1, in a paused-pool window).

The seam that makes the bookkeeping testable: `capture_round` takes a `runner`
callable that turns one scenario into a whole-run trace dict. The only
production runner is `sdk_runner`, which drives the real harness through
`bench_runner_sdk.run_trial`; a test injects a fake and sees the same paths, the
same budget arithmetic, the same statuses and the same capture record, with no
engine and no GPU. Everything this module is for — where files land, what a
scenario that never ran is *called* — is therefore measured without a model.

The bound: `DEFAULT_CAPTURE_BUDGET_SECONDS` is 600 s of wall clock per capture
round, the ruled "cap scenarios at 10 min GPU a round". It is measured as elapsed
wall clock because that is the only thing the capturer can observe while it
runs; the true GPU cost of a capture round is #1659's owed item 3 and will be
read off the first live capture rather than asserted here. The budget is checked
at scenario boundaries, and a scenario already in flight is allowed to finish —
aborting a live engine run mid-call is a worse instrument than one that starts
two scenarios instead of three.

Reserve note: a capture runs every scenario, including the ones in this month's
reserve. The hold-out's teeth are in the proposer never being shown a reserved
scenario's text (`hypothesis_generator` builds its prompt from the canonical
prompt surfaces plus the ledger, so it never sees this directory at all), not in
leaving it unmeasured — an unmeasured reserve is a missing axis value, which is
exactly the instrument failure clause 2 exists to report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.autoresearch import behavioural  # noqa: E402

#: The ruled 10 minutes. Per capture round, wall clock, not per scenario.
DEFAULT_CAPTURE_BUDGET_SECONDS = 600

# Statuses, and what each one obliges the scorecard to do with the scenario.
# Anything but CAPTURED leaves the scenario unscored, and the reason recorded
# here is what its scorecard row then quotes.
CAPTURED = "captured"
SKIPPED_BUDGET = "skipped_budget"
FAILED = "failed"
INVALID = "invalid"

#: The lists a grader can read a trace off. `behavioural`'s five graders read
#: exactly these, so an object carrying none of them is a trace in name only: it
#: would write a file that looks captured and scores as nothing.
OBSERVABLE_KEYS = ("turn_log", "durable_writes", "answers", "tool_calls", "events")


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_dir(cfg: Any, run_id: str) -> Path:
    """Where a capture's traces and its own record go: the path round_scorecard reads."""
    return Path(cfg.paths.research_root) / "behavioural_traces" / run_id


def default_plant_root(cfg: Any, run_id: str) -> Path:
    """Where planted state goes when the operator names no vault: the research root.

    A bench trial cannot write to the vault — `vault_write` is on the trial deny
    list — and the harness reads whatever vault it is configured with, so a
    behavioural probe that wants its planted file *seen* has to have it planted
    where the harness reads, which is a real write into a live tree. The default
    therefore plants into `behavioural_plant/<run_id>/` under the research root:
    the pipeline runs end to end and the record says plainly that the scenario
    state was planted somewhere the model was not looking. Point `--plant-into`
    at a vault for a measurement that reproduces the named failure.
    """
    return Path(cfg.paths.research_root) / "behavioural_plant" / run_id


def safe_scenario_filename(scenario: dict[str, Any]) -> str:
    """The scenario id as a filename component, or a refusal.

    Both the trace file and the planted-state file are named after the id, and a
    manifest is editable input to a job that writes files. An id of
    `../../.ssh/authorized_keys` would put a "trace" anywhere the process could
    write, so an id has to be a bare name — checked, not assumed, because no
    other reader of the manifest constrains it either.
    """
    sid = str(scenario.get("id") or "")
    if not sid or sid != Path(sid).name:
        raise ValueError(
            f"scenario id {sid!r} is not usable as a filename: the trace and the "
            f"planted-state file are named after it, so an id carrying a path "
            f"separator would write outside the capture directory")
    return sid


def plant_input(cfg: Any, scenario: dict[str, Any], *,
                plant_root: Path) -> str:
    """Write the scenario's declared pre-existing state; return where it landed.

    `planted_input` is what makes these scenarios whole-run rather than
    single-turn: the run has to discover the pre-existing state out of its own
    context, which a canned prompt cannot simulate. The frozen manifest declares
    that state as structured fields — `kind`, `entity`, `text`, `hedge_token`,
    `superseded_value` — and declares no path, so the capturer names the file
    after the scenario and writes the declared fields verbatim. `then_run` stays
    out of the file: it is the instruction for the run, not state the run has to
    discover, and a file that contained it would hand the run its own script.

    The path is relative to `plant_root`, which is what the run is told to read
    (`sdk_runner` puts it in the prompt), so a caller comparing what was planted
    against what was declared compares like for like.
    """
    planted = scenario.get("planted_input")
    if not isinstance(planted, dict) or not planted:
        raise ValueError(f"{scenario.get('id')}: scenario declares no planted_input")
    state = {k: v for k, v in planted.items() if k != "then_run"}
    if not state:
        raise ValueError(
            f"{scenario.get('id')}: planted_input declares only `then_run`, so there "
            f"is no pre-existing state to plant and nothing for the run to discover")
    name = safe_scenario_filename(scenario)
    base = Path(plant_root).resolve()
    target = (base / f"{name}.yaml").resolve()
    if not target.is_relative_to(base):
        raise ValueError(
            f"{scenario['id']}: planted state resolves outside the plant root "
            f"({base}); a scenario measures behaviour, it does not write anywhere else")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"# Planted pre-existing state for behavioural scenario '{name}' (#1659).\n"
        "# Written verbatim from the frozen manifest's planted_input. `then_run` is the\n"
        "# run's instruction and is deliberately not in this file.\n"
        + yaml.safe_dump(state, sort_keys=True, allow_unicode=True, width=100),
        encoding="utf-8")
    return f"{name}.yaml"


def _trace_problem(trace: Any, scenario_id: str) -> str | None:
    """A trace has to be the shape `load_traces` and the graders expect.

    Concretely: at least one of the five observable lists the graders read
    (`behavioural.GRADERS` reads exactly these) has to carry a row. A runner that
    hands back a bench-result object — `{"status": "success", "final_text": …}` —
    is a non-empty dict and would write a file that *reads* as a captured run
    while every grader on it reports `ran: 0`, which is the ambiguity clause 2
    exists to remove. The scenario keeps an explicit cause instead.
    """
    if not isinstance(trace, dict) or not any(
            isinstance(trace.get(key), list) and trace.get(key)
            for key in OBSERVABLE_KEYS):
        return (f"the runner returned no observable list to grade "
                f"({'/'.join(OBSERVABLE_KEYS)}), so no trace was written")
    # `load_traces` keys a file by the `scenario_id` INSIDE it, not by its name,
    # so a runner that returns one scenario's run under another's name would be
    # filed under the wrong scenario and graded against the wrong expectation —
    # a silent mis-scoring, which is worse than the missing file this guard
    # already refuses. A trace with no id written is still a trace (the file name
    # carries it through `load_traces`' fallback); a wrong one never is.
    declared = trace.get("scenario_id")
    if declared is not None and str(declared) != str(scenario_id):
        return (f"the trace the runner returned for this scenario declares "
                f"scenario_id={str(declared)!r}, which `load_traces` would file "
                f"under the other scenario and grade against the wrong expectation")
    return None


def capture_round(*, cfg: Any, run_id: str, runner: Callable[[dict[str, Any]], dict],
                  manifest: dict[str, Any] | None = None,
                  plant_root: Path | None = None,
                  budget_seconds: float = DEFAULT_CAPTURE_BUDGET_SECONDS,
                  clock: Callable[[], float] = time.monotonic,
                  engine: str = "injected") -> dict[str, Any]:
    """Run every scenario once, write its trace, and account for the rest.

    Returns the capture record — the same dict written to `capture.yaml` — with
    one row per scenario: `status` in captured / skipped_budget / failed /
    invalid, `seconds` of wall clock it took, and a `reason` for anything that
    did not produce a trace. That record is the only witness to a scenario that
    never ran, and `build_scorecard` quotes it, so the scorecard can tell
    "budget exhausted before this scenario started" apart from "this scenario
    scored nothing" instead of silently scoring the gap as a zero.

    A scenario whose capture raised is `failed` and the run continues: losing one
    trace is an instrument failure on one axis, while aborting the capture loses
    the whole round's measurement and hides which scenario was the problem.
    """
    manifest = manifest or behavioural.load_manifest()
    out = capture_dir(cfg, run_id)
    out.mkdir(parents=True, exist_ok=True)
    plant_root = Path(plant_root) if plant_root is not None else default_plant_root(cfg, run_id)
    started = clock()
    rows: list[dict[str, Any]] = []

    for scenario in manifest["scenarios"]:
        sid = str(scenario["id"])
        elapsed = clock() - started
        if elapsed >= budget_seconds:
            rows.append({"id": sid, "axis": scenario.get("axis"),
                         "checker": scenario.get("checker"),
                         "status": SKIPPED_BUDGET, "seconds": 0.0,
                         "trace": None,
                         "reason": f"capture budget of {budget_seconds:g} s exhausted "
                                   f"({elapsed:.1f} s elapsed) before this scenario "
                                   f"was started"})
            continue
        row: dict[str, Any] = {"id": sid, "axis": scenario.get("axis"),
                              "checker": scenario.get("checker"),
                              "status": CAPTURED, "trace": f"{sid}.yaml"}
        began = clock()
        try:
            name = safe_scenario_filename(scenario)
            planted = plant_input(cfg, scenario, plant_root=plant_root)
            row["planted"] = planted
            # The runner gets the path with the scenario: a run that is meant to
            # *discover* planted state has to be pointed at the file, and
            # `sdk_runner` is the one caller that cannot go looking for it.
            trace = runner({**scenario, "_planted": planted})
            problem = _trace_problem(trace, sid)
            if problem is not None:
                row["status"] = INVALID
                row["trace"] = None
                row["reason"] = problem
            else:
                (out / f"{name}.yaml").write_text(
                    yaml.safe_dump(trace, sort_keys=True, width=100), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — one scenario must not end the capture
            row["status"] = FAILED
            row["trace"] = None
            row["reason"] = f"{type(exc).__name__}: {exc}"
            row["traceback_tail"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
        row["seconds"] = round(clock() - began, 3)
        rows.append(row)

    elapsed = round(clock() - started, 3)
    meta = {
        "schema": behavioural.CAPTURE_SCHEMA,
        "run_id": run_id,
        "generated_at": _stamp(),
        "engine": engine,
        "budget_seconds": budget_seconds,
        "elapsed_seconds": elapsed,
        "budget_exhausted": elapsed >= budget_seconds,
        "suite": manifest.get("suite"),
        "scenarios_hash": manifest.get("_scenarios_hash"),
        "scenarios": rows,
        "label": (f"{sum(1 for r in rows if r['status'] == CAPTURED)} of {len(rows)} "
                  f"scenarios captured in {elapsed:.1f} s of a "
                  f"{budget_seconds:g} s budget"),
    }
    (out / behavioural.CAPTURE_META_FILENAME).write_text(
        yaml.safe_dump(meta, sort_keys=True, width=100), encoding="utf-8")
    return meta


# ─────────────── one bench trial → one whole-run trace (#1659) ─────────────
#
# `bench_runner_sdk.run_trial` answers with a TRIAL RECORD: `status`,
# `final_text`, `turns`, `tool_calls`, `denied_calls`, `unresolved_calls`, usage
# and timings (`bench_runner_sdk.py:587-626`). `behavioural`'s graders read four
# OBSERVABLE lists — `durable_writes`, `answers`, `tool_calls`, `events` — and
# `load_traces` keys every file by the `scenario_id` inside it, which a trial
# record does not carry at all. So handing a trial result straight to
# `capture_round` wrote a file that made the entire scorecard refuse:
# `load_traces` raises on a trace with no `scenario_id`, and `round_scorecard`
# turns that into `status: refused`. The mapper below is therefore not a nicety:
# without it the production runner's output is unreadable by its own scorer.

#: Tools whose SUCCESSFUL call is a durable write, and which two arguments carry
#: the location and the prose. `behavioural.durable_rows` reads only `path` and
#: `text`, so this table is the single place that knows how each of Lloyd's
#: writers spells those two things.
DURABLE_WRITE_ARGS: dict[str, tuple[str, str]] = {
    "vault_write": ("path", "content"),
    "Write": ("file_path", "content"),
    "Edit": ("file_path", "new_string"),
    "memory_add": ("file", "entry"),
    "memory_replace": ("file", "new_text"),
}


def durable_write_row(call: dict[str, Any]) -> dict[str, str] | None:
    """One successful write-shaped call as a `durable_writes` row, or None.

    A call the harness denied, or that came back `is_error`, wrote nothing. A
    grader that reads durable writes is asking what the run put on disk, so
    filing a refused write as one would hand the run credit for a record it
    never made — the exact false positive this suite exists to make impossible.
    """
    if call.get("denied") or call.get("is_error"):
        return None
    spec = DURABLE_WRITE_ARGS.get(str(call.get("name") or ""))
    if spec is None:
        return None
    args = call.get("args")
    if not isinstance(args, dict):
        return None
    path, text = str(args.get(spec[0]) or ""), str(args.get(spec[1]) or "")
    if not path or not text:
        return None
    return {"path": path, "text": text}


def trace_from_trial(trial: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """A trial record as the whole-run trace the graders read.

    `scenario_id` comes from the SCENARIO, never from the trial: the trial knows
    itself as `task_id="behavioural:<id>"`, and `load_traces` files a trace under
    whatever `scenario_id` the file carries, so copying the trial's own naming
    would grade one scenario against another's expectation.

    What is mapped, and what is deliberately not:

    * `tool_calls` — every call the trial made, including the ones the trial
      harness denied and the ones it never answered. The action-consistency axis
      asks whether the planted root reached the ARGUMENTS, and a denied call
      answers that with the arguments the model actually chose. The denial itself
      goes to `events` as `tool_denied`, so the fate of the call is readable
      beside the row it produced.
    * `answers` — the trial's `final_text` when it said anything. The harness
      keeps only the terminal answer, so that is the one row.
    * `durable_writes` — only from successful write-shaped calls, through
      `durable_write_row`. The bench harness denies the mutating tools, so a
      capture through `sdk_runner` measures no durable write, and the
      `uncertainty_preservation` axis reports an instrument failure instead of a
      score. Pointing a capture at a writable vault is the re-score's decision
      (#1659's owed item 1), not something this mapper can fake by trusting a
      refused call.
    * `events` — only what the trial observed: denials, unanswered calls, a
      timed-out or errored run. `route_blocked` and `plan_revised` are NOT
      invented here, because the harness emits no such pair; the scenario that
      reads them (`blocked-route-replan`) therefore scores `ran: 0` and is
      reported as the instrument failure clause 2 describes — which is the honest
      outcome, and the one that tells the re-score which event vocabulary a
      runner has to supply.

    `at_step` on a generated event is the call's position within the trial's own
    list of that kind (`tool_calls` / `denied_calls` / `unresolved_calls`), not a
    global turn index: the trial record keeps the three lists separately and
    interleaving them would be a guess about ordering it does not record.

    Raises ValueError when the trial produced neither a call nor an answer: that
    is a run that never happened, and `capture_round` records it as a failed
    scenario with this message as the cause, rather than writing an empty trace
    and reporting the scenario as captured.
    """
    sid = str(scenario["id"])
    status = str(trial.get("status") or "")
    calls = [c for c in (trial.get("tool_calls") or []) if isinstance(c, dict)]
    denied = [c for c in (trial.get("denied_calls") or []) if isinstance(c, dict)]
    unresolved = [c for c in (trial.get("unresolved_calls") or []) if isinstance(c, dict)]

    events: list[dict[str, Any]] = []
    for step, call in enumerate(denied):
        events.append({
            "kind": "tool_denied", "at_step": step,
            "detail": f"{call.get('name')} denied: "
                      f"{call.get('deny_reason') or call.get('deny_kind') or 'no reason recorded'}"})
    for step, call in enumerate(unresolved):
        events.append({"kind": "tool_unresolved", "at_step": step,
                       "detail": str(call.get("reason") or "no tool_result recorded")})
    if status and status != "success":
        events.append({"kind": f"run_{status}",
                       "at_step": int(trial.get("turns") or 0),
                       "detail": str(trial.get("error") or status)})

    final_text = str(trial.get("final_text") or "").strip()
    trace = {
        "scenario_id": sid,
        "captured_by": (f"bench trial ({trial.get('harness') or 'harness'},"
                        f" {trial.get('trial_id') or trial.get('task_id') or '?'},"
                        f" status={status or '?'})"),
        "durable_writes": [r for r in (durable_write_row(c) for c in calls) if r],
        "answers": [final_text] if final_text else [],
        "tool_calls": [{"name": str(c.get("name") or ""),
                        "args": (c.get("args") if isinstance(c.get("args"), dict) else {})}
                       for c in calls + denied + unresolved],
        "events": events,
        "trial": {key: trial.get(key) for key in
                  ("status", "turns", "duration_seconds", "stop_reason", "error",
                   "variant_id", "task_id", "session_id") if trial.get(key) is not None},
    }
    if not trace["tool_calls"] and not trace["answers"]:
        raise ValueError(
            f"trial for {sid} produced no tool call and no final answer "
            f"(status={status or '?'}, error={trial.get('error') or 'none'}), so "
            f"there is no whole-run trace to grade")
    return trace


# ─────────────────────── the production runner ───────────────────────────

def sdk_runner(cfg: Any, *, model: str | None = None,
               variant_id: str = "behavioural_capture",
               per_task_timeout: int = 300,
               max_agent_turns: int = 12) -> Callable[[dict[str, Any]], dict]:
    """The one runner that spends GPU: one whole agentic run per scenario.

    Builds a baseline overlay (the canonical prompt, no candidate edits — this
    measures behaviour, not a proposal) and hands each scenario's `then_run` to
    `bench_runner_sdk.run_trial`, which is the same harness path a bench trial
    takes. The scenario's `planted_input` is on disk before the call — see
    `plant_input` — so the run has to discover it the way a real session would.

    Each result goes through `trace_from_trial` before it is written, because a
    trial record is not a trace: it carries no `scenario_id` (so `load_traces`
    would refuse the whole directory), and it names its observations
    `final_text`/`denied_calls` rather than `answers`/`tool_calls`. Through this
    runner a capture measures the three scenarios whose evidence is an argument
    list or an answer — `act-on-known-fact`, `source-retention`,
    `stale-fact-action` — and reports the other two as instrument failures,
    because the trial harness denies the mutating tools
    (`uncertainty-hardening` needs a durable write) and emits no
    `route_blocked`/`plan_revised` pair (`blocked-route-replan` needs that event
    vocabulary). Both of those are decisions for the re-score — a writable vault
    to point `--plant-into` at, and an event source for the replan probe — and an
    instrument failure that says which is what clause 2 is for.

    What no test here can price is the run itself: `run_trial` on the real
    harness means GPU on an otherwise-idle primary in a paused-pool window, and
    the true wall-clock cost against the 600 s bound is #1659's owed item 3.
    """
    import asyncio

    from scripts.autoresearch import bench_runner_sdk, variant_sandbox

    baseline_id, overlay_dir = variant_sandbox.materialize_baseline(cfg)
    use_model = model or cfg.default_model

    def run(scenario: dict[str, Any]) -> dict[str, Any]:
        planted = scenario.get("_planted") or "(planted state not recorded)"
        # `then_run` is the scenario's own instruction and lives only in
        # planted_input — `plant_input` deliberately keeps it out of the file on
        # disk — so this is where the run is told what it is being asked to do,
        # along with where the state it has to discover was put.
        prompt = (f"Pre-existing state relevant to this session has been planted at "
                  f"`{planted}`; read it before you act.\n\n"
                  f"{scenario['planted_input']['then_run']}")
        task = {"id": f"behavioural:{scenario['id']}", "prompt": prompt,
                "category": "behavioural", "target_tools": [], "max_turns": max_agent_turns}
        trial = asyncio.run(bench_runner_sdk.run_trial(
            task, variant_id or baseline_id, overlay_dir, use_model,
            per_task_timeout=per_task_timeout, max_agent_turns=max_agent_turns))
        # The map is the difference between a directory the scorer can read and a
        # `status: refused` scorecard, so it happens here rather than in the
        # caller: see `trace_from_trial`.
        return trace_from_trial(trial, scenario)

    return run


def main(argv: list[str] | None = None) -> None:
    from scripts.autoresearch import run_round as _run_round

    ap = argparse.ArgumentParser(
        prog="behavioural_capture",
        description="Run the behavioural scenarios for real and write the traces "
                    "autoresearch's report-only rung scores (#1659).")
    ap.add_argument("--run-id", required=True,
                    help="The round or capture id the traces are filed under; "
                         "`round_scorecard` reads behavioural_traces/<run-id>/")
    ap.add_argument("--budget-seconds", type=float,
                    default=DEFAULT_CAPTURE_BUDGET_SECONDS,
                    help="Wall clock for the whole capture (ruled: 600 = 10 min)")
    ap.add_argument("--model", default=None, help="Defaults to autoresearch.default_model")
    ap.add_argument("--engine", default="claude-sdk")
    ap.add_argument("--yes", action="store_true",
                    help="Required: this runs real engine turns on the primary")
    args = ap.parse_args(argv)

    if not args.yes:
        ap.error("a capture spends real engine turns on the primary; pass --yes "
                 "explicitly (ruled: N=5 retro re-score in a paused-pool window)")
    cfg = _run_round.build_cfg()
    meta = capture_round(cfg=cfg, run_id=args.run_id,
                         runner=sdk_runner(cfg, model=args.model),
                         budget_seconds=args.budget_seconds, engine=args.engine)
    print(yaml.safe_dump(meta, sort_keys=True, width=100))
    unscored = [r for r in meta["scenarios"] if r["status"] != CAPTURED]
    if unscored:
        listed = ", ".join(f"{r['id']}={r['status']}" for r in unscored)
        print(f"# {len(unscored)} scenario(s) produced no trace and will report as "
              f"instrument failures: {listed}", file=sys.stderr)


if __name__ == "__main__":
    main()
