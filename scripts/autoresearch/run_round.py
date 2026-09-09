"""Autoresearch round orchestrator.

CLI:
    python -m scripts.autoresearch.run_round [--targets prompts] [--budget 60] [--dry-run]

MCP:
    autoresearch_round(targets=[...], budget_minutes=N, dry_run=bool) → calls run().
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .bench_runner import run_bench
from .bench_runner_sdk import DEFAULT_PER_TASK_TIMEOUT as SDK_PER_TASK_TIMEOUT
from .bench_runner_sdk import run_bench_sdk
from .common import (
    HARNESSES,
    AutoresearchConfig,
    find_last_promoted_variant,
    ledger_append,
    load_bench_tasks,
    load_config,
    now_iso,
    round_id,
    split_tasks_by_harness,
    validate_run_spec,
    write_run_spec,
    _run_spec_from_cfg,
)
from .hypothesis_generator import propose_variants
from .judge import aggregate_variant, judge_trace
from .promote import evaluate_promotion, promote
from .variant_sandbox import AnchorApplyError, materialize, materialize_baseline

logger = logging.getLogger("autoresearch.run_round")


def _group_traces_by_variant(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for t in traces:
        grouped.setdefault(t["variant_id"], []).append(t)
    return grouped


async def _run_trials(
    cfg: AutoresearchConfig,
    variant_pairs: list[tuple[str, Any]],
    tasks: list[dict[str, Any]],
    model: str,
    harness: str,
    max_parallel: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the (variant × task) matrix, split by harness routing.

    Returns `(direct_traces, sdk_traces)`. The second list is empty for the
    default `--harness direct`, so today's rounds are byte-for-byte unchanged.

    Both runners are awaited sequentially rather than concurrently: they share
    the primary vLLM slot, and the direct runner's cap of 3 exists precisely
    because vLLM does not honor client disconnects (see its docstring).
    Overlapping an agent-loop workload on top of it would push both past their
    timeouts.
    """
    direct_tasks, sdk_tasks = split_tasks_by_harness(tasks, harness)
    direct_traces: list[dict[str, Any]] = []
    sdk_traces: list[dict[str, Any]] = []

    if direct_tasks:
        direct_traces = await run_bench(
            cfg, variant_pairs, direct_tasks, model=model,
            max_parallel=max_parallel,
            per_task_timeout=300,
        )
    if sdk_tasks:
        logger.info("harness=%s: routing %d task(s) through the agent loop "
                    "(%d variants)", harness, len(sdk_tasks), len(variant_pairs))
        sdk_traces = await run_bench_sdk(
            cfg, variant_pairs, sdk_tasks, model=model,
            max_parallel=1,
            per_task_timeout=SDK_PER_TASK_TIMEOUT,
        )
    return direct_traces, sdk_traces


async def run(
    targets: list[str] | None = None,
    budget_minutes: int | None = None,
    max_variants: int | None = None,
    dry_run: bool = False,
    model: str | None = None,
    bench_limit: int | None = None,
    max_parallel: int = 4,
    harness: str = "direct",
) -> dict[str, Any]:
    cfg = load_config()
    cfg.paths.ensure()
    model = model or cfg.default_model

    rid = round_id()
    logger.info("=== autoresearch round %s (dry_run=%s harness=%s) ===", rid, dry_run, harness)

    # 1. Build and write run_spec.yaml
    spec = _run_spec_from_cfg(cfg, model, budget_minutes)
    err = validate_run_spec(spec)
    if err:
        logger.error("run_spec.yaml validation failed: %s", err)
        return {"round_id": rid, "error": f"run_spec validation: {err}"}
    spec["evaluation"]["harness"] = harness
    spec_path = write_run_spec(rid, cfg, spec)

    # Log run_spec to ledger
    ledger_append(cfg.paths.ledger_path, {
        "round_id": rid,
        "event": "spec",
        "spec_path": str(spec_path),
        "harness": harness,
        "writable_paths": spec["mutation_scope"]["writable_paths"],
        "created_at": now_iso(),
    })

    # 2. Look up last promoted variant for parent lineage (Part 2)
    parent_variant = find_last_promoted_variant(cfg.paths.ledger_path, cfg.paths.variants_dir)
    if parent_variant:
        logger.info("found last promoted variant for parent lineage: %s", parent_variant["variant_id"])
    else:
        logger.info("no prior promoted variant found — flat from baseline")

    tasks = load_bench_tasks(cfg.paths.bench_dir)
    if bench_limit:
        tasks = tasks[:bench_limit]
    if not tasks:
        msg = f"bench dir empty: {cfg.paths.bench_dir}"
        logger.error(msg)
        return {"round_id": rid, "error": msg}

    logger.info("loaded %d bench tasks", len(tasks))

    variants = await asyncio.to_thread(
        propose_variants,
        cfg, targets=targets, max_variants=max_variants or cfg.max_variants_per_round, model=model,
        parent_variant=parent_variant,
    )
    if not variants:
        logger.warning("hypothesis generator produced 0 variants — nothing to evaluate this round")

    # Materialize baseline + each variant as overlay dirs
    baseline_id, baseline_dir = materialize_baseline(cfg)
    variant_pairs: list[tuple[str, Path]] = [(baseline_id, baseline_dir)]
    dropped = 0
    for v in variants:
        try:
            overlay = materialize(cfg, v)
        except AnchorApplyError as exc:
            # #446: the parser bounds an edit's shape; only here can an anchor be
            # checked against the text it claims to quote. Zero or two matches
            # drops the variant whole — no fuzzy match, no partial write — and
            # the round goes on with the variants that did apply.
            logger.warning("dropping variant %s: %s", v.get("variant_id"), exc)
            dropped += 1
    if dropped:
        logger.warning("dropped %d of %d variants at anchored-edit apply", dropped, len(variants))

    # Fan out (variant × task), split by harness routing (#353)
    logger.info("running %d variants × %d tasks = %d trials (harness=%s)",
                len(variant_pairs), len(tasks), len(variant_pairs) * len(tasks), harness)
    direct_traces, sdk_traces = await _run_trials(
        cfg, variant_pairs, tasks, model, harness, max_parallel,
    )
    traces = direct_traces + sdk_traces

    # Judge each trace
    scored_traces: list[dict[str, Any]] = []
    for t in traces:
        task_by_id = {tk.get("id"): tk for tk in tasks}
        task = task_by_id.get(t["task_id"]) or {}
        score = judge_trace(task, t, rubric_model=model)
        scored_traces.append({**t, "_task": task, "_score": score})
        ledger_append(cfg.paths.ledger_path, {
            "round_id": rid,
            "variant_id": t["variant_id"],
            "task_id": t["task_id"],
            "task_category": t.get("task_category"),
            "harness": t.get("harness", "direct"),
            "tool_search_enabled": t.get("tool_search_enabled"),
            "trace_status": t["status"],
            "turns": t.get("turns"),
            "tool_call_count": len(t.get("tool_calls", [])),
            "denied_call_count": len(t.get("denied_calls", [])),
            "duration_seconds": t.get("duration_seconds"),
            "composite_score": score["composite_score"],
            "objective_score": score["objective_score"],
            "rubric_overall": score["rubric_overall"],
            "safety_critical": score.get("safety_critical"),
            "safety_passed": score.get("safety_passed"),
            "promoted": None,  # filled in after promotion decision
            "created_at": now_iso(),
        })

    # Aggregate per variant
    by_variant: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for st in scored_traces:
        by_variant.setdefault(st["variant_id"], []).append((st["_task"], st["_score"]))

    summaries: dict[str, dict[str, Any]] = {}
    for vid, pairs in by_variant.items():
        summaries[vid] = aggregate_variant(vid, pairs)

    baseline_summary = summaries.get(baseline_id) or {"mean_composite": 0.0, "per_task": []}

    # Evaluate each candidate vs baseline and pick the winner
    decisions: list[dict[str, Any]] = []
    best_variant: dict[str, Any] | None = None
    best_summary: dict[str, Any] | None = None
    best_overlay: Path | None = None

    for vid, overlay in variant_pairs:
        if vid == baseline_id:
            continue
        vs = summaries.get(vid)
        if not vs:
            continue
        should, reason = evaluate_promotion(cfg, baseline_summary, vs)
        decisions.append({
            "variant_id": vid,
            "mean_composite": vs.get("mean_composite"),
            "should_promote": should,
            "reason": reason,
        })
        if should and (best_summary is None or vs["mean_composite"] > best_summary["mean_composite"]):
            best_variant = next(v for v in variants if v["variant_id"] == vid)
            best_summary = vs
            best_overlay = overlay

    # Second-pass judge + promotion on the winner
    promotion_result: dict[str, Any] | None = None
    if best_variant and best_summary and best_overlay:
        promotion_result = promote(cfg, best_variant, best_overlay, best_summary, baseline_summary, dry_run=dry_run)

    # Write round summary markdown
    summary_file = cfg.paths.rounds_dir / f"{rid}.md"
    sdk_task_ids = sorted({t["task_id"] for t in sdk_traces})
    lines = [
        f"# Autoresearch round {rid}",
        f"- started_at: {now_iso()}",
        f"- model: {model}",
        f"- harness: {harness}",
        f"- tasks: {len(tasks)}",
        f"- tasks on the harness runner: {len(sdk_task_ids)}"
        + (f" ({', '.join(sdk_task_ids)})" if sdk_task_ids else ""),
        f"- variants proposed: {len(variants)}",
        f"- baseline mean composite: {baseline_summary.get('mean_composite', 0.0):.4f}",
        "",
        "## Variant summaries",
    ]
    for vid, summ in summaries.items():
        marker = " (baseline)" if vid == baseline_id else ""
        lines.append(f"- `{vid}`{marker}: mean={summ.get('mean_composite', 0.0):.4f}, "
                     f"safety={'pass' if summ.get('safety_passed') else 'fail'}, tasks={summ.get('task_count', 0)}")
    lines.append("")
    lines.append("## Promotion decisions")
    for d in decisions:
        lines.append(f"- `{d['variant_id']}`: {'PROMOTE' if d['should_promote'] else 'HOLD'} — {d['reason']}")
    if promotion_result:
        lines.append("")
        lines.append("## Promoted")
        lines.append(f"- variant: `{promotion_result['variant_id']}`")
        lines.append(f"- snapshot_dir: `{promotion_result.get('snapshot_dir')}`")
        lines.append(f"- applied_files: {promotion_result.get('applied_files')}")
        lines.append(f"- experiment_fact: `{promotion_result.get('experiment_fact')}`")
    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Patch ledger with final promotion decisions (cheap second pass — append another entry)
    promoted_vid = promotion_result["variant_id"] if promotion_result and not dry_run else None
    for d in decisions:
        ledger_append(cfg.paths.ledger_path, {
            "round_id": rid,
            "event": "decision",
            "variant_id": d["variant_id"],
            "should_promote": d["should_promote"],
            "reason": d["reason"],
            "promoted": promoted_vid == d["variant_id"],
            "created_at": now_iso(),
        })

    return {
        "round_id": rid,
        "summary_file": str(summary_file),
        "spec_file": str(spec_path),
        "variants_proposed": len(variants),
        "tasks_run": len(tasks),
        "baseline_mean": baseline_summary.get("mean_composite", 0.0),
        "decisions": decisions,
        "promoted": promotion_result,
        "parent_variant_id": parent_variant["variant_id"] if parent_variant else None,
        "dry_run": dry_run,
        "harness": harness,
        "tasks_on_harness_runner": len(sdk_task_ids),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one autoresearch round")
    parser.add_argument("--targets", nargs="*", default=["prompts"])
    parser.add_argument("--budget", type=int, default=None, help="Budget minutes (advisory; not a hard kill)")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--bench-limit", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument(
        "--harness", choices=list(HARNESSES), default="direct",
        help="Which bench runner scores the tasks. 'direct' (default) is the "
             "single-turn vLLM completion — unchanged history. 'auto' routes a "
             "task to the in-process agent loop when its frontmatter sets "
             "requires_runtime: true, so PreToolUse hooks and any other runtime "
             "mechanism are actually live for that trial. 'sdk' forces the "
             "agent-loop path for every task.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = asyncio.run(run(
        targets=args.targets,
        budget_minutes=args.budget,
        max_variants=args.max_variants,
        dry_run=args.dry_run,
        model=args.model,
        bench_limit=args.bench_limit,
        max_parallel=args.max_parallel,
        harness=args.harness,
    ))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    # Support: `python -m scripts.autoresearch.run_round` from /home/alansrobotlab/lloyd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
