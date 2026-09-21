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

from app.harness.bench_corpus import probe_ledger_fields

from . import auto_restore, post_promotion
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
from . import bench_split
# `slice_metrics` is imported by name, never reached through the module: the
# name `promote` is bound two lines lower to the promotion *function*, so
# `promote.slice_metrics(...)` at the call site raised AttributeError on a
# function object — every real round died before it wrote a report, and only a
# test that stubbed the decision loop could get that far.
from .promote import evaluate_promotion, promote, slice_metrics
from .variant_sandbox import AnchorApplyError, materialize, materialize_baseline

logger = logging.getLogger("autoresearch.run_round")

#: `post_promotion_check`'s "look it up yourself" default. A sentinel object rather
#: than `None`, because `None` is a real comparison value here — it means "there was
#: no promotion on record to compare against" and has its own report branch, so a
#: `None` default could not tell "nothing to compare" from "caller did not say".
_COMPUTE = object()


def _group_traces_by_variant(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for t in traces:
        grouped.setdefault(t["variant_id"], []).append(t)
    return grouped


def post_promotion_comparison(
    cfg: AutoresearchConfig, round_id: str, baseline_mean: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Look up the previous promotion and compare this round's baseline against it.

    Split out of :func:`post_promotion_check` because #1099 needs the verdict before
    anything is written: the lookup must precede this round's own `round_summary` row
    (a round must never be compared against itself, so `last_promotion` excludes
    `round_id` and the exclusion is worthless once the row exists), and the restore has
    to happen in the same window — after the comparison, before the record. Returns
    `(prior, comparison)`; `prior` is what tells the restore which snapshot and which
    vault commit it is undoing.
    """
    prior = post_promotion.last_promotion(
        cfg.paths.ledger_path, cfg.paths.rounds_dir, exclude_round=round_id,
    )
    return prior, post_promotion.compare(baseline_mean, prior, cfg.promotion_noise_floor)


def post_promotion_check(
    cfg: AutoresearchConfig,
    round_id: str,
    baseline_mean: float,
    promotion_result: dict[str, Any] | None,
    variant_summary: dict[str, Any] | None = None,
    candidate_overlay: Path | None = None,
    comparison: Any = _COMPUTE,
    restore: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, Any] | None]:
    """Compare this round's fresh baseline against the last promotion (#429).

    Returns `(report_lines, ledger_row, comparison)`. Recording and surfacing
    happen in this one call so the per-round comparison row is a property of
    every round rather than a step a round has to remember to take. The lookup
    runs *before* the write: a round must never be compared against itself.

    `candidate_overlay` is the winning candidate's overlay directory (None when the
    round had no candidate). This is where the #789 contract shape is measured,
    because this is the one caller that has both the live prompt files and the
    overlay in hand: `post_promotion` records and renders it but must not read
    prompt text or import the promote module (#429's separation test). The row is
    written before the drift block is built, so the block names this round too — the
    ratchet that refused a candidate read a history assembled *without* it, and a
    reader three promotions later wants to see the point that tripped it.

    `comparison` is the verdict from :func:`post_promotion_comparison`, passed by
    `run()` because the restore had to act on it before this call wrote the round's
    own row; `_COMPUTE` keeps a caller with no restore to run — a replay, a test —
    able to ask for the whole thing in one call. `restore` is #1099's outcome,
    rendered into this section by `post_promotion.report_section`: the restore happens
    *before* the record, but its verdict is reported in the section the record
    produced, because one section per round is what a reader follows.
    """
    import prompt_surface

    from .promote import contract_shape_fields

    if comparison is _COMPUTE:
        _, comparison = post_promotion_comparison(cfg, round_id, baseline_mean)
    lines = post_promotion.report_section(comparison, restore)
    row = post_promotion.record_round_summary(
        cfg, round_id, baseline_mean, promotion_result, variant_summary,
        contract_shape_fields(candidate_overlay),
    )
    lines += post_promotion.shape_report_lines(
        post_promotion.contract_shape_series(cfg.paths.ledger_path),
        prompt_surface.GATE_STACK_CEILING,
        prompt_surface.PROHIBITION_RATIO_CEILING,
        prompt_surface.CONTRACT_RISE_RUN,
    )
    return lines, row, comparison


def variant_surfaces(variant: dict[str, Any]) -> list[str]:
    """The prompt surface(s) a variant's proposal claims to edit.

    #680: a drop has to be attributable, and the only record of which surface a
    variant aimed at is its own edit list — the `AnchorApplyError` text names a
    path only for the edits that got as far as the canonical read, and a variant
    refused for spanning two surfaces names none. So the attribution reads the
    proposal, not the exception: the edit paths, falling back to
    `overlay_files` for a hand-built variant, falling back to a literal
    `"unknown"` rather than guessing a surface it was never told.

    A list, not a str, because the multi-surface rejection is exactly the case
    where one drop is attributable to more than one surface.
    """
    paths: list[str] = []
    for edit in variant.get("edits") or []:
        if isinstance(edit, dict):
            path = str(edit.get("path", ""))
            if path and path not in paths:
                paths.append(path)
    if not paths:
        paths = [str(p) for p in (variant.get("overlay_files") or {}) if p]
    return paths or ["unknown"]


def materialize_variants(
    cfg: AutoresearchConfig,
    variants: list[dict[str, Any]],
    baseline_pair: tuple[str, Path],
) -> tuple[list[tuple[str, Path]], dict[str, int]]:
    """Build the `(variant_id, overlay_dir)` list that a round actually benches.

    Returns `(variant_pairs, dropped_by_surface)`. #680 widened the second value
    from a bare int: every surface a proposed variant aimed at is seeded at 0
    and incremented on a drop, so a round that dropped 2 MEMORY variants and 0
    SOUL variants reports `{MEMORY.md: 2, SOUL.md: 0}` rather than `2`. The
    zeros are the point — an aggregate count cannot tell "the surface that is
    hard to anchor into" apart from "a surface nobody proposed", which is the
    distinction the fix has to be measured against. `sum(dropped_by_surface
    .values())` is the old aggregate.

    `variant_pairs` starts at the baseline —
    which `run()` needs the id of afterwards, so the caller passes the pair in
    rather than this function materializing it twice — and grows by one entry per
    variant `materialize` accepts. That list is the left operand of the
    (variant × task) matrix, so an entry missing here is a variant that is never
    scored, never compared to baseline, and never produces a decision row.

    #876: `fe40af3` (#446, 2026-09-09 13:08) reworked this loop and deleted the
    `variant_pairs.append` line without restoring it, so every round since has
    benched the baseline alone: the evaluation loop skips the baseline, so
    `decisions` stayed empty and the `event: "decision"` append wrote zero rows.
    The missing append is the whole defect; keep it pinned
    (`tests/test_autoresearch_variant_pairs.py`).

    #446's drop-whole semantics are unchanged: the parser bounds an edit's shape,
    and only here can an anchor be checked against the text it claims to quote.
    Zero or two matches raises `AnchorApplyError`, counts that variant against
    its surface, writes nothing, and the round goes on with the variants that did
    apply — no fuzzy match, no partial write.
    """
    variant_pairs: list[tuple[str, Path]] = [baseline_pair]
    dropped_by_surface: dict[str, int] = {}
    for v in variants:
        surfaces = variant_surfaces(v)
        for surface in surfaces:
            dropped_by_surface.setdefault(surface, 0)
        try:
            overlay = materialize(cfg, v)
        except AnchorApplyError as exc:
            logger.warning("dropping variant %s: %s", v.get("variant_id"), exc)
            for surface in surfaces:
                dropped_by_surface[surface] += 1
            continue
        variant_pairs.append((v["variant_id"], overlay))
    dropped = sum(dropped_by_surface.values())
    if dropped:
        logger.warning(
            "dropped %d of %d variants at anchored-edit apply (%s)",
            dropped, len(variants), _surface_drop_text(dropped_by_surface),
        )
    return variant_pairs, dropped_by_surface


def _surface_drop_text(dropped_by_surface: dict[str, int]) -> str:
    """`{MEMORY.md: 2, SOUL.md: 0}` → `MEMORY.md: 2, SOUL.md: 0`.

    One formatter shared by the log line and the round report, so the two cannot
    describe a round differently. Sorted so the text is stable for a test and for
    a diff across rounds.
    """
    return ", ".join(f"{surface}: {count}" for surface, count in sorted(dropped_by_surface.items()))


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


def record_split(cfg: AutoresearchConfig, all_tasks: list[dict], rid: str) -> dict:
    """Write this round's bench split and record it in the ledger; return the split.

    A function, not lines inside `run()`, so a test can drive the real write against
    the real bench and read back the artifact — a unit test over a hand-built split
    dict cannot tell you that `run()` never calls `write_split`, which is the mistake
    a test that fakes the whole round makes. Raises `RuntimeError` from `write_split`
    (no bench tasks, or the written artifact fails `verify`); the caller aborts the
    round, because a round with no slice has no veto and would promote anything —
    `compute_split` raises `RuntimeError` on a bench that yields no targeted or no
    held-out task, so a one-sided split cannot be written at all.
    """
    split = bench_split.write_split(cfg, all_tasks, rid)
    ledger_append(cfg.paths.ledger_path, {
        "round_id": rid,
        "event": "split",
        "split_hash": split["split_hash"],
        "targeted": split["targeted"],
        "heldout": split["heldout"],
        "rotated_into_heldout": split["rotated_into_heldout"],
        "created_at": now_iso(),
    })
    logger.info("split %s: %d targeted / %d held-out (hash %s…)", rid,
                len(split["targeted"]), len(split["heldout"]), split["split_hash"][:12])
    return split


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

    all_tasks = load_bench_tasks(cfg.paths.bench_dir)
    if not all_tasks:
        msg = f"bench dir empty: {cfg.paths.bench_dir}"
        logger.error(msg)
        return {"round_id": rid, "error": msg}

    # 3. Pin the held-out slice BEFORE anything is proposed (#549). Writing it
    # here, not at decision time, is the point: the split has to exist before the
    # proposer sees a score, or the hash only proves which slice was picked after
    # the results were known.
    #
    # The split is written from the FULL bench, and `--bench-limit` is applied
    # afterwards to the tasks this round actually runs. Truncating before the
    # split instead is the vacuous-gate shape this item was filed on, one layer
    # down from the one in the acceptance: a `--bench-limit 4` round would slice
    # 4 targeted tasks and no veto task, and a gate with nothing to refuse on
    # promotes anything. With the split over the whole bench, a truncated round
    # scores no veto task at all, `evaluate_promotion` refuses it with
    # `no_heldout_overlap` naming the slice, and the round still exercises the
    # plumbing — it just cannot promote, which is the honest verdict for a round
    # that never measured the half it is supposed to protect.
    tasks = all_tasks[:bench_limit] if bench_limit else list(all_tasks)
    if len(tasks) < len(all_tasks):
        logger.info("bench-limit %d: split over all %d tasks, %d evaluated this round",
                    bench_limit, len(all_tasks), len(tasks))

    logger.info("loaded %d bench tasks", len(tasks))

    try:
        split = record_split(cfg, all_tasks, rid)
    except RuntimeError as exc:
        logger.error("bench split failed: %s", exc)
        return {"round_id": rid, "error": f"bench split: {exc}"}

    variants = await asyncio.to_thread(
        propose_variants,
        cfg, targets=targets, max_variants=max_variants or cfg.max_variants_per_round, model=model,
        parent_variant=parent_variant,
    )
    if not variants:
        logger.warning("hypothesis generator produced 0 variants — nothing to evaluate this round")

    # Materialize baseline + each variant as overlay dirs. The returned pair list
    # is what `_run_trials` benches, so every variant that survives the anchored
    # edit has to be in it — #876.
    baseline_id, baseline_dir = materialize_baseline(cfg)
    variant_pairs, dropped_by_surface = materialize_variants(cfg, variants, (baseline_id, baseline_dir))
    dropped = sum(dropped_by_surface.values())

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
            # #651, from the same helper the on-demand runner uses. A round is
            # where the volume is, so this is the row that makes "did a variant
            # read its own grading" a queryable question rather than an
            # inspection of one CLI run. A direct trace has no tool channel and
            # gets the honest zeros.
            **probe_ledger_fields(t),
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

    # #429 + #1099: read the previous promotion's recorded mean *now*, before the
    # promotion decision below and long before this round's `round_summary` row is
    # appended. The order is load-bearing twice over: `last_promotion` excludes this
    # round by id, an exclusion that buys nothing once the row exists, and the restore
    # has to run in this window — a round that wrote its own comparison row first would
    # be recording a decline it had already acted on, and the report would describe a
    # contract it had already rewritten.
    prior_promotion, comparison = post_promotion_comparison(
        cfg, rid, float(baseline_summary.get("mean_composite", 0.0)))

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
        should, reason = evaluate_promotion(cfg, baseline_summary, vs, split=split)
        m = slice_metrics(baseline_summary, vs, split)
        decisions.append({
            "variant_id": vid,
            "mean_composite": vs.get("mean_composite"),
            "targeted_delta": m["targeted_delta"],
            "heldout_delta": m["heldout_delta"],
            "normalized_gain": m["normalized_gain"],
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

    # #1099, Alan's sign-off on #429's out-of-scope clause: a beyond-noise decline is
    # no longer only reported, it undoes the promotion that caused it — through
    # `promote.rollback`, which commits by the same validated vault route `promote`
    # uses, refuses any canonical file that changed after the promotion landed, and
    # never restores one promotion twice. Nothing here waits for a human; nothing here
    # copies a prompt file by hand. A `--dry-run` round never restores either.
    restore_outcome = auto_restore.restore_for_decline(
        cfg, rid, comparison, prior_promotion, dry_run=dry_run,
    )

    # #429: a promotion used to be the last time anything looked at it. Record this
    # round's comparison row and carry the verdict into the report below. The restore
    # above already ran (or declined, or refused) on the comparison computed before the
    # promotion decision; what is recorded here is the row plus that outcome, so the
    # report states the restore in the same section as the decline that caused it.
    # #789: the same row carries the contract shape — the live SOUL.md's two
    # ratios and this round's candidate's own two, read from `best_overlay`, which
    # is None whenever no candidate survived to the contract check.
    report_lines, summary_row, comparison = post_promotion_check(
        cfg,
        rid,
        float(baseline_summary.get("mean_composite", 0.0)),
        promotion_result,
        best_summary,
        best_overlay,
        comparison=comparison,
        restore=restore_outcome,
    )

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
        # #680: the aggregate said "2 dropped", which cannot distinguish "the
        # model invented spans" from "half this round's variants aimed at a file
        # it was only shown 4,000 chars of". Every proposed surface is listed,
        # zeros included, so a round that produced no MEMORY variants says so.
        f"- variants dropped by surface: {_surface_drop_text(dropped_by_surface) or '(none proposed)'}",
        f"- baseline mean composite: {baseline_summary.get('mean_composite', 0.0):.4f}",
        "",
        "## Variant summaries",
    ]
    for vid, summ in summaries.items():
        marker = " (baseline)" if vid == baseline_id else ""
        lines.append(f"- `{vid}`{marker}: mean={summ.get('mean_composite', 0.0):.4f}, "
                     f"safety={'pass' if summ.get('safety_passed') else 'fail'}, tasks={summ.get('task_count', 0)}")
    # The held-out slice is reported here and never to the proposer (#549): the
    # round is allowed to know which veto tasks declined, the optimizer is not.
    lines.append("")
    lines.append("## Bench split")
    lines.append(f"- split_hash: `{split['split_hash']}`")
    lines.append(f"- targeted ({len(split['targeted'])}): {', '.join(split['targeted'])}")
    lines.append(f"- held-out ({len(split['heldout'])}): {', '.join(split['heldout'])}"
                 f" (rotated in: {', '.join(split['rotated_into_heldout']) or 'none'})")
    heldout_scores = {p["task_id"]: p["composite_score"]
                      for p in baseline_summary.get("per_task", [])
                      if p["task_id"] in set(split["heldout"])}
    if heldout_scores:
        lines.append("- baseline held-out scores: " + ", ".join(
            f"{t}={heldout_scores[t]:.2f}" for t in sorted(heldout_scores)))
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
    lines.extend(report_lines)
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
        "variants_dropped": dropped,
        "variants_dropped_by_surface": dict(dropped_by_surface),
        "tasks_run": len(tasks),
        "baseline_mean": baseline_summary.get("mean_composite", 0.0),
        "decisions": decisions,
        "promoted": promotion_result,
        "post_promotion": comparison,
        # #1099: the restore verdict is part of the round's result, not a side effect
        # a caller has to go and find in the ledger. `status` is one of
        # `auto_restore`'s statuses and `vault_commit` is the sha when it landed one.
        "post_promotion_restore": (
            {k: restore_outcome.get(k) for k in
             ("status", "restored", "vault_commit", "reason", "restored_by")}
            if restore_outcome else None),
        "round_summary_row": summary_row,
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
