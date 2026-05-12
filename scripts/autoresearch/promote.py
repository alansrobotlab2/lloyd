"""Promotion pipeline — snapshot, atomic swap, fact write.

A promotion executes when a variant beats baseline on >= N% of bench tasks
and passes all safety probes. Before swapping, we snapshot the current state
(SOUL.md, MEMORY.md, USER.md) into `_pipeline/research/snapshots/<ts>/`.
After swap, we write the winning experiment as a fact under
`cfg.paths.facts_experiments_dir/<variant_id>/` (configured in config.yaml,
currently `~/lloyd/_pipeline/vault-derived/facts/experiments/`) so it's
queryable via the normal memory pipeline.

`rollback(snapshot_ts)` reverses a promotion by restoring files from the
named snapshot.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import LLOYD_HOME, AutoresearchConfig, now_iso

logger = logging.getLogger("autoresearch.promote")

# Canonical targets that can be overwritten by promotion
CANONICAL_PROMPTS = {
    "SOUL.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md",
    "MEMORY.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "MEMORY.md",
    "USER.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "USER.md",
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def evaluate_promotion(
    cfg: AutoresearchConfig,
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
) -> tuple[bool, str]:
    """Decide whether a variant should be promoted over baseline.

    Returns (should_promote, reason).
    """
    if not variant_summary.get("safety_passed", False) and cfg.promotion_require_safety_pass:
        return False, "safety_regression"

    baseline_mean = float(baseline_summary.get("mean_composite", 0.0))
    variant_mean = float(variant_summary.get("mean_composite", 0.0))
    delta = variant_mean - baseline_mean

    if delta < cfg.promotion_min_composite_delta:
        return False, f"insufficient_delta ({delta:+.4f} < {cfg.promotion_min_composite_delta})"

    # Win fraction — per-task: variant composite >= baseline composite
    baseline_per = {p["task_id"]: p["composite_score"] for p in baseline_summary.get("per_task", [])}
    wins = 0
    total = 0
    for p in variant_summary.get("per_task", []):
        task_id = p["task_id"]
        if task_id not in baseline_per:
            continue
        total += 1
        if p["composite_score"] > baseline_per[task_id]:
            wins += 1
    win_frac = (wins / total) if total else 0.0
    if win_frac < cfg.promotion_min_win_fraction:
        return False, f"insufficient_win_fraction ({win_frac:.2f} < {cfg.promotion_min_win_fraction})"

    return True, f"promote (delta={delta:+.4f}, win_frac={win_frac:.2f})"


def snapshot_current_prompts(cfg: AutoresearchConfig) -> Path:
    """Copy current SOUL/MEMORY/USER into a timestamped snapshot dir. Returns the dir."""
    snap_dir = cfg.paths.snapshots_dir / _ts()
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, src in CANONICAL_PROMPTS.items():
        if src.exists():
            shutil.copy2(src, snap_dir / name)
    (snap_dir / "snapshot.json").write_text(
        json.dumps({"created_at": now_iso(), "files": sorted(p.name for p in snap_dir.iterdir() if p.is_file())}, indent=2),
        encoding="utf-8",
    )
    logger.info("snapshotted canonical prompts into %s", snap_dir)
    return snap_dir


def apply_overlay(overlay_dir: Path) -> list[str]:
    """Copy variant overlay files onto canonical prompts. Returns list of applied files."""
    applied: list[str] = []
    for name, dest in CANONICAL_PROMPTS.items():
        src = overlay_dir / name
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            applied.append(name)
    return applied


def write_experiment_fact(
    cfg: AutoresearchConfig,
    variant: dict[str, Any],
    variant_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    snapshot_dir: Path,
) -> Path | None:
    """Write the promoted experiment as a fact file under cfg.paths.facts_experiments_dir/<id>/."""
    ex_dir = cfg.paths.facts_experiments_dir / variant["variant_id"]
    ex_dir.mkdir(parents=True, exist_ok=True)
    fact_file = ex_dir / f"{variant['variant_id']}-experiment.md"

    baseline_mean = baseline_summary.get("mean_composite", 0.0)
    variant_mean = variant_summary.get("mean_composite", 0.0)
    delta = variant_mean - baseline_mean

    frontmatter = {
        "type": "facts",
        "entity": variant["variant_id"],
        "category": "experiment",
        "last_updated": now_iso(),
        "facts": [
            {
                "fact": f"Autoresearch variant {variant['variant_id']} promoted ({variant.get('description', '')}). "
                        f"mean_composite {baseline_mean:.3f} → {variant_mean:.3f} (Δ {delta:+.3f}) over "
                        f"{variant_summary.get('task_count', 0)} bench tasks.",
                "confidence": 0.95,
                "category": "experiment",
                "id": f"exp-{variant['variant_id']}",
                "created_at": now_iso(),
                "valid_at": now_iso(),
                "invalid_at": None,
                "expired_at": None,
                "provenance": "EXTRACTED",
                "source_doc": str(snapshot_dir),
            }
        ],
    }
    import yaml

    body = (
        f"\n# {variant['variant_id']} - experiment\n\n"
        f"**Target surface:** {variant.get('target_surface', 'prompts')}\n"
        f"**Hypothesis:** {variant.get('hypothesis', '')}\n"
        f"**Snapshot:** `{snapshot_dir}`\n"
        f"**Baseline mean composite:** {baseline_mean:.3f}\n"
        f"**Variant mean composite:** {variant_mean:.3f}\n"
        f"**Delta:** {delta:+.3f}\n"
        f"**Task count:** {variant_summary.get('task_count', 0)}\n"
    )
    fact_file.write_text(f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n{body}", encoding="utf-8")
    logger.info("wrote experiment fact to %s", fact_file)
    return fact_file


def promote(
    cfg: AutoresearchConfig,
    variant: dict[str, Any],
    variant_overlay_dir: Path,
    variant_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full promotion pipeline. Returns a result dict."""
    result: dict[str, Any] = {
        "variant_id": variant["variant_id"],
        "dry_run": dry_run,
        "snapshot_dir": None,
        "applied_files": [],
        "experiment_fact": None,
    }
    if dry_run:
        logger.info("[dry-run] would promote %s", variant["variant_id"])
        return result

    snap = snapshot_current_prompts(cfg)
    applied = apply_overlay(variant_overlay_dir)
    fact = write_experiment_fact(cfg, variant, variant_summary, baseline_summary, snap)
    result["snapshot_dir"] = str(snap)
    result["applied_files"] = applied
    result["experiment_fact"] = str(fact) if fact else None
    logger.info("promoted %s: applied=%s snapshot=%s", variant["variant_id"], applied, snap)
    return result


def rollback(cfg: AutoresearchConfig, snapshot_ts: str) -> dict[str, Any]:
    """Restore canonical prompts from the named snapshot."""
    snap = cfg.paths.snapshots_dir / snapshot_ts
    if not snap.exists():
        return {"error": f"snapshot {snapshot_ts} not found"}
    restored: list[str] = []
    for name, dest in CANONICAL_PROMPTS.items():
        src = snap / name
        if src.exists():
            shutil.copy2(src, dest)
            restored.append(name)
    logger.info("rolled back %s files from %s", len(restored), snap)
    return {"snapshot": str(snap), "restored_files": restored}
