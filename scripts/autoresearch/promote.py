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
    saved = sorted(p.name for p in snap_dir.iterdir() if p.is_file())
    # A snapshot is the only rollback point a promotion has, and `mkdir` had
    # been the whole guarantee: a copy that raised still left a directory, so
    # `promote` went on to overwrite the live contract with nothing to restore.
    # 26 of the ledger's 83 promotions have no matching snapshot. Found while
    # writing `test_promote_refuses_when_the_snapshot_cannot_be_written` on
    # 2026-09-06 and carried as an xfail until now.
    if not any(name in CANONICAL_PROMPTS for name in saved):
        raise RuntimeError(
            f"snapshot {snap_dir} holds no prompt file — refusing to promote "
            f"with no rollback point (expected any of {sorted(CANONICAL_PROMPTS)})"
        )
    (snap_dir / "snapshot.json").write_text(
        json.dumps({"created_at": now_iso(), "files": saved}, indent=2),
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


def _prospective(overlay_dir: Path, name: str) -> str | None:
    """What `name` would contain after this overlay is applied."""
    src = overlay_dir / name
    if src.exists():
        return src.read_text(encoding="utf-8")
    dest = CANONICAL_PROMPTS.get(name)
    if dest and dest.exists():
        return dest.read_text(encoding="utf-8")
    return None


def contract_refusals(overlay_dir: Path) -> list[str]:
    """Why this variant must not be written over the live contract. [] = fine.

    This path is what produced #464 and #465. It writes Lloyd's identity files
    in the live vault on an hourly cadence with no gate, no test, no review and
    no revert, and on 2026-09-08 it promoted a variant whose own hypothesis was
    "add negative constraints to fix the shape benchmarks" — 64% gate stack,
    28% prohibition lines — over a contract that had been trimmed to 48%/19%
    ninety minutes earlier. A search over prompts is legitimate; landing the
    result unchecked is not, and the search cannot be trusted to police itself
    because bench score is exactly what it is optimising.

    Checked against the *prospective* text — the overlay's file where it has
    one, the live file where it does not — so a variant that only rewrites
    MEMORY.md is still judged against the SOUL.md it will sit beside.
    """
    try:
        import prompt_surface
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"prompt_surface unavailable, refusing to promote blind: {exc}"]
    soul = _prospective(overlay_dir, "SOUL.md")
    if soul is None:
        return ["no SOUL.md to check, in the overlay or on disk"]
    return prompt_surface.check_contract(soul, _prospective(overlay_dir, "MEMORY.md"))


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
    refusals = contract_refusals(variant_overlay_dir)
    if refusals:
        # Not an exception: a refused promotion is a normal outcome of a search
        # that proposed something out of bounds, and the round must carry on
        # evaluating the rest. It is logged at ERROR because a variant that wins
        # on bench score and still cannot be written is the signal that the
        # score and the constraint disagree — which is the whole finding.
        logger.error("REFUSED promotion of %s — %s",
                     variant["variant_id"], "; ".join(refusals))
        result["refused"] = refusals
        return result

    if dry_run:
        logger.info("[dry-run] would promote %s", variant["variant_id"])
        return result

    try:
        snap = snapshot_current_prompts(cfg)
    except (OSError, RuntimeError) as exc:
        logger.error("REFUSED promotion of %s — no rollback point: %s",
                     variant["variant_id"], exc)
        result["refused"] = [f"snapshot failed: {exc}"]
        return result
    applied = apply_overlay(variant_overlay_dir)
    result["snapshot_dir"] = str(snap)
    result["applied_files"] = applied

    # Commit through the vault route, which runs the real loaders and reverts
    # the paths if any of them fails. Before this, a promotion was an
    # uncommitted overwrite of a live tracked file: nothing recorded that it
    # had happened, `autoimplement_vault_revert` had no sha to undo, and the only
    # trace was a snapshot directory nobody was told about. The sha goes in the
    # autoimplement ledger as a `vault_land` event like every other vault change.
    try:
        from scripts.autoimplement import vault_round as VR

        # Derive the vault-relative paths rather than assuming `lloyd/<name>`,
        # and commit nothing when the canonical prompts are not in the vault at
        # all. `CANONICAL_PROMPTS` is redirected by tests and by the variant
        # sandbox, and a hardcoded prefix would have made this function run
        # `git add` against the real `~/obsidian` from inside a unit test.
        rel: list[str] = []
        vault_root = VR.VAULT.resolve()
        for name in applied:
            try:
                rel.append(str(CANONICAL_PROMPTS[name].resolve().relative_to(vault_root)))
            except ValueError:
                rel = []
                break
        if rel:
            landed = VR.land(
                rel,
                f"autoresearch: promote {variant['variant_id']}\n\n"
                f"{variant.get('description', '')}\n\n"
                f"Snapshot of the previous contract: {snap}",
            )
            result["vault_commit"] = landed.get("commit")
        else:
            # Not an error: the overlay sandbox and the tests both point the
            # canonical prompts somewhere that is not a git repo, and there is
            # nothing to commit there. The fact below is still written.
            logger.info("canonical prompts are outside %s — applied without a "
                        "vault commit", vault_root)
    except Exception as exc:  # VaultRoundError, or git refusing for any reason
        # `land` reverts the paths it validated before raising, so the live
        # contract is already back. Say so loudly and leave the snapshot.
        logger.error("promotion of %s did not land, contract restored: %s",
                     variant["variant_id"], exc)
        result["refused"] = [str(exc)]
        result["applied_files"] = []
        return result

    fact = write_experiment_fact(cfg, variant, variant_summary, baseline_summary, snap)
    result["experiment_fact"] = str(fact) if fact else None
    logger.info("promoted %s: applied=%s vault_commit=%s snapshot=%s",
                variant["variant_id"], applied, result.get("vault_commit"), snap)
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
