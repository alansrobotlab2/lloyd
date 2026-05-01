"""Variant sandbox — materializes a proposed variant as an overlay directory.

Each variant becomes a directory under `_pipeline/research/variants/<variant_id>/`
containing the overridden files. The bench runner points LLOYD_OVERLAY_DIR at
this directory; `prompt_builder.py` reads from the overlay and falls through to
the canonical vault for any file the variant did not override.

For the baseline (unmodified) evaluation, we materialize a "baseline" variant
whose overlay is an empty dir — so it falls through to the canonical files.
That keeps the bench runner path identical for baseline and candidates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .common import AutoresearchConfig, _canonical_prompt_paths, now_iso

logger = logging.getLogger("autoresearch.sandbox")


def _supported_relative_paths() -> set[str]:
    """Relative-path names a variant is allowed to overlay.

    Single source of truth: `_canonical_prompt_paths()` in common.py.
    Its keys are the relative names; its values are the absolute paths
    written into `run_spec.yaml`'s `mutation_scope.writable_paths`.
    Both the spec and the enforcement here resolve from the same dict
    so they cannot drift. (Skills/config overlays remain out of scope —
    `_canonical_prompt_paths()` is the gate for what to support.)

    #335 acceptance: writable-path enforcement is now spec-aligned.
    """
    return set(_canonical_prompt_paths().keys())


def materialize(cfg: AutoresearchConfig, variant: dict[str, Any]) -> Path:
    """Write the variant's overlay files to disk and return the overlay dir."""
    variant_id_val = variant["variant_id"]
    overlay_dir = cfg.paths.variants_dir / variant_id_val
    overlay_dir.mkdir(parents=True, exist_ok=True)

    supported = _supported_relative_paths()
    for rel_path, content in (variant.get("overlay_files") or {}).items():
        if rel_path not in supported:
            logger.warning(
                "variant %s: skipping unsupported path %s (not in spec writable_paths: %s)",
                variant_id_val, rel_path, sorted(supported),
            )
            continue
        dest = overlay_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    meta = {
        "variant_id": variant_id_val,
        "target_surface": variant.get("target_surface", "prompts"),
        "description": variant.get("description", ""),
        "hypothesis": variant.get("hypothesis", ""),
        "parent_variant_id": variant.get("parent_variant_id"),
        "overlay_files": sorted((variant.get("overlay_files") or {}).keys()),
        "created_at": variant.get("created_at", now_iso()),
    }
    (overlay_dir / "variant.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("materialized variant %s at %s", variant_id_val, overlay_dir)
    return overlay_dir


def materialize_baseline(cfg: AutoresearchConfig) -> tuple[str, Path]:
    """Create an empty overlay dir that falls through to the canonical vault."""
    baseline_id = f"BASELINE_{cfg.paths.variants_dir.stat().st_ctime:.0f}"
    overlay_dir = cfg.paths.variants_dir / baseline_id
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / "variant.json").write_text(
        json.dumps({
            "variant_id": baseline_id,
            "target_surface": "baseline",
            "description": "Unmodified canonical state.",
            "hypothesis": "control",
            "overlay_files": [],
            "created_at": now_iso(),
        }, indent=2),
        encoding="utf-8",
    )
    return baseline_id, overlay_dir
