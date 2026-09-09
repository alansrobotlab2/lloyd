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


class AnchorApplyError(ValueError):
    """An anchored edit cannot be applied mechanically and exactly.

    Raised when an anchor matches zero or more than once, when an edit is
    malformed, or when a variant's edits span more than one surface. The apply is
    all-or-nothing by construction: this is computed against the canonical text
    in memory and raised before anything is written, so a rejected variant never
    leaves a half-edited overlay behind. No fuzzy matching, no partial write —
    a zero-match anchor means the model invented a span it was never shown, and a
    two-match anchor means it quoted something ambiguous; guessing either way
    would put text into SOUL.md the proposal did not contain.
    """


def _group_edits_by_path(edits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Edit list → {surface: edits}, in first-seen order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edit in edits:
        grouped.setdefault(str(edit.get("path", "")), []).append(edit)
    return grouped


def _apply_anchored_edits(rel_path: str, edits: list[dict[str, Any]]) -> str:
    """Mechanically apply a variant's edits to the canonical text for `rel_path`.

    Each anchor must match the text as it stands when that edit is applied —
    exactly once, or the whole variant is refused (`AnchorApplyError`). Edits
    apply in list order, so a later anchor may sit in text an earlier edit just
    introduced; that is intended, and it is still an exact match either way.

    Returns the resulting full text. The overlay dir then holds that file, so
    everything downstream — the bench runner's LLOYD_OVERLAY_DIR, promote's
    `contract_refusals`, `apply_overlay` — keeps reading a materialized file and
    never needs to know the proposal was a patch.
    """
    canonical = _canonical_prompt_paths().get(rel_path)
    if canonical is None:
        raise AnchorApplyError(
            f"{rel_path}: not a spec-writable prompt surface "
            f"({sorted(_canonical_prompt_paths())})"
        )
    if not canonical.exists():
        raise AnchorApplyError(f"{rel_path}: canonical file is absent — an anchored edit needs text to anchor into")

    text = canonical.read_text(encoding="utf-8")
    for i, edit in enumerate(edits):
        anchor = edit.get("anchor")
        replacement = edit.get("replacement")
        if not isinstance(anchor, str) or not anchor:
            raise AnchorApplyError(f"{rel_path} edit {i}: empty or non-string anchor")
        if not isinstance(replacement, str):
            raise AnchorApplyError(f"{rel_path} edit {i}: replacement must be a string ('' to delete)")
        hits = text.count(anchor)
        if hits != 1:
            raise AnchorApplyError(
                f"{rel_path} edit {i}: anchor matched {hits} time(s), want exactly 1 "
                f"— no fuzzy match, no partial write: {anchor[:70]!r}"
            )
        text = text.replace(anchor, replacement, 1)
    return text


def materialize(cfg: AutoresearchConfig, variant: dict[str, Any]) -> Path:
    """Write the variant's overlay files to disk and return the overlay dir.

    A #446 variant carries `edits`, and the text they produce is what gets
    written; `overlay_files` is still honoured for hand-built variants and the
    baseline. An anchored edit that cannot be applied exactly raises
    `AnchorApplyError` and writes nothing.
    """
    variant_id_val = variant["variant_id"]
    overlay_dir = cfg.paths.variants_dir / variant_id_val
    overlay_dir.mkdir(parents=True, exist_ok=True)

    edits = variant.get("edits") or []
    overlay_files: dict[str, Any] = dict(variant.get("overlay_files") or {})
    if edits:
        grouped = _group_edits_by_path(edits)
        if len(grouped) > 1:
            raise AnchorApplyError(
                f"variant {variant_id_val}: edits span {len(grouped)} surfaces "
                f"{sorted(grouped)}; one surface per variant"
            )
        for rel_path, path_edits in grouped.items():
            overlay_files[rel_path] = _apply_anchored_edits(rel_path, path_edits)

    supported = _supported_relative_paths()
    for rel_path, content in overlay_files.items():
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
        "overlay_files": sorted(overlay_files.keys()),
        # The audit trail now has to carry the patch, because a #446 proposal no
        # longer contains the file it produced. Without this, a promoted variant
        # would be unattributable to the edit that made it.
        "edits": edits,
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
