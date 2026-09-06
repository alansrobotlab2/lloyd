"""Autoresearch promotion gate — the arithmetic that rewrites live prompts.

Why this file exists
--------------------
`scripts/autoresearch/` had zero tests. It runs unattended nightly, and its
`promote()` path copies variant files over SOUL.md / MEMORY.md / USER.md in the
live vault. On 2026-09-05 the ledger held 83 promotion decisions, 61 of them on
a score delta smaller than the run-to-run noise of an *unchanged* system (three
identical baseline runs scored 0.719 / 0.542 / 0.624 — spread 0.177, against a
`min_composite_delta` of 0.05).

Every one of those decisions came out of `evaluate_promotion()`, a pure function
with no assertion anywhere in the repo. This file is the check.

Isolation
---------
`promote.CANONICAL_PROMPTS` is a module-level dict of live vault paths resolved
at import time. The autouse fixture replaces it for every test in this module,
so nothing here can write to `~/obsidian/lloyd/`. The two tests that need to
*look* at the real values only read them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch import promote
from scripts.autoresearch.common import AutoresearchConfig, AutoresearchPaths

PROMPT_NAMES = ("SOUL.md", "MEMORY.md", "USER.md")

# The live spec, written down so a change to config.yaml has to update this too.
LIVE_MIN_COMPOSITE_DELTA = 0.05
LIVE_MIN_WIN_FRACTION = 0.50
LIVE_REQUIRE_SAFETY_PASS = True

# Measured 2026-09-05 from three identical baseline runs of the canonical prompts.
MEASURED_NOISE_SPREAD = 0.177


def make_cfg(tmp_path: Path, **over) -> AutoresearchConfig:
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench",
        research_root=tmp_path / "research",
        rounds_dir=tmp_path / "rounds",
        ledger_path=tmp_path / "ledger.jsonl",
        variants_dir=tmp_path / "variants",
        snapshots_dir=tmp_path / "snapshots",
        facts_experiments_dir=tmp_path / "facts-experiments",
    )
    kw = dict(
        paths=paths,
        default_model="primary",
        default_budget_minutes=120,
        max_variants_per_round=7,
        promotion_min_win_fraction=LIVE_MIN_WIN_FRACTION,
        promotion_min_composite_delta=LIVE_MIN_COMPOSITE_DELTA,
        promotion_require_safety_pass=LIVE_REQUIRE_SAFETY_PASS,
        tool_allowlist_consecutive_wins=2,
        targets=["prompts"],
    )
    kw.update(over)
    return AutoresearchConfig(**kw)


def summary(mean: float, wins_from: int | None = None, n: int = 11,
            task_ids: list[str] | None = None,
            scores: list[float] | None = None) -> dict:
    """A bench summary shaped like `judge.aggregate_variant`'s output.

    `mean_composite` and `per_task` are independent on purpose: the gate reads
    the mean for the delta check and per-task scores for the win fraction, so a
    test can drive one without the other.
    """
    ids = task_ids or [f"bench_{i:03d}" for i in range(n)]
    if scores is not None:
        per = [{"task_id": tid, "composite_score": sc} for tid, sc in zip(ids, scores)]
    else:
        per = [{"task_id": tid,
                "composite_score": 1.0 if (wins_from and i < wins_from) else 0.4}
               for i, tid in enumerate(ids)]
    return {"mean_composite": mean, "safety_passed": True, "task_count": len(per), "per_task": per}


@pytest.fixture(autouse=True)
def isolated_prompts(tmp_path, monkeypatch):
    """Point every canonical-prompt write at a tmp vault, never ~/obsidian."""
    vault = tmp_path / "vault-lloyd"
    vault.mkdir()
    targets = {}
    for name in PROMPT_NAMES:
        p = vault / name
        p.write_text(f"canonical {name}\n", encoding="utf-8")
        targets[name] = p
    monkeypatch.setattr(promote, "CANONICAL_PROMPTS", targets)
    return targets


# ── evaluate_promotion: the safety half ──────────────────────────────────────

def test_safety_regression_blocks_promotion_even_with_huge_delta(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    var = summary(0.99)
    var["safety_passed"] = False
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False
    assert reason == "safety_regression"


def test_safety_check_is_skipped_when_not_required(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path, promotion_require_safety_pass=False)
    var = summary(0.90, wins_from=11)
    var["safety_passed"] = False
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is True, reason


def test_missing_safety_field_is_treated_as_failed(isolated_prompts, tmp_path):
    """A summary that never set safety_passed must not slip through."""
    cfg = make_cfg(tmp_path)
    var = summary(0.99, wins_from=11)
    var.pop("safety_passed")
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False and reason == "safety_regression"


# ── evaluate_promotion: the delta half ───────────────────────────────────────

def test_delta_just_below_threshold_rejected(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    should, reason = promote.evaluate_promotion(cfg, summary(0.50), summary(0.54, wins_from=11))
    assert should is False
    assert "insufficient_delta" in reason


def test_delta_exactly_at_threshold_promotes(isolated_prompts, tmp_path):
    """`delta < min` is a strict comparison, so == threshold passes. Characterized
    here because it is an off-by-one magnet."""
    cfg = make_cfg(tmp_path)
    should, reason = promote.evaluate_promotion(cfg, summary(0.50), summary(0.55, wins_from=11))
    assert should is True, reason
    assert "promote" in reason


def test_negative_delta_rejected(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    should, reason = promote.evaluate_promotion(cfg, summary(0.70), summary(0.40, wins_from=11))
    assert should is False and "insufficient_delta" in reason


def test_missing_mean_composite_defaults_to_zero(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    should, _ = promote.evaluate_promotion(cfg, {}, {})
    assert should is False


def test_threshold_smaller_than_measured_noise_is_documented(isolated_prompts, tmp_path):
    """The live threshold accepts a change that noise alone can produce.

    Not a behavior assertion — a standing reminder. The measured spread of an
    *unchanged* system is 0.177; anything below that carries no information.
    """
    cfg = make_cfg(tmp_path)
    assert cfg.promotion_min_composite_delta < MEASURED_NOISE_SPREAD, (
        "config threshold moved — update this test and the noise measurement together"
    )
    # A pure-noise-sized improvement currently passes the delta gate:
    var = summary(0.50 + (MEASURED_NOISE_SPREAD / 2), wins_from=11)
    should, _ = promote.evaluate_promotion(cfg, summary(0.50), var)
    assert should is True, "noise-sized deltas are still promoted — see the gate TODO"


# ── evaluate_promotion: the win-fraction half ────────────────────────────────

def test_ties_are_not_wins(isolated_prompts, tmp_path):
    """`variant > baseline` is strict: an identical score does not count."""
    cfg = make_cfg(tmp_path)
    base = summary(0.10, scores=[0.4] * 11)
    var = summary(0.90, scores=[0.4] * 11)   # mean delta passes; every task ties
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "insufficient_win_fraction" in reason


def test_min_majority_of_eleven_tasks_is_enough(isolated_prompts, tmp_path):
    """The 1/11 granularity consequence: 6 of 11 beats a 0.50 threshold.

    This is the shape behind 58 of the 83 recorded promotions (`win_frac=0.55`).
    With per-task noise up to 1.0, it is a coin flip decided by the majority of
    coin flips — recorded here so raising the bench size is a visible change.
    """
    cfg = make_cfg(tmp_path)
    var = summary(0.90, wins_from=6)   # 6 better, 5 worse
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is True, reason
    assert "win_frac=0.55" in reason      # 6/11 = 0.545, shown at 2dp


def test_below_min_majority_rejected(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    var = summary(0.90, wins_from=5)   # 5/11 = 0.45
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False and "insufficient_win_fraction" in reason


def test_tasks_absent_from_baseline_are_not_counted(isolated_prompts, tmp_path):
    """A variant scored on tasks baseline never saw must not inflate win_frac."""
    cfg = make_cfg(tmp_path)
    base = summary(0.10, n=11)
    var = summary(0.90, wins_from=11, n=11,
                  task_ids=[f"extra_{i}" for i in range(11)])
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "0.00" in reason


def test_empty_variant_per_task_yields_zero_win_fraction(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    var = summary(0.99)
    var["per_task"] = []
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False and "insufficient_win_fraction" in reason


def test_all_three_gates_must_pass_together(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    # delta passes, safety passes, win fraction fails
    var = summary(0.90, wins_from=1)
    should, reason = promote.evaluate_promotion(cfg, summary(0.50), var)
    assert should is False and "insufficient_win_fraction" in reason


# ── snapshot / apply / rollback ──────────────────────────────────────────────

def test_snapshot_captures_all_present_files(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    manifest = json.loads((snap / "snapshot.json").read_text())
    # The manifest lists only the prompt files: `files` is computed by
    # iterdir() while building the JSON, before write_text() creates
    # snapshot.json itself. Matches every snapshot on disk (verified against
    # _pipeline/research/snapshots/20260905_141543/snapshot.json).
    assert sorted(PROMPT_NAMES) == manifest["files"]
    for name in PROMPT_NAMES:
        assert (snap / name).read_text() == f"canonical {name}\n"


def test_snapshot_survives_a_missing_source_without_losing_the_rest(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    isolated_prompts["USER.md"].unlink()
    snap = promote.snapshot_current_prompts(cfg)
    manifest = json.loads((snap / "snapshot.json").read_text())
    assert "SOUL.md" in manifest["files"] and "USER.md" not in manifest["files"]


def test_apply_overlay_overwrites_only_files_present_in_overlay(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("new soul\n", encoding="utf-8")
    applied = promote.apply_overlay(overlay)
    assert applied == ["SOUL.md"]
    assert isolated_prompts["SOUL.md"].read_text() == "new soul\n"
    assert isolated_prompts["MEMORY.md"].read_text() == "canonical MEMORY.md\n"


def test_snapshot_then_apply_then_rollback_round_trips(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    ts = snap.name
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    for name in PROMPT_NAMES:
        (overlay / name).write_text(f"variant {name}\n", encoding="utf-8")
    assert len(promote.apply_overlay(overlay)) == 3
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text().startswith("variant")

    result = promote.rollback(cfg, ts)
    assert sorted(result["restored_files"]) == sorted(PROMPT_NAMES)
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text() == f"canonical {name}\n"


def test_rollback_from_missing_snapshot_errors_without_touching_prompts(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    result = promote.rollback(cfg, "19700101_000000")
    assert "not found" in result["error"]
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text() == f"canonical {name}\n"


def test_rollback_restores_only_files_in_the_snapshot(isolated_prompts, tmp_path):
    """A partial snapshot must not blank the file it never captured."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    (snap / "MEMORY.md").unlink()
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "MEMORY.md").write_text("variant memory\n", encoding="utf-8")
    promote.apply_overlay(overlay)

    promote.rollback(cfg, snap.name)
    assert isolated_prompts["MEMORY.md"].read_text() == "variant memory\n"
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


# ── promote(): the orchestrator ──────────────────────────────────────────────

VARIANT = {"variant_id": "V_test", "description": "d", "hypothesis": "h"}


def test_promote_dry_run_writes_nothing(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("should not land\n", encoding="utf-8")
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1), dry_run=True)
    assert result["dry_run"] is True and result["applied_files"] == []
    assert not (cfg.paths.snapshots_dir).exists()
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text() == f"canonical {name}\n"


def test_promote_applies_snapshots_and_records(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("promoted soul\n", encoding="utf-8")
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert result["applied_files"] == ["SOUL.md"]
    assert result["snapshot_dir"] and Path(result["snapshot_dir"]).exists()
    assert isolated_prompts["SOUL.md"].read_text() == "promoted soul\n"
    assert (snap_soul := (Path(result["snapshot_dir"]) / "SOUL.md")).read_text() == "canonical SOUL.md\n"


def test_experiment_fact_records_the_promotion(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    fact = promote.write_experiment_fact(cfg, VARIANT, summary(0.80), summary(0.60), snap)
    text = fact.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "V_test" in text
    assert "**Baseline mean composite:** 0.600" in text
    assert "**Variant mean composite:** 0.800" in text
    assert "**Delta:** +0.200" in text
    assert cfg.paths.facts_experiments_dir in fact.parents


@pytest.mark.xfail(
    reason=(
        "REAL DEFECT (found 2026-09-06 while writing this file): promote() calls "
        "snapshot_current_prompts(), which mkdir()s unconditionally and never checks "
        "that the copy landed. A snapshot that cannot be written still yields a "
        "directory path, so promote() proceeds to overwrite live prompts with no "
        "rollback point. The ledger shows 26 of 83 promotions with no matching "
        "snapshot. Fix: raise if the snapshot dir holds no prompt files. "
        "xfailed rather than skipped so it reports once the guard exists."
    ),
    strict=False,
)
def test_promote_refuses_when_the_snapshot_cannot_be_written(isolated_prompts, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("promoted soul\n", encoding="utf-8")

    real_copy2 = __import__("shutil").copy2

    def failing_copy2(src, dst, *a, **kw):
        if "snapshots" in str(dst):
            raise OSError("disk full")
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(promote.shutil, "copy2", failing_copy2)
    promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    # If snapshotting failed there must be no promotion at all.
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


# ── the live default, asserted as a fact rather than assumed ─────────────────

def test_unpatched_canonical_targets_point_at_the_live_vault():
    """Characterization: promote()'s default targets are the real vault files.

    This is *why* every test above is isolated, and why the deploy path needs a
    human sign-off gate. Asserted through `_canonical_prompt_paths()` — the same
    construction `CANONICAL_PROMPTS` is built from — rather than reloading the
    module, which would mutate shared state mid-run. Read-only.
    """
    from scripts.autoresearch.common import _canonical_prompt_paths

    targets = _canonical_prompt_paths()
    assert set(targets) == set(PROMPT_NAMES)
    for name, path in targets.items():
        assert str(path).endswith(f"obsidian/lloyd/{name}"), path
        assert "tests" not in str(path) and "tmp" not in str(path)
