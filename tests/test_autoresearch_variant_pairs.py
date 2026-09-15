"""Backlog #876 — a round must bench the variants it materialized.

Commit `fe40af3` ("autoresearch: variants are anchored edits, not file echoes",
#446, 2026-09-09 13:08) reworked the materialize loop in
`scripts/autoresearch/run_round.py` and deleted one line without restoring it::

    -        variant_pairs.append((v["variant_id"], overlay))

`variant_pairs` then held the baseline alone from that point to
`_run_trials`, and the evaluation loop right after it skips the baseline — so
`decisions` was always empty, `evaluate_promotion` never ran, `promote` was
unreachable, and the `event: "decision"` append wrote zero rows. The live ledger
agrees: all 2,217 decision rows carry a real `V_*` id, and the newest is round
`R_20260908_181458`, the last round that ran before the commit.

Nothing caught it because every existing test drove the reader or the writer
directly. These two drive the loop that decides *what gets benched*: one the
function itself, one all the way through `run()` down to the trial matrix handed
to the bench runner and the decision rows the round appends to its own ledger —
the file `autoresearch_ledger_query` reads from a different process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from scripts.autoresearch import run_round, variant_sandbox as vs
from scripts.autoresearch.common import AutoresearchConfig, AutoresearchPaths

BASELINE_ID = "BASELINE_fixture"

#: Composite score per (variant_id, task_id). Baseline 0.40 on both tasks;
#: `V_a` beats it by +0.4000 on every task (promotable), `V_b` by +0.0300 on
#: average (below the 0.05 `promotion_min_composite_delta`, so it is judged and
#: held — a held variant still has to produce a decision row).
COMPOSITE = {
    (BASELINE_ID, "bench_a"): 0.40, (BASELINE_ID, "bench_b"): 0.40,
    ("V_a", "bench_a"): 0.80, ("V_a", "bench_b"): 0.80,
    ("V_b", "bench_a"): 0.42, ("V_b", "bench_b"): 0.44,
}


def make_cfg(tmp_path: Path, **over: Any) -> AutoresearchConfig:
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench",
        research_root=tmp_path / "research",
        rounds_dir=tmp_path / "rounds",
        ledger_path=tmp_path / "ledger.jsonl",
        variants_dir=tmp_path / "variants",
        snapshots_dir=tmp_path / "snapshots",
        facts_experiments_dir=tmp_path / "facts",
    )
    kw: dict[str, Any] = dict(
        paths=paths, default_model="primary", default_budget_minutes=60,
        max_variants_per_round=7, promotion_min_win_fraction=0.5,
        promotion_min_composite_delta=0.05, promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=2, targets=["prompts"],
    )
    kw.update(over)
    return AutoresearchConfig(**kw)


@pytest.fixture
def cfg(tmp_path):
    c = make_cfg(tmp_path)
    c.paths.ensure()
    c.paths.bench_dir.mkdir(parents=True, exist_ok=True)
    (c.paths.bench_dir / "bench_a.md").write_text(
        "---\nid: bench_a\ncategory: c\n---\nbody a\n", encoding="utf-8")
    (c.paths.bench_dir / "bench_b.md").write_text(
        "---\nid: bench_b\ncategory: c\n---\nbody b\n", encoding="utf-8")
    return c


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    """A stand-in vault: `Be direct.` appears twice, the rest once — all the
    exactly-once anchor rule needs to be testable (same fixture shape as
    `tests/test_autoresearch_sandbox.py`)."""
    root = tmp_path / "vault"
    root.mkdir()
    soul = root / "SOUL.md"
    soul.write_text(
        "# SOUL\n"
        "Be direct.\n"                        # ambiguous: appears twice
        "Never open with an apology.\n"
        "Be direct.\n",
        encoding="utf-8",
    )
    memory = root / "MEMORY.md"
    memory.write_text("# MEMORY\nA single unique note.\n", encoding="utf-8")
    monkeypatch.setattr(vs, "_canonical_prompt_paths", lambda: {
        "SOUL.md": soul, "MEMORY.md": memory,
    })
    return {"SOUL.md": soul, "MEMORY.md": memory}


def anchored(variant_id: str, anchor: str, replacement: str = "replaced", path: str = "SOUL.md"):
    return {
        "variant_id": variant_id,
        "target_surface": "prompts",
        "description": f"{variant_id} description",
        "hypothesis": f"{variant_id} hypothesis",
        "edits": [{"path": path, "anchor": anchor, "replacement": replacement}],
    }


def two_well_formed_and_two_unanchorable() -> list[dict[str, Any]]:
    """N = 2 variants whose anchor applies exactly once, M = 2 that cannot apply:
    one matching zero times (text the model invented), one matching twice
    (ambiguous). 4 proposals, 2 survivors."""
    return [
        anchored("V_a", "Never open with an apology.", "Open with the answer."),
        anchored("V_b", "A single unique note.", "A single clearer note.", path="MEMORY.md"),
        anchored("V_zero", "A span that does not exist at all."),
        anchored("V_two", "Be direct.", "Be blunt."),
    ]


def rows_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── clause 1: the materialize loop, on its own ───────────────────────────────

def test_the_bench_list_grows_by_one_per_variant_that_survives_the_anchored_edit(cfg, canonical):
    """`materialize_variants` returns `(variant_pairs, dropped)`; the pair list is
    the left operand of the (variant × task) matrix, so a variant that is absent
    from it is a variant no trial ever runs.

    Against main, where the append line is missing, this fails on the first
    assertion: the list comes back as the single baseline entry it has been since
    `fe40af3`.
    """
    variants = two_well_formed_and_two_unanchorable()
    pairs, dropped = run_round.materialize_variants(cfg, variants, (BASELINE_ID, cfg.paths.variants_dir))

    assert dropped == 2, "both unanchorable variants are counted as dropped"
    assert [vid for vid, _ in pairs] == [BASELINE_ID, "V_a", "V_b"]
    assert len(pairs) == 1 + len(variants) - dropped == 3

    by_id = dict(pairs)
    # The overlay a survivor maps to is the directory `materialize` wrote, holding
    # the *applied* text — not the proposal, and not the canonical file.
    assert by_id["V_a"] == cfg.paths.variants_dir / "V_a"
    assert (by_id["V_a"] / "SOUL.md").read_text(encoding="utf-8") == (
        "# SOUL\nBe direct.\nOpen with the answer.\nBe direct.\n")
    assert (by_id["V_b"] / "MEMORY.md").read_text(encoding="utf-8") == (
        "# MEMORY\nA single clearer note.\n")
    # #446's drop-whole semantics: a refused variant leaves no edited file behind.
    assert not (cfg.paths.variants_dir / "V_zero" / "SOUL.md").exists()
    assert not (cfg.paths.variants_dir / "V_two" / "SOUL.md").exists()
    # The baseline pair is passed through untouched — `run()` still needs its id.
    assert pairs[0] == (BASELINE_ID, cfg.paths.variants_dir)


def test_a_round_with_no_variants_still_benches_the_baseline_alone(cfg, canonical):
    """Zero proposals is not an error: the pair list is the baseline and nothing
    else, which is the only state in which today's rounds have been running."""
    pairs, dropped = run_round.materialize_variants(cfg, [], (BASELINE_ID, cfg.paths.variants_dir))
    assert pairs == [(BASELINE_ID, cfg.paths.variants_dir)]
    assert dropped == 0


# ── clause 1 + the acceptance, through the real run() ────────────────────────

def test_a_round_benches_every_survivor_and_writes_a_decision_row_for_each(
        cfg, canonical, monkeypatch, caplog):
    """Drives `run()` end to end with only the model calls stubbed: no trial
    reaches vLLM, no file leaves the vault. Everything between — the materialize
    loop, the harness split, the judge aggregation, the promotion gate, the report
    and the ledger append — is the code a live round runs.

    The stubs are the two process boundaries a test cannot cross from inside
    itself: `run_bench` (an HTTP completion against the primary vLLM slot) and
    `promote` (writes the canonical prompt files). `judge_trace` is a model call
    too; `aggregate_variant`, `evaluate_promotion` and `post_promotion_check` are
    not stubbed, so the decision each row reports is the real gate's verdict.
    """
    calls: dict[str, Any] = {}

    async def fake_run_bench(cfg_, variant_pairs, tasks, *, model, max_parallel, per_task_timeout):
        calls["variant_pairs"] = list(variant_pairs)
        calls["n_tasks"] = len(tasks)
        return [
            {"variant_id": vid, "task_id": t["id"], "status": "ok", "task_category": "c",
             "turns": 1, "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}
            for vid, _ in variant_pairs for t in tasks
        ]

    def fake_promote(cfg_, variant, overlay_dir, vsummary, bsummary, dry_run=False):
        calls["promote"] = (variant["variant_id"], Path(overlay_dir))
        return {"variant_id": variant["variant_id"], "snapshot_dir": str(cfg.paths.snapshots_dir / "snap"),
                "applied_files": ["SOUL.md"], "experiment_fact": None, "dry_run": False}

    monkeypatch.setattr(run_round, "load_config", lambda: cfg)
    monkeypatch.setattr(run_round, "propose_variants", lambda *a, **kw: two_well_formed_and_two_unanchorable())
    monkeypatch.setattr(run_round, "materialize_baseline", lambda c: (BASELINE_ID, c.paths.variants_dir))
    monkeypatch.setattr(run_round, "run_bench", fake_run_bench)
    monkeypatch.setattr(run_round, "run_bench_sdk", lambda *a, **kw: (_ for _ in ()).throw(AssertionError(
        "harness=direct must not route a task to the agent-loop runner")))
    monkeypatch.setattr(run_round, "promote", fake_promote)
    monkeypatch.setattr(run_round, "judge_trace", lambda task, t, rubric_model=None: {
        "composite_score": COMPOSITE[(t["variant_id"], t["task_id"])],
        "objective_score": 1.0, "rubric_score": 0.5, "rubric_overall": 0.5,
        "safety_critical": False, "safety_passed": True,
    })

    with caplog.at_level(logging.INFO, logger="autoresearch.run_round"):
        result = asyncio.run(run_round.run())

    # What the round handed the bench runner: 1 baseline + 2 survivors × 2 tasks.
    # Pre-fix this is 1 variant and 2 trials, which is the cost half of the bug —
    # a round finished in a third of the time because it benched nothing.
    assert [vid for vid, _ in calls["variant_pairs"]] == [BASELINE_ID, "V_a", "V_b"]
    assert calls["n_tasks"] == 2
    assert "running 3 variants × 2 tasks = 6 trials" in caplog.text
    assert "dropped 2 of 4 variants at anchored-edit apply" in caplog.text

    # The decisions the round reached, and the winner it promoted over them.
    assert [(d["variant_id"], d["should_promote"]) for d in result["decisions"]] == [
        ("V_a", True), ("V_b", False)]
    assert result["decisions"][1]["reason"] == "insufficient_delta (+0.0300 < 0.05)"
    assert result["variants_dropped"] == 2
    assert calls["promote"][0] == "V_a"
    assert calls["promote"][1] == cfg.paths.variants_dir / "V_a"
    assert result["promoted"]["variant_id"] == "V_a"

    # The row that never existed since fe40af3: one decision row per benched
    # variant, each naming a real variant, none naming the baseline.
    decisions = [r for r in rows_of(cfg.paths.ledger_path) if r.get("event") == "decision"]
    assert [(r["variant_id"], r["should_promote"], r["promoted"]) for r in decisions] == [
        ("V_a", True, True), ("V_b", False, False)]
    assert all(r["round_id"] == result["round_id"] for r in decisions)
    assert not any(r["variant_id"].startswith("BASELINE") for r in decisions)

    # ...and the human-readable half, on the bytes the round wrote: the PROMOTE
    # line is what `promotion_fp_rate.recorded_deltas` (#428) parses later, so a
    # promoted variant that never reaches the report is invisible to the sweep
    # that is supposed to catch it as a false positive.
    report = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")
    assert "`V_a`: PROMOTE" in report and "`V_b`: HOLD" in report
    from scripts.autoresearch import promotion_fp_rate as fp
    assert fp.recorded_deltas(cfg.paths.rounds_dir) == {(result["round_id"], "V_a"): 0.4}
