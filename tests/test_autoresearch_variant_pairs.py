"""Backlog #876 — a round must bench the variants it materialized.

Commit `fe40af3` ("autoresearch: variants are anchored edits, not file echoes",
#446, 2026-09-09 13:08) reworked the materialize loop in
`scripts/autoresearch/run_round.py` and deleted one line without restoring it::

    -        variant_pairs.append((v["variant_id"], overlay))

`variant_pairs` then held the baseline alone from that point to
`_run_trials`, and the evaluation loop right after it skips the baseline — so
`decisions` was always empty, `evaluate_promotion` never ran, `promote` was
unreachable, and the `event: "decision"` append wrote zero rows. Backlog #876's
triage measured the live ledger on 2026-09-13: all 2,217 `event: "decision"`
rows carry a real `V_*` id and the newest belongs to round `R_20260908_181458`,
the last round that ran before that commit. Those are its numbers, not this
file's — `_pipeline/` is gitignored, so nothing here re-measures them.

Nothing caught the defect because every existing test drove the reader or the
writer directly. The four tests here drive what the old ones never touched: two
on `materialize_variants` itself, one all the way through `run()` down to the
trial matrix handed to the bench runner and the decision rows the round appends
to its ledger — the file `autoresearch_ledger_query` reads from a different
process — and one on the round's write set, which is what keeps clause 3's
re-arming out of code.
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
    """`materialize_variants` returns `(variant_pairs, dropped_by_surface)`; the
    pair list is the left operand of the (variant × task) matrix, so a variant
    that is absent from it is a variant no trial ever runs.

    Against main, where the append line is missing, this fails on the first
    assertion: the list comes back as the single baseline entry it has been since
    `fe40af3`.
    """
    variants = two_well_formed_and_two_unanchorable()
    pairs, drops = run_round.materialize_variants(cfg, variants, (BASELINE_ID, cfg.paths.variants_dir))
    dropped = sum(drops.values())

    assert dropped == 2, "both unanchorable variants are counted as dropped"
    # #680: the two failures both aimed at SOUL.md, and the count says so — with
    # MEMORY.md present at zero because a variant proposed it and survived.
    assert drops == {"SOUL.md": 2, "MEMORY.md": 0}
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
    pairs, drops = run_round.materialize_variants(cfg, [], (BASELINE_ID, cfg.paths.variants_dir))
    assert pairs == [(BASELINE_ID, cfg.paths.variants_dir)]
    assert drops == {}, (
        "nothing proposed, so no surface is even named: the empty mapping is what "
        "tells a reader this round had no MEMORY variants to drop, as opposed to "
        "MEMORY variants that all applied (#680)"
    )


# ── clause 1 + the acceptance, through the real run() ────────────────────────

def drive_round(cfg, monkeypatch, caplog, variants_factory=two_well_formed_and_two_unanchorable):
    """Drive `run()` end to end with only the model calls stubbed: no trial
    reaches vLLM, no file leaves the vault. Everything between — the materialize
    loop, the harness split, the judge aggregation, the promotion gate, the report
    and the ledger append — is the code a live round runs.

    The stubs are the two process boundaries a test cannot cross from inside
    itself: `run_bench` (an HTTP completion against the primary vLLM slot) and
    `promote` (writes the canonical prompt files). `judge_trace` is a model call
    too; `aggregate_variant`, `evaluate_promotion` and `post_promotion_check` are
    not stubbed, so the decision each row reports is the real gate's verdict.

    Proposes 2 well-formed + 2 unanchorable variants by default, so a round that
    works reaches `promote` with `V_a`. `variants_factory` overrides the proposal
    list for a round that needs a different drop shape (#680). Any survivor must
    be named in COMPOSITE, since `judge_trace` looks its score up by id.
    Returns `(result, calls)`.
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
    monkeypatch.setattr(run_round, "propose_variants", lambda *a, **kw: variants_factory())
    def fake_materialize_baseline(c):
        """The real one returns a fresh subdirectory of the variants dir, so the
        baseline overlay must not be that dir itself."""
        overlay = c.paths.variants_dir / BASELINE_ID
        overlay.mkdir(parents=True, exist_ok=True)
        return BASELINE_ID, overlay

    monkeypatch.setattr(run_round, "materialize_baseline", fake_materialize_baseline)
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
    return result, calls


def test_a_round_benches_every_survivor_and_writes_a_decision_row_for_each(
        cfg, canonical, monkeypatch, caplog):
    """The round a live producer would run: 2 promotable/hold survivors benched,
    the 2 unanchorable ones dropped, one decision row each."""
    result, calls = drive_round(cfg, monkeypatch, caplog)

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


# ── clause 3: re-arming is not this code's to take ───────────────────────────

#: The gate a person flips after #876 lands (`workers.sources.autoresearch.enabled`
#: in `config.yaml`). `config.yaml` is on the self-modification loop's never-touch
#: list, so a round that could rewrite it could arm itself — one round per
#: `interval_seconds` against the single primary vLLM slot, with nobody deciding.
def test_a_round_writes_nothing_to_config_yaml(cfg, canonical, monkeypatch, caplog, tmp_path):
    """Clause 3, as behaviour rather than as a diff stat.

    The live-path test above drives a round that *promotes*, which is the strongest
    temptation for a future change to reach for the config: the round that just
    landed a winner is also the round that would want to arm its own producer. So
    drive that same round with `common.CONFIG_PATH` pointed at a copy of a config
    whose autoresearch source is switched off, and require the round to leave that
    file byte-for-byte alone.

    The static half is what catches the variant this run could not: a round that
    flips the flag through a path the test never pointed at. Every reference to
    `CONFIG_PATH` in the package must be the definition or a read.
    """
    from scripts.autoresearch import common

    config_copy = tmp_path / "config.yaml"
    config_copy.write_text(
        "workers:\n"
        "  sources:\n"
        "    autoresearch:\n"
        "      enabled: false\n"
        "      interval_seconds: 3600\n",
        encoding="utf-8",
    )
    before = config_copy.read_bytes()
    monkeypatch.setattr(common, "CONFIG_PATH", config_copy)

    drive_round(cfg, monkeypatch, caplog)

    assert config_copy.read_bytes() == before, (
        "the round wrote its own config: `workers.sources.autoresearch.enabled` is "
        "a person's call (backlog #876 clause 3)"
    )
    assert b"enabled: false" in config_copy.read_bytes()

    offenders = []
    for module in sorted((Path(common.__file__).parent).glob("*.py")):
        for lineno, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            if "CONFIG_PATH" not in line:
                continue
            if "CONFIG_PATH =" in line or ".read_text(" in line:
                continue
            offenders.append(f"{module.name}:{lineno}: {line.strip()}")
    assert offenders == [], (
        "a write path to config.yaml appeared in scripts/autoresearch; the loop may "
        f"never write that file. New references: {offenders}"
    )


# ── #680 clause 3: a drop names the surface it fell on ───────────────────────
#
# #680's premise is that MEMORY-targeted variants were dropped at a rate no one
# could see, because `variants_dropped` was one number for every surface at
# once. The count below is the observable half of that fix: it is seeded with
# every surface a round PROPOSED, so "MEMORY.md: 0" means its variants applied
# and an absent key means none were ever tried — the two shapes the aggregate
# could not tell apart.

@pytest.fixture
def memory_can_fail_to_anchor(tmp_path, monkeypatch):
    """A vault where MEMORY.md can produce both failure shapes — an anchor that
    matches zero times and one that matches twice — while SOUL.md's one variant
    applies. That is the asymmetry #680 is about, and the smallest fixture that
    makes a drop attributable to a named surface."""
    root = tmp_path / "vault680"
    root.mkdir()
    soul = root / "SOUL.md"
    soul.write_text("# SOUL\nNever open with an apology.\n", encoding="utf-8")
    memory = root / "MEMORY.md"
    memory.write_text(
        "# MEMORY\n"
        "A single unique note.\n"
        "Duplicated note.\n"
        "Duplicated note.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(vs, "_canonical_prompt_paths", lambda: {
        "SOUL.md": soul, "MEMORY.md": memory,
    })
    return {"SOUL.md": soul, "MEMORY.md": memory}


def two_memory_drops_and_one_soul_survivor() -> list[dict[str, Any]]:
    """1 SOUL variant that applies, 2 MEMORY variants that cannot: one anchor
    exists nowhere in the file, one appears twice."""
    return [
        anchored("V_a", "Never open with an apology.", "Open with the answer."),
        anchored("V_m_zero", "A calendar note that is not in the file.", path="MEMORY.md"),
        anchored("V_m_two", "Duplicated note.", "A clearer note.", path="MEMORY.md"),
    ]


def test_a_drop_is_counted_against_the_surface_the_variant_aimed_at(cfg, memory_can_fail_to_anchor):
    """2 MEMORY drops and 0 SOUL drops report `MEMORY.md: 2, SOUL.md: 0`.

    A pre-fix `materialize_variants` hands back the bare int 2, which is exactly
    the reading #680 says is unusable: it cannot say whether the failures were
    MEMORY's or SOUL's, so "MEMORY.md should not be the only surface that ever
    fails to apply" could not be checked against any round.
    """
    pairs, drops = run_round.materialize_variants(
        cfg, two_memory_drops_and_one_soul_survivor(), (BASELINE_ID, cfg.paths.variants_dir))

    assert drops == {"MEMORY.md": 2, "SOUL.md": 0}
    assert sum(drops.values()) == 2
    assert [vid for vid, _ in pairs] == [BASELINE_ID, "V_a"]
    # The survivor's overlay is the applied MEMORY text — proof the MEMORY edits
    # were attempted against that file and are the ones that failed.
    assert (cfg.paths.variants_dir / "V_a" / "SOUL.md").read_text(encoding="utf-8") == (
        "# SOUL\nOpen with the answer.\n")


def test_the_round_report_names_the_surface_of_its_drops_without_the_log(
        cfg, memory_can_fail_to_anchor, monkeypatch, caplog):
    """The same mapping reaches the round report's bytes.

    The report is the artifact a person or a sweep reads (`promotion_fp_rate`
    parses this very file); the log line is not kept per round. A reader must be
    able to see `MEMORY.md: 2, SOUL.md: 0` without opening a log, which is what
    clause 3 asks for.
    """
    result, _calls = drive_round(
        cfg, monkeypatch, caplog, variants_factory=two_memory_drops_and_one_soul_survivor)

    assert result["variants_dropped"] == 2, "the aggregate is still there"
    assert result["variants_dropped_by_surface"] == {"MEMORY.md": 2, "SOUL.md": 0}

    report = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")
    assert "- variants dropped by surface: MEMORY.md: 2, SOUL.md: 0" in report
    assert "dropped 2 of 3 variants at anchored-edit apply (MEMORY.md: 2, SOUL.md: 0)" in caplog.text
