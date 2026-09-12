"""Backlog #429 — the post-promotion comparison that used to not exist.

Nothing ever looked at a promotion after it landed: the live ledger's 30,953
rows contain 0 mentions of rollback/revert against 65 `"promoted": true` decision
rows, and `run_round.py` never read a prior round's score. These tests pin the
two halves of the fix — a machine-readable `round_summary` row per round, and a
round report that names a baseline decline past the noise floor and attributes it
to the promotion that caused it.

The producer is switched off (`workers.sources.autoresearch` is `enabled: false`,
tracked as #682), so **no test here runs a round against a model**. The replay
test instead drops a real round's own report onto disk and reads the comparison
out of it; the wiring test drives `run_round.run()` over stubbed collaborators
only far enough to reach the record-and-surface step.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.autoresearch import post_promotion, run_round
from scripts.autoresearch.common import DEFAULT_NOISE_FLOOR, AutoresearchConfig, AutoresearchPaths

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_ROUNDS_DIR = REPO_ROOT / "_pipeline" / "research" / "rounds"

#: Round R_20260908_165252 is the last promotion in the live ledger and the one
#: the item names as the replay subject. Copied verbatim from
#: `_pipeline/research/rounds/R_20260908_165252.md`; the numbers below (baseline
#: 0.4364, winner `V_20260908_165359_f45720` at 0.6145, snapshot 20260908_165708)
#: are that round's, not invented. `test_the_vendored_report_still_matches_the_live_artifact`
#: cross-checks this copy whenever the gitignored tree is present.
R_20260908_165252 = """# Autoresearch round R_20260908_165252
- started_at: 2026-09-08T16:57:08Z
- model: primary
- harness: direct
- tasks: 11
- tasks on the harness runner: 0
- variants proposed: 7
- baseline mean composite: 0.4364

## Variant summaries
- `BASELINE_1788882602` (baseline): mean=0.4364, safety=pass, tasks=11
- `V_20260908_165312_19878f`: mean=0.5795, safety=pass, tasks=11
- `V_20260908_165328_9fe455`: mean=0.5259, safety=pass, tasks=11
- `V_20260908_165352_178542`: mean=0.6077, safety=pass, tasks=11
- `V_20260908_165359_f45720`: mean=0.6145, safety=pass, tasks=11
- `V_20260908_165409_451da2`: mean=0.4986, safety=pass, tasks=11
- `V_20260908_165423_dcea63`: mean=0.6395, safety=pass, tasks=11
- `V_20260908_165445_bde451`: mean=0.7068, safety=pass, tasks=11

## Promotion decisions
- `V_20260908_165312_19878f`: HOLD — insufficient_win_fraction (0.27 < 0.5)
- `V_20260908_165328_9fe455`: HOLD — insufficient_win_fraction (0.36 < 0.5)
- `V_20260908_165352_178542`: HOLD — insufficient_win_fraction (0.36 < 0.5)
- `V_20260908_165359_f45720`: PROMOTE — promote (delta=+0.1781, win_frac=0.64)
- `V_20260908_165409_451da2`: HOLD — insufficient_win_fraction (0.27 < 0.5)
- `V_20260908_165423_dcea63`: HOLD — insufficient_win_fraction (0.45 < 0.5)
- `V_20260908_165445_bde451`: HOLD — insufficient_win_fraction (0.45 < 0.5)

## Promoted
- variant: `V_20260908_165359_f45720`
- snapshot_dir: `/home/alansrobotlab/lloyd/_pipeline/research/snapshots/20260908_165708`
- applied_files: ['SOUL.md']
- experiment_fact: `/home/alansrobotlab/lloyd/_pipeline/vault-derived/facts/Experiments/V_20260908_165359_f45720/V_20260908_165359_f45720-experiment.md`
"""

PROMOTED = "V_20260908_165359_f45720"
SNAPSHOT_DIR = "/home/alansrobotlab/lloyd/_pipeline/research/snapshots/20260908_165708"


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
        default_budget_minutes=60,
        max_variants_per_round=7,
        promotion_min_win_fraction=0.5,
        promotion_min_composite_delta=0.05,
        promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=2,
        targets=["prompts"],
    )
    kw.update(over)
    return AutoresearchConfig(**kw)


@pytest.fixture
def world(tmp_path):
    """One rounds dir carrying the real R_20260908_165252 report, plus a ledger."""
    cfg = make_cfg(tmp_path)
    cfg.paths.rounds_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.rounds_dir / "R_20260908_165252.md").write_text(R_20260908_165252, encoding="utf-8")
    cfg.paths.ledger_path.touch()
    return cfg


def landed(vid: str = PROMOTED, snapshot: str = SNAPSHOT_DIR) -> dict:
    return {"variant_id": vid, "snapshot_dir": snapshot, "applied_files": ["SOUL.md"]}


def rows_of(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def acceptance_check(path: Path) -> list[dict]:
    """The item's own check, verbatim in spirit: rows carrying baseline_mean, a
    promoted variant id, and that variant's recorded mean. Triage measured 0."""
    return [
        r for r in rows_of(path)
        if r.get("baseline_mean") is not None
        and r.get("promoted_variant_id")
        and r.get("promoted_variant_mean") is not None
    ]


# ── clause 1: the machine-readable row ───────────────────────────────────────

def test_the_item_s_acceptance_check_now_finds_a_row_with_all_three_numbers(world):
    cfg = world
    row = post_promotion.record_round_summary(
        cfg, "R_20260909_060000", 0.4364, landed(), {"mean_composite": 0.6145}
    )
    assert row["event"] == post_promotion.ROUND_SUMMARY_EVENT
    assert (row["baseline_mean"], row["promoted_variant_id"], row["promoted_variant_mean"]) == (
        0.4364, PROMOTED, 0.6145,
    )
    # The check that had to stop reproducing:
    assert len(acceptance_check(cfg.paths.ledger_path)) == 1
    found = acceptance_check(cfg.paths.ledger_path)[0]
    assert found["round_id"] == "R_20260909_060000"
    assert found["snapshot_dir"] == SNAPSHOT_DIR


def test_a_round_that_promoted_nothing_still_records_its_baseline(world):
    """Per-round, not per-promotion: a null row is the evidence the check ran."""
    cfg = world
    post_promotion.record_round_summary(cfg, "R_20260909_060000", 0.5100, None, None)
    post_promotion.record_round_summary(
        cfg, "R_20260909_070000", 0.5000,
        {"variant_id": PROMOTED, "snapshot_dir": None, "refused": ["contract"]},
        {"mean_composite": 0.6145},
    )
    rows = post_promotion.round_summary_rows(cfg.paths.ledger_path)
    assert [r["round_id"] for r in rows] == ["R_20260909_060000", "R_20260909_070000"]
    assert all(r["baseline_mean"] is not None for r in rows)
    # A refused promotion landed nothing, so it must not be recorded as one.
    assert all(r["promoted_variant_id"] is None for r in rows)
    assert acceptance_check(cfg.paths.ledger_path) == []


def test_the_noise_floor_is_the_measured_one_and_config_can_override_it(tmp_path, monkeypatch):
    from scripts.autoresearch import common

    assert DEFAULT_NOISE_FLOOR == 0.1389  # backlog #324, 84-round cross-round std
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "autoresearch:\n"
        "  bench_dir: b\n  research_root: r\n  rounds_dir: r\n  ledger_path: r/l\n"
        "  variants_dir: r/v\n  snapshots_dir: r/s\n  facts_experiments_dir: r/f\n"
        "  promotion:\n    noise_floor: 0.20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(common, "CONFIG_PATH", cfg_file)
    assert common.load_config().promotion_noise_floor == 0.20
    cfg_file.write_text(
        "autoresearch:\n"
        "  bench_dir: b\n  research_root: r\n  rounds_dir: r\n  ledger_path: r/l\n"
        "  variants_dir: r/v\n  snapshots_dir: r/s\n  facts_experiments_dir: r/f\n",
        encoding="utf-8",
    )
    assert common.load_config().promotion_noise_floor == DEFAULT_NOISE_FLOOR


# ── clause 5: replay an existing round, never a live round ───────────────────

def test_a_real_round_report_reads_back_as_a_promotion_record(world):
    record = post_promotion.promotion_record_from_report(
        world.paths.rounds_dir / "R_20260908_165252.md"
    )
    assert record["promoted_variant_id"] == PROMOTED
    assert record["promoted_variant_mean"] == 0.6145
    assert record["baseline_mean"] == 0.4364
    assert record["snapshot_dir"] == SNAPSHOT_DIR


def test_a_round_with_no_promoted_block_is_not_a_promotion(world):
    path = world.paths.rounds_dir / "R_20260909_050000.md"
    path.write_text(
        "# Autoresearch round R_20260909_050000\n- baseline mean composite: 0.5000\n\n"
        "## Variant summaries\n- `V_x`: mean=0.4000, safety=pass, tasks=11\n\n"
        "## Promotion decisions\n- `V_x`: HOLD — insufficient_delta\n",
        encoding="utf-8",
    )
    assert post_promotion.promotion_record_from_report(path) is None
    assert post_promotion.last_promotion(
        world.paths.ledger_path, world.paths.rounds_dir
    )["round_id"] == "R_20260908_165252"


def test_replaying_the_existing_round_produces_a_beyond_noise_decline(world):
    """R_20260908_165252 promoted on a 0.6145 mean; a later round whose baseline
    comes in at 0.4364 has fallen 0.1781 — past the 0.1389 floor. Produced from
    the report on disk, with no round run."""
    prior = post_promotion.last_promotion(world.paths.ledger_path, world.paths.rounds_dir)
    comparison = post_promotion.compare(0.4364, prior, DEFAULT_NOISE_FLOOR)
    assert comparison["decline"] == pytest.approx(0.1781, abs=1e-4)
    assert comparison["regression"] is True

    text = "\n".join(post_promotion.report_section(comparison))
    assert "BASELINE DECLINE PAST NOISE FLOOR" in text
    assert PROMOTED in text
    assert "R_20260908_165252" in text


def test_the_replay_survives_the_report_being_the_only_store(world):
    """The 65 promotions that predate this module have no ledger row, so the
    report is their only record; the check must still see them."""
    assert post_promotion.round_summary_rows(world.paths.ledger_path) == []
    assert post_promotion.last_promotion(
        world.paths.ledger_path, world.paths.rounds_dir
    )["promoted_variant_id"] == PROMOTED


def test_the_vendored_report_still_matches_the_live_artifact():
    """Guards against the fixture above drifting into a lie about R_20260908_165252.

    `_pipeline` is gitignored, so on a box without it there is nothing to compare
    and the fixture assertions above stand on their own.
    """
    live = LIVE_ROUNDS_DIR / "R_20260908_165252.md"
    if not live.exists():
        return
    assert live.read_text(encoding="utf-8") == R_20260908_165252


# ── clauses 2 and 3: the decline is named and attributed ─────────────────────

def test_a_decline_inside_the_floor_is_recorded_but_not_called_a_regression(world):
    """0.6145 → 0.5200 is 0.0945: fresh bench prompts drawn differently, not a
    regression the promotion caused. Naming it would be the false alarm the floor
    exists to prevent (#324: 0% cross-round baseline overlap)."""
    prior = post_promotion.last_promotion(world.paths.ledger_path, world.paths.rounds_dir)
    comparison = post_promotion.compare(0.5200, prior, DEFAULT_NOISE_FLOOR)
    assert comparison["regression"] is False
    text = "\n".join(post_promotion.report_section(comparison))
    assert "BASELINE DECLINE PAST NOISE FLOOR" not in text
    assert "within the noise floor" in text
    assert PROMOTED in text  # the comparison is still visible, not silently dropped


def test_the_decline_line_names_the_snapshot_it_could_be_restored_from(world):
    """Clause 3: attribution to one promotion, including the rollback point.
    `promote()` already resolved it into `snapshot_dir`; this is where a human
    reading the report gets it back."""
    prior = post_promotion.last_promotion(world.paths.ledger_path, world.paths.rounds_dir)
    comparison = post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR)
    text = "\n".join(post_promotion.report_section(comparison))
    assert SNAPSHOT_DIR in text
    assert "20260908_165708" in text
    assert 'autoresearch_rollback(snapshot_ts="20260908_165708")' in text


def test_the_check_records_and_surfaces_and_never_restores(world, monkeypatch):
    """The item's out-of-scope clause in test form: option (b) means no file
    write is on this path at all."""
    from scripts.autoresearch import promote

    def trip(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("post-promotion check must not restore anything")

    monkeypatch.setattr(promote, "rollback", trip)
    monkeypatch.setattr(promote, "apply_overlay", trip)
    prior = post_promotion.last_promotion(world.paths.ledger_path, world.paths.rounds_dir)
    post_promotion.report_section(post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR))
    assert "Nothing was restored" in "\n".join(
        post_promotion.report_section(post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR))
    )
    assert "promotion_fp_rate" in "\n".join(
        post_promotion.report_section(post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR))
    )


def test_a_promotion_with_no_snapshot_says_so_rather_than_inventing_one(world):
    """A decline on a promotion with no rollback point must say there is none.

    Not hypothetical in shape: the ledger's 65 `"promoted": true` rows carry no
    snapshot field whatsoever — the rollback point lives only as prose in the
    round report — so any promotion recovered from the *ledger* alone arrives
    here with an empty `snapshot_dir`.
    """
    prior = {
        "source": "ledger", "round_id": "R_20260908_165252", "baseline_mean": 0.4364,
        "promoted_variant_id": PROMOTED, "promoted_variant_mean": 0.6145, "snapshot_dir": "",
    }
    comparison = post_promotion.compare(0.3000, prior, DEFAULT_NOISE_FLOOR)
    text = "\n".join(post_promotion.report_section(comparison))
    assert "no snapshot directory on record" in text
    assert "autoresearch_rollback" not in text


def test_the_comparison_excludes_the_round_asking(world):
    """A round that promoted something must not be compared against itself —
    its own baseline against its own winner is the delta it already reported."""
    cfg = world
    post_promotion.record_round_summary(
        cfg, "R_20260909_060000", 0.4364, landed("V_self"), {"mean_composite": 0.9}
    )
    prior = post_promotion.last_promotion(
        cfg.paths.ledger_path, cfg.paths.rounds_dir, exclude_round="R_20260909_060000"
    )
    assert prior["round_id"] == "R_20260908_165252"


def test_a_ledger_row_supersedes_the_report_for_the_same_round(world):
    cfg = world
    post_promotion.record_round_summary(
        cfg, "R_20260908_165252", 0.4364, landed("V_corrected"), {"mean_composite": 0.7}
    )
    prior = post_promotion.last_promotion(cfg.paths.ledger_path, cfg.paths.rounds_dir)
    assert prior["promoted_variant_id"] == "V_corrected"
    assert prior["source"] == "ledger"


def test_with_nothing_on_record_the_report_says_so_instead_of_bluffing(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.paths.rounds_dir.mkdir(parents=True)
    cfg.paths.ledger_path.touch()
    assert post_promotion.last_promotion(cfg.paths.ledger_path, cfg.paths.rounds_dir) is None
    text = "\n".join(post_promotion.report_section(None))
    assert "no promotion on record" in text


# ── the wiring: run_round actually does this on every round ──────────────────

def test_post_promotion_check_writes_the_row_and_the_lines_together(world):
    lines, row, comparison = run_round.post_promotion_check(
        world, "R_20260909_060000", 0.4364, landed(), {"mean_composite": 0.6145}
    )
    assert "BASELINE DECLINE PAST NOISE FLOOR" in "\n".join(lines)
    assert row["promoted_variant_id"] == PROMOTED
    assert comparison["prior_round_id"] == "R_20260908_165252"
    assert len(acceptance_check(world.paths.ledger_path)) == 1


def test_run_round_records_and_surfaces_on_the_live_path(world, monkeypatch):
    """The whole point is that no round has to remember to do this. Drives the
    real `run()` far enough to reach the record-and-surface step, with the model
    calls stubbed — no live round, per clause 5."""
    cfg = world
    cfg.paths.bench_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.bench_dir / "bench_a.md").write_text("---\nid: bench_a\ncategory: c\n---\nbody\n", encoding="utf-8")

    async def fake_trials(cfg_, variant_pairs, tasks, model, harness, max_parallel):
        traces = [
            {"variant_id": vid, "task_id": t["id"], "status": "ok", "task_category": "c",
             "turns": 1, "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}
            for vid, _ in variant_pairs for t in tasks
        ]
        return traces, []

    monkeypatch.setattr(run_round, "load_config", lambda: cfg)
    monkeypatch.setattr(run_round, "propose_variants", lambda *a, **kw: [
        {"variant_id": "V_new", "description": "d", "hypothesis": "h"}
    ])
    monkeypatch.setattr(run_round, "_run_trials", fake_trials)
    monkeypatch.setattr(run_round, "judge_trace", lambda task, t, rubric_model=None: {
        "composite_score": 0.5, "objective_score": 1.0, "rubric_overall": 0.5,
        "safety_critical": False, "safety_passed": True,
    })
    monkeypatch.setattr(run_round, "aggregate_variant", lambda vid, pairs: {
        "mean_composite": 0.4364 if vid.startswith("BASELINE") else 0.7000,
        "per_task": [], "task_count": len(pairs), "safety_passed": True,
    })
    monkeypatch.setattr(run_round, "evaluate_promotion", lambda c, b, v: (True, "promote (delta=+0.2636, win_frac=1.00)"))
    monkeypatch.setattr(run_round, "materialize_baseline", lambda c: ("BASELINE_fixture", c.paths.variants_dir))
    monkeypatch.setattr(run_round, "materialize", lambda c, v: c.paths.variants_dir)
    monkeypatch.setattr(run_round, "promote", lambda c, v, overlay, vs, bs, dry_run=False: {
        "variant_id": v["variant_id"], "snapshot_dir": SNAPSHOT_DIR, "applied_files": ["SOUL.md"],
        "experiment_fact": None, "dry_run": False,
    })

    result = asyncio.run(run_round.run(bench_limit=1))

    report = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")
    assert "## Post-promotion check" in report
    assert "BASELINE DECLINE PAST NOISE FLOOR" in report
    assert PROMOTED in report and "R_20260908_165252" in report
    assert "20260908_165708" in report

    row = [r for r in rows_of(cfg.paths.ledger_path)
           if r.get("event") == post_promotion.ROUND_SUMMARY_EVENT]
    assert len(row) == 1
    assert row[0]["round_id"] == result["round_id"]
    assert row[0]["baseline_mean"] == 0.4364
    assert row[0]["noise_floor"] == DEFAULT_NOISE_FLOOR
    # This stub world stops short of a landing — `promote` is never reached, so
    # there is no promoted variant to attribute and the row says `null` rather
    # than inventing one. The landed case, where all three numbers are present,
    # is pinned one frame closer to here, at the single function `run()` calls:
    # see test_post_promotion_check_writes_the_row_and_the_lines_together.
    assert row[0]["promoted_variant_id"] is None
    assert result["promoted"] is None
    assert result["post_promotion"]["prior_round_id"] == "R_20260908_165252"
    assert result["post_promotion"]["decline"] == pytest.approx(0.1781, abs=1e-4)


def test_the_new_ledger_event_is_invisible_to_the_existing_readers(world):
    """`round_summary` rows carry no `composite_score` and no `task_id`, so the
    hypothesis generator's loser scan, the FP sweep and the MCP ledger query all
    keep their old inputs. A new row shape that leaked into either would
    silently change #428's published denominator."""
    from scripts.autoresearch import promotion_fp_rate as fp

    post_promotion.record_round_summary(
        world, "R_20260909_060000", 0.4364, landed(), {"mean_composite": 0.6145}
    )
    before = [r for r in fp._rows(world.paths.ledger_path)]
    assert fp.per_task_rows(world.paths.ledger_path) == []
    assert fp.decision_rows(world.paths.ledger_path) == []
    assert len(before) == 1  # the row is in the file; it is simply not their kind
    from scripts.autoresearch import hypothesis_generator as hg

    losers = hg._recent_ledger_losers(world.paths.ledger_path)
    assert losers == []
