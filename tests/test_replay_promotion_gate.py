"""The retrospective replay of the #549 gate — clause (c), and the seam it reads across.

Why this file exists
--------------------
``scripts/autoresearch/replay_promotion_gate.py`` re-runs every recorded
autoresearch promotion under the two-condition rule using only the per-task
scores already in the ledger, so the gate can be validated against six weeks of
decisions with zero new bench runs. Its own exit code is the assertion that the
2026-09-08 "negative constraints" variant (``V_20260908_165359_f45720``) is
refused. Before this file nothing in the suite imported the module: the flip
count, the headroom / normalized-gain columns and the named-variant refusal were
all unexercised, so a regression in ``_summary`` — say one that folded
``safety_critical`` in wrongly, or read the category off the wrong key — would
have printed a plausible flip count over a corpus it had silently rebuilt
wrong.

Two halves, because the script has two jobs
-------------------------------------------
* **The arithmetic**, over a synthetic ledger and overlay dir small enough that
  every printed number is hand-computable: 4 replayed decisions, 3 flips, and
  the breakdown is exactly ``score_gate`` 2 / ``contract_guard`` 1.
* **The seam.** The script is a separate process that reads ``ledger.jsonl`` and
  the stored variant overlays off disk. The synthetic run goes through
  ``subprocess`` with ``--ledger``/``--variants``, so the file reading, the exit
  code and ``prompt_surface`` resolution all run cross-process — against tmp
  state, never live. Then one run against the live corpus, which is clause (c)'s
  actual claim.

Named-variant isolation
-----------------------
The synthetic corpus uses the real ``NAMED_VARIANT`` id as one of its own
variants, which is what lets the assertion be tested in both directions: refused
→ exit 0, promoted → exit 1. A guard whose assertion can only pass is not an
assertion.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.autoresearch import bench_split
from scripts.autoresearch import replay_promotion_gate as rpg
from scripts.autoresearch.common import AutoresearchConfig, AutoresearchPaths, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# The synthetic bench mirrors the live one's axis: 6 tasks in the targeted
# categories (replay/synthetic) and 5 in the veto categories
# (adversarial/safety) — 4/4/2/1 live, same 6-rotating-into-5 split arithmetic.
TASK_IDS = [f"bench_{i:03d}" for i in range(11)]
CATS = ["replay", "synthetic", "replay", "synthetic", "replay", "synthetic",
        "adversarial", "safety", "adversarial", "safety", "adversarial"]
TASKS = [{"id": tid, "category": cat} for tid, cat in zip(TASK_IDS, CATS)]

BASE_SCORE = 0.50  # every baseline task; headroom is then exactly 0.500


def _tasks() -> list[dict]:
    return [dict(t) for t in TASKS]


def _rows(round_id: str, variant_id: str, scores: dict[str, float],
          *, promoted: bool = False) -> list[dict]:
    """Per-task ledger rows in the shape `judge` writes: task_id, task_category,
    composite_score, safety_critical, safety_passed."""
    cat = {t["id"]: t["category"] for t in TASKS}
    return [{
        "round_id": round_id,
        "variant_id": variant_id,
        "task_id": tid,
        "task_category": cat[tid],
        "composite_score": score,
        "safety_critical": cat[tid] == "safety",
        "safety_passed": True,
        "promoted": promoted,
    } for tid, score in scores.items()]


def _scores(round_id: str, *, targeted_delta: float,
            heldout_delta: float) -> dict[str, float]:
    """Score every task by which slice it lands in THIS round, so the slice means
    are exact no matter which tasks the rotation picked."""
    split = bench_split.compute_split(_tasks(), round_id)
    targeted = set(split["targeted"])
    return {
        tid: BASE_SCORE + (targeted_delta if tid in targeted else heldout_delta)
        for tid in TASK_IDS
    }


def write_world(tmp_path: Path, *, named_round: str | None = None,
                named_verdict_variant: str = "V_promote") -> Path:
    """A five-round synthetic corpus: four replayed decisions and one skip.

    ``named_round`` puts :data:`rpg.NAMED_VARIANT` in that round id under the
    score profile of ``named_verdict_variant``, so the named-variant assertion
    can be driven either way.
    """
    ledger = tmp_path / "ledger.jsonl"
    variants = tmp_path / "variants"
    variants.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, str, float, float, bool]] = [
        # round, variant, targeted Δ, held-out Δ, overlay breaches the contract
        ("R_20260101_000000", "V_promote", +0.10, +0.05, False),
        ("R_20260102_000000", "V_decline", +0.10, -0.05, False),
        ("R_20260103_000000", "V_flat", +0.00, +0.05, False),
        ("R_20260104_000000", "V_contract", +0.10, +0.05, True),
    ]
    lines: list[str] = []
    for round_id, vid, dt, dh, breach in plan:
        base = {tid: BASE_SCORE for tid in TASK_IDS}
        lines += [_rows(round_id, "BASELINE_V", base)]
        var = _scores(round_id, targeted_delta=dt, heldout_delta=dh)
        lines += [_rows(round_id, vid, var, promoted=True)]
        vdir = variants / vid
        vdir.mkdir(parents=True, exist_ok=True)
        if breach:
            # `prompt_surface.check_contract` refuses this on the gate-role and
            # load-bearing markers — a real overlay, not a stubbed refusal.
            (vdir / "SOUL.md").write_text("# Title\n\nsome prose\n", encoding="utf-8")
    # A round whose promoted variant has rows but whose baseline rows never made
    # it into the ledger: the replay must skip it, not score it as a refusal.
    lines += [_rows("R_20260105_000000", "V_orphan",
                    {tid: 0.9 for tid in TASK_IDS}, promoted=True)]

    if named_round:
        dt, dh = (+0.10, +0.05) if named_verdict_variant == "V_promote" else (+0.10, -0.05)
        lines += [_rows(named_round, "BASELINE_V", {t: BASE_SCORE for t in TASK_IDS})]
        lines += [_rows(named_round, rpg.NAMED_VARIANT,
                        _scores(named_round, targeted_delta=dt, heldout_delta=dh),
                        promoted=True)]

    ledger.write_text("\n".join(json.dumps(r) for group in lines for r in group) + "\n",
                      encoding="utf-8")
    return ledger


def run_cli(ledger: Path, variants: Path, *extra: str) -> subprocess.CompletedProcess:
    """The script as it actually runs: a fresh interpreter, reading state off disk."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.autoresearch.replay_promotion_gate",
         "--ledger", str(ledger), "--variants", str(variants), *extra],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )


@pytest.fixture
def cfg(tmp_path) -> AutoresearchConfig:
    return AutoresearchConfig(
        paths=AutoresearchPaths(
            bench_dir=tmp_path / "bench", research_root=tmp_path / "research",
            rounds_dir=tmp_path / "rounds", ledger_path=tmp_path / "ledger.jsonl",
            variants_dir=tmp_path / "variants", snapshots_dir=tmp_path / "snapshots",
            facts_experiments_dir=tmp_path / "facts",
        ),
        default_model="primary", default_budget_minutes=120, max_variants_per_round=7,
        promotion_min_win_fraction=0.5, promotion_min_composite_delta=0.05,
        promotion_require_safety_pass=True, tool_allowlist_consecutive_wins=2,
        targets=["prompts"],
    )


# ── _load_rows: the file-reading half ────────────────────────────────────────

def test_load_rows_keeps_valid_rows_and_drops_blanks_and_garbage(tmp_path):
    """One malformed line must not lose the corpus, and must not become a row.

    The live ledger is append-only JSONL written by several jobs; a torn final
    line is normal. Counting it as a row would put a score-less phantom into
    every downstream aggregate.
    """
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"round_id": "R_1", "task_id": "bench_000", "composite_score": 0.5}\n'
        "\n"
        "   \n"
        "{not json\n"
        '{"round_id": "R_1", "task_id": "bench_001", "composite_score": 0.7}',  # no trailing \n
        encoding="utf-8")
    rows = rpg._load_rows(ledger)
    assert [r["task_id"] for r in rows] == ["bench_000", "bench_001"]


def test_load_rows_returns_nothing_on_an_empty_ledger(tmp_path):
    """An empty corpus is a real answer; it is "no ledger" that `main` refuses with
    exit 2, and the two must stay distinguishable — a caller that reads an absent
    file as zero rows reports an unreadable ledger as a clean replay."""
    empty = tmp_path / "ledger.jsonl"
    empty.write_text("", encoding="utf-8")
    assert rpg._load_rows(empty) == []


# ── _summary: rebuilding the judge's shape from per-task rows ────────────────

def test_summary_rebuilds_mean_per_task_and_category(tmp_path):
    """The gate decides on `per_task[].category`, so a lost category is a lost
    slice — the split would fall back to nothing and the replay would promote."""
    rows = _rows("R_1", "V_a", {"bench_000": 0.4, "bench_006": 0.8})
    s = rpg._summary(rows)
    assert s["task_count"] == 2
    assert s["mean_composite"] == pytest.approx(0.6)
    assert [p["task_id"] for p in s["per_task"]] == ["bench_000", "bench_006"]
    assert [p["category"] for p in s["per_task"]] == ["replay", "adversarial"]


def test_summary_drops_rows_with_no_score_or_no_task_id():
    """A row missing its score or its id cannot contribute a mean; if it did, the
    mean would be computed over a population the round never ran."""
    rows = [{"round_id": "R", "task_id": "bench_000", "task_category": "replay",
             "composite_score": None},
            {"round_id": "R", "task_id": None, "task_category": "replay",
             "composite_score": 0.9},
            {"round_id": "R", "task_id": "bench_001", "task_category": "replay",
             "composite_score": "0.9"}]
    s = rpg._summary(rows)
    assert s["per_task"] == []
    assert s["mean_composite"] == 0.0 and s["task_count"] == 0


def test_the_safety_veto_needs_both_critical_and_a_failed_probe():
    """`safety_critical` is set on exactly one live task, so an unfailing probe on
    any of the other ten must not fail the summary — and a critical one that did
    fail must."""
    critical_fail = [{"task_id": "bench_010", "task_category": "safety",
                      "composite_score": 0.5, "safety_critical": True,
                      "safety_passed": False}]
    optional_fail = [{**critical_fail[0], "safety_critical": False}]
    assert rpg._summary(critical_fail)["safety_passed"] is False
    assert rpg._summary(optional_fail)["safety_passed"] is True


# ── replay: the flip that clause (c) counts ──────────────────────────────────

def test_a_heldout_regression_flips_a_recorded_promotion_to_the_score_gate(cfg, tmp_path):
    ledger = write_world(tmp_path)
    decisions = {d["variant_id"]: d for d in rpg.replay(cfg, ledger, tmp_path / "variants")}
    d = decisions["V_decline"]
    assert d["would_promote"] is False
    assert d["refused_by"] == "score_gate"
    assert d["score_reason"].startswith("heldout_decline")
    # Hand-computed: baseline 0.50 everywhere, targeted +0.10, veto -0.05.
    assert d["raw_delta"] == pytest.approx(0.10)
    assert d["heldout_delta"] == pytest.approx(-0.05)
    assert d["headroom"] == pytest.approx(0.50)
    assert d["normalized_gain"] == pytest.approx(0.20)


def test_a_targeted_tie_flips_it_too_even_though_the_veto_slice_rose(cfg, tmp_path):
    """Condition one is the gain condition; a flat targeted slice is not a gain
    however good the veto looks, and the old averaged gate would have promoted."""
    ledger = write_world(tmp_path)
    d = {x["variant_id"]: x for x in rpg.replay(cfg, ledger, tmp_path / "variants")}["V_flat"]
    assert d["would_promote"] is False and d["refused_by"] == "score_gate"
    assert d["score_reason"].startswith("targeted_no_gain")
    assert d["raw_delta"] == pytest.approx(0.0)
    assert d["heldout_delta"] == pytest.approx(0.05)


def test_a_contract_breaching_overlay_flips_the_decision_to_the_guard(cfg, tmp_path):
    """The replayed verdict is the whole accept path: the score gate passes this
    variant (identical scores to `V_promote`) and `prompt_surface` refuses it, so
    `refused_by` distinguishes the two conditions rather than conflating them."""
    ledger = write_world(tmp_path)
    d = {x["variant_id"]: x for x in rpg.replay(cfg, ledger, tmp_path / "variants")}["V_contract"]
    assert d["would_promote"] is False and d["refused_by"] == "contract_guard"
    assert d["score_reason"].startswith("promote")
    assert any("gate roles" in line or "removes behaviour" in line
               for line in d["contract_refusals"]), d["contract_refusals"]


def test_a_round_with_no_baseline_rows_is_skipped_not_counted_as_a_refuse(cfg, tmp_path):
    """Unmeasurable is not "refused" — a skip that read as a veto would inflate the
    flip count with rounds the ledger simply never scored."""
    ledger = write_world(tmp_path)
    d = {x["variant_id"]: x for x in rpg.replay(cfg, ledger, tmp_path / "variants")}["V_orphan"]
    assert d["skipped"] == "no baseline rows in the ledger"
    assert "would_promote" not in d


def test_the_replay_derives_the_same_split_the_gate_would_use(cfg, tmp_path):
    """`split_hash` is recomputed per round from the round id, and `verify` must
    accept it — a replay whose split fails its own hash is re-deriving something
    the gate never ran on."""
    ledger = write_world(tmp_path)
    decisions = rpg.replay(cfg, ledger, tmp_path / "variants")
    assert all(d["split_hash"] for d in decisions if not d["skipped"])
    assert len({d["split_hash"] for d in decisions if not d["skipped"]}) == len(decisions) - 1


# ── report: the numbers a human reads ────────────────────────────────────────

def test_the_report_prints_the_flip_count_and_its_breakdown(cfg, tmp_path, capsys):
    ledger = write_world(tmp_path)
    decisions = rpg.replay(cfg, ledger, tmp_path / "variants")
    rc = rpg.report(decisions, by_task=False)
    out = capsys.readouterr().out
    # 4 replayed (V_orphan skipped), 3 flip: two on the score gate, one on the guard.
    assert "recorded promotions replayed: 4   (skipped, unmeasurable: 1)" in out
    assert "flip to REJECT under the #549 gate: 3 (75%)" in out
    assert "still promote: 1" in out
    assert "score_gate      2" in out
    assert "contract_guard  1" in out
    assert rc == 1, "the synthetic corpus has no named variant, so the assertion cannot run"
    assert "NOT IN THE REPLAYED CORPUS" in out


def test_each_row_prints_headroom_and_normalized_gain_beside_the_raw_delta(cfg, tmp_path, capsys):
    """HarnessOpt-Bench's requirement, clause (c): a raw delta with no headroom
    cannot distinguish +0.05 off a 0.50 seed from +0.05 off a 0.85 one."""
    ledger = write_world(tmp_path)
    rpg.report(rpg.replay(cfg, ledger, tmp_path / "variants"), by_task=False)
    lines = capsys.readouterr().out.splitlines()
    row = next(" ".join(l.split()) for l in lines if l.startswith("R_20260101_000000"))
    assert row == ("R_20260101_000000 V_promote +0.1000 0.500 +20.0% +0.0500 PROMOTE"), row
    decline = next(" ".join(l.split()) for l in lines if l.startswith("R_20260102_000000"))
    assert decline == ("R_20260102_000000 V_decline +0.1000 0.500 +20.0% -0.0500 "
                       "REFUSE (score_gate)"), decline


def test_by_task_counts_only_the_veto_regressions_among_the_flips(cfg, tmp_path, capsys):
    """`--by-task` answers "which held-out regressions caused the flips", so it
    counts the held-out refusals and not the gain refusals — `V_flat` flipped on
    condition one with a rising veto slice and must not be tallied as a veto
    regression, or the breakdown would read as 2 vetoes out of 3 flips."""
    ledger = write_world(tmp_path)
    rpg.report(rpg.replay(cfg, ledger, tmp_path / "variants"), by_task=True)
    tail = capsys.readouterr().out.split("which veto tasks moved")[1]
    assert "heldout_decline      1" in tail
    assert "targeted_no_gain" not in tail


# ── the named-variant assertion, in both directions ──────────────────────────

def test_the_named_variant_assertion_passes_when_the_replay_refuses_it(cfg, tmp_path, capsys):
    """Refused by the held-out slice here — exit 0, verdict line printed."""
    ledger = write_world(tmp_path, named_round="R_20260106_000000",
                         named_verdict_variant="V_decline")
    decisions = rpg.replay(cfg, ledger, tmp_path / "variants")
    rc = rpg.report(decisions, by_task=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert f"named variant {rpg.NAMED_VARIANT}:" in out
    assert "replayed verdict: REFUSE | refused_by=score_gate" in out


def test_the_named_variant_assertion_can_fail_when_the_replay_promotes_it(cfg, tmp_path, capsys):
    """The guard that cannot fail is not a guard: the same script, same exit path,
    with a named variant the gate accepts, must return non-zero."""
    ledger = write_world(tmp_path, named_round="R_20260106_000000",
                         named_verdict_variant="V_promote")
    decisions = rpg.replay(cfg, ledger, tmp_path / "variants")
    rc = rpg.report(decisions, by_task=False)
    assert rc == 1
    assert "ASSERTION FAILED" in capsys.readouterr().out


def test_the_report_says_when_the_split_did_not_catch_the_named_variant(cfg, tmp_path, capsys):
    """The finding, printed rather than buried: on the live corpus the veto slice
    ROSE under the 09-08 variant and the contract guard is what refuses it. A
    report that let a reader conclude "the split stopped it" would be false."""
    ledger = write_world(tmp_path, named_round="R_20260106_000000",
                         named_verdict_variant="V_decline")
    d = next(x for x in rpg.replay(cfg, ledger, tmp_path / "variants")
             if x["variant_id"] == rpg.NAMED_VARIANT)
    d["refused_by"] = "contract_guard"  # as measured on the live corpus
    rpg.report([d], by_task=False)
    assert "the held-out slice did NOT catch this one" in capsys.readouterr().out


# ── the seam: a separate process reading the ledger and the overlays ─────────

def test_the_cli_reads_its_ledger_and_overlays_off_disk_in_a_fresh_process(tmp_path):
    """No stub, no import: a child interpreter, given only two paths, reproduces
    the whole flip breakdown — including the one decision that only the overlay on
    disk can refuse. An import-level test could not tell that the overlay was read
    rather than imagined."""
    ledger = write_world(tmp_path)
    proc = run_cli(ledger, tmp_path / "variants", "--json")
    assert proc.returncode == 0, proc.stderr
    decisions = {d["variant_id"]: d for d in json.loads(proc.stdout)}
    assert (decisions["V_promote"]["would_promote"],
            decisions["V_decline"]["refused_by"],
            decisions["V_flat"]["refused_by"],
            decisions["V_contract"]["refused_by"],
            decisions["V_orphan"]["skipped"]) == (
        True, "score_gate", "score_gate", "contract_guard",
        "no baseline rows in the ledger")

    text = run_cli(ledger, tmp_path / "variants")
    assert "flip to REJECT under the #549 gate: 3 (75%)" in text.stdout
    assert text.returncode == 1, "no named variant in this corpus → the assertion cannot run"


def test_the_cli_exits_zero_when_the_named_variant_it_must_refuse_is_on_disk(tmp_path):
    """The exit code is clause (c)'s assertion, produced by the child process
    against a real overlay dir, not by a test calling the function."""
    ledger = write_world(tmp_path, named_round="R_20260106_000000",
                         named_verdict_variant="V_decline")
    proc = run_cli(ledger, tmp_path / "variants")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"named variant {rpg.NAMED_VARIANT}:" in proc.stdout


def test_the_cli_names_the_ledger_that_is_missing_rather_than_reporting_zero(tmp_path):
    """`no ledger` must not read as "nothing to replay": an empty corpus and an
    absent file are different answers, and only one of them is exit 2."""
    proc = run_cli(tmp_path / "absent.jsonl", tmp_path / "variants")
    assert proc.returncode == 2
    assert "no ledger at" in proc.stderr


# ── the live corpus: clause (c)'s actual claim ───────────────────────────────

LIVE_LEDGER = load_config().paths.ledger_path


@pytest.mark.skipif(not LIVE_LEDGER.exists(),
                    reason="no autoresearch ledger on this machine")
def test_the_live_replay_refuses_the_20260908_negative_constraints_variant():
    """Clause (c), end to end, over the recorded corpus — zero new bench runs.

    Read-only: `--variants`/`--ledger` are not passed, so this runs against the
    real paths. Pinned to the historical fact, not to the flip count: the corpus
    is append-only, so the count grows the moment the loop is re-armed, while the
    09-08 decision cannot change.

    Measured 2026-09-19: 67 promotions replayed, 0 skipped, 42 flip to REJECT
    (27 contract_guard, 15 score_gate).
    """
    cfg = load_config()
    decisions = rpg.replay(cfg, cfg.paths.ledger_path, cfg.paths.variants_dir)
    replayed = [d for d in decisions if not d.get("skipped")]
    assert len(replayed) >= 65, (
        f"the acceptance names a 65-promotion corpus; the replay found {len(replayed)}")
    assert not any(d.get("skipped") for d in decisions), (
        "every recorded promotion has baseline rows; a skip means the replay "
        "silently lost a decision it should have graded")

    named = next((d for d in replayed if d["variant_id"] == rpg.NAMED_VARIANT), None)
    assert named is not None, "the 09-08 promotion is not in the replayed corpus"
    assert named["would_promote"] is False
    # Stated plainly because it is the finding, not a technicality: on this corpus
    # the veto slice ROSE under that variant (held-out +0.1470, targeted +0.2042),
    # so the guard that refuses it is the contract check #377 landed. The split
    # joins that guard; it did not, by itself, stop the 09-08 promotion.
    assert named["refused_by"] == "contract_guard"
    assert named["heldout_delta"] > 0
    assert named["headroom"] == pytest.approx(0.70)
    assert named["normalized_gain"] == pytest.approx(0.29167, abs=1e-4)


@pytest.mark.skipif(not LIVE_LEDGER.exists(),
                    reason="no autoresearch ledger on this machine")
def test_the_live_replay_flips_a_substantial_share_of_recorded_promotions():
    """The gate is not decorative: replaying it over the corpus must refuse
    recorded promotions, or clause (c)'s "how many flip" question has no content.

    A lower bound, not the exact count — see the test above for why the count
    moves and the 09-08 decision does not.
    """
    cfg = load_config()
    replayed = [d for d in rpg.replay(cfg, cfg.paths.ledger_path, cfg.paths.variants_dir)
                if not d.get("skipped")]
    flips = [d for d in replayed if not d["would_promote"]]
    assert len(flips) >= 30, f"only {len(flips)} of {len(replayed)} flip"
    assert all(d["refused_by"] in ("score_gate", "contract_guard") for d in flips)


@pytest.mark.skipif(not LIVE_LEDGER.exists(),
                    reason="no autoresearch ledger on this machine")
def test_the_script_run_against_live_state_exits_zero():
    """The same command the item's step 4 asks for, as a person would type it."""
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.autoresearch.replay_promotion_gate"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "replayed verdict: REFUSE" in proc.stdout
    assert "recorded promotions replayed: " in proc.stdout
