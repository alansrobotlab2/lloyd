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
import subprocess
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
    """Mechanics only. The overlay must be a contract the guard accepts, or the
    promotion is refused before it applies anything — see
    `tests/test_prompt_surface_guard.py`, which pins the refusal itself."""
    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text(GOOD_CONTRACT, encoding="utf-8")
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert result.get("refused") is None, result.get("refused")
    assert result["applied_files"] == ["SOUL.md"]
    assert result["snapshot_dir"] and Path(result["snapshot_dir"]).exists()
    assert isolated_prompts["SOUL.md"].read_text() == GOOD_CONTRACT
    assert (snap_soul := (Path(result["snapshot_dir"]) / "SOUL.md")).read_text() == "canonical SOUL.md\n"
    # `isolated_prompts` puts the canonical files outside `~/obsidian`, so no
    # vault commit is attempted. That is the property that keeps this unit test
    # from running `git add` against the live vault.
    assert result.get("vault_commit") is None


def test_promote_refuses_a_variant_that_breaks_the_contract(isolated_prompts, tmp_path):
    """The stub this test used to promote is exactly what must now be refused."""
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("promoted soul\n", encoding="utf-8")
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert result["refused"]
    assert result["applied_files"] == []
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


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


def test_promote_refuses_when_the_snapshot_cannot_be_written(isolated_prompts, tmp_path, monkeypatch):
    """Fixed 2026-09-08; xfailed since 2026-09-06.

    The overlay has to be a contract the guard accepts, or the promotion is
    refused one step earlier and this passes without touching the snapshot
    path at all — which is exactly how it started XPASSing.
    """
    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text(GOOD_CONTRACT, encoding="utf-8")

    real_copy2 = __import__("shutil").copy2

    def failing_copy2(src, dst, *a, **kw):
        if "snapshots" in str(dst):
            raise OSError("disk full")
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(promote.shutil, "copy2", failing_copy2)
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    # If snapshotting failed there must be no promotion at all.
    assert result["refused"] and "snapshot" in result["refused"][0]
    assert result["applied_files"] == []
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


def test_snapshot_current_prompts_raises_when_it_holds_no_prompt(isolated_prompts, tmp_path):
    """The `RuntimeError` half of the no-rollback-point guard (#429 clause 4).

    The test above forces `OSError` out of `copy2`; this is the other half — the
    one that costs *nothing* to trigger. A copy that silently no-ops, or a source
    that has gone missing, still leaves a directory behind, so only the content
    check can see it. It is the half that matters most: the ledger's 65
    `"promoted": true` rows carry no snapshot field at all, so a snapshot that
    exists only as a directory nobody recorded is invisible exactly like this one.
    """
    cfg = make_cfg(tmp_path)
    for path in isolated_prompts.values():
        path.unlink()
    with pytest.raises(RuntimeError, match="no rollback point"):
        promote.snapshot_current_prompts(cfg)


def test_promote_still_refuses_on_a_runtime_snapshot_error_and_applies_nothing(
    isolated_prompts, tmp_path, monkeypatch, caplog
):
    """#429 clause 4: the refusal must outlive any change to the promote path.

    `snapshot_current_prompts` raising `RuntimeError` is caught by the same
    handler as `OSError` and must produce the `REFUSED promotion … no rollback
    point` log and leave the live contract untouched. The overlay here is one the
    contract gate *accepts*, so the snapshot is the only thing standing between it
    and SOUL.md.
    """
    import shutil

    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text(GOOD_CONTRACT, encoding="utf-8")

    real_copy2 = shutil.copy2

    def silently_failing_copy2(src, dst, *a, **kw):
        # No exception, no write: the snapshot directory exists and is empty,
        # exactly the state the content check in snapshot_current_prompts is for.
        if "snapshots" in str(dst):
            return Path(dst)
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(promote.shutil, "copy2", silently_failing_copy2)
    caplog.set_level("ERROR", logger="autoresearch.promote")

    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))

    logged = [r.getMessage() for r in caplog.records]
    assert any("REFUSED promotion" in m and "no rollback point" in m for m in logged), (
        f"expected the no-rollback-point refusal in {logged}"
    )
    assert result["refused"] and "snapshot failed" in result["refused"][0]
    assert result["applied_files"] == []
    assert result["snapshot_dir"] is None
    # The overlay never got its turn on the contract.
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


# ── rollback through the vault route (#1009) ─────────────────────────────────
#
# `rollback()` used to be three `shutil.copy2` calls with no validator, no commit
# and no ledger line. Its targets are tracked vault files, so a restore left the
# contract dirty in the working tree while HEAD still pointed at the promotion —
# and `scripts/util/vault-commit.sh` runs `git add -A` over the vault from seven
# nightly skills, which lands that restore under an unrelated job's message. The
# tests below therefore run against a real git-backed vault: "the restore was
# committed, validated and recorded" is not observable in a plain tmp dir.

GOOD_MEMORY = "# Lloyd Long-Term Memory\n\n## Meta-Instructions\n- Measure before reporting.\n"
GOOD_USER = "# Alan\n\n- Prefers a scoped change over a rewrite.\n"

# Stripped of every gate-role heading and of the load-bearing markers: what a
# snapshot looks like once the contract has moved on underneath it.
GUTTED_SOUL = "# Lloyd Operating Contract\n\n## Core Identity\nBe helpful.\n"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def git_ok(repo, *args):
    """A git command that BUILDS the scenario. Its failure would otherwise show up
    only as a confusing assertion about a commit that was never made, so fail here
    with git's own message — a silently failing `add` is how a dirty-tree test
    starts passing for the wrong reason."""
    r = git(repo, *args)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stdout}{r.stderr}"
    return r


def vault_land_rows(tmp_path):
    """The `vault_land` rows the rollback under test appended, read off disk."""
    from scripts.automod import state as S
    return [e for e in S.read_events(path=tmp_path / "ledger.jsonl")
            if e.get("event") == "vault_land"]


@pytest.fixture
def vault_prompts(isolated_prompts, tmp_path, monkeypatch):
    """The canonical prompts as tracked files in a scratch vault repo.

    `isolated_prompts` is autouse and runs first; re-patching
    `CANONICAL_PROMPTS` here moves the targets *inside* a git tree, which is the
    only state in which the commit half of `rollback()` is reachable.

    The validators are NOT stubbed: the front-matter, prompt-surface and
    fresh-interpreter loader checks all run over the restored paths, which is the
    only way "the restore was validated" is a fact rather than an intention. The
    loader subprocess is cheap here because `LLOYD_HOME` still points at this
    checkout, so `build_system_prompt()` reads this repo's fallback prompt — one
    build, ~0.8 s — and a healthy tree returns no verdicts.
    `test_the_fresh_interpreter_loader_check_runs_inside_a_rollback` points
    `LLOYD_HOME` at a prompt builder that fails, to prove that check can refuse.
    """
    from scripts.automod import state as S, vault_round as V
    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    root = tmp_path / "obsidian"
    (root / "lloyd").mkdir(parents=True)
    git_ok(tmp_path, "init", "-q", "-b", "main", str(root))
    git_ok(root, "config", "user.email", "t@e.com")
    git_ok(root, "config", "user.name", "t")
    for name, text in (("SOUL.md", GOOD_CONTRACT), ("MEMORY.md", GOOD_MEMORY),
                       ("USER.md", GOOD_USER)):
        (root / "lloyd" / name).write_text(text, encoding="utf-8")
    git_ok(root, "add", "-A")
    git_ok(root, "commit", "-q", "-m", "base contract")
    monkeypatch.setattr(V, "VAULT", root)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(promote, "CANONICAL_PROMPTS",
                        {name: root / "lloyd" / name for name in PROMPT_NAMES})
    return root


def promote_into_vault(vault: Path, text: str = "variant") -> None:
    """Simulate a promotion that already landed: the files move and HEAD moves."""
    for name in PROMPT_NAMES:
        (vault / "lloyd" / name).write_text(f"{text} {name}\n", encoding="utf-8")
    git_ok(vault, "add", "-A")
    git_ok(vault, "commit", "-q", "-m", "autoresearch: promote V_x")


def dirty_paths(vault: Path) -> str:
    return git(vault, "status", "--porcelain", "--", *[f"lloyd/{n}" for n in PROMPT_NAMES]).stdout


# clause 1
def test_rollback_of_a_vault_backed_contract_leaves_no_uncommitted_change(vault_prompts, tmp_path):
    """The restored bytes equal the snapshot AND sit in a commit: nothing for a
    later `git add -A` to absorb under someone else's message."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)
    assert dirty_paths(vault_prompts) == "", "the fixture starts clean"

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused") is None, result
    for name in PROMPT_NAMES:
        live = vault_prompts / "lloyd" / name
        assert live.read_text(encoding="utf-8") == (snap / name).read_text(encoding="utf-8")
    assert dirty_paths(vault_prompts) == "", f"restore left the tree dirty: {dirty_paths(vault_prompts)}"


# clause 2
def test_a_committed_rollback_appends_one_vault_land_row_with_the_sha_and_ts(vault_prompts, tmp_path):
    """One row, `ok: true`, carrying the sha it created and the snapshot it came
    from — the audit line the raw copy never wrote."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)
    sha = result["vault_commit"]

    rows = vault_land_rows(tmp_path)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["ok"] is True and row["commit"] == sha, row
    assert snap.name in row["message"], row["message"]
    # The same ts is in the commit itself, so blame names the restore.
    subject = git(vault_prompts, "log", "-1", "--format=%s").stdout
    assert snap.name in subject, subject


# clause 3
def test_rollback_returns_the_vault_sha_and_keeps_the_other_two_keys(vault_prompts, tmp_path):
    """`snapshot` and `restored_files` mean what they meant before; `vault_commit`
    is new and resolves to a commit that changed exactly the prompt files."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)
    sha = result["vault_commit"]

    assert result["snapshot"] == str(snap)
    assert sorted(result["restored_files"]) == sorted(PROMPT_NAMES)
    assert sha and git(vault_prompts, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0
    shown = git(vault_prompts, "show", "--name-only", "--format=", sha).stdout.splitlines()
    assert sorted(line.strip() for line in shown if line.strip()) == sorted(
        f"lloyd/{name}" for name in PROMPT_NAMES)


# clause 3, across the process boundary the agent actually calls
def test_the_rollback_tool_result_carries_the_vault_sha(vault_prompts, tmp_path, monkeypatch):
    """`autoresearch_rollback` returns `promote.rollback()` verbatim, so the sha
    only reaches the caller if the handler's JSON does too."""
    import agent_mcp.autoresearch as AR

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)
    monkeypatch.setattr(AR, "_load_cfg", lambda: cfg)

    payload = json.loads(AR._handle_rollback({"snapshot_ts": snap.name}))

    assert payload.get("refused") is None, payload
    assert payload["vault_commit"] and payload["vault_commit"] == git(
        vault_prompts, "rev-parse", "HEAD").stdout.strip()
    assert sorted(payload["restored_files"]) == sorted(PROMPT_NAMES)


# clause 3, across the MCP endpoint the agent actually calls
async def test_the_mcp_endpoint_returns_the_vault_sha_and_its_own_description(vault_prompts, tmp_path, monkeypatch):
    """Through `call_tool`, not the handler: the agent sees what `text_result` put
    in `content[0].text`, and a refusal must not arrive as a transport error."""
    import agent_mcp.autoresearch as AR

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)
    monkeypatch.setattr(AR, "_load_cfg", lambda: cfg)

    rollback_tool = next(t for t in await AR.list_tools() if t.name == "autoresearch_rollback")
    assert rollback_tool.input_schema["required"] == ["snapshot_ts"]
    for claim in ("vault", "commit", "vault_land"):
        assert claim in rollback_tool.description, rollback_tool.description

    result = await AR.call_tool("autoresearch_rollback", {"snapshot_ts": snap.name})
    payload = json.loads(result.content[0].text)

    assert result.is_error is False, payload
    assert payload["vault_commit"] and payload["vault_commit"] == git(
        vault_prompts, "rev-parse", "HEAD").stdout.strip()
    assert payload["no_change"] is False
    assert sorted(payload["restored_files"]) == sorted(PROMPT_NAMES)
    assert git(vault_prompts, "status", "--porcelain").stdout == ""


async def test_a_refused_restore_crosses_the_mcp_endpoint_as_a_refusal_not_a_crash(vault_prompts, tmp_path, monkeypatch):
    """`refused` is a verdict about the content, so it must not arrive as `isError`
    (which reads as "the tool broke") — and it must still arrive with its reason."""
    import agent_mcp.autoresearch as AR

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    (snap / "SOUL.md").write_text(GUTTED_SOUL, encoding="utf-8")
    promote_into_vault(vault_prompts)
    monkeypatch.setattr(AR, "_load_cfg", lambda: cfg)

    result = await AR.call_tool("autoresearch_rollback", {"snapshot_ts": snap.name})
    payload = json.loads(result.content[0].text)

    assert result.is_error is False, payload
    assert payload.get("refused") and "gate roles" in payload["refused"][0], payload
    assert payload.get("vault_commit") is None
    assert git(vault_prompts, "status", "--porcelain").stdout == ""


# the fresh-interpreter loader rung, driven from inside a rollback
def test_the_fresh_interpreter_loader_check_runs_inside_a_rollback(vault_prompts, tmp_path, monkeypatch):
    """The loader check is a subprocess, so a rollback could have skipped it silently.

    `vault_round` runs that subprocess with `LLOYD_HOME` as its cwd, and the loader
    imports `prompt_builder` off `sys.path[0]` — so pointing `LLOYD_HOME` at a
    directory holding a prompt builder that returns a stub makes the real subprocess
    report the real verdict for a contract that no longer builds. Nothing is stubbed
    inside `land()`: the refusal below comes out of the subprocess's own stdout.
    """
    from scripts.automod import vault_round as V

    stub = tmp_path / "fallback-checkout"
    stub.mkdir()
    (stub / "prompt_builder.py").write_text(
        "def build_system_prompt(*a, **k):\n    return 'stub'\n", encoding="utf-8")
    monkeypatch.setattr(V, "LLOYD_HOME", stub)

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused"), result
    assert "system prompt failed to build" in result["refused"][0], result
    assert result["vault_commit"] is None and result["restored_files"] == []
    assert git(vault_prompts, "status", "--porcelain").stdout == "", "a refused restore left the tree dirty"
    assert (vault_prompts / "lloyd" / "SOUL.md").read_text(encoding="utf-8") == "variant SOUL.md\n"
    rows = vault_land_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["ok"] is False, rows


# clause 4
def test_a_restore_that_breaks_the_contract_is_refused_and_not_left_applied(vault_prompts, tmp_path):
    """An old snapshot predating a structural change must be refused, not applied:
    the tree ends back at HEAD and the result says why."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    (snap / "SOUL.md").write_text(GUTTED_SOUL, encoding="utf-8")
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused"), result
    assert "gate roles" in result["refused"][0], result["refused"]
    assert result.get("vault_commit") is None
    assert result["restored_files"] == [], "a refused restore did not happen"
    assert (vault_prompts / "lloyd" / "SOUL.md").read_text(encoding="utf-8") == "variant SOUL.md\n"
    assert dirty_paths(vault_prompts) == "", "a refused restore left the tree dirty"
    rows = vault_land_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["ok"] is False, rows


# clause 5
def test_a_rollback_onto_content_that_already_matches_head_reports_no_commit(vault_prompts, tmp_path):
    """A repeated rollback, or a snapshot equal to the live contract, is a no-op —
    `VaultRoundError("nothing to commit")` must not escape the tool as an error."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused") is None, result
    assert "error" not in result, result
    assert result["no_change"] is True and result["vault_commit"] is None
    assert sorted(result["restored_files"]) == sorted(PROMPT_NAMES)
    assert dirty_paths(vault_prompts) == ""
    assert [r["ok"] for r in vault_land_rows(tmp_path)] == [], "a no-op cannot ledger a landing"


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
