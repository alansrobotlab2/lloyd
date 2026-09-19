"""The held-out bench split — what reaches the proposer, and what may not.

#549 exists because the autoresearch proposer was handed the named, scored
failing bench tasks in its own prompt and was then judged on those same tasks by
those same tasks. These tests are the two halves of the fix: a split that is
pinned before anything is proposed (`split_hash`), and a leak check on the one
prompt block that used to print a veto task's id, category and score.

The arithmetic of the gate itself is in `test_autoresearch_promotion.py`; this
file is about which tasks the gate is even allowed to have learned from.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch import bench_split, hypothesis_generator as hg
from scripts.autoresearch.common import AutoresearchConfig, AutoresearchPaths

# The live bench's category census, measured 2026-09-10: 4 replay, 4 synthetic,
# 2 adversarial, 1 safety. Written down so a bench file added or recategorised
# shows up here as well as in the split itself.
LIVE_BENCH = {
    "bench_001_reply_greeting": "replay",
    "bench_002_recall_user_fact": "replay",
    "bench_004_replay_schedule_task": "replay",
    "bench_005_replay_memory_update": "replay",
    "bench_003_vault_recall": "synthetic",
    "bench_006_contradiction_check": "synthetic",
    "bench_007_skill_invocation": "synthetic",
    "bench_011_haiku_quantum": "synthetic",
    "bench_008_adversarial_gap": "adversarial",
    "bench_009_adversarial_probe": "adversarial",
    "bench_010_safety_destructive": "safety",
}


def tasks_fixture(names=LIVE_BENCH) -> list[dict]:
    return [{"id": tid, "category": cat} for tid, cat in names.items()]


@pytest.fixture
def cfg(tmp_path):
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench",
        research_root=tmp_path / "research",
        rounds_dir=tmp_path / "rounds",
        ledger_path=tmp_path / "ledger.jsonl",
        variants_dir=tmp_path / "variants",
        snapshots_dir=tmp_path / "snapshots",
        facts_experiments_dir=tmp_path / "facts-experiments",
    )
    paths.research_root.mkdir(parents=True, exist_ok=True)
    return AutoresearchConfig(
        paths=paths, default_model="primary", default_budget_minutes=120,
        max_variants_per_round=4, promotion_min_win_fraction=0.5,
        promotion_min_composite_delta=0.05, promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=2, targets=["prompts"],
    )


# ── the partition ────────────────────────────────────────────────────────────

def test_the_veto_categories_are_always_held_out():
    split = bench_split.compute_split(tasks_fixture(), "R_20260910_000000")
    assert {"bench_008_adversarial_gap", "bench_009_adversarial_probe",
            "bench_010_safety_destructive"} <= set(split["heldout"])


def test_the_split_partitions_the_bench_with_no_overlap_and_no_hole():
    """Every task is in exactly one pool. A task in neither is a task no condition
    reads, which is the failure mode `evaluate_promotion` refuses on — better not
    to produce one."""
    split = bench_split.compute_split(tasks_fixture(), "R_20260910_000000")
    both = set(split["targeted"]) & set(split["heldout"])
    union = set(split["targeted"]) | set(split["heldout"])
    assert both == set()
    assert union == set(LIVE_BENCH)


def test_slice_sizes_are_constant_across_rounds():
    """Category-level rotation is impossible at 4/4/2/1 — the slices would swing
    between 3/8 and 9/2 and every round would re-measure the gate on a different
    denominator. Task-level rotation keeps the sizes fixed; this is the assertion
    that keeps it that way."""
    for i in range(12):
        split = bench_split.compute_split(tasks_fixture(), f"R_20260910_{i:06d}")
        assert len(split["targeted"]) == 6, split
        assert len(split["heldout"]) == 5, split
        assert len(split["rotated_into_heldout"]) == bench_split.ROTATION_SIZE


def test_the_rotation_actually_rotates_so_no_task_is_permanently_privileged():
    """Over twelve rounds every targeted-category task has to serve in the veto at
    least once. A permanently exempt task is the thing the rotation exists to
    prevent, and `sha256(round_id || task_id)` is only honest if it moves."""
    served: set[str] = set()
    pool = {tid for tid, cat in LIVE_BENCH.items()
            if cat in bench_split.TARGETED_CATEGORIES}
    for i in range(60):
        split = bench_split.compute_split(tasks_fixture(), f"R_20260910_{i:06d}")
        served |= set(split["rotated_into_heldout"])
    assert served == pool, f"never rotated into the veto: {sorted(pool - served)}"


def test_the_split_is_deterministic_in_the_round_id():
    """Same round id, same split — so a replay can re-derive what was gated on."""
    a = bench_split.compute_split(tasks_fixture(), "R_20260910_074456")
    b = bench_split.compute_split(tasks_fixture(), "R_20260910_074456")
    assert a["targeted"] == b["targeted"] and a["heldout"] == b["heldout"]
    assert a["split_hash"] == b["split_hash"]


def test_a_different_round_id_selects_a_different_rotation_sometimes():
    """Guards against a rotation that is really a constant."""
    rotations = {tuple(bench_split.compute_split(tasks_fixture(), f"R_{i}")["rotated_into_heldout"])
                 for i in range(12)}
    assert len(rotations) > 1


# ── the hash ─────────────────────────────────────────────────────────────────

def test_verify_accepts_a_split_it_wrote():
    split = bench_split.compute_split(tasks_fixture(), "R_20260910_000000")
    assert bench_split.verify(split) is True


def test_verify_refuses_a_pool_edited_after_the_hash():
    """The whole reason the hash exists: a split may not be re-picked after the
    scores are in."""
    split = bench_split.compute_split(tasks_fixture(), "R_20260910_000000")
    split["heldout"] = [t for t in split["heldout"] if t != "bench_010_safety_destructive"]
    assert bench_split.verify(split) is False


def test_load_split_returns_nothing_for_a_tampered_file(cfg):
    split = bench_split.write_split(cfg, tasks_fixture(), "R_20260910_000000")
    on_disk = json.loads((cfg.paths.research_root / "bench_split.json").read_text())
    on_disk["heldout"] = on_disk["heldout"][:1]
    (cfg.paths.research_root / "bench_split.json").write_text(json.dumps(on_disk))
    assert bench_split.load_split(cfg) is None
    assert split["split_hash"]  # the write itself was fine; only the edit is not


def test_load_split_returns_nothing_when_absent(cfg):
    assert bench_split.load_split(cfg) is None


def test_write_split_lands_the_file_the_acceptance_names(cfg):
    """(b): `bench_split.json`, with a `split_hash`, readable back by round."""
    split = bench_split.write_split(cfg, tasks_fixture(), "R_20260910_074456")
    loaded = bench_split.load_split(cfg)
    assert loaded is not None
    assert loaded["round_id"] == "R_20260910_074456"
    assert loaded["split_hash"] == split["split_hash"]
    assert loaded["heldout"] == split["heldout"]


def test_a_one_sided_bench_refuses_to_produce_a_split():
    """A bench with only targeted categories would make the gate the pre-#549 gate
    again. Better to fail the round than to gate with no veto."""
    targeted_only = {"bench_001_reply_greeting": "replay", "bench_003_vault_recall": "synthetic"}
    with pytest.raises(RuntimeError, match="held-out"):
        bench_split.compute_split(tasks_fixture(targeted_only), "R_20260910_000000")


# ── the leak ─────────────────────────────────────────────────────────────────

def _ledger_with(rows):
    def _write(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return _write


def _baseline_row(task_id: str, category: str, score: float) -> dict:
    return {"round_id": "R_20260910_074456", "variant_id": "BASELINE_1",
            "task_id": task_id, "task_category": category,
            "composite_score": score, "safety_critical": category == "safety",
            "safety_passed": True}


def test_a_held_out_failure_never_reaches_the_proposer_prompt(cfg):
    """The defect in one assertion. Every task fails at 0.00; only the tasks this
    round leaves in the targeted pool may appear in the `_recent_baseline_failures`
    block, because a named failing task is a target and the veto tasks are scored
    on the same run. The visible set is read from the split rather than named, so
    the assertion does not quietly depend on which two tasks happened to rotate."""
    split = bench_split.write_split(cfg, tasks_fixture(), "R_20260910_074456")
    _ledger_with([_baseline_row(tid, cat, 0.0) for tid, cat in LIVE_BENCH.items()])(
        cfg.paths.ledger_path)

    rows = hg._recent_baseline_failures(
        cfg.paths.ledger_path, limit=20, exclude=bench_split.heldout_ids(cfg))
    assert {r["task_id"] for r in rows} == set(split["targeted"])
    assert not set(split["heldout"]) & {r["task_id"] for r in rows}


def test_the_block_is_built_from_the_filtered_rows_in_the_real_prompt(cfg, monkeypatch, tmp_path):
    """`_build_single_variant_prompt` is where the block is rendered, so the check
    belongs there and not only on the helper it calls."""
    _ledger_with([
        _baseline_row("bench_010_safety_destructive", "safety", 0.0),
        _baseline_row("bench_009_adversarial_probe", "adversarial", 0.0),
        _baseline_row("bench_003_vault_recall", "synthetic", 0.1),
    ])(cfg.paths.ledger_path)
    bench_split.write_split(cfg, tasks_fixture(), "R_20260910_074456")

    for attr, name in (("SOUL_PATH", "soul.md"), ("MEMORY_PATH", "memory.md"),
                       ("USER_PATH", "user.md"), ("CORRECTIONS_PATH", "corr.md"),
                       ("KNOWLEDGE_HEALTH_PATH", "kh.md")):
        f = tmp_path / name
        f.write_text(f"canonical {name}\n", encoding="utf-8")
        monkeypatch.setattr(hg, attr, f)

    prompt = hg._build_single_variant_prompt(cfg, ["prompts"])
    # Scoped to the failures block — lines of the rendered `- task=… category=…
    # composite=…` form. The prompt also carries a static sentence naming
    # bench_010_safety_destructive as the safety-critical task, which is a
    # separate disclosure with no score in it and its own backlog item; asserting
    # the whole prompt clean here would just be a failing test about the wrong bug.
    disclosed = {line.split()[1].removeprefix("task=")
                 for line in prompt.splitlines() if line.strip().startswith("- task=")}
    assert "bench_010_safety_destructive" not in disclosed
    assert "bench_009_adversarial_probe" not in disclosed
    assert "bench_003_vault_recall" in disclosed


def test_with_no_split_on_disk_the_veto_categories_are_still_withheld(cfg):
    """The fallback matters most when nobody set up: a hand-run generator with no
    round in flight must still not print the safety task."""
    _ledger_with([
        _baseline_row("bench_010_safety_destructive", "safety", 0.0),
        _baseline_row("bench_002_recall_user_fact", "replay", 0.0),
    ])(cfg.paths.ledger_path)
    bench_dir = cfg.paths.bench_dir
    bench_dir.mkdir(parents=True, exist_ok=True)
    for tid, cat in LIVE_BENCH.items():
        (bench_dir / f"{tid}.md").write_text(
            f"---\nid: {tid}\ncategory: {cat}\n---\n\nprompt: hello\n", encoding="utf-8")

    held = bench_split.heldout_ids(cfg)
    assert "bench_010_safety_destructive" in held
    rows = hg._recent_baseline_failures(cfg.paths.ledger_path, limit=6, exclude=held)
    assert {r["task_id"] for r in rows} == {"bench_002_recall_user_fact"}


def test_exclusion_happens_before_the_limit_is_spent(cfg):
    """Hiding the veto slice must cost the prompt signal volume, not the signal:
    the caller still gets `limit` rows, all of them aimable."""
    rows = [_baseline_row("bench_010_safety_destructive", "safety", 0.0)]
    rows += [_baseline_row(f"bench_{i:03d}_x", "replay", 0.0) for i in range(6)]
    _ledger_with(rows)(cfg.paths.ledger_path)
    out = hg._recent_baseline_failures(
        cfg.paths.ledger_path, limit=4,
        exclude={"bench_010_safety_destructive"})
    assert len(out) == 4
    assert all(r["task_id"] != "bench_010_safety_destructive" for r in out)


# ─────────────────────────────────────────────────────────────────────────────
# The round's seam: the code `run()` actually calls, over the REAL bench.
#
# The tests above are unit tests over synthetic task lists, so none of them can
# tell you that `run_round.run()` fails to write the artifact at all — a test
# that fakes the whole round and asserts on a hand-built split dict has that hole.
# These call the same `record_split` `run()` calls, against a bench directory
# that is the real 11-task one, and read back what landed on disk: the JSON
# artifact, its hash, and the ledger row. Read-only — the bench is only listed,
# and `research_root` is a tmp dir, so no split file is written to the live
# store and no live ledger is opened.
# ─────────────────────────────────────────────────────────────────────────────

REAL_BENCH = Path.home() / "obsidian" / "lloyd" / "bench"
#: A round id only fixes WHICH targeted tasks rotate into the veto; the pools
#: themselves come from the bench categories, so any id pins the seam.
ROUND = "R_20260919_120000"
requires_real_bench = pytest.mark.skipif(
    not REAL_BENCH.is_dir(), reason=f"no live bench at {REAL_BENCH}")


@requires_real_bench
def test_record_split_writes_the_artifact_and_the_ledger_row(cfg):
    """`record_split` writes bench_split.json AND appends an `event: split` ledger
    row carrying the same hash — the two surfaces the round's decision and the
    replay both read. Verified by re-reading, not by trusting the write."""
    from scripts.autoresearch import run_round
    from scripts.autoresearch.common import load_bench_tasks

    cfg.paths.ensure()
    tasks = load_bench_tasks(REAL_BENCH)
    assert len(tasks) == 11, f"expected the live 11-task bench, got {len(tasks)}"

    split = run_round.record_split(cfg, tasks, ROUND)

    artifact = json.loads(bench_split.split_path(cfg).read_text(encoding="utf-8"))
    assert artifact == split, "the file and the returned split disagree"
    assert bench_split.verify(artifact), "the written artifact fails its own hash check"
    assert artifact["split_hash"] == bench_split.compute_split(tasks, ROUND)["split_hash"]
    assert set(artifact["targeted"]) | set(artifact["heldout"]) == {
        t["id"] for t in tasks}, "every real bench task must be in exactly one pool"
    assert not set(artifact["targeted"]) & set(artifact["heldout"])

    rows = [json.loads(line) for line in
            cfg.paths.ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    split_rows = [r for r in rows if r.get("event") == "split"]
    assert len(split_rows) == 1, f"expected one split row, got {len(split_rows)}"
    assert split_rows[0]["split_hash"] == split["split_hash"]
    assert split_rows[0]["heldout"] == split["heldout"]


@requires_real_bench
def test_the_written_split_is_the_one_the_proposer_and_gate_load(cfg):
    """Round-trip through `load_split`: the proposer's leak filter and the replay
    both read the artifact back through it, and a tampered or unverifiable file
    loads as None — which silently turns the veto off. So the seam is pinned at
    the reader too, not only at the writer."""
    from scripts.autoresearch import run_round
    from scripts.autoresearch.common import load_bench_tasks

    cfg.paths.ensure()
    split = run_round.record_split(cfg, load_bench_tasks(REAL_BENCH), ROUND)

    assert bench_split.load_split(cfg) == split, "the artifact does not survive the reader"

    path = bench_split.split_path(cfg)
    tampered = dict(split, heldout=[t for t in split["heldout"] if t != split["heldout"][0]])
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert bench_split.load_split(cfg) is None, (
        "a split edited after the round started still loads — the leak filter and "
        "the veto would silently follow the edited slice")


def test_the_only_write_split_call_sits_inside_record_split():
    """Structural guard on the seam: `run()` must obtain its split by calling
    `record_split`, never by writing one inline. An inline `write_split` inside the
    round would produce a real artifact and a real veto while skipping the ledger
    row, and the round's own split could then not be re-read by the replay — the
    failure is invisible in the round's output, so it is pinned at the source."""
    import inspect
    import re

    from scripts.autoresearch import run_round

    src = inspect.getsource(run_round)
    calls = [m.start() for m in re.finditer(r"bench_split\.write_split\(", src)]
    assert len(calls) == 1, (
        f"expected exactly one write_split call site (inside record_split); "
        f"found {len(calls)} — a second one bypasses the ledger append")
    body = inspect.getsource(run_round.record_split)
    assert "bench_split.write_split(" in body
    assert "event" in body and '"split"' in body, (
        "record_split no longer appends the split event the replay reads back")
