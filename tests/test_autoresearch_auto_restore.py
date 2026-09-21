"""#1099 — a beyond-noise post-promotion decline restores the promotion itself.

What is under test
------------------
Backlog #429 built the *detector*: `post_promotion.compare` measures each round's
fresh baseline against the mean the last promotion recorded and says
`regression: true` when the decline is bigger than `promotion.noise_floor`
(0.0200 live). It stopped at reporting, reserving the restore for a human. Alan
signed that clause off on 2026-09-13, and this file pins what replaced it:
`auto_restore.restore_for_decline`, called from `run_round.run`, which hands the
snapshot to `promote.rollback(..., promotion=...)` — the same validated vault route
`promote` has used since #506 — and records one `promotion_restored` row.

The item's five clauses are the five sections below. The load-bearing distinction in
the middle of them is that a restore is a **vault commit**, not a file copy:
`vault_round.land` runs the prompt-surface validators and the real loaders, restores
the paths if any fails, commits exactly the files it was given on the vault's
`main`, and appends the `vault_land` event. Every test here therefore drives a real
git repo (a scratch vault under `tmp_path`) and the real validators — mocking `land`
would make clause 2 untestable, which is why nothing mocks it.

Isolation
---------
`vault_round.VAULT`, `automod.state.LEDGER_PATH`, `promote.CANONICAL_PROMPTS` and
`run_round.load_config` are redirected into `tmp_path` by the `env` fixture, so
nothing in this module can touch `~/obsidian` or the real ledger. The loader
subprocess still runs against this checkout — that is the point, not a leak. The
bench is the real `bench/` tree, read-only.

What the replay stubs, and why
------------------------------
`drive_round` drives the real `run_round.run`. Three calls are replaced, all of them
the half that costs a GPU-minute and a model: `propose_variants` (returns one
hand-built variant), `_run_trials` (returns traces instead of benching) and
`judge_trace` (returns the score the trace carries). The split, the summariser,
`evaluate_promotion`, the promotion decision, the `round_summary` write and the
restore are the real ones — which is the point. Clause 1 is about *ordering inside
`run()`* (compare → restore → record), and a test that faked the round could not see
that ordering at all.
"""
from __future__ import annotations

import ast
import asyncio
import json
import shutil
import textwrap
from pathlib import Path

import pytest

from scripts.autoresearch import auto_restore, post_promotion, promote, run_round
from scripts.automod import state as automod_state
from scripts.automod import vault_round

# Reused rather than re-derived: the git helpers, the config builder and the
# validator-clean prompt texts already encode the scratch-vault contract, and a
# second copy would be a second place to get wrong.
from tests.test_autoresearch_promotion import (
    GOOD_MEMORY, GOOD_USER, PROMPT_NAMES, git, git_ok, make_cfg, summary)
from tests.test_prompt_surface_guard import GOOD_CONTRACT

ROOT = Path(__file__).resolve().parent.parent

# A bench of our own, written into tmp_path by the fixture. Reading the real
# `~/obsidian/lloyd/bench` would make every assertion here depend on how many tasks
# the vault happens to hold today, and `bench_split.compute_split` refuses a bench
# with no targeted (`replay`/`synthetic`) or no held-out (`safety`/`adversarial`) half.
#
# The four replay tasks sort first by id, so with `--bench-limit 2` the round runs
# exactly `bench_a1` and `bench_a2` — both non-safety, so the veto half of the scored
# slice is empty, `heldout_delta` is None and `evaluate_promotion` refuses the
# candidate with `no_heldout_overlap`. That refusal is what keeps the replayed round
# from promoting its own candidate, and it holds for every `round_id`: the rotation
# into the veto cannot put a safety task into a slice that contains none.
BENCH_TASKS = [
    ("bench_a1", "replay", False), ("bench_a2", "replay", False),
    ("bench_a3", "replay", False), ("bench_a4", "replay", False),
    ("bench_s1", "safety", True), ("bench_s2", "safety", True),
]


def write_bench(dirn: Path) -> None:
    dirn.mkdir(parents=True, exist_ok=True)
    for task_id, category, safety in BENCH_TASKS:
        (dirn / f"{task_id}.md").write_text(
            "---\n"
            f"id: {task_id}\n"
            f"category: {category}\n"
            f"safety_critical: {str(safety).lower()}\n"
            "prompt: say the thing\n"
            "objective_checks: []\n"
            "---\n\nProse body.\n",
            encoding="utf-8")

# The numbers every round in this file is built from. `compare` decides on
# `decline > floor`, so a 0.6000 promotion measured against a 0.5000 baseline declines
# 0.1000 — five times the live floor — while a 0.5900 baseline declines 0.0100, half
# the floor. Stated once so a test's prose and its arithmetic cannot drift apart.
PROMOTED_MEAN = 0.6000
DECLINING_BASELINE = 0.5000
IN_FLOOR_BASELINE = 0.5900
NOISE_FLOOR = 0.0200

BAD_VARIANT = "V_bad_promotion"
BAD_ROUND = "R_20260901_000000"
SECOND_ROUND = "R_20260902_000000"
THIRD_ROUND = "R_20260903_000000"
REPLAY_VARIANT = {
    "variant_id": "V_replay",
    "description": "replayed candidate",
    "hypothesis": "replayed for the restore path",
    "overlay_files": {"SOUL.md": GOOD_CONTRACT},
}
PROMOTED_TEXT = "bad promoted content\n"


def head(vault: Path) -> str:
    return git(vault, "rev-parse", "HEAD").stdout.strip()


def subject(vault: Path) -> str:
    return git(vault, "log", "-1", "--format=%s").stdout.strip()


def body(vault: Path) -> str:
    return git(vault, "log", "-1", "--format=%b").stdout.strip()


def porcelain(vault: Path) -> str:
    return git(vault, "status", "--porcelain").stdout


def committed(vault: Path) -> list[str]:
    """The subject lines of every commit, oldest first."""
    return git(vault, "log", "--format=%s").stdout.splitlines()[::-1]


class Env:
    """The scratch vault, its config, and helpers for reading what a round did."""

    def __init__(self, cfg, vault, root):
        self.cfg = cfg
        self.vault = vault
        self.root = root
        self.promoted_commit = ""

    def read(self, name: str) -> str:
        return (self.vault / "lloyd" / name).read_text(encoding="utf-8")

    def restore_rows(self) -> list[dict]:
        return post_promotion.restore_rows(self.cfg.paths.ledger_path)

    def vault_land_rows(self) -> list[dict]:
        return [e for e in automod_state.read_events(path=self.root / "ledger.jsonl")
                if e.get("event") == "vault_land"]

    def round_reports(self) -> list[Path]:
        """Every round report on disk, oldest first — round ids are timestamps."""
        return sorted(self.cfg.paths.rounds_dir.glob("R_*.md"))

    def round_report(self) -> str:
        """The newest round report. Tests that run two rounds read the second one."""
        return self.round_reports()[-1].read_text(encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A vault repo holding the canonical contract, committed, clean.

    The three prompt files are validator-clean (`GOOD_CONTRACT` etc.), so a restore
    *back* to them passes the real loaders and a restore *away* from them can be
    refused by them: the validators run for real in both directions.
    """
    root = tmp_path / "obsidian"
    (root / "lloyd").mkdir(parents=True)
    git_ok(tmp_path, "init", "-q", "-b", "main", str(root))
    git_ok(root, "config", "user.email", "t@e.com")
    git_ok(root, "config", "user.name", "t")
    for name, text in (("SOUL.md", GOOD_CONTRACT), ("MEMORY.md", GOOD_MEMORY),
                       ("USER.md", GOOD_USER)):
        (root / "lloyd" / name).write_text(text, encoding="utf-8")
    git_ok(root, "add", "-A")
    git_ok(root, "commit", "-q", "-m", "contract before the bad promotion")

    cfg = make_cfg(tmp_path, promotion_noise_floor=NOISE_FLOOR)
    # `run()` calls `load_config()` out of run_round's own namespace, so this one
    # patch puts a whole round — spec, split, ledger, rounds, snapshots — under
    # tmp_path and nothing in this file can reach `~/obsidian` or the live ledger.
    monkeypatch.setattr(run_round, "load_config", lambda path=None: cfg)
    # The bench is this file's own, not the vault's: see BENCH_TASKS. `evaluate_promotion`
    # still refuses the replayed candidate for the real reason — a two-task slice has
    # no held-out overlap — but the reason does not depend on what the vault holds today.
    write_bench(tmp_path / "bench")
    cfg.paths.bench_dir = tmp_path / "bench"
    monkeypatch.setattr(vault_round, "VAULT", root)
    # The automod ledger sits *outside* the vault repo, where the live one does: a
    # `vault_land` row written inside the scratch repo would be swept up by `land`'s
    # commit and read back as a contract change this round made.
    monkeypatch.setattr(automod_state, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(promote, "CANONICAL_PROMPTS",
                        {name: root / "lloyd" / name for name in PROMPT_NAMES})
    return Env(cfg, root, tmp_path)


def promote_a_bad_variant(env: Env, *, variant_id: str = BAD_VARIANT,
                          round_id: str = BAD_ROUND, mean: float = PROMOTED_MEAN) -> str:
    """Stage the history a restoring round needs, through the real recorder.

    A promotion is three facts: the snapshot taken *before* it (the rollback point),
    the commit that landed it, and the `round_summary` row recording its mean. All
    three come from the functions a live round uses — `snapshot_current_prompts`, a
    vault commit, `record_round_summary` — because the reader under test
    (`last_promotion`, then `restore_for_decline`) has to work on what that writer
    actually emits, not on a hand-typed dict that happens to carry the keys I wanted.

    Returns the snapshot directory name.
    """
    snap = promote.snapshot_current_prompts(env.cfg)
    for name in PROMPT_NAMES:
        (env.vault / "lloyd" / name).write_text(PROMOTED_TEXT, encoding="utf-8")
    git_ok(env.vault, "add", "-A")
    git_ok(env.vault, "commit", "-q", "-m", f"autoresearch: promote {variant_id}")
    env.promoted_commit = head(env.vault)
    post_promotion.record_round_summary(
        env.cfg, round_id, 0.7000,
        {"variant_id": variant_id, "should_promote": True, "meets_contract": True,
         "snapshot_dir": str(snap), "vault_commit": env.promoted_commit,
         "applied_files": list(PROMPT_NAMES), "target_surface": "prompts"},
        summary(mean),
    )
    return snap.name


def drive_round(env: Env, baseline_mean: float, *, monkeypatch,
                variant_mean: float = 0.6300, dry_run: bool = False) -> dict:
    """Run a real autoresearch round whose benching is replaced by fixed scores.

    Only `propose_variants`, `_run_trials` and `judge_trace` are replaced. The split,
    the summariser, the promotion gate, the record and the restore all run.
    `bench_limit=2` is what keeps this round from promoting anything: a two-task
    slice has no held-out overlap, `evaluate_promotion` refuses it with
    `no_heldout_overlap`, and the round's own candidate can never be mistaken for the
    promotion being undone.
    """
    async def fake_trials(cfg, variant_pairs, tasks, model, harness, max_parallel):
        direct = []
        for index, (vid, _overlay) in enumerate(variant_pairs):
            # `materialize_variants` seeds the list with the baseline pair, so index
            # 0 is the baseline and every later pair is a candidate.
            score = baseline_mean if index == 0 else variant_mean
            for task in tasks:
                direct.append({
                    "variant_id": vid, "task_id": task["id"], "status": "ok",
                    "turns": 1, "tool_calls": [], "denied_calls": [],
                    "harness": "direct", "task_category": task.get("category"),
                    "preset_composite": score,
                })
        return direct, []

    def fake_judge(task, trace, rubric_model=None):
        score = float(trace["preset_composite"])
        return {"composite_score": score, "objective_score": score,
                "rubric_overall": score,
                "safety_critical": bool(task.get("safety_critical")),
                "safety_passed": True}

    monkeypatch.setattr(run_round, "_run_trials", fake_trials)
    monkeypatch.setattr(run_round, "judge_trace", fake_judge)
    monkeypatch.setattr(run_round, "propose_variants",
                        lambda cfg, **kwargs: [dict(REPLAY_VARIANT)])
    return asyncio.run(run_round.run(targets=["prompts"], bench_limit=2, dry_run=dry_run))


def restore(env: Env, baseline_mean: float = DECLINING_BASELINE, *,
            round_id: str = SECOND_ROUND, **kw) -> dict:
    """Compare and restore, exactly as `run_round.run` does, minus the benching."""
    prior = post_promotion.last_promotion(
        env.cfg.paths.ledger_path, env.cfg.paths.rounds_dir, exclude_round=round_id)
    comparison = post_promotion.compare(baseline_mean, prior, NOISE_FLOOR)
    return auto_restore.restore_for_decline(env.cfg, round_id, comparison, prior, **kw)


# ── clause 1: a beyond-noise decline restores, and nothing else does ──────────

def test_a_beyond_noise_decline_restores_the_promotion_without_a_human(env, monkeypatch):
    """`run()` itself does it: no human, no second command, no operator step."""
    promote_a_bad_variant(env)
    assert env.read("SOUL.md") == PROMOTED_TEXT, "the scenario must start promoted"

    result = drive_round(env, DECLINING_BASELINE, monkeypatch=monkeypatch)

    assert result["post_promotion"]["regression"] is True, result["post_promotion"]
    assert result["post_promotion_restore"]["status"] == "restored", result["post_promotion_restore"]
    assert env.read("SOUL.md") == GOOD_CONTRACT, "the promoted file was not undone"
    assert env.read("MEMORY.md") == GOOD_MEMORY and env.read("USER.md") == GOOD_USER
    assert porcelain(env.vault) == "", "a restore must not leave the vault dirty"


def test_the_restore_names_the_snapshot_and_the_sha_it_produced(env, monkeypatch):
    """The report, the ledger row and the round result agree on the rollback point."""
    snap_ts = promote_a_bad_variant(env)

    result = drive_round(env, DECLINING_BASELINE, monkeypatch=monkeypatch)

    row = env.restore_rows()[-1]
    assert row["snapshot_ts"] == snap_ts, row
    sha = head(env.vault)
    assert sha == row["vault_commit"] == result["post_promotion_restore"]["vault_commit"]
    report = env.round_report()
    assert "## Post-promotion check" in report
    assert snap_ts in report and BAD_VARIANT in report
    assert sha[:12] in report, "the report must carry the vault sha it landed"


def test_a_decline_within_the_noise_floor_restores_nothing(env, monkeypatch):
    """The floor still means something: 0.0100 below a 0.6000 promotion is noise."""
    promote_a_bad_variant(env)
    before = head(env.vault)

    result = drive_round(env, IN_FLOOR_BASELINE, monkeypatch=monkeypatch)

    assert result["post_promotion"]["regression"] is False, result["post_promotion"]
    assert env.restore_rows() == [], "a within-floor decline wrote a restore row"
    assert env.read("SOUL.md") == PROMOTED_TEXT, "a noise decline reverted work"
    assert head(env.vault) == before


def test_a_promotion_with_no_snapshot_on_record_is_refused_by_name(env, monkeypatch):
    """No rollback point is a refusal the round reports, never a guess at one."""
    snap = promote_a_bad_variant(env)
    shutil.rmtree(env.cfg.paths.snapshots_dir / snap)
    before = head(env.vault)

    result = drive_round(env, DECLINING_BASELINE, monkeypatch=monkeypatch)

    assert result["post_promotion_restore"]["status"] == "no_snapshot"
    assert env.restore_rows()[-1]["status"] == "no_snapshot"
    assert env.read("SOUL.md") == PROMOTED_TEXT
    assert head(env.vault) == before, "a refused restore must not commit"
    assert "NOT RESTORED" in env.round_report()


# ── clause 2: one revertable vault sha, through the validated route ───────────

def test_the_restore_is_one_revertable_commit_and_a_vault_land_event(env):
    snap_ts = promote_a_bad_variant(env)
    before = head(env.vault)

    outcome = restore(env)

    assert outcome["status"] == "restored", outcome
    after = head(env.vault)
    assert after != before
    assert porcelain(env.vault) == ""
    assert env.restore_rows()[-1]["vault_commit"] == after
    lands = env.vault_land_rows()
    assert len(lands) == 1 and lands[0]["ok"] is True, lands
    assert lands[0]["commit"] == after, lands
    assert lands[0]["paths"] == [f"lloyd/{n}" for n in PROMPT_NAMES], lands
    assert BAD_VARIANT in subject(env.vault), subject(env.vault)
    assert BAD_ROUND in subject(env.vault), subject(env.vault)
    assert snap_ts in subject(env.vault) + body(env.vault)
    # "one revert redoes the promotion" is a claim about git, so git answers it:
    # reverting the restore commit must put the promoted bytes back on every file.
    git_ok(env.vault, "revert", "--no-edit", after)
    assert porcelain(env.vault) == ""
    assert env.read("SOUL.md") == PROMOTED_TEXT, "the restore commit was not one revert"


def test_a_loader_failure_leaves_the_promoted_contract_in_place_and_says_so(env, monkeypatch):
    """The validators are not decoration: a restored contract that cannot build is
    put back, the round is told, and the promotion is *not* marked undone."""
    stub = env.root / "fallback-checkout"
    stub.mkdir()
    (stub / "prompt_builder.py").write_text(
        "def build_system_prompt(*a, **k):\n    return 'stub'\n", encoding="utf-8")
    monkeypatch.setattr(vault_round, "LLOYD_HOME", stub)
    promote_a_bad_variant(env)
    before = head(env.vault)

    outcome = restore(env)

    assert outcome["status"] == "refused", outcome
    assert "failed to build" in outcome["row"]["reason"], outcome["row"]
    assert env.read("SOUL.md") == PROMOTED_TEXT, "a refused restore left the contract modified"
    assert porcelain(env.vault) == "", "a refused restore left the vault dirty"
    assert head(env.vault) == before
    assert any("failed to build" in line for line in outcome["report_lines"]), outcome["report_lines"]
    # A refusal is not a restore: a later round must still be able to try, so the
    # promotion stays comparable and `already_restored` must not fire on this row.
    assert auto_restore.already_restored(env.cfg.paths.ledger_path, BAD_ROUND) is None


# ── clause 3: a file changed since the promotion is refused by name ───────────

def test_a_file_that_changed_after_the_promotion_is_refused_by_name(env):
    """Someone edited SOUL.md after the bad promotion landed. The restore undoes the
    two files nobody touched and reports the third rather than reverting their work."""
    promote_a_bad_variant(env)
    # A *valid* human edit: appending a line keeps SOUL.md buildable, so the only
    # thing standing between the restore and a commit is the changed-since-promotion
    # guard. (An edit that also broke the contract would refuse the whole restore at
    # the validator instead — pinned by
    # `test_a_loader_failure_leaves_the_promotion_in_place_and_says_so`.)
    human_edit = GOOD_CONTRACT + "\n## A section a human added afterwards\n\nBecause they had a reason.\n"
    assert human_edit != GOOD_CONTRACT
    (env.vault / "lloyd" / "SOUL.md").write_text(human_edit, encoding="utf-8")
    git_ok(env.vault, "add", "-A")
    git_ok(env.vault, "commit", "-q", "-m", "a human edit that landed after the promotion")
    human_head = head(env.vault)

    outcome = restore(env)

    assert outcome["status"] == "restored", outcome
    refused = outcome["row"]["refused_files"]
    assert [r["file"] for r in refused] == ["SOUL.md"], refused
    assert env.read("SOUL.md") == human_edit, "the guard was bypassed and work reverted"
    assert env.read("MEMORY.md") == GOOD_MEMORY and env.read("USER.md") == GOOD_USER
    assert "SOUL.md" in "\n".join(outcome["report_lines"])
    assert porcelain(env.vault) == "", "the refused file's bytes were left uncommitted"
    assert head(env.vault) != human_head


def test_changed_since_promotion_names_every_diverging_file(env):
    """The guard itself, at unit scope: the reference is the promotion's commit."""
    promote_a_bad_variant(env)
    (env.vault / "lloyd" / "MEMORY.md").write_text("drifted\n", encoding="utf-8")
    safe, refused = promote.changed_since_promotion(
        list(PROMPT_NAMES), [f"lloyd/{n}" for n in PROMPT_NAMES],
        env.vault, env.promoted_commit)

    assert [r["file"] for r in refused] == ["MEMORY.md"], refused
    assert sorted(safe) == ["SOUL.md", "USER.md"]


def test_an_uncommitted_edit_after_the_promotion_is_refused_and_never_committed(env):
    """The guard reads the working tree, not HEAD. Someone is mid-edit on USER.md:
    their bytes stay on disk, they are not swept into the restore's commit under the
    restore's message, and the two files nobody touched still get undone."""
    promote_a_bad_variant(env)
    work_in_progress = GOOD_USER + "\n## A section being drafted\n\nWork in progress.\n"
    (env.vault / "lloyd" / "USER.md").write_text(work_in_progress, encoding="utf-8")

    outcome = restore(env)

    assert outcome["status"] == "restored", outcome
    refused = {r["file"]: r["reason"] for r in outcome["row"]["refused_files"]}
    assert "USER.md" in refused, refused
    assert env.read("USER.md") == work_in_progress, "their draft was overwritten"
    # The draft stays exactly as it was found: still modified, still uncommitted. A
    # clean tree here would mean the restore had committed their work-in-progress.
    assert porcelain(env.vault) == " M lloyd/USER.md\n", porcelain(env.vault)
    committed_files = git_ok(env.vault, "show", "--name-only", "--format=", "HEAD").stdout
    assert "USER.md" not in committed_files, committed_files
    assert env.read("SOUL.md") == GOOD_CONTRACT


def test_a_restore_commits_only_the_files_it_restored(env):
    """With one file refused, the commit's file list must be exactly the two that
    were restored — the attribution rule #506 was filed on, applied to a restore:
    `land` is handed only what it is being asked to commit."""
    promote_a_bad_variant(env)
    before = head(env.vault)
    (env.vault / "lloyd" / "SOUL.md").write_text(
        GOOD_CONTRACT + "\n## A section added afterwards\n\nBecause they had a reason.\n",
        encoding="utf-8")

    outcome = restore(env)

    assert outcome["status"] == "restored", outcome
    files = git_ok(env.vault, "show", "--name-only", "--format=", "HEAD").stdout.split()
    assert sorted(f.split("/")[-1] for f in files) == ["MEMORY.md", "USER.md"], files
    assert outcome["row"]["restored_files"] == ["MEMORY.md", "USER.md"]
    assert head(env.vault) != before


def test_a_post_promotion_edit_that_breaks_the_contract_refuses_the_whole_restore(env):
    """The compound case, and the reason the restore is all-or-nothing: the human's
    SOUL.md is refused by the changed-since guard, and the two files that *could* be
    restored are validated as one surface with it — a contract that cannot build must
    not be committed half-restored. Nothing moves, and the refusal says why."""
    promote_a_bad_variant(env)
    broken = "# human rewrite that dropped the gate\n"
    (env.vault / "lloyd" / "SOUL.md").write_text(broken, encoding="utf-8")
    before = head(env.vault)

    outcome = restore(env)

    assert outcome["status"] == "refused", outcome
    assert env.read("SOUL.md") == broken
    assert env.read("MEMORY.md") == PROMOTED_TEXT, "a refused restore changed another file"
    assert env.read("USER.md") == PROMOTED_TEXT
    assert head(env.vault) == before, "a refused restore committed"
    assert "SOUL.md" in str(outcome["row"]["refused_files"])
    assert porcelain(env.vault) == " M lloyd/SOUL.md\n", porcelain(env.vault)


# ── clause 4: once per promotion, and never compared as the last promotion ────

def test_two_declining_rounds_restore_the_same_promotion_once(env, monkeypatch):
    """Round B undoes the promotion; round C neither undoes it again nor measures
    itself against it, because after B the live files are the pre-promotion contract."""
    snap_ts = promote_a_bad_variant(env)

    first = drive_round(env, DECLINING_BASELINE, monkeypatch=monkeypatch)
    restored_sha = head(env.vault)
    assert first["post_promotion_restore"]["status"] == "restored"
    assert env.restore_rows()[-1]["snapshot_ts"] == snap_ts

    second = drive_round(env, DECLINING_BASELINE, monkeypatch=monkeypatch)

    rows = env.restore_rows()
    assert len(rows) == 1, [r["status"] for r in rows]
    # Both are None because there is no longer a promotion left to compare against:
    # `last_promotion` subtracts the restored one, and `compare` on nothing is nothing
    # — not a silent pass, a stated absence (the report's own line says so).
    assert second["post_promotion"] is None, second["post_promotion"]
    assert second["post_promotion_restore"] is None, second["post_promotion_restore"]
    assert head(env.vault) == restored_sha, "the second round committed a second restore"
    assert env.read("SOUL.md") == GOOD_CONTRACT
    assert "no promotion on record" in env.round_report()


def test_the_action_level_guard_refuses_a_promotion_already_undone(env):
    """`last_promotion` is the primary exclusion; this is the guard at the point of
    action, for a caller that hands over its own comparison and prior record — a
    replay, or a promotion read back from a round report rather than the ledger. It
    must not commit a second restore of bytes that are already restored."""
    snap_ts = promote_a_bad_variant(env)
    outcome = restore(env)
    assert outcome["status"] == "restored"
    sha = head(env.vault)

    prior = {
        "round_id": BAD_ROUND, "promoted_variant_id": BAD_VARIANT,
        "snapshot_dir": str(env.cfg.paths.snapshots_dir / snap_ts),
        "promoted_variant_mean": PROMOTED_MEAN, "baseline_mean": 0.7000,
        "vault_commit": env.promoted_commit, "source": "report",
    }
    comparison = post_promotion.compare(DECLINING_BASELINE, prior, NOISE_FLOOR)
    again = auto_restore.restore_for_decline(env.cfg, THIRD_ROUND, comparison, prior)

    assert again["status"] == "already_restored", again
    assert again["restored_by"] == SECOND_ROUND
    assert head(env.vault) == sha, "a second restore was committed"
    assert len(env.restore_rows()) == 1, "the repeat restore was recorded as a second one"
    assert any("already restored" in line for line in again["report_lines"])


def test_restored_promotion_rounds_excludes_only_finished_restores(env):
    """The exclusion list is built from the statuses that mean *undone*."""
    promote_a_bad_variant(env)
    assert post_promotion.restored_promotion_rounds(env.cfg.paths.ledger_path) == set()

    restore(env)

    assert post_promotion.restored_promotion_rounds(env.cfg.paths.ledger_path) == {BAD_ROUND}
    assert post_promotion.last_promotion(
        env.cfg.paths.ledger_path, env.cfg.paths.rounds_dir,
        exclude_round=THIRD_ROUND) is None


def test_a_refused_restore_does_not_exclude_the_promotion(env):
    """A refusal leaves the contract promoted, so the promotion must stay visible as
    the thing to compare against — excluding it would hide the decline entirely."""
    snap = promote_a_bad_variant(env)
    shutil.rmtree(env.cfg.paths.snapshots_dir / snap)

    outcome = restore(env)

    assert outcome["status"] == "no_snapshot"
    assert post_promotion.restored_promotion_rounds(env.cfg.paths.ledger_path) == set()
    assert post_promotion.last_promotion(
        env.cfg.paths.ledger_path, env.cfg.paths.rounds_dir,
        exclude_round=THIRD_ROUND)["round_id"] == BAD_ROUND


# ── clause 5: the report says so; the manual tool cannot bypass the route ─────

def test_a_dry_run_reports_the_decline_and_restores_nothing(env, monkeypatch):
    """`--dry-run` is not consent to rewrite the contract; the decline and its
    rollback point are still reported, and the ledger carries the reason."""
    snap_ts = promote_a_bad_variant(env)
    before = head(env.vault)

    result = drive_round(env, DECLINING_BASELINE, monkeypatch=monkeypatch, dry_run=True)

    assert result["post_promotion_restore"]["status"] == "dry_run"
    assert env.read("SOUL.md") == PROMOTED_TEXT
    assert head(env.vault) == before
    report = env.round_report()
    assert snap_ts in report and "dry run" in report


def test_the_manual_tool_and_the_round_share_one_rollback_entry_point():
    """Clause 5's second half: exactly one function puts a snapshot's bytes onto the
    contract, and neither caller reaches past it.

    A copy still exists — a validated commit is not a `git revert`, the bytes have to
    be in the tree before `vault_round.land` can validate them — but it is the only
    one, it sits inside the function that lands through the vault route, and no caller
    performs a copy of its own. That is what "cannot bypass the vault route" means.
    """
    import inspect

    from agent_mcp import autoresearch as AR

    # Counted through the AST, not the source text: the docstring *names*
    # `shutil.copy2` to say what it replaced, and a text count would read that prose
    # as a second copy.
    copies = [
        node for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(promote.rollback))))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy2"
    ]
    assert len(copies) == 1, [ast.unparse(n) for n in copies]
    assert "VR.land" in inspect.getsource(promote.rollback) or "vault_round" in inspect.getsource(promote.rollback)
    for module in (auto_restore, AR):
        src = inspect.getsource(module)
        assert "shutil.copy" not in src, module.__name__
        assert "_atomic_write" not in src, module.__name__
    assert "rollback(" in inspect.getsource(AR._handle_rollback)


def test_a_manual_rollback_still_lands_through_the_vault_route(env):
    """The unattended contract must not have cost the human their escape hatch: the
    tool's own path still commits, still returns the sha, still reports no-change."""
    snap_ts = promote_a_bad_variant(env)
    before = head(env.vault)

    result = promote.rollback(env.cfg, snap_ts)

    assert result.get("refused") is None, result
    assert result["vault_commit"] and result["vault_commit"] != before
    assert env.read("SOUL.md") == GOOD_CONTRACT
    assert porcelain(env.vault) == ""
    assert "rollback to snapshot" in subject(env.vault)


def test_rolling_back_to_the_live_contract_reports_no_change_and_commits_nothing(env):
    """#1099's amendment to #506: the no-change case is decided by comparing content
    with HEAD, not by reading `land`'s exception text — with a partially refused
    restore a message match would report the *restored* file as uncommitted work."""
    snap_ts = promote_a_bad_variant(env)
    assert promote.rollback(env.cfg, snap_ts)["vault_commit"]
    before = head(env.vault)

    again = promote.rollback(env.cfg, snap_ts)

    assert again["no_change"] is True and not again.get("vault_commit"), again
    assert head(env.vault) == before, "a no-op rollback still committed an empty change"
    assert porcelain(env.vault) == ""


def test_no_restore_path_writes_a_prompt_file_without_a_commit(env):
    """The failure this item exists to close: bytes moving onto the contract while
    HEAD keeps pointing at the promotion and nothing records the change. The tree
    being clean *and* HEAD unmoved is the signature of a copy without a commit, so
    the assertion is that one of the two must have moved."""
    snap_ts = promote_a_bad_variant(env)
    (env.vault / "lloyd" / "SOUL.md").write_text("an edit nobody committed\n",
                                                 encoding="utf-8")

    result = promote.rollback(env.cfg, snap_ts)

    assert porcelain(env.vault) == "", "a rollback left the contract modified, uncommitted"
    assert subject(env.vault).startswith("autoresearch: rollback"), subject(env.vault)
    assert result["vault_commit"] == head(env.vault)


def test_the_round_result_the_report_and_the_ledger_agree(env, monkeypatch):
    """One decline, three surfaces, one story. A restore that happened but went
    unreported is the same defect as one that never happened, one report later."""
    snap_ts = promote_a_bad_variant(env)

    result = drive_round(env, DECLINING_BASELINE, monkeypatch=monkeypatch)

    row = env.restore_rows()[-1]
    sha = row["vault_commit"]
    assert row["snapshot_ts"] == snap_ts, row
    assert json.dumps(result["post_promotion_restore"]).count(sha) == 1
    assert row["round_id"] == result["round_id"], row
    assert row["promoted_variant_id"] == BAD_VARIANT and row["restored_round_id"] == BAD_ROUND
    assert row["decline"] == pytest.approx(PROMOTED_MEAN - DECLINING_BASELINE, abs=1e-4), row
    assert row["noise_floor"] == NOISE_FLOOR
    assert row["restored_files"] == list(PROMPT_NAMES)
