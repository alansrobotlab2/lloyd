"""#1549 clauses 1 and 2: the frozen scenario manifest, and its hash.

Two separate refusals, and the difference matters. Clause 1 is about SHAPE —
a scenario that declares no planted input cannot be scored, and a loader that
quietly skipped it would report a suite of four when the file holds five.
Clause 2 is about IDENTITY — a manifest edited after the fact would score a
run against scenarios nobody froze, and the scorecard would carry a hash that
described a different file than the one that produced it.

Both live across a boundary from the thing that reads them: `behavioural.py`
loads the manifest off disk and `run_round.py` prints what it was given, so a
unit test on the validator alone would still pass if the round loaded nothing,
or loaded a stale hash. The hash assertions therefore recompute the digest from
the bytes on disk and compare, rather than re-reading the field the loader wrote.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.autoresearch import behavioural as B

# The frozen manifest itself is the fixture: a copy of it in tmp is a different
# artifact than the one a round scores against, and the suite is only frozen if
# the checked-in bytes are the ones that load.
RAW_MANIFEST = yaml.safe_load(B.SCENARIOS_MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest(tmp_path: Path, payload: dict, *, name: str = "scenarios.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def tamper(mutate, tmp_path: Path) -> Path:
    """Apply `mutate` to a copy of the frozen manifest and re-sign it.

    Re-signing is what makes a shape test a shape test: with the stale hash left
    in place every refusal would be the hash check firing first, and the
    per-scenario validation would never be reached.
    """
    payload = copy.deepcopy(RAW_MANIFEST)
    mutate(payload)
    payload["scenarios_hash"] = B.scenarios_hash(payload)
    return write_manifest(tmp_path, payload)


def scenario_with(payload: dict, scenario_id: str) -> dict:
    return next(s for s in payload["scenarios"] if s["id"] == scenario_id)


# ── clause 1: the manifest is frozen, and every scenario declares its parts ──

def test_the_frozen_manifest_sits_at_the_fixed_repo_path_and_clears_the_floor():
    """`MIN_SCENARIOS` is a floor the loader enforces, not a comment."""
    manifest = B.load_manifest()
    assert len(manifest["scenarios"]) >= B.MIN_SCENARIOS, (
        f"the suite is {len(manifest['scenarios'])} scenarios; four is the floor "
        f"the item names, and a smaller set is an anecdote, not a scorecard")
    assert {a["axis"] for a in manifest["axes"]} == {
        "uncertainty_preservation", "source_retention",
        "action_consistency", "stale_fact_action"}


@pytest.mark.parametrize("scenario_id", [s["id"] for s in RAW_MANIFEST["scenarios"]])
def test_every_scenario_declares_all_four_of_its_scoring_parts(scenario_id):
    """Planted input, expected observation, axis and checker id — all four, non-empty."""
    manifest = B.load_manifest()
    scenario = scenario_with(manifest, scenario_id)
    for field in ("planted_input", "expected_observation", "axis", "checker"):
        assert scenario[field], f"{scenario_id}: `{field}` is blank"
    assert scenario["checker"] in B.GRADERS


@pytest.mark.parametrize("field", ["planted_input", "expected_observation", "axis", "checker"])
def test_the_loader_refuses_a_scenario_missing_a_field_and_names_its_id(field, tmp_path):
    """The refusal has to name the scenario: "a manifest is malformed" sends
    whoever reads it back to a 200-line JSON file, which is the failure this
    clause exists to remove."""
    def drop(payload):
        del scenario_with(payload, "source-retention")[field]

    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(tamper(drop, tmp_path))
    message = str(exc.value)
    assert "source-retention" in message, f"the refusal must name the scenario: {message}"
    assert f"`{field}`" in message, f"the refusal must name the field: {message}"


def test_a_scenario_with_no_id_is_named_by_position(tmp_path):
    """Without an id there is nothing else to name, and "no id" must not read
    as a green manifest."""
    def blank_id(payload):
        scenario_with(payload, "act-on-known-fact")["id"] = ""

    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(tamper(blank_id, tmp_path))
    assert "scenario[" in str(exc.value)


def test_a_checker_that_is_not_a_grader_is_refused_by_name(tmp_path):
    """A manifest may not point at a grader that does not exist, or at an LLM
    judge: `checker` resolves into `GRADERS`, which holds only pure functions."""
    def unknown(payload):
        scenario_with(payload, "stale-fact-action")["checker"] = "llm_rubs_the_answer"

    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(tamper(unknown, tmp_path))
    assert "llm_rubs_the_answer" in str(exc.value)
    assert "stale-fact-action" in str(exc.value)


def test_a_manifest_below_the_four_scenario_floor_is_refused(tmp_path):
    def cut(payload):
        payload["scenarios"] = payload["scenarios"][:B.MIN_SCENARIOS - 1]

    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(tamper(cut, tmp_path))
    assert f"{B.MIN_SCENARIOS - 1} scenarios" in str(exc.value)


def test_an_axis_no_scenario_is_scored_on_is_still_a_refusal(tmp_path):
    def bogus_axis(payload):
        scenario_with(payload, "source-retention")["axis"] = "vibes"

    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(tamper(bogus_axis, tmp_path))
    assert "vibes" in str(exc.value)


# ── clause 2: the hash is recomputed, and never trusted ─────────────────────

def test_the_manifest_is_tracked_not_ignored():
    """The suite is only frozen if the commit carries it. `.gitignore:42` ignores
    `*.json` repo-wide and `.gitignore` is denied to a round, so a JSON manifest
    would sit in a worktree unseen by `git add -A`: the round would gate green and
    land a `main` with no scenarios and failing tests."""
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch",
                              str(B.SCENARIOS_MANIFEST_PATH.relative_to(B.REPO_ROOT)),
                              str(B.BASELINE_PATH.relative_to(B.REPO_ROOT))],
                             cwd=B.REPO_ROOT, capture_output=True, text=True)
    assert tracked.returncode == 0, (tracked.stdout, tracked.stderr)
    assert B.SCENARIOS_MANIFEST_PATH.name == "scenarios.yaml", (
        "the frozen manifest must be a format this repo tracks without editing "
        "`.gitignore`, which a round may not do")


def test_the_recorded_hash_matches_the_bytes_on_disk_right_now():
    """The suite is frozen only if the checked-in file authenticates itself."""
    assert RAW_MANIFEST["scenarios_hash"] == B.scenarios_hash(RAW_MANIFEST)


def test_editing_a_scenario_without_resigning_refuses_to_score(tmp_path):
    """The whole point of the hash: an edited scenario changes what is scored,
    and a scorecard printed beside an unedited hash would describe a suite that
    is not the one that ran."""
    payload = copy.deepcopy(RAW_MANIFEST)
    scenario_with(payload, "uncertainty-hardening")["planted_input"]["hedge_token"] = "definitely"
    path = write_manifest(tmp_path, payload)  # hash deliberately left at the frozen value

    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(path)
    message = str(exc.value)
    assert "scenarios_hash mismatch" in message, message
    # The refusal prints BOTH digests: the one the file claims and the one the
    # bytes actually produce, so the reader can tell a typo from an edit.
    assert RAW_MANIFEST["scenarios_hash"][:16] in message


def test_a_manifest_with_no_recorded_hash_refuses_to_score(tmp_path):
    """A missing hash is a refusal, not a blank to be filled in by the loader."""
    payload = copy.deepcopy(RAW_MANIFEST)
    payload.pop("scenarios_hash")

    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(write_manifest(tmp_path, payload))
    assert "no `scenarios_hash` recorded" in str(exc.value)


def test_the_hash_returned_by_the_loader_is_the_recomputed_one(tmp_path):
    """`_scenarios_hash` must be a measurement of the file, not the field it
    just verified: a scorecard stamped with the recorded value would carry a
    digest nobody checked."""
    payload = copy.deepcopy(RAW_MANIFEST)
    payload["scenarios_hash"] = B.scenarios_hash(payload)
    path = write_manifest(tmp_path, payload)

    manifest = B.load_manifest(path)
    assert manifest["_scenarios_hash"] == B.scenarios_hash(yaml.safe_load(path.read_text()))


def test_scoring_against_a_tampered_manifest_returns_a_refusal_not_a_score(tmp_path):
    """The runner's half of clause 2: a hash disagreement is reported as the
    instrument refusing, with `guardrail_hit: None` — a refused instrument must
    never print as "no axis regressed"."""
    payload = copy.deepcopy(RAW_MANIFEST)
    payload["scenarios"] = payload["scenarios"][:1]
    path = write_manifest(tmp_path, payload)  # frozen hash, edited body

    scorecard = B.score_dir(path, B.REFERENCE_TRACES_DIR, B.BASELINE_PATH)
    assert scorecard["status"] == "refused"
    assert scorecard["guardrail_hit"] is None, (
        "a suite that could not be authenticated has not established that "
        "nothing declined")
    assert scorecard["denominator"] == 0
    assert "scenarios_hash mismatch" in scorecard["refusal"]


def test_a_missing_manifest_file_is_a_refusal(tmp_path):
    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(tmp_path / "nope.json")
    assert "not found" in str(exc.value)
