"""#1549 clauses 1 and 2: the frozen scenario manifest, and its hash.

Two separate refusals, and the difference matters. Clause 1 is about SHAPE —
a scenario that declares no planted input cannot be scored, and a loader that
quietly skipped it would report a suite one scenario short of the one the file
actually holds.
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
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.autoresearch import behavioural as B

# The frozen manifest itself is the fixture: a copy of it in tmp is a different
# artifact than the one a round scores against, and the suite is only frozen if
# the checked-in bytes are the ones that load.
RAW_MANIFEST = yaml.safe_load(B.SCENARIOS_MANIFEST_PATH.read_text(encoding="utf-8"))
#: The frozen digest, which is one of the two inputs a reserve seat is derived
#: from; the other is the month stamp. Tests that pin the seat name it explicitly
#: so a re-frozen manifest moves the seat rather than breaking the node.
DIGEST = B.scenarios_hash(RAW_MANIFEST)


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


#: What each axis must declare for the discrimination bar to be reachable at all.
#: The bar needs `denominator_a` AND `denominator_b` >= 2 on every axis, and an
#: axis's denominator is the number of scenarios declaring it, so 2 per axis is
#: the smallest suite that can ever clear it.
MIN_SCENARIOS_PER_AXIS = 2


def test_every_declared_axis_carries_at_least_two_scenarios():
    """#2368: one scenario per axis makes the bar unreachable, not merely unrun.

    `build_scorecard` appends exactly one value per scenario onto its axis, so the
    axis `denominator` is a count of SCENARIOS and never a count of rows: a
    one-scenario axis prints 1 through one capture or twenty, and the bar at
    `scripts/autoresearch/behavioural.py` needs 2 on both sides of EVERY declared
    axis. Before this node the suite declared `uncertainty_preservation`,
    `source_retention` and `stale_fact_action` at one scenario each, so no repeat
    pair — including the paused-pool window #2196 is owed — could have cleared it.
    Pinned here rather than only at a scorecard because this is where a scenario is
    added: an edit that drops an axis back to one scenario fails with that axis
    named, instead of three files away inside a pair comparison that merely prints
    a smaller number.

    The exact counts, not just the floor: 2/2/2/2 over the four axes is the whole
    shipped suite, so this also fails if a scenario is silently repointed to an
    axis that already had two and a thin one is left thin-looking-but-unequal.
    """
    per_axis = Counter(str(s["axis"]) for s in B.load_manifest()["scenarios"])
    assert per_axis == {"uncertainty_preservation": 2, "source_retention": 2,
                        "action_consistency": 2, "stale_fact_action": 2}, dict(per_axis)
    for axis, count in sorted(per_axis.items()):
        assert count >= MIN_SCENARIOS_PER_AXIS, (
            f"`{axis}` declares {count} scenario(s), so its axis denominator is "
            f"{count} however many captures run, and the discrimination bar's "
            f"`denominator >= {MIN_SCENARIOS_PER_AXIS}` can never be met on it")


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


# ── #1659 clause 4: reserve seats are derived from month + hash, and rotate ───

def _synthetic(count: int) -> dict:
    """A re-signed manifest-shaped payload with `count` ids, to size the seat."""
    payload = copy.deepcopy(RAW_MANIFEST)
    payload["scenarios"] = [
        copy.deepcopy(RAW_MANIFEST["scenarios"][0]) | {"id": f"sc-{i}"}
        for i in range(count)]
    payload["scenarios_hash"] = B.scenarios_hash(payload)
    return payload


@pytest.mark.parametrize("count,expected", [(1, 1), (5, 1), (6, 2), (10, 2),
                                            (11, 3), (12, 3), (15, 3)])
def test_the_seat_is_ceil_twenty_percent_and_never_empty(count, expected):
    """20% of the suite rounded up, and at least one scenario whatever its size.

    The floor is the load-bearing half at this suite's size: 20% of 5 rounds to
    one, and a rule that produced zero would quietly retire the hold-out. The
    ceiling is what stopped the 2-of-5 (40%) seat #1843 found and retired; at the
    8 scenarios #2368 shipped the seat is 2 again, which is 25% and inside it.
    """
    manifest = _synthetic(count)
    digest = B.scenarios_hash(manifest)
    for stamp in ("2026-08", "2026-12", "2031-01"):
        seat = B.reserved_scenario_ids(manifest["scenarios"], manifest_hash=digest,
                                       stamp=stamp)
        assert len(seat) == expected, (
            f"{count} scenarios at {stamp}: expected ceil(20%) = {expected} "
            f"reserved, got {seat}")
        assert set(seat) <= {s["id"] for s in manifest["scenarios"]}


#: The window the rotation is judged over: one calendar month per declared
#: scenario, starting at the first month the seat existed (2026-08). Derived from
#: the manifest rather than spelled out, because with a two-scenario seat a cycle
#: is NOT five months any more — see the node below.
def _convention_months(count: int) -> list[str]:
    return [f"{2026 + (8 + i - 1) // 12:04d}-{(8 + i - 1) % 12 + 1:02d}"
            for i in range(1, count + 1)]


def test_the_seat_rotates_every_month_across_the_whole_suite():
    """Every scenario is held out exactly once inside one cycle of the rotation.

    Rotation is the reason a seat exists: a fixed 20% is a scenario that stops
    being measured, not one measured out of sight of whoever proposes changes.

    One cycle is `len(ids)` MONTHS, not `len(ids)` scenarios — and that is the
    arithmetic #2368 moved when it took the suite from 5 to 8. A seat is
    `reserve_count(n)` = ceil(20%), so a 5-scenario suite had a one-scenario seat
    and five months covered it exactly; an 8-scenario suite has a TWO-scenario
    seat, so five months hold out at most ten slots of a different shape and, with
    the digest's ordering, miss scenarios entirely. Counting months off the
    scenario count is what makes the cycle the right length at any size, and the
    same month has to give the same answer twice.
    """
    manifest = B.load_manifest()
    ids = [str(s["id"]) for s in manifest["scenarios"]]
    months = _convention_months(len(ids))
    seats = [frozenset(B.reserved_scenario_ids(manifest["scenarios"],
                                               manifest_hash=DIGEST, stamp=m))
             for m in months]

    assert len(set(seats)) == len(months), (
        f"the seat repeated across consecutive months, so it is not rotating: {seats}")
    assert set().union(*seats) == set(ids), (
        "one full cycle of the rotation did not cover the suite: "
        f"{sorted(set(ids) - set().union(*seats))} were never held out")
    held = [len(s) for s in seats]
    assert set(held) == {B.reserve_seat_count(len(ids))}, (
        f"months of this cycle held out {sorted(set(held))} scenarios, not the "
        f"one seat size ceil({len(ids)}/{B.RESERVE_ONE_IN}) = "
        f"{B.reserve_seat_count(len(ids))}")
    # Coverage means UNIFORM, not merely non-empty. Eight months of a
    # two-scenario seat is sixteen slots over eight scenarios, so every scenario
    # sits out exactly two months of the cycle — a rotation that held one
    # scenario out eight times and another once would pass the union above.
    seat_size = B.reserve_seat_count(len(ids))
    held_by = Counter(sid for seat in seats for sid in seat)
    assert set(held_by) == set(ids), f"a non-scenario was held out: {sorted(held_by)}"
    assert set(held_by.values()) == {seat_size}, (
        f"over the {len(months)}-month cycle the suite was not covered evenly: "
        f"{dict(sorted(held_by.items()))}, expected every scenario held out "
        f"{seat_size} times")
    assert B.reserved_scenario_ids(manifest["scenarios"], manifest_hash=DIGEST,
                                    stamp="2026-10") == \
        B.reserved_scenario_ids(manifest["scenarios"], manifest_hash=DIGEST,
                                    stamp="2026-10"), \
        "the seat is drawn per call rather than derived from its inputs"


def test_the_seat_moves_when_the_frozen_manifest_hash_moves():
    """The hash is a real input, so re-freezing the suite reseats the rotation."""
    manifest = B.load_manifest()
    seats = {B.reserved_scenario_ids(manifest["scenarios"], manifest_hash=f"{i:064x}",
                                     stamp="2026-10")[0]
             for i in range(1, 13)}
    assert len(seats) > 1, (
        "the reserved scenario did not move across twelve different manifest "
        f"hashes ({seats}), so the hash is not an input to the seat")


def test_the_shipped_manifest_carries_no_hand_set_reserve_flags():
    """The seats the file used to hand out are gone; the rule lives in code.

    Three of the five scenarios declared `reserve:` by hand and two said true —
    40% of the suite against a ruled 20% — and no code read the flag, so the file
    asserted a hold-out nobody performed. A field nothing honours is worse than no
    field, because it reads as a control that is in place.
    """
    assert [s["id"] for s in RAW_MANIFEST["scenarios"] if "reserve" in s] == []
    assert RAW_MANIFEST["scenarios_hash"] != \
        "b5e57afd14bc030b0a64ff0cd9ce418388c0ad37c3a4d06cbbc94f48d949bcdc", \
        "retiring the flags has to move the frozen digest, or the digest in the " \
        "file is not the digest of the file"


def test_a_manifest_that_sets_reserve_by_hand_is_refused(tmp_path):
    """The flag is retired, so a manifest reintroducing it is rejected by name.

    Ignoring it silently would leave the file claiming a hold-out that
    `reserved_scenario_ids` does not perform — the state #1549 shipped in, and
    the reason the rotation was owed.
    """
    path = tamper(lambda p: scenario_with(p, "source-retention").__setitem__(
        "reserve", True), tmp_path)
    with pytest.raises(B.ScenarioManifestError) as exc:
        B.load_manifest(path)
    message = str(exc.value)
    assert "source-retention" in message, message
    assert "reserved_scenario_ids" in message, (
        "the refusal has to point at the rule that replaced the flag")


def test_the_scorecard_reports_the_seat_it_derived_and_the_month_it_derived_it_for():
    """`reserve_stamp` plus the seat, so a reader can reproduce the pick.

    The report used to print 'withholding mechanism deferred' beside a list that
    had been hand-set in a YAML file. A derived seat names its month, and the
    per-scenario flags have to agree with the derived set.
    """
    manifest = B.load_manifest()
    scorecard = B.build_scorecard(manifest=copy.deepcopy(manifest), traces={},
                                  baseline=B.load_pinned_baseline(),
                                  scenarios_digest=manifest["_scenarios_hash"],
                                  trace_source="reference", stamp="2026-11")
    seat = B.reserved_scenario_ids(manifest["scenarios"],
                                   manifest_hash=manifest["_scenarios_hash"],
                                   stamp="2026-11")

    assert scorecard["reserve_stamp"] == "2026-11"
    assert scorecard["reserved_scenarios"] == seat
    flagged = sorted(row["id"] for row in scorecard["scenarios"] if row["reserve"])
    assert flagged == sorted(seat), (
        f"the rows marked reserved disagree with the derived seat: {flagged} vs {seat}")
