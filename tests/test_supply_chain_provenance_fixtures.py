"""Clause 6 of #688: the fixture eval, and the claim it is allowed to make.

A provenance heuristic lives or dies on its false-block rate, and the only honest
way to know either rate is a set of names whose registry facts are *measured*
rather than imagined: planted squat-shaped names that must all be blocked, and the
repo's own dependency set as a control that must be blocked by none of them. So
this file asserts both rates at 100 % and 0 %, and then asserts four things about
the fixture set itself — things that would otherwise let a green eval stand for
nothing:

* every planted name is labelled with the shape it is meant to represent, and the
  fresh shape is present, because freshness is the fact the policy reads;
* every control fact was read off pypi.org, so the false-block rate is a
  measurement, not a claim about names nobody checked;
* no planted name's *measured* registry reality has quietly become unblockable (a
  name the registry now serves as old and multi-release is no longer a squat);
* the eval's clock is pinned, because a fixture whose pass depends on today's date
  fails months later for reasons no diff caused.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys

import pytest

from app.harness import supply_chain as sc

SUMMARY = sc.run_fixtures_eval(sc.read_fixtures())


def test_every_planted_squat_shaped_name_is_blocked():
    summary = SUMMARY

    assert summary["planted_total"] >= 20, (
        f"only {summary['planted_total']} planted names — the item asks for ~20, "
        "and a smaller set makes a 100% rate a coincidence")
    assert summary["planted_missed"] == [], (
        f"planted squat-shaped names the policy let through: "
        f"{summary['planted_missed']}")
    assert summary["block_rate"] == 1.0


def test_none_of_the_repos_own_dependencies_is_blocked():
    summary = SUMMARY

    assert summary["control_total"] >= 50, (
        f"the control set is {summary['control_total']} names; the repo declares "
        "over 180, and a control this file hand-picked could be cherry-picked")
    assert summary["control_false_blocks"] == [], (
        f"the policy would block the repo's own stack: {summary['control_false_blocks']}")
    assert summary["false_block_rate"] == 0.0


def test_the_two_rates_are_printed_together():
    """Threshold changes are reported by rate, per the item's step 5, so the printed
    output is the artefact a reviewer reads. A percentage rather than a bare count:
    `12/20` and `12/60` must never print as the same headline."""
    text = sc.format_fixtures_eval(SUMMARY)

    assert "block rate 100.0%" in text, text
    assert "false-block rate 0.0%" in text, text
    assert "clock:" in text, "the clock the rates were computed against is part of the claim"


def test_the_fixture_set_covers_both_halves_of_the_item_and_labels_them():
    """`invented` is the hallucinated-name half and `fresh` the published-yesterday
    half; the policy's two facts are registry silence and age, so an unlabelled set
    could be all of one shape and still print 100 %."""
    shapes = SUMMARY["planted_shapes"]

    assert {"invented", "typosquat", "fresh"} <= set(shapes), shapes
    assert shapes["fresh"] >= 5, shapes
    assert sum(shapes.values()) == SUMMARY["planted_total"], "every entry is labelled"


@pytest.mark.parametrize("summary_key", ["planted_missed", "control_false_blocks",
                                         "contradictions", "expectations_mismatch",
                                         "control_declared_missing"])
def test_the_committed_fixture_set_has_no_self_check_findings(summary_key):
    """Each of these is a way the two headline rates could be green and the eval
    still be measuring the wrong thing."""
    assert SUMMARY[summary_key] == [], (summary_key, SUMMARY[summary_key])


def test_the_eval_clock_is_pinned_so_the_rates_are_reproducible():
    assert SUMMARY["clock"] == "pinned", (
        "freshness is measured against now(); with a wall clock a fixture that "
        "passes today fails in ninety days with no diff behind it")


def test_the_control_is_the_dependency_files_and_nothing_invented():
    """The item's control is *the repo's own dependency set*, so the set is read
    from requirements.txt/requirements.lock here rather than carried in the YAML —
    a hand-copied list would drift into blocking a name the repo has since added,
    which is precisely the false block this eval exists to detect."""
    declared = set(sc.read_dependency_set(sc._repo_root()))
    control = {sc.normalize_dist_name(name) for name in SUMMARY["control_names"]}

    assert control, "an empty control set makes the false-block rate vacuous"
    assert control <= declared, sorted(control - declared)[:5]


def test_a_planted_name_whose_measured_reality_is_benign_is_a_contradiction():
    """A name becomes useless as a squat fixture the moment the registry serves it
    old and multi-release — it would have to be replaced, not re-labelled, and the
    eval says so instead of letting 100 % stand on a name that is no longer a squat."""
    entry = {"name": "nolongerasquat", "shape": "invented", "facts": {"exists": False},
             "reality": {"measured": True, "exists": True,
                         "first_release": "2015-01-01T00:00:00Z", "release_count": 40}}

    assert sc._reality_contradicts(entry, dt.datetime(2026, 9, 24, 12,
                                                      tzinfo=dt.timezone.utc))


def test_a_planted_name_with_synthetic_facts_is_not_a_contradiction():
    """The fresh-shaped fixtures are planted, so their facts are invented on purpose
    and `reality:` records what the registry really says. Only a *benign* measured
    reality is a contradiction; a reality that still gets blocked is fine, because
    the eval measures the policy, not the author's guess at the world."""
    entry = {"name": "fastflow", "shape": "fresh", "facts": {"exists": False},
             "reality": {"measured": True, "exists": True,
                         "first_release": "2026-08-03T13:43:39Z", "release_count": 1}}

    assert not sc._reality_contradicts(entry, dt.datetime(2026, 9, 24, 12,
                                                          tzinfo=dt.timezone.utc))


def test_measure_fixtures_keeps_planted_facts_and_records_reality_alongside():
    """The generator writes the facts the policy reads *and* what the registry says,
    in separate fields. Collapsing them would let a future `--measure` overwrite a
    planted fixture with the real registry answer and silently empty the set."""
    reg = sc.MappingRegistry({
        "brandnewinvented": sc.RegistryFacts(name="brandnewinvented", exists=False,
                                             source="pypi",
                                             observed_at=dt.datetime(
                                                 2026, 9, 24, tzinfo=dt.timezone.utc)),
    }, name="scripted")
    doc = {"evaluated_at": "2026-09-24T12:00:00Z",
           "planted": [{"name": "brandnewinvented", "shape": "invented",
                        "expect": "block", "facts": {"exists": False}}]}

    out = sc.measure_fixtures(doc, registry=reg, control_names=[])

    assert out["planted"][0]["reality"]["measured"] is True
    assert out["planted"][0]["expected_blocked"] is True
    assert out.get("contradictions", []) == []


def test_the_eval_is_reachable_from_the_documented_entry_point():
    """`fixtures` is the eval, and the module is the documented entry point — pinning
    the command to the function the nightly wiring names is what keeps a rename from
    quietly dropping the eval out of the run that reports it."""
    cmd = sc.fixtures_eval_command()

    assert cmd[-3:] == ["-m", "app.harness.supply_chain", "fixtures"], cmd
    assert cmd[0] == sys.executable
    proc = subprocess.run([*cmd], capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "block rate 100.0%" in proc.stdout, proc.stdout


if __name__ == "__main__":
    print(sc.format_fixtures_eval(SUMMARY))
