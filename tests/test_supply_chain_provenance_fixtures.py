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


# ── #3206 clause 5: the eval reports the `fresh-version` shape separately, and holds
#    every one of them while holding none of the tree's own frozen pins

def test_the_eval_holds_every_planted_fresh_version_name_and_prints_its_n():
    """#3206 clause 5, first half: 100 % of the planted fresh-version names are held, and the
    shape is its own line rather than a line buried inside the squat total.

    Both halves matter. A rate alone can hide a shape: with 61 planted names, two held and
    59 not would still be a shape the eval never names, and #688's own history is that a
    shape nobody prints is a shape nobody notices going quiet. `planted_shapes` is keyed by
    the doc's `shape:` field, so `fresh-version 2` is the printed n.
    """
    summary = sc.run_fixtures_eval(sc.read_fixtures())

    fresh = [n for n in summary["planted_shapes"] if n == sc.FRESH_VERSION_SHAPE]
    assert fresh == [sc.FRESH_VERSION_SHAPE], (
        f"the eval reports no `fresh-version` shape (shapes: {sorted(summary['planted_shapes'])});"
        " an attack class the eval cannot name is an attack class it cannot regress on")
    n = summary["planted_shapes"][sc.FRESH_VERSION_SHAPE]
    assert n == 2, f"the fixture set planted {n} fresh-version names, expected 2"
    names = set(summary["fresh_planted"])
    assert len(names) == n, f"fresh_planted {sorted(names)} does not match n={n}"
    missed = sorted(names & set(summary["planted_missed"]))
    assert not missed, (
        f"{n - len(missed)}/{n} planted fresh-version names held — these were allowed: "
        f"{missed}. Every one is a well-established name at a release two days old, which is"
        " exactly the artifact the 90-day and release-count facts pass")


def test_the_eval_holds_none_of_the_trees_own_frozen_pins_and_prints_that_n():
    """#3206 clause 5, second half: 0 of the repo's own `==` pins is held, over ALL of them
    (n printed), so the hold cannot be a control that blocks this box's own rebuild.

    The n is the point as much as the 0: an arm that measured three hand-picked pins would
    pass while `torch==2.13.0` bit the hold. And so is `unbounded`: each pin is priced at the
    commit that first recorded it (the youngest the artifact could be, read from this tree's
    own history), and a pin with no such floor is counted as UNBOUNDED rather than passed —
    otherwise "0 held" would be the same output whether the arm measured the pins or measured
    nothing, which is exactly how the first version of this arm failed (its history command was
    `git rev-list --follow`, which is not a rev-list option; git answered exit 129 with usage
    on stderr, the walk read stdout, and the arm reported `n=0`).
    """
    import datetime as dt
    arm = sc.run_fixtures_eval(sc.read_fixtures())["lock_pins"]
    on_disk = sc.lock_pins()

    assert arm["n"] == len(on_disk) and arm["n"] >= 100, (
        f"the arm measured {arm['n']} pins, the lock holds {len(on_disk)}; an arm over a "
        "subset cannot claim the tree's own setup is clear")
    assert arm["held"] == 0, (
        f"{arm['held']}/{arm['n']} of the repo's own pins are held: "
        f"{[h['name'] + '==' + h['version'] + ' (' + h['fact'] + ')' for h in arm['holding'][:5]]}")
    assert arm["ok"] and arm["file"] == sc.LOCK_PINS_FILE, arm
    assert arm["unbounded"] == 0, (
        f"{arm['unbounded']}/{arm['n']} pins had no commit-derived floor, so their age was "
        "never priced and the 0 above is an absence, not a result")
    assert arm["youngest_floor"], arm
    floor = dt.datetime.fromisoformat(arm["youngest_floor"])
    now = dt.datetime.now(dt.timezone.utc)
    assert (now - floor).days > sc.MIN_RELEASE_AGE_DAYS, (
        f"the youngest floor in the whole lock is {floor.date()}, inside the "
        f"{sc.MIN_RELEASE_AGE_DAYS}-day hold — the control arm would be pricing pins the hold"
        " could legitimately bite")


def test_every_lock_pin_is_dated_by_the_commit_that_first_recorded_it():
    """#3206 clause 5's floor, pinned as a measurement over this tree's history.

    `lock_pin_first_seen` walks `git log --follow` over the lock and keeps each `name==version`
    line at its FIRST sighting, so a pin that has been frozen since June is priced at June. That
    is a bound in the right direction: the artifact cannot predate the commit that recorded it,
    so if THAT is older than the hold, the real release date is too — the conclusion holds
    whatever PyPI says, which is what makes the arm decidable offline.

    A pin present only in the newest commit gets that commit's date, and `--follow` is what makes
    the walk see a rename: `requirements.txt` was renamed to `requirements.lock` in 2026-08-11
    `b6f04e8`, and the pins that predate the rename must not be attributed to that commit.
    """
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    on_disk = sc.lock_pins()
    floors = sc.lock_pin_first_seen(clock=now)

    assert len(on_disk) >= 100, len(on_disk)
    missing = [f"{n}=={v}" for n, v in on_disk if (n, v) not in floors]
    assert not missing, (
        f"{len(missing)} pins have no floor and are therefore unprovable, e.g. {missing[:5]}")
    assert all(f <= now for f in floors.values()), (
        "a floor in the future cannot bound an artifact's age from below")
    assert all((now - f).days > sc.MIN_RELEASE_AGE_DAYS for f in floors.values()), (
        f"the youngest floor {(max(floors.values())).date()} is inside the "
        f"{sc.MIN_RELEASE_AGE_DAYS}-day hold; an amendment made this week would pin its own"
        " date onto the artifact, which is not evidence about the artifact")
    # A clock in the past is not allowed to invent floors it cannot see.
    assert sc.lock_pin_first_seen(clock=now - dt.timedelta(days=365 * 3)) == {}, (
        "a clock before the lock's history still produced floors: the clock bound is what "
        "keeps a future commit from dating an artifact")


def test_a_fresh_version_fixture_whose_command_disagrees_fails_loudly():
    """#3206 clause 5's oracle cannot be satisfied by an inconsistent fixture.

    A doc that says `version: 2.2.0` while its command says `pip install mcp==2.1.9` would
    otherwise be priced on a version nobody asked for, and the 100 % would be a number about
    the fixture's typo.
    """
    import datetime as dt
    clock = dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc)
    facts = sc.RegistryFacts(name="mcp", exists=True, first_release=clock,
                             release_count=40, source="fixture", observed_at=clock,
                             version_upload_times={"2.2.0": clock - dt.timedelta(days=1)},
                             latest_version="2.2.0")
    with pytest.raises(ValueError, match="2.1.9"):
        sc.evaluate_version_hold("mcp", command="pip install mcp==2.1.9", facts=facts,
                                 now=clock)
    # And the agreeing case is held, so the raise is the only disagreement path.
    assert sc.evaluate_version_hold("mcp", command="pip install mcp==2.2.0", facts=facts,
                                    now=clock)


# ── #3206 clause 5's two n's, and the arm `measure_fixtures` stamps them from ──

def test_the_printed_report_carries_the_fresh_version_and_lock_pin_ns():
    """#3206 clause 5: "with both n's printed", and printed means in the report a reviewer
    reads, not only in the dict a test can reach.

    The two rates already printed cannot carry either number. `block rate 100.0%` is 63 planted
    names of six shapes, so the fresh-version arm could sit at 0/2 inside a headline that still
    reads 100 %, and the false-block rate is over the 180 *names* in `control:` — a claim about
    whether a name looks new, not about whether the hold bites a frozen `==` pin. So each arm
    gets its own line, with its own denominator named on it.
    """
    text = sc.format_fixtures_eval(SUMMARY)

    fresh_n = len(SUMMARY["fresh_planted"])
    assert f"fresh-version installs: {SUMMARY['fresh_held']}/{fresh_n}" in text, text
    assert fresh_n == 2 and SUMMARY["fresh_held"] == 2, (
        f"the printed line is only the claim it is if the numbers behind it are the item's: "
        f"held {SUMMARY['fresh_held']} of {fresh_n} planted")
    arm = SUMMARY["lock_pins"]
    assert f"{arm['file']} pins (control): {arm['held']}/{arm['n']}" in text, text
    assert arm["held"] == 0, "the printed 0 is a claim about the tree's own rebuild"
    # The bound travels with the number: `0/180` on its own would read as "measured old on
    # PyPI", which offline is precisely what was NOT measured. What was measured is the commit
    # that first recorded each pin, and the printed line says so and gives the youngest.
    assert str(arm["youngest_floor"])[:10] in text, text
    assert "commit that first recorded it" in text, text


def test_measure_fixtures_stamps_expected_blocked_from_the_arm_that_decides():
    """A `fresh-version` fixture is decided by the version hold in the GENERATOR too.

    `measure_fixtures` writes `expected_blocked:` into the committed document, and it used to
    ask only the name policy. On a fixture built for #3206 — first release years old, dozens of
    releases, the pinned artifact two days out — the name policy answers "clear", so the next
    `fixtures --measure` would have written `expected_blocked: false` beside `expect: block` on
    a fixture the eval holds. The two readers now choose their arm through one predicate.
    """
    clock = dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc)
    facts = sc.RegistryFacts(
        name="mcp-fork", exists=True,
        first_release=clock - dt.timedelta(days=1_200), release_count=41,
        source="pypi", observed_at=clock,
        version_upload_times={"2.2.0": clock - dt.timedelta(days=2)},
        latest_version="2.2.0")
    reg = sc.MappingRegistry({"mcp-fork": facts, "brandnewinvented": sc.RegistryFacts(
        name="brandnewinvented", exists=False, source="pypi", observed_at=clock)},
        name="scripted")
    doc = {"evaluated_at": clock.isoformat(), "planted": [
        {"name": "mcp-fork", "shape": sc.FRESH_VERSION_SHAPE, "expect": "block",
         "version": "2.2.0", "command": "pip install mcp-fork==2.2.0",
         "facts": {"exists": True, "measured": True,
                   "first_release": (clock - dt.timedelta(days=1_200)).isoformat(),
                   "release_count": 41, "observed_at": clock.isoformat(),
                   "latest_version": "2.2.0",
                   "versions": {"2.2.0": (clock - dt.timedelta(days=2)).isoformat()}}},
        {"name": "brandnewinvented", "shape": "invented", "expect": "block",
         "facts": {"exists": False}}]}

    out = sc.measure_fixtures(doc, registry=reg, control_names=[])

    assert out["planted"][0]["expected_blocked"] is True, (
        "a fresh-version fixture was stamped from the name policy — the committed document "
        "would then assert a verdict the eval does not produce")
    assert out["planted"][1]["expected_blocked"] is True, (
        "the name-shaped arm regressed: an unpublished name is still decided by the shipped "
        "name policy, not by the version hold")
    assert sc.entry_is_version_shaped(doc["planted"][0]) is True
    assert sc.entry_is_version_shaped(doc["planted"][1]) is False
