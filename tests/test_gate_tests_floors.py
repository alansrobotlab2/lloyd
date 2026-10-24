"""The `tests` rung's two pytest floors are derived, not absolute (#3250).

`architecture/testing.md`'s 2026-09-24 review log records a gate that collected
9,583-9,620 tests, which is the suite `PYTEST_MIN_COLLECTED = 1000` and
`PYTEST_MIN_PASSED = 1000` were budgeted against: a tenth of it. The green full
`tests` rungs of 2026-10-24 report `collected` 21,298-21,348, so the same two
literals had fallen under 5% of the thing they bound.

That matters because `collected` in a rung's data is NOT pytest's collection report.
`gate._PYTEST_COUNT_WORDS` documents it as the sum of what the run reports having RUN
plus SKIPPED, and `pytest -q` prints no collection report at all — so `collected`
cannot distinguish "the suite ran" from "a twentieth of it ran and the runner stopped
early". An xdist worker dying without printing a `FAILED` line is exactly that shape
(#2999, #3205), and `_refused_whole_suite_rerun` means the whole-suite re-run that
would have caught it is refused. A full run reporting 1,200 collected and 1,200 passed
of a ~20,000-node suite cleared both floors and the rung printed `all green`.

#3143 solved this exact problem for the skip rail six lines below those constants:
derive the figure from the rounds ledger each run, keep the literal as the floor, and
print every number in the detail. These nodes pin the same treatment for the two
pytest floors, including the constraint the skip rail never had to respect —
`scripts/maintenance/guard_vacuity.py` probes both constants by mutating the live
module, and its healthy fixture `HEALTHY = suite(6289, 6274, 13, 2)` bounds the
fraction at 6289/21337 = 0.294 when the probe runs standalone against the live store.

Everything here reads the store through `scripts.automod.state.ROUNDS_DIR` and nothing
else, the way `tests/test_gate_skip_ceiling.py` does: no mock of the derivation, no
path handed to the gate.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "automod" / "gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("automod_gate", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = _load_module()


def _counts(collected: int, passed: int, skipped: int = 40) -> dict:
    """What one whole-suite `pytest -q` reported, in the shape `_tests_pass` reads."""
    return {"passed": passed, "failed": 0, "errors": 0, "xfailed": 1,
            "tests_skipped": skipped, "collected": collected}


def _green(round_id: str, collected: int, passed: int, *, skipped: int = 40,
           data_extra: dict | None = None, detail: str | None = None,
           ok: bool = True) -> dict:
    """A `tests` rung for one round, as `gate.json` stores it.

    `detail` defaults to the green-run spelling. It matters: `_green_full_tests_data`
    reads the numbers out of the rung's `data` dict precisely because a detail string
    spells a green run's counts and a refusal's counts differently, so a base built by
    parsing prose would mix two kinds of row into one median.
    """
    return {"name": "tests", "ok": ok, "status": "passed" if ok else "failed",
            "detail": detail or (
                f"{passed} passed, 1 xfailed, {skipped} skipped (8 workers)"),
            "seconds": 500.0,
            "data": {"passed": passed, "failed": 0, "errors": 0, "xfailed": 1,
                     "tests_skipped": skipped, "collected": collected,
                     **(data_extra or {})}}


def _seed_ledger(tmp_path, monkeypatch, rows: dict[str, dict]) -> Path:
    """Stand a rounds store at `state.ROUNDS_DIR` holding `rows` (round id -> rung)."""
    rounds_dir = tmp_path / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    for round_id, rung in rows.items():
        d = rounds_dir / round_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "gate.json").write_text(json.dumps({"round_id": round_id,
                                                 "rungs": [rung]}), encoding="utf-8")
    monkeypatch.setattr(G.S, "ROUNDS_DIR", rounds_dir)
    return rounds_dir


def _no_ledger(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(G.S, "ROUNDS_DIR", tmp_path / "no-such-rounds-dir")


def _gate(round_id: str = "SM_UNDER_TEST"):
    return G.Gate(round_id, REPO_ROOT, "deadbeef")


# Five green full rows: `collected` 20,000, `passed` 19,900. With the module's own
# fraction that makes the applied floors int(0.25 * 20000) = 5,000 and
# int(0.25 * 19900) = 4,975 — the figures every claim below is measured against.
def _five_rows() -> dict[str, dict]:
    return {f"SM_202610{20 + i}_130000_abcdef": _green(f"SM_202610{20 + i}_130000_abcdef",
                                                       20000, 19900)
            for i in range(5)}


def test_a_truncated_full_run_is_refused_and_one_named_fraction_moves_both_floors(
        tmp_path, monkeypatch):
    """Clause 1: 1,200 of a 20,000-node suite is refused; the fraction is one constant.

    1,200 collected and 1,200 passed clears both absolute floors, so before #3250 this
    run was reported `all green` on a suite that had executed 6% of its nodes. Against
    the five green full rows of `collected` 20,000 / `passed` 19,900 the applied floors
    are 5,000 and 4,975, and the same counts are refused twice over — once by the
    collected floor, and, with collected lifted above it, by the passed floor too.

    Then the constant that binds the suite is set back out of the way and the very same
    counts PASS, which is what makes this a test of the derivation rather than of
    arithmetic: the refusal comes from the ledger, not from a bigger typed literal.

    Finally ONE module constant, `PYTEST_FLOOR_FRACTION`, is shown to move both figures
    together (0.5 -> 10,000 and 9,950). Two literals tuned apart is the lottery #2205
    named for the absolute floors, and it would make the probe's two guards for
    `PYTEST_MIN_COLLECTED` and `PYTEST_MIN_PASSED` disagree by construction.
    """
    _seed_ledger(tmp_path, monkeypatch, _five_rows())
    g = _gate()
    assert G.PYTEST_MIN_COLLECTED == 1000 and G.PYTEST_MIN_PASSED == 1000
    assert G.PYTEST_FLOOR_FRACTION == 0.25, "the fraction these figures are computed with"

    ok, detail, data = g._tests_pass(_counts(1200, 1200), None, {})
    assert ok is False, (
        f"1,200 collected of a 20,000-node suite cleared the absolute floors, which is "
        f"the hole #3250 closes: {detail!r}")
    assert "did the round delete tests?" in detail, detail
    assert "(floor 5000)" in detail, detail

    ok2, detail2, data2 = g._tests_pass(_counts(20000, 1200), None, {})
    assert ok2 is False and "was collected but not run" in detail2, detail2
    assert "(floor 4975)" in detail2, detail2

    monkeypatch.setattr(G, "PYTEST_FLOOR_FRACTION", 0.0)
    ok3, detail3, _ = g._tests_pass(_counts(1200, 1200), None, {})
    assert ok3 is True, (
        "with the fraction at 0 both floors fall back to the 1000 literals and these "
        f"counts must pass, or the refusal above is not the derivation's doing: {detail3!r}")

    monkeypatch.setattr(G, "PYTEST_FLOOR_FRACTION", 0.5)
    ok4, detail4, _ = g._tests_pass(_counts(6000, 6000), None, {})
    assert ok4 is False, (
        f"one fraction has to move BOTH floors; at 0.5 the collected floor is 10,000 and "
        f"6,000 is refused: {detail4!r}")
    assert "(floor 10000)" in detail4, detail4
    ok5, detail5, _ = g._tests_pass(_counts(10000, 6000), None, {})
    assert ok5 is False and "(floor 9950)" in detail5, (
        f"the passed floor moved to 9,950 by the same constant: {detail5!r}")


def test_the_base_is_the_five_newest_green_full_rows_and_not_this_round(
        tmp_path, monkeypatch):
    """Clause 2: exactly the rows `_green_full_tests_data` accepts, newest five, no self.

    Seven green full rows carry `collected` 1,000 through 7,000, so the five newest are
    3,000/4,000/5,000/6,000/7,000 and their median is 5,000. Then four rows that must
    NOT be read are seeded newest-first, each with a `collected` big enough to move that
    median if it were: a red `tests` rung (99,999), a green rung whose detail starts
    `REUSED` (88,888 — a second attempt's copy of numbers some other run already
    contributed), a green rung whose data carries a `partial` key (77,777 — the
    selection path, whose collected is a subtree by construction), and THIS round's own
    green rung (66,666), which sorts first by name so the exclusion cannot be an
    accident of scan order.

    A round that could join its own base is the failure mode the whole derivation turns
    on: one green full run would pin the base at its own `collected` forever.
    """
    rows = {f"SM_202610{i:02d}_130000_aaaaaa": _green(f"SM_202610{i:02d}_130000_aaaaaa",
                                                      1000 * i, 1000 * i)
            for i in range(1, 8)}
    rows["SM_Z10_130000_zzzzzz"] = _green("SM_Z10_130000_zzzzzz", 99999, 99999, ok=False)
    rows["SM_Z11_130000_zzzzzz"] = _green("SM_Z11_130000_zzzzzz", 88888, 88888,
                                          detail="REUSED from SM_Z10 (same head sha)")
    rows["SM_Z12_130000_zzzzzz"] = _green("SM_Z12_130000_zzzzzz", 77777, 77777,
                                          data_extra={"partial": True})
    rows["SM_ZZZ_UNDER_TEST"] = _green("SM_ZZZ_UNDER_TEST", 66666, 66666)
    _seed_ledger(tmp_path, monkeypatch, rows)

    base = G._pytest_base_median("collected", "SM_ZZZ_UNDER_TEST")
    assert base == 5000, (
        "the five newest ACCEPTED rows are collected 3,000-7,000, median 5,000; the red, "
        f"REUSED, `partial` and self rows are 99,999/88,888/77,777/66,666: {base}")
    assert G._pytest_base_median("passed", "SM_ZZZ_UNDER_TEST") == 5000

    floors = G._pytest_floors("SM_ZZZ_UNDER_TEST")
    assert floors == (1250, 1250, 5000, 5000), (
        f"each floor is the fraction applied to its own base (0.25 x 5,000 = 1,250), "
        f"with the bases returned alongside so the refusal can name them: {floors}")

    g = _gate("SM_ZZZ_UNDER_TEST")
    ok, _detail, _data = g._tests_pass(_counts(1250, 1250), None, {})
    assert ok is True, "a run exactly AT the derived floor is a pass, not a refusal"
    ok2, detail2, _ = g._tests_pass(_counts(1249, 1249), None, {})
    assert ok2 is False and "(floor 1250)" in detail2, detail2
    assert "SM_ZZZ_UNDER_TEST" not in detail2, (
        "the refusal names the figure and the base, not this round's own row: "
        f"{detail2!r}")


def test_with_no_ledger_both_floors_are_the_literals_read_from_the_module_at_call_time(
        tmp_path, monkeypatch):
    """Clause 3: no readable ledger -> exactly `PYTEST_MIN_COLLECTED`/`PYTEST_MIN_PASSED`.

    `tests/conftest.py` points `LLOYD_AUTOMOD_STATE` at scratch before
    `scripts.automod.state` is imported, so every unit test here and every subprocess
    the vacuity probe spawns under pytest sees a rounds dir with nothing in it. That is
    why the probe's `HEALTHY = suite(6289, 6274, 13, 2)` still measures what its note
    says it measures: a derived figure must be *absent*, not merely large, in that
    environment.

    The mutant half is the part that cannot be satisfied by a memoized or copied value:
    with `PYTEST_MIN_PASSED` set to 0 the way
    `scripts/maintenance/guard_vacuity.py:527` does, a collected-but-not-run input has
    to flip from BLOCK to PASS. A floor read once into an instance attribute would keep
    blocking, the probe would report `guard=BLOCK mutant=BLOCK`, and the ladder would
    lose the only witness that this guard governs anything.
    """
    _no_ledger(monkeypatch, tmp_path)
    g = _gate()
    assert G._pytest_floors("SM_UNDER_TEST") == (
        G.PYTEST_MIN_COLLECTED, G.PYTEST_MIN_PASSED, None, None), (
        "no ledger to read means the literals, exactly, with no base to name")

    collected_not_run = _counts(6289, 0)          # the probe's HEALTHY `collected`
    ok, _detail, _data = g._tests_pass(collected_not_run, None, {})
    assert ok is False, "the passed floor blocks a suite that collected and ran nothing"

    monkeypatch.setattr(G, "PYTEST_MIN_PASSED", 0)   # the probe's mutant
    ok_m, detail_m, _ = g._tests_pass(collected_not_run, None, {})
    assert ok_m is True, (
        "guard=BLOCK mutant=BLOCK would score the guard DEAD-BRANCH and the divergence "
        f"is what #2104 clause 4 asks for: {detail_m!r}")

    monkeypatch.setattr(G, "PYTEST_MIN_PASSED", 1000)
    monkeypatch.setattr(G, "PYTEST_MIN_COLLECTED", 0)
    # `collected` BELOW its floor while `passed` clears its own, which no real run can
    # report (`passed <= collected` always) — the reason the probe scores this guard
    # VACUOUS by construction. What the node pins is not a suite shape but the read:
    # the collected figure follows the module attribute, so the floor-at-0 mutant moves
    # it exactly as the passed one does.
    collapsed = _counts(999, 1000)   # collected 999, passed 1,000
    ok_c, detail_c, _ = g._tests_pass(collapsed, None, {})
    assert ok_c is True, (
        "the collected floor is read from the module the same way the passed floor is, "
        f"so its own floor-at-0 mutant moves it: {detail_c!r}")


def test_mutating_the_literal_on_the_module_moves_the_fallback_even_with_a_base(
        tmp_path, monkeypatch):
    """Clause 3's second half, the half the first commit could not exercise (#3250).

    `_one_pytest_floor` used to take its literal as a parameter, so a mutation of
    `gate.PYTEST_MIN_PASSED` was visible only in `_pytest_floors`, three frames up, and
    the fallback inside the helper still read the unmutated 1000. With no ledger that
    was invisible — both paths yielded 1000 — but with a base in the store the mutant's
    floor-at-0 verdict never materialised, which is precisely what
    `scripts/maintenance/guard_vacuity.py` needs in order to witness the guard.

    So the figure is now read from this module's own `globals()` at call time, and the
    node pins it in the one state where the two readings differ: a REAL base of 20,000
    collected / 19,900 passed, so the derived figures are 5,000 / 4,975, and then the
    literal set to 0. The floor has to become 0 — the derivation may not resurrect a
    deliberately disarmed guard — and a zero-pass run has to pass.
    """
    _seed_ledger(tmp_path, monkeypatch, _five_rows())
    assert G._pytest_floors("SM_UNDER_TEST")[:2] == (5000, 4975), (
        "a base has to be in play, or this node is the no-ledger case again")

    g = _gate("SM_UNDER_TEST")
    zero_pass = _counts(20_000, 0)
    ok, _detail, _ = g._tests_pass(zero_pass, None, {})
    assert ok is False, "the derived passed floor blocks it before the mutation"

    monkeypatch.setattr(G, "PYTEST_MIN_PASSED", 0)
    assert G._pytest_floors("SM_UNDER_TEST")[1] == 0, (
        "the fallback figure did not follow the module attribute, so the literal is "
        "being read somewhere other than the frame that applies it")
    ok_m, detail_m, _ = g._tests_pass(zero_pass, None, {})
    assert ok_m is True, (
        "guard=BLOCK mutant=BLOCK: the floor-at-0 mutant cannot disarm the guard while a "
        f"base is in the store, and the probe would score it DEAD-BRANCH: {detail_m!r}")

    # Same read on the collected side, whose mutant is the other pytest floor's.
    monkeypatch.setattr(G, "PYTEST_MIN_PASSED", 1000)
    monkeypatch.setattr(G, "PYTEST_MIN_COLLECTED", 0)
    assert G._pytest_floors("SM_UNDER_TEST")[0] == 0
    # passed 1,000 clears its own derived floor (4,975 is above it only after the base
    # is read), so pin the collected figure independently of the passed one.
    assert G._one_pytest_floor("passed", "SM_UNDER_TEST") == 4975, (
        "disarming collected must not disarm passed — the two read separate names")


def test_the_refusal_names_the_figure_its_base_and_its_floor_in_the_skip_rail_spelling(
        tmp_path, monkeypatch):
    """Clause 4: the detail carries every number, in the shape #3143 settled, and `gate.json` too.

    Three things a reader needs from one round's report, and none of them can be
    recovered from `gate.json` a week later if the detail names only the applied figure:
    the base it was derived from, the fraction, and the literal it may not fall under.
    `(base none, ...)` is the one spelling that distinguishes a figure that was derived
    from one that silently fell back, and the fallback is the state every unit test and
    every probe subprocess is in.

    The literal still appears as `floor {PYTEST_MIN_COLLECTED}`, which is what
    `tests/test_automod_gate.py:3035` asserts and what keeps a moved figure from quietly
    moving the goalposts; the applied figure appears as `(floor N)`, the prefix
    `tests/test_gate_tests_rung_parse.py:483` asserts. Both spellings are pinned here
    against each other so a rewording that satisfies one and breaks the other is caught
    in the same file that changed the prose.
    """
    _seed_ledger(tmp_path, monkeypatch, _five_rows())
    g = _gate()
    ok, detail, data = g._tests_pass(_counts(1200, 1200), None, {})
    assert ok is False, detail
    assert "collected floor 5000 (base 20000, fraction 0.25, floor 1000)" in detail, detail
    assert f"floor {G.PYTEST_MIN_COLLECTED}" in detail, (
        "tests/test_automod_gate.py:3035 asserts this exact substring: the refusal has "
        f"to name the number it fell back to: {detail!r}")
    assert detail.startswith("only 1200 tests collected (floor 5000)"), detail
    assert data["collected_floor"] == 5000 and data["collected_floor_base"] == 20000, data

    ok2, detail2, data2 = g._tests_pass(_counts(20000, 1200), None, {})
    assert ok2 is False, detail2
    assert "passed floor 4975 (base 19900, fraction 0.25, floor 1000)" in detail2, detail2
    assert f"floor {G.PYTEST_MIN_PASSED}" in detail2, detail2
    assert data2["passed_floor"] == 4975 and data2["passed_floor_base"] == 19900, data2

    _no_ledger(monkeypatch, tmp_path)
    ok3, detail3, data3 = g._tests_pass(_counts(500, 500), None, {})
    assert ok3 is False, detail3
    assert "collected floor 1000 (base none, fraction 0.25, floor 1000)" in detail3, (
        f"with no ledger the detail must SAY so, the way the skip rail's does: {detail3!r}")
    assert data3["collected_floor_base"] is None, data3

    note = G._pytest_floor_note(5000, 1000, 20000, "collected")
    skip_note = G._skip_ceiling_note(48, 40)
    for field in ("base", "floor"):
        assert f"{field} " in note and f"{field} " in skip_note, (note, skip_note)


def test_the_probe_measures_its_rung_fixtures_with_no_rounds_store_behind_them(tmp_path,
                                                                               monkeypatch):
    """The vacuity probe has to judge its pytest fixtures against the LITERAL floors.

    `scripts/maintenance/guard_vacuity.py` drives the real `Gate.rung_tests` over
    absolute suite shapes — `HEALTHY = suite(6289, 6274, 13, 2)` and `skipped_1600 =
    suite(6000, 4300, 1600, 100)` — and mutates only the literals. Since #3250 the
    guard's figure is derived from the rounds ledger, so a probe that can read a real
    store measures a guard at 5,325 against a mutant still holding 1,000: both verdicts
    BLOCK on every class, `PYTEST_MIN_PASSED class=collected_not_run` and
    `PYTEST_MAX_SKIPPED class=skipped_1600` fall from witnessed to DEAD-BRANCH, and the
    rate moves 3/13 -> 5/13 on an unchanged commit. Measured on the live store on
    2026-10-24, before the scrub was added. That is the report measuring its
    environment, which is what the probe's own `:88` promises it does not do.

    So the assertion is on the driver, not on a fraction: the probe points
    `gate_mod.S.ROUNDS_DIR` at a directory that does not exist, the same state
    `tests/conftest.py` puts pytest-mode in, and with no store the figure IS the
    literal — which is the whole point of #3250 clause 3.
    """
    scrub = re.search(
        r"gate_mod\.S\.ROUNDS_DIR\s*=\s*rung_tmp / \"([^\"]+)\"",
        (REPO_ROOT / "scripts" / "maintenance" / "guard_vacuity.py").read_text(
            encoding="utf-8"))
    assert scrub, (
        "the probe's rung driver no longer scrubs the rounds store, so its absolute "
        "fixtures are judged against a derived floor and three live guards score "
        "VACUOUS/DEAD-BRANCH on any host that has a ledger")
    scrubbed = Path(tempfile.gettempdir()) / "lloyd-guard-vacuity-rung" / scrub.group(1)
    assert not scrubbed.exists(), (
        f"the scrubbed path {scrubbed} exists, so the floors are derived after all")

    # With no store the figure is exactly the literal, and both fixtures clear it.
    _no_ledger(monkeypatch, tmp_path)
    floors = G._pytest_floors("GUARD_VACUITY_PROBE")
    assert floors[:2] == (G.PYTEST_MIN_COLLECTED, G.PYTEST_MIN_PASSED), floors
    assert G.PYTEST_MIN_PASSED < 4300 < 6274, (
        "skipped_1600 (4,300 passed) and HEALTHY (6,274 passed) must both PASS the "
        f"figure the probe measures; floors are {floors}")

    # And the reason the scrub is load-bearing rather than tidy, stated with numbers:
    # at the live base of 21,300 passed the derived figure is 5,325, which is ABOVE
    # skipped_1600's 4,300. So the fraction's real ceiling on this host is not the
    # 6289 / 21337 = 0.294 bound #3250's triage derived from the HEALTHY fixture — it
    # is 4300 / 21300 = 0.20, and only the scrub is what keeps the looser bound
    # sufficient. Anyone raising the fraction has to satisfy one of the two.
    assert int(G.PYTEST_FLOOR_FRACTION * 21300) > 4300, (
        "the derived figure no longer exceeds the probe's smaller fixture, so the scrub "
        "is no longer what saves it and the fixtures can be re-anchored on a derived floor")
    assert G.PYTEST_FLOOR_FRACTION * 21337 <= 6289, (
        "the fraction now blocks the probe's own HEALTHY fixture even with no store: "
        f"{G.PYTEST_FLOOR_FRACTION} x 21337 > 6289")