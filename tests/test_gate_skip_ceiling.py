"""#3143 — the tests rung's skip ceiling is derived from the base tree, not a literal.

The rail used to be `counts["tests_skipped"] > 40`, an absolute. The suite's own
skip count had moved 32/33 -> 38 -> 39 with no round at fault, because what gets
counted is a property of the tree: the ten `tests/test_dashboard_responsive.py`
pins skip whenever a worktree has no `web/node_modules`. On 2026-10-22 the ledger
reported 39 skips against that 40 — a margin of one — and the two refusals the item
names (`SM_20261012_084845`, `SM_20261013_022036`) are both in the rounds store as
`41 tests skipped (limit 40)`, which is a round refused for a fact about the box.
Every past near-miss had been answered by a bigger literal, which is the lottery
#2205 named, and #2214's `PYTEST_MAX_SKIPPED = 55` (`dbfddc60`) is still unlanded
because raising the figure is a human ruling.

What landed instead derives the margin: `max(PYTEST_MAX_SKIPPED, base +
PYTEST_SKIP_HEADROOM)`, `base` being the median `tests_skipped` of the newest green
FULL `tests` rungs in the rounds store. The nodes below pin each edge of that
sentence — the arithmetic, the row filter, the median, the floor's exact spelling
and the vacuity mutant that depends on it, the rail biting one skip past the
derived figure, the numbers riding into `gate.json`, and the probe following
`state.ROUNDS_DIR` and nothing else.

One thing every node here inherits: `tests/conftest.py` points
`LLOYD_AUTOMOD_STATE` at a scratch directory before `scripts.automod.state` is
imported, so no test reads the machine's real ledger. That is not an accident this
file leans on twice — it is also what leaves the no-ledger fallback (the floor,
exactly 40) as the shape every existing ceiling node in
`tests/test_gate_parallel_tests.py` and `tests/test_skip_inventory.py` still sees.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.automod import gate as G

ROOT = Path(__file__).resolve().parents[1]
GATE_SRC = (ROOT / "scripts" / "automod" / "gate.py").read_text(encoding="utf-8")
GUARD_SRC = (ROOT / "scripts" / "maintenance" / "guard_vacuity.py").read_text(
    encoding="utf-8")


# --------------------------------------------------------------------------- #
# Seeding a ledger
# --------------------------------------------------------------------------- #

def _row(skipped: int, *, passed: int = 20_000, extra: dict | None = None) -> dict:
    """A `tests` rung's `data`, big enough to clear the suite's other two floors."""
    data = {"passed": passed, "failed": 0, "errors": 0, "xfailed": 1,
            "tests_skipped": skipped, "collected": passed + skipped,
            "pin_findings": [], "workers": 8}
    data.update(extra or {})
    return data


def _green(round_id: str, skipped: int, **data_extra) -> dict:
    """One green FULL `tests` rung as it appears in a round's report."""
    return {"name": "tests", "ok": True,
            "detail": f"20000 passed, 1 xfailed, {skipped} skipped (8 workers)",
            "seconds": 500.0, "data": _row(skipped, extra=data_extra or None)}


def _red(round_id: str, skipped: int) -> dict:
    """A `tests` rung that did not pass: no row, but a real directory to read."""
    rung = _green(round_id, skipped)
    rung["ok"] = False
    return rung


def _seed_round(rounds_dir: Path, round_id: str, rung: dict | None = None, *,
                raw: str | None = None) -> Path:
    """Write one round's `gate.json` under `rounds_dir`; hand back its directory.

    `raw` writes bytes instead of a report (a half-written file), and `rung=None`
    leaves a report with no `tests` rung in it.
    """
    d = rounds_dir / round_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "gate.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return d
    rungs: list[dict] = [{"name": "preflight", "ok": True, "detail": "tolerated",
                          "seconds": 1.0}]
    if rung is not None:
        rungs.append(rung)
    report = {"round_id": round_id, "base": "b" * 40, "head": "h" * 40, "ok": True,
              "changed_paths": [], "venv": None, "rungs": rungs}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return d


def _seed_ledger(tmp_path, monkeypatch, rows: dict[str, dict | None]) -> Path:
    """Stand a rounds store at `state.ROUNDS_DIR` holding `rows` (id -> rung)."""
    rounds_dir = tmp_path / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    for rid, rung in rows.items():
        _seed_round(rounds_dir, rid, rung)
    monkeypatch.setattr(G.S, "ROUNDS_DIR", rounds_dir)
    return rounds_dir


def _bare_gate(tmp_path, monkeypatch) -> G.Gate:
    """A Gate over whatever `state.ROUNDS_DIR` currently points at."""
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    wt = tmp_path / "wt"
    (wt / "tests").mkdir(parents=True, exist_ok=True)
    return G.Gate("SM_T", wt, "b" * 40, live_root=tmp_path)


def _gate(tmp_path, monkeypatch, rows: dict[str, dict | None]) -> G.Gate:
    """A Gate pointed at a seeded ledger (an empty `rows` is the floor case)."""
    _seed_ledger(tmp_path, monkeypatch, rows)
    return _bare_gate(tmp_path, monkeypatch)


def _counts(skipped: int, *, passed: int = 20_000, collected: int | None = None,
            **over) -> dict:
    counts = {"passed": passed, "failed": 0, "errors": 0, "xfailed": 1,
              "tests_skipped": skipped,
              "collected": passed + skipped if collected is None else collected,
              "workers": 8, "parallel_only_failures": 0, "pin_findings": []}
    counts.update(over)
    return counts


def _five_at(skipped: int, start: int = 1) -> dict[str, dict]:
    """Five green full rounds at `skipped`, ids ascending from `start`.

    Zero-padded on purpose: the probe orders round directories by NAME, because a
    round id is its own timestamp (`SM_%Y%m%d_%H%M%S`), so an unpadded seed id would
    not sort where the test thinks it does.
    """
    return {f"SM_20261001_{i:06d}": _green(f"SM_20261001_{i:06d}", skipped)
            for i in range(start, start + 5)}


# --------------------------------------------------------------------------- #
# Clause 1 — the ceiling is max(floor, base + headroom) over the newest green rows
# --------------------------------------------------------------------------- #

def test_the_ceiling_is_the_base_plus_headroom_and_not_the_floor(tmp_path, monkeypatch):
    """Clause 1: a seeded base of 39 computes 47, and 47 is not the floor's 40.

    Both constants are read out of the module rather than typed, so widening either
    one lands here as a changed expectation instead of a node that contradicts
    nothing; `47 == max(40, 39 + 8)` is the clause's own arithmetic, restated by the
    test that grants it.
    """
    assert G.PYTEST_SKIP_HEADROOM == 8, "clause 1 fixes the headroom at 8"
    assert G.PYTEST_MAX_SKIPPED == 40, (
        "#3143 keeps 40 as the FLOOR; raising the figure is #2214's human ruling")
    g = _gate(tmp_path, monkeypatch, {
        **_five_at(39),
        "SM_20261001_000009": _green("SM_20261001_000009", 38),
    })
    ceiling, base = G._skip_ceiling(g.round_id)
    assert base == 39, f"the median of five rounds at 39 must be 39, got {base}"
    assert ceiling == max(G.PYTEST_MAX_SKIPPED, base + G.PYTEST_SKIP_HEADROOM) == 47, (
        f"a base of 39 with headroom 8 must put the rail at 47, not {ceiling} — the "
        "gap between 47 and 40 is the whole of this item")
    ok, detail, _data = g._tests_pass(_counts(41), None, {})
    assert ok is True, (
        f"41 skips was the refusal for a year; against a derived rail of 47 it is "
        f"inside it, which is exactly the drift #3143 exists to tolerate: {detail}")


def test_the_base_is_the_median_of_the_five_newest_rows_not_every_row(tmp_path,
                                                                      monkeypatch):
    """Clause 1: the probe reads the NEWEST five rows and takes the median of those.

    Two properties in one node, because both are a question about which rows: seven
    exist, the two older ones sit at 10 skips, and the answer is still 39. A probe
    that averaged the whole store would answer ~25 and tighten the rail below
    today's real count; a mean of the newest five would answer 37.4.
    """
    rows = {f"SM_20250101_{i:06d}": _green(f"SM_20250101_{i:06d}", 10)
            for i in (1, 2)}
    rows.update(_five_at(39, start=3))
    _seed_ledger(tmp_path, monkeypatch, rows)
    assert G._skip_base_median() == 39, (
        "the median of the five newest rows is 39; anything else read older rounds "
        "or took a mean")


def test_a_median_of_a_thin_sample_is_a_count_not_a_half(tmp_path, monkeypatch):
    """Clause 1: rows at 44 and 45 give a base of 44, so the rail is 52.

    `statistics.median` of an even sample is the mean of its two middle values,
    which is 44.5 — and a ceiling is something an integer skip count gets compared
    against. Truncating is also the only direction that cannot widen the rail on a
    thin sample: 52 here, where rounding up would have granted 53.
    """
    _seed_ledger(tmp_path, monkeypatch, {
        "SM_20261001_000002": _green("SM_20261001_000002", 44),
        "SM_20261001_000001": _green("SM_20261001_000001", 45),
    })
    ceiling, base = G._skip_ceiling()
    assert (ceiling, base) == (52, 44), (
        f"44 and 45 must truncate to a base of 44 and a rail of 52, got "
        f"{(ceiling, base)} — a float base or a rounded-up one is the bug")


def test_the_median_resists_two_rounds_that_took_all_their_headroom(tmp_path,
                                                                   monkeypatch):
    """The ratchet this design could have started, pinned shut by the median.

    A green round can never exceed the rail, so a rail that followed the MAXIMUM of
    recent rounds would climb by `PYTEST_SKIP_HEADROOM` every time two rounds
    happened to use their headroom — a skip rail that widens whenever it is
    approached is not a rail. Two rounds at 47 against three at 39 has to leave the
    answer at 39: three of the five have to move before the rail does.
    """
    _seed_ledger(tmp_path, monkeypatch, {
        "SM_20261003_000001": _green("SM_20261003_000001", 47),
        "SM_20261002_000001": _green("SM_20261002_000001", 47),
        "SM_20261001_000003": _green("SM_20261001_000003", 39),
        "SM_20261001_000002": _green("SM_20261001_000002", 39),
        "SM_20261001_000001": _green("SM_20261001_000001", 39),
    })
    ceiling, base = G._skip_ceiling()
    assert (ceiling, base) == (47, 39), (
        f"two rounds that used all their headroom must not widen the rail: "
        f"{(ceiling, base)}")


# --------------------------------------------------------------------------- #
# Clause 1's row filter — green, not REUSED, not partial
# --------------------------------------------------------------------------- #

def test_a_partial_run_is_never_a_base_row(tmp_path, monkeypatch):
    """Clause 1: `data.partial` disqualifies a row, and the ledger shows why.

    The partial branch of `_tests_pass` runs only the changed test files and puts
    THAT run's skip count in the same `tests_skipped` field the whole-suite rail
    reads: `SM_20261021_222704` carries `tests_skipped: 0` under a detail of
    `pytest (partial, 1 changed test file(s) …): 15 passed, 0 skipped`. Four such
    rows sitting on one honest row at 39 would answer 0 and drop the rail to the
    floor — tightening the rail on the very rounds it exists to protect.
    """
    rows = {f"SM_20261009_{i:06d}": _green(f"SM_20261009_{i:06d}", 0, partial=True)
            for i in range(1, 5)}
    rows["SM_20261001_000001"] = _green("SM_20261001_000001", 39)
    _seed_ledger(tmp_path, monkeypatch, rows)
    assert G._skip_base_median() == 39, (
        "a partial rung reports 0 skips for a whole suite; counting it answers 0")
    assert G._skip_ceiling() == (47, 39), (
        "the same store must yield the derived rail, not the floor")


def test_a_reused_verdict_a_red_rung_and_an_unreadable_file_are_not_rows(
        tmp_path, monkeypatch):
    """Clauses 1 and 2: whatever the probe cannot stand a row on, it refuses on.

    Each row is handed to the reader directly as well as through a whole ledger,
    because the ledger's answer (the floor) is the same answer for all four and
    cannot say WHICH filter fired. The control row is deliberately outside the
    store: it proves the reader accepts a good rung, so the four refusals below are
    the filter working and not a path that never resolved.
    """
    rounds = tmp_path / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    reused = _green("SM_20261010_000001", 88)
    reused["detail"] = "REUSED (same head as the previous gate)"
    red = _red("SM_20261010_000002", 99)
    _seed_round(rounds, "SM_20261010_000001", reused)
    _seed_round(rounds, "SM_20261010_000002", red)
    _seed_round(rounds, "SM_20261010_000003", None)
    _seed_round(rounds, "SM_20261010_000004",
                raw='{"round_id": "SM_20261010_000004", "r')
    control = _seed_round(tmp_path / "outside", "SM_20261010_000005",
                          _green("SM_20261010_000005", 60))
    monkeypatch.setattr(G.S, "ROUNDS_DIR", rounds)

    for rid, why in (("SM_20261010_000001", "a REUSED verdict reports another run's counts"),
                     ("SM_20261010_000002", "a red rung is not a green baseline"),
                     ("SM_20261010_000003", "a report with no tests rung says nothing"),
                     ("SM_20261010_000004", "a corrupt gate.json is unreadable, not zero")):
        assert G._green_full_tests_data(rounds / rid / "gate.json") is None, (
            f"{rid} is {why}, so it cannot be a base row")
    assert G._green_full_tests_data(control / "gate.json") is not None, (
        "the control: a green full rung IS a row, so the refusals above are the "
        "filter and not a path that never resolved")
    assert G._green_full_tests_data(rounds / "SM_19990101_000000" / "gate.json") is None, (
        "an absent gate.json is a round with no row in it, not an exception")
    assert G._skip_ceiling() == (40, None), (
        "with only refused rows inside the store the rail is the floor, 40")


# --------------------------------------------------------------------------- #
# Clause 2 — no readable ledger means exactly 40, and the name must survive
# --------------------------------------------------------------------------- #

def test_no_readable_ledger_leaves_the_ceiling_exactly_on_its_floor(tmp_path,
                                                                   monkeypatch):
    """Clause 2: an absent store, an empty one and a corrupt one all answer 40.

    Asserted through the rung as well as the helper, because 40 is only the clause
    if the rung then refuses 41 and accepts 40 against it. This is also the path
    every existing ceiling node in the suite takes — conftest points the automod
    state dir at scratch — which is why those nodes keep passing untouched.
    """
    for label, prep in (
            ("absent", lambda d: None),
            ("empty", lambda d: d.mkdir(parents=True, exist_ok=True)),
            ("corrupt", lambda d: (
                (d / "SM_20261011_000001").mkdir(parents=True),
                (d / "SM_20261011_000001" / "gate.json").write_text(
                    "{ not json", encoding="utf-8"))),
    ):
        rounds = tmp_path / f"rounds-{label}"
        prep(rounds)
        monkeypatch.setattr(G.S, "ROUNDS_DIR", rounds)
        assert G._skip_ceiling() == (40, None), (
            f"a {label} rounds store must put the rail on the floor, exactly 40")
        g = _bare_gate(tmp_path / label, monkeypatch)
        ok, detail, _ = g._tests_pass(_counts(41), None, {})
        assert ok is False and "(limit 40)" in detail, (
            f"with a {label} store the rail is the floor, so 41 is refused against "
            f"40: {detail}")
        ok40, detail40, _ = g._tests_pass(_counts(40), None, {})
        assert ok40 is True, f"a run at exactly the floor must pass: {detail40}"


def test_the_floor_constant_keeps_the_spelling_two_guards_match():
    """Clause 2: `PYTEST_MAX_SKIPPED = 40` is still assigned that way, exactly once.

    Not decoration. Four nodes grep this file's bytes for that assignment
    (`tests/test_gate_parallel_tests.py:926` and `:966-971`,
    `tests/test_skip_inventory.py:561` and `:991`), and
    `scripts/maintenance/guard_vacuity.py` declares the guard with
    `site_needle="PYTEST_MAX_SKIPPED"` and `symbol="PYTEST_MAX_SKIPPED"` over a
    docstring that says a name which no longer appears stops the run rather than
    reprinting a stale number. A derived ceiling that renamed the floor would have
    to rewrite a docstring arguing against moving the rail, which is the argument
    this item refuses to have.
    """
    assert re.findall(r"^PYTEST_MAX_SKIPPED = (\d+)$", GATE_SRC, re.M) == ["40"], (
        "gate.py must assign PYTEST_MAX_SKIPPED = 40 exactly once, at line start, for "
        "the four source-text pins and the vacuity guard's site_needle")
    assert 'site_needle="PYTEST_MAX_SKIPPED"' in GUARD_SRC, (
        "the vacuity guard's needle moved; this node is the other half of its match")
    assert 'mutant=lambda p: _patched_call(gate_mod, {"PYTEST_MAX_SKIPPED": 10 ** 12}' \
        in GUARD_SRC, (
        "the guard's 10**12 mutant is what the next node proves still bites; if its "
        "spelling changed, so did that proof")


def test_the_vacuity_mutant_still_diverges_on_1600_skips(tmp_path, monkeypatch):
    """Clause 2: patching the floor to 10**12 has to change the rung's answer.

    The mutant the vacuity guard really runs is `_patched_call(gate_mod,
    {"PYTEST_MAX_SKIPPED": 10 ** 12}, …)` over its `skipped_1600` input class. It
    diverges only if the ceiling is built from THAT module global read at call time:
    a ceiling frozen at import, or derived from a copy of the number, would keep
    refusing 1600 skips under the mutant, the guard would find itself vacuous, and
    `guard_vacuity.py` stops the ladder over that. So this node is why the helper
    reads `PYTEST_MAX_SKIPPED` inside its body rather than at import.
    """
    g = _gate(tmp_path, monkeypatch, _five_at(39))
    ok, _detail, _ = g._tests_pass(_counts(1600), None, {})
    assert ok is False, "1600 skips behind 6000 passes must be refused by the real rail"
    monkeypatch.setattr(G, "PYTEST_MAX_SKIPPED", 10 ** 12)
    ok_mutated, _detail2, _ = g._tests_pass(_counts(1600), None, {})
    assert ok_mutated is True, (
        "the 10**12 mutant accepted 1600 skips, which means the ceiling is no longer "
        "derived from the attribute the guard patches — the guard would report itself "
        "vacuous and stop the run")


# --------------------------------------------------------------------------- #
# Clause 3 — the rail bites one skip past the derived figure
# --------------------------------------------------------------------------- #

def test_the_rail_bites_one_skip_past_the_derived_figure(tmp_path, monkeypatch):
    """Clause 3: with base 39 -> 47 passes and 48 is refused, and again at base 33.

    Two bases, not one. A hard-coded 47 would satisfy the first and fail the second,
    which is the only way to tell a derived rail from a literal that got edited
    upward — the exact move #2205 called a lottery.
    """
    for base in (39, 33):
        g = _gate(tmp_path / f"base{base}", monkeypatch, _five_at(base))
        ceiling = base + 8
        ok, detail, _ = g._tests_pass(_counts(ceiling), None, {})
        assert ok is True, (
            f"{ceiling} skips is exactly base {base} + headroom 8: {detail}")
        bad, detail2, _ = g._tests_pass(_counts(ceiling + 1), None, {})
        assert bad is False, (
            f"one skip past the derived figure must still be refused: {detail2}")
        assert ("a round that skips its way to green is not a round that passed"
                in detail2), (
            f"the refusal lost the sentence that says what a skip rail is for: {detail2}")
        assert f"(limit {ceiling})" in detail2, (
            f"the refusal has to name the ceiling it refused against (limit "
            f"{ceiling}): {detail2}")
        assert "(limit 40)" not in detail2, (
            f"against a base of {base} the rail is not 40, and a refusal quoting 40 "
            f"is one no reader can check: {detail2}")


def test_a_round_is_never_its_own_base(tmp_path, monkeypatch):
    """A re-gated round must not grade its own skip count into its own rail.

    `gate.json` is written at the END of a ladder, so on a re-gate the previous
    attempt's report is sitting in `state.ROUNDS_DIR/<this round>` looking exactly
    like a green full round. Without the exclusion a round that skipped 77 and went
    green would raise its own base to 77 and hand itself a rail of 85; with it the
    store holds only itself, which is the same as holding nothing, and the floor
    refuses 41. Both sides are asserted, so the node cannot pass vacuously.
    """
    g = _gate(tmp_path, monkeypatch, {"SM_T": _green("SM_T", 77)})
    assert G._skip_ceiling(g.round_id) == (40, None), (
        "excluding this round must leave no rows, not one row at 77")
    assert G._skip_ceiling() == (85, 77), (
        "the control: without the exclusion this round's own 77 IS the base, so the "
        "assertion above is the exclusion working")
    ok, detail, _ = g._tests_pass(_counts(41), None, {})
    assert ok is False, (
        f"a round must not widen its own rail out of its own last attempt: {detail}")


def test_the_probe_stops_reading_at_the_scan_cap(tmp_path, monkeypatch):
    """The bound is a bound: sixty directories, newest first, whatever lies older.

    Sixty-one red rounds sit in front of five green ones and the probe answers the
    floor — the right verdict for a ledger that has gone quiet, and the only way to
    test a cap at all, since removing it lets the same store reach the green rows.
    Both sides run, so the node fails whether the cap is missing or set to zero.
    """
    greens = {f"SM_20261001_{i:06d}": _green(f"SM_20261001_{i:06d}", 39)
              for i in range(1, 6)}
    blocked = {**greens,
               **{f"SM_20990101_{i:06d}": _red(f"SM_20990101_{i:06d}", 10)
                  for i in range(1, 62)}}
    _seed_ledger(tmp_path / "capped", monkeypatch, blocked)
    assert G._skip_base_median() is None, (
        "61 newer red rounds must exhaust the sixty-directory scan")
    assert G._skip_ceiling() == (40, None)

    reachable = {**greens,
                 **{f"SM_20990101_{i:06d}": _red(f"SM_20990101_{i:06d}", 10)
                    for i in range(1, 60)}}
    _seed_ledger(tmp_path / "reachable", monkeypatch, reachable)
    assert G._skip_base_median() == 39, (
        "with 59 reds in front of them the green rows are inside the cap, so the "
        "None above is the bound and not a probe that reads nothing")


# --------------------------------------------------------------------------- #
# Clause 4 — the margin is readable out of one round's report
# --------------------------------------------------------------------------- #

def test_the_pass_detail_names_the_base_and_the_headroom_it_used(tmp_path, monkeypatch):
    """Clause 4: the numbers are in the string, not only in the data.

    The base comes from a store the retention sweep eats, so a verdict naming only a
    total cannot be checked a week later. The note is pinned in both of its shapes:
    derived, and the `(base none, …)` fallback — which has to LOOK different, or a
    reader cannot tell a rail that was derived from one that quietly fell back.
    """
    g = _gate(tmp_path, monkeypatch, _five_at(39))
    ok, detail, data = g._tests_pass(_counts(39), None, {})
    assert ok is True, detail
    assert "skip ceiling 47 (base 39, headroom 8, floor 40)" in detail, (
        f"the green detail must carry the base and the headroom it was judged by: "
        f"{detail}")
    assert (data["skip_ceiling"], data["skip_base"]) == (47, 39)
    assert (data["skip_headroom"], data["skip_floor"]) == (8, 40)

    empty = _gate(tmp_path / "none", monkeypatch, {})
    ok2, detail2, data2 = empty._tests_pass(_counts(39), None, {})
    assert ok2 is True, detail2
    assert "skip ceiling 40 (base none, headroom 8, floor 40)" in detail2, (
        f"the fallback has to say it fell back: {detail2}")
    assert data2["skip_base"] is None, (
        "a `base: 0` in the data would read as a suite that skips nothing")


def test_the_note_survives_the_trip_into_gate_json(tmp_path, monkeypatch):
    """The seam: rung detail -> GateReport -> `gate.json` -> the next round's base.

    `_tests_pass`'s strings mean nothing until they are in the file the review rung,
    the owed-check job and the next round's probe all read, so the report goes
    through `GateReport.to_dict` and `state.write_gate_report` — the one writer of
    that shape — and the assertions are made on the bytes on disk and on what the
    probe then answers. That last part is the feedback loop itself: a round that
    skipped 39 becomes tomorrow's base.
    """
    rounds = _seed_ledger(tmp_path, monkeypatch, _five_at(30, start=9))
    g = G.Gate("SM_20261015_120000", tmp_path / "wt", "b" * 40, live_root=tmp_path)
    ok, detail, data = g._tests_pass(_counts(39), None, {})
    assert ok is True, detail
    report = G.GateReport(round_id=g.round_id, base=g.base, head="h" * 40)
    report.rungs.append(G.RungResult("tests", ok, detail, 500.0, data))
    path = G.S.write_gate_report(g.round_id, report.to_dict())
    assert path == rounds / g.round_id / "gate.json", (
        "the seam under test is the file the next round reads, not a copy of it")

    stored = json.loads(path.read_text(encoding="utf-8"))
    rung = next(r for r in stored["rungs"] if r["name"] == "tests")
    assert "skip ceiling 40 (base 30, headroom 8, floor 40)" in rung["detail"], (
        f"the base and headroom did not survive the report's serialisation: "
        f"{rung['detail']}")
    assert rung["data"]["skip_base"] == 30 and rung["data"]["skip_ceiling"] == 40, (
        "the data has to carry the pair too, or the sentence is the only copy")

    assert G._skip_base_median(exclude_round=g.round_id) == 30, (
        "excluding this round, the five seeded rows at 30 are still the base")
    assert G._skip_base_median() == 30, (
        "and including it, the newest five rows are this 39 and four 30s, whose "
        "median is 30 — the file just written is being read back as a row")


# --------------------------------------------------------------------------- #
# Clause 5 — the ledger is redirectable by moving one module attribute
# --------------------------------------------------------------------------- #

def test_the_probe_follows_state_rounds_dir_and_nothing_else(tmp_path, monkeypatch):
    """Clause 5: moving `state.ROUNDS_DIR` moves the ledger, and nothing else does.

    `base_probe_excerpts_path` reads the store the same way for the same reason
    (the comment over `BASE_PROBE_EXCERPTS_FILE`): a probe that hard-coded
    `~/.local/state/lloyd-automod` could not be unit-tested without writing to the
    real rounds store, which is the one thing a test must never do. Both halves are
    here: a seeded store is followed, and an empty one is followed away from it.
    """
    seeded = _seed_ledger(tmp_path, monkeypatch, _five_at(39))
    assert G._skip_ceiling() == (47, 39)
    assert (seeded / "SM_20261001_000001" / "gate.json").is_file()

    monkeypatch.setattr(G.S, "ROUNDS_DIR", tmp_path / "moved")
    assert G._skip_ceiling() == (40, None), (
        "the seeded store is still on disk and full, and only the module attribute "
        "moved — a probe that reads anything but that attribute answers 47 here")


def test_the_other_floors_still_fire_under_a_derived_rail(tmp_path, monkeypatch):
    """The rails this change had to leave alone still refuse what they refused.

    The item's own check 4: `DASHBOARD_PINS_NOT_EXECUTED`, `PYTEST_MIN_COLLECTED`
    and `PYTEST_MIN_PASSED` unchanged and still firing. Both floors are exercised on
    a round whose skips sit well inside a derived rail, so the refusal cannot be the
    skip rail doing their job; and the pin finding has to reach the detail BESIDE
    the new clause, not behind it — the findings are the part a long detail gets to
    lose to the 2000-char cap (#2726), so their position is the assertion.
    """
    assert G.PYTEST_MIN_COLLECTED == 1000 and G.PYTEST_MIN_PASSED == 1000
    assert G.PIN_FINDING_PREFIX == "DASHBOARD_PINS_NOT_EXECUTED"
    g = _gate(tmp_path, monkeypatch, _five_at(39))

    ok, detail, _ = g._tests_pass(_counts(10, collected=5), None, {})
    assert ok is False and "collected" in detail and "1000" in detail, (
        f"the collected floor stopped firing: {detail}")
    ok2, detail2, _ = g._tests_pass(_counts(10, passed=5, collected=1015), None, {})
    assert ok2 is False and "collected but not run" in detail2, (
        f"the passed floor stopped firing: {detail2}")

    counts = _counts(39)
    counts["pin_findings"] = ["DASHBOARD_PINS_NOT_EXECUTED: 10 of 10 in "
                             "tests/test_dashboard_responsive.py"]
    ok3, detail3, _ = g._tests_pass(counts, None, {})
    assert ok3 is True, f"a finding is a finding, not a refusal: {detail3}"
    assert "DASHBOARD_PINS_NOT_EXECUTED" in detail3, (
        f"the pin finding vanished from a detail that now carries a ceiling clause: "
        f"{detail3}")
    assert "skip ceiling 47 (base 39, headroom 8, floor 40)" in detail3, detail3
    assert detail3.index("skip ceiling") < detail3.index("DASHBOARD_PINS_NOT_EXECUTED"), (
        f"the counts and the margin come first; the findings are the tail a clipped "
        f"detail loses: {detail3}")


def test_the_partial_branch_still_applies_no_whole_suite_floor(tmp_path, monkeypatch):
    """The derived rail is a whole-suite number, so the partial run must not meet it.

    `_tests_pass`'s partial branch runs one changed test file, which is why it
    applies none of the floors and why its rows are excluded from the base. A partial
    run at 0 skips has to pass, carry `partial: True`, and say nothing about a
    ceiling it never measured.
    """
    g = _gate(tmp_path, monkeypatch, _five_at(39))
    (g.worktree / "tests" / "test_dashboard_responsive.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8")
    g.report.changed_paths = ["tests/test_dashboard_responsive.py"]
    ok, detail, data = g._tests_pass(_counts(0), ["tests/test_dashboard_responsive.py"],
                                     {})
    assert ok is True and data.get("partial") is True, detail
    assert "skip ceiling" not in detail, (
        f"a partial run measured one file; a ceiling clause there would be a number "
        f"it cannot support: {detail}")
