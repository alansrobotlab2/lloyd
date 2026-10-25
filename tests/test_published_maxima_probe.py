"""Pins `scripts/maintenance/published_maxima_probe.py` (#3304).

The probe is deliberately NOT part of the promotion path: `tests/conftest.py:93`
points `LLOYD_AUTOMOD_STATE` at a scratch dir for every pytest run, so a
ledger-reading test is vacuously green inside a gate round — which is why the
recorder is off-suite and the live guard stays `over_budget()` in
`workers/sources/automod_regression.py`. What IS pinned here is the probe's own
contract, over a fixture ledger whose numbers are the ones the 2026-10-25
round read off the real thing: fresh max 1,160.230 ms and graded max 477.877 ms.

Two figures the tests exist to keep honest:

  * n=1,204 with 733 of those rows in the rotated archives — 471 in the live
    ledger alone, under 40 % of the population. A probe that read only
    `promotions.jsonl` would be the second version of the mistake #3037 found.
  * The drift it must catch is 905.3 against 1,160.230 (21.97 % under) and 401.3
    against 477.877 (16.02 % under). #3304 asked for a 25 % trigger and the item
    review refused it: a margin that size reads GREEN over both arms, so the
    rule under test is STRICT equality upward, and `test_a_25_percent_margin…`
    below is the regression that keeps anyone from re-adding one.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "scripts" / "maintenance" / "published_maxima_probe.py"
# The module whose published re-check command the probe must match, and
# whose provenance comment carries that command verbatim.
MODULE = ROOT / "workers" / "sources" / "automod_regression.py"


def _probe():
    spec = importlib.util.spec_from_file_location("published_maxima_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _probe()

#: The two maxima the live ledger carries, and the rows that carry them — the
#: same stamps and commits the refreshed publication cites.
FRESH_MAX = 1160.230
GRADED_MAX = 477.877
FRESH_STAMP = "2026-10-20T21:37:13Z"
GRADED_STAMP = "2026-10-21T22:13:46Z"
FRESH_COMMIT = "64914c67a4e7497925b4a2649d77a69409fcb8b5"
GRADED_COMMIT = "d2847a349274281193734f1b79900721731d99df"


def _row(ts: str, before: float, after: float, commit: str = "0" * 40,
         pin: str = "conf.d/agent-qmd-daemon.conf"):
    """One `regression_check` ledger line in the shape the writer emits. `ts` IS
    `created_at`: the first draft carried a separate default stamp, which pinned
    every row to 2026-10-01 and made the out-of-era fixture row count."""
    return json.dumps({
        "event": "regression_check",
        "created_at": ts,
        "commit": commit,
        "stage": "detached",
        "pin": {"daemon": {"source": pin}},
        "detail": {"latency_ms_avg": {"before": before, "after": after,
                                      "delta": after - before, "armed": False}},
    })


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    """A state dir: live ledger plus one rotated archive, and three rows that
    must NOT count (pre-era, a different pin, and a detail the writer never
    emits). Era-D rows in population: 4 — 3 live, 1 archived."""
    state = tmp_path / "state"
    state.mkdir()
    live = [
        _row(FRESH_STAMP, FRESH_MAX, 311.247, FRESH_COMMIT),
        _row(GRADED_STAMP, 737.177, GRADED_MAX, GRADED_COMMIT),
        _row("2026-10-11T05:17:04Z", 620.1, 227.7),
        # excluded: before the era floor
        _row("2026-09-21T23:59:59Z", 9999.9, 9999.9),
        # excluded: not the paired-check pin
        _row("2026-10-11T06:00:00Z", 8888.8, 8888.8,
             pin="conf.d/something-else.conf"),
        # excluded: an empty `detail`, which the module's truthy clause drops and
        # which both readers must therefore ignore. Kept EMPTY rather than a bare
        # string because the module's quoted command does `(r['detail'] or {}).get(…)`
        # and raises AttributeError on a str; the probe survives that shape (pinned
        # below) but it cannot live in a ledger both readers walk over.
        json.dumps({**json.loads(_row("2026-10-11T07:00:00Z", 7777.7, 7777.7)),
                    "detail": {}}),
    ]
    (state / "promotions.jsonl").write_text("\n".join(live) + "\n", encoding="utf-8")
    archived = [_row("2026-10-05T09:00:00Z", 500.0, 200.0)]
    (state / "promotions-archive-202610.jsonl.gz").write_bytes(
        gzip.compress(("\n".join(archived) + "\n").encode()))
    return state


def test_the_probe_and_the_module_command_agree_over_one_ledger(
        ledger: Path, tmp_path: Path):
    """Clause 1: the filter is the module's own, RUN rather than quoted.

    The first draft of this node compared the probe's predicate to the module's by
    eye and called it verbatim; the gate's review refused that as unpinned, and it was
    right — a probe can share every one of those substrings and still count a
    different population. So the extraction `test_automod_regression.py` already does
    is reused here and the published command is EXECUTED over this fixture, patched to
    read this directory rather than the real ledger. `Path.home` is patched, not
    `$HOME`: the command reads `pathlib.Path.home()/'.local/state/lloyd-automod'` and
    passwd wins over `$HOME` on this box — the same reason `app/data_root.py:140`
    refuses to read `$HOME`. The six numbers it prints must be the six the probe
    reports for the same files. That is equivalence, not resemblance: a floor moved by
    a minute, a dropped archive glob, `>` for `>=`, or one-decimal rounding that
    differs now fails here instead of silently publishing figures nobody agreed on.
    """
    src = MODULE.read_text(encoding="utf-8")
    # Anchored on the era-D re-check command's own import list, not on `-c "`: the
    # module publishes TWO `-c` commands (§8.1's vault-read latency probe at
    # `workers/sources/automod_regression.py:261` is the other), and a search for the
    # delimiter alone silently extracts the wrong one.
    head = '-c "import json,pathlib,gzip,statistics'
    start = src.index(head) + len('-c "')
    # +1 keeps the command's own closing paren: `index(')"')` lands on the `)`
    # that is glued to the quote, and dropping it truncates `print(` into a
    # SyntaxError.
    raw = src[start:src.index(')"', start) + 1]
    assert "regression_check" in raw and "promotions-archive-" in raw, (
        "the extracted command is not the era-D re-check")
    # Unwrap the comment: every continuation line of a command quoted in a comment
    # carries its `#     ` prefix, and a backslash-newline JOINED with a line that
    # starts with `#` comments out the rest of the joined logical line — the failure
    # is silent (exit 0, no output), which is the worst way for this rail to break.
    # Same unwrap `test_automod_regression.py` does for the same command.
    code = "\n".join(ln[1:].strip() if ln.lstrip().startswith("#") else ln
                      for ln in raw.split("\n"))

    home = tmp_path / "home"
    (home / ".local" / "state").mkdir(parents=True)
    (home / ".local" / "state" / "lloyd-automod").symlink_to(ledger)
    runner = tmp_path / "run-module-command.py"
    runner.write_text(
        "import pathlib\n"
        f"pathlib.Path.home = lambda: pathlib.Path({str(home)!r})\n"
        "exec(compile(" + repr(code) + ", '<module-recheck-command>', 'exec'))\n",
        encoding="utf-8")
    run = subprocess.run([sys.executable, str(runner)], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, f"the module's own command failed: {run.stderr[-400:]}"
    printed = run.stdout.split()
    assert len(printed) == 6, (
        f"the module command printed {len(printed)} fields, not n / fresh median / "
        f"graded median / fresh max / graded max / over-1600: {run.stdout!r}")

    measured = P.summarize(ledger)
    mine = [measured["n"], measured["fresh_median"], measured["graded_median"],
            measured["fresh_max"], measured["graded_max"],
            measured["over_bound_count"]]
    assert [float(x) for x in printed] == [float(x) for x in mine], (
        f"the probe and the module's published command disagree over one ledger: the "
        f"command printed {printed}, the probe reported {mine}. The era floor, the pin "
        "restriction, the archive glob, the `.gz` decompression or the one-decimal "
        "rounding drifted, and the published figures are no longer what that command "
        "answers — which is the #3037 failure with a fresh date on it")
    # The one place the probe is deliberately STRICTER than the quoted command: a
    # `detail` that is a bare string makes the module's `(r['detail'] or {}).get(…)`
    # raise AttributeError, while the probe skips the row. A scheduled job that dies on
    # one malformed line reads as a broken probe and stops bounding the drift entirely,
    # so the robustness is the point and is pinned here rather than assumed.
    ragged = tmp_path / "ragged"
    ragged.mkdir()
    (ragged / "promotions.jsonl").write_text(
        _row("2026-10-11T05:17:04Z", 620.1, 227.7) + "\n"
        + json.dumps({**json.loads(_row("2026-10-11T08:00:00Z", 7777.7, 7777.7)),
                      "detail": "no latency recorded"}) + "\n", encoding="utf-8")
    assert [r["before"] for r in P.era_d_readings(ragged)] == [620.1], (
        "a malformed `detail` must be skipped, not fatal: the recorder runs "
        "unsupervised and a crash there ends the bound's protection silently")

    assert 4 == len(P.era_d_readings(ledger)) == measured["n"], (
        "the fixture stopped being a real era-D population of 4 rows (3 live + 1 "
        f"archived): readings={len(P.era_d_readings(ledger))}, measured "
        f"n={measured['n']}. Agreement above would then be two readings of nothing")



def test_the_archives_are_what_the_maxima_could_hide(ledger: Path, tmp_path: Path):
    """A probe reading only `promotions.jsonl` is the bug, not a variant.

    Moving the fresh max into the archive and leaving the live file's own worst
    at 620.1 must not change what the probe reports: on the live ledger 733 of
    the 1,204 era rows live only in archives.
    """
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "promotions.jsonl").write_text(
        _row("2026-10-11T05:17:04Z", 620.1, 227.7) + "\n", encoding="utf-8")
    (moved / "promotions-archive-202610.jsonl.gz").write_bytes(gzip.compress(
        (_row(FRESH_STAMP, FRESH_MAX, 311.247, FRESH_COMMIT) + "\n"
         + _row(GRADED_STAMP, 737.177, GRADED_MAX, GRADED_COMMIT) + "\n").encode()))
    live_only = P.era_d_readings(moved)
    assert len(live_only) == 3
    assert max(r["before"] for r in live_only) == FRESH_MAX


def test_the_measurement_carries_the_stamp_and_commit_of_each_maximum(ledger: Path):
    """The publication cites a stamp and a commit per arm; the probe must be the
    thing that produced those citations, not a script that invents them."""
    m = P.summarize(ledger)
    assert m["n"] == 4
    assert m["fresh_max"] == 1160.2 and m["fresh_max_exact"] == 1160.23
    assert m["graded_max"] == 477.9 and m["graded_max_exact"] == 477.877
    assert m["fresh_max_created_at"] == FRESH_STAMP
    assert m["graded_max_created_at"] == GRADED_STAMP
    assert m["fresh_max_commit"] == FRESH_COMMIT[:9]
    assert m["graded_max_commit"] == GRADED_COMMIT[:9]
    assert m["over_bound_count"] == 0
    assert m["budget"] == 1600.0, "the probe must check against the armed bound"
    assert m["span"] == ["2026-10-05T09:00:00Z", GRADED_STAMP], \
        "span is min/max of the stamp: file order runs backwards across archives"


def test_two_invocations_append_exactly_two_rows(ledger: Path, tmp_path: Path,
                                                monkeypatch, capsys):
    """One row per invocation, dated, and its maxima are the ones it printed.

    The journal home is the production data root, which does not exist under
    `_pipeline/automod/` until this script makes it — so this asserts creation
    too: a first run that died FileNotFoundError would read as a broken probe.
    """
    root = tmp_path / "data-root"
    monkeypatch.setattr("app.data_root.production_data_root", lambda: root)
    journal = root / "_pipeline" / "automod" / "published-maxima.jsonl"
    assert not journal.parent.exists()

    for _ in range(2):
        P.main(["--state-dir", str(ledger)])
    rows = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    assert len(rows) == 2, "the probe must append exactly one row per invocation"
    assert {r for r in rows[0]} == {
        "ts", "n", "fresh_median", "fresh_max", "graded_median", "graded_max",
        "over_bound_count", "published_fresh", "published_graded", "budget"}, \
        "the journal's key set is its schema; adding a key silently is a schema change"
    # `ts` must be a real dated stamp on every row. Not asserted strictly
    # increasing: the stamp is second-granular and two runs in one loop share it.
    for row in rows:
        assert datetime.fromisoformat(row["ts"]).tzinfo is not None, \
            "each row is dated with its own UTC stamp"
    assert rows[0]["n"] == 4 and rows[1]["n"] == 4

    out = capsys.readouterr().out
    printed_fresh = _printed_max(out, "fresh (before)")
    printed_graded = _printed_max(out, "graded (after)")
    assert rows[1]["fresh_max"] == printed_fresh == 1160.2
    assert rows[1]["graded_max"] == printed_graded == 477.9


def _printed_max(stdout: str, label: str) -> float:
    """Read the `max <x>` off the probe's own line for `label`."""
    line = next(l for l in stdout.splitlines() if l.strip().startswith(label))
    chunk = line.split("max", 1)[1].split("(")[0].strip()
    return float(chunk)


def test_a_published_constant_below_its_arm_alarms_both_arms_both_numbers(ledger: Path):
    """Clause 2: the alarm names BOTH arms with BOTH figures.

    905.3 vs 1,160.2 and 401.3 vs 477.9 — the real 2026-10-20 publication against
    the real ledger. A per-arm-half message would let someone refresh one constant
    and leave its pair stamped to a different date.
    """
    m = P.summarize(ledger)
    problems = P.evaluate(m, {"fresh": 905.3, "graded": 401.3, "source": "x"}, 1600.0)
    text = " | ".join(problems)
    assert problems, "drift of 254.9 ms and 76.6 ms must not read green"
    for figure in ("905.3", "1160.2", "401.3", "477.9"):
        assert figure in text, f"{figure} missing from the alarm"
    assert "fresh" in text and "graded" in text, "both arms must be named"
    assert any("below the ledger" in p for p in problems)


def test_a_25_percent_margin_would_have_read_green_and_this_rule_does_not(ledger: Path):
    """The refused trigger, pinned. #3304 asked for '>25 % below'; at publication
    the drift was 21.97 % (fresh) and 16.02 % (graded), so that rule passes a set
    that is 254.93 ms stale on the fresh arm and 76.577 ms on the graded arm. This test fails the day someone reintroduces a margin.
    """
    m = P.summarize(ledger)
    drifted = P.evaluate(m, {"fresh": 905.3, "graded": 401.3, "source": "x"}, 1600.0)
    assert drifted, "a margin rule is back: 905.3 is 21.97 % under 1160.2"
    # And equality is NOT drift: a constant sitting exactly on its max is a
    # current publication, which is the state the refresh produces.
    green = P.evaluate(m, {"fresh": 1160.2, "graded": 477.9, "source": "x"}, 1600.0)
    assert green == [], f"a set equal to its maxima must pass, got {green}"
    # The finest drift this rule can see is ONE STEP OF THE PUBLISHED GRID, 0.1 ms:
    # the comparison is `published < round(max, 1)`, the same one decimal the module's
    # command prints and the same one decimal every figure in this file is published
    # at. Below the grid the rule is blind by construction — a record of 1,160.24
    # against a published 1,160.2 reads GREEN, because at that precision the two are
    # the same published figure. That is stated rather than fixed because fixing it
    # would make the alarm fire on figures nobody publishes: the alarm text carries
    # `exact` (1,160.23 here) so a reader can see the sub-grid truth when it matters.
    barely = P.evaluate(m, {"fresh": 1160.1, "graded": 477.9, "source": "x"}, 1600.0)
    assert barely, "strict inequality is the whole point of the margin-free rule"
    below_grid = P.evaluate(m, {"fresh": 1160.2, "graded": 477.9,
                               "source": "x"}, 1600.0)
    assert below_grid == [], (
        "the rule changed resolution under its own published figures: at one decimal "
        "a constant equal to its max is current, and this is the documented floor")
    assert m["fresh_max_exact"] == 1160.23 and m["fresh_max"] == 1160.2, (
        "the exact figure must stay carried beside the published one, or the sub-grid "
        "blind spot becomes invisible rather than documented")


def test_a_recorded_run_over_the_bound_alarms(ledger: Path, tmp_path: Path,
                                             monkeypatch, capsys):
    """Threshold 1: `over_bound_count > 0`, named with the bound it breached."""
    state = tmp_path / "over"
    state.mkdir()
    (state / "promotions.jsonl").write_text(
        _row("2026-10-24T03:00:00Z", 620.1, 1700.0) + "\n", encoding="utf-8")
    m = P.summarize(state)
    assert m["over_bound_count"] == 1
    problems = P.evaluate(m, {"fresh": 4408.0, "graded": 1700.0, "source": "x"}, 1600.0)
    assert any("over_bound_count=1" in p and "1600" in p for p in problems), problems
    monkeypatch.setattr("app.data_root.production_data_root",
                        lambda: tmp_path / "data-root")
    assert P.main(["--state-dir", str(state)]) == 1


def test_main_exits_nonzero_on_drift_and_zero_when_refreshed(ledger: Path,
                                                            tmp_path: Path,
                                                            monkeypatch, capsys):
    """The whole command's exit code, both ways, through `main`."""
    monkeypatch.setattr("app.data_root.production_data_root",
                        lambda: tmp_path / "data-root")
    monkeypatch.setattr(P, "published_constants",
                        lambda *a, **k: {"fresh": 905.3, "graded": 401.3,
                                         "source": "pinned by test"})
    assert P.main(["--state-dir", str(ledger)]) == 1
    monkeypatch.setattr(P, "published_constants",
                        lambda *a, **k: {"fresh": 1160.2, "graded": 477.9,
                                         "source": "pinned by test"})
    assert P.main(["--state-dir", str(ledger)]) == 0
    assert len((tmp_path / "data-root/_pipeline/automod"
                / "published-maxima.jsonl").read_text().splitlines()) == 2, \
        "a green run appends a row too — the journal is a series, not an alarm log"


def test_constants_that_cannot_be_read_are_an_alarm_not_a_green(tmp_path: Path):
    """No fallback literal: a probe that invents its expected value stops
    measuring. The first draft fell back to 1160.2 / 477.9 — figures that
    happened to equal the truth — and printed a green drift check it had not run.
    """
    bogus = tmp_path / "not-the-test-file.py"
    bogus.write_text("import os\nprint(os)\n", encoding="utf-8")
    got = P.published_constants(bogus)
    assert got["fresh"] is None and got["graded"] is None
    assert "UNREADABLE" in got["source"]
    m = P.summarize(tmp_path)
    problems = P.evaluate(m, got, 1600.0)
    assert any("cannot read the published constant" in p for p in problems), problems


def test_the_real_test_file_still_resolves(ledger: Path):
    """The parse must work on the file it actually reads, and the keys are
    `R.CONTEXT_PAIRED_CHECK` — attribute nodes, not string literals, which is
    what broke the first draft's `literal_eval`."""
    got = P.published_constants()
    assert got["fresh"] is not None, f"cannot read the published pair: {got['source']}"
    assert got["graded"] is not None
    assert got["fresh"] == 1160.2 and got["graded"] == 477.9, \
        "the probe must read the same figures the module publishes beside them"


def test_the_probe_is_off_the_promotion_path():
    """Clause 5's other half, pinned here as well as in the regression suite:
    nothing under `scripts/automod/` may import the probe or read its exit code.
    """
    hits = subprocess.run(
        ["git", "grep", "-rln", "published_maxima_probe", "--", "scripts/automod/"],
        cwd=ROOT, capture_output=True, text=True)
    assert hits.stdout.strip() == "", \
        f"the probe leaked onto the promotion path: {hits.stdout}"
    assert hits.returncode in (0, 1), hits.stderr


def test_the_journal_path_is_the_declared_home():
    """The item names the data root, not the automod state dir (the sibling
    recorder `gate_to_land_lag.py` writes beside the ledger instead). A move is a
    decision someone has to make out loud, so the path is pinned."""
    home = P.journal_path(Path("/root"))
    assert home == Path("/root/_pipeline/automod/published-maxima.jsonl")
