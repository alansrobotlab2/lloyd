"""#1204: one cold dashboard cycle decodes the automod ledger once, not 55 times.

`board_health` calls a dozen ledger-reading helpers (`triaged_ids`,
`released_ids`, `held_confirmations`, `implement_outcomes`, ...) and each one
went to `promotions.jsonl` with `read_text` plus a `json.loads` per line. At
triage that measured out to **55 `read_text` calls and 318,670 `json.loads`
against a 6,286,192-byte file, 3.79 s cumulative, inside one dashboard
request** — the single largest residue left after the loader swap, and the
cost the item's own section 4 said not to trim because it is not the ledger's
fault: it is re-decoding, not size.

The counter here is on `pathlib.Path.read_text` filtered to the ledger path,
which is what the shared row cache in `scripts/automod/state.py` actually
removes. Counting `json.loads` too, because a cache that read once but decoded
per line would still cost the 3.8 s.

**#2032: the decode count is attributed to the ledger, not global.** The
`json.loads` counter used to count every decode in the process for the length of
the cycle while the bound it fed was a *ledger-line* budget — and a dashboard
cycle decodes plenty of JSON that is not the ledger. `scorecard` row 16 reads
`<data root>/safety/denials.jsonl` and row 17 reads
`<data root>/supply-chain/provenance.jsonl`, both resolved through
`resolve_data_root_for_tree(LIVE_ROOT)`, so they follow `LLOYD_DATA` — which is
exactly what a gate run points at its own round data root (`gate._child_env`,
`scripts/automod/gate.py`). A full tests rung fills that root, and the node then
failed on its own host's bookkeeping: `SM_20261001_162133` (#2027) reported
`json.loads=4947` and `SM_20261001_170632` (#2029) `4772`, against a 4,000-line
budget, while `reads` stayed at its correct **1**. So the counter now tags each
decode with whether the string being decoded is one of the ledger's own lines,
the ceiling reads only the ledger-tagged number, and the untagged remainder is
reported beside it as its own figure. Attributing by content rather than by
call-site means a reader that stops passing ledger lines verbatim is caught by
the floor below rather than silently scoring zero.

The last two tests cross the two process boundaries this cache actually sits
on, which no in-process fixture can reach: `get_dashboard` asks for these rows
from `asyncio.to_thread` workers — different threads, same file, same instant —
and the rows themselves are appended by *other programs* (the gate, the
guardian, the MCP layer), so invalidation has to be a stat of a shared file and
not a process-local flag.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.routers import dashboard as dash
from scripts.automod import backlog as B
from scripts.automod import scorecard as SC
from scripts.automod import state as S

# Triage baseline, re-measured 2026-09-17, one cold cycle over the live board:
# 57 read_text calls / 331,569 json.loads / 3.79 s cumulative over a
# 6,462,191-byte ledger. Re-measure it with the counter in
# `_count_ledger_reads` wrapped around a real `dash._backlog(); dash._automod()`
# pair — the numbers move as the ledger grows (6,635,836 bytes on 2026-09-17
# 09:40Z), so they are a census, not a constant, and they appear in the failure
# messages only: no threshold here reads them.
BASELINE_CALLS = 57
BASELINE_LOADS = 331_569
BASELINE_BYTES = 6_462_191


@pytest.fixture(autouse=True)
def _clear_caches():
    """The section cache is module-global and the row cache is process-global;
    both would otherwise leak between tests and between fixture trees."""
    dash._cache.clear()
    B._ledger_cache_clear()
    yield
    dash._cache.clear()
    B._ledger_cache_clear()


@pytest.fixture
def ledger(tmp_path):
    """A real-shaped ledger, big enough that a re-decode is not free.

    2,000 rows across the six event types `board_health` and `scorecard.compute`
    actually ask for, so the helpers return real rows rather than passing on
    emptiness.
    """
    path = tmp_path / "promotions.jsonl"
    kinds = ["backlog_triage", "backlog_implement", "gate", "review", "promoted",
             "settled", "backlog_sweep", "vault_land"]
    lines = []
    for n in range(2000):
        lines.append(json.dumps({
            "ts": time.time() - n * 60,
            "event": kinds[n % len(kinds)],
            "item_id": 1000 + (n % 300),
            "round_id": f"SM_FAKE_{n}",
            "commit": f"{n:040x}",
            "phase": "finished",
            "verdict": "confirmed",
            "outcome": {"acceptance": "met", "clause_outcomes": [{"clause": 1, "outcome": "met"}]},
            "clauses": [{"clause": 1, "verdict": "met"}],
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ledger_lines(path: Path) -> frozenset[str]:
    """Every non-blank line of `path`, as a set for O(1) decode attribution.

    The strings are what `_decode_ledger` (`scripts/automod/state.py`) hands to
    `json.loads` — the raw `splitlines()` entries, stripped of nothing — so
    matching them by value is what identifies a ledger decode. A copy that only
    differs in surrounding whitespace still matches, which is fine: it is the
    same string as far as `json.loads` is concerned. What does NOT match is a
    re-serialised row (`json.dumps(json.loads(line))` rewrites key order and
    spacing) or a slice — and the real readers pass ledger lines through
    verbatim today, so attribution holds. It is also why
    `_assert_ledger_decode_bounds` carries a floor: attribution that stops
    matching for any reason drives the ledger count to zero, which the floor
    reports as the instrument's own failure instead of a quiet pass.
    """
    return frozenset(line for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip())


def _count_ledger_reads(monkeypatch, ledger: Path):
    """Count one cycle's ledger reads and attribute every JSON decode in it.

    Returns `(reads, all_decodes, ledger_decodes)`. `Path.read_text` is counted
    by path *identity*, not substring: a counter that also matched sibling files
    would rise for reasons the clause does not cover, and a number nobody can
    re-measure in the same breath is not a check.

    `json.loads` cannot be scoped that way — it is handed a string, not a path —
    so it is attributed by content instead: a decode counts against the ledger
    when the text being decoded is one of the ledger's own lines. Everything
    else is `other`, which is measured and reported but never charged to the
    ledger's budget (#2032, see the module docstring for the two gate runs this
    is a post-mortem of).
    """
    calls: list[int] = []
    loads: list[int] = []
    ledger_loads: list[int] = []
    original = Path.read_text
    mine = _ledger_lines(ledger)

    def counting_read(self, *args, **kwargs):
        if Path(self) == ledger:
            calls.append(len(calls))
        return original(self, *args, **kwargs)

    original_loads = json.loads

    def counting_loads(*args, **kwargs):
        loads.append(1)
        subject = args[0] if args else kwargs.get("s")
        if isinstance(subject, str) and subject in mine:
            ledger_loads.append(1)
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)
    monkeypatch.setattr(json, "loads", counting_loads)
    return calls, loads, ledger_loads


def _fixture_board(tmp_path: Path) -> Path:
    """A board of 150 real item files, so `_backlog()` has work to do."""
    board = tmp_path / "obsidian" / "backlog"
    board.mkdir(parents=True, exist_ok=True)
    real = sorted((Path.home() / "obsidian" / "backlog").glob("*.md"))
    for src in real[:150]:
        (board / src.name).write_text(src.read_text(encoding="utf-8", errors="replace"),
                                      encoding="utf-8")
    assert len(list(board.glob("*.md"))) > 0, "fixture board is empty"
    return board


def _gate_shaped_data_root(tmp_path: Path, monkeypatch, *, denials: int = 1200,
                           provenance: int = 1300) -> Path:
    """Point `LLOYD_DATA` at a root holding #2032's pollution: a denial journal
    and a provenance journal as full as a gate run's own round data root gets.

    The rows are the writers' shapes (`at` inside the scorecard's 7-day window),
    so `scorecard._denials`/`_provenance` decode every one of them rather than
    discarding the file wholesale. `LLOYD_DATA` is what makes it the real thing:
    `_denial_journal_default` and `_provenance_journal_default` resolve through
    `resolve_data_root_for_tree`, which honours that env var first, and the cycle
    is asserted to read *these* files — no `denials=`/`provenance=` argument is
    passed to `scorecard.compute`, because reproducing the gate's shape is the
    point.
    """
    root = tmp_path / "data-root"
    (root / "safety").mkdir(parents=True)
    (root / "supply-chain").mkdir(parents=True)
    now = time.time()
    with (root / "safety" / "denials.jsonl").open("w", encoding="utf-8") as fh:
        for n in range(denials):
            fh.write(json.dumps({
                "at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                    time.gmtime(now - n * 60)),
                "guard": "path_guard", "label": "protected path",
                "session_class": "autocode", "where": "hook"}) + "\n")
    with (root / "supply-chain" / "provenance.jsonl").open("w", encoding="utf-8") as fh:
        for n in range(provenance):
            fh.write(json.dumps({
                "at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                    time.gmtime(now - n * 60)),
                "session": f"s{n}", "command": f"pip install p{n}",
                "names": [{"name": f"pkg{n}", "outcome": "cleared",
                           "count": 1}]}) + "\n")
    monkeypatch.setenv("LLOYD_DATA", str(root))
    return root


def _cold_cycle_census(monkeypatch, ledger: Path, tmp_path: Path,
                       *, counter_for: Path | None = None) -> dict:
    """One cold `_backlog()` + `_automod()` cycle, and what it decoded.

    `counter_for` exists for one purpose: to attribute the cycle's decodes to a
    file *other* than the ledger the cycle reads, which is how the floor below is
    shown to bite. It defaults to `ledger`, i.e. the honest instrument.
    """
    board = _fixture_board(tmp_path)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(S, "LEDGER_PATH", ledger)
    calls, loads, ledger_loads = _count_ledger_reads(monkeypatch,
                                                     counter_for or ledger)

    dash._cache.clear()
    B._ledger_cache_clear()
    before = (len(calls), len(loads), len(ledger_loads))
    board_section = dash._backlog()
    automod_section = dash._automod()
    census = {
        "reads": len(calls) - before[0],
        "ledger_decodes": len(ledger_loads) - before[2],
        # The census clause (#2032): the two sources are two numbers, never a
        # sum, so the next failure names which one moved.
        "other_json_decodes": ((len(loads) - before[1])
                               - (len(ledger_loads) - before[2])),
        "total_json_decodes": len(loads) - before[1],
        "ledger_lines": _ledger_line_count(ledger),
        "board": board,
        "board_section": board_section,
        "automod_section": automod_section,
    }
    # What the gate's captured stdout will show next time this node is red. The
    # old line read `ledger reads=1 json.loads=4772`, and 4,772 was 2,000 ledger
    # lines plus 2,772 decodes of somebody else's journal.
    print(f"ledger reads={census['reads']} "
          f"ledger json.loads={census['ledger_decodes']} "
          f"other json.loads={census['other_json_decodes']} "
          f"total json.loads={census['total_json_decodes']} "
          f"ledger_lines={census['ledger_lines']} "
          f"ledger_bytes={ledger.stat().st_size}")
    return census


def _assert_ledger_decode_bounds(census: dict, *, where: str = "") -> None:
    """The floor and the ceiling on one cold cycle's ledger decodes.

    Floor at `lines`, ceiling at `lines`: exactly one `json.loads` per ledger
    line per cycle. The ceiling used to be `2 * lines` over a count of every
    decode in the process; that multiplier was slack for the foreign decodes that
    are now attributed elsewhere, so dropping it to 1 makes the ledger bound
    *tighter* than the old one and still passable — two forced re-reads decode
    4,000 of these 2,000 lines and fail it (see
    `test_a_second_ledger_decode_still_breaks_the_decode_ceiling`). The two
    bounds fail for different reasons and stay separate: the ceiling is the cache
    regressing, the floor is the attribution matching nothing.
    """
    lines = census["ledger_lines"]
    got = census["ledger_decodes"]
    # The two ways the floor trips say different things, so say which one
    # happened: with no ledger read attributed either, the counter was aimed at
    # the wrong file end to end, which is not a tagging bug.
    where_read = (
        "the cycle was not attributed a single ledger read either, so the "
        "counter is pointed at a file the cycle never opens"
        if census["reads"] == 0 else
        f"the cycle was attributed {census['reads']} ledger read(s), so the "
        "reads were there and the tags were not"
    )
    assert got >= lines, (
        f"{where}the cycle attributed {got} json.loads to a ledger of {lines} "
        "lines, so the ledger was never decoded into the counter at all — "
        f"{where_read}. A ceiling with no floor is the vacuous-denominator "
        "defect this file is about: without this assertion "
        "`ledger_decodes: 0` would satisfy any ceiling."
    )
    assert got <= lines, (
        f"{where}the cycle decoded the ledger's own lines {got} times against a "
        f"ledger of {lines} lines, i.e. more than one `json.loads` per line: the "
        f"shared row cache in scripts/automod/state.py is not serving every "
        f"reader. For the record the cycle also ran "
        f"{census['other_json_decodes']} json.loads on non-ledger JSON, over "
        f"{census['reads']} ledger read(s) — the number that used to be summed "
        f"into this bound and red the node from across the room. Triage "
        f"baseline for the whole cycle: {BASELINE_LOADS}."
    )


def test_one_dashboard_cycle_reads_the_ledger_once(tmp_path, monkeypatch, ledger):
    """`_backlog()` + `_automod()` cold, on the live board, one ledger file.

    This is the section pair the item timed at 10.75 s and 13.07 s
    respectively; every other section is sub-100 ms warm and irrelevant here.

    Two ceilings, on two different counts: `reads <= 1` is the read side, and
    the decode ceiling in `_assert_ledger_decode_bounds` is now charged only to
    the decodes attributed to the ledger — a cycle that also decodes 2,500 rows
    of the data root's denial and provenance journals still passes it, which is
    what red this node on two gate runs (#2032).
    """
    census = _cold_cycle_census(monkeypatch, ledger, tmp_path)
    board = census["board"]
    reads = census["reads"]
    board_section = census["board_section"]
    automod_section = census["automod_section"]

    # ── positive control, ahead of the ceilings ────────────────────────────
    # `reads <= 1` is satisfied by `reads == 0`, and `reads == 0` is also the
    # answer for a cycle that never reached the ledger at all: the repoint not
    # landing, the counter matching a different path, a fixture board that came
    # up empty. A ceiling with no floor is the vacuous-denominator defect this
    # item is filed about, so the same test first proves the thing it is
    # counting happened. Measured on this fixture: 1 read, 2,000 rows decoded,
    # 150 board files with 93 open, `events` 2,000.
    assert reads >= 1, (
        "the cycle decoded the ledger 0 times, so `reads <= 1` below is "
        "measuring nothing: either the section functions are not reading "
        f"{ledger} (check the `S.LEDGER_PATH` repoint and the counter's path "
        "identity in `_count_ledger_reads`) or both sections short-circuited "
        "before reaching it. Triage baseline for a real cycle: "
        f"{BASELINE_CALLS} reads."
    )
    assert board_section.get("total", 0) > 0, (
        f"`_backlog()` parsed {board_section.get('total', 0)} board files, so "
        "the board half of the cycle was inert and its ledger reads could "
        "legitimately be zero. The fixture copies 150 real item files; "
        f"`{board}` holding them is asserted above, so an empty count here is "
        "the reader failing, not the board being empty."
    )
    assert automod_section.get("events", 0) > 0, (
        f"`_automod()` reported {automod_section.get('events', 0)} ledger "
        "events for a fixture whose newest row is stamped ~now, so its "
        "ledger half was inert and `reads <= 1` would be an untested ceiling. "
        "`scorecard.compute` drops rows older than `since_days` (7.0) — "
        "fixture rows are stamped `time.time() - n * 60` and must stay inside "
        "that window for this clause to measure a decoding cycle."
    )

    assert reads <= 1, (
        f"one cold dashboard cycle decoded the ledger {reads} times; the "
        f"baseline at triage was {BASELINE_CALLS} read_text calls "
        f"({BASELINE_BYTES} bytes each, {BASELINE_LOADS} json.loads, 3.79 s "
        f"cumulative) and the fix is one shared row cache keyed on "
        f"(mtime_ns, size, inode) in scripts/automod/state.py. "
        f"`_ledger_events` and `scorecard._events` must both go through it."
    )
    # #2032: this bound is on the decodes attributed to the ledger, with a floor
    # at the same line count — see `_assert_ledger_decode_bounds` for why the
    # multiplier went from 2 to 1 rather than up.
    _assert_ledger_decode_bounds(census)


def _ledger_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ── #2032: the ceiling measures the ledger, not the whole process ──────────

def test_a_gate_shaped_data_root_does_not_count_against_the_ledger_ceiling(
        tmp_path, monkeypatch, ledger):
    """A cycle over a data root as full as a gate run's own passes (#2032 clause 1).

    The state that red this node twice was not in the ledger at all: a round's
    data root holds a denial journal and a provenance journal that
    `scorecard.compute` reads as rows 16 and 17 of the same `_automod()` call,
    and every one of those rows was charged to the ledger's line budget. Here
    that is 1,200 + 1,300 = 2,500 rows — the item's floor is >=1,000 each,
    >=2,000 combined, and a real round root measured 1,383 + 172 on the morning
    this was filed.

    The positive control is the interesting half: `other_json_decodes` must be
    the >=2,500 rows the fixture wrote, and the *unattributed* total must still
    breach the old `2 * lines` budget. If either stops being true the data root
    is not being read and this test has quietly become a no-op — which is the
    exact shape of the failure it is a post-mortem of.
    """
    root = _gate_shaped_data_root(tmp_path, monkeypatch, denials=1200, provenance=1300)
    # The two journals the cycle is about to read are THESE files, resolved the
    # way production resolves them (`LLOYD_DATA` through `resolve_data_root_for_tree`)
    # rather than handed to `compute` as an argument.
    assert SC._denial_journal_default() == root / "safety" / "denials.jsonl"
    assert SC._provenance_journal_default() == (root / "supply-chain"
                                               / "provenance.jsonl")

    census = _cold_cycle_census(monkeypatch, ledger, tmp_path)

    assert census["other_json_decodes"] >= 2_500, (
        f"only {census['other_json_decodes']} non-ledger JSON decodes for a "
        "data root holding 2,500 rows across the two journals, so the cycle "
        "never read the pollution this test exists to survive — the ceiling "
        "passing would then prove nothing. Check that "
        "`resolve_data_root_for_tree(LIVE_ROOT)` is still honouring "
        "`LLOYD_DATA` (`scripts/automod/scorecard.py`, rows 16 and 17)."
    )
    assert census["reads"] == 1, "the polluted cycle stopped reading the ledger once"
    assert census["ledger_decodes"] == census["ledger_lines"] > 0, (
        f"the cycle attributed {census['ledger_decodes']} decodes to a ledger of "
        f"{census['ledger_lines']} lines; the pollution must not move the "
        f"ledger-attributed count, which is the whole point of attributing"
    )
    # The old bound on the same cycle, kept as a witness: it is the number that
    # made a healthy cache look broken.
    assert census["total_json_decodes"] > 2 * census["ledger_lines"], (
        f"the unattributed total is {census['total_json_decodes']} against the "
        f"old `2 * lines` = {2 * census['ledger_lines']} budget, so this cycle "
        f"would NOT have reproduced the gate failure — the pollution fixture is "
        f"too small to be a control"
    )
    _assert_ledger_decode_bounds(census)


def test_a_second_ledger_decode_still_breaks_the_decode_ceiling(tmp_path, monkeypatch,
                                                                ledger):
    """#2032 clause 2: attribution must not become a licence to re-decode.

    The relaxation in `test_a_gate_shaped_data_root_...` is only honest if the
    bound it relaxed still bites the thing it was for. One eviction of the shared
    row cache mid-cycle — two `state.ledger_rows` calls that really come off disk
    — is the regression #1204 was filed about, and 4,000 `json.loads` over a
    2,000-line ledger must fail the ceiling rather than sit inside the old
    `2 * lines` slack exactly.
    """
    real_rows = S.ledger_rows
    evicted: list[int] = []

    def evict_once(path: Path | None = None, *args, **kwargs):
        target = Path(path) if path is not None else Path(S.LEDGER_PATH)
        rows = real_rows(target, *args, **kwargs)
        if target == ledger and not evicted:
            evicted.append(1)
            S.ledger_rows_reset()          # the next reader goes to disk again
        return rows

    monkeypatch.setattr(S, "ledger_rows", evict_once)
    census = _cold_cycle_census(monkeypatch, ledger, tmp_path)

    assert census["reads"] >= 2, (
        f"forcing one cache eviction mid-cycle produced {census['reads']} ledger "
        "reads, so the eviction never landed and the bound below would be "
        "asserted against a healthy cycle"
    )
    assert census["ledger_decodes"] >= 2 * census["ledger_lines"], (
        f"{census['ledger_decodes']} attributed decodes for "
        f"{census['reads']} ledger reads of a {census['ledger_lines']}-line file: "
        "each read decodes every line, so the counter is not seeing both decodes"
    )
    with pytest.raises(AssertionError) as exc:
        _assert_ledger_decode_bounds(census)
    assert "more than one `json.loads` per line" in str(exc.value), str(exc.value)
    # Clause 4's other half: the failure names the source that moved, in case.
    assert "non-ledger JSON" in str(exc.value)


def test_the_decode_floor_bites_when_attribution_matches_nothing(tmp_path, monkeypatch,
                                                                 ledger):
    """#2032 clause 3: the ledger count carries a floor, so 0 is not a pass.

    The counter attributes a decode by matching the string against the ledger's
    own lines. A reader that hands `json.loads` something derived — a slice, a
    normalised copy, a re-serialised row — or a repoint that lands on a different
    path than the one the cycle reads, and the attributed count falls to zero
    while the cycle is still decoding all 2,000 lines. A ceiling alone then
    reports a perfect cycle: `0 <= 2,000`. This is the instrument's own negative
    control, run by pointing the counter at a file the cycle never opens.
    """
    decoy = tmp_path / "some-other-ledger.jsonl"
    decoy.write_text(json.dumps({"event": "backlog_triage", "item_id": 1}) + "\n",
                     encoding="utf-8")

    census = _cold_cycle_census(monkeypatch, ledger, tmp_path, counter_for=decoy)

    # The mis-attribution really happened, and the work really was done.
    assert census["reads"] == 0 and census["ledger_decodes"] == 0
    assert census["total_json_decodes"] >= 2_000, (
        f"the cycle decoded {census['total_json_decodes']} JSON values in all, "
        "so the ledger was not decoded either and this is not the failure mode "
        "the floor exists to catch"
    )
    with pytest.raises(AssertionError) as exc:
        _assert_ledger_decode_bounds(census)
    assert "never decoded into the counter" in str(exc.value), str(exc.value)


def test_the_census_reports_ledger_and_non_ledger_decodes_separately(
        tmp_path, monkeypatch, ledger, capsys):
    """#2032 clause 4: one number per source, so the next failure is diagnosable.

    A single blended `json.loads` is what made the two gate failures unreadable:
    `4,947` and `4,772` said nothing about whether the cache regressed or the
    data root merely filled up. Both figures are printed by the census and both
    are in the dict, and they must be *different* numbers here — a census that
    reported the same value twice would satisfy any assertion on either.
    """
    _gate_shaped_data_root(tmp_path, monkeypatch, denials=1200, provenance=1300)
    census = _cold_cycle_census(monkeypatch, ledger, tmp_path)
    printed = capsys.readouterr().out

    assert census["ledger_decodes"] == census["ledger_lines"] > 0
    assert census["other_json_decodes"] >= 2_500
    assert census["ledger_decodes"] != census["other_json_decodes"]
    assert (f"ledger json.loads={census['ledger_decodes']}" in printed
            and f"other json.loads={census['other_json_decodes']}" in printed), printed
    assert f"ledger reads={census['reads']}" in printed

    # Parse the figures back off the line and require all four of them to be
    # there. Nothing arithmetic can be asserted across them: `other_json_decodes`
    # is *computed* as total minus ledger and the print formats those same dict
    # values, so any sum or printed-versus-dict comparison is true by
    # construction. The failure mode that is real is the line losing a figure —
    # which is how both gate failures went unread, at `json.loads=4772` and
    # nothing else.
    shown = re.findall(r"(ledger reads|ledger json\.loads|other json\.loads|"
                       r"total json\.loads)=(\d+)", printed)
    assert len(shown) == 4, f"census line is missing figures: {printed!r}"


def test_a_second_call_in_the_same_cycle_does_not_re_read_the_ledger(ledger):
    """The cache must be shared across helper functions, not per-helper.

    `board_health` reaches the ledger through eight different helpers; a cache
    that each one kept for itself would still read eight times. The assertion is
    on two calls to *different* helpers, which is the shape that broke.
    """
    B._ledger_cache_clear()
    B._ledger_events(ledger, "backlog_triage")
    B.triaged_ids(ledger)
    B.implement_outcomes(ledger)
    B.held_confirmations(ledger)
    reads = B._ledger_read_count(ledger)
    assert reads == 1, (
        f"four different ledger helpers caused {reads} ledger reads; the row "
        f"cache has to sit under `_ledger_events` itself, where all of them "
        f"land, not inside one helper"
    )


def test_appending_a_line_invalidates_the_cache(ledger):
    """An append grows the file, and a cycle must then see the new row.

    This is the failure mode a size-keyed cache would hide: a reader that
    served a stale row set forever would make `board_health` report a board
    that already moved, and the dashboard would look live while lying.
    """
    before = B._ledger_events(ledger, "backlog_sweep", require_item=False)
    assert B._ledger_events(ledger, "backlog_sweep", require_item=False) == before
    S.append_event({"event": "backlog_sweep", "note": "fresh"}, path=ledger)
    after = B._ledger_events(ledger, "backlog_sweep", require_item=False)
    assert len(after) == len(before) + 1, (
        f"an appended row was not visible to the next `_ledger_events` call "
        f"({len(before)} -> {len(after)}): the cache key has to change when the "
        f"ledger grows, or every panel reads a stale board"
    )


def test_scorecard_events_share_the_same_single_read(ledger):
    """`scorecard._events` must not keep its own second reader of the ledger.

    It is a separate module with its own `_events`, and its file header says
    "stdlib + yaml, no app imports" — which is why the cache lives in
    `scripts/automod/state.py` (stdlib-only, and the module that owns
    `LEDGER_PATH`) rather than in `backlog.py`, which imports `app.*`.
    """
    B._ledger_cache_clear()
    rows = SC._events(ledger)
    assert rows, "scorecard._events returned nothing for a populated ledger"
    assert B._ledger_read_count(ledger) == 1, (
        f"scorecard._events and backlog._ledger_events between them read the "
        f"ledger {B._ledger_read_count(ledger)} times — two readers, two "
        f"decodes, the 3.79 s back"
    )


# ── the boundaries this cache sits on ────────────────────────────────────

def test_concurrent_readers_cause_exactly_one_decode(ledger):
    """Threads arriving together decode once — the seam `get_dashboard` creates.

    `app/routers/dashboard.py`'s `get_dashboard` offloads `_backlog` and
    `_automod` through `_to_thread` (`asyncio.to_thread`), so two worker threads
    reach `state.ledger_rows` in the same instant and want the same rows. A
    check-then-decode-then-store is not atomic: the pre-fix shape read the
    cache, released the lock, and decoded outside it, so both threads found
    nothing cached and both spent the decode — and the clause this file pins is
    "at most once per cycle", not "usually once".

    The barrier releases all eight callers at once rather than letting them
    start at eight different moments, so the test asks about the race instead of
    about thread scheduling luck. The fixture is 2,000 rows, about 654 KB (its
    byte size is printed by the first test in this file), which decodes in
    single-digit milliseconds: long enough that a lock-free implementation has
    all eight threads inside the window together, short enough that the
    single-flight one costs one decode and a few blocked waits.
    """
    B._ledger_cache_clear()
    n_threads = 8
    gate = threading.Barrier(n_threads)
    results: list[int] = []
    results_lock = threading.Lock()

    def worker():
        gate.wait()
        rows = S.ledger_rows(ledger)
        with results_lock:
            results.append(len(rows))

    threads = [threading.Thread(target=worker, name=f"ledger-racer-{i}")
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "a ledger reader hung: single-flight deadlocked?"

    reads = B._ledger_read_count(ledger)
    print(f"{n_threads} concurrent readers -> {reads} ledger read(s), "
          f"{len(set(results))} distinct row-count(s)")
    assert len(results) == n_threads, "a racer died; the count below is not from all of them"
    assert set(results) == {_ledger_line_count(ledger)}, (
        f"the eight readers did not all get the full ledger: got counts "
        f"{sorted(set(results))} for a file of "
        f"{_ledger_line_count(ledger)} lines"
    )
    assert reads == 1, (
        f"{n_threads} threads arriving together caused {reads} ledger reads. "
        f"Baseline at triage was {BASELINE_CALLS} reads per cycle "
        f"({BASELINE_LOADS} json.loads, 3.79 s); one is the clause. The decode "
        f"has to happen while holding the lock that guards the cache "
        f"(`scripts/automod/state.py`), otherwise the two `asyncio.to_thread` "
        f"dashboard sections each decode the 6.3 MB file and the 3.8 s comes "
        f"back on every cold poll."
    )


def test_a_write_from_another_process_invalidates_the_cache(ledger):
    """A row appended by a different program must be visible to the next read.

    The writer of this file is not the reader: the gate, the guardian and the
    MCP layer all append to it from their own processes, while the backend that
    serves `/api/dashboard` holds the cache. So invalidation cannot depend on
    the writer cooperating with the cache — it has to be a stat of a file that
    somebody else grew. The in-process append test above cannot prove this,
    because there `append_event` and the cache share a module and could in
    principle conspire; a subprocess that never imports our code cannot.

    The child appends in the same shape `state.append_event` does — one JSON
    object per line, close, exit — and imports nothing from this repo, which is
    also why it needs no `PYTHONPATH`.
    """
    before = B._ledger_events(ledger, "backlog_sweep", require_item=False)
    assert B._ledger_read_count(ledger) == 1, "fixture did not start from one read"
    assert B._ledger_events(ledger, "backlog_sweep", require_item=False) == before
    assert B._ledger_read_count(ledger) == 1, "the repeat read: nothing was invalidated"

    child = (
        "import json, sys\n"
        "with open(sys.argv[1], 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps({'event': 'backlog_sweep', 'note':"
        " 'written by another process'}) + '\\n')\n"
    )
    proc = subprocess.run([sys.executable, "-c", child, str(ledger)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"child append failed: {proc.stderr[-500:]}"

    after = B._ledger_events(ledger, "backlog_sweep", require_item=False)
    reads = B._ledger_read_count(ledger)
    print(f"cross-process append: rows {len(before)} -> {len(after)}, reads {reads}")
    assert len(after) == len(before) + 1, (
        f"a row written by another process was invisible to the cache: "
        f"{len(before)} -> {len(after)} rows. The cache key is "
        f"(mtime_ns, size, inode) precisely so a foreign append — the gate, the "
        f"guardian, the MCP layer, all separate processes — is seen with no "
        f"cooperation from the writer; a board that already moved would "
        f"otherwise be reported as the live one."
    )
    assert reads == 2, (
        f"after a foreign append the cycle took {reads} reads (expected exactly "
        f"1 re-read, on top of the fixture's 1). Anything higher means the "
        f"invalidation is not the stat but a full cache eviction."
    )
