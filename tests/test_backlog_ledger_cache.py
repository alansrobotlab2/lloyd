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
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.routers import dashboard as dash
from scripts.automod import backlog as B
from scripts.automod import scorecard as SC
from scripts.automod import state as S

# Triage baseline, 2026-09-16, one cold cycle over the live board:
# 55 read_text calls / 318,670 json.loads over 6,286,192 bytes / 3.79 s.
BASELINE_CALLS = 55
BASELINE_LOADS = 318_670
BASELINE_BYTES = 6_286_192


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


def _count_ledger_reads(monkeypatch, ledger: Path):
    """Patch `Path.read_text` to count calls whose target *is* `ledger`.

    Identity, not substring: a counter that also matched sibling files would
    rise for reasons the clause does not cover, and a number nobody can
    re-measure in the same breath is not a check.
    """
    calls: list[int] = []
    loads: list[int] = []
    original = Path.read_text

    def counting_read(self, *args, **kwargs):
        if Path(self) == ledger:
            calls.append(len(calls))
        return original(self, *args, **kwargs)

    original_loads = json.loads

    def counting_loads(*args, **kwargs):
        loads.append(1)
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)
    monkeypatch.setattr(json, "loads", counting_loads)
    return calls, loads


def test_one_dashboard_cycle_reads_the_ledger_once(tmp_path, monkeypatch, ledger):
    """`_backlog()` + `_automod()` cold, on the live board, one ledger file.

    This is the section pair the item timed at 10.75 s and 13.07 s
    respectively; every other section is sub-100 ms warm and irrelevant here.
    """
    board = tmp_path / "obsidian" / "backlog"
    board.mkdir(parents=True)
    real = sorted((Path.home() / "obsidian" / "backlog").glob("*.md"))
    for src in real[:150]:
        (board / src.name).write_text(src.read_text(encoding="utf-8", errors="replace"),
                                      encoding="utf-8")
    assert len(list(board.glob("*.md"))) > 0, "fixture board is empty"

    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(S, "LEDGER_PATH", ledger)
    calls, loads = _count_ledger_reads(monkeypatch, ledger)

    dash._cache.clear()
    B._ledger_cache_clear()
    before = (len(calls), len(loads))
    dash._backlog()
    dash._automod()
    reads, decoded = len(calls) - before[0], len(loads) - before[1]

    print(f"ledger reads={reads} json.loads={decoded} ledger_bytes={ledger.stat().st_size}")
    assert reads <= 1, (
        f"one cold dashboard cycle decoded the ledger {reads} times; the "
        f"baseline at triage was {BASELINE_CALLS} read_text calls "
        f"({BASELINE_BYTES} bytes each, {BASELINE_LOADS} json.loads, 3.79 s "
        f"cumulative) and the fix is one shared row cache keyed on "
        f"(mtime_ns, size, inode) in scripts/automod/state.py. "
        f"`_ledger_events` and `scorecard._events` must both go through it."
    )
    assert decoded <= 2 * _ledger_line_count(ledger), (
        f"the cycle ran {decoded} json.loads against a ledger of "
        f"{_ledger_line_count(ledger)} lines: decoding the file once per cycle "
        f"means at most one json.loads per line. Triage baseline: "
        f"{BASELINE_LOADS}."
    )


def _ledger_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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
