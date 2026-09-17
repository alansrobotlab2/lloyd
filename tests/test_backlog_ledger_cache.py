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

The last two tests cross the two process boundaries this cache actually sits
on, which no in-process fixture can reach: `get_dashboard` asks for these rows
from `asyncio.to_thread` workers — different threads, same file, same instant —
and the rows themselves are appended by *other programs* (the gate, the
guardian, the MCP layer), so invalidation has to be a stat of a shared file and
not a process-local flag.
"""

from __future__ import annotations

import json
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
