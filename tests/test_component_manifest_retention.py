"""The manifest store stops growing with age (#1604).

`~/.local/state/lloyd-request-manifests/` held 2.4 GB across 9 dated files on
2026-09-27 — 45 MB to 531 MB per day, still being written — and nothing in the tree
deleted any of it: no prune in the writer, no autonomy task, and zero mentions in
`scripts/groundskeeper/retention-sweep.py`, whose own docstring says a store the
sweep has never heard of is an unbounded one. The bytes were not the problem (`df`
had 2.8 T free); unbounded retention of `sha256:` digests of vault text, email
bodies and user messages was, since the store's own `POLICY.md` says to treat them
"with the same confidentiality as ~/obsidian".

Each clause is pinned at the store, by writing real files and reading the directory
back rather than asserting on a return value:

* **the window, applied** — files past it gone, files inside it byte-identical, and
  the thread that owns the store calls the sweep when it starts
  (`test_a_sweep_deletes_every_dated_file_past_the_window_and_touches_no_other`,
  `test_a_writer_thread_sweeps_as_its_first_act`);
* **the window is config** — `harness.component_manifest.retention_days`, default
  14, and `0` deletes nothing
  (`test_the_window_is_the_config_key_and_defaults_to_fourteen_days`,
  `test_a_window_of_zero_disables_pruning_and_deletes_nothing`);
* **only the writer's own files** — `POLICY.md`, a name that is not a date, a date
  with the wrong extension, an impossible date, and a dated file outside
  `manifests/` all survive
  (`test_pruning_touches_only_date_named_ndjson_inside_manifests`).

Before this diff every test here fails at import for the same reason:
`cm.prune_store`, `cm.retention_days` and `cm._sweep_if_new_day` do not exist on
`app/component_manifest.py`, which is the fact the item's premise check measured as
zero matches for `def .*(prune|retention|cleanup|purge|sweep)|unlink|MAX_AGE` in
that file.
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import date, timedelta
from pathlib import Path

import pytest

from app import component_manifest as cm

#: Day offsets are measured back from this fixed "today", so a test says "13 days
#: old" instead of naming a date that stops being 13 days old next month. `_today`
#: is pinned to it in the fixture: the cutoff arithmetic the clauses are about is
#: then the thing under test, not the date the suite happened to run on.
BASE = date(2026, 9, 28)


def _stamp(days_ago: int) -> str:
    return (BASE - timedelta(days=days_ago)).isoformat()


def _write_day(store: Path, days_ago: int) -> Path:
    """A dated day file with known bytes, so byte-identity is checkable."""
    path = store / "manifests" / f"{_stamp(days_ago)}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request_id": _stamp(days_ago)}) + "\n",
                    encoding="utf-8")
    return path


def _set_config_window(monkeypatch, value):
    """Put one value at the real config key the clause names."""
    import app.config as cfg
    section = dict((cfg.CONFIG.get("harness") or {}).get("component_manifest") or {})
    if value is _ABSENT:
        section.pop("retention_days", None)
    else:
        section["retention_days"] = value
    harness = {**(cfg.CONFIG.get("harness") or {}), "component_manifest": section}
    monkeypatch.setattr(cfg, "CONFIG", {**cfg.CONFIG, "harness": harness})


class _Absent:
    pass


_ABSENT = _Absent()


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """A store of our own, the clock held still, and clean counters."""
    monkeypatch.setenv("LLOYD_MANIFEST_STORE", str(tmp_path / "manifest-store"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(cm, "_today", lambda: _stamp(0))
    cm.reset_stats()
    yield tmp_path / "manifest-store"
    cm.flush(timeout=5.0)
    cm.reset_stats()


def _names(store: Path) -> "set[str]":
    return {p.name for p in (store / "manifests").iterdir()}


def test_a_sweep_deletes_every_dated_file_past_the_window_and_touches_no_other(_store):
    """Clause 1. 14 days is the default window, so day 13 is the oldest keepable
    file and day 14 the newest deletable one: the boundary the clause is about, not
    a comfortably-distant pair.

    Driven through `_sweep_if_new_day("")`, the function the writer thread calls at
    startup and on its first wake after midnight — forcing the marker empty is what
    makes a sweep happen now rather than at midnight.
    """
    keep = [_write_day(_store, 0), _write_day(_store, 1), _write_day(_store, 13)]
    gone = [_write_day(_store, 14), _write_day(_store, 20), _write_day(_store, 400)]
    before = {p: p.read_bytes() for p in keep}
    assert all(p.is_file() for p in gone), "the out-of-window files were not written"

    assert cm._sweep_if_new_day("") == _stamp(0)

    assert not any(p.exists() for p in gone), [p.name for p in gone if p.exists()]
    for path, payload in before.items():
        assert path.read_bytes() == payload, f"{path.name} was inside the window"
    assert _names(_store) == {f"{_stamp(d)}.ndjson" for d in (0, 1, 13)}
    assert cm.stats()["pruned_files"] == 3
    assert cm.stats()["prune_errors"] == 0


def test_a_writer_thread_sweeps_as_its_first_act(_store, monkeypatch):
    """Clause 1's trigger: the clause says "when the writer starts", so the sweep
    has to be something the thread that owns the store actually does, not a
    function nothing calls.

    `_writer_started` is a process-lifetime flag, so resetting it would leave two
    threads draining one queue for the rest of the suite. Instead the thread body
    is run once on a named thread of this test's and the sweep is awaited on an
    event; the queue sentinel then lets it return instead of parking on the idle
    poll.
    """
    calls: "list[str]" = []
    swept = threading.Event()
    real = cm._prune_and_count

    def _spy(*, today: "str | None" = None) -> "dict[str, int]":
        calls.append(today or "")
        out = real(today=today)
        # Signalled after the real sweep returns: setting it first would let the
        # assertions below run while the deletion is still in flight.
        swept.set()
        return out

    monkeypatch.setattr(cm, "_prune_and_count", _spy)
    # A queue of this test's own, so the sentinel below can only be taken by the
    # thread started here. `cm._queue` is the module-wide one, and a None put on it
    # could be consumed by the backend's real writer thread instead — which would
    # park this thread forever and silently stop every later test's lines landing.
    mine = queue.Queue()
    monkeypatch.setattr(cm, "_queue", mine)
    old = _write_day(_store, 40)
    worker = threading.Thread(target=cm._writer_loop, name="sweep-observer")
    worker.start()
    try:
        assert swept.wait(5.0), "the writer started without sweeping the store"
        assert calls == [_stamp(0)], calls
        assert not old.exists(), "the sweep ran but deleted nothing"
    finally:
        mine.put(None)
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "the writer thread did not take its sentinel"


class _WakingQueue:
    """A queue that empties itself `wakes` times, then hands out the exit sentinel.

    `get` records the `timeout` it was called with, because the day rollover is only
    reachable if the wait is bounded: a plain `_queue.get()` parks the thread until a
    line arrives, and a backend that idles across midnight would keep appending into
    yesterday's file with nothing to sweep it.
    """

    def __init__(self, wakes: int):
        self.wakes = wakes
        self.timeouts: "list[float | None]" = []

    def get(self, timeout: "float | None" = None, **_kw):
        self.timeouts.append(timeout)
        if self.wakes > 0:
            self.wakes -= 1
            raise queue.Empty
        return None

    def put(self, _item) -> None:
        pass

    def task_done(self) -> None:
        pass


def test_the_writer_wakes_on_a_bounded_wait_and_sweeps_the_new_day(_store, monkeypatch):
    """The day-rollover half of the trigger, and the only thing standing between this
    module and a store that grows one extra day forever: the thread must wake on a
    timer, and when the date has moved it must sweep again."""
    calls: "list[str]" = []
    real = cm._prune_and_count
    dates = iter([_stamp(0), _stamp(1), _stamp(1)])

    def _spy(*, today: "str | None" = None) -> "dict[str, int]":
        calls.append(today or "")
        return real(today=today)

    q = _WakingQueue(wakes=1)
    monkeypatch.setattr(cm, "_today", lambda: next(dates))
    monkeypatch.setattr(cm, "_prune_and_count", _spy)
    monkeypatch.setattr(cm, "_queue", q)

    cm._writer_loop()

    assert calls == [_stamp(0), _stamp(1)], calls
    assert q.timeouts and all(t is not None for t in q.timeouts), q.timeouts
    assert all(0 < t <= 3600 for t in q.timeouts), q.timeouts


def test_the_window_is_the_config_key_and_defaults_to_fourteen_days(_store, monkeypatch):
    """Clause 2, first two thirds: an absent key means 14 and the sweep honours
    what the key says, so the window is a config edit rather than a code change.
    The 3-day case is checked against the directory, because a value that is read
    but not obeyed would pass an assertion on the getter alone."""
    _set_config_window(monkeypatch, _ABSENT)
    assert cm.retention_days() == cm.DEFAULT_RETENTION_DAYS == 14
    _write_day(_store, 13)
    _write_day(_store, 14)
    cm._sweep_if_new_day("")
    assert _names(_store) == {f"{_stamp(13)}.ndjson"}, "the default window is 14 days"

    _set_config_window(monkeypatch, 3)
    assert cm.retention_days() == 3
    _write_day(_store, 2)
    _write_day(_store, 3)
    cm._sweep_if_new_day("")
    assert _names(_store) == {f"{_stamp(2)}.ndjson"}, "a 3-day window keeps days 0-2"


def test_a_numeric_string_config_value_is_read_as_a_number(_store, monkeypatch):
    """Clause 2 as it will actually be edited: `retention_days: "7"` in config.yaml
    arrives as a string, and a window that silently fell back to 14 there would
    read as the key being ignored."""
    _set_config_window(monkeypatch, "7")
    assert cm.retention_days() == 7
    _write_day(_store, 6)
    _write_day(_store, 7)
    cm._sweep_if_new_day("")
    assert _names(_store) == {f"{_stamp(6)}.ndjson"}


def test_an_unparseable_window_falls_back_to_the_default(_store, monkeypatch):
    """Clause 2's edge, so the default is a real fallback and not the only path.
    `False` is the one that matters: YAML turns an empty `retention_days:` into
    `False`, and a value read as the integer 0 means "delete everything", so a
    config typo must never empty the store. `True`, a word, an empty list and a
    whole-section `None` all read as 14 for the same reason — none of them is a
    number, and the fallback is the default rather than zero.
    """
    for value in (False, True, "not-a-number", [], None):
        _set_config_window(monkeypatch, value)
        assert cm.retention_days() == 14, value


def test_a_window_of_zero_disables_pruning_and_deletes_nothing(_store, monkeypatch):
    """Clause 2, last third: `0` means off, not "everything is older than no days"
    — the reading that would empty the store on the next writer start. Driven
    through the trigger the thread uses, and the old files have to still be there."""
    _set_config_window(monkeypatch, 0)
    assert cm.retention_days() == 0
    old = [_write_day(_store, 3), _write_day(_store, 90)]
    before = {p: p.read_bytes() for p in old}

    cm._sweep_if_new_day("")

    for path, payload in before.items():
        assert path.read_bytes() == payload, f"{path.name} went anyway"
    assert cm.stats()["pruned_files"] == 0


def test_pruning_touches_only_date_named_ndjson_inside_manifests(_store):
    """Clause 3: the keep-list, each entry a different way of not being the
    writer's own day file — the policy document, a name that is not a date, a date
    with the wrong extension, a name with the right shape and an impossible day,
    and a dated file that is not under `manifests/` at all. Every one is older than
    any window, so any one of them vanishing is this clause failing."""
    _store.mkdir(parents=True, exist_ok=True)
    policy = _store / cm.POLICY_FILENAME
    policy.write_text(cm.RETENTION_POLICY, encoding="utf-8")
    manifests = _store / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    def _keep(path: Path, text: str) -> "tuple[Path, bytes, int]":
        path.write_text(text, encoding="utf-8")
        return (path, path.read_bytes(), cm.stats()["pruned_files"])

    keep = [
        (policy, policy.read_bytes(), 0),
        _keep(manifests / "README.md", "scratch, not a day file\n"),
        _keep(manifests / "2020-01-01.jsonl", '{"request_id": "jsonl"}\n'),
        _keep(manifests / "2020-13-45.ndjson", '{"request_id": "impossible-date"}\n'),
    ]
    outside = _store / "2019-01-01.ndjson"
    outside.write_text('{"request_id": "outside-manifests"}\n', encoding="utf-8")
    mine = _write_day(_store, 30)
    mine_size = mine.stat().st_size

    # Through the thread's own entry point, so the counters the sweep is judged by
    # are the ones this diff writes: `prune_store` is the deleter and reports to its
    # caller, and only `_prune_and_count` bumps `stats()`.
    swept = cm._prune_and_count(today=_stamp(0))
    assert swept["files"] == 1 and swept["errors"] == 0, swept
    assert swept["bytes"] == mine_size, (swept["bytes"], mine_size)

    assert not mine.exists(), "the writer's own out-of-window file was not pruned"
    for path, payload, _ in keep:
        assert path.is_file(), f"{path} was removed and clause 3 forbids it"
        assert path.read_bytes() == payload, f"{path} was modified"
    assert outside.is_file(), "pruning walked outside manifests/"
    assert cm.stats()["pruned_files"] == 1
    assert cm.stats()["prune_errors"] == 0, "none of these is a file to delete"


def test_the_sweep_runs_once_per_day_however_often_the_writer_wakes(_store):
    """The cadence the trigger is responsible for: the writer wakes on an idle poll
    every 15 minutes for as long as the backend runs, so a sweep must be one
    directory listing a day and not one per wake. A same-day call deletes nothing;
    a marker from a previous day is a new day and sweeps."""
    old = _write_day(_store, 30)
    assert cm._sweep_if_new_day("") == _stamp(0)
    assert not old.exists(), "the first call of the day has to sweep"
    again = _write_day(_store, 30)
    assert cm._sweep_if_new_day(_stamp(0)) == _stamp(0)
    assert again.is_file(), "a same-day wake must not re-sweep"
    assert cm._sweep_if_new_day(_stamp(1)) == _stamp(0)
    assert not again.exists(), "yesterday's marker is a new day and must sweep"


def test_an_unreadable_store_is_counted_and_never_reaches_a_request(_store, monkeypatch):
    """Retention may fail, and the store may keep growing — what it may not do is
    break a turn or the thread that owns the store. `iterdir` raising `EACCES` is
    the shape of a mount that moved, and the recorded line is the same
    `record_request` call a send site makes."""
    _write_day(_store, 30)
    denied = OSError(13, "Permission denied")

    def _raise(*_a, **_kw):
        raise denied

    monkeypatch.setattr(Path, "iterdir", _raise)
    assert cm._prune_and_count(today=_stamp(0)) == {"files": 0, "bytes": 0,
                                                    "errors": 1}
    assert cm.stats()["prune_errors"] == 1
    monkeypatch.undo()
    assert cm.record_request(
        base_url="http://127.0.0.1:8080", model="primary",
        payload={"messages": [{"role": "user", "content": "go"}]},
        session_id="retention", send_site="test_retention") is not None
