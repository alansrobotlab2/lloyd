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
  (`test_pruning_touches_only_date_named_ndjson_inside_manifests`);
* **the notice says what the code does** (#1781) — the copy of `POLICY.md` in the
  store is rewritten by the next recorded request when its contents differ from
  shipped `RETENTION_POLICY`, and is not written at all when they do not, so the
  file beside 3 GB of digests cannot go on describing a store that predates the
  deletion rules above
  (`test_recording_a_request_refreshes_a_notice_that_disagrees`,
  `test_a_current_policy_notice_is_not_rewritten_by_recorded_requests`).

Before this diff every test here fails at import for the same reason:
`cm.prune_store`, `cm.retention_days` and `cm._sweep_if_new_day` do not exist on
`app/component_manifest.py`, which is the fact the item's premise check measured as
zero matches for `def .*(prune|retention|cleanup|purge|sweep)|unlink|MAX_AGE` in
that file.
"""

from __future__ import annotations

import json
import os
import queue
import re
import statistics
import threading
from datetime import date, timedelta
from pathlib import Path

import pytest

from app import component_manifest as cm
from app.paths import VAULT_ROOT

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


# ── the notice says what the code does (#1781) ─────────────────────────────
#
# Everything above is the deletion the notice never mentioned; these two are the
# notice itself. `_write_policy` used to run `if not exists: write`, so the copy in
# the live store froze on the build that happened to create the directory:
# `~/.local/state/lloyd-request-manifests/POLICY.md` was 1,193 bytes dated
# 2026-09-19 — the shipped text minus #1604's deletion bullet — while `prune_store`
# beside it was deleting dated files daily. Clause 3's test above writes
# `RETENTION_POLICY` into the store and proves pruning leaves it alone; these prove
# the other two edges: a differing notice is replaced, and a matching one is not
# touched at all.

#: 2020-01-01T00:00:00Z in nanoseconds. A file's own mtime is what clause 2 is
#: judged on, so it is set to an instant no writer thread could produce by
#: working, and the assertion is "nothing touched this file" rather than "nothing
#: happened within this clock tick".
_EPOCH_2020_NS = 1_577_836_800_000_000_000


def _notice_before_1604() -> str:
    """The shipped notice with #1604's deletion bullet taken out.

    That is exactly the text the live store carries: the AST diff between the
    on-disk file and `RETENTION_POLICY` was measured as five added lines, all of
    them this bullet. Built from the constant rather than transcribed so the
    fixture is that one bullet's absence and nothing else, and cannot silently
    drift into a paraphrase of the old notice.
    """
    lines = cm.RETENTION_POLICY.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("* Dated files are DELETED"))
    end = start + 1
    while end < len(lines) and lines[end].startswith("  "):
        end += 1  # the bullet's own continuation lines belong to it
    return "".join(lines[:start] + lines[end:])


def _record(_store: Path, content: str) -> None:
    """One request through the door a send site uses, drained to disk.

    The clause is about the recorded-request path, so it is driven by
    `record_request` and the writer thread behind it — never by calling
    `_write_policy` directly, which would test a function nothing calls that way.
    """
    assert cm.record_request(
        base_url="http://127.0.0.1:8080", model="primary",
        payload={"messages": [{"role": "user", "content": content}]},
        session_id="notice", send_site="test_retention") is not None
    assert cm.flush(timeout=5.0), "the writer thread did not drain what it was handed"


def _manifest_lines(store: Path) -> int:
    """Manifest lines the store actually holds, counted from disk."""
    manifests = store / "manifests"
    if not manifests.is_dir():
        return 0
    return sum(1 for path in manifests.glob("*.ndjson")
               for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip())


def test_recording_a_request_refreshes_a_notice_that_disagrees(_store):
    """Clause 1: a notice whose contents differ from `RETENTION_POLICY` is left
    byte-equal to it by the next recorded request.

    Three stale shapes, chosen so that each one kills a comparison weaker than
    byte-identity: the live store's pre-#1604 text (345 bytes shorter, which is
    what the file at `~/.local/state/lloyd-request-manifests/POLICY.md` actually
    holds), the shipped text with its window changed from 14 days to 28 — same
    byte count, so it is the fixture that defeats an implementation comparing
    sizes rather than contents — and a one-line stub. Each is asserted to differ
    before the request, because a fixture that had quietly become the shipped text
    would turn this into a test that cannot fail.
    """
    _store.mkdir(parents=True, exist_ok=True)
    policy = _store / cm.POLICY_FILENAME
    shipped = cm.RETENTION_POLICY.encode("utf-8")
    pre_1604 = _notice_before_1604().encode("utf-8")
    wrong_window = cm.RETENTION_POLICY.replace("default 14", "default 28",
                                               1).encode("utf-8")
    assert len(pre_1604) == 1193 and b"retention_days" not in pre_1604, (
        "the pre-#1604 fixture has its deletion bullet back")
    assert wrong_window != shipped and len(wrong_window) == len(shipped), (
        "the one-number drift is supposed to be invisible to a size check")

    for stale in (pre_1604, wrong_window, b"# retention policy\n"):
        policy.write_bytes(stale)
        assert policy.read_bytes() != shipped, "the stale fixture is not stale"

        _record(_store, "refresh-me")

        assert policy.read_bytes() == shipped, (
            f"a {len(stale)}-byte notice survived a recorded request")
        assert cm.stats()["write_errors"] == 0, cm.stats()

    assert cm.stats()["lines_written"] == 3, cm.stats()
    assert _manifest_lines(_store) == 3, "refreshing the notice cost the store a line"
    assert cm.stats()["pruned_files"] == 0, "nothing here is a file to delete"


def test_a_current_policy_notice_is_not_rewritten_by_recorded_requests(_store):
    """Clause 2: when the notice already matches, recording requests rewrites it
    zero times — contents identical and mtime unmoved.

    The mtime is pinned to 2020-01-01 before the requests run, so an implementation
    that rewrote the same bytes on every request would move it and fail: the clause
    forbids the write, not merely a change of text. Two requests, because a
    per-request write is the cost the clause is bounding, and the line count is
    checked to prove the writer ran at all — an mtime that never moved because
    nothing was ever drained would otherwise read as a pass.
    """
    _store.mkdir(parents=True, exist_ok=True)
    policy = _store / cm.POLICY_FILENAME
    policy.write_text(cm.RETENTION_POLICY, encoding="utf-8")
    os.utime(policy, ns=(_EPOCH_2020_NS, _EPOCH_2020_NS))
    assert policy.stat().st_mtime_ns == _EPOCH_2020_NS, "the clock could not be pinned"

    for second in ("keep-1", "keep-2"):
        _record(_store, second)

    assert policy.stat().st_mtime_ns == _EPOCH_2020_NS, (
        "a current notice was rewritten: the request path does file I/O per line")
    assert policy.read_bytes() == cm.RETENTION_POLICY.encode("utf-8")
    assert cm.stats()["lines_written"] == 2, cm.stats()
    assert _manifest_lines(_store) == 2, "the requests did not reach the store"


# --------------------------------------------------------------------------- #
# #1880 clause 5: the bound stays 1024, stays a pure count, and its comment is
# priced in the unit the entries are actually stored in.
# --------------------------------------------------------------------------- #

#: The four component names the 2026-09-30 store actually carries on a
#: `turn_start` line (`SOUL.md`, `memories`, `skills_index`, `harness_hints`). A
#: session's entry is what those four cost digested, so this is the residency the
#: comment above `MAX_SESSIONS` is obliged to state — priced at the median session
#: size the witness bytes below carry, never at a number typed in here.
_MEASURED_NAMES = ("SOUL.md", "memories", "skills_index", "harness_hints")

#: The witness bytes for those figures: the first 3,018 lines of the 2026-09-30
#: manifest store — the window the triage scan read, and the store is append-only
#: so those lines are those bytes — reduced to `ts`, `session_id`,
#: `components_captured` and `components[].{name, bytes}`, with no digest, message
#: or model field. Committed because the live store rotates under 14-day retention
#: and a figure measured off it has no history to be checked against (clause 6).
#: Provenance and the re-derivation command:
#: `~/obsidian/backlog/data/2026-09-30.components-witness.md`.
_WITNESS = VAULT_ROOT / "backlog" / "data" / "2026-09-30.ndjson"


def _witness_turn_start_median():
    """(turn_start lines, distinct sessions, median component bytes) from the bytes.

    Reads the committed witness rather than the live store, and fails rather than
    skipping if it is missing: the figures in the `MAX_SESSIONS` comment are only
    as good as these bytes, and a skipped check would leave a quoted 42,662 sitting
    in the source with nothing behind it.
    """
    assert _WITNESS.is_file(), (
        f"{_WITNESS} is absent, so the residency figures quoted in "
        "app/component_manifest.py cannot be re-derived")
    peak: dict[str, int] = {}
    n_turn_start = 0
    for raw in _WITNESS.read_text(encoding="utf-8").splitlines():
        line = json.loads(raw)
        if line.get("components_captured") != "turn_start":
            continue
        n_turn_start += 1
        sid = line.get("session_id") or ""
        total = sum(int(c.get("bytes") or 0) for c in line.get("components") or [])
        peak[sid] = max(peak.get(sid, 0), total)
    assert n_turn_start == 2332 and len(peak) == 114, (
        f"the witness no longer holds the window the item quotes: "
        f"{n_turn_start} turn_start lines over {len(peak)} sessions")
    return n_turn_start, len(peak), statistics.median(peak.values())


def _component_entry_bytes():
    """(bytes for one stored pair, bytes for a whole session's entry), measured.

    The unit is canonical JSON — the same encoding the module digests components
    in — because that is the only byte cost a reader of the comment can check.
    Measured by handing `note_components` the shape the witness bytes describe (the
    four observed names at the observed median size), so if the seam is ever
    removed and raw text comes back through the door these numbers grow by two
    orders of magnitude and every band in the node that reads them goes red.
    """
    cm._reset_registry()
    median = int(_witness_turn_start_median()[2])
    cm.note_components("priced", {n: "x" * (median // len(_MEASURED_NAMES))
                                  for n in _MEASURED_NAMES})
    entry = cm.components_for("priced")["components"]
    assert entry, "note_components stored nothing to price"
    pair = len(cm.canonical_json(entry[_MEASURED_NAMES[0]]).encode("utf-8"))
    whole = len(cm.canonical_json(entry).encode("utf-8"))
    assert pair > 0 and whole >= pair, (pair, whole)
    return pair, whole


def _sentences(comment: str) -> "list[str]":
    """A comment block as sentences, with the `#:` prefixes folded away.

    Sentence-scoped, because a block-wide figure search is satisfiable by any
    number of the right size anywhere in the block: the block above `MAX_SESSIONS`
    carries the retired cap 128 and four observed session counts, and a bare
    "some figure near 98" check passes on those alone without the comment saying
    anything about a pair.
    """
    flat = " ".join(line.lstrip("#").strip() for line in comment.splitlines())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def _figures(text: str) -> "list[int]":
    return [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)]


def _comment_above(symbol: str) -> str:
    """The comment block sitting directly above `symbol` in the real source."""
    lines = Path(cm.__file__).read_text(encoding="utf-8").splitlines()
    at = next((i for i, ln in enumerate(lines)
               if ln.startswith(f"{symbol} =")), None)
    assert at is not None, f"{symbol} is not a module-level constant any more"
    start = at
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start:at])


def test_the_bound_stays_1024_and_stays_a_pure_count():
    """#1880 clause 5, first half: the number holds and nothing age-shaped entered.

    The cap's rationale moved underneath it — a session's entry now costs a
    digest pair where it cost 42,662 B of component text, so the residency
    pressure #1782 counted is gone and the bound could eventually be *raised*.
    Ruling that is owed work (#1782 entry 4), not this round's. What no round
    may do is quietly change the *kind* of bound: the autonomy and research
    rounds that run for hours are exactly the sessions an age- or activity-based
    expiry blanks, which is why the file has carried "the bound is a count, not
    an age" since #1782.

    So: fill the registry, backdate the newest hundred entries by a year, then
    push three more sessions through. The three that leave are the three oldest
    in insertion order; the backdated hundred stay; the size is exactly the cap.
    Under any expiry the backdated entries leave first, and this node reads that
    difference rather than trusting the code's intent.
    """
    assert cm.MAX_SESSIONS == 1024, (
        f"MAX_SESSIONS is {cm.MAX_SESSIONS}: this node is the guard that the "
        "number owed a ruling from #1782 entry 4 is still the number in the file")

    for n in range(cm.MAX_SESSIONS):
        cm.note_components(f"count-{n}", {"system_prompt": "p"})
    assert len(cm._registry) == cm.MAX_SESSIONS, len(cm._registry)
    for n in range(cm.MAX_SESSIONS - 100, cm.MAX_SESSIONS):
        cm._registry[f"count-{n}"]["at"] -= 365 * 86_400

    for n in range(3):
        cm.note_components(f"late-{n}", {"system_prompt": "p"})

    assert len(cm._registry) == cm.MAX_SESSIONS, (
        f"the registry holds {len(cm._registry)} entries against a cap of "
        f"{cm.MAX_SESSIONS}, so the bound is no longer the count it states")
    assert cm.stats()["evictions"] == 3, cm.stats()
    for n in range(3):
        assert cm.components_for(f"count-{n}") == {}, (
            f"count-{n} survived while something else left: the eviction order "
            "is no longer insertion order")
        assert cm.components_for(f"late-{n}"), "the newest entries were the ones dropped"
    for n in range(cm.MAX_SESSIONS - 100, cm.MAX_SESSIONS):
        assert cm.components_for(f"count-{n}"), (
            f"count-{n} was a year stale and left anyway: that is an expiry, and "
            "the bound is a count")
    assert cm.components_for("count-500"), "an entry left with nothing over the cap"
    cm._reset_registry()


def test_the_comment_prices_the_registry_in_digest_pairs_and_names_the_way_back():
    """#1880 clause 5, second half: the residency comment says what is stored now.

    Before #1880 the block above `MAX_SESSIONS` priced the cap in entries and in
    *component text*, because that is what an entry held, and said nothing about
    the unit entries are made of now — so a reader could not tell a megabyte of
    digests from forty-four megabytes of SOUL.md. Four things are pinned here, over
    the comment as it stands in `app/component_manifest.py`, each scoped to the
    sentence that has to carry it (`_sentences`) rather than to the block:

    * a sentence about a **pair** carrying 98 — the measured canonical-JSON cost of
      one stored `{sha256, bytes}` pair — and 449, the cost of a whole entry;
    * a sentence about the **worst case** carrying 459,776 = 1024 × 449;
    * a sentence about a **median** carrying the 42,662 B the committed witness
      bytes measure, so the raw-text cost the comment warns about is a number off
      those bytes and not a number in circulation;
    * the literal `dict(components)`, the assignment that put the raw text in the
      registry, named as the input that restores that cost if the digest seam is
      ever removed — the convention #1782 set for the prefetch half.

    Exact figures, sentence-scoped, and both halves matter. Sentence-scoped because
    block-wide this check is satisfiable by the retired cap 128 and the observed
    session counts 826/646/591 the same block carries: a mutation that deleted the
    pair figure and the worst-case figure and left those came out green under the
    block-wide version. Exact because "a hundred bytes or so" is a sentence about a
    pair that prices nothing — within a 3× band it passed too.
    """
    pair, whole = _component_entry_bytes()
    median = int(_witness_turn_start_median()[2])
    block = _comment_above("MAX_SESSIONS")
    assert block.strip(), "MAX_SESSIONS has no comment block above it any more"
    sentences = _sentences(block)
    assert any("1024" in s for s in sentences), block
    assert "dict(components)" in block, (
        "the comment lost the input it must name: `entry[\"components\"] = "
        "dict(components)` is the line that puts raw component text back in the "
        "registry, and the clause requires the comment to say so")

    def _in(needle: str) -> "list[int]":
        return [n for s in sentences if needle in s for n in _figures(s)]

    def _says(figures: "list[int]", want: int, what: str, where: str) -> None:
        assert want in figures, (
            f"no sentence about {where} states {what} as the measured {want} B "
            f"(it states {figures})")

    pairs = _in("pair")
    _says(pairs, pair, "one stored digest pair", "a pair")
    _says(pairs, whole, "a whole session's entry", "a pair")
    _says(_in("worst case"), cm.MAX_SESSIONS * whole,
          "the cap's whole residency", "the worst case")
    _says(_in("median"), median,
          "the raw-text cost per session, off the committed witness bytes", "a median")
    cm._reset_registry()
