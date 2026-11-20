"""`app/djev_shadow.py`: it records, and it can never reach its caller.

The recorder sits on `vault_recall`, `backlog_write_task` and the entity
sweep. The property worth pinning is not that it writes good rows — it is
that no failure of it, and no slowness of djev, can be felt by any of those
three.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone

import pytest

from app import djev, djev_shadow


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never the live log. `~/.local/state/lloyd-djev/shadow.jsonl` is the
    distribution the floors are read off, and 600 fixture rows in it is the
    same defect `_isolate_backlog_dedupe` exists for one file over."""
    monkeypatch.setattr(djev_shadow, "STATE_DIR", tmp_path)
    monkeypatch.setattr(djev_shadow, "SHADOW_LOG", tmp_path / "shadow.jsonl")
    monkeypatch.setattr(djev_shadow, "PENDING_DROPS", tmp_path / "drops.json")
    monkeypatch.setenv("LLOYD_DJEV_SHADOW", "1")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    djev_shadow.reset_for_tests()
    yield
    djev_shadow.reset_for_tests()


def _rows(tmp_path) -> list[dict]:
    p = tmp_path / "shadow.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _answered(monkeypatch, **over):
    calls = []
    def _ask(state, questions, **kw):
        calls.append((state, questions, kw))
        return djev.Answers(answers={}, latency_ms=1.0, server_ms=1.0,
                            prompt_tokens=1, chunks=[], uninformative=False,
                            floor=kw.get("floor"), seam=kw.get("seam", ""))
    monkeypatch.setattr(djev, "ask_sync", _ask)
    return calls


# ---------------------------------------------------------------------------
# It returns nothing and it never raises
# ---------------------------------------------------------------------------

def test_shadow_returns_none_so_no_caller_can_branch_on_it(monkeypatch, tmp_path):
    """Not a style point. A recorder that returned a truthy value would
    eventually be read by a seam, and then the observation would be a
    decision."""
    _answered(monkeypatch)
    out = djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}})
    assert out is None


def test_a_broken_queue_is_swallowed(monkeypatch):
    monkeypatch.setattr(djev_shadow, "_ensure_worker",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    assert djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}}) is None


def test_a_seam_callable_that_raises_still_writes_a_row(monkeypatch, tmp_path):
    """A seam's own lookup failing is news ABOUT that seam, not a reason to
    lose the observation that it happened."""
    _answered(monkeypatch)
    def _boom():
        raise OSError("disk gone")
    djev_shadow.shadow(seam="dedupe", state=_boom, questions={"q": {}},
                       actual={"merged_into": 4})
    djev_shadow.flush(5)
    rows = _rows(tmp_path)
    assert len(rows) == 1 and "OSError" in rows[0]["error"]
    assert rows[0]["actual"] == {"merged_into": 4}


# ---------------------------------------------------------------------------
# The bound: it drops rather than waits
# ---------------------------------------------------------------------------

def test_the_queue_drops_on_overflow_and_counts_it(monkeypatch):
    """djev is `--max-num-seqs 1` with unbounded queueing in front of it.
    A shadow call that waited would sit in front of a production decision —
    the old secondary's trap exactly."""
    monkeypatch.setattr(djev_shadow, "config",
                        lambda: {**djev_shadow.DEFAULTS, "queue_max": 2, "seams": {}})
    # A worker that never drains, so the bound is what is under test.
    monkeypatch.setattr(djev_shadow, "_ensure_worker",
                        lambda cfg: djev_shadow._queue or _stalled(2))
    for _ in range(10):
        djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}})
    st = djev_shadow.stats()
    assert st["dropped"] >= 8 and st["enqueued"] <= 2


def _stalled(maxsize):
    q = queue.Queue(maxsize=maxsize)
    djev_shadow._queue = q
    return q


def test_enqueue_is_fast_even_when_djev_is_slow(monkeypatch):
    """The hot path pays one `put_nowait`. The engine taking a second must
    not be visible to `vault_recall`."""
    def _slow(state, questions, **kw):
        time.sleep(0.5)
        return None
    monkeypatch.setattr(djev, "ask_sync", _slow)
    t0 = time.perf_counter()
    for _ in range(5):
        djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}})
    assert (time.perf_counter() - t0) < 0.05


# ---------------------------------------------------------------------------
# The worker does the expensive part
# ---------------------------------------------------------------------------

def test_callables_are_resolved_on_the_worker_not_the_caller(monkeypatch):
    """A seam enqueues ids and text it already holds; a disk read arrives as
    a callable. This pins WHERE it runs, which is the whole design."""
    calls = _answered(monkeypatch)
    where = {}
    def _state():
        where["thread"] = threading.current_thread().name
        return "built by the worker"
    djev_shadow.shadow(seam="rerank", state=_state, questions={"q": {}})
    djev_shadow.flush(5)
    assert where["thread"] == "djev-shadow"
    assert calls[0][0] == "built by the worker"


def test_the_floor_and_schema_hash_come_from_the_schema_registry(monkeypatch, tmp_path):
    """A seam names itself and nothing else. Three seams each remembering to
    pass their own floor is three places for it to go stale after one
    recalibration."""
    calls = _answered(monkeypatch)
    monkeypatch.setattr(djev_shadow, "_schema_for", lambda s: (0.42, "deadbeef"))
    djev_shadow.shadow(seam="entity", state="s", questions={"q": {}})
    djev_shadow.flush(5)
    assert calls[0][2]["floor"] == 0.42
    assert _rows(tmp_path)[0]["schema"] == "deadbeef"


# ---------------------------------------------------------------------------
# Two process shapes
# ---------------------------------------------------------------------------

def test_flush_drains_before_a_script_exits(monkeypatch, tmp_path):
    """The worker is a daemon thread, so a script takes its queue with it.
    Without this the entity seam — ~302 clusters a day, and the only one with
    true ground truth — would be the one that never recorded anything."""
    _answered(monkeypatch)
    for i in range(5):
        djev_shadow.shadow(seam="entity", state=f"s{i}", questions={"q": {}})
    assert djev_shadow.flush(10) == 0
    assert len(_rows(tmp_path)) == 5


def test_what_flush_cannot_drain_is_reported_on_the_next_row(monkeypatch, tmp_path):
    """A landing restart happens several times a night, and a silent loss
    reads exactly like a quiet seam."""
    djev_shadow._record_pending_drops(7)
    assert json.loads((tmp_path / "drops.json").read_text())["dropped"] == 7
    _answered(monkeypatch)
    djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}})
    djev_shadow.flush(5)
    rows = _rows(tmp_path)
    assert rows[0]["dropped_at_shutdown"] == 7
    # Read once and cleared, so the count is not carried forever.
    assert not (tmp_path / "drops.json").exists()


# ---------------------------------------------------------------------------
# Muting
# ---------------------------------------------------------------------------

def test_the_eval_mute_stops_everything(monkeypatch):
    """`run_eval.py` and every pinned-corpus arm set this. A replay recorded
    as production traffic would poison the very distribution the floors are
    read off — and the rerank arm calls `_vault_recall`, where the lead seam
    lives."""
    monkeypatch.setenv("LLOYD_DJEV_SHADOW", "0")
    def _never(*a, **k):  # pragma: no cover
        raise AssertionError("asked djev under the eval mute")
    monkeypatch.setattr(djev, "ask_sync", _never)
    assert djev_shadow.enabled("rerank") is False
    djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}})
    assert djev_shadow.flush(2) == 0


def test_the_mute_is_read_per_call_not_at_import(monkeypatch):
    """`run_eval.py` sets it in its own process before importing the handler,
    but the regression runner sets it for a CHILD. A module-level read would
    freeze whichever came first."""
    monkeypatch.setenv("LLOYD_DJEV_SHADOW", "0")
    assert djev_shadow.enabled("rerank") is False
    monkeypatch.setenv("LLOYD_DJEV_SHADOW", "1")
    assert djev_shadow.enabled("rerank") is True


def test_a_seam_absent_from_config_is_on(monkeypatch):
    """Adding a seam and forgetting its flag must not be the same as
    switching it off, which is the failure mode of every allow-list default."""
    monkeypatch.setattr(djev_shadow, "config",
                        lambda: {**djev_shadow.DEFAULTS, "seams": {"dedupe": False}})
    assert djev_shadow.seam_enabled("rerank") is True
    assert djev_shadow.seam_enabled("dedupe") is False


def test_the_master_switch_beats_every_seam_flag(monkeypatch):
    monkeypatch.setattr(djev_shadow, "config",
                        lambda: {**djev_shadow.DEFAULTS, "enabled": False,
                                 "seams": {"rerank": True}})
    assert djev_shadow.seam_enabled("rerank") is False


def test_a_disabled_slot_records_nothing(monkeypatch):
    monkeypatch.setattr(djev, "enabled", lambda: False)
    assert djev_shadow.enabled("rerank") is False


def test_log_rows_is_none_when_unreadable_not_zero(monkeypatch, tmp_path):
    """An unreadable log and an empty one are different answers."""
    assert djev_shadow._log_rows() == 0
    monkeypatch.setattr(djev_shadow, "SHADOW_LOG", tmp_path)   # a directory
    assert djev_shadow._log_rows() is None


# ---------------------------------------------------------------------------
# The log's own byte cap (#4410)
# ---------------------------------------------------------------------------

#: 20 bytes. The cap is a config value in MB, so asking for "over the cap" means a
#: cap smaller than one row, not a file big enough to fill a disk.
_TINY_MB = 20 / (1024 * 1024)


def _cap(monkeypatch, mb):
    monkeypatch.setattr(djev_shadow, "config",
                        lambda: {**djev_shadow.DEFAULTS, "seams": {},
                                 "max_log_mb": mb})


def _seed(tmp_path, *raw_lines) -> None:
    """Put these lines in the log exactly as written, so a test can ask whether the
    rotation changed one byte of them."""
    (tmp_path / "shadow.jsonl").write_text(
        "".join(ln + "\n" for ln in raw_lines), encoding="utf-8")


def _log_lines(tmp_path) -> list[str]:
    p = tmp_path / "shadow.jsonl"
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []


def _backups(tmp_path):
    return sorted(tmp_path.glob("shadow.jsonl.bak-*"))


def test_an_over_cap_log_is_rotated_aside_before_the_new_row_lands(monkeypatch, tmp_path):
    """#4410: `max_log_mb` was a comment. `git grep -n max_log_mb` returned only the
    `DEFAULTS` line and a doc note while the live store reached 478,739,826 bytes —
    1.78x the cap its own key declared — because `_write` opened `"a"` and appended with
    no size check of any kind, and no logrotate stanza, tmpfiles entry or timer covered
    the path either.

    The cap is read through `config()` at write time, so BOTH directions are asked here
    against the same little file: a small monkeypatched value must rotate, and the
    shipped 256 must not. A node that only ever shrank the cap would pass with the
    number hard-coded to anything smaller than a row.
    """
    seeded = ('{"ts": 1780000000.0, "seam": "dedupe", "n_questions": 1}',
              '{"ts": 1780000001.0, "seam": "entity", "n_questions": 2}')
    _seed(tmp_path, *seeded)
    _cap(monkeypatch, _TINY_MB)

    djev_shadow._write({"ts": 1780000002.0, "seam": "action_review"})

    baks = _backups(tmp_path)
    assert len(baks) == 1, f"expected exactly one backup, got {[b.name for b in baks]}"
    # The dated name the store's own hand rotation already used (`bak-20260927`), and
    # asserted by shape rather than against a date this test computed, which would flake
    # across a UTC midnight.
    assert baks[0].name.startswith("shadow.jsonl.bak-")
    stamp = baks[0].name[len("shadow.jsonl.bak-"):]
    assert len(stamp) == 8 and stamp.isdigit(), f"the backup is not dated: {baks[0].name}"
    assert baks[0].read_text(encoding="utf-8") == "".join(s + "\n" for s in seeded), (
        "the rotated file is not the file that was there; a rotation that loses the "
        "history it moved aside is a deletion wearing a rename")

    assert [r["seam"] for r in _rows(tmp_path)] == ["dedupe", "entity", "action_review"], (
        "the fresh log has to be the carried rows plus the new row, oldest first — the "
        "rename happens before the append, so the reader's tail window is never a short "
        "read of a file it already measured")

    # The other side of the same boundary. The filler matters: the shipped cap is 256
    # **MB**, and a three-row log also sits under 256 *bytes*, so it would stay green if
    # `_cap_bytes` lost the MB-to-byte conversion and began rotating on every append —
    # the failure mode its docstring names as the reason a garbage value has to mean
    # "do not bound" rather than "rotate always".
    grown = ['{"ts": %r, "seam": "rerank", "raw": "%s"}' % (1780000010.0 + k, "y" * 400)
             for k in range(20)]
    _seed(tmp_path, *grown)
    size = djev_shadow.SHADOW_LOG.stat().st_size
    assert 4096 < size < 256 * 1024 * 1024, (
        f"the under-cap fixture is {size} bytes, which no longer separates a cap read "
        "as MB from one read as bytes")
    before = _log_lines(tmp_path)

    _cap(monkeypatch, djev_shadow.DEFAULTS["max_log_mb"])
    djev_shadow._write({"ts": 1780000003.0, "seam": "action_review"})
    assert len(_backups(tmp_path)) == 1, (
        f"the shipped 256 MB cap rotated a {size}-byte file")
    assert _log_lines(tmp_path)[:len(before)] == before, (
        "an under-cap log was rewritten by a write that should only have appended")
    assert len(_rows(tmp_path)) == len(before) + 1


def test_a_rotation_carries_the_newest_row_per_seam_so_no_seam_goes_dark(monkeypatch,
                                                                        tmp_path):
    """The cut must not manufacture a #1372 alarm. `seam_log_stats` marks a seam `dark`
    on `rows == 0`, so a rotation that dropped the whole file would report EVERY seam dark
    at once — destroying the one liveness signal the recorder exists to publish.

    What is carried is the NEWEST row per seam, selected on the row's own `ts`, and it is
    carried as the raw line: the row IS the observation, and re-dumping it would rewrite
    the distribution the label-mass floors are read off. Rows the reader cannot use are
    not carried — a torn line, a seam this build does not know, a row with no timestamp —
    because carrying them would claim a liveness `seam_log_stats` does not grant them.
    """
    now = 1_780_000_100.0
    old = now - 200_000                    # older than DARK_AFTER_S (86400)
    dedupe_newest = ('{"seam": "dedupe", "actual": {"merged_into": 4}, '
                     '"schema": "abc\\u00e9f", "ts": %s, "n_questions": 1}' % now)
    _seed(tmp_path,
          '{"ts": %s, "seam": "dedupe", "n_questions": 9}' % old,
          dedupe_newest,
          '{"ts": %s, "seam": "entity", "meta": {"k": "only row, and stale"}}' % old,
          '{"ts": 1780000000, "seam": "rerank" oops a torn line',
          '{"ts": %s, "seam": "future_seam"}' % now,
          '{"seam": "rerank", "note": "no ts at all"}')
    _cap(monkeypatch, _TINY_MB)

    djev_shadow._write({"ts": now + 1, "seam": "rerank"})

    kept = _log_lines(tmp_path)
    assert len(kept) == 3, f"one row per carried seam plus the new one, got: {kept}"
    assert dedupe_newest in kept, (
        "the newest `dedupe` row was rewritten instead of carried: key order, the "
        "`\\u00e9` escape and the field set all have to survive the cut, because the "
        "rows are the sample the floors are measured on")
    assert not any('"n_questions": 9' in ln for ln in kept), (
        "an older row was carried for a seam that has a newer one — one row per seam is "
        "what keeps the new file small enough to be worth the rotation")
    assert not any("future_seam" in ln for ln in kept), (
        "a seam this build does not know was carried, so it would hold a row while "
        "being invisible to every verdict `seam_log_stats` reports")
    assert not any("no ts at all" in ln for ln in kept)

    seam = djev_shadow.seam_log_stats(now=now + 1)
    assert seam["dedupe"]["rows"] == 1 and seam["dedupe"]["dark"] is False, (
        "a seam whose newest row is one second old read dark across a rotation")
    assert seam["entity"]["rows"] == 1 and seam["entity"]["dark"] is True, (
        "carrying a row is not the same as carrying a live one: a seam whose only row "
        "is older than `DARK_AFTER_S` has to stay dark, or the bound would silently "
        "switch every genuinely quiet seam to healthy")
    assert seam["rerank"]["rows"] == 1 and seam["rerank"]["dark"] is False
    assert djev_shadow.stats()["log_rows"] == 3, (
        "`log_rows` counts the whole file, so past a cut it must count the new one; a "
        "total that still includes the history that moved aside is a wrong number")
    assert len(_backups(tmp_path)) == 1


def test_a_third_backup_prunes_the_oldest_by_mtime_not_by_name(monkeypatch, tmp_path):
    """Two is the ceiling `MAX_BACKUPS` declares — the precedent on disk
    (`shadow.jsonl.bak-20260927`, 12,477,488 bytes) is a hand rotation that has pruned
    nothing since, so a rotation that copied it would only move the leak sideways.

    And the deletion is ordered by `st_mtime`, not by name: a second rotation the same
    day is named `.bak-<YYYYMMDD>-2`, which string-sorts BEFORE the plain-dated file it
    actually followed, so a name-ordered prune keeps the stale backup and throws away the
    history written last.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    plain = tmp_path / f"shadow.jsonl.bak-{stamp}"
    suffixed = tmp_path / f"shadow.jsonl.bak-{stamp}-2"
    for p in (plain, suffixed):
        p.write_text('{"ts": 1780000000.0, "seam": "dedupe"}\n', encoding="utf-8")
    stale = time.time() - 7_200
    os.utime(plain, (stale + 7_200, stale + 7_200))     # newest by name…
    os.utime(suffixed, (stale, stale))                  # …oldest by mtime
    _seed(tmp_path, '{"ts": 1780000001.0, "seam": "dedupe", "n_questions": 3}')
    _cap(monkeypatch, _TINY_MB)

    djev_shadow._write({"ts": 1780000002.0, "seam": "entity"})

    assert len(_rows(tmp_path)) == 2, "the rotation did not happen at all"
    survivors = [b.name for b in _backups(tmp_path)]
    assert len(survivors) == 2, f"more than MAX_BACKUPS survived: {survivors}"
    assert suffixed.name not in survivors, (
        f"the deleted backup was the newest by name, not the newest by mtime: "
        f"{survivors}. `-2` sorts before the plain-dated file it followed, so ordering "
        "the prune by name keeps the stale copy and discards the newer history")
    assert plain.name in survivors


def test_the_worker_exits_instead_of_spinning_when_its_queue_is_cleared(monkeypatch):
    """The busy-wait this caught. `_run` read the module global on every
    iteration and looped on `None` with no sleep, so a worker left alive by
    `reset_for_tests()` spun a core at 100% for the life of the process — one
    pytest worker at 99% CPU starving the other seven.
    """
    _answered(monkeypatch)
    djev_shadow.shadow(seam="rerank", state="s", questions={"q": {}})
    djev_shadow.flush(5)
    worker = djev_shadow._worker
    assert worker is not None and worker.is_alive()
    # What `reset_for_tests` does, and what a second `_ensure_worker` would do
    # if the queue were ever replaced under a live worker.
    djev_shadow._queue = None
    worker.join(timeout=12)      # one `get` timeout plus slack
    assert not worker.is_alive(), "the worker kept running with no queue"
