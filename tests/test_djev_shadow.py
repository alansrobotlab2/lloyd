"""`app/djev_shadow.py`: it records, and it can never reach its caller.

The recorder sits on `vault_recall`, `backlog_write_task` and the entity
sweep. The property worth pinning is not that it writes good rows — it is
that no failure of it, and no slowness of djev, can be felt by any of those
three.
"""

from __future__ import annotations

import json
import queue
import threading
import time

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
