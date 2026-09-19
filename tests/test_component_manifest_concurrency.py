"""Several OS processes, one day file: the append seam #581 had to prove.

The writer thread is one per *process*, and the processes are many: the backend,
`agent_mcp/main.py`, `workers/sources/*.py` and the automod round subprocess each
run a `stream_chat` loop and each appends to
`<store>/manifests/<date>.ndjson`. They share nothing but the path, so no queue
orders them and no lock in this module's own state protects them. Every record has
to arrive whole, which holds only if a record is one `write(2)` under an exclusive
lock — a property of the bytes this store writes, not of CPython's buffer size.

A line here is genuinely large: measured at 22,737 bytes with 131 advertised tools,
five times `PIPE_BUF` (4,096 on this box) and nearly three times the 8 KiB stdio
buffer. So the question is not academic, and a 22 KB record is what these tests
append rather than a short one — a small record survives almost any writer.
"""
from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import component_manifest as cm  # noqa: E402

PROCS, PER_PROC = 8, 12
# Above PIPE_BUF and above the 8 KiB stdio buffer, in the range of a real line.
TARGET_BYTES = 22_000


def _child(store: str, proc: int, n: int, target: int) -> None:
    """Append `n` real-shaped records to the shared day file, from its own process."""
    os.environ["LLOYD_MANIFEST_STORE"] = store
    sys.path.insert(0, str(REPO))
    from app import component_manifest as child_cm
    day = child_cm._day_file(Path(store))
    day.parent.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rec = json.dumps({"request_id": f"{proc}-{i}", "proc": proc, "seq": i,
                          "session": f"20260919_120000_worker_t{i % 3}_s",
                          "filler": "x" * target})
        child_cm._append_line(Path(store), rec)


def _record_bytes(cm_module, store: Path) -> tuple[Path, bytes]:
    """One real manifest line, padded to `TARGET_BYTES`, from the real writer."""
    cm_module._reset_tools_memo()
    cm_module.note_components("20260919_120000_probe_s", {"SOUL.md": "s" * 200})
    line = cm_module.record_request(
        base_url="http://127.0.0.1:8096", model="primary",
        session_id="20260919_120000_probe_s", iteration=1,
        send_site="tests/test_component_manifest_concurrency.py",
        payload={"model": "primary", "messages": [{"role": "user", "content": "go"}]})
    assert line is not None, "the writer recorded nothing"
    body = dict(line)
    body["filler"] = "x" * max(0, TARGET_BYTES - len(json.dumps(body)))
    day = cm_module._day_file(store)
    day.parent.mkdir(parents=True, exist_ok=True)
    return day, json.dumps(body).encode("utf-8")


def test_a_record_is_written_in_one_syscall_not_chunked_by_a_buffer(monkeypatch, tmp_path):
    """The property that makes the cross-process case safe, tested without processes.

    A buffered text-mode write of a 22 KB line reaches the kernel as two or more
    `write(2)` calls, and between two of them another process's bytes may land: the
    resulting file has the right total length and the wrong records. Here the
    append must issue exactly one `os.write` for the whole record, so an interleave
    has nowhere to happen. Chunking a write is the regression this pins, and a test
    that only appended concurrently would very likely not notice it — on this box a
    torn write needs an unlucky schedule, not a broken design.
    """
    store = tmp_path / "store"
    day, payload = _record_bytes(cm, store)
    # Drain the writer thread first: with its own queued record still in flight it
    # would append during the counted window below, and its bytes would be counted
    # as part of this record.
    assert cm.flush(timeout=8.0), "the writer thread did not drain"
    before = day.read_bytes() if day.exists() else b""

    calls: list[int] = []
    real_write = os.write

    def counting_write(fd: int, data) -> int:
        calls.append(len(bytes(data)))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", counting_write)
    cm._append_line(store, payload.decode("utf-8"))

    assert sum(calls) == len(payload) + 1, (
        f"the record reached the kernel as {len(calls)} writes totalling "
        f"{sum(calls)} bytes of {len(payload) + 1}: a record split across syscalls "
        f"is a record another process can cut in half")
    assert len(calls) == 1, calls
    assert day.read_bytes()[len(before):] == payload + b"\n"


def test_appending_blocks_while_another_process_holds_the_file(monkeypatch, tmp_path):
    """The lock is taken by *this* writer, not assumed.

    `O_APPEND` alone still permits a torn record if one write is split, so the
    exclusive lock is what makes the seam safe whatever the kernel does with a long
    write. A test of the guarantee, not of the comment: hold `LOCK_EX` on the day
    file from this process, then ask the writer to append. If it returns anyway it
    never asked for the lock and provides nothing another appender can rely on.
    `flock` locks the open-file description, so the blocker's own fd cannot see its
    own lock — which is exactly why an unpatched writer would sail past it.
    """
    store = tmp_path / "store"
    day, payload = _record_bytes(cm, store)
    holder = os.open(day, os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)

    done = threading.Event()

    def append() -> None:
        cm._append_line(store, payload.decode("utf-8"))
        done.set()

    monkeypatch.setattr(cm, "_bump", lambda *a, **k: None)
    thread = threading.Thread(target=append, daemon=True)
    thread.start()
    try:
        blocked = not done.wait(timeout=0.75)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)
    thread.join(timeout=10)

    assert blocked, ("_append_line wrote while another fd held LOCK_EX, so it takes "
                     "no exclusive lock and the multi-process guarantee is prose")
    assert done.is_set(), "the append never completed once the lock was released"
    assert day.read_bytes() == payload + b"\n"


def test_eight_processes_appending_records_at_real_size_lose_nothing(tmp_path):
    """The cross-process seam itself: real processes, one day file, no lost or torn line.

    Each child sets the store root in its own environment and calls the module's
    single append point — the same call the writer thread makes after its own queue
    drains — so nothing here shares memory with anything else in the test. Every one
    of the `PROCS * PER_PROC` records has to be present and parseable, and every
    `request_id` unique: a lost record is a dropped request, and an unparseable line
    is a store that `prompt_diff` can no longer read.
    """
    store = tmp_path / "store"
    (store / "manifests").mkdir(parents=True)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_child, args=(str(store), p, PER_PROC, TARGET_BYTES))
             for p in range(PROCS)]
    for proc in procs:
        proc.start()
    deadline = time.time() + 120
    for proc in procs:
        proc.join(timeout=max(1.0, deadline - time.time()))
    assert all(not proc.is_alive() for proc in procs), "a child hung mid-append"
    assert all(proc.exitcode == 0 for proc in procs), [p.exitcode for p in procs]

    lines = (store / "manifests" / cm._day_file(store).name).read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == PROCS * PER_PROC, (
        f"{len(lines)} of {PROCS * PER_PROC} records landed; a lost line is a lost "
        f"request")
    ids = [json.loads(line)["request_id"] for line in lines]
    assert len(set(ids)) == len(ids), "two processes wrote the same request id"
    assert sorted(ids) == sorted(f"{p}-{i}" for p in range(PROCS)
                                 for i in range(PER_PROC))


def test_a_torn_line_would_be_caught(tmp_path):
    """The detector above works: a hand-interleaved store fails it, so it is not vacuous.

    Every concurrency test earns its keep by failing on the thing it claims to
    detect. The check is that each record parses and each `request_id` is unique, so
    writing two records as one line — what an interleave between chunked writes
    produces — must trip it.
    """
    store = tmp_path / "store"
    (store / "manifests").mkdir(parents=True)
    day = store / "manifests" / cm._day_file(store).name
    a = json.dumps({"request_id": "0-0", "filler": "x" * 100})
    b = json.dumps({"request_id": "1-0", "filler": "y" * 100})
    day.write_text(a[:50] + b + "\n" + a[50:] + "\n", encoding="utf-8")

    lines = day.read_text(encoding="utf-8").splitlines()
    torn = 0
    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            torn += 1
    assert torn == 2, (
        "a deliberately interleaved pair was read back as valid NDJSON, so the "
        "intactness check in the test above is checking nothing")
