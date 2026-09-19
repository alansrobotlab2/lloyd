"""Pins for backlog #899: one guarded writer for the groundskeeper queue.

The queue is rebuilt nightly and rewritten by two consumers under
``scripts/memory/`` whose caller has never been identified. On 2026-03-30 a
consumer wrote only the four items it had processed back over a queue holding
~2,700 and the loss was unrecoverable; the survey grew a verify-then-rename
guard, the consumers never did, and one of them still re-dumped the whole queue
with a bare ``json.dump`` nightly. These tests pin the guarded writer
(``scripts/groundskeeper/queue_io.py``), that both consumers reach the queue
only through it, that each stamped item logs its own reason, and that every
write leaves an attribution row.

Nothing here touches the live queue: the consumers take their queue path from
``GROUNDKEEPER_QUEUE``, so each test runs a real subprocess against a fixture in
``tmp_path`` — which is also the seam (env var -> child process -> files on
disk) the change crosses.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.groundskeeper import queue_io
from scripts.groundskeeper.queue_io import QueueWriteError, write_queue_atomic

REPO = Path(__file__).resolve().parents[1]
PROCESS_SCRIPT = REPO / "scripts/memory/process-groundskeeper-queue.py"
BATCH_SCRIPT = REPO / "scripts/memory/batch-process-orphans.py"
SURVEY_SCRIPT = REPO / "scripts/groundskeeper/groundskeeper-survey.py"

#: Every process that writes the queue. Before #899 the first two finished with
#: a bare ``open(queue, 'w'); json.dump`` and the third kept its verify-then-
#: rename inline where no other writer could inherit it.
QUEUE_WRITERS = [PROCESS_SCRIPT, BATCH_SCRIPT, SURVEY_SCRIPT]

REASON_SURVEY_BUG = "hub-page-linked-survey-bug"
REASON_ORGANIZED = "legitimate organized project file in folder hierarchy"
GENERATED_AT = "2026-09-18T07:15:02.114375"
UTC_SECOND = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _orphan(item_id: int, source: str, status: str = "pending") -> dict:
    return {"id": item_id, "type": "ORPHAN_FILE", "source_file": source, "status": status}


def _queue(items: list[dict]) -> dict:
    return {
        "generated_at": GENERATED_AT,
        "items": items,
        "health_score": {"overall": 62.5, "orphan_files": 0},
    }


def _write_fixture(tmp_path: Path, items: list[dict]) -> Path:
    """A fixture queue in its own directory, shaped like the live one."""
    queue_path = tmp_path / "groundskeeper-queue.json"
    queue_path.write_text(json.dumps(_queue(items), indent=2), encoding="utf-8")
    return queue_path


def _run_consumer(script: Path, queue_path: Path) -> subprocess.CompletedProcess:
    """Run a consumer for real, pointed at the fixture by environment."""
    env = dict(os.environ, GROUNDKEEPER_QUEUE=str(queue_path))
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"{script.name} exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return proc


def _rows(path: Path) -> list[dict]:
    assert path.exists(), f"{path.name} was never written"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_survey_module(tmp_path: Path):
    """Import the survey pointed at a fixture queue, so importing cannot touch the live one.

    Loaded under a private module name and dropped from ``sys.modules`` again:
    ``main()`` runs a vault scan that no test here wants, but the module-level
    constants are what the queue-write call reads, and those are what the pin
    needs to see.
    """
    import importlib.util

    monkey_env = dict(
        os.environ, GROUNDKEEPER_QUEUE=str(tmp_path / "groundskeeper-queue.json")
    )
    saved = os.environ.get("GROUNDKEEPER_QUEUE")
    os.environ.update(monkey_env)
    try:
        spec = importlib.util.spec_from_file_location("_gk_survey_under_test", SURVEY_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            os.environ.pop("GROUNDKEEPER_QUEUE", None)
        else:
            os.environ["GROUNDKEEPER_QUEUE"] = saved
        sys.modules.pop("_gk_survey_under_test", None)
    return module


# --- clause 1: the writer verifies before it renames -------------------------


def test_write_queue_atomic_renames_only_after_the_temp_copy_reads_back(tmp_path):
    """A good write lands, no temp file survives, and the row says who did it."""
    queue_path = _write_fixture(tmp_path, [_orphan(1, "knowledge/a.md", "skipped")])

    row = write_queue_atomic(
        queue_path,
        _queue([_orphan(1, "knowledge/a.md"), _orphan(2, "projects/b.md")]),
        stamped=2,
    )

    written = json.loads(queue_path.read_text(encoding="utf-8"))
    assert [i["id"] for i in written["items"]] == [1, 2]
    assert written["generated_at"] == GENERATED_AT, "a stamp must not drop the rebuild's stamp"
    assert not Path(f"{queue_path}.tmp").exists(), "the temp file must not outlive the rename"
    assert row["stamped"] == 2
    assert row["items"] == 2


@pytest.mark.parametrize(
    "broken_serialise, expected, note",
    [
        (
            lambda queue, fh: json.dump(
                {**queue, "items": queue["items"][:1]}, fh, indent=2
            ),
            "expected 4, got 1",
            "well-formed JSON holding only the one item a loop processed — the "
            "2026-03-30 shape: 4 items written back over a queue of ~2,700",
        ),
        (
            lambda queue, fh: fh.write(json.dumps(queue, indent=2)[:40]),
            "failed",
            "a write cut off halfway, which re-reads as unparseable JSON",
        ),
    ],
    ids=["short-queue", "truncated-json"],
)
def test_a_bad_temp_copy_is_refused_and_the_live_queue_is_byte_identical(
    tmp_path, monkeypatch, broken_serialise, expected, note
):
    """The guard is the rename's precondition: fail the copy, keep the queue."""
    queue_path = _write_fixture(
        tmp_path, [_orphan(1, "knowledge/a.md"), _orphan(2, "projects/b.md"), _orphan(3, "x.md")]
    )
    before = queue_path.read_bytes()
    monkeypatch.setattr(queue_io, "_serialise", broken_serialise)

    with pytest.raises(QueueWriteError) as raised:
        write_queue_atomic(
            queue_path,
            _queue([_orphan(1, "a"), _orphan(2, "b"), _orphan(3, "c"), _orphan(4, "d")]),
            stamped=4,
        )

    assert expected in str(raised.value), note
    assert queue_path.read_bytes() == before, "a refused write must leave the queue untouched"
    assert not Path(f"{queue_path}.tmp").exists(), "the refused temp file must be deleted"
    assert not (tmp_path / queue_io.WRITES_LOG_NAME).exists(), (
        "a write that never happened must not be attributed"
    )


def test_a_queue_with_no_items_list_is_refused(tmp_path):
    """'Empty' and 'wrong shape' are different answers, and only one is writable."""
    queue_path = _write_fixture(tmp_path, [_orphan(1, "knowledge/a.md")])
    before = queue_path.read_bytes()

    with pytest.raises(QueueWriteError, match="no 'items' list"):
        write_queue_atomic(queue_path, {"generated_at": GENERATED_AT})

    assert queue_path.read_bytes() == before


# --- clause 2: the consumers reach the queue only through the writer ---------


@pytest.mark.parametrize(
    "script", QUEUE_WRITERS, ids=[p.name for p in QUEUE_WRITERS]
)
def test_no_writer_holds_a_direct_queue_write(script):
    """No bare open(queue,'w'), json.dump or rename survives in any writer."""
    src = script.read_text(encoding="utf-8")

    assert not re.search(
        r"open\(\s*(queue_path|QUEUE_PATH|QUEUE_OUTPUT|['\"][^'\"]*groundskeeper-queue\.json['\"])"
        r"\s*,\s*['\"]w['\"]",
        src,
    ), f"{script.name} still opens the queue for writing directly"
    assert not re.search(r"json\.dump\(", src), (
        f"{script.name} still serialises the queue itself instead of calling the writer"
    )
    assert "os.rename(" not in src, (
        f"{script.name} renames by hand; only queue_io may rename"
    )
    assert "write_queue_atomic(" in src, f"{script.name} does not use the guarded writer"


def test_every_writer_resolves_the_name_to_the_one_shared_implementation(tmp_path):
    """The guard is shared by import, not copied — a copy is what drifted before."""
    survey = _load_survey_module(tmp_path)

    assert survey.write_queue_atomic is queue_io.write_queue_atomic
    assert survey.QueueWriteError is queue_io.QueueWriteError
    assert survey.QUEUE_OUTPUT == str(tmp_path / "groundskeeper-queue.json"), (
        "the survey must take its queue path from the same seam the consumers use"
    )
    for script in (PROCESS_SCRIPT, BATCH_SCRIPT):
        src = script.read_text(encoding="utf-8")
        assert "from scripts.groundskeeper.queue_io import" in src, script.name


@pytest.mark.parametrize(
    "script, zero_line, signal",
    [
        (PROCESS_SCRIPT, "- Processed: 0", "SIGNAL:TASK_COMPLETE"),
        (BATCH_SCRIPT, "Processed 0 items", "Processed 0 items"),
    ],
    ids=["survey-bug", "batch"],
)
def test_a_consumer_exits_zero_on_a_queue_with_no_pending_orphans(
    script, zero_line, signal, tmp_path
):
    """Tonight's pass found 0 pending orphans; the nightly job must not start failing."""
    queue_path = _write_fixture(
        tmp_path,
        [
            _orphan(1, "knowledge/a.md", "skipped"),
            _orphan(2, "projects/b.md", "done"),
            {"id": 3, "type": "BROKEN_LINK", "source_file": "notes/c.md", "status": "pending"},
        ],
    )

    proc = _run_consumer(script, queue_path)

    assert zero_line in proc.stdout, proc.stdout
    assert signal in proc.stdout, proc.stdout
    written = json.loads(queue_path.read_text(encoding="utf-8"))
    assert written["items_processed"] == 0
    assert [i["status"] for i in written["items"]] == ["skipped", "done", "pending"], (
        "a pass with nothing to stamp must not change any item's status"
    )
    assert not (tmp_path / queue_io.PROCESS_LOG_NAME).exists(), (
        "an empty pass must not write process-log rows it cannot account for"
    )
    writes = _rows(tmp_path / queue_io.WRITES_LOG_NAME)
    assert len(writes) == 1 and writes[0]["stamped"] == 0


#: Dropped into the child's PYTHONPATH so the guard fires inside a real run of
#: a consumer, not only inside a test call: the same half-written serialisation
#: the 2026-03-30 incident produced, one interpreter out from the queue file.
_TRUNCATE_SITECUSTOMIZE = '''"""Test injection: make the queue writer's serialise step stop halfway."""
import json


def _install():
    import scripts.groundskeeper.queue_io as qio

    def _half(queue, fh):
        fh.write(json.dumps(queue, indent=2)[:40])

    qio._serialise = _half


_install()
'''


@pytest.mark.parametrize(
    "script", [PROCESS_SCRIPT, BATCH_SCRIPT], ids=["survey-bug", "batch"]
)
def test_a_consumer_that_cannot_write_a_whole_queue_fails_loudly_and_keeps_the_queue(
    script, tmp_path
):
    """End to end: a torn write aborts the nightly run instead of installing itself.

    Covers the ordering the clause depends on as well — no process-log row may
    exist for a stamp that never reached the queue.
    """
    queue_path = _write_fixture(
        tmp_path, [_orphan(401, "projects/alpha/one.md"), _orphan(402, "knowledge/two.md")]
    )
    before = queue_path.read_bytes()
    (tmp_path / "sitecustomize.py").write_text(_TRUNCATE_SITECUSTOMIZE, encoding="utf-8")
    env = dict(
        os.environ,
        GROUNDKEEPER_QUEUE=str(queue_path),
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(REPO)]),
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode != 0, (
        "a consumer whose write was refused must not report success: "
        f"exit {proc.returncode}\nstdout: {proc.stdout}"
    )
    assert "QueueWriteError" in proc.stderr, proc.stderr[-800:]
    assert "SIGNAL:TASK_COMPLETE" not in proc.stdout, (
        "the completion signal is the task's 'this run did its job' line"
    )
    assert queue_path.read_bytes() == before, "the queue must be byte-identical after a refusal"
    assert not Path(f"{queue_path}.tmp").exists()
    assert not (tmp_path / queue_io.WRITES_LOG_NAME).exists()
    assert not (tmp_path / queue_io.PROCESS_LOG_NAME).exists(), (
        "no stamp may be logged for a write that did not land"
    )


# --- clause 3: every stamped item logs its own reason -----------------------


def test_a_pass_that_stamps_both_kinds_leaves_both_reasons_in_the_log(tmp_path):
    """The 2026-09-11 queue held two reason strings; the log held one of them."""
    queue_path = _write_fixture(
        tmp_path,
        [
            _orphan(101, "projects/alpha/one.md"),
            _orphan(102, "knowledge/two.md"),
            _orphan(103, "notes/three.md"),
            _orphan(104, "loose.md"),
        ],
    )

    _run_consumer(BATCH_SCRIPT, queue_path)
    batch_rows = _rows(tmp_path / queue_io.PROCESS_LOG_NAME)
    assert [r["item_id"] for r in batch_rows] == [101, 102], (
        "the batch consumer stamped two items and logged none of them before #899"
    )
    assert {r["reason"] for r in batch_rows} == {REASON_ORGANIZED}

    _run_consumer(PROCESS_SCRIPT, queue_path)
    rows = _rows(tmp_path / queue_io.PROCESS_LOG_NAME)

    reasons = {r["reason"] for r in rows}
    assert reasons == {REASON_ORGANIZED, REASON_SURVEY_BUG}, (
        f"expected both reason strings, measured {reasons}"
    )
    assert len(rows) == 4, "one row per stamped item, not one reason for the whole run"

    by_id = {i["id"]: i for i in json.loads(queue_path.read_text(encoding="utf-8"))["items"]}
    for row in rows:
        assert row["status"] == "skipped"
        assert row["processed_at"] == by_id[row["item_id"]]["processed_at"]
        assert row["reason"] == by_id[row["item_id"]]["reason"], (
            "a log row must carry the reason the queue carries for that item"
        )


def test_the_batch_cap_still_bounds_one_run(tmp_path):
    """Logging must not change the 25-item cap the batch consumer runs under."""
    queue_path = _write_fixture(
        tmp_path, [_orphan(i, f"projects/p{i}.md") for i in range(1, 27)]
    )

    _run_consumer(BATCH_SCRIPT, queue_path)

    assert len(_rows(tmp_path / queue_io.PROCESS_LOG_NAME)) == 25, "the cap is 25"
    items = json.loads(queue_path.read_text(encoding="utf-8"))["items"]
    assert sum(1 for i in items if i["status"] == "skipped") == 25
    assert [i for i in items if i["status"] == "pending"][0]["id"] == 26, (
        "the 26th orphan must still be pending: one run still stamps at most 25"
    )


# --- clause 4: every write is attributable from the sidecar ------------------


def test_every_queue_write_appends_one_attribution_row_naming_its_writer(tmp_path):
    """Two runs, two rows, each naming the script, pid, stamps and queue it read."""
    queue_path = _write_fixture(
        tmp_path, [_orphan(201, "knowledge/a.md"), _orphan(202, "notes/b.md")]
    )

    _run_consumer(BATCH_SCRIPT, queue_path)
    _run_consumer(PROCESS_SCRIPT, queue_path)

    rows = _rows(tmp_path / queue_io.WRITES_LOG_NAME)
    assert [r["writer"] for r in rows] == [
        "batch-process-orphans.py",
        "process-groundskeeper-queue.py",
    ], "the sidecar must name which consumer wrote the queue"
    assert [r["stamped"] for r in rows] == [1, 1]
    for row in rows:
        assert isinstance(row["pid"], int) and row["pid"] > 0
        assert row["pid"] != os.getpid(), "the writer runs in its own process"
        assert row["queue_generated_at"] == GENERATED_AT, (
            "the row must carry the generated_at the writer read, so a rebuild is traceable"
        )
        assert row["items"] == 2
        assert row["queue"] == str(queue_path)
        assert UTC_SECOND.match(row["written_at"]), row["written_at"]
        stamp = datetime.strptime(row["written_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        assert abs((datetime.now(timezone.utc) - stamp).total_seconds()) < 300, (
            f"{row['written_at']} is not a UTC timestamp"
        )


def test_the_writer_names_the_script_that_called_it(tmp_path, monkeypatch):
    """Attribution comes from the process, so an import cannot forge another writer."""
    queue_path = _write_fixture(tmp_path, [_orphan(301, "knowledge/a.md")])
    monkeypatch.setattr(sys, "argv", ["/somewhere/groundskeeper-survey.py", "extra"])

    row = write_queue_atomic(queue_path, _queue([_orphan(301, "knowledge/a.md")]))

    assert row["writer"] == "groundskeeper-survey.py"
    assert row["stamped"] == 0, "an unstamped rewrite still has to be attributed"

    monkeypatch.setattr(sys, "argv", [])
    anonymous = write_queue_atomic(queue_path, _queue([_orphan(301, "knowledge/a.md")]))
    assert anonymous["writer"] == "<unknown>", (
        "a writer that cannot name itself must say so, not stay silent"
    )
    assert len(_rows(tmp_path / queue_io.WRITES_LOG_NAME)) == 2
