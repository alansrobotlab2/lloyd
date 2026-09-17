"""#526 — a SIGTERM drain must not be reported as a mass LLM failure.

The batch runner consumed every future through one `except Exception`, and
`concurrent.futures.CancelledError` **is** an Exception subclass (see
test_cancelled_error_is_an_exception_below), so the futures the drain path
cancels at classify-v4-batch.py:344-349 were tallied into `fail`. Task #74's
run at 2026-09-09T01:53Z printed `2455 ok, 6239 failed` when 6,237 of those
were `CancelledError()` and 2 were real — a clean timeout reading as
catastrophic, with the genuine failures buried in the noise.

The fix separates the channels: drain cancels go to `cancelled (drain)`,
`fail` means a failure. These tests drive the real `main()` with a stubbed
classifier so the accounting, not the LLM, is what is under test.
"""
import importlib.util
import json
import re
import sys
import time
import types
from concurrent.futures import CancelledError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "classify-v4-batch.py"

# `Classified: 2455 ok, 2 failed, 1061 skipped (cache hit), 6237 cancelled
# (drain) in 1421.3s`
SUMMARY = re.compile(
    r"Classified: (\d+) ok, (\d+) failed, (\d+) skipped \(cache hit\), "
    r"(\d+) cancelled \(drain\) in [\d.]+s"
)


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """Import the runner as a module (hyphen in filename → load by path).

    The kg store is configured only because importing `classify-relationships-v4`
    resolves store paths; `main()` is handed a fake one either way."""
    monkeypatch.setenv("LLOYD_KG_DB", str(tmp_path / "kg.sqlite"))
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    name = "classify_v4_batch_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)
    spec.loader.exec_module(m)
    # The runner installs SIGTERM/SIGINT handlers on the process; a test
    # process keeps its own.
    monkeypatch.setattr(m, "_install_signal_handlers", lambda: None)
    m._stop.clear()
    try:
        yield m
    finally:
        m._stop.clear()
        kg_store.reset()


def _setup(mod, monkeypatch, tmp_path, n_edges):
    """Give `main()` n_edges fake `mentions` pairs and a tmp output file.

    Returns the output path. Concurrency is pinned to 1 so completion order
    equals submission order, which is what makes the drain assertions exact
    rather than eventual."""
    edges = [{"source": f"src{i}", "target": f"tgt{i}", "type": "mentions",
              "provenance": "EXTRACTED"} for i in range(n_edges)]
    store = types.SimpleNamespace(edges=types.SimpleNamespace(
        all=lambda: list(edges)))
    monkeypatch.setattr(mod, "_v4", types.SimpleNamespace(
        _kg_store=lambda: store, VOCABULARY=mod._v4.VOCABULARY))
    # Real _prepare_candidate reads fact snippets off disk; the pair and a
    # stable context_hash are all the accounting under test needs.
    monkeypatch.setattr(mod, "_prepare_candidate",
                        lambda edge, prior, max_ctx: (edge, "ctx", "ch"))
    out = tmp_path / "classified-v4-test.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "classify-v4-batch.py", "--no-resume", "--concurrency", "1",
        "--output", str(out)])
    return out


def _verdict(edge_type="uses"):
    return {"type": edge_type, "confidence": 0.9, "reason": "test"}


def _run(mod, capsys, out):
    rc = mod.main()
    return rc, capsys.readouterr().out, out


def _counts(stdout):
    match = SUMMARY.search(stdout)
    assert match, f"no `Classified:` line in the expected shape:\n{stdout}"
    ok, fail, skipped, cancelled = (int(g) for g in match.groups())
    return ok, fail, skipped, cancelled


def test_cancelled_error_is_an_exception():
    """Why this needed a separate except clause at all.

    If a future Python makes CancelledError a BaseException (like
    asyncio's), the blanket handler stops swallowing it and the bug this
    file exists to prevent becomes impossible — this assertion is then
    stale information, not a failing requirement, but the drain tests
    above still hold either way."""
    assert issubclass(CancelledError, Exception)


def test_clean_drain_reports_cancelled_not_failed(mod, monkeypatch, tmp_path,
                                                  capsys):
    """A run that lost nothing must print `0 failed`."""
    total = 10
    out = _setup(mod, monkeypatch, tmp_path, total)

    def fake_classify(src, tgt, context, endpoint, model, timeout,
                      skip_direction_check=False):
        # The first call stands in for "SIGTERM arrived": ask the runner to
        # drain, then hold the single worker busy long enough for main to
        # cancel everything still queued behind this one.
        if src == "src0":
            mod._stop.set()
            time.sleep(0.2)
        return _verdict()

    monkeypatch.setattr(mod, "classify_edge_v4", fake_classify)
    _rc, stdout, out_path = _run(mod, capsys, out)

    ok, fail, skipped, cancelled = _counts(stdout)
    assert fail == 0, f"drain cancels leaked into `failed`:\n{stdout}"
    assert ok >= 1, f"nothing completed; the drain never ran:\n{stdout}"
    assert ok + cancelled == total, f"work was double-counted:\n{stdout}"
    assert skipped == 0
    # Resume semantics are unchanged by this fix: a cancelled pair wrote
    # nothing, so the next run picks it up.
    written = len(out_path.read_text().splitlines())
    assert written == ok


def test_mixed_run_counts_only_genuine_failures(mod, monkeypatch, tmp_path,
                                                capsys):
    """The incident shape: real failures reported, cancels separated.

    src4 genuinely fails; the drain arrives at src5. Everything after that
    is cancelled pending work."""
    total = 10
    out = _setup(mod, monkeypatch, tmp_path, total)

    def fake_classify(src, tgt, context, endpoint, model, timeout,
                      skip_direction_check=False):
        if src == "src4":
            raise RuntimeError("endpoint hiccup")
        if src == "src5":
            mod._stop.set()
            time.sleep(0.2)
        return _verdict()

    monkeypatch.setattr(mod, "classify_edge_v4", fake_classify)
    rc, stdout, _ = _run(mod, capsys, out)

    ok, fail, skipped, cancelled = _counts(stdout)
    assert fail == 1, f"the one real failure is not the only failure:\n{stdout}"
    assert ok == 5, f"expected src0..src3 + src5 to classify:\n{stdout}"
    assert cancelled == 4, f"4 pairs were pending at the drain:\n{stdout}"
    assert ok + fail + cancelled == total
    # A real failure still makes the run non-zero. Exit-status *meaning*
    # on a drain is backlog #652, not this test.
    assert rc == 1


def test_worker_side_exception_still_counts_as_failed(mod, monkeypatch,
                                                      tmp_path, capsys):
    """_process_one's own exception branch (the `reason=repr(exc)` path) is
    a real failure and stays in `fail`."""
    out = _setup(mod, monkeypatch, tmp_path, 3)

    def fake_classify(src, tgt, context, endpoint, model, timeout,
                      skip_direction_check=False):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(mod, "classify_edge_v4", fake_classify)
    rc, stdout, out_path = _run(mod, capsys, out)

    ok, fail, skipped, cancelled = _counts(stdout)
    assert (ok, fail, skipped, cancelled) == (0, 3, 0, 0)
    assert rc == 1
    assert not out_path.exists() or not out_path.read_text().strip()


def test_out_of_vocabulary_verdict_still_counts_as_failed(
        mod, monkeypatch, tmp_path, capsys):
    """A verdict outside _v4.VOCABULARY is a failed classification, not a
    cancel — it stays in `fail` (#526 acceptance)."""
    out = _setup(mod, monkeypatch, tmp_path, 3)

    def fake_classify(src, tgt, context, endpoint, model, timeout,
                      skip_direction_check=False):
        return _verdict("banana")

    monkeypatch.setattr(mod, "classify_edge_v4", fake_classify)
    rc, stdout, out_path = _run(mod, capsys, out)

    ok, fail, skipped, cancelled = _counts(stdout)
    assert (ok, fail, skipped, cancelled) == (0, 3, 0, 0)
    assert rc == 1
    assert not out_path.exists() or not out_path.read_text().strip()


def test_endpoint_down_path_still_reports_failures(
        mod, monkeypatch, tmp_path, capsys):
    """The consecutive-failure outage guard must keep seeing real failures.

    Cancelled work landing in `fail` would trip this guard on a drain;
    real failures must still trip it."""
    out = _setup(mod, monkeypatch, tmp_path, mod.CONSECUTIVE_FAILURE_LIMIT + 2)

    def fake_classify(src, tgt, context, endpoint, model, timeout,
                      skip_direction_check=False):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mod, "classify_edge_v4", fake_classify)
    monkeypatch.setattr(mod, "_endpoint_alive", lambda *a, **k: False)
    rc, stdout, _ = _run(mod, capsys, out)

    ok, fail, skipped, cancelled = _counts(stdout)
    assert fail >= mod.CONSECUTIVE_FAILURE_LIMIT
    assert "[stopped] endpoint" in stdout
    assert rc == 1


# ---------------------------------------------------------------------------
# #1206 — the resume-order partition must not disturb this accounting. The
# candidates now reach the pool never-judged first, judged after, so a drain
# that used to land inside the unjudged work now lands after it: skipped, ok
# and cancelled are all filled by the same run, which is what makes "the four
# channels still add up" a claim about the reordered runner rather than about
# the old one (which could not produce this shape at all). See tests/test_classify_v4_batch_order.py for the ordering
# itself; this is the accounting half of that change.
# ---------------------------------------------------------------------------

def _setup_mixed(mod, monkeypatch, tmp_path, judged, drain_on=None):
    """11 `mentions` pairs (ids ascending) and a real resume filter.

    Unlike `_setup`, this drives the real `_prepare_candidate` against a
    stubbed `_build_context_and_hash` and a stubbed resume map, so
    skipped-vs-reclassified is decided by hash comparison exactly as in
    production — and `--no-resume` is deliberately absent, since the resume
    map is the thing under test. `_load_existing_records` is stubbed because
    the real one globs `~/lloyd/_pipeline/memory-graph/classified-v4*.jsonl`,
    i.e. live production data.

    Concurrency 1 keeps execution order equal to submission order, so the
    drain point is exact. `drain_on` names a source whose classification raises
    the drain signal, standing in for the SIGTERM a `timeout 1400` sends at the
    end of task #74's cycle."""
    edges = [{"id": 100 + i, "source": f"src{i}", "target": f"tgt{i}",
              "type": "mentions", "provenance": "EXTRACTED"}
             for i in range(11)]
    store = types.SimpleNamespace(edges=types.SimpleNamespace(
        all=lambda: list(edges)))
    monkeypatch.setattr(mod, "_v4", types.SimpleNamespace(
        _kg_store=lambda: store, VOCABULARY=mod._v4.VOCABULARY))
    monkeypatch.setattr(mod, "_load_existing_records", lambda: dict(judged))

    monkeypatch.setattr(mod, "_build_context_and_hash",
                        lambda edge, max_ctx: (f"ctx-{edge['source']}",
                                               f"hash-{edge['source']}"))

    def fake_classify(src, tgt, context, endpoint, model, timeout,
                      skip_direction_check=False):
        if drain_on is not None and src == drain_on:
            # Same seam the drain tests above use: stand in for "SIGTERM
            # arrived" and hold the single worker busy, so the sweep that
            # cancels what is still pending has something left to cancel.
            mod._stop.set()
            time.sleep(0.2)
        return {"type": "uses", "confidence": 0.9, "reason": "test"}

    monkeypatch.setattr(mod, "classify_edge_v4", fake_classify)
    out = tmp_path / "classified-v4-mixed.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "classify-v4-batch.py", "--concurrency", "1", "--output", str(out)])
    return out


def test_mixed_judged_and_unjudged_run_still_balances_all_four_lanes(
        mod, monkeypatch, tmp_path, capsys):
    """#1206 acceptance 5: ok + failed + skipped + cancelled == candidate count
    for a set that is part judged (unchanged hash), part judged (churned hash)
    and part never-judged — and an unchanged-hash pair writes nothing.

    Submission order after the partition is src4..src9 (the six never-judged
    pairs), then src0, src1, src2 (up-to-date), src3 (churned) and src10
    (up-to-date). The drain fires inside src3's classification, so src10 is
    still pending and the run ends with three of the four lanes non-zero and
    every candidate accounted for: 7 ok, 0 failed, 3 skipped, 1 cancelled =
    11 candidates."""
    judged = {("src0", "tgt0"): "hash-src0",
              ("src1", "tgt1"): "hash-src1",
              ("src2", "tgt2"): "hash-src2",
              ("src3", "tgt3"): "stale-hash",
              ("src10", "tgt10"): "hash-src10"}
    out = _setup_mixed(mod, monkeypatch, tmp_path, judged, drain_on="src3")
    rc = mod.main()
    stdout = capsys.readouterr().out

    ok, fail, skipped, cancelled = _counts(stdout)
    total = 11
    assert ok + fail + skipped + cancelled == total, (
        f"the four lanes do not add up to the candidate count:\n{stdout}")
    assert (ok, fail, skipped, cancelled) == (7, 0, 3, 1), (
        "expected the six never-judged pairs plus the one churned pair as ok, "
        f"the three up-to-date pairs as skipped and src10 cancelled:\n{stdout}")
    # A clean drain writes nothing but real classifications: the three
    # unchanged-hash pairs and the cancelled pair contribute no lines.
    written = [json.loads(line)["source"]
               for line in out.read_text().splitlines() if line.strip()]
    assert len(written) == ok == 7, f"writes and `ok` disagree: {written}"
    assert "src0" not in written and "src1" not in written
    assert "src2" not in written and "src10" not in written, (
        "an unchanged-hash pair wrote a record")
    assert "src3" in written, "the churned pair was not re-classified"
    assert rc == 0, f"a clean drain with re-classification must exit 0: {rc}"
