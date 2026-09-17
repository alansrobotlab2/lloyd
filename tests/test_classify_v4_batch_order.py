"""#1206 — fresh pairs must be classified before the judged prefix is hashed.

`edges.all()` returns rows in ascending id (`app/kg_store.py:512` — `ORDER BY
id`) and the runner submitted candidates in exactly that order, so every cycle
re-walked the same leading block of already-judged pairs before it reached any
new work. Membership in the resume map is a dict lookup that costs nothing,
but that lookup only happens *after* `_prepare_candidate` has paid
`_build_context_and_hash` — the fact-tree walk and SHA1 that decide whether a
pair is a cache hit. Task #74's run of 2026-09-17 spent 6,559 of its 7,516
candidates on cache-hit skips inside a 1,400 s budget, and the triage probe
(200 sampled candidates each side) measured 186 ms mean per judged pair
against 98 ms for a fresh one: the judged prefix is the *expensive* half, so
the window was almost entirely spent hashing context the run then threw away.

The fix partitions the candidate list on one free key — pair *membership* in
the resume map — before submitting it. It cannot partition further into
hash-changed vs hash-unchanged pairs, because that distinction is only knowable
by paying the hash; the second group still carries the skip-vs-reclassify
decision inside `_prepare_candidate`, unchanged. These tests drive the real
`main()` with a stubbed classifier, a stubbed context builder and a pool that
records submission order, so what is under test is the order work reaches the
pool in — plus the resume semantics the reorder must not disturb.
"""
import importlib.util
import json
import re
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "classify-v4-batch.py"

SUMMARY = re.compile(
    r"Classified: (\d+) ok, (\d+) failed, (\d+) skipped \(cache hit\), "
    r"(\d+) cancelled \(drain\) in [\d.]+s"
)

# Edge ids in the fake store start here so the tests can assert on a real
# `id` column (`edges.all()` is `SELECT * … ORDER BY id`) rather than on list
# position. Row order in the fake store is ascending id, which is what the
# live store guarantees.
FIRST_ID = 100


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """Import the runner as a module (hyphen in filename → load by path)."""
    monkeypatch.setenv("LLOYD_KG_DB", str(tmp_path / "kg.sqlite"))
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    name = "classify_v4_batch_order_under_test"
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


class _RecordingPool:
    """`ThreadPoolExecutor` stand-in that records the order candidates arrive.

    `main()` submits from its own thread, so `submitted` is the literal order
    the futures dict was built in — no race with the workers. Everything else
    delegates to a real pool, so `as_completed`, cancellation and the drain
    path all still behave as production."""

    def __init__(self, *args, **kwargs):
        self._pool = ThreadPoolExecutor(*args, **kwargs)
        self.submitted = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append(args[0])
        return self._pool.submit(fn, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._pool.__exit__(*exc)


def _pair(i):
    return (f"src{i}", f"tgt{i}")


def _setup(mod, monkeypatch, tmp_path, n_edges, judged, *, concurrency=1):
    """Give `main()` n_edges `mentions` pairs and a resume map.

    `judged` maps a pair to the `context_hash` its prior record carries
    (`None` = a legacy record with no hash). `_build_context_and_hash` is
    stubbed to a per-source hash, so a pair is a cache hit exactly when
    `judged[pair] == f"hash-{source}"` and needs re-classification when it
    differs — which is how the hash-churn and legacy clauses stay testable.

    Concurrency is pinned to 1 by default: one worker makes execution order
    equal submission order, which is what makes "the judged prefix was hashed
    last" an exact claim instead of an eventual one. The drain half of the
    accounting lives in tests/test_classify_v4_batch_drain.py."""
    edges = [{"id": FIRST_ID + i, "source": f"src{i}", "target": f"tgt{i}",
              "type": "mentions", "provenance": "EXTRACTED"}
             for i in range(n_edges)]
    store = types.SimpleNamespace(edges=types.SimpleNamespace(
        all=lambda: list(edges)))
    monkeypatch.setattr(mod, "_v4", types.SimpleNamespace(
        _kg_store=lambda: store, VOCABULARY=mod._v4.VOCABULARY))
    monkeypatch.setattr(mod, "_load_existing_records", lambda: dict(judged))

    hashed = []

    def fake_build(edge, max_ctx_chars):
        hashed.append(edge)
        return f"ctx-{edge['source']}", f"hash-{edge['source']}"

    # The real `_prepare_candidate` stays in place: membership, the hash
    # comparison and the legacy `prior_hash is None` rule are all under test.
    monkeypatch.setattr(mod, "_build_context_and_hash", fake_build)

    pools = []

    def recording_pool(*args, **kwargs):
        pool = _RecordingPool(*args, **kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(mod, "ThreadPoolExecutor", recording_pool)

    def fake_classify(src, tgt, context, endpoint, model, timeout,
                      skip_direction_check=False):
        return {"type": "uses", "confidence": 0.9, "reason": "test"}

    monkeypatch.setattr(mod, "classify_edge_v4", fake_classify)

    out = tmp_path / "classified-v4-test.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "classify-v4-batch.py", "--concurrency", str(concurrency),
        "--output", str(out)])
    return types.SimpleNamespace(out=out, pools=pools, hashed=hashed)


def _counts(stdout):
    match = SUMMARY.search(stdout)
    assert match, f"no `Classified:` line in the expected shape:\n{stdout}"
    ok, fail, skipped, cancelled = (int(g) for g in match.groups())
    return ok, fail, skipped, cancelled


def _written(out_path):
    if not out_path.exists():
        return []
    return [json.loads(line) for line in out_path.read_text().splitlines()
            if line.strip()]


def _prefix_fixture():
    """12 pairs, ids ascending, with a contiguous judged block at the head.

    Pairs 0-5 are a contiguous judged run (the shape of a real cycle's
    leading window), 6-7 and 9-10 are unjudged, and 8 and 11 are judged too —
    the judged tail is what makes the partition, rather than a sort, visible.
    Every judged pair carries a matching hash, so the run skips all of them."""
    n = 12
    judged_idx = [0, 1, 2, 3, 4, 5, 8, 11]
    fresh_idx = [6, 7, 9, 10]
    judged = {_pair(i): f"hash-src{i}" for i in judged_idx}
    return n, judged, fresh_idx, judged_idx


def test_absent_pairs_are_submitted_before_any_pair_present_in_the_resume_map(
        mod, monkeypatch, tmp_path, capsys):
    """Clause 1: nothing in the judged prefix reaches the pool ahead of a pair
    the resume map has never seen.

    On today's code this fails: submission is pure id order, so pairs 0-5 —
    all already judged — are the first six futures."""
    n, judged, fresh_idx, judged_idx = _prefix_fixture()
    env = _setup(mod, monkeypatch, tmp_path, n, judged)
    mod.main()
    capsys.readouterr()

    order = [(e["source"], e["target"]) for e in env.pools[0].submitted]
    fresh_positions = [i for i, p in enumerate(order) if p not in judged]
    judged_positions = [i for i, p in enumerate(order) if p in judged]
    assert len(fresh_positions) == len(fresh_idx)
    assert len(judged_positions) == len(judged_idx)
    assert max(fresh_positions) < min(judged_positions), (
        "an already-judged pair was submitted before an unjudged one: "
        f"{order}")


def test_partition_is_stable_so_each_group_keeps_ascending_edge_id(
        mod, monkeypatch, tmp_path, capsys):
    """Clause 2: the reorder is a partition, not a sort — within each group
    candidates still go in ascending edge id, so an unjudged tail never
    overtakes an unjudged head."""
    n, judged, fresh_idx, judged_idx = _prefix_fixture()
    env = _setup(mod, monkeypatch, tmp_path, n, judged)
    mod.main()
    capsys.readouterr()

    submitted = env.pools[0].submitted
    assert [e["id"] for e in submitted] == [
        FIRST_ID + i for i in [*fresh_idx, *judged_idx]]
    fresh_ids = [e["id"] for e in submitted
                 if (e["source"], e["target"]) not in judged]
    judged_ids = [e["id"] for e in submitted
                  if (e["source"], e["target"]) in judged]
    assert fresh_ids == sorted(fresh_ids), "unjudged group lost id order"
    assert judged_ids == sorted(judged_ids), "judged group lost id order"


def test_the_judged_prefix_is_not_hashed_until_the_unjudged_pairs_are_done(
        mod, monkeypatch, tmp_path, capsys):
    """The acceptance sentence itself, measured at the cost seam: the run must
    not spend a single `_build_context_and_hash` call on the judged prefix
    before every fresh pair has been through it.

    This is the CPU the item is about — 186 ms mean per judged pair against
    98 ms for a fresh one — so it is pinned where the cost is paid, not only
    in the submit order that causes it."""
    n, judged, fresh_idx, judged_idx = _prefix_fixture()
    env = _setup(mod, monkeypatch, tmp_path, n, judged)
    mod.main()
    capsys.readouterr()

    hashed = [(e["source"], e["target"]) for e in env.hashed]
    n_fresh = len(fresh_idx)
    assert hashed[:n_fresh] == [_pair(i) for i in fresh_idx], (
        f"hash cost was spent before the fresh pairs: {hashed}")
    assert set(hashed[n_fresh:]) == {_pair(i) for i in judged_idx}


def test_a_changed_context_hash_is_still_reclassified_and_written(
        mod, monkeypatch, tmp_path, capsys):
    """Clause 3: moving judged pairs to the back of the queue must not stop a
    pair whose facts changed from being re-judged, nor a pair whose facts are
    unchanged from being reported as a cache hit."""
    judged = {_pair(0): "stale-hash", _pair(1): "hash-src1"}
    env = _setup(mod, monkeypatch, tmp_path, 3, judged)
    mod.main()
    stdout = capsys.readouterr().out

    ok, fail, skipped, cancelled = _counts(stdout)
    assert (ok, fail, skipped, cancelled) == (2, 0, 1, 0), (
        "expected src0 re-classified on a changed hash, src1 skipped as a "
        f"cache hit and src2 classified fresh:\n{stdout}")

    written = {r["source"]: r for r in _written(env.out)}
    assert set(written) == {"src0", "src2"}
    assert written["src0"]["context_hash"] == "hash-src0", (
        "the reclassified pair was written against a context hash the run "
        "never built")
    assert "src1" not in written, "a cache-hit pair wrote a record"


def test_a_legacy_record_with_no_context_hash_stays_up_to_date(
        mod, monkeypatch, tmp_path, capsys):
    """Clause 4: a resume record whose `context_hash` is None predates
    fingerprinting and is treated as up-to-date — re-classifying it would turn
    every old file into a stampede, and the reorder must not change that."""
    judged = {_pair(0): None}
    env = _setup(mod, monkeypatch, tmp_path, 2, judged)
    mod.main()
    stdout = capsys.readouterr().out

    ok, fail, skipped, cancelled = _counts(stdout)
    assert (ok, fail, skipped, cancelled) == (1, 0, 1, 0), (
        f"the legacy pair was re-classified rather than skipped:\n{stdout}")
    written = [r["source"] for r in _written(env.out)]
    assert written == ["src1"], "a legacy up-to-date pair wrote a record"


def test_a_run_with_no_resume_map_keeps_pure_id_order(
        mod, monkeypatch, tmp_path, capsys):
    """`--no-resume` has an empty resume map, so every pair is in the same
    partition and the partition must be a pass-through — this runner's
    documented `--no-resume` behaviour is the only thing a power-user run
    relies on, and it must not become an arbitrary order."""
    env = _setup(mod, monkeypatch, tmp_path, 6, {})
    mod.main()
    stdout = capsys.readouterr().out

    submitted = env.pools[0].submitted
    assert [e["id"] for e in submitted] == [FIRST_ID + i for i in range(6)]
    assert _counts(stdout) == (6, 0, 0, 0)
