"""`--limit N` must mean N NEW judgments (#535).

The bug: `main()` sliced the sorted candidate list (`candidates[: args.limit]`)
*before* the judgment loop, and the loop's verdict-cache-hit branch `continue`d
while still occupying one of the N slots. Every already-judged pair at the head of
the list therefore consumed budget, the slice never advanced, and the backlog
drained at the rate definitions happened to change rather than at the rate the
limit claims.

Measured on 2026-09-15 over the then-live 2026-09-08 candidate file (566,170
pairs) and verdict cache (2,000 entries): 4,515 pairs cleared the `--min-score
4.0` floor, the next `--limit 2000` slice took 2,000 of them, and 580 of those
were cache hits — so the run attempted 1,420 LLM calls on a 2,000-judgment budget
and printed a line that said nothing about it. After the change the same command
selected 2,000 pairs with the 580 passed over and not charged, 0 of the selected
pairs in the cache, and 1,935 eligible pairs left above the floor.

These tests pin the contract on a fixture pool, so they fail without the change
and read no store: slice selection computes `_cache_key` and takes the first N
pairs whose key is ABSENT, cache hits are counted and reported separately from
the budget, and the run line reports judged/cached/remaining-above-floor. The
live-store replay the round that wrote them carried is not kept: it pinned the
09-08 pool's row count and a `~/lloyd/_pipeline` path, and the runtime data
moved to `~/lloyd-data` on 2026-09-22 while the KG rebuild of 09-23 shrank the
pool to 40,812 rows (557 above the floor), so `--limit 2000` no longer binds
there. Re-measure by hand with `select_candidates(pool, 4.0, N, cache)` over the
newest `semantic-entity-candidates-*.jsonl` under `PIPELINE_DIR / "memory-graph"`
— from a process whose LLOYD_DATA is the live root, or every definition reads ""
and the cache is 0 % warm by key.

The budget the fix is sized against: task #67 runs with `timeout_seconds: 3000`.
Judgment latency measured live on 2026-09-15 over 6 uncached pairs: 0.64 / 0.65 /
0.70 / 0.72 / 0.92 / 1.07 s (median 0.71 s), and 0.49–0.57 s on an idle engine
on 2026-09-23; candidate generation ~43 s. `--limit 2000` therefore lands well
inside 3,000 s, so the task's `--limit` needs no decrease and the change buys no
larger one: it makes 2,000 mean 2,000 NEW judgments rather than the 1,420 the
slice actually bought. Cache hits are free, so a warm cache costs LESS than
2,000 judgments' worth of LLM time.
"""
import contextlib
import io
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
SER = ROOT / "scripts" / "memory" / "semantic-entity-resolution.py"


@pytest.fixture(scope="module")
def ser():
    name = "ser_limit_under_test"
    spec = importlib.util.spec_from_file_location(name, SER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cand(i: int) -> dict:
    # Distinct `a` side per index; the shared `b` side keeps the pool realistic
    # (one hot entity against many) without changing key determinism.
    return {"a": f"Entity {i:05d}", "b": "Shared Counterpart",
            "score": 10.0 - i * 1e-6, "signals": ["jaccard"]}


@pytest.fixture
def harness(ser, tmp_path, monkeypatch):
    """Drive `ser.main()` over a fixture candidate file: argv in, no LLM, no
    graph, no real verdict cache, nothing written outside tmp_path.

    Returns a callable `(n_total, n_cached, limit)` that seeds the verdict cache
    the way a previous run would and returns
    `(exit_code, stdout, judged_pairs, the_candidate_list)`.
    """
    monkeypatch.setattr(ser, "PIPELINE_ROOT", tmp_path)
    monkeypatch.setattr(ser, "CANDIDATE_LOG", tmp_path / "cand.jsonl")
    monkeypatch.setattr(ser, "JUDGMENT_LOG", tmp_path / "judg.jsonl")
    monkeypatch.setattr(ser, "PROPOSAL_LOG", tmp_path / "prop.jsonl")
    monkeypatch.setattr(ser, "PROPOSAL_CUMULATIVE", tmp_path / "prop-cumulative.jsonl")
    monkeypatch.setattr(ser, "PROPOSAL_LATEST", tmp_path / "prop-latest.jsonl")
    # Deterministic definitions ⇒ a pair's _cache_key depends only on its names.
    monkeypatch.setattr(ser, "_definition", lambda entity: f"defn:{entity}")

    state: dict = {"cache": {}, "judged": []}

    def fake_verdict_cache(path=None):
        return dict(state["cache"])

    def fake_append(rec, path=None):
        state["cache"][rec["key"]] = rec

    def fake_judge(pair, endpoint, model, timeout):
        state["judged"].append(pair)
        return {"verdict": "distinct", "confidence": 0.2, "reason": "fake"}

    monkeypatch.setattr(ser, "load_verdict_cache", fake_verdict_cache)
    monkeypatch.setattr(ser, "append_verdict", fake_append)
    monkeypatch.setattr(ser, "judge_pair", fake_judge)
    # Nothing about the entity store is needed to fill a slice; stay hermetic.
    monkeypatch.setattr(ser, "list_entities", lambda: ["Shared Counterpart"])
    monkeypatch.setattr(ser, "load_aliases", lambda: {})
    monkeypatch.setattr(ser, "load_graph", lambda: {"edges": []})
    monkeypatch.setattr(ser, "build_neighbors", lambda graph: {})
    monkeypatch.setattr(ser, "get_store", lambda: None, raising=False)
    monkeypatch.setattr(ser, "count_facts", lambda entity: 0)

    def run(n_total: int, n_cached: int, limit: int | None):
        cands = [_cand(i) for i in range(n_total)]
        cand_file = tmp_path / "in.jsonl"
        with cand_file.open("w") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
        for c in cands[:n_cached]:
            key = ser._cache_key(c["a"], c["b"], f"defn:{c['a']}", f"defn:{c['b']}")
            state["cache"][key] = {"key": key, "a": c["a"], "b": c["b"],
                                   "verdict": "distinct", "confidence": 0.2,
                                   "reason": "previously judged"}
        argv = ["semantic-entity-resolution.py", "--from-candidates", str(cand_file)]
        if limit is not None:
            argv += ["--limit", str(limit)]
        monkeypatch.setattr(sys, "argv", argv)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ser.main()
        return rc, buf.getvalue(), list(state["judged"]), cands

    return run


def _line(out: str, needle: str) -> str:
    hits = [ln for ln in out.splitlines() if needle in ln]
    assert hits, f"no line containing {needle!r} in output:\n{out}"
    return hits[0]


# --- clause 1: the budget buys new judgments -------------------------------


def test_the_slice_advances_between_two_runs(harness, tmp_path):
    """The property #535 denied: two runs over one pool must judge DIFFERENT pairs.

    Run one judges the head and appends its verdicts; run two, over the same
    candidate file with no other change, must step past them. Under the old code
    both runs took `candidates[:limit]` — the same ten pairs, judged twice, while
    the backlog behind them never moved. Also the file seam: each run's judgment
    log holds that run's ten rows only, and no row carries the selection key.
    """
    rc1, out1, judged1, cands = harness(n_total=100, n_cached=0, limit=10)
    assert rc1 == 0
    assert [p["a"] for p in judged1] == [c["a"] for c in cands[:10]]

    rc2, out2, judged2_all, _ = harness(n_total=100, n_cached=0, limit=10)
    assert rc2 == 0
    judged2 = judged2_all[len(judged1):]
    assert [p["a"] for p in judged2] == [c["a"] for c in cands[10:20]], \
        "the second run re-judged the first run's slice"

    line2 = _line(out2, "remaining-above-floor")
    assert "judged=10" in line2, line2
    assert "cached=10" in line2, line2  # run one's verdicts, passed over, not charged
    assert "remaining-above-floor=80" in line2, line2  # 100 − 10 judged − 10 judged

    rows = [json.loads(ln) for ln in
            (tmp_path / "judg.jsonl").read_text().splitlines() if ln.strip()]
    assert len(rows) == 10, f"run two's judgment log has {len(rows)} rows, expected its own 10"
    assert [r["a"] for r in rows] == [c["a"] for c in cands[10:20]]
    assert not any("_cache_key" in r for r in rows), "the selection key leaked to disk"


def test_limit_counts_uncached_pairs_not_slots(harness):
    """2,000 cached pairs at the head must not eat a 50-judgment budget."""
    rc, out, judged, cands = harness(n_total=3000, n_cached=2000, limit=50)
    assert rc == 0
    assert len(judged) == 50, f"--limit 50 must make 50 new judgments, got {len(judged)}"
    assert [p["a"] for p in judged] == [c["a"] for c in cands[2000:2050]]


def test_the_slice_skips_every_cached_pair(harness, ser):
    """The pairs that were judged before are exactly the ones not re-judged, and
    the key selection computed does not leak into the judgment log."""
    rc, out, judged, cands = harness(n_total=3000, n_cached=2000, limit=50)
    for p in judged:
        assert "_cache_key" not in p, "the selection key must not leak into the judgment log"
    cached_keys = {ser._cache_key(c["a"], c["b"], f"defn:{c['a']}", f"defn:{c['b']}")
                   for c in cands[:2000]}
    judged_keys = {ser._cache_key(p["a"], p["b"], f"defn:{p['a']}", f"defn:{p['b']}")
                   for p in judged}
    assert not (judged_keys & cached_keys), "a cache-hit pair was charged to the budget"


def test_empty_cache_slices_from_the_head(harness):
    """With nothing cached the slice is the ordinary head of the eligible pool."""
    rc, out, judged, cands = harness(n_total=100, n_cached=0, limit=10)
    assert [p["a"] for p in judged] == [c["a"] for c in cands[:10]]


def test_limit_larger_than_uncached_pool_judges_what_exists(harness):
    rc, out, judged, cands = harness(n_total=100, n_cached=90, limit=50)
    assert len(judged) == 10, f"only 10 uncached pairs exist, got {len(judged)}"
    assert [p["a"] for p in judged] == [c["a"] for c in cands[90:]]


def test_no_limit_still_serves_and_reports_cached_pairs(harness):
    """Unlimited runs keep cached verdicts: the cache is not consulted to cut the
    slice there (no budget to protect), and their verdicts still reach the file."""
    rc, out, judged, cands = harness(n_total=120, n_cached=20, limit=None)
    assert len(judged) == 100, "only the 20 cached pairs should have skipped the LLM"
    assert "from_cache=20" in _line(out, "[summary]"), out


def test_floor_is_applied_before_the_cache(ser, monkeypatch):
    """A cached pair below the floor is never even key-computed: the floor is the
    first lock (#879), the cache is the second (#535)."""
    monkeypatch.setattr(ser, "_definition", lambda entity: "d")
    pool = [{"a": "a1", "b": "b1", "score": 9.0},
            {"a": "a2", "b": "b2", "score": 1.0},
            {"a": "a3", "b": "b3", "score": 5.0}]
    cached = {ser._cache_key("a3", "b3", "d", "d"): {"verdict": "distinct"}}
    selected, above, skipped = ser.select_candidates(
        pool, min_score=4.0, limit=2, cache=cached)
    assert above == 2, "a1 and a3 clear the floor; a2 never reaches the cache"
    assert [c["a"] for c in selected] == ["a1"]
    assert skipped == 1


def test_a_pair_the_pool_names_twice_is_not_two_judgments(ser, monkeypatch):
    """A duplicate row is not new work either, so it cannot spend the budget."""
    monkeypatch.setattr(ser, "_definition", lambda entity: "d")
    pool = [{"a": "dup", "b": "counterpart", "score": 9.0}] * 3 + \
           [{"a": "next", "b": "counterpart", "score": 8.0}]
    selected, above, skipped = ser.select_candidates(
        pool, min_score=0.0, limit=10, cache={})
    assert [c["a"] for c in selected] == ["dup", "next"]
    assert (above, skipped) == (4, 2)


# --- clause 1: the two report lines ---------------------------------------


def test_limit_line_counts_uncached_pairs(harness):
    """`limited to first N` must count the uncached slice, not the raw head."""
    rc, out, judged, _ = harness(n_total=3000, n_cached=2000, limit=50)
    line = _line(out, "limited to first")
    assert "first 50 uncached" in line, line
    assert "2000 already-judged-or-duplicate pairs skipped" in line, line


def test_run_line_reports_judged_cached_remaining_above_floor(harness):
    rc, out, judged, _ = harness(n_total=3000, n_cached=2000, limit=50)
    line = _line(out, "remaining-above-floor")
    assert "judged=50" in line, line
    assert "cached=2000" in line, line
    # 3,000 eligible: 50 judged + 2,000 already judged = 950 still queued.
    assert "remaining-above-floor=950" in line, line


def test_summary_line_counts_cache_separately_from_the_budget(harness):
    """#879's `[summary]` line under #535's accounting: `from_cache` is every pair
    this run did not pay for, and the remaining number excludes them too."""
    rc, out, judged, _ = harness(n_total=3000, n_cached=2000, limit=50)
    line = _line(out, "[summary]")
    assert "newly_judged=50" in line, line
    assert "from_cache=2000" in line, line
    assert "above_floor_remaining=950" in line, line


def test_a_run_with_nothing_cached_charges_the_whole_budget_and_reports_zero_skipped(
        harness):
    """The skipped number must be 0 rather than absent, so a reader can tell a
    fresh cache from a drained pool."""
    rc, out, judged, _ = harness(n_total=100, n_cached=0, limit=10)
    assert ("0 already-judged-or-duplicate pairs skipped"
            in _line(out, "limited to first")), out
    assert "cached=0" in _line(out, "remaining-above-floor"), out


# --- the cost of the new step ----------------------------------------------


def test_selection_reads_each_definition_once(ser, monkeypatch):
    """Selecting over a 3,000-pair pool must not re-open one entity's definition
    per pair: a definition is a file read, and un-memoised that is what would make
    the new selection step costlier than the judgment it is protecting."""
    calls: list[str] = []

    def counting(entity: str) -> str:
        calls.append(entity)
        return f"defn:{entity}"

    monkeypatch.setattr(ser, "_definition", counting)
    pool = [_cand(i) for i in range(3000)]
    selected, above, skipped = ser.select_candidates(pool, min_score=0.0, limit=50, cache={})
    assert len(selected) == 50
    # 50 selected pairs, each with a unique `a` and the same `b`: 51 reads, not 100.
    assert calls.count("Shared Counterpart") == 1, "the shared side was re-read per pair"
    assert len(calls) == 51, f"expected 50 unique + 1 shared reads, got {len(calls)}"


def test_limit_advertises_the_new_judgment_contract(ser):
    """The seam the autonomy task uses is argv, so the help text is part of the
    contract: `--limit` is a budget of NEW judgments."""
    help_text = ser.build_arg_parser().format_help()
    assert "Budget of NEW judgments" in help_text, help_text
