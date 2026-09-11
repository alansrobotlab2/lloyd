"""The clustering pass: deterministic, offline, and the one place the
loop's split items are put back together.

84 parent items had produced 282 children by 2026-09-11, and #788, #795 and
#799 were one finding filed by three re-runs of #549. Nothing could
reassemble them. This pass reads three signals already on disk — qmd's
stored vectors, the file paths items name, the parent named in their first
line — and writes `clusters.json` for group triage to consume.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, cluster as CL, state as S


def write_item(d: Path, item_id, *, status="draft", days_old=3, tags=("backlog",),
               name="A thing", body="Do the thing.", board="lloyd", first_line=None) -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": board, "tags": list(tags)}
    p = d / f"{item_id}-{name.lower().replace(' ', '-')[:30]}.md"
    lead = f"{first_line}\n\n" if first_line else ""
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{lead}{body}\n",
                 encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(CL, "CLUSTERS_PATH", tmp_path / "clusters.json")
    monkeypatch.setattr(CL, "JUDGMENTS_PATH", tmp_path / "judgments.jsonl")
    monkeypatch.setattr(CL, "QMD_DB", tmp_path / "nope.sqlite")
    return d


def _items():
    return CL.clusterable_items(S.LEDGER_PATH)


# ── signals ────────────────────────────────────────────────────────────────

def test_named_paths_normalises_to_basename_and_drops_the_stoplist():
    body = ("see `app/harness/loop.py:123` and `loop.py`, also `agent_mcp/vault.py:9-12`, "
            "`config.yaml`, `web/src/api.ts` and `not/a/path`")
    assert CL.named_paths(body) == {"loop.py", "vault.py", "api.ts"}


@pytest.mark.parametrize("line, want", [
    ("Split from #549 during automod triage on 2026-09-10.", 549),
    ("Split out of #400 during triage.", 400),
    ("Found while implementing #570 (round SM_1).", 570),
    ("Nothing to see here; see #12 for context.", None),
])
def test_parse_parent_reads_all_three_spellings_from_the_first_line_only(line, want):
    assert CL.parse_parent(f"# title\n\n{line}\n\nLater text says Split from #999.") == want


def test_persist_parents_writes_once_and_never_touches_a_yaml_broken_file(isolated):
    p = write_item(isolated, 10, first_line="Split from #549 during automod triage on 2026-09-10.")
    q = write_item(isolated, 11, name="Broken", first_line="Split from #549 during triage.")
    q.write_text(q.read_text().replace("tags:", "tags: [unclosed\nx:", 1))
    items = [B.load_item(p), B.load_item(q)]
    assert CL.persist_parents(items) == 1
    assert yaml.safe_load(p.read_text().split("---")[1])["parent"] == 549
    assert "[unclosed" in q.read_text(), "the broken file is untouched"
    assert CL.persist_parents([B.load_item(p)]) == 0, "idempotent"


def test_candidate_pairs_shared_paths_and_common_parent_form_edges_without_vectors(isolated):
    write_item(isolated, 1, body="`app/a.py:1` and `app/b.py:2`")
    write_item(isolated, 2, body="`a.py` and `b.py` again")
    write_item(isolated, 3, body="`app/a.py`", first_line="Split from #9 during triage.")
    write_item(isolated, 4, body="unrelated", first_line="Split from #9 during triage.")
    write_item(isolated, 5, body="also unrelated")
    pairs = CL.candidate_pairs(_items(), {}, threshold=0.75)
    edges = {(p["a"], p["b"]): p["reasons"] for p in pairs}
    assert edges[(1, 2)] == ["paths"], "two shared paths"
    assert (1, 3) not in edges, "one shared path with no cosine is not enough"
    assert edges[(3, 4)] == ["parent"]


def test_a_common_parent_is_an_edge_on_its_own(isolated):
    """Four re-runs of #549 filed twelve children in 110 minutes; they are
    one consolidation job whatever their pairwise cosine. The first cut
    demanded a near-threshold cosine and dropped that family entirely."""
    import numpy as np
    write_item(isolated, 3, body="x", first_line="Split from #9 during triage.")
    write_item(isolated, 4, body="y", first_line="Split from #9 during triage.")
    v = np.array([1.0, 0.0]); w = np.array([0.0, 1.0])   # cosine 0
    pairs = CL.candidate_pairs(_items(), {3: v, 4: w}, threshold=0.75)
    assert pairs and pairs[0]["reasons"] == ["parent"] and pairs[0]["parent"] == 9


def test_one_shared_path_needs_a_near_threshold_cosine(isolated):
    import numpy as np
    write_item(isolated, 1, body="`app/a.py:1`")
    write_item(isolated, 2, body="`app/a.py:5`")
    v = np.array([1.0, 0.0]); w = np.array([0.8, 0.6])   # cosine 0.8
    pairs = CL.candidate_pairs(_items(), {1: v, 2: w}, threshold=0.85)
    assert pairs and pairs[0]["reasons"] == ["paths"] and pairs[0]["cosine"] == 0.8
    pairs = CL.candidate_pairs(_items(), {1: v, 2: w}, threshold=0.95)
    assert pairs == [], "0.8 is below 0.95 - 0.10"


def test_a_giant_component_is_peeled_into_hub_groups_not_trimmed(isolated):
    """The first cut kept the twelve highest-degree nodes and listed the
    rest as overflow, which dropped most of the live board from the night's
    output. Every item with an edge lands in some group."""
    for i in range(1, 16):
        write_item(isolated, i, body="`app/x.py` `app/y.py`")
    data = CL.build_clusters(items=_items(), vecs={}, judge=False, persist_parents_=False)
    sizes = sorted(len(c["item_ids"]) for c in data["clusters"])
    assert sizes == [3, CL.MAX_CLUSTER_SIZE]
    assert sorted(i for c in data["clusters"] for i in c["item_ids"]) == list(range(1, 16))
    big = max(data["clusters"], key=lambda c: len(c["item_ids"]))
    assert set(big["anchor_paths"]) == {"x.py", "y.py"}


def test_peeling_keeps_a_hubs_tight_neighbourhood_together():
    # Hub 1 has strong edges to 2..4 and weak edges to 5..7; 5..7 are strong
    # among themselves. With max_size 4 the strong four come out together.
    pairs = []
    for j in (2, 3, 4):
        pairs.append({"a": 1, "b": j, "cosine": 0.9, "reasons": ["cosine"], "shared_paths": []})
    for j in (5, 6, 7):
        pairs.append({"a": 1, "b": j, "cosine": 0.1, "reasons": ["parent"], "shared_paths": []})
    for a, b in ((5, 6), (6, 7), (5, 7)):
        pairs.append({"a": a, "b": b, "cosine": 0.95, "reasons": ["cosine"], "shared_paths": []})
    groups = CL.peel(pairs, max_size=4)
    assert groups == [{1, 2, 3, 4}, {5, 6, 7}]


def test_a_pair_is_a_cluster_and_a_singleton_is_not(isolated):
    """A pair of duplicates is the most valuable cluster there is: one
    group-triage turn closes one of them with certainty (#795/#799)."""
    write_item(isolated, 1, body="`app/x.py` `app/y.py`")
    write_item(isolated, 2, body="`app/x.py` `app/y.py`")
    write_item(isolated, 3, body="nothing shared")
    data = CL.build_clusters(items=_items(), vecs={}, judge=False, persist_parents_=False)
    assert [c["item_ids"] for c in data["clusters"]] == [[1, 2]] and data["pairs_candidate"] == 1


def test_cluster_id_is_stable_under_membership_order():
    assert CL.cluster_id([3, 1, 2]) == CL.cluster_id((1, 2, 3)) == CL.cluster_id(["2", 3, 1])


# ── who is clusterable ─────────────────────────────────────────────────────

def test_items_with_a_group_umbrellas_needs_human_expired_and_triaged_are_never_clustered(isolated):
    write_item(isolated, 1)
    p = write_item(isolated, 2)
    B.update_frontmatter(p, {"group": 50})
    write_item(isolated, 3, tags=("backlog", "umbrella"))
    write_item(isolated, 4, tags=("backlog", B.NEEDS_HUMAN_TAG))
    write_item(isolated, 5, tags=("backlog", B.EXPIRED_TAG))
    write_item(isolated, 6)
    S.append_event({"event": "backlog_triage", "item_id": 6, "verdict": "stale"}, path=S.LEDGER_PATH)
    write_item(isolated, 7, status="up_next")
    assert [i.id for i in _items()] == [1]


def test_quarantined_self_spawned_items_are_clusterable(isolated):
    """The deliberate exception to test_backlog_spawn_loop: quarantine asks
    'is this stale?', clustering asks 'is this the same work?'."""
    write_item(isolated, 1, days_old=0, tags=("backlog", "spawned-by-triage"))
    assert B.select_candidate(S.LEDGER_PATH) is None
    assert [i.id for i in _items()] == [1]


# ── the judge ──────────────────────────────────────────────────────────────

def _three_cosine_only(isolated):
    import numpy as np
    for i in (1, 2, 3):
        write_item(isolated, i, body=f"item {i}")
    v = np.array([1.0, 0.0]); w = np.array([0.96, 0.28]); u = np.array([0.94, 0.34])
    return _items(), {1: v, 2: w, 3: u}   # pairwise cosines ~0.96/0.94/0.998


def test_judge_distinct_drops_the_edge_and_an_error_keeps_it(isolated):
    items, vecs = _three_cosine_only(isolated)

    # An error first: nothing is cached, every edge stands.
    def broken(a, b, pair):
        return {"error": "connection refused"}
    data = CL.build_clusters(items=items, vecs=vecs, judge=True, judge_fn=broken,
                             persist_parents_=False, threshold=0.93)
    assert data["judge_errors"] == 2 and len(data["clusters"]) == 1
    assert CL.load_judgments() == {}, "an error is not a verdict and is not cached"

    calls = []

    def judge(a, b, pair):
        calls.append((a.id, b.id))
        return {"verdict": "distinct", "confidence": 0.9, "reason": "no"}
    data = CL.build_clusters(items=items, vecs=vecs, judge=True, judge_fn=judge,
                             persist_parents_=False, threshold=0.93)
    # (2,3) at 0.998 is not ambiguous and is never judged; the other two are.
    assert data["pairs_judged"] == 2 and sorted(calls) == [(1, 2), (1, 3)]
    assert [c["item_ids"] for c in data["clusters"]] == [[2, 3]], "the unjudged strong pair stands"


def test_strong_edges_are_not_judged(isolated):
    items, vecs = _three_cosine_only(isolated)
    seen = []
    data = CL.build_clusters(items=items, vecs=vecs, judge=True,
                             judge_fn=lambda a, b, p: seen.append(1) or {"verdict": "same", "confidence": 1, "reason": ""},
                             persist_parents_=False, threshold=0.80)
    assert seen == [] and data["pairs_judged"] == 0 and len(data["clusters"]) == 1


def test_judgments_are_cached_by_pair_and_body_hash(isolated):
    items, vecs = _three_cosine_only(isolated)
    n = {"calls": 0}

    def judge(a, b, pair):
        n["calls"] += 1
        return {"verdict": "same", "confidence": 1.0, "reason": "twin"}
    CL.build_clusters(items=items, vecs=vecs, judge=True, judge_fn=judge, persist_parents_=False, threshold=0.93)
    data = CL.build_clusters(items=items, vecs=vecs, judge=True, judge_fn=judge, persist_parents_=False, threshold=0.93)
    assert n["calls"] == 2 and data["judge_cached"] == 2
    assert sorted(map(sorted, data["clusters"][0]["duplicates"])) == [[1, 2], [1, 3]]


def test_judge_pair_parses_the_model_and_fails_soft():
    import io
    a = B.Item(path=Path("a"), id=1, name="A", status="draft", priority="m", created="", body="a")
    b = B.Item(path=Path("b"), id=2, name="B", status="draft", priority="m", created="", body="b")

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    body = json.dumps({"choices": [{"message": {"content": json.dumps(
        {"verdict": "Related", "confidence": "0.7", "reason": "r"})}}]}).encode()
    out = CL.judge_pair(a, b, {"cosine": 0.8}, endpoint="http://x", model="m",
                        urlopen=lambda req, timeout: Resp(body))
    assert out == {"verdict": "related", "confidence": 0.7, "reason": "r"}

    def boom(req, timeout):
        raise OSError("down")
    assert "error" in CL.judge_pair(a, b, {}, endpoint="http://x", model="m", urlopen=boom)


# ── output ─────────────────────────────────────────────────────────────────

def test_write_clusters_is_atomic_and_load_tolerates_a_missing_file(isolated, tmp_path):
    assert CL.load_clusters() == {}
    data = CL.build_clusters(items=[], vecs={}, judge=False, persist_parents_=False)
    p = CL.write_clusters(data)
    assert p == CL.CLUSTERS_PATH and not list(tmp_path.glob("*.tmp"))
    assert CL.load_clusters()["schema"] == CL.SCHEMA and CL.load_clusters()["clusters"] == []
    p.write_text("{not json")
    assert CL.load_clusters() == {}


def test_missing_vectors_degrade_to_paths_and_parents(isolated):
    write_item(isolated, 1, body="`app/x.py` `app/y.py`")
    write_item(isolated, 2, body="`app/x.py` `app/y.py`")
    write_item(isolated, 3, body="`app/x.py` `app/y.py`")
    data = CL.build_clusters(items=_items(), judge=False, persist_parents_=False)
    assert data["vectors_found"] == 0 and len(data["clusters"]) == 1


def test_the_round_cli_has_the_cluster_subcommand():
    import inspect
    from scripts.automod import round as R
    src = inspect.getsource(R.main)
    assert 'sub.add_parser("cluster"' in src and "CL.main" in src


# ── the nightly source ─────────────────────────────────────────────────────

class _Queue:
    def __init__(self):
        self.rows = []

    def enqueue(self, **kw):
        self.rows.append(kw)
        return len(self.rows)


class _Item:
    def __init__(self, payload=None):
        self.payload = payload or {}


def test_nightly_source_skips_when_clusters_json_is_fresh(isolated):
    from workers.sources import backlog_cluster as SRC
    q = _Queue()
    asyncio.run(SRC.enqueue_if_due(q, {"min_age_seconds": 72000}))
    assert len(q.rows) == 1 and q.rows[0]["dedup_key"] == SRC.DEDUP_KEY
    CL.write_clusters(CL.build_clusters(items=[], vecs={}, judge=False, persist_parents_=False))
    asyncio.run(SRC.enqueue_if_due(q, {"min_age_seconds": 72000}))
    assert len(q.rows) == 1, "a fresh file means no second run"
    asyncio.run(SRC.enqueue_if_due(q, {"min_age_seconds": 0}))
    assert len(q.rows) == 2


def test_nightly_source_runs_off_the_loop_thread_and_records_the_event(isolated):
    import inspect
    from workers.sources import backlog_cluster as SRC
    assert "to_thread" in inspect.getsource(SRC.execute)
    write_item(isolated, 1, body="`app/x.py` `app/y.py`")
    write_item(isolated, 2, body="`app/x.py` `app/y.py`")
    write_item(isolated, 3, body="`app/x.py` `app/y.py`")
    out = asyncio.run(SRC.execute(_Item({"judge": False})))
    assert out["status"] == "success" and "1 clusters over 3 items" in out["summary"]
    assert "no vectors available" in out["summary"]
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["event"] == "backlog_cluster" and ev["clusters"] == 1 and ev["items"] == 3
    assert CL.load_clusters()["clusters"][0]["item_ids"] == [1, 2, 3]


def test_the_source_is_registered_with_the_pool_interface():
    import workers.sources as sources_pkg
    mod = sources_pkg.SOURCE_REGISTRY["backlog-cluster"]
    assert mod.NAME == "backlog-cluster" and not getattr(mod, "LONG_LIVED", False)
