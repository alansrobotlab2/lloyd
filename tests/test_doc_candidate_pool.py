"""#504 — the document leg's *pre-ranking* pool must be able to contain the answer.

`kg-maintenance-tasks` ("which autonomy tasks maintain the knowledge graph?")
expects five autonomy task files and has scored `doc_hit: false` /
`first_doc_rank: null` in every eval artifact since it was written. The triage that
confirmed it proved the miss is candidate generation, not ranking: the same daemon
returns those files at ranks 2, 3, 6, 8 and 20 for a name-shaped query, and returns
none of them for this query at any depth the client can consume.

The mechanism is the unrestricted search (`_qmd_daemon_search`). It asked qmd for a
**global** top-k — `collections: []` — whose ANN width qmd derives from `limit`
(`annVecScan(db, embedding, limit * 3)`). That is not "top-k of each collection", it
is "top-k of every chunk in the index, whichever collection owns it", and
`subliminal` re-indexes the whole vault, so it spends most of that k on copies of
documents the reply already carries. `_qmd_normalize_global` then discards every row
whose head is not a vault segment. Measured on the pre-fix tree, live daemon, 2026-09-14:
`_qmd_daemon_search(KG_QUERY, 20, VAULT_SEGMENTS)` — the eval's own ask — answered 36
rows, 21 distinct paths, **17 of them inside a vault segment**, and none of the five
task files among those 17.

Naming the segments switches qmd to its per-collection path, where each named
collection is scanned for its own quota before fusion (`ftsLimit = limit * 10`, and
an exact vector scan under 20k vectors per collection) — the per-collection fairness
#1005 asked for, so a 242-file collection stops competing against the whole index for
slots. It also makes `limit` mean the *pool* rather than the answer, which has to be
said to the reranker explicitly: qmd caps its own ranking window at
`RERANK_CANDIDATE_LIMIT = 40`, applied as `fused.slice(0, candidateLimit)` *before*
it reranks, so a deep ask that leaves the default still only ever ranks 40
candidates. Recorded against the live daemon, named segments, pool 240, rerank on,
the same wire text the client sends: the five files enter the reply at ranks 14, 17,
35, 39 and 175. Zero of the five at a pool of 40.

The rows below are that recorded arm, in reply order, with the task files at their
observed ranks. What is under test there is everything downstream of the daemon's
answer — the depth of the request, the pool handed to the final ranker, the fold back
onto the segment view, and the slice in between. Those tests can only fail on a
client-side cut, because the reply they assert against is one this file wrote. Two
things hold the daemon-side half, and neither is a unit test:

  - `test_the_live_daemon_hands_the_five_task_files_to_the_pre_rank_pool`, at the end
    of the file, POSTs the real payload through the real `_vault_recall` to the index
    production queries and asserts the same five paths enter the pre-rank pool. Run by
    hand (2026-09-14): 235-row pool, the five at ranks 14, 17, 35, 39 and 175 — the
    recording's own numbers. It is marked `live_vault` because a round under test does
    not control that corpus, and the gate runs `-m "not live_vault"`, so this file is
    NOT the thing that stops a retrieval regression in review.
  - The thing that does is the eval: `doc_hit_rate`, `mrr_doc`, `ndcg10` and
    `doc_recall_avg` are ARMED metrics of the paired promotion comparison
    (`workers/sources/automod_regression.py:101`), measured on a pinned corpus on
    every promotion, and `kg-maintenance-tasks` is one of its 20 queries. A daemon
    that stops handing those five files to the pool moves `doc_hit_rate` 1.00 -> 0.95
    and a promotion fails on it. That is where retrieval is gated; this file gates the
    client, which is the half the eval cannot localise.
"""
import urllib.error

import pytest

from agent_mcp import vault

# The eval query verbatim, from eval/vault_recall_queries.yaml — the string
# `_vault_recall` is asked to answer as `kg-maintenance-tasks`.
KG_QUERY = "which autonomy tasks maintain the knowledge graph?"

# The five files that query expects, each at the rank it occupies in the recorded
# 240-deep named-collection reply (measured 2026-09-14, live daemon, both search legs
# on, rerank on). `autonomy/67-` at 175 is what makes the pool depth load-bearing:
# any pool narrower than ~180 loses it again, silently.
RECORDED_RANKS = {
    "autonomy/24-data-pipeline.md": 14,
    "autonomy/48-entity-resolution-sweep.md": 17,
    "autonomy/74-kg-mention-classifier.md": 35,
    "autonomy/51-conversation-relation-linking.md": 39,
    "autonomy/67-semantic-entity-resolution.md": 175,
}
EXPECTED_TASKS = list(RECORDED_RANKS)
RECORDED_REPLY_DEPTH = 240


def _snapshot_reply(depth=RECORDED_REPLY_DEPTH):
    """The recorded reply in order: filler documents with the task files slotted at
    their measured ranks. One row per path, scores descending, so a fold or a slice
    that treats a deep row differently from a shallow one is exercised.
    """
    by_rank = {rank: path for path, rank in RECORDED_RANKS.items()}
    filler_heads = ["memory", "knowledge", "backlog", "skills", "architecture",
                    "projects", "lloyd", "work", "people", "personal"]
    rows = []
    for rank in range(1, depth + 1):
        path = by_rank.get(rank) or (
            f"{filler_heads[rank % len(filler_heads)]}/filler-{rank:03d}.md")
        rows.append({
            "file": f"qmd://{path}",
            "title": path,
            "snippet": "kg maintenance",
            "score": round(max(0.01, 1.0 - rank * 0.004), 4),
        })
    return rows



@pytest.fixture
def collection_fusion(monkeypatch):
    """The request #504 measured: one ranked list per collection, a 240-row pool.

    Since 2026-09-19 the recall's default is global fusion over a 40-row pool
    (`vault.RECALL_QMD_FUSION`, with the eval that moved it), and
    `RECALL_QMD_FUSION = "collection"` is the kill switch that restores this
    request exactly. The recorded ranks below are per-collection fusion's — 30 to
    227 — and mean nothing under a global ranking, so the tests that replay them
    pin the path they were recorded on: the switch has to keep delivering what
    #504 bought, or it is not a way back. tests/test_recall_global_fusion.py pins
    the default.
    """
    monkeypatch.setattr(vault, "RECALL_QMD_FUSION", "collection")

@pytest.fixture
def wire(monkeypatch):
    """Capture every payload the client sends; answer with the recorded reply."""
    seen = []

    def fake_post(payload):
        seen.append(payload)
        return _snapshot_reply()

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    return seen


def _paths_seen_by_the_demoter(monkeypatch):
    """Record the pool in the config production actually runs.

    `RECALL_GRAPH_RERANK` is False, so `_graph_rerank` is not the last function to
    see every candidate there: with `demote_daily_logs` on (the default) that is the
    `_DAILY_LOG_RE.search` inside the demote loop, which runs once per document in
    the pool before the sort and the `[:limit]` slice. Spying its pattern is the
    only seam onto the pre-ranking pool in that config.
    """
    seen: list = []

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def search(self, text, *a, **kw):
            seen.append(text or "")
            return self._inner.search(text, *a, **kw)

    monkeypatch.setattr(vault, "_DAILY_LOG_RE", Spy(vault._DAILY_LOG_RE))
    return seen


# ── The request: name the segments, ask for a pool, widen the rerank window ───

def test_unrestricted_search_names_the_segments_it_is_willing_to_keep(wire):
    """The request may not ask for rows the client is going to discard.

    The old shape sent `collections: []` — one global scan across all fifteen
    collections, including `sessions` (629 files), `autonomy-runs` (4,339) and
    `skill-cards` (189), every one of which `_qmd_normalize_global` throws away.
    Those rows are not merely waste: they occupy slots in a top-k whose width is
    derived from `limit`, which is how the documents of a 242-file collection get
    pushed out of the reply while its *neighbours* (`autonomy/39-`) stay in.
    """
    vault._qmd_daemon_search(KG_QUERY, 20, list(vault.VAULT_SEGMENTS))
    assert wire[0]["collections"] == list(vault.VAULT_SEGMENTS), (
        "an unrestricted search must name the segments it filters to, so qmd scans "
        "each one for its own quota instead of spending a global top-k on collections "
        "the client discards"
    )


def test_unrestricted_search_over_asks_and_widens_the_rerank_window_with_it(wire):
    """`limit` is the size of the answer; the rerank pool is a separate number.

    Both halves had to move together. The ask has to be deeper than the answer (at a
    40-row pool none of the five arrive; at 240 all five do, ranks 14-175), and qmd's
    `candidateLimit` has to be raised with it — `fused.slice(0, candidateLimit)` runs
    before the rerank, at a default of 40. The recorded arm that asked deep and left
    the default at 40 returned 0 of the 5: a deep reply whose extra depth was never
    ranked is the same miss with a bigger payload.
    """
    vault._qmd_daemon_search(KG_QUERY, 20, list(vault.VAULT_SEGMENTS))
    asked = wire[0]["limit"]
    assert asked >= 20 * 3, (
        f"asked for {asked} rows to answer a 20-document question; the recorded ranks "
        "put the five expected files between 14 and 175")
    assert wire[0].get("candidateLimit") == asked, (
        "the rerank window has to be as wide as the ask, or qmd re-ranks only its "
        f"default 40 candidates and the other {asked - 40} rows are never ranked")


def test_the_pool_is_bounded_so_a_small_ask_stays_cheap(wire):
    """The pool is a ceiling, not a multiplier that runs away.

    `_lookup_entity_facts` calls this same function six times with `limit=2` on a
    recall that opts into the graph lookup, and `vault_search` calls it with
    `max_results=10`. Neither is asking for a 240-row pool. The width is bounded, and
    a small caller still gets a pool a few times the size of its answer.
    """
    vault._qmd_daemon_search("entity resolution", 2, list(vault.VAULT_SEGMENTS))
    small = wire[-1]["limit"]
    # The WIDTH, as a number. `small == 2 * QMD_POOL_FACTOR` (the form here before)
    # only failed if the factor stopped applying: it stayed green for any factor, so
    # it described the rule and pinned no pool. 6 rows is what a `limit=2` caller
    # actually gets and what the paired latency measurement priced (+2 ms per call,
    # six calls per entity). Changing the pool a hot caller pays for is now an edit
    # that has to happen twice.
    assert small == 6, (
        f"a limit=2 caller (`_lookup_entity_facts` calls it six times per entity) got "
        f"a {small}-row ask, not 6: the pool is 3x the answer, capped at 240")
    assert small == 2 * vault.QMD_POOL_FACTOR, (
        f"a limit-2 caller got a {small}-row ask; the pool is QMD_POOL_FACTOR "
        f"(={vault.QMD_POOL_FACTOR}) times the answer, so a small caller gets a pool "
        "rather than only its answer")
    assert 2 < small < vault.QMD_POOL_MAX, (
        f"{small} is not strictly between the answer (2) and the ceiling "
        f"({vault.QMD_POOL_MAX}): if the factor reached the ceiling then `min` clamps "
        "every ask in the tree and QMD_POOL_FACTOR is decorative")

    # The ceiling, falsified by an input above it rather than by one sitting on it.
    vault._qmd_daemon_search("some vault question", 4 * vault.QMD_POOL_MAX,
                             list(vault.VAULT_SEGMENTS))
    assert wire[-1]["limit"] == 240, (
        f"a caller asking for {4 * vault.QMD_POOL_MAX} rows got "
        f"{wire[-1]['limit']}, not the 240-row ceiling: above the cap the factor must "
        "stop multiplying, or QMD_POOL_MAX bounds nothing")
    assert wire[-1]["limit"] == vault.QMD_POOL_MAX, (
        "the 240 written here and QMD_POOL_MAX disagree; the pool the recall doc leg "
        "asks for is the number a human reads in this file")
    # And the reachable version of the same ask — `RECALL_DOC_POOL` is exactly
    # `QMD_POOL_MAX`, which is what `_vault_recall`'s doc leg hands this function.
    vault._qmd_daemon_search("some vault question", vault.QMD_POOL_MAX,
                             list(vault.VAULT_SEGMENTS))
    assert wire[-1]["limit"] == vault.QMD_POOL_MAX, (
        "the deep ask is a fixed pool: a caller already asking for the whole pool must "
        "not have the factor applied on top of it")


def test_scope_restricted_search_keeps_its_own_shape(wire):
    """Unchanged: a deliberate scope names its subset and its own depth.

    A caller that scoped to `backlog` is asking for exactly the hits a wide scan might
    drop, and it gets `[:limit]` back with no fold — so the pool belongs to the
    unrestricted path only. `prefetch` runs nine collections (including `sessions`)
    through this function on the latency-critical turn path and must not inherit it.
    """
    vault._qmd_daemon_search("guardian rollback", 10, ["backlog", "architecture"])
    assert wire[0]["collections"] == ["backlog", "architecture"]
    assert wire[0]["limit"] == 10
    # Direct index, no default: `.get("candidateLimit", 10) == 10` was satisfied by
    # the key being ABSENT, which is the one thing this asserts against — an absent
    # key hands the width back to the daemon's default 40, wider than the answer.
    assert "candidateLimit" in wire[0], (
        "the rerank window is stated explicitly on every arm, restricted included; a "
        "payload that omits it lets the daemon's default decide the width")
    assert wire[0]["candidateLimit"] == 10, (
        f"a restricted search asked to rank {wire[0]['candidateLimit']} candidates for "
        "a 10-document answer; its pool is its depth, so prefetch does not inherit the "
        "widened ask")


def test_the_recall_doc_leg_asks_for_the_whole_pool(wire, collection_fusion):
    """The leg that feeds the final ranker is the one that must ask deep.

    `QMD_POOL_FACTOR` on `limit` alone would give this query a 60-row pool, which the
    recording shows holding three of the five. So `_vault_recall` asks for
    `RECALL_DOC_POOL` documents and slices its pre-rank pool to the same number: the
    ask, the fold and the slice are one width, stated once.
    """
    vault._vault_recall({"query": KG_QUERY, "limit": 20, "grep_code": False,
                         "include_facts": False, "expand_graph": False})
    assert wire[0]["limit"] >= vault.RECALL_DOC_POOL, (
        f"the doc leg asked for {wire[0]['limit']} rows but the pre-rank pool is "
        f"{vault.RECALL_DOC_POOL}: whatever is between the two is unreachable")


# ── The pool: what the final ranker is actually handed ────────────────────────

def test_the_five_task_files_reach_the_pre_rank_candidate_pool(wire, monkeypatch, collection_fusion):
    """Clause 5 of #504, asserted one stage before the ranking that hides it.

    The returned top-10 is the wrong place to assert: `_graph_rerank` can hoist a
    document into view or drop one out, so a top-10 assertion passes and fails for
    reasons that have nothing to do with candidate generation. This asserts on the list
    handed *to* the ranker. Narrow the pool afterwards — revert to a global scan, drop
    the depth, trim the folded reply back to `limit` or to `limit * 3`, or leave
    `candidateLimit` at qmd's default 40 — and `autonomy/67-` at recorded rank 175
    falls out and this fails, instead of the eval quietly reporting
    `first_doc_rank: null` for a fortnight.
    """
    handed = {}

    def spy_rank(documents, seed_entities, weighted_neighbors, alpha=0.5):
        handed["paths"] = [d.get("path") or "" for d in documents]
        return documents

    monkeypatch.setattr(vault, "_graph_rerank", spy_rank)
    out = vault._vault_recall({"query": KG_QUERY, "limit": 20, "grep_code": False,
                               "include_facts": False, "expand_graph": False,
                               "graph_rerank": True})

    pool = handed.get("paths") or []
    missing = [p for p in EXPECTED_TASKS if p not in pool]
    assert not missing, (
        f"{len(missing)} of the five KG-maintenance task files never reached the "
        f"pre-ranking candidate pool ({len(pool)} rows): {missing}"
    )
    assert len(pool) == vault.RECALL_DOC_POOL, (
        f"pre-rank pool is {len(pool)} rows, not the {vault.RECALL_DOC_POOL} this "
        f"query needs: `autonomy/67-semantic-entity-resolution.md` arrives at rank "
        f"{RECORDED_RANKS['autonomy/67-semantic-entity-resolution.md']}, so any pool "
        "narrower than that silently loses it again — which is the failure mode this "
        "test exists to catch, and the `// 2` floor it had before accepted 120"
    )
    assert len(pool) > RECORDED_RANKS["autonomy/67-semantic-entity-resolution.md"], (
        "the pool is shallower than the deepest expected file, so that file cannot be "
        "in it regardless of the reply"
    )
    assert out["documents"], "the ranker still has to return an answer"


def test_the_production_config_hands_the_same_pool_to_its_final_slice(
        wire, monkeypatch, collection_fusion):
    """The same property in the config the eval runs, where nothing is called
    `_graph_rerank`.

    With `RECALL_GRAPH_RERANK` False the final ranking is demote -> sort ->
    `[:limit]`, so the candidate set is whatever reaches the demote loop. It must be
    the same width as the rerank arm's, or the fix would only exist in a config
    nobody runs.
    """
    seen = _paths_seen_by_the_demoter(monkeypatch)
    out = vault._vault_recall({"query": KG_QUERY, "limit": 20, "grep_code": False,
                               "include_facts": False, "expand_graph": False})
    assert vault.RECALL_GRAPH_RERANK is False, (
        "RECALL_GRAPH_RERANK moved: this arm and the one above are now the same test")
    missing = [p for p in EXPECTED_TASKS if p not in seen]
    assert not missing, (
        f"{len(missing)} of the five expected files never reached the pre-slice pool "
        f"({len(seen)} rows) in the default config: {missing}"
    )
    assert len(out["documents"]) <= 20, "the answer is still `limit` documents"


def test_the_fold_still_drops_rows_outside_the_segment_view(monkeypatch):
    """Naming collections is not the same as trusting the reply.

    The discard rule stays, and stays exercised: `sessions/` and `skill-cards/` rows
    turn up whenever the daemon fans out for another caller's reason, and the point of
    this change is that the vault view never contains them. The discarded
    `skill-cards/autonomy-data-pipeline.md` card is the stand-in that used to be the
    closest thing this query could return for `autonomy/24-data-pipeline.md`.
    """
    def fake_post(payload):
        return [
            {"file": "qmd://sessions/2026-09-08/leak.md", "title": "s",
             "snippet": "", "score": 2.0},
            {"file": "qmd://skill-cards/autonomy-data-pipeline.md", "title": "c",
             "snippet": "", "score": 1.5},
            {"file": "qmd://autonomy/24-data-pipeline.md", "title": "t",
             "snippet": "", "score": 1.0},
        ]

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    out = vault._qmd_daemon_search(KG_QUERY, 20, list(vault.VAULT_SEGMENTS))
    assert [r["file"] for r in out] == ["qmd://autonomy/24-data-pipeline.md"]


def test_a_daemon_that_does_not_answer_is_still_not_a_zero_hit():
    """The outage contract survives a wider payload.

    A wider request is more surface, so this pins the one property that must hold for
    any payload change: no answer is an exception, never an empty list.
    """
    def boom(payload):
        raise urllib.error.URLError("connection refused")

    original = vault._qmd_post
    vault._qmd_post = boom
    try:
        with pytest.raises(vault.QmdUnavailable):
            vault._qmd_daemon_search(KG_QUERY, 20, list(vault.VAULT_SEGMENTS))
    finally:
        vault._qmd_post = original


# The five expectations exactly as `eval/vault_recall_queries.yaml` writes them for
# `kg-maintenance-tasks`: prefixes rather than filenames, so the assertion below is
# the eval's own contract and not a transcription of it.
EVAL_DOC_PREFIXES = ("autonomy/24-", "autonomy/48-", "autonomy/51-",
                     "autonomy/67-", "autonomy/74-")


@pytest.mark.live_vault
def test_the_live_daemon_hands_the_five_task_files_to_the_pre_rank_pool(monkeypatch, collection_fusion):
    """The half every test above cannot check: the daemon, over the real seam.

    Each assertion above answers a reply this file wrote, so the only failure they can
    report is a client-side cut. This one posts the payload `_vault_recall` really
    builds — `VAULT_SEGMENTS` named, `limit` and `candidateLimit` both at
    `RECALL_DOC_POOL` (240), rerank on — to the index on :8181, and asserts the five
    files reach the pool the final ranker is handed. It is the only assertion here that
    can fail because retrieval, rather than the client, stopped returning them.

    Marked `live_vault`, which the automod gate excludes with `-m "not live_vault"`: a
    round under test does not control this corpus, and nightly jobs rewrite it. Not
    skip-guarded on the daemon being up — same reading as `test_conversation_relations`
    — because an unreachable daemon is exactly the outage this test exists to catch.
    Run by hand against the corpus production is serving; cite the run in the round
    report, do not count the gate as having run it.
    """
    payloads: list = []
    real_post = vault._qmd_post

    def spy_post(payload):
        payloads.append(payload)
        return real_post(payload)          # straight through: the POST is the point

    monkeypatch.setattr(vault, "_qmd_post", spy_post)
    seen = _paths_seen_by_the_demoter(monkeypatch)

    out = vault._vault_recall({"query": KG_QUERY, "limit": 20, "grep_code": False,
                               "include_facts": False, "expand_graph": False})

    assert payloads, "no query POST reached the qmd daemon on :8181"
    doc_leg = payloads[0]
    assert doc_leg["collections"] == list(vault.VAULT_SEGMENTS), (
        "the live doc leg did not name the segments, so it is asking for a global "
        "top-k again — the shape that returned none of the five at any depth")
    assert doc_leg["limit"] >= vault.RECALL_DOC_POOL, (
        f"the live doc leg asked for {doc_leg['limit']} rows, below the "
        f"{vault.RECALL_DOC_POOL}-row pool the recorded ranks require")
    assert doc_leg.get("candidateLimit") == doc_leg["limit"], (
        "the live ask did not widen qmd's own rerank window with it, so rows past "
        "the daemon's default 40 candidates come back unranked")

    pool = list(dict.fromkeys(seen))
    assert pool, (
        f"the doc leg POSTed and the pre-rank pool is empty — the daemon answered "
        f"{len(payloads)} request(s) and nothing survived the fold")
    missing = [p for p in EVAL_DOC_PREFIXES
               if not any(path.startswith(p) for path in pool)]
    assert not missing, (
        f"{len(missing)} of the five expectations never reached the pre-rank pool of "
        f"{len(pool)} real rows (pool width {vault.RECALL_DOC_POOL}): {missing}"
    )
    assert out["documents"], "the query still has to return an answer"


@pytest.mark.live_vault
def test_the_eval_scorer_itself_reports_the_target_query_hitting():
    """The claim of #504, asserted at the scorer rather than at the pool.

    Everything above asserts a *pool*, which is the mechanism; the acceptance is what
    `eval/run_eval.py` reports. So this runs the eval's own `_score` over the eval's
    own yaml entry for `kg-maintenance-tasks` — the expectations are read from the
    file, not copied into this test, which is what keeps clause 4 honest: it cannot
    drift into matching what the retriever happens to return, because it asks the
    retriever and then lets the yaml decide.

    Marked `live_vault` and therefore excluded from the gate run: it POSTs to the
    daemon on :8181 against a corpus the nightly jobs rewrite. Run by hand and cited
    in the round report. `ndcg10` is deliberately NOT asserted: `_ndcg_at_k` scores
    only the ten returned rows and the first expected file lands at rank 17, so this
    record reads `ndcg10: 0.0` while `doc_hit` is true — see the finding appended to
    #504. `doc_hit` and `first_doc_rank` are the two numbers this item is about.
    """
    import sys
    from pathlib import Path

    import yaml

    root = Path(vault.__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from eval.run_eval import _score

    specs = yaml.safe_load(
        (root / "eval" / "vault_recall_queries.yaml").read_text())["queries"]
    spec = next(s for s in specs if s.get("id") == "kg-maintenance-tasks")

    out = vault._vault_recall({"query": spec["query"], "limit": 20,
                               "grep_code": False, "include_facts": False,
                               "expand_graph": False})
    scoring = _score(spec, out)

    assert scoring["doc_hit"] is True, (
        f"the eval's scorer still reports doc_hit false for {spec['query']!r} against "
        f"the live corpus — the expectation is unmet, not unmeasured: "
        f"{[d.get('path') for d in out['documents'][:10]]}"
    )
    assert scoring["first_doc_rank"] is not None, (
        "doc_hit true with a null first_doc_rank is a scorer bug, not a pass")
    assert scoring["docs_matched"], (
        "no expected doc matched, so the scorer and the yaml disagree about the query")


# ── #1129: the price that justified this widening is re-stated on fresh queries ──

def test_the_widenings_price_is_stated_as_a_fresh_query_not_a_cached_repeat():
    """The figures behind `RECALL_DOC_POOL` = 240 are fresh, and say which arm.

    `agent_mcp/vault.py` priced the named arm at 110 ms (40-row pool) and 165 ms
    (240-row pool), labelled "rerank on, warm", and that ~55 ms delta is what
    justified asking qmd to cross-encode 240 pairs per recall. The measured cost
    of the same widening is nightly `latency_ms_avg` 707.8 ms → 4,368.4 ms — 6.2x
    — and the reason the comment could read 55 ms is that its samples were
    repeats of one query the daemon had already embedded. `_qmd_daemon_search`'s
    own docstring states the mechanism: the daemon caches query embeddings, so a
    sequential A/B measures arm order, not arm.

    So a price in this file is admissible only if it names the sample kind, and
    the retired cached pair may not return. This is a whole-file check rather than
    a near-the-constant grep on purpose: the figures lived in two places (the
    `RECALL_DOC_POOL` comment and the `_qmd_daemon_search` docstring) and the
    defect was that neither said which sample it was.

    Provenance of the replacement numbers, measured by this round on 2026-09-18
    against the live daemon, named segments, a fresh never-seen query text per
    sample (3 samples per cell): 2,161-2,291 ms at pool 40 rerank-on,
    3,672-4,027 ms at pool 240 rerank-on, 119-183 ms for the identical repeat of
    a query just run at 240, and 150-210 ms at pool 240 with the rerank skipped.
    """
    import inspect
    from pathlib import Path

    src = Path(vault.__file__).read_text(encoding="utf-8")

    for stale in ("110 ms", "140 ms", "165 ms", "577 ms"):
        assert stale not in src, (
            f"{stale} is back in vault.py — that figure was a cached repeat, and "
            "it is the number that made a 6x latency step look like 55 ms")

    doc = inspect.getsource(vault._qmd_daemon_search)
    assert "2,161-2,291 ms FRESH named at a 40-row pool" in doc, (
        "the fresh 40-row price is gone from the seam it prices")
    assert "3,672-4,027 ms FRESH named at" in doc, (
        "the fresh 240-row price is gone from the seam it prices")
    assert "119-183 ms CACHED" in doc, "the cached-repeat counter-sample is gone"
    assert "150-210 ms FRESH rerank-OFF" in doc, "the rerank-OFF price is gone"

    # And the same four figures where the pool constant itself is justified, so a
    # reader who lands on `RECALL_DOC_POOL` and never opens the function gets the
    # fresh numbers too.
    pool_comment = src[src.index('"Measured free"'):src.index("QMD_POOL_FACTOR = 3")]
    # Not `assert pool_comment`: both `src.index()` anchors above raise ValueError
    # before it could run, so a non-empty slice was guaranteed and the assertion
    # could not fail. What must hold instead is that this block points at the real
    # ceiling — the nightly figure has to be the one the owning module currently
    # holds, read from that module rather than transcribed, so the comment cannot
    # cite a budget that was moved or never written.
    from workers.sources import automod_regression as R
    nightly_ceiling = f"{R.LATENCY_BUDGET_MS[R.CONTEXT_NIGHTLY]:,.0f} ms"
    assert nightly_ceiling in pool_comment, (
        f"the block beside RECALL_DOC_POOL no longer names the {nightly_ceiling} "
        "ceiling the budget module holds, so a reader who lands here cannot find "
        "where the price is graded")
    assert "2,161-2,291 ms" in pool_comment and "3,672-4,027 ms" in pool_comment, (
        "the pool comment no longer carries the fresh prices")
    assert "CACHED" in pool_comment, "the pool comment does not name its arm"

    # The seam may not state a duration that reads as a current price without
    # naming its sample kind. Four exact assertions above pin the fresh prices and
    # the cached counter-sample; a general per-line sweep over the whole docstring
    # was tried first and is deliberately NOT here: it also catches the two
    # dated 2026-09-07 pairs and the disowned other-arm history that this change
    # is not entitled to re-measure, and a rule that needs a three-line lookback
    # to tell those apart is not a rule a reader can predict. Extending coverage
    # to those pairs is a finding on #1129, not a clause of it.

    # The two numbers the comment defers to must actually exist as a budget, so
    # the comment cannot cite a ceiling nobody wrote. (`R` is in scope from the
    # ceiling check above.)
    assert set(R.LATENCY_BUDGET_MS) == {R.CONTEXT_NIGHTLY, R.CONTEXT_PAIRED_CHECK}
