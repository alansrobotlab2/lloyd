"""The `djev_rerank` arm on `_vault_recall`, and the shadow seam beside it.

Two things the plan's own first draft got wrong, both pinned here:

  * the arm has to be a PARAMETER of the handler, not a log. The adoption
    question is "does djev's ordering beat qmd's own reranker on the labelled
    set", and a shadow record cannot be scored against labels.
  * the shadow hook has to sit where production's decision is made. Its first
    draft went inside `_graph_rerank`, which is only reached under
    `if graph_rerank:` while `RECALL_GRAPH_RERANK` is False — so it would
    have fired zero times and read as a quiet seam.
"""

from __future__ import annotations


from agent_mcp import vault
from app import djev


DOCS = [{"path": f"p{i}.md", "title": f"t{i}", "snippet": f"s{i}", "score": 1.0 - i / 100}
        for i in range(20)]


# ---------------------------------------------------------------------------
# The arm ships off
# ---------------------------------------------------------------------------

def test_the_arm_is_off_by_default():
    """Flipping this constant IS the adoption decision, and it is a separate
    commit with the eval result in its message."""
    assert vault.RECALL_DJEV_RERANK is False


def test_the_default_top_is_inside_the_safe_window():
    assert vault.RECALL_DJEV_RERANK_TOP <= djev.RANK_MAX_N


def test_the_eval_exposes_the_arm_and_records_it_as_not_production():
    """A djev-reranked baseline must not compare as production's by accident."""
    import eval.run_eval as ev
    ns = ev.build_parser().parse_args(["--djev-rerank"])
    assert ns.djev_rerank is True
    assert ev.build_run_config(ns)["matches_production_defaults"] is False
    assert ev.build_run_config(ns)["djev_rerank"] is True
    off = ev.build_parser().parse_args([])
    assert off.djev_rerank == vault.RECALL_DJEV_RERANK
    assert off.djev_rerank_top == vault.RECALL_DJEV_RERANK_TOP


def test_the_eval_mutes_the_shadow_recorder():
    """A pinned-corpus run recorded as production traffic would poison the
    distribution the floors are read off — and the rerank arm calls
    `_vault_recall`, which is exactly where the lead seam lives."""
    import os
    import eval.run_eval  # noqa: F401 — the import is what sets it
    assert os.environ.get("LLOYD_DJEV_SHADOW") == "0"


def test_the_pinned_corpus_env_mutes_it_too():
    """One place, because every regression arm, noise run and warm-up passes
    through `env_for` — a second copy at each caller is how they drift."""
    from scripts.automod import evalpin
    pin = evalpin.PinnedCorpus.__new__(evalpin.PinnedCorpus)
    pin.overlay = "/tmp/overlay.yaml"
    # `name` because `env_for` now also names the index FILE the daemon was started
    # on (#1374); it read only `overlay` before, so a hand-built pin could leave the
    # rest of the object unset. Set here rather than defaulted inside `env_for`: a
    # silently-defaulted name there is how a genuinely missing name would end up
    # recorded as a corpus identity.
    pin.name = evalpin.PIN_INDEX_NAME
    assert pin.env_for({})["LLOYD_DJEV_SHADOW"] == "0"


# ---------------------------------------------------------------------------
# The arm reorders only the head, and fails open
# ---------------------------------------------------------------------------

def test_the_arm_reorders_only_the_head_slice(monkeypatch):
    """djev is a final-stage reranker over a shortlist and nothing else:
    fanning a 240-row pool across canvas chunks and sorting the union
    produces an artefact that looks exactly like a ranking."""
    monkeypatch.setattr(djev, "rank",
                        lambda q, c, **k: [{"index": i} for i in reversed(range(len(c)))])
    out = vault._djev_rerank_pool(list(DOCS), "q", 5)
    assert [d["path"] for d in out[:5]] == [f"p{i}.md" for i in (4, 3, 2, 1, 0)]
    assert [d["path"] for d in out[5:]] == [d["path"] for d in DOCS[5:]]


def test_none_from_the_client_leaves_the_order_untouched(monkeypatch):
    """`None` means the engine did not answer — or answered across a canvas
    split, which the client refuses to sort. Either way the pool keeps the
    order qmd gave it."""
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)
    assert vault._djev_rerank_pool(list(DOCS), "q", 12) == DOCS


def test_an_exception_in_the_arm_never_fails_the_recall(monkeypatch):
    monkeypatch.setattr(djev, "rank",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert vault._djev_rerank_pool(list(DOCS), "q", 12) == DOCS


def test_the_arm_never_exceeds_the_client_ceiling(monkeypatch):
    """`djev.rank` raises above 16 rather than truncating, so the arm has to
    clamp before calling it or a `--djev-rerank-top 40` would fail every
    query instead of ranking 16."""
    seen = {}
    def _rank(q, c, **k):
        seen["n"] = len(c)
        return [{"index": i} for i in range(len(c))]
    monkeypatch.setattr(djev, "rank", _rank)
    vault._djev_rerank_pool(list(DOCS), "q", 40)
    assert seen["n"] <= djev.RANK_MAX_N


def test_a_pool_too_small_to_reorder_is_left_alone(monkeypatch):
    def _never(*a, **k):  # pragma: no cover
        raise AssertionError("asked djev to rank one candidate")
    monkeypatch.setattr(djev, "rank", _never)
    assert vault._djev_rerank_pool(DOCS[:1], "q", 12) == DOCS[:1]


def test_the_arm_sends_the_text_it_already_holds(monkeypatch):
    """No disk read. The point of ranking a shortlist is that everything it
    needs came back with the pool."""
    seen = {}
    monkeypatch.setattr(djev, "rank",
                        lambda q, c, **k: seen.setdefault("c", c) and None or None)
    vault._djev_rerank_pool(list(DOCS), "q", 3)
    assert seen["c"] == ["t0\ns0", "t1\ns1", "t2\ns2"]


# ---------------------------------------------------------------------------
# The shadow seam
# ---------------------------------------------------------------------------

def test_the_shadow_hook_is_not_inside_the_graph_rerank_branch():
    """The bug the plan caught in its own first draft. `_graph_rerank` is
    dead code in production, so a hook there fires zero times."""
    import inspect
    src = inspect.getsource(vault._graph_rerank)
    assert "djev" not in src


def test_the_shadow_hook_runs_on_the_production_path(monkeypatch):
    """Both `vault_recall` and `memory_ops.recall` pass through the slice,
    and `RECALL_GRAPH_RERANK` is False, so this is the ordering production
    actually returns."""
    import inspect
    src = inspect.getsource(vault._vault_recall)
    assert "_djev_shadow_rerank(documents, query)" in src
    # Above the slice, so `documents` still holds the pool.
    assert src.index("_djev_shadow_rerank") < src.index("documents = documents[:limit]")


def test_the_shadow_hook_records_productions_order(monkeypatch):
    sent = {}
    monkeypatch.setattr("app.djev_shadow.enabled", lambda seam="": True)
    monkeypatch.setattr("app.djev_shadow.shadow",
                        lambda **kw: sent.update(kw))
    vault._djev_shadow_rerank(list(DOCS), "a query")
    assert sent["seam"] == "rerank"
    assert sent["actual"] == [d["path"] for d in DOCS[:djev.RANK_DEFAULT_N]]
    # The state is a CALLABLE, so the concatenation happens on the worker.
    assert callable(sent["state"]) and callable(sent["questions"])
    assert "a query" in sent["state"]()


def test_the_shadow_hook_returns_none_and_swallows_everything(monkeypatch):
    monkeypatch.setattr("app.djev_shadow.enabled",
                        lambda seam="": (_ for _ in ()).throw(RuntimeError("boom")))
    assert vault._djev_shadow_rerank(list(DOCS), "q") is None


def test_the_arm_and_the_shadow_are_exclusive():
    """With the arm ON djev IS the decision, and a row comparing djev's
    ordering against djev's ordering is not an observation."""
    import inspect
    src = inspect.getsource(vault._vault_recall)
    i = src.index("elif djev_rerank:")
    block = src[i:i + 500]
    assert "_djev_rerank_pool" in block and "elif reranker is None:" in block
    assert block.index("_djev_rerank_pool") < block.index("_djev_shadow_rerank")
    # djev AS the ranker (#1336) is decided before either, and excludes both.
    assert src.index('if ranker == "djev":') < i
