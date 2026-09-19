"""agent_mcp.retrieval + the fact tools' ranking behaviour.

There were no tests here at all, which is how `_graph_rerank`'s god-node
penalty stayed a no-op for four months and `fact_profile` kept returning
5,489 facts for one entity.
"""
import asyncio
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import facts as facts_mod  # noqa: E402
from agent_mcp import retrieval, vault  # noqa: E402
from app import kg_store  # noqa: E402


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A temp facts tree + store, wired into every module that reads them."""
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    import agent_mcp._shared as shared
    monkeypatch.setattr(shared, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    st = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, st
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None


def _write_facts(root, entity, category, facts):
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity, "category": category, "facts": facts}
    (d / f"{entity}-{category}.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n")


# ── _graph_rerank ────────────────────────────────────────────────────────────

def test_graph_rerank_penalty_applies_to_a_cased_hub_entity(world):
    """`voters` is keyed lowercase; the degree map was cased, so every lookup
    missed and every voter was divided by log(1+1+e) — the same constant.
    The penalty existed in the code and did nothing."""
    _, st = world
    # `Lloyd` is a hub (many edges); `Alfie` is specific (one edge).
    for i in range(40):
        st.edges.add({"source": "Lloyd", "target": f"N{i}", "type": "mentions"}, origin="t")
    st.edges.add({"source": "Alfie", "target": "Block", "type": "uses"}, origin="t")

    ci = retrieval.get_entity_edge_counts_ci()
    assert ci["lloyd"] == 40 and ci["alfie"] == 1, ci

    docs = [{"path": "a.md", "snippet": "a note about Lloyd", "score": 1.0},
            {"path": "b.md", "snippet": "a note about Alfie", "score": 1.0}]
    out = vault._graph_rerank(list(docs), ["Lloyd", "Alfie"], [], alpha=0.0)
    # The specific entity's document must outrank the hub's.
    assert [d["path"] for d in out] == ["b.md", "a.md"]
    assert out[0]["_topo_score"] > out[1]["_topo_score"]

    # And with the penalty absent, the two are indistinguishable — which is
    # exactly what the cased-map lookup produced for four months.
    import agent_mcp.vault as v
    orig = v._edge_counts_or_empty
    try:
        v._edge_counts_or_empty = lambda ci=False: {}
        flat = v._graph_rerank(list(docs), ["Lloyd", "Alfie"], [], alpha=0.0)
        assert flat[0]["_topo_score"] == flat[1]["_topo_score"]
    finally:
        v._edge_counts_or_empty = orig


def test_graph_rerank_is_a_noop_without_voters(world):
    docs = [{"path": "a.md", "text": "x", "score": 1.0}]
    assert vault._graph_rerank(docs, [], [], alpha=0.3) is docs


def test_production_defaults_are_what_the_eval_imports():
    """The eval ran rerank off / alpha 0.5 while production ran on / 0.3, so
    a retrieval regression could not appear in the nightly numbers.

    `RECALL_SEED_TOP_K` joins the list because the same drift already existed
    one knob over: production seeded 10 entities and the eval scored 5 (#843).
    Importing the constant is the fix; this line is what stops the import being
    dropped and a literal restored."""
    import eval.run_eval as ev
    assert ev.RECALL_GRAPH_RERANK is vault.RECALL_GRAPH_RERANK
    assert ev.RECALL_RERANK_ALPHA == vault.RECALL_RERANK_ALPHA
    assert ev.RECALL_GRAPH_TOP_K == vault.RECALL_GRAPH_TOP_K
    assert ev.RECALL_GRAPH_HOPS == vault.RECALL_GRAPH_HOPS
    assert ev.RECALL_SEED_TOP_K == vault.RECALL_SEED_TOP_K


# ── extract_entities_from_query ──────────────────────────────────────────────

def test_generic_single_words_do_not_win_a_full_name_match(world):
    """`memory`, `graph`, `session` are real entity dirs. A single generic
    word scored 5.0 — the full-name bonus — and pushed the query's actual
    subject out of the top-k."""
    root, _ = world
    for name in ("memory", "graph", "Knowledge Graph Store"):
        _write_facts(root, name, "state", [{"fact": "x", "id": "stat-001"}])
    ranked = dict(retrieval.extract_entities_from_query(
        "how does the knowledge graph store work in memory"))
    assert "Knowledge Graph Store" in ranked
    assert ranked.get("memory", 0) < 5.0
    assert ranked.get("graph", 0) < 5.0
    assert max(ranked, key=ranked.get) == "Knowledge Graph Store"


def test_task_ids_dispatch_to_the_canonical_form(world):
    root, _ = world
    _write_facts(root, "Task #67", "state", [{"fact": "x", "id": "stat-001"}])
    ranked = dict(retrieval.extract_entities_from_query("what happened with task 67"))
    assert ranked["Task #67"] == 10.0


# ── How many query entities become seeds (#843) ──────────────────────────────

SEED_NAMES = [f"seedcap{n}" for n in range(12)]


def _seed_spy(vault_mod, monkeypatch) -> list[list[str]]:
    """Install the spy that lets a test read the seed list `_vault_recall`
    built, and return the list of seed lists it records.

    The width is captured at `graph_weighted_neighbors` — the one call that
    receives the WHOLE seed list (the fact legs slice it per entity). The recall
    result carries facts and neighbours but never the seeds, so this is the only
    call the width can be read off. qmd is stubbed: the document leg needs a
    daemon and these tests are about seeding, not retrieval breadth.

    The caller then invokes the handler however it wants — directly, or through
    `call_tool` / `memory_ops.recall` to cross a process boundary — and reads
    `handed[0]` afterwards."""
    from agent_mcp import _shared as shared

    shared._invalidate_entity_dirs_cache()
    retrieval._entity_index_cache = None
    handed: list[list[str]] = []
    real = vault_mod.graph_weighted_neighbors

    def spy(entities, *a, **k):
        handed.append(list(entities))
        return real(entities, *a, **k)

    monkeypatch.setattr(vault_mod, "graph_weighted_neighbors", spy)
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", lambda *a, **k: [])
    return handed


def _seeds_of(handed: list[list[str]]) -> list[str]:
    assert handed, "_vault_recall seeded nothing, so the width is unmeasured"
    return handed[0]


def _twelve_entity_query(world):
    """Seed a 12-entity fixture and return the query naming all of them — wide
    enough that any plausible budget truncates it, which is the only condition
    under which a seed count measures anything."""
    root, _ = world
    for name in SEED_NAMES:
        _write_facts(root, name, "state", [{"fact": "x", "id": "stat-001"}])
    query = " ".join(SEED_NAMES)
    extracted = len(retrieval.extract_entities_from_query(query))
    assert extracted > vault.RECALL_SEED_TOP_K, (
        f"the fixture extracts {extracted} entities, no more than the width "
        f"{vault.RECALL_SEED_TOP_K}, so no truncation is observable — widen "
        "SEED_NAMES rather than reporting a pass")
    return query


def test_the_recall_path_seeds_at_the_shared_constant(world, monkeypatch):
    """Clause 1. Production sliced the extractor at a literal `[:10]` while the
    eval scored a literal `[:5]` off the same extractor (#843), so every
    `entity_hit_rate` ever published described a seed set nothing served. The
    width now has one name; a query naming 12 extractable entities must hand the
    recall path exactly `RECALL_SEED_TOP_K` of them."""
    query = _twelve_entity_query(world)
    handed = _seed_spy(vault, monkeypatch)
    vault._vault_recall({"query": query, "expand_graph": True})
    assert len(_seeds_of(handed)) == vault.RECALL_SEED_TOP_K


def test_the_seed_width_moves_with_the_constant_it_names(world, monkeypatch):
    """Clause 1, the derivation half. Equality with 10 is not derivation: at a
    retired literal `[:10]` the count assertion above passes just as well, which
    is what the review of the first #843 round caught. Move the constant and the
    width must move with it — a slice bound by a literal does not."""
    query = _twelve_entity_query(world)
    monkeypatch.setattr(vault, "RECALL_SEED_TOP_K", 7)
    handed = _seed_spy(vault, monkeypatch)
    vault._vault_recall({"query": query, "expand_graph": True})
    assert len(_seeds_of(handed)) == 7, (
        "RECALL_SEED_TOP_K says 7 and the recall path seeded otherwise, so the "
        "width is still a literal inside _vault_recall")


@pytest.mark.parametrize("requested", [3, 12],
                         ids=["narrower than production", "wider than production"])
def test_an_explicit_width_is_honoured_both_directions(world, monkeypatch, requested):
    """The keyword is not advisory, and not clamped to production's number.

    The companion tests above only ever leave the width unset or move the
    constant, so a `_vault_recall` that read the keyword and then threw it away —
    seeding at `RECALL_SEED_TOP_K` regardless — passed every one of them, and a
    version that clamped a caller's width down to the constant passed too. Both
    mutations matter here rather than only in the abstract: the eval is the caller,
    and `--seed-top-k 3` against a recall path that silently seeded 10 would put
    the artifact's seeds and retrieval's seeds back a part, which is #843 with a
    different number. The wider case is what a raised constant will need on the
    day it is raised."""
    query = _twelve_entity_query(world)
    ranked = [e for e, _ in retrieval.extract_entities_from_query(query)]
    assert len(ranked) >= max(3, requested), (
        f"the fixture only ranks {len(ranked)} entities, so a width of {requested} "
        "is unmeasurable")
    handed = _seed_spy(vault, monkeypatch)
    vault._vault_recall({"query": query, "expand_graph": True},
                        seed_top_k=requested)
    seeds = _seeds_of(handed)
    assert len(seeds) == requested, (
        f"asked for {requested} seeds and got {len(seeds)}: the keyword reached "
        "retrieval and was not the width it seeded at")
    assert seeds == ranked[:requested], (
        "the count moved but the set is not the ranked prefix every caller "
        "assumes it to be")


async def test_a_client_cannot_move_the_seed_set_through_the_tool_boundary(
        world, monkeypatch):
    """The seam the first #843 round was refused for. `_vault_recall` is
    registered as the `vault_recall` handler and `call_tool` hands it the
    client's raw argument dict; `memory_ops.recall` forwards its params the same
    way (memory_ops.py:105). Reading the width out of `params` would therefore
    have made an undocumented key able to change retrieval — a client that sent
    `seed_top_k: 2` would have gotten 2 seeds where the schema offers no such
    parameter. The width is keyword-only now, so neither boundary may move it."""
    from agent_mcp import memory_ops

    query = _twelve_entity_query(world)
    width = vault.RECALL_SEED_TOP_K

    handed = _seed_spy(vault, monkeypatch)
    await vault.call_tool("vault_recall",
                          {"query": query, "expand_graph": True, "seed_top_k": 2})
    assert len(_seeds_of(handed)) == width, (
        "a stray `seed_top_k` in the MCP arguments moved the seeding width, "
        "which no tool schema declares and no eval asked for")

    handed = _seed_spy(vault, monkeypatch)
    memory_ops.recall({"query": query, "expand_graph": True, "seed_top_k": 2})
    assert len(_seeds_of(handed)) == width, (
        "memory_ops.recall forwards its params untouched, so the width must not "
        "be readable from them")


def test_the_width_is_keyword_only_and_stays_out_of_the_schema():
    """The other side of the same boundary: keyword-only has to stay keyword-only.
    A positional third argument, or a `seed_top_k` key carried in `params`, is
    the shape that would put the knob back on the wire, where it is reachable by
    any client that invents the key."""
    sig = inspect.signature(vault._vault_recall)
    assert sig.parameters["seed_top_k"].kind is inspect.Parameter.KEYWORD_ONLY, (
        "a positional or dict-carried width is an agent-settable retrieval knob "
        "that the vault_recall schema does not declare")
    tool = next(t for t in asyncio.run(vault.list_tools())
                if t.name == "vault_recall")
    # `inputSchema` is the wire name; the mcp SDK's pydantic model exposes it as
    # `input_schema`, and reading the wrong one raises AttributeError rather than
    # returning an empty dict, which is the failure that would fake a pass here.
    assert "seed_top_k" not in tool.input_schema["properties"], (
        "the width must stay out of the tool schema, which is exactly why it is "
        "a keyword argument and not a parameter read from params")


# ── A class name must not consume a query about one instance (#1024) ─────────

ITEM_QUERY = "tell me about backlog item 363"


def _backlog_world(root, *, with_specific=True, other_ids=60):
    """The live shape: one `Task #363` row beside the `Backlog Item`/`Backlog`
    class rows, whose facts are per-record facts about OTHER items.

    The live store files an item's facts under `Backlog Item` as well as under
    `Task #N`, so 168 and 125 facts sit on the class rows and every one of them
    reads `Backlog item #1121 was …` — text that contains the query's own words,
    which is why the god-node token filter keeps 139 of 144. `other_ids` writes
    enough facts to clear FACT_GODNODE_THRESHOLD (50) so that filter runs here
    too. Returns the ids the class rows name.
    """
    named = [str(900 + i) for i in range(other_ids)]
    generic = [{"fact": f"Backlog item #{n} was created on the lloyd board "
                          f"with priority High.", "id": f"g{i:03d}",
                "confidence": 1.0}
               for i, n in enumerate(named)]
    _write_facts(root, "Backlog Item", "activity", generic)
    _write_facts(root, "Backlog", "activity",
                 generic[:40] + [{"fact": "Backlog board has 1200 items.",
                                  "id": "b000", "confidence": 1.0}])
    if with_specific:
        _write_facts(root, "Task #363", "state", [
            {"fact": "Task #363 was created on the lloyd board with Medium priority.",
             "id": "s001", "confidence": 1.0},
            {"fact": "Task #363 implementation plan was reordered to fix a dependency.",
             "id": "s002", "confidence": 1.0},
        ])
    for extra in ("Backlog Task #363", "Issue #363", "#363"):
        _write_facts(root, extra, "state", [
            {"fact": f"{extra} #363 is the queried backlog item.", "id": "x001",
             "confidence": 1.0}])
    return named


def _recall_fact_items(monkeypatch):
    """Run the real `_vault_recall` fact leg with the doc leg silenced.

    The doc leg is a qmd daemon call; empty-list is that function's genuine
    zero-hit answer, not an outage, so the seed-fact leg still runs end to end —
    `seed_entities` → `_collect(…, FACT_GODNODE_THRESHOLD)` → `_rank(…,
    FACT_RANK_CAP_SEED)` — which is the path the clause is about.
    """
    from agent_mcp import vault as vault_mod
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search",
                        lambda *a, **k: [])
    return vault_mod._vault_recall({
        "query": ITEM_QUERY, "limit": 5, "grep_code": False,
        "include_facts": True, "expand_graph": False, "graph_rerank": False})


def test_a_multiword_class_name_is_not_a_full_name_seed(world):
    """`Backlog Item` and `Backlog` are real entity directories, so a query that
    names ONE item full-matched the class name at 5.6/5.35 and every ranked slot
    came from those two rows: the pre-fix probe on 2026-09-18 printed
    `Counter({'Backlog Item': 10})` over a 318-fact pool while `Task #363` seeded
    at 10.0 and won zero slots. Neither class row may be seeded at all when the
    query names an instance — a demoted score would not be enough, because
    `seed_entities` is the top 10 by rank whatever the number."""
    root, _ = world
    _backlog_world(root)
    ranked = dict(retrieval.extract_entities_from_query(ITEM_QUERY))
    assert "Backlog Item" not in ranked, (
        f"the class row was seeded at {ranked.get('Backlog Item')}; the query names"
        " item 363, not the class its facts also sit under")
    assert "Backlog" not in ranked
    above_one = {e: s for e, s in ranked.items() if s > 1.0}
    assert above_one and all("#363" in e or e.endswith("363") for e in above_one), (
        f"seeds scoring above 1.0 must all be the queried instance, got {above_one}")


def test_a_topic_row_survives_a_query_about_one_item(world):
    """The rule must cost nothing it is not fixing. Every-word-generic alone
    matched 144 live entity directories carrying 6,705 facts (both re-measured
    2026-09-18) — including `Vault` (1,736), `Data Pipeline` (1,238) and
    `Knowledge Graph` (621), which have facts about themselves and are what such a
    query is actually asking about.

    So the rule requires a record noun as well as all-generic words, which matches
    12 directories and 363 facts (also re-measured): ten whose name contains
    `backlog` (`Backlog`, `Backlog Item`, `Backlog Board`, `vault-backlog`, …) plus
    `Agent Facts Records` and `Queue Entries`. This pins the half of that boundary
    the class-name test cannot see: a topic whose every word is generic stays
    seeded even while the same query names one item.
    """
    root, _ = world
    _backlog_world(root)
    for name in ("Knowledge Graph", "Data Pipeline"):
        _write_facts(root, name, "state", [{"fact": "x", "id": "stat-001"}])
    ranked = dict(retrieval.extract_entities_from_query(
        "how does the knowledge graph build the data pipeline for backlog item 363"))
    for topic in ("Knowledge Graph", "Data Pipeline"):
        assert topic in ranked, (
            f"{topic} was suppressed by the class-name rule although it names no"
            " record class")
    assert retrieval._is_generic_seed_name("knowledge graph") is False
    assert retrieval._is_generic_seed_name("backlog item") is True
    assert retrieval._is_generic_seed_name("task #363") is False, (
        "a name carrying the queried id must never be class-suppressed")


def test_a_class_row_never_ranks_above_the_instance_it_describes(world, monkeypatch):
    """The slot-occupancy clause: with the class rows and the specific row in the
    same fixture, the specific row wins strictly more of FACT_RANK_CAP_SEED than
    any class-named seed, and no ranked fact names an item other than 363.

    Before the change this printed `{'Backlog Item': 10}` on the live store with
    10 of 10 facts naming a different item id, all tied at fact_score 0.500
    against the specific row's best 0.333 — a per-entity cap could not fix that,
    since 5 slots for each class row still leaves the instance zero.
    """
    root, _ = world
    named = _backlog_world(root)
    facts = _recall_fact_items(monkeypatch)["facts"]
    assert facts, "the queried item has facts; an empty answer is not the fix"
    owners = Counter(f["entity"] for f in facts)
    for class_row in ("Backlog Item", "Backlog"):
        assert owners[class_row] < owners["Task #363"], (
            f"{class_row} held {owners[class_row]} of {len(facts)} slots against "
            f"Task #363's {owners['Task #363']}: the class row outranks the item "
            f"the query names (all owners: {dict(owners)})")
    wrong = [f for f in facts if any(f"#{n}" in f["fact"] for n in named)]
    assert not wrong, (
        f"{len(wrong)} of {len(facts)} ranked facts name an item id other than 363")


async def test_the_mcp_tool_boundary_delivers_no_other_item_facts(world, monkeypatch):
    """The process boundary the agent actually crosses: the MCP aggregator calls
    `vault.call_tool`, which runs the handler on a worker thread
    (`asyncio.to_thread`) and serialises the answer through `_wrap`'s single
    `json.dumps`. A suppression that only survived a direct in-process call to
    `_vault_recall` would still ship other items' facts to the caller, so this
    asserts on the deserialised MCP payload, not on the handler's dict.

    Same fixture and same requirement as the slot-occupancy test above.
    """
    root, _ = world
    named = _backlog_world(root)
    monkeypatch.setattr(vault, "_qmd_daemon_search", lambda *a, **k: [])
    result = await vault.call_tool("vault_recall", {
        "query": ITEM_QUERY, "limit": 5, "grep_code": False,
        "include_facts": True, "expand_graph": False, "graph_rerank": False})
    assert not result.is_error, result.content[0].text[:400]
    facts = json.loads(result.content[0].text)["facts"]
    assert facts, "the queried item has facts; an empty answer is not the fix"
    wrong = [f for f in facts if any(f"#{n}" in f["fact"] for n in named)]
    assert not wrong, (
        f"{len(wrong)} of {len(facts)} MCP-delivered facts name an item id other"
        " than 363")
    assert not [f for f in facts if f["entity"] in ("Backlog Item", "Backlog")], (
        f"a class row held a slot across the tool boundary: "
        f"{Counter(f['entity'] for f in facts)}")


def test_no_row_for_the_named_id_emits_no_other_item_facts(world, monkeypatch):
    """A query for a recently created item, whose facts have not been written
    yet, used to answer with 100% other-item facts: `_get_facts_sync` returns
    nothing for the id, so the class rows were the entire seed set. Suppressing
    them leaves NO facts — which is the required outcome, not other items' facts.
    """
    root, _ = world
    named = _backlog_world(root, with_specific=False)
    out = _recall_fact_items(monkeypatch)
    wrong = [f for f in out["facts"] if any(f"#{n}" in f["fact"] for n in named)]
    assert not wrong, (
        f"{len(wrong)} facts name an item other than 363 with no row for 363;"
        f" owners were {sorted({f['entity'] for f in out['facts']})}")


def test_a_query_that_names_no_instance_still_seeds_the_class_row(world):
    """The guard is scoped to instance queries. `what is the backlog item board`
    names no id, so `Backlog Item` keeps the answer it has always given —
    otherwise this would be a retrieval regression on every class-level question.
    The live store holds 144 entity directories whose every word is generic,
    carrying 6,705 facts (both re-measured 2026-09-18); this rule reaches only the
    12 of those that also name a record class, and only when the query names one
    record."""
    root, _ = world
    _backlog_world(root)
    ranked = dict(retrieval.extract_entities_from_query(
        "what does the backlog item board hold"))
    assert ranked.get("Backlog Item", 0) >= 5.0, (
        f"a class question lost its class row: {ranked}")


def test_degree_breaks_ties_deterministically(world):
    root, st = world
    for name in ("Alpha One", "Alpha Two"):
        _write_facts(root, name, "state", [{"fact": "x", "id": "stat-001"}])
    st.edges.add({"source": "Alpha Two", "target": "Hub", "type": "uses"}, origin="t")
    ranked = retrieval.extract_entities_from_query("alpha")
    names = [n for n, _ in ranked]
    assert names.index("Alpha Two") < names.index("Alpha One")


# ── graph_weighted_neighbors ─────────────────────────────────────────────────

def test_weighted_neighbors_decay_by_hop_and_skip_expired(world):
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "uses", "confidence": 1.0}, origin="t")
    st.edges.add({"source": "B", "target": "C", "type": "uses", "confidence": 1.0}, origin="t")
    gone = st.edges.add({"source": "A", "target": "Gone", "type": "uses"}, origin="t")
    st.edges.expire(gone, "test")

    one = dict(retrieval.graph_weighted_neighbors(["A"], top_k=5, hops=1))
    assert set(one) == {"B"}
    two = dict(retrieval.graph_weighted_neighbors(["A"], top_k=5, hops=2))
    assert set(two) == {"B", "C"}
    assert two["C"] < two["B"], "second hop must decay"
    assert "Gone" not in two


def test_edge_type_weights_order_neighbours(world):
    _, st = world
    st.edges.add({"source": "A", "target": "Typed", "type": "depends_on", "confidence": 1.0}, origin="t")
    st.edges.add({"source": "A", "target": "Weak", "type": "co_mentioned", "confidence": 1.0}, origin="t")
    ranked = retrieval.graph_weighted_neighbors(["A"], top_k=2)
    assert [n for n, _ in ranked] == ["Typed", "Weak"]


# ── directed traversal (backlog #396) ────────────────────────────────────────
#
# Direction has lived on every edge since the SQLite rewrite; both traversal
# sites read it only to pick "the other end", so the retrieval path could never
# answer "what depends on X" / "what did X supersede" even where the data
# supports it. These tests pin the direction argument and, just as importantly,
# the default: everything above and every existing caller (vault_recall → the
# nightly retrieval eval) must keep walking symmetrically.

def _names(pairs):
    return [n for n, _ in pairs]


def test_weighted_neighbors_out_follows_source_to_target(world):
    """A →(depends_on) B. `out` from A reaches B; `out` from B reaches nothing,
    because the arrow does not point out of B."""
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "depends_on", "confidence": 1.0}, origin="t")

    assert _names(retrieval.graph_weighted_neighbors(["A"], top_k=5, direction="out")) == ["B"]
    assert retrieval.graph_weighted_neighbors(["B"], top_k=5, direction="out") == []


def test_weighted_neighbors_in_is_the_mirror_of_out(world):
    """Same edge, opposite query: `in` from B reaches A, `in` from A is empty.
    This is the "what depends on X" shape that was unanswerable before #396."""
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "depends_on", "confidence": 1.0}, origin="t")

    assert _names(retrieval.graph_weighted_neighbors(["B"], top_k=5, direction="in")) == ["A"]
    assert retrieval.graph_weighted_neighbors(["A"], top_k=5, direction="in") == []


def test_graph_expand_entities_honours_direction(world):
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "uses", "confidence": 1.0}, origin="t")

    assert retrieval.graph_expand_entities(["A"], direction="out") == ["B"]
    assert retrieval.graph_expand_entities(["B"], direction="out") == []
    assert retrieval.graph_expand_entities(["B"], direction="in") == ["A"]
    assert retrieval.graph_expand_entities(["A"], direction="in") == []


def test_direction_is_consistent_across_hops(world):
    """A → B → C. A directed multi-hop walk never reverses mid-path: `out` from
    A gets both, `out` from B gets only C (not back to A), `in` from C gets
    both walking backwards."""
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "uses", "confidence": 1.0}, origin="t")
    st.edges.add({"source": "B", "target": "C", "type": "uses", "confidence": 1.0}, origin="t")

    assert sorted(_names(retrieval.graph_weighted_neighbors(["A"], top_k=5, hops=2, direction="out"))) == ["B", "C"]
    assert _names(retrieval.graph_weighted_neighbors(["B"], top_k=5, hops=2, direction="out")) == ["C"]
    assert _names(retrieval.graph_weighted_neighbors(["A"], top_k=5, hops=2, direction="in")) == []
    assert sorted(_names(retrieval.graph_weighted_neighbors(["C"], top_k=5, hops=2, direction="in"))) == ["A", "B"]


SYMMETRIC_TYPES = ("mentions", "related_to", "discusses", "competes_with",
                   "co_mentioned", "wiki_link_co_occurrence", "some_unknown_type")


def test_symmetric_types_are_excluded_from_a_directed_query(world):
    """34% of live edges are directional; the other 66% (mentions, related_to,
    discusses, competes_with, cooccurrence) have endpoint order as an
    extraction artefact. Honouring direction over the whole graph would be
    two-thirds noise, so a directed query is a *typed* query. That treatment is
    specified in `_directed_neighbor`'s docstring, not incidental — this test is
    the other half of the contract."""
    _, st = world
    directional = sorted(retrieval.DIRECTIONAL_EDGE_TYPES)
    for i, etype in enumerate(directional):
        st.edges.add({"source": f"Dir_{i}", "target": "Hub", "type": etype,
                      "confidence": 1.0}, origin="t")
    for i, etype in enumerate(SYMMETRIC_TYPES):
        st.edges.add({"source": f"Sym_{i}", "target": "Hub", "type": etype,
                      "confidence": 1.0}, origin="t")

    got = set(_names(retrieval.graph_weighted_neighbors(["Hub"], top_k=50, direction="in")))
    assert got == {f"Dir_{i}" for i in range(len(directional))}, (
        "a directed query must return only directional sources, got "
        f"{sorted(got)}"
    )
    assert set(_names(retrieval.graph_weighted_neighbors(["Hub"], top_k=50, direction="out"))) == set()

    # ... and the symmetric neighbours are still reachable the default way.
    both = set(_names(retrieval.graph_weighted_neighbors(["Hub"], top_k=50)))
    assert both == {f"Dir_{i}" for i in range(len(directional))} | {
        f"Sym_{i}" for i in range(len(SYMMETRIC_TYPES))
    }
    assert set(retrieval.graph_expand_entities(["Hub"], direction="in")) == {
        f"Dir_{i}" for i in range(len(directional))
    }


def test_symmetric_neighbours_survive_under_the_default(world):
    """Guardrail for the acceptance criterion that matters most: `both` is the
    default and must still return the symmetric neighbourhood, because that is
    what vault_recall asks for and what the nightly eval measures."""
    _, st = world
    st.edges.add({"source": "A", "target": "Symmetric", "type": "mentions", "confidence": 1.0}, origin="t")
    st.edges.add({"source": "A", "target": "Directional", "type": "uses", "confidence": 1.0}, origin="t")

    default = retrieval.graph_weighted_neighbors(["A"], top_k=5)
    assert set(_names(default)) == {"Symmetric", "Directional"}
    assert default == retrieval.graph_weighted_neighbors(["A"], top_k=5, direction="both")
    assert set(retrieval.graph_expand_entities(["A"])) == {"Symmetric", "Directional"}
    assert retrieval.graph_expand_entities(["A"], direction="both") == retrieval.graph_expand_entities(["A"])


def test_direction_filters_edges_but_not_scoring(world):
    """Direction selects which edges are traversable; it must not silently
    rescore the ones that are. Same edge, same weight under both and out."""
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "uses", "confidence": 0.8}, origin="t")

    both = dict(retrieval.graph_weighted_neighbors(["A"], top_k=5, direction="both"))
    out = dict(retrieval.graph_weighted_neighbors(["A"], top_k=5, direction="out"))
    assert out == both


def test_bad_direction_raises_instead_of_going_symmetric(world):
    """`_fact_relationships` ignores an unrecognised direction. Here a typo
    would have returned a symmetric walk to a caller that asked for a directed
    one — a confident wrong answer to exactly the question the parameter is
    for."""
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "uses", "confidence": 1.0}, origin="t")
    with pytest.raises(ValueError):
        retrieval.graph_weighted_neighbors(["A"], direction="incoming")
    with pytest.raises(ValueError):
        retrieval.graph_expand_entities(["A"], direction="sideways")


def test_direction_case_is_normalised(world):
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "uses", "confidence": 1.0}, origin="t")
    assert _names(retrieval.graph_weighted_neighbors(["A"], top_k=5, direction=" OUT ")) == ["B"]
    assert _names(retrieval.graph_weighted_neighbors(["A"], top_k=5, direction=None)) == ["B"]




def test_get_facts_sync_temporal_filters(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "current"},
        {"id": "stat-002", "fact": "expired", "expired_at": "2026-02-01"},
        {"id": "stat-003", "fact": "invalid", "invalid_at": "2026-02-01"},
        {"id": "stat-004", "fact": "future", "valid_at": "2027-01-01"},
    ])
    now = [f["id"] for f in retrieval.get_facts_sync("Lloyd")["facts"]]
    assert now == ["stat-001", "stat-004"]        # valid_at is not a future gate by default
    at = [f["id"] for f in retrieval.get_facts_sync("Lloyd", as_of="2026-01-15")["facts"]]
    assert at == ["stat-001", "stat-002", "stat-003"]
    every = retrieval.get_facts_sync("Lloyd", include_expired=True)["facts"]
    assert len(every) == 4
    assert retrieval.get_facts_sync("Nobody")["facts"] == []


# ── fact_profile ─────────────────────────────────────────────────────────────

def test_fact_profile_caps_a_god_node(world):
    """Uncapped, this returned every fact an entity had — 5,489 for `Lloyd`
    — straight into the model's context."""
    root, _ = world
    _write_facts(root, "Lloyd", "state",
                 [{"id": f"stat-{i:03d}", "fact": f"fact number {i}", "category": "state"}
                  for i in range(1, 61)])
    out = facts_mod._fact_profile({"entity": "Lloyd"})
    assert out["fact_count"] == 60
    assert len(out["categories"]["state"]) == retrieval.FACT_RANK_CAP_SEED
    assert out["truncated_categories"] == {"state": 60}
    assert "hint" in out and "showing" in out["summary"]


def test_fact_profile_ranks_by_query_when_given(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state",
                 [{"id": f"stat-{i:03d}", "fact": f"filler {i}", "category": "state"}
                  for i in range(1, 40)]
                 + [{"id": "stat-099", "category": "state",
                     "fact": "Lloyd serves models through vLLM"}])
    out = facts_mod._fact_profile({"entity": "Lloyd", "query": "vLLM serving"})
    assert out["categories"]["state"][0]["id"] == "stat-099"


# ── fact_resolve ─────────────────────────────────────────────────────────────

def test_fact_resolve_reports_by_default(world):
    """It defaulted to auto_resolve=True, so a call that reads like a query
    silently expired facts."""
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "the feature is enabled", "confidence": 0.9},
        {"id": "stat-002", "fact": "the feature is disabled", "confidence": 0.5},
    ])
    out = facts_mod._fact_resolve({"entity": "Lloyd"})
    assert out["resolved"] == 0 and out["remaining"] >= 1
    fm = yaml.safe_load((root / "Lloyd" / "Lloyd-state.md").read_text().split("---")[1])
    assert all(not f.get("invalid_at") and not f.get("expired_at") for f in fm["facts"])


def test_fact_resolve_sets_invalid_at_only(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "the feature is enabled", "confidence": 0.9},
        {"id": "stat-002", "fact": "the feature is disabled", "confidence": 0.5},
    ])
    out = facts_mod._fact_resolve({"entity": "Lloyd", "auto_resolve": True})
    assert out["resolved"] == 1
    fm = yaml.safe_load((root / "Lloyd" / "Lloyd-state.md").read_text().split("---")[1])
    by_id = {f["id"]: f for f in fm["facts"]}
    assert by_id["stat-002"]["invalid_at"] and not by_id["stat-002"].get("expired_at")
    assert not by_id["stat-001"].get("invalid_at")


def test_fact_resolve_leaves_equal_confidence_alone(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "the feature is enabled", "confidence": 0.9},
        {"id": "stat-002", "fact": "the feature is disabled", "confidence": 0.9},
    ])
    assert facts_mod._fact_resolve({"entity": "Lloyd", "auto_resolve": True})["resolved"] == 0


def test_the_contradiction_scan_is_refused_on_a_god_node(world):
    """O(n squared) over 5,489 facts is 15 million comparisons — measured at
    113 seconds through MCP, and it reported 32,857 `contradictions` that
    were almost all the overlap heuristic firing on similar phrasing.
    Refused in BOTH modes: the report path ran the scan too."""
    root, _ = world
    _write_facts(root, "Lloyd", "state",
                 [{"id": f"stat-{i:03d}", "fact": f"fact {i} is enabled",
                   "confidence": 0.9, "category": "state"}
                  for i in range(1, retrieval.FACT_GODNODE_THRESHOLD + 5)])
    for params in ({"entity": "Lloyd", "auto_resolve": True}, {"entity": "Lloyd"}):
        out = facts_mod._fact_resolve(params)
        assert "error" in out and "refused" in out["error"], params
    check = facts_mod._fact_check({"entity": "Lloyd"})
    assert "error" in check and "refused" in check["error"]
    # a narrower slice is still scannable
    _write_facts(root, "Lloyd", "goal", [
        {"id": "goal-001", "fact": "the goal is enabled", "confidence": 0.9, "category": "goal"},
        {"id": "goal-002", "fact": "the goal is disabled", "confidence": 0.5, "category": "goal"},
    ])
    retrieval.invalidate_fact_file_cache()
    narrow = facts_mod._fact_check({"entity": "Lloyd", "category": "goal"})
    assert "error" not in narrow and narrow["checked"] == 2


# ── entity kind ──────────────────────────────────────────────────────────────

def test_system_names_are_not_typed_as_people():
    """`Knowledge Graph`, `Claude Code` and `Isaac Lab` all satisfy the
    `First Last` shape. The v4 classifier's PERSON test ran before its SYSTEM
    test, so all three were typed PERSON — and the type gates
    ROLE_BLOCKED_VERBS, which is how `created_by` between two systems passed.
    """
    from app.entity_kind import derive_entity_type, derive_kind
    for name in ("Knowledge Graph", "Claude Code", "Isaac Lab", "Intel Pipeline",
                 "QMD Daemon", "Morning Briefing System"):
        assert derive_kind(name) == "system", name
        assert derive_entity_type(name) == "SYSTEM", name


def test_person_needs_corroboration(monkeypatch):
    from app import entity_kind
    monkeypatch.setattr(entity_kind, "_people_cache", {"jane doe"})
    assert entity_kind.derive_kind("Jane Doe") == "person"
    # same shape, no note and no people/ source -> not a person
    assert entity_kind.derive_kind("Isaac Lab") == "system"
    # the source document settles it either way
    assert entity_kind.derive_kind("Someone New", "people/someone-new.md") == "person"
    assert entity_kind.derive_kind("Jane Doe", "projects/x.md") == "project"


def test_kind_recognises_tasks_docs_and_skills():
    from app.entity_kind import derive_kind
    assert derive_kind("Task #67") == "task"
    assert derive_kind("Autonomy Task #33") == "task"
    assert derive_kind("backlog_item_235") == "task"
    assert derive_kind("server.py") == "doc"
    assert derive_kind("knowledge/ai/note.md") == "doc"
    assert derive_kind("nightly-reflection") == "skill"
    assert derive_kind("vLLM") == "system"


# ── every fact write updates the index ───────────────────────────────────────

def test_fact_invalidate_updates_the_index(world):
    """The markdown is the fact layer, but facts_idx is what the router and
    fact_profile read. fact_invalidate wrote expired_at to the file and left
    the index alone, so the Memory page went on showing the fact as current."""
    root, st = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "Lloyd runs on Ollama", "confidence": 0.9, "category": "state"},
        {"id": "stat-002", "fact": "Lloyd runs on vLLM", "confidence": 0.9, "category": "state"},
    ])
    st.facts_idx.reindex(root=root)
    assert len(st.facts_idx.for_entity("Lloyd")) == 2

    out = facts_mod._fact_invalidate({
        "entity": "Lloyd", "ended": "2026-09-04", "fact_substring": "ollama",
        "reason": "moved to vLLM"})
    assert out["expired_count"] == 1

    live = st.facts_idx.for_entity("Lloyd")
    assert [f["fact_id"] for f in live] == ["stat-002"], "the index still serves the expired fact"
    everything = st.facts_idx.for_entity("Lloyd", include_expired=True)
    assert len(everything) == 2
    expired = next(f for f in everything if f["fact_id"] == "stat-001")
    assert expired["expired_at"] == "2026-09-04"


def test_fact_add_and_resolve_also_update_the_index(world):
    root, st = world
    add = facts_mod._fact_add({"entity": "Newthing", "category": "state", "fact": "it exists"})
    assert add["success"]
    assert [f["fact"] for f in st.facts_idx.for_entity("Newthing")] == ["it exists"]

    _write_facts(root, "Pair", "state", [
        {"id": "stat-001", "fact": "the flag is enabled", "confidence": 0.9, "category": "state"},
        {"id": "stat-002", "fact": "the flag is disabled", "confidence": 0.4, "category": "state"},
    ])
    st.facts_idx.reindex(root=root)
    retrieval.invalidate_fact_file_cache()
    assert facts_mod._fact_resolve({"entity": "Pair", "auto_resolve": True})["resolved"] == 1
    assert [f["fact_id"] for f in st.facts_idx.for_entity("Pair")] == ["stat-001"]


# ── #877: the four unwired retrieval symbols stay deleted ────────────────────
#
# The names are assembled below rather than spelled out. Acceptance clause 1 of
# #877 is a repo-wide `grep --include=*.py` for these exact literals, so a test
# that wrote them in the clear would itself be the match that keeps that grep
# red. Every assembled string here resolves to the symbol it checks for; none
# of them is a placeholder.
UNWIRED_INTENT_SYMBOLS = tuple(
    f"_INTENT_{kind}_RE" for kind in ("FACTUAL", "TEMPORAL", "CONCEPTUAL")
)
UNWIRED_FUSION_SYMBOL = "_rrf" + "_fuse"
UNWIRED_FUSION_SCORE_KEY = "rrf" + "_score"
UNWIRED_LITERALS = UNWIRED_INTENT_SYMBOLS + (
    UNWIRED_FUSION_SYMBOL, UNWIRED_FUSION_SCORE_KEY)

# The vault surface as it stood at base 3d95c9c1, measured before the deletion.
VAULT_TOOL_NAMES = [
    "vault_overview", "vault_read", "vault_recall", "vault_search", "vault_write",
]


def test_vault_no_longer_defines_the_unwired_retrieval_symbols():
    """Three intent regexes and one RRF function reached this file already
    unwired (the `memory.py` split, 82ef902) and never gained a caller. Their
    presence made `grep rrf` / `grep INTENT` over the MCP layer read as though
    MCP-level rank fusion or intent routing were wired. Neither is: what fuses
    lives in the QMD fork (`reciprocalRankFusion` in qmd/src/store.ts)."""
    for name in UNWIRED_INTENT_SYMBOLS + (UNWIRED_FUSION_SYMBOL,):
        assert not hasattr(vault, name), f"agent_mcp.vault still defines {name}"

    # The fusion function's output key was a dict key, not a symbol, so
    # hasattr alone would miss it if it were ever re-emitted.
    src = Path(vault.__file__).read_text()
    for literal in UNWIRED_LITERALS:
        assert literal not in src, f"{literal} still appears in agent_mcp/vault.py"


def test_the_checkout_holds_no_reference_to_the_dead_retrieval_symbols():
    """Clause 1 as the item states it — over the whole checkout, not just the
    one file, with `.venvs` excluded exactly as the item's command excludes it.
    This is the test that catches the deletion being undone by a re-wiring
    somewhere else in the tree."""
    hits = []
    for path in ROOT.rglob("*.py"):
        # `is_file`: the live tree's fact layer holds an entity directory named
        # `Router.py` (_pipeline/vault-derived/facts/), which rglob matches.
        if ".venvs" in path.parts or not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        for literal in UNWIRED_LITERALS:
            if literal in text:
                hits.append(f"{path.relative_to(ROOT)}: {literal}")
    assert hits == [], "dead retrieval symbols still referenced: " + "; ".join(hits)


def _code_names(fn) -> set:
    """Every name reachable from a function's code object, including its nested
    functions — `_vault_recall` builds its retrieval legs as closures, so the
    outer `co_names` alone would not see a call made from inside one."""
    names: set = set()
    stack, seen = [fn.__code__], set()
    while stack:
        code = stack.pop()
        if id(code) in seen:
            continue
        seen.add(id(code))
        names.update(code.co_names)
        stack.extend(c for c in code.co_consts if type(c).__name__ == "code")
    return names


def test_the_recall_leg_the_eval_scores_does_no_rank_fusion():
    """#877 is a pure deletion, so the claim acceptance rests on is that the
    thing deleted never ran. This states that about the one leg
    `eval/run_eval.py` scores: `_vault_recall` merges QMD, code-grep and graph
    legs by dedupe-append, first-wins, and the only MCP-side rescoring is
    `_graph_rerank`, which is default-off (`RECALL_GRAPH_RERANK`). Nothing on
    that path fused ranks before this round and nothing does after — wiring the
    deleted function into it is exactly what would move a metric, and that is
    what this catches."""
    referenced = _code_names(vault._vault_recall)
    for literal in UNWIRED_LITERALS:
        assert literal not in referenced, (
            f"{literal} is referenced on the recall path the eval scores")
    assert vault.RECALL_GRAPH_RERANK is False, (
        "graph rerank flipped default-on; the eval numbers this round measured "
        "were taken with it off, so the base-vs-changed comparison no longer holds")


def test_the_vault_tool_surface_is_unchanged_by_the_deletion():
    """Clause 2. None of the four symbols was reachable from a tool, so
    deleting them may not move the advertised surface in either direction: not
    one tool lost, and no new tool standing in for what was removed."""
    import agent_mcp.main as mcp_main

    tools = asyncio.run(mcp_main.list_tools())
    got = sorted(t.name for t in tools if t.name.startswith("vault"))
    assert len(VAULT_TOOL_NAMES) == 5, "the pinned list itself is no longer five"
    assert got == VAULT_TOOL_NAMES, f"vault tool surface changed: {got}"
