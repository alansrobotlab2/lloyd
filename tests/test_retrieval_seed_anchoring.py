"""Query-seed extraction resolves through the store's alias table (#1260).

The eval's entity leg sat at 0.5 across eight nightly runs while its doc leg
saturated at 1.0. The legs diverged because six of twenty queries had NO expected
entity among the seeds `_extract_entities_from_query` produced: the extractor
ranked entity DIRECTORY names by substring and token overlap and never asked the
store what a name IS, so `Relationship Graph` seeded on `Relationship Graph` and
left `Knowledge Graph` — where that entity's facts, edges and documents live —
unreachable at any seed budget. `app/kg_store.py` holds ~4,000 alias rows saying
exactly that, and `store().resolve` was not called on this path.
"""
import json
import os
import subprocess
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.kg_store as ks  # noqa: E402
from agent_mcp import retrieval  # noqa: E402

# `_pipeline/` is gitignored, so a git worktree — the automod round this runs in —
# has neither `kg.sqlite` nor a fact tree. Same rule `app/uptake.py:lloyd_root` and
# `tests/test_eval_corpus_guard.py` apply: measure the live tree, because a
# worktree's absence of the store is not a measurement of the corpus.
LIVE = Path.home() / "lloyd" / "_pipeline" / "vault-derived"
LIVE_FACTS = LIVE / "facts"
LIVE_KG_DB = LIVE / "kg.sqlite"


# The extractor's own alias-map reader, captured before any test patches it. A folded
# call rebinds to THIS: it reads `retrieval.store`, which the fixture has already
# pointed at the tmp store, so a folded call folds against the fixture's alias rows.
REAL_ALIAS_SURFACE_MAP = retrieval._alias_surface_map


def _write_entity(root: Path, name: str, fact: str) -> None:
    """One entity's fact file, in the shape `agent_mcp/retrieval.py` parses."""
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / f"{name}-state.md").write_text(
        "---\n"
        f"entity: {name}\n"
        "facts:\n"
        f"- fact: {fact}\n"
        "  confidence: 0.9\n"
        "  provenance: STATED\n"
        "  created_at: '2026-09-01T00:00:00'\n"
        "---\n"
        f"\n# {name}\n",
        encoding="utf-8")


def _point_the_extractor_at(tmp_path, monkeypatch, entities, aliases=()):
    """A fact tree over `entities` plus a store holding `aliases`, wired in.

    Built through the named routes (`KGStore`, `entities.register`, `aliases.set`)
    rather than hand-written sqlite, so an alias row is what a real sweep would
    write. `retrieval` imports `FACTS_ROOT` and `store` into its own namespace, so
    BOTH names are patched there — patching only `app.kg_store.store` would leave
    the extractor holding the old function and the test would pass or fail for the
    wrong reason.

    The name list comes from the real directory read, not from a stand-in. The fold
    gates a canonical on "does a directory by that name exist", and that answer
    comes from `agent_mcp._shared._get_entity_dirs_cached` — a cached scan whose
    validity is decided by the mtime of the `FACTS_ROOT` it was last pointed at
    (`_shared.py:577-599`). So this patches `_shared.FACTS_ROOT` (the root the scan
    walks) and resets that cache: the fold is then exercised through the same
    code path production uses, including the cache's staleness rule. Patching
    `_get_entity_dirs_cached` itself would have tested the fold's reaction to a
    list of names, which is narrower than the claim being made and would have
    stayed green if the real read never saw the tmp tree. `agent_mcp.facts` holds
    its own import of the name and is repointed too, because `_vault_recall` reads
    facts through it.
    """
    import agent_mcp._shared as shared
    import agent_mcp.facts as facts_mod

    facts = tmp_path / "facts"
    facts.mkdir(parents=True)
    for name in entities:
        _write_entity(facts, name, f"{name} holds the facts about itself")
    db = tmp_path / "kg.sqlite"
    st = ks.KGStore(db)
    for name in entities:
        st.entities.register(name, kind="system")
    for surface, canonical in aliases:
        st.aliases.set(surface, canonical, kind="semantic", origin="test")
    st.close()
    st = ks.KGStore(db)
    for mod in (retrieval, shared, facts_mod):
        monkeypatch.setattr(mod, "FACTS_ROOT", facts, raising=False)
    monkeypatch.setattr(retrieval, "store", lambda: st)
    # Setup-side reset is enough: the cache's own validity rule compares the mtime
    # of the root it last scanned, so once monkeypatch restores the real root the
    # next reader rescans rather than reusing this test's names.
    shared._invalidate_entity_dirs_cache()
    return st


def test_a_query_phrased_as_an_alias_seeds_on_the_alias_canonical(tmp_path, monkeypatch):
    """Clause 1's mechanism: an alias surface seeds on the entity's canonical.

    `Relationship Graph` is in the alias table pointing at `Knowledge Graph`, so a
    query worded with the alias must put the canonical into the seed list. Before
    this the query seeded only on the alias surface and on substring neighbours of
    it, and the canonical appeared only if the query happened to contain it.
    """
    _point_the_extractor_at(
        tmp_path, monkeypatch, ["Knowledge Graph", "Relationship Graph"],
        [("Relationship Graph", "Knowledge Graph")])
    seeds = [n for n, _ in retrieval.extract_entities_from_query(
        "what is the quality of the relationship graph") or []]
    assert "Knowledge Graph" in seeds, (
        f"the alias surface was scored and its canonical never reached the list: {seeds}")


def test_the_alias_seed_is_kept_beside_its_canonical(tmp_path, monkeypatch):
    """The fold is additive, because the doc leg must not move.

    The alias's own directory holds facts too. Substituting the canonical for the
    surface would trade a seed the doc leg already hits for the entity-leg gain,
    and the nightly trend cannot read an entity number that moved the doc number.
    """
    _point_the_extractor_at(
        tmp_path, monkeypatch, ["Knowledge Graph", "Relationship Graph"],
        [("Relationship Graph", "Knowledge Graph")])
    seeds = [n for n, _ in retrieval.extract_entities_from_query(
        "what is the quality of the relationship graph") or []]
    assert "Knowledge Graph" in seeds and "Relationship Graph" in seeds, seeds


def test_an_alias_to_a_name_with_no_facts_never_becomes_a_seed(tmp_path, monkeypatch):
    """The canonical has to be readable, or the fold seeds on nothing.

    `Dangling Index` is an alias whose target has no directory. Seeding on it would
    spend a slot on an entity with no facts to read — turning an alias hit into a
    guaranteed fact-leg miss, which is a worse failure than the one being fixed.
    """
    _point_the_extractor_at(
        tmp_path, monkeypatch, ["Relationship Index", "Dangling Index"],
        [("Dangling Index", "Nowhere Entity")])
    seeds = [n for n, _ in retrieval.extract_entities_from_query(
        "what is in the dangling index") or []]
    assert "Nowhere Entity" not in seeds, seeds


def test_an_unreadable_store_seeds_lexically_rather_than_reporting_no_seeds(tmp_path, monkeypatch):
    """No store must not read as "this query names no entity".

    `_alias_surface_map` folds nothing when the store will not open, and the
    extractor still ranks directory names. A path that raised instead would take
    the seed list — and with it every entity-leg score — to zero on a broken store,
    reporting an infrastructure fault as a retrieval regression.
    """
    _point_the_extractor_at(tmp_path, monkeypatch, ["Knowledge Graph"])

    def _refuse():
        raise ks.StoreUnavailable("no store on this path")

    monkeypatch.setattr(retrieval, "store", _refuse)
    out = retrieval.extract_entities_from_query("what is the knowledge graph")
    assert any("knowledge graph" in n.lower() for n, _ in out), out


def _run_against_the_live_corpus(script: str) -> dict:
    """Run `script` with the corpus, the fact tree and the store the eval scores.

    A subprocess because `agent_mcp._shared.FACTS_ROOT` and the configured store
    path are process-global: setting them in-process would repoint every later test
    in the suite, which is the failure `tests/test_kg_store.py` keeps visible. The
    `LLOYD_*` env vars are read at import, so this is the only route in.
    """
    assert LIVE_FACTS.is_dir() and LIVE_KG_DB.is_file(), (
        f"no store to measure at {LIVE}. These assertions are about the entity table "
        "the nightly eval scores against; an absent store is not a verdict about the "
        "corpus and must not be reported as one")
    env = dict(os.environ)
    env["LLOYD_FACTS_ROOT"] = str(LIVE_FACTS)
    env["LLOYD_KG_DB"] = str(LIVE_KG_DB)
    res = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    return json.loads(res.stdout.strip().splitlines()[-1])


def _recall_fact_entities(monkeypatch, tmp_path, entities, aliases, query, width, *, fold=True):
    """Ask `vault._vault_recall` what facts it returns at an explicit seed width.

    This is the seam the fold's extra seed is spent on. `RECALL_SEED_TOP_K` is a
    slice in a different module from the one that builds the ranking, so a test that
    stopped at `extract_entities_from_query` would have verified a list nobody
    consumes: the ranking is additive only up to the width, and past it a canonical
    takes a slot a lexically-matched name used to hold. Driving the real tool tells us
    which happened, and the observable is the returned FACTS — a seed that survives
    the cut contributes its entity's facts, one that does not contributes nothing and
    leaves no trace in the seed list.
    """
    import agent_mcp.vault as vault

    # Its own tree per call: a test that reads the fold off and then on re-enters here
    # with the same `tmp_path`, and a second `facts/` under the same root would collide.
    _point_the_extractor_at(tmp_path / ("folded" if fold else "unfolded"),
                            monkeypatch, entities, list(aliases.items()))
    # Bound on BOTH branches. A `monkeypatch` patch lives for the whole test, so setting
    # it only on the unfolded branch leaves the folded call reading an empty map — the
    # reading comes out as a no-op fold and the test fails claiming the fold is free.
    # The folded branch therefore rebinds to the module's real function.
    monkeypatch.setattr(retrieval, "_alias_surface_map",
                        (lambda: {}) if not fold else REAL_ALIAS_SURFACE_MAP)
    monkeypatch.setattr(vault, "_qmd_daemon_search", lambda *a, **k: [])
    out = vault._vault_recall({"query": query, "limit": 5, "grep_code": False},
                              seed_top_k=width)
    assert "error" not in out, out
    return {f.get("entity") for f in out.get("facts") or []}


# Three entity directories, a query that names two of them, and one alias row. The
# two names the query spells out score 5.9 and 5.55 — the width decides, not the score.
_SEAM_ENTITIES = ["Relationship Graph", "Knowledge Graph", "Noise Floor"]
_SEAM_ALIASES = {"relationship graph": "Knowledge Graph"}
_SEAM_QUERY = "what's the noise floor in our relationship graph?"


def test_the_folded_canonical_reaches_the_facts_the_real_seeder_asks_for(tmp_path, monkeypatch):
    """Finding 1 of the review, additive half: the canonical is a seed `vault` uses.

    With room for all three names the fold changes the recall's OUTPUT and not only
    its input: the canonical's own fact file is retrieved, because `vault` asked for
    that entity's facts by name and the store answered. This is the reading the #1199
    fix depends on — a query phrased as an alias reaching the entity it means.
    """
    got = _recall_fact_entities(monkeypatch, tmp_path, _SEAM_ENTITIES, _SEAM_ALIASES,
                                _SEAM_QUERY, width=3)
    assert {"Knowledge Graph", "Relationship Graph", "Noise Floor"} <= got, got


def test_the_fold_costs_a_seed_when_the_width_has_no_room_for_it(tmp_path, monkeypatch):
    """Finding 1 of the review, priced half: at a tight width the extra seed is not free.

    Cut the width to two and the fold's canonical takes a slot — `Noise Floor`'s fact
    stops coming back, and the test asserts that rather than excusing it. The bound
    that keeps the price honest, and what the comment in `extract_entities_from_query`
    claims, is that a canonical is scored IDENTICALLY to the alias surface that earned
    it and never above it, so the seed it displaces is one the ranking had already
    placed no higher than it. The unfolded pair is the control: without the fold
    `Noise Floor` held that slot because nothing had folded `Knowledge Graph` in.
    """
    without = _recall_fact_entities(monkeypatch, tmp_path, _SEAM_ENTITIES, _SEAM_ALIASES,
                                    _SEAM_QUERY, width=2, fold=False)
    assert without == {"Relationship Graph", "Noise Floor"}, without
    with_fold = _recall_fact_entities(monkeypatch, tmp_path, _SEAM_ENTITIES, _SEAM_ALIASES,
                                      _SEAM_QUERY, width=2)
    assert "Knowledge Graph" in with_fold, with_fold
    assert "Noise Floor" not in with_fold, (
        "the fold is supposed to cost the tail seed at this width — if nothing is "
        "displaced any more this test is asserting a price nobody pays")


_WIDTH_SCRIPT = """
import json, sys, yaml
sys.path.insert(0, '.')
from agent_mcp import retrieval
from agent_mcp.vault import RECALL_SEED_TOP_K

qs = yaml.safe_load(open('eval/vault_recall_queries.yaml'))['queries']
real = retrieval._alias_surface_map
displaced = []
for q in qs:
    retrieval._alias_surface_map = lambda: {}
    base = retrieval.extract_entities_from_query(q['query']) or []
    retrieval._alias_surface_map = real
    now = retrieval.extract_entities_from_query(q['query']) or []
    bs, ns = dict(base), dict(now)
    bk = [e for e, _ in base[:RECALL_SEED_TOP_K]]
    nk = [e for e, _ in now[:RECALL_SEED_TOP_K]]
    # Evicted from the window, and everything the fold put INTO the window at a score
    # it did not previously hold there — either a name absent from the unfolded
    # ranking entirely, or one the fold raised. Both are what a lost seed gave its
    # slot to, and they are scored differently, so both carry their two scores.
    lost = [(e, bs[e]) for e in bk if e not in nk]
    gained = [(e, bs.get(e), ns[e]) for e in nk
              if e not in bk or ns[e] > bs.get(e, 0.0) + 1e-9]
    if lost:
        displaced.append({'id': q['id'],
                     'lost': [[e, round(s, 4)] for e, s in lost],
                     'gained': [[e, (None if b is None else round(b, 4)), round(f, 4)]
                                for e, b, f in gained]})
print(json.dumps({'width': RECALL_SEED_TOP_K, 'displaced': displaced}))
"""


def test_the_fold_never_costs_a_seed_that_outscored_the_canonical_that_replaced_it():
    """The bound, measured over the shipped corpus at the shipped width.

    `RECALL_SEED_TOP_K` truncates in another module, so "the fold is additive" is only
    true where there is room. This reads the real store and real entity tree through
    the real extractor and asks, for every shipped query, which seeds the fold
    actually costs at production width — then checks the strongest statement that
    survives contact with the tie-break: a seed the fold displaces never scored higher
    than the best seed the fold seated. Where that is false the fold is spending a
    matched seed to buy a weaker canonical, and the fold needs a score floor — which is
    exactly the question the seam-unverified finding raised.
    """
    out = _run_against_the_live_corpus(_WIDTH_SCRIPT)
    displaced = out["displaced"]
    assert out["width"] == 10, out["width"]
    for row in displaced:
        max_lost = max(s for _e, s in row["lost"])
        max_gained = max(after for _e, _before, after in row["gained"])
        assert max_gained >= max_lost - 1e-9, (
            f"{row['id']}: the fold evicted {row['lost']} (best {max_lost}) for "
            f"{row['gained']} (best {max_gained}) — a matched seed spent on a weaker "
            "canonical, which means a canonical is now being scored ABOVE the alias "
            "surface that earned it")
    # Positive control: a displacement really is being paid, so the loop above is not a
    # green over an empty list. `autonomy-pipeline` is the query that pays, and its
    # exact shape is pinned: `Autonomy Data Pipeline` was already in the ranking at
    # 0.3333 (rank 12, token overlap), the fold raised it to 0.5 (its alias surface
    # `Pipeline`), and the width then dropped `Lloyd's autonomy pipeline` at 0.3333.
    # The gold entity was promoted and the tail name gave way to it — the ordering
    # doing its job, not a match traded away.
    paying = [row["id"] for row in displaced]
    assert paying == ["autonomy-pipeline"], displaced
    row = displaced[0]
    assert row["gained"] == [["Autonomy Data Pipeline", 0.3333, 0.5]], row
    assert row["lost"] == [["Lloyd's autonomy pipeline", 0.3333]], row


_CORPUS_SCRIPT = """
import json, sys
sys.path.insert(0, '.')
import yaml
from agent_mcp.retrieval import extract_entities_from_query
from agent_mcp import vault
import eval.run_eval as ev
from app.kg_store import store

qs = yaml.safe_load(open('eval/vault_recall_queries.yaml'))['queries']
recs = []
for q in qs:
    seeds = [e for e, _ in (extract_entities_from_query(q['query']) or [])
             [:vault.RECALL_SEED_TOP_K]]
    recs.append({'id': q['id'], 'category': q.get('category', '?'),
                 'seeds_extracted': seeds,
                 'expected': {'entities': q.get('expect_entities') or []}})
anchorless = ev.anchorless_queries(recs)
# The pre-#1319 half is the first 20 entries in file order, which
# tests/test_eval_corpus_guard.py::test_the_gold_set_is_big_enough_for_its_own_power_claim
# pins as exactly the original 20 ids. Splitting here rather than re-listing them
# keeps one source of truth for what "the original corpus" means.
anchorless_first20 = [a for a in anchorless if a in {q["id"] for q in qs[:20]}]

st = store()
names = set(st.entities.all())
unresolvable = []
for q in qs:
    for e in (q.get('expect_entities') or []):
        c = st.resolve(e)
        if c is None or c not in names:
            unresolvable.append({'id': q['id'], 'expect': e, 'resolved': c})
print(json.dumps({'anchorless': anchorless,
                  'anchorless_first20': anchorless_first20,
                  'n_queries': len(qs),
                  'unresolvable': unresolvable}))
"""


def test_the_anchorless_residue_survives_the_corpus_growth_and_is_pinned():
    """#1260's five are still exactly the anchorless ones among the original 20.

    Clause 1's number was 6 → 5 over the 20-query corpus: `graph-quality` is the
    alias case that contract removed (the query says `relationship graph`, the
    alias table resolves that to `Knowledge Graph`, and the gold entity IS
    `Knowledge Graph`), and the five survivors have gold entities related to the
    query only semantically — no lexical path exists at any seed budget.

    #1319 grew the corpus to 87 queries and 19 of the 67 additions are anchorless
    for the same reason — a gold entity no seed extractor can reach lexically — so
    the residue is 24 of 87 and the entity leg's measured ceiling moves from
    (20−5)/20 = 0.750 to (87−24)/87 = 0.724. Both halves are pinned: the original
    five by id (a gold that silently stops, or starts, being lexically reachable is
    exactly what #1260 watches), and the grown residue in full, because that count
    is the denominator every baseline artifact carries. Reaching zero remains
    #1164's recall arm, not this contract.
    """
    out = _run_against_the_live_corpus(_CORPUS_SCRIPT)
    assert out["anchorless_first20"] == [
        "kg-maintenance-tasks", "qwen38-local-serving", "relationships-location",
        "memory-persistence", "robotics-projects"], out["anchorless_first20"]
    assert out["anchorless"] == [
        "kg-maintenance-tasks", "qwen38-local-serving", "relationships-location",
        "memory-persistence", "robotics-projects",
        "browser-tool-validation", "three-d-printing-calibration",
        "config-yaml-readonly", "dream-to-skill-edit", "automod-to-entity-guard",
        "job-that-changes-its-own-code", "stop-auto-merging-entities",
        "facts-that-contradict", "browser-tool-falls-back",
        "skill-that-never-improves", "gpu-ram-thin-should-not-reboot",
        "alarm-comes-back-after-fixed", "djev-decision-engine-integration",
        "ambient-prefetch-ttl-reclaim", "isaac-gr00t-n17",
        "retrieval-seed-anchoring-contract", "eval-corpus-naming-conventions",
        "eval-north-star-candidate", "retrieval-eval-item-1085",
    ], out["anchorless"]
    ceiling = (out["n_queries"] - len(out["anchorless"])) / out["n_queries"]
    assert (len(out["anchorless"]), out["n_queries"], round(ceiling, 3)) == (
        24, 87, 0.724
    ), f"{len(out['anchorless'])} anchorless of {out['n_queries']} queries"


def test_every_gold_entity_name_resolves_to_an_entity_the_store_can_return():
    """Clause 3: no expectation names something the store cannot return.

    Every `expect_entities` string must resolve through `store().resolve` onto a
    name that IS an entity row. `Nightly Reflection` was the last offender — a real
    directory, but an ALIAS row whose canonical is `Nightly Reflection Pipeline`, so
    the scorer was asked to find a spelling retrieval does not emit. Query TEXT is
    unchanged throughout: the audit rule at the top of the corpus file forbids
    substituting an easier question, and editing a query would silently change what
    every other test in this file measures.
    """
    out = _run_against_the_live_corpus(_CORPUS_SCRIPT)
    assert out["unresolvable"] == [], out["unresolvable"]
