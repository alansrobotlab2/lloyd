"""#541 — the counterfactual-perturbation rater on the retrieval eval.

Two numbers per query, both over entity/fact-level output only:

  counterfactual_moved_rate   the changed constraint actually moved what the
                              retriever pulled;
  counterfactual_pinned_rate  the constraints that did NOT change pulled the
                              same rows they did before.

The pinned half is the one a doc-level metric cannot see: the doc corpus is the
same vault either way, so a forward-only "did it change" rater rewards an
extractor that churns everything on every edit. Both directions, or neither
number means anything.

These tests pin the committed perturbation records (one per query, deterministic,
never re-derived from live graph data at eval time so a nightly diff stays a
diff) and the two metric definitions.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.counterfactual as cf  # noqa: E402
import eval.run_eval as ev  # noqa: E402

QUERIES = ROOT / "eval" / "vault_recall_queries.yaml"
RECORDS = ROOT / "eval" / "counterfactual_perturbations.yaml"
GENERATOR = ROOT / "eval" / "counterfactual.py"


def _specs():
    return yaml.safe_load(QUERIES.read_text())["queries"]


# The rater is sized against the live corpus, never a literal (#763 clause 4).
# This used to carry `MIN_CORPUS_N = 78` — the floor the trend audit's 80%-power
# claim needs (#1319, the exact McNemar search in scripts/eval_trend_stats.py) —
# and that assertion belonged where the corpus is under review, not in the
# rater's suite: here it could only fail for a reason this file has no power to
# fix, so every corpus resize reddened the counterfactual tests, and a suite
# that reddens on an unrelated change is a suite nobody reads. The power floor is
# still enforced, once, in tests/test_eval_corpus_guard.py::
# test_the_gold_set_is_big_enough_for_its_own_power_claim via
# GOLD_SET_MIN_QUERIES.
CORPUS_N = len(_specs())


# ── the committed perturbation set ───────────────────────────────────────────

def test_one_perturbation_per_query_with_the_declared_shape():
    """One variant per query, one changed axis each, over the WHOLE corpus.

    The count is the live corpus size, never a literal 20. The generator was
    written for the 20-query corpus and its records raise `ValueError` for an id
    with no PLAN entry, so a corpus change that forgets the rater leaves the
    nightly scoring one number of queries while its summary reports
    `counterfactual_n_moved` for another — one summary describing two different
    corpora. The corpus-size floor behind the trend audit's power claim (#1319)
    is guarded in `tests/test_eval_corpus_guard.py`, where resizing the corpus is
    the change under review; asserting it here could only redden the rater's own
    suite for a reason this file has no power to fix (#763 clause 4).
    """
    specs = {s["id"]: s for s in _specs()}
    recs = cf.load_records(RECORDS)
    assert len(recs) == len(specs) == CORPUS_N, (
        f"records {len(recs)}, corpus {CORPUS_N}")
    assert set(recs) == set(specs)
    for qid, rec in recs.items():
        for key in ("axis_changed", "old_value", "new_value",
                    "expected_to_move", "expected_pinned"):
            assert key in rec, f"{qid} missing {key}"
        assert rec["axis_changed"] in cf.AXES, (qid, rec["axis_changed"])
        assert isinstance(rec["expected_to_move"], list)
        assert isinstance(rec["expected_pinned"], list)


def test_plan_covers_every_query_and_nothing_outside_it():
    """The PLAN/corpus join, as a set difference that names EVERY gap, plus the
    size comparison a corpus resize moves together with it (#763 clause 4).

    `build_perturbations` raises on the first id with no plan entry, so it stops
    there; a growth that added 67 queries would be reported one id at a time.
    """
    specs = _specs()
    ids = {s["id"] for s in specs}
    assert sorted(set(cf.PLAN) - ids) == [], "PLAN entries with no query in the corpus"
    assert sorted(ids - set(cf.PLAN)) == [], "corpus queries with no PLAN entry"
    assert len(cf.PLAN) == len(specs), (
        f"the plan holds {len(cf.PLAN)} entries, the live corpus "
        f"{len(specs)} — the rater and the eval are describing different sets")


def test_cli_check_reports_no_drift():
    """`python eval/counterfactual.py --check`: the committed file IS the generator.

    Goes through the same entry point a person runs rather than only the
    in-process comparison, so a `--write` that emits something `--check` cannot
    reproduce — a header, a key order, a path the generator would not write — is
    caught here.
    """
    out = subprocess.run([sys.executable, str(GENERATOR), "--check"],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "records match the generator" in out.stdout, out.stdout


# ── the --verify audit, over a real store ────────────────────────────────────
#
# Everything under this heading seeds a store; nothing here injects a dict. The
# audit's entire job is to read the alias map, so a test that hands
# `verify_siblings` a hand-written resolver proves nothing about the one
# production uses — which is exactly how `--verify` shipped unable to run at
# all: `select distinct canonical` returns a one-column ROW, `.lower()` on that
# tuple raises AttributeError, and every sibling claim in this file rested on a
# fake that cannot fail that way (#1046).

#: An alias SURFACE row. Proves the resolver reads the aliases table at all,
#: and certifies the `tts-voice-cloning` swap by its surface, not by its name.
SEEDED_ALIAS = ("Piper TTS", "Piper Text-to-Speech")
#: A canonical registered as an entity with NO alias row: the case the table
#: alone cannot answer, and the case #762 measured as a manufactured label.
SEEDED_ENTITY = "Groundskeeper Research"
#: The two committed swaps those two rows certify, in the order the store
#: reaches them.
SEEDED_VERIFIED = ("tts-voice-cloning", "research-queue-to-vault-note")


def _entity_axis_records():
    return [r for r in cf.load_records(RECORDS).values()
            if r["axis_changed"] == cf.ENTITY_AXIS]


def _verify_store(tmp_path, *, alias=SEEDED_ALIAS, entity=SEEDED_ENTITY):
    """A real, seeded knowledge store, installed as the process default.

    `kg_store.configure` is the provisioning route — it creates the schema,
    which `store()` deliberately refuses to do for an absent path (#1236) — and
    conftest's `_isolate_default_store` restores the default path afterwards, so
    repointing the process default is the documented pattern for wanting a store
    (tests/test_api_contracts.py:551). This is what lets `main(["--verify"])`
    audit a graph this file controls: the derived store lives under `_pipeline/`
    and is gitignored, so it is absent from a self-modification worktree, and
    the fallback this test used to reach for built a one-table sqlite file no
    `store()` would ever open.
    """
    from app import kg_store

    db = kg_store.configure(tmp_path / "kg.sqlite")
    if alias:
        db.aliases.set(alias[0], alias[1], kind="manual", origin="test")
    if entity:
        db.entities.register(entity)
    return db


def _seeded_store_file(path, *, alias=SEEDED_ALIAS, entity=SEEDED_ENTITY):
    """Provision and seed a store at `path`, then close it so a child owns it.

    `KGStore(path)` is the provisioning constructor — the same one
    tests/test_kg_store.py:27 builds its `db` fixture with — as opposed to
    `store()`, which refuses to invent a schema (#1236).
    """
    from app.kg_store import KGStore

    db = KGStore(path)
    if alias:
        db.aliases.set(alias[0], alias[1], kind="manual", origin="test")
    if entity:
        db.entities.register(entity)
    db.close()
    return path


def _run_verify_script(store):
    """One real `python eval/counterfactual.py --verify` process.

    `LLOYD_KG_DB` (`app/paths.py:49`) is the documented override for the store
    path, which is what makes the child deterministic: the derived store under
    `_pipeline/` is gitignored, so it exists in the main checkout and not in a
    self-modification worktree, and pointing the child at a file this test wrote
    means the same assertions hold in both trees.
    """
    env = dict(os.environ)
    if store is not None:
        env["LLOYD_KG_DB"] = str(store)
    return subprocess.run([sys.executable, str(GENERATOR), "--verify"],
                          capture_output=True, text=True, timeout=600, env=env)


def _store_where_every_swap_resolves(tmp_path):
    """A store holding one entity per entity-axis `new_value`, so every
    swapped-in value is a registered canonical and the audit has nothing to
    flag — the shape the live store measures today (39 entity-axis pairs,
    0 unverified, `all_lower()` 27,428 entries, read 2026-09-22)."""
    db = _verify_store(tmp_path, alias=None, entity=None)
    for rec in _entity_axis_records():
        db.entities.register(rec["new_value"])
    return db


#: `eval/` scripts that open qmd's retrieval index with sqlite3 — a different
#: database from the knowledge store this rule is about. Each is a measurement
#: harness that copies or builds a scratch index; the test below proves none of
#: their connects names a knowledge store.
QMD_INDEX_READERS = {
    "embed_side_index.py",          # #1493: side index built beside a copy
    "contextual_titles.py",         # #1494: sub-corpus copy with situating titles
    "contextual_full_corpus.py",    # #1494: full-corpus scratch indexes
}


def test_no_script_under_eval_opens_the_knowledge_store_file():
    """`eval/` reaches the graph through `app.kg_store`, never through sqlite3.

    The rule is stated in CLAUDE.md — "Nothing opens the store except
    `app.kg_store`" — and `--verify` was the last violator: it opened
    `VAULT_KG_DB` read-only and hand-queried `aliases`. Living that close to raw
    sqlite is what let one expression — `.lower()` over a one-column ROW instead
    of the row's field — raise AttributeError on every invocation from the file's
    first commit (`af1e8c1`, 2026-09-09) until an unrelated gold-set commit
    hand-indexed it (`98fa216b`, 2026-09-20), with the suite green throughout:
    no test called the production resolver. Going through the store is also what
    makes an absent derived store raise `StoreUnavailable` rather than answer
    zero rows (#1236), so a missing build can never read as a clean audit.
    """
    scanned = sorted((p, p.read_text()) for p in (ROOT / "eval").rglob("*.py"))
    # Denominator first. An empty or renamed `eval/` would leave `offenders`
    # empty and the assertion below green — a scan whose zero cannot be
    # distinguished from "the scan saw nothing" is no verdict, so the file this
    # rule is about has to be among what the scan read.
    assert "counterfactual.py" in [p.name for p, _ in scanned], scanned
    offenders = [p.name for p, text in scanned
                 if ("import sqlite3" in text or "sqlite3.connect" in text)
                 and p.name not in QMD_INDEX_READERS]
    assert offenders == [], f"{offenders} open the knowledge store directly"
    # The named exceptions open qmd's OWN index (a VACUUM copy of it, or a
    # scratch index they built), never a knowledge store: no connect line of
    # theirs may name one, so the exemption cannot grow into the thing it exempts.
    for p, text in scanned:
        if p.name in QMD_INDEX_READERS:
            bad = [ln.strip() for ln in text.splitlines()
                   if "sqlite3.connect" in ln and ("kg" in ln.lower() or "KG_DB" in ln)]
            assert bad == [], f"{p.name} connects to a knowledge store: {bad}"
    # And the predicate is not vacuous: the module that owns the boundary trips
    # it, through a different path than the one under audit.
    kg_text = (ROOT / "app" / "kg_store.py").read_text()
    assert "import sqlite3" in kg_text or "sqlite3.connect" in kg_text


def test_alias_resolver_resolves_a_surface_and_a_canonical_with_no_row(tmp_path):
    """The production resolver, end-to-end over a seeded store — no fake.

    Both legs of `aliases.all_lower()` have to work, because the audit needs
    both: a registered alias SURFACE resolves to its canonical, and a canonical
    with no alias row at all resolves to ITSELF. The second leg is the reason
    the resolver reads the store's map instead of the `aliases` table — that
    table stores variant surfaces, so an entity nobody ever aliased is simply
    absent from it, and a table-only resolver calls it unknown. #762 measured
    that false negative inventing an `entity_swap_seed_set_unchanged` label for
    `qmd`, whose five surfaces are all entities with no alias row. Case and
    surrounding whitespace still fold, and an unknown name is still None.
    """
    db = _verify_store(tmp_path)
    assert db.aliases.for_canonical(SEEDED_ENTITY) == []      # the premise: no row
    resolve = cf.alias_resolver()
    assert resolve(SEEDED_ALIAS[0]) == SEEDED_ALIAS[1]        # via the surface row
    assert resolve(SEEDED_ENTITY) == SEEDED_ENTITY            # via entities.name
    assert resolve(f"  {SEEDED_ENTITY.lower()}\n") == SEEDED_ENTITY
    assert resolve("no such thing") is None
    assert resolve("") is None


def test_verify_siblings_reaches_a_verdict_on_the_committed_corpus(tmp_path):
    """The audit runs to a verdict of ZERO over the committed corpus.

    With one entity registered per entity-axis `new_value`, every swapped-in
    value is a canonical and each resolves to one its `old_value` does not — the
    risk-1 rule #537 spells out for exactly these labels, and the verdict the
    live store returns today (39 entity-axis pairs, 0 unverified, read
    2026-09-22). This test used to read `cf.kg_db_path()` and, when the derived
    store was absent — always, inside a worktree — fall back to hand-writing a
    one-table sqlite file; the audit now reads the store, so the store is what
    the test seeds.
    """
    _store_where_every_swap_resolves(tmp_path)
    unverified = cf.verify_siblings(list(cf.load_records(RECORDS).values()),
                                    cf.alias_resolver())
    assert [r["id"] for r in unverified] == [], unverified


def test_verify_cli_prints_a_verdict_over_a_seeded_store(tmp_path, capsys):
    """`main(["--verify"])` prints its verdict line and exits 1 — it used to raise.

    Seeded PARTIALLY — two swaps certified, the rest unresolvable — so the
    printed fraction is the store's answer and not a tautology. Both the exit
    code and the `N of M entity-axis pairs unverified against <db>` line are the
    contract: the AttributeError at the old resolver meant neither was ever
    printed, against any store, from the day the file landed.
    """
    db = _verify_store(tmp_path)
    rc = cf.main(["--verify"])
    out = capsys.readouterr().out
    verdict = re.fullmatch(
        r"(\d+) of (\d+) entity-axis pairs unverified against (.*)",
        out.strip().splitlines()[-1])
    assert verdict, out
    n_unverified, n_pairs, against = verdict.groups()
    all_ids = {r["id"] for r in _entity_axis_records()}
    assert int(n_pairs) == len(all_ids)
    assert int(n_unverified) == len(all_ids) - len(SEEDED_VERIFIED)
    assert Path(against) == db.path
    assert sorted(re.findall(r"UNVERIFIED ([\w-]+):", out)) == sorted(
        all_ids - set(SEEDED_VERIFIED)), out
    assert rc == 1, out


def test_verify_as_a_script_resolves_through_the_store(tmp_path):
    """One REAL `python eval/counterfactual.py --verify`, over a store it did not pick.

    This is the boundary the in-process calls cannot reach. Run as a script the
    file has `sys.path[0] == eval/`, so `from app import kg_store` only works
    because `_kg_store_module` inserts the repo root first — under pytest the
    root is already on `sys.path`, so every `cf.main(["--verify"])` call passes
    with or without that line. And `--verify` is only ever invoked as a script:
    a whole-checkout grep finds no automated caller, so a broken import here is
    the audit silently dying again, in the one shape anyone would use.

    Same partial seed as the in-process test — two rows, so two swaps certify and
    the rest do not — read by the child through `LLOYD_KG_DB`. A failure here
    reads as a traceback or a `ModuleNotFoundError: app`, which is exactly the
    class of death #1046 is about.
    """
    path = _seeded_store_file(tmp_path / "kg.sqlite")
    out = _run_verify_script(path)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "Traceback" not in out.stderr, out.stderr
    all_ids = {r["id"] for r in _entity_axis_records()}
    verdict = re.fullmatch(
        r"(\d+) of (\d+) entity-axis pairs unverified against (.*)",
        out.stdout.strip().splitlines()[-1])
    assert verdict, out.stdout
    n_unverified, n_pairs, against = verdict.groups()
    assert int(n_pairs) == len(all_ids)
    assert int(n_unverified) == len(all_ids) - len(SEEDED_VERIFIED)
    assert Path(against) == path, out.stdout
    flagged = set(re.findall(r"UNVERIFIED ([\w-]+):", out.stdout))
    assert flagged == all_ids - set(SEEDED_VERIFIED), out.stdout


def test_verify_as_a_script_reports_an_absent_store_without_a_traceback(tmp_path):
    """A store that is not there exits 2 with a sentence, not a stack.

    The audit's other real-world state: `_pipeline/vault-derived/kg.sqlite` is
    gitignored, so in a self-modification worktree — and on any machine where the
    nightly rebuild has not run — it simply does not exist. The raw-connection
    version answered that with `db.exists()` and a line of its own; the store
    raises `StoreUnavailable` instead (#1236), because the connection route could
    also invent an empty database and report 39 unverified pairs from a file that
    was never built. So this pins both halves: `main` catches it and exits 2, and
    nothing escapes as a traceback — an uncaught raise here would exit 1, the
    same code the audit returns when it genuinely finds an unfair swap.
    """
    missing = tmp_path / "not-built" / "kg.sqlite"
    out = _run_verify_script(missing)
    assert "Traceback" not in out.stderr, out.stderr
    assert out.returncode == 2, out.stdout + out.stderr
    assert "no alias table to audit the siblings against" in out.stdout, out.stdout
    assert "unverified against" not in out.stdout, out.stdout


def test_verify_cli_exits_zero_when_every_swap_resolves(tmp_path, capsys):
    """`0 of N … unverified` and exit 0 — the audit's quiet branch, also pinned.

    Same route as the failing case above with a store that answers every
    swapped-in value. Without it the only exit code anyone had ever seen from
    `--verify` would be the loud one, and `return 1 if bad else 0` has two
    halves.
    """
    _store_where_every_swap_resolves(tmp_path)
    assert cf.main(["--verify"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^0 of \d+ entity-axis pairs unverified against ", out, re.M), out
    assert "UNVERIFIED" not in out


def test_verify_cli_count_is_the_store_backed_audit_count(tmp_path, capsys):
    """--verify's number IS `verify_siblings` driven by `aliases.all_lower()`.

    Same store, measured twice: once through the CLI, once here, with a resolver
    built directly from `store().aliases.all_lower()` — the map, and the one
    helper that binds a map to the resolver shape, not a re-typed copy of it, so
    what this compares is the product and not a duplicate of it. The equality is only
    worth pinning because a third resolver disagrees — one built from the
    `aliases` table alone, surface_lc to canonical with no entity self-identity,
    which is what the pre-fix resolver was. It loses
    `research-queue-to-vault-note`: `Groundskeeper Research` is a registered
    entity holding no alias row, so the table has nothing to answer with, and
    #762 measured that exact gap manufacturing a label rather than a verdict.
    """
    db = _verify_store(tmp_path)
    from app import kg_store

    rc = cf.main(["--verify"])
    out = capsys.readouterr().out
    printed = int(re.match(r"(\d+) of ", out.strip().splitlines()[-1]).group(1))
    assert rc == 1, out

    recs = list(cf.load_records(RECORDS).values())
    direct = {r["id"] for r in cf.verify_siblings(
        recs, cf._resolver_from_map(kg_store.store().aliases.all_lower()))}
    assert printed == len(direct)
    assert direct == {r["id"] for r in cf.verify_siblings(
        recs, cf.alias_resolver())}, "the CLI and the store disagree on the same file"

    table_only = {r["surface_lc"]: r["canonical"] for r in db.aliases.rows()}
    only_table = {r["id"] for r in cf.verify_siblings(
        recs, lambda name: table_only.get((name or "").strip().lower()))}
    assert only_table == direct | {SEEDED_VERIFIED[1]}, (direct, only_table)


def test_a_scored_run_reports_counterfactual_n_over_the_whole_corpus(monkeypatch):
    """One scored run over the live corpus rates EVERY query, not 20 of them.

    The clause this pins is the one a yaml-only growth cannot satisfy: the
    summary reports `counterfactual_n_moved`, and if any query id is missing
    from the plan the runtime degrades it to `counterfactual_error: "no
    perturbation record for this query id"` and drops it from the denominator.
    The nightly then prints one number over 20 queries and another over 87 and
    reads as though it described a single corpus. So the whole production loop
    runs here — `run_eval` with only the retriever injected, then `summarize` —
    and its denominator has to equal the corpus size.
    """
    specs = _specs()

    def retrieve(params):
        # The rater reads only fact-attributed entities and their text, so the
        # stub echoes the query text as the ATTRIBUTED ENTITY: an entity swap
        # then finds its swapped-in sibling among the rows the variant added, a
        # swap with no named target sees a different attributed set, and a
        # pinned constraint matches the same single text in both arms.
        return {"documents": [{"path": "knowledge/whatever.md"}],
                "entities": [], "entity_facts": [],
                "facts": [{"entity": params["query"], "text": "one fact"}],
                "graph_neighbors_used": []}

    monkeypatch.setattr(ev, "_vault_recall", lambda params, **kw: retrieve(params))
    out = ev.summarize(ev.run_eval(specs, limit=len(specs)))["overall"]
    assert out["n_queries"] == CORPUS_N
    assert out["counterfactual_n_moved"] == CORPUS_N, (
        "the scored run rated fewer queries than the corpus holds — the rater "
        "is describing a smaller corpus than the eval is")
    assert out["counterfactual_moved_rate"] == 1.0
    # A twin that was never recalled would leave the pinned leg empty, so a
    # non-zero pinned denominator is the evidence both arms ran.
    assert out["counterfactual_n_pinned"] > 0


def test_each_perturbation_changes_exactly_one_named_constraint():
    """The variant differs from the original by exactly one substitution.

    A generator that quietly changes two things measures two things, and the
    axis attribution — the whole point of the rater — is gone.
    """
    specs = {s["id"]: s for s in _specs()}
    for qid, rec in cf.load_records(RECORDS).items():
        query = specs[qid]["query"]
        old, new = rec["old_value"], rec["new_value"]
        assert old.lower() in query.lower(), f"{qid}: old_value not in the query"
        rebuilt = cf.apply_perturbation(query, rec)
        assert rebuilt != query, f"{qid}: perturbation changed nothing"
        assert rebuilt == cf.apply_perturbation(query, rec), f"{qid}: not deterministic"
        before = query.lower().count(old.lower())
        after = rebuilt.lower().count(old.lower())
        # an overlap (QMD -> QMD Index) legitimately leaves the substring inside
        # the replacement; anything else means the swap did not take.
        assert after == before - 1 or old.lower() in new.lower(), qid
        assert rebuilt.lower().count(new.lower()) >= 1, qid


def test_generator_is_deterministic_and_matches_the_committed_file():
    """The generator is committed and reproducible; the records are committed.

    A record set that re-derived itself from live graph data each night would
    change the measurement underneath the trend line — the same defect the
    nightly compare step guards against for the query set itself.
    """
    built = {r["id"]: r for r in cf.build_perturbations(_specs())}
    committed = cf.load_records(RECORDS)
    assert set(built) == set(committed)
    for qid, rec in built.items():
        assert rec == committed[qid], f"{qid}: committed record drifted from generator"


def test_entity_swap_siblings_are_declared_as_alias_table_pairs():
    """Risk 1 in #541: an unfair sibling manufactures false failures, so every
    entity-axis pair declares where its sibling came from. Resolution against
    the live table is a separate, runnable check (verify_siblings) — this one
    stays hermetic so the suite does not need the store."""
    recs = cf.load_records(RECORDS)
    entity_axis = {q: r for q, r in recs.items() if r["axis_changed"] == "entity"}
    assert entity_axis, "no entity-swap perturbations — the #537 evidence cannot exist"
    for qid, rec in entity_axis.items():
        assert rec.get("sibling_source") == "kg_alias_table", qid
        assert rec["old_value"] != rec["new_value"], qid
    for rec in recs.values():
        if rec["axis_changed"] != "entity":
            assert "sibling_source" not in rec, "non-entity axis claims a sibling"


def test_sibling_verifier_flags_a_pair_that_is_not_in_the_table():
    """The audit half of risk 1, tested against an injected resolver rather than
    the live 3.9k-row table."""
    table = {"knowledge graph": "Knowledge Graph", "entity graph": "Entity Graph",
             "kg": "Knowledge Graph", "qmd": "QMD"}
    resolve = lambda name: table.get((name or "").strip().lower())
    good = {"axis_changed": "entity", "old_value": "Knowledge Graph",
            "new_value": "Entity Graph"}
    invented = {"axis_changed": "entity", "old_value": "Knowledge Graph",
                "new_value": "Frobnicator Prime"}
    # two surfaces of ONE canonical is a rename, not a sibling swap
    rename = {"axis_changed": "entity", "old_value": "Knowledge Graph", "new_value": "KG"}
    assert cf.verify_siblings([good], resolve) == []
    bad = cf.verify_siblings([invented, rename, good], resolve)
    assert [r["new_value"] for r in bad] == ["Frobnicator Prime", "KG"]


def test_non_entity_axes_are_not_held_to_the_sibling_audit():
    """date/qualifier/artifact swaps name no sibling; the audit has nothing to
    check and must not report them."""
    rec = {"axis_changed": "date", "old_value": "this week", "new_value": "last quarter"}
    assert cf.verify_siblings([rec], lambda name: None) == []


def test_perturbation_generation_touches_no_ground_truth(tmp_path):
    """Guardrail in #541: generation must not touch the query set or the corpus.

    The generator is a pure function of the query text plus the checked-in axis
    plan; it opens no store and writes only where it is told to.
    """
    before = QUERIES.read_text()
    built = cf.build_perturbations(_specs())
    assert QUERIES.read_text() == before
    out = tmp_path / "perts.yaml"
    cf.emit_records(built, out)
    assert cf.load_records(out) == {r["id"]: r for r in built}


# ── the two metrics ──────────────────────────────────────────────────────────

def _result(entities, fact_text="f"):
    return {"facts": [{"entity": e, "text": f"{fact_text} {e}"} for e in entities],
            "graph_expanded_facts": [], "graph_neighbors_used": [],
            "documents": [{"path": "a/b.md"}]}


def test_moved_is_scored_on_retrieved_entities_not_doc_paths():
    """Risk 3 in #541: the doc corpus is the same vault, so a doc-level pinned
    comparison would read as spurious success. Only entity/fact output counts."""
    rec = {"axis_changed": "entity", "old_value": "Knowledge Graph",
           "new_value": "Entity Graph", "expected_to_move": ["Entity Graph"],
           "expected_pinned": ["Data Pipeline"]}
    orig = _result(["Knowledge Graph", "Data Pipeline"])
    moved_entities = _result(["Entity Graph", "Data Pipeline"])
    same_entities = _result(["Knowledge Graph", "Data Pipeline"])

    out = cf.score_pair(rec, orig, ["Old Seed"], moved_entities, ["New Seed"])
    assert out["counterfactual_moved"] is True
    assert out["counterfactual_pinned"] is True

    stayed = cf.score_pair(rec, orig, ["Old Seed"], same_entities, ["New Seed"])
    assert stayed["counterfactual_moved"] is False
    assert stayed["counterfactual_pinned"] is True
    assert stayed["retrieved_unchanged"] is True
    # every arm above returned the identical document path: docs are not signal
    assert stayed["retrieved"] == stayed["retrieved_variant"]


def test_pinned_fails_when_an_unchanged_axis_churns():
    """The backward half. A retriever that re-pulls everything on any edit is
    not sensitive to the constraint — it is noise, and a moved-only rater would
    score it a perfect 1.0."""
    rec = {"axis_changed": "entity", "old_value": "vLLM", "new_value": "TensorRT-LLM",
           "expected_to_move": ["TensorRT-LLM"], "expected_pinned": ["lloyd"]}
    orig = _result(["vLLM", "Lloyd"], fact_text="same")
    churned = _result(["TensorRT-LLM", "Lloyd"], fact_text="different")
    out = cf.score_pair(rec, orig, ["vLLM"], churned, ["TensorRT-LLM"])
    assert out["counterfactual_moved"] is True
    assert out["counterfactual_pinned"] is False


def test_unscored_pinned_is_none_and_never_a_vacuous_pass():
    """A query with nothing left to pin contributes to moved_rate only. Scoring
    it as a pass would inflate pinned_rate with queries that tested nothing."""
    rec = {"axis_changed": "entity", "old_value": "QMD", "new_value": "QMD Index",
           "expected_to_move": ["QMD Index"], "expected_pinned": []}
    out = cf.score_pair(rec, _result(["QMD"]), ["QMD"], _result(["QMD Index"]), ["QMD Index"])
    assert out["counterfactual_moved"] is True
    assert out["counterfactual_pinned"] is None
    assert out["pinned_unscored"] is True


def test_non_entity_axes_score_moved_as_any_change():
    """date / qualifier / artifact axes name no new entity row, so the moved
    half is "the pulled entity-level output differs at all"."""
    rec = {"axis_changed": "date", "old_value": "this week", "new_value": "last quarter",
           "expected_to_move": [], "expected_pinned": ["Autonomy System"]}
    changed = cf.score_pair(rec, _result(["Autonomy System"]), ["s"],
                            _result(["Autonomy Data Pipeline"]), ["s"])
    assert changed["counterfactual_moved"] is True
    unchanged = cf.score_pair(rec, _result(["Autonomy System"]), ["s"],
                              _result(["Autonomy System"]), ["s"])
    assert unchanged["counterfactual_moved"] is False
    assert unchanged["counterfactual_pinned"] is True


def test_move_tolerates_the_stores_canonical_name_but_not_an_old_row():
    """Two things pull in opposite directions and the added-set settles both.

    Tolerant, because the store's canonical for 'Semantic Entity Resolution' is
    'semantic-entity-resolution-via-graph-embeddings' — refusing to call that a
    move manufactures a false failure (risk 1). Restricted to the rows the
    variant attributed and the original did not, because 'QMD' was already
    attributed when the query said QMD, and crediting it for a swap to
    'QMD Index' would score a no-change run as sensitive.
    """
    rec = {"axis_changed": "entity", "old_value": "QMD", "new_value": "QMD Search",
           "expected_to_move": ["QMD Search"], "expected_pinned": []}

    def res(entities):
        return _result(entities)

    # already-attributed 'QMD' must not satisfy a swap to 'QMD Search'
    no_move = cf.score_pair(rec, res(["QMD"]), ["QMD"], res(["QMD"]), ["QMD Search"])
    assert no_move["counterfactual_moved"] is False

    # a newly-attributed row whose canonical is longer still counts
    long_canon = {"facts": [{"entity": "semantic-entity-resolution-via-graph-embeddings",
                             "text": "x"}],
                  "graph_expanded_facts": [], "graph_neighbors_used": [],
                  "documents": []}
    out = cf.score_pair({"axis_changed": "entity", "old_value": "Entity Resolution Sweep",
                        "new_value": "Semantic Entity Resolution",
                        "expected_to_move": ["Semantic Entity Resolution"],
                        "expected_pinned": []},
                       res(["Entity Resolution Sweep"]), ["s"], long_canon, ["s"])
    assert out["counterfactual_moved"] is True


# ── the record the nightly compare step reads ────────────────────────────────

def test_summarize_carries_both_rates_and_its_own_n():
    """Each rate is averaged over the queries scoreable for that half, so
    summary carries those denominators beside the numbers."""
    def rec(moved, pinned):
        return {"id": "q", "category": "single", "latency_ms": 100.0, "error": None,
                "scoring": {"entity_hit": True, "doc_hit": True, "entity_recall": 1.0,
                            "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                            "fact_entity_recall": 1.0, "first_doc_rank": 1,
                            "counterfactual_moved_rate": moved,
                            "counterfactual_pinned_rate": pinned}}
    o = ev.summarize([rec(1.0, 1.0), rec(0.0, None), rec(1.0, 0.0)])["overall"]
    assert o["counterfactual_moved_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert o["counterfactual_pinned_rate"] == pytest.approx(0.5)
    assert o["counterfactual_n_moved"] == 3
    assert o["counterfactual_n_pinned"] == 2


def test_summarize_emits_both_keys_even_with_no_counterfactual_data():
    """Old records and the automod baseline arm carry no perturbation block.
    The keys must still be present — so the nightly compare reads a key that
    exists — with null rather than 0.0: absent is not zero."""
    rec = {"id": "q", "category": "single", "latency_ms": 10.0, "error": None,
           "scoring": {"entity_hit": True, "doc_hit": True, "entity_recall": 1.0,
                       "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                       "fact_entity_recall": 1.0, "first_doc_rank": 1}}
    o = ev.summarize([rec])["overall"]
    assert "counterfactual_moved_rate" in o and "counterfactual_pinned_rate" in o
    assert o["counterfactual_moved_rate"] is None
    assert o["counterfactual_n_moved"] == 0


def test_failure_labels_name_the_swaps_that_left_the_seed_set_alone():
    """The deliverable is a labelled defect list, and the #537-relevant label is
    an entity-name swap the seed extractor did not notice."""
    labelled = cf.label_failures([
        {"id": "swap-blind", "counterfactual": {"axis_changed": "entity",
                                                "seed_moved": False,
                                                "counterfactual_moved": False,
                                                "counterfactual_pinned": True}},
        {"id": "swap-seen", "counterfactual": {"axis_changed": "entity",
                                               "seed_moved": True,
                                               "counterfactual_moved": True,
                                               "counterfactual_pinned": True}},
        {"id": "pinned-churn", "counterfactual": {"axis_changed": "qualifier",
                                                  "seed_moved": False,
                                                  "counterfactual_moved": True,
                                                  "counterfactual_pinned": False}},
    ])
    by_id = {row["id"]: row for row in labelled}
    assert by_id["swap-blind"]["label"] == "entity_swap_seed_set_unchanged"
    assert "swap-blind" in cf.identity_keying_evidence(labelled)
    assert "swap-seen" not in cf.identity_keying_evidence(labelled)
    assert by_id["swap-seen"]["label"] is None
    assert by_id["pinned-churn"]["label"] == "pinned_axis_churned"


# ── #763: the pinned leg's two coverage errors, and its denominator ──────────

# The three entity swaps whose swapped-in sibling keeps part of the original
# term, so the surviving part is a constraint the perturbed query really does
# still name. `backlog-363` pins "Backlog" rather than the two-word
# "Backlog Item" for a measured reason, recorded at the PLAN entry: the original
# arm attributes `backlogtask363`, in which the normalized "backlogitem" is not a
# substring, so the two-word form scores `present=False->True` — the clause-2
# defect imported instead of removed.
SURVIVING_TYPE_PINS = {"qmd": "QMD", "backlog-363": "Backlog",
                       "entity-resolution-sweep": "Entity Resolution"}

# The rows each arm attributed for these three queries in
# `eval/baselines/nightly-20260921-20260921-063958.json`
# (`counterfactual.retrieved` and `.retrieved_variant`), copied here so the
# non-vacuity claim below is checkable without a store: a pin must match a row on
# BOTH sides, because `score_pair` also passes a pin that matched nothing in
# either arm (`present=False->False`) — the vacuity #763 leaves to a person.
MEASURED_ROWS = {
    "qmd": (["bun", "ldlibrarypath", "lexicalsearchbackend", "qmd",
             "qwen3embedding", "structuredsearch"],
            ["industrialaiassistant", "ldlibrarypath", "lexicalsearchbackend",
             "qmd", "qmdsearch", "qwen3embedding", "structuredsearch"]),
    "backlog-363": (["backlogtask363", "currenttask", "task363",
                     "task363tgsragmultihopreasoning", "task380kgdensification",
                     "tgsrag", "tgsragretrievallevers"],
                    ["313", "backlogitem313", "task313"]),
    "entity-resolution-sweep": (
        ["aliastable", "cleanup", "entityresolution", "entityresolutionscript",
         "entityresolutionsweep", "lloydv4classifier", "task320"],
        ["autonomytask67", "cleanup", "entityresolution", "entityresolutionscript",
         "lloydv4classifier", "robomd",
         "semanticentityresolutionviagraphembeddings"]),
}


def _perturbed_text(qid: str) -> str:
    """The variant text the generator produces for one corpus query."""
    spec = next(s for s in _specs() if s["id"] == qid)
    _axis, old, new, _move, _pins = cf.PLAN[qid]
    return cf.apply_perturbation(spec["query"],
                                 {"old_value": old, "new_value": new, "id": qid})


def _stubbed_run(monkeypatch, specs):
    """The production loop with only the retriever injected.

    The stub echoes the query text as the ATTRIBUTED ENTITY, so presence is
    judged against each arm's own text and nothing else — the same shape
    `test_a_scored_run_reports_counterfactual_n_over_the_whole_corpus` uses.
    """
    def retrieve(params):
        return {"documents": [{"path": "knowledge/whatever.md"}],
                "entities": [], "entity_facts": [],
                "facts": [{"entity": params["query"], "text": "one fact"}],
                "graph_neighbors_used": []}

    monkeypatch.setattr(ev, "_vault_recall", lambda params, **kw: retrieve(params))
    records = ev.run_eval(specs, limit=len(specs))
    return records, ev.summarize(records)


def test_the_three_entity_swaps_whose_type_survives_their_swap_are_scored(monkeypatch):
    """Clause 1: `qmd`, `backlog-363` and `entity-resolution-sweep` each carry a
    non-empty `expected_pinned` that survives into its own perturbed query, so
    those three ids no longer score `nothing_pinned`.

    Three things are asserted, because the first alone would pass on a pin
    invented out of thin air:

      * the pin is the PLAN's declared pin, and a case-insensitive substring of
        the variant text the generator will actually run;
      * it matches a row BOTH arms attributed in the shipping baseline, so the
        pass is evidence about retrieval rather than a pin that matched nothing
        in either arm;
      * the production loop now returns a non-null pinned leg for the id, which
        is what takes it out of `nothing_pinned` in `counterfactual_failures`.
    """
    specs = _specs()
    for qid, pin in SURVIVING_TYPE_PINS.items():
        assert cf.PLAN[qid][4] == [pin], (qid, cf.PLAN[qid][4])
        perturbed = _perturbed_text(qid)
        assert pin.lower() in perturbed.lower(), (qid, pin, perturbed)
        orig_rows, var_rows = MEASURED_ROWS[qid]
        assert cf._match(pin, {cf._norm(r) for r in orig_rows}), (qid, orig_rows)
        assert cf._match(pin, {cf._norm(r) for r in var_rows}), (qid, var_rows)

    records, summary = _stubbed_run(monkeypatch, specs)
    by_id = {r["id"]: r for r in records}
    labels = {row["id"]: row["label"] for row in cf.label_failures(records)}
    for qid in SURVIVING_TYPE_PINS:
        block = by_id[qid]["counterfactual"]
        assert block["pinned_unscored"] is False, qid
        assert block["counterfactual_pinned"] is not None, qid
        assert labels[qid] != "nothing_pinned", (qid, labels[qid])
    # The denominator still means "actually scored", so it equals the entries
    # that declare a pin — derived from the plan, never a number in this file.
    declared = sum(1 for entry in cf.PLAN.values() if entry[4])
    o = summary["overall"]
    assert o["counterfactual_n_pinned"] == declared, (o["counterfactual_n_pinned"],
                                                     declared)
    assert o["counterfactual_n_moved"] == len(specs), o["counterfactual_n_moved"]


def test_no_plan_entry_pins_a_term_its_own_swap_deletes_beyond_the_five_named():
    """Clause 2: the only entries whose declared pin is absent from their own
    perturbed query are the five enumerated non-entity ones, so
    `autonomy-pipeline` no longer pins "autonomy".

    `autonomy-pipeline` is the live false alarm this clause closes. Its swap
    rewrites "describe the autonomy pipeline end to end" into "describe the Data
    Pipeline end to end", so the variant text holds no "autonomy" to hold
    constant, and `nightly-20260917` and `nightly-20260921` each booked the row as
    `pinned_axis_churned / autonomy: present=True->False` — a defect reported on
    the one axis that was supposed to move. The five survivors are qualifier and
    artifact entries whose pin names an entity the query implies rather than
    repeats; they are legitimate expectations and `PINS_ABSENT_BY_DESIGN` says so
    per entry. No ENTITY-axis entry may be in that set, because on that axis a pin
    outside the variant text is by construction the constraint that moved.
    """
    specs = _specs()
    offenders = cf.pins_absent_from_their_own_swap(specs)
    assert sorted(offenders) == sorted(cf.PINS_ABSENT_BY_DESIGN), offenders
    assert set(cf.PINS_ABSENT_BY_DESIGN) == {
        "qwen38-local-serving", "yaml-scalar-block-indent",
        "check-that-cannot-see-its-input", "self-referential-check-catalogue",
        "skill-mining-to-promotion"}
    assert cf.PLAN["autonomy-pipeline"][4] == [], cf.PLAN["autonomy-pipeline"]
    assert "autonomy-pipeline" not in offenders
    # The pin is gone; the swap's own `old_value` keeps the word, because the
    # word is what the swap moves.
    assert "autonomy" not in str(cf.PLAN["autonomy-pipeline"][4]), \
        cf.PLAN["autonomy-pipeline"][4]
    for qid in cf.PINS_ABSENT_BY_DESIGN:
        assert cf.PLAN[qid][0] != cf.ENTITY_AXIS, (
            qid, "an entity-axis entry pins a term its own swap deletes")

    # The audit must be able to FAIL. `autonomy-pipeline` is the offender this
    # clause removes, so it is injected back as a synthetic sixth entry here and
    # has to come back named — otherwise the five could be passing because the
    # function never finds anything.
    probe_specs = [{"id": "probe", "query": "describe the autonomy pipeline"}]
    probe_plan = {"probe": ("entity", "autonomy pipeline", "Data Pipeline",
                            ["Data Pipeline"], ["autonomy"])}
    assert cf.pins_absent_from_their_own_swap(probe_specs, probe_plan) == {
        "probe": ["autonomy"]}
    # ... and a pin that DOES survive its own swap is not reported.
    ok_plan = {"probe": ("entity", "autonomy pipeline", "Data Pipeline",
                         ["Data Pipeline"], ["pipeline"])}
    assert cf.pins_absent_from_their_own_swap(probe_specs, ok_plan) == {}


def test_the_printed_counterfactual_line_shows_each_rate_out_of_the_query_set(monkeypatch,
                                                                              capsys):
    """Clause 3: the console summary prints `pinned=0.90 (n=50/81)`, and moved's
    out-of-total is on the same line.

    `(n=50)` alone — what `eval/run_eval.py` has printed since `af1e8c1`
    (2026-09-09) — shows the count but not the coverage. The two legs of this
    rater are scored over DIFFERENT query populations by design (an entry with no
    `expected_pinned` feeds moved only), so without the out-of-total a reader
    comparing the two trend lines is comparing an unknown pair of populations,
    which is the reading #763 was filed on. The unscored remainder has to stay
    VISIBLE: the fraction carries the shortfall, and the residual pair is not
    quietly promoted into the denominator.
    """
    specs = _specs()
    records, summary = _stubbed_run(monkeypatch, specs)
    ev.print_table(records, summary)
    out = capsys.readouterr().out
    o = summary["overall"]
    total = o["n_queries"]
    line = next((ln for ln in out.splitlines() if "counterfactual:" in ln), None)
    assert line is not None, out
    # Both denominators are fractions of the run's own query total, in the order
    # (moved, pinned), with no bare `(n=N)` left behind.
    assert re.findall(r"\(n=(\d+)/(\d+)\)", line) == [
        (str(o["counterfactual_n_moved"]), str(total)),
        (str(o["counterfactual_n_pinned"]), str(total))], line
    # The residual unscored pair shows as a shortfall rather than as coverage:
    # the pinned denominator is strictly under the total, and the two entries the
    # item leaves to a person are why.
    assert o["counterfactual_n_pinned"] < total, line
    for qid in ("inner-voice", "vault-recall"):
        assert cf.PLAN[qid][4] == [], qid


def test_the_counterfactual_suite_is_sized_by_the_corpus_it_reads():
    """Clause 4: no hard-coded corpus size anywhere in this file, so a corpus
    resize cannot redden the rater's suite.

    This file used to carry `MIN_CORPUS_N = 78` and assert the live corpus was at
    least it, which meant the rater's own tests failed whenever an unrelated job
    trimmed or grew `eval/vault_recall_queries.yaml`. 78 is the exact-McNemar
    floor for a 0.10 paired change at 80 % power (`scripts/eval_trend_stats.py`),
    a property of the CORPUS, and it is guarded once in
    `tests/test_eval_corpus_guard.py::test_the_gold_set_is_big_enough_for_its_own_power_claim`,
    where a resize is the change under review. What belongs HERE is agreement:
    plan entries, committed records and corpus queries all one size.
    """
    specs = _specs()
    assert len(cf.PLAN) == len(specs), (len(cf.PLAN), len(specs))
    assert len(cf.load_records(RECORDS)) == len(specs), (
        len(cf.load_records(RECORDS)), len(specs))
    src = Path(__file__).read_text()
    literals = re.findall(r"^(?:MIN_)?CORPUS_N\s*=\s*\d+", src, re.MULTILINE)
    assert literals == [], (
        f"the suite hard-codes a corpus size: {literals} — assert against "
        "`len(_specs())` instead")
