"""eval/run_eval.py must say which corpus it scored, and refuse an empty one.

The blind spot this pins: with `RECALL_GRAPH_RERANK = False` the graph never
reorders documents, and the document leg queries the qmd daemon at an absolute
external URL. So a run against an entirely empty fact tree and an empty graph
store produces a well-formed record with `errors: 0` whose mrr_doc, ndcg10,
doc_hit_rate and every per-category mrr_doc are IDENTICAL to a real run. Only
entity_hit_rate, entity_recall_avg and fact_entity_recall_avg collapse — three
numbers nobody reads first. An empty corpus was therefore indistinguishable
from a healthy one in the headline metrics.

Everything here goes through subprocess rather than importing the module:
`app.paths` reads LLOYD_FACTS_ROOT / LLOYD_KG_DB at IMPORT time, so an
in-process test would have to win a race with the import, and the CLI contract
(exit code, message, flag) is the thing that actually protects the operator.

The interpreter is resolved, never assumed. `.venvs/` is gitignored
(.gitignore:3), so a `git worktree` — including the automod round worktree the
gate runs the suite in — has no venv of its own. The module-level skip that used
to key off `ROOT/.venvs` existing therefore removed every test in this file from
exactly the tree where a regression here would have been caught before it
landed, and the run still reported green (#498 clause 4). Use the tree's own venv
when it has one; otherwise the interpreter already running pytest, which
provably has the dependencies: the rest of this suite imports `agent_mcp` out of
this tree in-process. `SCRIPT`, `cwd` and the `LLOYD_*` env overrides still point
every subprocess at THIS tree, so only the interpreter is borrowed, never the
code under test.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_VENV = ROOT / ".venvs" / "lloyd" / "bin" / "python"
PY = _VENV if _VENV.exists() else Path(sys.executable)
SCRIPT = ROOT / "eval" / "run_eval.py"
BASELINES = ROOT / "eval" / "baselines"


def _queries_file(tmp_path: Path) -> Path:
    """One query, with both expectation lists populated.

    print_table formats entity_recall_avg / doc_recall_avg unconditionally, and
    `_score` returns None for an expectation list that is empty — so a query
    with no expectations would crash the summary for reasons unrelated to what
    is under test.
    """
    p = tmp_path / "queries.yaml"
    p.write_text(
        "queries:\n"
        "  - id: guard-probe\n"
        "    query: what is lloyd\n"
        "    category: single\n"
        "    expect_entities: [Lloyd]\n"
        "    expect_docs: [lloyd]\n"
    )
    return p


def _provision_store(db: Path, env: dict) -> None:
    """Create the store this run is about to score, through the named route.

    Since #1236 the reader `store()` refuses an absent database rather than
    letting sqlite invent one, so a run that means "score against this empty
    store" has to create it, the same way `_run` creates the empty facts root
    beside it. An existing file (the unreadable-store test below) is left
    exactly as it is. Subprocess for the reason in the module docstring: this
    file imports nothing out of the tree it tests.
    """
    subprocess.run(
        [str(PY), "-c",
         "import sys; from app.kg_store import KGStore; KGStore(sys.argv[1]).close()",
         str(db)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
        check=True,
    )


def _run(tmp_path: Path, *args: str,
         kg_db: Path | None = None,
         facts_root: Path | None = None) -> subprocess.CompletedProcess:
    """Run the eval against an empty facts root unless one is named.

    `facts_root` exists for the #1250 tests below, which need the readable fact
    tree and the store's index of it to DISAGREE — the index is what
    `corpus.facts` counts, and the read path is what produced the six zeroed
    arms. Same env either way.
    """
    facts = facts_root if facts_root is not None else tmp_path / "facts"
    facts.mkdir(exist_ok=True)
    db = kg_db if kg_db is not None else tmp_path / "kg.sqlite"
    env = dict(os.environ)
    env["LLOYD_FACTS_ROOT"] = str(facts)
    env["LLOYD_KG_DB"] = str(db)
    env["LLOYD_VOICE_ALERTS"] = "0"
    if not db.exists():
        _provision_store(db, env)
    return subprocess.run(
        [str(PY), str(SCRIPT), "--queries", str(_queries_file(tmp_path)), *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300,
    )


def _cleanup(label: str) -> None:
    for f in BASELINES.glob(f"{label}-*.json"):
        f.unlink()


def test_empty_corpus_refuses_and_names_both_resolved_paths(tmp_path):
    """The refusal must name the tree it actually read.

    Both paths come from env overrides, so 'empty corpus' without them leaves
    the operator guessing which of two trees was scored — the whole failure is
    that the run read the wrong one."""
    proc = _run(tmp_path, "--label", "pytest-empty-corpus")
    out = proc.stdout + proc.stderr

    assert proc.returncode != 0, out
    assert str(tmp_path / "facts") in out, out
    assert str(tmp_path / "kg.sqlite") in out, out
    # It refused *before* scoring: no record was written, nothing was queried.
    assert not list(BASELINES.glob("pytest-empty-corpus-*.json"))
    assert "--allow-empty-corpus" in out


def test_allow_empty_corpus_completes_and_records_corpus_ok_false(tmp_path):
    """Measuring the no-graph baseline on purpose stays possible — that is how
    this blind spot was found — but the record says so."""
    label = "pytest-allow-empty"
    _cleanup(label)
    try:
        proc = _run(tmp_path, "--label", label, "--allow-empty-corpus")
        assert proc.returncode == 0, proc.stdout + proc.stderr

        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        rec = json.loads(written[0].read_text())

        assert rec["corpus_ok"] is False
        assert rec["corpus"]["facts_root"] == str(tmp_path / "facts")
        assert rec["corpus"]["kg_db"] == str(tmp_path / "kg.sqlite")
        assert rec["corpus"]["entity_dirs"] == 0
        assert rec["corpus"]["entities"] == 0
        assert rec["corpus"]["edges_active"] == 0
        # Provenance reaches the terminal too, not only the JSON.
        assert "[info] corpus" in proc.stdout
    finally:
        _cleanup(label)


def _production_knobs() -> dict:
    """Production's four retrieval defaults, read the way the eval reads them:
    from `agent_mcp.vault`, in a process of its own. This file is subprocess-only
    because `app.paths` reads LLOYD_FACTS_ROOT / LLOYD_KG_DB at import time, and
    restating the values here is the drift this round exists to remove."""
    code = (
        "import json;"
        "from agent_mcp import vault;"
        "print(json.dumps({'graph_rerank': vault.RECALL_GRAPH_RERANK,"
        " 'rerank_alpha': vault.RECALL_RERANK_ALPHA,"
        " 'graph_top_k': vault.RECALL_GRAPH_TOP_K,"
        " 'graph_hops': vault.RECALL_GRAPH_HOPS}))"
    )
    proc = subprocess.run([str(PY), "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_default_run_records_production_config(tmp_path):
    """The nightly CLI path is what the trend compares on, and #498 changed only
    how run_eval()'s defaults are DERIVED — so a default invocation must still
    record production's config for all four knobs.

    --allow-empty-corpus is here solely so the run completes without depending on
    the live fact tree; every knob asserted is still the default.

    `matches_production_defaults` is asserted True because a default run matches
    production on every knob the flag compares. Graph expansion is the exception
    and is no longer hidden inside that flag: production defaults it False, the
    eval runs it on, and #1000 moved the disagreement into its own
    `expand_graph_matches_production` field."""
    label = "pytest-default-config"
    _cleanup(label)
    try:
        proc = _run(tmp_path, "--label", label, "--allow-empty-corpus")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        rec = json.loads(written[0].read_text())

        for knob, value in _production_knobs().items():
            assert knob in rec, knob
            assert rec[knob] == value, (knob, rec[knob], value)
        assert rec["matches_production_defaults"] is True
        # The graph leg runs expanded by the eval's own choice, not production's.
        # Since #1000 the artifact says so beside the value rather than claiming
        # parity: RECALL_EXPAND_GRAPH is False, so this run matches production on
        # the six compared knobs and explicitly does not on this one.
        assert rec["expand_graph"] is True
        assert rec["expand_graph_matches_production"] is False
    finally:
        _cleanup(label)


def test_unreadable_store_is_its_own_failure_and_ignores_the_flag(tmp_path):
    """`StoreUnavailable` is 'I could not read it', not 'it is empty'.

    Collapsing the two would let --allow-empty-corpus — a flag that means the
    second — silently excuse the first, which is the exact substitution
    kg_store.StoreUnavailable exists to prevent."""
    bad = tmp_path / "not-a-database.sqlite"
    bad.write_bytes(b"this is not a sqlite file, not even close\n" * 64)

    for extra in ([], ["--allow-empty-corpus"]):
        proc = _run(tmp_path, "--label", "pytest-bad-store", *extra, kg_db=bad)
        out = proc.stdout + proc.stderr
        assert proc.returncode != 0, out
        assert "unreadable" in out.lower(), out
        assert str(bad) in out, out
        assert not list(BASELINES.glob("pytest-bad-store-*.json"))


CORPUS_KEYS = {"facts_root", "kg_db", "entity_dirs", "edges_total",
               "edges_active", "aliases", "entities", "facts"}


def _assert_corpus_shape(rec: dict) -> None:
    """The one assertion pair this file's shape test exists for, shared by both
    of its arms so the fallback arm cannot be a weaker test than the historical
    one."""
    assert CORPUS_KEYS <= set(rec["corpus"]), sorted(CORPUS_KEYS - set(rec["corpus"]))
    assert isinstance(rec["corpus_ok"], bool)


def test_a_healthy_run_record_carries_its_corpus(tmp_path):
    """Guards the *shape* of a run record: a reader that trusts `corpus` needs
    the keys to be there, and every baseline written before commit 60f7139 (the
    commit that added this file) has no `corpus` key at all — so anything reading
    that directory must read defensively.

    Arm 1 (the live checkout): assert on the newest historical record that has a
    `corpus` block, which costs no eval run.

    Arm 2 (a fresh tree): `eval/baselines/` is gitignored (.gitignore:92), so in
    a round worktree the directory does not exist and there is no history to
    read. There the same `_assert_corpus_shape` runs against a record this test
    writes itself, via the same empty-corpus CLI path the test above uses. A
    skip is not an alternative: skipping in a worktree is exactly how #498
    clause 4's blind spot worked — the suite stayed green in the one tree where
    the gate would have acted on a failure."""
    recs = sorted(BASELINES.glob("*.json"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    for path in recs:
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(rec, dict) or "corpus" not in rec:
            continue
        _assert_corpus_shape(rec)
        return

    label = "pytest-shape-fallback"
    _cleanup(label)
    try:
        proc = _run(tmp_path, "--label", label, "--allow-empty-corpus")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        _assert_corpus_shape(json.loads(written[0].read_text()))
    finally:
        _cleanup(label)


def _latency_budgets() -> dict:
    """The two ceilings, read from the owning module in a process of its own.

    Restating the numbers here is the drift this file's `_production_knobs`
    already exists to avoid, and the owning module is in `workers/`, not in the
    eval script — which is exactly the seam #1129 joined."""
    code = (
        "import json;"
        "from workers.sources import automod_regression as R;"
        "print(json.dumps({'budgets': R.LATENCY_BUDGET_MS,"
        " 'nightly': R.CONTEXT_NIGHTLY, 'field': R.OVER_BUDGET_FIELD}))"
    )
    proc = subprocess.run([str(PY), "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_nightly_artifact_and_stdout_carry_the_latency_verdict(tmp_path):
    """The nightly runner is `latency_ms_avg`'s consumer, across a process seam.

    #1129's premise is that the field was written into 20 nightly artifacts and
    read by nobody, which is how #504's 708 ms -> 4,230 ms step landed with every
    rung green. A constant in `workers/` does not close that; the process that
    produces the number has to write the verdict next to it. So this drives the
    real CLI (subprocess, because `app.paths` reads LLOYD_FACTS_ROOT/LLOYD_KG_DB
    at import time) and asserts BOTH surfaces a reader has: the `latency_budget`
    block in the artifact, naming the nightly context and the ceiling the owning
    module holds, and the printed line, naming the same ceiling and the same flag.

    `--allow-empty-corpus` is here only so the run completes without the live fact
    tree; the latency is whatever the run actually took, and `over` is asserted
    against the budget rather than hardcoded — either answer is a pass here. What
    must be impossible is a nightly record with an average and no verdict.
    """
    known = _latency_budgets()
    budget = known["budgets"][known["nightly"]]
    label = "pytest-latency-verdict"
    _cleanup(label)
    try:
        proc = _run(tmp_path, "--label", label, "--allow-empty-corpus")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        rec = json.loads(written[0].read_text())

        verdict = rec["latency_budget"]
        avg = rec["summary"]["overall"]["latency_ms_avg"]
        assert verdict["context"] == known["nightly"]
        assert verdict["budget_ms"] == budget, (
            f"the artifact names a {verdict['budget_ms']} ms ceiling; the owning "
            f"module holds {budget}")
        assert verdict["latency_ms_avg"] == avg, (
            "the verdict was computed on a different average than the one the "
            "record reports")
        assert verdict["over"] is (avg > budget), (
            "the flag does not match its own two numbers — the ceiling is strict, "
            "so an average exactly ON it is inside budget, and `>=` here would "
            "fail the runner for a number the owning module calls compliant")

        line = [ln for ln in proc.stdout.splitlines() if "budget" in ln.lower()]
        assert line, f"nothing about the budget was printed:\n{proc.stdout}"
        printed = " ".join(line)
        assert f"{budget:,.0f} ms" in printed, printed
        assert ("OVER BUDGET" if verdict["over"] else "inside budget") in printed, printed
        assert f"{avg:,.0f} ms" in printed, printed
    finally:
        _cleanup(label)


# ---------------------------------------------------------------------------
# The fact leg (#1250)
# ---------------------------------------------------------------------------

def _provision_fact_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A real fact tree AND a store that indexes it — then the tree is emptied.

    Returns (facts_root, kg_db) with the index holding one fact and the readable
    tree holding none, which is the exact disagreement that hid the six zeroed
    arms of 2026-09-18: `corpus.facts` is the store's index count and stays
    non-zero, while the read path the eval exercises returns nothing. Built
    through the named routes (`KGStore`, `facts_idx.reindex`) rather than by
    hand-writing sqlite, so the index is what a real rebuild would have written.

    The file shape is the one `agent_mcp/retrieval.py` parses: YAML frontmatter
    with a `facts:` list, under `<facts_root>/<entity-slug>/<Entity>-<category>.md`.
    """
    facts = tmp_path / "fact-tree"
    (facts / "lloyd").mkdir(parents=True, exist_ok=True)
    (facts / "lloyd" / "Lloyd-state.md").write_text(
        "---\n"
        "entity: Lloyd\n"
        "facts:\n"
        "- fact: Lloyd is the agent that runs this box\n"
        "  confidence: 0.9\n"
        "  provenance: STATED\n"
        "  created_at: '2026-09-01T00:00:00'\n"
        "---\n"
        "\n"
        "# Lloyd\n",
        encoding="utf-8")
    db = tmp_path / "kg-indexed.sqlite"
    env = dict(os.environ)
    env["LLOYD_FACTS_ROOT"] = str(facts)
    env["LLOYD_KG_DB"] = str(db)
    _provision_store(db, env)
    subprocess.run(
        [str(PY), "-c",
         "import sys, pathlib\n"
         "from app.kg_store import KGStore\n"
         "s = KGStore(sys.argv[1])\n"
         "s.entities.register('Lloyd', kind='system')\n"
         "s.edges.add({'source': 'Lloyd', 'target': 'Mission Control',"
         " 'type': 'documents', 'origin': 'test'})\n"
         "s.facts_idx.reindex(root=pathlib.Path(sys.argv[2]))\n"
         "assert s.stats()['facts'] == 1, s.stats()\n"
         "s.close()\n",
         str(db), str(facts)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180, check=True)
    # The index now claims a fact the readable tree no longer holds. Same shape
    # as the gate arm: the corpus is populated, the read path is not.
    shutil.rmtree(facts / "lloyd")
    (facts / "lloyd").mkdir()
    return facts, db


def test_a_zeroed_fact_leg_is_recorded_as_unscoreable_not_as_zero(tmp_path):
    """Every fact leg empty while the store indexes facts = no measurement (#1250).

    Six `automod-check` arms on 2026-09-18 recorded `fact_entity_recall_avg:
    0.0` — a number, non-null, scored — with per-query `n_facts: 0`,
    `error: null` on all 20 queries, `errors: 0` and `corpus_ok: true`, against
    a corpus naming 315,462 facts. `corpus_ok` could not catch that because it is
    `bool(corpus["edges_active"]) and bool(corpus["entities"])`: the graph half
    only. And the paired check's `evaluate()` has no zero-guard, so the 0.0
    became a regression and a rollback reason for a commit that touched nothing
    in the fact path. This drives the real CLI over a world where the index and
    the read path disagree, and asserts the artifact cannot be mistaken for a
    measurement: null metric, `corpus_ok: false`, and the resolved facts root
    printed — the line that lets a reader see WHICH root read nothing.
    """
    label = "pytest-zeroed-fact-leg"
    _cleanup(label)
    facts, db = _provision_fact_tree(tmp_path)
    try:
        proc = _run(tmp_path, "--label", label, kg_db=db, facts_root=facts)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, "the run wrote no artifact"
        rec = json.loads(written[0].read_text())

        assert rec["corpus"]["facts"] > 0, (
            "the world this test needs is a NON-EMPTY fact index beside an "
            "unreadable fact tree; a zero here means the fixture did not index")
        assert rec["summary"]["overall"]["fact_entity_recall_avg"] is None, (
            "a fact leg that read nothing was recorded as a scored number — "
            "the exact defect: 'did not measure' and 'measured zero' read alike")
        assert rec["corpus_ok"] is False, (
            "an arm that scored nothing must not report a corpus it could read; "
            "this is the flag the paired check refuses on")
        assert rec["fact_leg"]["empty"] is True
        assert rec["fact_leg"]["n_facts_total"] == 0
        assert rec["fact_leg"]["facts_in_corpus"] == rec["corpus"]["facts"]

        per_query = [r["result_summary"] for r in rec["records"]]
        assert all(q["n_facts"] == 0 for q in per_query), per_query
        # The two new per-query fields exist on a leg that read nothing, so a
        # reader can tell "0 read" from "0 matched" from the artifact alone.
        assert all("n_fact_reads_failed" in q and "fact_read_first_error" in q
                   for q in per_query), per_query

        assert "facts_root =" in proc.stderr, (
            f"the refusal must name the root it read:\n{proc.stderr}")
        assert str(facts) in proc.stderr, (
            f"the printed facts root is not the one this run was given:\n{proc.stderr}")
        # The refusal adds a verdict, it does not remove anything: a reader who
        # already depends on the `corpus` block and the `corpus_ok` flag still
        # finds both, which is what this file's shape helper checks.
        _assert_corpus_shape(rec)
    finally:
        _cleanup(label)


def test_one_query_with_a_legitimate_zero_fact_leg_keeps_its_number(tmp_path):
    """A per-query zero is a real score; only an all-zero leg is not (#1250).

    The healthy arm's per-query fact counts are nineteen 10s and one 1 — zero
    queries at 0 — so a guard keyed per-query would null a legitimate result.
    This world has two queries: one whose entity has facts, one whose entity
    directory exists but holds no fact files (an entity-resolution miss, a real
    0.0 for that query). The run must keep its numeric average — 1.0 and 0.0
    over two queries — with `corpus_ok` true and the empty query NAMED in
    `empty_fact_queries` rather than the leg declared unscorable.
    """
    label = "pytest-partial-fact-leg"
    _cleanup(label)
    facts = tmp_path / "fact-tree-partial"
    (facts / "lloyd").mkdir(parents=True, exist_ok=True)
    (facts / "lloyd" / "Lloyd-state.md").write_text(
        "---\nentity: Lloyd\nfacts:\n"
        "- fact: Lloyd is the agent that runs this box\n  confidence: 0.9\n"
        "  provenance: STATED\n  created_at: '2026-09-01T00:00:00'\n---\n\n# Lloyd\n",
        encoding="utf-8")
    (facts / "Zzzghost").mkdir()  # seedable entity, genuinely no facts: a real 0
    db = tmp_path / "kg-partial.sqlite"
    env = dict(os.environ)
    env.update({"LLOYD_FACTS_ROOT": str(facts), "LLOYD_KG_DB": str(db)})
    _provision_store(db, env)
    subprocess.run(
        [str(PY), "-c",
         "import sys, pathlib\n"
         "from app.kg_store import KGStore\n"
         "s = KGStore(sys.argv[1])\n"
         "s.entities.register('Lloyd', kind='system')\n"
         "s.entities.register('Zzzghost', kind='system')\n"
         "s.edges.add({'source': 'Lloyd', 'target': 'Mission Control',"
         " 'type': 'documents', 'origin': 'test'})\n"
         "s.facts_idx.reindex(root=pathlib.Path(sys.argv[2]))\n"
         "assert s.stats()['facts'] == 1, s.stats()\n"
         "s.close()\n",
         str(db), str(facts)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180, check=True)
    queries = tmp_path / "queries-mixed.yaml"
    queries.write_text(
        "queries:\n"
        "  - id: has-facts\n"
        "    query: what is lloyd\n"
        "    category: single\n"
        "    expect_entities: [Lloyd]\n"
        "    expect_docs: [lloyd]\n"
        "  - id: no-facts\n"
        "    query: what is zzzghost\n"
        "    category: single\n"
        "    expect_entities: [Zzzghost]\n"
        "    expect_docs: [lloyd]\n"
    )
    try:
        proc = _run(tmp_path, "--queries", str(queries), "--label", label,
                    kg_db=db, facts_root=facts)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, "the run wrote no artifact"
        rec = json.loads(written[0].read_text())

        per_query = {r["id"]: r for r in rec["records"]}
        assert {r["result_summary"]["n_facts"] > 0 for r in rec["records"]} == {True, False}, (
            "this test needs one query with facts and one without; the world "
            f"produced {[(k, v['result_summary']['n_facts']) for k, v in per_query.items()]}")
        scored = [r["scoring"]["fact_entity_recall"] for r in rec["records"]
                  if r["scoring"]["fact_entity_recall"] is not None]
        expected = sum(scored) / len(scored)
        assert rec["summary"]["overall"]["fact_entity_recall_avg"] == pytest.approx(expected), (
            "a run where SOME queries returned facts had its real number taken "
            "away — the guard must be total-across-queries, not per-query")
        assert rec["corpus_ok"] is True, (
            "a partial fact leg is a score, and `corpus_ok: false` would refuse "
            "a legitimately imperfect run")
        assert rec["fact_leg"]["empty"] is False
        assert rec["fact_leg"]["n_facts_total"] > 0
        assert "no-facts" in rec["fact_leg"]["empty_fact_queries"], rec["fact_leg"]
        assert rec["fact_leg"]["n_fact_reads_failed_total"] == 0, (
            "an entity that simply has no fact files is not a failed read; "
            "counting it would make every healthy arm look broken")
        assert "fact leg read NOTHING" not in proc.stderr, proc.stderr
    finally:
        _cleanup(label)


# ── #878: the ENTITY leg of the corpus is satisfiable ─────────────────────────
#
# `entity_hit` is containment: `eval/run_eval.py:_score` scores an expectation by
# `exp in got` over the entities the run actually returned, and every entity name
# a run can return exists in the entity store. So an `expect_entities` entry that
# NO entity name contains can never be true — no retrieval improvement can score
# it, and the query silently lowers `entity_hit_rate` and `entity_recall_avg`,
# which then gets read as a seed-identification defect. It did: #513's
# `Backlog Item #363` — a real vault PATH that is not a real ENTITY NAME — was 1/8
# of the measured defect behind #400, was "retargeted" on 2026-08-06 without
# landing, and by 2026-09-16 was armed in TWO queries (`backlog-363`,
# `tgs-rag-state`).
#
# The doc leg has had a satisfiability guard since #399/#504, and it covers ONE
# query (`test_the_retargeted_eval_query_is_satisfiable`, in
# `tests/test_automod_hardening.py`). Nothing guarded the entity leg for ANY
# query, which is exactly why the entity-side instance slipped through a guard
# written for the doc side. So the check below is a loop over every query, and no
# query id appears in it — naming one query is the hole this closes.
#
# Unlike the rest of this file, these tests read the store IN PROCESS. The
# subprocess rule above exists because `app.paths` reads LLOYD_FACTS_ROOT /
# LLOYD_KG_DB at import time and the CLI contract is what protects an operator;
# here the thing under test is a corpus-vs-store invariant with no CLI, and the
# store must be read through `app.kg_store` — the module that owns the one
# connection, because six programs rewriting the same JSON is what produced the
# 2026-08-22 wipe. So: `app.kg_store` only, never a sqlite handle.

import inspect  # noqa: E402

import yaml  # noqa: E402

CORPUS = ROOT / "eval" / "vault_recall_queries.yaml"
# `_pipeline/` is gitignored (.gitignore:25), so a git worktree — including the
# automod round the gate runs this suite in — has no `kg.sqlite` at all. The
# fallback is `app/uptake.py:lloyd_root`'s rule for the same fact: measure the
# live tree, because the transcripts and the store live there and the worktree's
# absence of them is not a measurement of anything. Without it this file's entity
# guard would refuse in every worktree, which is a red suite that tells you
# nothing about the corpus.
LIVE_KG_DB = Path.home() / "lloyd" / "_pipeline" / "vault-derived" / "kg.sqlite"


def _entity_store_names(getter=None, *, allow_live_fallback: bool = True) -> tuple[list[str], str]:
    """Every entity name in the store the eval scores against, plus where it came from.

    Every open is `app.kg_store`'s — `KGStore(path)` for a path this function
    resolved, `store()` for the refusal — so a store that will not open surfaces as
    `StoreUnavailable` rather than as an empty list. That distinction is the whole
    of clause 5: "this expectation is unreachable" and "I could not look" must not
    arrive at the same place. A test passes its own getter — one that raises, or
    one holding zero rows — to pin either side of it, and a named getter NEVER
    falls back, or the test would read the real graph behind its own fake and pass
    for the wrong reason.

    The fallback opens `KGStore(path)` rather than `configure(path)`: the latter
    repoints the process-wide default, and a test must not swap the knowledge
    graph out from under the rest of the suite. What it must not do, in either
    branch, is open the file itself — `kg.sqlite` has one opener, and the six that
    used to share it produced the 2026-08-22 wipe.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import app.kg_store as ks

    from app.paths import VAULT_KG_DB

    if getter is not None:
        st = getter()
        return list(st.entities.all()), str(getattr(st, "path", "?"))

    # No getter: read the store THIS TREE'S eval resolves against, and nothing
    # else. Going through `store()` directly would be wrong here for a reason worth
    # naming: the rest of this file provisions throwaway stores with
    # `kg_store.configure(...)`, which repoints the process-wide default, so by the
    # time this runs the "default" store may be some closed tmp_path database from
    # another test. The path is resolved from `app.paths` — which honours
    # LLOYD_KG_DB — and opened through `KGStore`, one connection, closed here.
    # `is_file()` before opening is `kg_store._require_database`'s own rule:
    # `KGStore` CREATES an absent file, and a created-empty store is the false
    # clean bill, so an absent path is never opened, it is refused below.
    if allow_live_fallback:
        for path in (Path(VAULT_KG_DB), LIVE_KG_DB):
            if path.is_file():
                st = ks.KGStore(path)
                try:
                    return list(st.entities.all()), str(st.path)
                finally:
                    st.close()

    # Nothing readable on either path (or a caller switched the fallback off):
    # the sanctioned reader produces the refusal, and it names the path it looked
    # for. Deciding "absent" here instead would be the very substitution clause 5
    # forbids — an unavailable store must not become an unsatisfiable corpus.
    st = ks.store()
    return list(st.entities.all()), str(st.path)


def _entity_satisfiability_report(corpus: Path = CORPUS, getter=None, *,
                                  allow_live_fallback: bool = True) -> dict:
    """Which expectations no entity name CONTAINS — the scorer's own rule.

    Containment, not equality: `_score` matches a normalized expectation as a
    substring of a normalized returned name, so `TGS-RAG Implementation` is
    reachable through the row `#363 TGS-RAG Implementation` even though no entity
    is named exactly that. A guard written as an equality lookup would alarm on
    satisfiable expectations and force a needless corpus edit — and the retargets
    it demanded would be substitutions of an easier target, which the 2026-08-06
    audit rule at the top of the corpus file forbids.

    Returns the report rather than asserting, so a caller can assert on the shape
    (`entity_names`, `expectations`) as well as the verdict: a 0-hit loop over a
    corpus that was never read is indistinguishable from a clean corpus unless the
    denominator is printed beside it.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from eval.run_eval import _norm  # the scorer's own normalization, not a copy

    names, where = _entity_store_names(getter, allow_live_fallback=allow_live_fallback)
    specs = yaml.safe_load(Path(corpus).read_text())["queries"]
    expectations = [(str(s["id"]), str(e))
                    for s in specs for e in (s.get("expect_entities") or [])]
    if not names:
        raise AssertionError(
            f"the entity store at {where} returned 0 entity rows while the corpus at "
            f"{corpus} carries {len(expectations)} expectations. Reporting every "
            "expectation absent from a store with no rows is not a verdict — it is "
            "the same false finding an unreadable store would produce, and #878 "
            "exists to remove that class.")
    normed = [_norm(n) for n in names]
    dead = [{"query": qid, "expect": exp} for qid, exp in expectations
            if not any(_norm(exp) in n for n in normed)]
    return {"store": where, "corpus": str(corpus), "queries": len(specs),
            "entity_names": len(names), "expectations": len(expectations), "dead": dead}


def _assert_entity_expectations_satisfiable(report: dict) -> None:
    """Name the query AND the expectation, or the failure is not actionable."""
    dead = report["dead"]
    assert not dead, (
        f"{len(dead)} of {report['expectations']} `expect_entities` entries across "
        f"{report['queries']} queries of {report['corpus']} are contained in no "
        f"entity name in {report['store']} ({report['entity_names']} rows), so the "
        "scorer can never report them as matched: they lower `entity_hit_rate` and "
        "`entity_recall_avg` no matter how retrieval improves, and the movement "
        "reads as a retrieval regression. Re-point each at a name the store "
        "resolves, or delete it if it is factually wrong — never at an easier "
        "target. Offenders: "
        + "; ".join(f"{d['query']} -> {d['expect']!r}" for d in dead))


def test_every_expect_entities_name_is_contained_in_a_store_entity_name():
    """#878: the corpus is checked against the store for its ENTITY leg, entirely.

    The failure this pins is the one that already happened twice: an expectation
    naming an entity that does not exist is armed forever, is scored as a
    retrieval failure, and the number is read as a seed-identification defect.
    `entity_recall` is halved as well as `entity_hit` lost — `backlog-363` sat at
    0.5 with `entities_matched: ["task #363"]` while reporting a pass.
    """
    report = _entity_satisfiability_report()
    assert report["queries"] >= 20, report          # the loop read the whole corpus
    assert report["expectations"] >= 40, report     # not one query, not one expectation
    _assert_entity_expectations_satisfiable(report)


def test_the_guard_loops_the_whole_corpus_and_names_no_query_id():
    """A guard naming one query id is the hole #878 exists to close.

    `test_the_retargeted_eval_query_is_satisfiable` is the doc-side precedent and
    it is a single-query assertion: the entity-side variant of the same defect
    (#513, then #878's second armed query) walked straight past it. So this pins
    the shape — every expectation examined, no corpus query id written into the
    checker's source — not just the current verdict.
    """
    report = _entity_satisfiability_report()
    specs = yaml.safe_load(CORPUS.read_text())["queries"]
    per_query = {str(s["id"]): len(s.get("expect_entities") or []) for s in specs}
    assert report["expectations"] == sum(per_query.values()), (
        "the guard examined a different number of expectations than the corpus "
        f"holds: {report['expectations']} vs {sum(per_query.values())}")

    src = inspect.getsource(_entity_satisfiability_report) + inspect.getsource(
        _assert_entity_expectations_satisfiable) + inspect.getsource(_entity_store_names)
    named = [qid for qid in per_query if qid in src]
    assert not named, f"the checker names corpus query id(s) {named}; it must loop"


def test_the_guard_fails_on_a_synthetic_query_no_entity_can_satisfy(tmp_path):
    """Non-vacuity: the checker actually fails, and says which query and which name.

    The expectation is the real historical offender, and the query id is a slug
    that exists nowhere in the corpus — so a pass here can only mean the checker
    is not looking.
    """
    corpus = tmp_path / "dead-expectation.yaml"
    corpus.write_text(
        "queries:\n"
        "  - id: unsatisfiable-probe\n"
        "    query: tell me about the thing that is not in the graph\n"
        "    category: single\n"
        "    expect_entities: [\"Backlog Item #363\"]\n"
        "    expect_docs: [lloyd]\n"
    )
    report = _entity_satisfiability_report(corpus)
    assert report["expectations"] == 1, report
    with pytest.raises(AssertionError) as exc:
        _assert_entity_expectations_satisfiable(report)
    msg = str(exc.value)
    assert "unsatisfiable-probe" in msg, msg
    assert "Backlog Item #363" in msg, msg


def test_a_containment_reachable_expectation_is_not_reported_unsatisfiable(tmp_path):
    """The guard is containment, or it alarms on expectations retrieval can meet.

    #513 was filed to delete a whole class of these, and a guard on strict names would
    re-file it every night against names that are not defects.

    Synthetic, with one reachable-only-by-containment gold and one reachable by no
    route, in the same report. The shipped corpus used to supply the first case
    (`TGS-RAG Implementation`, contained in the row `#363 TGS-RAG Implementation`) and
    the test read its target out of that corpus, which meant the assertion could not
    go red: once #1260 re-pointed the gold to a real row the target was simply not
    there and the test returned early. Here the assertion always runs, and one report
    proves both directions — containment is admitted, an unreachable name is still
    caught.
    """
    corpus = tmp_path / "containment.yaml"
    corpus.write_text(
        "queries:\n"
        "  - id: containment-probe\n"
        "    query: tell me about the backlog\n"
        "    category: single\n"
        '    expect_entities: ["Lloyd Backlog"]\n'
        "    expect_docs: [lloyd]\n"
        "  - id: unreachable-probe\n"
        "    query: tell me about the thing that is not in the graph\n"
        "    category: single\n"
        '    expect_entities: ["Backlog Item #363"]\n'
        "    expect_docs: [lloyd]\n"
    )

    class ContainmentOnlyStore:
        """One entity whose NAME CONTAINS the gold, and nothing else.

        `Lloyd Backlog` is neither an entity row nor an alias surface, so only the
        scorer's own containment rule reaches it — `run_eval._score` tests the
        expectation as a substring of a returned name, and `Lloyd Backlog System` is
        such a name.
        """
        class _A:
            @staticmethod
            def lookup(name):
                return None
        class _E:
            @staticmethod
            def all():
                return ["Lloyd Backlog System", "Knowledge Graph"]
        aliases = _A()
        entities = _E()
        version = 1

    report = _entity_satisfiability_report(corpus, getter=lambda: ContainmentOnlyStore)
    assert report["expectations"] == 2, report
    # A list of dicts: the report is JSON-shaped so it can travel into a run record.
    assert report["dead"] == [{"query": "unreachable-probe",
                               "expect": "Backlog Item #363"}], report
    with pytest.raises(AssertionError) as exc:
        _assert_entity_expectations_satisfiable(report)
    msg = str(exc.value)
    assert "unreachable-probe -> 'Backlog Item #363'" in msg, msg
    assert "containment-probe" not in msg, (
        f"the guard flagged a containment-reachable expectation, which is the "
        f"#513 false-positive class: {msg}")

def test_an_unreadable_store_raises_rather_than_reporting_every_name_absent(monkeypatch):
    """Clause: a store that will not open must not read as an unsatisfiable corpus.

    The failure this removes is the one a naive guard would add: an empty entity
    list makes every expectation in the corpus look dead at once, a far bigger and
    far less true alarm than the names it exists to catch. Same shape as
    `test_unreadable_store_is_its_own_failure_and_ignores_the_flag` above, one
    level down: `StoreUnavailable` means "I could not read it".

    Three spellings of the sanctioned route are exercised, and the third is the one
    the shipped corpus test uses: the DEFAULT route with the worktree fallback still
    switched ON. Patching `store` alone could not prove that route, because the
    fallback opens `KGStore(path)` — so both names in `app.kg_store` are patched,
    which is exactly the pair the checker is allowed to touch. A guard that quietly
    re-opened the live graph behind a refusing store — reaching for a path instead of
    admitting it could not read one — fails here rather than passing and reporting a
    verdict it cannot justify.
    """
    import app.kg_store as ks

    def refusing(*_args, **_kwargs):
        raise ks.StoreUnavailable("no database at /nowhere/kg.sqlite")

    monkeypatch.setattr(ks, "store", refusing)
    monkeypatch.delenv("LLOYD_KG_DB", raising=False)

    with pytest.raises(ks.StoreUnavailable):
        _entity_satisfiability_report(getter=ks.store)
    with pytest.raises(ks.StoreUnavailable):
        _entity_satisfiability_report(allow_live_fallback=False)

    monkeypatch.setattr(ks, "KGStore", refusing)
    with pytest.raises(ks.StoreUnavailable):
        _entity_satisfiability_report()          # shipped default: no escape hatch

    # The route is `app.kg_store`, never a handle of its own: no sqlite import,
    # no `.conn`, no `execute(` anywhere in the checker, and the default really is
    # `ks.store()` rather than some second opener.
    src = (inspect.getsource(_entity_store_names)
           + inspect.getsource(_entity_satisfiability_report))
    assert "sqlite3" not in src, src
    assert ".conn" not in src and ".execute(" not in src, src
    assert "ks.store()" in src, src


def test_the_default_route_is_the_live_store_the_eval_scores_against():
    """The guard reads a real store with real rows, or it guards nothing.

    A checker whose default silently resolved to an empty or synthetic store would
    report 0 expectations dead and look exactly like a healthy corpus — the
    denominator assertions in the corpus test are here for that reason, and this
    one names the store it read so the figure is attributable.
    """
    names, where = _entity_store_names()
    assert len(names) > 1000, f"{where} returned {len(names)} entity names"
    assert "kg.sqlite" in where, where
    assert yaml.safe_load(CORPUS.read_text())["queries"], CORPUS


def test_a_store_that_opens_with_no_entity_rows_is_refused_as_a_verdict(monkeypatch):
    """Zero rows is a different fact from zero matches, and must not become one.

    `KGStore(path)` CREATES an absent database, which is how a worktree run once
    reported `duplicate_rows: 0` about a store that was not there
    (`app/uptake.py:1413-1418`). An empty `entities` table would otherwise arrive
    as "every expectation is unsatisfiable" — 43 findings, all of them about the
    reader — so the refusal is on the row count as well as on the open.
    """
    class EmptyStore:
        path = Path("empty-kg.sqlite")

        class entities:  # noqa: N801 - mirrors the store's attribute spelling
            @staticmethod
            def all():
                return []

    with pytest.raises(AssertionError) as exc:
        _entity_satisfiability_report(getter=EmptyStore)
    assert "0 entity rows" in str(exc.value), exc.value
    assert "not a verdict" in str(exc.value), exc.value


# ── #1319: the DOCUMENT leg gets the same guard the entity leg has ───────────
#
# `expect_docs` is scored per label — `doc_recall` is hits-over-labels
# (`eval/run_eval.py:267`) and `ndcg10` takes the label list as its ideal set
# (`:250`) — so a label that matches no vault path is not a neutral miss: it is
# a permanent subtraction from every run's measured ceiling, attributed to
# retrieval. Before this guard existed, 13 of the 50 labels in the 20-query
# corpus were code paths (`agent_mcp/vault.py`, `app/harness/loop.py`,
# `classifier-v4`) that the doc scorer could never satisfy, capping
# `doc_recall_avg` at 37/50 = 0.74 while the measured value read 0.593 — and
# nothing in the suite could see it, because the entity leg's guard below reads
# the ENTITY store and never a document path.
#
# Same rule as the scorer, deliberately re-implemented rather than imported:
# `_norm(label) in _norm(path)` with `-` folded to `_`, from
# `eval/run_eval.py:158` (`_norm`) and `:210`/`:213` (both sides normalised,
# substring test). Importing run_eval here would make the guard read the tree
# it guards, which is the discipline the module docstring keeps this file to;
# the three lines below are the whole rule and the comments name the lines to
# check it against.

def _doc_norm(s: str) -> str:
    """Mirror of `eval/run_eval.py::_norm` — lowercase, `-` and `_` equivalent."""
    return str(s or "").lower().replace("-", "_")


def _doc_label_satisfiability_report(
        specs_path: Path = CORPUS,
        vault_root: Path = Path.home() / "obsidian") -> dict:
    """Walk the vault once and return which `expect_docs` labels match no path.

    `path` is every path under `vault_root` (files AND directories,
    vault-relative), minus any path with a dot-prefixed component. That is
    qmd's own indexing rule (`qmd/src/cli/qmd.ts`, the `parts.some(part =>
    part.startsWith("."))` filter after the glob), so `skills/.archived/**` is
    NOT a location the index can hold: the live index had 0 such documents on
    2026-09-21, and the five labels this walk used to accept there could never
    be returned. `.git/**` falls under the same rule. `walked_paths` is returned
    so the zero is checkable in the same breath as the verdict — a denominator
    of zero is not a pass.
    """
    specs = yaml.safe_load(specs_path.read_text())["queries"]
    labels = [(str(s.get("id")), str(d))
              for s in specs for d in (s.get("expect_docs") or [])]
    walked, normed = 0, []
    for p in vault_root.rglob("*"):
        rel = p.relative_to(vault_root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        walked += 1
        normed.append(_doc_norm(str(rel)))
    dead = [{"query": qid, "label": label} for qid, label in labels
            if not any(_doc_norm(label) in n for n in normed)]
    return {"vault": str(vault_root), "corpus": str(specs_path),
            "queries": len(specs), "labels": len(labels),
            "walked_paths": walked, "dead": dead}


def test_committed_corpus_has_no_unresolvable_document_label():
    """Every `expect_docs` label in the committed corpus matches a real path.

    The failure message names the query AND the label, because a list of bare
    labels cannot be acted on: the reader has to open the query to judge whether
    the label is wrong or the note moved.
    """
    rep = _doc_label_satisfiability_report()
    assert rep["labels"] > 0, rep
    assert rep["walked_paths"] > 100, (
        f"the walk under {rep['vault']} found {rep['walked_paths']} paths; a "
        "walk that reaches nothing reports every label as dead OR as clean "
        "depending on the comparison, and neither is a verdict (#878's shape)")
    assert rep["dead"] == [], (
        f"{len(rep['dead'])} of {rep['labels']} expect_docs labels in "
        f"{rep['corpus']} match no path under {rep['vault']} (walked "
        f"{rep['walked_paths']} paths): "
        + "; ".join(f"{d['query']}: {d['label']!r}" for d in rep["dead"])
        + " — a label that cannot match is subtracted from doc_recall and "
        "ndcg@10 by every future run, so re-point it at the note the answer "
        "would cite, or delete it, rather than leaving the ceiling capped.")


def test_a_label_only_a_dot_directory_holds_is_dead(tmp_path):
    """qmd never indexes a dot-directory, so a label that resolves only there is
    unreachable and must be reported, not accepted (gold audit, 2026-09-21)."""
    vault = tmp_path / "vault"
    (vault / "skills" / ".archived" / "old-skill").mkdir(parents=True)
    (vault / "skills" / ".archived" / "old-skill" / "SKILL.md").write_text("x")
    (vault / "skills" / "live-skill").mkdir(parents=True)
    (vault / "skills" / "live-skill" / "SKILL.md").write_text("x")
    corpus = tmp_path / "q.yaml"
    corpus.write_text(yaml.safe_dump({"queries": [
        {"id": "a", "query": "q", "expect_docs": ["skills/.archived/old-skill/SKILL.md"]},
        {"id": "b", "query": "q", "expect_docs": ["skills/live-skill/SKILL.md"]}]}))
    rep = _doc_label_satisfiability_report(specs_path=corpus, vault_root=vault)
    assert rep["dead"] == [{"query": "a", "label": "skills/.archived/old-skill/SKILL.md"}], rep


def test_the_document_label_guard_fails_on_an_unresolvable_label(tmp_path):
    """A synthetic unresolvable label is caught by name, and a resolvable one is not.

    Both halves matter. The negative half is the point — a guard that never
    fires is indistinguishable from a corpus with no dead labels, which is
    exactly how 13 dead labels survived an entire month of nightly runs. The
    positive half is what stops the guard from being a tautology that flags
    everything.
    """
    vault = tmp_path / "vault"
    (vault / "knowledge" / "software").mkdir(parents=True)
    (vault / "knowledge" / "software" / "live-note.md").write_text("body\n")
    corpus = tmp_path / "queries.yaml"
    corpus.write_text(
        "queries:\n"
        "  - id: resolvable-one\n"
        "    query: what does the live note say\n"
        "    category: single\n"
        "    expect_entities: []\n"
        "    expect_docs: [knowledge/software/live-note.md]\n"
        "  - id: dead-one\n"
        "    query: what does the missing note say\n"
        "    category: single\n"
        "    expect_entities: []\n"
        "    expect_docs: [knowledge/software/never-written.md]\n"
    )
    rep = _doc_label_satisfiability_report(specs_path=corpus, vault_root=vault)
    assert rep["walked_paths"] >= 2, rep          # positive control: it walked
    assert [d["query"] for d in rep["dead"]] == ["dead-one"], rep["dead"]
    assert rep["dead"][0]["label"] == "knowledge/software/never-written.md"

    # A dead label must be named by BOTH its query and the label itself: a
    # reader who is only told "one label is dead" cannot fix it.
    rendered = "; ".join(f"{d['query']}: {d['label']!r}" for d in rep["dead"])
    assert "dead-one" in rendered and "never-written" in rendered

    # Folding `-`/`_` is the scorer's tolerance, so a label that differs from
    # the path only in that spelling must NOT be flagged — otherwise this guard
    # is stricter than the metric it predicts and alarms on a passing label.
    hyphen = tmp_path / "hyphen.yaml"
    hyphen.write_text(
        "queries:\n"
        "  - id: hyphen-spelling\n"
        "    query: what does the live note say\n"
        "    category: single\n"
        "    expect_entities: []\n"
        "    expect_docs: [knowledge/software/live_note.md]\n"
    )
    assert _doc_label_satisfiability_report(specs_path=hyphen,
                                            vault_root=vault)["dead"] == []



# The 20 ids the corpus opened with, in committed file order, and the SHA-256 of
# their `query` strings joined by "\n" as of `b49a7a36` — the last commit before
# #1319 grew the set. Pinning the digest is how byte-identity is checked without
# a second copy of the answers in the tree: a re-worded original query changes
# the digest even when the new wording still resolves, so the 2026-09-04 ->
# 2026-09-17 series stays joinable on the queries it was measured with.
ORIGINAL_GOLD_IDS = [
    "backlog-363", "entity-resolution-sweep", "inner-voice", "vault-recall",
    "qmd", "kg-maintenance-tasks", "lloyd-vllm-rel", "harness-tools",
    "nightly-reflection", "qwen38-local-serving", "tgs-rag-state",
    "graph-quality", "classifier-v4", "godnode-threshold",
    "relationships-location", "memory-persistence", "autonomy-pipeline",
    "robotics-projects", "this-week-autonomy", "backlog-overview",
]
ORIGINAL_QUERY_DIGEST = \
    "sha256:2382e4eabb68242034c3b958d75aed12a7caa1131ff1aa57959240e657e29e74"

# The exact-McNemar search in scripts/eval_trend_stats.py prints this: queries
# needed for a 0.10 paired change at 80 % power, alpha 0.05. Below it the
# nightly's trend verdicts are unsupported by its own audit, which is the defect
# #1319 was filed to close — so it belongs in the corpus guard, where a later
# trim that drops back under it fails loudly instead of quietly re-decorating
# the nightly.
GOLD_SET_MIN_QUERIES = 78


def test_the_gold_set_is_big_enough_for_its_own_power_claim(tmp_path):
    """>= 78 gold queries, the original 20 first and byte-identical (#1319).

    Two halves, both from the trend audit's own arithmetic: the corpus has to
    clear the n its sizing says is required for a 0.10 paired change at 80 %
    power, and the ids the 2026-09-04 -> 2026-09-17 series was measured with have
    to remain the first 20 entries with their `query` strings unchanged.
    `expect_*` labels may legitimately be re-pointed — the dead-doc repair did
    exactly that — but the query text is what the old series joined on, so it is
    frozen by digest.
    """
    report = _gold_set_shape_report()
    assert report["n_queries"] >= GOLD_SET_MIN_QUERIES, (
        f"the gold set holds {report['n_queries']} queries, under the "
        f"{GOLD_SET_MIN_QUERIES} this script's own power search requires — the "
        "nightly would be back to emitting trend verdicts its audit withholds")
    assert report["misplaced_original_ids"] == [], report["misplaced_original_ids"]
    assert report["missing_original_ids"] == [], report["missing_original_ids"]
    assert report["query_digest"] == ORIGINAL_QUERY_DIGEST, (
        "a pre-growth query string changed. Its expect_* labels may be "
        "re-pointed; the query itself may not — the whole "
        "2026-09-04 -> 2026-09-17 series joined on that text.")

    # The shape check must be able to FAIL. A corpus that re-orders the
    # originals, or drops one, is the failure this guards, so both are injected
    # here and named in the report rather than passing silently.
    base = [{"id": qid, "query": f"query {i}", "category": "single",
             "expect_entities": ["Alpha"], "expect_docs": ["lloyd"]}
            for i, qid in enumerate(ORIGINAL_GOLD_IDS)]
    filler = [{"id": f"grown-{i}", "query": f"grown query {i}", "category": "single",
               "expect_entities": ["Alpha"], "expect_docs": ["lloyd"]}
              for i in range(GOLD_SET_MIN_QUERIES - len(ORIGINAL_GOLD_IDS))]
    good = _gold_set_shape_report(specs_path=_write_corpus(tmp_path, base + filler))
    assert good["n_queries"] == GOLD_SET_MIN_QUERIES, good
    assert good["misplaced_original_ids"] == [], good
    assert good["missing_original_ids"] == [], good

    rotated = _gold_set_shape_report(specs_path=_write_corpus(
        tmp_path, filler[:1] + base + filler[1:]))
    assert rotated["misplaced_original_ids"], (
        "the original ids are no longer the first 20 in file order and the "
        "guard did not notice")

    dropped = _gold_set_shape_report(specs_path=_write_corpus(
        tmp_path, [q for q in base if q["id"] != "qmd"] + filler))
    assert dropped["missing_original_ids"] == ["qmd"], dropped

    short = _gold_set_shape_report(specs_path=_write_corpus(
        tmp_path, base + filler[:GOLD_SET_MIN_QUERIES - len(base) - 1]))
    assert short["n_queries"] == GOLD_SET_MIN_QUERIES - 1, short


def _write_corpus(tmp_path: Path, specs: list[dict], name: str = "gold.yaml") -> Path:
    """Emit a corpus YAML with the given specs, in the given order."""
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"queries": specs}, sort_keys=False))
    return p


def _gold_set_shape_report(specs_path: Path | None = None) -> dict:
    """Size, the first ids in file order, and a digest of the originals' text.

    Reads the corpus YAML only — no store, no vault, no retriever — so it cannot
    manufacture a label, and a digest of the ORIGINAL ids' query text means a
    re-worded original is caught even when the new wording still resolves.
    """
    import hashlib

    corpus = Path(specs_path or CORPUS)
    specs = yaml.safe_load(corpus.read_text())["queries"]
    ids = [str(s["id"]) for s in specs]
    originals = [s for s in specs if str(s["id"]) in ORIGINAL_GOLD_IDS]
    digest = "sha256:" + hashlib.sha256(
        "\n".join(str(s["query"]) for s in originals).encode()).hexdigest()
    first = ids[:len(ORIGINAL_GOLD_IDS)]
    return {"corpus": str(corpus), "n_queries": len(specs),
            "first_ids": first,
            "query_digest": digest,
            "misplaced_original_ids": [qid for qid in ORIGINAL_GOLD_IDS
                                       if qid not in first],
            "missing_original_ids": [qid for qid in ORIGINAL_GOLD_IDS
                                     if qid not in ids]}


# The nightly retrieval-eval autonomy task, read from the live vault. It has no
# worktree, so the only tree that can answer a question about it is the vault.
NIGHTLY_TASK_GLOB = "autonomy/82-nightly-retrieval-eval*.md"
VAULT_ROOT = Path.home() / "obsidian"

# Measured on the 20-query corpus: two consecutive nightly runs took 259.4 s and
# 260.4 s, i.e. ~13.0 s per query counting BOTH arms (the reference and its
# perturbed twin). The constant below is that measured cost per query, rounded
# up, and it is what makes the timeout claim arithmetic rather than reassurance.
MEASURED_SECONDS_PER_QUERY = 13.0
NIGHTLY_TIMEOUT_MIN_SECONDS = 1800


def test_the_nightly_timeout_fits_the_corpus_it_now_has():
    """The nightly's own ceiling has to accommodate the grown gold set.

    The task ran `timeout_seconds: 900` against a 20-query corpus that took
    ~260 s. Scaling that measured per-query cost by the corpus that exists now
    is well past 900 s, so a run would have been killed mid-flight and written
    no baseline at all — the failure mode is a silent gap in the series, not a
    loud error, which is the worst thing a nightly trend instrument can do.

    The assertion is therefore on the ARITHMETIC (ceiling vs measured cost x
    queries) and not on a bare constant: a later growth that outgrows this
    timeout fails here even though 1800 is still in the file.
    """
    task = _nightly_task_timeout()
    n_queries = _gold_set_shape_report()["n_queries"]
    needed = MEASURED_SECONDS_PER_QUERY * n_queries
    assert task["timeout_seconds"] >= NIGHTLY_TIMEOUT_MIN_SECONDS, (
        f"{task['path']} declares timeout_seconds: "
        f"{task['timeout_seconds']}, under the {NIGHTLY_TIMEOUT_MIN_SECONDS} s "
        "floor the grown corpus needs")
    assert task["timeout_seconds"] >= needed, (
        f"{task['path']} allows {task['timeout_seconds']} s for a run whose "
        f"{n_queries} queries cost ~{needed:.0f} s at the measured "
        f"{MEASURED_SECONDS_PER_QUERY} s/query (both arms). The nightly would "
        "time out and write no baseline — a gap in the series, not an error.")


def _nightly_task_timeout() -> dict:
    """timeout_seconds from the live nightly retrieval-eval autonomy task."""
    import yaml as _yaml

    matches = sorted(VAULT_ROOT.glob(NIGHTLY_TASK_GLOB))
    if not matches:
        raise AssertionError(
            f"no file matches {VAULT_ROOT / NIGHTLY_TASK_GLOB}. The nightly retrieval eval "
            "task is where the corpus's runtime budget lives; a guard that "
            "skipped when it could not find the file would report a verdict it "
            "cannot justify.")
    path = matches[0]
    text = path.read_text()
    fm = text.split("---", 2)[1]
    meta = _yaml.safe_load(fm) or {}
    timeout = meta.get("timeout_seconds")
    if not isinstance(timeout, (int, float)):
        raise AssertionError(
            f"{path} has no numeric timeout_seconds ({timeout!r}); the run's "
            "ceiling is undeclared, so nothing bounds the grown corpus")
    return {"path": str(path), "timeout_seconds": float(timeout),
            "n_matches": len(matches)}


def _production_expand_graph() -> bool:
    """`RECALL_EXPAND_GRAPH`, read in a subprocess like `_production_knobs`.

    Deliberately not folded into that helper: its caller asserts that every
    entry equals what the run recorded, and the whole claim #1000 exists to make
    honest is that this is the one knob the eval does NOT run at production's
    default. Folding it in would either fail that caller or force the eval's
    default to flip.
    """
    code = ("import json; from agent_mcp import vault; "
            "print(json.dumps(vault.RECALL_EXPAND_GRAPH))")
    proc = subprocess.run([str(PY), "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("argv, expect_expanded, expect_match", [
    # The default invocation: graph expanded (the eval's own measurement choice),
    # production defaults it closed, so the honest claim is "does not match".
    ([], True, False),
    # A run that does match production says so — the field is a comparison, not
    # a constant restatement, so it has to be able to go both ways.
    (["--no-graph"], False, True),
])
def test_the_written_baseline_never_claims_graph_parity_it_lacks(
        tmp_path, argv, expect_expanded, expect_match):
    """Unconstructable pair (#1000): the run's `expand_graph` differs from
    `RECALL_EXPAND_GRAPH` while the artifact claims production-match for it.

    Before the fix the artifact carried the run's value plus a conjunction whose
    graph term was `and not args.no_graph` — a `store_true` flag compared with
    its own absence, true on every run that did not pass the flag. Every
    baseline in the tree therefore stamped graph parity that production does not
    have, and `app/uptake.py`'s `retrieval_gate()` picks its retrieval
    non-regression pool off that field, while `scripts/memory/kg_rebuild.py`
    reports it as before/after evidence.

    Written through the real CLI to a real baseline file, because the defect was
    in the artifact rather than in a function: reading it back off disk is the
    only way a test can catch the claim rather than the code that makes it.
    """
    label = "pytest-expand-graph-honesty"
    _cleanup(label)
    try:
        proc = _run(tmp_path, "--label", label, "--allow-empty-corpus", *argv)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        rec = json.loads(written[0].read_text())

        production = _production_expand_graph()
        assert rec["expand_graph"] is expect_expanded
        assert (rec["expand_graph"] == production) is expect_match
        assert rec["expand_graph_matches_production"] is expect_match
        # Both rows still match production on the six compared knobs: the knob
        # moved OUT of the conjunction, it was not folded into it. Folding it in
        # would make every nightly run report a non-production configuration.
        assert rec["matches_production_defaults"] is True
        assert not (rec["expand_graph"] != production
                    and rec["expand_graph_matches_production"] is not False), (
            "the artifact claims graph parity while running the opposite of "
            "production's default")
    finally:
        _cleanup(label)
