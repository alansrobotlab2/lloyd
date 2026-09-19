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

    `matches_production_defaults` is asserted True because a default run is
    exactly the case that flag exists for. #1000 may make a graph-on run report
    False — production defaults expand_graph False while the eval runs it on —
    and when that lands, this line moves with it."""
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
        # The graph leg runs expanded by the eval's own choice, not production's:
        # there is no RECALL_EXPAND_GRAPH to compare it against (#1000).
        assert rec["expand_graph"] is True
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
