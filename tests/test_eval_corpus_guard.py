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
