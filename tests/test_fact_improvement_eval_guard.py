"""#1250 clause 3 has a second reader, and it cannot round a null.

Clause 3 makes `eval/run_eval.py` record `fact_entity_recall_avg: null` instead of
`0.0` when its fact leg read nothing on a corpus that has facts. That writer is not
the only place the number is consumed: `agent_mcp/fact_improvement.py::
_fact_entity_recall` imports `run_eval.py` by path and calls `summarize()` itself,
and its next statement was `round(summary["overall"]["fact_entity_recall_avg"], 4)`.

A null is not a measurement, and `round(None, 4)` raises `TypeError`. The blanket
handler under that call caught it and logged
`improve: fact_entity_recall could not be measured: type NoneType doesn't define
__round__` — reporting a broken harness and destroying the actual reason, which is
the same destruction this item is filed under. So clause 3's null has to be answered
here too, in the words the eval artifact uses.

Subprocess, for two reasons: `agent_mcp.fact_improvement` imports `app.config`,
which binds an engine socket at import (same reason `tests/test_eval_scorer.py`
subprocesses its fact-path coverage), and the guard under test reads the *resolved*
facts root and store paths, which only an env set before import can control. The
retrieval daemon is stubbed inside; the store is read, never written.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "eval" / "vault_recall_queries.yaml").exists(),
    reason="the eval query set is what the improve loop scores")

BODY = '''
import importlib.machinery
import json
import sys
from pathlib import Path

sys.path.insert(0, {repo!r})
_records = json.loads({records!r})
_summary = json.loads({summary!r})

# `agent_mcp.search`, imported inside `_fact_entity_recall`, is stubbed: it would
# otherwise reach the live retrieval daemon, and this is a test of the metric
# reader, not of the daemon.
_search = type(sys)("agent_mcp.search")
async def _stub_search(query, max_results=5):
    return [{{"file": "knowledge/glossary.md", "score": 1.0}}]
_search.search_daemon = _stub_search
sys.modules["agent_mcp.search"] = _search

# `fact_improvement` loads `eval/run_eval.py` BY PATH, so the loader is the only
# seam. Everything the reader asks of that module — `_corpus_provenance`,
# `fact_leg_read_nothing`, the resolved roots — stays production code over the env
# this harness set; only the two functions that would run a real eval are swapped.
# That is what makes this a test of the READER'S guard and not a restatement of it.
_real_exec = importlib.machinery.SourceFileLoader.exec_module
def _exec(self, module):
    _real_exec(self, module)
    if str(getattr(module, "__file__", "")).endswith("eval/run_eval.py"):
        module.run_eval = lambda queries, limit=20, **kw: _records
        module.summarize = lambda recs: _summary
importlib.machinery.SourceFileLoader.exec_module = _exec

import agent_mcp.fact_improvement as fi
print(json.dumps({{"metric": fi._fact_entity_recall(limit=3),
                  "queries_path_exists": Path({queries!r}).exists()}}))
'''


def _call(records: list[dict], summary: dict, *, kg_db: Path | None = None):
    """Run `_fact_entity_recall` with a fabricated eval result and return (out, log)."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    if kg_db is not None:
        env["LLOYD_KG_DB"] = str(kg_db)
    proc = subprocess.run(
        [sys.executable, "-c", BODY.format(
            repo=str(REPO_ROOT),
            records=json.dumps(records),
            summary=json.dumps(summary),
            queries=str(REPO_ROOT / "eval" / "vault_recall_queries.yaml"))],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def _record(n_facts: int, fer: float | None = None) -> dict:
    return {"id": "what-is-glossary", "scoring": {"fact_entity_recall": fer},
            "result_summary": {"n_facts": n_facts, "n_docs": 1}, "error": None}


def _summary(avg) -> dict:
    return {"overall": {"errors": 0, "fact_entity_recall_avg": avg}}


#: The live fact index, read-only, for the case that needs a NON-empty one.
#: `KGStore(path)` is the deliberate-creation route; pointing the reader at a
#: store that does not exist raises `StoreUnavailable` — a real behavior this file
#: must not confuse with the empty-leg case, since a store that will not open is a
#: different refusal from a leg that read nothing from one that would.
LIVE_KG_DB = REPO_ROOT / "_pipeline" / "vault-derived" / "kg.sqlite"


def _make_empty_store(path: Path) -> Path:
    from app.kg_store import KGStore
    KGStore(path)
    return path


def _store_with_facts(path: Path, n: int = 3) -> Path:
    """A real fact index holding `n` rows, on the canonical schema.

    `KGStore(path)` is `app.kg_store`'s own creation route and applies its own
    `_SCHEMA`; the rows go in through the store's own transaction, not a schema
    this file invents. `run_eval`'s `store.count_facts()` then answers from a
    store that IS there and IS non-empty, which is the only way to build the
    clause-3 state — `n_facts` 0 on every query against a corpus that names
    facts — without touching the live vault.

    A store that does not exist is deliberately NOT the stand-in for this:
    `KGStore` raises `StoreUnavailable` there, which the improve loop's blanket
    handler reports as "could not be measured: no knowledge-graph database at
    …" — a store that will not open is a different refusal from a leg that read
    nothing from one that would, and conflating them would let this test pass on
    the wrong path.
    """
    from app.kg_store import KGStore
    st = KGStore(path)
    with st.transaction() as conn:
        for i in range(n):
            conn.execute(
                "INSERT INTO facts_idx (entity, category, text_hash, fact, file_path) "
                "VALUES (?, ?, ?, ?, ?)",
                ("Glossary", "glossary", f"sha-test-{i}", f"test fact {i}",
                 f"memory/test-{i}.md"))
    assert st.facts_idx.count() == n, "the fixture store must be non-empty to mean anything"
    return path


def test_a_null_fact_metric_on_a_store_with_facts_is_reported_as_none(tmp_path):
    """The clause-3 state, read by the improve loop: None, and it names WHY.

    The records report zero facts on every query and the fact store indexes facts,
    so `fact_leg_read_nothing` — the writer's own guard, imported not copied — says
    the leg read nothing, which is the state `run_eval` nulls the metric for.
    Before this change the null reached `round(None, 4)` and the only trace was a
    log line naming a TypeError.
    """
    out, log = _call([_record(0), _record(0), _record(0)], _summary(None),
                     kg_db=_store_with_facts(tmp_path / "kg" / "kg.sqlite"))
    assert out["queries_path_exists"], (
        "without the real query set the reader scores nothing and both readings pass")
    assert out["metric"] is None, (
        "a fact leg that read nothing is not a measurement; the improve loop must "
        f"return null, got {out['metric']!r}")
    assert "read nothing" in log, (
        "the refusal must name the empty fact leg rather than a TypeError — a "
        f"destroyed reason is the defect this item is filed under: {log[-700:]}")


def test_a_numbered_fact_metric_still_comes_through(tmp_path):
    """The guard must not silence the healthy path: a real average stays a float."""
    recs = [_record(3, 1.0), _record(2, 0.5), _record(1, 0.0)]
    out, log = _call(recs, _summary(0.5))
    assert out["metric"] == 0.5, out
    assert "read nothing" not in log, log[-500:]


def test_a_null_this_reader_cannot_explain_is_still_never_a_zero(tmp_path):
    """A null with NO empty-leg evidence must not be laundered into a number.

    An empty fact index makes `fact_leg_read_nothing` False — clause 3 keys on a
    NON-empty index, because an empty tree is a different condition from an
    unreadable one — so a null here has no available explanation. It still returns
    null (a non-measurement is never 0.0) and the log still has to show it as the
    anomaly it is.
    """
    empty_db = tmp_path / "kg" / "kg.sqlite"
    empty_db.parent.mkdir(parents=True)
    out, log = _call([_record(0)], _summary(None), kg_db=empty_db)
    assert out["metric"] is None, out
    assert "could not be measured" in log, (
        "an unexplained null has to be visible as an anomaly, not swallowed: "
        f"{log[-700:]}")
