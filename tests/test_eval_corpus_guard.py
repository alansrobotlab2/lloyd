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
import contextlib
import json
import os
import re
import shutil
import statistics
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from app.paths import (ACCOUNT_HOME, LIVE_CHECKOUT, LLOYD_HOME,  # noqa: E402
                      production_data_root)

ROOT = Path(__file__).resolve().parent.parent
_VENV = ROOT / ".venvs" / "lloyd" / "bin" / "python"
PY = _VENV if _VENV.exists() else Path(sys.executable)
SCRIPT = ROOT / "eval" / "run_eval.py"
# run_eval.py writes its record under the data root the child inherits.
from app.paths import EVAL_BASELINES_DIR as BASELINES  # noqa: E402


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
         facts_root: Path | None = None,
         env_extra: dict | None = None) -> subprocess.CompletedProcess:
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
    # `env_extra` exists for #1374: the document-corpus probe is pointed at a
    # health URL the test controls, so the assertions below are about what the
    # run RECORDED rather than about how busy the live index happens to be.
    env.update(env_extra or {})
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


# ---------------------------------------------------------------------------
# The document half of the corpus (#1374)
#
# Everything above this line is about the fact half: an empty graph made a run
# indistinguishable from a healthy one. This section is about the OTHER input the
# headline numbers come from. `doc_hit`, `ndcg10` and `mrr_doc` are scored against
# the qmd daemon, which re-embeds continuously — 38 768 vectors at 14:11Z and
# 39 210 at 14:41Z on 2026-09-22, same process, up 54,860 s, no restart — and the
# artifact recorded none of it. So two nightlies could be compared as if the
# document corpus had held still when it had not. The keyword leg of the same
# score reads a checkout (`LLOYD_CODE_ROOT`), and that was never recorded either.
#
# A local HTTP server stands in for the daemon rather than a patched `urlopen`,
# because the thing under test is a run's answer about a service: a mock that
# replaces the client also removes the failure mode (no answer at all) that most
# of the nightly artifacts will eventually hit.
# ---------------------------------------------------------------------------

#: A daemon's `/health` body in the shape the live qmd answers with.
HEALTHY_HEALTH = {
    "status": "ok", "uptime": 57241,
    "rerank": {"ranked": 117, "fallbacks": 0},
    "vecIndex": {"vectors": 39_210, "fullBuilds": 1, "incrementalRefreshes": 243},
}


class _HealthHandler(BaseHTTPRequestHandler):
    """Answers GET /health with whatever body its server was handed."""

    def do_GET(self):
        body = json.dumps(self.server.body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Silence: a test fixture getting a request is not a log line."""


def _make_qmd_index(path: Path, *, rows: int = 7,
                    table: str = "content_vectors") -> Path:
    """Build a sqlite file shaped like qmd's index: a `content_vectors` table
    holding `rows` rows.

    A fixture that only *looks* like a database — a byte blob wearing a .sqlite
    suffix — makes the row-count probe return None forever. Then a renamed or
    missing table in production would be recorded as `unknown`, and every
    assertion written against the blob would still pass. The real qmd index has
    this table under this name; the recorder opens it `mode=ro`, exactly as it
    opens the live file while qmd's watcher is writing it.
    """
    import sqlite3

    con = sqlite3.connect(str(path))
    try:
        con.execute(f"create table {table} (id integer primary key, hash text)")
        con.executemany(f"insert into {table} (hash) values (?)",
                        [(f"row-{i}",) for i in range(rows)])
        con.commit()
    finally:
        con.close()
    return path


@contextlib.contextmanager
def _health_server(body: dict):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    srv.body = body
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/health"
    finally:
        srv.shutdown()
        srv.server_close()


@contextlib.contextmanager
def _closed_health_port():
    """A health URL on a port nothing is listening on.

    Bound then released, so the address is real and the connection is refused —
    what a stopped or restarting daemon looks like from the eval's side.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield f"http://127.0.0.1:{port}/health"


def _provision_healthy_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A populated fact tree AND a store that indexes it — and it stays populated.

    `test_a_zero_vector_count_...` needs the two halves of `corpus_ok` to be
    separable: the fact side has to be genuinely non-zero so that a false flag can
    only have come from the document side. `_provision_fact_tree` below is the same
    fixture emptied on purpose, which is the wrong world for that test.
    """
    facts = tmp_path / "healthy-facts"
    (facts / "lloyd").mkdir(parents=True)
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
    db = tmp_path / "kg-healthy.sqlite"
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
         "st = s.stats()\n"
         "assert st['facts'] == 1 and st['edges_active'] == 1, st\n"
         "assert st['entities'] >= 1, st\n"
         "s.close()\n",
         str(db), str(facts)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180,
        check=True)
    return facts, db


def test_a_completed_run_records_the_document_it_scored(tmp_path):
    """Clause 1: the artifact names the corpus the document leg actually searched.

    Three things had no home in the artifact before this: how many vectors the
    daemon held, which index answered, and which checkout the keyword leg grepped.
    Each is asserted with a value only the probe could have produced — a vector
    count from the fixture daemon's own body, and a code root the child was told
    about through the same env var `agent_mcp/vault.py` reads, so the recorder
    cannot be quietly reading a different tree than the scorer did.
    """
    label = "pytest-doc-corpus"
    _cleanup(label)
    facts, db = _provision_healthy_corpus(tmp_path)
    # A REAL sqlite index carrying a `content_vectors` table, not a byte blob with
    # a .sqlite suffix — see `_make_qmd_index`. Seven rows, asserted below.
    index = _make_qmd_index(tmp_path / "evalpin-snapshot.sqlite", rows=7)
    pinned_code_root = tmp_path / "armed-tree"
    with _health_server(HEALTHY_HEALTH) as url:
        proc = _run(tmp_path, "--label", label, kg_db=db, facts_root=facts,
                    env_extra={"LLOYD_QMD_HEALTH_URL": url,
                               "LLOYD_QMD_INDEX": str(index),
                               "LLOYD_CODE_ROOT": str(pinned_code_root)})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        rec = json.loads(written[0].read_text())

    doc = rec["corpus"]["doc"]
    assert doc["vectors"] == HEALTHY_HEALTH["vecIndex"]["vectors"], doc
    assert doc["health_url"] == url, doc
    # The index by three identities, because the path alone does not prove the
    # file did not change underneath the run: `index_mtime` does, and
    # `content_vectors` is the file's own count beside the daemon's — the pair is
    # what distinguishes "the corpus grew and was re-embedded" from "the file grew
    # and the daemon has not read it yet".
    assert doc["index_path"] == str(index), doc
    assert doc["index_size_bytes"] == index.stat().st_size, doc
    assert doc["index_mtime"], "the index has to be identifiable by more than its path"
    assert doc["index_reason"] is None, doc
    # The file's OWN row count, recorded beside the daemon's: the pair is what
    # tells "the corpus grew and was re-embedded" from "the file grew and the
    # daemon has not read it yet". Seven is what the fixture was built with, so
    # this asserts through a read-only sqlite open of a real database — the seam
    # production crosses against a file its watcher is writing.
    assert doc["content_vectors"] == 7, doc
    assert Path(doc["code_root"]) == pinned_code_root, doc
    # Identity, not liveness — the third contract in `app/doc_corpus.py`. `uptime`,
    # the rerank leg's counters and the vecIndex's fullBuilds/incrementalRefreshes
    # all count the daemon PROCESS: a restart over an index that did not move resets
    # incrementalRefreshes to 0 and takes fullBuilds to 1 while `vectors` stands
    # still. `eval/ci_backtest.py:125` canonicalises this whole block to decide
    # whether a pair gets a PAIRED test, so shipping any of them would cost the
    # instrument its power on exactly the same-day pairs #1374 is about.
    assert "uptime" not in doc and "uptime_s" not in doc, doc
    assert not [k for k in doc if k.startswith("rerank_")], doc
    assert "full_builds" not in doc and "incremental_refreshes" not in doc, doc
    assert rec["corpus_ok"] is True
    # Stdout too: the line is where an operator reading a nightly log sees the
    # corpus, and a fact-only line is what let this go unnoticed for a month.
    assert f"vectors={HEALTHY_HEALTH['vecIndex']['vectors']}" in proc.stdout, proc.stdout


def test_a_zero_vector_count_empties_the_corpus_even_with_a_healthy_graph(tmp_path):
    """Clause 2: the document half can empty the corpus on its own.

    The fixture's fact side is provably populated (the assertions below), so a
    false `corpus_ok` here has exactly one possible source — the daemon that
    answered 0 vectors. That is the case the fact-only guard could never see: an
    index that was wiped or rebuilt-empty scores every document query as a miss
    and reported, until now, `corpus_ok: true`.
    """
    empty = {**HEALTHY_HEALTH, "vecIndex": {**HEALTHY_HEALTH["vecIndex"],
                                           "vectors": 0}}
    label = "pytest-zero-vectors"
    _cleanup(label)
    facts, db = _provision_healthy_corpus(tmp_path)
    with _health_server(empty) as url:
        proc = _run(tmp_path, "--label", label, kg_db=db, facts_root=facts,
                    env_extra={"LLOYD_QMD_HEALTH_URL": url})
        out = proc.stdout + proc.stderr
        assert proc.returncode != 0, out
        assert "empty corpus" in out, out
        # The refusal names the half that emptied, with the address and the index
        # it looked at: the fact half of this world is intact, so a message that
        # only said "empty corpus" would send the operator to the wrong store.
        assert "doc corpus = vectors=0" in out, out
        assert url in out, out
        assert "empty-document-corpus baseline" in out, out
        # Refused before scoring: an empty index would produce a run of all-miss
        # document scores, which is the indistinguishable-from-real artifact this
        # file exists to prevent.
        assert not list(BASELINES.glob(f"{label}-*.json")), out

        # The flag means "measure the empty corpus deliberately" and now governs
        # the document half as it has always governed the fact half.
        proc = _run(tmp_path, "--label", label, "--allow-empty-corpus",
                    kg_db=db, facts_root=facts,
                    env_extra={"LLOYD_QMD_HEALTH_URL": url})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        rec = json.loads(written[0].read_text())

    # The fact half is populated in this world: one indexed fact, one active edge,
    # and the two entities that edge joins. Every term `corpus_ok` used to consult
    # is non-zero here, so the false flag below has one possible cause.
    assert rec["corpus"]["facts"] == 1, rec["corpus"]
    assert rec["corpus"]["edges_active"] == 1, rec["corpus"]
    assert rec["corpus"]["entities"] >= 1, rec["corpus"]
    assert rec["corpus"]["doc"]["vectors"] == 0
    assert rec["corpus_ok"] is False, (
        "an empty document corpus is an empty corpus: every doc_hit, ndcg10 and "
        "mrr_doc in the artifact would be a miss scored against nothing")


def test_an_unanswerable_probe_records_null_and_still_scores(tmp_path):
    """Clause 3 at the writer's end: no answer is recorded as null, never as 0.

    The daemon being down is not an empty corpus — it is an unanswered question —
    and the two must not land in the same place in the artifact. `0` here would
    make a stopped daemon look like a wiped index (and, three days later, would
    make the pair `doc_vectors +0` and read as a still corpus on both sides).
    """
    label = "pytest-doc-unanswerable"
    _cleanup(label)
    facts, db = _provision_healthy_corpus(tmp_path)
    with _closed_health_port() as url:
        proc = _run(tmp_path, "--label", label, kg_db=db, facts_root=facts,
                    env_extra={"LLOYD_QMD_HEALTH_URL": url})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, "an unanswerable probe must not stop the run"
        rec = json.loads(written[0].read_text())

    assert rec["corpus"]["doc"] is None, rec["corpus"]
    assert rec["corpus_ok"] is True, (
        "an unmeasured document corpus is unknown, not empty: the fact half still "
        "stands and the run still scored something")
    assert "vectors=unknown" in proc.stdout, proc.stdout
    assert "vectors=0" not in proc.stdout, proc.stdout


def test_the_document_block_is_the_identity_ci_backtest_pairs_on(tmp_path):
    """The process boundary #1374 does not name: this block is also a fingerprint.

    `eval/ci_backtest.py:125` canonicalises the WHOLE `corpus` dict and runs a PAIRED
    test only when two artifacts' fingerprints match; anything else falls back to an
    independent test with far less power. So the block has a second consumer with a
    demand the trend audit never made of it: equal fingerprints for an unchanged
    corpus, different ones when the vectors moved. The +442 drift this item measured
    on 2026-09-22 was invisible to that consumer as well.

    All three artifacts are REAL `run_eval` runs against the fixture daemon — two on
    one health answer, one on an answer 442 vectors higher. A `paired` assertion
    proves nothing if the artifacts it compares were assembled by this test instead
    of by the writer under test.
    """
    sys.path.insert(0, str(ROOT))
    import eval.ci_backtest as ci_backtest

    facts, db = _provision_healthy_corpus(tmp_path)
    index = _make_qmd_index(tmp_path / "index.sqlite", rows=7)
    # ONE daemon for all three runs, its body mutated in place between them. The
    # block records the `health_url` it reached as part of the corpus's identity, and
    # a fresh fixture server per run hands each run a different ephemeral port — the
    # fingerprint would then move while the corpus stood still, which is the same
    # defect as shipping `uptime` and is precisely what the equality assertion below
    # exists to catch. One daemon, one port, three answers.
    daemon_body = json.loads(json.dumps(HEALTHY_HEALTH))

    def _artifact(url: str, label: str) -> Path:
        _cleanup(label)
        proc = _run(tmp_path, "--label", label, kg_db=db, facts_root=facts,
                    env_extra={"LLOYD_QMD_HEALTH_URL": url,
                               "LLOYD_QMD_INDEX": str(index)})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        written = list(BASELINES.glob(f"{label}-*.json"))
        assert len(written) == 1, written
        return written[0]

    with _health_server(daemon_body) as url:
        paths = [_artifact(url, "pytest-bt-a"), _artifact(url, "pytest-bt-b")]
        daemon_body["vecIndex"] = dict(
            HEALTHY_HEALTH["vecIndex"],
            vectors=HEALTHY_HEALTH["vecIndex"]["vectors"] + 442)
        paths.append(_artifact(url, "pytest-bt-c"))
    try:
        a, b, c = (ci_backtest.Run(pth) for pth in paths)
        assert a.corpus is not None and a.records, a.path
        # The pair has to be joinable on query ids, or `paired` below is the default
        # answer `corpus_state` falls to rather than one it earned.
        assert set(a.records) == set(b.records), "the artifacts join on ids"
        assert a.corpus == b.corpus, (
            "two runs on an unchanged corpus must produce the same fingerprint. This "
            "is the assertion that breaks if anyone re-adds a per-process counter "
            "like `uptime` or `incrementalRefreshes` to `collect`: these runs are "
            "seconds apart, so any elapsed-time or process-lifetime field would "
            "differ while the corpus did not, and the backtest would quietly stop "
            "pairing nightlies.")
        assert ci_backtest.corpus_state(a, b) == "paired"
        # +442 vectors is the drift measured between 14:11Z and 14:41Z on
        # 2026-09-22 inside one daemon process that never restarted. The backtest
        # could not see it; now it refuses the pair instead of scoring it as change.
        assert ci_backtest.corpus_state(a, c) == "unpaired-corpus", (
            "a daemon that re-embedded 442 vectors is a different corpus")
    finally:
        for pth in paths:
            pth.unlink(missing_ok=True)


def test_the_health_url_follows_the_recall_to_the_pin_port(monkeypatch):
    """The probe must describe the daemon the RECALL used, not a port named here.

    Every subprocess test above reaches the probe through `LLOYD_QMD_HEALTH_URL`, so
    none of them exercises the derivation at all: the branch that survives a
    `PinnedCorpus` — whose daemon answers on :8182 while the live one answers on
    :8181 — would never be seen. If the two ever disagreed, one artifact would carry
    two corpora's provenance and look entirely healthy, which is the failure #1374 is
    filed under rather than one it fixes.
    """
    from app import doc_corpus

    monkeypatch.delenv(doc_corpus.HEALTH_URL_ENV, raising=False)
    assert doc_corpus.health_url_for("http://localhost:8181/query") == \
        "http://localhost:8181/health"
    assert doc_corpus.health_url_for("http://localhost:8182/query") == \
        "http://localhost:8182/health", "the PIN port, not the live one"
    # An explicit override still wins, and an unusable URL falls back to the live
    # daemon rather than raising inside a run that is about to score 87 queries.
    monkeypatch.setenv(doc_corpus.HEALTH_URL_ENV, "http://127.0.0.1:9/health")
    assert doc_corpus.health_url_for("http://localhost:8181/query") == \
        "http://127.0.0.1:9/health"
    monkeypatch.delenv(doc_corpus.HEALTH_URL_ENV, raising=False)
    # An empty URL yields an empty answer, not a raise and not a guessed port: the
    # run records `doc: null` and the audit prints `unknown`, rather than dying over
    # a provenance field or describing a daemon it never queried.
    assert doc_corpus.health_url_for("") == ""
    assert doc_corpus.collect(query_url="") is None


def test_a_pinned_run_never_inherits_the_live_index_path(monkeypatch, tmp_path):
    """Under an overlay, the DEFAULT index file belongs to a different daemon.

    `PinnedCorpus.env_for` names the snapshot through `LLOYD_QMD_INDEX`; when that
    did not happen, recording `~/.cache/qmd/index.sqlite` would put the live index's
    path, mtime and row count on an artifact whose every doc number came from a
    frozen copy — a provenance field that reads as precise and is wrong. So the
    answer is null plus a reason, and the reason is a sentence a reader of the
    artifact can act on.
    """
    from app import doc_corpus

    named = tmp_path / "evalpin.sqlite"
    named.write_bytes(b"x" * 64)
    monkeypatch.setenv(doc_corpus.INDEX_PATH_ENV, str(named))
    monkeypatch.setenv(doc_corpus.CONFIG_OVERLAY_ENV, str(tmp_path / "o.yaml"))
    assert doc_corpus.index_identity()["index_path"] == str(named), \
        "an explicitly named pin is used, overlay or not"

    monkeypatch.delenv(doc_corpus.INDEX_PATH_ENV)
    ident = doc_corpus.index_identity()
    assert ident["index_path"] is None, ident
    assert doc_corpus.PIN_UNNAMED_REASON in ident["index_reason"]
    assert ident["content_vectors"] is None and ident["index_size_bytes"] is None

    monkeypatch.delenv(doc_corpus.CONFIG_OVERLAY_ENV)
    assert doc_corpus.index_identity()["index_path"] == str(doc_corpus.LIVE_INDEX), \
        "with no overlay the live default is correct, and is still used"


def test_a_missing_content_vectors_table_records_unknown_not_zero(tmp_path):
    """The row count is a probe, and a probe with no table is *unknown*.

    Recording 0 here would sit beside a live daemon's `vectors: 39575` in the same
    artifact and read as "the file has nothing in it", which is a stronger and
    falsler claim than "I could not count".
    """
    from app import doc_corpus

    index = _make_qmd_index(tmp_path / "no-table.sqlite", rows=3, table="other_stuff")
    ident = doc_corpus.index_identity(index)
    assert ident["content_vectors"] is None, ident
    assert ident["index_path"] == str(index), "the file is still identifiable"


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

from tests._live_data import (KG_ENTITY_FLOOR, require_live_data,
                              require_live_entity_volume)

CORPUS = ROOT / "eval" / "vault_recall_queries.yaml"
# `_pipeline/` is gitignored (.gitignore:25), so a git worktree — including the
# automod round the gate runs this suite in — has no `kg.sqlite` at all. The
# fallback is `app/uptake.py:lloyd_root`'s rule for the same fact: measure the
# live tree, because the transcripts and the store live there and the worktree's
# absence of them is not a measurement of anything. Without it this file's entity
# guard would refuse in every worktree, which is a red suite that tells you
# nothing about the corpus.
LIVE_KG_DB = production_data_root() / "_pipeline" / "vault-derived" / "kg.sqlite"


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
                    names = list(st.entities.all())
                    where = str(st.path)
                finally:
                    st.close()
                # Present-but-stub is a third case that neither `require_live_data`
                # (which asks only whether the file is there) nor the zero-row
                # refusal below can see, and it is the one that blocks a promotion:
                # a 30-entity stub of unrelated rows makes 96 of this corpus's 100
                # `expect_entities` entries look unreachable, so the failure reads
                # as a corpus defect when it is a measurement of the store. The
                # helper owns the number; zero rows deliberately falls through to
                # the refusal below, and a caller that passed its own getter never
                # reaches this line, so no synthetic store can ever be skipped.
                require_live_entity_volume(names, path)
                return names, where

        # Neither route holds a store. The refusal below is still the right answer
        # for a caller that switched the fallback OFF — that caller is asserting the
        # refusal happens — but for the live route it became permanent on
        # 2026-09-22: kg.sqlite went with the tree and no copy exists on the box
        # (`find ~ -name kg.sqlite` returns nothing), so this reproduced red at base
        # in every round. The skip keeps the distinction clause 5 is about: an
        # unavailable store still does not become an unsatisfiable corpus, it just
        # says so by name instead of by blocking every promotion.
        require_live_data(LIVE_KG_DB, "the knowledge-graph store the eval scores "
                          f"against (also looked at {Path(VAULT_KG_DB)})", kind="file")

    # Nothing readable on either path (or a caller switched the fallback off):
    # the sanctioned reader produces the refusal, and it names the path it looked
    # for. Deciding "absent" here instead would be the very substitution clause 5
    # forbids — an unavailable store must not become an unsatisfiable corpus.
    st = ks.store()
    return list(st.entities.all()), str(st.path)


def _named_store(names):
    """A synthetic store holding exactly `names`, for a guard that must not read the graph.

    A guard that supplies its own corpus (`tmp_path`) has to supply its own store too.
    Before #1394 the ones below did not, so their verdict depended on how many rows
    the machine's live store happened to hold — and once a 30-row stub appeared, the
    non-vacuity probes were reading that stub and could not tell a working checker
    from a broken one. Passing a getter also puts them on the route where the live
    volume floor does not apply, which is the point: a probe that must fail can never
    be skipped out of existence by a stub on disk.
    """
    class _Entities:
        @staticmethod
        def all():
            return list(names)

    class _Store:
        path = Path("synthetic-kg.sqlite")
        entities = _Entities()

    return _Store


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
    report = _entity_satisfiability_report(getter=_named_store(["Knowledge Graph"]))
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
    report = _entity_satisfiability_report(
        corpus, getter=_named_store(["Knowledge Graph", "Lloyd Backlog System"]))
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


class _RowStore:
    """A stand-in `KGStore` opened at `path`, holding exactly `n_rows` entity rows.

    #1394 needs to exercise the LIVE route at a chosen row count — the count is the
    thing that decides skip versus verdict — and the only live store on the box is the
    30-row stub left by the 2026-09-22 loss. So the sanctioned opener is patched, which
    is the same pair `test_an_unreadable_store_raises_rather_than_reporting_every_name_absent`
    patches: `app.kg_store` stays the one module that opens the file.
    """

    opened: list = []

    def __init__(self, path, n_rows=30, prefix="Leftover Stub Entity"):
        self.path = Path(path)
        self._n = n_rows
        self._prefix = prefix
        _RowStore.opened.append(str(path))

    class _E:
        def __init__(self, owner):
            self._owner = owner

        def all(self):
            return [f"{self._owner._prefix} {i:05d}" for i in range(self._owner._n)]

    @property
    def entities(self):
        return _RowStore._E(self)

    def close(self):
        return None


def _patch_live_store(monkeypatch, n_rows, prefix="Leftover Stub Entity"):
    """Point the sanctioned opener at a store of `n_rows` rows; `store()` still refuses."""
    import app.kg_store as ks

    def refusing(*_a, **_k):
        raise ks.StoreUnavailable("the process-wide default is not this test's store")

    monkeypatch.setattr(ks, "store", refusing)
    monkeypatch.setattr(ks, "KGStore",
                        lambda path: _RowStore(path, n_rows, prefix))
    _RowStore.opened = []


def test_a_below_floor_live_store_skips_naming_floor_count_and_path(monkeypatch):
    """Clause 1: a present store below the floor is a named skip, not a red node.

    The failure this removes is the one that blocked every promotion on 2026-09-23: a
    30-entity / 0-edge stub sat at the live path, `require_live_data` saw a file and
    said nothing, and the guard reported 96 of 100 corpus expectations as unreachable
    — a statement about the store, reproduced red at base in four consecutive rounds.
    The skip has to carry all three numbers a reader needs (floor, observed rows, the
    path it opened) or it is indistinguishable from the guard having not looked.
    """
    _patch_live_store(monkeypatch, 30)

    with pytest.raises(pytest.skip.Exception) as exc:
        _entity_satisfiability_report()
    reason = str(exc.value)
    assert "holds 30 entity rows" in reason, reason
    assert f"the {KG_ENTITY_FLOOR}-entity floor" in reason, reason
    assert _RowStore.opened, "the guard never opened a store, so the path in its " \
        "skip reason could not have come from the route that read it"
    assert _RowStore.opened[0] in reason, f"{_RowStore.opened[0]!r} missing from {reason}"


def test_a_below_floor_store_reached_through_a_getter_is_a_verdict_not_a_skip(monkeypatch):
    """Clause 3's other half: the floor lives on the live route only.

    A caller that names its own store is asserting something about a store it built.
    If the machine's stub could skip that call, the probe would be reading the graph
    behind its own fake and passing for the wrong reason — the rule the getter branch
    of `_entity_store_names` already states for refusals, extended to skips.
    """
    _patch_live_store(monkeypatch, 30)

    try:
        report = _entity_satisfiability_report(getter=_named_store(["Knowledge Graph"]))
    except pytest.skip.Exception as exc:
        pytest.fail(f"a one-row getter store was skipped by the live volume floor, "
                    f"which means the guard read the machine's graph behind its own "
                    f"fake: {exc}")
    assert report["entity_names"] == 1, report
    # And it is a verdict, not a shrug: one row against the shipped corpus leaves most
    # expectations unreachable, the containment rule still reaches the rest, and the
    # failure names BOTH halves of an offender — its corpus query id and the expectation
    # text. The offender is read out of the report instead of written here: `9b028e9`
    # re-pointed 22 gold names against the rebuilt graph and dropped 5, so a hard-coded
    # `query -> 'name'` pair in this assert would go red on the next re-extraction for a
    # reason that has nothing to do with this clause. What must not move is that the
    # message renders the pair the report holds, with the denominator beside it.
    assert report["dead"], report
    assert len(report["dead"]) < report["expectations"], (
        f"with one row named `Knowledge Graph`, containment should still reach "
        f"several expectations; every one was reported dead instead: {report}")
    offender = report["dead"][0]
    with pytest.raises(AssertionError) as exc:
        _assert_entity_expectations_satisfiable(report)
    msg = str(exc.value)
    assert f"{offender['query']} -> {offender['expect']!r}" in msg, (
        f"the failure did not name both halves of the offender it reported "
        f"({offender}): {msg}")
    assert f"{len(report['dead'])} of {report['expectations']}" in msg, msg


def test_a_live_store_at_the_floor_reports_a_verdict_and_does_not_skip(monkeypatch):
    """The floor is a floor, not an off-switch: at the floor the guard answers.

    Without this pair the skip could be widened to "always skip" and every node in
    this file would still be green, which is the failure #878 was filed against.
    """
    _patch_live_store(monkeypatch, KG_ENTITY_FLOOR)

    try:
        report = _entity_satisfiability_report()
    except pytest.skip.Exception as exc:
        pytest.fail(f"a store holding exactly the {KG_ENTITY_FLOOR}-entity floor "
                    f"skipped instead of answering: {exc}")
    assert report["entity_names"] == KG_ENTITY_FLOOR, report
    # The stub names are not in the corpus, so the verdict here is a failing one —
    # a floor-passing store must still be able to lose.
    assert report["dead"], report
    with pytest.raises(AssertionError) as exc:
        _assert_entity_expectations_satisfiable(report)
    assert "expect_entities" in str(exc.value), exc.value


def test_both_kg_guards_route_through_the_one_floor_helper():
    """Clause 4: one floor, defined once, called by both guard files.

    Source-shape in this file's existing `inspect.getsource` style, because the thing
    to pin is that the two guards cannot disagree: a second literal anywhere in either
    guard is a floor someone can change in one place, which is how a guard silently
    becomes an always-skip.
    """
    import tests._live_data as live_data

    assert live_data.KG_ENTITY_FLOOR == KG_ENTITY_FLOOR == 1000, live_data.KG_ENTITY_FLOOR

    helper = inspect.getsource(live_data.require_live_entity_volume)
    assert "KG_ENTITY_FLOOR" in helper, helper
    assert "pytest.skip" in helper, helper
    assert "0 < count < KG_ENTITY_FLOOR" in helper, (
        f"the helper must skip only ABOVE zero rows — an empty store is a failing "
        f"verdict, not a skip: {helper}")

    guard_a = inspect.getsource(_entity_store_names)
    assert "require_live_entity_volume(" in guard_a, guard_a
    assert "KG_ENTITY_FLOOR" not in guard_a and "1000" not in guard_a, (
        f"the corpus guard re-states the floor instead of calling the helper: {guard_a}")

    import tests.test_retrieval_seed_anchoring as seed_anchoring

    guard_b = inspect.getsource(seed_anchoring._run_against_the_live_corpus)
    assert "require_live_entity_volume(" in guard_b, guard_b
    assert "KG_ENTITY_FLOOR" not in guard_b and "1000" not in guard_b, (
        f"the seed-anchoring runner re-states the floor instead of calling the "
        f"helper: {guard_b}")


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


# #1354: labels whose answer is a doc in the LLOYD CHECKOUT, not the vault.
# `{label prefix: checkout-relative directory}`. The prefix is the name of the qmd
# collection that would index that directory, so it is exactly the path the
# scorer would see (`vault_recall` returns `<collection>/<rel>`). A label carrying
# it is resolved against the checkout ONLY: the vault holds stale copies of these
# docs (`projects/lloyd/architecture/qmd.md`, `architecture/*.md`), and a label
# satisfied by one of those would pass this guard while pointing the metric at the
# wrong document. The hyphen keeps the scorer anchored the same way —
# `lloyd_architecture/` normalised is contained in no vault path — which
# `test_a_checkout_label_is_not_satisfied_by_a_stale_vault_copy` pins.
CHECKOUT_LABEL_ROOTS = {"lloyd-architecture/": "architecture"}


#: Passed as `collections` by a caller with no collection list to check against —
#: the holdout corpus today. Distinct from `[]`, which would be an assertion that
#: NO collection is deployed, and distinct from `None`, which means the list could
#: not be read (#696's three-way shape again).
NO_COLLECTION_LIST = object()


def _deployed_collections(index_path: Path = None) -> list[dict]:
    """`[{name, root}]` for the collections the qmd index actually serves.

    Re-implemented per this file's standing convention (header at
    `tests/test_eval_corpus_guard.py:1567-1573`): the guard reads the table the
    deployment writes instead of importing the instrument's reader, so the guard
    cannot be green because it agrees with the code it guards. The two
    implementations are pinned against each other by
    `test_the_guard_and_the_eval_agree_on_which_collections_are_deployed`.

    `store_collections` is qmd's own table — the one the live proof command
    `select name, path from store_collections` answers with — opened `mode=ro`.
    Raises rather than returning `[]` when the file or the table is missing: an
    empty list would assert that nothing is deployed and every label in the corpus
    would read unsearchable, which is the same fabrication
    `test_an_unreadable_store_raises_rather_than_reporting_every_name_absent`
    refuses for the entity leg.
    """
    import sqlite3
    path = Path(index_path) if index_path is not None \
        else Path.home() / ".cache" / "qmd" / "index.sqlite"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select name, path from store_collections order by name").fetchall()
    finally:
        con.close()
    return [{"name": str(n), "root": str(p)} for n, p in rows]


def _served_paths(collections: list[dict]) -> list[str]:
    """Every path a deployed collection could return, shaped as the retriever shapes
    it (`<collection>/<root-relative path>`) and normalised.

    Mirrors `indexed_paths` in `eval/run_eval.py` deliberately, for the reason at
    :1567-1573: a guard and an instrument sharing an import cannot disagree, so "the
    guard waved it through and the artifact named it" is only catchable when the rule
    is held twice. The prefix is the shape: qmd returns `<name>/<path relative to that
    collection's root>`, which is why the whole-vault `subliminal` collection serves a
    vault-relative label like `projects/lloyd/architecture/qmd.md` — an earlier draft
    that matched a label's first segment against a collection NAME called that label
    unsearchable, because no collection is named `projects`.
    """
    pool = []
    for coll in collections:
        name = str(coll.get("name") or "")
        root = Path(str(coll.get("root") or coll.get("path") or "")).expanduser()
        if not name or not root.is_dir():
            continue
        pool.extend(_walk_normed(root, name + "/"))
    return pool


def _label_returnable(label: str, pool: list[str]) -> bool:
    """Could a deployed collection hand back something this gold label scores?

    Containment, which is this guard's own doc rule everywhere else (see `want in n`
    below) and the substring semantics `_doc_pair_satisfied` implements in the scorer.
    Labels are SUBSTRINGS, not paths — `expect_docs: qmd` is legal and matches
    `knowledge/qmd.md` — so a `Path.exists()` test on the label cannot ask this
    question: the first draft of #1599 did exactly that and called 12 of 86 queries
    unreturnable where only 5 are, and a finding wrong in that direction hands a
    reader a list of gold to delete, which is the corpus change the item reserves for
    a person.
    """
    want = _doc_norm(label)
    return any(want in n for n in pool)


def _walk_normed(root: Path, prefix: str = "") -> list[str]:
    """Every path under `root` (files and dirs), `prefix` + root-relative,
    normalised, minus any dot-prefixed component — qmd's own indexing rule."""
    out = []
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        out.append(_doc_norm(prefix + str(rel)))
    return out


def _doc_label_satisfiability_report(
        specs_path: Path = CORPUS,
        vault_root: Path = Path.home() / "obsidian",
        checkout_root: Path = LLOYD_HOME,
        collections: list[dict] = NO_COLLECTION_LIST) -> dict:
    """Walk both roots once and return which `expect_docs` labels match no path.

    THREE states, not two (#1599). A label can match no path anywhere (`dead`, what
    this guard has always failed on); can match a path under a root while NO
    deployed, searchable qmd collection indexes that root (`unsearchable` — filed
    and green-lit for three nights because existence and searchability were the same
    question here); or match a path a deployed collection serves (`satisfied_by`).
    `unsearchable` entries carry the query id, the label and the root that resolved
    it, and are excluded from `dead` so an unresolvable label and an unservable one
    cannot be told apart by a reader who only counts.

    `collections` is what makes the middle state expressible at all: which
    collections are deployed is a fact about the machine, not this repo, so a caller
    that has no list passes `NO_COLLECTION_LIST` and gets `unsearchable: []` plus
    `collections_checked: None` — a check that did not run, which is not the same
    thing as a clean corpus. The live list comes from `_deployed_collections()`.

    Two named roots (#1354). A label that starts with a `CHECKOUT_LABEL_ROOTS`
    prefix is resolved against that directory of the lloyd checkout and nowhere
    else; every other label against the vault. `satisfied_by` records, per label,
    which root answered it (`vault` / `checkout`), so a label that silently moves
    between roots is visible; a label neither root holds is in `dead`.
    `checkout_root` defaults to `app.paths.LLOYD_HOME`, the tree this suite is
    running from — in a worktree that is the worktree, which carries the docs.

    `path` is every path under a root (files AND directories, root-relative),
    minus any path with a dot-prefixed component. That is qmd's own indexing rule
    (`qmd/src/cli/qmd.ts`, the `parts.some(part => part.startsWith("."))` filter
    after the glob), so `skills/.archived/**` is NOT a location the index can
    hold: the live index had 0 such documents on 2026-09-21, and the five labels
    this walk used to accept there could never be returned. `.git/**` falls
    under the same rule. `walked_paths` (vault) and `walked_checkout_paths` are
    returned so the zero is checkable in the same breath as the verdict — a
    denominator of zero is not a pass.
    """
    specs = yaml.safe_load(specs_path.read_text())["queries"]
    labels = [(str(s.get("id")), str(d))
              for s in specs for d in (s.get("expect_docs") or [])]
    vault_normed = _walk_normed(vault_root)
    checkout_normed = [n for prefix, sub in CHECKOUT_LABEL_ROOTS.items()
                       for n in _walk_normed(checkout_root / sub, prefix)]
    checkout_prefixes = tuple(_doc_norm(p) for p in CHECKOUT_LABEL_ROOTS)
    # The third corpus this report reads. The vault and the checkout answer "does a
    # path like this exist"; only the deployed collections answer "could one be
    # returned", and it costs one walk of every served root — the same price the two
    # root walks above already pay.
    served_pool = ([] if collections is NO_COLLECTION_LIST
                   else _served_paths(collections))
    dead, satisfied_by, unsearchable = [], [], []
    for qid, label in labels:
        want = _doc_norm(label)
        root, pool = (("checkout", checkout_normed)
                      if want.startswith(checkout_prefixes)
                      else ("vault", vault_normed))
        if any(want in n for n in pool):
            # Resolved: a root holds something the scorer could match. Whether a
            # deployed collection can RETURN such a path is the separate question
            # #1599 asks, and only the collection list can answer it.
            if (collections is NO_COLLECTION_LIST
                    or _label_returnable(label, served_pool)):
                satisfied_by.append({"query": qid, "label": label, "root": root})
            else:
                unsearchable.append({"query": qid, "label": label, "root": root})
        else:
            dead.append({"query": qid, "label": label})
    return {"vault": str(vault_root), "checkout": str(checkout_root),
            "corpus": str(specs_path),
            "queries": len(specs), "labels": len(labels),
            "walked_paths": len(vault_normed),
            "walked_checkout_paths": len(checkout_normed),
            "collections_checked": (None if collections is NO_COLLECTION_LIST
                                    else len(collections)),
            "satisfied_by": satisfied_by, "unsearchable": unsearchable,
            "dead": dead}


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
    checkout_labels = [d for d in rep["satisfied_by"] + rep["dead"]
                       if any(d["label"].startswith(pfx) for pfx in CHECKOUT_LABEL_ROOTS)]
    if checkout_labels:
        assert rep["walked_checkout_paths"] > 10, (
            f"the corpus names {len(checkout_labels)} checkout labels but the walk "
            f"under {rep['checkout']} found {rep['walked_checkout_paths']} paths")
    assert rep["dead"] == [], (
        f"{len(rep['dead'])} of {rep['labels']} expect_docs labels in "
        f"{rep['corpus']} match no path under {rep['vault']} or "
        f"{rep['checkout']} (walked {rep['walked_paths']} + "
        f"{rep['walked_checkout_paths']} paths): "
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



# #1354 clause 1: the harness, the automod gate, retrieval, qmd and djev — one
# current checkout doc each, named by the label convention above.
CHECKOUT_DOCS_REQUIRED = ("harness.md", "automod.md", "retrieval.md", "qmd.md", "djev.md")


def test_a_checkout_rooted_label_is_attributed_to_the_checkout_and_never_the_vault(
        tmp_path):
    """A checkout-prefixed label resolves against the CHECKOUT, not the vault — the
    half of #1354 clause 1 that outlives the five queries it was written on.

    This node used to read
    `test_the_corpus_asks_questions_the_checkout_architecture_docs_answer` and asserted
    ">= 5 gold queries resolve to current checkout docs", which was #1354's deliverable.
    Alan's ruling on #1662 — "(c) Drop or relabel the five now: the labels are the bug"
    — retires those five (#1354's own, whose answer lives in `~/lloyd/architecture/`,
    which no deployed collection indexes), so the presence assertion is now false by
    design and re-pointing it is required. What stays is the mechanism the presence
    happened to exercise: `projects/lloyd/architecture/qmd.md` sitting in the VAULT is a
    stale copy, and a label that matches it must never be reported satisfied by the
    vault. The trap does not need the five queries to exist to be reachable, so it is
    exercised here on a fixture tree.

    The live corpus is asserted empty of checkout-rooted labels, which is #1662's
    positive claim, and the fixture is what makes that emptiness a measurement rather
    than an untested default.
    """
    rep = _doc_label_satisfiability_report()
    by_checkout = [d for d in rep["satisfied_by"] if d["root"] == "checkout"]
    assert by_checkout == [], (
        f"labels resolved against the checkout: {by_checkout}. #1662 retired the five "
        "that were there, so a new one means either a collection was deployed (then "
        "the retirement note in the gold file and the five ids it names are stale too) "
        "or a checkout doc was indexed by nothing and the query scores a permanent miss")

    # The calibration that makes the emptiness above mean something: `root` has to be
    # capable of reading "checkout" at all, or `by_checkout == []` is a property of the
    # guard rather than of the corpus. One fixture query naming a doc the fixture
    # checkout holds — the stale-vault-copy trap is the neighbouring node's business,
    # not this one's.
    vault, checkout = _two_root_fixture(tmp_path)
    fixture = _doc_label_satisfiability_report(
        specs_path=_write_corpus(tmp_path, [
            {"id": "checkout-side", "query": "how does the harness freeze its prompt",
             "expect_docs": ["lloyd-architecture/harness.md"]},
            {"id": "vault-side", "query": "what does the live note say",
             "expect_docs": ["knowledge/live-note.md"]}]),
        vault_root=vault, checkout_root=checkout)
    resolved = {d["query"]: d["root"] for d in fixture["satisfied_by"]}
    assert resolved == {"checkout-side": "checkout", "vault-side": "vault"}, (
        f"expected the fixture's checkout-prefixed label to resolve against the "
        f"checkout and its vault-rooted label against the vault — `root` is only "
        f"evidence if it can say both; got {fixture['satisfied_by']} "
        f"/ dead={fixture['dead']}")


def _two_root_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A vault holding a stale `projects/lloyd/architecture/qmd.md` and a real
    note, and a checkout whose `architecture/` holds only `harness.md`."""
    vault = tmp_path / "vault"
    (vault / "projects" / "lloyd" / "architecture").mkdir(parents=True)
    (vault / "projects" / "lloyd" / "architecture" / "qmd.md").write_text("stale\n")
    (vault / "architecture").mkdir()
    (vault / "architecture" / "harness.md").write_text("stale too\n")
    (vault / "knowledge").mkdir()
    (vault / "knowledge" / "live-note.md").write_text("x\n")
    checkout = tmp_path / "checkout"
    (checkout / "architecture").mkdir(parents=True)
    (checkout / "architecture" / "harness.md").write_text("current\n")
    return vault, checkout


def test_a_checkout_label_is_not_satisfied_by_a_stale_vault_copy(tmp_path):
    """#1354 clause 3: a label naming a checkout doc that does not exist is dead
    even though the vault's `projects/lloyd/architecture/` holds a same-named
    file — a stale document must never be scored as the current one's answer."""
    vault, checkout = _two_root_fixture(tmp_path)
    corpus = _write_corpus(tmp_path, [
        {"id": "gone", "query": "q", "expect_docs": ["lloyd-architecture/qmd.md"]},
        {"id": "live", "query": "q", "expect_docs": ["lloyd-architecture/harness.md"]},
    ])
    rep = _doc_label_satisfiability_report(specs_path=corpus, vault_root=vault,
                                           checkout_root=checkout)
    assert rep["dead"] == [{"query": "gone", "label": "lloyd-architecture/qmd.md"}], rep
    assert rep["satisfied_by"] == [
        {"query": "live", "label": "lloyd-architecture/harness.md", "root": "checkout"}], rep

    # The scorer's own rule (`_doc_pair_satisfied`: normalised substring of the
    # returned path) is anchored by the same prefix. The stale vault copy's path,
    # as recall would return it, does not contain the label...
    stale_returned = _doc_norm("projects/lloyd/architecture/qmd.md")
    assert _doc_norm("lloyd-architecture/qmd.md") not in stale_returned
    # ...the collection's own path does...
    assert _doc_norm("lloyd-architecture/qmd.md") in _doc_norm("lloyd-architecture/qmd.md")
    # ...and the unanchored spelling is exactly the trap the prefix avoids.
    assert _doc_norm("lloyd/architecture/qmd.md") in stale_returned


def test_the_guard_names_the_root_that_satisfied_each_label_and_neither_is_dead(tmp_path):
    """#1354 clause 2: two named roots, the answering root recorded per label, and
    a label present under neither still reported dead."""
    vault, checkout = _two_root_fixture(tmp_path)
    corpus = _write_corpus(tmp_path, [
        {"id": "v", "query": "q", "expect_docs": ["knowledge/live-note.md"]},
        {"id": "c", "query": "q", "expect_docs": ["lloyd-architecture/harness.md"]},
        {"id": "neither", "query": "q",
         "expect_docs": ["knowledge/never-written.md", "lloyd-architecture/never-written.md"]},
    ])
    rep = _doc_label_satisfiability_report(specs_path=corpus, vault_root=vault,
                                           checkout_root=checkout)
    assert {(d["query"], d["root"]) for d in rep["satisfied_by"]} == {
        ("v", "vault"), ("c", "checkout")}, rep
    assert rep["dead"] == [
        {"query": "neither", "label": "knowledge/never-written.md"},
        {"query": "neither", "label": "lloyd-architecture/never-written.md"}], rep
    assert rep["walked_paths"] >= 5 and rep["walked_checkout_paths"] == 1, rep

    # A vault label is never satisfied by the checkout: `architecture/harness.md`
    # resolves in this vault, and with the vault copy gone it is dead even though
    # the checkout still has that file.
    (vault / "architecture" / "harness.md").unlink()
    (vault / "architecture").rmdir()
    only_checkout = _write_corpus(tmp_path, [
        {"id": "vault-label", "query": "q", "expect_docs": ["architecture/harness.md"]}],
        name="only.yaml")
    rep2 = _doc_label_satisfiability_report(specs_path=only_checkout, vault_root=vault,
                                            checkout_root=checkout)
    assert rep2["dead"] == [{"query": "vault-label", "label": "architecture/harness.md"}], rep2


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


# ---------------------------------------------------------------------------
# #1412: the HOLDOUT leg goes through the same two guards, reported as counts
# ---------------------------------------------------------------------------
# The holdout corpus is only worth having while no job reads its per-query rows,
# and a guard failure message is a per-query row: it lands in pytest output, in the
# gate's rung report and in a round's transcript. So the holdout file is checked by
# the SAME report functions as the dev file (one definition of "satisfiable"), and
# asserted through a message that carries the counts and never a query id.

HOLDOUT_CORPUS = ROOT / "eval" / "vault_recall_holdout_queries.yaml"
HOLDOUT_MIN_QUERIES = 20


def _assert_holdout_satisfiable_by_count(report: dict, what: str, total_key: str) -> None:
    # An explicit raise, not `assert not dead`: pytest's assertion rewriting prints
    # the operand, and the operand is the list of reserved ids.
    dead = report["dead"]
    if dead:
        raise AssertionError(
        f"{len(dead)} of {report[total_key]} {what} across {report['queries']} holdout "
        f"queries of {report['corpus']} cannot be satisfied — re-point or delete them "
        "WITHOUT reading the retriever's output for these queries, then rewrite the "
        "manifest (`python -m eval.retrieval_holdout write`). Ids are withheld on "
        "purpose: this file is the reserved holdout leg (#1412).")


def test_the_holdout_corpus_has_the_dev_schema_and_disjoint_ids():
    holdout = yaml.safe_load(HOLDOUT_CORPUS.read_text())["queries"]
    dev = yaml.safe_load(CORPUS.read_text())["queries"]
    assert len(holdout) >= HOLDOUT_MIN_QUERIES, len(holdout)
    for spec in holdout:
        assert {"id", "query", "category", "expect_entities", "expect_docs"} <= set(spec), (
            "a holdout query is missing a dev-schema key")
        assert isinstance(spec["expect_entities"], list) and isinstance(spec["expect_docs"], list)
        assert spec["expect_docs"], "every holdout query names at least one document"
    ids = [str(s["id"]) for s in holdout]
    assert len(set(ids)) == len(ids)
    overlap = set(ids) & {str(s["id"]) for s in dev}
    assert not overlap, f"{len(overlap)} holdout id(s) also sit in the dev corpus"


def test_every_holdout_entity_expectation_is_satisfiable_reported_as_a_count():
    report = _entity_satisfiability_report(HOLDOUT_CORPUS)
    assert report["queries"] >= HOLDOUT_MIN_QUERIES, report["queries"]
    assert report["expectations"] >= 10, report["expectations"]   # the loop read them
    _assert_holdout_satisfiable_by_count(report, "`expect_entities` entries", "expectations")


def test_every_holdout_document_label_is_satisfiable_reported_as_a_count():
    rep = _doc_label_satisfiability_report(specs_path=HOLDOUT_CORPUS)
    assert rep["labels"] >= HOLDOUT_MIN_QUERIES and rep["walked_paths"] > 100, (
        rep["labels"], rep["walked_paths"])
    _assert_holdout_satisfiable_by_count(rep, "`expect_docs` labels", "labels")


def test_the_holdout_guard_fails_on_a_dead_expectation_and_names_no_id(tmp_path):
    """Non-vacuity, and the leak shape: the failure is real and carries no id."""
    corpus = tmp_path / "holdout.yaml"
    corpus.write_text(yaml.safe_dump({"queries": [
        {"id": "reserved-probe-alpha", "query": "q", "category": "single",
         "expect_entities": ["Backlog Item #363"], "expect_docs": ["lloyd"]},
        {"id": "reserved-probe-beta", "query": "q", "category": "single",
         "expect_entities": ["Knowledge Graph"], "expect_docs": ["lloyd"]}]}))
    report = _entity_satisfiability_report(corpus, getter=_named_store(["Knowledge Graph"]))
    assert report["expectations"] == 2 and len(report["dead"]) == 1, report
    with pytest.raises(AssertionError) as exc:
        _assert_holdout_satisfiable_by_count(report, "`expect_entities` entries", "expectations")
    msg = str(exc.value)
    assert "1 of 2" in msg, msg
    assert "reserved-probe" not in msg and "Backlog Item #363" not in msg, msg


# ── #1599: a label can RESOLVE and still be unreturnable ────────────────────
#
# Until tonight this guard asked one question of a gold doc label — does some path
# under a named root match it — and answered yes for five labels that no deployed
# qmd collection indexes: `lloyd-architecture/{harness,automod,retrieval,qmd,djev}.md`
# sat in `satisfied_by` with `root: checkout`, `dead` was empty, and the test at
# `test_committed_corpus_has_no_unresolvable_document_label` was green while those
# five scored `doc_hit: false` with `docs_matched: []` on every one of the three
# nights they have existed (measured 2026-09-27: `doc_hit_rate` 0.6279 with the five
# ids, 0.6667 without). Existence and searchability were the same question, so the
# condition could not be reported. The middle state below is the second question.

def test_a_label_that_resolves_but_no_deployed_collection_covers_it_is_unsearchable(tmp_path):
    """Clause 1 of #1599: three states, not two, on a fixture corpus.

    The four labels are the four shapes that exist in the committed corpus, and the
    assertion that matters is that `dead` stays EMPTY while `unsearchable` is not:
    an unresolvable label and an unservable one are different findings, and the
    second is the one this guard was blind to.
    """
    vault, checkout = _two_root_fixture(tmp_path)
    corpus = _write_corpus(tmp_path, [
        {"id": "vault-note", "query": "q", "expect_docs": ["knowledge/live-note.md"]},
        # A vault-relative label with NO `projects` collection deployed, served by
        # the whole-vault collection. This is the false-alarm trap: `subliminal` is
        # rooted at the vault itself, so a rule that only ever matched a label's
        # first segment against a collection NAME would file this as unsearchable.
        {"id": "vault-relative", "query": "q",
         "expect_docs": ["projects/lloyd/architecture/qmd.md"]},
        # The #1599 shape: resolves against the checkout, indexes nothing.
        {"id": "checkout-doc", "query": "q",
         "expect_docs": ["lloyd-architecture/harness.md"]},
        {"id": "never-written", "query": "q",
         "expect_docs": ["knowledge/never-written.md"]},
    ])
    collections = [
        {"name": "knowledge", "root": str(vault / "knowledge")},
        {"name": "subliminal", "root": str(vault)},
        {"name": "architecture", "root": str(vault / "architecture")},
    ]
    rep = _doc_label_satisfiability_report(
        specs_path=corpus, vault_root=vault, checkout_root=checkout,
        collections=collections)
    assert rep["collections_checked"] == 3, rep
    assert rep["satisfied_by"] == [
        {"query": "vault-note", "label": "knowledge/live-note.md", "root": "vault"},
        {"query": "vault-relative", "label": "projects/lloyd/architecture/qmd.md",
         "root": "vault"},
    ], rep["satisfied_by"]
    assert rep["unsearchable"] == [
        {"query": "checkout-doc", "label": "lloyd-architecture/harness.md",
         "root": "checkout"},
    ], rep["unsearchable"]
    assert rep["dead"] == [{"query": "never-written",
                            "label": "knowledge/never-written.md"}], rep["dead"]

    # And the finding is about the DEPLOYMENT, not the label: the moment a person
    # registers the collection the label names — option (a) on #1599, which a round
    # may not do — the same corpus reports no unsearchable label at all, through the
    # same predicate. Nothing here rewrites a gold label to get that result.
    rep2 = _doc_label_satisfiability_report(
        specs_path=corpus, vault_root=vault, checkout_root=checkout,
        collections=collections + [
            {"name": "lloyd-architecture", "root": str(checkout / "architecture")}])
    assert rep2["unsearchable"] == [], rep2
    # Registering a collection moves a label from unsearchable to satisfied and
    # NOTHING else: `never-written` matches no path under either root, so it stays
    # dead under every deployment. The two findings are independent, and an
    # assertion that let this one empty would not notice a report conflating them.
    assert rep2["dead"] == rep["dead"], rep2["dead"]
    assert {"query": "checkout-doc", "label": "lloyd-architecture/harness.md",
            "root": "checkout"} in rep2["satisfied_by"], rep2["satisfied_by"]


def test_the_committed_corpus_reports_three_states_with_no_dead_label(tmp_path):
    """Clause 1's committed-corpus half: with the live collection list the corpus
    still has NO dead label, and every label lands in exactly one of the three
    states.

    The invariant is the half that cannot rot silently: `CHECKOUT_LABEL_ROOTS` makes
    a label resolvable against the checkout tree, and if no deployed collection is
    named for that prefix then EVERY label under it is a permanent doc miss — so
    they must all be reported as unsearchable, never satisfied. That conditional
    needs no knowledge of the five ids, so a sixth checkout label filed tomorrow is
    covered by the same line, which is the recurrence #1599 exists to stop.

    #1662 then retired the five, which is what the conditional was instantiated on.
    The corpus has no checkout-prefixed label left, so `under` is legitimately empty
    here and the loop reads as an implication with no antecedent — which is exactly
    how a guard goes quietly toothless. The injection at the end of this node is what
    keeps the conditional honest: a label named on a real checkout doc that no query
    ever referenced must come back `unsearchable` while no collection is named for
    the prefix, and the same label must be `satisfied_by` once one is.
    """
    colls = _deployed_collections()
    assert colls, "no deployed collection list readable; the check has no denominator"
    plain = _doc_label_satisfiability_report()
    assert plain["collections_checked"] is None, (
        "a caller that passed no list must report a check that did not run")
    assert plain["unsearchable"] == [], (
        "without a collection list there is nothing to check against; reporting "
        "labels as unsearchable would be an invented finding")
    rep = _doc_label_satisfiability_report(collections=colls)
    assert rep["labels"] > 0, rep
    assert rep["walked_paths"] > 100, rep
    assert rep["collections_checked"] == len(colls)
    assert rep["dead"] == [], (
        f"{len(rep['dead'])} of {rep['labels']} expect_docs labels match no path: "
        + "; ".join(f"{d['query']} -> {d['label']}" for d in rep["dead"]))

    states = {name: {(d["query"], d["label"]) for d in rep[name]}
              for name in ("satisfied_by", "unsearchable", "dead")}
    total = set().union(*states.values())
    assert len(total) == rep["labels"], (
        "a label reached two states or none, so the three-way split is not a "
        f"partition: {rep['labels']} labels, {len(total)} classified")
    for name in ("satisfied_by", "unsearchable"):
        for other in ("satisfied_by", "unsearchable"):
            if other != name:
                assert not (states[name] & states[other]), (name, other,
                                                            states[name] & states[other])
    for prefix in CHECKOUT_LABEL_ROOTS:
        named = [c for c in colls
                 if _doc_norm(c["name"]) == _doc_norm(prefix.rstrip("/"))]
        under = {k for k in total if k[1].startswith(prefix)}
        if not named:
            assert under <= states["unsearchable"], (
                f"{len(under)} labels resolve under the checkout prefix {prefix} "
                f"while no collection is named {prefix!r}, so each is a permanent "
                f"doc miss; reported: {sorted(states['unsearchable'])}")
            # Vacuity stated, not assumed. On the committed corpus `under` is EMPTY
            # since #1662 retired the five, so the assertion above has no antecedent
            # here — which is precisely how a guard of this shape goes silent. Two
            # things keep it honest: the injection at the end of this node, which
            # manufactures a label to prove the conditional bites, and this line,
            # which fails the moment a checkout-prefixed label reappears with no
            # collection behind it, i.e. the moment the retirement is undone.
            assert under == set(), (
                f"{sorted(k[1] for k in under)}: a checkout-prefixed label is in the "
                f"corpus while no deployed collection is named {prefix!r}. Each such "
                f"label scores a permanent doc miss — the state #1662 retired five "
                f"queries to clear. Name the collection or retire the label.")
    # Post-#1662 the corpus asserts no permanent doc miss at all. This is NOT the
    # assertion #1599 left here, which named the five query ids as the corpus's
    # instance of the condition: those queries are retired, so that set is empty by
    # construction and an `<=`-style inclusion over it would have gone red for
    # doing the work and green for not reading the file.
    unsearched = {d["query"] for d in rep["unsearchable"]}
    assert unsearched == set(), (
        f"{len(unsearched)} gold queries score a doc leg no deployed collection can "
        f"answer: {sorted(states['unsearchable'])}. #1662 retired the five that were "
        "here; a new one is either the same bug again or the collection still is not "
        "deployed, and both are a finding, not a corpus to keep")
    assert unsearched <= {str(s.get("id")) for s in
                          yaml.safe_load(CORPUS.read_text())["queries"]}, unsearched

    # The implication above must be able to FAIL, and after the drop nothing in the
    # corpus can make it. So the sixth label is manufactured: the first checkout doc
    # in file order under `architecture/` — a real file in this tree, one no query has
    # ever named, so it is a label that has never existed in the corpus and the
    # conditional is being asked about a case the corpus itself cannot supply — under
    # the checkout prefix, against the live list, which still names no
    # `lloyd-architecture` collection. Reported `unsearchable` is the conditional
    # firing; a label that came back `satisfied_by` here would mean the guard had
    # quietly stopped resolving the prefix, which is how the five went unnoticed for
    # four days.
    docs = sorted(p.name for p in (LLOYD_HOME / "architecture").glob("*.md"))
    assert docs, f"no checkout docs under {LLOYD_HOME / 'architecture'} to inject"
    injected_label = f"lloyd-architecture/{docs[0]}"
    corpus6 = tmp_path / "sixth-label.yaml"
    corpus6.write_text(yaml.safe_dump({"queries": [{
        "id": "sixth-checkout-label", "query": "injected", "category": "technical",
        "expect_entities": [], "expect_docs": [injected_label]}]}, sort_keys=False))
    six = _doc_label_satisfiability_report(specs_path=corpus6, collections=colls)
    assert [(d["query"], d["label"]) for d in six["unsearchable"]] == [
        ("sixth-checkout-label", injected_label)], six
    assert six["satisfied_by"] == [], (
        f"{injected_label} read satisfied with no collection named for the prefix, "
        "so the checkout half of the walk is not being resolved as a second root")
    # And registering the collection is the fix, not a re-label: same injected
    # corpus, one more collection, and the finding is gone through the same rule
    # #1662's whole premise rested on.
    served = _doc_label_satisfiability_report(
        specs_path=corpus6,
        collections=colls + [{"name": "lloyd-architecture",
                              "root": str(LLOYD_HOME / "architecture")}])
    assert served["unsearchable"] == [], served
    assert [d["label"] for d in served["satisfied_by"]] == [injected_label], served
    # Bounded against the corpus size the way #606 insists a gold-side finding be.
    # 5 of 144 labels is a work list; 144 of 144 is an empty or broken collection
    # list, and the two must not be printable as the same sentence — a reader told
    # "every gold doc in the corpus is unreturnable" would be advised to delete the
    # gold set, which is the corpus change this item reserves for a person. A list
    # read from the wrong index, or a `store_collections` rename, trips here.
    assert len(states["unsearchable"]) <= rep["labels"] // 4, (
        f"{len(states['unsearchable'])} of {rep['labels']} labels read unsearchable "
        f"against {len(colls)} collections — past a quarter of the corpus this is a "
        "deployment fault being reported as a corpus finding, not #1599's five")


def test_the_guard_and_the_eval_agree_on_which_collections_are_deployed(tmp_path):
    """The seam between the two implementations of "is this gold returnable".

    The guard re-implements the deployed-collection read and the covering rule
    rather than importing them (the discipline at the top of this file), so the pair
    can drift: a guard that sees a collection the artifact does not green-lights a
    label the nightly then reports unreturnable, and the two instruments disagree
    about the corpus on the same night. This pins them to one answer on one fixture
    index — the only place the comparison does not also depend on the live
    deployment. The import is inside the test so this module's import-time posture
    (the guard reads the tree, never imports it) is unchanged.
    """
    import sys
    sys.path.insert(0, str(LLOYD_HOME))
    import eval.run_eval as ev

    # Roots that exist, because the pool rule walks them: a fixture pointing at
    # `/vault` would make both sides return an empty pool and every assertion about
    # a SERVED label would pass on nothing. This is the shape the live index has —
    # `subliminal` rooted at the whole vault, `knowledge` rooted inside it — which is
    # what makes a vault-relative label legal at all.
    vault = tmp_path / "vault"
    (vault / "knowledge").mkdir(parents=True)
    (vault / "knowledge" / "live-note.md").write_text("x\n")
    # A dot-prefixed directory, which qmd does not index (`parts.some(part =>
    # part.startsWith("."))` in qmd.ts) and which both sides must therefore drop
    # identically — the pool equality below is what pins that the two walks share the
    # rule, since a fixture without one would leave it untested.
    (vault / ".archived").mkdir()
    (vault / ".archived" / "old-note.md").write_text("x\n")
    db = tmp_path / "index.sqlite"
    import sqlite3
    con = sqlite3.connect(db)
    con.execute("create table store_collections (id INTEGER PRIMARY KEY, name TEXT, "
                "path TEXT, include TEXT, ignore TEXT)")
    con.executemany("insert into store_collections (name, path, include) values (?,?,?)",
                    [("knowledge", str(vault / "knowledge"), "*.md"),
                     ("subliminal", str(vault), "**/*.md")])
    con.commit()
    con.close()

    here = _deployed_collections(index_path=db)
    assert here == [{"name": "knowledge", "root": str(vault / "knowledge")},
                    {"name": "subliminal", "root": str(vault)}], here
    # The eval reaches the index through its owner, `app.doc_corpus` — never through
    # sqlite of its own, which is what
    # `test_no_script_under_eval_opens_the_knowledge_store_file` exists to keep true.
    # So the seam to pin is guard-vs-owner: if either side's read of
    # `store_collections` changes shape, the guard can green-light a label the
    # artifact then names, and both numbers are in the same morning note.
    from app import doc_corpus
    assert doc_corpus.deployed_collections(index_path=db) == here, (
        "the guard and the module the eval reads through disagree about which "
        "collections are deployed")
    # The pool rule is duplicated on purpose (the guard's discipline), so the two
    # pools must be the same set of strings and the two verdicts the same answer on
    # the same labels — a served one and the unserved one the whole item is about.
    assert _served_paths(here) == ev.indexed_paths(here), (
        "the guard and the eval disagree about which paths a deployed collection can "
        "return, so they can disagree about the corpus on the same night")
    for label in ("knowledge/live-note.md", "live-note.md",
                  "lloyd-architecture/harness.md"):
        assert _label_returnable(label, _served_paths(here)) \
            == ev._label_returnable(label, ev.indexed_paths(here)), label
    assert _label_returnable("knowledge/live-note.md", _served_paths(here)), here
    assert not _label_returnable("lloyd-architecture/harness.md", _served_paths(here))
    # Stated as a fact about the pools and not only as their equality, because
    # equality would also hold if BOTH sides dropped everything: `live-note.md` above
    # says the pools are not empty, this says the dot-rule is the reason one file is
    # missing from them.
    assert not any("archived" in p for p in _served_paths(here)), _served_paths(here)
    # And the owner answers None, not [], for an index that is not there: `[]` would
    # assert nothing is deployed and every label in the corpus would read unreturnable.
    assert doc_corpus.deployed_collections(index_path=tmp_path / "absent.sqlite") is None


#: The five gold doc labels #1599 found unreturnable, recorded as RETIRED. #1662
#: carried Alan's ruling on the item — "(c) Drop or relabel the five now: the labels
#: are the bug. File (a) as its own proposal, deployed only if it measurably helps
#: Lloyd's code-work recall" — and the drop is what landed, so the five are no longer
#: in `queries:`. This tuple is the difference between a decision and a disappearance:
#: in a diff an absent query and a retired one are the same five deleted blocks, and
#: only one of them can be audited afterwards. #1599 pinned these ids as EXPECTED TO
#: EXIST while the question was open; the question is closed, and what is pinned now is
#: the record. Option (a) is still a person's call — registering an
#: `lloyd-architecture` collection is what would make these five answerable, and a
#: corpus that re-adds them behind a deployed collection is that option being taken,
#: not this record being broken.
RETIRED_GOLD_LABELS_1662 = (
    ("harness-system-prompt-frozen", "lloyd-architecture/harness.md"),
    ("automod-gate-rungs", "lloyd-architecture/automod.md"),
    ("recall-doc-pool-ordering", "lloyd-architecture/retrieval.md"),
    ("qmd-embed-model-switch", "lloyd-architecture/qmd.md"),
    ("djev-rank-not-gate", "lloyd-architecture/djev.md"),
)

#: What the corpus file itself must carry for the ids above to read as retired. A
#: retirement that lives only in a test file is a claim; one written where the queries
#: used to be is a record a reader of the corpus finds.
#: Anchored on the block heading rather than the bare item number: the number appears
#: in the file's size note too, and a window opened there would read the wrong note.
RETIREMENT_NOTE_WHO = "checkout-doc queries: RETIRED by #1662"


def test_the_five_checkout_doc_queries_are_retired_by_record_and_still_a_tripwire(
        tmp_path):
    """#1662 clause 3: the presence tripwire becomes a retirement record.

    #1599 wrote this node to stop its own finding being made true by deleting what it
    was about, and pinned the five as expected to EXIST. Alan's ruling then chose
    exactly that option — the labels are the bug — so the pin had to be re-pointed
    rather than removed: what stays is (a) the five ids, visible as retired, with the
    authority that retired them named, and (b) the tripwire's actual purpose, which was
    never "these five exist" but "a checkout-prefixed gold label must not sit in the
    corpus unserved without anyone noticing". That half is exercised here by injection,
    because after the drop the corpus contains no checkout-prefixed label at all and the
    rule would otherwise be asserted over an empty set forever.

    The injection is a label that has never existed in the corpus, so the node can
    fail; the deletion it guards against is checked two ways, since the queries are
    gone and a missing entry is otherwise indistinguishable from the file having never
    had one.
    """
    raw = CORPUS.read_text()
    specs = yaml.safe_load(raw)["queries"]
    ids = {str(s.get("id")) for s in specs}
    labels = {str(d) for s in specs for d in (s.get("expect_docs") or [])}

    # (a) retired, not re-pointed: the ids are out of `queries:`, and so is every
    # checkout-prefixed label they carried. Re-pointing one onto some in-corpus
    # document would have kept the query and quietly changed what it tests.
    still_there = [qid for qid, _label in RETIRED_GOLD_LABELS_1662 if qid in ids]
    assert still_there == [], (
        f"{still_there} are back in {CORPUS} with no retirement note changed, so "
        "either option (a) was taken by someone who did not update this record or "
        "the drop is being undone quietly — both are a person's decision, and the "
        "note in the corpus is where it has to be written")
    checkout_labels = [d for d in labels if any(
        d.startswith(prefix) for prefix in CHECKOUT_LABEL_ROOTS)]
    assert checkout_labels == [], (
        f"{checkout_labels} resolve under a checkout prefix while no collection is "
        "named for it, which is the permanent doc miss #1599 filed and #1662 retired")

    # (b) the record: the file the queries were dropped from says so, names the
    # authority, and names all five ids. Deleting gold and deleting the note that
    # explained the deletion is the same edit to a reader six weeks on, so the note
    # is asserted with the same weight the presence of the queries used to be.
    assert RETIREMENT_NOTE_WHO in raw, (
        f"{CORPUS} carries no note naming {RETIREMENT_NOTE_WHO}: the five are absent "
        "and nothing in the corpus says why, which is a silent shrink by another name")
    note = raw[raw.index(RETIREMENT_NOTE_WHO):]
    unrecorded = [f"{qid} -> {label}" for qid, label in RETIRED_GOLD_LABELS_1662
                  if qid not in note or label not in note]
    assert unrecorded == [], (
        f"{unrecorded} are gone from the corpus and named nowhere near the retirement "
        f"note in {CORPUS}")
    # The five docs the record retires are #1354's five, and each is a REAL file in
    # this checkout — the retirement is therefore recorded against documents that
    # exist and could be answered again under option (a), not against labels that
    # never resolved. A fact about the tree, not a second literal: the same basename
    # list is checked for existence here and named by the note above.
    absent = [b for b in (label.rsplit("/", 1)[-1]
                          for _, label in RETIRED_GOLD_LABELS_1662)
              if not (LLOYD_HOME / "architecture" / b).is_file()]
    assert absent == [], (
        f"{absent}: the retirement names a document that is not in "
        f"{LLOYD_HOME / 'architecture'}, so the record points at nothing to restore")
    assert ({label.rsplit("/", 1)[-1] for _, label in RETIRED_GOLD_LABELS_1662}
            == set(CHECKOUT_DOCS_REQUIRED)), CHECKOUT_DOCS_REQUIRED

    # (c) the tripwire, still able to bite. One injected corpus, three labels: a
    # checkout doc that exists but no collection serves (the exact shape #1599
    # found), a checkout-prefixed name no root holds at all (so the node does not
    # depend on a file anyone might add), and a vault-served label that must stay
    # clean or the check would be reporting the whole corpus.
    served_doc = sorted(p.name for p in (LLOYD_HOME / "architecture").glob("*.md"))
    assert served_doc, f"{LLOYD_HOME / 'architecture'} has no docs to inject"
    corpus_x = tmp_path / "tripwire.yaml"
    corpus_x.write_text(yaml.safe_dump({"queries": [
        {"id": "re-added", "query": "q", "category": "technical",
         "expect_entities": [], "expect_docs": [f"lloyd-architecture/{served_doc[0]}"]},
        {"id": "never-written", "query": "q", "category": "technical",
         "expect_entities": [], "expect_docs": ["lloyd-architecture/no-such-doc.md"]},
        {"id": "served", "query": "q", "category": "technical",
         "expect_entities": [], "expect_docs": ["knowledge/"]},
    ]}, sort_keys=False))
    colls = _deployed_collections()
    assert colls, "no deployed collection list readable; the tripwire has no denominator"
    rep = _doc_label_satisfiability_report(specs_path=corpus_x, collections=colls)
    fired = {d["query"] for k in ("unsearchable", "dead") for d in rep[k]}
    assert fired == {"re-added", "never-written"}, rep
    assert [d["query"] for d in rep["satisfied_by"]] == ["served"], rep

    # And the fix is a deployment, not a re-label: register the collection the prefix
    # names and the existing doc stops being a finding, while the name no root holds
    # stays one. The two must move differently or the report is conflating a dead
    # label with an unserved one.
    fixed = _doc_label_satisfiability_report(
        specs_path=corpus_x,
        collections=colls + [{"name": "lloyd-architecture",
                              "root": str(LLOYD_HOME / "architecture")}])
    assert "re-added" not in {d["query"] for k in ("unsearchable", "dead")
                              for d in fixed[k]}, fixed
    assert "never-written" in {d["query"] for d in fixed["dead"]}, fixed


def test_no_gold_query_scores_a_doc_leg_no_deployed_collection_can_answer():
    """#1662 clause 1: the gold set carries no `expect_docs` label a deployed
    collection could never return — run through the instrument's own predicate, not
    the guard's private copy of it.

    The guard elsewhere in this file re-implements the collection read and the
    covering rule so it cannot be green by agreeing with the code it guards, and that
    split is pinned against itself by
    `test_the_guard_and_the_eval_agree_on_which_collections_are_deployed`. This node is
    deliberately the opposite: `eval/run_eval.py::gold_docs_unreturnable` is the
    function whose output the nightly artifact prints, so the acceptance is measured on
    it, against `app.doc_corpus.deployed_collections()` — the live list, the same one
    the artifact names as its denominator. The two readings agreeing on the live corpus
    is the whole point of having both.

    `gold_docs_unreturnable` reads gold off each record's `expected.docs`, the way
    `anchorless_queries` and `gold_bearing_rates` do, so the records built here
    normalise `expect_docs` into that shape rather than feeding the function a key it
    ignores.

    An unreadable index SKIPS this node and prints no verdict. It does not pass:
    `deployed_collections()` answers None precisely so that "the index would not
    open" cannot be filed as "every label is returnable", which is the same three-state
    discipline the artifact keeps for `gold_doc_unreturnable: null`.
    """
    import eval.run_eval as ev
    from app import doc_corpus

    colls = doc_corpus.deployed_collections()
    if colls is None:
        pytest.skip(
            "no deployed qmd collection list could be read: the returnability of a "
            "gold label is a fact about the deployment, and a corpus asserted clean "
            "against an unreadable index would be a clean bill printed by the thing "
            "that could not read it")
    assert colls, "an empty collection list is not a deployment to check against"

    specs = yaml.safe_load(CORPUS.read_text())["queries"]
    records = [{"id": s.get("id"), "expected": {"docs": list(s.get("expect_docs") or [])}}
               for s in specs]
    assert records, f"{CORPUS} carries no queries to check"
    bad = ev.gold_docs_unreturnable(records, colls)
    assert bad == [], (
        f"{len(bad)} gold labels no deployed collection can return: "
        + "; ".join(f"{b['query']} <- {b['label']}" for b in bad)
        + ". Each is a permanent doc miss: the query scores 0 on the document leg "
        "however good retrieval gets, so it drags doc_hit_rate, doc_recall_avg, "
        "mrr_doc and ndcg10 as a standing penalty and a trend can never clear it")

    # The check must be able to fail. `[]` is also what a rule that matches nothing,
    # or a pool walk that found no paths, prints — so the same call is given one
    # record it must name: a retired checkout label, which is the exact string #1662
    # removed and the exact string `lloyd-architecture/harness.md` that no vault-rooted
    # collection serves.
    pool = ev.indexed_paths(colls)
    assert pool, "the deployed collections yielded no paths, so nothing below is a test"
    named = ev.gold_docs_unreturnable(
        records + [{"id": "injected-checkout-label",
                    "expected": {"docs": ["lloyd-architecture/harness.md"]}}], colls)
    assert [n["query"] for n in named] == ["injected-checkout-label"], named

    # ...and one it must not: a label copied straight out of the live pool. Without
    # this the injection above could be satisfied by a predicate that calls everything
    # unreturnable, which is the failure #1599's first draft actually had (it called 12
    # queries unreturnable off a `Path.exists()` test over substring labels, on a corpus
    # of the 86 queries the file held then).
    good = sorted(pool)[0]
    assert ev.gold_docs_unreturnable(
        [{"id": "injected-served-label", "expected": {"docs": [good]}}], colls) == [], (
        f"{good} was taken from the pool the check itself was given, so the rule is "
        "not reading that pool")


def test_the_gold_file_states_the_size_it_actually_has(tmp_path):
    """#1662 clause 5: the header's stated query count equals the count the file
    carries.

    The note #1354 left said "81 -> 86 queries ... DO NOT COMPARE ABSOLUTE VALUES
    ACROSS THIS POINT" beside the five queries it had just added. Retiring them makes
    that sentence a lie in two directions at once — the size is 81 again and the
    discontinuity it warns about is now the wrong one — and a stale size line is worse
    than none, because the next reader trusts the number instead of counting. So the
    count is asserted against the file rather than against a constant: this node passes
    at 81 today and at 82 whenever a query is added, and fails on the day the line and
    the file disagree, which is the only thing that makes the line worth keeping.

    The #1748 label-policy prose is asserted in the same read, because the edit that
    rewrites a header is the edit that casually deletes the paragraph above it.
    """
    stated, actual = _gold_header_size()
    assert stated is not None, (
        f"{CORPUS} carries no '# GOLD SET SIZE:' line; the header's count is where a "
        "reader gets the denominator before they count, and #1662 is the reason it has "
        "to be re-measured rather than re-copied")
    assert stated == actual, (
        f"the gold header states {stated} queries and the file carries {actual}")

    raw = CORPUS.read_text()
    after = raw.split("# GOLD SET SIZE:", 1)[1].split("\nqueries:", 1)[0]
    assert "DO NOT COMPARE ABSOLUTE VALUES" in after, (
        "the size line carries no do-not-compare warning, so a reader comparing "
        "tonight's doc means with 2026-09-27's has nothing telling them the "
        "denominator moved")
    assert "1662" in after, (
        "the discontinuity is not attributed to the change that caused it")
    assert "LABEL POLICY (#1748)" in raw, (
        "the one-way ratchet on gold label shape was dropped with the size note; it is "
        "what stops an unreturnable label being widened until it passes")

    # Must be able to fail, three ways a stale line actually goes stale: the count
    # left behind by a query added elsewhere, the line deleted outright, and a
    # do-not-compare note dropped while the number survives. Each is asserted on a
    # copy, because the file itself is the thing under test.
    stale = raw.replace(f"# GOLD SET SIZE: {stated} queries",
                        f"# GOLD SET SIZE: {stated + 1} queries", 1)
    assert _gold_header_size(_write_text(tmp_path, "stale.yaml", stale)) \
        == (stated + 1, actual)
    gone = raw.replace(f"# GOLD SET SIZE: {stated} queries", "# GOLD SET SIZE: ?", 1)
    assert _gold_header_size(_write_text(tmp_path, "unreadable.yaml", gone))[0] is None
    # The warning assertion above can only fail if the region it reads is the size
    # block and not the whole header — an earlier note at the top of this file carries
    # the same all-caps phrase, and a slice that ran to the end of the file would find
    # that one instead. So the slice is calibrated: it starts at the size marker, ends
    # at the body, and is a small fraction of the file.
    assert after.startswith(f" {stated} queries"), after[:80]
    assert "\nqueries:" not in after and len(after) < len(raw) // 4, (
        f"the size block was read over {len(after)} of {len(raw)} chars, so the "
        "do-not-compare phrase it found may be some other note's")
    missing_line = "\n".join(ln for ln in raw.splitlines()
                             if not ln.startswith("# GOLD SET SIZE: "))
    assert _gold_header_size(_write_text(tmp_path, "nolines.yaml", missing_line)) \
        == (None, actual), "a file with no size line must read as None, not as 0"


def _write_text(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _gold_header_size(path: Path = CORPUS) -> tuple:
    """(the count the header states, the count of `- id:` entries), or (None, n)."""
    raw = path.read_text()
    body = raw.split("\nqueries:", 1)
    actual = sum(1 for ln in (body[1] if len(body) > 1 else raw).splitlines()
                 if ln.startswith("  - id:"))
    for ln in raw.splitlines():
        if ln.startswith("# GOLD SET SIZE: "):
            head = ln.split("# GOLD SET SIZE: ", 1)[1].split(" ")[0]
            return (int(head) if head.isdigit() else None), actual
    return None, actual

# ── #1748: the gold LABEL shape is a one-way ratchet, not a whole-file rule ───
#
# #1662's clause 4 asked that EVERY `expect_docs` entry in this corpus contain a
# `/` and a `.md` basename. That is false of the file and was false before #1662
# existed: 23 of its labels are substrings, and the file's own schema line says
# "expect_docs: substrings any of which should appear in returned document paths".
# Narrowing them onto single documents would move `doc_hit_rate`, `doc_recall_avg`,
# `mrr_doc` and `ndcg10` in one step — the meaning of the metric, not a defect
# fix — so no diff can satisfy that literal without doing #1662's unrelated work.
# What the clause was FOR is the ratchet below: a round that finds a query
# unreturnable must not be able to "fix" it by widening its gold label to a
# substring, so the substring-shaped set is pinned as of #1662 and may only
# SHRINK. The 23 stay legal; a new one, or an old one re-pointed onto a
# substring, is what reddens.

_GOLD_YAML = Path(__file__).resolve().parents[1] / "eval" / "vault_recall_queries.yaml"

#: Every substring-shaped `expect_docs` label on 2026-09-28, before any part of
#: #1662 landed (`git log efcee660 -- eval/vault_recall_queries.yaml` carries none
#: of that item's work, and the corpus one-liner prints the same 86 queries / 23
#: labels this frozenset was generated from). The set may shrink — narrowing one
#: of these onto a single document is fine, and the same commit that does it
#: deletes its entry here — and may never gain a member by any other edit.
_PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS = frozenset({
    "backlog-overview -> backlog/",
    "entity-resolution-sweep -> autonomy/48-entity-resolution",
    "entity-resolution-sweep -> skills/entity-resolution-sweep",
    "inner-voice -> inner_voice",
    "inner-voice -> lloyd/inner_voice",
    "kg-maintenance-tasks -> autonomy/24-",
    "kg-maintenance-tasks -> autonomy/48-",
    "kg-maintenance-tasks -> autonomy/51-",
    "kg-maintenance-tasks -> autonomy/67-",
    "kg-maintenance-tasks -> autonomy/74-",
    "nightly-reflection -> autonomy/39-",
    "nightly-reflection -> autonomy/42-",
    "nightly-reflection -> nightly-reflection",
    "qmd -> qmd",
    "qwen38-local-serving -> hybrid-gdn-qsa-inference-serving",
    "qwen38-local-serving -> qwen38-27b-local-24gb",
    "qwen38-local-serving -> qwen38-flash-next-engram",
    "robotics-projects -> GR00T",
    "robotics-projects -> robot",
    "robotics-projects -> robotics",
    "tgs-rag-state -> 363-implement-tgs-rag",
    "tgs-rag-state -> neuro-symbolic-rag",
    "tgs-rag-state -> tgs-rag",
})


def _gold_label_pairs(path: Path = _GOLD_YAML) -> set:
    """Every `query id -> expect_docs label` pair the corpus asserts."""
    import yaml

    queries = yaml.safe_load(path.read_text())["queries"]
    return {f"{q['id']} -> {d}" for q in queries for d in (q.get("expect_docs") or [])}


def _substring_shaped_gold_labels(pairs: set) -> set:
    """The pairs whose label is a substring rather than a named document.

    The predicate is #1662's clause-4 literal inverted: a label names one document
    iff it carries a `/` and its basename ends in `.md`. Kept in this file rather
    than imported so the ratchet cannot be disarmed by editing the scorer.
    """
    return {p for p in pairs
            if "/" not in p.split(" -> ", 1)[1]
            or not p.split(" -> ", 1)[1].rsplit("/", 1)[-1].endswith(".md")}


def _gold_label_ratchet_violations(pairs: set) -> tuple:
    """`(grown, narrowed)` for one corpus's label pairs — the ratchet's two halves.

    Pure over a pair set so the failure modes can be exercised on a synthetic
    corpus without touching the real one, which is what makes clause 3 checkable:
    an instrument whose only passing state is the current file has not been shown
    to be able to fail.
    """
    shaped = _substring_shaped_gold_labels(pairs)
    grown = sorted(shaped - _PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS)
    narrowed = sorted(_PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS - shaped)
    return grown, narrowed


def test_no_gold_label_was_widened_into_a_substring_to_pass_the_returnability_guard():
    """#1748 clauses 2 and 3: the substring-shaped label set is one-way.

    Two directions, both must be able to fail, because the failure this ratchet
    exists to prevent is a round that meets an unreturnable gold label and
    *widens* it — which turns a doc miss into a doc hit while looking like a
    label-quality improvement:

    * **Grow:** any substring-shaped label that is not one of the pinned 23 — a
      new query given `qmd`-style gold, or a pinned label re-pointed onto a
      substring — reddens the first assert.
    * **Narrow without editing the pin:** a pinned label replaced by a named
      document (legitimate work) reddens the second assert until the same commit
      deletes its entry from the frozenset, which is what keeps the pinned set
      from silently rotting into a list of labels the corpus no longer carries.
    """
    grown, narrowed = _gold_label_ratchet_violations(_gold_label_pairs())
    assert not grown, (
        f"{len(grown)} new substring-shaped expect_docs label(s): {grown[:5]} — a "
        "label added after #1662 must name one document (`<dir>/<name>.md`); "
        "widening gold is how an unreturnable query starts passing as retrieved")

    assert not narrowed, (
        f"{len(narrowed)} pinned pre-#1662 label(s) are no longer in the corpus: "
        f"{narrowed[:5]} — narrowing a legacy substring label onto a document is "
        "allowed, but delete its entry from "
        "_PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS in the same commit, and expect "
        "doc_hit_rate/doc_recall_avg/mrr_doc/ndcg10 to move at that point")


def _gold_header(text: str) -> str:
    """The leading comment block of a gold-file text, up to its first data line."""
    lines = text.split("\n")
    end = next((i for i, ln in enumerate(lines)
                if ln.strip() and not ln.lstrip().startswith("#")), len(lines))
    return "\n".join(lines[:end])


def _expect_docs_block(header: str) -> str:
    """The `expect_docs:` schema entry and its continuation lines, nothing else.

    Scoped to this one entry on purpose: a guard that greps the whole comment
    region is satisfied by any sentence that happens to contain the words, which is
    the weakness #1748's own review left as an advisory on the first version of
    this pin.
    """
    lines = header.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if ln.lstrip().lstrip("#").strip().startswith("expect_docs:")), None)
    if start is None:
        return ""
    out = [lines[start]]
    for ln in lines[start + 1:]:
        body = ln.lstrip()
        if not body.startswith("#"):
            break
        # Indent is counted AFTER the leading `#`, since every comment line has zero
        # spaces before it. Schema keys sit at `#   key:`; continuations are deeper.
        after = body[1:]
        indent = len(after) - len(after.lstrip())
        if indent <= 3 and ":" in after.strip().split(" ")[0]:
            break
        out.append(ln)
    return "\n".join(out)


def _gold_header_problems(text: str) -> list:
    """Every way a gold-file header fails to carry the #1748 label policy.

    Pure over the file's text, for the same reason `_gold_label_ratchet_violations`
    is: an instrument whose only passing state is today's file has not been shown
    able to fail, and the specific state this has to reject is one where the policy
    EXISTS but the schema line above it still reads as a free pass — which is
    exactly what `eval/vault_recall_queries.yaml` looked like after the first #1748
    landing, and the reason the clause was re-offered.

    Three checks, all of them about a reader who stops at the `expect_docs:` line:
    the match rule there has to disclaim itself, the label rule has to name one
    document and the shrinking pinned set, and the count the prose quotes has to be
    the count the guard pins AND the count the corpus actually carries.
    """
    import re

    problems: list[str] = []
    block = _expect_docs_block(_gold_header(text))
    if not block:
        return ["the header has no `expect_docs:` schema entry"]
    # Flatten comment-by-comment: joining raw lines leaves each line's `#` marker in
    # the prose, and a sentence wrapped across two lines stops matching anything
    # (measured — "must name one # document").
    flat = " ".join(ln.lstrip().lstrip("#").strip()
                    for ln in block.split("\n")).lower()

    # 1. The licence itself. Split at the policy marker: what is ABOVE it is the
    #    only prose a reader of the schema line alone ever sees.
    above, sep, below = flat.partition("label policy")
    if not sep:
        problems.append("the expect_docs entry states no LABEL POLICY")
        below = ""
    if "not a licence" not in above and "not a license" not in above:
        problems.append(
            "the match rule at the expect_docs: line still licenses a substring — "
            "it must say in its own sentence that it scores a label rather than "
            "permitting one")

    # 2. What the policy has to promise, in the policy's own prose.
    for needle, what in (
            ("name one document", "a new label must name one document"),
            ("legacy", "the 23 are legacy exceptions, not a precedent"),
            ("pinned", "the legacy set is a pinned set"),
            ("shrink", "the pinned set may shrink"),
            ("never grow", "the pinned set may never grow")):
        if needle not in below:
            problems.append(f"the label policy does not state: {what}")

    # 3. The quoted count against the two instruments that carry it. A header
    #    saying 23 over a guard pinning 24 is a second instrument disagreeing with
    #    the first, and the old pin's `"23" in header` — a substring match over the
    #    whole comment region — could not tell those apart.
    counts = {int(n) for n in re.findall(
        r"(?<![\w.#-])(\d+)\s+(?:labels?\b|substrings?\b)", below)}
    pinned = len(_PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS)
    in_corpus = len(_substring_shaped_gold_labels(_gold_label_pairs()))
    if counts != {pinned}:
        problems.append(
            f"the policy quotes {sorted(counts) or 'no'} legacy label count(s); "
            f"the guard pins {pinned}")
    if pinned != in_corpus:
        problems.append(
            f"the guard pins {pinned} legacy labels but the corpus carries "
            f"{in_corpus} — the ratchet and its fixture disagree")
    return problems


_HEADER_BEFORE_1748 = """\
#   expect_docs: substrings any of which should appear in returned document paths
#     (relative to ~/obsidian/) — at least one match counts as a hit
"""

_HEADER_POLICY_BENEATH_A_BARE_LICENCE = """\
#   expect_docs: substrings any of which should appear in returned document paths
#     (relative to ~/obsidian/) — at least one match counts as a hit
#     LABEL POLICY (#1748), which that match rule does not state: a NEW or
#     RE-POINTED expect_docs label must name one document — a path with a `/` and
#     a `.md` basename, e.g. `knowledge/foo.md`. The 23 labels that predate #1662
#     and are bare substrings (qmd -> qmd, robotics-projects -> robot, …) are
#     legacy: they stay legal only as the pinned set in
#     tests/test_eval_corpus_guard.py::_PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS,
#     which that guard holds may SHRINK and may NEVER GROW.
"""


def test_the_gold_header_states_the_ratchet_instead_of_licensing_any_substring():
    """#1748 clause 4: the file's own schema line must not read as a free pass.

    `expect_docs: substrings any of which should appear in returned document
    paths` is still true of retrieval semantics and stays, but it is the MATCH
    rule, and until now it stood at `:12` as the whole of what a reader of that
    line sees while the label rule sat underneath it. The next round adds a
    substring for that reason, and only the ratchet catches it — after that round
    has committed and re-scored the trend. So the disclaimer lives in the match
    rule's own sentence now, and this node reads the header the way that reader
    does: the `expect_docs:` entry and nothing else.
    """
    problems = _gold_header_problems(_GOLD_YAML.read_text())
    assert problems == [], problems


def test_the_gold_header_guard_reddens_on_a_header_that_only_borrows_the_policy():
    """Clause 4's fail-both-ways half: the guard must reject the two headers that
    shipped before this, not merely pass the one that ships now.

    `_HEADER_BEFORE_1748` is the entry as it stood before #1748 touched it, and
    `_HEADER_POLICY_BENEATH_A_BARE_LICENCE` is verbatim the entry this round
    replaced — policy present, count correct, and the schema line above it still an
    unqualified licence. Both must produce the licence problem: that is the state
    the clause was re-offered over, and a node that only asserted the current
    file's prose could not have distinguished it from a fix.
    """
    bare = _gold_header_problems(_HEADER_BEFORE_1748)
    assert any("label policy" in p for p in bare), bare
    assert any("licenses a substring" in p for p in bare), bare

    borrowed = _gold_header_problems(_HEADER_POLICY_BENEATH_A_BARE_LICENCE)
    assert borrowed == [
        "the match rule at the expect_docs: line still licenses a substring — "
        "it must say in its own sentence that it scores a label rather than "
        "permitting one"], borrowed


def test_the_gold_header_guard_reddens_when_the_quoted_count_drifts():
    """Clause 4's third check, made falsifiable: prose that quotes a count the
    guard no longer pins is a problem, and so is a guard pinning a count the corpus
    does not carry.

    The count is rewritten in the fixture header rather than the frozenset, so the
    disagreement is created in exactly one of the two instruments the assert names.
    """
    drifted = _HEADER_POLICY_BENEATH_A_BARE_LICENCE.replace(
        "The 23 labels", "The 24 labels")
    problems = _gold_header_problems(drifted)
    assert any("quotes [24]" in p and "pins 23" in p for p in problems), problems

    empty = _gold_header_problems(_HEADER_POLICY_BENEATH_A_BARE_LICENCE.replace(
        "The 23 labels", "These legacy labels"))
    assert any("no legacy label count" in p for p in empty), empty


def test_the_gold_label_ratchet_reddens_on_a_new_substring_label():
    """#1748 clause 3, grow direction: an injected substring-shaped label is `grown`.

    `brand-new-query -> some-slug` is exactly what a round under pressure writes
    when a gold path turns out not to be returnable, and it is legal under the
    file's match rule — which is the whole reason the ratchet has to be the guard.
    """
    grown, narrowed = _gold_label_ratchet_violations(
        _gold_label_pairs() | {"brand-new-query -> some-slug"})
    assert grown == ["brand-new-query -> some-slug"], grown
    assert narrowed == [], narrowed


def test_the_gold_label_ratchet_reddens_when_a_pinned_label_narrows_unpinned():
    """#1748 clause 3, narrow direction: a pinned label pointed at a document, with
    the pinned frozenset left untouched, is `narrowed` — and stays legal once the
    pin is edited in the same change.
    """
    pairs = _gold_label_pairs()
    pinned = sorted(_PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS)[0]
    qid, _ = pinned.split(" -> ", 1)
    narrowed_away = (pairs - {pinned}) | {f"{qid} -> knowledge/some-doc.md"}
    grown, narrowed = _gold_label_ratchet_violations(narrowed_away)
    assert grown == [], grown
    assert narrowed == [pinned], narrowed

    # The same corpus is legal once the pin is edited in the same change, which is
    # the sanctioned route: narrow the label, drop its entry from the frozenset.
    rest = _PRE_1662_SUBSTRING_SHAPED_GOLD_LABELS - {pinned}
    shaped_after = _substring_shaped_gold_labels(narrowed_away)
    assert sorted(shaped_after - rest) == [] and sorted(rest - shaped_after) == [], (
        sorted(shaped_after ^ rest))


# ---------------------------------------------------------------------------
# #3076: a label's collection has to be one the RECALL ASKS FOR, not merely one the
# index serves.
#
# The side-index A/B published at `eval/measurements/lloyd-architecture-ab-2026-10-20.md`
# built an arm whose qmd config declared `lloyd-architecture` rooted at
# `~/lloyd/architecture` (36 docs, 997 chunk vectors, `qmd --index arm ls
# lloyd-architecture` lists all 36, and a query restricted to that collection returns
# them). Scored through the real document leg, ten probe queries over those same 36
# documents came back 0.0 on every metric — on the arm exactly as on the control. The
# collection was deployed, the documents were embedded and retrievable, and the recall
# still never returned one, because `agent_mcp/vault.py:1007` sends
# `list(VAULT_SEGMENTS) + extra` as the payload's `collections:` and `VAULT_SEGMENTS`
# (:86-89) is the eleven vault segments. `app/doc_corpus.py::deployed_collections` reads
# the index's own `store_collections` table, which is a WIDER set than the one the
# retriever names: on the live index of 2026-10-20 the index serves 14 and the recall
# names 11, and `autonomy-runs`, `sessions` and `subliminal` are the three the daemon
# holds without the unrestricted search ever asking for them.
#
# So every returnability check that reads `store_collections` — including
# `eval/run_eval.py::gold_docs_unreturnable`, whose output the artifact prints as
# `gold_doc_unreturnable` — calls a label returnable when the retriever cannot request
# it. A gold label rooted at `subliminal/` would be counted available, score 0 forever,
# and read as a retrieval miss. That is #1599's permanent-miss shape arriving through a
# door the #1662 guard cannot see, and it is why the trial's arm could measure zero on
# its gain leg and look like a verdict.
RECALL_SEGMENT_SOURCE = ROOT / "agent_mcp" / "vault.py"

#: Collections the qmd index serves that the recall's unrestricted document leg never
#: names in its `collections:` list. Each is indexed on purpose and reaches an answer
#: another way: `sessions` is chat transcripts, which the recall asks for only as the
#: episodic floor (`agent_mcp/vault.py:320` `RECALL_EPISODIC_FLOOR`, gated by
#: `retrieval.recall.episodic_floors`), a route that pins one row per floor and is not a
#: lane a document label can be scored on; `subliminal` re-indexes the vault for the
#: background pass; `autonomy-runs` is run output. A gold label rooted at one of these is
#: a permanent doc miss and must be named here or the corpus gains one quietly. When the
#: owed ruling on #3076 deploys `lloyd-architecture`, its name has to join
#: `VAULT_SEGMENTS` in the same change — deploying it in qmd's config alone puts it in
#: this set, which is what
#: `test_a_collection_deployed_but_not_named_leaves_its_label_unreachable` pins.
COLLECTIONS_INDEXED_BUT_NOT_RECALLED = {"autonomy-runs", "sessions", "subliminal"}


def _recall_named_collections() -> set:
    """The collection names `vault_recall`/`vault_search` put in the qmd payload.

    Read off `agent_mcp/vault.py`'s source by AST, not imported: this file's standing
    discipline (`:1567-1573`) is that a guard may not be green by agreeing with the code
    it guards, and importing `VAULT_SEGMENTS` would let a rename in that module silently
    rename this guard's expectation with it. Parsing keeps the literal in one place while
    still failing if the assignment is deleted or stops being a list of strings.
    """
    import ast

    tree = ast.parse(RECALL_SEGMENT_SOURCE.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "VAULT_SEGMENTS"):
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    raise AssertionError(
        f"no `VAULT_SEGMENTS = [...]` literal in {RECALL_SEGMENT_SOURCE}: the guard has "
        "no denominator, and a corpus silently unchecked against a renamed list is the "
        "failure this node exists to catch")


def _label_root(label: str):
    """The collection a label is rooted at, or None for a bare-basename label — the same
    split the scorer's substring test (`run_eval.py:478-486`) makes against the
    `<collection>/<rel>` paths qmd returns."""
    return label.split("/", 1)[0] if "/" in label else None


def _labels_the_recall_cannot_request(pairs: set, named: set) -> list:
    """`query id -> label` pairs whose root is a collection the recall does not name.

    The one predicate both the corpus check and its mutation test call, so neither can
    drift from the other: a guard that re-implements its own negative in the test is a
    guard that cannot fail.
    """
    return sorted((qid, lab) for qid, lab in (p.split(" -> ", 1) for p in pairs)
                  if _label_root(lab) is not None and _label_root(lab) not in named)


def test_no_gold_label_names_a_collection_the_recall_never_requests():
    """#3076: every rooted gold label lives under a collection `vault_recall` names, and
    every name the recall asks for is one the index serves.

    The property fails in both directions and each is a different bug. A rooted label
    outside the recall's list is a permanent doc miss no amount of retrieval quality can
    clear — the arm of the trial above, seen from the corpus side. A name in the recall's
    list that the index does not serve is the mirror: the payload asks for a collection
    that holds nothing, and every real collection shares its candidate quota.
    """
    named = _recall_named_collections()
    assert len(named) == 11, (
        f"the recall names {len(named)} collections ({sorted(named)}); this node's claim "
        "about the eleven vault segments and its read of `VAULT_SEGMENTS` have to be "
        "re-read together, not re-literalled")

    pairs = _gold_label_pairs()
    rooted = [lab for _, lab in (p.split(" -> ", 1) for p in pairs)
              if _label_root(lab) is not None]
    assert rooted, f"{CORPUS} carries no rooted doc label — check the corpus loaded"
    unrecalled = _labels_the_recall_cannot_request(pairs, named)
    assert unrecalled == [], (
        f"{unrecalled}: the label is rooted at a collection the recall never asks the "
        "daemon for, so the query scores 0 on the document leg forever. Either the label "
        "belongs under a named segment, or the collection joins VAULT_SEGMENTS in the "
        "same change — deploying it in qmd's config alone does not make it retrievable "
        "(#3076, eval/measurements/lloyd-architecture-ab-2026-10-20.md)")

    served = {c["name"] for c in _deployed_collections()}
    assert served, "the index reports no collections at all; the subset check is vacuous"
    absent = sorted(named - served)
    assert absent == [], (
        f"{absent}: the recall asks for a collection the qmd index does not serve, so the "
        "ask is dead weight in every unrestricted search's collection list")
    gap = served - named
    assert gap == COLLECTIONS_INDEXED_BUT_NOT_RECALLED, (
        f"the index serves {sorted(gap)} that the recall does not name, and the "
        "documented set says otherwise. A new one is a scoped-only collection no gold "
        "label may be rooted at; a removed one means the recall grew a segment, which "
        "changes the denominator of every doc-leg number in the trend")


def test_a_collection_deployed_but_not_named_leaves_its_label_unreachable():
    """The check has to be able to fail, and what fixes it is the recall's list.

    One injected corpus, two labels under the same collection name: the name deployed but
    not named by the recall (the trial's arm, exactly) and the name both deployed and
    named (what an option-(a) deploy has to look like to do any good). A rule that
    reported both rows, or neither, would be describing the deployment rather than
    retrieval, and the trial would still be reading its own zero as a verdict.
    """
    named = set(_recall_named_collections())
    served = {c["name"] for c in _deployed_collections()}
    assert served, "no deployed collection list to inject against"
    assert "lloyd-architecture" not in served, (
        "the trial's premise has changed: a lloyd-architecture collection is deployed. "
        "The owed ruling on #3076 decided that, and this node's control row has to be "
        "re-read against the deployed set instead of asserted absent")

    injected = {"qmd -> lloyd-architecture/qmd.md", "kg -> knowledge/ai/x.md",
                "bare -> workers.md"}
    # The un-named collection's label is flagged; a named segment's is not; a bare
    # basename is not (the scorer's substring test can still reach it, which is the
    # separate #1662 substring-shape problem, not this one).
    assert _labels_the_recall_cannot_request(injected, named) == [
        ("qmd", "lloyd-architecture/qmd.md")], named
    # Name it, and the row clears with nothing else about the corpus changing. That is
    # the whole difference between the trial's zero and its ten hits.
    assert _labels_the_recall_cannot_request(injected, named | {"lloyd-architecture"}) == []
    # And it is still not reachable if the index does not serve it: naming it in
    # `VAULT_SEGMENTS` over an index without the collection is the `control_seg` row of
    # that trial — 0.0 on every metric. Both halves, or the label is a miss.


# ---------------------------------------------------------------------------
# The #3076 trial's own artifacts, pinned rather than narrated.
#
# `eval/measurements/lloyd-architecture-ab-2026-10-20.md` is a measurement, and a
# measurement quoted in prose is a claim. This node reads the twelve artifacts and the
# arm's scratch qmd config off disk and asserts the three things the note's acceptance
# depends on: the arm declared the collection rooted where the note says, the two side
# indices differ by exactly that one collection, and the note prints the 36-file listing
# rather than summarising it. Like every guard here it reads derived data, so
# `require_live_data` skips by name on a checkout that never regenerated them.
#: The trial's scratch lives in the PRODUCTION data root, not under `$HOME`: the gate
#: runs this suite with `HOME` at the round's farm, where `<round>/home/lloyd-data` is
#: its own directory holding 16 entries and NO `eval/3076`, so every node below skipped
#: instead of grading (`43 tests skipped (limit 40)`). `production_data_root()`
#: (imported at :43) is `ACCOUNT_HOME / "lloyd-data"` and names the real one, which is
#: also what the trial runner itself wrote through.
#: ...and the vault the note and the probe live in, and the qmd config the no-touch
#: proof reads, are anchored the same way for the same reason: the gate does NOT run
#: this file in the real home. `scripts/automod/gate.py:1595` sets `"HOME": str(home)`
#: with `home` = `<round>/home`, the farm `scripts/automod/worktree.py:78
#: ensure_round_home` builds. Measured in this round's farm: `obsidian` and `.config`
#: are SYMLINKS to the live ones, so a path built from `$HOME` names the same bytes but
#: a DIFFERENT STRING — which is how SM_20261020_175859 lost two nodes, comparing a
#: path the note printed verbatim: `'/home/alansrobotlab/.config/qmd/index.yml:14' !=
#: '/home/alansrobotlab/lloyd-work/SM_20261020_175859/home/.config/qmd/index.yml:14'`.
#: `lloyd-data` is not a symlink: it is its own directory, holding 16 entries and no
#: `eval/3076`, so the same reads there are absent outright — the `43 tests skipped
#: (limit 40)` that cost the round before that one its gate. Either way the read is
#: wrong: clause 5's claim is about `~/.config/qmd/index.yml`, and certifying a symlink
#: by its round-home name proves nothing about the file the daemon actually opens.
#: `ACCOUNT_HOME` is passwd-derived and ignores `$HOME` (`app/data_root.py:77`), which
#: is why `production_data_root()` was stable and these lines were not; it is also what
#: the corpus's own header rule says `~/` means —
#: `tests/test_architecture_coverage_doc_claims.py:257`.
TRIAL_VAULT = ACCOUNT_HOME / "obsidian"
TRIAL_DIR = production_data_root() / "eval" / "3076"
TRIAL_ARTIFACTS = TRIAL_DIR / "data" / "eval" / "baselines"
TRIAL_NOTE = TRIAL_VAULT / "eval" / "measurements" / "lloyd-architecture-ab-2026-10-20.md"
TRIAL_PROBE = TRIAL_NOTE.with_name("lloyd-architecture-ab-2026-10-20-probe.yaml")


def test_the_3076_side_index_ab_artifacts_record_what_the_note_claims():
    """One snapshot, two side indices, 14 vs 15 — and a 36-line listing, not a summary."""
    import yaml

    require_live_data(TRIAL_ARTIFACTS, "the #3076 side-index A/B artifact directory")
    require_live_data(TRIAL_NOTE, "the #3076 measurement note", kind="file")
    require_live_data(TRIAL_PROBE, "the #3076 probe set", kind="file")

    # The two scratch configs beside the two indices. The fork reads
    # `$QMD_CONFIG_DIR/<index-name>.yml`, so each side has its own file, and each was
    # written from the live config verbatim — the arm's with exactly one entry added.
    conf_dir = TRIAL_DIR / "idx" / "conf"
    live = yaml.safe_load(TRIAL_LIVE_YML.read_text())
    ctrl_conf = yaml.safe_load((conf_dir / "control.yml").read_text())
    arm_conf = yaml.safe_load((conf_dir / "arm.yml").read_text())
    assert len(ctrl_conf["collections"]) == len(live["collections"]), (
        "the control's scratch config is not the live config's collection set, so the "
        "two arms did not start from one snapshot")
    assert len(arm_conf["collections"]) == len(ctrl_conf["collections"]) + 1, (
        f"the arm config differs from the control's by "
        f"{len(arm_conf['collections']) - len(ctrl_conf['collections'])} collection(s), "
        "not one: the A/B would be measuring something other than adding this one")
    arm = arm_conf["collections"]["lloyd-architecture"]
    assert arm["path"] == "/home/alansrobotlab/lloyd/architecture", arm
    assert arm["pattern"] == "**/*.md", arm
    assert set(arm_conf["collections"]) - set(ctrl_conf["collections"]) == {
        "lloyd-architecture"}, sorted(set(arm_conf["collections"]) ^ set(
            ctrl_conf["collections"]))
    assert "lloyd-architecture" not in live["collections"], (
        "the live qmd config now declares the collection, so the trial's whole premise "
        "— an A/B with no live-config edit — has ended: this node has to be re-read "
        "against a deployed collection, not re-asserted")

    def head(label_prefix):
        import glob
        files = sorted((Path(f) for f in glob.glob(
            str(TRIAL_ARTIFACTS / f"{label_prefix}-*.json"))), key=lambda f: f.stat().st_mtime)
        assert files, f"no scored artifact matching {label_prefix}"
        art = json.loads(files[-1].read_text())
        return art, files[-1]

    control, cfile = head("x3076-81-control")
    arm_art, afile = head("x3076-81-arm")
    assert control["summary"]["overall"]["n_queries"] == 81, cfile.name
    assert arm_art["summary"]["overall"]["n_queries"] == 81, afile.name
    # The 14-vs-15 count is the only evidence that the arm is the arm: without
    # LLOYD_QMD_INDEX beside LLOYD_CONFIG_OVERLAY both report None (#3076's trap).
    assert control["summary"]["overall"]["gold_doc_collections"] == 14, cfile.name
    assert arm_art["summary"]["overall"]["gold_doc_collections"] == 15, afile.name
    assert cfile.name != afile.name
    # Each side scored its OWN index, and only the arm's holds the new collection.
    import sqlite3
    for art, want in ((control, False), (arm_art, True)):
        path = (art["corpus"]["doc"]["index_path"])
        assert path.startswith(str(TRIAL_DIR / "idx")), path
        cols = [r[0] for r in sqlite3.connect(f"file:{path}?mode=ro", uri=True).execute(
            "SELECT DISTINCT collection FROM documents")]
        assert ("lloyd-architecture" in cols) is want, (path, sorted(cols))

    # The note: the same two counts, the cost leg's tie stated, and the listing printed
    # at full length — an elided "36 lines total" is a claim, 36 `qmd://` URIs are the
    # command's output.
    text = TRIAL_NOTE.read_text()
    assert "control 14, arm 15" in text
    assert text.count("qmd://lloyd-architecture/") >= 36, (
        "the note summarises the `ls` listing instead of printing it; clause 1 asks for "
        "the listing that shows 36 docs")
    probe = yaml.safe_load(TRIAL_PROBE.read_text())
    assert 8 <= len(probe["queries"]) <= 12, len(probe["queries"])
    anchored = [d for q in probe["queries"] for d in q["expect_docs"]]
    assert all(d.startswith("lloyd-architecture/") and d.endswith(".md") for d in anchored), anchored
    assert len(anchored) == len(probe["queries"]) == 10, anchored


# --------------------------------------------------------------------------- #
# 2026-10-20 round SM_20261020_175859 — #3076's five acceptance clauses, each
# pinned by a node that re-derives its numbers from the trial's own artifacts
# rather than trusting the note that prints them.
#
# WHY this section exists. The 2026-10-20T08:39 issue of
# `eval/measurements/lloyd-architecture-ab-2026-10-20.md` was landed with every
# artifact in place and still reported clauses 1-5 unmet, because the prose had
# drifted from the bytes beside it:
#   * the `qmd --index arm ls lloyd-architecture` listing was rekeyed with every
#     path printed twice — 72 `qmd://lloyd-architecture/` URIs for the 36 the
#     command emitted — and the node above accepted ANY count >= 36, so it passed
#     straight through a listing that showed each document twice;
#   * the two metric tables were a doc-gold-population (n=74) mean printed under
#     "as recorded in the artifact JSONs", and the delta columns were the paired
#     estimator over all 81, so one row read `mrr_doc 0.317 -> 0.323` beside a
#     delta of +0.0151 whose own endpoints are 0.2867 and 0.3018;
#   * the no-touch section printed output its commands cannot produce:
#     `grep -c -- '- name: '` returns **0** for both qmd configs, not 14 (this
#     fork's yml is a mapping, not a `- name:` list), and the quoted gold-set
#     header line does not exist in `eval/vault_recall_queries.yaml`.
# So each node below compares the note against an artifact, and where the clause
# is about a command's output it re-runs the command. They also carry the
# measurement's own load-bearing claims — that the arm's 36 docs are in the arm
# index and not the control's, that the probe's labels are root-anchored because
# doc matching is a substring test, and that a deployed-but-unsegmented
# collection is unreachable through `vault_recall` (#3075 option (a) as written).
# --------------------------------------------------------------------------- #

TRIAL_INGEST = TRIAL_DIR / "ingest.json"
TRIAL_SCORE = TRIAL_DIR / "score.json"
TRIAL_REACH = TRIAL_DIR / "reach.json"
TRIAL_RUNNER = TRIAL_DIR / "run_trial.py"
# The named-collection upper bound (#3086). Its `summary` is a broken tally — see the
# block above the two nodes at the end of this file — so the ONLY usable evidence in it
# is the per-query `top5`, which is what those nodes re-rank. Deliberately NOT added to
# `_3076_require_trial()`: the nodes in that block never read this file, and a guard's
# skip set has to name only what it actually opens, or an absent artifact silences
# guards that were perfectly capable of grading themselves.
TRIAL_UPPER = TRIAL_DIR / "upper.json"
# The two qmd collection configs: the one the daemon reads, and the tracked copy
# the nightly drift check compares it against. Neither may name lloyd-architecture
# while this trial's verdict is "nothing was deployed".
TRIAL_LIVE_YML = ACCOUNT_HOME / ".config" / "qmd" / "index.yml"
# The checkout the trial ran in. NOT `ROOT`, which at the gate is the round's worktree
# whose paths the note never printed, and NOT `Path.home()/"lloyd"`, which at the gate
# is `<round>/home/lloyd` — the worktree again, reached through the mirrored home.
# `app.paths.LIVE_CHECKOUT` is `ACCOUNT_HOME / "lloyd"`, the one production answer.
PRODUCTION_REPO = LIVE_CHECKOUT
TRIAL_TEMPLATE_YML = PRODUCTION_REPO / "agent-services" / "conf" / "qmd-index.yml"
DOC_METRICS = ("doc_hit_rate", "doc_recall_avg", "mrr_doc", "ndcg10")
# The arm's whole content: the checkout's architecture docs, declared in a scratch
# config. #3076 names this root and pattern verbatim, so it is not a constant to
# widen — a retry that roots the collection somewhere else is a different trial.
ARM_COLLECTION_ROOT = "/home/alansrobotlab/lloyd/architecture"
ARM_COLLECTION_PATTERN = "**/*.md"
# Every scored run the note reports, keyed by the trial's label prefix. `_arm`
# names the arm so a mislabelled artifact cannot silently swap the two sides: the
# control is the one whose index lacks the collection.
TRIAL_RUN_LABELS = {
    "x3076-81-control": "control", "x3076-81-arm": "arm",
    "x3076r-81-control": "control", "x3076r-81-arm_plain": "arm",
    "x3076r-81-arm_seg": "arm", "x3076r-81-control_seg": "control",
    "x3076-probe-control": "control", "x3076-probe-arm": "arm",
    "x3076r-probe-control": "control", "x3076r-probe-arm_plain": "arm",
    "x3076r-probe-arm_seg": "arm", "x3076r-probe-control_seg": "control",
}
# Which daemon each side was allowed to reach. Production stays at 8181; an arm
# that scored through production's daemon would be measuring the live config,
# which is the one thing this trial promises not to do.
ARM_PORTS = {"control": 8193, "arm": 8194}


def _3076_require_trial() -> None:
    """Skip by name on a machine that holds none of this trial's derived data, and
    say WHICH piece is missing: `TRIAL_NOTE` absent and `TRIAL_ARTIFACTS` absent are
    different statements, and the previous node already skips on the first of them."""
    require_live_data(TRIAL_ARTIFACTS, "the #3076 side-index A/B artifact directory")
    require_live_data(TRIAL_NOTE, "the #3076 measurement note", kind="file")
    require_live_data(TRIAL_PROBE, "the #3076 probe set", kind="file")
    require_live_data(TRIAL_INGEST, "the #3076 ingest record", kind="file")
    require_live_data(TRIAL_SCORE, "the #3076 first-pass score record", kind="file")
    require_live_data(TRIAL_REACH, "the #3076 head-to-head score record", kind="file")
    require_live_data(TRIAL_RUNNER, "the #3076 trial runner", kind="file")


def _3076_artifact(label_prefix: str):
    """The newest scored artifact JSON for a run label, with its path."""
    files = sorted(TRIAL_ARTIFACTS.glob(f"{label_prefix}-*.json"))
    assert files, f"no scored artifact matching {label_prefix}"
    return json.loads(files[-1].read_text()), files[-1]


def _3076_run_record(label_prefix: str) -> dict:
    """What the note's `run:` anchor must carry about one run, off that run's JSON:
    which file, which side, how many collections the recall could name, how many
    queries scored, how many gold docs the leg could not return, which index the
    daemon served, and which daemon answered. Clause 2's evidence, all of it."""
    art, path = _3076_artifact(label_prefix)
    assert art["label"] == label_prefix, (path.name, art["label"])
    side = TRIAL_RUN_LABELS[label_prefix]
    o = art["summary"]["overall"]
    corpus = art["corpus"]["doc"]
    assert int(corpus["health_url"].rsplit(":", 1)[1].split("/")[0]) == ARM_PORTS[side]
    assert corpus["index_path"] == str(TRIAL_DIR / "idx" / f"{side}.sqlite"), corpus
    return {"artifact": path.name, "arm": side,
            "gold_doc_collections": o["gold_doc_collections"],
            "n_queries": o["n_queries"],
            "gold_doc_unreturnable": len(o.get("gold_doc_unreturnable") or []),
            "index_path": corpus["index_path"], "health_url": corpus["health_url"]}


def _3076_table_row(label_prefix: str) -> dict:
    """One table row: the collection count, the query count, and the four doc metrics
    exactly as that artifact's `summary.overall` recorded them — no re-meaning."""
    art, _ = _3076_artifact(label_prefix)
    o = art["summary"]["overall"]
    return {"gold_doc_collections": o["gold_doc_collections"], "n_queries": o["n_queries"],
            **{m: o[m] for m in DOC_METRICS}}


def _3076_anchor_map(text: str, kind: str) -> dict:
    """`{key: payload}` for every `<!-- 3076:<kind>:<key> --> {json}` note line.

    The anchors exist so a test can find the note's copy of a number without
    regexing prose: the `file:line` of a markdown section moves the moment the
    section is edited, and a guard that greps for a heading is a guard that goes
    red on a typo.
    """
    out: dict[str, dict] = {}
    for line in text.splitlines():
        m = re.match(rf"^<!-- 3076:{re.escape(kind)}:(\S+) -->\s*(\{{.*\}})\s*$", line)
        if m:
            assert m.group(1) not in out, f"duplicate anchor {kind}:{m.group(1)}"
            out[m.group(1)] = json.loads(m.group(2))
    return out


def _3076_fenced_blocks(text: str) -> list[list[str]]:
    """The inner lines of every ```-fenced block in the note, in order."""
    blocks: list[list[str]] = []
    current = None
    for line in text.splitlines():
        if current is None and line.startswith("```"):
            current = []
        elif current is not None and line.startswith("```"):
            blocks.append(current)
            current = None
        elif current is not None:
            current.append(line)
    assert current is None, "the note ends inside an unterminated fence"
    return blocks


def _3076_listing_block(text: str) -> list[str]:
    """The note's one block that is a pure `qmd://lloyd-architecture/` listing.

    Clause 1 wants the `ls` listing printed; this finds it structurally (a fenced
    block whose every line is one of those URIs) instead of counting URIs across
    the whole file, which is what let a duplicated listing through.
    """
    found = [b for b in _3076_fenced_blocks(text)
             if b and all("qmd://lloyd-architecture/" in l for l in b)]
    assert len(found) == 1, f"expected one arm listing block, found {len(found)}"
    return found[0]


def _3076_fenced_after(text: str, marker: str) -> list[str]:
    """The fenced block that follows the note's line beginning `marker`."""
    lines = text.splitlines()
    hits = [k for k, l in enumerate(lines) if l.startswith(marker)]
    assert len(hits) == 1, f"want exactly one line starting {marker!r}, found {len(hits)}"
    rest = lines[hits[0] + 1:]
    start = next((k for k, l in enumerate(rest) if l.startswith("```")), None)
    assert start is not None, f"no fence follows {marker!r}"
    out = []
    for line in rest[start + 1:]:
        if line.startswith("```"):
            return out
        out.append(line)
    raise AssertionError(f"unterminated fence after {marker!r}")


def _3076_paired(control: dict, cand: dict) -> dict:
    """The trial's own paired estimator (`run_trial.py:278`), re-run here.

    Re-derived rather than read from `reach.json` so a note that printed a
    swapped-arm or wrong-population delta fails here even if the artifact it
    cites was regenerated to match: endpoints are the per-record means over the
    records BOTH arms scored, which for `doc_hit_rate`, `mrr_doc` and `ndcg10` is
    all 81 and for `doc_recall_avg` is only the 74 with doc gold — the #1600
    population divergence, applied deliberately, and the reason a table cell and
    a delta can legitimately disagree while both are right.
    """
    a = {r["id"]: r for r in control["records"] if not r.get("error")}
    b = {r["id"]: r for r in cand["records"] if not r.get("error")}
    out: dict[str, dict] = {}
    for metric, field in (("doc_hit_rate", "doc_hit"), ("doc_recall_avg", "doc_recall"),
                          ("mrr_doc", "rr_doc"), ("ndcg10", "ndcg10")):
        pairs = []
        for i in a:
            if i not in b:
                continue
            x, y = a[i]["scoring"].get(field), b[i]["scoring"].get(field)
            if x is None or y is None:
                continue
            pairs.append((float(x), float(y)))
        assert pairs, f"no paired {metric} values at all"
        out[metric] = {
            "control": round(statistics.fmean(x for x, _ in pairs), 4),
            "arm": round(statistics.fmean(y for _, y in pairs), 4),
            "delta": round(statistics.fmean(y - x for x, y in pairs), 4),
            "n": len(pairs),
            "wins": sum(1 for x, y in pairs if y > x),
            "losses": sum(1 for x, y in pairs if y < x),
        }
    return out


def test_the_3076_note_prints_the_arm_listing_the_command_emitted():
    """Clause 1: the dated note records one arm and one control run built from ONE
    snapshot, the arm's scratch config declaring `lloyd-architecture` rooted
    `/home/alansrobotlab/lloyd/architecture` with pattern `**/*.md`, and the arm's
    `qmd --index arm ls lloyd-architecture` printed as 36 documents.

    The listing is compared to `ingest.json`'s captured stdout line for line. The
    previous issue of the note rekeyed it and printed every path twice, and the
    count-based assertion it satisfied (`>= 36` URIs) said nothing about whether
    the 36 shown were the 36 emitted — 72 lines of 36 documents passed it.
    """
    _3076_require_trial()

    # One snapshot, two indices: the arm config is the control config plus exactly
    # the one entry the item specifies, in the same order. Anything else and the
    # two arms differ by more than the collection, which was the whole point.
    ctrl = yaml.safe_load((TRIAL_DIR / "idx/conf/control.yml").read_text())["collections"]
    arm = yaml.safe_load((TRIAL_DIR / "idx/conf/arm.yml").read_text())["collections"]
    assert list(arm) == list(ctrl) + ["lloyd-architecture"], (list(ctrl), list(arm))
    assert len(ctrl) == 14 and len(arm) == 15, (len(ctrl), len(arm))
    declared = arm["lloyd-architecture"]
    assert declared["path"] == ARM_COLLECTION_ROOT, declared
    assert declared["pattern"] == ARM_COLLECTION_PATTERN, declared

    # The command's own bytes, and the arm index's own count of what it holds.
    ing = json.loads(TRIAL_INGEST.read_text())
    ls = ing["ls_stdout"].rstrip("\n").splitlines()
    assert len(ls) == ing["ls_md_lines"] == 36, (len(ls), ing["ls_md_lines"])
    assert ing["ls_rc"] == 0, ing["ls_rc"]
    names = [l.rsplit("/", 1)[-1].strip() for l in ls]
    assert len(set(names)) == 36, "the captured stdout itself repeats a document"
    assert ing["arm_docs_in_collection"] == 36, ing["arm_docs_in_collection"]
    assert sorted(names) == sorted(p.name for p in (ROOT / "architecture").rglob("*.md")), (
        "the collection's 36 docs are not the checkout's 36 architecture docs")
    # The retracted claim: the first issue said `workers.md` and `qmd.md` were absent
    # from the ingest and called it "a defect in the arm's ingest". Both are listed.
    for name in ("qmd.md", "workers.md"):
        assert name in names, name

    text = TRIAL_NOTE.read_text()
    listing = _3076_listing_block(text)
    assert listing == ls, (
        f"the note's listing is not the command's stdout ({len(listing)} lines against "
        f"{len(ls)}); first difference: "
        + next((f"{a!r} != {b!r}" for a, b in zip(listing, ls) if a != b), "length"))
    assert listing.count(listing[0]) == 1, "the printed listing repeats a line"
    assert "36 lines, one per document" in text, "the note's own count of the listing"
    # Clause 1's root and pattern must be readable in the note, not just on disk.
    assert ARM_COLLECTION_ROOT in text and ARM_COLLECTION_PATTERN in text


def _3076_func_body(src: str, name: str) -> str:
    """One top-level function's source, so an env var set by its neighbour cannot
    stand in for one set by it. `_run_arm` and `_run` are the two places an arm
    process is built; a slice that runs to the next `def ` is what makes the
    assertion about each of them mean what it says."""
    start = src.index(f"def {name}(")
    rest = src[start + len(f"def {name}("):]
    end = rest.index("\ndef ")
    return src[start:start + len(f"def {name}(") + end]


def test_both_3076_arms_name_the_overlay_and_the_index_and_the_note_prints_14_vs_15():
    """Clause 2: both arm run environments name `LLOYD_CONFIG_OVERLAY` AND
    `LLOYD_QMD_INDEX`, and the note prints each artifact's `gold_doc_collections`
    (control 14, arm 15) rather than None.

    The second variable is not decoration. `app/doc_corpus.py:88-96 index_path_for`
    ends `return None if os.environ.get(CONFIG_OVERLAY_ENV) else LIVE_INDEX`, so an
    overlay WITHOUT `LLOYD_QMD_INDEX` makes `deployed_collections()` return None and
    `run_eval.py:1944-1953` write `gold_doc_collections: null` — every arm then
    reports nothing, and the 14-vs-15 count is the only evidence that the arm is the
    arm. That is why this node reads the pair out of the runner source AND out of
    each artifact, and why the note's printed copy has to match, field for field.
    """
    _3076_require_trial()

    src = TRIAL_RUNNER.read_text()
    for fn in ("_run_arm", "_run"):
        body = _3076_func_body(src, fn)
        assert "LLOYD_CONFIG_OVERLAY" in body, f"{fn} sets no config overlay"
        assert "LLOYD_QMD_INDEX" in body, f"{fn} sets no index path: every artifact " \
            "would report gold_doc_collections None and the arms would be indistinguishable"

    text = TRIAL_NOTE.read_text()
    assert "LLOYD_CONFIG_OVERLAY" in text and "LLOYD_QMD_INDEX" in text
    printed = _3076_anchor_map(text, "run")
    assert set(printed) == set(TRIAL_RUN_LABELS), (
        f"note prints {sorted(set(printed) ^ set(TRIAL_RUN_LABELS))} differently")
    for label, side in TRIAL_RUN_LABELS.items():
        want = _3076_run_record(label)
        assert printed[label] == want, (label, printed[label], want)
        assert want["gold_doc_collections"] == (14 if side == "control" else 15), want
        assert want["gold_doc_collections"] is not None, label
    assert {p["gold_doc_collections"] for k, p in printed.items()
            if TRIAL_RUN_LABELS[k] == "control"} == {14}
    assert {p["gold_doc_collections"] for k, p in printed.items()
            if TRIAL_RUN_LABELS[k] == "arm"} == {15}
    assert "control 14, arm 15" in text


def test_the_3076_cost_table_is_the_artifacts_said_over_the_same_81_queries():
    """Clause 3: the note tabulates `doc_hit_rate` / `doc_recall_avg` / `mrr_doc` /
    `ndcg10` for both sides over the SAME 81 queries with the delta stated, and says
    on its face that this leg can only tie or lose.

    Three separate ways this measurement was previously mis-transcribed, and one
    invariant that is not about transcription at all:

      * the table is compared cell by cell to `summary.overall`, so a doc-gold-population
        mean can no longer be printed under the heading "as recorded in the artifact";
      * the delta rows are compared to the trial's paired estimator RE-RUN here over the
        per-record `scoring` values, so the two populations can be quoted together only
        when each is labelled — the note's own `<!-- 3076:pair:... -->` lines carry them;
      * "the same 81 queries" is checked as set equality of the scored record ids, not as
        two `n_queries` fields that happen to agree;
      * and the leg really is one-sided: `tests/test_eval_corpus_guard.py` forbids a
        `lloyd-architecture/` gold label while no such collection is deployed, so the arm
        has nothing new to hit on the 81. That is why the gain denominator is the probe.
    """
    _3076_require_trial()

    text = TRIAL_NOTE.read_text()
    cells: dict[str, dict] = {}
    for line in text.splitlines():
        m = re.match(r"^\| `(x3076r?-[\w-]+)` \|(.*)\|$", line.strip())
        if m:
            vals = [c.strip() for c in m.group(2).split("|")]
            assert len(vals) == 6, (line, vals)
            cells[m.group(1)] = dict(zip(
                ("gold_doc_collections", "n_queries", *DOC_METRICS),
                [int(vals[0]), int(vals[1])] + [float(v) for v in vals[2:]]))
    assert set(cells) == set(TRIAL_RUN_LABELS), sorted(set(cells) ^ set(TRIAL_RUN_LABELS))
    for label, row in cells.items():
        assert row == _3076_table_row(label), (label, row)

    # The same queries, not merely the same count of them.
    for corpus, want_n in (("81", 81), ("probe", 10)):
        labels = [k for k, s in TRIAL_RUN_LABELS.items()
                  if (corpus == "81") == ("81" in k)]
        id_sets = [set(r["id"] for r in _3076_artifact(l)[0]["records"]) for l in labels]
        assert all(s == id_sets[0] for s in id_sets), corpus
        assert len(id_sets[0]) == want_n, (corpus, len(id_sets[0]))

    # Each printed delta against the estimator recomputed from raw records.
    # Every delta the note shows, against the number the trial itself recorded.
    reach = json.loads(TRIAL_REACH.read_text())["corpora"]
    score = json.loads(TRIAL_SCORE.read_text())["corpora"]
    pairs = _3076_anchor_map(text, "pair")
    stored = {
        "81/x3076": score["81"]["paired"],
        "probe/x3076": score["probe"]["paired"],
        "81/x3076r/arm_plain": reach["81"]["paired_vs_control"]["arm_plain"],
        "81/x3076r/arm_seg": reach["81"]["paired_vs_control"]["arm_seg"],
        "81/x3076r/control_seg": reach["81"]["paired_vs_control"]["control_seg"],
        "probe/x3076r/arm_plain": reach["probe"]["paired_vs_control"]["arm_plain"],
        "probe/x3076r/arm_seg_vs_control_seg": reach["probe"]["paired_seg_vs_control_seg"],
        "probe/x3076r/arm_seg_vs_plainarm": reach["probe"]["paired_seg_vs_plainarm"],
    }
    assert set(pairs) == set(stored), sorted(set(pairs) ^ set(stored))
    for key, want in stored.items():
        assert pairs[key] == want, key
    recompute = {
        "81/x3076r/arm_plain": ("x3076r-81-control", "x3076r-81-arm_plain"),
        "81/x3076r/arm_seg": ("x3076r-81-control_seg", "x3076r-81-arm_seg"),
        "probe/x3076r/arm_seg_vs_control_seg": ("x3076r-probe-control_seg",
                                               "x3076r-probe-arm_seg"),
    }
    for key, (ctrl_label, arm_label) in recompute.items():
        mine = _3076_paired(_3076_artifact(ctrl_label)[0], _3076_artifact(arm_label)[0])
        for metric in DOC_METRICS:
            got, printed = mine[metric], pairs[key][metric]
            for field in ("control", "arm", "delta", "n", "wins", "losses"):
                assert printed[field] == got[field], (key, metric, field,
                                                      printed[field], got[field])

    # The one-sidedness the clause asks the note to admit out loud, plus the fact
    # that makes it true and the denominators the two populations actually have.
    assert "can only tie or lose" in text
    corpus_labels = [d for q in yaml.safe_load(CORPUS.read_text())["queries"]
                     for d in (q.get("expect_docs") or [])]
    assert not [l for l in corpus_labels if l.startswith("lloyd-architecture/")], (
        "an anchored lloyd-architecture/ gold label is in the corpus, so the cost leg is "
        "no longer one-sided — which is only legal after a deployed collection")
    assert sum(1 for l in CORPUS.read_text().splitlines() if l.startswith("  - id: ")) == 81
    gold_docs = [q for q in yaml.safe_load(CORPUS.read_text())["queries"] if q.get("expect_docs")]
    assert len(gold_docs) == 74, len(gold_docs)


def test_the_3076_gain_probe_is_root_anchored_outside_the_repo_and_its_twins_are_quoted():
    """Clause 4: an 8-12 query probe over the 26 checkout-only basenames, stored
    OUTSIDE `~/lloyd` and passed via `--queries`, scored on both sides, every label
    root-anchored `lloyd-architecture/<basename>`, with the control's returned paths
    quoted for `qmd.md` and `workers.md`.

    The anchoring is the clause. `_doc_pair_satisfied` (`eval/run_eval.py:478-486`)
    matches doc paths with a SUBSTRING test, and `qmd.md` and `workers.md` also exist
    at `~/obsidian/projects/lloyd/architecture/` — indexed by the deployed `projects`
    collection on BOTH arms — so a bare `workers.md` label would be satisfied by the
    stale vault twin and reported as a gain for a collection that returned nothing.
    The node checks the control's quoted top-10 in both directions: no
    `lloyd-architecture/` path may appear in it, and the twin
    `projects/lloyd/architecture/workers.md` MUST appear, which is the false pass and
    the proof that the anchor is what killed it.
    """
    _3076_require_trial()

    assert TRIAL_PROBE.resolve().is_relative_to(TRIAL_VAULT), TRIAL_PROBE
    assert not TRIAL_PROBE.resolve().is_relative_to(ROOT), (
        "the probe lives in the repo, which the item's step 2 forbids")
    assert TRIAL_PROBE.name == "lloyd-architecture-ab-2026-10-20-probe.yaml"
    probe = yaml.safe_load(TRIAL_PROBE.read_text())
    assert 8 <= len(probe["queries"]) <= 12, len(probe["queries"])
    labels = [d for q in probe["queries"] for d in q["expect_docs"]]
    assert len(labels) == len(probe["queries"]), "one gold doc per probe query"
    assert all(l.startswith("lloyd-architecture/") for l in labels), labels

    checkout = {p.name for p in (ROOT / "architecture").rglob("*.md")}
    vault_twins = {p.name for p in (TRIAL_VAULT / "architecture").rglob("*.md")}
    checkout_only = checkout - vault_twins
    assert len(checkout_only) == 26, sorted(checkout_only)
    got = {l.split("/")[-1] for l in labels}
    assert got <= checkout_only, sorted(got - checkout_only)
    assert {"qmd.md", "workers.md"} <= checkout_only
    # The stale twins that make the anchoring load-bearing live one directory
    # deeper, under the deployed `projects` collection — not in the vault's own
    # `architecture` segment, which is what the 26 are measured against.
    projects_twins = {p.name for p in (TRIAL_VAULT / "projects" / "lloyd" / "architecture").rglob("*.md")}
    assert {"qmd.md", "workers.md"} <= projects_twins, sorted(projects_twins)

    assert '"--queries"' in TRIAL_RUNNER.read_text(), "the runner never passes --queries"

    twins = json.loads(TRIAL_REACH.read_text())["corpora"]["probe"]["twins"]
    text = TRIAL_NOTE.read_text()
    for side in ("control", "arm_plain", "arm_seg"):
        for query_id in ("qmd", "workers"):
            want = twins[side][query_id]["doc_paths_top10"]
            quoted = _3076_fenced_after(
                text, f"<!-- 3076:twin:{side}:{query_id} -->")
            assert quoted == want, (side, query_id, quoted, want)
            head = [l for l in text.splitlines()
                    if l.startswith(f"<!-- 3076:twin:{side}:{query_id} -->")][0]
            assert f"doc_hit={twins[side][query_id]['doc_hit']}" in head, head
            assert f"expected_docs=['lloyd-architecture/{query_id}.md']" in head, head
    control_paths = twins["control"]["workers"]["doc_paths_top10"]
    assert not [p for p in control_paths if p.startswith("lloyd-architecture/")], control_paths
    assert "projects/lloyd/architecture/workers.md" in control_paths, (
        "the vault twin is gone from the control's top-10, so this node no longer "
        "proves the anchored label is what prevented the false pass")
    for side in ("arm_plain",):
        for query_id in ("qmd", "workers"):
            assert not [p for p in twins[side][query_id]["doc_paths_top10"]
                        if p.startswith("lloyd-architecture/")], side
    for query_id in ("qmd", "workers"):
        arm_seg = twins["arm_seg"][query_id]["doc_paths_top10"]
        assert arm_seg[0] == f"lloyd-architecture/{query_id}.md", arm_seg


def test_the_3076_note_states_the_two_numbers_and_the_live_config_is_untouched():
    """Clause 5: the decision is two numbers over their own denominators (cost =
    delta over the 81, gain = delta over the probe) with the ship/revert rule beside
    them, and the note proves `~/.config/qmd/index.yml` was never edited.

    The no-touch proof is re-measured here, not believed. The first issue printed
    `grep -c -- '- name: '` output claiming 14 and 14 for the two qmd configs; that
    command returns **0** for them, because this fork's `index.yml` is a mapping
    (`memory:` / `path:` / `pattern:`), not a `- name:` list — a no-touch section that
    prints numbers its commands cannot produce is worse than no section. So this node
    runs its own checks: the string `lloyd-architecture` occurs in neither config, the
    live config still parses to 14 collections, the template still agrees with it, the
    gold file still says 81 and has 81 entries, and every number in the note's own
    no-touch block equals what the same command says now.
    """
    _3076_require_trial()

    for path in (TRIAL_LIVE_YML, TRIAL_TEMPLATE_YML):
        assert "lloyd-architecture" not in path.read_text(), (
            f"{path} now declares the collection: the trial's no-touch claim is over "
            "and the note must be superseded, not re-asserted")

    live = yaml.safe_load(TRIAL_LIVE_YML.read_text())["collections"]
    assert len(live) == 14, sorted(live)

    text = TRIAL_NOTE.read_text()
    block = _3076_fenced_after(text, "Re-run fresh at")
    # Commands and their outputs, in the order they appear: a block of `$ cmd` lines
    # followed by the lines that cmd printed. Read sequentially, because a dict built
    # by filtering the whole tail is a dict whose first command "prints" every number
    # in the block — which is how the previous issue's no-touch section could stand.
    cmds: list[tuple[str, list[str]]] = []
    for line in block:
        stripped = line.strip()
        if stripped.startswith("$ "):
            cmds.append((stripped[2:], []))
        elif stripped and cmds:
            cmds[-1][1].append(stripped)
    assert len({c for c, _ in cmds}) == len(cmds), [c for c, _ in cmds]
    printed = dict(cmds)

    def says(command: str, *expected: str):
        hits = [k for k in printed if command in k]
        assert len(hits) == 1, (command, sorted(printed))
        assert printed[hits[0]] == list(expected), (command, printed[hits[0]], expected)

    says("grep -c lloyd-architecture ~/.config/qmd/index.yml", "0", "rc=1")
    says("grep -c lloyd-architecture ~/lloyd/agent-services/conf/qmd-index.yml", "0", "rc=1")
    says("grep -cE '^  [A-Za-z0-9_-]+:$'",
         f"{TRIAL_LIVE_YML}:14", f"{PRODUCTION_REPO}/agent-services/conf/qmd-index.yml:14")
    says("qmd collection list | head -1", "Collections (14):")
    says("qmd collection list | grep -c 'qmd://'", "14")
    says("qmd collection list | grep -c 'lloyd-architecture'", "0")
    says("grep -c '^  - id: ' ~/lloyd/eval/vault_recall_queries.yaml", "81")
    says("find ~/lloyd/architecture -name '*.md' | wc -l",
         str(len(list((PRODUCTION_REPO / "architecture").rglob("*.md")))))
    header = [l for l in CORPUS.read_text().splitlines() if "GOLD SET SIZE" in l]
    assert len(header) == 1 and " 81 queries" in header[0], header
    says('grep -n "GOLD SET SIZE"', header[0].strip())

    # The two numbers, each over its own denominator, and the rule they are read by.
    cost = _3076_paired(_3076_artifact("x3076r-81-control_seg")[0],
                        _3076_artifact("x3076r-81-arm_seg")[0])
    gain = _3076_paired(_3076_artifact("x3076r-probe-control_seg")[0],
                        _3076_artifact("x3076r-probe-arm_seg")[0])
    lines = text.splitlines()
    for word, stats in (("COST", cost), ("GAIN", gain)):
        at = [k for k, l in enumerate(lines)
              if l.startswith(f"{word}  over the") and "paired bootstrap" in l]
        assert len(at) == 1, (word, len(at))
        # A set note may write a negative with a typographic minus; the number is
        # the number, so fold it rather than forbidding the typography.
        body = lines[at[0] + 1].replace("\u2212", "-")
        for metric in DOC_METRICS:
            assert f"{metric} {stats[metric]['delta']:+.4f}" in body, (word, metric, body)
    assert "COST  over the 81-query gold set" in text and "GAIN  over the 10-query probe" in text
    for phrase in ("Ship is licensed only", "Revert / close #3075"):
        assert phrase in text, phrase


"""Append the node that pins the ANCHORING itself across a process boundary.

Why this node exists: the previous round of #3076 failed its `tests` rung at the gate
with two nodes red, and the reason was not the measurement — it was that the note and
the probe set were read through `Path.home()`, and the gate runs this file with `HOME`
pointed at `<round>/home`, whose `obsidian` is a clone of the vault taken when the
round OPENED (`gate.py:1595` sets `"HOME": str(home)`; `worktree.ensure_round_home:78`
builds the farm). The trial's note was repaired by a vault land AFTER the code round
opened, so the gate graded superseded prose, and an earlier configuration skipped all
six nodes for the same reason. `ACCOUNT_HOME` / `production_data_root()` are the
production answer whatever `$HOME` says.

A node that merely asserts `TRIAL_VAULT == ACCOUNT_HOME / "obsidian"` in-process is
graded on the same interpreter that got it right by luck; the failure mode lives in a
CHILD process whose environment differs. So this node launches `python -c` with a
deliberately bogus `HOME` and asks the module, imported in that child, which paths it
resolves — the exact seam the gate crosses.
"""
from pathlib import Path

PROBE = r"""
import sys, json
sys.path.insert(0, %(rootr)s)
import tests.test_eval_corpus_guard as m
print(json.dumps({
    "home": __import__("os").path.expanduser("~"),
    "vault": str(m.TRIAL_VAULT),
    "note": str(m.TRIAL_NOTE),
    "probe": str(m.TRIAL_PROBE),
    "live_yml": str(m.TRIAL_LIVE_YML),
    "repo": str(m.PRODUCTION_REPO),
    "dir": str(m.TRIAL_DIR),
    "account": str(m.ACCOUNT_HOME),
}))
"""


def test_the_3076_trial_paths_survive_the_home_the_gate_runs_them_under():
    """The six #3076 nodes must resolve the SAME production paths in a child whose
    `$HOME` is not the real one, because the gate runs this file under exactly such a
    `$HOME` (`scripts/automod/gate.py:1595`) — so a trial read that follows `$HOME`
    resolves through the round's farm instead of the account home, which is where
    SM_20261020_175859's two nodes failed and the six before them skipped.

    The child prints eight values. Six are the trial's own paths and each must equal
    the constant this module holds however the child's `$HOME` is set: `TRIAL_VAULT`
    (the vault the note and probe live in), `TRIAL_NOTE` and `TRIAL_PROBE` (clause 1
    and clause 4's documents), `TRIAL_DIR` (the trial's scratch, production-anchored at
    `ecddc4d3`), `TRIAL_LIVE_YML` (`~/.config/qmd/index.yml`, the no-touch proof's
    subject — a mirrored copy would let the guard "prove" the file was untouched by
    reading a clone of it) and `PRODUCTION_REPO` (`app.paths.LIVE_CHECKOUT`, whose paths
    the note prints verbatim). The seventh is the child's `os.path.expanduser("~")`,
    which must be the bogus one — proof the environment took — and the eighth is the
    child's `ACCOUNT_HOME`, which under that bogus `$HOME` must still equal THIS
    process's `ACCOUNT_HOME`: `app/data_root.py:77` derives it from `getpwuid`. That
    difference between two homes inside one dict is the whole property the six anchors
    lean on, and the reason the comparison is against the constant and never against
    `Path.home()` — at the gate those two are different directories.
    """
    import json
    import subprocess

    bogus = Path("/tmp/definitely-not-the-real-home-3076")
    assert bogus != Path.home(), "the fixture home must differ from the real one"

    proc = subprocess.run(
        [str(PY), "-c", PROBE % {"rootr": repr(str(ROOT))}],
        cwd=str(ROOT), env={"PATH": "/usr/bin:/bin", "HOME": str(bogus),
                            "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])

    assert got["home"] == str(bogus), got
    # Each expectation is built from the constant THIS MODULE anchored, never from
    # `Path.home()`. The gate runs this file with `$HOME` at the round's mirrored home
    # (`scripts/automod/gate.py:1595` → `<round>/home`), so there the parent's
    # `Path.home()` is NOT the account home, and a claim written with `Path.home()`
    # accuses a correctly-anchored child of following the bogus `$HOME`. That is this
    # node's own defect restated inside the assertion meant to catch it: its first gate
    # run failed at `account`, then at
    # `got["vault"] == str(Path.home() / "obsidian")` — `'/home/alansrobotlab/obsidian'
    # != '/home/alansrobotlab/lloyd-work/<round>/home/obsidian'` — so on a bare shell it
    # could only ever pass where the bug is invisible. A $HOME-independence claim needs
    # a $HOME-independent name on BOTH sides of the `==`.
    assert got["account"] == str(ACCOUNT_HOME), (
        f"the child's ACCOUNT_HOME ({got['account']}) is not this process's "
        f"({ACCOUNT_HOME}), so app.data_root stopped being $HOME-independent and "
        "every anchor below is a lie")
    for key, mine in (("vault", TRIAL_VAULT), ("note", TRIAL_NOTE),
                      ("probe", TRIAL_PROBE), ("live_yml", TRIAL_LIVE_YML),
                      ("repo", PRODUCTION_REPO), ("dir", TRIAL_DIR)):
        assert got[key] == str(mine), (key, got[key], str(mine))
    assert got["note"].endswith("/eval/measurements/lloyd-architecture-ab-2026-10-20.md"), got
    assert got["probe"].endswith("-probe.yaml") and got["probe"].startswith(got["vault"]), got
    # The case that exists only at the gate: when this process is itself under a
    # steered `$HOME`, an anchored path and a `$HOME`-built path are different FILES, so
    # "nothing followed it" becomes a claim that can fail. On a bare shell
    # `Path.home() == ACCOUNT_HOME` and this adds nothing to the loop above.
    if Path.home() != ACCOUNT_HOME:
        for key in ("vault", "note", "probe", "live_yml", "repo", "dir"):
            assert not Path(got[key]).is_relative_to(Path.home()), (
                key, got[key], f"resolves under the steered home {Path.home()}, so the "
                "gate would be grading a mirror of the very file it is certifying")
    # And the anchored files are really readable: a guard that "passes" because it
    # skipped is what cost the previous round its gate.
    for key in ("note", "probe", "live_yml"):
        assert Path(got[key]).is_file(), (key, got[key])


# --------------------------------------------------------------------------- #
# 2026-10-20 round SM_20261020_234603 — #3086: the one tally #3076's repair left
# behind. §3 of the measurement note asserts "The upper bound is genuinely high"
# and prints, three lines under that assertion, the artifact's own summary as
# `{"queries":10,"gold_in_top10":0,"gold_top1":0,"mean_reciprocal_rank":0.0}`.
# That zero is not a measurement. The trial's runner ranks each query with
#     rank = next((i + 1 for i, f in enumerate(files) if f.endswith("/" + gold)), None)
# (`run_trial.py:527`), and the `files` it tests are the RAW hits the fork returned:
# `qmd://lloyd-architecture/<name>?index=arm`. It is the `?index=arm` SUFFIX, not the
# `qmd://` scheme, that defeats every `endswith`, so all ten `rank_in_named_collection`
# values are null and the summary counts 0 / 0 / 0.0 for a run that was in fact finding
# the gold document. Read past the bug, the SAME artifact's per-query `top5` puts gold
# in the top five for 9 of 10 queries and at rank 1 for 7 of 10. The runner is untracked
# runtime scratch under `~/lloyd-data/eval/3076/` — outside the repo and outside the
# vault — so its summary will keep reading zero, and the note is the only durable place
# the correction can live. Which is also the hazard: a re-run regenerates `upper.json`,
# and then the quoted zero ages out together with the explanation beside it. So the
# note quotes both numbers, and the node below recomputes them from the artifact and
# goes red the moment the note's copy stops matching — a stale zero becomes a failing
# test, not a silent citation.
#
# What the tally may NOT say: anything about rank 6-10. `top5` is truncated at five
# hits, so the one miss (`research-pipeline`) is only ever shown as "not in the first
# five". A note sentence writing the recomputed count as `gold_in_top10 = 9` would put
# a new unbacked number where the old one was corrected, so clause 4's ban is checked
# over the whole section rather than only over the sentences that carry a figure.
# --------------------------------------------------------------------------- #

#: The spellings that would turn a top-5 tally into a ten-deep claim. Checked against
#: every non-anchor line of §3, so the bound holds across the section's prose and not
#: just across the one sentence that quotes the figures.
UPPER_OVERREACH = ("top-10", "top 10", "top ten", "in 10", "in ten")

#: The §3 needles that make the artifact's own zero explainable rather than merely
#: quoted: the file that computes it, the line that computes it, the expression that is
#: wrong, the suffix that makes it wrong, and the field that comes back null.
UPPER_BUG_NEEDLES = ("run_trial.py", "run_trial.py:527", "endswith", "?index=arm",
                     "rank_in_named_collection", "untracked")

#: The anchor key §3 carries its recomputed copy under, and the keys it must carry.
UPPER_RECOMPUTED_KEYS = ("queries", "gold_in_top5", "gold_top1", "misses")

#: Attribute names through which a guard could read a steered `$HOME`. `Path.home` and
#: `os.path.expanduser`/`expandvars` are the three spellings that resolve to whatever
#: home the interpreter was handed, so refusing these ATTRIBUTES — rather than a
#: rendered call string — is the ban that can actually fire. See the node that uses it.
HOME_READER_ATTRS = ("home", "expanduser", "expandvars")


def _3076_home_reads(tree) -> set[str]:
    """The attribute names in `tree` through which a path could be built off a steered
    `$HOME` — the ONLY spelling of this ban that can fail.

    The first attempt of this round wrote it as `not [c for c in called if
    c.endswith("home()")]` over `ast.unparse(node.func)`, and `ast.unparse` of a
    `Path.home()` call's FUNC is `Path.home`: the parentheses belong to the Call node,
    not to its func, so no source has ever produced a rendered func ending in
    `home()`. Review caught it; the review rung is right that an assertion which cannot
    fail is worse than none, because it reads as coverage. Matching the ATTRIBUTE is the
    same ban with a failure mode: whatever you do to the result — call it, divide it
    into a path, assign it — the read of `home`/`expanduser`/`expandvars` is an
    `ast.Attribute` node in the source. It is a helper rather than an inline expression
    so the node can run it against a positive control, not only against the node it is
    guarding.
    """
    import ast

    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in HOME_READER_ATTRS}


def _3076_upper_names(top5) -> list[str]:
    """One hit list as the basenames it names, gold-comparable.

    `run_trial.py:528-530` already strips the `qmd://lloyd-architecture/` prefix on its
    way into `top5`, so the stored strings are `<name>?index=arm` and the `?index=`
    suffix is ALL that is left to strip. The trailing split on `/` is a deliberate
    no-op today: it keeps this reader correct if a re-run ever stores the raw URI, and
    it means a node never fails for the wrong reason by insisting on a prefix the
    artifact stopped writing.
    """
    return [h.split("?index=")[0].split("/")[-1] for h in (top5 or [])]


def _3076_upper_tally(artifact: dict) -> dict:
    """gold-in-top-5 and rank-1 over `queries[].top5` — the artifact read past its own
    broken summary, from the per-query lists rather than from any `rank` field (every
    one of which is null, which is the bug, not the result)."""
    queries = artifact["queries"]
    hit5, rank1, misses = 0, 0, []
    for q in queries:
        names = _3076_upper_names(q.get("top5"))
        # The shape the arithmetic above depends on, asserted rather than assumed: a
        # hit list that came back longer than five, or still carrying a scheme or a
        # suffix, would make `gold_in_top5` a number computed on junk.
        assert len(names) <= 5, (q["id"], names)
        assert all("?" not in n and "/" not in n for n in names), (q["id"], names)
        if q["gold"] in names:
            hit5 += 1
        else:
            misses.append(q["id"])
        if names and names[0] == q["gold"]:
            rank1 += 1
    return {"queries": len(queries), "gold_in_top5": hit5, "gold_top1": rank1,
            "misses": sorted(misses)}


def _3076_section(text: str, heading: str) -> str:
    """The note's one section that opens with `heading`, up to the next `## ` heading.

    A section, not a `file:line` slice: this note has already been re-issued once
    (`aeaceeae` repaired its listing, its tables and its no-touch block), and every
    line number in it moved. The heading is the address that survives an edit.
    """
    lines = text.splitlines()
    at = [k for k, l in enumerate(lines) if l.startswith(heading)]
    assert len(at) == 1, f"want exactly one line starting {heading!r}, found {len(at)}"
    rest = lines[at[0] + 1:]
    end = next((k for k, l in enumerate(rest) if l.startswith("## ")), len(rest))
    return "\n".join(rest[:end])


def test_the_3076_note_requotes_the_upper_bound_the_summary_cannot_report():
    """Clauses 1-4: §3 quotes the named-collection upper bound as `upper.json` actually
    supports it — gold in top-5 for 9 of 10 queries and gold at rank 1 for 7 of 10 — on
    a `<!-- 3076:upper:recomputed -->` anchor the existing anchor parser reads, keeps the
    artifact's own all-zero summary verbatim beside it, and says in the same section why
    that zero is a path-form bug in the untracked runner.

    Both directions are pinned. The note's anchor must equal the tally recomputed here
    from `queries[].top5`, so a re-run that changes the artifact turns this node red
    instead of leaving the note quoting a stale number; and the note's copy of the
    artifact's `summary` must equal the artifact's `summary`, so the zero the note
    prints stays the zero the file holds. Neither figure may be worded as a ten-deep
    result: `top5` stops at five hits, so the artifact cannot support a rank 6-10 claim
    about the one miss, `research-pipeline`.
    """
    require_live_data(TRIAL_UPPER, "the #3076 named-collection upper-bound artifact",
                      kind="file")
    require_live_data(TRIAL_NOTE, "the #3076 measurement note", kind="file")

    art = json.loads(TRIAL_UPPER.read_text())
    want = _3076_upper_tally(art)
    # The ANCHOR MAP IS READ OUT OF §3, NOT OUT OF THE WHOLE NOTE. Both clauses are
    # §3-clauses: clause 1 asks for the recomputed line in that section and clause 2
    # asks for the zero "in the same section" as its explanation, so an anchor lifted
    # into §4 or an appendix has to be red, not green-and-shelved. The parser scans the
    # text it is handed, so handing it the section is what makes placement checkable;
    # the first attempt of this round parsed the whole file and the review rung named
    # it as the one clause-1 gap. (The zero line is separately re-checked as a §3 line
    # below, so a §3-scoped map and a §3-scoped line test cannot disagree.)
    sec = _3076_section(TRIAL_NOTE.read_text(), "## 3.")
    anchors = _3076_anchor_map(sec, "upper")

    # Clause 1: the recomputed figures are there, on an anchor line, and they are the
    # artifact's own numbers — 9 of 10 in top-5 and 7 of 10 at rank 1 as this run stands.
    assert "recomputed" in anchors, (
        f"§3 carries no `<!-- 3076:upper:recomputed -->` anchor (keys present: "
        f"{sorted(anchors)}), so the section still quotes only the broken summary")
    rec = anchors["recomputed"]
    assert tuple(rec) == UPPER_RECOMPUTED_KEYS, sorted(rec)
    assert rec == want, (
        f"the note's anchor says {rec} but recomputing upper.json's per-query top5 "
        f"gives {want}: the tally in the note is not the artifact's tally")
    assert want["queries"] == 10, want
    assert (want["gold_in_top5"], want["gold_top1"]) == (9, 7), (
        f"the artifact now tallies {want['gold_in_top5']}/{want['queries']} in top-5 and "
        f"{want['gold_top1']}/{want['queries']} at rank 1, not the 9 and 7 the note and "
        "the item both quote — the trial was re-run, and §3 has to be rewritten, not "
        "this assertion widened")
    assert want["misses"] == ["research-pipeline"], want

    # Clause 2: the artifact's own summary is still quoted, verbatim, still zero, and
    # still the artifact's own — and the section states why it reads that way.
    assert "summary" in anchors, f"the verbatim zero line was dropped: {sorted(anchors)}"
    assert anchors["summary"] == art["summary"], (
        f"the note prints {anchors['summary']} where upper.json holds {art['summary']}: "
        "a re-run changed the artifact and the note's quotation of it is now stale")
    assert anchors["summary"]["gold_in_top10"] == 0 == anchors["summary"]["gold_top1"], (
        "the summary stopped being the broken one — if the runner was fixed at source, "
        "§3's explanation is obsolete and has to be rewritten, not left beside a zero "
        "that no longer exists")
    assert anchors["summary"]["mean_reciprocal_rank"] == 0.0, anchors["summary"]
    assert [q.get("rank_in_named_collection") for q in art["queries"]] == [None] * len(
        art["queries"]), "some rank is no longer null, so the endswith bug is gone"
    for needle in UPPER_BUG_NEEDLES:
        assert needle in sec, f"§3 no longer names {needle!r}: the zero is quoted but " \
                              "no longer explained, and a reader is left to cite it"
    assert any(l.startswith("<!-- 3076:upper:summary -->") for l in sec.splitlines()), \
        "the zero must stay visible as its own line in §3, not paraphrased into prose"

    # Clause 1 again, in prose: the two figures are quoted as counts of ten queries,
    # each beside the window that supports it — 9 next to top-5, 7 next to rank 1.
    for num, windows in ((want["gold_in_top5"], ("top-5", "top 5", "top five")),
                         (want["gold_top1"], ("rank 1", "rank-1", "first place"))):
        quoting = [l for l in sec.splitlines()
                   if re.search(rf"\b{num} of {want['queries']}\b", l)]
        assert quoting, f"§3 never states {num} of {want['queries']} in words"
        for line in quoting:
            assert any(w in line for w in windows), (num, windows, line)

    # Clause 4: nowhere in the section does any figure become a ten-deep claim. Only
    # anchor lines are exempt, and they are the artifact's own key names.
    for line in sec.splitlines():
        if line.startswith("<!-- 3076:"):
            continue
        assert not any(s in line for s in UPPER_OVERREACH), (
            f"§3 says something ten-deep on a source truncated at five hits: {line!r}")


def test_the_3076_upper_bound_guard_is_anchored_read_only_and_launches_nothing():
    """Clause 5, read off the AST of the node above rather than off its prose: it opens
    `upper.json` and the note through production-anchored constants, skips BY NAME
    through `require_live_data` when either is absent, imports no way to launch a
    process, and calls no `$HOME`-derived path constructor — the gate runs this file
    with `HOME` at the round's farm (`scripts/automod/gate.py:1595`), where a steered
    home names a mirror or nothing at all.

    `ast` and not a substring scan, because the node's own docstring has to be ALLOWED
    to name the things it must not do: a `grep`-style check would fail for writing this
    paragraph. What is checked is the call graph of the compiled function, which is the
    thing that could actually re-run a trial or spend a GPU.
    """
    import ast
    import inspect

    target = test_the_3076_note_requotes_the_upper_bound_the_summary_cannot_report
    tree = ast.parse(inspect.getsource(target))
    called = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    # No way to run anything: clause 3 is a read of an existing artifact, and a node
    # that shells out is a node that could re-run the trial the item forbids.
    assert not [c for c in called if c.split(".")[0] in
                {"subprocess", "os", "pty", "signal"}], called
    assert not {"subprocess", "os", "pty"} & imported, sorted(imported)
    # No path built from the environment: `ROOT` is this tree, and at the gate that is
    # the round's worktree, not the trial's scratch or the vault.
    assert "ROOT" not in names, sorted(names)
    # Finding (a), the review rung's own words: the previous commit's
    # `not [c for c in called if c.endswith("home()")]` could never fail, because
    # `ast.unparse` of a `Path.home()` call's func is `Path.home` — the parens are the
    # Call's, not the func's. The ban now reads the ATTRIBUTE, and runs against a
    # positive control in the same breath: a snippet holding all three steerable-home
    # spellings has to be flagged by the same helper, or the assertion below is being
    # satisfied by absence rather than by detection. The control is the part an
    # unfalsifiable assertion was missing.
    assert not _3076_home_reads(tree), _3076_home_reads(tree)
    control = ast.parse(
        "from pathlib import Path\nimport os\n"
        "a = Path.home() / 'lloyd-data'\n"
        "b = os.path.expanduser('~/lloyd-data')\n"
        "c = os.path.expandvars('$HOME/lloyd-data')\n"
        "d = Path.home().joinpath('x')\n")
    assert _3076_home_reads(control) == {"home", "expanduser", "expandvars"}, (
        _3076_home_reads(control), "the ban detects nothing, so it bans nothing")

    # Skips by name, on both of the two files it reads, both of which are files.
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "require_live_data"]
    guarded = {ast.unparse(call.args[0]) for call in guards if call.args}
    assert {"TRIAL_UPPER", "TRIAL_NOTE"} <= guarded, (sorted(guarded), len(guards))
    for call in guards:
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert kwargs.get("kind") == "'file'", ast.unparse(call)
    assert all(isinstance(call.args[-1], ast.Constant) for call in guards
               if call.args), "each skip names WHICH piece is missing"

    # And the constants it reads through are production-anchored, whatever `$HOME`
    # the interpreter was started with.
    assert TRIAL_UPPER == TRIAL_DIR / "upper.json"
    assert TRIAL_DIR == production_data_root() / "eval" / "3076"
    for path in (TRIAL_UPPER, TRIAL_NOTE):
        assert path.is_absolute() and path.is_relative_to(ACCOUNT_HOME), (
            path, ACCOUNT_HOME, "not anchored at the account home, so at the gate this "
            "guard would be certifying a mirror of the file it is certifying")
