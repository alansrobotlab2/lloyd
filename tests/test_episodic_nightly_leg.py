"""#4404 — the nightly can now SEE the episodic floor.

`retrieval.recall.episodic_floors` is ON in production (`config.yaml:2888`,
read by `recall_episodic_floors()` at `agent_mcp/vault.py:421`, floor
`RECALL_EPISODIC_FLOOR = {"sessions": 1}` at `:417`). Until now the nightly
retrieval eval never asked whether it still worked: `git grep -n
"episodic\\|RECALL_EPISODIC" -- eval/run_eval.py` returned 0 hits, and no gold
row in `eval/vault_recall_queries.yaml` can name a sessions document — two
guards pin that (`tests/test_eval_corpus_guard.py:1767`, `:3265`), so a floor
that silently stopped returning chat transcripts left every nightly doc-leg
number exactly flat. Its live behaviour had been measured once, by hand, on
2026-09-25 (`eval/measurements/episodic-recall-2026-09-25.md`).

This file pins the two things the item asked for, and nothing else:

  * the nightly writes the leg's own `doc_hit`/`mrr` into the named artifact
    field `episodic_floor_leg`, computed over the generated proxy set, with no
    live qmd or model call in the test path (the retrieval seam is the same
    `ev._vault_recall` the doc leg is patched at);
  * the leg's items come from `run_episodic_recall_eval.episodic_items()` — the
    daily-note proxy generator — and NEVER `synth_items()`, which needs the
    primary model and produced 59 of 68 mislabelled rows when it was used
    (#1319). `synth_items` is armed to raise below, so a wiring change that
    routes the nightly through it fails here instead of quietly costing a model
    call every night.

Report-only is deliberate and is NOT a shortcut: `tests/conftest.py:815` sets
`LLOYD_RECALL_EPISODIC=0`, so no pytest node can observe floor-on behaviour and
no test can gate on it. The pin is the artifact's own field; the witness that
the nightly can now see the floor is the first live nightly's leg, scored twice
(#4404's owed check).
"""
import ast
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import doc_corpus  # noqa: E402
import eval.run_eval as ev  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────

_DATE = "2026-09-22"
# Two transcripts on the same day, two auto-captures. Both are three-part chat
# ids, which is what `episodic_items` filters on (worker exports before
# 2026-09-10 are two-part and cannot be resolved to a capture minute).
_HIT = "20260922_203910_iv4437.md"
_MISS = "20260922_210500_iv9999.md"

_HIT_TEXT = (
    "user: the retention sweep kept reporting a clean fleet while the "
    "grounding pass never ran, so check the scheduler row before the sweep "
    "itself\n"
    "assistant: the sweep is healthy and the scheduler had the job parked in "
    "draft, which is why every per-task number stayed flat across the whole "
    "window\n"
)
_MISS_TEXT = (
    "user: why did the wake word stop answering after the reboot\n"
    "assistant: the ALSA card index moved, so the listener is attached to a "
    "device that no longer exists\n"
)


def _daily_note(captures: list[tuple[str, str]]) -> str:
    """A daily note with `### Session HH:MM <name> — Auto-captured` blocks.

    The paragraph under each header is the secondary model's summary; the proxy
    generator takes its FIRST SENTENCE as the query and resolves the label to
    the transcript that holds at least half of its content words. The transcript
    below therefore carries the paragraph verbatim, which makes the label
    resolution deterministic rather than a second opinion about matching.
    """
    parts = ["---", "segment: memory", "type: daily", "---", ""]
    for hhmm, name, para in captures:
        parts += [f"### Session {hhmm} {name} — Auto-captured", "", para, ""]
    return "\n".join(parts)


def _proxy_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A temp vault's `memory/` and a temp qmd `sessions` collection: 2 items."""
    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / f"{_DATE}.md").write_text(_daily_note([
        ("20:39", "alpha", _HIT_TEXT.split("\n")[0].split(" ", 1)[1].rstrip()
         + ". Nothing else was said."),
        ("21:05", "beta", _MISS_TEXT.split("\n")[0].split(" ", 1)[1].rstrip()
         + ". Nothing further was recorded."),
    ]), encoding="utf-8")
    sessions = tmp_path / "derived" / "sessions" / _DATE
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / _HIT).write_text(_HIT_TEXT, encoding="utf-8")
    (sessions / _MISS).write_text(_MISS_TEXT, encoding="utf-8")
    return memory, tmp_path / "derived" / "sessions"


def _fake_index(tmp_path: Path, sessions_root: Path) -> Path:
    """A qmd-shaped index whose `sessions` collection is rooted in the fixture.

    The real deployment roots `sessions` at
    `~/lloyd-data/_pipeline/vault-derived/sessions` and `~/obsidian/sessions` is
    an empty directory, so the fixture names its own root and the run is
    pointed at this file by `LLOYD_QMD_INDEX` — the same env var
    `app.doc_corpus.INDEX_PATH_ENV` uses for a pinned corpus. No real index is
    opened by anything in this file.
    """
    db = tmp_path / "fixture-index.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table store_collections (name text, path text)")
    con.executemany("insert into store_collections values (?, ?)",
                    [("sessions", str(sessions_root))])
    con.commit()
    con.close()
    return db


# ── clause 1: the leg's numbers reach the named artifact field ───────────────

def _leg_only_recall(hits: list[str], calls: list[str]):
    """A stand-in retrieval that returns `hits` and records every query."""
    def _recall(params, **kw):
        calls.append(params["query"])
        return {"documents": [{"path": p} for p in hits], "facts": [],
                "entities": []}
    return _recall


def test_the_floor_leg_scores_the_proxy_set_into_its_named_field(tmp_path, monkeypatch):
    """Two proxy items, one returned, one not: the field carries 0.5 and 0.5.

    The expected values are recomputed here by hand — two items, a hit for the
    first at rank 1 — so the node cannot pass on a leg that scores nothing or
    scores everything. `episodic_floor_leg` is the field the item named, and the
    module constant is what `main` writes, so a rename that drops the field from
    the artifact fails here rather than in six weeks of trend data.
    """
    memory, sessions = _proxy_tree(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(ev, "_vault_recall",
                        _leg_only_recall([f"sessions/{_DATE}/{_HIT}"], calls))

    out: dict = {}
    field = ev.attach_episodic_floor_leg(
        out, memory_dir=memory, sessions_root=sessions, k=10)

    leg = field[ev.EPISODIC_FLOOR_KEY]
    assert leg["ran"] is True, f"the leg refused on a valid fixture: {leg}"
    assert leg["n_items"] == 2, (
        "the proxy set resolved to 2 auto-captures; a different count means the "
        "fixture stopped being the generated set this leg is specified to score")
    assert leg["doc_hit"] == pytest.approx(0.5)
    assert leg["mrr"] == pytest.approx(0.5)
    assert leg["n_errors"] == 0
    # One returned slot per item is a transcript, which is the whole point of
    # the floor; a leg that scored only vault notes would say 0.
    assert leg["sessions_slots"] == 2
    assert len(calls) == 2, "one recall call per proxy item, no counterfactual twin"
    assert all(q for q in calls), "every proxy item's query reached the recall"


def test_the_leg_never_calls_the_real_recall_and_never_a_model(tmp_path, monkeypatch):
    """No qmd, no daemon, no primary model: the seam is the recall and nothing else.

    `ev._vault_recall` is replaced with a raiser, so the leg cannot reach the
    live daemon; `synth_items` is the only path in that module that opens an
    HTTP client to the primary, and it is armed to raise too. The leg still has
    to produce numbers through its injected `recall_fn`, which is what
    `main` does not pass — so this node also pins that an injected seam is
    honoured instead of being shadowed by the module-level name.
    """
    memory, sessions = _proxy_tree(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("the leg reached the live recall seam")

    import eval.run_episodic_recall_eval as epi
    monkeypatch.setattr(ev, "_vault_recall", _boom)
    monkeypatch.setattr(epi, "synth_items", _boom)

    field = ev.attach_episodic_floor_leg(
        {}, memory_dir=memory, sessions_root=sessions, k=10,
        recall_fn=lambda params, **kw: {
            "documents": [{"path": f"sessions/{_DATE}/{_MISS}"}], "facts": []})
    leg = field[ev.EPISODIC_FLOOR_KEY]
    assert leg["ran"] is True and leg["doc_hit"] == pytest.approx(0.5)


# ── clause 2: the items are the generated proxy set, never the synthetic one ─

def test_the_legs_items_come_from_episodic_items_not_synth_items(tmp_path, monkeypatch):
    """`episodic_items` is called with the fixture's roots; `synth_items` raises.

    The nightly's path has no `--synth-file` argument at all, because it does
    not shell out to `run_episodic_recall_eval.py`'s CLI — it imports the
    generator. That makes the unreachable-by-argv claim checkable two ways, and
    both are asserted: the raising stand-in (a wiring change fires it), and the
    absence of the flag spelling in the leg's own source (the nightly's
    invocation cannot grow one by accident).
    """
    import inspect

    import eval.run_episodic_recall_eval as epi

    seen: list[tuple] = []
    real = epi.episodic_items

    def _spy(memory_dir, sessions_root, window_min):
        seen.append((Path(str(memory_dir)), Path(str(sessions_root)), window_min))
        return real(memory_dir, sessions_root, window_min)

    def _boom(*a, **k):
        raise AssertionError("the nightly leg routed through synth_items")

    monkeypatch.setattr(epi, "episodic_items", _spy)
    monkeypatch.setattr(epi, "synth_items", _boom)

    memory, sessions = _proxy_tree(tmp_path)
    monkeypatch.setattr(ev, "_vault_recall", _leg_only_recall([], []))
    leg = ev.attach_episodic_floor_leg(
        {}, memory_dir=memory, sessions_root=sessions, k=10,
        window_min=45)[ev.EPISODIC_FLOOR_KEY]

    assert seen == [(memory, sessions, 45)], (
        "the leg must call the proxy generator with the memory dir and the "
        f"sessions collection root it was given; got {seen}")
    assert leg["item_source"] == "episodic_items"
    # AST, not substring: the leg's own docstring USES the words "synth_items"
    # and "--synth-file" to say they are forbidden, so a text search fires on the
    # prohibition rather than on the routing. A call graph cannot be prose.
    tree = ast.parse(inspect.getsource(ev.episodic_proxy_items) + "\n"
                     + inspect.getsource(ev.attach_episodic_floor_leg))
    called = ({n.func.id for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
              | {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)})
    assert "episodic_items" in called, "the generator is reached by attribute call"
    assert "synth_items" not in called, "the leg routed through the synthetic set"
    assert not any(c.split(".")[-1] in {"Popen", "system", "run", "check_output",
                                        "popen"} for c in called), (
        "the leg runs in-process; shelling out to the script's CLI would let a "
        "--synth-file appear on its path without changing this module")
    cli_flags = [a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                 for a in list(n.args) + [k.value for k in n.keywords]
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value.startswith("--")]
    assert cli_flags == [], f"the leg passes CLI flags it does not need: {cli_flags}"


def test_an_empty_proxy_set_records_nulls_and_a_reason(tmp_path, monkeypatch):
    """No proxy items is not a floor score of zero.

    `doc_hit: 0.0` over zero items and a floor that stopped returning
    transcripts are the same reading, which is the indistinguishability this
    whole item is about; the leg therefore reports `ran: false` with the reason
    and nulls. The empty case is reachable on a fresh machine (no daily notes
    yet, or a sessions collection that is not deployed), so the branch is not
    hypothetical.
    """
    empty_memory = tmp_path / "memory"
    empty_memory.mkdir()
    empty_sessions = tmp_path / "derived" / "sessions"
    empty_sessions.mkdir(parents=True)
    monkeypatch.setattr(ev, "_vault_recall", _leg_only_recall([], []))

    leg = ev.attach_episodic_floor_leg(
        {}, memory_dir=empty_memory, sessions_root=empty_sessions,
        k=10)[ev.EPISODIC_FLOOR_KEY]
    assert leg["ran"] is False
    assert leg["n_items"] == 0
    assert leg["doc_hit"] is None and leg["mrr"] is None
    assert leg["reason"], "an unrun leg has to say why"


def test_a_missing_sessions_root_refuses_the_leg_with_a_reason(tmp_path, monkeypatch):
    """A collection that is not deployed cannot produce a confident zero either."""
    memory = tmp_path / "memory"
    memory.mkdir()
    monkeypatch.setattr(ev, "_vault_recall", _leg_only_recall([], []))
    leg = ev.attach_episodic_floor_leg(
        {}, memory_dir=memory, sessions_root=tmp_path / "nowhere",
        k=10)[ev.EPISODIC_FLOOR_KEY]
    assert leg["ran"] is False and leg["doc_hit"] is None
    assert "sessions" in leg["reason"]


# ── clause 1 end-to-end: the field is in the artifact main writes ────────────

class _OneQueryStore:
    """A corpus-only stand-in for `app.kg_store.store()`.

    Same shape `tests/test_counterfactual_eval.py::_CorpusOnlyStore` uses: the
    nightly refuses to score an empty corpus, and this answers the provenance
    read without opening the knowledge store.
    """

    def stats(self):
        return {"entities_total": 1, "edges_total": 1, "edges_active": 1,
                "aliases": 0, "facts": 1}

    def resolve(self, text):
        return text

    def active_edges(self):
        return []

    def neighbours(self, *a, **k):
        return {}


def test_main_writes_the_floor_leg_field_into_the_baseline(tmp_path, monkeypatch):
    """One real `ev.main()` over a one-query corpus, then read the JSON back.

    The unit nodes above prove the emitter works; this proves the nightly's
    command actually calls it, because a leg nobody invokes would otherwise be
    invisible to every reader that globs `nightly-*.json`. The artifact is read
    off disk, so the assertion is about the file the trend stats read.
    """
    memory, sessions = _proxy_tree(tmp_path)
    index = _fake_index(tmp_path, sessions)
    monkeypatch.setenv(doc_corpus.INDEX_PATH_ENV, str(index))

    gold = "Nightly Retrieval Eval"
    monkeypatch.setattr(ev, "_recall_seeds", lambda q, k: [gold])
    monkeypatch.setattr(ev, "_semantic_seed_k", lambda: 1)
    monkeypatch.setattr(ev, "_vault_recall", lambda params, **kw: {
        "entities": [],
        "facts": [{"entity": gold, "text": "t", "source": "memory/x.md"}],
        "documents": [{"path": "memory/2026-09-26.md"},
                      {"path": f"sessions/{_DATE}/{_HIT}"}],
        "graph_expanded_facts": [], "graph_neighbors_used": [],
        "fact_read_coverage": {"attempted": 1, "read": 1},
        "graph_expansion": {}})
    monkeypatch.setattr(ev, "store", _OneQueryStore)
    monkeypatch.setattr(ev, "VAULT_ROOT", tmp_path)
    out = tmp_path / "baselines"
    monkeypatch.setattr(ev, "EVAL_BASELINES_DIR", out)
    monkeypatch.setattr(ev, "LLOYD_CODE_ROOT", tmp_path)

    qf = tmp_path / "q.yaml"
    qf.write_text(
        "queries:\n"
        "  - id: floor-leg-probe\n"
        "    query: what did the nightly eval find\n"
        "    category: entity\n"
        f"    expect_entities: [{gold}]\n"
        "    expect_docs: [memory/2026-09-26.md]\n")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--queries", str(qf),
                                     "--label", "floor-leg",
                                     "--allow-empty-corpus"])
    assert ev.main() == 0, "a report-only leg never fails the run it reports on"
    written = sorted(out.glob("floor-leg-*.json"))
    assert written, f"main wrote no artifact into {out}"
    blob = json.loads(written[-1].read_text())

    leg = blob[ev.EPISODIC_FLOOR_KEY]
    assert leg["n_items"] == 2, (
        "main resolved the proxy set from the run's own vault memory dir and "
        "the sessions root read out of the index this run scores")
    # Hand-recomputed: 2 proxy items; the stand-in returns
    # [memory/…, sessions/<date>/_HIT] for every query, so item 1 hits at rank 2
    # (doc 1.0, rr 0.5) and item 2, whose label is the OTHER transcript, is a miss
    # (doc 0.0, rr 0.0). doc_hit = 1/2 = 0.5, mrr = 0.5/2 = 0.25 — the MRR is the
    # half of the pair that would notice a floor that returns the transcript last.
    assert leg["doc_hit"] == pytest.approx(0.5)
    assert leg["mrr"] == pytest.approx(0.25)
    assert leg["item_source"] == "episodic_items"
    # The floor's own state, in the artifact beside the number: a night that
    # scored the leg with the floor forced off is readable as that night.
    assert leg["episodic_floors_on"] is False, (
        "conftest forces LLOYD_RECALL_EPISODIC=0, so a test-run artifact must "
        "say the floor was off rather than imply today's production state")
    # The doc leg still scores ONLY the gold corpus: 1 query, the fixture's.
    assert blob["summary"]["overall"]["n_queries"] == 1
