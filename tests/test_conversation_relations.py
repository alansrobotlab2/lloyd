"""Stage 1/2 extraction and landing of scripts/memory/conversation_relations.py.

Backlog #420: the extractor only recognised tool names that left the pool in
the harness cutover (`vault_read`, `file_grep`, …) and only read the `path`
param key, while live trajectories record `Read`/`Write`/`Edit` with the path
under `file_path`. It extracted 7 pairs across 19 trajectory files, autonomy
task #51 exited 0 on 4 MB of unreadable input, and the watermark advanced past
every day it had been blind to. Each test below names the #420 clause it pins.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import retrieval  # noqa: E402
from app import kg_store  # noqa: E402

SCRIPT = ROOT / "scripts" / "memory" / "conversation_relations.py"
TRAJECTORY_DIR = Path.home() / "lloyd" / "_pipeline" / "trajectories"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cr(request):
    """A fresh module instance per test — these tests monkeypatch module
    globals (VAULT, TRAJECTORY_DIR, PROPOSALS_FILE) and must not leak them —
    unloaded again on teardown so sys.modules does not accumulate one copy per
    test."""
    name = f"cr_{request.node.name.replace('[', '_').replace(']', '_')}"
    mod = _load(name)
    yield mod
    sys.modules.pop(name, None)


@pytest.fixture
def vault(cr, tmp_path, monkeypatch):
    """A vault root in tmp_path with three docs and one skill, so
    normalize_vault_path's existence check is hermetic. Fixture params use the
    `~/obsidian/…` spelling the trajectories really carry."""
    root = tmp_path / "vault"
    for rel in ("knowledge/a.md", "knowledge/b.md", "knowledge/c.md",
                "skills/code-review/SKILL.md"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# doc\n")
    monkeypatch.setattr(cr, "VAULT", root)
    return root


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Process-default edge store in a temp file (same setup as
    test_edge_readers.py: entity resolution must not reach the real corpus)."""
    facts = tmp_path / "facts"
    facts.mkdir()
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts)
    import agent_mcp._shared as shared
    monkeypatch.setattr(shared, "FACTS_ROOT", facts)
    shared._invalidate_entity_dirs_cache()
    s = kg_store.configure(tmp_path / "kg.sqlite")
    yield s
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()


def _tool(name, seq, params_summary=None, result_summary=""):
    return {"name": name, "sequence": seq,
            "params_summary": params_summary or {},
            "result_summary": result_summary}


def _entry(tools, session_key="20260912_010101_abc", date="2026-09-12"):
    return {"session_key": session_key, "timestamp": f"{date}T01:01:01Z",
            "tools": tools}


def _write_traj(dir_path: Path, date: str, entries):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{date}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


# ── clause 1: the SDK tool surface is recognised by its file_path param ───────

def test_sdk_read_and_edit_by_file_path_yield_one_pair(cr, vault, tmp_path):
    """Both halves at once: the SDK names *and* the `file_path` key. Fixing the
    name list alone yields 0 pairs, which is why this is one test."""
    entry = _entry([
        _tool("Read", 3, {"file_path": "~/obsidian/knowledge/a.md"}),
        _tool("Edit", 4, {"file_path": "~/obsidian/knowledge/b.md"}),
    ])

    docs = cr.extract_vault_docs_from_trajectory(entry)
    assert [d["path"] for d in docs] == ["knowledge/a.md", "knowledge/b.md"]

    traj = tmp_path / "trajectories"
    _write_traj(traj, "2026-09-12", [entry])
    pairs = cr.extract_co_access_pairs(traj, since_date=None)
    assert len(pairs) == 1
    assert sorted([pairs[0]["doc_a"], pairs[0]["doc_b"]]) == [
        "knowledge/a.md", "knowledge/b.md"]
    assert len(cr.aggregate_pairs(pairs)) == 1


def test_write_tool_path_key_and_no_self_pair(cr, vault, tmp_path):
    """`Write` counts too. Every access is a document entry — two accesses to
    one doc contribute two pair-rows with it (co_access_count then records 2),
    but a doc is never paired with itself."""
    entry = _entry([
        _tool("Write", 1, {"file_path": "~/obsidian/knowledge/a.md"}),
        _tool("Read", 2, {"file_path": "~/obsidian/knowledge/a.md"}),
        _tool("Edit", 3, {"file_path": "~/obsidian/knowledge/c.md"}),
    ])
    assert [d["path"] for d in cr.extract_vault_docs_from_trajectory(entry)] == [
        "knowledge/a.md", "knowledge/a.md", "knowledge/c.md"]

    traj = tmp_path / "trajectories"
    _write_traj(traj, "2026-09-12", [entry])
    pairs = cr.extract_co_access_pairs(traj, since_date=None)
    assert len(pairs) == 2  # a<->c once per a access
    assert all(p["doc_a"] != p["doc_b"] for p in pairs)
    # The count is aggregated, not per-pair: both a<->c rows land in one group.
    groups = list(cr.aggregate_pairs(pairs).values())
    assert len(groups) == 1 and groups[0]["co_access_count"] == 2


# ── clause 2: the legacy vocabulary and the `path` key still work ────────────

def test_legacy_names_and_skill_reads_still_yield_their_docs(cr, vault):
    """Widening the vocabulary must not regress the names that still occur
    (vault_read / vault_write / skills_read)."""
    entry = _entry([
        _tool("vault_read", 1, {"path": "~/obsidian/knowledge/a.md"}),
        _tool("vault_write", 2, {"path": "~/obsidian/knowledge/b.md"}),
        _tool("skills_read", 3, {"name": "code-review"}),
    ])
    docs = cr.extract_vault_docs_from_trajectory(entry)
    assert [d["path"] for d in docs] == [
        "knowledge/a.md", "knowledge/b.md", "skills/code-review/SKILL.md"]


def test_path_key_wins_when_both_keys_are_present(cr, vault):
    """A call carrying both `path` and `file_path` contributes one doc, not two."""
    entry = _entry([
        _tool("Read", 1, {"path": "~/obsidian/knowledge/a.md",
                          "file_path": "~/obsidian/knowledge/a.md"}),
        _tool("Edit", 2, {"file_path": "~/obsidian/knowledge/b.md"}),
    ])
    assert [d["path"] for d in cr.extract_vault_docs_from_trajectory(entry)] == [
        "knowledge/a.md", "knowledge/b.md"]


def test_non_vault_and_missing_paths_are_still_rejected(cr, vault):
    """The widened name set is only safe because normalize_vault_path still
    screens: repo paths, non-markdown, and files that do not exist stay out."""
    entry = _entry([
        _tool("Read", 1, {"file_path": "~/lloyd/app/kg_store.py"}),
        _tool("Edit", 2, {"file_path": "~/obsidian/knowledge/nope.md"}),
        _tool("Write", 3, {"file_path": "~/obsidian/knowledge/notes.txt"}),
        _tool("Read", 4, {"file_path": "[truncated: 2048 chars]"}),
    ])
    assert cr.extract_vault_docs_from_trajectory(entry) == []


def test_search_result_branch_is_inert_and_stays_so(cr, vault):
    """Defect record, not a behaviour to preserve (#420 finding).

    The `vault_search`/`vault_recall`/`file_grep` branch parses bare
    `segment/….md` strings out of result text, and normalize_vault_path only
    accepts a path carrying a vault prefix — so that branch yields nothing
    whatever its membership, and adding `Grep` to it would be a no-op. A future
    round that wants search hits must fix the normalization in the same change;
    this test fails the moment the two halves stop agreeing, which is the point.
    """
    assert cr.normalize_vault_path("~/obsidian/knowledge/a.md") == "knowledge/a.md"
    assert cr.normalize_vault_path("knowledge/a.md") is None
    entry = _entry([
        _tool("file_grep", 1, {"pattern": "watermark"},
              "knowledge/a.md:12:- the watermark advances\n"
              "knowledge/b.md:44:blind extraction\n"),
    ])
    assert cr.extract_vault_docs_from_trajectory(entry) == []


# ── clause 3: the live corpus now yields a material signal ───────────────────

def test_extractor_scales_on_a_synthetic_corpus(cr, vault, tmp_path):
    """The clause-3 floor, pinned hermetically so it always runs.

    4 files x 15 SDK-named accesses: a vocabulary that missed `Read`/`Edit`
    yields 0 pairs here, and the thresholds are the ones #420 was graded on.
    """
    traj = tmp_path / "trajectories"
    for day in range(1, 5):
        entries = []
        for col in range(15):
            rel = f"knowledge/d{day}_{col}.md"
            doc = vault / rel
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text("# doc\n")
        tools = [_tool("Read", seq,
                       {"file_path": f"~/obsidian/knowledge/d{day}_{seq}.md"})
                 for seq in range(15)]
        entries.append(_entry(tools, session_key=f"2026090{day}_111111_x"))
        _write_traj(traj, f"2026-09-0{day}", entries)

    pairs = cr.extract_co_access_pairs(traj, since_date=None)
    assert len(pairs) >= 200, f"only {len(pairs)} raw pairs"
    assert len(cr.aggregate_pairs(pairs)) >= 100


@pytest.mark.live_vault
@pytest.mark.skipif(not TRAJECTORY_DIR.is_dir(), reason="trajectory corpus not present")
def test_live_trajectories_yield_material_co_access_signal(cr):
    """The acceptance measurement on the real corpus: 7 pairs / 1 aggregate
    before #420, 1144 / 256 after. Reads the live trajectories and the live
    vault (path existence is part of normalize_vault_path), so it is marked
    `live_vault` like the other tests that assert over files no round controls.
    """
    pairs = cr.extract_co_access_pairs(TRAJECTORY_DIR, since_date=None)
    assert len(pairs) >= 200, f"only {len(pairs)} raw pairs from {TRAJECTORY_DIR}"
    assert len(cr.aggregate_pairs(pairs)) >= 100


# ── clause 4: the watermark only advances past files that yielded docs ───────

def test_watermark_advances_only_past_productive_files(cr, vault, tmp_path, monkeypatch):
    """The newest file on disk yields no docs, so the watermark must stop at
    2026-09-11: advancing it unconditionally burned every blind day (#420)."""
    traj = tmp_path / "trajectories"
    _write_traj(traj, "2026-09-11", [
        _entry([_tool("Read", 1, {"file_path": "~/obsidian/knowledge/a.md"}),
                _tool("Edit", 2, {"file_path": "~/obsidian/knowledge/b.md"})],
               session_key="20260911_010101_aaa", date="2026-09-11")])
    _write_traj(traj, "2026-09-12", [
        _entry([_tool("Bash", 1, {"command": "ls"})],
               session_key="20260912_010101_bbb", date="2026-09-12")])
    props = tmp_path / "proposals.json"
    monkeypatch.setattr(cr, "TRAJECTORY_DIR", traj)
    monkeypatch.setattr(cr, "PROPOSALS_FILE", props)

    cr.cmd_incremental()
    wm = json.loads(props.read_text())["watermark"]
    assert wm["last_trajectory_date"] == "2026-09-11", wm

    # …and it is not stuck there: a productive day moves it again.
    _write_traj(traj, "2026-09-13", [
        _entry([_tool("Read", 1, {"file_path": "~/obsidian/knowledge/b.md"}),
                _tool("Edit", 2, {"file_path": "~/obsidian/knowledge/c.md"})],
               session_key="20260913_010101_ccc", date="2026-09-13")])
    cr.cmd_incremental()
    wm = json.loads(props.read_text())["watermark"]
    assert wm["last_trajectory_date"] == "2026-09-13", wm


def test_watermark_counts_pairs_under_an_honest_key(cr, vault, tmp_path, monkeypatch):
    """The key held a pair count under the name `sessions_processed`."""
    traj = tmp_path / "trajectories"
    _write_traj(traj, "2026-09-11", [
        _entry([_tool("Read", 1, {"file_path": "~/obsidian/knowledge/a.md"}),
                _tool("Edit", 2, {"file_path": "~/obsidian/knowledge/b.md"}),
                _tool("Write", 3, {"file_path": "~/obsidian/knowledge/c.md"})],
               date="2026-09-11")])
    props = tmp_path / "proposals.json"
    monkeypatch.setattr(cr, "TRAJECTORY_DIR", traj)
    monkeypatch.setattr(cr, "PROPOSALS_FILE", props)

    cr.cmd_incremental()
    wm = json.loads(props.read_text())["watermark"]
    assert "sessions_processed" not in wm
    assert wm["pairs_processed"] == 3  # a-b, a-c, b-c from one row


# ── clauses 5 & 6: landing persists the classifier's type and provenance ─────

def test_approved_proposal_lands_with_the_classifiers_own_type(store, cr):
    """Proposals store the Stage-2 result under `type`; landing read
    `relation_type`, which no proposal has, so every edge became co_accessed."""
    props = [{"source": "knowledge/a.md", "target": "knowledge/b.md",
              "status": "approved", "type": "supersedes", "confidence": 0.9,
              "reason": "a replaces b",
              "evidence": {"sessions": ["20260910_010203_abc"]}}]
    assert cr.land_approved_edges(props) == 1
    assert store.edges.find_active("knowledge/a.md", "knowledge/b.md", "supersedes")
    assert store.edges.find_active("knowledge/a.md", "knowledge/b.md", "co_accessed") is None


def test_landed_edges_carry_a_non_null_source_doc_from_evidence_sessions(store, cr):
    props = [{"source": "knowledge/a.md", "target": "knowledge/b.md",
              "status": "approved", "type": "related-to", "confidence": 0.9,
              "evidence": {"sessions": ["20260910_010203_abc",
                                        "autonomy_51_20260911"]}}]
    assert cr.land_approved_edges(props) == 1
    rows = list(store.conn.execute(
        "select id, type, source_doc from edges where origin='conversation'"))
    assert len(rows) == 1
    assert rows[0]["source_doc"], "source_doc must not be NULL"
    assert "20260910_010203_abc" in rows[0]["source_doc"]
    assert "2026-09-10" in rows[0]["source_doc"]


def test_legacy_evidence_trajectory_key_still_lands(store, cr):
    """The pre-existing landing path pinned by test_edge_readers.py stays wired."""
    props = [{"source": "a.md", "target": "b.md", "status": "approved",
              "confidence": 0.9, "evidence_trajectory": "traj/2026-09-01.jsonl"}]
    assert cr.land_approved_edges(props) == 1
    edge = store.edges.find_active("a.md", "b.md", "co_accessed")
    assert edge["source_doc"] == "traj/2026-09-01.jsonl"


def test_unattributable_approved_proposal_does_not_land(store, cr):
    """Nothing in the proposal says where the pair came from, so landing it
    would recreate the NULL-source_doc rows; it stays approved instead."""
    props = [{"source": "a.md", "target": "b.md", "status": "approved",
              "confidence": 0.9}]
    assert cr.land_approved_edges(props) == 0
    assert list(store.conn.execute(
        "select id from edges where origin='conversation'")) == []
    assert props[0].get("edge_id") is None


# ── Stage 2 endpoint/model come from config, not the constant ────────────────

def _task_file(dir_path: Path, model: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    f = dir_path / "51-conversation-relation-linking.md"
    f.write_text("---\nname: Conversation Relation Linking\n"
                 "skill_name: conversation-relation-linking\n"
                 f"model: {model}\n---\n\n# body\n", encoding="utf-8")
    return f


def _config_endpoint(alias: str) -> str:
    """The endpoint config.yaml says for an alias — the contract under test,
    so the assertions below track the file rather than a port someone may move."""
    from app.config import MODEL_CONFIGS
    cfg = MODEL_CONFIGS.get(alias) or {}
    base = (cfg.get("base_url")
            or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL") or "")
    return base.rstrip("/") + "/v1/chat/completions"


def test_endpoint_follows_the_task_model_declaration(cr, tmp_path, monkeypatch):
    """#51's frontmatter is the decision; the constant at :41 ignored it."""
    adir = tmp_path / "autonomy"
    _task_file(adir, "secondary")
    monkeypatch.setattr(cr, "AUTONOMY_DIR", adir)
    endpoint, model = cr.resolve_llm_target()
    assert model == "secondary"
    assert endpoint == _config_endpoint("secondary")
    assert endpoint != cr.LLM_ENDPOINT          # not the hardcoded constant
    assert _config_endpoint("secondary") != _config_endpoint("primary")

    _task_file(adir, "primary")
    assert cr.resolve_llm_target()[0] == _config_endpoint("primary")


def test_disabled_secondary_alias_moves_with_the_switch(cr, tmp_path, monkeypatch):
    """`secondary_enabled: false` rewrites the alias to primary in app.config;
    Stage 2 must follow the rewrite rather than post to a slot that is off.
    The constant could not have honoured either state."""
    from app import config as app_config
    adir = tmp_path / "autonomy"
    _task_file(adir, "secondary")
    monkeypatch.setattr(cr, "AUTONOMY_DIR", adir)
    monkeypatch.setitem(app_config.CONFIG, "secondary_enabled", False)
    monkeypatch.setattr(app_config, "_ALIAS_REWRITES_LOGGED", set(), raising=True)
    endpoint, model = cr.resolve_llm_target()
    assert (endpoint, model) == (_config_endpoint("primary"), "primary")


def test_endpoint_falls_back_to_the_constant_without_a_task_file(cr, tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "AUTONOMY_DIR", tmp_path / "missing")
    assert cr.resolve_llm_target() == (cr.LLM_ENDPOINT, cr.LLM_MODEL)


def test_stage2_posts_to_the_resolved_endpoint(cr, tmp_path, monkeypatch):
    """The HTTP seam: a real classify call must go where config says, not to
    the port the constant carried."""
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(
                {"type": "supersedes", "reason": "r", "confidence": 0.9})}}]}).encode()

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return FakeResp()

    adir = tmp_path / "autonomy"
    _task_file(adir, "secondary")
    monkeypatch.setattr(cr, "AUTONOMY_DIR", adir)
    # urllib.request is process-global, so the fake is installed only around the
    # call itself and the assertions run after the context has restored it.
    with monkeypatch.context() as ctx:
        ctx.setattr(cr.urllib.request, "urlopen", fake_urlopen)
        out = cr.classify_relationship("knowledge/a.md", "knowledge/b.md", "ctx")

    assert captured["url"] == _config_endpoint("secondary")
    assert captured["url"] != cr.LLM_ENDPOINT
    assert captured["body"]["model"] == "secondary"
    assert out["type"] == "supersedes"
