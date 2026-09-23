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

from tests._live_data import require_live_data, require_live_volume
from app.paths import production_data_root  # noqa: E402

SCRIPT = ROOT / "scripts" / "memory" / "conversation_relations.py"
TRAJECTORY_DIR = production_data_root() / "_pipeline" / "trajectories"


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

    6 files x 18 SDK-named accesses: a vocabulary that missed `Read`/`Edit`
    yields 0 pairs here. The assertions stay at the floors #420 was graded on
    (200 / 100) while the corpus yields 918 / 198, so losing one whole day's
    worth of pairs is a red for a reason and not a red because the twin was
    built 8 aggregates above its own bar (round SM_20260912_092119 advisory).
    """
    traj = tmp_path / "trajectories"
    for day in range(1, 7):
        for col in range(18):
            rel = f"knowledge/d{day}_{col}.md"
            doc = vault / rel
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text("# doc\n")
        tools = [_tool("Read", seq,
                       {"file_path": f"~/obsidian/knowledge/d{day}_{seq}.md"})
                 for seq in range(18)]
        _write_traj(traj, f"2026-09-0{day}",
                    [_entry(tools, session_key=f"2026090{day}_111111_x")])

    pairs = cr.extract_co_access_pairs(traj, since_date=None)
    assert len(pairs) >= 200, f"only {len(pairs)} raw pairs"
    assert len(cr.aggregate_pairs(pairs)) >= 100


@pytest.mark.live_vault
def test_live_trajectories_yield_material_co_access_signal(cr):
    """The acceptance measurement on the real corpus: 7 pairs / 1 aggregate
    before #420, 1144 / 256 after. Reads the live trajectories and the live
    vault (path existence is part of normalize_vault_path), so it is marked
    `live_vault` like the other tests that assert over files no round controls.

    Not `skipif`-guarded: a corpus that is absent or moved must be a failure,
    never a silent pass — the same reason #551's vault scan fails loudly on an
    empty tree. The gate deselects this by marker, not by absence.
    """
    # "Absent or moved must be a failure" was written for a corpus that could be
    # moved back. The 2026-09-22 deletion took _pipeline/trajectories and the
    # 09-11..09-22 mining window with it, and no run regenerates mined trajectories
    # from sessions that are also gone. Absence is now a fact about the machine.
    # A present corpus below the floor still names both numbers rather than passing.
    require_live_data(TRAJECTORY_DIR, "the mined trajectory corpus")
    files = sorted(TRAJECTORY_DIR.glob("*.jsonl"))
    require_live_volume(files, 5, TRAJECTORY_DIR, "the mined trajectory corpus")
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


# ── #773 clauses 1-4: an auto-accepted edge says it was never reviewed ───────

def _aged_proposal(cr, **over):
    """A Stage-2-classified pending proposal that clears the 48h age gate, so
    `auto_approve_strong` is decided by its confidence alone."""
    p = {"source": "knowledge/a.md", "target": "knowledge/b.md",
         "status": "pending", "type": "related-to", "confidence": 0.95,
         "reason": "read together in one session",
         "evidence": {"sessions": ["20260910_010203_abc"], "aggregate_weight": 1.0},
         "proposed_at": (cr.datetime.now(cr.timezone.utc)
                         - cr.timedelta(days=3)).isoformat()}
    p.update(over)
    return p


def _conversation_edge(store):
    edge = store.edges.find_active("knowledge/a.md", "knowledge/b.md", "related-to")
    assert edge, "the approved proposal did not land a related-to edge"
    return edge


def test_auto_approved_edge_says_its_acceptance_was_automatic_and_unreviewed(store, cr):
    """Clauses 1 & 2. The gate that admitted the edge is the classifier's own
    score against a threshold; nothing reads a proposal during the 48 h it waits,
    so the landed row has to say so or it reads downstream like an accepted one.
    The marker round-trips through both the row dict and the sqlite `extra`
    column — `app/kg_store.py` persists non-column keys there, which is the
    whole reason no migration is needed."""
    weak = _aged_proposal(cr, target="knowledge/c.md", confidence=0.5)
    props = [_aged_proposal(cr), weak]
    assert cr.auto_approve_strong(props) == 1
    assert props[0]["accepted_by"] == "auto", props[0]
    assert "auto_approved@0.85" in props[0]["auto_acceptance"], props[0]
    # The proposal it did not flip keeps both its status and its emptiness.
    assert weak["status"] == "pending" and "accepted_by" not in weak

    assert cr.land_approved_edges(props) == 1
    edge = _conversation_edge(store)
    assert edge["accepted_by"] == "auto", edge
    assert edge["auto_acceptance"] == "auto_approved@0.85, unreviewed", edge

    row = store.conn.execute(
        "select extra from edges where origin='conversation'").fetchone()
    assert row["extra"], "marker never reached the extra column"
    assert json.loads(row["extra"])["accepted_by"] == "auto", row["extra"]
    assert json.loads(row["extra"])["auto_acceptance"] == (
        "auto_approved@0.85, unreviewed"), row["extra"]


def test_the_marker_names_the_threshold_that_admitted_this_edge(store, cr):
    """Clause 2, the derivation half: the threshold in the marker is the one
    passed to this call, so an edge admits the gate that actually let it
    through rather than quoting a literal that may no longer be the one in use."""
    props = [_aged_proposal(cr)]
    assert cr.auto_approve_strong(props, threshold=0.92) == 1
    assert cr.land_approved_edges(props) == 1
    marker = _conversation_edge(store)["auto_acceptance"]
    assert marker == "auto_approved@0.92, unreviewed", marker
    assert "0.85" not in marker, marker


def test_proposal_approved_without_the_mark_lands_unmarked(store, cr):
    """Clause 3, the discriminating half: a proposal already `status: approved`
    that carries no auto mark lands with no marker at all. A field stamped on
    every conversation edge distinguishes nothing, which is the difference
    between this fix and a comment."""
    props = [_aged_proposal(cr, status="approved")]
    assert cr.land_approved_edges(props) == 1
    edge = _conversation_edge(store)
    assert "accepted_by" not in edge, edge
    assert "auto_acceptance" not in edge, edge
    row = store.conn.execute(
        "select extra from edges where origin='conversation'").fetchone()
    assert row["extra"] is None, f"unmarked edge carries extra: {row['extra']}"


def test_the_edge_marker_is_derived_from_the_proposal_mark(store, cr):
    """Clause 3's derivation half: the edge marker comes from the mark on the
    proposal, not from re-running the gate at landing time. A proposal marked
    `accepted_by: auto` with no gate text of its own is recorded against the
    shipped default rather than left nameless — an edge whose row cannot say
    which gate admitted it is the defect this item is about."""
    props = [_aged_proposal(cr, status="approved", accepted_by=cr.AUTO_ACCEPTED_BY)]
    assert "auto_acceptance" not in props[0]
    assert cr.land_approved_edges(props) == 1
    edge = _conversation_edge(store)
    assert edge["accepted_by"] == "auto", edge
    assert edge["auto_acceptance"] == cr.auto_acceptance_marker(
        cr.DEFAULT_AUTO_APPROVE_THRESHOLD), edge
    assert edge["auto_acceptance"] == "auto_approved@0.85, unreviewed", edge


def test_the_field_is_the_discrimination_not_a_re_score(store, cr):
    """Clause 4. The store has no human-verified band to sit below (0 rows at
    `origin='manual'`; EXTRACTED edges average 0.917), and
    `app/routers/entities.py` filters the entity graph on `min_confidence`, so
    capping an auto-accepted edge below 0.85 would rank it under
    machine-extracted edges and hide it from anyone who raised the slider. The
    score is copied through untouched; the marker is what separates them."""
    props = [_aged_proposal(cr, confidence=0.95)]
    assert cr.auto_approve_strong(props) == 1
    assert cr.land_approved_edges(props) == 1
    edge = _conversation_edge(store)
    assert edge["confidence"] == 0.95, edge
    assert edge["provenance"] == "INFERRED", edge
    assert edge["accepted_by"] == "auto", edge


# ── Stage 2 endpoint/model come from config, not the constant ────────────────

def _task_file(dir_path: Path, model: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    f = dir_path / "51-conversation-relation-linking.md"
    f.write_text("---\nname: Conversation Relation Linking\n"
                 "skill_name: conversation-relation-linking\n"
                 f"model: {model}\n---\n\n# body\n", encoding="utf-8")
    return f


def _config_endpoint(alias: str) -> str:
    """The endpoint config.yaml declares for an alias, read straight from the
    parsed config — the contract under test, so these assertions track the file
    rather than a port someone may move.

    Deliberately *no* fallback chain here. `resolve_llm_target()` derives its URL
    from `base_url` with an `env.ANTHROPIC_BASE_URL` fallback; repeating that
    precedence in the test means the two can drift together and stay green while
    both are wrong (round SM_20260912_092119 advisory). `base_url` is the key
    both model slots actually populate; if one stops being, this assert names it
    instead of quietly resolving through a different key.
    """
    from app.config import MODEL_CONFIGS
    base = MODEL_CONFIGS[alias]["base_url"]
    assert base, (
        f"MODEL_CONFIGS[{alias!r}].base_url is empty; the endpoint assertions "
        "below would be vacuous")
    return base.rstrip("/") + "/v1/chat/completions"


def test_endpoint_follows_the_task_model_declaration(cr, tmp_path, monkeypatch):
    """#51's frontmatter is the decision; the constant at :41 ignored it."""
    # Pin the slot ON. This test is about alias resolution WHILE the secondary
    # is enabled, and `resolve_model_alias` rewrites secondary -> primary when
    # it is not — so inheriting the live `secondary_enabled` makes the test's
    # subject depend on production config. It read true for as long as the slot
    # was in use and both assertions below went red the day it was switched off
    # (2026-09-20), with nothing about this code changed. Its sibling
    # `test_disabled_secondary_alias_moves_with_the_switch` already pins the
    # false case; this is the other half of the same contract.
    from app import config as app_config
    monkeypatch.setitem(app_config.CONFIG, "secondary_enabled", True)
    monkeypatch.setattr(app_config, "_ALIAS_REWRITES_LOGGED", set(), raising=True)
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

    # Pin the slot ON. This test is about alias resolution WHILE the secondary
    # is enabled, and `resolve_model_alias` rewrites secondary -> primary when
    # it is not — so inheriting the live `secondary_enabled` makes the test's
    # subject depend on production config. It read true for as long as the slot
    # was in use and both assertions below went red the day it was switched off
    # (2026-09-20), with nothing about this code changed. Its sibling
    # `test_disabled_secondary_alias_moves_with_the_switch` already pins the
    # false case; this is the other half of the same contract.
    from app import config as app_config
    monkeypatch.setitem(app_config.CONFIG, "secondary_enabled", True)
    monkeypatch.setattr(app_config, "_ALIAS_REWRITES_LOGGED", set(), raising=True)
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


# ── #1039: a skill doc is verifiable by the bare name its session carries ─────
#
# Stage 1 normalises a `skills_read` to the 3-part `skills/<name>/SKILL.md`, but
# a session file records the call argument `{"name": "<name>"}` and the skill's
# own text — never that path. The evidence matcher compared full paths only, so
# a pair of two skill docs could be proposed by Stage 1 and then never verified
# against the session it came from: `cmd_classify` skips a candidate whose
# window is empty and never asks the model. #420 fixed the mirror-image defect
# on the Stage 1 side and left this one.
#
# Measured at triage over `_pipeline/conversation-relation-proposals.json`
# (414 proposals): 83 skill-skill pairs, 81 already find a window because the
# path text reaches the session some other way (injected skill context, a
# `skills_search` result, an absolute-path `Read`). The 2 that returned None are
# reproduced below with their real doc paths and session key:
# `skills/codebase-inspection` + `skills/requesting-code-review` in session
# `20260905_151355_iv5174`, and `skills/backlog-premise-triage` +
# `skills/selfmod-change-own-code` in an e2e session.


def _session_file(dir_path: Path, *messages, key: str = "20260922_010101_sess") -> Path:
    """Write a session file in the shape the session logger emits.

    The tool-call form is copied from
    `~/lloyd/sessions/20260905_151355_iv5174.json` — one of the two sessions the
    matcher fails on today — where the assistant message carries a `tool_calls`
    array whose `arguments` is a JSON *string*. That nesting is the whole point:
    it is where the bare skill name lives and where no path ever appears.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    f = dir_path / f"{key}.json"
    f.write_text(json.dumps({"session_id": key, "messages": list(messages)}),
                 encoding="utf-8")
    return f


def _asked(text: str) -> dict:
    return {"id": "u1", "role": "user",
            "content": [{"type": "text", "text": text}]}


def _answered(text: str) -> dict:
    return {"id": "a1", "role": "assistant",
            "content": [{"type": "text", "text": text}]}


def _tool_says(text: str) -> dict:
    return {"id": "t1", "role": "tool",
            "content": [{"type": "text", "text": text}]}


def _skills_read(skill: str, call_id: str = "call_1") -> dict:
    """The assistant message a `skills_read` call leaves behind."""
    return {"id": call_id, "role": "assistant", "content": [],
            "tool_calls": [{"id": call_id, "call_id": call_id, "type": "function",
                            "function": {"name": "skills_read",
                                         "arguments": json.dumps({"name": skill})}}]}


def _needle_test_proposal(source: str, target: str, session_key: str) -> dict:
    """A Stage-2 candidate: co-access evidence, above the LLM threshold, pending."""
    return {"source": source, "target": target, "type": None,
            "confidence": 0.0, "reason": "",
            "status": "pending", "classification_source": "co-access",
            "signal_strength": "weak",
            "evidence": {"sessions": [session_key], "aggregate_weight": 0.9}}


def _capturing_llm(monkeypatch, cr) -> dict:
    """Install a fake `/chat/completions` that records every request.

    Returns the capture dict; `calls` is what a test asserts on to prove whether
    Stage 2 crossed the HTTP seam at all.
    """
    captured = {"calls": []}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(
                {"type": "related-to", "reason": "read together",
                 "confidence": 0.8})}}]}).encode()

    def fake_urlopen(req, timeout=None):
        captured["calls"].append({"url": req.full_url,
                                  "body": json.loads(req.data.decode())})
        return FakeResp()

    # urllib.request is process-global, so the fake goes on only for the call.
    monkeypatch.setattr(cr.urllib.request, "urlopen", fake_urlopen)
    # The endpoint is not the subject here, and resolving it reads live config
    # and the `secondary_enabled` switch (see the endpoint tests above), so the
    # seam is pinned with a fixed target instead.
    monkeypatch.setattr(cr, "resolve_llm_target",
                        lambda: ("http://127.0.0.1:1/v1/chat/completions", "fixture-model"))
    return captured


def test_a_skill_pair_verifies_from_the_bare_name_in_the_call_args(cr, tmp_path):
    """Clause 1: no path string anywhere, so the argument `{"name": "b"}` is the
    only trace of either skill. Pre-fix the matcher found no hit message and
    returned None, and Stage 2 dropped the pair unread.

    With single-letter skill names the derived needle `a` also matches ordinary
    text (`assistant`, `name`), so on its own this assertion would pass under any
    matcher that matched generic prose; the needle derivation below is what
    removes that reading, and `test_the_bare_name_is_the_documents_own_skill_name`
    — the real pair from the triage measurement, whose names cannot collide by
    accident — is what pins it behaviourally.
    """
    session = _session_file(
        tmp_path,
        _asked("compare the two skills I just opened"),
        _skills_read("b"),
        _answered("one of them is about review"),
    )
    assert cr.evidence_needles("skills/b/SKILL.md") == ["skills/b/skill.md", "b"], (
        "the needle list is what this matcher searches; pin it before the window")
    window = cr.extract_conversation_context(
        session, "skills/a/SKILL.md", "skills/b/SKILL.md")
    assert window is not None, (
        "no evidence window: the matcher still knows only the 3-part path, "
        "which this session never contains")
    assert "compare the two skills I just opened" in window, window


def test_the_bare_name_is_the_documents_own_skill_name(cr, tmp_path):
    """The 2 real proposals the fix rescues, and the negative that keeps the
    rescue honest: a session that read a *different* skill is not evidence.

    The names here are the ones in the live proposals file, so no needle can
    match by accident the way a single-letter name would.
    """
    docs = ("skills/codebase-inspection/SKILL.md",
            "skills/requesting-code-review/SKILL.md")

    hit = _session_file(
        tmp_path,
        _asked("how should I review this diff?"),
        _skills_read("requesting-code-review"),
        _answered("read one skill"),
        key="20260905_151355_iv5174",
    )
    assert cr.extract_conversation_context(hit, *docs) is not None

    miss = _session_file(
        tmp_path,
        _asked("how should I review this diff?"),
        _skills_read("youtube-transcript"),
        _answered("read one skill"),
        key="unrelated",
    )
    assert cr.extract_conversation_context(miss, *docs) is None, (
        "a session that read an unrelated skill is not co-access evidence for "
        "this pair")


def test_the_pair_still_verifies_from_the_three_part_path_alone(cr, tmp_path):
    """Clause 2: the case the 81 already-verifiable proposals rely on — the
    3-part path arrives inside an absolute path (injected skill context, a
    `skills_search` result, an absolute `Read`) and the bare name is nowhere.

    Behaviourally the two needles cannot be separated for a skill doc: any text
    carrying `skills/<name>/SKILL.md` carries `<name>` as a substring, so no
    session can present the path without also presenting the name. What this test
    can pin, and does, is that the path is still a needle at all — pre-fix it was
    the only one, and a fix that replaced path matching with name matching would
    leave the behavioural half of this test green. The assertion below is
    therefore the load-bearing one: the path is the first entry of
    `evidence_needles`, and the name is an addition to it, not a substitution."""
    session = _session_file(
        tmp_path,
        _asked("which one covers diffs?"),
        _tool_says("loaded /home/alansrobotlab/obsidian/skills/requesting-code-review/SKILL.md"),
        _answered("the review skill"),
    )
    window = cr.extract_conversation_context(
        session,
        "skills/codebase-inspection/SKILL.md",
        "skills/requesting-code-review/SKILL.md")
    assert window is not None
    assert "which one covers diffs?" in window, window
    assert cr.evidence_needles("skills/requesting-code-review/SKILL.md")[0] == \
        "skills/requesting-code-review/skill.md"


def test_a_bare_name_is_derived_only_from_a_three_part_skill_doc(cr, tmp_path):
    """Clause 3: the derivation is what makes skill docs verifiable, so it must
    not be reachable from any other shape of path — otherwise every vault doc
    pair whose basenames appear in prose would start matching.

    The first pair are vault docs that happen to be named after skills, and the
    session's only mention of them is a `skills_read` argument carrying exactly
    those names; the second pair is a 4-part path, which is not a skill doc
    however it ends.
    """
    lookalikes = ("knowledge/codebase-inspection.md",
                  "knowledge/requesting-code-review.md")
    named = _session_file(
        tmp_path,
        _asked("compare them"),
        _skills_read("codebase-inspection"),
        _skills_read("requesting-code-review", call_id="call_2"),
        key="lookalikes",
    )
    assert cr.extract_conversation_context(named, *lookalikes) is None, (
        "a bare name was derived from a non-skill doc, so prose that merely "
        "names a file now counts as accessing it")

    too_deep = ("skills/nested/extra/SKILL.md", "knowledge/z.md")
    deep_session = _session_file(
        tmp_path,
        _asked("compare them"),
        _skills_read("extra"),
        key="too_deep",
    )
    assert cr.extract_conversation_context(deep_session, *too_deep) is None

    # Structural half: the needle list itself, one entry for an ordinary doc and
    # two for a 3-part skill doc, with the path always first.
    assert cr.evidence_needles("knowledge/a.md") == ["knowledge/a.md"]
    assert cr.evidence_needles("skills//SKILL.md") == ["skills//skill.md"], (
        "an empty skill name would add the empty needle, which matches every "
        "message in every session")
    assert cr.evidence_needles("skills/code-review/SKILL.md") == [
        "skills/code-review/skill.md", "code-review"]


def test_no_mention_of_either_skill_still_yields_no_window(cr, tmp_path):
    """Clause 4: the session names a third skill and neither path, so the pair
    stays unread and Stage 2 must skip it without spending a classification
    call. The sibling test below pins that the same skip really does mean no
    HTTP request."""
    session = _session_file(
        tmp_path,
        _asked("what tooling is out there?"),
        _skills_read("songsee"),
        _answered("one skill listed"),
        key="no_pair",
    )
    assert cr.extract_conversation_context(
        session, "skills/arxiv/SKILL.md", "skills/discord/SKILL.md") is None


def test_stage2_skips_an_unverifiable_skill_pair_without_an_llm_call(cr, tmp_path,
                                                                    monkeypatch):
    """Clause 4's other half, across the HTTP seam: the skip is what keeps a
    pair the matcher cannot read from ever reaching the model.

    The spy is what makes `calls == []` mean *the matcher declined*: without it an
    empty capture is equally a pair that never entered the candidate set."""
    calls = _capturing_llm(monkeypatch, cr)
    consulted = []
    real_matcher = cr.extract_conversation_context

    def spy(session_path, doc_a, doc_b, max_chars=4000):
        consulted.append((doc_a, doc_b))
        return real_matcher(session_path, doc_a, doc_b, max_chars)

    monkeypatch.setattr(cr, "extract_conversation_context", spy)
    key = "no_pair_stage2"
    _session_file(tmp_path, _asked("what tooling is out there?"),
                  _skills_read("songsee"), key=key)
    monkeypatch.setattr(cr, "LLOYD_SESSIONS", tmp_path)
    props = tmp_path / "proposals.json"
    props.write_text(json.dumps(
        {"watermark": {}, "stats": {}, "proposals": [
            _needle_test_proposal("skills/arxiv/SKILL.md",
                                  "skills/discord/SKILL.md", key)]}),
        encoding="utf-8")
    monkeypatch.setattr(cr, "PROPOSALS_FILE", props)

    cr.cmd_classify()

    assert consulted == [("skills/arxiv/SKILL.md", "skills/discord/SKILL.md")], (
        "the pair never reached the matcher, so the skip proves nothing about it")
    assert calls["calls"] == [], "Stage 2 classified a pair it could not read"
    assert json.loads(props.read_text())["proposals"][0]["status"] == "pending"


def test_stage2_classifies_a_skill_pair_whose_session_names_it_barely(cr, tmp_path,
                                                                     monkeypatch):
    """The payoff, across the same seam: the real rescued proposal — the
    `codebase-inspection` + `requesting-code-review` pair from session
    `20260905_151355_iv5174`, reproduced with the messages that session really
    holds — now reaches the model instead of being skipped. Before the needle
    derivation this asserted zero calls; the pair was proposed, weighty enough
    to classify, and silently dropped."""
    calls = _capturing_llm(monkeypatch, cr)
    key = "20260905_151355_iv5174"
    _session_file(tmp_path,
                  _asked("how should I review this diff?"),
                  _skills_read("codebase-inspection"),
                  _skills_read("requesting-code-review", call_id="call_2"),
                  _answered("both are relevant"),
                  key=key)
    monkeypatch.setattr(cr, "LLOYD_SESSIONS", tmp_path)
    props = tmp_path / "proposals.json"
    proposal = _needle_test_proposal(
        "skills/codebase-inspection/SKILL.md",
        "skills/requesting-code-review/SKILL.md", key)
    props.write_text(json.dumps({"watermark": {}, "stats": {},
                                 "proposals": [proposal]}), encoding="utf-8")
    monkeypatch.setattr(cr, "PROPOSALS_FILE", props)

    cr.cmd_classify()

    assert len(calls["calls"]) == 1, calls["calls"]
    body = calls["calls"][0]["body"]
    assert body["model"] == "fixture-model"
    assert "how should I review this diff?" in json.dumps(body), (
        "the window that reached the model was not the one around the hit")
    landed = json.loads(props.read_text())["proposals"][0]
    assert landed["classification_source"] == "llm"
    assert landed["type"] == "related-to"


def test_stage2_posts_the_rescued_window_over_a_real_socket(cr, tmp_path, monkeypatch):
    """The same seam with nothing faked on it.

    Every other Stage-2 test here fakes `urlopen` at the call site, so the socket,
    the request line and the *resolved* endpoint were never actually crossed.
    Nothing on the HTTP path is faked below: `resolve_llm_target()` reads `model:`
    from the task file, resolves the alias through app.config, and
    `classify_relationship` posts to whatever it returns. The one thing redirected
    is the config *value* of `base_url`, pointed at a port this test is listening
    on — the resolution code and the client code are the production ones.

    For #1039 the assertion that matters is inside the request body: the window
    that a bare skill name rescued is what a real request carried to a real
    engine-shaped endpoint.
    """
    import http.server
    import threading
    from app import config as app_config

    received = {}

    class Responder(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            received["method"] = self.command
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
            payload = json.dumps({"choices": [{"message": {"content": json.dumps(
                {"type": "related-to", "reason": "read together",
                 "confidence": 0.8})}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass  # the default handler logs every request to stderr

    server = http.server.HTTPServer(("127.0.0.1", 0), Responder)
    serve = threading.Thread(target=server.serve_forever, daemon=True)
    serve.start()
    key = "20260905_151355_iv5174"
    props = tmp_path / "proposals.json"
    try:
        monkeypatch.setattr(app_config, "_ALIAS_REWRITES_LOGGED", set(), raising=True)
        monkeypatch.setitem(app_config.MODEL_CONFIGS["primary"], "base_url",
                            f"http://127.0.0.1:{server.server_address[1]}")
        adir = tmp_path / "autonomy"
        _task_file(adir, "primary")
        monkeypatch.setattr(cr, "AUTONOMY_DIR", adir)

        _session_file(tmp_path,
                      _asked("how should I review this diff?"),
                      _skills_read("codebase-inspection"),
                      _skills_read("requesting-code-review", call_id="call_2"),
                      _answered("both are relevant"), key=key)
        monkeypatch.setattr(cr, "LLOYD_SESSIONS", tmp_path)
        props.write_text(json.dumps(
            {"watermark": {}, "stats": {}, "proposals": [
                _needle_test_proposal("skills/codebase-inspection/SKILL.md",
                                      "skills/requesting-code-review/SKILL.md", key)]}),
            encoding="utf-8")
        monkeypatch.setattr(cr, "PROPOSALS_FILE", props)

        cr.cmd_classify()
    finally:
        server.shutdown()
        server.server_close()
        serve.join(timeout=5)

    assert (received.get("method"), received.get("path")) == ("POST", "/v1/chat/completions"), received
    assert received["body"]["model"] == "primary", (
        "the body should name the alias the task file declared, resolved by "
        "app.config — not the module's fallback constant")
    assert "how should I review this diff?" in json.dumps(received["body"]), (
        "the window the bare-name needle rescued did not reach the wire")
    landed = json.loads(props.read_text())["proposals"][0]
    assert landed["classification_source"] == "llm"
    assert landed["type"] == "related-to"
