"""#4033: the conversation linker's note-to-note edges are served beside the note.

`origin='conversation'` edges join a vault note path to a vault note path, and no
name-keyed graph reader can reach a path. `agent_mcp._document_links` serves them
where a document is the unit: `vault_read` of a note carries `linked_notes`, each
`{path, type, direction}`, from one `store().edges.active(either=<path>)`.

Every test runs against a fixture store (`kg_store.configure(tmp_path/...)`) and a
scratch vault; nothing here opens the live store.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import _document_links as DL  # noqa: E402
from agent_mcp import builtin_fs as FS  # noqa: E402
from app import kg_store  # noqa: E402

CLASSIFIER_JOB = "autonomy/74-kg-mention-classifier.md"
CLASSIFIER_SKILL = "skills/kg-mention-classifier/SKILL.md"
LONE_NOTE = "knowledge/software/a-note-nobody-linked.md"


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A scratch vault holding three notes, and an empty fixture store."""
    import agent_mcp.vault as V

    vault_root = tmp_path / "vault"
    for rel in (CLASSIFIER_JOB, CLASSIFIER_SKILL, LONE_NOTE):
        p = vault_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {rel}\n\nbody of {rel}\n")
    monkeypatch.setattr(V, "VAULT", vault_root)
    st = kg_store.configure(tmp_path / "kg.sqlite")
    FS.reset_read_records()
    yield V, st
    FS.reset_read_records()
    kg_store.reset()


def _conv(st, source, target, type_="related_to", confidence=0.9):
    return st.edges.add({"source": source, "target": target, "type": type_,
                         "confidence": confidence, "provenance": "INFERRED"},
                        origin=DL.LINK_ORIGIN)


def test_the_origin_is_the_writers_spelling():
    """Restated rather than imported (the aggregator does not import scripts/);
    pinned here so a rename on the writer's side cannot silently serve nothing."""
    from scripts.memory import conversation_relations
    assert DL.LINK_ORIGIN == conversation_relations.EDGE_ORIGIN


# ── clause 1: a linked note lists its links, from either end ─────────────────

def test_a_note_at_either_end_of_a_conversation_edge_lists_the_other(world):
    V, st = world
    _conv(st, CLASSIFIER_JOB, CLASSIFIER_SKILL, "implements", 0.95)

    at_source = V._vault_read({"path": CLASSIFIER_JOB})
    assert at_source["linked_notes"] == [
        {"path": CLASSIFIER_SKILL, "type": "implements", "direction": "out"}]
    assert "body of" in at_source["text"]

    # The target end gets its source: `either=` matches source OR target.
    at_target = V._vault_read({"path": CLASSIFIER_SKILL})
    assert at_target["linked_notes"] == [
        {"path": CLASSIFIER_JOB, "type": "implements", "direction": "in"}]


def test_the_links_come_from_one_either_lookup_on_the_store(world, monkeypatch):
    """The bounded-latency half: one `edges.active(either=<path>)` per note."""
    V, st = world
    _conv(st, CLASSIFIER_JOB, CLASSIFIER_SKILL)
    calls: list[dict] = []
    real = st.edges.active

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(st.edges, "active", spy)
    V._vault_read({"path": CLASSIFIER_SKILL})
    assert calls == [{"either": CLASSIFIER_SKILL}]


def test_an_edge_of_another_origin_is_not_a_linked_note(world):
    V, st = world
    st.edges.add({"source": CLASSIFIER_JOB, "target": CLASSIFIER_SKILL,
                  "type": "mentions", "confidence": 0.99}, origin="classifier")
    assert "linked_notes" not in V._vault_read({"path": CLASSIFIER_JOB})


# ── clause 2: no edge, no links, the read otherwise unchanged ────────────────

def test_a_note_with_no_conversation_edge_reads_exactly_as_before(world):
    V, st = world
    _conv(st, CLASSIFIER_JOB, CLASSIFIER_SKILL)  # an edge elsewhere in the store
    out = V._vault_read({"path": LONE_NOTE})
    assert out == {"path": LONE_NOTE,
                   "text": f"# {LONE_NOTE}\n\nbody of {LONE_NOTE}\n"}
    # A partial read is the same partial read it always was.
    assert V._vault_read({"path": LONE_NOTE, "start_line": 3, "num_lines": 1}) == {
        "path": LONE_NOTE, "text": f"body of {LONE_NOTE}"}


def test_an_unreadable_store_is_named_and_the_note_still_reads(world, monkeypatch):
    """Not an empty graph: the store's failure is named in place of the links,
    and the note — which is what was asked for — is served."""
    V, st = world

    def broken(*a, **k):
        raise kg_store.StoreUnavailable("read failed on kg.sqlite: disk I/O error")

    monkeypatch.setattr(st.edges, "active", broken)
    out = V._vault_read({"path": CLASSIFIER_JOB})
    assert "error" not in out
    assert out["text"].startswith(f"# {CLASSIFIER_JOB}")
    assert "linked_notes" not in out
    assert "StoreUnavailable" in out["linked_notes_unavailable"]
    assert "disk I/O error" in out["linked_notes_unavailable"]


def test_an_absent_store_is_named_too(world, monkeypatch):
    """`store()` itself raising (no database at the path, #1236) is the same case."""
    V, _st = world
    monkeypatch.setattr(kg_store, "store", lambda: (_ for _ in ()).throw(
        kg_store.StoreUnavailable("no database at /nowhere/kg.sqlite")))
    out = V._vault_read({"path": CLASSIFIER_JOB})
    assert out["text"].startswith(f"# {CLASSIFIER_JOB}")
    assert "no database" in out["linked_notes_unavailable"]


# ── clause 3: capped at 8, highest confidence first, expired never listed ───

def test_links_are_capped_at_eight_highest_confidence(world):
    V, st = world
    others = [f"backlog/{i:04d}-linked.md" for i in range(12)]
    for i, other in enumerate(others):
        _conv(st, CLASSIFIER_JOB, other, confidence=0.50 + i * 0.01)
    links = V._vault_read({"path": CLASSIFIER_JOB})["linked_notes"]
    assert DL.LINKS_CAP == 8
    assert [l["path"] for l in links] == list(reversed(others))[:8]


def test_an_expired_edge_is_never_listed(world):
    V, st = world
    keep = _conv(st, CLASSIFIER_JOB, CLASSIFIER_SKILL, confidence=0.6)
    gone = _conv(st, CLASSIFIER_JOB, LONE_NOTE, confidence=0.99)
    assert st.edges.expire(gone, "test: retired")
    links = V._vault_read({"path": CLASSIFIER_JOB})["linked_notes"]
    assert [l["path"] for l in links] == [CLASSIFIER_SKILL]
    assert keep != gone
    # And from the other end of the expired edge: nothing at all.
    assert "linked_notes" not in V._vault_read({"path": LONE_NOTE})


def test_the_either_lookup_is_answered_from_the_endpoint_indexes(world, monkeypatch):
    """The latency bound is a query plan, not a hope. Without the unary `+` on the
    expired filter the planner served `either=` from `edges_expired` — every
    active row walked, 26-35 ms per note on the live store's 66k edges, ~0.6 s on
    a 20-document recall — instead of a MULTI-INDEX OR on source/target (0.11 ms)."""
    _V, st = world
    _conv(st, CLASSIFIER_JOB, CLASSIFIER_SKILL)
    issued: list[tuple] = []
    real = st._query

    def spy(sql, params=()):
        issued.append((sql, params))
        return real(sql, params)

    monkeypatch.setattr(st, "_query", spy)
    assert st.edges.active(either=CLASSIFIER_SKILL)
    (sql, params), = issued
    plan = " | ".join(str(tuple(r)[3]) for r in
                      st.conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall())
    assert "edges_expired" not in plan, plan
    assert "edges_source" in plan and "edges_target" in plan, plan


# ── the eval's linked-notes arm (eval/run_eval.py --linked-notes) ────────────

def _eval_result():
    return {"documents": [
        {"path": "skills/dream-consolidation/SKILL.md", "linked_notes": [
            {"path": "knowledge/software/guardian-data-damage-false-trip.md",
             "type": "related_to", "direction": "in"},
            {"path": "autonomy/47-dream-consolidation.md",  # already ranked: not twice
             "type": "implements", "direction": "in"}]},
        {"path": "autonomy/47-dream-consolidation.md"},
        {"path": "backlog/3093-x.md"}], "facts": []}


def test_the_eval_scores_the_served_list_unless_the_arm_is_asked_for():
    import eval.run_eval as ev
    res = _eval_result()
    assert ev._with_linked_notes(res, "off") is res
    gold = {"expect_docs": ["knowledge/software/guardian-data-damage-false-trip.md",
                            "skills/dream-consolidation/SKILL.md"]}
    assert ev._score(gold, ev._with_linked_notes(res, "off"))["doc_recall"] == 0.5
    assert ev._score(gold, ev._with_linked_notes(res, "tail"))["doc_recall"] == 1.0


def test_tail_moves_no_ranked_document_and_inline_follows_the_carrier():
    import eval.run_eval as ev
    linked = "knowledge/software/guardian-data-damage-false-trip.md"
    tail = [d["path"] for d in ev._with_linked_notes(_eval_result(), "tail")["documents"]]
    assert tail == ["skills/dream-consolidation/SKILL.md", "autonomy/47-dream-consolidation.md",
                    "backlog/3093-x.md", linked]
    inline = [d["path"] for d in ev._with_linked_notes(_eval_result(), "inline")["documents"]]
    assert inline == ["skills/dream-consolidation/SKILL.md", linked,
                      "autonomy/47-dream-consolidation.md", "backlog/3093-x.md"]
    with pytest.raises(ValueError):
        ev._with_linked_notes(_eval_result(), "everywhere")
