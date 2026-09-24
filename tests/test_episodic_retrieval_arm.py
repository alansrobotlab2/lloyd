"""The episodic arm's three mechanisms (#675): turns, expansion, gold.

Pure functions over strings; nothing here reaches qmd or the corpus.
"""
from __future__ import annotations

from eval.episodic_arm import (
    ALL_KINDS, PROSE_KINDS, corpus_presence, est_tokens, expand, gold_flags,
    labels_for, matched_turn, parse_turns, score_query, _norm)

DOC = """# 20260915_130917_iv184b
# 2026-09-15T06:09:17
# model: primary

user: how is the knee doing after physio
lloyd: Checking your health notes.
tool_call: Read(file_path=/x/health.md)
  → [OK] no acute pain
plenty of scar tissue
user: and the shoulder?
  → [ERROR] file not found
lloyd: No shoulder notes exist yet.
user: ok thanks
lloyd: Anytime, the TGS-RAG item can wait.
"""


def _turns():
    return parse_turns(DOC)


# ── clause 2: the turn unit ────────────────────────────────────────────────

def test_user_and_lloyd_lines_open_turns_and_the_header_is_dropped():
    turns = _turns()
    assert [t.kind for t in turns] == ["user", "lloyd", "user", "lloyd", "user", "lloyd"]
    assert "# model" not in "".join(t.text() for t in turns)


def test_tool_lines_attach_to_the_turn_they_follow():
    t = _turns()
    kinds = [k for k, _ in t[1].parts]
    assert kinds == ["lloyd", "tool_call", "tool_result"]
    # A continuation line stays with the tool result it continues.
    assert "plenty of scar tissue" in t[1].parts[2][1]
    # `[ERROR]` (what the exporter writes) and `[ERR]` both attach.
    assert [k for k, _ in t[2].parts] == ["user", "tool_result"]
    assert [k for k, _ in parse_turns("user: a\n  → [ERR] b\n")[0].parts] == ["user", "tool_result"]


def test_include_kinds_decides_what_the_scored_text_holds():
    t = _turns()[1]
    assert "scar tissue" in t.text(ALL_KINDS)
    assert "scar tissue" not in t.text(PROSE_KINDS)
    assert t.text(PROSE_KINDS) == "lloyd: Checking your health notes."


def test_turn_spans_cover_their_lines():
    turns = _turns()
    pos = DOC.index("plenty of scar")
    assert [i for i, t in enumerate(turns) if t.start <= pos < t.end] == [1]


# ── clause 1: expansion ───────────────────────────────────────────────────

def test_k0_returns_only_the_matched_turn():
    ep = expand(_turns(), 2, 0)
    assert ep.turn_indices == [2]
    assert ep.text == _turns()[2].text()


def test_k_turns_either_side_in_file_order():
    ep = expand(_turns(), 2, 1)
    assert ep.turn_indices == [1, 2, 3]
    ep2 = expand(_turns(), 2, 2)
    assert ep2.turn_indices == [0, 1, 2, 3, 4]


def test_clamped_at_file_boundaries():
    assert expand(_turns(), 0, 5).turn_indices == [0, 1, 2, 3, 4, 5]
    assert expand(_turns(), 5, 2).turn_indices == [3, 4, 5]


def test_truncated_to_the_token_budget_and_stays_contiguous():
    turns = _turns()
    tight = est_tokens(turns[2].text()) + est_tokens(turns[1].text())
    ep = expand(turns, 2, 5, budget=tight)
    assert ep.turn_indices == [1, 2]      # before-1 fit, after-1 did not: stop
    assert ep.truncated and ep.tokens <= tight
    # A matched turn bigger than the budget is cut to it.
    big = parse_turns("user: " + "x" * 4000 + "\nlloyd: hi\n")
    ep = expand(big, 0, 2, budget=100)
    assert ep.turn_indices == [0] and ep.tokens <= 100 and ep.truncated


def test_matched_turn_prefers_query_overlap_inside_the_chunk():
    turns = _turns()
    # A chunk spanning every turn resolves to the one naming the query terms.
    assert matched_turn(turns, 0, len(DOC), "shoulder notes") == 3
    # A position with no overlapping span falls back to the nearest turn.
    assert matched_turn(turns, len(DOC) + 50, 1, "zzz") == 5


# ── clause 3: deterministic gold, zero-gold counted ────────────────────────

SPEC = {"expect_entities": ["TGS-RAG"],
        "expect_docs": ["backlog/363-implement-tgs-rag-visited-memory.md"]}


def test_labels_include_the_doc_stem():
    ents, docs = labels_for(SPEC)
    assert ents == ["TGS-RAG"]
    assert docs == [["backlog/363-implement-tgs-rag-visited-memory.md",
                     "363-implement-tgs-rag-visited-memory"]]


def test_an_episode_is_gold_on_an_entity_or_a_doc_substring():
    assert gold_flags(SPEC, "the tgs_rag item")["gold"]          # normalised
    assert gold_flags(SPEC, "see 363-implement-tgs-rag-visited-memory")["docs"]
    assert not gold_flags(SPEC, "nothing relevant")["gold"]


def test_gold_is_deterministic():
    a = [gold_flags(SPEC, t.text()) for t in _turns()]
    b = [gold_flags(SPEC, t.text()) for t in _turns()]
    assert a == b


def test_zero_gold_is_a_corpus_property():
    corpus = [_norm(t.text()) for t in _turns()]
    assert corpus_presence(SPEC, corpus)["any"]
    assert not corpus_presence({"expect_entities": ["Nowhere"]}, corpus)["any"]


def test_score_query_ranks_gold_episodes():
    turns = _turns()
    eps = [expand(turns, 0, 0), expand(turns, 5, 0)]    # gold is second
    s = score_query(SPEC, eps)
    assert s["entity_hit"] and s["entity_recall"] == 1.0
    assert s["rr_doc"] == 0.5 and s["first_gold_rank"] == 2
    assert 0 < s["ndcg10"] < 1
    assert score_query(SPEC, [expand(turns, 0, 0)])["rr_doc"] == 0.0
    # The window is the mechanism: k=0 on turn 4 misses, k=1 reaches turn 5.
    assert not score_query(SPEC, [expand(turns, 4, 0)])["entity_hit"]
    assert score_query(SPEC, [expand(turns, 4, 1)])["entity_hit"]
