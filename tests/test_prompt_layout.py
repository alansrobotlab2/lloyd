"""P1 — cross-turn prefix reuse (architecture/harness.md, Review 2026-09-24).

The system prompt is the head of every request's prefix, so anything in it
that changes between turns re-prefills the whole conversation behind it. P1
moves the session's mutable state (goal/plan/todos) and the memory files'
edits out of it, behind switches that ship as today's layout.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompt_builder as pb  # noqa: E402
from app import memory_snapshot, prefix_miss, prompt_layout  # noqa: E402
from app.config import CONFIG  # noqa: E402
from app.routers._messages_subliminal import (  # noqa: E402
    _build_subliminal_entry,
    _extract_subliminal_prefix,
    _split_subliminal,
)
from app.sessions_io import SessionTurn  # noqa: E402


def _turn(turn_id: str) -> SessionTurn:
    return SessionTurn(turn_id=turn_id, source="user", payload={},
                       enqueued_at=datetime.now())


TODOS = [{"content": "write the test", "status": "in_progress"},
         {"content": "land it", "status": "pending"}]
GOAL = {"text": "ship P1"}


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A canonical memory dir we control, no SOUL.md, no overlay."""
    mem = tmp_path / "vault"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("fact one\nfact two\n")
    (mem / "USER.md").write_text("Alan likes tests\n")
    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", mem)
    monkeypatch.setattr(pb, "_load_soul", lambda overlay=None: "SOUL")
    monkeypatch.delenv("LLOYD_OVERLAY_DIR", raising=False)
    from app import paths
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path / "sessions")
    return mem


def _layout(monkeypatch, **cfg):
    harness = dict(CONFIG.get("harness") or {})
    harness["prompt_layout"] = cfg
    monkeypatch.setitem(CONFIG, "harness", harness)


def _build(**kw):
    kw.setdefault("include_skills_index", False)
    return pb.build_system_prompt(**kw)


def _legacy_system_prompt(todos, plan, goal) -> str:
    """Today's layout, assembled by hand from the same renderers: SOUL,
    memory, goal, plan, todos, then the harness paragraphs unchanged."""
    memories = pb._load_memories(None, soul="SOUL")
    head = ["SOUL", f"<memory>\n{memories}\n</memory>"]
    head += [b for b in (pb._format_goal_block(goal), pb._format_plan_block(plan),
                         pb._format_active_todos(todos)) if b]
    bare = pb.build_system_prompt(include_skills_index=False,
                                  session_state="user_tail")
    hints = bare.split(f"<memory>\n{memories}\n</memory>\n\n", 1)[1]
    return "\n\n".join(head + [hints])


def test_system_head_is_byte_identical_to_today(vault, monkeypatch):
    _layout(monkeypatch)  # no keys: the default
    got = _build(todos=TODOS, goal=GOAL)
    assert got == _legacy_system_prompt(TODOS, None, GOAL)
    assert got == _build(todos=TODOS, goal=GOAL, session_state="system_head")
    assert "<session_state>" not in got
    # An unknown value is today's layout, not a new one.
    _layout(monkeypatch, session_state="sideways")
    assert _build(todos=TODOS, goal=GOAL) == got


def test_system_tail_puts_one_state_block_after_the_static_hints(vault, monkeypatch):
    _layout(monkeypatch, session_state="system_tail")
    got = _build(todos=TODOS, goal=GOAL)
    block = pb.build_session_state_block(TODOS, None, GOAL)
    assert got.endswith("\n\n" + block)
    assert got.index("Turn discipline") < got.index("<session_state>")
    # Everything before the block is the stateless prompt, byte for byte.
    assert got[: -len(block) - 2] == _build(session_state="user_tail")


def test_system_prompt_is_byte_stable_across_state_changes_in_user_tail(vault, monkeypatch):
    _layout(monkeypatch, session_state="user_tail")
    a = _build(todos=TODOS, goal=GOAL)
    b = _build(todos=TODOS[:1], goal={"text": "something else"},
               plan={"plan_mode": True})
    c = _build()
    assert a == b == c
    assert "<active_todos>" not in a and "<goal>" not in a


def test_state_block_rides_at_the_tail_of_the_user_message_and_in_the_subliminal_row(
        vault, monkeypatch):
    _layout(monkeypatch, session_state="user_tail")
    text = "what next?"
    tail = prompt_layout.turn_tail(TODOS, None, GOAL)
    assert tail.startswith("<session_state>")
    sent = prompt_layout.append_turn_tail("<context>vault</context>\n\n" + text, tail)
    assert sent.endswith("\n\n" + tail)

    prefix, got_tail = _split_subliminal(sent, text)
    assert prefix == "<context>vault</context>"
    assert got_tail == tail
    row = _build_subliminal_entry(_turn("t1"),
                                  prefix, "now", tail=got_tail)
    shown = row["content"][0]["text"]
    assert "<context>vault</context>" in shown and shown.endswith(tail)
    assert row["subliminal"]["tail_chars"] == len(tail)
    assert "state" in row["subliminal"]["sources"]

    # No prefix at all: the tail alone is the row, the text is not in it.
    prefix, got_tail = _split_subliminal(prompt_layout.append_turn_tail(text, tail), text)
    assert (prefix, got_tail) == ("", tail)


def test_no_tail_by_default_and_split_matches_the_old_extractor(vault, monkeypatch):
    _layout(monkeypatch)
    assert prompt_layout.turn_tail(TODOS, None, GOAL) == ""
    for sent, text in [("<context>x</context>\n\nhi", "hi"), ("hi", "hi"),
                       ('<ambient priority="notable">\nhi\n</ambient>\n\nrest', "hi")]:
        assert _split_subliminal(sent, text) == (_extract_subliminal_prefix(sent, text), "")
    row = _build_subliminal_entry(_turn("t"),
                                  "<context>x</context>", "now")
    assert "tail_chars" not in row["subliminal"]


def test_snapshot_is_taken_once_and_reused(vault, monkeypatch):
    _layout(monkeypatch, freeze_memory=True)
    mem, note = memory_snapshot.frozen_memories("s1", "mission-control")
    assert note == ""
    path = memory_snapshot.snapshot_path("s1")
    assert path.read_text() == mem and "fact one" in mem
    path.write_text(mem + "\nmarker")  # a later turn reads the file, not the vault
    again, _ = memory_snapshot.frozen_memories("s1", "mission-control")
    assert again.endswith("marker")


def test_a_memory_edit_yields_a_delta_note_not_a_new_prompt(vault, monkeypatch):
    _layout(monkeypatch, freeze_memory=True)
    mem, _ = memory_snapshot.frozen_memories("s2", "mission-control")
    before = _build(**prompt_layout.mem_kwargs(mem))
    (vault / "MEMORY.md").write_text("fact one\nfact two\nfact three\n")
    mem2, note = memory_snapshot.frozen_memories("s2", "mission-control")
    after = _build(**prompt_layout.mem_kwargs(mem2))
    assert before == after
    assert note.startswith("<memory_delta>") and "fact three" in note
    assert "+1 lines" in note
    tail = prompt_layout.turn_tail(None, None, None, note)
    assert tail == note  # rides the user message even in system_head


def test_frozen_first_turn_matches_the_live_prompt(vault, monkeypatch):
    _layout(monkeypatch, freeze_memory=True)
    mem, _ = memory_snapshot.frozen_memories("s3", "mission-control")
    assert _build(**prompt_layout.mem_kwargs(mem)) == _build()


def test_worker_snapshot_omits_user_md(vault, monkeypatch):
    _layout(monkeypatch, freeze_memory=True)
    mem, _ = memory_snapshot.frozen_memories("20260924_120000_autocode_ab12", "worker")
    assert "fact one" in mem
    assert "Alan likes tests" not in mem


def test_freeze_off_passes_nothing(vault, monkeypatch):
    _layout(monkeypatch)
    assert memory_snapshot.frozen_memories("s4", "mission-control") == (None, "")
    assert prompt_layout.mem_kwargs(None) == {}
    assert not memory_snapshot.snapshot_path("s4").exists()


def test_delta_note_is_bounded():
    note = memory_snapshot.memory_delta_note("a", "a\n" + "x" * 5000, max_chars=100)
    assert len(note) < 400 and "truncated" in note


def test_turn_start_prefix_is_emitted_once_per_turn(monkeypatch):
    monkeypatch.setattr(prefix_miss, "_cfg", lambda: {"enabled": False})
    events: list = []
    t = prefix_miss.TurnMissTracker()
    for i in (1, 2, 3):
        prefix_miss.record_iteration(
            t, i, {"input_tokens": 120_000, "cache_read": 110_000 if i == 1 else 0},
            log=lambda n, d: events.append((n, d)), ttft_ms=812)
    starts = [d for n, d in events if n == "brain1.turn_start_prefix"]
    assert starts == [{"iteration": 1, "input_tokens": 120_000,
                       "cache_read": 110_000, "reuse": 0.917, "ttft_ms": 812}]


def test_replay_injected_context_rejoins_the_subliminal_row():
    tail = "<session_state>\n<goal>g</goal>\n</session_state>"
    user = {"role": "user", "turn_id": "t1", "content": [{"type": "text", "text": "hi"}]}
    sub = _build_subliminal_entry(_turn("t1"),
                                  "<context>c</context>", "now", tail=tail)
    other = {"role": "user", "turn_id": "t2", "content": [{"type": "text", "text": "yo"}]}
    out = prompt_layout.replay_injected([user, other], [user, sub, other])
    assert out[0]["content"][0]["text"] == f"<context>c</context>\n\nhi\n\n{tail}"
    assert out[1] is other
    assert user["content"][0]["text"] == "hi"  # the persisted row is untouched


def test_replay_is_off_by_default(monkeypatch):
    _layout(monkeypatch)
    assert prompt_layout.replay_injected_context() is False
