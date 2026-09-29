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

from app import prompt_builder as pb  # noqa: E402
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


# ── #1786: what the module's own annotations say about these placements ──────
#
# The rollout that set `session_state: system_tail` in config.yaml (2026-09-25,
# `ca7eb481`) touched only that file, and the annotations in these two modules
# went on asserting `system_head` was today's layout — `session_state_layout`'s
# docstring even called its own fallback "today's layout", so a reader was told
# an unreadable config value still yields what production renders. Only the doc
# was fixed, and only `tests/test_harness_doc_claims.py` checked, which reads the
# doc. These nodes close the gap where the layout is actually resolved. They
# assert no rendered byte moved, so they belong with this suite and had to pass
# unmodified alongside every test above it.

FALLBACK_LAYOUT = "system_head"     # the pre-rollout placement the code returns
STATE_MARK = "<active_todos>"       # where the session-state block starts
STATE_CLOSE = "</session_state>"    # and where it ends, closing the tail
MEMORY_DELTA_MARK = "<memory_delta>"  # the tail's other half
STATIC_TAIL = "Turn discipline:"    # last static line of the harness prompts


def _fallback_docstring() -> str:
    """`session_state_layout`'s docstring, flattened. Read through `ast`, not the
    file text: the function's own `return` lines mention the same layout name, so
    a whole-file grep cannot tell a docstring claim from the code it describes."""
    import ast
    tree = ast.parse(Path(pb.__file__).read_text(encoding="utf-8"))
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "session_state_layout"), None)
    assert node is not None, "session_state_layout is gone from prompt_builder"
    doc = ast.get_docstring(node)
    assert doc, "session_state_layout has no docstring to be wrong in"
    return " ".join(doc.split())


def test_the_fallback_is_the_pre_rollout_layout_and_the_docstring_says_so(vault,
                                                                          monkeypatch):
    """Clause 4: the fallback VALUE is right and stays; the word "today's" was not.

    Both unresolvable shapes still resolve to `system_head` — an unknown value,
    and a missing `prompt_layout` block entirely (the path the CLI scripts take,
    which is why the `except` is there). That behaviour must not change: raising
    on a typo would take the whole prompt down to protect a layout choice. What
    changed is the docstring's description of it, so a reader no longer concludes
    that a silent fallback is what production renders anyway.
    """
    cfg = CONFIG.setdefault("harness", {}).setdefault("prompt_layout", {})
    original = dict(cfg)
    try:
        cfg["session_state"] = "system_upside_down"
        assert pb.session_state_layout() == FALLBACK_LAYOUT, (
            "an unknown layout no longer falls back — the behaviour this clause "
            "requires unchanged")
        cfg.pop("session_state")
        assert pb.session_state_layout() == FALLBACK_LAYOUT, (
            "a missing `session_state` key no longer falls back — the CLI path "
            "that never brings up the app package would now pick a placement at "
            "random")

        doc = _fallback_docstring()
        assert FALLBACK_LAYOUT in doc, (
            f"the docstring no longer names the value it falls back to "
            f"({FALLBACK_LAYOUT!r}), so it describes an outcome it never states")
        assert "unreadable" in doc and "unknown" in doc, (
            "the docstring no longer says WHEN the fallback fires, which is the "
            f"thing #1786 had to fix: {doc[:200]!r}")
        for stale in ("today's layout", "today's output", "shipped default"):
            assert stale not in doc, (
                f"the docstring calls the fallback {stale!r} again — an unreadable "
                "config value does not yield what is in force, it yields the "
                "layout this switch shipped with")
    finally:
        cfg.clear()
        cfg.update(original)


def test_every_unreadable_config_shape_falls_back_to_the_same_layout(
        vault, monkeypatch):
    """Clause 4's behaviour half, branch by branch instead of one representative.

    `session_state_layout` reaches `system_head` three distinct ways, and #1786's
    claim — the fallback is a safe default, not what is in force — has to hold on
    all three: the key is absent from a config that loaded fine (the `.get`
    default), the value is present but is not one of the three layouts (the
    warning branch), and `CONFIG` cannot be read at all (the `except`, reached by
    replacing the module attribute the function imports so every read raises).
    Deleting that `except` turns the third case red, which the mapping mutations
    alone could never do.
    """
    import app.config as app_config

    cfg = CONFIG.setdefault("harness", {})
    original = dict(cfg)
    try:
        cfg.pop("prompt_layout", None)
        assert pb.session_state_layout() == FALLBACK_LAYOUT, (
            "with no `prompt_layout` block the resolver stopped falling back, so "
            "a config that never loaded P1 at all now picks a placement some "
            "other way")

        cfg["prompt_layout"] = {"session_state": "system_upside_down"}
        assert pb.session_state_layout() == FALLBACK_LAYOUT, (
            "an unknown value stopped falling back")

        cfg["prompt_layout"] = {"session_state": 42}
        assert pb.session_state_layout() == FALLBACK_LAYOUT, (
            "a non-string layout value no longer falls back")

        class _Unreadable:
            def get(self, *a, **k):
                raise RuntimeError("config unavailable")

        monkeypatch.setattr(app_config, "CONFIG", _Unreadable())
        assert pb.session_state_layout() == FALLBACK_LAYOUT, (
            "the `except Exception` branch stopped returning the fallback, so a "
            "CLI script that never brings up the app package now raises instead "
            "of rendering — the one outcome the docstring calls worse than the "
            "old cache behaviour")
    finally:
        cfg.clear()
        cfg.update(original)


def test_the_unknown_value_warns_and_the_absent_key_stays_quiet(vault, caplog):
    """The docstring's other half — "the unknown case logs a warning" — as behaviour.

    Both halves of that sentence are code, so both are checked: the unknown value
    logs what it fell back from and to, and an absent `prompt_layout` logs
    nothing, because the CLI scripts that never load the app config take that path
    on every prompt and a warning there would page on every turn.
    """
    import logging

    cfg = CONFIG.setdefault("harness", {})
    original = dict(cfg)
    try:
        with caplog.at_level(logging.WARNING, logger="lloyd.prompt"):
            cfg["prompt_layout"] = {"session_state": "system_upside_down"}
            pb.session_state_layout()
        assert any("system_upside_down" in r.getMessage()
                   and "system_head" in r.getMessage()
                   for r in caplog.records if r.levelno >= logging.WARNING), (
            "the unknown-value path no longer says in the log what it fell back "
            "from and to, so 'the unknown case logs a warning' is a claim "
            "nothing checks")

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="lloyd.prompt"):
            cfg.pop("prompt_layout", None)
            assert pb.session_state_layout() == FALLBACK_LAYOUT
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
            "an absent `prompt_layout` now warns on every read, which would page "
            "on every CLI script that never loads the app config — the quiet path "
            "is what the docstring means by rendering the old layout silently")
    finally:
        cfg.clear()
        cfg.update(original)


def test_the_render_under_the_shipped_config_is_the_one_the_prose_describes(
        vault):
    """Clause 1's byte claim, on the config production actually loads.

    The other nodes in this file drive the renderer with a layout the test
    picks; this one passes no `session_state` at all, so the placement arrives
    through `CONFIG` the way a live turn's does and is never named here — which
    is the rule #1786 exists to enforce, applied to its own test. It then asserts
    the two byte facts a caller sees, derived from the config's own values rather
    than from a layout written into this file: where the state block sits in the
    system prompt, and whether the turn tail is empty.

    The expectations are DERIVED, never skipped. A node that jumped out of the
    run the moment the switch moved would stop pinning anything at the exact
    moment it mattered, so `shipped` selects which of the three placements has to
    hold and `freeze_memory` selects what the tail has to contain — flip either
    in `config.yaml` and this node asserts the new truth, which is the whole
    point of leaving the value in one place.
    """
    shipped = pb.session_state_layout()          # resolved from CONFIG, not named
    assert shipped in pb.SESSION_STATE_LAYOUTS, (
        f"the shipped layout {shipped!r} is not in the enum, so production is "
        "rendering a placement no annotation describes")
    frozen = bool(CONFIG["harness"]["prompt_layout"]["freeze_memory"])

    system = _build(todos=TODOS, plan=None, goal=GOAL)
    tail = prompt_layout.turn_tail(TODOS, None, GOAL)

    if shipped == "system_tail":
        assert system.rstrip().endswith(STATE_CLOSE), (
            f"production's system prompt does not end on the session-state block "
            f"(last 45 chars: {system[-45:]!r}); the annotation prices a tail "
            "invalidation that the shipped render does not do")
        assert STATE_MARK not in system[:system.index(STATIC_TAIL)], (
            "the shipped render also places state above the static text, so "
            "`system_tail` is no longer only a tail")
    elif shipped == "system_head":
        assert system.index(STATE_MARK) < system.index(STATIC_TAIL), (
            "the shipped `system_head` render puts the block beneath the static "
            "text, which is the placement its annotation does not describe")
    else:
        assert STATE_MARK not in system and "<goal>" not in system, (
            f"the shipped `{shipped}` render carries state in the system prompt, "
            "so the byte-stability its annotation prices is not what callers get")

    state_in_tail = "state" in tail or STATE_MARK in tail
    assert state_in_tail == (shipped == "user_tail"), (
        f"`turn_tail` returned {tail[:70]!r} under shipped={shipped!r}: the "
        "session-state half of the tail is present exactly when the placement is "
        "`user_tail` and not otherwise, which is the first of the two reasons "
        "prompt_layout's docstring gives")
    if not frozen:
        assert MEMORY_DELTA_MARK not in tail, (
            "an unfrozen config still put a `<memory_delta>` note on the user "
            "message, so the second reason the docstring gives for an empty tail "
            "is false")


def test_each_placement_renders_where_its_annotation_says_it_does(vault):
    """Clause 1: the rewrite prices each layout, and a price is only honest if
    the block is where the prose says it is.

    This is the byte-level guarantee the clause asks for, checked against the
    annotations as written now rather than against a layout the prose once
    claimed was current: `system_head` puts the state ahead of the harness
    markers, `system_tail` puts it after every static byte, `user_tail` leaves it
    out of the system prompt altogether. Reorder the rendering in
    `prompt_builder` without retyping the annotation and the two halves of this
    node part company — which is the only way a comment like this earns its keep.
    """
    # `STATIC_TAIL` is the last thing the static prompt says, so "after
    # everything static" has an anchor that cannot be satisfied by accident.
    system = _build(session_state="system_head", todos=TODOS, goal=GOAL)
    assert STATE_MARK in system and STATIC_TAIL in system, (
        "the render under `system_head` lost one of the two landmarks this node "
        f"orders ({STATE_MARK!r} / {STATIC_TAIL!r}) — the fixture changed shape "
        "and the ordering assert below would be comparing positions in a prompt "
        "that no longer has both halves")
    assert system.index(STATE_MARK) < system.index(STATIC_TAIL), (
        "`system_head` no longer renders the state ahead of the static harness "
        "text, so its annotation describing a prefix shared with that text is "
        "describing a placement the code does not implement")

    tail_layout = _build(session_state="system_tail", todos=TODOS, goal=GOAL)
    assert tail_layout.index(STATIC_TAIL) < tail_layout.index(STATE_MARK), (
        "`system_tail` no longer renders beneath the static text; its whole "
        "justification is that a state edit invalidates only the tail under it, "
        "and the annotation says so")
    assert tail_layout.rstrip().endswith(STATE_CLOSE), (
        f"the rendered `system_tail` prompt does not END on {STATE_CLOSE!r} "
        f"(last 45 chars: {tail_layout[-45:]!r}), so the state block is no "
        "longer everything that comes after the static text, and 'only the tail "
        "of the system prompt is invalidated' would be describing a block that "
        "is not the tail")
    assert STATE_MARK not in tail_layout[:tail_layout.index(STATIC_TAIL)], (
        "`system_tail` is also rendering a state block ABOVE the static text, so "
        "what its annotation prices is no longer the render it claims")

    assert STATIC_TAIL in system[system.index(STATE_MARK):], (
        "under `system_head` no static text follows the state block, so the two "
        "placements now render identically and the annotations describing "
        "different cache costs are describing one behaviour")

    user_tail = _build(session_state="user_tail", todos=TODOS, goal=GOAL)
    assert STATE_MARK not in user_tail and "<goal>" not in user_tail, (
        "`user_tail` is rendering state inside the system prompt, so the "
        "byte-stability its annotation prices is not what callers get")
