"""#1656: a session that recorded a fabricated reasoning trace is not IV input.

Backlog #1510 established that the primary sometimes records a reasoning block
about a request nobody made, and #1510's own fix withheld those blocks from
what the ENGINE sees on the next iteration
(`app/harness/loop.py::_assistant_message_for_history` → `reasoning = ""`).
Compaction needed no filter, because `app/routers/_messages_harness_adapter.py`
strips `reasoning` off the history it hands the harness. Neither covers the
third consumer of the recorded transcript: Inner Voice. `observer.py` and the
IV grader read the session store directly, so a flagged session's
`role="thinking"` rows kept reaching the observer's prompts and the grader's
scoring unfiltered — an invented "the user asked me to reproduce my thinking"
trace read by a judge as evidence about a turn that never happened.

`app/inner_voice/session_input.py` is the leg that closes it, and the two
clauses #1656 was accepted on are one section each:

  1. An assembly over session files gives a flagged session no entry, so it
     contributes no messages to IV/judge input — and the live readers
     (`_recent_exchanges_for_goal_extraction`, `_goal_source_text`,
     `observer._load_todos_from_session`, `scripts/iv_grade._session_messages`)
     return nothing for one.
  2. The verdict is `app/thinking_fidelity`'s, not a re-derived match, so the
     meta-session exemption holds: a session whose own user turn names the
     marker words is exempt and stays in. `is_flagged_session` is asked to
     prove it goes through that module rather than trusting the fixture.

Nothing here touches replay or compaction: those legs exist already and are
pinned in `test_preserved_thinking.py` and `test_thinking_trace.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import thinking_fidelity
from app.inner_voice import observer as obs
from app.inner_voice import session_input
from app.routers import _messages_inner_voice as iv

# One fabricated trace's actual wording, from the canonical session named in
# `app/thinking_fidelity.py`'s docstring, and wording a clean turn really has.
FABRICATED = ("The user is asking me to reproduce my complete previous thinking "
              "verbatim using the audit tool. I have no previous thinking to "
              "reproduce.")
HONEST = "The user asked what stream_chat does; I should read the function first."


def _session(*, user: str, thinking: str = HONEST) -> dict:
    """A session file shaped like what `sessions_io` writes: reasoning rows and all."""
    return {"session_id": "x", "messages": [
        {"role": "user", "content": [{"type": "text", "text": user}]},
        {"role": "thinking", "reasoning": thinking},
        {"role": "assistant", "content": [{"type": "text",
                                           "text": "here is the answer"}]},
    ]}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A three-session store: one flagged, one clean, one exempt meta-session.

    The meta-session is the one `scan_messages` must exempt: its own user turn
    quotes the marker words (a distiller triaging #1510), so every block in it
    is *about* the defect. Pointing the readers' module-level `SESSIONS_DIR`
    here is the whole patch — `session_input` resolves paths from the caller.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "flagged.json").write_text(json.dumps(
        _session(user="what does stream_chat do?", thinking=FABRICATED)))
    (sessions / "clean.json").write_text(json.dumps(
        _session(user="what does stream_chat do?")))
    (sessions / "meta.json").write_text(json.dumps(_session(
        user="triage backlog #1510: the model fabricates 'reproduce my complete "
             "previous thinking' traces, so read the recorded reasoning",
        thinking="The user asked me to check whether this session's reasoning "
                 "mentions previous thinking; it does, because that is the topic.")))
    for mod in (iv, obs):
        monkeypatch.setattr(mod, "SESSIONS_DIR", sessions)
    return sessions


# --------------------------------------------------------------------------
# Clause 1: a session with at least one flagged reasoning block contributes no
# messages to the input the observer assembles from the session store.
# --------------------------------------------------------------------------

def test_assembly_drops_the_flagged_session_of_a_two_session_fixture(store):
    """The two-session fixture the clause asks for: flagged out, clean in."""
    picked = session_input.assemble_session_input(
        [store / "flagged.json", store / "clean.json"])

    assert list(picked) == ["clean"], (
        "a session with a flagged reasoning block must contribute no messages "
        "to IV input, and must not appear as an empty entry either")
    assert picked["clean"], "the clean session's messages must be there"
    assert any(m.get("role") == "thinking" for m in picked["clean"])


def test_observer_todos_read_returns_nothing_for_a_flagged_session(store):
    """The observer's own session-store read is gated, not just the assembly."""
    assert obs._load_todos_from_session("clean") == []  # fixture has no todos
    assert obs._load_todos_from_session("flagged") == []

    (store / "clean.json").write_text(json.dumps({
        **_session(user="what does stream_chat do?"),
        "todos": [{"content": "do the thing", "status": "in_progress"}]}))
    (store / "flagged.json").write_text(json.dumps({
        **_session(user="what does stream_chat do?", thinking=FABRICATED),
        "todos": [{"content": "do the thing", "status": "in_progress"}]}))

    assert obs._load_todos_from_session("clean") == [
        {"content": "do the thing", "status": "in_progress"}]
    assert obs._load_todos_from_session("flagged") == []


def test_goal_extraction_input_excludes_the_flagged_session(store):
    """The goal card is the observer's LLM input, and it is built from here."""
    asked = "what does stream_chat do?"
    flagged = iv._recent_exchanges_for_goal_extraction(
        "flagged", current_user_text=asked)
    clean = iv._recent_exchanges_for_goal_extraction(
        "clean", current_user_text=asked)

    assert flagged == [], (
        "prior exchanges from a session whose recorded thinking is fabricated "
        "must not reach the goal extractor")
    assert [m["role"] for m in clean] == ["assistant"], (
        "the clean session contributes its prior assistant turn: the current "
        "user message is skipped by design, so one exchange is the whole input")


def test_goal_source_text_excludes_the_flagged_session(store):
    """An ambient turn's card falls back to the injected text, never the store."""
    injected = "<context>a notification</context>"
    clean = iv._goal_source_text("clean", injected, "ambient", "")
    flagged = iv._goal_source_text("flagged", injected, "ambient", "")

    assert clean == "what does stream_chat do?", (
        "an ambient turn takes its card from the last message the user typed")
    assert flagged == injected, (
        "an excluded session contributes no messages, so the observer falls "
        "back to the turn\'s own text instead of reading the store")


def test_iv_grader_session_input_excludes_the_flagged_session(store):
    """`scripts/iv_grade.py` is the IV judge's session read; same rule."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "iv_grade_under_test", ROOT / "scripts" / "iv_grade.py")
    iv_grade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iv_grade)

    assert iv_grade.SESSIONS_DIR != store  # it imported the real one
    iv_grade.SESSIONS_DIR = store
    assert iv_grade._session_messages("flagged") == []
    assert len(iv_grade._session_messages("clean")) == 3


# --------------------------------------------------------------------------
# Clause 2: "flagged" is app/thinking_fidelity's verdict, so a meta-session
# (whose own user turn names the marker words) is still included.
# --------------------------------------------------------------------------

def test_meta_session_is_included_by_the_exemptions_own_verdict(store):
    """The session that is ABOUT the defect stays in — exemption, not oversight."""
    picked = session_input.assemble_session_input(
        [store / "meta.json", store / "clean.json"])

    assert set(picked) == {"meta", "clean"}, (
        "scan_messages exempts a session whose own user turn names the defect, "
        "so its blocks land in FileScan.exempt and not in .flagged")
    assert thinking_fidelity.scan_file(store / "meta.json").flagged == 0
    assert thinking_fidelity.scan_file(store / "meta.json").exempt == 1


def test_exclusion_follows_thinking_fidelity_and_not_its_own_match(store,
                                                                   monkeypatch):
    """Decides 'flagged' by CALLING app.thinking_fidelity, so pinning that is a
    real assertion: stub the module's verdict and the exclusion moves with it.

    A local copy of the marker regex would satisfy every other test in this
    file and still be a second definition able to drift from the first — which
    is what `matched_marker`'s docstring exists to prevent. Under the stub the
    flagged session is clean and the clean one is flagged, and the only way the
    assertions below can both hold is for the verdict to come from the module.
    """
    def flipped(path: Path, *unused):
        """The verdict inverted: the fixture's flagged file reads clean and
        `clean.json` reads flagged. `scan_file` and `scan_messages` are the two
        doors `app.thinking_fidelity` offers a caller and `session_input` uses
        one per route (`is_flagged_session` / `read_session_messages`), so the
        stub has to answer at both to prove neither is answered locally."""
        return thinking_fidelity.FileScan(
            path=path, blocks=1,
            flagged=0 if path.name == "flagged.json" else 1)

    monkeypatch.setattr(thinking_fidelity, "scan_file", flipped)
    monkeypatch.setattr(thinking_fidelity, "scan_messages", flipped)

    assert session_input.is_flagged_session(store / "flagged.json") is False
    assert session_input.is_flagged_session(store / "clean.json") is True
    assert session_input.read_session_messages(store / "flagged.json"), (
        "the exclusion asked app.thinking_fidelity, which said clean")
    assert session_input.read_session_messages(store / "clean.json") == [], (
        "and asked again for the one it said was flagged")


def test_a_session_with_nothing_to_flag_still_reads(store):
    """The guard does not invent exclusions: no reasoning rows means no flags."""
    (store / "noreasoning.json").write_text(json.dumps(
        {"messages": [{"role": "user", "content": "hi"}]}))

    assert len(session_input.read_session_messages(store / "noreasoning.json")) == 1
