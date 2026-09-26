"""Tests for #552 — did each durable memory entry / skill get honored?

The item's claim was that Lloyd measures every input surface and no *outcome*
surface: nothing asks whether an entry that landed in the prompt was ever
followed, or disputed afterward. So these tests are all about the measurement
existing, being computed from real logged evidence rather than assertion, and
being honest about the half it cannot compute.

**Exactly one test POSTs to the secondary engine**: the step-2 acceptance
measurement, `test_live_engine_scores_the_corpus_and_reports_every_way_precision_was_measured`.
It carries no mark on purpose — it is the number the item is accepted on, so it
runs on the gate, and if the engine is asleep it **fails, naming the engine**.
There is no skip and no opt-out: an earlier version asserted whichever branch it
landed in and recorded nothing about which, so an asleep model and a passing
grader were indistinguishable in the report. The fail-closed behaviour of a
silent engine is pinned hermetically by the stand-in-transport tests, so
refusing to grade the live measurement against an absent model costs no coverage.

Everything else here reaches the engine only through an injected transport or a
replayed fixture, including the two probe-exit tests: `_replay_engine` answers
`classify_dispute` *and* `classify_dispute_raw` with the replies recorded on the
measurement date, so the zero-shot pass and the replay verifier run without a
model anywhere. That is also why the "do the committed replies still hold"
check is a probe step (`uptake_probe.verify_replay`, recorded in every emitted
table as `classifier.replay`) and not a test: asking a live model whether it
still agrees with its own recorded answer is a property of the measurement run,
and inside the suite it would make a hermetic replay claim depend on which GGUF
happens to be loaded. Two further tests read the live `~/obsidian` vault and
carry the tree's existing `live_vault` mark, which the automod gate excludes.
Other tests read live session and baseline *data*; none of those is the engine.

Nothing in this file skips, xfails, or asserts `True`; the loopback HTTP seam is
crossed by a test that runs whether or not a model is loaded, and the precision
claim is replayable from recorded replies.
"""

from __future__ import annotations

import json
import os
import re
import sys
from urllib.parse import urlparse
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import uptake  # noqa: E402



# ---------------------------------------------------------------- corpus ----

def _write_session(root: Path, name: str, messages: list[dict], source=None) -> Path:
    doc = {
        "session_id": name,
        "id": name,
        "title": "",
        "source": source,
        "created_at": "2026-09-10T00:00:00+00:00",
        "last_active": "2026-09-10T00:00:00+00:00",
        "messages": messages,
    }
    p = root / "sessions" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def _user(text: str, source=None) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}], "source": source}


def _asst(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _asst_tools_only() -> dict:
    """The shape the harness writes when the agent ran tools and never wrote an
    answer: an assistant message whose text part is empty."""
    return {"role": "assistant", "content": [{"type": "text", "text": ""}],
            "tool_calls": [{"id": "t1", "name": "Bash", "args": {}}]}





def test_human_turn_filter_drops_synthetic_user_turns(tmp_path):
    """The corpus is about a *human* disputing something.

    Two shapes leak through a naive `role == "user"` filter and both were
    present in the real 30-day population: inner-voice injections arrive as
    user-role messages carrying `source`, and autonomy task prompts arrive as
    user-role messages whose text opens with `[SYSTEM:`. If either were kept,
    the classifier would spend its whole positive class on Lloyd's own
    scaffolding text and the uptake table would attribute disputes to entries
    that were never questioned by anyone.
    """
    _write_session(tmp_path, "s_a", [
        _asst("Here is the summary you asked for."),
        _user("wrong, that is not what the file says"),
        _user("[INNER VOICE] Stop: you have run 3 variations of the same search",
              source="inner_voice_inject"),
        _user("[SYSTEM: You are executing autonomy task #1] Do the thing",
              source="autonomy"),
        _user("<context><skill name=\"x\">…</skill></context> ambient nudge",
              source="ambient"),
        _user("hey there"),
    ])
    turns = uptake.human_turns(root=tmp_path, days=3650)
    texts = [t.user_text for t in turns]
    assert any("wrong, that is not" in t for t in texts), texts
    assert not any("INNER VOICE" in t for t in texts), texts
    assert not any("autonomy task" in t for t in texts), texts
    assert not any("<context>" in t for t in texts), texts
    assert any("hey there" in t for t in texts)


def test_turn_carries_the_previous_assistant_prose(tmp_path):
    """A dispute is decided *against* what was just delivered — uReview graded
    the reply to a comment, not the reply alone. `continue` after a real answer
    is a nudge; `continue` after a turn that produced no answer is the user
    having to re-ask for work Lloyd promised."""
    _write_session(tmp_path, "s_b", [
        _user("check the thing"),
        _asst("Checked. It is fine."),
        _user("please continue"),
    ])
    turns = uptake.human_turns(root=tmp_path, days=3650)
    again = [t for t in turns if t.user_text == "please continue"][0]
    assert again.prev_assistant == "Checked. It is fine."
    assert again.turn_id.endswith("#2")


def test_candidate_extraction_seeds_but_does_not_decide(tmp_path):
    """The cue list is a recall-oriented screen. It has to catch an explicit
    correction and a re-prompt; it is allowed to also catch requests, which is
    why the labels are hand-made and the classifier is the decider."""
    _write_session(tmp_path, "s_c", [
        _asst("Built cleanly, `dist/` has the manifest."),
        _user("Failed to load extension — Could not load manifest."),
        _asst_tools_only(),
        _user("please continue"),
        _asst("A 2.5 lb plate is 2.5 inches across."),
        _user("what about the dimensions of a 2.5 lb plate?"),
    ])
    turns = uptake.human_turns(root=tmp_path, days=3650)
    cands = uptake.candidate_disputes(turns)
    got = {c.user_text for c in cands}
    assert any("Failed to load extension" in t for t in got), got
    # The re-prompt follows a turn that produced no prose, so it is the user
    # re-asking for promised work. The same words after the plate answer above
    # must not screen in.
    assert any(t == "please continue" for t in got), got
    assert not any("2.5 lb plate" in t for t in got), got
    again = [t for t in turns if t.user_text == "please continue"][0]
    assert again.prev_is_no_answer is True
    answered = [t for t in turns if "2.5 lb plate" in t.user_text][0]
    assert answered.prev_is_no_answer is False


# ------------------------------------------------------------- classifier ----

def test_precision_recall_math_is_typed_from_the_confusion_matrix():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    preds = [1, 1, 1, 0, 1, 0, 0, 0]
    m = uptake.precision_recall(labels, preds)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)


def test_precision_recall_refuses_a_denominator_that_cannot_be_divided():
    """A run with no predicted positives has no precision. Reporting 0.0 would
    read as "the classifier is bad"; reporting 1.0 would read as a pass. Both
    are false — the run measured nothing, so it must say so."""
    m = uptake.precision_recall([1, 1, 0], [0, 0, 0])
    assert m["precision"] is None
    assert m["measured"] is False


def test_a_grader_that_barely_fires_cannot_pass_on_precision_alone():
    """The first measured run of this probe scored precision 1.00 / recall 0.09
    by answering NOT to nearly everything. Under the item's literal acceptance
    (precision >= 0.70) that run *passes* and emits a table saying no durable
    entry is ever disputed — a measurement reporting health because it detects
    nothing. So both floors are enforced, and they are enforced by the one
    function the probe itself calls. Asserting `FLOOR > 0.0` grades two literals
    against themselves, and re-typing the boolean here would leave the production
    gate untested while looking like it was covered.
    """
    import scripts.uptake_probe as probe

    assert probe.uptake is uptake, "the probe must gate on the shared floors"
    # The always-NOT shape that scored 1.00 precision cannot clear the recall floor.
    always_not = uptake.precision_recall([1] * 23 + [0] * 23, [0] * 46)
    assert always_not["precision"] is None, "no predicted positive has no precision"
    assert uptake.classifier_clears_floors(always_not) is False

    # Each clause of the gate, discriminated rather than tautologised.
    def metrics(precision, recall, measured=True):
        return {"precision": precision, "recall": recall, "measured": measured}

    assert uptake.classifier_clears_floors(metrics(1.0, 0.0)) is False, "recall floor"
    assert uptake.classifier_clears_floors(metrics(0.69, 1.0)) is False, "precision floor"
    assert uptake.classifier_clears_floors(metrics(None, 1.0)) is False, "no denominator"
    assert uptake.classifier_clears_floors(metrics(1.0, None)) is False, "no denominator"
    assert uptake.classifier_clears_floors(metrics(0.70, 0.50, measured=False)) is False
    assert uptake.classifier_clears_floors(None) is False
    assert uptake.classifier_clears_floors(metrics(0.70, 0.50)) is True, "the floors themselves"
    # Reached through the probe's own module, so the probe cannot be gating on
    # something else.
    assert probe.uptake.classifier_clears_floors(always_not) is False


@pytest.mark.parametrize("reply,want", [
    ("DISPUTE", True),
    ("dispute", True),
    ("NOT_DISPUTE", False),
    ("NOT DISPUTE", False),
    ("Verdict: not_dispute, the user is asking a new question.", False),
    ("DISPUTE — the user contradicts the delivered result", True),
])
def test_classifier_reads_a_binary_verdict_not_prose(reply, want):
    got = uptake._parse_verdict(reply)
    assert got is want


def test_classifier_calls_the_secondary_engine_with_thinking_disabled():
    """Seam: loopback HTTP to the llama.cpp secondary slot. The tree's own
    client (`app/secondary_models`) sets `chat_template_kwargs.enable_thinking`
    because this model emits a separate `reasoning_content` stream, and a
    thinking block that opens with "Thinking Process:" would parse as a verdict
    on nothing. This test crosses the seam with a fake transport and pins the
    payload the seam requires."""
    seen = {}

    def fake_transport(payload):
        seen.update(payload)
        return {"choices": [{"message": {"content": "DISPUTE"}}]}

    got = uptake.classify_dispute("Built cleanly.", "no, manifest will not load",
                                  transport=fake_transport)
    assert got is True
    assert seen["chat_template_kwargs"]["enable_thinking"] is False
    assert seen["model"] == uptake.SECONDARY_MODEL
    assert "Built cleanly." in seen["messages"][-1]["content"]


def test_classifier_fails_closed_when_the_engine_is_unreachable():
    """A dead engine must not silently read as "no disputes" — that would make
    every entry look perfectly honored, which is the exact failure mode the
    item is about."""
    def dead(payload):
        raise OSError("connection refused")

    assert uptake.classify_dispute("a", "b", transport=dead) is None


# ------------------------------------------------------------ attribution ----

def _mk_turn(idx, text, prev, *, skills=(), ctx_titles=(), session="s1"):
    # Whatever presence a fixture claims, it claims from a persisted injected
    # block — the reader can produce `injected_skills`/`vault_context` no other way
    # now, so a fixture with evidence and no block would describe a state the
    # parser cannot emit, and `reach` would be lying about it.
    return uptake.Turn(
        session=session, session_source=None, ts=f"2026-09-10T0{idx % 10}:00:00+00:00",
        user_text=text, prev_assistant=prev, ordinal=idx,
        injected_skills=list(skills), vault_context=list(ctx_titles),
        injections_seen=1 if (skills or ctx_titles) else 0,
    )


def test_verdicts_cannot_leak_across_sessions_that_share_an_ordinal():
    """The decisive defect, pinned. The whole suite was green while it shipped.

    `Turn.ordinal` counts human turns **within one session**, so it restarts at 1
    everywhere. Measured on the live 30-day window: 220 turns over 132 sessions
    occupy ordinals 1-10, so the 24 screened candidates collapsed into 9 ordinal
    keys — and the join, `flagged.get(t.ordinal)`, charged every session that had
    a turn at a colliding position with some other session's verdict. The
    committed artifact reported 20 disputes when 9 had actually been classified.

    A single-session test cannot see this no matter how many turns it has, which
    is why every attribution test here used to build one session. Two sessions,
    same ordinal, one verdict: the counts below only come out right if the join
    is keyed per turn.
    """
    a = _mk_turn(1, "wrong, that is not what the file says", "Here you are.",
                 skills=["voice-mode"], session="sess_a")
    b = _mk_turn(1, "pull this transcript please", None,
                 skills=["voice-mode"], session="sess_b")
    table = uptake.build_uptake_table(
        turns=[a, b],
        # Only `a` was ever screened. `b` sits at the same ordinal and must
        # inherit nothing from it.
        dispute_flags={"sess_a#1": True}, memory_entries=[],
        skills_read={"sess_a": {"voice-mode": 1}, "sess_b": {"voice-mode": 1}})
    row = [r for r in table["entries"] if r["entry"] == "skill:voice-mode"][0]
    assert row["present_in_turns"] == 2, row
    assert row["disputes"] == 1, "a verdict from sess_a was charged to sess_b's turn"
    assert row["dispute_rate"] == pytest.approx(0.5), row
    assert row["present_turn_ids"] == ["sess_a#1", "sess_b#1"], row
    corpus = table["corpus"]
    assert corpus["disputes"] == 1, corpus
    assert corpus["classified_turns"] == 1, corpus
    assert corpus["unscreened_turns"] == 1, corpus
    assert corpus["verdict_keyed_on"] == "turn_id", corpus

    # A caller that hands back ordinal keys — the shape that caused this — is a
    # programming error, and refusing is the only honest outcome: it can neither
    # match nothing (silent zero disputes) nor match the wrong turn.
    with pytest.raises(ValueError, match="not turn_ids"):
        uptake.build_uptake_table(turns=[a, b], dispute_flags={1: True},
                                  memory_entries=[], skills_read={})
    with pytest.raises(ValueError, match="not turn_ids"):
        uptake.build_uptake_table(turns=[a, b], dispute_flags={"sess_c#1": True},
                                  memory_entries=[], skills_read={})
    with pytest.raises(ValueError, match="duplicate turn_id"):
        uptake.build_uptake_table(turns=[a, a], dispute_flags={}, memory_entries=[],
                                  skills_read={})


def test_dispute_is_attributed_only_to_entries_present_in_that_turn():
    """"Present in the turn" is the whole crux, and the item names the risk:
    USER.md and MEMORY.md are in every prompt, so an unweighted join hands one
    dispute to every entry in force. Presence therefore stays evidence-bound —
    a skill is present only if it was injected or read in that session, a note
    only if it arrived in that turn's prefetch — and the always-in-force memory
    entries carry an explicit presence source plus a text-overlap weight so a
    reader can tell the two kinds apart."""
    t1 = _mk_turn(1, "wrong, the restart target is wrong", "Restarted livekit.",
                  skills=["voice-mode"], ctx_titles=["knowledge/software/livekit.md"],
                  session="s1")
    t2 = _mk_turn(2, "pull this transcript please", None,
                  skills=["youtube-transcript"], session="s1")
    table = uptake.build_uptake_table(
        turns=[t1, t2],
        dispute_flags={"s1#1": True, "s1#2": False},
        memory_entries=[uptake.Entry("lloyd/MEMORY.md", "restart lloyd-agent-worker")],
        # The real reader's shape: session → {skill: earliest human turn it was
        # read during}. Passing {} here — as this test used to — left the entire
        # per-session join unexercised while its name promised exactly that, so
        # the session-boundary bug could have shipped under a green test.
        # Same skill names, read only in ANOTHER session: they must contribute no
        # presence to these two turns.
        skills_read={"elsewhere": {"voice-mode": 1, "youtube-transcript": 1}},
    )
    rows = {r["entry"]: r for r in table["entries"]}

    vm = rows["skill:voice-mode"]
    assert vm["present_in_turns"] == 1 and vm["disputes"] == 1
    assert vm["dispute_rate"] == pytest.approx(1.0)

    yt = rows["skill:youtube-transcript"]
    # Present via the injected-context route on t2 only.
    assert yt["present_in_turns"] == 1 and yt["disputes"] == 0
    for row in (vm, yt):
        assert row["presence_bound"] == "injected_this_turn", row
    assert uptake.PRESENCE_BOUNDS >= {r["presence_bound"] for r in table["entries"]
                                     if r["kind"] == "skill"}

    # Which turns the join actually credited, per row. Asserting that another
    # skill's name is absent from this row's JSON could never fail — one row's
    # dict never contains another skill's name — so the assertion is on the row's
    # own evidence: `yt` is present only on the non-dispute turn, and `vm` only on
    # the dispute turn. A join that leaked t1's dispute into `yt`, or credited t2
    # to `vm`, moves these lists.
    assert yt["present_turn_ids"] == ["s1#2"], yt
    assert vm["present_turn_ids"] == ["s1#1"], vm
    assert (yt["name"], yt["kind"]) == ("youtube-transcript", "skill"), yt


def test_always_in_force_memory_entries_are_flagged_and_weighted():
    table = uptake.build_uptake_table(
        turns=[_mk_turn(1, "wrong, the restart target is wrong", "Restarted livekit."),
               _mk_turn(2, "what is 2+2", None)],
        dispute_flags={"s1#1": True, "s1#2": False},
        memory_entries=[uptake.Entry("lloyd/MEMORY.md", "Restart lloyd-agent-worker for voice")],
        skills_read={},
    )
    row = table["entries"][0]
    assert row["present_in_turns"] == 2
    assert row["disputes"] == 1
    assert row["presence_source"] == uptake.ALWAYS_IN_FORCE
    # The overlap weight is what separates "this entry was questioned" from
    # "this entry was in the prompt when something else was questioned".
    assert row["weighted_disputes"] < row["disputes"]
    assert row["overlap_max"] > 0.0


def test_skill_presence_is_derived_from_skills_read_event_payloads(tmp_path):
    """Seam: `event_logs/*.events.jsonl`, written by a different process than
    the reader. #435's per-injection telemetry does not exist yet, so skill
    presence is derived from the `skills_read` tool-call payloads in the turn
    event log, exactly as the triage directed. The reader must key on the
    `name` inside `data.args` — a JSON string the writer double-encoded — and
    must ignore a Bash command that merely mentions the tool."""
    ev = tmp_path / "event_logs" / "s1.events.jsonl"
    ev.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        # The turn boundary the ordinal is counted from — same file, same order,
        # so no clock assumption is involved. Only source=="user" counts.
        {"ts": "x", "session_id": "s1", "event": uptake.TURN_EVENT, "turn_id": "t1",
         "data": {"text": "do the thing", "source": "user"}},
        {"ts": "x", "session_id": "s1", "event": uptake.TURN_EVENT, "turn_id": "t1b",
         "data": {"text": "inner-voice follow-up", "source": "inner_voice"}},
        {"ts": "x", "session_id": "s1", "event": "brain1.tool_call_proposed",
         "turn_id": "t1", "data": {"tool_call_id": "a", "name": "skills_read",
                                   "args": json.dumps({"name": "voice-mode",
                                                       "summary": "reading"})}},
        {"ts": "x", "session_id": "s1", "event": uptake.TURN_EVENT, "turn_id": "t2",
         "data": {"text": "and now this", "source": "user"}},
        {"ts": "x", "session_id": "s1", "event": "brain1.tool_call_proposed",
         "turn_id": "t2", "data": {"tool_call_id": "c", "name": "skills_read",
                                   "args": json.dumps({"name": "youtube-transcript"})}},
        {"ts": "x", "session_id": "s1", "event": "brain1.tool_call_proposed",
         "turn_id": "t2", "data": {"tool_call_id": "b", "name": "Bash",
                                   "args": json.dumps({"command": "grep skills_read app/"})}},
        "this line is not json at all\n",
    ]
    ev.write_text("\n".join(json.dumps(l) if isinstance(l, dict) else l for l in lines))

    got = uptake.skills_read_by_session(tmp_path)
    # value = earliest human turn during which the skill was read: voice-mode
    # during turn 1, youtube-transcript not until turn 2. A set here would be the
    # old shape, which is what let a turn-2 dispute be charged to a turn-9 read.
    assert got == {"s1": {"voice-mode": 1, "youtube-transcript": 2}}, got


def test_a_skill_read_late_in_a_session_is_not_charged_to_earlier_disputes():
    """The defect this replaces: presence was a per-session set, so a skill first
    opened on turn 3 was credited into turns 1 and 2 — inflating the divisor and
    the numerator of the same row at once. A dispute before the read must land on
    nothing."""
    t1 = _mk_turn(1, "wrong, that is not what I asked", "Here you are.", session="s1")
    t2 = _mk_turn(2, "thanks", None, session="s1")
    table = uptake.build_uptake_table(
        turns=[t1, t2], dispute_flags={"s1#1": True, "s1#2": False}, memory_entries=[],
        skills_read={"s1": {"voice-mode": 2}},   # first read during turn 2
    )
    rows = [r for r in table["entries"] if r["kind"] == "skill"]
    assert len(rows) == 1, rows
    # Present from turn 2 only — and crucially NOT charged with the turn-1 dispute.
    assert rows[0]["present_in_turns"] == 1 and rows[0]["disputes"] == 0, rows[0]


def test_a_turn_reading_a_skill_takes_the_blame_for_that_turn_only():
    t1 = _mk_turn(1, "first", None, session="s1")
    t2 = _mk_turn(2, "wrong again", "ok", session="s1")
    t3 = _mk_turn(3, "fine", None, session="s1")
    table = uptake.build_uptake_table(
        turns=[t1, t2, t3], dispute_flags={"s1#1": False, "s1#2": True, "s1#3": False},
        memory_entries=[], skills_read={"s1": {"voice-mode": 2}})
    row = [r for r in table["entries"] if r["entry"] == "skill:voice-mode"][0]
    assert row["present_in_turns"] == 2, "turn 1, before the read, was counted"
    assert row["disputes"] == 1
    assert row["dispute_rate"] == pytest.approx(0.5)
    assert row["presence_bound"] == "causal_event_order"


def test_coverage_block_reports_what_the_table_actually_covers():
    table = uptake.build_uptake_table(
        turns=[_mk_turn(1, "wrong", "ok", skills=["voice-mode"])],
        dispute_flags={"s1#1": True},
        memory_entries=[uptake.Entry("lloyd/MEMORY.md", "a"), uptake.Entry("lloyd/MEMORY.md", "b")],
        skills_read={"s1": {"voice-mode": 1, "voice-clone-sample": 1}},
        active_skills=["voice-mode", "voice-clone-sample", "restart-lloyd", "obsidian"],
    )
    cov = table["coverage"]
    assert cov["user_md_entries"]["covered"] == 2 and cov["user_md_entries"]["total"] == 2
    assert cov["active_skills"]["covered"] == 2 and cov["active_skills"]["total"] == 4
    # The half the item cannot ask for yet must be labelled, not hidden.
    assert cov["active_skills"]["presence_source"] == uptake.SKILL_PRESENCE_PROXY
    assert "435" in cov["active_skills"]["note"]


def test_the_memory_denominator_counts_what_the_grammar_misses(tmp_path):
    """`user_md_entries.total` used to be `len(entries)`, so the "≥ 80 % of
    USER.md entries" clause was this regex grading itself. The tally of
    bullet-shaped lines the grammar does NOT reach now has to be in the
    denominator, and the ratio has to move when it does."""
    doc = tmp_path / "probe" / "MEMORY.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "# T\n\n"
        "- a top-level bullet that is long enough\n"
        "  - an indented sub bullet that is also long\n"
        "- too short\n"
    )
    tally: dict[str, dict[str, int]] = {}
    entries = uptake.memory_entries(root=tmp_path, docs=("probe/MEMORY.md",), tally=tally)
    assert len(entries) == 1, entries
    assert tally == {"probe/MEMORY.md": {"indented_bullets": 1, "short_bullets": 1}}, tally

    cov = uptake._memory_coverage(entries, tally)
    assert cov["covered"] == 1 and cov["total"] == 3
    assert cov["ratio"] == pytest.approx(1 / 3, abs=5e-5)  # published at 4 dp
    assert cov["skipped"] == {"indented_bullets": 1, "short_bullets": 1}
    # `covered` is presence, not uptake — the block has to say so, or a reader
    # files 100 % under "the entries are being honored".
    assert "NOT evidence an entry was honored" in cov["denominator_note"]


def test_the_memory_denominator_is_reported_per_document(tmp_path):
    """The acceptance clause is "≥ 80 % of **USER.md** entries". One flat tally
    lets a clean MEMORY.md carry a USER.md the grammar barely reads, so the
    denominator is per document and the two ratios are independent."""
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / "USER.md").write_text(
        "- a top-level bullet that is long enough\n"
        "  - an indented sub bullet that is also long\n"
        "  - another indented sub bullet that is long too\n"
    )
    (tmp_path / "probe" / "MEMORY.md").write_text(
        "- first bullet of memory that is long\n"
        "- second bullet of memory that is long\n"
    )
    tally: dict[str, dict[str, int]] = {}
    entries = uptake.memory_entries(
        root=tmp_path, docs=("probe/USER.md", "probe/MEMORY.md"), tally=tally)
    cov = uptake._memory_coverage(entries, tally)
    assert cov["by_doc"]["probe/USER.md"] == {
        "covered": 1, "total": 3, "ratio": pytest.approx(1 / 3, abs=5e-5),
        "skipped": {"indented_bullets": 2, "short_bullets": 0}}
    assert cov["by_doc"]["probe/MEMORY.md"]["ratio"] == 1.0
    # A flat ratio in the middle is not what the clause gets graded on; both
    # numbers have to be there to read.
    assert cov["ratio"] == pytest.approx(0.6, abs=5e-5)


def test_the_committed_classifier_numbers_recompute_from_their_own_lists():
    """The committed table is the acceptance evidence, so it has to be
    self-checking: precision/recall must agree with the confusion counts they are
    derived from, and the disagreement turn ids must be present in the label file
    with the labels the counts claim. A block that only carries `precision: 1.0`
    can be re-typed by hand; this one cannot."""
    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    j = json.loads(files[-1].read_text())
    cls = j["classifier"]
    tp, fp, fn = cls["tp"], cls["fp"], cls["fn"]
    assert cls["precision"] == pytest.approx(tp / (tp + fp))
    assert cls["recall"] == pytest.approx(tp / (tp + fn))

    dis = cls["disagreements"]
    assert len(dis["false_positive"]) == fp, "the fp count has no matching ids"
    assert len(dis["false_negative"]) == fn, "the fn count has no matching ids"
    labels = {i["turn_id"]: int(i["label"])
              for i in json.loads((REPO / cls["labels_file"]).read_text())["items"]}
    for tid in dis["false_positive"]:
        assert labels[tid] == 0, f"{tid} listed as a false positive but labeled dispute"
    for tid in dis["false_negative"]:
        assert labels[tid] == 1, f"{tid} listed as a miss but labeled a non-dispute"
    assert set(dis["false_negative"]) & set(dis["false_positive"]) == set()

    # The artifact's self-declared pass is the production gate's verdict on its
    # own numbers — not a `passed: true` someone typed next to the metrics.
    assert cls["passed"] is uptake.classifier_clears_floors({
        "precision": cls["precision"], "recall": cls["recall"], "measured": True}), cls
    # And the join it reports is the fixed one: counts that cannot exceed what was
    # classified, keyed per turn rather than per position.
    corpus = j["corpus"]
    assert corpus["verdict_keyed_on"] == "turn_id", corpus
    assert corpus["disputes"] <= corpus["classified_turns"] <= corpus["turns"], corpus


def test_every_row_matches_the_declared_contract_hermetically():
    """The gate-running half of the skill↔table seam. The consumer's prose is
    checked against the artifact only by a `live_vault` test, which the gate
    deselects — so on the gate's own run nothing protected the schema at all.
    This needs no vault, no engine and no transcripts: rows built from fixtures
    must match `ROW_KEYS` exactly and take a `presence_source` from the declared
    set, so renaming a field fails the suite on any box, and the committed
    artifact is held to the same contract."""
    turns = [_mk_turn(1, "wrong, that restart target is wrong", "Restarted.",
                      skills=["voice-mode"],
                      ctx_titles=["knowledge/software/livekit.md"], session="s1"),
             _mk_turn(2, "thanks", None, session="s1")]
    table = uptake.build_uptake_table(
        turns=turns, dispute_flags={"s1#1": True, "s1#2": False},
        memory_entries=[uptake.Entry("lloyd/MEMORY.md", "restart lloyd-agent-worker")],
        skills_read={"s1": {"voice-mode": 1}},
        active_skills=["voice-mode"])
    kinds = {r["kind"] for r in table["entries"]}
    assert {"memory_entry", "skill", "note"} <= kinds, f"one row shape untested: {kinds}"

    for row in table["entries"]:
        assert uptake.ROW_KEYS <= set(row), (
            f"{row['entry']} is missing {uptake.ROW_KEYS - set(row)}")
        assert row["presence_source"] in uptake.PRESENCE_SOURCES, row["presence_source"]
        if row["kind"] == "skill":
            assert row["presence_bound"] in uptake.PRESENCE_BOUNDS, row

    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    for row in json.loads(files[-1].read_text())["entries"]:
        assert uptake.ROW_KEYS <= set(row), (
            f"{row['entry']} is missing {uptake.ROW_KEYS - set(row)} in the artifact")
        assert row["presence_source"] in uptake.PRESENCE_SOURCES, row["presence_source"]


# ---------------------------------------------------------------- gate ------

#: Sentinel for "this night has no `semantic_seeding` key at all" (#1547) —
#: distinct from a night that records `{"enabled": False, "k": 0}`.
_NO_SEEDING = object()


def _mk_baseline(path: Path, label: str, doc_hr: float, ndcg: float, *, prod=True,
                 limit=20, seeding=_NO_SEEDING):
    """One baseline artifact, at the shape `run_eval.py` actually writes.

    `seeding` is omitted (not written as null) by default because that is the
    shape of every nightly on disk before #1547: the key is ABSENT, and absent
    is the state the gate must not read as "seeding was off".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "label": label, "ran_at": "2026-09-09T06:00:00+00:00", "limit": limit,
        "matches_production_defaults": prod, "graph_rerank": False,
        "summary": {"overall": {"n_queries": limit, "doc_hit_rate": doc_hr,
                                "ndcg10": ndcg, "entity_hit_rate": 0.5}},
    }
    # Top level, spread the way `main()` spreads `build_run_config(args)` — the
    # reader's half of the seam pinned writer-side in
    # tests/test_eval_scorer.py::test_the_seeding_record_lands_at_the_artifact_top_level_the_reader_reads.
    if seeding is not _NO_SEEDING:
        doc["semantic_seeding"] = seeding
    path.write_text(json.dumps(doc))


def _mk_night(d: Path, i: int, label: str, doc_hr: float, ndcg: float, **kw) -> None:
    """One fixture night whose mtime is strictly later than its predecessor's.

    `retrieval_gate` orders its pool by mtime, so a fixture that wants to say
    "newest" has to set it; `os.utime` is the only honest way to do that here.
    """
    p = d / f"{label}-{i}.json"
    _mk_baseline(p, label, doc_hr, ndcg, **kw)
    os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))


def test_retrieval_gate_is_derived_from_the_newest_baseline_plus_its_noise_band(tmp_path):
    """A gate that hard-codes `doc_hit_rate >= 0.95` is decided by noise: on six
    identical-config nights the metric alone spans 0.85-0.95. The re-based gate
    is the newest measurement minus the spread of comparable nights, so the
    tolerance is a measured quantity, not a remembered one.

    And the sample it measured has to be readable from the artifact (#1220): every
    metric block names the window it is capped to and the nights actually pooled,
    so a quoted `band` cannot travel without its denominator.
    """
    d = tmp_path / "baselines"
    for i, (lab, dh, nd) in enumerate([
        ("nightly-20260904", 0.95, 0.591), ("nightly-20260905", 0.90, 0.557),
        ("nightly-20260906", 0.90, 0.536), ("nightly-20260907", 0.90, 0.562),
        ("nightly-20260908", 0.95, 0.595), ("nightly-20260909", 0.85, 0.536),
    ]):
        _mk_night(d, i, lab, dh, nd)

    gate = uptake.retrieval_gate(baselines_dir=d)
    assert gate["latest"]["label"] == "nightly-20260909"
    assert gate["nights"] == 6
    # Six nights here, below the cap, so pooled == comparable and both are named.
    assert gate["nights_in_shape"] == 6
    assert gate["window_nights"] == uptake.RETRIEVAL_GATE_WINDOW_NIGHTS
    for metric in ("doc_hit_rate", "ndcg10"):
        blk = gate[metric]
        assert blk["window_nights"] == uptake.RETRIEVAL_GATE_WINDOW_NIGHTS, metric
        assert blk["n_nights"] == 6, f"{metric}: band published without its sample size"
    assert gate["doc_hit_rate"]["band"] == pytest.approx(0.10)
    assert gate["doc_hit_rate"]["floor"] == pytest.approx(0.75)
    assert gate["ndcg10"]["band"] == pytest.approx(0.059)
    assert gate["hardcoded_gate_would_have_failed_nights"] >= 4


def test_retrieval_gate_ignores_nights_it_cannot_compare(tmp_path):
    """A 40-query run or a rerank-on run is a different experiment; folding it
    into the spread would widen the band with somebody else's variance."""
    d = tmp_path / "baselines"
    _mk_baseline(d / "nightly-a.json", "nightly-a", 0.95, 0.59)
    _mk_baseline(d / "nightly-b.json", "nightly-b", 0.85, 0.53)
    _mk_baseline(d / "nightly-wide.json", "nightly-wide", 0.40, 0.20, limit=40)
    _mk_baseline(d / "nightly-off.json", "nightly-off", 0.40, 0.20, prod=False)
    gate = uptake.retrieval_gate(baselines_dir=d)
    assert gate["nights"] == 2
    assert gate["nights_in_shape"] == 2
    assert gate["doc_hit_rate"]["floor"] == pytest.approx(0.75)


def test_a_seeding_on_night_is_never_banded_with_an_unrecorded_one(tmp_path):
    """#1547 clauses 3 and 4: the comparability shape carries the recorded seeding.

    #1486 changed the SEED DEFINITION the entity scores are built from, and did it
    between two nights that share `limit: 20` and both read
    `matches_production_defaults: true` — so `nightly-20260925` (seeding off) and
    `nightly-20260926` (seeding on) landed in one shape and one band while
    `entity_hit_rate` moved 0.337 -> 0.500, `entity_recall_avg` 0.384 -> 0.579 and
    `anchorless_query_count` 25 -> 16 across them. The fixture is those two files:
    same limit, both production-config, one recording `k: 3` and one holding no
    key at all — and no key is NOT a claim that seeding was off, it is the absence
    of a measurement, which is why it is its own band rather than `k=0`. The
    published pool is one night, not two.
    """
    d = tmp_path / "baselines"
    _mk_night(d, 1, "nightly-20260925", 0.90, 0.55)          # pre-#1547: no key
    _mk_night(d, 2, "nightly-20260926", 0.88, 0.54,
              seeding={"enabled": True, "k": 3})

    gate = uptake.retrieval_gate(baselines_dir=d)
    assert gate["nights"] == 1, f"a k=3 night banded with an unrecorded one: {gate}"
    assert gate["nights_in_shape"] == 1, gate
    assert gate["shape"]["semantic_seeding"] == "enabled=True,k=3", gate["shape"]
    assert gate["latest"]["label"] == "nightly-20260926", gate
    for metric in ("doc_hit_rate", "ndcg10"):
        # The band's denominator has to say one as well, or the separation stops
        # being visible one level down where the floor is quoted.
        assert gate[metric]["n_nights"] == 1, (metric, gate[metric])


def test_the_published_shape_names_the_seeding_its_band_was_computed_over(tmp_path):
    """#1547 clause 4, second half: a one-night or excluded-night pool has to be
    readable as a regime boundary rather than as lost data.

    Shape is still chosen MODALLY — an odd newest night must not set the
    reference, which is what `test_retrieval_gate_ignores_nights_it_cannot_compare`
    pins for a 40-query arm — so three pre-#1547 nights plus one recording `k: 3`
    band as the three unrecorded nights and the newest night is excluded on
    regime. That exclusion is only honest if the block says which seeding the band
    was computed over: `nights: 3` beside `semantic_seeding: "unrecorded"` names
    it, where a shape block that stopped at `limit` would leave a reader to
    discover it by diffing the directory. (Whether the pre-2026-09-26 entity-side
    nights should be annotated as pre-re-base or dropped from the published
    window instead is a person's call, carried on #1547.)
    """
    d = tmp_path / "baselines"
    for i, lab in enumerate(("nightly-20260923", "nightly-20260924",
                             "nightly-20260925")):
        _mk_night(d, i, lab, 0.90 - 0.02 * i, 0.55 - 0.01 * i)
    _mk_night(d, 3, "nightly-20260926", 0.86, 0.53,
              seeding={"enabled": True, "k": 3})

    gate = uptake.retrieval_gate(baselines_dir=d)
    assert gate["shape"]["semantic_seeding"] == uptake.SEEDING_SHAPE_UNRECORDED
    assert gate["nights"] == 3, gate
    assert gate["nights_in_shape"] == 3, gate
    assert gate["latest"]["label"] == "nightly-20260925", (
        "the k=3 night is excluded from the unrecorded band; the shape block is "
        "what says so, and `latest` must not quietly claim the newest night")

    # A night recording seeding OFF is a third regime, not a synonym for the
    # unrecorded one: pooling them would invent the measurement it lacks. The
    # recorded night is the NEWEST here on purpose — with one night per shape the
    # modal choice breaks on mtime, so this also pins that a recorded `k=0` beats
    # an unrecorded night rather than merging with it.
    off = tmp_path / "off"
    _mk_night(off, 1, "nightly-20261001", 0.90, 0.55)          # no key
    _mk_night(off, 2, "nightly-20261002", 0.88, 0.54,
              seeding={"enabled": False, "k": 0})
    g2 = uptake.retrieval_gate(baselines_dir=off)
    assert g2["shape"]["semantic_seeding"] == "enabled=False,k=0", g2["shape"]
    assert g2["nights"] == 1 and g2["nights_in_shape"] == 1, g2
    assert g2["latest"]["label"] == "nightly-20261002", g2


def test_the_band_window_excludes_nights_older_than_the_window(tmp_path):
    """Comparability of run shape is a filter, not a window (#1220).

    The real directory on 2026-09-17 held fourteen same-shape nights whose
    `corpus.facts` had grown 205,693 -> 314,653: half the pool was scored against
    a corpus that no longer existed, and the band it published (0.15) was three
    times the newest-seven band (0.05). So a fixture of M+4 comparable nights,
    whose OLDEST is far outside the recent regime, must publish the band of the
    newest M — the 0.50 night must not reach `min`, and the band must be the
    window's spread, not the history's.
    """
    m = uptake.RETRIEVAL_GATE_WINDOW_NIGHTS
    total = m + 4
    d = tmp_path / "baselines"
    #: Oldest first. Index 0 is the obsolete-regime night; the rest alternate two
    #: rates so that ANY window of two or more holds both, keeping the expected
    #: band independent of the value chosen for M.
    nights = [("nightly-old-regime", 0.50, 0.200)] + [
        (f"nightly-{i:02d}", 0.95 if i % 2 else 0.90, 0.58 if i % 2 else 0.55)
        for i in range(1, total)]
    assert m < total, "the fixture is only a window test if the cap bites"
    for i, (lab, dh, nd) in enumerate(nights):
        _mk_night(d, i, lab, dh, nd)

    gate = uptake.retrieval_gate(baselines_dir=d)
    assert gate["nights_in_shape"] == total
    assert gate["nights"] == m
    assert gate["doc_hit_rate"]["n_nights"] == m
    assert gate["doc_hit_rate"]["window_nights"] == m

    doc = gate["doc_hit_rate"]
    newest_doc = [dh for _, dh, _ in nights][-m:]
    assert doc["min"] > 0.50, "the out-of-band night leaked into the band"
    assert doc["min"] == pytest.approx(min(newest_doc))
    assert doc["max"] == pytest.approx(max(newest_doc))
    assert doc["band"] == pytest.approx(max(newest_doc) - min(newest_doc))
    assert doc["band"] == pytest.approx(0.05)
    # The uncapped band this fixture would produce — 0.45, the number the cap
    # exists to refuse — is strictly wider than what is published.
    all_doc = [dh for _, dh, _ in nights]
    assert doc["band"] < max(all_doc) - min(all_doc)

    nd = gate["ndcg10"]
    newest_nd = [nd_ for _, _, nd_ in nights][-m:]
    assert nd["n_nights"] == m
    assert (nd["min"], nd["max"]) == (pytest.approx(min(newest_nd), abs=1e-4),
                                      pytest.approx(max(newest_nd), abs=1e-4))
    assert nd["band"] == pytest.approx(max(newest_nd) - min(newest_nd), abs=1e-4)


def test_a_saturated_or_spreadless_metric_says_its_floor_means_nothing(tmp_path):
    """A floor on a saturated rate is not a tolerance (#1220 symptom 2).

    `doc_hit_rate` read 1.0 on five straight nights to 2026-09-18 while
    `n_queries` was 20, so one query is 0.05 and the metric's own grid is coarser
    than any floor the gate could name; `latest - band` there is 1.0, which reads
    as the strictest gate ever passed and discriminates nothing at all. A
    one-night pool is the other spreadless shape — band 0 by definition. Both have
    to be reported as such, and every block states the grid the metric moves on.
    """
    pinned = tmp_path / "pinned"
    for i, (lab, dh, nd) in enumerate([("nightly-20260914", 1.0, 0.52),
                                       ("nightly-20260915", 1.0, 0.50),
                                       ("nightly-20260916", 1.0, 0.51)]):
        _mk_night(pinned, i, lab, dh, nd)
    gate = uptake.retrieval_gate(baselines_dir=pinned)
    doc = gate["doc_hit_rate"]
    assert doc["latest"] == 1.0 and doc["band"] == 0.0
    assert doc["at_ceiling"] is True, doc
    assert "ceiling" in doc["ceiling_reason"], doc
    assert doc["n_queries"] == 20                 # `_mk_baseline` stamps n_queries = limit
    assert doc["granularity"] == pytest.approx(0.05), "1/20: one query is five points"
    # Per metric, not per night: ndcg10 moved across the same three nights, so it
    # keeps a real band and a floor that means something.
    assert gate["ndcg10"]["at_ceiling"] is False
    assert gate["ndcg10"]["ceiling_reason"] is None

    solo = tmp_path / "solo"
    _mk_night(solo, 0, "nightly-alone", 0.72, 0.40)
    s = uptake.retrieval_gate(baselines_dir=solo)["doc_hit_rate"]
    assert s["latest"] == pytest.approx(0.72) and s["band"] == 0.0
    assert s["at_ceiling"] is True, "a one-night band is not a measured tolerance"
    assert "no spread" in s["ceiling_reason"], s

    discriminating = tmp_path / "spread"
    for i, (lab, dh, nd) in enumerate([("nightly-a", 0.95, 0.55), ("nightly-b", 0.88, 0.50)]):
        _mk_night(discriminating, i, lab, dh, nd)
    ok = uptake.retrieval_gate(baselines_dir=discriminating)["doc_hit_rate"]
    assert ok["at_ceiling"] is False and ok["ceiling_reason"] is None, ok
    assert ok["band"] == pytest.approx(0.07) and ok["granularity"] == pytest.approx(0.05)


def test_the_probe_carries_the_window_and_the_ceiling_flag_into_the_table(tmp_path):
    """Seam: the gate is computed here, the table is written there, and the
    consolidator that decides what to rewrite reads only the table.

    #1220's triage named how a key dies on that seam: `uptake_probe` used to copy
    four named top-level keys plus two whole metric blocks, so a new top-level
    field — the window count, which is the denominator of every `band` in the
    artifact — would be absent from every committed table while present in the
    gate's own return value, and only a key parked inside a metric block would
    arrive. The copy is now everything-but-the-machine-path, and this pins that
    the exclusion list is the only thing dropped, so the next key added upstream
    cannot go missing quietly.
    """
    import scripts.uptake_probe as probe

    d = tmp_path / "baselines"
    for i, (lab, dh, nd) in enumerate([("nightly-20260914", 1.0, 0.52),
                                       ("nightly-20260915", 1.0, 0.50)]):
        _mk_night(d, i, lab, dh, nd)
    gate = uptake.retrieval_gate(baselines_dir=d)
    block = probe.retrieval_gate_block(gate)

    assert block["window_nights"] == uptake.RETRIEVAL_GATE_WINDOW_NIGHTS
    assert block["nights"] == gate["nights"] == 2
    assert block["nights_in_shape"] == 2
    assert block["shape"] == gate["shape"], "which pool the band came from must travel"
    for metric in ("doc_hit_rate", "ndcg10"):
        assert block[metric] == gate[metric], f"{metric} block was not copied whole"
        assert block[metric]["window_nights"] == uptake.RETRIEVAL_GATE_WINDOW_NIGHTS
    assert block["doc_hit_rate"]["at_ceiling"] is True
    assert block["doc_hit_rate"]["granularity"] == pytest.approx(0.05)
    assert set(gate) - set(block) == set(probe.GATE_TABLE_EXCLUDED_KEYS)
    assert probe.GATE_TABLE_EXCLUDED_KEYS == ("baselines_dir",), \
        "the excluded key must be a machine-local path, not a measurement"


def test_a_directory_of_one_off_runs_raises_rather_than_banding(tmp_path):
    """Only a `nightly-*.json` is a night (#1220).

    The live baselines directory after the 2026-09-22 deletion held one surviving
    night beside A/B arms and a kg-rebuild `after` snapshot, all of them written
    at `limit=20, matches_production_defaults=True`, so the leftovers shared the
    night's modal shape. That is what made `basis = prod or nightly or rows`
    dangerous rather than merely untidy: as soon as a directory held no night,
    the leftovers answered for the nightly spread by themselves. Measured on this
    box before the change, `retrieval_gate('/tmp/t1220')` — the five non-night
    files copied out of that directory and nothing else — answered `nights: 5`,
    doc band 0.025, floor 0.691: a "measured tolerance" assembled from five
    one-off runs, each scored against whatever store was standing the minute it
    ran. The honest answer for zero nights is `NoBaselines`.
    """
    d = tmp_path / "baselines"
    for name, dh, nd in [("abba-A1-empty", 0.700, 0.400),
                         ("abba-B1-rebuild", 0.690, 0.390),
                         ("kg-rebuild-before", 0.660, 0.370),
                         ("rebuild-after", 0.716, 0.416)]:
        _mk_baseline(d / f"{name}-20260923.json", name, dh, nd)

    with pytest.raises(uptake.NoBaselines) as ei:
        uptake.retrieval_gate(baselines_dir=d)
    msg = str(ei.value)
    assert "nightly-*.json" in msg, msg
    assert "4 eval JSON" in msg, f"the refusal must name what it refused to use: {msg}"

    # One night added to the same directory is a one-night sample: the runs beside
    # it stay out of the band, and the count the reader sees is 1, not 5.
    _mk_night(d, 9, "nightly-20260923", 0.716, 0.416)
    gate = uptake.retrieval_gate(baselines_dir=d)
    assert gate["nights"] == 1, gate
    assert gate["nights_in_shape"] == 1, gate
    assert gate["doc_hit_rate"]["n_nights"] == 1, gate
    assert gate["latest"]["label"] == "nightly-20260923"


def test_a_missing_nightly_leaves_a_named_absence_in_the_table_not_a_crash(tmp_path):
    """Seam, other edge: the gate refusing is right, the probe dying is not.

    `retrieval_gate` now raises on a baselines directory with no night (#1220),
    and `uptake_probe.main` reads it *after* the classifier has been scored.
    Letting that refusal propagate would cost the whole `uptake-<date>.json` — the
    artifact the nightly consolidator reads its keep/archive decisions from — over
    a retrieval sample that has nothing to do with the classifier figures inside
    it. So the refusal is transcribed under `classifier.retrieval_gate` carrying
    its reason: an absence a reader can act on, not a missing key that reads like
    an oversight, and not a lost table.

    Both edges are asserted on the written artifact. `main` cannot be driven here
    (it scores the classifier against the secondary engine first), so what is
    pinned is the one function `main` calls to assign the key, with the value
    pushed through `write_table` and read back by path — the boundary the
    consolidation job actually reads, which a whitelist copy could cross while
    writing nothing.
    """
    import scripts.uptake_probe as probe

    def table_with_gate(out_dir: Path, classifier: dict) -> dict:
        path = uptake.write_table(
            uptake.build_uptake_table(turns=[_mk_turn(1, "wrong", "ok")],
                                      dispute_flags={"s1#1": True},
                                      memory_entries=[], skills_read={}),
            out_dir=out_dir, date="2026-09-24", classifier=classifier)
        return json.loads(path.read_text())

    d = tmp_path / "baselines"
    _mk_baseline(d / "abba-B1-rebuild-20260923.json", "abba-B1-rebuild", 0.690, 0.390)

    block = probe.attach_retrieval_gate({}, lambda: uptake.retrieval_gate(baselines_dir=d))
    assert set(block) == {probe.GATE_TABLE_KEY}, block
    field = block[probe.GATE_TABLE_KEY]
    assert set(field) == {probe.GATE_ABSENT_KEY}, field
    assert "nightly-*.json" in field[probe.GATE_ABSENT_KEY], field
    assert "doc_hit_rate" not in field and "floor" not in field, \
        f"a refused gate must not publish a tolerance: {field}"

    j = table_with_gate(tmp_path / "eval" / "absent", block)
    assert j["classifier"][probe.GATE_TABLE_KEY][probe.GATE_ABSENT_KEY] \
        .startswith("no nightly-*.json"), j["classifier"]

    # The other edge, same call and same artifact: nights present yield the whole
    # block under the same key, so the absent shape is that directory's state and
    # not how the key always looks. Two nights, because one night spans no spread
    # and would flag itself non-discriminating (correctly — see
    # test_a_saturated_or_spreadless_metric_says_its_floor_means_nothing).
    _mk_night(d, 11, "nightly-20260923", 0.950, 0.520)
    _mk_night(d, 12, "nightly-20260924", 0.900, 0.500)
    ok = probe.attach_retrieval_gate({}, lambda: uptake.retrieval_gate(baselines_dir=d))
    gate_blk = ok[probe.GATE_TABLE_KEY]
    assert probe.GATE_ABSENT_KEY not in gate_blk, ok
    assert gate_blk["nights"] == 2, gate_blk
    j2 = table_with_gate(tmp_path / "eval" / "present", ok)
    written = j2["classifier"]["retrieval_gate"]["doc_hit_rate"]
    assert written["window_nights"] == uptake.RETRIEVAL_GATE_WINDOW_NIGHTS, written
    assert written["at_ceiling"] is False and written["granularity"] == pytest.approx(0.05), written
    # floor = latest - band = 0.90 - (0.95 - 0.90)
    assert written["floor"] == pytest.approx(0.85), written


def test_the_skill_states_the_window_the_code_applies_and_the_ceiling_caveat():
    """The instruction a nightly run obeys is pinned to the code it describes.

    `skills/nightly-reflection-knowledge-write/SKILL.md` is read as procedure by the
    consolidation job, and its previous version told the reader the band was computed
    "over the newest 6 of the nightly files" at a time when the code applied no cap
    whatsoever — so every pass/fail claim written from that sentence cited a sample
    size nobody had measured. The skill lives in the vault, a live tree with no
    worktree, so nothing but a text assertion stops the retired sentence coming back
    (same shape as `tests/test_retrieval_eval_skill_ci_contract.py`). The number is
    asserted against the constant rather than typed in: change
    `RETRIEVAL_GATE_WINDOW_NIGHTS` and this goes red until the prose follows.
    """
    skill = (Path(os.environ.get("LLOYD_VAULT", Path.home() / "obsidian"))
             / "skills" / "nightly-reflection-knowledge-write" / "SKILL.md")
    if not skill.exists():
        # Same convention as the sibling this copies: the vault is not part of the
        # checkout, so its absence is a named skip, not an error.
        pytest.skip(f"vault skill not present at {skill}")
    text = skill.read_text()

    assert "newest 6" not in text, "the uncapped window is back in the instruction"
    assert f"the newest {uptake.RETRIEVAL_GATE_WINDOW_NIGHTS} comparable nights" in text, \
        "the skill no longer states the window retrieval_gate() applies"
    # The two readings that turn a floor into a meaningless number, both named.
    assert "at_ceiling" in text and "discriminates nothing" in text, \
        "the skill still tells the reader to quote a floor it cannot check"
    assert "granularity" in text, "no statement of the metric's own step size"
    # Only a nightly is a night, and zero nights is a refusal, not a band.
    assert "`nightly-*.json` only" in text and "NoBaselines" in text, \
        "the skill still describes a band that may be computed from A/B arms"


def test_live_nightly_band_is_a_capped_window_of_the_real_files():
    """The re-basing has to be true of the real files, not only of fixtures — and
    after #1220 the thing owed changed: the published band must equal the
    newest-M recomputation, never the whole history's.

    This guard used to assert `gate["doc_hit_rate"]["floor"] < 0.95`. That is not
    a statement about the window: with the doc leg at its ceiling and a 0.05 band,
    `latest - band` lands on exactly 0.95 and a strict `<` fails on correct
    behaviour — the trap the item's triage named. What is checkable instead is
    recomputed here from the files themselves, at whatever volume this box holds.

    Meaningful in both states rather than erroring on a clean checkout:
    `eval/baselines/` is gitignored, so a tree without it must still see the
    documented behaviour — `retrieval_gate()` refusing to invent a band.
    """
    assert uptake.RETRIEVAL_GATE_HARDCODE == 0.95  # the number being replaced
    baselines = uptake.lloyd_root() / "eval" / "baselines"
    # Nightly-only (#1220): the guard's old form was
    # `if not any(baselines.glob("*.json")): expect a refusal; return`, and it
    # carried the claim that the gate reads every `*.json` with metrics. That
    # sentence described `basis = prod or nightly or rows`, which is the fallback
    # this item removed — keeping it here would have pinned the bug as behaviour.
    # Both shapes are now distinguished on purpose: a directory with no JSON at
    # all is gitignored state a checkout may legitimately lack (the documented
    # refusal, then stop), while a directory with JSONs but no NIGHT is exactly
    # the 2026-09-23 state that used to publish a band from A/B arms, so it must
    # fail here rather than return quietly.
    if not any(baselines.glob("*.json")):
        with pytest.raises(uptake.NoBaselines):
            uptake.retrieval_gate()
        return
    assert any(baselines.glob("nightly-*.json")), (
        f"{baselines} holds eval JSONs but no nightly-*.json — the gate must "
        "refuse, and the leftovers must not answer for the nightly spread")
    gate = uptake.retrieval_gate()
    window = uptake.RETRIEVAL_GATE_WINDOW_NIGHTS

    def newest_values(metric: str) -> list[float]:
        """The same pool from the other direction: nightly, production-config,
        this shape's `limit` AND its `semantic_seeding`, newest first, capped at M
        — read off the files and sorted by mtime here, not returned by the
        function under test.

        The seeding term is part of the model as of #1547, not a nicety: the gate
        bands by it now, so a recomputation that stopped at `limit` would describe
        a different set of files than the one the band came over and the
        `n_nights`/`min`/`max` equality checks below would be comparing the gate
        against a pool it never used, red the first night that records a seeding.
        The token is re-derived here from the documented shape rather than read
        back through `uptake._seeding_shape_key`, so a producer whose token
        drifts from the published one is caught by this and not by itself.
        """
        rows = []
        for p in baselines.glob("nightly-*.json"):
            doc = json.loads(p.read_text())
            if doc.get("limit") != gate["shape"]["limit"]:
                continue
            if not bool(doc.get("matches_production_defaults")):
                continue
            rec = doc.get("semantic_seeding")
            token = (uptake.SEEDING_SHAPE_UNRECORDED
                     if not isinstance(rec, dict) or not isinstance(rec.get("k"), int)
                     or isinstance(rec.get("k"), bool)
                     else f"enabled={bool(rec.get('enabled'))},k={rec['k']}")
            if token != gate["shape"]["semantic_seeding"]:
                continue
            v = ((doc.get("summary") or {}).get("overall") or {}).get(metric)
            if isinstance(v, (int, float)):
                rows.append((p.stat().st_mtime, float(v)))
        rows.sort(reverse=True)
        return [v for _, v in rows][:window]

    assert gate["window_nights"] == window
    assert gate["nights"] == min(window, gate["nights_in_shape"]), gate
    for metric in ("doc_hit_rate", "ndcg10"):
        blk = gate[metric]
        vals = newest_values(metric)
        assert vals, f"no live {metric} value found under {baselines}"
        assert blk["n_nights"] == len(vals), (metric, blk, len(vals))
        assert blk["window_nights"] == window, metric
        assert blk["min"] == pytest.approx(min(vals), abs=1e-4), metric
        assert blk["max"] == pytest.approx(max(vals), abs=1e-4), metric
        assert blk["band"] == pytest.approx(max(vals) - min(vals), abs=1e-4), metric
        assert blk["floor"] == pytest.approx(blk["latest"] - blk["band"], abs=1e-4), metric
        n_q = blk.get("n_queries")
        assert isinstance(n_q, int) and n_q > 0, (
            f"{metric}: the live baseline records no n_queries, so the metric's "
            f"step size is unmeasurable and granularity cannot be checked: {blk}")
        assert blk["granularity"] == pytest.approx(1.0 / n_q, abs=1e-4), metric
        # Semantics, not the flag's own formula re-derived from the block's own
        # fields: `vals` is recomputed from the files above, newest first, so
        # `vals[0] >= 1.0` and `spread <= 0.0` are the two states the flag is
        # supposed to report, established without trusting `blk`.
        spread = max(vals) - min(vals)
        assert blk["at_ceiling"] is (vals[0] >= 1.0 or spread <= 0.0), \
            f"{metric}: rate {vals[0]} over a {spread}-wide pool, flagged {blk}"
        if blk["at_ceiling"]:
            # Which of the two fired has to be the one that holds: a saturated
            # rate says ceiling, a spreadless pool says spread.
            assert ("ceiling" in blk["ceiling_reason"]) is (vals[0] >= 1.0), \
                (metric, blk["ceiling_reason"], vals[0])

    # Two claims are about volume rather than shape, and this box held a single
    # nightly after the 2026-09-22 deletion: a named skip naming the floor is the
    # honest answer, and it heals as the nightly runs rebuild the sample.
    from tests._live_data import require_live_volume
    require_live_volume(sorted(baselines.glob("nightly-*.json")), 3, baselines,
                        "the nightly retrieval baselines")
    assert gate["nights"] >= 3, gate
    # The hard-coded 0.95 really would have fired on at least one pooled night —
    # the observation #801 was filed on, now scoped to the nights the band uses.
    assert gate["hardcoded_gate_would_have_failed_nights"] >= 1, gate


# ------------------------------------------------------------ emitted table -

def test_emitted_table_lands_under_eval_uptake_with_a_probe_timestamp(tmp_path):
    table = uptake.build_uptake_table(turns=[_mk_turn(1, "wrong", "ok", skills=["obsidian"])],
                                      dispute_flags={"s1#1": True}, memory_entries=[],
                                      skills_read={})
    path = uptake.write_table(table, out_dir=tmp_path / "eval" / "uptake", date="2026-09-11",
                             classifier={"precision": 0.9, "recall": 0.6})
    assert path.name == "uptake-2026-09-11.json"
    j = json.loads(path.read_text())
    assert j["probe_timestamp"]
    assert j["item"] == "552"
    assert j["classifier"]["precision"] is not None


def test_live_uptake_table_exists_and_carries_per_entry_keys():
    """The reproduction the triage named: `ls ~/lloyd/eval/uptake/` shows a
    dated JSON, and each row carries present_in_turns / disputes / dispute_rate.

    This asserts the *measured step-1/step-2 outcomes* off committed evidence, so
    the acceptance clauses are pinned by a test that can fail on any machine and
    does not need the secondary engine awake — the live-engine test next to it is
    the cross-check, not the pin.
    """
    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    assert files, "no dated uptake table under eval/uptake/"
    j = json.loads(files[-1].read_text())
    rows = j["entries"]
    assert rows, j
    for key in ("entry", "present_in_turns", "disputes", "dispute_rate", "presence_source"):
        assert key in rows[0], rows[0]
    assert re.match(r"\d{4}-\d{2}-\d{2}", j["probe_timestamp"])

    cls = j["classifier"]
    assert cls["threshold"] == 0.70
    # Step 2's stop condition, as actually measured and recorded.
    assert cls["precision"] is not None and cls["precision"] >= uptake.PRECISION_FLOOR, cls
    assert cls["n_positives"] >= 20, "acceptance needs >= 20 hand-labeled disputes"
    assert cls["recall"] is not None and cls["recall"] >= uptake.RECALL_FLOOR, cls
    assert cls["passed"] is True, cls
    # The exemplars were written against this corpus's shapes, so the number is
    # in-sample; the flag is what stops a reader quoting it as a clean holdout.
    assert cls["prompt_tuned_on_labels"] is True, cls
    # Coverage halves, each with the source that produced it.
    cov = j["coverage"]
    assert cov["user_md_entries"]["ratio"] >= 0.8, cov
    assert "435" in cov["active_skills"]["note"], cov


def test_the_committed_artifact_points_at_evidence_that_still_exists(tmp_path):
    """Every path in the committed report must resolve after the round that
    produced it is gone. The first artifact recorded `labels_file` inside
    `~/lloyd-work/SM_2026…`, which the abort deletes: an evidence pointer to a
    deleted worktree reads as auditable and is not."""
    import scripts.uptake_probe as probe

    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    labels_file = json.loads(files[-1].read_text())["classifier"]["labels_file"]
    assert labels_file, "no labels file recorded"
    assert not Path(labels_file).is_absolute(), (
        f"labels_file must be repo-relative, got {labels_file}")
    assert (REPO / labels_file).is_file(), f"{labels_file} does not resolve"
    assert probe._repo_relative(REPO / "eval" / "x.json") == "eval/x.json"
    assert Path(probe._repo_relative("/elsewhere/x.json")).is_absolute()


# ------------------------------------------------------- probe exit codes --



#: How many times the last `_replay_engine` stand-in was reached through each
#: entry point. A module-level handoff because `main()` is what calls the probe's
#: internals; the emit test reads it to prove the zero-shot pass and the replay
#: verifier ran THROUGH the fixture rather than past it to a live socket.
_REPLAY_CALLS: dict[str, dict[str, int]] = {}


def _always_not_engine(monkeypatch) -> None:
    """The always-NOT shape the item's own stop condition exists to catch.

    Both entry points again: the stop test reaches `classify_dispute_raw` through
    the zero-shot pass, and leaving that unpatched would have it asking the live
    slot what a fixture that never POSTs is supposed to have decided.
    """
    def fake(prev_assistant, user_text, *, transport=None, examples=True):
        return False

    def fake_raw(prev_assistant, user_text, *, transport=None, examples=True):
        return False, "NOT"

    monkeypatch.setattr(uptake, "classify_dispute", fake)
    monkeypatch.setattr(uptake, "classify_dispute_raw", fake_raw)


def test_probe_stops_below_the_precision_floor_and_writes_no_uptake_table(tmp_path, monkeypatch):
    """Step 2 is 'stop here if precision < 0.70', and the stop must be behavior:
    exit 3, a classifier report as the honest artifact, and — the part that
    matters — *no uptake table*. A table emitted by a classifier that cannot
    tell a correction from a new request is a machine for attributing blame at
    random, and the always-NOT shape is exactly what the committed corpus scores
    when the engine answers NOT to everything."""
    import scripts.uptake_probe as probe

    _always_not_engine(monkeypatch)
    out = tmp_path / "uptake"

    assert probe.main(["--eval-only", "--out-dir", str(out)]) == 3
    assert not list(out.glob("uptake-*.json")), "an eval-only run writes nothing"

    assert probe.main(["--out-dir", str(out), "--days", "30"]) == 3
    assert not list(out.glob("uptake-*.json")), "table written despite the failed floor"
    report = json.loads((out / "classifier-report.json").read_text())
    assert report["classifier"]["precision"] is None, "no positive predicted => no precision"
    assert report["classifier"]["passed"] is False




def test_coverage_states_the_reach_of_the_note_half(tmp_path):
    """`notes_seen: 0` was the most misleading number in the table, and the fix is
    not a disclaimer — it is a measured reach.

    Presence IS persisted (`role="subliminal"` messages), so the honest statement
    is not "unmeasurable" but "here is how many turns in this window actually
    persisted an injected block". A zero with `turns_with_block: 12/12` in front of
    it means notes were prefetched and never disputed; a zero with `0/12` means this
    window says nothing about notes at all. Those two readings are the difference
    between an entry being pruned and an entry being left alone, and before the
    reach went out neither could be told from the other.
    """
    table = uptake.build_uptake_table(
        turns=[_mk_turn(1, "wrong", "ok")],          # no injected block at all
        dispute_flags={"s1#1": True}, memory_entries=[], skills_read={})
    notes = table["coverage"]["prefetch_notes"]
    assert notes["entries_identified"] == 0
    assert notes["turns_total"] == 1
    # A turn built without any injected block: no note evidence, and the table says
    # so as *reach*, not as a verdict about the notes.
    assert notes["turns_with_block"] == 0
    assert notes["reach"] == 0.0
    assert notes["presence_source"] == uptake.NOTE_PRESENCE_UNREACHABLE
    assert "no_injected_block" in notes["presence_source"]
    assert "contributes no note evidence" in notes["note"]


def test_turns_that_persisted_no_injected_block_are_counted_as_no_reach(tmp_path):
    """The reach denominator must be turns, not blocks.

    Two blocks landing on one turn is one turn's worth of evidence. Counting blocks
    would report reach of 2/1 and let a table claim more coverage than it has.
    """
    a = _mk_turn(1, "wrong", "ok", ctx_titles=["knowledge/a.md"])
    a.injections_seen = 2                      # two blocks, one turn
    b = _mk_turn(2, "next thing", "ok", session="s2")
    table = uptake.build_uptake_table(
        turns=[a, b], dispute_flags={"s1#1": True}, memory_entries=[], skills_read={})
    notes = table["coverage"]["prefetch_notes"]
    assert notes["turns_with_block"] == 1 and notes["turns_total"] == 2
    assert notes["reach"] == pytest.approx(0.5)


def test_a_prefetched_note_is_attributed_when_evidence_does_exist():
    """The other direction: if a title *does* reach us it must be credited, so
    the honest-emptiness change cannot quietly disable the note half."""
    t = _mk_turn(1, "wrong, that note is stale", "Here is the note.",
                 ctx_titles=["knowledge/software/livekit.md"])
    table = uptake.build_uptake_table(turns=[t], dispute_flags={"s1#1": True},
                                      memory_entries=[], skills_read={})
    row = [r for r in table["entries"] if r["kind"] == "note"][0]
    assert row["presence_source"] == "prefetch:vault_context"
    assert row["present_in_turns"] == 1 and row["disputes"] == 1
    assert table["coverage"]["prefetch_notes"]["presence_source"] == "prefetch:vault_context"


def test_an_absent_store_is_reported_absent_and_never_created(tmp_path, monkeypatch):
    """A store path that does not exist must not become `duplicate_rows: 0`.

    `KGStore` opens lazily and creates the file. Run from an automod worktree,
    where `_pipeline/` is not checked out, that produced an empty 77 KB
    `kg.sqlite` beside the code and a table asserting the knowledge graph had no
    duplicate fact rows — a false clean bill of the exact shape the 2026-08-22
    wipe established is the worst possible report from this subsystem.
    """
    import app.paths as paths

    missing = tmp_path / "nope" / "kg.sqlite"
    monkeypatch.setattr(paths, "VAULT_KG_DB", missing)
    # Also point the live-tree fallback away from the real store, so this pins
    # "nothing found" rather than "found it in the live checkout after all".
    monkeypatch.setenv("LLOYD_ROOT", str(tmp_path))
    block = uptake.store_sizes()["facts_idx"]
    assert block.get("error"), block
    assert "duplicate_rows" not in block, block
    assert block.get("probed_at"), block
    assert not missing.exists(), "the probe created a store that was not there"


class _Undo:
    def __init__(self, obj, name, old):
        self.obj, self.name, self.old = obj, name, old

    def undo(self):
        setattr(self.obj, self.name, self.old)


def _patch_endpoint(module, url, model):
    old = module._endpoint
    module._endpoint = lambda job: (url, model)
    return _Undo(module, "_endpoint", old)


def test_recorded_engine_replies_reproduce_the_measured_precision():
    """The precision claim is replayable without the engine and without trusting
    the fixture's own verdict field.

    What this replaces: a test that asserted precision off numbers committed in
    the same diff, with the only fresh run stubbing the engine with an oracle that
    copied the labels. Neither could catch a wrong grader. Here the fixture
    carries the engine's **literal reply per turn** (`engine_raw`, captured by
    `run_classifier_eval(record_raw=True)`); the production parser turns those
    replies into verdicts and the confusion matrix is recomputed from scratch.
    Change `_parse_verdict` or the floors and this moves — including the verdict
    itself, which goes through the production gate rather than an inline boolean.
    """
    doc = json.loads((REPO / "eval" / "uptake" / "labels"
                      / "hand-2026-09-11.json").read_text())
    items, replay = doc["items"], doc["engine_replay"]
    assert len(items) >= 20, items
    assert all("engine_raw" in i for i in items), "a label without its reply cannot be replayed"

    predictions = {i["turn_id"]: bool(uptake._parse_verdict(i["engine_raw"])) for i in items}
    m = uptake.precision_recall([int(i["label"]) for i in items],
                                [int(predictions[i["turn_id"]]) for i in items])
    assert m["measured"] is True, m
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (
        replay["tp"], replay["fp"], replay["fn"], replay["tn"]), m
    assert m["precision"] == pytest.approx(replay["precision"], abs=1e-4), m
    assert m["recall"] == pytest.approx(replay["recall"], abs=1e-4), m
    assert m["n_positives"] >= 20, m
    assert uptake.classifier_clears_floors(m) is True, m
    # In-sample, disclosed rather than renamed away.
    assert replay["prompt_tuned_on_labels"] is True, replay
    assert replay["engine"], replay


def test_the_secondary_engine_http_seam_is_crossed_by_a_request_that_always_runs():
    """The loopback POST to `/v1/chat/completions`, tested without needing a model.

    The only test that used to cross this seam self-skipped whenever the engine
    was silent, so a gate run could pass with the request builder, the endpoint
    resolver and the response parse all untouched. Here a real HTTP server on a
    real socket answers a canned completion and `classify_dispute_raw` runs with
    `transport=None` — the actual `_post_secondary`, `urllib.request`, and the
    `choices[0]["message"]["content"]` extraction. It fails on a payload or
    endpoint regression with no model anywhere in the picture.
    """
    import http.server
    import threading
    import app.secondary_models as sm

    seen: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self, message):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen["path"] = self.path
            seen["ctype"] = self.headers.get("Content-Type")
            seen["body"] = json.loads(raw)
            payload = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):  # noqa: N802
            self._answer({"role": "assistant", "content": "DISPUTE"})

        def log_message(self, *args):  # silence
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    undo = _patch_endpoint(sm, f"http://127.0.0.1:{server.server_address[1]}"
                              "/v1/chat/completions", "served-model-name")
    try:
        verdict, raw = uptake.classify_dispute_raw("Deployed it.", "no, that broke login")
        assert verdict is True and raw == "DISPUTE", (verdict, raw)
        body = seen["body"]
        assert seen["path"] == "/v1/chat/completions", seen
        assert seen["ctype"] == "application/json", seen
        # The resolver's model name wins over the payload's `"secondary"`: llama.cpp
        # 404s the alias, and nothing read this body before now.
        assert body["model"] == "served-model-name", body
        assert body["temperature"] == 0.0 and body["max_tokens"] == 16, body
        assert body["chat_template_kwargs"] == {"enable_thinking": False}, body
        assert uptake._CLASSIFY_SYSTEM in body["messages"][0]["content"], body
        assert "no, that broke login" in body["messages"][1]["content"], body
    finally:
        undo.undo()
        server.shutdown()
        server.server_close()

    # A reply that is only a reasoning stream — what this model gives with thinking
    # enabled — yields no verdict rather than a false "no dispute".
    class ReasoningOnly(Handler):
        def do_POST(self):  # noqa: N802
            self._answer({"role": "assistant", "content": "",
                          "reasoning_content": "The user is happy, so NOT"})

    quiet = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ReasoningOnly)
    threading.Thread(target=quiet.serve_forever, daemon=True).start()
    undo = _patch_endpoint(sm, f"http://127.0.0.1:{quiet.server_address[1]}"
                              "/v1/chat/completions", "m")
    try:
        assert uptake.classify_dispute("ok", "ok thanks") is None
    finally:
        undo.undo()
        quiet.shutdown()
        quiet.server_close()


def test_skill_presence_survives_the_event_log_writer_it_is_read_from(tmp_path, monkeypatch):
    """The reader's other half: a real file written by the real writer.

    `skills_read_by_session` parses JSON that `app.event_log.log_event` wrote in a
    different process, from a payload shape that writer owns — `data.args` is a
    JSON *string* nested inside the record. Author-written line fixtures pin only
    that the reader agrees with itself, so a writer-side rename of `data.args`, of
    the event name, or of `data.source` would silently zero the entire skill half
    of the table with every such fixture still green. This writes through the
    production writer and reads it back.
    """
    from app import event_log

    monkeypatch.setattr(event_log, "EVENT_LOGS_DIR", tmp_path / "event_logs")
    monkeypatch.setattr(event_log, "BLOBS_DIR", tmp_path / "event_logs" / "blobs")

    event_log.log_event("sess9", uptake.TURN_EVENT, {"source": "user", "len": 30})
    event_log.log_event("sess9", "brain1.tool_call_proposed", {
        "tool_call_id": "c1", "name": "skills_read",
        "args": json.dumps({"name": "voice-mode"}), "context_tokens": 1200})
    event_log.log_event("sess9", uptake.TURN_EVENT, {"source": "user", "len": 22})
    event_log.log_event("sess9", uptake.TURN_EVENT, {"source": "ambient", "len": 22})
    # A line that does not parse must not be fatal: these files contain them.
    with (tmp_path / "event_logs" / "sess9.events.jsonl").open("a") as fh:
        fh.write("{not json\n")
    event_log.log_event("sess9", "brain1.tool_call_proposed", {
        "name": "Read", "args": '{"file_path": "/tmp/x"}'})

    assert (tmp_path / "event_logs" / "sess9.events.jsonl").is_file()
    assert uptake.TURN_EVENT == "brain1.user_prompt_received"
    assert uptake.skills_read_by_session(root=tmp_path) == {"sess9": {"voice-mode": 1}}


def test_an_unpaired_store_count_is_refused_where_the_table_is_written(tmp_path):
    """The guard, not only its output. The artifact-level assertion only ever saw
    a well-formed block, so `write_table`'s refusal could be deleted with the
    suite green."""
    with pytest.raises(ValueError, match="probed_at"):
        uptake.write_table({"entries": []}, out_dir=tmp_path / "uptake",
                           stores={"facts_idx": {"duplicate_group_count": 12}})
    assert not (tmp_path / "uptake").exists(), "a refused write leaves no file behind"
    out = uptake.write_table({"entries": []}, out_dir=tmp_path / "uptake",
                             stores={"facts_idx": {"fact_rows": 3,
                                                   "probed_at": "2026-09-11T09:00:00+00:00"}})
    assert json.loads(out.read_text())["stores"]["facts_idx"]["fact_rows"] == 3


def test_store_size_figures_in_the_table_are_always_paired_with_a_timestamp():
    """Standing rule in this repo: a KG or facts count with no probe timestamp
    is a stale figure wearing a current one. The table must not be able to
    carry one without it."""
    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    j = json.loads(files[-1].read_text())
    stores = j.get("stores")
    # `for name, block in (j.get("stores") or {}).items()` iterated zero times on
    # a table with no stores block and still reported green — the loop was a
    # no-op that looked like a check. Require the block, and require the store
    # the item quotes duplicate counts from.
    assert isinstance(stores, dict) and stores, "committed table carries no stores block"
    assert "facts_idx" in stores, sorted(stores)
    quoted = 0
    for name, block in stores.items():
        numbers = {k: v for k, v in block.items() if isinstance(v, int)}
        if numbers:
            quoted += len(numbers)
            assert block.get("probed_at"), f"{name}{sorted(numbers)} has no probe timestamp"
    assert quoted, "no store figure at all — nothing here pins the timestamp rule"


# ------------------------------------------------- labeled corpus integrity -

def test_hand_labeled_corpus_covers_the_item_s_minimum():
    labels = uptake.load_labels()
    positives = [l for l in labels if l["label"] == 1]
    assert len(labels) >= 40, len(labels)
    assert len(positives) >= 20, len(positives)
    # Every label must point at a turn id of the documented shape, so a later
    # run can re-open the transcript behind it.
    for l in labels:
        assert re.match(r"^.+#\d+$", l["turn_id"]), l
    # `labeled_by` used to be asserted equal to a string here. That assertion
    # could not fail for a fabricated corpus: the string is part of the file it is
    # checking. The claim "this was hand-labeled from real turns" is falsifiable
    # only against the transcript store, which is what the test below does.




def test_validate_labels_fails_on_a_fabricated_or_stale_label(tmp_path, monkeypatch):
    """The check must be able to fail, in both directions."""
    real = uptake.Turn(session="s9", ts="2026-09-01T00:00:00+00:00",
                       user_text="the retry budget is not what you said it is",
                       prev_assistant="I raised the retry budget to 5 and rebuilt the "
                                      "client so the timeout no longer compounds.",
                       ordinal=1, session_source=None)
    index = {"s9#1": real}
    good = {"turn_id": "s9#1", "label": 1,
            "user_text": real.user_text, "prev_assistant": real.prev_assistant}
    assert uptake.validate_labels([good], index)["ok"] is True

    invented = {**good, "user_text": "a sentence no transcript contains " + "x" * 40}
    out = uptake.validate_labels([invented], index)
    assert out["ok"] is False and out["excerpt_mismatch_turn_ids"] == ["s9#1"], out

    rolled_off = {"turn_id": "s9#2", "label": 0, "user_text": "anything",
                  "prev_assistant": "anything"}
    out = uptake.validate_labels([rolled_off], index)
    # An unresolvable label is a FAILURE, not a skip: a corpus that shrinks as
    # transcripts age out would quietly re-weight the precision figure.
    assert out["ok"] is False and out["unresolved_turn_ids"] == ["s9#2"], out

    # A truncated excerpt is legitimate — it is the beginning of the real turn.
    long_prefix = real.prev_assistant[:45]
    assert len(long_prefix) >= 40
    assert uptake.validate_labels([{**good, "prev_assistant": long_prefix}],
                                  index)["ok"] is True


def test_live_engine_scores_the_corpus_and_reports_every_way_precision_was_measured():
    """Step 2's stop condition against the real secondary engine.

    Three numbers must clear the floor before this passes, because the artifact
    quotes all three and the first one alone flatters the prompt:

      * `metrics`    — the deployed prompt over the whole labeled corpus (in-sample:
                       `prompt_tuned_on_labels: true`, the exemplars were written
                       against these turn shapes);
      * `holdout`    — the labeled turns the prompt does not quote;
      * `zero_shot`  — those same turns with the few-shot block removed, which is
                       the only figure the prompt cannot have read out of itself;
      * `pipeline`   — screen + classifier together, reported but not gated: the
                       cue screen drops most labeled disputes, and the item's stop
                       condition is about the classifier. Its recall is what the
                       table's lower-bound note is computed from.
    """
    import scripts.uptake_probe as probe
    from app.config import CONFIG

    # THREE states, not two, and this one is checked first.
    #
    # `secondary_enabled: false` does not merely stop the engine — it makes
    # `resolve_model_alias` rewrite secondary -> primary, so `secondary_endpoint()`
    # returns :8096 and every assertion below would be measured against the
    # PRIMARY and reported under a test named for the secondary. That is not a
    # missing measurement, it is a confidently wrong one, and it is strictly
    # worse than the red this replaces: on 2026-09-20, with the slot switched
    # off, this test went GREEN on the live tree in 15s having scored the wrong
    # engine, while the worktrees the gate cuts still read `true` from the
    # committed config.yaml and hard-failed on :8091.
    #
    # So a retired slot skips, and the skip is not the escape hatch the comment
    # below rejects: that one hid a silent engine the deployment still expected
    # to be up. This one records that the subject of the measurement has been
    # taken out of service, which is a fact about the tree, read from the same
    # tracked switch the gate's worktree reads.
    if not CONFIG.get("secondary_enabled", False):
        pytest.skip(
            "secondary_enabled is false: the slot this measures is retired and "
            "the alias now resolves to primary, so a run here would score the "
            "wrong engine. Re-enable the slot to re-take Step 2's measurement.")

    # No skip, and no environment opt-out. An earlier shape asserted whichever
    # branch it landed in and recorded nothing about which, so a gate run that
    # caught :8091 asleep reported a pass on a measurement that had not happened;
    # the escape hatch added to fix that turned a missing measurement into a
    # green-skipped one, which is the same claim with a log line attached. The
    # fail-closed behaviour of a silent engine is pinned hermetically by the
    # stand-in-transport tests above, so failing here costs no coverage.
    if uptake.classify_dispute("Built it, works now.", "it 404s on me") is None:
        pytest.fail(f"secondary engine at {uptake.secondary_endpoint()} is not answering. "
                    "Step 2's acceptance measurement cannot be claimed without it: "
                    "start the secondary slot before gating this.")

    result = probe.run_classifier_eval()
    m = result["metrics"]
    assert m["n_labels_unresolvable"] == 0, m
    assert m["n_unanswered"] == 0, m
    assert m["n_positives"] >= 20, m
    assert m["precision"] is not None and m["precision"] >= uptake.PRECISION_FLOOR, m
    assert m["recall"] is not None and m["recall"] >= uptake.RECALL_FLOOR, m

    ho, zs = result["holdout"], result["zero_shot"]
    assert ho["measured"] is True and zs["measured"] is True, (ho, zs)
    assert ho["n"] >= 20, f"holdout below the item's 20-item minimum: {ho}"
    assert ho["n_positives"] >= 10, ho
    assert ho["precision"] >= uptake.PRECISION_FLOOR, ho
    assert zs["precision"] >= uptake.PRECISION_FLOOR, zs
    assert zs["n"] == ho["n"], (zs, ho)     # same turns, block removed
    assert result["split_note"], "contamination must be described, not assumed away"

    pl = result["pipeline"]
    assert pl["measured"] is True and pl["screened"] > 0, pl
    # The screen can only lose recall: every pipeline true positive is also a
    # classifier true positive, so the deployed number is bounded by the model's.
    assert pl["tp"] <= m["tp"], (pl, m)
    assert pl["recall"] <= m["recall"], (pl, m)

    assert result["labels_ok"] is True, result["labels_check"]
    assert result["passed"] is True, {k: result[k] for k in
                                      ("metrics", "holdout", "zero_shot", "pipeline")}




@pytest.mark.live_vault
def test_knowledge_write_skill_cites_a_per_entry_uptake_figure():
    """The consolidator is the consumer. If it does not read the table, the
    measurement is a dashboard and the blind rewrite continues."""
    skill = Path.home() / "obsidian/skills/nightly-reflection-knowledge-write/SKILL.md"
    text = skill.read_text()
    assert "eval/uptake" in text
    assert "dispute_rate" in text
    assert re.search(r"uptake-\d{4}-\d{2}-\d{2}\.json|uptake-\*", text), "no dated table cited"
    # The re-based gate has to be readable from the file too, or the skill keeps
    # quoting a threshold the noise band already broke.
    assert "retrieval_gate" in text or "noise band" in text


@pytest.mark.live_vault
def test_the_skill_s_descriptions_of_the_table_match_what_the_code_emits():
    """Seam: the consumer lives in the vault, the producer lives here, and the
    only thing between them is prose. `nightly-reflection-knowledge-write` names
    the row fields and the `presence_source` values it expects to read; if the
    code renames one, the consolidator reads `undefined`, decides on nothing, and
    its completion note still looks like it cited a figure. So the skill's own
    vocabulary is checked against the committed artifact, not against memory."""
    skill = Path.home() / "obsidian/skills/nightly-reflection-knowledge-write/SKILL.md"
    text = skill.read_text()
    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    j = json.loads(files[-1].read_text())
    row_keys = set(j["entries"][0])
    sources = {r["presence_source"] for r in j["entries"]}

    # Every backticked field name the skill lists for a row must be a real key.
    cited = set(re.findall(r"`([a-z_]{4,})`", text))
    row_fields = cited & {"present_in_turns", "disputes", "dispute_rate",
                          "weighted_disputes", "presence_source"}
    assert len(row_fields) >= 4, f"skill cites no row contract: {sorted(cited)[:12]}"
    missing = row_fields - row_keys
    assert not missing, f"skill cites fields the table does not emit: {missing}"

    # Every presence_source the skill quotes must be one the code really writes.
    quoted = {s for s in ("always_in_force:system_prompt", "prefetch:vault_context")
              if s in text}
    assert quoted, "skill quotes no presence_source at all"
    for src in quoted:
        emitted = any(s.startswith(src) for s in sources)
        # A source may legitimately have no rows, but only if the table says so
        # in so many words. "The skill describes it and the data is silent" is
        # how an unmeasurable half turns into a clean-looking zero.
        declared_unevaluable = (
            src.startswith("prefetch:")
            and j["coverage"]["prefetch_notes"]["presence_source"]
            == uptake.NOTE_PRESENCE_UNREACHABLE)
        assert emitted or declared_unevaluable, (
            f"skill cites {src}; table emits {sorted(sources)} and does not "
            "declare that half unevaluable")

    # And the gate it is told to run must be callable under that name.
    if "app.uptake" in text and "retrieval_gate" in text:
        from app.uptake import retrieval_gate  # noqa: F401
        assert callable(retrieval_gate)


# ------------------------------------------- presence: the persisted block --

#: What `app/routers/_messages_subliminal.py:144` + the writer actually put in a
#: transcript: one message per injected block, role "subliminal", content a list of
#: text blocks. Verified against the live store before this was written — 59 such
#: blocks in the 120 most recent session files, 48 carrying a note list.
def _subliminal_block(skills=(), titles=()):
    parts = []
    parts.extend(f'<skill name="{s}" score="9.0">\nbody of {s}\n</skill>' for s in skills)
    if titles:
        parts.append("<vault-context>\n"
                     + "\n".join(f"- **{t}** (score: 1.00): excerpt" for t in titles)
                     + "\n</vault-context>")
    return {"role": "subliminal",
            "content": [{"type": "text", "text": "\n".join(parts)}],
            "timestamp": "2026-09-10T01:00:05+00:00"}


def test_the_persisted_subliminal_message_is_what_binds_presence(tmp_path):
    """Presence evidence lives in its OWN message, not in a field on the user turn.

    The reader used to look for `msg["subliminal"]` on the user message, which the
    writer never emits — so every skill/note lookup in the module parsed nothing,
    `skill_injections` and `vault_context` were empty on every real transcript, and
    the table's note half concluded the data did not exist. This fixture is the
    shape on disk; the old reader returns empty lists against it, which is what
    makes the assertion able to fail.
    """
    _write_session(tmp_path, "s5", [
        _user("do the thing"),
        _subliminal_block(skills=["voice-mode"], titles=["knowledge/software/x.md"]),
        _asst("on it"),
    ])
    turn = uptake.human_turns(root=tmp_path)[0]
    assert turn.injected_skills == ["voice-mode"], turn.injected_skills
    assert turn.vault_context == ["knowledge/software/x.md"], turn.vault_context
    assert turn.injections_seen == 1


def test_a_subliminal_block_never_creates_a_turn_of_its_own(tmp_path):
    """The block is evidence attached to a turn, not a user turn.

    Counting it as one would let a session with three injected blocks report three
    human turns of presence for one request — and a block with no preceding user
    turn has nothing to be evidence about.
    """
    _write_session(tmp_path, "s6", [
        _subliminal_block(titles=["knowledge/orphan.md"]),      # no turn to bind to
        _user("first"),
        _subliminal_block(titles=["knowledge/a.md"]),
        _subliminal_block(titles=["knowledge/b.md"]),           # two blocks, one turn
        _asst("ok"),
        _user("second"),
    ])
    turns = uptake.human_turns(root=tmp_path)
    assert [t.user_text for t in turns] == ["first", "second"], turns
    assert turns[0].injections_seen == 2
    assert turns[0].vault_context == ["knowledge/a.md", "knowledge/b.md"]
    assert turns[1].vault_context == [] and turns[1].injections_seen == 0


@pytest.mark.asyncio
async def test_the_production_writer_and_the_reader_agree_on_one_block(tmp_path,
                                                                       monkeypatch):
    """The seam, crossed by the thing that writes and the thing that reads.

    Every other presence test builds a transcript by hand, so all it proves is that
    this module parses its own fixture. This one writes through
    `app.sessions_io._append_messages` — the function that actually produces
    `sessions/*.json`, with its schema repair, its dedupe against an in-memory
    transcript, and its per-file lock — and reads the file back through
    `uptake.human_turns`. If the writer ever nests the block, renames the role, or
    drops the content list, this fails here instead of quietly emptying the
    skill/note halves of the table.
    """
    import app.sessions_io as sio

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(sio, "SESSIONS_DIR", sessions)
    await sio._save_session_meta("s7", "test-model", "do the thing")
    await sio._append_messages("s7", [
        {"role": "user", "content": "do the thing",
         "timestamp": "2026-09-10T01:00:00+00:00"},
        _subliminal_block(skills=["voice-mode"], titles=["knowledge/a.md"]),
        {"role": "assistant", "content": "on it",
         "timestamp": "2026-09-10T01:00:10+00:00"},
    ])
    on_disk = json.loads((sessions / "s7.json").read_text())
    roles = [m.get("role") for m in on_disk["messages"]]
    assert "subliminal" in roles, f"the writer did not persist the block: {roles}"

    turns = uptake.human_turns(root=tmp_path)
    assert len(turns) == 1, turns
    assert turns[0].injected_skills == ["voice-mode"], turns[0].injected_skills
    assert turns[0].vault_context == ["knowledge/a.md"], turns[0].vault_context
    # And the note half of a real table built from that file is no longer "0 notes,
    # unevaluable": it is one note, present, with reach to say so.
    table = uptake.build_uptake_table(
        turns=turns, dispute_flags={}, memory_entries=[], skills_read={})
    assert table["coverage"]["prefetch_notes"]["entries_identified"] == 1
    assert table["coverage"]["prefetch_notes"]["presence_source"] == "prefetch:vault_context"


# --------------------------------------- holdout, zero-shot and the gate ----

def test_prompt_leak_finds_quoted_cases_including_the_short_ones():
    """`continue` is two words, and a ≥4-word window cannot see it.

    The exemplars encode exactly that case, so if short turns were exempt from the
    leak check the most contaminated items in the corpus would be the ones declared
    held out — which would make the holdout number flattering in the one direction
    the item's own risk section warns about.
    """
    prompt = "Rules here.\nWorked examples\nUSER: please continue\nLLM: NOT\n"
    assert uptake.prompt_leak("please continue", prompt) == "please continue"
    assert uptake.prompt_leak("continue", prompt) == "continue"
    assert uptake.prompt_leak("please carry on", prompt) is None
    assert uptake.prompt_leak(
        "the deploy finished but the pods are crash-looping", prompt) is None


def test_holdout_split_moves_quoted_turns_to_dev_and_keeps_them_visible():
    quoted = {"turn_id": "a#1", "label": 1, "user_text": "please continue"}
    clean = {"turn_id": "a#2", "label": 1,
             "user_text": "the retry budget is not what you told me it was"}
    prompt = uptake._CLASSIFY_RULES + "\nWorked examples\nUSER: please continue\n"
    split = uptake.holdout_split([quoted, clean], prompt=prompt)
    assert [i["turn_id"] for i in split["holdout"]] == ["a#2"], split
    assert split["dev"][0]["prompt_leak"] == "please continue"
    assert split["n_holdout_positives"] == 1
    assert "verbatim" in split["note"]


def test_pipeline_confusion_counts_a_screened_away_dispute_as_a_miss():
    """The screen's misses are the pipeline's misses.

    Grading the classifier by calling it directly on every labeled turn reported
    recall 0.57 for a system measured end to end at 0.39, and the table's
    lower-bound note was computed from the flattering one.
    """
    # 4 labeled positives; the screen forwards 2; the classifier gets both right.
    m = uptake.pipeline_confusion([1, 1, 1, 1], [1, 1, 0, 0], [True, True, None, None])
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 2), m
    assert m["precision"] == 1.0 and m["recall"] == 0.5, m
    assert m["screened"] == 2 and m["unanswered"] == 0, m
    # An engine that will not answer is not a correct negative.
    m2 = uptake.pipeline_confusion([1, 1], [1, 1], [None, True])
    assert m2["unanswered"] == 1 and m2["tp"] == 1 and m2["fn"] == 1, m2
    with pytest.raises(ValueError):
        uptake.pipeline_confusion([1], [1], [])


def test_examples_false_is_what_strips_the_few_shot_block():
    """`zero_shot` means "block removed" only if the call really removes it.

    Captures the payload instead of calling the model: if the flag ever stops being
    threaded, the unexemplified number silently becomes a second copy of the
    in-sample one and the gate that exists to catch in-sample precision is grading
    a tautology.
    """
    seen: list[dict] = []

    def fake(payload):
        seen.append(payload)
        return {"choices": [{"text": "NOT"}]}

    uptake.classify_dispute("prev", "next", transport=fake, examples=False)
    assert "Worked examples" not in seen[-1]["messages"][0]["content"], seen[-1]
    uptake.classify_dispute("prev", "next", transport=fake)
    assert "Worked examples" in seen[-1]["messages"][0]["content"]
    assert uptake._CLASSIFY_RULES and uptake._CLASSIFY_EXAMPLES
    assert uptake._CLASSIFY_SYSTEM.startswith(uptake._CLASSIFY_RULES)
    assert uptake._CLASSIFY_EXAMPLES in uptake._CLASSIFY_SYSTEM
    assert uptake._EXAMPLE_MARKER not in uptake._CLASSIFY_RULES


def test_measurement_gate_refuses_any_measurement_that_is_missing_or_weak():
    """Step 2's stop condition, one clause per way the single number used to lie."""
    strong = {"precision": 1.0, "recall": 0.55, "measured": True, "n_positives": 23}
    full = {"labels_ok": True, "classifier": strong,
            "holdout": {"precision": 1.0, "measured": True},
            "zero_shot": {"precision": 0.9, "measured": True}}
    assert uptake.measurement_clears_floors(full) is True
    assert uptake.measurement_clears_floors(None) is False

    assert uptake.measurement_clears_floors({**full, "labels_ok": False}) is False
    assert uptake.measurement_clears_floors(
        {**full, "classifier": {**strong, "precision": 0.6}}) is False
    assert uptake.measurement_clears_floors(
        {**full, "classifier": {**strong, "recall": 0.2, "precision": 1.0}}) is False
    # Not measured is not passed: an absent holdout used to be indistinguishable
    # from a holdout that cleared.
    for key in ("holdout", "zero_shot"):
        assert uptake.measurement_clears_floors({**full, key: None}) is False, key
        assert uptake.measurement_clears_floors(
            {**full, key: {"measured": False, "precision": None}}) is False, key
        assert uptake.measurement_clears_floors(
            {**full, key: {"measured": True, "precision": 0.5}}) is False, key


@pytest.mark.asyncio
async def test_zero_shot_pass_reports_unmeasured_when_the_engine_is_silent(tmp_path):
    """An absent grader must not become a precision of 0.0 or of 1.0.

    0.0 would fail the floor for an environmental reason and send the next run
    hunting for a model bug; 1.0 would pass the floor for no reason at all. This is
    the hermetic half of what the live test refuses to assert by inference.
    """
    import scripts.uptake_probe as probe

    turn = uptake.Turn(session="s1", ts="2026-09-01T00:00:00+00:00",
                       user_text="that is not what it does", prev_assistant="done",
                       ordinal=1, session_source=None)
    silent = lambda payload: (_ for _ in ()).throw(OSError("connection refused"))
    out = probe._zero_shot_pass({"s1#1"}, {"s1#1": turn}, {"s1#1": 1}, transport=silent)
    assert out["measured"] is False and out["precision"] is None, out
    assert uptake.measurement_clears_floors(
        {"labels_ok": True, "classifier": {"precision": 1.0, "recall": 1.0,
                                           "measured": True, "n_positives": 20},
         "holdout": {"precision": 1.0, "measured": True},
         "zero_shot": out}) is False


def test_store_sizes_counts_real_duplicate_rows_from_a_populated_store(tmp_path,
                                                                       monkeypatch):
    """The duplicate-row figure was pinned only on its absent branch.

    The other branch runs a `GROUP BY text_hash` query that no fixture ever
    exercised, so a broken query, a wrong column, or a store that reports one row
    per group would all have passed. `facts_idx` duplication is the number the item
    quotes (8,102 dup rows) and the number a consolidation is judged against, so it
    gets a store with duplicates in it — populated through `app.kg_store`, because
    nothing in this tree is allowed to open `kg.sqlite` any other way.
    """
    import app.paths
    from app.kg_store import KGStore

    facts = tmp_path / "facts"
    db = tmp_path / "kg.sqlite"

    def write(entity, category, facts_list):
        import yaml
        d = facts / entity
        d.mkdir(parents=True, exist_ok=True)
        fm = {"type": "facts", "entity": entity, "category": category,
              "facts": facts_list}
        p = d / f"{entity}-{category}.md"
        p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity} - {category}\n")
        return p

    # The same fact TEXT in two files: that is what `text_hash` duplication means
    # here (the per-file `fact_id` counter is not an identity — see MEMORY.md).
    write("Lloyd", "state", [
        {"id": "stat-001", "fact": "shared claim", "created_at": "2026-01-01"},
        {"id": "stat-002", "fact": "distinct claim", "created_at": "2026-01-01"}])
    write("Lloyd", "goal", [
        {"id": "goal-001", "fact": "shared claim", "created_at": "2026-01-01"}])
    st = KGStore(db)
    try:
        st.facts_idx.reindex(root=facts)
        assert st.facts_idx.count() == 3, st.facts_idx.count()
    finally:
        st.close()

    monkeypatch.setattr(app.paths, "VAULT_KG_DB", db)
    out = uptake.store_sizes()
    f = out["facts_idx"]
    assert f.get("error") is None, f
    assert f["fact_rows"] == 3, f
    assert f["duplicate_text_hash_groups"] == 1, f
    assert f["duplicate_rows"] == 1, f
    assert f["probed_at"], "a count without its timestamp is the defect, not the number"
    assert "ages" in f["note"]


def test_the_secondary_url_the_measurement_uses_is_the_one_the_tree_resolves():
    """`_endpoint()` is patched out everywhere else, so nothing pinned which slot the
    acceptance measurement actually runs against.

    A classifier graded against the wrong engine is a precision figure about a model
    nobody serves, and `uptake` does not own that URL — `app.secondary_models` does.
    Called unpached, so resolution itself is what is asserted, plus that the URL
    uptake would post to is the one the resolver returns.
    """
    from app.secondary_models import _endpoint

    url, model = _endpoint("uptake")
    assert url.startswith(("http://", "https://")), url
    assert urlparse(url).port, f"no port in a localhost slot URL: {url}"
    assert model, "an empty model name means the slot was never resolved"
    assert uptake.secondary_endpoint() == url, (uptake.secondary_endpoint(), url)

    # And the resolver's failure mode is describable, not exception-shaped: a
    # misconfigured slot must be reportable, because a bare exception in a test
    # message reads as "the test is broken" rather than "the engine is not there".
    def boom(job):
        raise RuntimeError("no slot configured")
    import app.secondary_models as sm
    saved = sm._endpoint
    try:
        sm._endpoint = boom
        described = uptake.secondary_endpoint()
    finally:
        sm._endpoint = saved
    assert described.startswith("<unresolved:") and "no slot configured" in described


def test_a_worktree_with_no_sessions_falls_back_instead_of_reporting_a_clean_table(tmp_path):
    """The fallback that keeps a round from measuring nothing and calling it health.

    A gate runs in a worktree whose `sessions/` is empty or absent. Without the
    live-checkout fallback the probe would read zero turns, find zero disputes, and
    write a flawless uptake table over an empty corpus — the shape of every guard in
    this tree that reads its own missing input and reports a verdict. This is the
    root every `sessions/`, `event_logs/` and `eval/baselines/` read hangs off, and
    no test crossed it.
    """
    import app.uptake as U

    # A real stand-in for a round's worktree: the directory exists, `sessions/`
    # exists inside it, and there is nothing in there to read.
    empty = tmp_path / "worktree"
    (empty / "sessions").mkdir(parents=True)
    assert U._has_sessions(empty) is False, "fixture must have an EMPTY sessions dir"
    resolved = U.lloyd_root()
    assert resolved != empty, "fell back to nothing: the guard did not fire"
    assert U._has_sessions(resolved) is True, (
        f"lloyd_root() resolved to {resolved}, which has no sessions either — every "
        "table read downstream of this would be silently empty")
    # And the fallback is not decorative: reading turns through it yields a corpus,
    # not the zero that the worktree path would have given.
    assert len(U.human_turns(days=900)) > 0, resolved


def test_a_worktree_holding_only_the_gates_canary_session_is_not_a_transcript_store(tmp_path):
    """#1213, 2026-09-17: `canary_smoke` leaves `sessions/canary_*.json` in the
    round's worktree, and the next full-suite run there took that one turn for
    the corpus instead of falling back to the live store."""
    (tmp_path / "sessions").mkdir()
    assert uptake._has_sessions(tmp_path) is False
    (tmp_path / "sessions" / "canary_1789667418_68bba5.json").write_text("{}")
    assert uptake._has_sessions(tmp_path) is False, "the gate's own smoke turn is not a corpus"
    (tmp_path / "sessions" / "20260917_101500_ab12cd.json").write_text("{}")
    assert uptake._has_sessions(tmp_path) is True


def test_a_worktree_holding_only_review_grader_sessions_falls_back(tmp_path, monkeypatch):
    """#1375: the review rung's grader records land in the round's
    `sessions/`, and 18 of them read as the corpus — `human_turns()` 0 and
    five nodes red for that round only. `lloyd_root()` must still fall back."""
    from app import paths

    d = tmp_path / "sessions"
    d.mkdir()
    (d / "canary_1789667418_68bba5.json").write_text("{}")
    (d / "20260922_064347_review_f8b2.json").write_text(json.dumps(
        {"session_id": "20260922_064347_review_f8b2", "platform": "worker",
         "source": "automod-review"}))
    assert uptake._has_sessions(tmp_path) is False
    monkeypatch.delenv("LLOYD_ROOT", raising=False)
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(paths, "production_data_root", lambda: tmp_path / "prod")
    assert uptake.lloyd_root() == tmp_path / "prod"
    (d / "20260922_070000_ab12cd.json").write_text("{}")
    assert uptake.lloyd_root() == tmp_path


# ------------------------------- #1195: what a `weighted_disputes` may mean --

def test_a_skill_row_with_no_text_overlap_reports_no_signal_not_a_small_constant():
    """Defect 1, clause 1: the `+ 0.05` calibration floor fabricated readings.

    In the committed table `eval/uptake/uptake-2026-09-17.json`, 4 of the 90
    skill rows sit at exactly 0.0500 and `skill:code-review` reports
    `weighted_disputes 0.05` on `overlap_max 0.0` — uptake credit composed
    entirely of a constant that arrived in the first implementation commit
    (`1e21fc1`) with no stated rationale and nothing pinning it. A constant
    cannot be averaged into a headline: it sorts *above* a row that genuinely
    measured zero, which is how a note row ended up scoring below a skill row
    that never overlapped anything. No overlap is no signal, and null is the
    only value that says so without becoming a number.
    """
    t = _mk_turn(1, "wrong, that is not what I asked at all", "Here you are.",
                 skills=["voice-mode"])
    # The fixture's own premise, checked: this turn shares no word-gram with the
    # skill name, so a nonzero weight here can only come from a constant.
    assert uptake.overlap("voice-mode", t.user_text) == 0.0
    table = uptake.build_uptake_table(
        turns=[t], dispute_flags={"s1#1": True}, memory_entries=[],
        skills_read={"s1": {"voice-mode": 1}})
    row = [r for r in table["entries"] if r["entry"] == "skill:voice-mode"][0]
    assert row["disputes"] == 1, row
    assert row["weighted_disputes"] is None, row
    assert row["overlap_max"] == 0.0, row


def test_the_skill_weighting_carries_no_calibration_constant():
    """Clause 1's second half: the constant is *gone*, not merely unused.

    At triage `grep -n "0.05" app/uptake.py` returned exactly one line — the
    skill weighting — so a module-wide check on that literal and on the
    `max(overlap(…), <constant>)` shape both have a denominator of one.

    The check reads *numeric literals*, not the file's text: this module's
    comments and docstrings name the constant they removed, and a prose mention
    is not a calibration floor. Tokenising first is what makes the assertion
    about the code rather than about the writeup.
    """
    import inspect
    import io
    import tokenize

    src = inspect.getsource(uptake)
    code = " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))
    assert "0.05" not in code, "a 0.05 numeric literal is back in app/uptake.py"
    assert re.search(r"max\s*\(\s*overlap\s*\(", code) is None, \
        "a floor under overlap() is back: a channel with no signal must emit null"


#: Invented here, in this test, for this test. It is in no committed
#: `eval/uptake/*.json`, no transcript and no other fixture — clause 2's
#: externally-sourced check has to be a reference the instrument did not write,
#: or "the probe agrees with the probe's last table" would be the whole result.
_TOKEN = "quokka-spline-7741"


def test_a_shared_token_makes_a_weight_and_deleting_it_takes_the_weight_away():
    """Clause 2: dropping the floor must not silence genuine signal.

    The same synthetic turn, twice — once carrying a token the entry also
    carries, once with that token deleted. The first run's weight is the summed
    per-turn overlaps; the second's is null. Inject-then-remove is the pair that
    shows the weight tracks the text rather than the row's existence, and it is
    built from turns this test invents, not from any artifact under
    `eval/uptake/`.
    """
    entry = uptake.Entry("lloyd/MEMORY.md", _TOKEN)
    t1 = "wrong, the quokka-spline-7741 is stale"
    t2 = "no, quokka-spline-7741 again"
    with_tok = [_mk_turn(1, t1, "Done.", session="s1"),
                _mk_turn(2, t2, "Fixed.", session="s1")]
    table = uptake.build_uptake_table(
        turns=with_tok, dispute_flags={"s1#1": True, "s1#2": True},
        memory_entries=[entry], skills_read={})
    row = table["entries"][0]
    assert row["presence_source"] == uptake.ALWAYS_IN_FORCE, row
    assert row["disputes"] == 2, row
    # Hand-computed, NOT recomputed from uptake.overlap: recomputing it here would
    # make the assertion a tautology that no change to the scorer could fail.
    # `_TOKENS` is `[a-z0-9_]+`, so "quokka-spline-7741" is the three tokens
    # quokka / spline / 7741 plus their two bigrams = 5 grams, and turn 1's seven
    # tokens ("wrong the quokka spline 7741 is stale") are 7 + 6 = 13 grams that
    # contain all five. Turn 2 is five tokens = 5 + 4 = 9 grams, again containing
    # all five. Jaccard is therefore 5/13 + 5/9 = 110/117.
    expected = 5 / 13 + 5 / 9
    assert expected == pytest.approx(110 / 117)
    assert row["weighted_disputes"] == pytest.approx(expected, abs=5e-5), row

    # Same entries, same turns, token deleted from both: nothing is left to match,
    # and the honest answer is no signal — not the 0.05 that used to be paid out.
    without_tok = [
        _mk_turn(1, "wrong, the is stale", "Done.", session="s1"),
        _mk_turn(2, "no, again", "Fixed.", session="s1"),
    ]
    off = uptake.build_uptake_table(
        turns=without_tok, dispute_flags={"s1#1": True, "s1#2": True},
        memory_entries=[entry], skills_read={})["entries"][0]
    assert off["disputes"] == 2, off
    assert off["weighted_disputes"] is None, off
    assert off["overlap_max"] == 0.0, off


def test_the_skill_channel_pays_a_shared_token_and_nulls_without_one():
    """Clause 2 across the seam that had the floor: a skill row is scored on the
    skill's name, so the token goes into the name.

    One row with a real overlap must keep its number — a null-everywhere change
    would look like an improvement while quietly disabling the channel.
    """
    on = uptake.build_uptake_table(
        turns=[_mk_turn(1, f"wrong, rerun {_TOKEN} please", "Ran it.", session="s1")],
        dispute_flags={"s1#1": True}, memory_entries=[],
        skills_read={"s1": {_TOKEN: 1}})
    row = [r for r in on["entries"] if r["entry"] == f"skill:{_TOKEN}"][0]
    # Again hand-computed: "wrong rerun quokka spline 7741 please" is 6 tokens =
    # 6 + 5 = 11 grams, and the entry's 5 grams are all inside it, so the row's
    # single disputed turn is worth 5/11.
    assert row["weighted_disputes"] == pytest.approx(5 / 11, abs=5e-5), row

    off = uptake.build_uptake_table(
        turns=[_mk_turn(1, "wrong, rerun it please", "Ran it.", session="s1")],
        dispute_flags={"s1#1": True}, memory_entries=[],
        skills_read={"s1": {_TOKEN: 1}})
    off_row = [r for r in off["entries"] if r["entry"] == f"skill:{_TOKEN}"][0]
    assert off_row["disputes"] == 1 and off_row["weighted_disputes"] is None, off_row


def test_overlap_itself_scores_hand_counted_grams():
    """The one honest limit of an inject/remove pair whose expected value is
    produced by `overlap()`: it pins aggregation and null-ing and cannot catch a
    change to `overlap()`. These are the literals that can.

    Every number is gram counts read off the definition — word-set Jaccard over
    unigrams plus bigrams, `_TOKENS` = `[a-z0-9_]+`, so a hyphen splits rather
    than joins:

      * identical text shares every gram with itself, and the union is the same
        set, so it is 1.0 whatever the wording;
      * disjoint text shares nothing -> 0.0, and an empty side is 0.0 rather than
        a ZeroDivisionError, which is what would silently delete every weight;
      * "alpha beta" inside "alpha beta gamma delta" is 3 shared grams (alpha,
        beta, "alpha beta") over 7 union grams (3 + 4 more from the longer text);
      * "quokka-spline-7741" is the tokens quokka / spline / 7741 plus two
        bigrams = 5 grams, and "wrong, the quokka-spline-7741 is stale" is 7
        tokens = 13 grams holding all five, so 5/13 — the number the row above
        is built from.
    """
    assert uptake.overlap("wrist torque spec", "Wrist Torque Spec") == 1.0
    assert uptake.overlap("alpha beta", "gamma delta") == 0.0
    assert uptake.overlap("", "anything at all") == 0.0
    assert uptake.overlap("alpha beta", "alpha beta gamma delta") == pytest.approx(3 / 7)
    assert uptake.overlap(_TOKEN, "wrong, the quokka-spline-7741 is stale") \
        == pytest.approx(5 / 13)


def _note_block(title, path="", excerpt=""):
    """One persisted `<vault-context>` block in production's own form.

    `prefetch.py` writes `- **<title>** (score: N, file: <vault path>): <excerpt>`
    and `_messages_subliminal.py:144` stores that line verbatim; both the path and
    the excerpt are in the string the reader used to throw away.
    """
    meta = f"score: 0.87, file: {path}" if path else "score: 0.87"
    line = f"- **{title}** ({meta})" + (f": {excerpt}" if excerpt else "")
    return {"role": "subliminal",
            "content": [{"type": "text",
                         "text": f"<vault-context>\n{line}\n</vault-context>"}],
            "timestamp": "2026-09-10T01:00:05+00:00"}


def test_the_note_reader_keeps_the_path_and_excerpt_of_an_injected_line():
    """Clause 3, first half: `_vault_context_titles` captures the whole line.

    The title stays the row's identity — `note:<title>` is how every consumer
    addresses the row — so the captured path and excerpt ride *beside* it rather
    than replacing it.
    """
    block = ("<vault-context>\n- **Wrist Joint Torque Spec** (score: 0.87, "
             "file: knowledge/rig/torque.md): the elbow harmonic drive takes 4 Nm\n"
             "</vault-context>")
    got = uptake._vault_context_titles(block)
    assert [str(g) for g in got] == ["Wrist Joint Torque Spec"], got
    assert got[0] == "Wrist Joint Torque Spec", "the title must still equal the row key"
    assert got[0].path == "knowledge/rig/torque.md", got[0].path
    assert "harmonic drive" in got[0].excerpt, got[0].excerpt
    assert "harmonic drive" in got[0].scored_text, got[0].scored_text

    # A title-only line, the shape the older fixtures emit, still parses and says
    # which basis it was scored on.
    bare = uptake._vault_context_titles("<vault-context>\n- **A Bare Note Title**\n"
                                        "</vault-context>")
    assert bare[0].path == "" and bare[0].excerpt == "", bare[0]
    assert bare[0].scored_text == "A Bare Note Title", bare[0].scored_text


def test_a_disputed_note_is_weighted_against_the_injected_line_not_its_title(tmp_path):
    """Clause 3, second half: defect 2, on a turn built through the real reader.

    13 of the 14 presence-verified disputed note rows in the committed table
    weighed exactly 0.0000 — including rows whose *excerpt* was the thing under
    dispute — because the score was `overlap(title, user_text)` and the title is
    a heading, 10 of them a date. Here the title shares nothing with the
    correction and the excerpt is the correction's subject: title-scored the row
    is null, line-scored it is not.
    """
    _write_session(tmp_path, "s9", [
        _user("wrong, the harmonic drive spec you quoted is stale"),
        _note_block("Wrist Notes", "knowledge/rig/torque.md",
                    "the harmonic drive needs 4 Nm on the elbow"),
        _asst("here is the note."),
    ])
    turns = uptake.human_turns(root=tmp_path)
    assert turns[0].vault_context == ["Wrist Notes"], turns[0].vault_context
    assert uptake.overlap("Wrist Notes", turns[0].user_text) == 0.0, \
        "the title must be the answer the old scorer would have given"

    table = uptake.build_uptake_table(turns=turns, dispute_flags={"s9#1": True},
                                      memory_entries=[], skills_read={})
    row = [r for r in table["entries"] if r["kind"] == "note"][0]
    assert row["entry"] == "note:Wrist Notes", row
    assert row["source_doc"] == "knowledge/rig/torque.md", row
    assert row["scored_on"] == "line", row
    assert row["disputes"] == 1, row
    assert row["weighted_disputes"] > 0.0, row
    assert row["overlap_max"] > 0.0, row


def test_a_note_row_says_whether_it_was_scored_on_its_line_or_its_title(tmp_path):
    """The row has to state its own scoring basis, or clause 3's prohibition in
    the consumer skill is unfalsifiable prose.

    A note that arrived with neither path nor excerpt can only be title-scored;
    the row says `title`, and the skill tells the consolidator not to rest a keep
    on such a weight.
    """
    _write_session(tmp_path, "s10", [
        _user("wrong, that is not what the note says"),
        _note_block("2026-04-21 Daily Notes"),
        _asst("ok."),
    ])
    table = uptake.build_uptake_table(uptake.human_turns(root=tmp_path),
                                      dispute_flags={"s10#1": True},
                                      memory_entries=[], skills_read={})
    row = [r for r in table["entries"] if r["kind"] == "note"][0]
    assert row["scored_on"] == "title", row
    assert row["weighted_disputes"] is None, row


def test_a_note_shown_two_ways_across_turns_reports_its_basis_as_mixed(tmp_path):
    """`scored_on: mixed` — the branch where a note's own rows disagree.

    Two disputed turns, the same note, two different injected lines: one carries
    the path and body excerpt, the other only the bold title. Neither `line` nor
    `title` describes that row honestly, and a reader told one or the other would
    read a half-measured weight as a measured one. The pair below is what makes
    `mixed` falsifiable: strip the excerpt from both turns and the same fixture
    collapses to `title` with a null weight.
    """
    _write_session(tmp_path, "s11", [
        _user("wrong, the harmonic drive spec you quoted is stale"),
        _note_block("Wrist Notes", "knowledge/rig/torque.md",
                    "the harmonic drive needs 4 Nm on the elbow"),
        _asst("here is the note."),
        _user("that is not what I asked at all"),
        _note_block("Wrist Notes"),
        _asst("ok."),
    ])
    table = uptake.build_uptake_table(uptake.human_turns(root=tmp_path),
                                      dispute_flags={"s11#1": True, "s11#2": True},
                                      memory_entries=[], skills_read={})
    row = [r for r in table["entries"] if r["kind"] == "note"][0]
    assert row["disputes"] == 2, row
    assert row["scored_on"] == "mixed", row
    # The title turn shares no gram with its correction, so the weight the row
    # reports is the line turn's alone — a `mixed` basis is not an average.
    assert row["weighted_disputes"] > 0.0, row
    assert row["overlap_max"] > 0.0, row

    # Same two turns, both stripped to the bare title: one basis, no signal. The
    # second fixture lives under its own root so this build cannot see s11's
    # line-bearing turns and report `mixed` for the wrong reason.
    bare = tmp_path / "bare"
    _write_session(bare, "s12", [
        _user("wrong, the harmonic drive spec you quoted is stale"),
        _note_block("Wrist Notes"),
        _asst("here is the note."),
        _user("that is not what I asked at all"),
        _note_block("Wrist Notes"),
        _asst("ok."),
    ])
    stripped = uptake.build_uptake_table(uptake.human_turns(root=bare),
                                         dispute_flags={"s12#1": True, "s12#2": True},
                                         memory_entries=[], skills_read={})
    srow = [r for r in stripped["entries"] if r["kind"] == "note"][0]
    assert srow["disputes"] == 2, srow
    assert srow["scored_on"] == "title", srow
    assert srow["weighted_disputes"] is None, srow


def _three_channel_table():
    """One row from each channel, so the per-source block has a denominator."""
    return uptake.build_uptake_table(
        turns=[_mk_turn(1, f"wrong, {_TOKEN} is stale", "Done.", skills=[_TOKEN],
                        ctx_titles=["2026-04-21 Daily Notes"])],
        dispute_flags={"s1#1": True},
        memory_entries=[uptake.Entry("lloyd/MEMORY.md", _TOKEN)],
        skills_read={"s1": {_TOKEN: 1}})


def test_the_table_breaks_its_weights_out_per_presence_source():
    """Clause 4: per-channel numbers, because pooling them ranks channels.

    Measured in the committed table: the largest `weighted_disputes` per source
    is 0.2308 / 0.0505 / 0.0294 — an ~8× gap that is a property of which channel
    a row belongs to, not of uptake. Any single ranking over all rows is decided
    by that, and a reader with one sorted list cannot see it happening.
    """
    table = _three_channel_table()
    assert {r["presence_source"] for r in table["entries"]} <= set(uptake.PRESENCE_SOURCES)
    bps = table["coverage"]["by_presence_source"]
    # Every channel a row can carry gets a block, including one with no rows at
    # all: a missing key reads as "not measured", which is a different claim from
    # "measured, and nothing landed here".
    assert set(bps) == set(uptake.PRESENCE_SOURCES), bps
    for src, blk in bps.items():
        assert {"rows", "rows_with_weight", "rows_null_weight", "max_weight"} <= set(blk), (src, blk)
    assert all(b["rows"] >= 1 for b in bps.values()), bps
    mem = bps[uptake.ALWAYS_IN_FORCE]
    assert mem["rows"] == 1 and mem["rows_with_weight"] == 1, mem
    assert mem["max_weight"] is not None and mem["max_weight"] > 0.0, mem
    note = bps[uptake.NOTE_PRESENCE_EMITTED]
    assert note["rows"] == 1 and note["rows_with_weight"] == 0, note
    assert note["rows_null_weight"] == 1 and note["max_weight"] is None, note


def test_the_glossary_says_a_null_is_no_signal_and_the_channels_are_not_one_scale(tmp_path):
    """Clause 4's second half: the table's own glossary promised something the
    code does not do.

    It read "dispute counts discounted by lexical overlap between the entry text
    and the correction" — true of exactly one channel. For skills a constant
    could *raise* a zero, and for notes "the entry text" was a title. The
    glossary now travels with every table `write_table` emits, so a consumer
    reading the JSON alone is told the truth about both.
    """
    text = uptake.GLOSSARY["weighted_disputes"]
    assert "null" in text and "not zero" in text, text
    assert "no signal" in text, text
    assert "presence_source" in text, text
    assert "never rank across" in text or "not on one scale" in text, text

    path = uptake.write_table(_three_channel_table(), out_dir=tmp_path, date="2026-09-21")
    j = json.loads(path.read_text())
    assert j["glossary"] == uptake.GLOSSARY, j["glossary"]
    assert set(j["coverage"]["by_presence_source"]) == set(uptake.PRESENCE_SOURCES)


def _resolve(doc: dict, dotted: str):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def test_a_current_table_carries_the_keys_its_generation_promises(tmp_path):
    """The seam the review rung found: prose that outruns the bytes.

    Clause 5 put three reading rules into the consumer skill — read
    `coverage.by_presence_source`, treat `weighted_disputes: null` as no signal,
    check `scored_on` before resting a keep on a note row. None of those keys
    existed in any committed table, because no probe had run since the scorer
    that emits them shipped. A rule about keys that are not in the file a job
    reads is not enforced by anything. So the artifact now names its own scorer
    generation, and this asserts against a table written by *this* code that the
    generation and the keys it promises travel together.
    """
    assert uptake.SCORER_GENERATION == 2, "bump the corpus rule with the constant"
    path = uptake.write_table(_three_channel_table(), out_dir=tmp_path,
                              date="2026-09-21")
    j = json.loads(path.read_text())
    assert j["scorer_generation"] == uptake.SCORER_GENERATION, j["scorer_generation"]
    for dotted in ("coverage.by_presence_source", "glossary.weighted_disputes",
                   "glossary.scorer_generation"):
        assert _resolve(j, dotted), f"{dotted} missing from a generation-2 table"
    notes = [r for r in j["entries"] if r.get("kind") == "note"]
    assert notes and all("scored_on" in r for r in notes), notes
    # And the glossary explains the stamp, in terms of what its absence means —
    # that is the sentence that makes an old table readable rather than wrong.
    g = uptake.GLOSSARY["scorer_generation"]
    assert "generation 1" in g and "Absent" in g, g


def test_every_committed_table_declares_a_generation_and_only_claims_its_own_keys():
    """The same rule read off the committed corpus, not a fixture.

    Generation 1 predates `scorer_generation` itself, so absence *is* its
    declaration — and the point of pinning this is that the generation-1 tables
    must not contain the generation-2 keys (a stamped-old/unstamped-new mix would
    let a job read a floor-composed 0.05 as measured signal). The positive control
    is the count: the loop must have looked at at least one file, and at least one
    file must be generation 1, or the absence branch of this test proves nothing.
    """
    files = sorted((REPO / "eval/uptake").glob("uptake-*.json"))
    assert files, "no committed table to read the generation rule against"
    gens = []
    for f in files:
        j = json.loads(f.read_text())
        gen = int(j.get("scorer_generation", 1))
        gens.append(gen)
        rows = j["entries"]
        bps = "by_presence_source" in j.get("coverage", {})
        gloss = "glossary" in j
        note_keys = {"scored_on" in r for r in rows if r.get("kind") == "note"}
        nulls = [r for r in rows if r.get("weighted_disputes") is None]
        # Generation 1 emitted a glossary too — the one that was false for two of
        # the three channels — so its presence proves nothing. The keys that
        # distinguish a generation are the ones a reading rule depends on.
        if gen == 1:
            assert "scorer_generation" not in j, f"{f.name} declares 1 and carries the key"
            assert not bps, f"{f.name} is generation 1 but emits by_presence_source"
            assert not any(note_keys), f"{f.name} is generation 1 but rows carry scored_on"
            assert not nulls, f"{f.name} is generation 1 but has null weights"
            floor = [r for r in rows if r.get("presence_source", "").startswith("proxy:")
                     and r.get("weighted_disputes") == 0.05 and r.get("overlap_max") == 0.0]
            assert floor, f"{f.name} is gen 1 with no floor-only skill row: the " \
                "0.05 this generation pays is no longer visible in it"
        else:
            assert gen == uptake.SCORER_GENERATION, f"{f.name} declares {gen}"
            assert bps and gloss, f"{f.name} claims gen {gen} without its keys"
            assert not [r for r in rows if r.get("weighted_disputes") == 0.05
                        and r.get("overlap_max") == 0.0], \
                f"{f.name} is stamped {gen} and still pays a floor"
    assert 1 in gens, "every committed table is stamped: the absence rule is untested"


@pytest.mark.live_vault
def test_the_knowledge_write_skill_reads_a_null_as_no_signal_not_zero(tmp_path):
    """Clause 5: the producer's vocabulary change is only safe if the one live
    consumer is told. `nightly-reflection-knowledge-write` mandates citing
    `weighted_disputes` for every keep / merge / archive decision, so a null it
    reads as `0` becomes an argument for pruning a row that was never measured —
    and the row that reads as zero sorts *lowest*, which is the direction that
    gets an entry archived.

    This reads the vault, so it is `live_vault` and does not run under the gate's
    `-m "not live_vault"` (pytest.ini: a round cannot change the vault). The
    hermetic counterparts that DO run at the gate are
    `test_a_current_table_carries_the_keys_its_generation_promises` and
    `test_every_committed_table_declares_a_generation_and_only_claims_its_own_keys`:
    together they pin that every path this prose orders the job to read exists in
    a table the current scorer emits, and that the tables which do NOT have it
    declare the generation they belong to — which is the seam the review rung
    found open, where the prose was written for keys no committed table had.
    """
    text = (Path.home() / "obsidian/skills/nightly-reflection-knowledge-write/SKILL.md").read_text()
    assert "null" in text and "no signal, not zero" in text, "null semantics not stated"
    assert "never rank across" in text, "cross-channel ranking not forbidden"
    assert "scored_on" in text and "title" in text, \
        "a title-scored note row is not told to be non-load-bearing"

    # The prose has to say which scorer a rule applies to, because the committed
    # tables are generation 1 and say nothing about it.
    assert "scorer_generation" in text, "the skill never says how to tell the generations apart"
    assert "generation 1" in text and "0.05" in text, \
        "the skill does not name the floor its tables may still be showing"

    # Every `coverage.*` path the skill orders the job to read must resolve in a
    # table this code emits. A named path that does not exist is not a missing
    # figure — the job reads nothing, explains the gap, and its citation still
    # names the key, which is how prose silently becomes fiction.
    j = json.loads(uptake.write_table(_three_channel_table(), out_dir=tmp_path,
                                      date="2026-09-21").read_text())
    cites = set(re.findall(r"`(coverage\.[a-z_]+)`", text))
    assert "coverage.by_presence_source" in cites, sorted(cites)
    for dotted in cites:
        assert _resolve(j, dotted), f"skill orders the job to read {dotted}; it is not emitted"

    # And a named sub-key under a declared channel must be a real channel, since
    # the block is what the skill tells the job to rank within.
    channels = set(uptake.PRESENCE_SOURCES)
    for src in re.findall(r"coverage\.by_presence_source/([A-Za-z0-9_:+*]+)", text):
        assert src in channels or src == "<one declared source>", (src, sorted(channels))


def test_the_probe_stamps_the_engine_that_answered_and_refuses_a_rerouted_measurement(
        monkeypatch):
    """#1310: `engine` came from the `secondary` constant, so with the slot
    retired every table said `secondary` over numbers the primary produced.
    The resolved endpoint and model are recorded, and a reroute cannot be
    `measured: true` or pass."""
    import scripts.uptake_probe as probe
    from app import secondary_models

    def _block():
        return {"metrics": {"measured": True, "precision": 1.0},
                "holdout": {"measured": True}, "zero_shot": {"measured": True},
                "passed": True}

    monkeypatch.setattr(secondary_models, "_endpoint", lambda job: (
        "http://127.0.0.1:8096/v1/chat/completions", "primary"))
    out = probe.stamp_engine(_block())
    assert out["engine"] == "primary" and out["engine_alias"] == "secondary"
    assert out["engine_endpoint"].startswith("http://127.0.0.1:8096")
    assert out["engine_rerouted"] is True and out["passed"] is False
    for key in ("metrics", "holdout", "zero_shot"):
        assert out[key]["measured"] is False, key
        assert "not a measurement of the secondary" in out[key]["unmeasured_reason"]
    assert out["metrics"]["precision"] == 1.0, "the number is kept, only disowned"

    monkeypatch.setattr(secondary_models, "_endpoint", lambda job: (
        "http://127.0.0.1:8091/v1/chat/completions", "secondary"))
    out = probe.stamp_engine(_block())
    assert out["engine"] == "secondary" and out["engine_rerouted"] is False
    assert out["passed"] is True and out["metrics"]["measured"] is True
