"""Tests for #552 — did each durable memory entry / skill get honored?

The item's claim was that Lloyd measures every input surface and no *outcome*
surface: nothing asks whether an entry that landed in the prompt was ever
followed, or disputed afterward. So these tests are all about the measurement
existing, being computed from real logged evidence rather than assertion, and
being honest about the half it cannot compute.

One test here needs the secondary LLM to be answering and two read the live
`~/obsidian` vault; neither is under a round's control. The vault readers carry
the tree's existing `live_vault` mark, which the automod gate excludes. The
engine reader carries no mark — it is the acceptance measurement for #552, so it
should run on the gate. With the engine awake the floors must clear; with it down
the test **fails**, naming the engine, unless `UPTAKE_ALLOW_SILENT_ENGINE=1` records
that the box is offline on purpose (then it skips, visibly, instead of passing). It
used to assert whichever branch it landed in and record nothing about which, so an
asleep model and a passing grader were indistinguishable in the report. The
fail-closed behaviour of a silent engine is pinned hermetically by the
stand-in-transport tests, so refusing to grade the live measurement against an
absent model costs no coverage. Nothing else in this file skips, xfails, or asserts
`True`; the loopback HTTP seam is crossed by a test that runs whether or not a model
is loaded, and the precision claim is replayable from recorded replies.
"""

from __future__ import annotations

import json
import os
import re
import sys
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

def _mk_baseline(path: Path, label: str, doc_hr: float, ndcg: float, *, prod=True, limit=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "label": label, "ran_at": "2026-09-09T06:00:00+00:00", "limit": limit,
        "matches_production_defaults": prod, "graph_rerank": False,
        "summary": {"overall": {"n_queries": limit, "doc_hit_rate": doc_hr,
                                "ndcg10": ndcg, "entity_hit_rate": 0.5}},
    }))


def test_retrieval_gate_is_derived_from_the_newest_baseline_plus_its_noise_band(tmp_path):
    """A gate that hard-codes `doc_hit_rate >= 0.95` is decided by noise: on six
    identical-config nights the metric alone spans 0.85-0.95. The re-based gate
    is the newest measurement minus the spread of comparable nights, so the
    tolerance is a measured quantity, not a remembered one."""
    d = tmp_path / "baselines"
    for i, (lab, dh, nd) in enumerate([
        ("nightly-20260904", 0.95, 0.591), ("nightly-20260905", 0.90, 0.557),
        ("nightly-20260906", 0.90, 0.536), ("nightly-20260907", 0.90, 0.562),
        ("nightly-20260908", 0.95, 0.595), ("nightly-20260909", 0.85, 0.536),
    ]):
        _mk_baseline(d / f"{lab}-x.json", lab, dh, nd)
        os.utime(d / f"{lab}-x.json", (1_700_000_000 + i, 1_700_000_000 + i))

    gate = uptake.retrieval_gate(baselines_dir=d)
    assert gate["latest"]["label"] == "nightly-20260909"
    assert gate["nights"] == 6
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
    assert gate["doc_hit_rate"]["floor"] == pytest.approx(0.75)


def test_live_nightly_band_is_not_the_hardcoded_threshold():
    """The re-basing has to be true of the real files, not only of fixtures.

    Meaningful in both states rather than erroring on a clean checkout:
    `eval/baselines/` is gitignored, so a tree without it must still see the
    documented behaviour — `retrieval_gate()` refusing to invent a band. On a box
    that has run the nightly eval, the real numbers are asserted.
    """
    assert uptake.RETRIEVAL_GATE_HARDCODE == 0.95  # the number being replaced
    baselines = uptake.lloyd_root() / "eval" / "baselines"
    if not any(baselines.glob("nightly-*.json")):
        with pytest.raises(uptake.NoBaselines):
            uptake.retrieval_gate()
        return
    gate = uptake.retrieval_gate()
    assert gate["nights"] >= 3, gate
    assert gate["doc_hit_rate"]["floor"] < 0.95, gate
    assert gate["doc_hit_rate"]["floor"] == pytest.approx(
        gate["doc_hit_rate"]["latest"] - gate["doc_hit_rate"]["band"], abs=1e-4), gate
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

def _replay_engine(monkeypatch) -> dict:
    """Stand in the engine by replaying its **recorded** replies.

    What this replaces is `_oracle_classifier(want_positives=True)`, which
    reproduced the hand labels exactly: the table-writing path was therefore only
    ever demonstrated with a flawless grader, scoring 1.0 by construction, and a
    pipeline bug that flipped every verdict would have stayed green. Replaying the
    literal replies the secondary slot gave on the measurement date is different
    in kind — the grader misses 10 of 23 disputes — so the emit path runs under a
    grader that is wrong, and the confusion matrix it must produce is a recorded
    fact of the fixture rather than a restatement of the labels.

    Keyed on the exact `(prev_assistant, user_text)` pair the classifier is called
    with; 60-char text prefixes collide in this corpus (46 labels, 37 distinct
    prefixes, 2 shared across classes), so a text-keyed stand-in could not be
    declared exactly. Only the engine is stubbed — `main()`'s screen, cache, join
    and write path all run.
    """
    import scripts.uptake_probe as probe

    labels = probe.uptake.load_labels()
    index = probe._corpus_index()
    by_pair: dict[tuple, str] = {}
    recorded = 0
    for item in labels:
        if "engine_raw" not in item:
            continue
        turn = index.get(item["turn_id"])
        if turn is None:
            continue
        key = (turn.prev_assistant, turn.user_text)
        if key in by_pair:
            # Two labeled turns share this exchange. That is only harmless if the
            # engine gave the same reply to both — otherwise a stand-in keyed on
            # what `classify_dispute` actually receives cannot say which reply
            # belongs to which turn, and the replay would be inventing one.
            assert by_pair[key] == item["engine_raw"], (
                f"ambiguous replay pair for {item['turn_id']}: the same exchange "
                f"has two different recorded replies")
        by_pair[key] = item["engine_raw"]
        recorded += 1
    assert recorded == len(labels), f"{recorded}/{len(labels)} labels carry a reply"

    def fake(prev_assistant, user_text, *, transport=None):
        return uptake._parse_verdict(by_pair.get((prev_assistant, user_text), "NOT"))

    monkeypatch.setattr(uptake, "classify_dispute", fake)
    return json.loads((REPO / "eval" / "uptake" / "labels"
                       / "hand-2026-09-11.json").read_text())["engine_replay"]


def _always_not_engine(monkeypatch) -> None:
    """The always-NOT shape the item's own stop condition exists to catch."""
    def fake(prev_assistant, user_text, *, transport=None):
        return False
    monkeypatch.setattr(uptake, "classify_dispute", fake)


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


def test_probe_writes_a_dated_table_when_the_floor_clears(tmp_path, monkeypatch):
    """The other side of the same contract, run end to end through `main()` under
    a grader that MISSES: the recorded replies score recall 0.565, so this
    exercises the emit path with a fallible classifier and still asserts the
    emitted numbers are the recorded ones to four decimals."""
    import scripts.uptake_probe as probe

    recorded = _replay_engine(monkeypatch)
    out = tmp_path / "uptake"
    assert probe.main(["--out-dir", str(out), "--days", "30"]) == 0

    files = sorted(out.glob("uptake-*.json"))
    assert len(files) == 1, files
    assert re.match(r"uptake-\d{4}-\d{2}-\d{2}\.json$", files[0].name)
    j = json.loads(files[0].read_text())
    cls = j["classifier"]
    assert j["item"] == "552"
    assert (cls["tp"], cls["fp"], cls["fn"]) == (recorded["tp"], recorded["fp"], recorded["fn"]), cls
    assert cls["recall"] < 1.0, "a grader that misses nothing is not a grader"
    assert cls["precision"] == pytest.approx(recorded["precision"], abs=1e-4), cls
    assert cls["recall"] == pytest.approx(recorded["recall"], abs=1e-4), cls
    assert cls["n_unanswered"] == 0, cls
    assert j["glossary"]["dispute_rate"], "the rate ships without its meaning"
    assert any(r["kind"] == "memory_entry" for r in j["entries"]), j["entries"][:2]
    # And the join it wrote is turn_id-keyed with counts that cannot exceed the
    # turns it actually classified.
    corpus = j["corpus"]
    assert corpus["verdict_keyed_on"] == "turn_id", corpus
    assert corpus["disputes"] <= corpus["classified_turns"], corpus


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
    module._endpoint = lambda: (url, model)
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


def test_every_committed_label_re_resolves_to_its_transcript(tmp_path, monkeypatch):
    """The labels file cannot attest itself, and this is the check that does.

    The store is an artifact the labeler does not write, so re-resolving each
    `turn_id` there and comparing the stored excerpt turns "I labeled this by hand"
    into "this is what that turn actually said" — the check a reviewer had to
    perform by hand over all 46 turns on the previous head, now in the suite.
    """
    labels = uptake.load_labels()
    index = {t.turn_id: t for t in uptake.human_turns(days=900)}
    check = uptake.validate_labels(labels, index)
    assert check["unresolved_turn_ids"] == [], check
    assert check["excerpt_mismatch_turn_ids"] == [], check
    assert check["ok"] is True and check["n"] == len(labels), check
    assert check["n_resolved"] == len(labels), (
        "the corpus silently shrank as transcripts rolled off the store; the "
        "precision figure is quoted against this denominator")


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


#: Set this only when the secondary engine is deliberately down (an offline bench,
#: a machine being moved). Until it is set again, a silent engine FAILS this test
#: rather than passing it. The previous version asserted whichever branch it
#: happened to land in and recorded nothing about which one that was, so a gate run
#: that caught :8091 asleep reported a pass on a measurement that had not happened.
#: The fail-closed semantics of a silent engine are still pinned — hermetically, by
#: `_oracle_classifier` driving the same pipeline with a stand-in transport.
ALLOW_SILENT_ENGINE = "UPTAKE_ALLOW_SILENT_ENGINE"


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

    awake = uptake.classify_dispute("Built it, works now.", "it 404s on me") is not None
    if not awake:
        # The previous version asserted whichever branch it happened to land in and
        # recorded nothing about which, so a gate run that caught :8091 asleep
        # reported a pass on a measurement that had not happened. Down is now loud
        # by default; the opt-out exists for a box that is genuinely offline and
        # makes the omission visible as a skip rather than a pass.
        if os.environ.get(ALLOW_SILENT_ENGINE) == "1":
            pytest.skip(f"{ALLOW_SILENT_ENGINE}=1: {uptake.SECONDARY_URL} is "
                        "deliberately down, so the uptake measurement did not run")
        pytest.fail(f"secondary engine at {uptake.SECONDARY_URL} is not answering. "
                    "Step 2's acceptance measurement cannot be claimed without it. "
                    f"Start it, or set {ALLOW_SILENT_ENGINE}=1 if this box is offline "
                    "on purpose — the fail-closed behaviour of a silent engine is "
                    "pinned hermetically by the stand-in-transport tests above.")

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


def test_recorded_engine_replies_still_reproduce_on_the_live_engine():
    """Closes the loop the recorded matrix could not: the fixture reproduces itself,
    which proves nothing about whether the engine still says that.

    Re-asks a sample of labeled turns with the deployed prompt and requires the
    answer to match the committed `engine_raw`. The engine is temperature 0 on a
    pinned slot, so a mismatch is the model or the prompt having moved — which is
    exactly when a committed precision figure stops being re-quotable.
    """
    awake = uptake.classify_dispute("Built it, works now.", "it 404s on me") is not None
    if not awake:
        pytest.skip(f"secondary engine down; {ALLOW_SILENT_ENGINE} governs the "
                    "measurement test, which fails rather than silently skipping")
    labels = uptake.load_labels()
    index = {t.turn_id: t for t in uptake.human_turns(days=900)}
    checked = 0
    for item in labels:
        raw = item.get("engine_raw")
        turn = index.get(item["turn_id"])
        if raw is None or turn is None:
            continue
        verdict, fresh = uptake.classify_dispute_raw(turn.prev_assistant,
                                                     turn.user_text)
        assert verdict == uptake._parse_verdict(raw), (
            f"{item['turn_id']}: recorded {raw!r}, engine now answers {fresh!r}")
        checked += 1
        if checked >= 6:
            break
    assert checked >= 6, f"only {checked} labeled turns resolvable to re-ask"


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

#: What `app/routers/_messages_subliminal.py:86` + the writer actually put in a
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
