"""Tests for #552 — did each durable memory entry / skill get honored?

The item's claim was that Lloyd measures every input surface and no *outcome*
surface: nothing asks whether an entry that landed in the prompt was ever
followed, or disputed afterward. So these tests are all about the measurement
existing, being computed from real logged evidence rather than assertion, and
being honest about the half it cannot compute.

Marks: `live_engine` needs the secondary LLM answering on its port, and
`live_vault` reads `~/obsidian` — neither is under a round's control, so the
automod gate's hard rung skips both (`-m "not live_vault"` plus
`-m "not live_engine"`).
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
    nothing. So both floors are enforced together, from constants the probe
    shares rather than numbers it re-declares."""
    import scripts.uptake_probe as probe

    assert probe.uptake is uptake, "the probe must gate on the shared floors"
    assert uptake.RECALL_FLOOR > 0.0 and uptake.PRECISION_FLOOR > 0.0
    # The always-NOT shape that scored 1.00 precision cannot clear the recall floor.
    m = uptake.precision_recall([1] * 23 + [0] * 23, [0] * 46)
    assert m["recall"] == 0.0
    assert not (m["measured"]
                and (m["precision"] or 0) >= uptake.PRECISION_FLOOR
                and (m["recall"] or 0) >= uptake.RECALL_FLOOR)


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
    return uptake.Turn(
        session=session, session_source=None, ts=f"2026-09-10T0{idx % 10}:00:00+00:00",
        user_text=text, prev_assistant=prev, ordinal=idx,
        injected_skills=list(skills), vault_context=list(ctx_titles),
    )


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
        dispute_flags={1: True, 2: False},
        memory_entries=[uptake.Entry("lloyd/MEMORY.md", "restart lloyd-agent-worker")],
        skills_read={},
    )
    rows = {r["entry"]: r for r in table["entries"]}

    vm = rows["skill:voice-mode"]
    assert vm["present_in_turns"] == 1 and vm["disputes"] == 1
    assert vm["dispute_rate"] == pytest.approx(1.0)

    yt = rows["skill:youtube-transcript"]
    assert yt["present_in_turns"] == 1 and yt["disputes"] == 0

    assert "skill:voice-mode" not in json.dumps(yt)


def test_always_in_force_memory_entries_are_flagged_and_weighted():
    table = uptake.build_uptake_table(
        turns=[_mk_turn(1, "wrong, the restart target is wrong", "Restarted livekit."),
               _mk_turn(2, "what is 2+2", None)],
        dispute_flags={1: True, 2: False},
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
        {"ts": "x", "session_id": "s1", "event": "brain1.tool_call_proposed",
         "turn_id": "t1", "data": {"tool_call_id": "a", "name": "skills_read",
                                   "args": json.dumps({"name": "voice-mode",
                                                       "summary": "reading"})}},
        {"ts": "x", "session_id": "s1", "event": "brain1.tool_call_proposed",
         "turn_id": "t2", "data": {"tool_call_id": "b", "name": "Bash",
                                   "args": json.dumps({"command": "grep skills_read app/"})}},
        "this line is not json at all\n",
    ]
    ev.write_text("\n".join(json.dumps(l) if isinstance(l, dict) else l for l in lines))

    got = uptake.skills_read_by_session(tmp_path)
    assert got == {"s1": {"voice-mode"}}


def test_coverage_block_reports_what_the_table_actually_covers():
    table = uptake.build_uptake_table(
        turns=[_mk_turn(1, "wrong", "ok", skills=["voice-mode"])],
        dispute_flags={1: True},
        memory_entries=[uptake.Entry("lloyd/MEMORY.md", "a"), uptake.Entry("lloyd/MEMORY.md", "b")],
        skills_read={"s1": {"voice-mode", "voice-clone-sample"}},
        active_skills=["voice-mode", "voice-clone-sample", "restart-lloyd", "obsidian"],
    )
    cov = table["coverage"]
    assert cov["user_md_entries"]["covered"] == 2 and cov["user_md_entries"]["total"] == 2
    assert cov["active_skills"]["covered"] == 2 and cov["active_skills"]["total"] == 4
    # The half the item cannot ask for yet must be labelled, not hidden.
    assert cov["active_skills"]["presence_source"] == uptake.SKILL_PRESENCE_PROXY
    assert "435" in cov["active_skills"]["note"]


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
    """The re-basing has to be true of the real files, not only of fixtures."""
    gate = uptake.retrieval_gate()
    assert gate["nights"] >= 3, gate
    assert gate["doc_hit_rate"]["floor"] < 0.95, gate
    assert uptake.RETRIEVAL_GATE_HARDCODE == 0.95  # the number being replaced


# ------------------------------------------------------------ emitted table -

def test_emitted_table_lands_under_eval_uptake_with_a_probe_timestamp(tmp_path):
    table = uptake.build_uptake_table(turns=[_mk_turn(1, "wrong", "ok", skills=["obsidian"])],
                                      dispute_flags={1: True}, memory_entries=[],
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
    dated JSON, and each row carries present_in_turns / disputes / dispute_rate."""
    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    assert files, "no dated uptake table under eval/uptake/"
    j = json.loads(files[-1].read_text())
    rows = j["entries"]
    assert rows, j
    for key in ("entry", "present_in_turns", "disputes", "dispute_rate", "presence_source"):
        assert key in rows[0], rows[0]
    assert re.match(r"\d{4}-\d{2}-\d{2}", j["probe_timestamp"])
    assert j["classifier"]["precision"] is not None
    assert j["classifier"]["threshold"] == 0.70


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


def test_store_size_figures_in_the_table_are_always_paired_with_a_timestamp():
    """Standing rule in this repo: a KG or facts count with no probe timestamp
    is a stale figure wearing a current one. The table must not be able to
    carry one without it."""
    files = sorted((REPO / "eval" / "uptake").glob("uptake-*.json"))
    j = json.loads(files[-1].read_text())
    for name, block in (j.get("stores") or {}).items():
        for metric, value in block.items():
            if isinstance(value, int):
                assert block.get("probed_at"), f"{name}.{metric} has no probe timestamp"


# ------------------------------------------------- labeled corpus integrity -

def test_hand_labeled_corpus_covers_the_item_s_minimum():
    labels = uptake.load_labels()
    positives = [l for l in labels if l["label"] == 1]
    assert len(labels) >= 40, len(labels)
    assert len(positives) >= 20, len(positives)
    assert {l["labeled_by"] for l in labels} == {"hand:alan-turns-2026-09-11"}
    # Every label must point at a turn id of the documented shape, so a later
    # run can re-open the transcript behind it.
    for l in labels:
        assert re.match(r"^.+#\d+$", l["turn_id"]), l


@pytest.mark.live_engine
def test_live_holdout_precision_clears_the_stopping_threshold():
    """Step 2's stop condition, run against the real secondary engine: if
    precision < 0.70 the pipeline is not supposed to go further, and an uptake
    table built on a noisier classifier would be a machine for attributing
    blame at random."""
    import scripts.uptake_probe as probe
    result = probe.run_classifier_eval()
    assert result["metrics"]["n_positives"] >= 20, result["metrics"]
    assert result["metrics"]["precision"] is not None, result["metrics"]
    assert result["metrics"]["precision"] >= 0.70, result["metrics"]


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
