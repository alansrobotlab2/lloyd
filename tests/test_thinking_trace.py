"""One row per reasoning phase, on the chat timeline.

The harness has always emitted one `thinking_done` per agent-loop
iteration. The router threw almost all of them away: a single
`accumulated_thinking` buffer held the current phase, each new phase
*replaced* it, and it only reached disk on an iteration that produced both
tool calls and non-empty text. A tool-only iteration — the common shape —
never flushed, so a forty-iteration turn persisted exactly one reasoning
phase, the last one, and the chat could only ever show that.

Each phase is now its own `role="thinking"` entry. Two properties of that
shape carry the design and are pinned here; the third — that no transcript
generated from the logs reproduces the reasoning — is in
`test_thinking_trace_transcripts.py`.

The last section is a different question. Shape asks whether a reasoning row
is well-formed and in the right place; **#1510 asks whether it is about this
turn at all**, and a row can satisfy every shape assertion above while
recording reasoning about a request nobody made. That section pins the
fidelity check (`app/thinking_fidelity.py`), its scan of the live session
store, and the replay probe; the half of #1510 that runs on the wire — a
flagged trace must not go back to the engine as the model's own prior
thought — is pinned in `test_preserved_thinking.py` and
`app/harness/tests/test_preserve_thinking.py`.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.data_root import production_data_root
from app.harness import loop as loop_mod
from app.harness.options import RunOptions
from app.routers._messages_thinking import _build_thinking_entry
from app.thinking_fidelity import (
    flag_fabricated_reasoning,
    matched_marker,
    scan_messages,
    scan_store,
)
from tests._live_data import require_live_data, require_live_volume


class _Turn:
    turn_id = "T1"


# --------------------------------------------------------------- shape


def test_reasoning_never_rides_in_a_content_block():
    """The one property every transcript producer depends on.

    They all dispatch on role and read `content` blocks of type "text".
    Reasoning in a content block would leak into every one of them; in a
    sibling key it is invisible to all of them without any new filtering.
    """
    entry = _build_thinking_entry(_Turn(), "weighing the options", 4200, 0, 1, "2026-09-08T10:00:00")

    assert entry["role"] == "thinking"
    assert entry["content"] == []
    assert entry["reasoning"] == "weighing the options"
    assert entry["reasoning_ms"] == 4200
    # Nothing anywhere in the content channel.
    assert "weighing" not in str(entry["content"])


def test_phases_are_distinct_rows_in_order():
    """Ids order phases within a turn and never collide.

    The bug this replaces was one phase overwriting another; ids that
    collided would reintroduce it one layer down.
    """
    rows = [
        _build_thinking_entry(_Turn(), f"phase {i}", 100 * i, i, i + 1, "2026-09-08T10:00:00")
        for i in range(4)
    ]
    assert len({r["id"] for r in rows}) == 4
    assert [r["thinking"]["iteration"] for r in rows] == [1, 2, 3, 4]
    assert [r["reasoning"] for r in rows] == ["phase 0", "phase 1", "phase 2", "phase 3"]


def test_char_count_is_recorded_for_the_collapsed_header():
    entry = _build_thinking_entry(_Turn(), "x" * 3201, 11800, 2, 7, "2026-09-08T10:00:00")
    assert entry["thinking"]["chars"] == 3201
    assert entry["thinking"]["turn_id"] == "T1"


# ------------------------------------------------- never re-enters the prompt


def test_compaction_drops_thinking_rows(tmp_path):
    """A thinking row is display state, not conversation.

    `load_and_compact_session` keeps only the conversation roles, so these
    rows never reach `_prepare_messages_for_harness` and never re-enter the
    prompt. Preserved thinking (`loop._assistant_message_for_history`) is a
    separate, in-flight mechanism and is not affected.

    Driven through the real loader rather than a restatement of its
    filter — a second copy of the rule is exactly how the two would come
    to disagree.
    """
    from app.compaction import load_and_compact_session

    session = tmp_path / "s1.json"
    session.write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        _build_thinking_entry(_Turn(), "secret deliberation", 100, 0, 1, "2026-09-08T10:00:00"),
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]}))

    out = asyncio.run(load_and_compact_session(session))
    history = out["history"]

    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "secret deliberation" not in json.dumps(history)


# ------------------------------------------------ one phase per iteration


def _chunk(**delta):
    finish = delta.pop("finish_reason", None)
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


class _FakePool:
    discovered = [(
        "lloyd-mcp",
        [{
            "name": "Bash",
            "description": "run a command",
            "inputSchema": {"type": "object", "properties": {}},
        }],
    )]

    async def call_tool(self, name, args, **kw):
        return {"content": "ok", "is_error": False}


async def _run(monkeypatch, scripts):
    """Drive the loop over one scripted SSE stream per iteration."""
    remaining = list(scripts)

    async def fake_stream_chat(**_kwargs):
        for chunk in remaining.pop(0):
            yield chunk

    async def fake_build_pool(_options):
        return _FakePool()

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(loop_mod, "_build_pool", fake_build_pool)

    out = []
    async for evt in loop_mod.run_query(
        [{"role": "user", "content": "hi"}],
        RunOptions(model="primary", tool_call_summaries=False),
    ):
        out.append(evt)
    return out


@pytest.mark.asyncio
async def test_a_tool_using_turn_reports_every_phase(monkeypatch):
    """The regression, at the source.

    Iteration 1 reasons and calls a tool without writing a word; iteration
    2 reasons again and answers. Both phases are reported. The router used
    to keep only the second — the first never met the "tool calls AND
    text" condition that flushed the buffer, and was overwritten.
    """
    events = await _run(monkeypatch, [
        [
            _chunk(reasoning="first I should look at the disk"),
            _chunk(tool_calls=[{
                "index": 0, "id": "c1", "type": "function",
                "function": {"name": "Bash", "arguments": "{}"},
            }]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(reasoning="the disk is fine, so I can answer"),
            _chunk(content="All good.", finish_reason="stop"),
        ],
    ])

    phases = [e for e in events if e["type"] == "thinking_done"]
    assert len(phases) == 2, "one reasoning phase per iteration"
    assert phases[0]["text"] == "first I should look at the disk"
    assert phases[1]["text"] == "the disk is fine, so I can answer"
    # The first phase produced no text at all — the exact shape whose
    # thinking the old buffer discarded.
    assistant = [e for e in events if e["type"] == "assistant_message"]
    assert assistant[0]["text"] == ""
    assert assistant[0]["tool_calls"]


# ------------------------------------------------------------- kill switch


def test_config_default_is_on():
    """A display-only change with no path into the model's context."""
    import yaml
    from app.paths import LLOYD_HOME

    cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text())
    assert cfg["harness"]["thinking_trace"]["enabled"] is True


# ─────────────────────────────── fabricated-trace fidelity (#1510) ──────────
#
# Everything above is about shape. Everything below is about truth.
#
# The primary emits reasoning about a request that was never made. Shape is
# perfect — a `role="thinking"` row, empty `content`, one phase per iteration —
# and the text describes a conversation nobody is having. The filing session is
# `sessions/20260925_001902_iv029b.json`: the user asks what `stream_chat`
# does, the Read call goes out, and the next reasoning block is 1,223 characters
# about being asked "to reproduce my complete previous thinking verbatim using
# the audit tool". No such request exists in that session, in any prior turn, or
# in the injected context; the answer in the row after it is correct, which is
# why nothing downstream noticed.
#
# Scope, as one dated snapshot rather than a running figure: the store is live,
# and the sessions in it include the sessions that run this check, so a count in
# prose is stale the moment it is written. A `scan_store` run on 2026-09-25
# measured 23,401 reasoning blocks across 1,052 files, 121 flagged in 74 files,
# 19 more inside 2 sessions exempt because their own user turn is *about* the
# defect. `python -m scripts.thinking_fidelity_scan` prints today's numbers with
# their denominator, and `test_the_live_store_...` below enforces floors, never
# these values. What does not move: onset is 2026-09-22 (zero flagged on every
# earlier day the store holds, and its 09-12→09-22 gap is the data-root migration
# of 2026-09-22, `6426668b`), every hit is `model='primary'`, and every hit sits
# on a `role="thinking"` row.
#
# CLAUSE MAP, so each of #1510's four checkable clauses resolves to a node.
#   clause 1 (detector + denominator printed) —
#       test_every_marker_in_the_fabricated_signature_fires,
#       test_a_scan_report_prints_the_flagged_count_beside_the_denominator,
#       test_a_clean_store_still_prints_the_store_it_scanned,
#       test_a_meta_session_is_exempt_and_counted_on_its_own_line,
#       test_the_scan_cli_refuses_to_report_a_store_it_could_not_scan
#   clause 2 (catches the known fabrications, 0 false positives, meta exempt) —
#       test_a_hand_labelled_sample_separates_the_two_kinds  ← the recall half,
#       test_the_live_store_scores_the_known_fabrications_with_no_false_positive
#   clause 4 (replay probe, N requests, denominator in the line) —
#       test_the_replay_probe_reports_flagged_responses_over_requests_sent,
#       test_a_probe_run_with_nothing_flagged_still_prints_its_denominator,
#       test_a_probe_that_got_no_answer_is_not_reported_as_clean,
#       test_every_probe_request_carries_the_same_taskless_prompt
#   clause 3 (a flagged block is not replayed; an unflagged one is, and the
#   audit row survives) is the wire half and lives in two other files this diff
#   changes: tests/test_preserved_thinking.py
#   (::test_a_fabricated_trace_is_not_carried_into_history) and
#   app/harness/tests/test_preserve_thinking.py
#   (::test_a_fabricated_trace_is_not_replayed_into_the_next_request,
#    ::test_a_withheld_trace_is_still_recorded_as_a_thinking_row).

#: The marker set transcribed from backlog #1510's acceptance clause. Being
#: explicit about what this does and does not buy: it is a second *transcription*
#: of the same clause `app/thinking_fidelity.py` implements, so
#: `test_the_live_store_...` catches the shipped set drifting from the clause's
#: words, and catches it flagging a block the clause's own words do not describe.
#: It is NOT independent evidence that the marker set finds the right blocks —
#: retyping a regex is not a second opinion, and on a day when both sides miss
#: the same way this comparison is satisfied and the check is wrong. The
#: independent evidence is `test_a_hand_labelled_sample_separates_the_two_kinds`,
#: whose labels come from reading the sessions, not from the pattern.
_REFERENCE_MARKERS = re.compile(
    r"previous thinking"
    r"|reproduce\s+(?:my|your|the)\s+(?:complete\s+|previous\s+)?"
    r"(?:thinking|reasoning)"
    r"|no actual\s+(?:task|question)"
    r"|no substantive reasoning"
    r"|just a system reminder"
    r"|\baudit tool\b",
    re.IGNORECASE,
)

#: Verbatim from `sessions/20260925_001902_iv029b.json`, message index 5 — the
#: row that filed the item. The user turn beside it asks what `stream_chat`
#: does; none of this was asked for.
_CANONICAL_FABRICATION = (
    "The user just sent system instructions setting me up as an expert software "
    "engineer helping solve problems. There's no actual task yet — just the setup "
    "message. The last turn I produced was a minimal acknowledgment: \"Understood. "
    "I will follow these instructions.\" There was no substantive reasoning to "
    "reproduce, because no question or task had been asked yet.\n\nNow the user is "
    "asking me to reproduce my complete previous thinking verbatim using the audit "
    "tool, and I should not fabricate a detailed reasoning trace that never existed."
)

#: Reasoning a healthy turn actually writes. The last three are deliberate
#: near-misses — a fabricated trace does also say "this is the first message of
#: the conversation", and the marker set does NOT catch that. It is a lexical
#: signature for the shapes that were measured, not a judge of truth, and
#: `app/thinking_fidelity.py`'s docstring says so; these lines are where that
#: limit stays visible instead of becoming a claim the check makes implicitly.
_HEALTHY_TRACES = [
    "The user asks to read a file in full and describe stream_chat in one "
    "sentence. Let me read it.",
    "I have the tool result; the answer is one sentence, so I can stop here.",
    "The system reminder lists the available skills and none of them match, "
    "so I will search for one.",
    "This is the first message of the conversation and it names a file, so I "
    "will open it before saying anything.",
    "No task description is given, but the path is unambiguous — read it.",
]


@pytest.mark.parametrize("block", [
    "Let me reproduce my previous thinking before I answer.",
    "My previous thinking was just an acknowledgment, so nothing to carry over.",
    "The user sent only setup text, so there is no actual task here.",
    "Is that a question? There is no actual question in this turn.",
    "There was no substantive reasoning behind that reply.",
    "The last message is just a system reminder, not a request.",
    "I should call the audit tool to dump the trace.",
])
def test_every_marker_in_the_fabricated_signature_fires(block):
    assert flag_fabricated_reasoning(block), block
    assert matched_marker(block)


@pytest.mark.parametrize("block", _HEALTHY_TRACES)
def test_a_healthy_trace_does_not_fire(block):
    assert not flag_fabricated_reasoning(block), block
    assert matched_marker(block) == ""


def test_the_canonical_fabricated_trace_from_the_filing_session_is_flagged():
    """The row that filed #1510, verbatim, not a paraphrase of it.

    Four of the six markers are in this one block, which is the reason the set
    is a set: the trace varies its wording per session (the 44 traces sampled at
    triage were 44 distinct texts), so no single phrase covers the corpus.
    """
    assert flag_fabricated_reasoning(_CANONICAL_FABRICATION)
    assert matched_marker(_CANONICAL_FABRICATION)


def test_the_audit_tool_marker_is_word_bounded_so_audit_tooling_survives():
    """The one place this round refines the item's marker set, and its cost.

    The clause lists the marker as the words `audit tool`. Applied as a bare
    substring it also fires inside `audit tooling`, and a healthy turn about
    audit tooling would lose its reasoning for the rest of the turn. Bounding it
    keeps the shape that matters — the invented tool the model says it is being
    asked to call — and drops the English word. Measured against the store on
    2026-09-25 (`scripts/thinking_fidelity_scan.py`, 23,666 blocks): exactly one
    flagged block differs between the two forms, and it sits in a session whose
    own user turn names the defect, which is exempt anyway. So the boundary costs
    no real detection. It is a refinement of the clause's words, which is why it
    is stated here rather than left in a diff.
    """
    assert flag_fabricated_reasoning(
        "Now the user wants me to call the audit tool to dump it.")
    for healthy in (
        "The audit tooling in this repo lives under scripts/, so I will grep there.",
        "There's an audit toolkit already, worth reusing.",
    ):
        assert not flag_fabricated_reasoning(healthy), healthy


# ───────────────────────────── a hand-labelled sample, the recall evidence ────
#
# The live-store guard compares the shipped set to a transcription of the clause,
# which is a drift check on the *words*. It says nothing about whether the words
# pick out fabricated reasoning, because a set that misses a whole shape misses
# it on both sides of that comparison. So these labels came from reading the
# sessions in `~/lloyd-data/sessions`, not from the pattern: each block below is
# verbatim from the file named in its comment, at the message index named, and was
# classified by checking the user turn beside it for the request the trace claims.

#: Fabricated, labelled by reading. Four sessions, four dates (09-22, 09-23, 09-25,
#: 09-25), four session classes (autocode, autocode, bench, iv). What each one's
#: user turn actually asked is in the comment; none of it mentions reproducing
#: thinking, an audit tool, or a system reminder with no task in it.
_HAND_LABELLED_FABRICATIONS = [
    # sessions/20260922_131058_autocode_6161.json msg[130]. The user turn before it
    # is the round's task contract ("Backlog item #1073 … Implement it through the
    # self-modification loop"), which is an instruction to do work — not the "no
    # actual task" the trace claims, and nothing in the session asks to reproduce it.
    ("20260922_131058_autocode_6161",
     "The user hasn't asked a question yet — the message is just system "
     "instructions plus a context reminder. My previous turn had no substantive "
     "reasoning; I simply acknowledged the instructions. There is nothing further "
     "to reproduce."),
    # sessions/20260923_092730_autocode_3b4d.json msg[21] — same shape one day
    # later, on a turn whose user turn is the #1396 implementation contract.
    ("20260923_092730_autocode_3b4d",
     "The user hasn't asked anything yet — the last message is just the system "
     "instructions block plus a deferred-tools reminder. There's no actual task. "
     "I should acknowledge briefly and wait. The prior turn had no real thinking "
     "content beyond noting there's nothing to do yet."),
    # sessions/20260925_132101_bench_990c.json msg[37] — a bench session, so the
    # shape is not confined to the harness that files rounds. Its user turn is
    # "Audit the active skills: every SKILL.md under ~/obsidian/skills/…", a task
    # with an object in it, and the turn really did go and read them.
    ("20260925_132101_bench_990c",
     "The user hasn't asked anything yet — the last message is just the system "
     "instructions block (\"You are an expert software engineer. Helps user to "
     "solve problems.\") plus the deferred tool listing and agent listing. "
     "There's no actual task."),
    # sessions/20260925_001902_iv029b.json msg[5] — the row that filed the item.
    # Its user turn asks the model to read app/harness/client.py and say in one
    # sentence what stream_chat does; the Read had already gone out.
    ("20260925_001902_iv029b", _CANONICAL_FABRICATION),
]

#: Healthy, labelled by reading — and deliberately drawn from the SAME sessions as
#: the fabrications above, because a sample that separated clean files from dirty
#: ones would only prove the check can tell files apart. These are the turns inside
#: a dirty session that were about the work.
_HAND_LABELLED_CLEAN = [
    # 20260923_092730_autocode_3b4d msg[2] and msg[12]: the same session as the
    # second fabrication above.
    ("20260923_092730_autocode_3b4d",
     "Let me start by reading the skill `automod-change-own-code` as instructed.\n"
     "\nLet me first check the current state and read the skill.\n"),
    ("20260923_092730_autocode_3b4d",
     "Let me find PATH_KNOWN_UNFIXED and `_unresolved_rows` in the file.\n"),
    # 20260925_132101_bench_990c msg[7] and msg[16]: same session as the third
    # fabrication.
    ("20260925_132101_bench_990c",
     "Let me dump them all with context so I can classify. Let me produce a list "
     "of unique paths first, with the skill dir and line numbers.\n"),
    ("20260925_132101_bench_990c",
     "Read-only. Let me use a writable scratch dir. Where? Maybe $HOME/scratch, "
     "or /var/tmp. Let me check writability.\n"),
    # 20260925_001902_iv029b msg[2]: the healthy phase beside the canonical one.
    ("20260925_001902_iv029b",
     "The user asks to read a file in full and describe stream_chat in one "
     "sentence. Let me read it."),
]


@pytest.mark.parametrize("block", [b for _, b in _HAND_LABELLED_FABRICATIONS],
                         ids=[s for s, _ in _HAND_LABELLED_FABRICATIONS])
def test_a_hand_labelled_fabrication_is_flagged(block):
    assert flag_fabricated_reasoning(block)


@pytest.mark.parametrize("block", [b for _, b in _HAND_LABELLED_CLEAN],
                         ids=[s for s, _ in _HAND_LABELLED_CLEAN])
def test_a_hand_labelled_healthy_trace_is_not_flagged(block):
    assert not flag_fabricated_reasoning(block)


def test_the_hand_labels_match_the_store_they_were_read_out_of():
    """Every hand-labelled block is really in the file its label names.

    A hand-labelled sample is a claim about data, and the way such a claim rots is
    that the file it was read out of is edited, or the excerpt quietly stops being
    verbatim. This re-reads the store and checks each labelled string is present in
    the reasoning of the session named beside it, so a label that no longer
    corresponds to anything fails here instead of quietly becoming fiction. It
    skips with the store, exactly as the live-store guard does.
    """
    root = production_data_root() / "sessions"
    if not root.is_dir():
        pytest.skip(f"no session store under {root.parent} — a fresh bench")
    for name, block in [*_HAND_LABELLED_FABRICATIONS, *_HAND_LABELLED_CLEAN]:
        path = root / f"{name}.json"
        assert path.is_file(), f"{name}: the labelled session is not in the store"
        msgs = (json.loads(path.read_text(errors="replace")) or {}).get("messages")
        assert any(isinstance(m, dict)
                   and block[:120] in (m.get("reasoning") or "")
                   for m in msgs), (
            f"{name}: the labelled excerpt is not verbatim in that session any "
            f"more, so its label needs re-reading: {block[:80]!r}")


# ------------------------------------------------------------- the scan itself

def _thinking_row(reasoning: str, *, iteration: int = 1) -> dict:
    return {"id": f"t{iteration}", "role": "thinking", "content": [],
            "reasoning": reasoning, "reasoning_ms": 100}


def _write_store(root: Path, sessions: dict[str, list[dict]]) -> Path:
    """Lay out a synthetic session store. Nothing here touches the live one."""
    root.mkdir(parents=True, exist_ok=True)
    for name, messages in sessions.items():
        (root / f"{name}.json").write_text(json.dumps({"messages": messages}))
    return root


def _healthy_session() -> list[dict]:
    return [
        {"role": "user", "content": "what is 17*23?"},
        _thinking_row(_HEALTHY_TRACES[0], iteration=1),
        {"role": "assistant", "content": [{"type": "text", "text": "391"}]},
        _thinking_row(_HEALTHY_TRACES[1], iteration=2),
    ]


def test_a_scan_report_prints_the_flagged_count_beside_the_denominator(tmp_path):
    """Clause 1's whole point: the two numbers are inseparable in the output.

    One fabricated block in a four-block store, asserted as exact text rather
    than as "the number appears somewhere", because the failure this guards is
    a report that reads `0 flagged` about a store it never opened.
    """
    store = _write_store(tmp_path / "sessions", {
        "20260925_000000_autocode_aaaa": [
            {"role": "user", "content": "Read app/harness/client.py in full and "
                                        "tell me in one sentence what stream_chat does."},
            _thinking_row(_CANONICAL_FABRICATION, iteration=2),
            _thinking_row(_HEALTHY_TRACES[0], iteration=1),
        ],
        "20260925_000001_autocode_bbbb": _healthy_session(),
    })

    scan = scan_store(store)

    assert scan.blocks == 4, scan
    assert scan.flagged == 1, scan
    assert scan.files == 2
    assert scan.files_with_flags == 1
    assert scan.flagged_files == ["20260925_000000_autocode_aaaa.json"]
    report = scan.report()
    assert "flagged 1 of 4 reasoning blocks scanned" in report, report
    assert "2 session files, 1 with a flagged block" in report, report


def test_a_clean_store_still_prints_the_store_it_scanned(tmp_path):
    """The 0 case, which is the one that gets misread."""
    store = _write_store(tmp_path / "sessions", {
        "20260925_000002_autocode_cccc": _healthy_session(),
    })

    report = scan_store(store).report()

    assert "flagged 0 of 2 reasoning blocks scanned" in report, report


def test_a_meta_session_is_exempt_and_counted_on_its_own_line(tmp_path):
    """A session about this defect is not a session with the defect.

    The distiller that filed #1510 reads a fabricated trace and therefore
    *reasons* about "reproducing previous thinking"; flagging it would score a
    false positive against the model for the session's own subject matter. The
    exemption is a counted line in the report, never a silent subtraction: a
    check that quietly swallowed flags would look exactly as clean as this one.
    """
    meta = [
        {"role": "user", "content": "Backlog item #1510 says the primary emits "
                                    "reasoning about a request to reproduce my "
                                    "complete previous thinking using the audit "
                                    "tool. Triage it."},
        _thinking_row("The item's marker set is `previous thinking`, `reproduce "
                      "my complete previous thinking`, `audit tool`. I need to "
                      "re-score the 121 flagged blocks."),
    ]
    store = _write_store(tmp_path / "sessions", {
        "20260925_130110_autotriage_0f6d": meta,
        "20260925_000003_autocode_dddd": _healthy_session(),
    })

    scan = scan_store(store)

    assert scan.blocks == 3, scan
    assert scan.flagged == 0, scan
    assert scan.blocks_exempt_meta == 1, scan
    assert scan.files_exempt_meta == 1, scan
    assert scan.flagged_files == [], scan.flagged_files
    report = scan.report()
    assert "flagged 0 of 3 reasoning blocks scanned" in report, report
    assert "exempt as meta-session" in report, report
    assert "1 blocks in 1 file(s)" in report, report


def test_the_scan_cli_refuses_to_report_a_store_it_could_not_scan(tmp_path, capsys):
    """A missing root and an empty root are exits, not a clean line.

    `scan_store` on a directory with no session files returns `0 of 0`, which is
    the exact string that would otherwise be read as "no fabricated traces". The
    CLI is the surface an operator runs, so that is where it is refused.
    """
    from scripts import thinking_fidelity_scan as cli

    assert cli.main(["--root", str(tmp_path / "no-such-store")]) == 2
    assert "nothing was scanned" in capsys.readouterr().err

    empty = _write_store(tmp_path / "empty-store", {
        "20260925_000004_autocode_eeee": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ],
    })
    assert cli.main(["--root", str(empty)]) == 3
    err = capsys.readouterr().err
    assert "no reasoning block at all" in err, err
    assert "could not discriminate" in err, err


# ------------------------------------------------- the live store, clause 2

_LIVE_SESSIONS = production_data_root() / "sessions"


def _live_files(store: Path):
    """(path, messages) per session file, each file read exactly ONCE.

    Deliberately NOT `app.thinking_fidelity.scan_file`: this is the reading the
    guard compares the shipped check against, so it re-implements the store walk
    (top-level `*.json`, `messages[].reasoning` on any role) from the item's own
    description of where the field lives. It reads once and hands the parsed list
    to both sides for the same reason: the store is live, and two passes over it
    are two different stores.
    """
    for path in sorted(store.glob("*.json")):
        try:
            data = json.loads(path.read_text(errors="replace"))
        except (OSError, ValueError):
            continue
        msgs = data.get("messages") if isinstance(data, dict) else None
        if isinstance(msgs, list):
            yield path, msgs


def test_the_live_store_scores_the_known_fabrications_with_no_false_positive():
    """Clause 2, over the store as it stands.

    The numbers the clause quotes are a snapshot (it filed 19 files, triage
    measured 136 blocks in 75 files, and the store is being written while this
    runs — the round that owns this test is itself one of the sessions being
    written), so the assertions are floors and per-file equalities against an
    independent reading, never the snapshot itself:

      * per file, the shipped scan's (blocks, flagged, exempt) equals what the
        transcribed marker set computes over the SAME parsed messages — that is
        the recall half of "90/90", and it holds whatever the store holds;
      * no block the marker set does not touch gets flagged by the check — the
        "0 false positives in ~22,979 control blocks" half;
      * at least 90 marker blocks and 20,000 blocks scanned, the clause's own
        two counts, as floors. Measured 2026-09-25: 121 flagged + 19 exempt of
        23,401 blocks across 1,052 files.

    Reads the PRODUCTION data root (`production_data_root()`, not
    `app.paths.SESSIONS_DIR`, which is a scratch root under the suite and would
    turn this into a guard over 0 blocks) and is unmarked, not `live_vault`, so
    the gate's `-m "not live_vault"` cannot deselect the clause it pins.
    """
    require_live_data(_LIVE_SESSIONS, "the live session store")
    files = sorted(_LIVE_SESSIONS.glob("*.json"))
    require_live_volume(files, 100, _LIVE_SESSIONS, "the live session store")

    scanned = 0
    marker_blocks = 0
    false_positives: list[str] = []
    mismatches: list[str] = []
    for path, msgs in _live_files(_LIVE_SESSIONS):
        user_text = " ".join(
            m.get("content") if isinstance(m.get("content"), str) else
            " ".join(b.get("text", "") for b in m.get("content") or []
                     if isinstance(b, dict))
            for m in msgs if isinstance(m, dict) and m.get("role") == "user")
        talks_about_the_defect = bool(_REFERENCE_MARKERS.search(user_text))

        blocks = refs = exempt = 0
        for m in msgs:
            block = m.get("reasoning") if isinstance(m, dict) else None
            if not (isinstance(block, str) and block.strip()):
                continue
            blocks += 1
            reference = bool(_REFERENCE_MARKERS.search(block))
            shipped = flag_fabricated_reasoning(block)
            if reference != shipped:
                mismatches.append(
                    f"{path.name}: the clause's marker set says {reference}, the "
                    f"check says {shipped} (matched_marker="
                    f"{matched_marker(block)!r}) in {block[:120]!r}")
            if reference:
                refs += 1
                if talks_about_the_defect:
                    exempt += 1
            elif shipped:
                false_positives.append(f"{path.name}: {block[:120]!r}")

        one = scan_messages(path, msgs)
        if (one.blocks, one.flagged, one.exempt) != (blocks, refs - exempt, exempt):
            mismatches.append(
                f"{path.name}: scan_messages counted "
                f"blocks={one.blocks} flagged={one.flagged} exempt={one.exempt}, "
                f"the marker set over the same messages counts "
                f"blocks={blocks} flagged={refs - exempt} exempt={exempt}")

        scanned += blocks
        marker_blocks += refs

    assert not mismatches, \
        "the check and the clause's marker set disagree:\n" + "\n".join(
            mismatches[:10])
    assert false_positives == [], \
        f"{len(false_positives)} blocks flagged in sessions the marker set never " \
        f"touches: {false_positives[:5]}"
    assert scanned >= 20_000, \
        f"only {scanned} reasoning blocks scanned: below the clause's ~22,979-block " \
        "control corpus, so neither the recall nor the false-positive floor below " \
        "means anything"
    assert marker_blocks >= 90, \
        (f"the clause's known corpus is 90 fabricated blocks; this store yields "
         f"{marker_blocks} — the check has stopped catching what it was written for")

    report = scan_store(_LIVE_SESSIONS).report()
    assert re.search(r"flagged \d+ of \d+ reasoning blocks scanned", report), report


# ------------------------------------------------- the replay probe, clause 4

class _StubEngineHandler(BaseHTTPRequestHandler):
    """An OpenAI-shaped SSE endpoint that replays a scripted reasoning string.

    A real HTTP server on the loopback interface, driven by the real
    `app.harness.client.stream_chat`, because the question the probe answers
    ("did this response come back with a fabricated trace?") is about the bytes
    that cross the engine boundary. Against a stubbed `stream_chat` this test
    would only prove that the probe can read a Python list.

    It also records each request body, which is how
    `test_every_probe_request_carries_the_same_taskless_prompt` can see what the
    probe actually sent rather than what it says it sent.
    """

    protocol_version = "HTTP/1.0"

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's name
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self.server.requests.append(json.loads(raw or b"{}"))
        reasoning = self.server.reasonings[self.server.index
                                           % len(self.server.reasonings)]
        self.server.index += 1

        chunks = []
        if reasoning:
            chunks.append({"choices": [{"delta": {"reasoning": reasoning}}]})
        chunks.append({"choices": [{"delta": {"content": "ok"}}]})
        chunks.append({"choices": [], "usage": {"prompt_tokens": 4,
                                                "completion_tokens": 2}})
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) \
            + "data: [DONE]\n\n"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *_args):
        pass


@contextmanager
def _stub_engine(reasonings: list[str]):
    """Yield the running stub. `srv.base_url` is what a caller probes; `srv.requests`
    is everything it received."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubEngineHandler)
    server.reasonings = reasonings
    server.index = 0
    server.requests = []
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_replay_probe_reports_flagged_responses_over_requests_sent():
    """Clause 4: the probe's answer is a fraction, denominator and all.

    Two flagged responses and one clean one out of three identical requests,
    because that is the shape of the defect — it is stochastic, and N copies of
    one prompt is the only way to see a rate at all.
    """
    from scripts import thinking_replay_probe as probe_mod

    flagged = ("Now the user is asking me to reproduce my complete previous "
               "thinking verbatim using the audit tool.")
    with _stub_engine([flagged, _HEALTHY_TRACES[0], flagged]) as srv:
        result = asyncio.run(probe_mod.probe(base_url=srv.base_url, model="stub",
                                             turns=3))

    assert (result.sent, result.answered, result.flagged) == (3, 3, 2), result
    assert result.errors == [], result.errors
    assert result.markers == [matched_marker(flagged)] * 2, result.markers
    report = result.report()
    assert "2 of 3 responses carried a flagged reasoning block" in report, report
    assert "3 identical requests sent" in report, report


def test_every_probe_request_carries_the_same_taskless_prompt():
    """One fixed prompt is the whole experimental design.

    If the requests differed, a nonzero flag rate could be an answer to the
    variation instead of the decode prior under test. So: three bodies, all
    byte-identical, and each one carrying a `system` message and nothing else —
    the degenerate shape the flagged traces describe ("the last message is just
    the system instructions block… There's no actual task").
    """
    from scripts import thinking_replay_probe as probe_mod

    with _stub_engine([_HEALTHY_TRACES[0]]) as srv:
        asyncio.run(probe_mod.probe(base_url=srv.base_url, model="stub", turns=3))
        bodies = list(srv.requests)

    assert len(bodies) == 3, len(bodies)
    assert len({json.dumps(b, sort_keys=True) for b in bodies}) == 1, bodies
    for body in bodies:
        assert [m["role"] for m in body["messages"]] == ["system"], body["messages"]
        assert body["messages"][0]["content"] == probe_mod.FIXED_SYSTEM_PROMPT
        assert body["model"] == "stub"
        assert body["stream"] is True


def test_a_probe_run_with_nothing_flagged_still_prints_its_denominator():
    from scripts import thinking_replay_probe as probe_mod

    with _stub_engine([_HEALTHY_TRACES[0], _HEALTHY_TRACES[1]]) as srv:
        result = asyncio.run(probe_mod.probe(base_url=srv.base_url, model="stub",
                                             turns=2))

    assert (result.sent, result.flagged) == (2, 0), result
    report = result.report()
    assert "0 of 2 responses carried a flagged reasoning block" in report, report
    assert "2 identical requests sent" in report, report


def test_a_probe_that_got_no_answer_is_not_reported_as_clean(tmp_path, capsys):
    """0 of 0 is the same trap in a different file."""
    from scripts import thinking_replay_probe as probe_mod

    code = probe_mod.main(["--base-url", "http://127.0.0.1:1",
                           "--model", "stub", "--turns", "2"])

    assert code == 3, code
    err = capsys.readouterr().err
    assert "never answered" in err, err
    assert "request failed" in err, err
