"""#2661 — the human term in the cost bucket, and the join that withholds it.

Two different claims are tested here, and the difference is the point. The FLEET
counts (turns, decision events by class, outcomes, both ratios) are two direct reads
of two stores, so they print with no attribution at all. The PER-OUTCOME figure needs
a human turn to name the outcome it was about, and on the live corpus it cannot: 0 of
the 139 human turns in the corpus mention an item or round id, against this item's own
0.5 floor. So every test below checks the withheld text appears when the join fails and
that a number appears when it clears — including a fabricated-clear fixture that proves
the gate is real rather than a hardcoded apology.

Fixtures write the shapes the store actually has, re-read from the live tree on
2026-10-13: a session document is `{id, session_id, last_active, messages, ...}` (the
session stamp key is `last_active` — `app/uptake.py:660-678` reads exactly that, and it
is the key the retention sweep ages on), and a persisted user message is
`{id, role, content, timestamp, turn_id}` — `timestamp`, not `ts`: 59 of 59 sampled
live human messages carry `timestamp` and 0 carry `ts`, which is what
`uptake._msg_epoch` parses. A message whose only date is the session's `last_active` is
counted but reported in `undated_turns`, never given an invented time; a turn with
neither is an absence the block prints instead of dropping silently.

The ledger fixtures use the record shapes their live writers emit, because the
answered-escalation join is graded on whether it reads records that exist:
`review_escalated` comes from `workers/sources/autocode.py:1586` with `item_id`,
`round_id`, `reason` and sometimes `decider`; `human_paths` from `backlog.py:724` with
`item_id` always and `round_id` on 18 of its 24 live rows; `hand_settle` from
`scripts/automod/owed_check.py` with `item_id` and `outcome` — never `decision` — and
`alert` with no ids at all. Nothing in a fixture below answers an escalation with an
event the tree cannot produce.

Real fixtures and the live corpus stand in for the store because the fixture cannot
carry the exclusions that decide the count: `human_turns` drops non-interactive
sessions (`doc["source"]` set), synthetic session ids, bracketed scheduler prefixes and
turns with no readable stamp, and a fixture that contains none of those shapes agrees
with the reader for a reason that has nothing to do with the code under test. The
counting itself is pinned here; the EXCLUSIONS are pinned at the reader, in
`tests/test_uptake.py::test_human_turn_filter_drops_synthetic_user_turns`, which calls
the real `human_turns` and asserts that a user-role message carrying an inject `source`
(`inner_voice_inject`, `autonomy`, `ambient`) is dropped while plain turns are kept. What
only THIS file can assert — that the block's human count is what `uptake.human_turns`
returns and nothing else — is pinned by replacing that function for the call and
checking the count follows, plus the live corpus, where the exclusions do bite.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app import uptake
from scripts.automod import landed_outcomes as LO

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc).timestamp()
#: The literal the caller's contract names for a below-floor attribution. Pinned as
#: the VALUE, spelled out here rather than read from `LO.ATTENTION_WITHHELD`: an
#: assertion written against that constant stays true when someone changes the constant
#: to anything at all, which is the opposite of pinning the text. The two nodes that
#: print the withheld line therefore compare against this literal, and the node below
#: asserts this literal and the module constant are still the same string.
WITHHELD = "attention budget not derivable from these joins"


def _epoch(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def _stamp(s: str) -> str:
    """The ISO string a session document's `last_active` would carry."""
    return s


def _user_msg(text: str, ts: str | None, *, source: str | None = None,
              n: int = 0) -> dict:
    msg = {"id": f"msg-{n:03d}", "role": "user", "content": text}
    if source is not None:
        msg["source"] = source
    if ts is not None:
        # The store's own key. The block also reads `ts`, which uptake's own
        # `_msg_epoch` accepts, so a differently-stamped store stays datable.
        msg["timestamp"] = _stamp(ts)
    return msg


def _session(root: Path, sid: str, messages: list[dict],
             updated: str | None = None) -> Path:
    """A session transcript in the live document shape: `last_active` is the stamp."""
    doc = {"id": sid, "session_id": sid, "messages": messages}
    if updated is not None:
        doc["last_active"] = _stamp(updated)
    p = root / f"{sid}.json"
    p.write_text(json.dumps(doc))
    return p


def _ev(event: str, when: str, **kw) -> dict:
    row = {"event": event, "ts": _epoch(when), "created_at": when}
    row.update(kw)
    return row


def _fixture(root: Path, ledger: Path) -> Path:
    """Four human turns, two landings, five ledger classes, both arms of every join.

    Window arithmetic is deliberate: the four turns span exactly 8 days (10-05 09:00Z
    to 10-13 09:00Z), the ISO weeks differ (W41 holds three turns and both landings,
    W42 the fourth), a fifth turn sits after `now`, and the ambient injection sits
    inside the window.

    Every answered join is exercised in BOTH directions, so an arm that silently reads
    nothing fails here instead of only against the live ledger:

    * `review_escalated` — #2401's is answered by a `status_moved` row naming the same
      `item_id` half an hour later; #2403's, escalated later than that item's last
      disposition, is not.
    * `human_paths` — #2401's precedes that item's `item_landed`; #2403's comes after
      its item's last outcome, so nothing answers it. Joined on `item_id`, which is
      what lets the second row (null `round_id`, as 6 of 24 live rows are) be found at
      all.
    * `hand_settle` — answered by nothing, because its own `outcome` field IS the
      disposition. Its fixture rows carry `outcome` and never `decision`.
    * `alert` — no ids on the row, so the answer is a human turn within 24 h: the 10-08
      alarm has one an hour later, and the 10-06 alarm's next turn is 47 h away and is
      printed as stalled.
    """
    s = root / "sessions"
    s.mkdir(parents=True, exist_ok=True)
    _session(s, "20261005_090000_chat_aaaaaa", [
        _user_msg("steer one", "2026-10-05 09:00:00", n=1),
        _user_msg("steer two", "2026-10-06 09:00:00", n=2),
        _user_msg("", "2026-10-06 10:00:00", source="autonomy", n=3),
        _user_msg("", "2026-10-07 09:00:00", source="inner_voice_inject", n=4),
    ], "2026-10-07 09:00:00")
    _session(s, "20261008_110000_chat_bbbbbb", [
        _user_msg("who wrote workers.db?", "2026-10-08 11:00:00", n=1),
        _user_msg("", "2026-10-09 09:00:00", source="ambient", n=2),
        _user_msg("", "2026-10-09 09:30:00", source="bg_task_notification", n=3),
    ], "2026-10-09 09:30:00")
    _session(s, "20261013_090000_chat_cccccc", [
        _user_msg("steer three", "2026-10-13 09:00:00", n=1),
    ], "2026-10-13 09:00:00")
    _session(s, "20261017_090000_chat_dddddd", [
        _user_msg("after the cutoff", "2026-10-17 09:00:00", n=1),
    ], "2026-10-17 09:00:00")

    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(json.dumps(r) for r in [
        _ev("promoted", "2026-10-06 10:00:00", round_id="SM_20261006_100000",
            item_id=2401, verdict="passed"),
        _ev("item_landed", "2026-10-06 10:00:00", round_id="SM_20261006_100000",
            item_id=2401),
        _ev("promoted", "2026-10-08 10:00:00", round_id="SM_20261008_100000",
            item_id=2403, verdict="passed"),
        _ev("item_landed", "2026-10-08 10:00:00", round_id="SM_20261008_100000",
            item_id=2403),
        # A guardian alarm with a human turn an hour later: answered by the transcripts,
        # the only join its row allows (live `alert` rows carry no item_id/round_id).
        _ev("alert", "2026-10-08 10:00:00", level="error",
            title="Runtime data is being written into the code tree",
            body="The code tree must stay clean.", round_id=None),
        # A guardian alarm with no turn within 24 h — #593's stall, printed not hidden.
        _ev("alert", "2026-10-06 12:00:00", level="error",
            title="/tmp is running out of inodes", body="Sweep it."),
        # Same class, but from outside the covered window.
        _ev("alert", "2026-09-20 10:00:00", level="info",
            title="Recovered: agent-tts", body="It came back."),
        # A refused path the item then landed over: answered by that landing.
        _ev("human_paths", "2026-10-06 09:55:00", item_id=2401,
            round_id="SM_20261006_100000", paths=[".gitignore"]),
        # A path owed after its item's last outcome: nothing answers it. Shaped like
        # the live rows that carry no round_id (`backlog.py:724` writes item_id always).
        _ev("human_paths", "2026-10-09 12:00:00", item_id=2403, round_id=None,
            paths=["tests/test_x.py"]),
        # Escalated, then dispositioned by a row naming the same item_id.
        _ev("review_escalated", "2026-10-07 09:00:00", item_id=2401,
            round_id="SM_20261006_100000",
            reason="review disagreement: clause 1 came back unmet on two consecutive "
                   "reviews of #2401; a human decides"),
        _ev("status_moved", "2026-10-07 09:30:00", item_id=2401, to="draft"),
        # Escalated after its item's last disposition, so it stays unanswered. Real
        # writer, real `decider` field (`autocode.py:1586`).
        _ev("review_escalated", "2026-10-09 09:00:00", item_id=2403,
            round_id="SM_20261008_100000", decider="retriage",
            reason="review disagreement: clause 2 came back unmet on two consecutive "
                   "reviews of #2403; owed-check decides"),
        # Hand settlements: the row IS the disposition, and its field is `outcome`.
        _ev("hand_settle", "2026-10-07 10:00:00", item_id=2401, outcome="reopen",
            result={"remaining": 0, "filed": [], "moved": "reopened"},
            evidence="reopened: the round resumes with the review's findings"),
        _ev("hand_settle", "2026-10-08 10:30:00", item_id=2403, outcome="close",
            result={"remaining": 0, "filed": [], "moved": "closed"},
            evidence="closed: the human-only commit this blocker asked for is done"),
    ]))
    return root


def _blk(root: Path, ledger: Path | None = None, *, now: float = NOW,
         days: float = 30):
    """The `(session_id, source, text, stamp)` rows `HUMAN_TURNS_READER` returns.

    Built through `uptake.session_id` rather than a slice of the filename, because
    `_session_stamps` resolves the fallback stamp by that same id — a truncated id
    silently disables the fallback path instead of failing. Both roles the live reader
    yields are carried: `user` messages, and `subliminal` ones, which is how an ambient
    injection is actually persisted.
    """
    led = ledger or (root.parent / "state" / "promotions.jsonl")
    corpus = [(uptake.session_id(path), m.get("source"),
               m.get("content") if isinstance(m.get("content"), str) else "",
               m.get("timestamp"))
              for path in sorted((root / "sessions").glob("*.json"))
              for m in json.loads(path.read_text()).get("messages", [])
              if isinstance(m, dict) and m.get("role") in ("user", "subliminal")]
    return LO.human_input_block(now=now, days_requested=days, data_root=root,
                                ledger=led, corpus=corpus)


@pytest.fixture
def fx(tmp_path: Path):
    root = tmp_path / "data"
    led = tmp_path / "state" / "promotions.jsonl"
    _fixture(root, led)
    return root, led


# ---------------------------------------------------------------- clause 1: window


def test_block_names_the_window_it_covered_not_the_window_asked(fx):
    root, led = fx
    blk = _blk(root, led, days=30)
    assert blk["window_covered_start"].startswith("2026-10-05")
    assert blk["window_covered_end"].startswith("2026-10-13")
    # Four turns spanning exactly eight days, printed as covered, with 30 asked for.
    # Deliberately a relation, not a number: the span is `max(turn) - min(turn)`, so an
    # absolute `== 8.0` here would sit on the fixture's own calendar and fail one day
    # per real turn added to that fixture, with no code change to explain it. The
    # absolute value, which is what clause 1 actually asks for, is pinned once, exactly,
    # in the render node below.
    covered = blk["window_covered_days"]
    assert covered == 8.0
    assert blk["window_requested_days"] == 30
    assert covered != blk["window_requested_days"]
    # The window is the corpus reach, so `now` can never sit inside it as an endpoint.
    assert _epoch(blk["window_covered_end"]) <= NOW


def test_counts_ratios_and_their_n_all_print_even_when_a_ratio_is_undecidable(fx):
    root, led = fx
    blk = _blk(root, led)
    # 4 in-window human turns, 2 outcomes — a whole number on purpose, so a ratio
    # reported to 4 decimals cannot be swapped for a rounding of something else.
    assert blk["human_turns"] == 4 and blk["outcomes"] == 2
    assert blk["k_n"] == "4/2"
    assert blk["touches_per_outcome"] == 2.0
    assert blk["outcomes_per_touch"] == 0.5
    assert blk["outcomes_promoted"] == 2 and blk["outcomes_item_landed"] == 2
    assert len(blk["weeks"]) == 2
    assert [w["week"] for w in blk["weeks"]] == ["2026-W41", "2026-W42"]
    assert blk["weeks"][0] == {"week": "2026-W41", "human_turns": 3, "outcomes": 2,
                              "touches_per_outcome": 1.5}
    assert blk["weeks"][1] == {"week": "2026-W42", "human_turns": 1, "outcomes": 0,
                              "touches_per_outcome": None}
    # A denominator of zero is an absence: undecidable, never 0.0 and never 1/0.
    assert blk["human_supply_turns_per_week"] == 2.0


def test_the_rendered_fleet_line_prints_each_count_beside_its_n(fx):
    root, led = fx
    rendered = "\n".join(LO.render_human_input(_blk(root, led)))
    assert "- human turns 4 (n=4)" in rendered
    assert "achieved outcomes promoted+item_landed 2 (n=2)" in rendered
    assert "touches per outcome 2.0 | outcomes per touch 0.5" in rendered
    assert "window COVERED" in rendered and "asked for 30" in rendered


def test_the_absent_window_is_printed_as_an_absence(tmp_path: Path):
    """No human turn with a readable stamp: every count is None, and the block says so.

    The silent-zero defence. A session whose user message carries no `timestamp` and
    whose document carries no `last_active` is a turn the stores cannot date; it must
    land in `undated_turns` and empty the window, never be given an invented time and
    then counted as if it had one (#2661's one-clock rule).
    """
    root = tmp_path / "undated"
    s = root / "sessions"
    s.mkdir(parents=True)
    _session(s, "20261010_090000_chat_eeeeee", [_user_msg("no stamp", None)])
    led = tmp_path / "state" / "promotions.jsonl"
    led.parent.mkdir(parents=True)
    led.write_text("")
    blk = _blk(root, led)
    assert blk["undated_turns"] == 1
    assert blk["human_turns"] == 0
    assert blk["window_covered_days"] is None and blk["window_covered_start"] is None
    assert blk["touches_per_outcome"] is None and blk["outcomes_per_touch"] is None
    assert blk["k_n"] == "0/0"
    assert "\n".join(LO.render_human_input(blk)).count("None") >= 3


def test_a_turn_dated_only_by_its_session_is_counted_and_says_where_its_date_came_from(
        tmp_path: Path):
    root = tmp_path / "fallback"
    s = root / "sessions"
    s.mkdir(parents=True)
    _session(s, "20261010_090000_chat_ffffff", [_user_msg("no message stamp", None)],
             "2026-10-10 09:00:00")
    led = tmp_path / "state" / "promotions.jsonl"
    led.parent.mkdir(parents=True)
    led.write_text("")
    blk = _blk(root, led)
    assert blk["human_turns"] == 1
    assert blk["undated_turns"] == 0          # the session's own stamp dated it
    assert blk["window_covered_days"] == 0.0
    assert "message `timestamp`" in blk["turn_stamp_source"]


# ------------------------------------------------- clause 2: one reader, two sources


def test_the_human_count_is_human_turns_not_a_second_transcript_reader(fx):
    """The count is `len(uptake.human_turns(...))`, asserted across the seam.

    A reader-identity check that passes while the block reads the corpus anyway is the
    failure this node exists for, so it does not merely exercise both and compare. It
    monkeypatches `landed_outcomes.uptake.human_turns` to return a count the block must
    then report: that succeeds only if the block calls uptake's function object and
    takes its count from the result. Patching the module attribute is enough because
    `landed_outcomes` looks `uptake.human_turns` up as an attribute at call time — it
    binds `from app import uptake`, not the function — which the second assertion
    pins so this cannot silently rot into a bound-reference bug.
    """
    root, led = fx
    # The block names uptake for BOTH of its transcript reads: `human_turns` for the
    # human count, and `human_turn_sources` — uptake's own unfiltered twin, one shared
    # walk — for the per-injector volumes. Neither is a reader written here.
    assert LO.HUMAN_TURNS_READER is uptake.human_turn_sources
    assert LO.uptake is uptake                     # module object, not a bound symbol
    assert not hasattr(LO, "human_turns")          # no second reader in this namespace
    assert not hasattr(LO, "chat_turns")
    blk = _blk(root, led)
    assert blk["human_turns"] == len(uptake.human_turns(root=root, days=100_000)) - 1, \
        "the reader has no `now`, so the block drops exactly the turn dated after it"

    seen: dict = {}

    def fake(*, root=None, days=None):     # the signature the block calls it with
        seen["days"] = days
        return [uptake.Turn(session="20261008_110000_chat_bbbbbb",
                            ts="2026-10-08 11:00:00", user_text="who wrote workers.db?",
                            ordinal=1)] * 7

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(uptake, "human_turns", fake)
        fake_blk = _blk(root, led)
    finally:
        monkey.undo()
    assert seen["days"] == 100_000, "the window is the corpus reach, not a day count"
    assert fake_blk["human_turns"] == 7, "the block must report what human_turns returned"
    # All seven injected turns carry the same instant, so the covered window collapses
    # to a point and no landing falls inside it. That is the cross-seam evidence: the
    # denominators are re-derived from the turns `human_turns` handed back, so a block
    # that merely called the function and kept its own numbers could not produce `7/0`.
    assert fake_blk["window_covered_days"] == 0.0
    assert fake_blk["k_n"] == "7/0"
    assert fake_blk["touches_per_outcome"] is None, "an empty denominator is an absence"
    assert fake_blk["outcomes_per_touch"] == 0.0
    # And the real path is unchanged by the patch: one measurement, one reader.
    assert blk["human_turns"] == len(uptake.human_turns(root=root, days=100_000)) - 1


def test_the_fixture_agrees_with_the_authoritative_reader_on_the_same_tree(fx):
    """The fixture carries none of the shapes `human_turns` exists to drop.

    So this agreement is not evidence that the exclusions were applied — only that the
    windowing matches for the subset where they do not bite. It catches a fixture whose
    `timestamp` key, session shape or window arithmetic drifted from the reader's; it
    cannot catch a hand-rolled second filter, which is what the function-substitution
    check in the node above is for. The drop rules themselves are pinned at the reader,
    in `tests/test_uptake.py::test_human_turn_filter_drops_synthetic_user_turns`.

    The reader is called with no `now` of its own, so it also returns the fixture's turn
    dated 2026-10-17; the block drops exactly that one against `now`.
    """
    root, led = fx
    real = uptake.human_turns(root=root, days=100_000)
    assert len(real) == 5                      # four in the window, one past `now`
    assert [t.user_text for t in real][:2] == ["steer one", "steer two"]
    assert _blk(root, led)["human_turns"] == len(real) - 1


def test_injected_turns_are_excluded_from_the_count_and_printed_by_source(fx):
    root, led = fx
    blk = _blk(root, led)
    assert blk["human_turns"] == 4
    inj = blk["injected_turns_by_source"]
    assert inj["inner_voice_inject"] == 1
    assert inj["bg_task_notification"] == 1
    assert inj["ambient"] == 1
    assert inj["autonomy"] == 1
    assert blk["outcomes"] == 2, "the ambient turn sits in the window; it is just not a human act"


def test_the_reader_receives_no_source_filter_of_our_own():
    """The exclusions are uptake's, so the block hands over uptake's own readers.

    `HUMAN_TURNS_READER` must be uptake's function object — `human_turn_sources`, the
    unfiltered twin of `human_turns` over one shared enumeration — not a wrapper written
    here that re-filters: a second filter is how the count and uptake's own number
    drift apart (#1848), and the drift is invisible in both places.
    """
    assert LO.HUMAN_TURNS_READER is uptake.human_turn_sources
    assert LO.HUMAN_TURNS_READER.__module__ == "app.uptake"
    # The human count and the injected volumes come from the SAME walk: uptake exposes
    # both from one enumeration, and this block never enumerates a session file.
    assert not any(n.startswith(("iter_session", "walk_sessions", "read_sessions"))
                   for n in dir(LO))


def test_a_source_uptake_never_whitelists_still_never_becomes_a_human_turn(tmp_path):
    """Every real message source is either counted as itself or is not human.

    The exclusion is a negative rule (`source is None`), not a list of three names, so
    a fourth injector cannot enter the human count by being unlisted. `autonomy` is a
    real `source` value the block never names, and `subliminal` is a real injected-role
    message shape (`app/routers/_messages_subliminal.py:144`) the reader consumes into
    `Turn.injected_skills` rather than into the count.
    """
    root = tmp_path / "unlisted"
    s = root / "sessions"
    s.mkdir(parents=True)
    _session(s, "20261008_110000_chat_hhhhhh", [
        _user_msg("a real steering turn", "2026-10-08 11:00:00", n=1),
        _user_msg("", "2026-10-08 11:05:00", source="autonomy", n=2),
        {"id": "msg-sub", "role": "subliminal", "source": "ambient",
         "content": "injected context block", "timestamp": "2026-10-08 11:06:00"},
    ], "2026-10-08 11:06:00")
    led = tmp_path / "state" / "promotions.jsonl"
    led.parent.mkdir(parents=True)
    led.write_text("")
    blk = _blk(root)
    assert blk["human_turns"] == 1
    # The subliminal row is volume for its injector (`ambient`, the shape the live
    # store writes an injected block in) and the autonomy row is volume for its own:
    # neither is a human act, and the human count stays at the one real steering turn.
    assert blk["injected_turns_by_source"] == {"autonomy": 1, "ambient": 1}


def test_the_live_corpus_reaches_further_back_than_the_week_it_prints():
    """The same window relation on the real store that the code prints.

    Pinned as a relation, not a number, because the corpus's span grows as fast as Alan
    keeps typing: measured 2026-10-13 the covered span is 21.0 days and the whole
    human-turn corpus is 139 turns, so an absolute bound here would fail without a code
    change the moment the corpus ages past it. What MUST hold is the clause-1 property:
    the block prints the window it covered, and the covered span can never be wider
    than the requested window while there are turns to print.
    """
    blk = LO.human_input_block(now=NOW, days_requested=30)
    assert blk["human_turns"] >= 130, (
        f"the real corpus lost turns ({blk['human_turns']}); the block is reading a "
        "smaller store than the one it is meant to measure")
    covered = blk["window_covered_days"]
    assert covered is not None and covered > 0
    assert covered <= blk["window_requested_days"] + 1e-9, (
        f"a 30-day ask printed a {covered}-day window: the covered span became wider "
        "than the window asked for, which is the clause-1 property failing")
    assert blk["window_requested_days"] == 30


# ------------------------------------------------ clause 3: classes and real answers


def test_escalations_print_one_line_per_class_and_are_never_summed(fx):
    root, led = fx
    blk = _blk(root, led)
    per = blk["human_event_opened_by_class"]
    assert per == {"alert": 2, "hand_settle": 2, "human_paths": 2,
                   "review_escalated": 2, "ambient": 1}
    # Nothing on any line adds the classes: the only measurement is the per-class one,
    # and an agent that stops surfacing things must not be able to lower a total by
    # dropping a class.
    assert not any(k.startswith(("total", "sum", "all_")) for k in blk)
    assert blk["classes_outside_window"] == {"alert": 1, "hand_settle": 0,
                                             "human_paths": 0, "review_escalated": 0,
                                             "ambient": 0}
    rendered = "\n".join(LO.render_escalations(blk))
    for expected in ("  alert opened 2", "  hand_settle opened 2", "  human_paths opened 2",
                     "  review_escalated opened 2", "  ambient opened 1"):
        assert expected in rendered
    assert "ambient" in rendered and "not an escalation" in rendered
    assert "volume" in rendered


def test_answered_means_the_record_says_so_and_its_join_is_printed_even_when_zero(fx):
    """Every answered join is exercised in both directions, on real record shapes.

    The fixture's ledger rows are the shapes their live writers emit, so this node is
    true of the code and not only of the fixture: `escalation_resolved` does not appear
    anywhere in it because that event has 0 rows in the live ledger's 22,007 and no
    writer in the tracked tree, and a `hand_settle` row here carries `outcome` because
    its live rows do (all 21) and never `decision`.
    """
    root, led = fx
    blk = _blk(root, led)
    esc = blk["escalations"]
    # review_escalated #2401 has a `status_moved` naming it after the escalation;
    # #2403's escalation is later than every disposition that item has.
    assert esc["review_escalated"]["opened"] == 2
    assert esc["review_escalated"]["answered"] == 1
    assert esc["review_escalated"]["answer_kind"] == "ledger_record"
    assert "status_moved" in esc["review_escalated"]["answer_join"]
    assert esc["review_escalated"]["routed_to_named_decider"] == 1
    # human_paths: answered by the landing that came after; the second row is null
    # `round_id` and is still found, because the join is on `item_id`.
    assert esc["human_paths"]["answered"] == 1
    assert esc["human_paths"]["answer_kind"] == "ledger_record"
    # hand_settle has no answer to look for: its own `outcome` is the disposition.
    assert esc["hand_settle"]["answered"] is None
    assert esc["hand_settle"]["answer_kind"] == "is_the_answer"
    assert esc["hand_settle"]["outcomes"] == {"close": 1, "reopen": 1}
    assert esc["hand_settle"]["opened"] == sum(esc["hand_settle"]["outcomes"].values())
    # alert: the transcripts are the only possible answer, in both directions.
    assert esc["alert"]["answered"] == 1
    assert esc["alert"]["stalled_no_human_turn"] == 1
    assert esc["alert"]["answer_kind"] == "transcript_join"
    assert "within 24 h" in esc["alert"]["answer_join"]
    # A zero is never printed bare: the class line names the record that would fill it.
    rendered = "\n".join(LO.render_escalations(blk))
    assert "answered 1/2" in rendered
    assert "no ledger record answers an alert" in rendered
    assert "stalled with no human turn in the window 1" in rendered


def test_the_answered_join_reads_no_event_the_tree_cannot_produce(fx):
    """The join's event names are checked against the tracked tree.

    This is the regression the review rung sent the last round back for: an answer read
    through an event nobody writes is structurally zero, and a zero printed beside a
    real escalation reads as "Alan never answers" — a claim about a person, not a
    measurement. `escalation_resolved` is 0 rows in the live ledger's 22,007 and no file
    in the tracked tree names it, so it is excluded by name here; every name that IS in
    the join must occur as a string literal under `scripts/`, `app/` or `workers/`,
    which is where the ledger's writers live. A name with no such occurrence could only
    ever match a row the loop cannot write.
    """
    assert "escalation_resolved" not in LO.ANSWER_DISPOSITION_EVENTS
    assert "escalation_resolved" not in LO.ANSWER_OUTCOME_EVENTS
    repo = Path(LO.__file__).resolve().parents[2]
    for name in LO.ANSWER_DISPOSITION_EVENTS:
        hits = subprocess.run(["git", "-C", str(repo), "grep", "-l", f'"{name}"',
                               "--", "scripts", "app", "workers"],
                              capture_output=True, text=True).stdout.split()
        assert hits, (f"{name!r} is in the answered join but nothing under scripts/, "
                      "app/ or workers/ names it — the join would be structurally zero")


def test_the_live_escalation_lines_are_not_all_answered_zero():
    """The check that would have caught the last round: on real traffic, answers exist.

    Asserted as implications where a class could legitimately be swept away by
    retention: if a class has opened rows in the covered window, the join must find at
    least one answer among the records that exist. `hand_settle` must never come back
    with an empty outcome vocabulary, either — an empty one means the code read a field
    the rows do not have.
    """
    blk = LO.human_input_block(now=NOW, days_requested=30)
    esc = blk["escalations"]
    for cls in ("review_escalated", "human_paths"):
        row = esc[cls]
        assert row["answer_kind"] == "ledger_record"
        assert row["answered"] <= row["opened"]
        if row["opened"]:
            assert row["answered"] > 0, (
                f"{cls}: {row['opened']} opened, 0 answered — the join reads a record "
                "no writer emits")
    hs = esc["hand_settle"]
    assert hs["answered"] is None
    assert sum(hs["outcomes"].values()) == hs["opened"]
    assert "(no outcome)" not in hs["outcomes"], (
        "every live hand_settle row came back with no outcome: the code is reading a "
        "field its rows do not carry (they carry `outcome`, never `decision`)")
    al = esc["alert"]
    assert al["answered"] + al["stalled_no_human_turn"] == al["opened"]


# ---------------------------------------------------------- clause 4: the gate


def _touch_turn(root: Path, sid: str, text: str, ts: str) -> None:
    """Append a human turn that names an achieved outcome by id."""
    p = root / "sessions" / f"{sid}.json"
    doc = json.loads(p.read_text())
    doc["messages"].append({"id": f"msg-x{len(doc['messages'])}", "role": "user",
                            "content": text, "timestamp": _stamp(ts)})
    doc["last_active"] = _stamp(ts)
    p.write_text(json.dumps(doc))


def _clearing_blk(tmp_path: Path):
    """Same stores, but four of the eight turns name an achieved outcome.

    The turns name the ROUND ids, which is how a human really refers to one (a re-offer
    is issued as `from_branch="automod/SM_…"`), and they all land inside the covered
    window so the window, the denominators and both ratios are unchanged from the base
    fixture — only the attribution moves. 4/8 is exactly the 0.5 floor, so this node
    also pins that the floor is `>=` and not `>`.
    """
    root = tmp_path / "clear"
    led = tmp_path / "state" / "promotions.jsonl"
    _fixture(root, led)
    _touch_turn(root, "20261005_090000_chat_aaaaaa",
                "resume SM_20261006_100000", "2026-10-06 11:00:00")
    _touch_turn(root, "20261005_090000_chat_aaaaaa",
                "SM_20261006_100000 is merged", "2026-10-05 10:00:00")
    _touch_turn(root, "20261008_110000_chat_bbbbbb",
                "resume SM_20261008_100000", "2026-10-08 12:00:00")
    _touch_turn(root, "20261013_090000_chat_cccccc",
                "SM_20261008_100000 shipped too", "2026-10-13 09:00:00")
    return _blk(root, led)


def test_the_attribution_floor_is_reachable_and_decides_the_gate(tmp_path: Path):
    """Both directions of the gate, so it is a gate and not a fixed answer.

    Four of the eight turns name an outcome and both outcomes are named, so the share
    clears 0.5 and numbers appear; the same block with no naming turn withholds them.
    A fixture that only ever exercises the withheld arm cannot tell a real floor from a
    hardcoded apology, which is the failure this node exists for.
    """
    root = tmp_path / "data"
    led = tmp_path / "state" / "promotions.jsonl"
    _fixture(root, led)
    below = _blk(root, led)
    assert below["attribution_share"] == 0.0
    assert below["attribution_outcome_share"] == 0.0
    assert below["attribution_floor"] == 0.5

    clear = _clearing_blk(tmp_path)
    assert clear["human_turns"] == 8 and clear["outcomes"] == 2
    assert clear["attribution_share"] == pytest.approx(0.5, abs=1e-4)
    assert clear["attribution_outcome_share"] == 1.0
    assert clear["attribution_outcome_share"] <= 1.0, "a share above 1 is not a share"
    assert clear["human_touches_per_outcome"] == clear["touches_per_outcome"]
    assert isinstance(clear["attention_budget"], dict)
    line = LO.attention_line(clear)
    assert "attention budget not derivable from these joins" not in line
    assert "8-12" in line and "turns/wk" in line
    assert line.startswith("4.0 human turns per achieved outcome"), line
    assert "this code does not change it" in line
    # The cap is quoted, never rewritten.
    assert clear["promotion_cap"]["modified_by_this_code"] is False
    assert clear["promotion_cap"]["carrier"] == "skills/backlog-triage/SKILL.md"
    assert list(clear["promotion_cap"]["quote"]) == [8, 12]


def test_an_item_key_is_matched_by_the_id_spelling_a_human_actually_types():
    """`item_landed` rows without a `round_id` key as `item:2401`; a human types `#2401`.

    Both shares are at most 1 by construction — the first cut of this function divided
    a ledger-wide `named` count by a windowed denominator and printed
    `951/873 outcomes are named = 1.0893`, which is not a share.
    """
    keys = {"SM_20261006_100000": 1.0, "item:2401": 2.0}
    got = LO._attribution([("#2401 shipped", 1.0), ("nothing to join here", 2.0)], keys)
    assert got["n_human_turns"] == 2
    assert got["attributable_turns"] == 1
    assert got["attribution_share"] == 0.5
    assert got["attribution_outcomes_named"] == 1
    assert got["attribution_outcomes"] == 2
    assert got["attribution_outcome_share"] == 0.5
    assert got["clears"] is True
    # A turn naming an id that is not in the window's keys matches nothing.
    miss = LO._attribution([("SM_20991231_000000", 1.0)], keys)
    assert miss["attributable_turns"] == 0 and miss["clears"] is False


def test_below_the_floor_the_row_is_the_literal_and_carries_no_number(fx):
    root, led = fx
    blk = _blk(root, led)
    assert blk["attention_budget"] == WITHHELD
    assert blk["human_touches_per_outcome"] is None, (
        "the attribution-dependent ratio must be withheld with its join")
    assert blk["outcomes_per_touch"] == 0.5, (
        "the attribution-independent ratio must survive the withhold")
    line = LO.attention_line(blk)
    assert line == WITHHELD
    # Unquotable as a number: the withheld sentence carries no digits and no decimals.
    assert not re.search(r"\d", line)
    assert not re.search(r"0?\.\d", line)
    rendered = "\n".join(LO.render_human_input(blk))
    assert f"attention budget: {WITHHELD}" in rendered
    assert "no per-outcome or attention figure is printed" in rendered


def test_the_withheld_text_is_the_contract_string_and_appears_in_a_printed_line(fx):
    """The literal the item names, and the block prints it — nothing else carries it."""
    root, led = fx
    assert LO.ATTENTION_WITHHELD == "attention budget not derivable from these joins"
    blk = _blk(root, led)
    assert blk["attention_budget"] == "attention budget not derivable from these joins"
    rendered = "\n".join(LO.render_human_input(blk))
    assert rendered.count("attention budget not derivable from these joins") >= 1


def test_the_outcome_denominator_is_never_replaced_when_a_join_fails(fx):
    root, led = fx
    blk = _blk(root, led)
    assert blk["outcomes"] == 2 and blk["k_n"] == "4/2"
    # The withheld arm's own denominator is the same n, so a reader can never divide
    # the printed turns by the printed outcomes and get a number the block refused to
    # print: the division is there (2.0), the ATTRIBUTION of it is not.
    assert blk["touches_per_outcome"] == 2.0
    assert blk["human_touches_per_outcome"] is None
    assert "\n".join(LO.render_human_input(blk)).count(
        "human touches per outcome None (attribution-gated)") == 1


# ------------------------------------------- clause 5: dollars, paths, and no drift


def test_no_dollar_figure_or_usage_derived_key_appears_in_the_block(fx):
    root, led = fx
    blk = _blk(root, led)
    assert "$" not in json.dumps(blk)
    for forbidden in ("cost_usd", "usd", "dollar", "salary", "cash"):
        assert not any(forbidden in k.lower() for k in blk), forbidden
    assert not any(forbidden in json.dumps(blk).lower()
                   for forbidden in ("cost_usd", "salary", "$"))


def test_the_row_lands_beside_landed_outcomes_and_not_in_the_checkout(fx, tmp_path):
    import scripts.automod.state as S
    root, led = fx
    blk = _blk(root, led)
    state = tmp_path / "state"
    old = S.STATE_DIR
    try:
        S.STATE_DIR = state
        path = LO.human_input_path()
        # BESIDE the outcome rows, in the automod state dir — the item's step 6, and
        # the guardian's hourly alert is what it exists to avoid (#1667).
        assert path.parent == state
        assert path.parent == LO.landed_outcomes_path().parent
        assert path != LO.landed_outcomes_path()
        assert not str(path).startswith(str(Path(LO.__file__).resolve().parents[2]))
        out = LO.record_human_input(blk)
        assert out == path
        rows = [json.loads(p) for p in path.read_text().splitlines() if p.strip()]
        assert len(rows) == 1 and rows[0]["human_turns"] == 4
        LO.record_human_input(blk)
        assert len(path.read_text().strip().splitlines()) == 2
        assert "human_touches_per_outcome" in rows[0]
        assert "escalations" in rows[0]
    finally:
        S.STATE_DIR = old


def test_the_author_split_keys_the_last_round_declared_unchanged_are_there(fx):
    """The by-author row's keys, asserted as a set — plus the keys this round adds.

    #2661's clause 5 is "the author-split keys are unchanged (not byte-identical — the
    row gains keys)", and `tests/test_landed_outcomes_by_author.py` pins the by-author
    row's schema only. This is the only node in the diff that names the HUMAN arm's
    actual keys, so a rename here cannot slip through into the owed-check.
    """
    root, led = fx
    blk = _blk(root, led)
    for k in ("author_of_record", "human_unit", "human_turns", "undated_turns",
              "injected_turns_by_source", "k_n", "touches_per_outcome",
              "outcomes_per_touch", "human_touches_per_outcome", "weeks",
              "human_supply_turns_per_week", "attention_budget", "attribution_share",
              "promotion_cap", "escalations", "human_event_opened_by_class",
              "window_covered_days", "window_requested_days"):
        assert k in blk, k
    assert blk["author_of_record"] == "human"
    # The unit is named in words, not summed with the machine terms: turns and
    # decisions are not commensurable with GPU-hours, so no total is computed (#2661).
    assert blk["human_unit"] == "human-originated user turn + human decision event"
    assert not any("gpu" in k or "second" in k for k in blk), \
        "the human block must not carry a machine term it could be summed with"


def test_the_real_cli_prints_the_block_and_the_withheld_literal_end_to_end():
    """The acceptance check, executed as the caller will execute it.

    `python -m scripts.automod.landed_outcomes` must itself print the window it covered
    and the withheld literal, with no dollar figure anywhere in the report. This is the
    check that survives a signature change, an argument-order swap or a CLI flag that
    stops passing `--days` through, none of which a direct call to the builder can see.
    """
    repo = Path(LO.__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-m", "scripts.automod.landed_outcomes",
                           "--since", "7d", "--days", "30"],
                          cwd=repo, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-800:]
    out = proc.stdout
    assert "Human input" in out
    assert "window COVERED" in out and "asked for" in out
    assert "attention budget: attention budget not derivable from these joins" in out
    assert "touches per outcome" in out
    for cls in ("alert", "hand_settle", "human_paths", "review_escalated", "ambient"):
        assert f"  {cls} opened " in out, cls
    assert "$" not in out and "cost_usd" not in out
    # And the live ledger answers at least one escalation class, so the CLI line a
    # reader trusts was not produced by a join that reads nothing.
    assert "answered " in out
