"""#2717 clause 1 — the uptake probe persists the per-turn dispute map it computes.

`uptake.build_uptake_table` decides, turn by turn, whether a run disputed the assistant's
work, and for two years it kept only the tally: `corpus.disputes` said how many, a row's
`disputes` said how many of ITS runs, and nothing in the written artifact said WHICH turns.
That is the join #2717 needs — "this item was retrieved for that run, and that run went
badly" — and it could not be reconstructed from a table by anyone, including the author of
the run. The consumer that needed it therefore re-ran the classifier to get an answer, which
is the engine grading its own output (#2269) and, as it happened, a call with the wrong
arity that crashed on first use.

The fix is one key. These nodes pin that the map is written, that it survives the round trip
through `write_table` to the file a later process opens, that the counts beside it are the
counts a reader needs to know how much of the window the map actually covers, and that its
meaning travels in the artifact's own glossary rather than in a comment.

Run from a worktree, never from `~/lloyd` (the live checkout is the running system):
    HOME=/home/alansrobotlab .venvs/lloyd/bin/python -m pytest \\
      tests/test_uptake_verdict_map.py -p no:cacheprovider -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from app import uptake  # noqa: E402


def _turns():
    """Three turns, one session: disputed, cleared, and screened-but-unanswered.

    The third is the case a count cannot carry. `disputes` can say 1 of 2, and neither
    number says that a turn the cue screen reached came back from the engine with nothing —
    which is missing evidence, not a clean bill.
    """
    return [
        uptake.Turn(session="s1", ts=None, user_text="no, the other way", ordinal=1,
                    vault_context=["Alpha rule"], injections_seen=1),
        uptake.Turn(session="s1", ts=None, user_text="carry on", ordinal=2,
                    vault_context=["Alpha rule", "Beta rule"], injections_seen=1),
        uptake.Turn(session="s1", ts=None, user_text="and the tts tree?", ordinal=3,
                    vault_context=["Beta rule"], injections_seen=1),
    ]


def _entries():
    return [uptake.Entry(source="lloyd/USER.md",
                         text="Prefer the larger batch when the queue looks long")]


FLAGS = {"s1#1": True, "s1#2": False, "s1#3": None}


def _table():
    return uptake.build_uptake_table(turns=_turns(), dispute_flags=dict(FLAGS),
                                     memory_entries=_entries())


def test_the_written_table_carries_the_very_map_the_run_computed():
    """Clause 1, first half: the map is IN the table, keyed on turn_id, verdicts intact."""
    table = _table()
    key = uptake.DISPUTE_MAP_KEY
    assert key in table, sorted(table)
    assert table[key] == {"s1#1": True, "s1#2": False, "s1#3": None}, table[key]
    # Keyed on the run's own turn ids, and the artifact says so — `<session>#<ordinal>` is
    # per-run, which is what makes the join reconstructible and what makes these ids NOT a
    # durable event key a later reader could replay against the transcript store.
    assert table["corpus"]["verdict_keyed_on"] == "turn_id", table["corpus"]
    assert table["corpus"]["dispute_turn_ids"] == ["s1#1"], table["corpus"]


def test_the_table_says_how_much_of_the_window_the_map_actually_covers():
    """Clause 1, second half: a reader can tell a small window from an unanswered one.

    A map over three turns of which one was disputed is a different claim depending on
    whether the other two were *cleared*, *unanswered*, or *never screened*. All three counts
    are in `corpus`, and `engine_unanswered` counts the `None` verdicts rather than folding
    them into the negative class — which is the rule the counts beside it now have to agree
    with.
    """
    corpus = _table()["corpus"]
    assert corpus["turns"] == 3, corpus
    assert corpus["disputes"] == 1, corpus
    assert corpus["classified_turns"] == 2, corpus
    assert corpus["engine_unanswered"] == 1, corpus
    assert corpus["unscreened_turns"] == 0, corpus
    assert (corpus["classified_turns"] + corpus["engine_unanswered"]
            + corpus["unscreened_turns"] == corpus["turns"]), corpus


def test_the_map_survives_the_file_a_later_process_opens(tmp_path):
    """The process boundary clause 1 exists for: written by the probe, read back by a job.

    `write_table` → bytes → `json.loads` is the whole distance between the run that
    classified the turns and the candidate artifact that must not re-classify them. A map
    that only existed in memory would leave that second process exactly where it started,
    which is where the consumer script that crashed was standing.
    """
    path = uptake.write_table(_table(), out_dir=tmp_path, date="2026-10-14")
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back[uptake.DISPUTE_MAP_KEY] == {"s1#1": True, "s1#2": False, "s1#3": None}, (
        sorted(back))
    assert back["corpus"]["dispute_turn_ids"] == ["s1#1"], back["corpus"]
    # JSON has no `None` key problem and no tuple keys: the verdicts come back as the same
    # three-valued thing, not as strings.
    assert [type(v) for v in back[uptake.DISPUTE_MAP_KEY].values()] == [bool, bool, type(None)]


def test_the_maps_meaning_travels_in_the_artifact_not_in_a_comment():
    """A consumer reads the glossary, not the source: both new keys must be in it.

    `eval/uptake/*.json` is read by a skill on a later day by something that has not opened
    `app/uptake.py`, which is why the writer carries its own field meanings. A map whose
    three-valued semantics are undocumented is a map that gets coalesced to false.
    """
    table = _table()
    for field in (uptake.DISPUTE_MAP_KEY, "dispute_turn_ids"):
        assert field in uptake.GLOSSARY, field
        assert field in table["glossary"], sorted(table["glossary"])
    assert "null" in uptake.GLOSSARY[uptake.DISPUTE_MAP_KEY].lower(), (
        uptake.GLOSSARY[uptake.DISPUTE_MAP_KEY])
    assert "unavailable" in uptake.GLOSSARY[uptake.DISPUTE_MAP_KEY].lower(), (
        "the glossary must say what a consumer does when the key is absent, or a table "
        "without a map reads as a window with no disputes")


def test_a_table_written_without_a_map_is_distinguishable_from_a_window_of_no_disputes():
    """The negative control for the whole item: absence has a shape, and it is not zero.

    A pre-#2717 table has no `dispute_verdicts` key at all while still counting its
    disputes. If "no map" and "nothing disputed" produced the same bytes, every committed
    artifact would be evidence that memory is healthy, and the consumer's `unavailable`
    branch would be decoration.
    """
    without = _table()
    without.pop(uptake.DISPUTE_MAP_KEY)
    clean = uptake.build_uptake_table(
        turns=_turns(), dispute_flags={"s1#1": False, "s1#2": False, "s1#3": False},
        memory_entries=_entries())
    assert uptake.DISPUTE_MAP_KEY in clean
    assert clean[uptake.DISPUTE_MAP_KEY] and without.get(uptake.DISPUTE_MAP_KEY) is None
    assert clean["corpus"]["disputes"] == 0 and without["corpus"]["disputes"] == 1
    json.dumps(without)  # still a well-formed artifact; the key is additive, not required


@pytest.mark.parametrize("flags,expect", [
    ({"s1#1": True, "s1#2": True}, ["s1#1", "s1#2"]),
    ({"s1#1": False}, []),
])
def test_the_disputed_id_list_follows_the_map_rather_than_reranking_it(flags, expect):
    """`corpus.dispute_turn_ids` is the map's own true-valued subset, sorted.

    Sorted so two runs over one corpus write byte-identical files: an artifact that shuffles
    its own evidence between runs cannot be diffed, and this instrument's whole use is the
    diff between two windows.
    """
    table = uptake.build_uptake_table(turns=_turns(), dispute_flags=dict(flags),
                                      memory_entries=_entries())
    assert table["corpus"]["dispute_turn_ids"] == expect, table["corpus"]
    assert list(table[uptake.DISPUTE_MAP_KEY]) == sorted(table[uptake.DISPUTE_MAP_KEY])
