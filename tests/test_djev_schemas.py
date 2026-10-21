"""`eval/djev/schemas.py`: option order is part of the schema, and the hash
is what makes that a test rather than a convention.

Measured 2026-09-20: reversing the option order of one question moved P by
0.324 on average and 0.851 at worst, and dropped accuracy at a fixed 0.5
cutoff from 0.80 to 0.50. Every framing had a different optimal threshold
(0.39, 0.03, 0.30, 0.01). So a threshold belongs to an exact wording AND an
exact option order, and a rewording that kept the old number would be
reporting a measurement of a different question.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from eval.djev import schemas

#: The CLI the drift message tells a reader to run. Read off the file rather
#: than restated here, so the check below fails when the surface moves.
REPLAY_CLI = Path(__file__).resolve().parents[1] / "eval" / "djev" / "replay.py"

#: A CLI token, matched identically on both sides of the check below. The
#: character class is deliberately wider than `[a-z-]`: an option may be
#: capitalised or digit-initial one day, and a pattern too narrow to see such a
#: token in the message would pass the subset check by ignoring it — which is
#: the hole this rail exists to close.
_FLAG_TOKEN = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")


def _flag_tokens(text: str) -> set[str]:
    return set(_FLAG_TOKEN.findall(text))


def _defined_flags() -> set[str]:
    """The options `replay.py`'s argparse actually accepts, read off the file
    so the check tracks the CLI and not this file's memory of it."""
    return set(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9][A-Za-z0-9-]*)"',
                          REPLAY_CLI.read_text()))


def _drift_message() -> str:
    """What a reader is told to do when a schema's wording moved under the
    numbers calibrated against it."""
    drifted = schemas.Schema(name="example", spec={"q": {"type": "noul"}},
                             calibrated_hash="0" * 16)
    with pytest.raises(schemas.SchemaDrift) as excinfo:
        drifted.check()
    return str(excinfo.value)


def test_every_schema_is_current():
    """The whole point of `calibrated_hash`. A wording or option-order edit
    that leaves the numbers behind fails here and nowhere else."""
    for name, s in schemas.SCHEMAS.items():
        s.check()   # raises SchemaDrift


def test_reordering_options_changes_the_hash():
    """`sort_keys=False` in `Schema.hash` is load-bearing, not an oversight:
    sorting would make the one change most likely to invalidate a threshold
    invisible to the check that exists to catch it."""
    q = schemas.ENTITY.spec["same_entity"]
    reversed_criteria = dict(reversed(list(q["criteria"].items())))
    flipped = schemas.Schema(
        name="entity", seam="entity",
        spec={"same_entity": {**q, "criteria": reversed_criteria}})
    assert flipped.hash != schemas.ENTITY.hash
    with pytest.raises(schemas.SchemaDrift):
        schemas.Schema(name="x", spec=flipped.spec,
                       calibrated_hash=schemas.ENTITY.hash).check()


def test_rewording_changes_the_hash():
    q = schemas.DEDUPE.spec["same_finding"]
    reworded = schemas.Schema(
        name="dedupe",
        spec={"same_finding": {**q, "instructions": q["instructions"] + "?"}})
    assert reworded.hash != schemas.DEDUPE.hash


def test_drift_message_names_only_flags_replay_actually_accepts():
    """#3113. The message used to tell the reader to run
    `eval/djev/replay.py --corpus ... --calibrate`, and no such flag has ever
    existed: the whole argparse surface is `--corpus`, `--limit`, `--balance`,
    `--swap-probe`, `--batch`, `--seed`, `--floors`, `--out`. So the one
    message that tells a calibrating operator how to re-measure a threshold
    ended in an argparse error, and the move left to whoever hit it was to
    clear `calibrated_hash` — the exact state this check exists to prevent.

    Not every reader of this rail is mid-decision: `check()` has two callers,
    the replay CLI and this file, and the live client only ever *computes*
    `drifted` in `as_dict()` without raising. That narrows the audience to the
    person doing the calibration, which is who the message has to serve.

    Both sets come from the one `_FLAG_TOKEN` pattern, so a token in the
    message can only pass by being an option argparse really defines; the flag
    list is read off `replay.py` rather than hardcoded here, so the test fails
    when the CLI moves and not when this file's prose does.
    """
    msg = _drift_message()
    defined = _defined_flags()
    # Positive control: the pattern and the path both resolved, so the
    # subset check below is not vacuous.
    assert {"--corpus", "--batch", "--floors"} <= defined, \
        f"could not read replay.py's flags off {REPLAY_CLI}: {sorted(defined)}"
    assert "--calibrate" not in defined
    named = _flag_tokens(msg)
    assert named, f"the drift message names no flag at all: {msg!r}"
    assert named <= defined, \
        f"the message names flags replay.py does not accept: {sorted(named - defined)}"


def test_drift_message_states_both_recalibration_steps():
    """`architecture/djev.md` §9.3 steps 4-5 are the procedure, and it is two
    steps: replay **at the seam's own request shape** (`--corpus <c> --batch
    <decisions per canvas the seam sends>`), then write `threshold`,
    `label_mass_floor`, `calibrated_on` and `calibrated_hash` into the schema
    BY HAND. A message naming only a command drops the half that actually
    moves `calibrated_hash`, and `replay.py` reports `best_threshold` without
    writing anything, so a reader who ran the first step alone would still be
    holding the old hash."""
    msg = _drift_message()
    assert "--corpus" in msg and "--batch" in msg
    assert "seam" in msg.lower()
    assert "hand-write" in msg.lower()
    for field in ("threshold", "label_mass_floor", "calibrated_on",
                  "calibrated_hash"):
        assert field in msg, f"{field} is not named in the instruction: {msg!r}"


def test_the_replay_cli_hands_its_operator_the_same_actionable_text(monkeypatch):
    """The one production caller of `check()` is `eval/djev/replay.py:660` and
    nothing catches `SchemaDrift` on the way out of `main()`, so for the
    operator who hits this rail the exception text IS the interface. This
    drives that entry point — argv through argparse, the corpus loop, the
    lookup, the raise — and requires the text that comes out to clear the same
    two rails the tests above pin on the raise site.

    In-process, deliberately: `main()` is called, not spawned, so argparse
    parsing and the uncaught raise are covered and the traceback rendering on
    the operator's stderr is not. What would NOT be caught by asserting the
    obvious (that the text equals `_drift_message()` — same schema, same f-
    string, so it cannot disagree) is a CLI that swallows the exception, names
    a flag argparse would refuse, or loses the second step.
    """
    from eval.djev import replay

    monkeypatch.setattr(replay.djev, "reachable", lambda: True)
    monkeypatch.setattr(schemas, "get", lambda name: schemas.Schema(
        name="example", spec={"q": {"type": "noul"}}, calibrated_hash="0" * 16))
    monkeypatch.setattr(sys, "argv", ["replay.py", "--corpus", "reverted"])
    with pytest.raises(schemas.SchemaDrift) as excinfo:
        replay.main()
    msg = str(excinfo.value)
    # The drift that surfaced is the schema this CLI looked up, not some other
    # raise on the way there.
    assert "'example'" in msg, msg
    assert _flag_tokens(msg) <= _defined_flags(), \
        f"the CLI surfaced flags argparse would refuse: {msg!r}"
    assert "--corpus" in msg and "--batch" in msg and "hand-write" in msg.lower(), (
        f"the operator was handed an instruction missing a step: {msg!r}")


def test_the_negative_option_is_listed_first_in_every_pair_schema():
    """The arm that measured AUC 0.833, against 0.795 for the reverse and an
    optimal threshold of 0.03 — a number that is noise. Order-AVERAGING is
    not the fix: it lands between the two arms and `acc@0.5` stays 0.53."""
    for name in ("dedupe", "entity", "clusters"):
        q = next(iter(schemas.SCHEMAS[name].spec.values()))
        first = list(q["criteria"])[0]
        assert first in ("different", "distinct"), f"{name} lists {first} first"


def test_no_schema_may_gate_a_production_decision_yet():
    """`gate_ready` is a separate field from `threshold is not None` because a
    measured threshold and a threshold that may DECIDE something are
    different claims. Every schema here carries a measured number and none
    may act on it."""
    for name, s in schemas.SCHEMAS.items():
        assert s.gate_ready is False, name
        assert s.gate_blocked_reason.strip(), name


def test_a_calibrated_schema_names_what_it_was_calibrated_on():
    """A threshold with no provenance cannot be re-checked, and three of
    these were measured against another MODEL's opinions rather than against
    truth."""
    for name, s in schemas.SCHEMAS.items():
        if s.threshold is not None or s.label_mass_floor is not None:
            assert s.calibrated_on.strip(), name
            assert s.calibrated_hash == s.hash, name


def test_the_rerank_schema_has_no_floor_and_says_why():
    """The one place the design's own rule was tested rather than restated:
    listwise `label_mass` at n=12-16 has measured 0.446, 0.807, 0.965 and
    1.000 across four runs on different corpora. No floor read off one of
    them means anything for the others, so the shadow rows set it."""
    assert schemas.RERANK.label_mass_floor is None
    assert schemas.RERANK.threshold is None


def test_every_live_seam_has_a_schema():
    """A seam with no schema records nothing, which is how a new seam fails:
    silently uninstrumented rather than instrumented against a shape nobody
    froze."""
    from app import djev_shadow
    for seam in djev_shadow.SEAMS:
        assert seam in schemas.BY_SEAM, seam
        assert schemas.hash_for(seam)


def test_rank_levels_are_ordered_worst_first():
    """`score` returns an expected value over the level INDEX, so reversing
    this list reverses every ranking silently."""
    from app import djev
    assert list(djev.RANK_LEVELS) == schemas.RANK_LEVELS
    assert schemas.RANK_LEVELS[0] == "irrelevant"
    assert schemas.RANK_LEVELS[-1] == "directly answers it"


def test_the_clusters_schema_asks_what_its_corpus_labels():
    """382 `distinct`, 221 `related`, 6 `same`, and no `duplicate` verdict at
    any row. A `duplicate`-vs-`distinct` framing measured 12 usable pairs out
    of 609 and reported AUC 1.000 off the six `same` rows."""
    options = list(schemas.CLUSTERS.spec["related_item"]["criteria"])
    assert options == ["distinct", "related"]
