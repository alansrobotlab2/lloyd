#!/usr/bin/env python3
"""#2620 clause 3: every domain ships a positive control that reddens ITS OWN
postcondition, and a control that stays green is itself a test failure.

This file is the reason #644's objection does not apply to this instrument. That
triage ruled auto-generated rows are "another way to produce green tests that
assert nothing"; the answer is not to argue, it is that a domain reporting zero
failures here reports zero failures ONLY if a seeded break was shown to light its
detector up first. 0 failures with a live control is evidence. 0 failures without
one is not evidence of anything, which is the standing "a 0-hit grep needs a
positive control" rule pointed at this instrument itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _substrate_fuzz import (BY_ID, CONTROLS, CONTROLLED_DOMAINS,  # noqa: E402
                             REAL, sf_corpus)

#: A substring of the filesystem base task's payload that survives turning its first byte into
#: a different one — the point of the fixture is a postcondition that cannot notice the fault.
WRITTEN_NEEDLE = "ritten by arm A"


def test_every_domain_ships_at_least_one_control():
    """Every domain the corpus DECLARES ships a proven detector.

    Declared minus shipped, not shipped-minus-shipped: `CONTROLLED_DOMAINS` is computed from
    the same rows as the domains, so subtracting it from the domains that happen to be present
    asks a question the corpus cannot lose. Against `sf_corpus.DOMAINS` the node can actually
    fail — by a domain losing its control, or by a control being deleted and the domain then
    quietly dropped from the declared list to keep the count honest.
    """
    uncovered = sorted(set(sf_corpus.DOMAINS) - set(CONTROLLED_DOMAINS))
    assert not uncovered, f"domains with no control: {uncovered}"
    unshipped = sorted({t.domain for t in REAL} - set(CONTROLLED_DOMAINS))
    assert not unshipped, f"shipped domains with no control: {unshipped}"


def test_every_control_is_injected_at_the_tool_boundary():
    """A break inside the runner's own bookkeeping would prove the bookkeeping works
    and nothing about the substrate, so the corpus states where each break lands."""
    for c in CONTROLS:
        assert c.control_at == "tool_boundary", (c.id, c.control_at)
        assert c.control in ("strip_at_nul", "clip_to_prefix", "flip_one_byte"), c.id


def test_a_control_that_stays_green_is_a_test_failure():
    """Clause 3's own rule, executed: every control in the shipped corpus must come
    back RED from a real arm-A run.

    This is not a tautology reading the control's own expectation. The control task
    carries the SAME postcondition as its base and the SAME expected value, and the
    break is applied to the bytes crossing the boundary; so the only way it goes red
    is the detector reading the boundary. A control whose base were redefined to
    match the break would go green, and this node fails.

    Two ways a control can be dead rather than red are refused here, not folded into
    `ok is False`: a run that SKIPPED the task (its break never applied to any payload,
    so nothing was seeded) and a run whose redness came from the runner noticing the
    postcondition stayed green (`rule: control stayed green`, a dead detector reported
    as a defect). Both leave the domain without a live detector, which is the state this
    file exists to make impossible.
    """
    from eval.substrate_fuzz import runner as R
    for c in CONTROLS:
        rec = R.run_task(c)
        assert rec["skipped"] is None, (
            f"{c.id}: the run skipped the control ({rec['skipped']}: {rec['evidence']}), so "
            f"{c.control!r} was never seeded at any boundary — a control that cannot fire "
            "leaves its domain's detector dark while the row still reads as coverage")
        assert rec["ok"] is False, (
            f"{c.id} ({c.control} over {c.control_for}) stayed green: the seeded break "
            f"did not redden the shipped postcondition, so that detector is dead")
        assert "stayed green" not in (rec["rule"] or ""), (
            f"{c.id}: red only because the runner saw the postcondition survive the break "
            f"({rec['rule']!r}) — the seeded fault is real, the detector is not")
        assert rec["control"] == c.control, rec
        assert rec["control_break_verified"] is True, (
            f"{c.id}: the task went red, but the run cannot show that {c.control!r} changed "
            f"the bytes the tool answered with — a control that may not have fired is not a "
            f"live detector, and the artifact's `controls_verified` count would be inflated "
            "for its domain")
        assert rec["failure"]["layer"] in R.LAYERS, rec["failure"]


def test_the_verified_count_is_a_judgement_about_the_observed_payload():
    """The count clause 3's green board rests on is computed from what the tool answered, not
    from a value derived out of the postcondition.

    The first shipped version took `bytes.fromhex(post.expected)` as the payload — 32 bytes of
    expected DIGEST, which nobody sent — and so reported `byte_roundtrip controls 0 verified` on
    the full arm-A run for a control that had just produced an `interface` verdict, while
    reporting `verified` for any break that can alter a hash by construction. A count that
    cannot be wrong is a green test asserting nothing, which is #644's objection aimed back at
    this instrument. The assertion below is what that bug fails: every shipped control's break
    has to change the bytes of the answer it was handed, and `strip_at_nul` — the one break that
    only fires on a payload carrying a NUL — is the case the old code got backwards."""
    from eval.substrate_fuzz import runner as R
    nul_ctl = [c for c in CONTROLS if c.control == "strip_at_nul"]
    assert nul_ctl, "the NUL control is the one break that cannot be assumed to apply"
    rec = R.run_task(nul_ctl[0])
    assert rec["steps"][-1]["tool_recorded"] is True, rec["steps"][-1]
    assert rec["control_break_verified"] is True, (
        f"{nul_ctl[0].id}: the tool answered with bytes containing a NUL and the break was "
        f"applied to them, so a `False` here means the count is being computed from something "
        "other than the payload that crossed the boundary")


def test_a_control_whose_postcondition_cannot_see_the_break_is_reported_red():
    """The other edge of clause 3, and it needs no real defect: a control whose break fires
    but whose shipped postcondition is too weak to notice must come back as a RED RECORD.

    The row below is the shipped filesystem control with its `stdout_sha256` postcondition
    replaced by a substring one. `flip_one_byte` flips the payload's first byte, so the
    substring assertion survives a seeded fault: exactly the green control clause 3 refuses to
    let pass. Silently re-classifying it as a skip would be worse than letting it pass — a skip
    is excluded from `k / N`, so the domain's number would stay clean while its detector went
    dark — and a record with no layer would be the `unknown` bucket clause 4 forbids.
    """
    from eval.substrate_fuzz import runner as R
    doc = json.loads(sf_corpus.CORPUS_PATH.read_text(encoding="utf-8"))
    for row in doc["tasks"]:
        if row["id"] == "ctl-fs-write-flipped":
            row["postcondition"] = {"kind": "shown_contains", "expected": WRITTEN_NEEDLE}
    tasks = sf_corpus.load_corpus(doc["tasks"], check_cross_row=False)
    ctl = [t for t in tasks if t.id == "ctl-fs-write-flipped"][0]
    rec = R.run_task(ctl)
    assert rec["skipped"] is None, rec["evidence"]
    assert rec["ok"] is False, (
        "a weak postcondition survived a seeded break and the run called it a pass")
    assert "stayed green" in rec["rule"], rec["rule"]
    assert rec["defect"] is True, "a detector that cannot fail is not a legitimate limit"
    assert rec["layer"] == "harness", (
        f"a dead control is a defect in this instrument, and the artifact's vocabulary has "
        f"no bucket for it besides the one that owns the instrument: got {rec['layer']!r}")
    assert json.dumps(rec)  # the record survives the artefact's json.dump


@pytest.mark.parametrize("kind", ["strip_at_nul", "clip_to_prefix", "flip_one_byte"])
def test_each_break_kind_is_implemented_and_changes_bytes(kind):
    """A named break with no implementation would report itself as exercised."""
    from eval.substrate_fuzz.runner import _break_bytes  # noqa: PLC0415
    raw = b"alpha\x00omega-payload"
    broken = _break_bytes(raw, kind)
    assert broken != raw, f"{kind} left the payload untouched"


def test_a_control_never_breaks_the_runners_own_bookkeeping():
    """Positive control for this file: an unbroken task still reads green, so a
    blanket `ok is False` assertion cannot be what makes the controls test pass."""
    from eval.substrate_fuzz import runner as R
    base = BY_ID[CONTROLS[0].control_for]
    rec = R.run_task(base)
    assert rec["ok"] is True, rec["failure"]


def test_loader_refuses_a_control_that_could_not_redden_anything():
    """`exit_shape` reads a flag and `file_exists` reads an inode: no byte break
    touches either, so the loader refuses that pairing instead of shipping a control
    that can never fail — which is the exact defect class #644 ruled in."""
    base = sf_corpus.load_corpus()
    ctl = [c for c in base if c.control_for == "proc-spawn-and-wait"][0]
    row = dict(ctl.raw)
    row["postcondition"] = {"kind": "exit_shape", "is_error": False}
    with pytest.raises(sf_corpus.CorpusError, match="reads no bytes"):
        sf_corpus.load_corpus(_replace(base, ctl.id, row))


def _replace(tasks, tid, row):
    import json
    import tempfile
    d = Path(tempfile.mkdtemp())
    rows = [dict(t.raw) for t in tasks]
    for i, r in enumerate(rows):
        if r["id"] == tid:
            rows[i] = row
    p = d / "corpus.json"
    p.write_text(json.dumps({"version": sf_corpus.CORPUS_VERSION, "item": 2620,
                             "tasks": rows}))
    return p
