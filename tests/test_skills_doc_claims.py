"""What `architecture/skills.md` may say about the absent `skill_dispatch` flag.

Dispatch ships with no `harness.skill_dispatch.enabled` key, so nothing routes
a protocol until somebody writes one. For a fortnight the doc explained the
absence as an errand — it ended "Both keys are set by a human" — while #750 had
already settled the question on measurement: the flag was **decided against**,
and the ruling is in that item's `owed_settled` (`follow_up: 2142`). A reader who
finds a to-do where a decision was made re-litigates it, and the next round that
wants to "finish the feature" finds no bar to argue with. So the paragraph
carries the decision, its numbers, and the condition that would reopen it.

Two claims are pinned, plus the evidence behind them:

  * the measured decision — `spurious_rate` 0.04% quoted **beside** its dispatch
    count, projected compliance delta +0, the only firing rule already at 100% —
    with no sentence left framing the absent key as somebody's pending task;
  * the re-enable bar, both halves of it: firing on non-meta traffic AND
    measured compliance below 100%, named with the probe command that measures
    it. A condition missing either half admits traffic the 2026-10-03 run
    actually showed, so the extractor demands both.

The numbers came from outside this repo, so they have a witness: the probe's own
JSON is committed at `~/obsidian/backlog/data/probe_2142.json` and re-read here,
and the doc's quoted rate and count are compared to it. That comparison is the
point of the file — a doc quoting a number no committed artifact carries is a
sentence nobody can re-check, and the denominator moves with the live session
corpus (9153 → 9250 between #750's ruling and this round), so the count has to be
re-derivable and not merely asserted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "skills.md"

#: The command that decides enablement, cited by the paragraph and checked to
#: exist — a condition naming a probe that has since moved is a bar nobody can
#: apply.
PROBE_CMD = "eval/run_skill_dispatch_probe.py"

#: The probe's rate, structurally, with its `n` in the same sentence. The
#: numbers stay generic on purpose: extraction is a shape, and which values are
#: true is what the assertions below say.
_RATE_WITH_N_RE = re.compile(
    # `[^\d]{0,10}` is the gap between the word the doc uses for the metric and
    # its value — "`spurious_rate` 0.04%" — not a licence to search the paragraph
    # for any number: the value still has to carry a `%` and be followed, inside
    # one sentence, by a count of dispatches.
    r"spurious[^\d]{0,10}(\d+(?:\.\d+)?)\s*%[^.]{0,40}?(\d{3,6})\s+"
    r"(?:replayed |globally )?dispatches",
    re.I,
)

#: Framings that turn an absent flag into somebody's errand. The #750 ruling
#: made it a measured decision, so a paragraph carrying one of these is back to
#: reporting a to-do. Deliberately narrower than the word "human": `enabled()`'s
#: docstring still says a human sets these keys, and that is a true statement
#: about *authority* — the paragraph keeps the same fact as "the automod
#: preflight denies the runtime config" — whereas these frames present the
#: missing flag as work nobody has done yet.
PENDING_HUMAN_RES = (
    re.compile(r"\b(?:set|sets|setting|enabled|enable|flip|flipped|flipping)\b[^.]{0,30}\bby a human\b", re.I),
    re.compile(r"\bhuman (?:must|should|sets|to set|needs to|has to|will)\b", re.I),
    re.compile(r"\b(?:waiting|pending|blocked|owed|left)\b[^.]{0,20}\b(?:a |the )?(?:human|owner|someone|anyone)\b", re.I),
)


def default_off_paragraph() -> str:
    """The `**Default-off…**` paragraph, whole, wrapped flat.

    Matched on its bold lead-in and run to the blank line, so a rewrite that
    split the decision into a second paragraph would leave the extractor reading
    half the claim and passing anyway.
    """
    text = DOC.read_text(encoding="utf-8")
    m = re.search(r"\*\*Default-off[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text)
    assert m, "architecture/skills.md has no **Default-off… paragraph"
    flat = " ".join(m.group(0).replace("**", "").split())
    assert len(flat) > len("**Default-off".replace("**", "")), (
        "the **Default-off** lead-in has no prose after it")
    return flat


def _sentences(text: str) -> list[str]:
    """Sentences, with the doc's ~80-column wraps collapsed away first: a
    wrapped sentence and an absent sentence are different findings."""
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]


def pending_human_claims(paragraph: str) -> list[str]:
    """Sentences presenting the absent key as work a person still owes."""
    return [s for s in _sentences(paragraph)
            if any(r.search(s) for r in PENDING_HUMAN_RES)]


def reenable_conditions(paragraph: str) -> list[str]:
    """Sentences stating a complete bar for turning dispatch on.

    Complete means both halves present: a scope that excludes the automod
    meta-traffic every one of the 2026-10-03 triggering dispatches was, and a
    compliance figure short of perfect. Either half alone admits the traffic that
    run already showed, which is why a sentence with only one is not counted.
    """
    out = []
    for s in _sentences(paragraph):
        if not re.search(r"\benabl", s, re.I):
            continue
        if re.search(r"\bnon-?meta\b", s, re.I) and re.search(r"below\s+100\s*%", s, re.I):
            out.append(s)
    return out


def quoted_rate_and_count(paragraph: str) -> tuple[float, int]:
    """(spurious rate, dispatch count) as the doc quotes them, or a failure.

    Fails when the rate stands alone: #750's ruling named 0.04% over 9153 and
    the live corpus now replays 9250, so a rate whose `n` the reader cannot
    recover is a number from no run at all.
    """
    m = _RATE_WITH_N_RE.search(paragraph)
    assert m, (
        "the **Default-off** paragraph must quote the probe's spurious rate with the "
        "dispatch count in the same sentence — the corpus is the live session store, "
        "so a bare rate is not re-checkable")
    return float(m.group(1)), int(m.group(2))


def test_the_default_off_paragraph_records_a_decision_not_a_pending_human():
    """Clause 1: the absence is on the record as decided against, with the run's
    numbers, and no sentence leaves it as somebody's errand."""
    flat = default_off_paragraph()
    assert "Both keys are set by a human" not in flat, (
        "the paragraph is back to ending on the errand #750 ruled shut")
    assert pending_human_claims(flat) == [], (
        f"a sentence frames the absent key as pending work: {pending_human_claims(flat)}")
    assert re.search(r"decided against", flat, re.I), (
        "nothing says the flag was decided against, which is the whole record")
    assert quoted_rate_and_count(flat) == (0.04, 9250), "the quoted run drifted"
    assert re.search(r"projected[^.]{0,30}delta \+?0\b", flat, re.I), (
        "the projected compliance delta is no longer stated — the reason the "
        "decision cost nothing to take")
    assert re.search(r"already at 100\s*%", flat, re.I), (
        "the one protocol that fired is no longer said to be already compliant")
    # The detectors bite: the sentence this clause exists to keep out is caught
    # rather than merely absent, and a bare rate fails the extraction. A pin that
    # only grepped the retired string would survive a reworded reinstatement.
    assert pending_human_claims(
        "Both keys are set by a human — the automod preflight denies the runtime config."), (
        "the old framing slipped past the detector")
    assert pending_human_claims(
        "Enabling it is pending a human, since the preflight denies the config."), (
        "a reworded errand slipped past the detector")
    try:
        quoted_rate_and_count("`spurious_rate` 0.04%, and the flag stays off.")
    except AssertionError:
        pass
    else:
        raise AssertionError("a spurious rate with no dispatch count was accepted")


def test_the_default_off_paragraph_states_what_would_reopen_it():
    """Clause 2: the re-enable bar, both halves, with the probe that measures it
    named — so the decision is falsifiable instead of permanent, and a future run
    that does meet the bar has a written bar to meet."""
    flat = default_off_paragraph()
    conds = reenable_conditions(flat)
    assert conds, "the paragraph states no bar for enabling a rule again"
    assert all(PROBE_CMD in s for s in conds), (
        f"a stated condition does not name the probe that decides it: {conds}")
    assert (ROOT / PROBE_CMD).is_file(), f"{PROBE_CMD} is cited but is not in the tree"
    assert "firing" in conds[0] or "trigger" in conds[0].lower(), (
        "the condition does not require the rule to fire on the traffic")
    # Half-bars are refused, so a condition cannot quietly weaken to one of them.
    assert reenable_conditions(
        "Re-enable a rule when a replayed corpus shows it firing on non-meta traffic."
    ) == [], "a bar with no compliance half was accepted"
    assert reenable_conditions(
        "Re-enable a rule when measured compliance is below 100%."
    ) == [], "a bar with no non-meta half was accepted"
    assert reenable_conditions(
        "Re-enable a rule when it fires on non-meta traffic with compliance below 90%."
    ) == [], "a bar with the wrong threshold was accepted"


def test_the_committed_probe_witness_backs_the_quoted_numbers():
    """The decision's evidence left this repo the day the probe wrote its JSON,
    so its bytes are committed and re-read here. This is the check the paragraph
    cannot do for itself: the doc's quoted rate and count have to be the
    witness's, and the reopen bar the paragraph names has to be the bar that run
    actually failed.

    Deliberately unmarked, and the exemption is this: `pytest.ini` declares
    `live_vault` for assertions about a board a nightly or autoresearch job can
    rewrite between rounds, and the gate runs `-m "not live_vault"` — so marking
    this node is how the pin would certify nothing, since the gate would never
    run it. What the mark protects against is not the hazard here. The witness is
    a measurement snapshot at a fixed path (`backlog/data/probe_2142.json`) that
    nothing rewrites: the probe's next run goes to a fresh file under
    `_pipeline/skill-dispatch-probe/`, and `retention.py` never deletes
    `backlog/data/*` — the sweep's own table has no entry for it, which is what
    makes the copy a durable witness rather than a file that can vanish. A drift
    this node *can* catch is the doc being edited to numbers the probe never
    reported, which is the rot it is written against."""
    import board_presence

    witness = board_presence.BOARD_DIR / "data" / "probe_2142.json"
    assert witness.is_file(), (
        f"{witness} is the only history behind the numbers in architecture/skills.md; "
        "a decision recorded with an unbacked number is a sentence nobody can re-check")
    rep = json.loads(witness.read_text(encoding="utf-8"))
    assert rep["dispatches_replayed"] == 9250, rep["dispatches_replayed"]
    assert rep["spurious_rate_pct"] == 0.04, rep["spurious_rate_pct"]
    assert rep["dispatches_triggered"] == 4, rep["dispatches_triggered"]
    assert rep["dispatches_triggered"] == sum(
        p["triggered"] for p in rep["per_protocol"].values()), "the report contradicts itself"

    rate, count = quoted_rate_and_count(default_off_paragraph())
    assert rate == rep["spurious_rate_pct"], "the doc's rate and the committed probe run disagree"
    assert count == rep["dispatches_replayed"], (
        f"the doc quotes {count} dispatches, the committed run replayed "
        f"{rep['dispatches_replayed']} — the corpus moves, so this comparison is "
        "what keeps the quote honest")

    firing = {n: p for n, p in rep["per_protocol"].items() if p["triggered"] > 0}
    assert firing, "no rule fired, which is a different decision than the one recorded"
    for name, p in firing.items():
        assert p["compliance_before_pct"] == 100.0, (
            f"{name} fired with imperfect compliance: the parked decision no longer holds")
        assert p["compliance_after_predicted_pct"] == p["compliance_before_pct"], (
            f"{name} has a non-zero projected delta: dispatch would buy something")
    # The bar in the paragraph, applied to the witness: nothing met it, which is
    # what makes the absence a decision rather than a backlog of one.
    assert not any(
        p["triggered"] > 0 and (p["compliance_before_pct"] or 100.0) < 100.0
        for p in rep["per_protocol"].values()), (
        "a protocol fired on this corpus with compliance below 100% — enablement is "
        "an open question again, and the paragraph must not read as settled")
