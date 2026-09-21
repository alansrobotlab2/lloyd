"""The nightly retrieval-eval report must trigger on an interval, not a delta (#696).

Autonomy task #82 runs `skills/retrieval-eval/SKILL.md` every night and its report
is the surface where a retrieval regression claim is authored. The shipped version
of that skill triggered on `|Δ| > 0.05` and called a move `flat` when it was below
0.02; on this 20-query eval the 95 % interval on a hit rate of 0.50 is
[0.299, 0.701], so the trigger fired on one query of twenty and the 09-09 report
opened with "third consecutive night of entity-side decline. Not noise." over three
flipped queries. `eval/ci_backtest.py` replayed that rule over the baselines on
disk and counted 73 of 82 verdicts indistinguishable, 20 of the 21 entity-side ones.

The rule is text, so the test is text — the same shape as
`tests/test_ambient_disclosure_contract.py`: a nightly run reads the file at
runtime, so nothing but a test keeps the retired trigger from coming back. Two
kinds of failure are pinned: the fixed threshold returning as an instruction, and
the skill drifting from the field names the writer actually emits — a report told
to quote `confidence` while `eval/run_eval.py` writes `ci` quotes nothing.

The vault is read-only and gitignored state is not in this checkout, so the file
is resolved the way the audit resolves its baselines: an override, else the live
checkout.
"""
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path(os.environ.get("LLOYD_VAULT", Path.home() / "obsidian")) / "skills" / "retrieval-eval" / "SKILL.md"

pytestmark = pytest.mark.skipif(not SKILL.exists(), reason=f"vault skill not present at {SKILL}")

TEXT = SKILL.read_text()


def section(heading: str) -> str:
    """The body under an H2/H3 heading, up to the next heading of any level.

    Bounded on purpose: a keyword scanned across the whole file would pass on a
    mention in `## Notes` while Step 3 said something else, and Step 3 is where
    the verdict gets authored.
    """
    m = re.search(rf"^#{{2,3}} {re.escape(heading)}.*$", TEXT, re.MULTILINE)
    assert m, f"no section headed {heading!r} in {SKILL}"
    rest = TEXT[m.end():]
    nxt = re.search(r"^#{2,3} ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


# ── the trigger is an interval ───────────────────────────────────────────────

# Phrases that WERE the trigger, verbatim from the shipped skill. Imperative
# forms: they tell the run to call something out on a fixed number.
FIXED_TRIGGER = re.compile(
    r"moved more than 0\.\d+|more than 0\.05 in either direction|flat within 0\.02"
)
# What the file must say INSTEAD.
def test_step_three_triggers_on_intervals_not_a_fixed_delta():
    body = section("Step 3")
    assert "interval" in body.lower(), body[:200]
    assert "exclude" in body.lower() and "zero" in body.lower(), body[:200]
    # The retired trigger is mentioned ONLY where the file retires it — i.e. in
    # the same breath as the reason it cannot be a threshold. This is the
    # positive control on the negative assertion below: without it, a typo in
    # FIXED_TRIGGER passes by matching nothing, which is the 0-hit-grep failure
    # mode this repo has catalogued.
    blocks = [b for b in TEXT.split("\n\n") if FIXED_TRIGGER.search(b)]
    assert blocks, "FIXED_TRIGGER matched nothing: the regex is broken, not the skill"
    for block in blocks:
        low = block.lower()
        assert ("retir" in low or "resolution limit" in low or "not a threshold" in low
                or "pre-#608" in low), block[:300]


def test_step_four_does_not_trigger_on_a_fixed_delta():
    """The reporting contract's only trigger is the test; a number sneaks back in
    as a shortcut whenever someone wants a cheaper rule."""
    body = section("Step 4")
    assert "paired test rejects" in body, body[:300]
    assert "not confident enough to decide" in body, body[:300]


# ── the skill names what the writer actually emits ───────────────────────────

def test_the_skill_names_the_ci95_field_the_eval_writes():
    """#696: the interval only reaches the report if the report is told where it
    lives. `eval/run_eval.py` writes `summary.overall.ci95` with `ci` and `n`
    keys; a skill that names some other key sends every nightly looking for a
    field that does not exist."""
    assert "ci95" in TEXT, TEXT[:200]
    body = section("Where the interval for one night lives")
    assert "summary.overall.ci95" in body
    assert "`ci`" in body or "[lo, hi]" in body, body[:300]
    assert "no verdict" in body, body[:300]


def test_the_skill_says_night_over_night_is_unpaired():
    """Cross-night comparisons must use the wider independent interval. A paired
    interval across a corpus shift manufactures precision the two runs do not
    have, and `doc_hit_rate` 0.85 → 1.00 between 09-09 and 09-14 with no code
    change is what that looks like when it is quoted tightly."""
    body = section("Night-over-night is UNPAIRED")
    assert "independent" in body.lower(), body[:200]
    assert "UNPAIRED" in body or "unpaired" in body, body[:200]
    # The pairing test the code actually applies, named the way the code names
    # it, so a reader who opens eval/ci_backtest.py finds the same vocabulary.
    for label in ("paired", "unpaired-corpus", "unpaired-ids", "unpaired-drift-unknown"):
        assert label in body, label


def test_the_skill_keeps_the_reporting_tools_named():
    """The two tools the verdict depends on must be named by path, and the
    backtest — the artifact that says how often the old rule was wrong — has to
    be reachable from the report step or it is written once and never read."""
    assert "scripts/eval_trend_stats.py" in TEXT
    assert "eval/ci_backtest.py" in TEXT


# ── the finding stays a finding ──────────────────────────────────────────────

def test_an_interval_worth_calling_out_is_still_required_to_be_reported():
    """The risk in #696 is that wide intervals become a blinder. The clause that
    forbids that must name both halves: attribute a cause, and never answer
    [SILENT]."""
    body = section("Step 4")
    assert "[SILENT]" in body, body[-400:]
    assert "excludes zero" in body, body[-400:]
    # A named cause is required for anything called out — a mechanism, not a
    # time window; "third night in a row" is explicitly not a test.
    assert "third night in a row" in body.lower(), body[-600:]


def test_a_called_out_metric_still_has_to_name_a_most_likely_cause():
    """Clause 5's other half, and the half #608 deleted: vault commit ca43ca6b
    rewrote Step 4 to ask only for the drift term "with the reason stated". An
    interval says a move is real; it does not say why. A report that stops at
    "real, cause unknown" is what lets a one-query collapse read as weather, so
    the requirement is pinned here rather than trusted to survive the next prose
    rewrite."""
    step4 = TEXT[TEXT.index("## Step 4: Report"):TEXT.index("## Notes")]
    assert re.search(r"most likely cause", step4), (
        "Step 4 no longer requires a cause for a metric it calls out — #608's "
        "vault commit ca43ca6b removed it and this round must put it back")
