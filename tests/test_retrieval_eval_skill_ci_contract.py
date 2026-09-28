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


# ── #1600 / #1663: one rate per leg, over the gold-bearing subset ────────────

def _overall_with_gold_mixed_in() -> dict:
    """A real `summarize()` pass over two queries — one carrying entity gold, one
    carrying none — so the field names and the numbers checked below come out of
    what the writer emits, not out of a list this test wrote.

    The second query is the interesting one: it has no entity gold, so since #1663
    it is absent from the entity leg's denominator entirely while still counting on
    the document leg. Imported here rather than at module scope: the rest of this
    file is pure text and stays that way.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    import eval.run_eval as ev
    return ev.summarize([
        {"id": "with-gold", "category": "single", "latency_ms": 10.0, "error": None,
         "expected": {"entities": ["Knowledge Graph"], "docs": ["knowledge/kg.md"]},
         "scoring": {"entity_hit": True, "doc_hit": True, "entity_recall": 1.0,
                     "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                     "fact_entity_recall": 1.0, "first_doc_rank": 1,
                     "entity_hit_retrieval_carried": True}},
        {"id": "no-entity-gold", "category": "single", "latency_ms": 10.0, "error": None,
         "expected": {"entities": [], "docs": ["knowledge/kg.md"]},
         "scoring": {"entity_hit": False, "doc_hit": True, "entity_recall": None,
                     "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                     "fact_entity_recall": None, "first_doc_rank": 1,
                     "entity_hit_retrieval_carried": False}},
    ])["overall"]


RATE_LEGS = ("entity_hit_rate", "entity_hit_rate_retrieval_carried",
             "doc_hit_rate", "mrr_doc", "ndcg10")


def test_the_skill_names_the_denominator_field_the_eval_emits():
    """#1663 clauses 1 and 4: the artifact now carries ONE rate per leg plus its
    gold-bearing denominator in `ci95`, and the skill names that field.

    The names come from `summarize()`'s own keys, so the writer renaming or
    re-introducing a field fails this test along with the prose — #696's failure
    mode (`a report told to quote `confidence` while eval/run_eval.py writes `ci`
    quotes nothing`). The exact-value asserts are the positive control: were the
    writer emitting no `ci95` block at all, the loop over the five legs would have
    nothing to check and would pass.

    The values are the re-base in two queries: the entity leg is 1.0 over the one
    query that carries entity gold (`ci95.entity_hit_rate.n == 1`), where the
    all-records denominator gave 0.5 over two; the document leg is unchanged at 1.0
    over two, because both queries carry doc gold.
    """
    overall = _overall_with_gold_mixed_in()
    assert not [k for k in overall
                if k.endswith("_gold_bearing") or k.endswith("_gold_bearing_n")], (
        "a companion key is back beside a headline that is now the same reading")
    assert overall["entity_hit_rate"] == 1.0
    assert overall["entity_hit_rate_retrieval_carried"] == 1.0
    assert overall["doc_hit_rate"] == 1.0
    ci = overall["ci95"]
    assert [ci[m]["n"] for m in RATE_LEGS] == [1, 1, 2, 2, 2], ci
    for metric in RATE_LEGS:
        assert f"`ci95.{metric}.n`" in TEXT, (
            f"skill does not name the denominator the writer emits: {metric}")


def test_the_skill_carries_one_number_per_leg_and_no_companion_instruction():
    """#1663 clause 4, the page half: the skill section that used to tell the
    nightly to print a companion beside the headline now tells it to print the rate
    with its `n`, and says out loud that the companions are gone.

    Scoped to its own section so a stray mention in `## Notes` cannot stand in for
    the report step. The negative asserts are the point: "print the gold-bearing
    rate beside the headline" surviving in the section would have the report print
    one leg twice, which after the re-base is either a duplicate or a contradiction.
    """
    body = section("Which population each scored rate is divided over")
    low = body.lower()
    assert "quote each rate with its `n`" in low, body[:400]
    assert "beside the headline" not in low, body[:400]
    assert "companions are gone" in low, body[:400]
    assert "null" in low and "never `0.0`" in body, body[:400]
    assert "no verdict" in body, body[:400]
    # #1600's two escape clauses are retired: the section may no longer claim the
    # denominator policy is un-owned, nor that the rank scores were left behind.
    assert "reserves" not in low, body[-900:]
    assert "no companion" not in low, body[-900:]


def test_the_skill_books_the_denominator_re_base_by_date():
    """#1663 clause 5: a denominator change is invisible to the trend tool, so the
    skill's dated re-base list is the only record of it, and it has to say which
    night is the first one scored the new way.

    `scripts/eval_trend_stats.py` joins nights by `records[].id` and diffs the
    `corpus` block, so five rates re-based over a stable 86-query corpus prints
    nothing: the break is `entity_hit_rate` 0.488 → 0.636 with every id intact. This
    is the same shape as the 2026-09-26 seed re-base, which is why the bullet has to
    sit in the same list rather than in the section that explains the fields.
    """
    notes = TEXT[TEXT.index("## Notes"):]
    seed = notes.index("The seed definition moved under the entity leg")
    rebase_at = notes.index("The denominator definition moved on 2026-09-28")
    assert rebase_at > seed, "the #1663 bullet is not in the re-base list"
    bullet = notes[rebase_at:notes.index("\n- ", rebase_at) + 1]
    for metric in RATE_LEGS:
        assert metric in bullet, (metric, bullet[:200])
    assert "nightly-20260929" in bullet, bullet
    assert "pre-re-base" in bullet, bullet
    # The prose #1600 wrote to refuse the re-base must not survive it.
    assert "Why this is a companion and not a re-base" not in TEXT
    assert "have **no** companion" not in TEXT


# ── #1599: the unreturnable-gold fields the writer emits ────────────────────

def test_the_skill_names_the_unreturnable_gold_fields_the_eval_emits():
    """#1599: the nightly has to report which gold no deployed collection can
    return, and a report told to quote some other name quotes nothing.

    The names come out of `summarize()`'s own keys rather than a list this test
    wrote, so a writer renaming a field fails along with the prose — #696's failure
    mode (`a report told to quote `confidence` while eval/run_eval.py writes `ci`
    quotes nothing`) applied to #1599's fields. The exact-name assert below is the
    positive control: were `summarize` emitting none of them, the loop would have
    nothing to check and would pass.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    import eval.run_eval as ev
    overall = ev.summarize([{"id": "q", "category": "single", "latency_ms": 1.0,
                             "error": None,
                             "expected": {"entities": [], "docs": ["knowledge/x.md"]},
                             "scoring": {"entity_hit": False, "doc_hit": True,
                                         "entity_recall": None, "doc_recall": 1.0,
                                         "rr_doc": 1.0, "ndcg10": 1.0,
                                         "first_doc_rank": 1,
                                         "fact_entity_recall": None}}])["overall"]
    emitted = sorted(k for k in overall if k.startswith("gold_doc"))
    assert emitted == ["gold_doc_collections", "gold_doc_unreturnable",
                       "gold_doc_unreturnable_query_ids"], emitted
    # Emitted even when no collection list was supplied — as nulls. A key that only
    # appeared on the checked path would be invisible to a report reading a run that
    # could not open the index, which is the case that most needs a sentence.
    assert overall["gold_doc_unreturnable"] is None, overall
    assert overall["gold_doc_collections"] is None, overall
    for name in emitted:
        assert f"`{name}`" in TEXT, f"skill does not name the field the writer emits: {name}"


def test_the_skill_reports_the_finding_without_touching_the_gold_set():
    """#1599's instruction half, scoped to its own section so a mention in
    `## Notes` cannot stand in for the report step saying it.

    The half that matters most is the prohibition: an `unreturnable` count goes to
    zero by registering the collection, by re-pointing the labels onto indexed
    copies, or by deleting the queries — and only the first two are fixes. The other
    two raise `doc_hit_rate` with retrieval standing still, so the report step has to
    say out loud that it is not the one to make that move.
    """
    body = section("Which gold the deployed collections cannot return")
    low = body.lower()
    assert "never means editing the gold set" in low, body[:400]
    assert "person's" in body, body[:400]
    assert "no verdict" in body, body[:400]
    # Both of the two moves that legitimately end the finding are named, so a reader
    # is told what is being waited on rather than that something is broken.
    assert "register" in low and "re-point" in low, body[:500]
