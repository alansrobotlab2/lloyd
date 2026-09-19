"""Backlog #701 — the `opposing_terms` trigger must mean whole-word opposition.

`_detect_contradictions_sync` classifies a fact pair two ways:
`opposing_terms:<a>/<b>` from `_OPPOSING_PAIRS`, or `high_overlap_potential_update`
from token overlap. Only the first is admissible evidence downstream —
`fact_improvement.REQUIRE_OPPOSING_TERMS` is the single gate between a detected
pair and a planned expiry, and `fact_resolve(auto_resolve=true)` marks on any pair
at all — so the classification carries the whole safety argument for those writes.

Until #701 it was bare substring containment over lowercased text:

    if (pair[0] in t1 and pair[1] in t2) or (pair[1] in t1 and pair[0] in t2):

Two of the seven pairs contain their own opposite (`inactive` ⊃ `active`,
`unsupported` ⊃ `supported`), so two sentences that agreed were classified as
opposing each other, and `yes`/`no` matched inside `another`, `notable` and
`denote`. Measured on the live store on 2026-09-15: 2 of 215 pairs were flagged
`opposing_terms` and both were false positives — a 43% success rate against a
statement about causes of failure, and a content claim against a backlink
bookkeeping note — and both became planned expiries.

So this file pins three things, in the order the clauses are written: a pair of
sentences that merely share a term is not classified; a larger word containing a
term is not a match; and a genuine opposition still is.

Run: REPO=$PWD ~/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_fact_contradiction_detector.py -q
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reached as attributes of the module, not imported names: a test file that
# fails at import proves nothing about behaviour, and these tests exist to fail
# on the classification, which is what #701 changed.
from agent_mcp import facts as facts_mod                                # noqa: E402

_OPPOSING_PAIRS = facts_mod._OPPOSING_PAIRS
_detect_contradictions_sync = facts_mod._detect_contradictions_sync


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A temp facts tree the tool seam can read. The detector itself takes a
    list; the `_fact_check` seam at the bottom of the file needs a tree, and a
    classification proved only on a hand-built list is a classification the tool
    path was never shown."""
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    import agent_mcp._shared as shared
    from agent_mcp import retrieval
    monkeypatch.setattr(shared, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    yield facts_root
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None


def _reasons(*texts):
    """Classify exactly these facts, the way the improvement loop calls it: one
    list in, the pairs out, no filesystem and no entity alias to widen the set."""
    facts = [{"fact": t, "id": f"f{i:03d}", "category": "state"}
             for i, t in enumerate(texts)]
    out = _detect_contradictions_sync("Probe", facts=facts)
    return [c["reason"] for c in out["contradictions"]]


def _opposing(*texts):
    return [r for r in _reasons(*texts) if r.startswith("opposing_terms")]


# ── clause 1: agreeing sentences are not oppositions ─────────────────────────
#
# Each term is now matched `(?<!\w)<term>(?!\w)` — the `(?!\w)` half is what the
# `\b` shorthand would get wrong after `s`, where the lookbehind does not hold.

def test_two_sentences_that_only_say_unsupported_are_not_opposing_terms():
    """The proving command from the item. `unsupported` contains `supported`,
    so before #701 these two agreeing sentences were an 'opposition' and, with
    confidences differing, a planned expiry."""
    assert _opposing("The feature is unsupported.",
                     "Both paths are unsupported, measured 09-09.") == []


def test_two_sentences_that_only_say_inactive_are_not_opposing_terms():
    """The second self-collapsing pair: `inactive` contains `active`."""
    assert _opposing("The gate is inactive by default.",
                     "The gate stays inactive until armed.") == []


def test_a_substring_only_pair_falls_through_to_the_overlap_class():
    """Not classified as an opposition is not the same as not reported: the
    high-overlap class still sees a pair phrased alike, and downstream that
    class is reported and never acted on. The fix narrows one trigger; it does
    not blind the scan. These two sentences share only `inactive`, so before
    #701 they were an 'opposition' and now land in the overlap class — 6 of 7
    tokens in common, above the 0.6 threshold."""
    reasons = _reasons("The gate stays inactive until armed.",
                       "The gate stays inactive until armed manually.")
    assert reasons == ["high_overlap_potential_update"], reasons


# ── clause 2: a term inside a larger word is not a match ─────────────────────

@pytest.mark.parametrize("other", [
    "That is another matter entirely.",   # 'no' inside 'another'
    "It is a notable difference.",        # 'no' inside 'notable'
    "We denote the term by x.",           # 'no' inside 'denote'
])
def test_yes_is_not_opposed_by_a_word_containing_no(other):
    assert _opposing("The answer is yes.", other) == []


def test_true_is_not_opposed_by_a_word_containing_false():
    """`falsehood` starts with `false`; the trailing `hood` is a word character,
    so the term does not stand alone and the pair is not an opposition."""
    assert _opposing("The flag is true.", "The claim is a falsehood.") == []


def test_the_bare_terms_still_pair_when_standalone():
    """The other side of the boundary: a standalone `no` is still `no`. Without
    this the parametrised misses above could be satisfied by dropping `yes`/`no`
    from the list, which is not what the item asked for."""
    assert _opposing("The answer is yes.", "The answer is no.") == [
        "opposing_terms:yes/no"]


# ── clause 3: the guard still catches what it exists to catch ────────────────

def test_a_real_opposition_still_classifies_as_opposing_terms():
    """"The gate is enabled." vs "The gate is disabled." is the pair this guard
    was written for. Both terms are whole words here, so bounding must not
    cost the trigger its one real case."""
    assert _opposing("The gate is enabled.", "The gate is disabled.") == [
        "opposing_terms:enabled/disabled"]


def test_a_whole_word_self_collapsing_pair_is_still_an_opposition():
    """"is active" vs "is inactive" stays detectable, which is why the pairs
    were bounded rather than dropped or split."""
    assert _opposing("The daemon is active.", "The daemon is inactive.") == [
        "opposing_terms:active/inactive"]
    assert _opposing("The port is supported.", "The port is unsupported.") == [
        "opposing_terms:supported/unsupported"]


def test_all_seven_pairs_remain_listed_in_both_directions():
    """Nothing was dropped and nothing was split. The list is the guard's whole
    vocabulary; shrinking it to dodge a false positive would be the fix the item
    explicitly rejected."""
    assert _OPPOSING_PAIRS == [
        ("yes", "no"), ("true", "false"), ("enabled", "disabled"),
        ("active", "inactive"), ("supported", "unsupported"),
        ("working", "broken"), ("success", "failure")]


@pytest.mark.parametrize("left,right", [
    ("The gate is enabled.", "The gate is disabled."),
    ("The daemon is active.", "The daemon is inactive."),
    ("The port is supported.", "The port is unsupported."),
    ("The build is working.", "The build is broken."),
    ("The run was a success.", "The run was a failure."),
    ("The answer is yes.", "The answer is no."),
    ("The flag is true.", "The flag is false."),
])
def test_every_pair_trips_on_a_whole_word_opposition(left, right):
    """Purpose clause, one node per pair: for each of the seven, two plain
    sentences that put the terms in opposition still produce a reason starting
    `opposing_terms:`, in either order."""
    reasons = _opposing(left, right)
    assert len(reasons) == 1, (left, right, reasons)
    assert reasons[0].startswith("opposing_terms:"), reasons


def test_the_reason_names_the_pair_that_fired():
    """The trigger string is the reviewer's handle on the decision, so it has to
    identify a pair and not merely say 'opposing terms'."""
    assert _opposing("The build is working.", "The build is broken.") == [
        "opposing_terms:working/broken"]


# ── the seam the tool path actually uses ─────────────────────────────────────

def test_the_bounded_classification_is_what_fact_check_returns(world):
    """`fact_check` is the read surface an agent calls, and `fact_resolve`
    auto-resolves over the same list. Both go through
    `_detect_contradictions_sync`, so the boundary is proved here rather than
    assumed: two stored facts that merely share `unsupported` come back through
    the tool with no `opposing_terms` reason, while the stored enabled/disabled
    pair still does."""
    root = world
    d = root / "Probe"
    d.mkdir(parents=True)
    fm = {"type": "facts", "entity": "Probe", "category": "state", "facts": [
        {"id": "stat-001", "category": "state", "confidence": 0.9,
         "fact": "The feature is unsupported.", "created_at": "2026-09-01T00:00:00"},
        {"id": "stat-002", "category": "state", "confidence": 0.95,
         "fact": "Both paths are unsupported, measured 09-09.",
         "created_at": "2026-09-02T00:00:00"},
        {"id": "stat-003", "category": "state", "confidence": 0.9,
         "fact": "The gate is enabled.", "created_at": "2026-09-03T00:00:00"},
        {"id": "stat-004", "category": "state", "confidence": 0.8,
         "fact": "The gate is disabled.", "created_at": "2026-09-04T00:00:00"},
    ]}
    (d / "Probe-state.md").write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n",
                                      encoding="utf-8")
    out = facts_mod._fact_check({"entity": "Probe", "category": "state"})
    reasons = sorted(c["reason"] for c in out["contradictions"])
    assert "opposing_terms:enabled/disabled" in reasons, reasons
    assert not [r for r in reasons if "supported" in r], reasons


def test_opposing_reason_is_the_single_matcher_shared_by_callers():
    """One matcher, not a bounded copy beside the old one. The detector and any
    caller that wants the trigger go through `_opposing_reason`, so a second
    substring comparison cannot reappear beside the guarded one."""
    assert facts_mod._opposing_reason("the gate is enabled.", "the gate is disabled.") \
        == "opposing_terms:enabled/disabled"
    assert facts_mod._opposing_reason("both are unsupported here",
                                      "the feature is unsupported") is None
