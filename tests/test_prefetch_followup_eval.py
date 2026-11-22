"""Pins for the multi-turn prefetch gold set and its A/B runner (#4032).

Two things are pinned here and one in tests/test_prefetch.py:

  * the gold set's schema, its follow-up bounds and its content-word bound, on
    EVERY record, plus the existence of at least one should-be-empty case
    (clause 1);
  * that the runner builds a `SessionFocus` from the case's own prefix rather
    than from live session state, calls both legs twice per case under the two
    arms, passes the two arms DIFFERENT query strings, and that the focus arm's
    string carries a keyword from the prefix while the raw arm's does not
    (clause 2) — all with the legs stubbed, so no fact store or skill inventory
    is touched at collection time.

The unit under test is loaded from eval/run_prefetch_followup_eval.py by path, the
route tests/test_run_prefetch_eval_quiet_logging.py and
tests/test_tool_choice_eval_cost.py already use for a runner script: a test that
imported it by module name would depend on whether pytest happened to have `eval/`
on `sys.path`, and a test that shelled out to it could not assert which query
strings reached the legs.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import prefetch  # noqa: E402

GOLD = ROOT / "eval" / "prefetch_followup_queries.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "pf_followup_eval", ROOT / "eval" / "run_prefetch_followup_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pf_followup_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load_runner()

# ── clause 1: the gold set itself ─────────────────────────────────────────────


def test_gold_set_has_at_least_ten_well_formed_cases():
    cases = fu.load_cases(GOLD)
    assert len(cases) >= fu.MIN_CASES, (
        f"the gold set has {len(cases)} cases; clause 1 requires at least "
        f"{fu.MIN_CASES}")
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        assert c.prefix and all(p.strip() for p in c.prefix)
        assert c.source, f"{c.id}: every case names the transcript or probe it came from"
        assert c.scores_facts or c.scores_skills, f"{c.id} is scored by neither leg"


def test_every_follow_up_is_a_turn_that_reaches_the_legs():
    """10-60 characters, and above MIN_MESSAGE_LEN is not decoration.

    `_prefetch_prepare` returns None below MIN_MESSAGE_LEN, so a follow-up
    shorter than that scores a path no real turn ever takes.
    """
    assert fu.MIN_FOLLOW_UP_CHARS == prefetch.MIN_MESSAGE_LEN == 10
    cases = fu.load_cases(GOLD)
    for c in cases:
        assert 10 <= len(c.follow_up) <= fu.MAX_FOLLOW_UP_CHARS, (
            f"{c.id}: follow_up is {len(c.follow_up)} chars "
            f"({c.follow_up!r})")


def test_every_follow_up_is_under_the_content_word_bound():
    """Counted by prefetch's own noise-filtered counter, on every record.

    "and what about the other one?" is one content word (`other`) to
    `_extract_focus_keywords`, six to `str.split` — the count that decides
    whether the focus arm fires at all is the module's, so that is the count the
    bound is checked against. At the bound (8) enrichment would not fire, so the
    bound is validated strictly below it.
    """
    cases = fu.load_cases(GOLD)
    for c in cases:
        n = fu.content_words(c.follow_up)
        assert 0 < n < prefetch.FOLLOWUP_MAX_CONTENT_WORDS, (
            f"{c.id}: {n} content words in {c.follow_up!r}; the focus arm only "
            f"fires below {prefetch.FOLLOWUP_MAX_CONTENT_WORDS}")
    # The filed case is the shape this whole item is about: six words typed,
    # one content word counted.
    filed = next(c for c in cases if c.id == "qmd-daemon-other-one")
    assert fu.content_words(filed.follow_up) == 1


def test_gold_set_carries_should_be_empty_cases():
    """At least one case whose correct answer is that the leg returns nothing.

    Without these the set measures only hit rate, and on a follow-up the facts
    leg's failure is selecting the WRONG entity, not selecting none — that half
    would score as a wash.
    """
    cases = fu.load_cases(GOLD)
    facts_empty = [c for c in cases if c.facts_must_be_empty]
    assert facts_empty, "no facts_must_be_empty case in the gold set"
    assert any(c.skills_must_be_empty for c in cases), (
        "no skills_must_be_empty case in the gold set")
    for c in facts_empty:
        assert not c.expect_entities, (
            f"{c.id} asks for both a target and an empty block")


def test_loader_rejects_a_record_that_would_score_nothing(tmp_path):
    """The bound is a real gate, not a comment: a bad record fails the load."""
    import yaml

    good = yaml.safe_load(GOLD.read_text())
    base = good["cases"][0]

    too_short = dict(base, id="short", follow_up="what now?")  # 9 chars
    # Eight content words to `_extract_focus_keywords` (vault, indexer, fall,
    # behind, qmd, collection, sweep, re-run) inside 52 characters: past the
    # character bound it must still be refused by the content-word one.
    too_many_words = dict(base, id="long",
                          follow_up="vault indexer fall behind qmd collection "
                                    "sweep re-run")
    assert fu.content_words(too_many_words["follow_up"]) == 8
    unmeasurable = {k: v for k, v in dict(base, id="bare").items()
                    if k not in ("expect_entities", "expect_skills",
                                 "facts_must_be_empty", "skills_must_be_empty")}

    for bad, why in ((too_short, "under MIN_MESSAGE_LEN"),
                     (too_many_words, "at or over the content-word bound"),
                     (unmeasurable, "scored by neither leg")):
        p = tmp_path / "gold.yaml"
        p.write_text(yaml.safe_dump({"cases": [dict(base, id="keep"), bad]}))
        with pytest.raises(ValueError) as ei:
            fu.load_cases(p)
        assert bad["id"] in str(ei.value), f"{why}: {ei.value}"

    # And a set with no should-be-empty case is refused wholesale.
    no_empty = [dict(c, facts_must_be_empty=False, skills_must_be_empty=False,
                     expect_entities=c.get("expect_entities") or ["anything"],
                     expect_skills=c.get("expect_skills"))
                for c in good["cases"]]
    p = tmp_path / "gold2.yaml"
    p.write_text(yaml.safe_dump({"cases": no_empty}))
    with pytest.raises(ValueError, match="must-be-empty"):
        fu.load_cases(p)


# ── clause 2: the runner builds focus per case and arms the two legs ─────────


class _Recorder:
    """A stubbed leg that remembers every query it was handed."""

    def __init__(self, returns=()):
        self.calls = []
        self.returns = list(returns)

    def __call__(self, query, *args):
        self.calls.append(query)
        return list(self.returns)


def _one_case():
    return fu.Case(
        id="fixture",
        prefix=["the qmd collection daemon keeps timing out on the vault sweep"],
        follow_up="and what about the other one?",
        expect_entities=["qmd daemon"],
        expect_skills=["qmd-index-maintenance"])


def test_the_two_arms_hand_the_legs_different_queries_and_focus_carries_the_prefix():
    """Both legs, twice per case, with the focus arm's keyword absent from raw."""
    case = _one_case()
    facts, skills = _Recorder(["FACTS BLOCK"]), _Recorder(["some-skill"])
    focus = fu.build_focus(case.prefix)
    raw = fu.run_case(case, "raw", facts_fn=facts, skills_fn=skills)
    focus_row = fu.run_case(case, "focus", facts_fn=facts, skills_fn=skills,
                            focus=focus)

    assert facts.calls == [case.follow_up, focus_row["query"]]
    assert skills.calls == [case.follow_up, focus_row["query"]]
    assert facts.calls[0] != facts.calls[1], "the arms collapsed to one query"
    # Both arms reach BOTH legs, twice per case, once each.
    assert len(facts.calls) == len(skills.calls) == 2

    keyword = prefetch._extract_focus_keywords(case.prefix[0])[0]
    assert keyword in focus_row["query"], (
        f"the focus arm lost the prefix keyword {keyword!r}: {focus_row['query']!r}")
    assert keyword not in raw["query"], (
        f"the raw arm leaked the prefix keyword {keyword!r} into {raw['query']!r}")
    # The raw arm is exactly the typed message — the string the legs get today.
    assert raw["query"] == case.follow_up


def test_the_focus_arm_equals_what_production_returns_with_the_key_on(monkeypatch):
    """The harness's focus arm and production's chooser are the same string.

    Without this the runner could score a query the shipped code never sends, and
    the A/B would be measuring a third thing.
    """
    import app.config as cfg

    case = _one_case()
    focus = fu.build_focus(case.prefix)
    monkeypatch.setitem(cfg.CONFIG, "prefetch", {"followup": {"focus_legs": True}})
    assert prefetch.followup_focus_legs_enabled() is True
    assert prefetch._facts_skills_query(case.follow_up, focus) == fu.arm_query(
        case, "focus", focus=focus)
    monkeypatch.setitem(cfg.CONFIG, "prefetch", {"followup": {"focus_legs": False}})
    assert prefetch._facts_skills_query(case.follow_up, focus) == fu.arm_query(
        case, "raw")


def test_focus_is_built_from_the_case_prefix_and_never_from_live_state(monkeypatch):
    """A fresh tracker per case, in a session id that no real turn uses.

    prefetch's focus tracker is in-memory per session with LRU eviction at 50
    sessions and is gone on restart, so a runner that read it would be scoring
    whatever the last real turn happened to leave behind.
    """
    built = []

    class _Spy:
        def __init__(self):
            self.turns = []
            built.append(self)

        def update(self, text):
            self.turns.append(text)

        def enrich_query(self, q):
            return q + " enriched"

    cases = [
        fu.Case(id="a", prefix=["first alpha turn"], follow_up="and the other alpha one?",
                expect_entities=["alpha"]),
        fu.Case(id="b", prefix=["first beta turn", "second beta turn"],
                follow_up="and the other beta one?", expect_entities=["beta"]),
    ]
    rows = fu.run_arms(cases, ("focus",), facts_fn=_Recorder(["x"]),
                       skills_fn=_Recorder(["s"]), focus_factory=_Spy)

    assert len(rows) == 2, "one row per case per arm"
    assert [b.turns for b in built] == [["first alpha turn"],
                                        ["first beta turn", "second beta turn"]], (
        "each case's focus must carry exactly its own prefix, in order")
    assert all("enriched" in r["query"] for r in rows)
    assert len(built) == 2, "one focus per case, not one shared focus"


def test_run_arms_scores_both_legs_and_reports_a_rate_per_arm_per_leg():
    """The reported numbers are per arm and per leg, targets apart from empties."""
    case_target = fu.Case(
        id="target", prefix=["the qmd collection daemon keeps timing out"],
        follow_up="and what about the other one?",
        expect_entities=["qmd daemon"], expect_skills=["qmd-index-maintenance"])
    case_empty = fu.Case(
        id="empty", prefix=["Scott. Thank you. Yeah."],
        follow_up="Hey there. Leave it alone please.",
        facts_must_be_empty=True, skills_must_be_empty=True)
    cases = [case_target, case_empty]

    def facts(query):
        return ["<facts>qmd daemon restart</facts>"] if "qmd" in query else ["wrong entity"]

    def skills(query, session_id):
        return ["qmd-index-maintenance"] if "qmd" in query else []

    rows = fu.run_arms(cases, ("raw", "focus"), facts_fn=facts, skills_fn=skills)
    assert len(rows) == 4, "2 cases x 2 arms"
    summary = fu.summarise(cases, rows)

    assert summary["raw"]["facts"]["target_hits"] == 0
    assert summary["raw"]["facts"]["target_n"] == 1
    assert summary["focus"]["facts"]["target_hits"] == 1
    assert summary["focus"]["facts"]["target_rate"] == 1.0
    # The must-be-empty case is counted separately from the target case, per leg.
    assert summary["raw"]["facts"]["empty_n"] == 1
    assert summary["raw"]["facts"]["empty_ok"] == 0     # wrong entity returned
    assert summary["focus"]["skills"]["empty_ok"] == 1  # nothing offered
    assert summary["focus"]["facts"]["empty_n"] == 1
    assert set(summary["focus"]) == {"facts", "skills"}
    assert "facts" in fu.render_table(summary)


# ── #4296: the skills target scorer must read the shape production returns ───


def _production_shape(*names, score: float = 0.91):
    """A skills leg handing back what production's leg actually returns.

    `app/prefetch._search_skills_for_turn` is annotated
    `-> list[tuple[float, dict]]` and returns `(score, skill_dict)` pairs. Every stub
    in this file before #4296 returned plain strings instead, which is why the
    scorer's `s in names` test never failed here while scoring 0.0 in production.
    """
    return [(score, {"name": n, "description": f"{n} body"}) for n in names]


def _perfect_skills_leg(names):
    """A leg that always offers exactly `names`, in production's pair shape.

    The one gold case that expects no skill gets an empty list from this leg, which is
    what its must-be-empty clause asks for — so a perfect leg satisfies that case too.
    """
    pairs = _production_shape(*names)
    return lambda query, session_id: list(pairs)


def test_a_skills_hit_is_counted_against_the_name_inside_the_production_pair_shape():
    """A pair naming the expected skill is a hit; a pair naming another is not.

    The bug this pins: the scorer tested `s in names` where `names` was the list of
    `(score, dict)` pairs, so a slug was compared against tuples and every skills
    TARGET case scored False in both arms, whatever the leg returned — the tied
    skills arm that made #4296's adoption rule unsatisfiable by any run.
    """
    case = fu.Case(
        id="shape", prefix=["the qmd collection daemon keeps timing out"],
        follow_up="and what about the other one?",
        expect_skills=["qmd-index-maintenance"])

    hit = fu.run_case(case, "raw",
                      skills_fn=lambda q, s: _production_shape("qmd-index-maintenance"))
    assert hit["skills_hit"] is True, (
        "a (score, dict) pair naming the expected skill is a hit; the scorer is still "
        f"comparing the slug against the pair itself: {hit['skills_returned']}")

    miss = fu.run_case(case, "raw",
                       skills_fn=lambda q, s: _production_shape("some-other-skill"))
    assert miss["skills_hit"] is False, (
        "a pair naming a DIFFERENT skill must not count as the expected one")

    # The plain-string shape every pre-#4296 stub returns — and what
    # default_skills_fn's own annotation still claims — has to keep scoring too,
    # or fixing the production shape strands the harness's own stubs.
    assert fu.run_case(case, "raw", skills_fn=lambda q, s: [
        "qmd-index-maintenance"])["skills_hit"] is True, (
        "the string shape the existing stubs return stopped scoring")


def test_the_skills_must_be_empty_measure_reads_the_raw_return_not_the_names():
    """[] is a hit and one (score, dict) pair is a miss, exactly as before #4296.

    Kept on the RAW list the leg returned, deliberately: once the target scorer starts
    digging inside pairs, a measure that followed it would call an unnameable pair
    empty and quietly start awarding hits for garbage. The two branches must not
    converge, so the pair with no usable name is asserted here as a miss.
    """
    case = fu.Case(
        id="empty", prefix=["Scott. Thank you. Yeah."],
        follow_up="Hey there. Leave it alone please.",
        skills_must_be_empty=True)

    assert fu.run_case(case, "raw", skills_fn=lambda q, s: [])["skills_hit"] is True, (
        "a leg that offers nothing is what a must-be-empty case asks for")
    assert fu.run_case(case, "raw", skills_fn=lambda q, s: _production_shape(
        "qmd-index-maintenance"))["skills_hit"] is False, (
        "one offered skill is a miss for a must-be-empty case")
    assert fu.run_case(case, "raw", skills_fn=lambda q, s: [
        (0.5, {"score": 1.0})])["skills_hit"] is False, (
        "a pair with no name still fills the slot; it is not an empty answer")


def test_a_perfect_skills_leg_scores_the_shipped_gold_set_at_1_00_on_both_arms():
    """The number #4296 was blocked on: the arms can now differ on skills at all.

    Over the shipped 12 gold cases — 3 facts targets, 9 skills targets, 2 facts
    must-be-empty, 1 skills must-be-empty — a leg offering each case's own expected
    skill in production's pair shape scored skills target_rate 0.00 for BOTH arms
    before the fix, so "focus > raw on the corrected skills rate" could not be
    satisfied by any run. The facts leg is deliberately run empty here, which pins its
    two rates independently of the skills shape: 3 targets missed, 2 must-be-empty
    cases satisfied. The skills must-be-empty measure is n=1 and is asserted as that
    one case's outcome, not presented as a rate.
    """
    cases = fu.load_cases(GOLD)
    assert len(cases) == 12
    assert sum(1 for c in cases if c.expect_skills) == 9
    assert sum(1 for c in cases if c.expect_entities) == 3
    assert sum(1 for c in cases if c.skills_must_be_empty) == 1

    rows = []
    for case in cases:
        rows += fu.run_arms([case], ("raw", "focus"),
                            facts_fn=lambda q: [],
                            skills_fn=_perfect_skills_leg(case.expect_skills))
    assert len(rows) == 24, "12 cases x 2 arms"
    summary = fu.summarise(cases, rows)

    for arm in ("raw", "focus"):
        skills = summary[arm]["skills"]
        assert skills["target_n"] == 9
        assert skills["target_rate"] == 1.0, (
            f"the {arm} arm scores a perfect skills leg at "
            f"{skills['target_rate']}, not 1.0 — the pair shape is still unscored")
        # n=1: the case that expects no skill is offered nothing by a perfect leg,
        # so its must-be-empty clause is satisfied. Recorded as an outcome.
        assert skills["empty_n"] == 1
        assert skills["empty_ok"] == 1, (
            "the one skills must-be-empty case expects no skill, so a perfect leg "
            "offers it none and satisfies it")
        # Facts scoring is untouched by the skills fix.
        facts = summary[arm]["facts"]
        assert facts["target_n"] == 3 and facts["target_rate"] == 0.0
        assert facts["empty_n"] == 2 and facts["empty_rate"] == 1.0


def test_dry_run_never_touches_a_store(monkeypatch):
    """`--dry-run` is the mode a reviewer can run anywhere; it must not read KG."""
    def _boom(*a, **k):
        raise AssertionError("dry run reached the fact store")

    monkeypatch.setattr(prefetch, "_search_facts", _boom)
    rc = fu.main(["--dry-run", "--gold", str(GOLD), "--arm", "raw,focus",
                  "--ids", "qmd-daemon-other-one"])
    assert rc == 0
