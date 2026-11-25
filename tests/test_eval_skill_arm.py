"""#4399: the `skills_search` nightly arm in `eval/run_eval.py` (#4031's unlanded core).

`skills_search` carries real traffic (132 `"name": "skills_search"` calls in
`~/lloyd-data/sessions/*.json` as measured 2026-11-20, `.json.gz` archives excluded so
that figure is a floor) while the nightly recorded no number of its own: hit@1 0.722
lived only in prose (`eval/measurements/skills-search-embed-2026-11-10.md`). `--skill-arm`
closes that by scoring `eval/skill_match_queries.yaml` through the scorer that already
exists and publishing `summary.overall.skill_hit_at_1` into the nightly artifact.

What these nodes pin, clause by clause:

1. the flag writes the key into `summary.overall` of the artifact a REAL `main()` run
   writes, and a run without the flag writes no `skill`-bearing key at all;
2. the published value is `eval/run_skill_match_eval.py`'s OWN published number over one
   shared fixture — asserted as equality of the two published values with the same arm
   and the same labelled denominator, never against a formula restated here;
3. a query file that cannot be read, a scorer that raises, and the runner's stale-label
   exit all write the key as `null` and print `skill=absent` — never `skill_hit_at_1=0`,
   the same rule as `unverified=absent` at `eval/run_eval.py:1866`;
4. (in `tests/test_automod_regression.py`) the key stays out of the compared metric set;
5. the flag is a report-only arm: it adds no retrieval knob, so
   `matches_production_defaults` stays true and the nightly keeps its uptake pool.

Which arm: `tool_arm(True)` with its declared defaults (`weight=2.0`, `k=16`) is exactly
the arm `eval/run_skill_match_eval.py --tool-arm` publishes as `tool_embed_w2` — the
embedding re-order production serves (`config.yaml` `skills.search_embed_rerank: true`).
The same command also publishes `tool_lexical` (0.481 against 0.722), so a pinned arm
name is what keeps a future reader from comparing the nightly against the wrong arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

#: Two skills whose descriptions share the tokens of FIXTURE_TURN, so
#: `_skills_search_scored` scores BOTH for every turn — which is the precondition for
#: the re-order below (`_skills_search_scored` skips `rerank` when one candidate or fewer
#: survives lexically, and an arm that silently ran lexical would make the equality
#: below true for the wrong reason).
SKILLS = [
    {"name": "alpha-skill", "description": "restart lloyd services safely",
     "category": "ops", "tags": [], "body": "alpha body"},
    {"name": "beta-skill", "description": "check lloyd services health",
     "category": "ops", "tags": [], "body": "beta body"},
]
FIXTURE_TURN = "lloyd services"

#: The re-order's answer for every turn: `beta-skill` first. `score()` counts hit@1 over
#: labelled turns only, so with one labelled turn expecting alpha and one expecting beta
#: the published hit@1 is 0.5 — a value that separates the readings a wrong
#: implementation could produce: hit@1 over labelled turns is 0.5, hit@1 over ALL records
#: would be 0.25, and hit@5 would be 1.0.
EMBED_ORDER = ["beta-skill", "alpha-skill"]
EXPECTED_HIT_AT_1 = 0.5
EXPECTED_LABELLED = 2

#: `TURN_B` adds a token only `beta-skill`'s description carries, so the LEXICAL arm
#: ranks beta first on it and scores 1.0 over the two labelled turns, while the re-ordered
#: arm books beta first on `TURN_A` too and misses the turn that expects alpha. Equal
#: turns would make both arms 0.5 and the arm identity below uncheckable; this is what
#: makes `..._not_the_lexical_one` a difference rather than a coincidence.
TURN_B = "lloyd services health"

#: r3 carries no expected skill and no `expected_empty`, so it is in neither
#: denominator — the record that makes "same labelled denominator" a checkable claim
#: rather than a sentence.
FIXTURE_RECORDS = [
    {"id": "sx1", "turn": FIXTURE_TURN, "expected_skills": ["alpha-skill"]},
    {"id": "sx2", "turn": TURN_B, "expected_skills": ["beta-skill"]},
    {"id": "sx3", "turn": FIXTURE_TURN, "expected_skills": []},
    {"id": "sx4", "turn": FIXTURE_TURN, "expected_skills": [], "expected_empty": True},
]


@pytest.fixture
def fake_search(monkeypatch):
    """The embed arm over two skills, with the embedder kept out of the process.

    `_build_vectors` and `rerank` are the two calls in the arm's path that reach the
    model, and conftest's autouse fixture keeps `app.qwen3_embed` out of test
    processes; replacing them with a booked order is what makes the arm deterministic
    without replacing `score`, `load_records`, `check_labels` or `tool_arm` — the four
    callables the arm under test is required to USE.
    """
    from agent_mcp import skills as skills_mod
    from app import skill_embed_rank

    monkeypatch.setattr(skills_mod, "load_skills_cached", lambda: [dict(s) for s in SKILLS])
    monkeypatch.setattr(skills_mod, "_iter_skills", lambda: iter([dict(s) for s in SKILLS]))
    monkeypatch.setattr(skill_embed_rank, "_build_vectors", lambda skills: None)

    def fake_rerank(query, scored, skills, cfg):
        by_name = {sk["name"]: sk for _score, sk in scored}
        assert len(by_name) == len(SKILLS), (
            f"the lexical stage dropped a candidate for {query!r}, so the re-order "
            f"being booked here is not the one the arm ran: {sorted(by_name)}")
        return [(0.0, by_name[n]) for n in EMBED_ORDER if n in by_name]

    monkeypatch.setattr(skill_embed_rank, "rerank", fake_rerank)
    return dict(EMBED_ORDER=EMBED_ORDER)


@pytest.fixture
def fixture_queries(tmp_path):
    path = tmp_path / "skill_arm_fixture.yaml"
    path.write_text(yaml.safe_dump({"records": FIXTURE_RECORDS}, sort_keys=False))
    return path


def _skill_eval():
    from eval import run_skill_match_eval
    return run_skill_match_eval


# ---------------------------------------------------------------- clause 2


def test_the_published_value_is_the_scorer_module_own_number(fixture_queries, fake_search):
    """Clause 2: one shared fixture, two published numbers, asserted EQUAL.

    The reference side calls `run_skill_match_eval`'s own `score` over its own
    `load_records`/`tool_arm` — the code path `--tool-arm` runs to publish
    `tool_embed_w2`. The arm side is the callable `run_eval.main` uses. A second
    implementation of the hit@1 rule in `run_eval.py` (or a restated denominator) fails
    here even when both numbers are wrong, because this asserts one published value
    against the other, not against a number computed in this file.
    """
    import eval.run_eval as run_eval
    sm = _skill_eval()

    reference = sm.score(sm.load_records(fixture_queries), sm.tool_arm(True))["hit@1"]
    published = run_eval.score_skill_arm(fixture_queries)["hit@1"]
    assert published == reference == EXPECTED_HIT_AT_1, (published, reference)


def test_the_arm_scores_the_embedding_re_order_not_the_lexical_one(fixture_queries,
                                                                   fake_search):
    """The arm is `tool_embed_w2`, not `tool_lexical` — pinned by order, not by name.

    `eval/run_skill_match_eval.py:443-445` publishes two hit@1s under `--tool-arm`, and
    production serves the re-order, so the nightly number is the re-ordered one. Here the
    re-order books `beta-skill` first on every turn, including the turn that expects
    `alpha-skill`, so its hit@1 over the two labelled turns is 0.5. The lexical arm scores
    both turns (alpha wins the equal-score tie on turn 1 by `_skills_search_scored`'s
    stable sort, beta wins turn 2 on its own token), so it publishes 1.0 over the SAME
    denominator of 2 — the two arms are separable only by order, never by population.
    """
    import eval.run_eval as run_eval
    sm = _skill_eval()

    arm = run_eval.score_skill_arm(fixture_queries)
    lexical = sm.score(sm.load_records(fixture_queries), sm.tool_arm(False))
    assert arm["hit@1"] == EXPECTED_HIT_AT_1
    assert lexical["hit@1"] == 1.0 != EXPECTED_HIT_AT_1, (
        "the lexical arm no longer scores 1.0 over this fixture, so it stops "
        "distinguishing itself from the re-ordered arm and the arm identity is unpinned")
    # Same labelled denominator on both arms: the difference is order, not population.
    assert arm["labelled_turns"] == lexical["labelled_turns"] == EXPECTED_LABELLED


def test_the_arm_names_the_published_arm_it_reproduces(fixture_queries, fake_search):
    """The artifact says WHICH published arm it is, because two arms exist."""
    import eval.run_eval as run_eval
    summary = {"overall": {}}
    assert run_eval.attach_skill_arm(summary, enabled=True,
                                     queries_path=fixture_queries) is True
    o = summary["overall"]
    assert o["skill_arm_name"] == run_eval.SKILL_ARM_NAME == "tool_embed_w2"
    assert o["skill_arm_labelled_turns"] == EXPECTED_LABELLED


def test_the_printed_line_carries_the_value_and_its_denominator(fixture_queries,
                                                                fake_search):
    """Step 1 of the skill quotes this line verbatim, so its shape is a contract."""
    import eval.run_eval as run_eval
    summary = {"overall": {}}
    run_eval.attach_skill_arm(summary, enabled=True, queries_path=fixture_queries)
    line = run_eval.skill_arm_line(summary["overall"])
    assert line == (f"         skill arm: skill={EXPECTED_HIT_AT_1}  "
                    f"(hit@1 over {EXPECTED_LABELLED} labelled turns, "
                    f"arm={run_eval.SKILL_ARM_NAME})"), repr(line)


# ---------------------------------------------------------------- clause 3


def test_a_query_file_that_cannot_be_read_gives_null_and_absent(tmp_path, fake_search):
    """Clause 3, the load half: absent, never 0.

    A missing corpus file and a scorer that measured zero hits are different facts, and
    `skill_hit_at_1: 0.0` would be read as "skills_search got worse" on a night where it
    never ran — the same indistinguishability `unverified=absent` exists for.
    """
    import eval.run_eval as run_eval
    missing = tmp_path / "no-such-corpus.yaml"
    summary = {"overall": {}}
    assert run_eval.attach_skill_arm(summary, enabled=True,
                                     queries_path=missing) is True
    o = summary["overall"]
    assert o["skill_hit_at_1"] is None
    assert "skill_hit_at_1" in o
    assert run_eval.skill_arm_line(o).startswith("         skill arm: skill=absent ")
    assert "skill_hit_at_1=0" not in run_eval.skill_arm_line(o)
    assert str(missing) in o["skill_arm_note"]


def test_a_stale_label_gives_null_and_absent(fake_search, tmp_path):
    """Clause 3, the runner's exit-2 half: one retired skill and the arm is absent.

    `run_skill_match_eval.main` prints the stale labels and returns 2 rather than
    scoring them. `score_skill_arm` raises the same refusal in process, so a retired or
    renamed skill reaches the nightly as `skill=absent` plus the label that broke, not
    as a denominator change that reads as a hit@1 move.
    """
    import eval.run_eval as run_eval
    stale = tmp_path / "stale.yaml"
    stale.write_text(yaml.safe_dump({"records": [
        {"id": "sx1", "turn": FIXTURE_TURN, "expected_skills": ["retired-skill"]},
        {"id": "sx2", "turn": FIXTURE_TURN, "expected_skills": ["beta-skill"]},
    ]}))
    summary = {"overall": {}}
    assert run_eval.attach_skill_arm(summary, enabled=True, queries_path=stale) is True
    o = summary["overall"]
    assert o["skill_hit_at_1"] is None
    assert "retired-skill" in o["skill_arm_note"], o["skill_arm_note"]
    assert run_eval.skill_arm_line(o).startswith("         skill arm: skill=absent ")


def test_a_scorer_that_raises_gives_null_and_absent(monkeypatch, fixture_queries,
                                                    fake_search):
    """Clause 3, the raise half — and the run survives it.

    An embedder that does not answer is the expected production failure: `tool_arm(True)`
    waits on `_build_vectors` with a 60 s bound and then asks for one embed per turn, so
    a model outage arrives here as an exception, not as a low score.
    """
    import eval.run_eval as run_eval
    sm = _skill_eval()

    def boom(*a, **k):
        raise RuntimeError("embedder unreachable")
    monkeypatch.setattr(sm, "score", boom)

    summary = {"overall": {}}
    assert run_eval.attach_skill_arm(summary, enabled=True,
                                     queries_path=fixture_queries) is True
    o = summary["overall"]
    assert o["skill_hit_at_1"] is None
    assert "RuntimeError" in o["skill_arm_note"] and "embedder unreachable" in o["skill_arm_note"]
    assert run_eval.skill_arm_line(o) == ("         skill arm: skill=absent "
                                          f"({o['skill_arm_note']})")


def test_a_labelled_population_of_zero_is_absent_not_zero(fake_search, tmp_path):
    """`score` returns null over zero labelled turns; the arm must not turn that into 0.

    A corpus of expected-empty controls only is a real shape (`check_labels` is clean on
    it), and hit@1 over nothing is no verdict — `rate()` in the scorer already says so.
    """
    import eval.run_eval as run_eval
    empty = tmp_path / "empty.yaml"
    empty.write_text(yaml.safe_dump({"records": [
        {"id": "sx4", "turn": FIXTURE_TURN, "expected_skills": [],
         "expected_empty": True}]}))
    summary = {"overall": {}}
    run_eval.attach_skill_arm(summary, enabled=True, queries_path=empty)
    assert summary["overall"]["skill_hit_at_1"] is None
    assert run_eval.skill_arm_line(summary["overall"]).startswith(
        "         skill arm: skill=absent ")


def test_the_absent_rule_reuses_the_existing_absent_vocabulary():
    """The clause names `unverified=absent` as the shape to copy: same word, same rule.

    Pinned as bytes rather than as behaviour, because the behaviour is pinned above and
    what drifts is someone inventing a second spelling (`none`, `-`, `n/a`, `0`).
    """
    src = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
    assert "skill=absent" in src, "the printed absent token is the contract the skill quotes"
    assert "unverified=absent" in src, "the existing rule this one is shaped like"
    arm_src = src[src.index("def skill_arm_line"):src.index("def skill_arm_line") + 1200]
    assert "None" in arm_src and "absent" in arm_src


# ---------------------------------------------------------------- clause 1


def _drive_a_nightly(tmp_path, monkeypatch, label, extra_argv):
    """Run the REAL `eval/run_eval.py` end to end over a one-query fixture.

    `build_parser`, `build_run_config`, `summarize`, the artifact write and the JSON on
    disk are all real; only the corpus touch and the embedder are replaced, because what
    is under test is whether the flag reaches the artifact's `summary.overall` — and the
    seam between `main()` and the file it writes is exactly where a flag that only
    affects stdout would hide.
    """
    import sys as _sys

    import eval.run_eval as ev

    def fake_recall(params, **kw):
        return {"entities": [], "facts": [{"entity": "AlphaSkill", "text": "t",
                                          "source": "memory/x.md"}],
                "documents": [{"path": "memory/daily/2026-11-01.md"}],
                "graph_expanded_facts": [], "graph_neighbors_used": [],
                "fact_read_coverage": {"attempted": 1, "read": 1},
                "graph_expansion": {}}

    class _S:
        def stats(self):
            return {"entities_total": 1, "edges_total": 1, "edges_active": 1,
                    "aliases": 0, "facts": 1}

        def resolve(self, text):
            return text

        def active_edges(self):
            return []

        def neighbours(self, *a, **k):
            return {}

    qf = tmp_path / "q.yaml"
    qf.write_text(
        "queries:\n"
        "  - id: skill-arm-night\n"
        "    query: which skill restarts lloyd services\n"
        "    category: entity\n"
        "    expect_entities: [AlphaSkill]\n"
        "    expect_docs: [memory/daily/2026-11-01.md]\n")
    out = tmp_path / "baselines"
    monkeypatch.setattr(ev, "_vault_recall", fake_recall)
    monkeypatch.setattr(ev, "_recall_seeds", lambda q, k: [])
    monkeypatch.setattr(ev, "_semantic_seed_k", lambda: 0)
    monkeypatch.setattr(ev, "store", _S)
    monkeypatch.setattr(ev, "EVAL_BASELINES_DIR", out)
    monkeypatch.setattr(ev, "LLOYD_CODE_ROOT", tmp_path)
    # The arm's corpus reaches `main` through the module constant, so the flag is the
    # only thing distinguishing these two runs.
    corpus = tmp_path / "nightly-skill-corpus.yaml"
    corpus.write_text(yaml.safe_dump({"records": FIXTURE_RECORDS}, sort_keys=False))
    monkeypatch.setattr(ev, "SKILL_ARM_QUERIES", corpus)
    monkeypatch.setattr(_sys, "argv",
                        ["run_eval.py", "--queries", str(qf), "--label", label,
                         "--allow-empty-corpus", *extra_argv])
    assert ev.main() == 0
    return json.loads(next(iter(sorted(out.glob(f"{label}-*.json")))).read_text())


def test_the_flag_writes_skill_hit_at_1_into_the_artifact(tmp_path, monkeypatch,
                                                          fake_search):
    """Clause 1's green case, read off the written file rather than off stdout."""
    blob = _drive_a_nightly(tmp_path, monkeypatch, "nightly-skill-arm-on",
                            ["--skill-arm"])
    o = blob["summary"]["overall"]
    assert o["skill_hit_at_1"] == EXPECTED_HIT_AT_1, o.get("skill_hit_at_1")
    assert o["skill_arm_name"] == "tool_embed_w2"
    assert o["skill_arm_labelled_turns"] == EXPECTED_LABELLED
    # Report-only in the strongest sense available to a flag: the night is still
    # production's configuration, so `app/uptake.py`'s `retrieval_gate()` keeps it in
    # the non-regression pool. A `--skill-arm` that cost the nightly its own nights is
    # the failure mode the SKILL.md sentence exists to prevent.
    assert blob["matches_production_defaults"] is True, blob["matches_production_defaults"]


def test_a_run_without_the_flag_writes_no_skill_key_at_all(tmp_path, monkeypatch,
                                                           fake_search):
    """Clause 1's negative case: absent, not null, not 0.

    The default run must stay byte-comparable with the whole 2026-09-23-on series, so
    the arm's fields may not exist unless the flag asked for them. A `null` here would
    also read as "scored and found nothing" to `scripts/eval_trend_stats.py`.
    """
    blob = _drive_a_nightly(tmp_path, monkeypatch, "nightly-skill-arm-off", [])
    o = blob["summary"]["overall"]
    assert not [k for k in o if "skill" in k], sorted(o)
    assert not [k for k in blob if "skill" in k], sorted(blob)


def test_the_flag_is_declared_and_reaches_the_artifact_call():
    """The producer half as bytes: `build_parser` declares the flag and `main` passes it.

    A parser built by hand in a test can only assert about values the test invented
    (#999), so this pins the wiring instead of re-declaring it: the flag is one of the
    parser's own actions, and the `attach_skill_arm` call in `main` is fed
    `args.skill_arm`.
    """
    src = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
    assert '"--skill-arm"' in src, "the flag must be declared by build_parser()"
    assert "attach_skill_arm(" in src
    main_src = src[src.index("def main("):]
    assert "attach_skill_arm" in main_src, "the arm must be attached by main(), not a test"
    assert "args.skill_arm" in main_src

    import eval.run_eval as ev
    ns = ev.build_parser().parse_args(["--skill-arm"])
    assert ns.skill_arm is True
    assert ev.build_parser().parse_args([]).skill_arm is False


def test_the_flag_is_not_a_term_of_the_production_defaults_conjunction():
    """Clause 5's code half: `--skill-arm` must never enter the six-knob conjunction.

    `build_run_config`'s conjunction is what `app/uptake.py`'s `retrieval_gate()` reads
    to pick its pool, and every term in it is a parsed retrieval knob against an imported
    `RECALL_*` constant. `skill_arm` sets none of the six, so it belongs to
    `expand_graph`/`semantic_seeding`/`observed_ranker`: recorded, never compared. If
    someone folds it in, the conjunction reads false on the nightly and uptake's band
    stops seeing every future night — which is why this is pinned as bytes and not only
    by the artifact test above.
    """
    src = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
    cfg = src[src.index("def build_run_config"):src.index("def main(")]
    conjunction = cfg[cfg.index("matches_production_defaults"):]
    assert "skill_arm" not in conjunction, (
        "`--skill-arm` is a report-only arm; a term of the conjunction would drop every "
        "nightly night out of app/uptake.py's retrieval_gate() pool")
    assert "skill_arm" not in cfg, (
        "the flag is not even recorded in the run's retrieval shape: it scores a "
        "different corpus and must not read as a retrieval knob")


def test_the_arm_reuses_the_scorer_module_rather_than_copying_it():
    """The seam test for clause 2: the arm imports the scorer, and imports its loader.

    `graph_explain` is blind across the `import eval.run_skill_match_eval` seam, so the
    reuse is pinned as bytes: `score_skill_arm` must call the other module's
    `score`/`tool_arm`/`load_records`/`check_labels`. A future reimplementation inside
    `run_eval.py` would keep every value test above green only until the two rules
    drifted, which is the failure mode #4031 was opened for.
    """
    src = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
    body = src[src.index("def score_skill_arm"):src.index("def attach_skill_arm")]
    for called in ("run_skill_match_eval", "load_records", "check_labels",
                   "active_skill_names", "tool_arm", "score("):
        assert called in body, f"`score_skill_arm` no longer reaches {called!r}"
    assert Path(ROOT / "eval" / "run_skill_match_eval.py").is_file()
    # The scorer's own output key, as an indexing expression: `run_eval` reads the report
    # the other module returns and never derives the number itself. Prose may say hit@1;
    # code here may not spell the key, because the moment it does, `run_eval` has an
    # implementation of the rule to keep in step with someone else's.
    assert '"hit@1"' not in body, (
        "score_skill_arm must return the scorer's report, not compute hit@1 itself")


def test_the_corpus_the_arm_scores_is_the_one_the_scorer_ships():
    """`SKILL_ARM_QUERIES` is the scorer's own file, not a copy of its path."""
    import eval.run_eval as ev
    assert ev.SKILL_ARM_QUERIES == ROOT / "eval" / "skill_match_queries.yaml"
    assert ev.SKILL_ARM_QUERIES.is_file()
