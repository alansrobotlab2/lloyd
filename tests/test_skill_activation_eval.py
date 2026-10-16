"""#711: the per-skill activation eval — triggers, misses, false triggers.

Fixture skills go through the real loader and the real matcher
(`prefetch._search_skills` with a substituted inventory), so these pin the
definitions without reading the live vault. The checked-in corpus is checked
for its contract from the YAML alone.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from scripts import skill_activation as A

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "eval" / "run_skill_activation_eval.py"


def _runner():
    spec = importlib.util.spec_from_file_location("run_skill_activation_eval", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skill(name, desc):
    return A.skill_from_text(name, f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n")


TEA = "tea-brewing"
RECORDS = [
    {"id": "p1", "turn": "how long should I steep oolong", "expected_skills": [TEA]},
    {"id": "p2", "turn": "what temperature for green leaves", "expected_skills": [TEA]},
    {"id": "p3", "turn": "steep the oolong again please", "expected_skills": [TEA]},
    {"id": "p4", "turn": "brew a pot of tea for me", "expected_skills": [TEA]},
    {"id": "p5", "turn": "the green leaves taste bitter, what temperature", "expected_skills": [TEA]},
    {"id": "n1", "turn": "make coffee in the french press", "expected_skills": ["coffee-making"]},
    {"id": "n2", "turn": "espresso shot is too sour", "expected_skills": ["coffee-making"]},
    {"id": "n3", "turn": "grind coffee beans coarse for french press", "expected_skills": ["coffee-making"]},
    {"id": "n4", "turn": "restart the backend server now", "expected_skills": [], "expected_empty": True},
    {"id": "n5", "turn": "what's the weather tomorrow afternoon", "expected_skills": [], "expected_empty": True},
]


def _inventory(tea_desc):
    return [_skill("coffee-making", "coffee french press espresso grinder beans"),
            _skill(TEA, tea_desc)]


def test_counts_misses_and_false_triggers_on_a_fixture():
    cases = A.build_cases(RECORDS, [TEA])
    res = A.evaluate(cases, _inventory("steep oolong leaves, coffee espresso french press"))
    t = res["skills"][TEA]
    # p2 and p5 need `green`/`temperature`, which this description lacks.
    assert (t["positives"], t["triggers"], t["misses"]) == (5, 3, 2)
    # n1 and n3 carry three coffee words each, enough for the excerpt slot.
    assert (t["negatives"], t["false_triggers"]) == (5, 2)
    assert t["recall"] == 0.6 and t["false_trigger_rate"] == 0.4
    by_id = {r["id"]: r for r in t["rows"]}
    assert by_id["n1"]["winner"] == "coffee-making" and by_id["n1"]["pass"] is False
    assert by_id["n1"]["expected"] == ["coffee-making"], "a negative names who should have fired"
    assert by_id["n4"]["expected"] == [] and by_id["n4"]["pass"] is True
    assert res["corpus"]["false_triggers"] == 2 and res["corpus"]["recall"] == 0.6


def test_a_rate_over_nothing_prints_its_denominator():
    cases = A.build_cases(RECORDS[5:], [TEA])  # negatives only
    res = A.evaluate(cases, _inventory("steep oolong leaves"))
    t = res["skills"][TEA]
    assert t["recall"] is None and t["false_trigger_rate"] == 0.0
    lines = "\n".join(_runner().render(res))
    assert "n/a (denominator 0)" in lines
    assert "0.0000" in lines


def test_positives_without_negatives_fail_the_run(monkeypatch, capsys):
    ev = _runner()
    only_pos = [r for r in RECORDS if r["expected_skills"] == [TEA]]
    monkeypatch.setattr(A, "load_spec", lambda *a, **k: {"skills": {TEA: {}}, "extra_records": []})
    monkeypatch.setattr(A, "load_records", lambda *a, **k: only_pos)
    monkeypatch.setattr(A, "base_skills", lambda: _inventory("steep oolong leaves"))
    monkeypatch.setattr("app.prefetch._get_skills_cached", lambda: _inventory("steep oolong leaves"))
    assert ev.main([]) == 1
    assert "positives and no negatives" in capsys.readouterr().err
    assert A.contract_errors(A.build_cases(only_pos, [TEA])) == [
        f"{TEA}: 5 positives and no negatives"]


def test_two_runs_report_identical_counts_and_rows():
    cases = A.build_cases(RECORDS, [TEA])
    inv = _inventory("steep oolong and green leaves at the right temperature")
    ev = _runner()
    one = ev.build_report(A.evaluate(cases, inv))
    two = ev.build_report(A.evaluate(cases, inv))
    one.pop("measured_at"), two.pop("measured_at")
    assert one == two
    row = one["rows"][0]
    assert set(row) >= {"id", "label", "winner", "pass"}
    assert one["matches_production_defaults"] is True
    assert {"graph_rerank", "rerank_alpha", "graph_top_k", "graph_hops"} <= set(one)


def test_the_baseline_lives_under_the_data_root():
    from app.paths import EVAL_BASELINES_DIR
    assert _runner().baseline_path() == EVAL_BASELINES_DIR / "skill_activation_baseline.json"


def test_activation_is_what_prefetch_injects(monkeypatch):
    """Top skill at the first threshold, the second only at SKILL_THRESHOLD_SECOND."""
    from app import prefetch
    a, b = {"name": "a"}, {"name": "b"}
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks, skills=None: [(9.0, a), (prefetch.SKILL_THRESHOLD_SECOND - 0.1, b)])
    assert A.injected("a long enough turn") == ["a"]
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks, skills=None: [(9.0, a), (prefetch.SKILL_THRESHOLD_SECOND, b)])
    assert A.injected("a long enough turn") == ["a", "b"]
    # Since #435 `_search_skills` returns every offer above the report floor;
    # one below SKILL_THRESHOLD_FIRST is offered, never injected.
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks, skills=None: [(prefetch.SKILL_THRESHOLD_FIRST - 0.1, a)])
    assert A.injected("a long enough turn") == []
    assert A.injected("hi") == [], "prefetch skips a message under MIN_MESSAGE_LEN"


def test_the_checked_in_corpus_is_derived_and_meets_its_contract():
    spec = A.load_spec()
    assert len(spec["skills"]) >= 5
    raw = yaml.safe_load(A.CASES.read_text())
    assert set(raw) <= {"skills", "extra_records"}, "cases come from the #557 set, not a copy"
    queries = {r["id"] for r in yaml.safe_load(A.QUERIES.read_text())["records"]}
    assert not queries & {r["id"] for r in spec["extra_records"]}
    assert all("authored" in r["source"] for r in spec["extra_records"])
    cases = A.build_cases(A.load_records(spec), spec["skills"])
    assert A.contract_errors(cases, 5) == []
    assert all(isinstance(e.get("recall_floor"), float) for e in spec["skills"].values())


@pytest.mark.live_vault
def test_every_covered_skill_is_live():
    active = {s["name"] for s in A.base_skills()}
    assert set(A.load_spec()["skills"]) <= active


# ── #2812: the three recurring worker-job skills join the corpus ──────────────
#
# The five #711 shipped are the skills the #557 labelled turns happened to cover.
# The three added here (youtube-digest, paper-digest, nightly-skills-management)
# are the ones that corpus cannot reach: they are the protocols the recurring
# worker jobs run, and a job's dispatch never hands an interactive turn to the
# matcher, so `git grep` for their slugs in eval/skill_match_queries.yaml returns
# nothing and every one of their positives had to be authored.
#
# The nodes split on what needs the live vault. The corpus contract — keys, ids,
# provenance, per-skill case counts against the shipped MIN_CASES — is YAML and
# runs everywhere. Whether the matcher actually fires these skills on their own
# positives, and does not fire them on everything else, is a measurement over the
# 197-skill inventory, so it is `live_vault`-marked like
# `test_every_covered_skill_is_live`, and the numbers are read out of the runner's
# own printed table rather than restated here.

NEW = ["youtube-digest", "paper-digest", "nightly-skills-management"]
ORIGINAL = ["voice-mode", "local-llm-gotchas", "poisoned-worker-troubleshoot",
            "system-health-check", "voice-clone-sample"]


def _added_records(spec):
    """The `extra_records` #2812 added, i.e. everything past the one #711 shipped."""
    return [r for r in spec["extra_records"] if r["id"] != "sac001"]


def test_the_three_recurring_job_skills_are_covered_with_measured_float_floors():
    """Clause 1: the `skills:` map is the allowlist the gate reads, so a skill is
    blockable exactly when it appears there.

    Eight entries, the five #711 shipped plus the three worker-job skills, each with
    a float `recall_floor`. The floor is a measurement, never a wish — which is why
    the type is asserted, not just the key: `A.gate` skips its recall check when the
    floor is missing, and a string `"1.0"` would sit there doing nothing while
    reading fine in the table.
    """
    spec = A.load_spec()
    assert set(spec["skills"]) == set(ORIGINAL) | set(NEW), (
        "the corpus's coverage drifted: it names the gate's allowlist, so a skill "
        "that leaves this map silently stops being scored when its SKILL.md is "
        "rewritten — drop one only on purpose")
    for slug in NEW:
        floor = spec["skills"][slug].get("recall_floor")
        assert isinstance(floor, float) and not isinstance(floor, bool), (
            f"{slug} has no measured float recall_floor ({floor!r}); the corpus's one "
            f"job is to carry a number a baseline run produced")
    labelled = {s for r in yaml.safe_load(A.QUERIES.read_text())["records"]
                for s in (r.get("expected_skills") or [])}
    assert not labelled & set(NEW), (
        f"{sorted(labelled & set(NEW))} now has a labelled turn in "
        f"{A.QUERIES.name}, so its authored positives are no longer the only "
        f"evidence for it — fold the real turn into that file and re-run the baseline "
        f"instead of leaving a hand-made floor standing as the measurement")


def test_the_added_cases_are_authored_records_and_meet_the_shipped_five_case_floor():
    """Clauses 2, 3 and 5 as far as the YAML can show them: provenance, and the
    corpus against `contract_errors(cases, 5)` with no exception for anybody.

    Fifteen authored positives — five per new skill, because `MIN_CASES = 5` is the
    runner's own floor and a single-positive skill is what it refuses — plus three
    near-miss negatives. The negatives are the part worth pinning: `build_cases`
    makes every record that does not name a skill a negative for it, so a new skill
    arrives with ~89 of them whether anyone authors any or not, and what could only
    be authored are the turns that look like the new skill's subject and belong to a
    different skill (an upload, an arxiv fetch, a lint). Those are the cases that
    would catch a skill tuned to fire on every turn.
    """
    spec = A.load_spec()
    raw = yaml.safe_load(A.CASES.read_text())
    assert set(raw) == {"skills", "extra_records"}, (
        "the top-level keys moved, and scripts/automod/vault_round.py's activation "
        "scoring parses this file through load_spec on every vault landing")

    added = _added_records(spec)
    assert len(added) == 18, (
        f"#2812 added 18 records (15 positives, 3 near-miss negatives); found "
        f"{len(added)}")
    ids = {r["id"] for r in added}
    assert len(ids) == len(added), "two added records share an id"
    queries = {r["id"] for r in yaml.safe_load(A.QUERIES.read_text())["records"]}
    assert not queries & ids, "an added record reuses a #557 transcript id"
    assert all("authored" in r["source"] for r in added), (
        "an added record does not say so in its source; a positive nobody can tell "
        "was hand-made is a positive nobody can defend when it starts failing")

    for slug in NEW:
        pos = [r for r in added if slug in (r.get("expected_skills") or [])]
        assert len(pos) == 5, (
            f"{slug} has {len(pos)} authored positives; 5 is what "
            f"`contract_errors(cases, MIN_CASES)` needs, and every positive of a "
            f"5-case skill is worth 0.2 of its floor")
        near = [r for r in added if not (set(r.get("expected_skills") or []) & set(NEW))
                and r.get("expected_skills")]
        assert len(near) >= 3, (
            "the near-miss negatives are gone; without them every negative a new "
            "skill has is a turn about something plainly else, and a skill that "
            "fires on any youtube or paper word would score clean")
        for r in near:
            assert not set(r["expected_skills"]) & set(spec["skills"]), (
                f"{r['id']} expects {r['expected_skills']}, all of which are covered, "
                f"so it is a positive for a covered skill rather than a negative "
                f"shaped like the new ones")

    cases = A.build_cases(A.load_records(spec), spec["skills"])
    assert A.contract_errors(cases, _runner().MIN_CASES) == [], (
        "the corpus no longer meets its own floor with no per-skill exception")
    assert _runner().MIN_CASES == 5, (
        "MIN_CASES moved; growing the corpus was the task, lowering its bar was not")
    for slug in NEW:
        pos = sum(c["should_trigger"] for c in cases[slug])
        neg = len(cases[slug]) - pos
        assert pos >= 5 and neg >= 5, f"{slug}: {pos} positives / {neg} negatives"


@pytest.mark.live_vault
def test_the_runner_scores_the_three_new_skills_on_the_live_matcher(capsys):
    """Clauses 2 and 4, over the real matcher: the boundary this growth is supposed
    to be worth crossing is YAML -> `build_cases` -> `prefetch._search_skills` over
    the live inventory -> the printed table, and the table is what a person reads at
    a vault landing, so the assertion parses the table rather than the dict behind it.

    Two numbers per new skill, and both matter: recall above 0 (the positives fire)
    and a false-trigger rate strictly below 1.0. A skill that injected on every turn
    would have a perfect recall, which is precisely how a greedy skill could join
    this corpus and look like coverage. `test_a_rate_over_nothing_prints_its_
    denominator` covers the other failure shape: a rate with no population prints its
    denominator instead of a number, and must not read as a clean 0 here.
    """
    ev = _runner()
    assert ev.main([]) == 0, "the runner refused the grown corpus (see its stderr)"
    rows = _table_rows(capsys.readouterr().out)
    active = {s["name"] for s in A.base_skills()}
    for slug in NEW:
        row = rows[slug]
        assert row["positives"] >= 5 and row["negatives"] >= 5, f"{slug}: {row}"
        assert row["recall"] is not None and row["recall"] > 0, (
            f"{slug} fired on none of its own positives: {row} — the case count is "
            f"met but the corpus scores nothing, so a rewrite of its SKILL.md would "
            f"be gated against a floor nobody can reach")
        assert row["false_trigger_rate"] is not None \
            and row["false_trigger_rate"] < 1.0, (
            f"{slug} injects on every negative it was given ({row}); a skill that "
            f"fires on every turn gets full marks on recall, which is the one way to "
            f"pass this corpus by being useless")
    covered = set(A.load_spec()["skills"])
    assert covered <= active and len(covered) < len(active), (
        f"{len(covered)} covered against {len(active)} active is no longer a strict "
        f"minority — this corpus is an allowlist of the skills someone labelled, not "
        f"a duty to cover all 197, and the gate would be refusing landings on a "
        f"number nobody authored")


_TABLE_RE = re.compile(
    r"^(?P<skill>\S+)\s+(?P<positives>\d+)\s+(?P<triggers>\d+)\s+(?P<misses>\d+)\s+"
    r"(?P<negatives>\d+)\s+(?P<false_triggers>\d+)\s+"
    r"(?P<recall>[\d.]+|n/a \(denominator \d+\)) / "
    r"(?P<false_trigger_rate>[\d.]+|n/a \(denominator \d+\))$")


def _table_rows(text: str) -> dict:
    """The runner's printed table, keyed by skill. `n/a` becomes None, so an empty
    population fails the caller's comparison instead of comparing as a string."""
    rows = {}
    for line in text.splitlines():
        m = _TABLE_RE.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        out = {"skill": g["skill"]}
        for k in ("positives", "triggers", "misses", "negatives", "false_triggers"):
            out[k] = int(g[k])
        for k in ("recall", "false_trigger_rate"):
            out[k] = None if g[k].startswith("n/a") else float(g[k])
        rows[g["skill"]] = out
    assert rows, f"no runner table found in:\n{text[:400]}"
    return rows
