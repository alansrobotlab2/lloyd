"""Tests for the durable-artifact funnel census (#2507).

`scripts/funnel_census.py` joins four instruments that each used to publish a
number in their own artifact — #711's matcher score, #435's offer/loaded
telemetry, #552's uptake attribution, #2378's priced landings — into one dated
stage read. The clauses below are that item's acceptance, one test each, pinned
against fixtures rather than the live stores: the census reads the production
event log, `workers.db` and the vault, none of which a round controls.

Hermetic by construction: every store is a `tmp_path` fixture passed in as an
argument, `app.paths.DATA_ROOT` is monkeypatched for the default-path case, and
the clause about the *checkout* asserts a refusal — so nothing here ever creates
a file under the tree it is testing.

Run: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest \
        tests/test_funnel_census.py -q
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import paths  # noqa: E402
from eval.stats import wilson_ci  # noqa: E402
from scripts import funnel_census as fc  # noqa: E402

RUNS_DDL = """
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL,
  started_at TEXT NOT NULL, completed_at TEXT NOT NULL, duration_seconds REAL)
"""


# ── fixtures ──────────────────────────────────────────────────────────────────

def _stamp(**delta) -> str:
    """An event stamp in the shape `app.event_log` writes (UTC, ms, trailing Z)."""
    ts = datetime.now(timezone.utc) - timedelta(**delta)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def _skill(root: Path, name: str, *, status: str = "active") -> None:
    """One skill directory as the loader walks it (#1294's rules)."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A fixture skill called {name}.\n"
        f"category: test\ntags:\n- test\nstatus: {status}\n---\n"
        f"# {name}\n\nBody.\n", encoding="utf-8")


def _match(name: str, *, landed: bool, body: bool = False,
           score: float = 8.0, ts: str | None = None) -> dict:
    return {"ts": ts or _stamp(hours=2), "event": "prefetch.skill_match",
            "data": {"skill": name, "score": score, "landed": landed,
                     "injected_body": body}}


def _embedded(name: str, route: str, chars: int = 1200) -> dict:
    return {"ts": _stamp(hours=2), "event": "skill.embedded",
            "data": {"route": route, "skill": name, "embedded_chars": chars}}


def _events(root: Path, session_id: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{session_id}.events.jsonl").open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _workers_db(path: Path, rows: list[tuple]) -> Path:
    """`workers.db`'s `runs` table with exactly the rows a case needs.

    Rows are (run_id, source, started_at, duration_seconds), same column types as
    production, so the TEXT comparison on `started_at` is being tested rather
    than assumed.
    """
    con = sqlite3.connect(path)
    con.execute(RUNS_DDL)
    for run_id, source, started, secs in rows:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)",
                    (run_id, source, "done", started, started, secs))
    con.commit()
    con.close()
    return path


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(
        timespec="seconds")


def _empty_events(tmp_path: Path) -> Path:
    d = tmp_path / "bare_event_logs"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def corpus(tmp_path: Path) -> dict:
    """A skills root, an event log and a workers.db, sized so every stage bites.

    4 skill directories (3 served, 1 retired in place) → produced 3/4.
    4 offer rows, 2 of them landed → surfaced 2/4, so offers-that-never-landed is
    2/4 and is the widest loss in the fixture. `retired-offer` is offered but is
    not in the cohort, and `gamma` reaches a prompt on the uncapped autonomy
    route only: both exercise the cohort intersection rather than only the happy
    path. The db holds two producing sources, one non-producing source, and one
    run 400 days old that the window must exclude.
    """
    skills = tmp_path / "skills"
    for name in ("alpha", "beta", "gamma"):
        _skill(skills, name)
    _skill(skills, "retired-offer", status="archived")
    events = tmp_path / "event_logs"
    _events(events, "sess-a", [
        _match("alpha", landed=True, body=True),
        _match("alpha", landed=True, body=False),
        _match("beta", landed=False),
        _match("retired-offer", landed=False),
        _embedded("alpha", "prefetch", 600),
        _embedded("gamma", "autonomy_task", 9000),
    ])
    db = _workers_db(tmp_path / "workers.db", [
        ("r1", "scheduled-task", _iso(1), 3600.0),
        ("r2", "scheduled-task", _iso(2), 5400.0),
        ("r3", "autocode", _iso(3), 7200.0),
        ("r4", "queue-maintenance", _iso(1), 600.0),      # not a producer
        ("r5", "autocode", _iso(400), 86400.0),           # outside the window
    ])
    return {"skills": skills, "events": events, "db": db, "tmp": tmp_path}


def _argv(case: dict, *extra: str) -> list[str]:
    return ["--skills-root", str(case["skills"]),
            "--event-root", str(case["events"]),
            "--workers-db", str(case["db"]), *extra]


def _census(case: dict) -> dict:
    return fc.build_census(30, event_root=case["events"], workers_db=case["db"],
                           skills_roots=[case["skills"]])


def _stages(census: dict) -> dict[str, dict]:
    return {s["stage"]: s for s in census["stages"]}


# ── clause 1: one dated artifact under DATA_ROOT, >=3 Wilson stages, cannot_see ─

def test_clause_1_writes_one_dated_artifact_under_the_data_root(tmp_path, monkeypatch,
                                                                corpus, capsys):
    """Clause 1: running the module writes ONE dated JSON artifact under
    `app.paths.DATA_ROOT` and nothing under the checkout."""
    data_root = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA_ROOT", data_root)
    assert fc.main(_argv(corpus)) == 0

    written = sorted(p for p in data_root.rglob("*") if p.is_file())
    assert len(written) == 1, f"expected one artifact, got {[str(p) for p in written]}"
    path = written[0]
    assert path.parent == (data_root / "eval" / "funnel").resolve()
    assert re.fullmatch(r"funnel-\d{4}-\d{2}-\d{2}\.json", path.name)
    census = json.loads(path.read_text())
    assert census["artifact"] == "funnel-census" and census["item"] == "#2507"
    assert census["artifact_class"] == "SKILL.md"
    assert Path(census["artifact_path"]) == path
    assert f"artifact: {path}" in capsys.readouterr().out


def test_clause_1_three_stages_are_k_over_n_with_a_wilson_interval(corpus):
    """Clause 1: >=3 stages, each `k/n`, each interval from
    `eval.stats.wilson_ci`, over a window the report prints."""
    census = _census(corpus)
    measured = [s for s in census["stages"] if s["measured"]]
    assert len(measured) >= 3, [s["stage"] for s in measured]
    for st in measured:
        lo, hi = wilson_ci(st["k"], st["n"])
        assert st["ci95"] == [round(lo, 6), round(hi, 6)], st["stage"]
        assert st["rate"] == round(st["k"] / st["n"], 6)
        assert st["numerator"] and st["denominator"]

    by_name = _stages(census)
    assert (by_name["produced"]["k"], by_name["produced"]["n"]) == (3, 4)
    assert (by_name["offered"]["k"], by_name["offered"]["n"]) == (2, 3)
    assert (by_name["surfaced"]["k"], by_name["surfaced"]["n"]) == (2, 4)

    w = census["window"]
    assert w["days_asked"] == 30 and w["since"] < w["until"]
    text = fc.render_text(census)
    assert f"window: asked 30d, {w['since']} -> {w['until']}" in text


def test_clause_1_cannot_see_names_the_used_surrogate_and_the_archive_policy(corpus):
    """Clause 1: a `cannot_see` section naming the used stage's surrogate
    predicate WITH its blind spot, and the retention-sweep archive policy that
    bounds the window's survivorship."""
    see = _census(corpus)["cannot_see"]
    surrogate = see["used_surrogate"]
    assert "tool call" in surrogate["predicate_v1_considered"]
    assert "sessions/*.json" in surrogate["predicate_v1_considered"]
    assert "#630" in surrogate["blind_spot"]
    assert "never implies" in surrogate["blind_spot"]

    policy = see["retention_policy_bounding_the_window"]
    assert "90 days" in policy["sessions_store"]
    assert "SESSION_ARCHIVE_AGE_DAYS" in policy["sessions_store"]
    assert "survivorship" in policy
    assert "correct_but_never_applicable" in see


# ── clause 1's other half: nothing lands under ~/lloyd ────────────────────────

def test_clause_1_refuses_to_write_inside_the_checkout(tmp_path):
    """Clause 1: the artifact lands only under the data root (#2378's ruling on
    in-tree runtime data), enforced by the writer rather than by convention."""
    inside = ROOT / "eval" / "funnel"
    with pytest.raises(fc.CensusWriteRefused):
        fc.resolve_out_dir(str(inside))

    assert fc.main(["--out-dir", str(ROOT / "eval" / "funnel")]) == 2
    assert not inside.exists(), "the refusal still created a directory in the tree"

    assert fc.resolve_out_dir(str(tmp_path / "funnel")) == (tmp_path / "funnel").resolve()
    assert fc.resolve_out_dir(None, data_root=tmp_path / "d") == (
        tmp_path / "d" / "eval" / "funnel").resolve()


# ── clause 2: an unmeasurable denominator is a named refusal, never a 0 ───────

def test_clause_2_an_empty_denominator_is_a_named_refusal_not_a_rate(tmp_path):
    """Clause 2: a stage whose denominator is 0 carries a named refusal with its
    denominator instead of a rate, and no stage renders `0` for a denominator it
    never had."""
    skills = tmp_path / "skills"
    _skill(skills, "alpha")
    census = fc.build_census(30, event_root=_empty_events(tmp_path),
                            workers_db=_workers_db(tmp_path / "w.db", []),
                            skills_roots=[skills])
    by_name = _stages(census)
    surfaced = by_name["surfaced"]
    assert surfaced["measured"] is False
    assert surfaced["refusal"] == "no_offer_rows_in_window"
    assert surfaced["denominator"] == 0
    for banned in ("k", "n", "rate", "ci95"):
        assert banned not in surfaced, f"{banned} would read as a measured zero"

    benefited = by_name["benefited"]
    assert benefited["measured"] is False and benefited["denominator"] == 0
    assert benefited["refusal"] in ("no_uptake_table", "uptake_table_not_read_in_v1")
    used = by_name["used"]
    assert used["measured"] is False and "denominator" in used

    text = fc.render_text(census)
    assert "REFUSAL no_offer_rows_in_window (denominator 0)" in text
    assert not re.search(r"stage \w+: 0/0", text), text


def test_clause_2_a_measured_zero_is_still_a_rate(corpus):
    """The other side of clause 2: `k = 0` over a real denominator IS printed, as
    a rate with an interval — refusing that would erase a real finding."""
    events = corpus["tmp"] / "never" / "event_logs"
    _events(events, "s", [_match("alpha", landed=False),
                          _match("beta", landed=False)])
    census = fc.build_census(30, event_root=events, workers_db=corpus["db"],
                            skills_roots=[corpus["skills"]])
    surfaced = _stages(census)["surfaced"]
    assert surfaced["measured"] is True
    assert (surfaced["k"], surfaced["n"], surfaced["rate"]) == (0, 2, 0.0)
    assert surfaced["ci95"] == [round(w, 6) for w in wilson_ci(0, 2)]
    assert surfaced["ci95"][1] > 0, "Wilson must not report a perfect miss as a certainty"


# ── clause 3: the surfaced leg reads the events that actually exist ───────────

def test_clause_3_surfaced_comes_through_skill_injection_counts(corpus, monkeypatch):
    """Clause 3: the surfaced stage is computed over `prefetch.skill_match`
    (`landed`, `injected_body`) and `skill.embedded` THROUGH
    `app.skill_telemetry.skill_injection_counts`, not `app.event_log.read_events`,
    and publishes offers-that-never-landed as its own loss share."""
    import app.event_log as event_log

    def _no_ui_reader(*a, **k):            # the 200-row page must never be used
        raise AssertionError("the census read the paginated UI event reader")

    monkeypatch.setattr(event_log, "read_events", _no_ui_reader)
    calls: list[tuple] = []
    real = fc.skill_injection_counts

    def _spy(root, days):
        calls.append((Path(root), days))
        return real(root, days)

    monkeypatch.setattr(fc, "skill_injection_counts", _spy)
    census = _census(corpus)
    assert calls == [(corpus["events"], 30)], "the surfaced leg bypassed the reader"

    surfaced = _stages(census)["surfaced"]
    assert (surfaced["k"], surfaced["n"]) == (2, 4)
    loss = census["loss_shares"]["surfaced"]
    assert (loss["k"], loss["n"]) == (2, 4), "offers that never landed, as its own share"
    assert loss["label"] == "offers that never landed"
    assert loss["ci95"] == [round(w, 6) for w in wilson_ci(2, 4)]

    rows = census["surface_detail"]["embed_rows"]
    assert rows["landed_rows"] == {"with_injected_body": 1,
                                   "without_injected_body": 1}
    assert rows["rows_by_route"] == {"prefetch": 1, "autonomy_task": 1}
    routes = census["surface_detail"]["embed_routes"]
    assert routes["in_cohort"] == {"prefetch": 1}
    assert routes["excluded_named"] == {"autonomy_task": 1}
    assert routes["unclassified"] == {}


def test_clause_3_the_cohort_intersection_keeps_out_what_the_window_cannot_see(corpus):
    """An artifact that only ever reaches a prompt on the uncapped autonomy route
    is counted, named and EXCLUDED: the per-turn `used` predicate this census
    declines could never see it, and dividing by it anyway would fake a funnel
    leg. Whether v2 includes it is a ruling on #2507, not a decision here."""
    census = _census(corpus)
    detail = census["surface_detail"]
    assert "gamma" in detail["out_of_cohort_artifacts"]
    assert "gamma" not in detail["artifacts_reaching_surfaced"]
    assert detail["artifacts_reaching_surfaced"] == ["alpha"]
    assert "retired-offer" in census["stages"][0]["not_in_circulation"]
    assert "retired-offer" in census["stages"][1]["offered_not_in_cohort"]
    # An offered name outside the cohort may never inflate the offered numerator.
    offered = _stages(census)["offered"]
    assert (offered["k"], offered["n"]) == (2, 3)


# ── clause 4: one line, one stage ─────────────────────────────────────────────

def test_clause_4_the_report_prints_one_dominant_loss_stage_line(corpus):
    """Clause 4: the report prints one line
    `dominant loss stage: <name> (k/n)` naming exactly one stage and no other
    candidate."""
    census = _census(corpus)
    text = fc.render_text(census)
    lines = [line for line in text.splitlines()
             if line.startswith("dominant loss stage:")]
    assert len(lines) == 1, text
    m = re.fullmatch(r"dominant loss stage: ([a-z_]+) \((\d+)/(\d+)\)", lines[0])
    assert m, f"the line is not the shape the acceptance names: {lines[0]!r}"
    stage, k, n = m.group(1), int(m.group(2)), int(m.group(3))
    assert stage == "surfaced"
    loss = census["loss_shares"][stage]
    assert (k, n) == (loss["k"], loss["n"])
    for other in fc.FUNNEL_ORDER:
        if other != stage:
            assert other not in lines[0], f"{other} is a runner-up on the line"
    assert census["dominant_loss_stage"]["stage"] == stage


def test_clause_4_no_dominant_stage_is_claimed_when_nothing_was_measured(tmp_path):
    """With no denominator anywhere the line refuses itself rather than naming a
    winner the data cannot support."""
    census = fc.build_census(30, event_root=_empty_events(tmp_path),
                            workers_db=_workers_db(tmp_path / "w.db", []),
                            skills_roots=[tmp_path / "no_skills"])
    text = fc.render_text(census)
    assert text.count("dominant loss stage:") == 1
    assert "dominant loss stage: UNMEASURED (no stage had a denominator)" in text
    assert census["dominant_loss_stage"] is None


# ── clause 5: the priced line ─────────────────────────────────────────────────

def test_clause_5_the_cost_line_carries_its_query_sources_and_no_dollar_figure(corpus):
    """Clause 5: the cost line embeds the SQL it executed, names its source list
    (including `scheduled-task`) and its window, states GPU-hours and the
    reached-use count, and contains no dollar figure."""
    census = _census(corpus)
    cost = census["cost"]
    assert "scheduled-task" in cost["sources"]
    assert "'scheduled-task'" in cost["sql"]
    assert cost["sources"] == list(fc.ARTIFACT_PRODUCING_SOURCES)
    assert cost["window_start"] == census["window"]["since"]

    # The printed query reproduces the printed numbers, from the printed string.
    con = sqlite3.connect(corpus["db"])
    rows = list(con.execute(cost["sql"]))
    con.close()
    assert [{"source": r[0], "runs": int(r[1]), "gpu_hours": round(float(r[2]), 4)}
            for r in rows] == cost["per_source"]
    assert cost["per_source"] == [
        {"source": "scheduled-task", "runs": 2, "gpu_hours": 2.5},
        {"source": "autocode", "runs": 1, "gpu_hours": 2.0}]
    assert cost["gpu_hours"] == 4.5 and cost["runs"] == 3
    assert "queue-maintenance" not in json.dumps(cost["per_source"])
    assert cost["store_span_rows"] == 5, "the store's own depth is printed too"

    assert cost["reached_use"]["count"] == 1
    assert cost["reached_use"]["taken_at_stage"] == "surfaced"
    assert "unmeasured" in cost["reached_use"]["why"]
    assert cost["gpu_hours_per_artifact_reaching_use"] == 4.5

    blob = json.dumps(census) + fc.render_text(census)
    assert "$" not in blob, "the census printed a dollar figure"
    assert "GPU-hours" in blob
    out = fc.render_text(census)
    assert f"cost sql: {cost['sql']}" in out
    assert "reached use: 1 artifacts" in out


def test_clause_5_an_empty_cost_denominator_refuses_rather_than_printing_zero(tmp_path):
    """No hours and no reached-use artifacts is a refusal, not a price of 0 — the
    rule clause 2 states for the stages applies to the ratio as well."""
    census = fc.build_census(30, event_root=_empty_events(tmp_path),
                            workers_db=_workers_db(tmp_path / "w.db", []),
                            skills_roots=[tmp_path / "no_skills"])
    cost = census["cost"]
    assert cost["gpu_hours"] == 0.0
    assert cost["reached_use"]["count"] == 0
    assert cost["gpu_hours_per_artifact_reaching_use"] is None
    assert "ratio_refusal" in cost
    assert "price: REFUSAL" in fc.render_text(census)
