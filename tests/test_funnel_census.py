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


# ── the uptake store the `benefited` stage reads (#2733) ─────────────────────

def _classifier_block(*, rerouted: bool = False,
                      measured: bool = True) -> dict:
    """A classifier block in the shape `scripts/uptake_probe.py` hands
    `write_table`: engine keys at block level, `measured` only inside sub-blocks.

    `stamp_engine` is the probe's own stamper, called with provenance supplied so
    it never resolves an endpoint, and the block is built with the same keys the
    probe builds — so a fixture stamped `engine_rerouted: true` here is rerouted
    in exactly the way a real table is, including the cascade that sets every
    sub-block's `measured` to false beside it.
    """
    import scripts.uptake_probe as probe
    from app import uptake
    block = {
        "passed": not rerouted and measured,
        "metrics": {"measured": measured, "precision": 0.82, "recall": 0.60},
        "holdout": {"measured": measured, "precision": 0.79},
        "zero_shot": {"measured": measured, "precision": 0.74},
    }
    return probe.stamp_engine(block, provenance={
        "alias": "secondary", "endpoint": "http://127.0.0.1:9/v1/chat/completions",
        "resolved_model": ("other-engine" if rerouted
                           else uptake.EXPECTED_SCORING_ENGINE),
        "expected_engine": uptake.EXPECTED_SCORING_ENGINE,
        "expected_engine_reason": "fixture", "rerouted": rerouted})


def _uptake_table(dir_: Path, name: str, *, days_ago: float = 1.0,
                  classifier: dict | None = None,
                  audit_only: bool = False,
                  probe_timestamp: str | None = None) -> Path:
    """One real uptake table, written by the instrument's OWN writer.

    Hand-spelling the dict would grade the stage against a schema the probe never
    emits: `app.uptake.write_table` puts `audit_only` at table level,
    `engine_rerouted` inside `classifier` and `measured` only inside that block's
    sub-blocks, so a flat fixture would pass a reader that misses two of the
    three keys. `probe_timestamp` is set through `extra` to decouple the AGE from
    the filename (`write_table` names the file off `date` and stamps the
    timestamp itself), which is how a >30-day table is built at all; an explicit
    stamp of `""` or junk is how the two unreadable-stamp cases are.
    """
    from app import uptake
    extra: dict = {"probe_timestamp": (
        probe_timestamp if probe_timestamp is not None else
        (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(
            timespec="seconds"))}
    if audit_only:
        extra["audit_only"] = True
        extra["audit_only_reason"] = "fixture: emitted under --ignore-precision-floor"
    return uptake.write_table({"corpus": {"turns": 40, "disputes": 3},
                               "coverage": {}, "entries": []},
                              out_dir=dir_, date=name[len("uptake-"):-len(".json")],
                              classifier=classifier, extra=extra)


@pytest.fixture(autouse=True)
def _uptake_store_repointed(tmp_path: Path, monkeypatch) -> Path:
    """Every test in this file reads a fixture uptake store, never the real one.

    The three committed tables under `eval/uptake/` are dated 2026-09-11/-17/-20,
    so any test that read them would pin a number the calendar moves: the newest
    crossed `UPTAKE_TABLE_MAX_AGE_DAYS` on 2026-10-20 and would have flipped this
    file's expectations from a fresh table to a stale one all by itself. The
    writer's constant is repointed for the whole module; the one test that exists
    to prove THAT constant is the route repoints it again to its own fixture.
    """
    import app.uptake as uptake
    d = tmp_path / "uptake_store"
    _uptake_table(d, "uptake-2026-09-20.json", days_ago=1.0)
    monkeypatch.setattr(uptake, "UPTAKE_ARTIFACT_DIR", d)
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


# ── #2733: the benefited stage reads the store WHERE ITS WRITER DECLARES IT ───
#
# Until #2733 this stage defaulted to `app.paths.DATA_ROOT / "eval" / "uptake"`,
# a path nothing ever wrote, while `app/uptake.py`'s own `write_table` outputs to
# `UPTAKE_ARTIFACT_DIR` — inside the code tree. The false absence was the only
# outcome reachable from `build_census`, which passed no directory at all. These
# five clauses are the item's acceptance; the leg still never produces a rate,
# because the join gap is a separate gap from the path.

def test_2733_clause_1_the_store_resolves_through_the_writer_s_own_constant(
        tmp_path, monkeypatch):
    """Clause 1: repointing `app.uptake.UPTAKE_ARTIFACT_DIR` moves the stage, so
    the store is resolved through the writer's constant and not a re-spelled
    data-root path. `DATA_ROOT` is pinned to a directory that does not exist, so
    the pre-#2733 default would have died on `no_uptake_table` naming it."""
    import app.uptake as uptake
    fixture = tmp_path / "writer_declared_store"
    _uptake_table(fixture, "uptake-2026-10-10.json", days_ago=2.0)
    monkeypatch.setattr(uptake, "UPTAKE_ARTIFACT_DIR", fixture)
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "no_data_root_here")

    out = fc.benefited_stage()

    assert out["uptake_store"] == str(fixture)
    assert out["refusal"] != "no_uptake_table", out["detail"]
    assert str(tmp_path / "no_data_root_here") not in json.dumps(out)
    assert out["uptake_table"] == "uptake-2026-10-10.json"


def test_2733_clause_2_the_uptake_dir_flag_selects_the_store_end_to_end(
        tmp_path, monkeypatch, corpus):
    """Clause 2, the CLI leg: `--uptake-dir DIR` is what the stage reads once the
    bytes are on disk, not just what `main` parsed — and no store path is
    re-spelled from the data root on the way."""
    chosen = tmp_path / "chosen_store"
    _uptake_table(chosen, "uptake-2026-10-05.json", days_ago=1.0)
    data_root = tmp_path / "data_root"
    monkeypatch.setattr(paths, "DATA_ROOT", data_root)

    assert fc.main(["--skills-root", str(corpus["skills"]),
                    "--event-root", str(corpus["events"]),
                    "--workers-db", str(corpus["db"]),
                    "--uptake-dir", str(chosen)]) == 0
    written = sorted((data_root / "eval" / "funnel").glob("funnel-*.json"))
    assert len(written) == 1, f"one artifact expected, got {[str(p) for p in written]}"
    stage = _stages(json.loads(written[0].read_text()))["benefited"]
    assert stage["uptake_store"] == str(chosen)
    assert stage["uptake_table"] == "uptake-2026-10-05.json"
    assert stage["refusal"] == "uptake_table_not_read_in_v1"


def test_2733_clause_2_the_default_is_the_writer_s_constant_and_the_flag_is_threaded(
        tmp_path, monkeypatch, corpus):
    """Clause 2, the two seams behind the flag: the CLI default resolves to
    `app.uptake.UPTAKE_ARTIFACT_DIR` — proved by repointing that constant and
    watching the stage follow it with no flag given — and `build_census` hands
    `uptake_dir` down to the stage rather than dropping it."""
    import app.uptake as uptake
    chosen = tmp_path / "chosen_store"
    _uptake_table(chosen, "uptake-2026-10-05.json", days_ago=1.0)
    monkeypatch.setattr(uptake, "UPTAKE_ARTIFACT_DIR", chosen)
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "no_data_root_here")

    seen: list[dict] = []
    real = fc.benefited_stage

    def _spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(fc, "benefited_stage", _spy)
    argv = ["--skills-root", str(corpus["skills"]),
            "--event-root", str(corpus["events"]),
            "--workers-db", str(corpus["db"])]
    assert fc.main(argv) == 0
    assert len(seen) == 1, f"the stage ran {len(seen)} times, expected exactly once"
    assert seen[0]["uptake_dir"] is None, "the CLI default must not invent a path"

    default = fc.benefited_stage(seen[0]["uptake_dir"])
    assert default["uptake_store"] == str(uptake.UPTAKE_ARTIFACT_DIR)
    assert default["uptake_store"] != str(paths.DATA_ROOT / "eval" / "uptake")

    seen.clear()
    fc.build_census(30, event_root=corpus["events"], workers_db=corpus["db"],
                    skills_roots=[corpus["skills"]], uptake_dir=chosen)
    assert len(seen) == 1, "build_census did not reach the benefited stage"
    assert seen[0]["uptake_dir"] == chosen, "build_census dropped the store"


def test_2733_clause_3_the_newest_table_is_named_with_its_age_and_key_states(
        tmp_path):
    """Clause 3: the store's newest table is named by FILENAME, aged on its own
    `probe_timestamp`, and every citability key is reported in the state the FILE
    states — where `absent` means the FILE omits it at EVERY depth it writes to,
    proved by the same table shape carrying the keys one level down."""
    store = tmp_path / "uptake"
    _uptake_table(store, "uptake-2026-09-01.json", days_ago=1.0)
    _uptake_table(store, "uptake-2026-10-10.json", days_ago=4.0)

    out = fc.benefited_stage(store)

    assert out["uptake_table"] == "uptake-2026-10-10.json"
    assert out["uptake_table_age_days"] == 4, "aged on probe_timestamp, not mtime"
    assert out["uptake_citability"] == {"engine_rerouted": "absent",
                                        "measured": "absent",
                                        "audit_only": "absent"}
    assert "uptake-2026-10-10.json" in out["detail"]
    assert "uptake-2026-09-01.json" not in json.dumps(out)
    for key in ("engine_rerouted", "measured", "audit_only"):
        assert f"{key}=absent" in out["detail"]
    assert "citable" not in json.dumps(out["uptake_citability"])

    # Positive control: the keys the writer puts BELOW the top level —
    # `engine_rerouted` in `classifier`, `measured` in its sub-blocks — are read
    # there, not missed. Without this leg, `absent` above is ambiguous between
    # "the file omits the key" and "this reader never looks where it lives".
    deep = tmp_path / "deep_store"
    _uptake_table(deep, "uptake-2026-10-10.json", days_ago=1.0,
                  classifier=_classifier_block(rerouted=True))
    stated = fc.benefited_stage(deep)["uptake_citability"]
    assert stated["engine_rerouted"] == "true", stated
    assert stated["measured"] == "false", (
        f"a rerouted stamp also sets every sub-block to measured=false; the "
        f"reader must see that too, got {stated}")
    assert stated["audit_only"] == "absent", stated


def test_2733_clause_4_a_stale_or_uncitable_table_refuses_by_name(tmp_path):
    """Clause 4: a table past `UPTAKE_TABLE_MAX_AGE_DAYS`, or stamped
    `engine_rerouted: true` / `audit_only: true` / `measured: false`, or with no
    readable `probe_timestamp`, is a NAMED refusal carrying its denominator and no
    rate — never a 0.0 that would read as a measured zero. An absent key is not
    the same thing as a cleared one, so an unstamped table is held for the join
    gap rather than refused on citability, and an undated table is refused because
    its age is unknowable, not because it is old."""
    stale = tmp_path / "stale"
    _uptake_table(stale, "uptake-2026-06-01.json",
                  days_ago=fc.UPTAKE_TABLE_MAX_AGE_DAYS + 5)
    out = fc.benefited_stage(stale)
    assert out["refusal"] == "uptake_table_uncitable"
    assert out["denominator"] == 0
    assert "uptake-2026-06-01.json" in out["detail"]

    rerouted = tmp_path / "rerouted"
    _uptake_table(rerouted, "uptake-2026-10-10.json", days_ago=2.0,
                  classifier=_classifier_block(rerouted=True))
    assert fc.benefited_stage(rerouted)["refusal"] == "uptake_table_uncitable"

    audit = tmp_path / "audit"
    _uptake_table(audit, "uptake-2026-10-10.json", days_ago=2.0, audit_only=True,
                  classifier=_classifier_block())
    assert fc.benefited_stage(audit)["refusal"] == "uptake_table_uncitable"

    measured_false = tmp_path / "unmeasured"
    _uptake_table(measured_false, "uptake-2026-10-10.json", days_ago=2.0,
                  classifier=_classifier_block(measured=False))
    assert (fc.benefited_stage(measured_false)["refusal"]
            == "uptake_table_uncitable")

    # A table whose classifier ran but stamped no `engine_rerouted` at all — the
    # pre-#1850 shape — must not be read as cleared, and must not refuse either:
    # the stage's own reason for holding it is the join gap, named separately.
    absent_keys = tmp_path / "absent_keys"
    _uptake_table(absent_keys, "uptake-2026-09-20.json", days_ago=2.0,
                  classifier={"passed": True,
                              "metrics": {"precision": 0.8, "recall": 0.6}})
    held = fc.benefited_stage(absent_keys)
    assert held["refusal"] == "uptake_table_not_read_in_v1", "absent keys must not refuse"
    assert held["uptake_citability"]["engine_rerouted"] == "absent", held

    undated = tmp_path / "undated"
    _uptake_table(undated, "uptake-2026-10-10.json", probe_timestamp="not-a-date")
    assert fc.benefited_stage(undated)["refusal"] == "uptake_table_uncitable"

    unstamped = tmp_path / "unstamped"
    _uptake_table(unstamped, "uptake-2026-10-10.json")
    (unstamped / "uptake-2026-10-10.json").write_text(
        json.dumps({"entries": [], "corpus": {"turns": 40}}), encoding="utf-8")
    assert fc.benefited_stage(unstamped)["refusal"] == "uptake_table_uncitable"

    for case in (stale, rerouted, audit, measured_false, undated, unstamped):
        stage = fc.benefited_stage(case)
        assert stage["measured"] is False
        assert stage["denominator"] == 0
        for banned in ("k", "n", "rate", "ci95"):
            assert banned not in stage, f"{banned} would read as a measured zero"


def test_2733_clause_5_an_empty_store_still_refuses_by_name_and_no_branch_rates(
        tmp_path):
    """Clause 5: a store directory that exists but holds no table is still the
    named `no_uptake_table` refusal naming THAT directory, and `measured: false`
    with `denominator: 0` survives every one of the stage's five outcomes."""
    empty = tmp_path / "empty_store"
    empty.mkdir()
    out = fc.benefited_stage(empty)
    assert out["refusal"] == "no_uptake_table"
    assert str(empty) in out["detail"]
    assert out["uptake_store"] == str(empty)

    missing = tmp_path / "never_created"
    assert fc.benefited_stage(missing)["refusal"] == "no_uptake_table"

    outcomes = [fc.benefited_stage(empty), fc.benefited_stage(missing)]
    stale = tmp_path / "stale2"
    _uptake_table(stale, "uptake-2026-01-01.json",
                  days_ago=fc.UPTAKE_TABLE_MAX_AGE_DAYS + 1)
    outcomes.append(fc.benefited_stage(stale))
    unreadable_dir = tmp_path / "broken_store"
    _ = _uptake_table(unreadable_dir, "uptake-2026-10-10.json", days_ago=1.0)
    (unreadable_dir / "uptake-2026-10-10.json").write_text("{not json",
                                                           encoding="utf-8")
    outcomes.append(fc.benefited_stage(unreadable_dir))
    fine = tmp_path / "fine_store"
    _uptake_table(fine, "uptake-2026-10-10.json", days_ago=1.0)
    outcomes.append(fc.benefited_stage(fine))

    assert [o["refusal"] for o in outcomes] == [
        "no_uptake_table", "no_uptake_table", "uptake_table_uncitable",
        "uptake_table_unreadable", "uptake_table_not_read_in_v1"]
    for stage in outcomes:
        assert stage["measured"] is False
        assert stage["denominator"] == 0
        assert stage["stage"] == "benefited"
        for banned in ("k", "n", "rate", "ci95"):
            assert banned not in stage, f"{banned} would read as a measured zero"
        assert stage["unit"] == "artifact"


# ── #2732: the two shares the census computed and never printed ───────────────
#
# Both numbers were already in the artifact and reachable from no report. Measured
# over the census's own 30-day window (`build_census(30)` against the real event
# store, 2026-10-15T04:17Z): the excerpt-only split inside `surfaced` is 3,141 of
# 6,297 landed rows — 49.9% — and the out-of-cohort embed share is 734 of 7,042
# embed rows — 10.4%. The counts drift by the minute and the shares do not: the
# ruling that started this cites 3,131 of 6,277, the triage measured 3,138 of 6,291
# two hours earlier, and all three are 49.9%.
#
# #2507 ruled both questions at 2026-10-15T02:47:52 (its #2 and #4 entries, the
# latter naming this item as its follow-up) and the rulings were never written back
# into the module, so every reader met a note saying the question was still
# undecided, and the owed-check job went on re-asking it.
#
# What the five clauses below hold at once: the lines print, they print the numbers
# the published dict holds, and neither one becomes a stage or a dominant-loss
# candidate.

def _lines(text: str, prefix: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(prefix)]


def test_the_excerpt_only_share_prints_from_the_published_dict(corpus):
    """Clause 1: one `surfaced excerpt-only: k/n` line, taken from the census dict.

    The fixture's two landed rows are one body render and one excerpt render, so
    the honest share is 1/2 — and it must be read out of
    `surface_detail.embed_rows.landed_rows`, not recomputed in the renderer: a k
    written twice is two numbers, and the JSON artifact is the one a later report
    parses.
    """
    census = _census(corpus)
    landed = census["surface_detail"]["embed_rows"]["landed_rows"]
    lines = _lines(fc.render_text(census), "surfaced excerpt-only: ")
    assert len(lines) == 1, f"exactly one such line, got {lines}"

    k, n = landed["without_injected_body"], (landed["with_injected_body"]
                                             + landed["without_injected_body"])
    assert f": {k}/{n} " in lines[0], f"{lines[0]!r} != {k}/{n} from {landed}"
    # The line says what it is a share OF, since the denominator is not the
    # funnel's and a bare k/n would read as if it were.
    assert "landed rows" in lines[0], lines[0]


def test_the_line_follows_the_dict_not_a_recount(tmp_path, monkeypatch):
    """The same claim from the other side: mutate the dict, and the printed share
    moves with it. A renderer that recounted the events would keep printing the
    old split and the artifact and the report would disagree silently."""
    census = fc.build_census(30, event_root=_empty_events(tmp_path),
                             workers_db=_workers_db(tmp_path / "w.db", []),
                             skills_roots=[tmp_path / "no_skills"])
    landed = census["surface_detail"]["embed_rows"]["landed_rows"]
    landed.update({"with_injected_body": 5, "without_injected_body": 7})
    line = _lines(fc.render_text(census), "surfaced excerpt-only: ")[0]
    assert ": 7/12 " in line, line
    # And it never silently divides by zero: an empty split is a named refusal,
    # which is clause 2's rule for the stages applied to a share.
    landed.update({"with_injected_body": 0, "without_injected_body": 0})
    line = _lines(fc.render_text(census), "surfaced excerpt-only: ")[0]
    assert "REFUSAL" in line and "0/0" not in line, line


def test_the_out_of_cohort_share_prints_as_the_cohort_boundary(corpus):
    """Clause 2: the embed-route exclusion prints as k/n and says it is a cohort
    boundary, not a funnel stage.

    The fixture's two `skill.embedded` rows are one chat-route row (`alpha` on
    `prefetch`) and one out-of-cohort row (`gamma` on `autonomy_task`), so the
    share is 1/2 — small on purpose: what is under test is which side of the
    boundary a row lands on, not the live magnitude, which is 734 of 7,036 rows
    over the real 30-day window (measured 2026-10-15). Summing over
    `OUT_OF_COHORT_EMBED_ROUTES` rather than subtracting the chat routes from the
    total is the point: an unrecognised route is neither in nor out, and the line
    must not be able to claim it silently.
    """
    census = _census(corpus)
    per_route = census["surface_detail"]["embed_rows"]["rows_by_route"]
    k = sum(per_route.get(r, 0) for r in fc.OUT_OF_COHORT_EMBED_ROUTES)
    n = sum(per_route.values())
    lines = _lines(fc.render_text(census), "out-of-cohort embed rows: ")
    assert len(lines) == 1, f"exactly one such line, got {lines}"
    assert f": {k}/{n} " in lines[0], f"{lines[0]!r} != {k}/{n} from {per_route}"
    assert (k, n) == (1, 2), f"the fixture's cohort split moved: {per_route}"
    assert "cohort boundary" in lines[0] and "not a funnel stage" in lines[0]
    assert "stage " not in lines[0].split(" - ")[0], (
        f"the label reads as a stage: {lines[0]!r}")


def test_the_two_new_lines_add_no_stage_and_no_candidate(corpus):
    """Clause 3: printing a share must not change a single number the report
    already printed — the arithmetic of the funnel is the artifact's whole point.

    Checked against the derivation, not a remembered string: `stages[]` is still
    the five funnel legs, `loss_shares` still has the keys the measured stages
    give it, and the one `dominant loss stage:` line still names what
    `dominant_loss(loss_shares(stages))` returns with its own k/n. The two new
    lines appear in none of those three.
    """
    census = _census(corpus)
    stages = census["stages"]
    assert [s["stage"] for s in stages] == list(fc.FUNNEL_ORDER), (
        "a new stage entered the funnel: a printed share is not a leg")
    assert len(census["loss_shares"]) == sum(
        1 for s in stages if s.get("measured") and s.get("loss")), (
        "the keys of loss_shares are no longer the measured stages' losses, so "
        "`dominant_loss` is being fed a different set than the stages imply")

    text = fc.render_text(census)
    # Both counts read the artifact, never a remembered number: `FUNNEL_ORDER` is
    # the order `stages[]` is built in and `loss_shares` is keyed by stage, so a
    # printed line the census does not carry, or a census loss the report dropped,
    # fails here — and neither count depends on how many lines the renderer happens
    # to add.
    assert [ln.split(" ")[1].rstrip(":") for ln in _lines(text, "stage ")] == \
        [s["stage"] for s in stages] == list(fc.FUNNEL_ORDER), text
    assert [ln.split(" ")[2].split(":")[0] for ln in _lines(text, "loss share ")] \
        == list(census["loss_shares"]), text
    for label in ("surfaced excerpt-only", "out-of-cohort embed rows"):
        assert label not in census["loss_shares"], f"{label} became a loss-share key"
        assert label not in census["stages"][0], f"{label} entered a stage dict"
        assert not any(label in str(v) for v in census["loss_shares"].values())

    dom = _lines(text, "dominant loss stage:")
    assert len(dom) == 1, text
    expected = fc.dominant_loss(fc.loss_shares(stages))
    assert expected == census["dominant_loss_stage"], (
        "the printed stage is no longer what the derivation returns")
    assert expected["stage"] == "surfaced", expected
    assert dom[0] == (f"dominant loss stage: {expected['stage']} "
                      f"({expected['k']}/{expected['n']})"), dom[0]


def test_no_cannot_see_value_still_calls_a_ruling_open(corpus):
    """Clause 4: the ruled texts read as ruled.

    `cannot_see` is the section a reader reaches for on a number they do not
    trust, so an entry that ended "an open ruling on #2507" was the census asking
    a question the owed-check job had settled at 2026-10-15T02:47:52 — and the next
    reader, finding it unanswered, re-opens the whole taxonomy. Nothing in the
    section may say that now: the two ruled entries name the decision and the
    mechanism behind it.
    """
    see = _census(corpus)["cannot_see"]
    assert isinstance(see, dict) and see, "cannot_see vanished"
    assert not [k for k, v in see.items() if "open ruling" in json.dumps(v)], (
        "a settled ruling is still advertised as open")

    excerpt = see["landed_without_injected_body"]
    assert "render mode" in excerpt and "not a stage" in excerpt, excerpt
    assert "app/prefetch.py:733-734" in excerpt, (
        "the ruling has to name the mechanism that settles it: both flags come "
        "from one `mode`, not from two measurements")

    classes = see["artifact_classes_not_censused"]
    assert "SKILL.md-only" in classes, classes
    assert "no per-turn predicate" in classes, classes
    assert "eval/run_eval.py" in classes, (
        "the reason has to name the instrument that already covers those classes")


def test_the_cohort_text_reads_as_ruled_without_taking_2733_s_text():
    """Clause 5: the three cohort notes are decided prose, and the only `open
    ruling` strings left in the module are #2733's.

    The grep this item was filed with — `open ruling on #2507` returning 0 hits —
    is not the check, and it was never achievable by this item alone. At this
    round's base the module carried the phrase on 7 lines while that literal grep
    matched only 5 of them: one note sat mid-sentence across a line break in
    `_scan_embed_rows`'s docstring, and an eighth note (`pending cohort ruling` in
    the name-cap comment) used different wording entirely, so the old check could
    pass with stale text still in the file. Three of the 7 belong to
    `benefited_stage` — its docstring plus the two branches of its refusal — which
    #2733 owns in a round that is in flight, so they are left exactly as they are.

    The pin therefore is positional, not a line count: no `pending cohort` string
    anywhere, no `open ruling` string outside `benefited_stage`'s own span, and
    `benefited_stage`'s still present (which is what proves this round did not
    reach into #2733's text to make a grep look clean).
    """
    src = (ROOT / "scripts" / "funnel_census.py").read_text(encoding="utf-8")
    assert "pending cohort" not in src, "a ruled note still says pending"

    # The one remaining `open ruling` phrasing must be #2733's and nowhere else:
    # pinning it to the function's own source span rather than to a line count,
    # because a line count silently entitles any future line to one.
    start = src.index("def benefited_stage")
    end = src.index("\ndef ", start)
    body, outside = src[start:end], src[:start] + src[end:]
    assert "open ruling" in body, "#2733's text was edited by this round"
    assert "open ruling" not in outside, (
        "a settled ruling is still advertised as open outside benefited_stage")

    # The cohort constant's own comment states the boundary as a decision, with the
    # condition that would reopen it — that condition is what makes it a ruling
    # rather than a preference.
    lines_ = src.splitlines()
    at = next(i for i, ln in enumerate(lines_)
              if ln.startswith("OUT_OF_COHORT_EMBED_ROUTES = ("))
    top = at
    while lines_[top - 1].lstrip().startswith("#:"):
        top -= 1
    block = "\n".join(lines_[top:at])
    assert "chat-only UNTIL" in block and "per-run" in block, block
    assert "Whether they belong" not in block, block
    doc_start = src.index("def _scan_embed_rows")
    doc = src[doc_start:src.index("\ndef ", doc_start)]
    assert "RENDER MODE" in doc and "app/prefetch.py:733-734" in doc, doc
