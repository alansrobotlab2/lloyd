"""The cost-outlier trigger: the replay, the arming rule, and the recursion guard.

Three things are pinned here, one per clause of #2510:

* the **replay command** — what a human runs before trusting the trigger, so its
  output has to name the window it ACTUALLY held. `workers.db` has never held 30
  days on this box (`MIN(completed_at)` is 2026-09-22, ~18 days as this item was
  triaged), and `app/routers/workers.py:174` already documents that a per-source
  number printed under a 90-day label rests on whatever hours the store holds. A
  replay that printed "last 30 days" over an 18-day store would be the same
  mistake one layer up, so the held window is the headline, not a footnote;
* the **arming rule** — a source needs `min_samples` (40) completed runs in the
  window AND the run's tokens above `margin` (1.5x) that source's own window P90.
  Both bounds are load-bearing and measured, not guessed: at 1.0xP90 a hold-out
  replay over the live stores fires on 197 of 1,969 attributed runs (10.0%), which
  is ~34 diagnostic runs a day, because "above your own P90" is by construction
  one run in ten; at 2.0x it fires on 0, which makes the trigger dead. 1.5x is the
  measured middle (10 fires in 18 days) and the sample floor is what stops a
  12-run source firing on noise;
* the **recursion guard** — runs of the diagnostic source are outside the
  distribution that arms it. The talk this came from never addresses it: one
  expensive investigation would otherwise raise the very threshold that watches
  it, and the investigation's own cost is the thing most likely to be an outlier.

Tokens, never dollars. `cost_usd` is `0.0` on every row of the usage store —
`app/usage_store.py:1306-1308` says so in the file: "a per-skill cost would be a
column of zeros dressed up as a cost. Tokens are the only magnitude this store
actually carries." A `cost_usd > P90` trigger here has P90 = 0 and never fires,
which is the item's own kill switch tripping on the metric rather than the idea.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import cost_outlier as CO
from app import usage_store
from workers.queue import WorkQueue, new_run_id

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "scripts" / "maintenance" / "cost_replay.py"


# ---------------------------------------------------------------------------
# Fixture stores built through the real writers, not by hand-rolled SQL.
# A test that invented its own schema could pass against a table the shipped
# code never wrote.
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _split(tokens: int, turns: int):
    """`tokens` split over `turns` usage rows, exactly, as (in, out) per turn.

    Asserted on by the P90 figures below, so the arithmetic has to close: each
    turn's input+output sums to its share and the shares sum to `tokens`.
    """
    share, rem = divmod(tokens, turns)
    for i in range(turns):
        tot = share + (rem if i == 0 else 0)
        half = tot // 2
        yield i, tot - half, half


def _make_stores(tmp_path, monkeypatch):
    """A scratch workers.db + usage.db, both reached through the shipped writers."""
    wdb = tmp_path / "workers.db"
    udb = tmp_path / "usage.db"
    monkeypatch.setattr(usage_store, "DB_PATH", udb)
    return wdb, udb


def _record(queue, wdb, *, source, tokens, day_offset, status="success",
            turn_count=6):
    """One completed run whose sessions total `tokens`, plus its usage rows.

    `day_offset` counts back from now. The run row carries `session_ids` the way
    `workers/pool.py:1458` does it, because that key is the only join this whole
    mechanism has between a run and a cost: nothing in `runs` records tokens.
    """
    now = datetime.now(timezone.utc) - timedelta(days=day_offset)
    item_id = queue.enqueue(source, "k")
    queue.claim_next(worker_id="w1")
    queue.mark_running(item_id)
    sid = f"worker:{source}:{abs(hash((source, day_offset, tokens))) % 10**8:08d}"
    queue.record_run(
        run_id=new_run_id(source), queue_id=item_id, source=source, status=status,
        started_at=_iso(now - timedelta(seconds=90)), completed_at=_iso(now),
        duration_seconds=90.0, summary="ok",
        meta_json=json.dumps({"session_ids": [sid]}))
    # `turn_count` rows whose SUM is exactly `tokens`, because the unit is a run's
    # total and the item asks where the tokens WENT, which only exists per turn.
    # The split is exact on purpose: an inexact fixture would make every expected
    # P90 in this file off by whatever the rounding chose.
    for _i, inp, outp in _split(tokens, turn_count):
        usage_store.record_usage(session_id=sid, model="primary",
                                 input_tokens=inp, output_tokens=outp,
                                 num_turns=_i + 1)
    return sid


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Queue plus 44 `autocode` runs and 12 `bench-mine` runs, one at 3x its own P90.

    44 clears the 40-sample floor; 12 sits under it deliberately, so a source with
    a real outlier in a thin source is in the fixture and cannot be forgotten.
    """
    wdb, udb = _make_stores(tmp_path, monkeypatch)
    q = WorkQueue(wdb)
    for i in range(44):
        _record(q, wdb, source="autocode", tokens=100_000 + i, day_offset=i * 0.3)
    for i in range(12):
        _record(q, wdb, source="bench-mine", tokens=50_000 + i, day_offset=i * 0.4)
    return {"queue": q, "workers_db": wdb, "usage_db": udb}


def _outlier(queue, wdb, *, source="autocode", tokens=4_000_000):
    """Append one enormous run to `source` and return its run_id."""
    now = datetime.now(timezone.utc)
    item_id = queue.enqueue(source, "k")
    queue.claim_next(worker_id="w1")
    queue.mark_running(item_id)
    run_id = new_run_id(source)
    sid = f"worker:{source}:outlier{abs(tokens) % 10**6:06d}"
    queue.record_run(
        run_id=run_id, queue_id=item_id, source=source, status="success",
        started_at=_iso(now - timedelta(seconds=600)), completed_at=_iso(now),
        duration_seconds=600.0, summary="big",
        meta_json=json.dumps({"session_ids": [sid]}))
    for i, inp, outp in _split(tokens, 4):
        usage_store.record_usage(session_id=sid, model="primary",
                                 input_tokens=inp, output_tokens=outp,
                                 num_turns=i + 1)
    return run_id


# ---------------------------------------------------------------------------
# Clause 1 — the replay prints per-source count, token P90 and would-be fires,
# and names the window it HELD.
# ---------------------------------------------------------------------------

def _run_cli(*args, env_extra=None):
    import os
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(REPLAY), *args],
                          capture_output=True, text=True, cwd=ROOT, env=env)


def test_the_replay_reports_the_run_through_the_real_cli(stores, tmp_path):
    """Clause 1, across the process boundary.

    Every other node in this file calls the library; this one execs the script,
    because the argparse entry and the store-path flags are the part a caller
    actually touches. A test that only called `CO.replay()` in-process would pass
    with a `main()` that could not be invoked at all — the seam #2508's review
    named as untested on the sibling eval tool.
    """
    out = _run_cli("--workers-db", str(stores["workers_db"]),
                   "--usage-db", str(stores["usage_db"]), "--days", "30")
    assert out.returncode == 0, out.stderr[-1200:]
    assert "autocode" in out.stdout, out.stdout
    row = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("autocode")]
    assert len(row) == 1, out.stdout
    cells = row[0].split()
    # source, samples, p90 tokens, fires — each a real number off this fixture.
    assert cells[0] == "autocode", cells
    assert cells[1] == "44", f"sample count should be the 44 runs on disk: {cells}"
    p90 = int(cells[2].replace(",", ""))   # the table groups thousands
    assert 100_000 < p90 < 110_000, f"token P90 off a 100k..100k+43 fixture: {cells}"
    assert cells[3] == "0", f"no outlier was injected into this fixture: {cells}"


def test_the_replay_fires_on_the_run_above_margin_and_names_the_stores(stores):
    """The same readout has to be a real predictor, not a table of counts."""
    _outlier(stores["queue"], stores["workers_db"], tokens=4_000_000)
    rep = CO.replay(stores["workers_db"], stores["usage_db"], window_days=30.0)
    auto = rep["sources"]["autocode"]
    assert auto["samples"] == 45, auto
    assert auto["fires"] == 1, auto
    assert auto["armed"], "45 samples clears the 40 floor, so the source is armable"
    assert auto["p90_tokens"] and auto["p90_tokens"] > 100_000, auto


def test_the_replay_prints_the_window_it_held_not_the_window_asked(stores):
    """Clause 1's second half, and the part that cannot be fudged.

    The fixture's runs sit inside a couple of weeks, and the command is asked for
    30 days. The output must say which of the two it computed over: this box's
    `workers.db` has never held 30 days, so a replay that echoed `--days 30` in
    its own header would report a coverage the store does not have.
    """
    _outlier(stores["queue"], stores["workers_db"])
    out = _run_cli("--workers-db", str(stores["workers_db"]),
                   "--usage-db", str(stores["usage_db"]), "--days", "30")
    assert out.returncode == 0, out.stderr[-1200:]
    assert "held" in out.stdout, out.stdout
    assert "requested" in out.stdout, out.stdout
    rep = CO.replay(stores["workers_db"], stores["usage_db"], window_days=30.0)
    assert rep["window_days_requested"] == pytest.approx(30.0)
    assert 0.0 < rep["window_days_held"] < 30.0, rep
    held_line = [ln for ln in out.stdout.splitlines() if "held" in ln][0]
    assert f"{rep['window_days_held']:.1f}" in held_line, held_line
    assert "first_completed" in held_line or rep["first_completed_at"] in held_line, \
        held_line


def test_an_unattributed_run_is_counted_as_unattributed_and_never_as_zero(stores):
    """A run with no session in the usage store has no cost, which is not a cost of 0.

    Attribution is a join through `meta_json`, and triage measured 355 of the last
    400 successful runs matching. The remainder are not cheap runs; they are
    unknown. Read as 0 they would sit at the bottom of the distribution and pull
    the P90 down, arming more often than the data supports — so they are held out
    of the distribution and counted in the open.
    """
    q, wdb = stores["queue"], stores["workers_db"]
    now = datetime.now(timezone.utc)
    for i in range(3):
        item_id = q.enqueue("autocode", "k")
        q.claim_next(worker_id="w1")
        q.mark_running(item_id)
        q.record_run(run_id=new_run_id("autocode"), queue_id=item_id,
                     source="autocode", status="success",
                     started_at=_iso(now), completed_at=_iso(now),
                     duration_seconds=30.0, summary="no session recorded",
                     meta_json=json.dumps({"session_ids": []}))
    rep = CO.replay(wdb, stores["usage_db"], window_days=30.0)
    assert rep["sources"]["autocode"]["samples"] == 44, "unattributed rows stay out"
    assert rep["unattributed"] == 3, rep


def test_the_replay_is_read_only_on_both_stores(stores):
    """The command is a reader. It runs against the LIVE queue and usage stores.

    `usage_store.read_rows_readonly` exists precisely so an offline reader cannot
    migrate or write a live `usage.db`; a replay that opened the store read-write
    would run its schema migration against the file the engine is writing to.
    """
    before = {p: p.stat().st_mtime_ns for p in (stores["workers_db"],
                                                stores["usage_db"])}
    CO.replay(stores["workers_db"], stores["usage_db"], window_days=30.0)
    for p, ns in before.items():
        assert p.stat().st_mtime_ns == ns, f"{p} was touched by a read"


# ---------------------------------------------------------------------------
# Clause 2 — >=40 samples AND >1.5xP90, each bound on its own.
# ---------------------------------------------------------------------------

def test_a_source_with_40_samples_and_a_run_above_1_5x_p90_arms(stores):
    """The positive case, end to end through the arming decision."""
    run_id = _outlier(stores["queue"], stores["workers_db"], tokens=4_000_000)
    row = [r for r in stores["queue"].list_runs(limit=200) if r["run_id"] == run_id][0]
    fired = CO.arm_after_run(stores["queue"], run=row, workers_db=stores["workers_db"],
                             usage_db=stores["usage_db"], cfg={})
    assert fired is not None, "45 samples and 4M tokens against a ~100k P90 must arm"


def test_a_source_with_39_samples_arms_nothing_even_on_a_huge_run(tmp_path, monkeypatch):
    """The floor, alone. 39 is one short of 40, and the run is 40x the median.

    Without the floor this is exactly the noise case the item's Risks name: a
    P90 over 12 runs is not a distribution. One enormous run would arm a source
    that has almost no history to be unusual against.
    """
    wdb, udb = _make_stores(tmp_path, monkeypatch)
    q = WorkQueue(wdb)
    for i in range(38):
        _record(q, wdb, source="arch-review", tokens=100_000 + i, day_offset=i * 0.2)
    run_id = _outlier(q, wdb, source="arch-review", tokens=4_000_000)
    row = [r for r in q.list_runs(limit=200) if r["run_id"] == run_id][0]
    assert CO.arm_after_run(q, run=row, workers_db=wdb, usage_db=udb, cfg={}) is None
    assert CO.should_arm(tokens=4_000_000, samples=39, p90_tokens=100_000,
                         min_samples=40, margin=1.5) is False
    assert CO.should_arm(tokens=4_000_000, samples=40, p90_tokens=100_000,
                         min_samples=40, margin=1.5) is True


def test_a_run_at_1_2x_p90_arms_nothing(stores):
    """The margin, alone: above P90 is not the same as above 1.5x P90.

    1.2x is deliberately inside that gap, because the gap is the whole design
    finding — exceeding your own source's P90 is ~10% of every run by
    construction, and at that rate the loop spends more on investigations than it
    can ever learn.
    """
    _outlier(stores["queue"], stores["workers_db"], tokens=120_000)
    assert CO.should_arm(tokens=120_000, samples=45, p90_tokens=100_000,
                         min_samples=40, margin=1.5) is False
    assert CO.should_arm(tokens=150_001, samples=45, p90_tokens=100_000,
                         min_samples=40, margin=1.5) is True


def test_a_run_with_no_attributable_tokens_arms_nothing(stores):
    """No magnitude, no verdict — and the reason has to come back, not a silent None.

    A run whose sessions are missing from `usage` is the 11% triage found, not a
    free run. Treating it as zero tokens would put it below every threshold and
    report the source as calm.
    """
    q, wdb = stores["queue"], stores["workers_db"]
    now = datetime.now(timezone.utc)
    item_id = q.enqueue("autocode", "k")
    q.claim_next(worker_id="w1")
    q.mark_running(item_id)
    q.record_run(run_id=new_run_id("autocode"), queue_id=item_id, source="autocode",
                 status="success", started_at=_iso(now), completed_at=_iso(now),
                 duration_seconds=30.0, meta_json=json.dumps({"session_ids": ["nope"]}))
    row = [r for r in q.list_runs(limit=200)][0]
    assert CO.arm_after_run(q, run=row, workers_db=wdb, usage_db=stores["usage_db"],
                            cfg={}) is None


def test_a_failed_run_is_out_of_the_distribution_and_cannot_arm(stores):
    """The population is successful completed runs, as the replay was measured over.

    Triage's 355-of-400 attribution sample is over successful runs; a poisoned or
    timed-out row's token count is a record of an attempt dying, not of a job's
    cost, and mixing the two would move the P90 the trigger is judged by.
    """
    q, wdb = stores["queue"], stores["workers_db"]
    now = datetime.now(timezone.utc)
    item_id = q.enqueue("autocode", "k")
    q.claim_next(worker_id="w1")
    q.mark_running(item_id)
    q.record_run(run_id=new_run_id("autocode"), queue_id=item_id, source="autocode",
                 status="failed", started_at=_iso(now), completed_at=_iso(now),
                 duration_seconds=30.0, meta_json=json.dumps({"session_ids": ["x"]}))
    usage_store.record_usage(session_id="x", model="primary",
                             input_tokens=5_000_000, output_tokens=5_000_000)
    rep = CO.replay(wdb, stores["usage_db"], window_days=30.0)
    assert rep["sources"]["autocode"]["samples"] == 44, rep
    row = [r for r in q.list_runs(limit=300) if r["status"] == "failed"][0]
    assert CO.arm_after_run(q, run=row, workers_db=wdb, usage_db=stores["usage_db"],
                            cfg={}) is None


# ---------------------------------------------------------------------------
# Clause 4 — the recursion guard.
# ---------------------------------------------------------------------------

def test_an_expensive_investigation_run_cannot_arm_another_one(stores):
    """The guard, as a difference in behaviour rather than a constant.

    Same token count, same sample count, two sources: an `autocode` run of 4M
    tokens arms, a `cost-investigate` run of 4M tokens does not. Delete the
    exclusion and the second assertion fails — which is the regress this exists
    for, since an investigation is the run in the fleet most likely to be the
    outlier, and each one firing the next is how the loop eats the GPU it is
    meant to be auditing.
    """
    big = 4_000_000
    auto_run = _outlier(stores["queue"], stores["workers_db"], source="autocode",
                        tokens=big)
    inv_run = _outlier(stores["queue"], stores["workers_db"],
                       source=CO.SOURCE, tokens=big)
    rows = {r["run_id"]: r for r in stores["queue"].list_runs(limit=200)}
    assert CO.arm_after_run(stores["queue"], run=rows[auto_run],
                            workers_db=stores["workers_db"],
                            usage_db=stores["usage_db"], cfg={}) is not None
    assert CO.arm_after_run(stores["queue"], run=rows[inv_run],
                            workers_db=stores["workers_db"],
                            usage_db=stores["usage_db"], cfg={}) is None


def test_investigation_runs_are_out_of_every_source_s_distribution(stores):
    """The exclusion is over the whole scan, not only over the source's own rows.

    A per-source filter would still let an investigation row into a fleet-wide
    total, and the printed P90 is what a human decides the margin from. Here 40
    fat investigations must move `autocode`'s sample count not at all, and must
    not appear as their own row.
    """
    for i in range(40):
        _record(stores["queue"], stores["workers_db"], source=CO.SOURCE,
                tokens=900_000, day_offset=i * 0.1)
    rep = CO.replay(stores["workers_db"], stores["usage_db"], window_days=30.0)
    assert CO.SOURCE not in rep["sources"], rep["sources"]
    assert rep["sources"]["autocode"]["samples"] == 44, rep
    assert CO.SOURCE in CO.EXCLUDED_SOURCES


def test_the_replay_prints_no_row_for_the_investigation_source(stores):
    """The printed table is the guard a human reads, so it says the same thing.

    Forty 900k-token investigations are the loudest rows in the store; if any of
    them reached the table, the next reading of "which source is expensive" would
    be answered by the auditing loop's own bill.
    """
    for i in range(40):
        _record(stores["queue"], stores["workers_db"], source=CO.SOURCE,
                tokens=900_000, day_offset=i * 0.1)
    out = _run_cli("--workers-db", str(stores["workers_db"]),
                   "--usage-db", str(stores["usage_db"]))
    assert out.returncode == 0, out.stderr[-1200:]
    rows = [ln.split()[0] for ln in out.stdout.splitlines()
            if ln.strip().startswith(("autocode", "bench-mine", CO.SOURCE))]
    assert CO.SOURCE not in rows, out.stdout
    assert "autocode" in rows


# ---------------------------------------------------------------------------
# The percentile rule itself, because "P90" is not one definition.
# ---------------------------------------------------------------------------

def test_the_percentile_is_the_nearest_rank_rule_this_box_already_uses():
    """Same rule as the local `_pct` in `usage_store.ttft_summary` (#1746 lines
    1166-1172), which is a closure and cannot be imported. Two percentile helpers
    that disagree by one index would make the replay and the dashboard's latency
    percentiles non-comparable, so the rule is stated and pinned rather than
    silently re-derived.
    """
    assert CO.percentile([], 0.9) is None
    assert CO.percentile([10], 0.9) == 10
    assert CO.percentile(list(range(1, 11)), 0.9) == 9
    assert CO.percentile([100_000] * 44 + [4_000_000], 0.9) == 100_000


def test_the_defaults_are_the_measured_numbers_and_say_so_in_the_file():
    """The three tuning constants, pinned against the measurements that set them.

    These are not style constants: 1.0x measured 197/1,969 (10.0%), 1.5x measured
    10 fires in 18 days, 2.0x measured 0. A later edit that nudges `margin` to
    1.0 silently puts ~34 diagnostic runs a day on the GPU, so the value is
    checked here with the evidence beside it.
    """
    assert CO.DEFAULT_MARGIN == 1.5
    assert CO.DEFAULT_MIN_SAMPLES == 40
    assert CO.DEFAULT_WINDOW_DAYS == 30.0
    src = (ROOT / "app" / "cost_outlier.py").read_text()
    assert "10.0%" in src and "197" in src, "the measurement behind the margin"


# ---------------------------------------------------------------------------
# The diagnostic record itself: the item's fourth acceptance sentence, and the
# only output of this whole loop that a human reads.
# ---------------------------------------------------------------------------

def _fired_payload(queue, wdb, udb, *, source="autocode", tokens=4_000_000):
    """Arm the trigger the way the pool does and hand back the queued payload."""
    for i in range(44):
        _record(queue, wdb, source=source, tokens=100_000 + i, day_offset=i * 0.3)
    _record(queue, wdb, source=source, tokens=tokens, day_offset=0)
    # The newest row is the outlier: `_record` writes one run per call and the
    # seeded ones are all older by construction.
    row = queue.list_runs(limit=500)[0]
    assert row["source"] == source
    # `usage_db` named explicitly: `arm_after_run` defaults it to
    # `app.paths.USAGE_DB`, and a test that let it default would read the live
    # fleet's store from a scratch fixture.
    out = CO.arm_after_run(queue, run=row, usage_db=udb, cfg={})
    assert out, "the fixture did not arm, so the record below was never owed"
    item = [i for i in queue.list_items(limit=50) if i.source == CO.SOURCE][0]
    return item.payload


def test_the_diagnostic_writes_one_record_naming_where_and_one_candidate(
        tmp_path, monkeypatch):
    """The record names where the tokens went AND one improvement candidate.

    Two things this pins that a "record was written" assertion would not: the
    per-session token table has to carry the run's OWN figures (the item asks
    where turns and tokens went, and that exists only per usage row), and the
    candidate must quote a figure rather than gesture — a candidate with no
    number beside it is a guess about a run that is already finished.
    """
    wdb, udb = _make_stores(tmp_path, monkeypatch)
    queue = WorkQueue(wdb)
    payload = _fired_payload(queue, wdb, udb)
    rec = tmp_path / "rec.md"
    monkeypatch.setattr(CO, "record_path", lambda run_id: rec)
    res = CO.investigate(payload, usage_db=udb)

    assert rec.exists(), "no record for a fired run"
    assert res["artifact_path"] == str(rec)
    txt = rec.read_text()
    assert txt.startswith(f"# Cost outlier — `{payload['run_id']}` (autocode)")
    assert "## Where the tokens went" in txt and "## One improvement candidate" in txt
    assert f"{payload['tokens']:,} tokens" in txt, "the trigger's own figure"
    # The table is the run's rows: the session id and its real per-turn numbers.
    rows = CO.usage_rows(udb, payload["session_ids"])
    assert len(rows) == 6, f"expected the fired run's 6 usage rows, got {len(rows)}"
    assert f"{rows[0]['input_tokens']:,}" in txt, "a usage figure is missing"
    cand = CO.analyse(rows)["candidate"]
    assert cand["rule"] in txt and cand["text"] in txt
    assert any(ch.isdigit() for ch in cand["text"]), "a candidate with no figure"
    # Dollars are not part of this: the column is zero on every row of the store.
    assert "$" not in txt and "cost_usd" not in txt


def test_the_record_says_the_investigator_never_scores_itself(tmp_path, monkeypatch):
    """The recursion guard has to be READABLE, not merely true.

    A reader of one record is the person who later decides whether to widen the
    exclusion list, so the record states the rule it was measured under.
    """
    wdb, udb = _make_stores(tmp_path, monkeypatch)
    queue = WorkQueue(wdb)
    payload = _fired_payload(queue, wdb, udb)
    rec = tmp_path / "rec.md"
    monkeypatch.setattr(CO, "record_path", lambda run_id: rec)
    CO.investigate(payload, usage_db=udb)
    txt = rec.read_text()
    assert f"`{CO.SOURCE}` runs are excluded" in txt
    assert "No model turn" in txt, (
        "the record must say what it cost to produce, since the whole loop is "
        "justified by costing less than what it explains")


def test_the_candidate_rules_are_ordered_and_every_one_carries_a_figure():
    """The candidate list is a priority order over causes, and each rule's text is
    a template with at least one substituted number.

    The order matters: a run that both hit its turn cap and re-read an uncached
    prefix gets the turn-cap sentence, because finishing the job is the cheaper
    fix. An unordered `if` chain would report whichever the author noticed first.
    """
    keys = [k for k, _ in CO._CANDIDATE_RULES]
    assert keys == ["turn_cap", "compaction", "prefix_cache", "tool_errors", "refill"]
    for _k, tmpl in CO._CANDIDATE_RULES:
        assert "{" in tmpl, f"{_k}: a candidate with no figure to check"
    rows = [{"session_id": "s", "ts": "t", "model": "m", "input_tokens": 10_000,
             "output_tokens": 2_000, "cache_create": 0, "cache_read": 0,
             "num_turns": 3, "reprefill_tokens": 0, "prefix_misses": 0,
             "compaction": 0, "stop_reason": "max_turns", "tool_calls": 4,
             "tool_errors": 0, "tool_errors_by_class": None, "skills": None}] * 3
    v = CO.analyse(rows)
    assert v["candidate"]["rule"] == "turn_cap", v["candidate"]
    assert "3" in v["candidate"]["text"]


def test_a_run_with_no_rule_matched_still_names_a_candidate(tmp_path, monkeypatch):
    """The `spread` fallback, because an empty candidate field is the failure the
    whole record exists to avoid: the item asks for one named improvement, and a
    record that found nothing has to say which measurement to take next rather
    than stay quiet.
    """
    wdb, udb = _make_stores(tmp_path, monkeypatch)
    queue = WorkQueue(wdb)
    payload = _fired_payload(queue, wdb, udb)
    rows = CO.usage_rows(udb, payload["session_ids"])
    for r in rows:
        r.update(cache_read=10 ** 9, reprefill_tokens=0, compaction=0,
                 stop_reason="end_turn", tool_calls=10, tool_errors=0)
    v = CO.analyse(rows)
    assert v["candidate"]["rule"] == "spread", v["candidate"]
    assert "candidate is the task's size itself" in v["candidate"]["text"]


# ---------------------------------------------------------------------------
# The `compaction` column is one turn's JSON freeing record (#1078), not a count.
#
# Every site pinned below used to read the column with `int()`, so a real fire of
# this diagnostic raised `ValueError: invalid literal for int() with base 10:
# '{"mechanisms": [], "turn_start": …` — the identical string in the `error` column
# of the quarantined `cost-investigate` queue row (workers.db id 8980, 3 attempts,
# enqueued 2026-10-13T03:45:32Z) — and died upstream of its one write, which is why
# `~/lloyd-data/_pipeline/cost-outliers/` had never existed (#2738). The seeds above
# are all ints, which is how a green suite shipped that crash: no test here had
# ever put a blob in the column.
# ---------------------------------------------------------------------------

def _turn_start_blob(turn_start_freed: int, relief_freed: int = 0) -> str:
    """A `compaction` value in the shape the live column holds.

    Copied field for field out of a real row of `~/lloyd-data/usage.db` read
    2026-10-15 (row 11,553, which freed 11,343 tokens at turn start and 33,618
    through the intra-turn relief ladder); only the two freed-token numbers are
    parameterised, so a test can ask for exact arithmetic and still be handing the
    reader the shape that raised. Written as JSON text because that is what the
    column stores: TEXT holding an object, normalised by `usage_store`'s writer.
    """
    rec = {
        "mechanisms": ["microcompact"],
        "relief": [],
        "relief_passes": 0,
        "relief_tokens_freed": relief_freed,
        "turn_start": {
            "context_window": 262_144, "flushed_before_summary": False,
            "mechanisms": ["microcompact"], "microcompact_reduced": 0,
            "microcompact_sidecar": False, "microcompacted": 21, "ran": True,
            "restored_files": 0, "summarize_attempted": False,
            "summarize_outcome": "under_threshold", "summarized": False,
            "summary_covered_rows": 0, "summary_folds": 0, "summary_reused": False,
            "threshold": 210_144, "tokens_after": 28_261, "tokens_before": 39_604,
            "tokens_freed": turn_start_freed, "truncated": False,
        },
    }
    if relief_freed:
        rec["mechanisms"].append("relief:intra_turn")
        rec["relief_passes"] = 1
        rec["relief"] = [{"freed_tokens": relief_freed, "passes": 2,
                          "rearm": 151_303, "reason": "intra_turn",
                          "rungs": ["tool_results:11", "reasoning:6"],
                          "target": 109_274, "used_after": 108_211,
                          "used_before": 151_313}]
    return json.dumps(rec)


def _relief_only_blob(relief_freed: int) -> str:
    """The OTHER live shape: the relief ladder ran mid-turn, so the record has no
    `turn_start` key at all. 60 of the 8,393 populated rows on this box are this
    shape (measured 2026-10-15, `json_extract(compaction,'$.turn_start') IS NULL`),
    and 506 of the 1,145 rows that freed anything freed only through it — so a
    reader that indexes `record["turn_start"]` KeyError's on these, and one that
    merely looks under it reports them as turns that freed nothing.
    """
    return json.dumps({
        "mechanisms": ["relief:intra_turn"],
        "relief": [{"freed_tokens": relief_freed, "passes": 2, "rearm": 151_303,
                    "reason": "intra_turn",
                    "rungs": ["tool_results:11", "reasoning:6", "reasoning:2"],
                    "target": 109_274, "used_after": 108_211,
                    "used_before": 151_313}],
        "relief_passes": 1,
        "relief_tokens_freed": relief_freed,
    })


def _usage_row(compaction=None, *, inp=10_000, outp=2_000, stop="end_turn", **over):
    """One usage row as `usage_rows` hands it to `analyse`: every column present.

    `cache_read` is large by default so `prefix_cache` stays unlit and a candidate
    assertion says which rule fired rather than which one the fixture forgot to
    switch off.
    """
    row = {"session_id": "s", "ts": "t", "model": "m", "input_tokens": inp,
           "output_tokens": outp, "cache_create": 0, "cache_read": 10 ** 7,
           "num_turns": 3, "reprefill_tokens": 0, "prefix_misses": 0,
           "compaction": compaction, "stop_reason": stop, "tool_calls": 4,
           "tool_errors": 0, "tool_errors_by_class": None, "skills": None}
    row.update(over)
    return row


def test_a_live_compaction_blob_is_read_as_a_record_and_never_cast_to_a_number():
    """Clause 1: the blob is the crash, so the blob is the seed.

    Before #2738 both rows below raised on sight, and `analyse` is called before
    the record's only write, so a fire produced no artifact at all. The relief-only
    row is here because it is the second live shape, not a variation of the first:
    it has no `turn_start` block to read.
    """
    rows = [_usage_row(_turn_start_blob(11_343, 33_618)),
            _usage_row(_relief_only_blob(43_102)),
            _usage_row()]                      # the NULL the store also holds
    v = CO.analyse(rows)
    assert isinstance(v, dict), "a blob turned the readout into an exception"
    # Two rows freed something; the third recorded nothing.
    assert v["figures"]["compact"] == 2, v["figures"]
    assert v["candidate"]["rule"] == "compaction", v["candidate"]

    # And no reader of this column may cast it again, in this module or later:
    # `usage.compaction` is TEXT (`app/usage_store.py`'s own CREATE TABLE), so a
    # line that casts it is a line that raises on 100% of the live rows.
    src = (ROOT / "app" / "cost_outlier.py").read_text(encoding="utf-8")
    cast = [ln.strip() for ln in src.splitlines()
            if "int(" in ln and "compaction" in ln]
    assert cast == [], f"the column holds a record, not a count: {cast}"


def test_the_compaction_rule_needs_a_record_that_actually_freed_something():
    """Clause 2: presence is not cost.

    8,393 of the live usage rows carry a record and 1,145 of those freed anything
    (measured 2026-10-15), so arming this rule on a record merely being present
    would light it on 86% of runs — ahead of `prefix_cache`, whose <50%-cached
    threshold is the one that actually discriminates, so the record would name the
    wrong fix on most of the runs it wrote.

    The quiet rows are not empty records: each says a pass RAN (`ran: true`, 21
    things microcompacted) and freed 0, which is the case that reads as evidence
    and is not.
    """
    quiet = CO.analyse([_usage_row(_turn_start_blob(0, 0)),
                        _usage_row(_relief_only_blob(0))])
    assert quiet["figures"]["compact"] == 0, quiet["figures"]
    assert quiet["checks"]["compaction"] is False, quiet["checks"]
    assert quiet["candidate"]["rule"] != "compaction", quiet["candidate"]

    relief = CO.analyse([_usage_row(_relief_only_blob(43_102))])
    assert relief["checks"]["compaction"] is True, relief["checks"]
    assert relief["candidate"]["rule"] == "compaction", relief["candidate"]


def test_compact_pct_is_freed_tokens_as_a_share_of_the_run():
    """Clause 3: 10,000 tokens freed of a 100,000-token run is 10.0%.

    Not a sum of the column: summing a JSON blob is arithmetically impossible, and
    summing the mere presence of one would report 1 or 2 as a percentage. And the
    number must not move when a row whose record freed nothing joins the run — the
    zero-token row below is not invented, the live store has 10 rows with
    `input_tokens+output_tokens = 0` and every one of them carries a compaction
    record (measured 2026-10-15), which is exactly the row that would inflate a
    presence-counting figure's row count without adding a token of spend.
    """
    run = [_usage_row(_turn_start_blob(6_000, 4_000), inp=90_000, outp=10_000)]
    v = CO.analyse(run)
    assert v["total_tokens"] == 100_000, v["total_tokens"]
    assert v["figures"]["compact_pct"] == 10.0, v["figures"]

    with_freebie = run + [_usage_row(_turn_start_blob(0, 0), inp=0, outp=0)]
    v2 = CO.analyse(with_freebie)
    assert v2["figures"]["compact_pct"] == 10.0, v2["figures"]
    assert v2["figures"]["compact"] == 1, "the row that freed nothing is not compacted"


def test_a_fired_run_with_a_live_blob_writes_its_one_record(tmp_path, monkeypatch):
    """Clause 4: the deliverable, across the store seam that produced it.

    Arming happens the way the pool arms (`_fired_payload`), the blob row is written
    by the shipped writer into the fired run's own session, and `investigate` then
    re-reads that store through `usage_rows` — so this test crosses the two
    boundaries the crash sat between (sqlite TEXT -> dict -> figures -> markdown)
    rather than calling `analyse` with a dict built in the test.
    """
    wdb, udb = _make_stores(tmp_path, monkeypatch)
    queue = WorkQueue(wdb)
    payload = _fired_payload(queue, wdb, udb)
    usage_store.record_usage(session_id=payload["session_ids"][0], model="primary",
                             input_tokens=1_000, output_tokens=500, num_turns=7,
                             compaction=json.loads(_turn_start_blob(11_343, 33_618)))

    out = tmp_path / "cost-outliers"
    monkeypatch.setattr(CO, "record_path", lambda run_id: out / f"{run_id}.md")
    res = CO.investigate(payload, usage_db=udb)

    # Exactly one record, because the item is a file count over this directory.
    files = sorted(p.name for p in out.iterdir())
    assert len(files) == 1, f"expected one record for one fire, got {files}"
    txt = (out / files[0]).read_text()
    assert res["artifact_path"] == str(out / files[0])

    cand = [ln for ln in txt.splitlines() if ln.startswith("**`")]
    assert cand, "no candidate line"
    assert cand[0].startswith("**`compaction`**"), cand[0]
    assert "{" not in cand[0], f"an unsubstituted template reached the record: {cand}"
    assert "44,961" in txt, "the compaction cell must show the tokens it freed"
    assert '"mechanisms"' not in txt, "the raw blob does not belong in a record"


def test_a_freeing_blob_beside_the_int_seeds_still_loses_to_the_turn_cap():
    """Clause 5: the priority order is a judgement about causes, and a blob is now
    in the same list of seeds the ints are in. A run that hit its turn cap AND
    rewrote its history still gets the turn-cap sentence — finishing the job is the
    cheaper fix — so reading `compaction` correctly must not have promoted it.
    """
    order = [k for k, _ in CO._CANDIDATE_RULES]
    assert order.index("turn_cap") < order.index("compaction"), order
    v = CO.analyse([_usage_row(_turn_start_blob(11_343, 33_618), stop="max_turns"),
                    _usage_row(0),            # the int seed, still readable
                    _usage_row(_relief_only_blob(43_102))])
    assert v["checks"]["turn_cap"] and v["checks"]["compaction"], v["checks"]
    assert v["candidate"]["rule"] == "turn_cap", v["candidate"]
