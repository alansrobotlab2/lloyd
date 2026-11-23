"""Measured close: a landed improvement item is settled by its bench, not a session.

No engine, no clock, no vault, no git: fitness rows are stamped off a fixed
epoch, the ledger is a list, "does this row contain the landing" is a lambda,
and the board is a `tmp_path` directory only where the write is under test.

Run: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_measured_close.py -q
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, fitness as F, fitness_intake as FI
from scripts.automod import measured_close as MC, owed as O, scorecard as SC, state as S
from workers.sources import owed_check as OC

T0 = 1_000_000_000.0
DAY = 86400.0
LANDED = T0 + 10 * DAY
SHA = "ab12cd34" + "0" * 32
FACTS = {"item_id": 7, "bench": "alpha", "commit": SHA, "landed_ts": LANDED, "round_id": "SM_r"}
QUIET = {"alpha": {"status": "ok", "spread": 0.02}}
ANY = lambda commit: None        # noqa: E731 — judge every row by time alone


def _row(value, day, *, bench="alpha", n=100, higher=True, commit="c" * 40, **kw):
    ts = T0 + day * DAY
    k = round(value * n)
    lo, hi = F.wilson(k, n)
    return {"kind": "bench", "bench": bench, "status": F.MEASURED, "value": value, "n": n,
            "k": k, "ci_low": lo, "ci_high": hi, "higher_is_better": higher, "ts": ts,
            "commit": commit, "recorded_at": f"day-{day:g}", "run_id": f"run-{day:g}", **kw}


# ── ruling ───────────────────────────────────────────────────────────────────

def test_met_when_the_first_row_after_landing_moved_the_right_way_outside_noise():
    r = MC.ruling(FACTS, [_row(0.60, 9), _row(0.85, 11), _row(0.10, 12)], QUIET)
    assert r["verdict"] == MC.MET and r["delta"] == 0.25
    # the FIRST post-landing row decides: day 12 has other landings in it
    assert (r["pre"]["value"], r["post"]["value"]) == (0.60, 0.85)


def test_lower_is_better_reads_the_direction_off_the_row():
    rows = [_row(0.60, 9, higher=False), _row(0.30, 11, higher=False)]
    assert MC.ruling(FACTS, rows, QUIET)["verdict"] == MC.MET
    worse = [_row(0.30, 9, higher=False), _row(0.60, 11, higher=False)]
    r = MC.ruling(FACTS, worse, QUIET)
    assert (r["verdict"], r["direction"], r["outside_noise"]) == (MC.REJECTED, "regressed", True)


@pytest.mark.parametrize("after, why", [
    (0.61, "the delta is inside the noise floor"),
    (0.60, "the delta is inside the noise floor"),
    (0.66, "the post-landing row's 95% interval covers the pre-landing value"),
    (0.30, "moved the wrong way, outside the noise floor"),
])
def test_rejected_when_it_did_not_move_outside_noise_in_the_right_direction(after, why):
    r = MC.ruling(FACTS, [_row(0.60, 9), _row(after, 11)], QUIET)
    assert (r["verdict"], r["why"]) == (MC.REJECTED, why)


def test_pending_until_a_row_at_or_after_the_landing_exists():
    r = MC.ruling(FACTS, [_row(0.60, 8), _row(0.60, 9)], QUIET)
    assert (r["verdict"], r["why"], r["can_rule"]) == (
        MC.PENDING, "no fitness row at or after the landing", True)


def test_a_row_whose_code_does_not_contain_the_landing_is_not_a_post_row():
    rows = [_row(0.60, 9), _row(0.90, 11, commit="old"), _row(0.90, 12, commit="new")]
    r = MC.ruling(FACTS, rows, QUIET, contains=lambda row: row["commit"] == "new")
    assert r["verdict"] == MC.MET and r["post"]["commit"] == "new"
    none = MC.ruling(FACTS, rows[:2], QUIET, contains=lambda row: row["commit"] == "new")
    assert none["verdict"] == MC.PENDING


@pytest.mark.parametrize("noise, why", [
    ({}, "`alpha` has no measured noise floor"),
    ({"alpha": {"status": "insufficient"}}, "`alpha` has no measured noise floor"),
    ({"alpha": {"status": "too-noisy", "spread": 0.4}}, "`alpha` is flagged too-noisy"),
])
def test_no_floor_no_ruling(noise, why):
    r = MC.ruling(FACTS, [_row(0.60, 9), _row(0.95, 11)], noise)
    assert (r["verdict"], r["why"], r["can_rule"]) == (MC.PENDING, why, False)


def test_no_row_before_the_landing_cannot_be_ruled():
    r = MC.ruling(FACTS, [_row(0.95, 11)], QUIET)
    assert (r["verdict"], r["can_rule"]) == (MC.PENDING, False)


def test_a_row_with_no_value_is_no_row():
    rows = [_row(0.60, 9), {**_row(0.0, 11), "status": "timeout", "value": None}]
    assert MC.ruling(FACTS, rows, QUIET)["verdict"] == MC.PENDING


# ── candidates and the per-item decision ─────────────────────────────────────

def _landed(item_id, *, ts=LANDED, acceptance="met", commit=SHA, **kw):
    return {"event": "item_landed", "item_id": item_id, "ts": ts, "acceptance": acceptance,
            "commit": commit, "round_id": "SM_r", "closed": True, **kw}


def test_only_met_landings_of_improvement_items_with_a_manifest_metric():
    events = [_landed(1), _landed(2), _landed(3), _landed(4, acceptance="not_met"),
              _landed(5, vault=True), _landed(6), {"event": "promoted", "item_id": 1, "ts": 1.0}]
    classes = {1: ("improvement", "alpha"), 2: ("fix", ""), 3: ("improvement", "made up"),
               4: ("improvement", "alpha"), 5: ("improvement", "alpha")}      # 6: unclassified
    assert [c["item_id"] for c in MC.candidates(events, classes, ["alpha"])] == [1]


def test_the_newest_landing_of_an_item_is_the_one_measured():
    events = [_landed(1, ts=LANDED, commit="old"), _landed(1, ts=LANDED + DAY, commit="new")]
    got = MC.candidates(events, {1: ("improvement", "alpha")}, ["alpha"])
    assert [(c["commit"], c["landed_ts"]) for c in got] == [("new", LANDED + DAY)]
    # … and a verdict recorded for the earlier landing does not answer for it
    assert MC.state_of({MC.FM_KEY: {"commit": "old", "verdict": "met"}}, "new") == ""


@pytest.mark.parametrize("state, verdict, age_days, want", [
    ("", MC.PENDING, 1, "open"),
    (MC.PENDING, MC.PENDING, 1, "wait"),
    (MC.PENDING, MC.PENDING, 14, MC.UNMEASURED),       # bounded: released, never held for ever
    ("", MC.MET, 1, MC.MET),
    (MC.PENDING, MC.REJECTED, 3, MC.REJECTED),
    (MC.MET, MC.REJECTED, 3, "skip"),                  # ruled once
    (MC.UNMEASURED, MC.MET, 20, "skip"),
    ("", MC.PENDING, 40, "skip"),                      # history from before the switch
    ("", MC.MET, 40, "skip"),
])
def test_decide(state, verdict, age_days, want):
    got = MC.decide(FACTS, state, {"verdict": verdict}, now=LANDED + age_days * DAY)
    assert got == want


# ── the pass, with fakes ─────────────────────────────────────────────────────

class Fake:
    def __init__(self, fronts=None, open_fronts=()):
        self.fronts = fronts or {7: {"class": "improvement", "metric": "alpha"}}
        self.open_fronts = list(open_fronts)
        self.opened, self.settled, self.filed, self.events = [], [], [], []

    def run(self, rows, *, now=LANDED + DAY, noise=QUIET, events=None, **kw):
        paths = {iid: Path(f"/nonexistent/{iid}.md") for iid in self.fronts}
        return MC.settle_pass(
            now=now, rows=rows, noise=noise, benches=["alpha"],
            events=[_landed(7)] if events is None else events,
            board=(paths, self.fronts, self.open_fronts), contains_for=ANY,
            opener=lambda path, facts, **k: self._open(facts),
            settler=lambda path, facts, verdict, r, **k: self._settle(facts, verdict, r, k),
            filer=self._file, append=self.events.append, **kw)

    def _open(self, facts):
        self.opened.append(facts["item_id"])
        self.fronts[facts["item_id"]][MC.FM_KEY] = {"commit": facts["commit"],
                                                   "verdict": MC.PENDING}

    def _settle(self, facts, verdict, r, kw):
        self.settled.append((facts["item_id"], verdict, kw.get("follow_up"), kw.get("note")))
        self.fronts[facts["item_id"]][MC.FM_KEY] = {"commit": facts["commit"], "verdict": verdict}

    def _file(self, name, body, *, bench, cls):
        self.filed.append({"name": name, "body": body, "bench": bench, "cls": cls})
        return 900 + len(self.filed)

    def verdicts(self):
        return [e["verdict"] for e in self.events]


def test_a_landing_with_no_row_yet_opens_the_debt_once_and_waits():
    fake = Fake()
    assert [r["verdict"] for r in fake.run([_row(0.60, 9)])] == [MC.OPENED]
    assert fake.run([_row(0.60, 9)]) == []                 # pending: nothing more to say
    assert (fake.opened, fake.settled, fake.verdicts()) == ([7], [], [MC.OPENED])


def test_met_settles_and_files_nothing():
    fake = Fake()
    fake.run([_row(0.60, 9)])
    out = fake.run([_row(0.60, 9), _row(0.85, 11)], now=LANDED + 2 * DAY)
    assert [r["verdict"] for r in out] == [MC.MET]
    assert fake.settled == [(7, MC.MET, None, "")] and not fake.filed
    assert fake.verdicts() == [MC.OPENED, MC.MET]
    assert fake.run([_row(0.60, 9), _row(0.85, 11)], now=LANDED + 3 * DAY) == []   # ruled once


def test_a_ruling_on_the_first_pass_still_records_the_landing():
    fake = Fake()
    fake.run([_row(0.60, 9), _row(0.85, 11)])
    assert fake.opened == [7] and fake.verdicts() == [MC.OPENED, MC.MET]


def test_rejected_records_the_numbers_and_files_one_revert_proposal():
    fake = Fake()
    out = fake.run([_row(0.60, 9), _row(0.61, 11)])
    assert out[0]["verdict"] == MC.REJECTED and out[0]["follow_up"] == 901
    assert "before 0.600 (n=100" in out[0]["summary"] and "after 0.610" in out[0]["summary"]
    assert fake.settled == [(7, MC.REJECTED, 901, "")]
    (p,) = fake.filed
    assert p["cls"] == "simplify" and p["bench"] == "alpha"
    assert p["name"].startswith("Propose reverting ab12cd34 (#7): `alpha` did not move")
    assert "not a rollback request: nothing was reverted" in p["body"]
    assert not B.is_derived_name(p["name"])


def test_a_landing_that_made_the_bench_worse_proposes_the_revert_as_an_improvement():
    fake = Fake()
    fake.run([_row(0.60, 9), _row(0.30, 11)])
    (p,) = fake.filed
    assert p["cls"] == "improvement" and "got worse after it landed" in p["name"]
    assert "back within its noise floor (0.020) of the 0.600" in p["body"]


def test_no_revert_proposal_while_an_open_item_names_the_bench():
    fake = Fake(open_fronts=[(55, {"metric": "alpha"})])
    out = fake.run([_row(0.60, 9), _row(0.61, 11)])
    assert out[0]["verdict"] == MC.REJECTED and out[0]["follow_up"] is None and not fake.filed
    assert "open item #55 already names `alpha`" in fake.settled[0][3]


def test_one_revert_proposal_per_pass_and_then_one_per_bench():
    fronts = {7: {"class": "improvement", "metric": "alpha"},
              8: {"class": "improvement", "metric": "beta"}}
    fake = Fake(fronts)
    rows = [_row(0.60, 9), _row(0.61, 11), _row(0.60, 9, bench="beta"), _row(0.61, 11, bench="beta")]
    noise = {**QUIET, "beta": {"status": "ok", "spread": 0.02}}
    out = MC.settle_pass(
        now=LANDED + DAY, rows=rows, noise=noise, benches=["alpha", "beta"],
        events=[_landed(7), _landed(8)],
        board=({7: Path("/x/7.md"), 8: Path("/x/8.md")}, fronts, []), contains_for=ANY,
        opener=lambda *a, **k: None, settler=lambda *a, **k: None,
        filer=fake._file, append=fake.events.append)
    assert [(r["item_id"], r["verdict"], r["follow_up"]) for r in out] == [
        (7, MC.REJECTED, 901), (8, MC.REJECTED, None)]
    assert out[1]["note"] == "No revert proposal filed: this pass already filed one."


def test_pending_past_the_bound_is_released_unmeasured_with_the_reason():
    fake = Fake()
    fake.run([_row(0.60, 9)])
    out = fake.run([_row(0.60, 9)], now=LANDED + 14 * DAY)
    assert out[0]["verdict"] == MC.UNMEASURED
    assert fake.settled[0][1] == MC.UNMEASURED and not fake.filed
    assert fake.settled[0][3] == ("Released unmeasured after 14 d: "
                                  "no fitness row at or after the landing.")


def test_a_bench_with_no_floor_is_released_not_ruled():
    fake = Fake()
    rows = [_row(0.60, 9), _row(0.95, 11)]
    assert [r["verdict"] for r in fake.run(rows, noise={})] == [MC.OPENED]
    out = fake.run(rows, noise={}, now=LANDED + 15 * DAY)
    assert out[0]["verdict"] == MC.UNMEASURED and "no measured noise floor" in fake.settled[0][3]


def test_a_dry_run_reports_and_writes_nothing():
    fake = Fake()
    out = fake.run([_row(0.60, 9), _row(0.61, 11)], dry_run=True)
    assert [r["verdict"] for r in out] == [MC.REJECTED]
    assert not (fake.opened or fake.settled or fake.filed or fake.events)
    assert [r["verdict"] for r in fake.run([_row(0.60, 9)], dry_run=True)] == ["open"]


# ── the writes, on a real (scratch) board ────────────────────────────────────

@pytest.fixture
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    fm = {"status": "done", "priority": "medium", "board": "lloyd", "tags": ["backlog"],
          "class": "improvement", "metric": "alpha", "created": "2001-09-01T00:00:00",
          "automod_landed": SHA}
    path = d / "7-improve-alpha.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Improve alpha\n",
                    encoding="utf-8")
    O.add_owed(path, ["a hand check in scripts/x.py that another job owns"])
    return d, path


def _pass(rows, now, **kw):
    return MC.settle_pass(now=now, rows=rows, noise=QUIET, benches=["alpha"],
                          events=[_landed(7)], contains_for=ANY, **kw)


def test_the_debt_is_an_owed_entry_dated_past_the_bound(board):
    _, path = board
    _pass([_row(0.60, 9)], LANDED + DAY)
    fm = B.front_matter_of(path)
    assert fm[MC.FM_KEY]["verdict"] == MC.PENDING and fm[MC.FM_KEY]["commit"] == SHA
    mine = [e for e in O.entries_of(fm) if e["what"] == MC.owed_text("alpha", SHA)]
    assert len(mine) == 1 and mine[0]["kind"] == "check"
    assert B.names_a_demand(mine[0]["what"])
    due = datetime.fromisoformat(mine[0]["recheck_after"].replace("Z", "+00:00"))
    assert due.timestamp() == LANDED + 15 * DAY       # the session sees it only after the release
    at = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc)   # noqa: E731
    assert not O._due(mine[0], at(LANDED + 14 * DAY)) and O._due(mine[0], at(LANDED + 15 * DAY))
    # a second pass does not open it twice
    _pass([_row(0.60, 9)], LANDED + 2 * DAY)
    assert len(O.entries_of(B.front_matter_of(path))) == 2


def test_rejected_leaves_the_numbers_on_the_item_and_a_tagged_draft_on_the_board(board):
    d, path = board
    _pass([_row(0.60, 9)], LANDED + DAY)
    (out,) = _pass([_row(0.60, 9), _row(0.61, 11)], LANDED + 2 * DAY)
    fm = B.front_matter_of(path)
    block = fm[MC.FM_KEY]
    assert (block["verdict"], block["pre"]["value"], block["post"]["value"], block["delta"]) == (
        MC.REJECTED, 0.60, 0.61, 0.01)
    assert block["follow_up"] == out["follow_up"]
    assert fm["status"] == "done"                      # a clean close: the status is not moved
    # the other job's entry is untouched — not re-dated, not dropped
    (other,) = O.entries_of(fm)
    assert other["what"].startswith("a hand check") and not other.get("recheck_after")
    (record,) = fm[O.SETTLED_KEY]
    assert record["outcome"] == "ruling" and "delta +0.010" in record["evidence"]
    child = B.load_item(next(d.glob(f"{out['follow_up']}-*.md")))
    assert child.status == "draft" and FI.FITNESS_SPAWN_TAG in child.tags
    assert B.is_loop_spawned(child) and not B.is_quarantined(child)
    assert B.front_matter_of(child.path)[FI.FM_BENCH] == "alpha"
    assert [e["verdict"] for e in S.ledger_rows() if e.get("event") == MC.EVENT] == [
        MC.OPENED, MC.REJECTED]
    # the proposal now holds the bench: intake files nothing against it either
    assert FI.open_bench_items(FI.open_fronts()) == {"alpha": [child.id]}


def test_met_settles_the_entry_as_settled(board):
    _, path = board
    _pass([_row(0.60, 9), _row(0.85, 11)], LANDED + 2 * DAY)
    fm = B.front_matter_of(path)
    assert fm[MC.FM_KEY]["verdict"] == MC.MET
    assert [r["outcome"] for r in fm[O.SETTLED_KEY]] == ["settled"]
    assert len(O.entries_of(fm)) == 1


# ── the switch and the tick ──────────────────────────────────────────────────

def test_the_switch_ships_off_and_reads_both_spellings():
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    assert MC.enabled(cfg["automod"]["landing"]) is False
    assert MC.enabled({"measured_close": True}) and MC.enabled({"measured_close": "on"})
    assert not MC.enabled({"measured_close": "off"}) and not MC.enabled({})


def test_the_tick_does_nothing_when_off_or_dry_and_is_throttled_when_on(monkeypatch):
    passes = []
    monkeypatch.setattr(MC, "settle_pass", lambda **kw: passes.append(kw) or [])
    monkeypatch.setattr(MC, "_last_tick", 0.0)
    monkeypatch.setattr(MC, "enabled", lambda cfg=None: False)
    MC.tick(monotonic=lambda: 100.0)
    monkeypatch.setattr(MC, "enabled", lambda cfg=None: True)
    MC.tick(apply=False, monotonic=lambda: 100.0)
    assert passes == []
    MC.tick(monotonic=lambda: 100.0)
    MC.tick(monotonic=lambda: 100.0 + MC.TICK_EVERY_S - 1)
    assert len(passes) == 1
    MC.tick(monotonic=lambda: 100.0 + MC.TICK_EVERY_S)
    assert len(passes) == 2


def test_the_owed_check_tick_calls_it_off_the_loop_and_survives_its_failure(monkeypatch):
    seen = []
    monkeypatch.setattr(OC, "_next_owings", lambda batch, skip: [])
    monkeypatch.setattr(MC, "tick", lambda *, apply: seen.append(apply) or [])
    assert asyncio.run(OC.enqueue_if_due(None, {"apply": True})) is None
    assert seen == [True]

    def boom(*, apply):
        raise RuntimeError("series unreadable")
    monkeypatch.setattr(MC, "tick", boom)
    assert asyncio.run(OC.enqueue_if_due(None, {"apply": True})) is None


# ── scorecard row 25 ─────────────────────────────────────────────────────────

def test_the_scorecard_row_counts_landed_met_rejected_and_pending():
    def e(item, verdict, ts, **kw):
        return {"event": MC.EVENT, "item_id": item, "commit": f"c{item}", "verdict": verdict,
                "ts": ts, **kw}
    since = T0
    events = [e(1, MC.OPENED, T0 + 1), e(1, MC.MET, T0 + 2),
              e(2, MC.OPENED, T0 + 1), e(2, MC.REJECTED, T0 + 2, follow_up=9),
              e(3, MC.OPENED, T0 + 1),                           # pending
              e(4, MC.OPENED, T0 - 9 * DAY),                     # opened before the window, still pending
              e(5, MC.OPENED, T0 - 9 * DAY), e(5, MC.UNMEASURED, T0 + 5),
              {"event": "item_landed", "item_id": 1, "ts": T0 + 1}]
    block = MC.scorecard_block(events, since)
    assert block == {"tracked": True, "landed": 3, "met": 1, "rejected": 1, "unmeasured": 1,
                     "pending": 2, "proposals": 1}
    line = MC.render_scorecard_row(block)
    assert line.startswith("| 25 | measured closes | 1 met / 1 rejected of 3 landed |")
    assert "2 pending a fitness row" in line


def test_the_row_is_appended_last_and_says_so_when_nothing_was_measured(tmp_path):
    assert MC.scorecard_block([], T0)["tracked"] is False
    empty = tmp_path / "ledger.jsonl"
    empty.write_text("")
    full = SC.compute(since_days=7, ledger=empty, backlog_dir=tmp_path / "nope",
                      repo=tmp_path, now=T0)
    assert full["measured_closes"]["tracked"] is False
    assert SC.render(full).splitlines()[-1].startswith("| 25 | measured closes | — |")
    assert SC._render_measured_closes({}).startswith("| 25 | measured closes | — |")
