"""Fitness-driven intake: the measurement files at most one item a day.

No engine, no clock, no vault: the series is a list of rows stamped off a fixed
epoch, the ledger is a list, the board is a `tmp_path` directory (only where the
write itself is under test) and the filer is a fake everywhere else.

Run: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_fitness_intake.py -q
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scripts.automod import backlog as B, fitness as F, fitness_intake as FI, state as S
from scripts.automod import source_yield as SY
from workers.queue import QueueItem
from workers.sources import fitness as SRC

T0 = 1_000_000_000.0
DAY = 86400.0
NOW = T0 + 30 * DAY


def _row(bench, value, days_ago, *, n=100, higher=True, run=None, commit="c" * 40):
    ts = NOW - days_ago * DAY
    k = round(value * n)
    lo, hi = F.wilson(k, n)
    return {"schema": 1, "kind": "bench", "bench": bench, "status": F.MEASURED,
            "value": value, "n": n, "k": k, "ci_low": lo, "ci_high": hi,
            "higher_is_better": higher, "ts": ts, "commit": commit,
            "recorded_at": f"day-{days_ago:g}", "run_id": run or f"run-{days_ago:g}"}


def _noise(**benches):
    return {name: ({"status": "ok", "spread": v} if isinstance(v, float) else {"status": v})
            for name, v in benches.items()}


class Board:
    """A fake filer and a fake ledger."""

    def __init__(self):
        self.filed, self.events = [], []

    def filer(self, name, body, *, bench, cls="improvement"):
        self.filed.append({"name": name, "body": body, "bench": bench, "cls": cls})
        return 5000 + len(self.filed)

    def run(self, rows, noise, *, fronts=(), events=(), now=NOW, **kw):
        names = sorted({r["bench"] for r in rows})
        return FI.run(now=now, rows=rows, noise=noise, names=names,
                      events=list(events) + self.events, fronts=list(fronts),
                      filer=self.filer, **kw,
                      # the real `state.append_event` stamps `ts`; so does this one
                      append=lambda e: self.events.append({"ts": now, **e}))

    def journal(self):
        return [{k: v for k, v in e.items() if k != "ts"} for e in self.events]


# two benches that both regressed outside noise; beta by more
REGRESSED = [
    _row("alpha", 0.90, 9), _row("alpha", 0.70, 1),
    _row("beta", 0.90, 9), _row("beta", 0.50, 1),
]
QUIET = _noise(alpha=0.02, beta=0.02)


def test_the_most_regressed_bench_gets_the_one_item():
    board = Board()
    out = board.run(REGRESSED, QUIET)
    assert (out["decision"], out["bench"], out["item_id"]) == ("filed", "beta", 5001)
    assert len(board.filed) == 1 and board.filed[0]["cls"] == "improvement"
    assert board.journal() == [out] and out["event"] == FI.INTAKE_EVENT


def test_the_body_carries_both_values_the_rerun_the_cases_and_the_measured_clause():
    board = Board()
    landing = {"event": "promoted", "ts": NOW - 5 * DAY, "round_id": "SM_x",
               "commit": "deadbeef" * 5, "title": "Change the matcher"}
    board.run(REGRESSED, QUIET, events=[landing])
    body, name = board.filed[0]["body"], board.filed[0]["name"]
    assert "0.500 (n=100, 95% CI [" in body and "0.900 (n=100, 95% CI [" in body
    assert "`deadbeef` SM_x Change the matcher" in body
    assert "python -m scripts.automod.fitness run --once --only beta" in body
    assert FI.NO_CASE_ROWS in body                   # `beta` is not a manifest bench
    # baseline 0.500 plus the 0.020 floor
    assert "1. `beta` is above 0.520 " in body and "on a `fitness run` after landing" in body
    assert not B.is_derived_name(name)


def test_the_recorded_fitness_move_supplies_the_landings():
    board = Board()
    move = {"event": F.MOVE_EVENT, "ts": NOW, "bench": "beta", "to_run": "run-1",
            "from_run": "run-9", "landings": [{"commit": "abc12345ff", "round_id": "SM_m",
                                              "title": "From the move event"}]}
    board.run(REGRESSED, QUIET, events=[move])
    assert "`abc12345` SM_m From the move event" in board.filed[0]["body"]


def test_no_landing_between_is_said_not_left_blank():
    board = Board()
    board.run(REGRESSED, QUIET)
    assert "None recorded in the ledger between the two measurements" in board.filed[0]["body"]


def test_a_lower_is_better_bench_gets_a_below_clause():
    rows = [_row("canary", 0.10, 9, higher=False), _row("canary", 0.40, 1, higher=False)]
    board = Board()
    out = board.run(rows, _noise(canary=0.05))
    assert out["target"] == {"direction": "below", "value": 0.35}
    assert "1. `canary` is below 0.350 " in board.filed[0]["body"]


def test_one_open_item_per_bench_falls_through_to_the_next_candidate():
    board = Board()
    out = board.run(REGRESSED, QUIET, fronts=[(41, {"metric": "beta", "class": "improvement"})])
    assert (out["decision"], out["bench"]) == ("filed", "alpha")
    assert any("beta: open item #41" in s for s in out["skipped"])


def test_the_fitness_bench_key_holds_the_bench_after_triage_dropped_the_metric():
    board = Board()
    fronts = [(41, {FI.FM_BENCH: "beta", "class": "fix"}), (42, {"metric": "alpha"})]
    out = board.run(REGRESSED, QUIET, fronts=fronts)
    assert out["decision"] == "held" and not board.filed
    assert out["why"] == "every bench that moved already has an open item"
    assert board.journal() == [out]                   # a hold is journaled too


def test_at_most_one_item_a_day_counting_the_revert_proposals():
    board = Board()
    assert board.run(REGRESSED, QUIET)["decision"] == "filed"
    again = board.run(REGRESSED, QUIET, now=NOW + 3600)
    assert again["decision"] == "held" and "one item per day: #5001" in again["why"]
    assert len(board.filed) == 1
    other = Board()
    proposal = {"event": FI.CLOSE_EVENT, "ts": NOW - 3600, "verdict": "rejected", "follow_up": 77}
    held = other.run(REGRESSED, QUIET, events=[proposal])
    assert held["decision"] == "held" and "#77" in held["why"]
    # a day later the budget is back
    assert Board().run(REGRESSED, QUIET, events=[{**proposal, "ts": NOW - DAY - 1}]
                       )["decision"] == "filed"


@pytest.mark.parametrize("noise", [
    {"alpha": {"status": "too-noisy", "spread": 0.3}, "beta": {"status": "too-noisy", "spread": 0.3}},
    {},                                                           # never measured
    {"alpha": {"status": "insufficient"}, "beta": {"status": "insufficient"}},
])
def test_a_bench_with_no_usable_floor_is_never_filed_against(noise):
    board = Board()
    out = board.run(REGRESSED, noise, file_lowest=True)
    assert out["decision"] == "held" and not board.filed


def test_an_improvement_or_a_move_inside_noise_files_nothing():
    rows = [_row("alpha", 0.70, 9), _row("alpha", 0.90, 1),      # improved
            _row("beta", 0.90, 9), _row("beta", 0.89, 1)]        # inside the floor
    out = Board().run(rows, QUIET)
    assert out["decision"] == "held"
    assert out["why"] == "no bench regressed outside its noise floor"


def test_the_trailing_best_arm_is_behind_its_own_switch():
    # No 7-day move (the row a week back is as low as today), but 20 days ago it was 0.90.
    rows = [_row("alpha", 0.90, 20), _row("alpha", 0.70, 8), _row("alpha", 0.70, 1)]
    off = Board()
    assert off.run(rows, QUIET)["decision"] == "held" and not off.filed
    on = Board()
    out = on.run(rows, QUIET, file_lowest=True)
    assert (out["decision"], out["kind"], out["reference_value"]) == ("filed", "below-best", 0.90)
    assert "30-day best 0.900" in on.filed[0]["name"]
    assert "Best of the last 30 days: 0.900" in on.filed[0]["body"]


def test_a_gap_to_the_trailing_best_inside_noise_is_not_a_gap():
    rows = [_row("alpha", 0.72, 20), _row("alpha", 0.70, 8), _row("alpha", 0.70, 1)]
    assert FI.below_best(rows, "alpha", QUIET) is None
    assert FI.below_best(rows, "alpha", {}) is None


def test_a_dry_run_files_nothing_and_journals_nothing():
    board = Board()
    out = board.run(REGRESSED, QUIET, dry_run=True)
    assert out["decision"] == "would-file" and "## Acceptance clauses" in out["body"]
    assert not board.filed and not board.events


# ── the manifest table ───────────────────────────────────────────────────────

def test_every_manifest_bench_says_where_its_failing_cases_are():
    names = {b.name for b in F.load_manifest()}
    assert names == set(FI.CASE_ROWS), (
        "a manifest bench was added or removed: check whether its artifact has per-case "
        "rows and say so in fitness_intake.CASE_ROWS")
    assert FI.case_rows("refusal_detour").startswith(FI.NO_CASE_ROWS)
    assert FI.case_rows("not_a_bench") == FI.NO_CASE_ROWS


# ── the write, on a real (scratch) board ─────────────────────────────────────

@pytest.fixture
def board_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def test_the_filed_item_is_a_draft_the_inflow_brakes_can_see(board_dir):
    out = FI.run(now=NOW, rows=REGRESSED, noise=QUIET, names=["alpha", "beta"], events=[])
    item = B.load_item(next(board_dir.glob(f"{out['item_id']}-*.md")))
    fm = B.front_matter_of(item.path)
    assert item.status == "draft" and FI.FITNESS_SPAWN_TAG in item.tags
    assert (fm["class"], fm["metric"], fm[FI.FM_BENCH]) == ("improvement", "beta", "beta")
    assert B.loop_spawn_tag(item.tags) == FI.FITNESS_SPAWN_TAG   # expiry and the gauge
    assert B.is_loop_spawned(item) and not B.is_quarantined(item)
    assert SY.source_of(item.tags, item.body) == "fitness"       # scorecard row 21's bucket
    # journaled in the real ledger, and the open item now holds its bench
    assert [e["decision"] for e in S.ledger_rows() if e.get("event") == FI.INTAKE_EVENT] == ["filed"]
    assert FI.open_bench_items(FI.open_fronts()) == {"beta": [item.id]}


def test_a_revert_proposal_keeps_its_bench_without_a_metric(board_dir):
    iid = FI.file_item("Propose reverting abc", "body", bench="beta", cls="simplify")
    fm = B.front_matter_of(next(board_dir.glob(f"{iid}-*.md")))
    assert fm["class"] == "simplify" and "metric" not in fm and fm[FI.FM_BENCH] == "beta"
    assert FI.open_bench_items(FI.open_fronts()) == {"beta": [iid]}


# ── the hook in the nightly source ───────────────────────────────────────────

def test_the_switch_ships_off_and_an_off_row_is_unchanged():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    src = cfg["workers"]["sources"]["fitness"]
    assert src["file_items"] is False and src["file_lowest"] is False
    assert SRC.intake_flags(src) == {} and SRC.intake_flags({}) == {}
    assert SRC.intake_flags({"file_items": True}) == {"file_items": True, "file_lowest": False}


def _execute(monkeypatch, payload, output):
    calls = []

    async def fake_cli(argv, timeout):
        return 0, output

    monkeypatch.setattr(SRC, "_run_cli", fake_cli)
    monkeypatch.setattr(SRC, "_cap_seconds", lambda: 4500)
    monkeypatch.setattr(SRC, "_file_items",
                        lambda lowest: calls.append(lowest) or {"decision": "filed", "item_id": 9})
    item = QueueItem(id=1, source="fitness", kind="fitness", priority=85, payload=payload,
                     dedup_key=None, state="claimed", attempts=1, enqueued_at="",
                     claimed_at=None, claimed_by=None, completed_at=None, error=None)
    return asyncio.run(SRC.execute(item)), calls


MEASURED = 'FITNESS_RESULT {"run_id": "r1", "measured": 2, "rows": 2, "not_measured": []}'
NOTHING = 'FITNESS_RESULT {"run_id": "r1", "measured": 0, "rows": 2, "not_measured": ["a: failed"]}'


def test_the_hook_runs_only_when_the_row_carries_the_switch(monkeypatch):
    out, calls = _execute(monkeypatch, {}, MEASURED)
    assert out["status"] == "success" and calls == [] and "intake" not in out["meta"]
    out, calls = _execute(monkeypatch, {"file_items": True, "file_lowest": True}, MEASURED)
    assert calls == [True] and out["meta"]["intake"] == {"decision": "filed", "item_id": 9}
    assert out["summary"].endswith("intake filed #9")


def test_a_run_that_measured_nothing_files_nothing(monkeypatch):
    out, calls = _execute(monkeypatch, {"file_items": True}, NOTHING)
    assert out["status"] == "failed" and calls == []


def test_a_broken_intake_costs_a_warning_not_the_run(monkeypatch):
    def boom(**kw):
        raise RuntimeError("board unreadable")
    monkeypatch.setattr(FI, "run", boom)
    assert SRC._file_items(False) == {"decision": "error", "why": "RuntimeError: board unreadable"}
