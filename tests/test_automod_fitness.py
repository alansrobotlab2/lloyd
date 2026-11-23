"""The behaviour-fitness runner, its noise floors, its moves and its schedule.

Nothing here reaches an engine, the backend, the network or a real bench: every
child process is a fake `runner`, the pool state is a dict, the lock is a stub,
the ledger is a list, and every store is under `tmp_path`. The clock is pinned
far in the past, so a fake artifact written "now" is always newer than the run
that reads it and no node depends on the day the suite ran.

Run: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_automod_fitness.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.automod import fitness as F
from workers.queue import QueueItem, WorkQueue

ROOT = Path(__file__).resolve().parent.parent

#: 2001-09-09. Rows are stamped off this; real file mtimes are always later.
T0 = 1_000_000_000.0
NOW = datetime.fromtimestamp(T0, tz=timezone.utc)
DAY = 86400.0

PAUSED = {"paused": True, "in_flight_count": 0, "kv_gate": {"engaged": False}}
RUNNING = {"paused": False, "in_flight_count": 2, "kv_gate": {"engaged": False}}


class Clock:
    def __init__(self, start: float = T0, step: float = 1.0):
        self.t, self.step = start, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class StubLock:
    def __init__(self):
        self.acquired, self.released = 0, 0

    def __call__(self, wait_s: float):
        self.acquired += 1
        return self

    def release(self):
        self.released += 1


def _bench(name="alpha", *, engine=True, higher=True, timeout=100.0, **kw) -> F.Bench:
    return F.Bench(
        name=name, covers="test", engine=engine, higher_is_better=higher,
        timeout_s=timeout, safety="fake",
        commands=kw.pop("commands", (("{python}", "bench.py", "--out", "{scratch}/out.json"),)),
        read=kw.pop("read", {"from": "file", "path": "{scratch}/out.json",
                             "value": "summary.rate", "k": "summary.k", "n": "summary.n"}),
        **kw)


def _writer(values: dict[str, list], calls: list | None = None):
    """A fake runner: writes `{summary: {rate, k, n}}` to the path after `--out`,
    taking each bench's values in order. A value of None writes nothing; the
    string "timeout" returns no exit code; an int is a bare exit code."""
    def runner(argv, *, env, cwd, timeout):
        if calls is not None:
            calls.append({"argv": list(argv), "env": dict(env), "timeout": timeout})
        out = Path(argv[argv.index("--out") + 1])
        name = out.parent.name.rsplit("-", 1)[0]
        v = values[name].pop(0)
        if v == "timeout":
            return None, "still going"
        if isinstance(v, int) and not isinstance(v, bool):
            return v, "boom\nTraceback: nope"
        if v is not None:
            k, n = v
            out.write_text(json.dumps({"summary": {"rate": k / n if n else None,
                                                   "k": k, "n": n}}), encoding="utf-8")
        return 0, "ok"
    return runner


def _deps(tmp_path, values, *, pool=PAUSED, lock=None, calls=None, moves=None) -> F.Deps:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    pools = pool if callable(pool) else (lambda: pool)
    return F.Deps(pool=pools, health=lambda: {"commit": "backendsha"},
                  commit=lambda: "treesha", runner=_writer(values, calls),
                  clock=Clock(), sleep=lambda s: None, lock=lock or StubLock(),
                  series=tmp_path / "fitness.jsonl", data_root=tmp_path / "data",
                  make_scratch=lambda: scratch,
                  record_moves=moves or (lambda **kw: []))


# ── the manifest ──────────────────────────────────────────────────────────────

def test_the_tracked_manifest_loads_and_fits_the_nightly_budget():
    benches = F.load_manifest()
    names = [b.name for b in benches]
    assert 6 <= len(benches) <= 10
    assert len(set(names)) == len(names)
    assert sum(b.engine_minutes for b in benches) <= 60
    # The timeouts are the hard bound, and the nightly source's cap has to hold
    # them: a run whose benches could each time out is still stopped by budget.
    for b in benches:
        assert b.safety.strip(), b.name
        assert b.timeout_s <= 1800, b.name
        assert 0 < b.noise_max_spread <= 0.25, b.name
        assert b.read.get("value") and b.read.get("n"), b.name
        assert "LLOYD_DATA" not in b.env
    assert {"tool_choice", "injection_canary", "refusal_detour", "skill_match"} <= set(names)
    assert [b.engine for b in benches] == sorted(b.engine for b in benches), (
        "offline benches first: the budget drops whatever is last on a slow night")


def test_every_manifest_command_names_a_script_that_exists_and_no_placeholder_is_unknown():
    known = {"python", "root", "scratch", "data_root"}
    for b in F.load_manifest():
        for argv in b.commands:
            assert argv[0] == "{python}", (b.name, argv)
            assert (ROOT / argv[1]).is_file(), f"{b.name}: {argv[1]} is not in the tree"
        texts = [a for argv in b.commands for a in argv] + list(b.env.values())
        if b.read.get("path"):
            texts.append(str(b.read["path"]))
        for text in texts:
            for name in re.findall(r"\{([a-z_]+)(?::\d+)?\}", text):
                assert name in known | {"days_ago"}, (b.name, text)


def test_the_canary_entry_redirects_its_rows_away_from_the_tracked_corpus():
    """`eval/canary_rows_guard.py` refuses a non-window caller that did not set
    LLOYD_CANARY_ROWS, and a nightly that appended to the tracked corpus would
    dirty the live tree and stop automod."""
    canary = next(b for b in F.load_manifest() if b.name == "injection_canary")
    assert canary.env["LLOYD_CANARY_ROWS"].startswith("{scratch}/")
    assert [argv[2] for argv in canary.commands] == ["run", "grade"]
    assert canary.read["from"] == "stdout"


def _manifest(tmp_path, **override) -> Path:
    entry = {"name": "a", "covers": "x", "engine": False, "safety": "offline",
             "commands": [["{python}", "x.py"]],
             "read": {"from": "file", "path": "{scratch}/o.json", "value": "r", "n": "n"},
             "higher_is_better": True, "timeout_s": 10, "engine_minutes": 1}
    entry.update(override)
    entry = {k: v for k, v in entry.items() if v is not None}
    import yaml
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"schema": 1, "budget_engine_minutes": 5,
                                    "benches": [entry]}), encoding="utf-8")
    return path


@pytest.mark.parametrize("override, needle", [
    ({"env": {"LLOYD_DATA": "/x"}}, "LLOYD_DATA"),
    ({"safety": ""}, "safety"),
    ({"name": "has.dot"}, "must match"),
    ({"read": {"from": "file", "path": "p", "value": "r"}}, "denominator"),
    ({"read": {"from": "socket", "value": "r", "n": "n"}}, "read.from"),
    ({"commands": ["python x.py"]}, "argv lists"),
    ({"timeout_s": 0}, "timeout_s"),
    ({"higher_is_better": None}, "higher_is_better"),
    ({"engine_minutes": 99}, "over the declared budget"),
])
def test_the_loader_refuses_an_entry_it_would_be_unsafe_or_meaningless_to_run(
        tmp_path, override, needle):
    with pytest.raises(F.ManifestError) as exc:
        F.load_manifest(_manifest(tmp_path, **override))
    assert needle in str(exc.value)


def test_select_refuses_an_unknown_bench_name():
    benches = [_bench("alpha"), _bench("beta")]
    assert [b.name for b in F.select(benches, ["beta"])] == ["beta"]
    with pytest.raises(F.ManifestError):
        F.select(benches, ["gamma"])


# ── reading a value ───────────────────────────────────────────────────────────

def test_wilson_matches_the_documented_interval_and_refuses_n_zero():
    lo, hi = F.wilson(10, 20)
    assert (round(lo, 4), round(hi, 4)) == (0.2993, 0.7007)
    lo, hi = F.wilson(20, 20)
    assert round(lo, 4) == 0.8389 and hi == 1.0
    assert F.wilson(0, 0) is None


def test_read_value_takes_rate_k_and_n_and_never_reads_nothing_as_zero():
    b = _bench()
    got = F.read_value(b, {"summary": {"rate": 0.75, "k": 15, "n": 20}})
    assert got["status"] == "measured" and got["value"] == 0.75 and got["n"] == 20
    assert got["k"] == 15 and got["ci_low"] < 0.75 < got["ci_high"]
    assert got["ci_kind"] == "wilson"

    # Counts only: the rate is k / n.
    assert F.read_value(b, {"summary": {"k": 5, "n": 20}})["value"] == 0.25
    # n of zero, a missing rate, and a non-share are each `no-value` with value None.
    for doc in ({"summary": {"rate": None, "k": 0, "n": 0}},
                {"summary": {"n": 20}}, {"summary": {"rate": 17.6, "n": 20}}, {}):
        got = F.read_value(b, doc)
        assert got["status"] == "no-value" and got["value"] is None, doc


def test_a_mean_of_fractions_says_its_interval_is_approximate():
    b = _bench(binomial=False, read={"from": "file", "path": "p", "value": "r@5", "n": "n"})
    got = F.read_value(b, {"r@5": 0.6863, "n": 51})
    assert got["status"] == "measured" and got["k"] is None
    assert got["ci_kind"].startswith("wilson-approx")


def test_trailing_json_reads_the_object_after_the_prose():
    text = "episodes 20  attacks 2/16\n\narm table {x}\n\n" + json.dumps(
        {"attack_success_rate": 0.125, "attack_success": 2, "attack_scenarios": 16,
         "arms": {"enforce-off": {"rates": {}}}}, indent=2)
    assert F.trailing_json(text)["attack_scenarios"] == 16
    assert F.trailing_json("no json here") is None


def test_expand_fills_placeholders_and_dates_from_the_run_clock():
    ctx = {"scratch": "/s", "python": "/py"}
    assert F.expand("{scratch}/x-{days_ago:30}.json", ctx, now=NOW) == "/s/x-2001-08-10.json"


# ── the pool precondition ─────────────────────────────────────────────────────

def test_the_hand_run_needs_a_paused_idle_pool():
    assert F.pool_precondition(PAUSED, mode="paused")[0] is True
    ok, why = F.pool_precondition(RUNNING, mode="paused")
    assert ok is False and "not paused" in why and "--force" in why
    ok, why = F.pool_precondition({"paused": True, "in_flight_count": 1}, mode="paused")
    assert ok is False and "still in flight" in why
    ok, why = F.pool_precondition(None, mode="paused")
    assert ok is False and "unreadable" in why
    assert F.pool_precondition(None, mode="force") == (True, "forced")


def test_the_pool_job_mode_allows_only_its_own_row_and_respects_the_kv_gate():
    own = {"paused": False, "in_flight_count": 1, "kv_gate": {"engaged": False}}
    assert F.pool_precondition(own, mode="pool-job")[0] is True
    assert F.pool_precondition(RUNNING, mode="pool-job") == (False, "1 other job(s) in flight")
    hot = {"paused": False, "in_flight_count": 1, "kv_gate": {"engaged": True}}
    assert F.pool_precondition(hot, mode="pool-job") == (False, "the KV gate is engaged")


def test_pool_state_reads_the_pool_block_of_the_status_route():
    seen = []

    def fetch(url):
        seen.append(url)
        return {"initialized": True, "pool": {"paused": True, "in_flight_count": 0}}

    assert F.pool_state(fetch) == {"paused": True, "in_flight_count": 0}
    assert seen == ["http://127.0.0.1:8080/api/workers/status"]
    assert F.pool_state(lambda url: None) is None
    assert F.pool_state(lambda url: {"initialized": False}) is None


def test_the_lock_is_the_regression_runners_own():
    from scripts.automod import state as S
    from workers.sources import automod_regression as AR
    assert F.lock_path() == S.STATE_DIR / AR.REGRESSION_LOCK
    assert F.lock_path().name == "regression.lock"


# ── run ───────────────────────────────────────────────────────────────────────

def test_a_run_appends_one_row_per_bench_then_one_summary_row(tmp_path):
    calls: list = []
    lock = StubLock()
    deps = _deps(tmp_path, {"alpha": [(15, 20)], "beta": [(3, 30)]}, lock=lock, calls=calls)
    out = F.execute_run([_bench("alpha"), _bench("beta", higher=False)], now=NOW, deps=deps)

    assert out["exit"] == F.EXIT_OK and out["refused"] is None
    rows = F.read_rows(deps.series)
    assert [r["kind"] for r in rows] == ["bench", "bench", "summary"]
    a, b, summary = rows
    assert (a["bench"], a["value"], a["n"], a["k"]) == ("alpha", 0.75, 20, 15)
    assert a["ci_low"] < 0.75 < a["ci_high"]
    assert a["commit"] == "treesha" and a["backend_commit"] == "backendsha"
    assert a["run_id"] == b["run_id"] == summary["run_id"] == out["run_id"]
    assert a["duration_s"] > 0 and a["status"] == "measured" and a["mode"] == "paused"
    assert a["pool"] == {"paused": True, "in_flight": 0}
    assert b["higher_is_better"] is False

    # Sequential, in manifest order, under one lock taken once and released once.
    assert [Path(c["argv"][1]).name for c in calls] == ["bench.py", "bench.py"]
    assert [Path(c["argv"][3]).parent.name for c in calls] == ["alpha-1", "beta-1"]
    assert (lock.acquired, lock.released) == (1, 1)

    # The summary row is the shape the instrument index's rate reader takes.
    assert summary["benches"]["alpha"]["rate"] == 0.75
    assert summary["benches"]["alpha"]["n"] == 20
    assert set(summary["benches"]["beta"]) >= {"rate", "n", "k", "ci_low", "ci_high"}
    assert rows[-1] is summary, "the summary is the LAST row: the index reads the newest"


def test_repeat_runs_each_bench_n_times_and_only_one_summary(tmp_path):
    deps = _deps(tmp_path, {"alpha": [(10, 20), (12, 20), (11, 20)]})
    F.execute_run([_bench("alpha")], repeat=3, now=NOW, deps=deps)
    rows = F.read_rows(deps.series)
    assert [r.get("rep") for r in rows if r["kind"] == "bench"] == [1, 2, 3]
    assert sum(r["kind"] == "summary" for r in rows) == 1
    assert rows[-1]["benches"]["alpha"]["rate"] == 0.55, "the newest repeat"


def test_an_unpaused_pool_refuses_before_the_lock_and_before_any_bench(tmp_path):
    calls: list = []
    lock = StubLock()
    deps = _deps(tmp_path, {"alpha": [(1, 2)]}, pool=RUNNING, lock=lock, calls=calls)
    out = F.execute_run([_bench("alpha")], now=NOW, deps=deps)
    assert out["exit"] == F.EXIT_REFUSED and "not paused" in out["refused"]
    assert calls == [] and lock.acquired == 0
    assert not deps.series.exists(), "a refused run appends nothing"


def test_force_runs_on_an_unpaused_pool_and_stamps_every_row(tmp_path):
    deps = _deps(tmp_path, {"alpha": [(1, 2)]}, pool=RUNNING)
    out = F.execute_run([_bench("alpha")], mode="force", now=NOW, deps=deps)
    assert out["exit"] == F.EXIT_OK
    assert F.read_rows(deps.series)[0]["mode"] == "force"


def test_offline_benches_need_neither_the_pool_nor_the_lock(tmp_path):
    lock = StubLock()
    polled = []
    deps = _deps(tmp_path, {"alpha": [(4, 5)]}, lock=lock,
                 pool=lambda: polled.append(1) or RUNNING)
    out = F.execute_run([_bench("alpha", engine=False)], now=NOW, deps=deps)
    assert out["exit"] == F.EXIT_OK
    assert lock.acquired == 0 and polled == []


def test_a_held_lock_exits_3_and_runs_nothing(tmp_path):
    calls: list = []

    def held(wait_s):
        raise RuntimeError("self-modification lock held: regression-runner-1")

    deps = _deps(tmp_path, {"alpha": [(1, 2)]}, lock=held, calls=calls)
    out = F.execute_run([_bench("alpha")], now=NOW, deps=deps)
    assert out["exit"] == F.EXIT_LOCK_HELD and "regression.lock is held" in out["refused"]
    assert calls == [] and not deps.series.exists()


def test_a_bench_that_did_not_measure_has_a_status_and_no_value(tmp_path):
    deps = _deps(tmp_path, {"t": ["timeout"], "c": [2], "m": [None], "z": [(0, 0)],
                            "ok": [(9, 10)]})
    out = F.execute_run([_bench(n) for n in ("t", "c", "m", "z", "ok")], now=NOW, deps=deps)
    assert out["exit"] == F.EXIT_SOME_FAILED
    rows = {r["bench"]: r for r in F.read_rows(deps.series) if r["kind"] == "bench"}
    assert rows["t"]["status"] == "timeout" and "killed" in rows["t"]["detail"]
    assert rows["c"]["status"] == "failed" and "exit 2" in rows["c"]["detail"]
    assert rows["m"]["status"] == "no-artifact"
    assert rows["z"]["status"] == "no-value"
    for name in ("t", "c", "m", "z"):
        assert rows[name]["value"] is None, f"{name}: a failed bench must never read as 0"
    assert rows["ok"]["value"] == 0.9, "one bench failing does not stop the next"
    summary = F.read_rows(deps.series)[-1]
    assert set(summary["benches"]) == {"ok"}, "an unmeasured bench is absent, not zero"


def test_an_artifact_older_than_the_run_is_some_other_runs_number(tmp_path):
    target = tmp_path / "old.json"
    target.write_text(json.dumps({"summary": {"rate": 1.0, "k": 5, "n": 5}}), encoding="utf-8")
    os.utime(target, (T0 - 500, T0 - 500))
    b = _bench(commands=(("{python}", "x.py"),),
               read={"from": "file", "path": str(target), "value": "summary.rate",
                     "n": "summary.n"})
    got = F.run_bench(b, scratch=tmp_path, now=NOW, data_root=tmp_path,
                      runner=lambda argv, **kw: (0, ""), clock=Clock())
    assert got["status"] == "no-artifact" and got["value"] is None


def test_a_glob_artifact_reads_the_newest_file_this_run_wrote(tmp_path):
    store = tmp_path / "tool-choice"
    store.mkdir()
    stale = store / "fitness-20010101-000000.json"
    stale.write_text(json.dumps({"s": {"r": 0.1, "n": 20}}), encoding="utf-8")
    os.utime(stale, (T0 - 500, T0 - 500))

    def runner(argv, **kw):
        (store / "fitness-20010909-014640.json").write_text(
            json.dumps({"s": {"r": 0.8, "n": 20}}), encoding="utf-8")
        return 0, ""

    b = _bench(commands=(("{python}", "x.py"),),
               read={"from": "file", "path": "{data_root}/tool-choice/fitness-*.json",
                     "value": "s.r", "n": "s.n"})
    got = F.run_bench(b, scratch=tmp_path, now=NOW, data_root=tmp_path, runner=runner,
                      clock=Clock())
    assert got["status"] == "measured" and got["value"] == 0.8


def test_a_stdout_bench_runs_its_commands_in_order_with_the_declared_env(tmp_path):
    calls: list = []

    def runner(argv, *, env, cwd, timeout):
        calls.append((argv[2], env.get("LLOYD_CANARY_ROWS"), "LLOYD_DATA" in env))
        return 0, "table\n\n" + json.dumps({"rate": 0.25, "k": 4, "n": 16}, indent=2)

    b = _bench(commands=(("{python}", "c.py", "run"), ("{python}", "c.py", "grade")),
               env={"LLOYD_CANARY_ROWS": "{scratch}/rows.jsonl"}, higher=False,
               read={"from": "stdout", "value": "rate", "k": "k", "n": "n"})
    got = F.run_bench(b, scratch=tmp_path, now=NOW, data_root=tmp_path, runner=runner,
                      clock=Clock(), base_env={"PATH": "/usr/bin"})
    assert got["status"] == "measured" and got["value"] == 0.25 and got["n"] == 16
    assert calls == [("run", f"{tmp_path}/rows.jsonl", False),
                     ("grade", f"{tmp_path}/rows.jsonl", False)]


def test_a_failed_first_command_stops_the_bench_before_the_second(tmp_path):
    seen = []

    def runner(argv, **kw):
        seen.append(argv[2])
        return 2, "arm refused: sandbox not enforced"

    b = _bench(commands=(("{python}", "c.py", "run"), ("{python}", "c.py", "grade")),
               read={"from": "stdout", "value": "rate", "n": "n"})
    got = F.run_bench(b, scratch=tmp_path, now=NOW, data_root=tmp_path, runner=runner,
                      clock=Clock())
    assert seen == ["run"] and got["status"] == "failed"
    assert "sandbox not enforced" in got["detail"] and got["value"] is None


def test_an_accepted_nonzero_exit_still_reads_the_artifact(tmp_path):
    deps = _deps(tmp_path, {"alpha": [(8, 10)]})
    inner = deps.runner
    deps.runner = lambda argv, **kw: (1, inner(argv, **kw)[1])
    F.execute_run([_bench("alpha", ok_exit_codes=(0, 1))], now=NOW, deps=deps)
    assert F.read_rows(deps.series)[0]["value"] == 0.8


def test_the_pool_job_skips_a_bench_it_cannot_get_a_quiet_engine_for(tmp_path):
    states = iter([RUNNING,                                   # the opening read
                   {"paused": False, "in_flight_count": 3},   # alpha: busy, no wait
                   {"paused": False, "in_flight_count": 1}])  # beta: only this job
    deps = _deps(tmp_path, {"alpha": [(1, 2)], "beta": [(1, 2)]}, pool=lambda: next(states))
    out = F.execute_run([_bench("alpha"), _bench("beta")], mode="pool-job", now=NOW,
                        deps=deps, quiet_wait_s=0.0)
    rows = {r["bench"]: r for r in out["rows"]}
    assert rows["alpha"]["status"] == "skipped" and rows["alpha"]["value"] is None
    assert "2 other job(s) in flight" in rows["alpha"]["detail"]
    assert rows["beta"]["status"] == "measured"
    assert rows["beta"]["pool"] == {"paused": False, "in_flight": 1}


def test_the_budget_stops_new_benches_instead_of_being_killed_mid_bench(tmp_path):
    deps = _deps(tmp_path, {"alpha": [(1, 2)], "beta": [(1, 2)]})
    deps.clock = Clock(step=40.0)
    out = F.execute_run([_bench("alpha", timeout=100), _bench("beta", timeout=100)],
                        now=NOW, deps=deps, budget_s=250.0)
    by = {r["bench"]: r for r in out["rows"]}
    assert by["alpha"]["status"] == "measured"
    assert by["beta"]["status"] == "skipped" and "budget" in by["beta"]["detail"]


def test_a_redirected_series_never_reaches_the_live_ledger(tmp_path, monkeypatch):
    """`record_moves` defaults to the live ledger, so a run whose series was
    redirected must not call it unless the caller named a recorder."""
    called = []
    monkeypatch.setattr(F, "record_moves", lambda **kw: called.append(kw) or [])
    deps = _deps(tmp_path, {"alpha": [(1, 2)]})
    deps.record_moves = None
    F.execute_run([_bench("alpha")], now=NOW, deps=deps)
    assert called == []


def test_the_summary_carries_a_recent_bench_forward_and_drops_an_old_one():
    def row(bench, ts, value):
        return {"kind": "bench", "bench": bench, "status": "measured", "value": value,
                "n": 20, "k": None, "ts": ts, "run_id": f"r{ts}",
                "recorded_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()}

    rows = [row("alpha", T0 - 2 * 3600, 0.7), row("beta", T0 - 3 * DAY, 0.4)]
    s = F.summary_row(rows, ["alpha", "beta"], run_id="now", now=NOW, commit="c")
    assert s["benches"]["alpha"]["rate"] == 0.7, "an --only run must not blank the others"
    assert "beta" not in s["benches"], "a bench not measured for 3 d is no-value, not carried"


# ── noise ─────────────────────────────────────────────────────────────────────

def test_noise_of_records_min_max_stdev_and_flags_a_wide_spread():
    quiet = F.noise_of([0.70, 0.72, 0.71, 0.70, 0.72], max_spread=0.05)
    assert quiet["status"] == "ok" and quiet["repeats"] == 5
    assert (quiet["min"], quiet["max"], quiet["spread"]) == (0.70, 0.72, 0.02)
    assert quiet["stdev"] == pytest.approx(0.01, abs=1e-3)

    loud = F.noise_of([0.50, 0.80, 0.65], max_spread=0.05)
    assert loud["status"] == "too-noisy" and loud["spread"] == 0.3

    thin = F.noise_of([0.5, 0.6], max_spread=0.05)
    assert thin["status"] == "insufficient" and thin["spread"] is None


def test_the_noise_command_writes_floors_and_adds_nothing_to_the_series(tmp_path):
    noise_file = tmp_path / "fitness_noise.json"
    deps = _deps(tmp_path, {"alpha": [(14, 20), (15, 20), (14, 20), (15, 20), (14, 20)],
                            "beta": [(2, 20), (12, 20), "timeout", (4, 20), (6, 20)]})
    out = F.execute_noise([_bench("alpha", noise_max_spread=0.10),
                           _bench("beta", noise_max_spread=0.10)],
                          repeat=5, now=NOW, deps=deps, noise_file=noise_file)
    assert out["refused"] is None
    assert not deps.series.exists(), "five repeats on one night are not a trend"
    doc = json.loads(noise_file.read_text(encoding="utf-8"))
    assert doc["alpha"]["status"] == "ok" and doc["alpha"]["spread"] == 0.05
    assert doc["alpha"]["commit"] == "treesha" and len(doc["alpha"]["duration_s"]) == 5
    assert doc["beta"]["status"] == "too-noisy"
    assert doc["beta"]["repeats"] == 4 and doc["beta"]["attempted"] == 5
    assert doc["beta"]["failures"] and doc["beta"]["failures"][0].startswith("timeout")

    # A later run of one bench keeps the other's floor.
    deps2 = _deps(tmp_path, {"alpha": [(10, 20), (10, 20), (10, 20)]})
    F.execute_noise([_bench("alpha")], repeat=3, now=NOW, deps=deps2, noise_file=noise_file)
    doc = json.loads(noise_file.read_text(encoding="utf-8"))
    assert doc["alpha"]["spread"] == 0.0 and doc["beta"]["status"] == "too-noisy"


# ── deltas and moves ──────────────────────────────────────────────────────────

def _row(bench, days_ago, value, *, higher=True, run=None, commit=None, status="measured",
         n=20):
    ts = T0 - days_ago * DAY
    lo, hi = F.wilson(value * n, n) if status == "measured" else (None, None)
    return {"kind": "bench", "bench": bench, "status": status,
            "value": value if status == "measured" else None, "n": n,
            "ci_low": lo, "ci_high": hi, "higher_is_better": higher,
            "ts": ts, "recorded_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "run_id": run or f"run-{days_ago:g}", "commit": commit or f"sha-{days_ago:g}"}


OK_NOISE = {"alpha": {"status": "ok", "spread": 0.05},
            "beta": {"status": "ok", "spread": 0.05},
            "loud": {"status": "too-noisy", "spread": 0.4}}


def test_moves_reports_only_deltas_outside_the_noise_floor():
    rows = [_row("alpha", 8, 0.80), _row("alpha", 0, 0.60),                 # -0.20, regressed
            _row("beta", 8, 0.30, higher=False), _row("beta", 0, 0.28, higher=False),
            _row("loud", 8, 0.20), _row("loud", 0, 0.90),                   # too-noisy
            _row("nofloor", 8, 0.20), _row("nofloor", 0, 0.90),             # no floor
            _row("young", 2, 0.20), _row("young", 0, 0.90)]                 # no 7 d reference
    names = ["alpha", "beta", "loud", "nofloor", "young", "never"]
    ds = {d["bench"]: d for d in F.deltas(T0, rows=rows, noise=OK_NOISE, names=names)}

    assert ds["alpha"]["delta"] == -0.2 and ds["alpha"]["outside_noise"] is True
    assert ds["alpha"]["direction"] == "regressed"
    assert ds["beta"]["delta"] == -0.02 and ds["beta"]["outside_noise"] is False
    assert ds["beta"]["direction"] == "improved", "down is better for this bench"
    assert ds["loud"]["outside_noise"] is None and "too-noisy" in ds["loud"]["why"]
    assert ds["nofloor"]["outside_noise"] is None and "noise floor" in ds["nofloor"]["why"]
    assert ds["young"]["delta"] is None and "7 d older" in ds["young"]["why"]
    assert ds["never"]["latest"] is None and ds["never"]["why"].startswith("unmeasured")

    moved = F.moves(T0, rows=rows, noise=OK_NOISE, names=names)
    assert [m["bench"] for m in moved] == ["alpha"]
    assert moved[0]["reference"]["commit"] == "sha-8" and moved[0]["latest"]["commit"] == "sha-0"


def test_a_delta_the_latest_interval_covers_is_not_a_move_even_past_the_spread():
    """A deterministic bench on a fixed set measures a spread of zero, and a bench
    over live traffic measures a spread that says nothing about next week's
    traffic. So the spread alone is not the floor: the latest row's own Wilson
    interval has to exclude the old value too."""
    zero = {"alpha": {"status": "ok", "spread": 0.0}}
    small = [_row("alpha", 8, 0.80, n=20), _row("alpha", 0, 0.70, n=20)]
    d = F.bench_delta(small, "alpha", zero, now=T0)
    assert d["delta"] == -0.1 and d["outside_noise"] is False
    assert "interval covers" in d["why"]
    assert F.moves(T0, rows=small, noise=zero, names=["alpha"]) == []

    # The same ten points over a corpus large enough to resolve them is a move.
    large = [_row("alpha", 8, 0.80, n=400), _row("alpha", 0, 0.70, n=400)]
    assert F.bench_delta(large, "alpha", zero, now=T0)["outside_noise"] is True


def test_moves_is_pure_given_its_inputs():
    rows = [_row("alpha", 8, 0.80), _row("alpha", 0, 0.60)]
    before = json.dumps(rows, sort_keys=True)
    one = F.moves(T0, rows=rows, noise=OK_NOISE, names=["alpha"])
    two = F.moves(T0, rows=rows, noise=OK_NOISE, names=["alpha"])
    assert one == two and json.dumps(rows, sort_keys=True) == before


def test_the_reference_is_the_newest_row_at_least_seven_days_older_and_skips_failures():
    rows = [_row("alpha", 20, 0.10), _row("alpha", 9, 0.50),
            _row("alpha", 8, None, status="timeout"), _row("alpha", 3, 0.99),
            _row("alpha", 0, 0.70)]
    d = F.bench_delta(rows, "alpha", OK_NOISE, now=T0)
    assert d["reference"]["value"] == 0.50 and d["delta"] == 0.2
    assert d["direction"] == "improved"


# ── attribution ───────────────────────────────────────────────────────────────

def _series(tmp_path, rows) -> Path:
    path = tmp_path / "fitness.jsonl"
    F.append_rows(rows, path)
    return path


LEDGER = [
    {"event": "promoted", "ts": T0 - 9 * DAY, "round_id": "SM_old", "commit": "c0", "title": "before"},
    {"event": "promoted", "ts": T0 - 5 * DAY, "round_id": "SM_a", "commit": "c1", "title": "first"},
    {"event": "gate", "ts": T0 - 4 * DAY, "round_id": "SM_b"},
    {"event": "promoted", "ts": T0 - 1 * DAY, "round_id": "SM_b", "commit": "c2", "title": "second"},
    {"event": "promoted", "ts": T0 + 1 * DAY, "round_id": "SM_new", "commit": "c3", "title": "after"},
]


def test_a_regression_outside_noise_is_recorded_with_the_landings_between(tmp_path):
    series = _series(tmp_path, [_row("alpha", 8, 0.80), _row("alpha", 0, 0.60),
                                _row("beta", 8, 0.30, higher=False),
                                _row("beta", 0, 0.10, higher=False)])   # improved 0.20
    written: list = []
    out = F.record_moves(now=NOW, series=series, noise=OK_NOISE, names=["alpha", "beta"],
                         events=LEDGER, append=written.append)
    assert out == written and len(written) == 1, "an improvement is not attributed"
    e = written[0]
    assert e["event"] == "fitness_move" == F.MOVE_EVENT
    assert (e["bench"], e["delta"], e["direction"]) == ("alpha", -0.2, "regressed")
    assert (e["from_commit"], e["to_commit"]) == ("sha-8", "sha-0")
    assert [l["round_id"] for l in e["landings"]] == ["SM_a", "SM_b"]
    assert [l["commit"] for l in e["landings"]] == ["c1", "c2"]
    assert e["noise_spread"] == 0.05 and e["value"] == 0.60 and e["reference_value"] == 0.80
    assert "no bisect, no rollback" in e["action"]


def test_the_same_move_is_recorded_once(tmp_path):
    series = _series(tmp_path, [_row("alpha", 8, 0.80), _row("alpha", 0, 0.60)])
    ledger = list(LEDGER)
    for _ in range(2):
        F.record_moves(now=NOW, series=series, noise=OK_NOISE, names=["alpha"],
                       events=ledger, append=ledger.append)
    assert sum(e.get("event") == "fitness_move" for e in ledger) == 1


def test_nothing_is_recorded_inside_the_noise_floor_or_without_one(tmp_path):
    series = _series(tmp_path, [_row("alpha", 8, 0.80), _row("alpha", 0, 0.78),
                                _row("nofloor", 8, 0.9), _row("nofloor", 0, 0.1)])
    written: list = []
    F.record_moves(now=NOW, series=series, noise=OK_NOISE, names=["alpha", "nofloor"],
                   events=LEDGER, append=written.append)
    assert written == []


def test_recording_a_move_only_appends_a_ledger_event():
    """The attribution scaffold records. Its source names no rollback request, no
    bisect and no round command."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(F))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("rollback_request", "request_rollback", "spawn_detached", "bisect"):
        assert banned not in names, banned


# ── report and the scorecard row ──────────────────────────────────────────────

def test_the_report_says_unmeasured_rather_than_zero():
    text = F.render_report(F.deltas(T0, rows=[], noise={}, names=["alpha"]))
    line = next(l for l in text.splitlines() if l.startswith("alpha"))
    assert "unmeasured" in line and "0.000" not in line


def test_the_report_names_a_move_and_an_in_noise_delta():
    rows = [_row("alpha", 8, 0.80), _row("alpha", 0, 0.60),
            _row("beta", 8, 0.30, higher=False), _row("beta", 0, 0.28, higher=False)]
    text = F.render_report(F.deltas(T0, rows=rows, noise=OK_NOISE, names=["alpha", "beta"]))
    alpha = next(l for l in text.splitlines() if l.startswith("alpha"))
    beta = next(l for l in text.splitlines() if l.startswith("beta"))
    assert "MOVED (regressed)" in alpha and "-0.200" in alpha
    assert "inside the noise floor" in beta


def test_scorecard_row_22_is_unmeasured_with_no_rows_and_never_zero():
    from scripts.automod import scorecard as SC
    empty = F.scorecard_block(T0, rows=[], noise={}, names=["alpha", "beta"])
    assert empty["measured"] is False
    for block in ({}, None, empty, {"measured": False}):
        (line,) = SC._render_fitness_row({"fitness": block})
        assert line.startswith("| 22 | fitness | unmeasured |"), line
        assert " 0" not in line
    assert SC._render_fitness_row({}) == SC._render_fitness_row({"fitness": {}}), (
        "a row recorded before row 22 existed renders the same way")


def test_scorecard_row_22_carries_each_bench_value_and_its_seven_day_delta():
    from scripts.automod import scorecard as SC
    rows = [_row("alpha", 8, 0.80), _row("alpha", 0, 0.60),
            _row("beta", 8, 0.30, higher=False), _row("beta", 0, 0.28, higher=False),
            _row("young", 0, 0.5)]
    block = F.scorecard_block(T0, rows=rows, noise=OK_NOISE,
                              names=["alpha", "beta", "young", "never"])
    assert block["measured"] is True and block["moved"] == ["alpha"]
    (line,) = SC._render_fitness_row({"fitness": block})
    assert line.startswith("| 22 | fitness | 3 of 4 benches, 1 moved |")
    assert "alpha 0.600 (n 20; -0.200 over 7 d, OUTSIDE noise (regressed))" in line
    assert "beta 0.280 (n 20; -0.020 over 7 d, inside noise)" in line
    assert "young 0.500 (n 20; no 7 d delta yet)" in line
    assert "never unmeasured" in line


def test_the_scorecard_renders_row_22_last_and_computes_the_block(monkeypatch):
    from scripts.automod import scorecard as SC
    monkeypatch.setattr(F, "scorecard_block", lambda now: {"measured": False, "benches": []})
    assert SC._fitness(T0) == {"measured": False, "benches": []}

    def boom(now):
        raise OSError("series unreadable")

    monkeypatch.setattr(F, "scorecard_block", boom)
    assert SC._fitness(T0) == {}, "the scorecard never fails on this row"
    src = (ROOT / "scripts" / "automod" / "scorecard.py").read_text(encoding="utf-8")
    assert '"fitness": _fitness(now)' in src
    assert "_render_growth_rows(row) + _render_fitness_row(row)" in src


# ── report-only ───────────────────────────────────────────────────────────────

def test_the_stores_hang_off_the_data_root():
    from app import paths
    assert F.series_path() == paths.FITNESS_SERIES == paths.DATA_ROOT / "eval" / "fitness.jsonl"
    assert F.noise_path() == paths.FITNESS_NOISE_PATH == paths.DATA_ROOT / "eval" / "fitness_noise.json"


def test_no_gate_promoter_or_guardian_reads_the_fitness_stores():
    """Report-only, like the index it feeds: the series and the noise file are
    read by `fitness report`, the scorecard and the instrument index — never by a
    path that gates, lands or rolls back."""
    needles = ("FITNESS_SERIES", "FITNESS_NOISE", "fitness.jsonl", "fitness_noise",
               "automod.fitness", "import fitness", "fitness_move")
    files = [ROOT / "scripts" / "automod" / n for n in
             ("gate.py", "promote.py", "round.py", "spec.py", "review.py")]
    files += sorted((ROOT / "agent-services" / "guardian").glob("*.py"))
    files += [ROOT / "workers" / "sources" / "automod_regression.py"]
    read = 0
    for path in files:
        if not path.is_file():
            continue
        read += 1
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, f"{path.relative_to(ROOT)} names {needle}"
    assert read >= 6, "the scan opened too few files to mean anything"


# ── the nightly source ────────────────────────────────────────────────────────

def _source():
    import workers.sources as sources
    return sources.SOURCE_REGISTRY["fitness"]


def _queued(q: WorkQueue) -> list[dict]:
    with q._connect() as conn:
        rows = conn.execute("SELECT id, payload_json FROM queue WHERE source=?",
                            ("fitness",)).fetchall()
    return [dict(r) for r in rows]


def _item() -> QueueItem:
    return QueueItem(id=1, source="fitness", kind="fitness", priority=85, payload={},
                     state="running", attempts=1, enqueued_at="", claimed_at=None,
                     claimed_by=None, not_before=None, dedup_key=None,
                     completed_at=None, error=None)


def test_the_source_is_registered_long_lived_and_ships_disabled():
    import yaml
    src = _source()
    assert src.NAME == "fitness" and src.LONG_LIVED is True
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    block = cfg["workers"]["sources"]["fitness"]
    assert block["enabled"] is False, "the kill switch ships off until floors are measured"
    assert block["max_inflight"] == 1 and "inner_voice" not in block
    assert block["window_start_hour"] == src.DEFAULT_WINDOW_START_HOUR
    assert block["window_end_hour"] == src.DEFAULT_WINDOW_END_HOUR
    assert block["min_interval_seconds"] == src.DEFAULT_MIN_INTERVAL_SECONDS


def test_the_sources_timers_sit_under_the_pools_cap_with_room_for_the_budget():
    import yaml
    src = _source()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cap = cfg["workers"]["sources"]["fitness"]["max_duration_seconds"]
    timeout, budget = src.timers(cap)
    assert timeout == cap - 60, "the subprocess dies 60 s before the pool's own timer"
    assert budget < timeout
    engine_s = 60 * sum(b.engine_minutes for b in F.load_manifest())
    assert engine_s <= 3600 <= budget, "the 60-minute engine budget fits under the cap"
    assert cfg["workers"]["slots"] == 6, "this change does not touch the pool's slots"


def test_the_source_runs_the_script_in_pool_job_mode_with_its_budget():
    src = _source()
    argv = src.build_argv(4320)
    assert argv[1:6] == ["-m", "scripts.automod.fitness", "run", "--once", "--pool-job"]
    assert argv[argv.index("--budget-seconds") + 1] == "4320"
    assert "--force" not in argv
    # And the script's own parser accepts exactly that line.
    parsed = None
    import argparse
    real = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        nonlocal parsed
        parsed = real(self, args, namespace)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            F.main(argv[3:])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert parsed.cmd == "run" and parsed.pool_job is True and parsed.budget_seconds == 4320.0


def test_enqueue_only_inside_the_quiet_window_and_once_a_night(tmp_path):
    src = _source()
    q = WorkQueue(tmp_path / "fitness.db")
    cfg = {"window_start_hour": 1, "window_end_hour": 4, "min_interval_seconds": 72000}
    t = datetime(2026, 11, 23, 9, 0, tzinfo=timezone.utc)

    asyncio.run(src.enqueue_if_due(q, cfg, local_hour=14, now=t))
    assert _queued(q) == [], "noon is not the quiet window"
    asyncio.run(src.enqueue_if_due(q, cfg, local_hour=4, now=t))
    assert _queued(q) == [], "the end hour is exclusive"

    asyncio.run(src.enqueue_if_due(q, cfg, local_hour=1, now=t))
    rows = _queued(q)
    assert len(rows) == 1
    assert "--pool-job" in json.loads(rows[0]["payload_json"])["command"]

    q.mark_completed(rows[0]["id"])
    asyncio.run(src.enqueue_if_due(q, cfg, local_hour=2, now=t + timedelta(hours=1)))
    assert len(_queued(q)) == 1, "one run a night, held by the watermark"
    asyncio.run(src.enqueue_if_due(q, cfg, local_hour=1, now=t + timedelta(hours=24)))
    assert len(_queued(q)) == 2


def test_the_window_wraps_past_midnight():
    src = _source()
    assert src.in_window(23, 22, 3) and src.in_window(2, 22, 3)
    assert not src.in_window(3, 22, 3) and not src.in_window(12, 22, 3)
    assert not src.in_window(5, 5, 5)


def _fake_cli(monkeypatch, code: int, output: str, seen: dict | None = None):
    src = _source()

    async def fake(argv, timeout):
        if seen is not None:
            seen.update(argv=argv, timeout=timeout)
        return code, output

    monkeypatch.setattr(src, "_run_cli", fake)
    monkeypatch.setattr(src, "_cap_seconds", lambda: 4500)


def test_execute_reports_what_the_run_measured(monkeypatch):
    src = _source()
    seen: dict = {}
    line = 'FITNESS_RESULT ' + json.dumps({"run_id": "r1", "measured": 6, "rows": 7,
                                           "not_measured": ["tool_choice: skipped"]})
    _fake_cli(monkeypatch, 1, "fitness run r1:\n  ...\n" + line + "\n", seen)
    result = asyncio.run(src.execute(_item()))
    assert result["status"] == "success"
    assert "6 of 7 benches measured" in result["summary"]
    assert "tool_choice: skipped" in result["summary"]
    assert result["meta"]["run_id"] == "r1"
    assert seen["timeout"] == 4500 - 60
    assert seen["argv"][seen["argv"].index("--budget-seconds") + 1] == str(4500 - 60 - 120)


def test_execute_returns_a_failure_when_nothing_was_measured(monkeypatch):
    src = _source()
    _fake_cli(monkeypatch, 3, "refused: regression.lock is held (x); nothing was run\n")
    held = asyncio.run(src.execute(_item()))
    assert held["status"] == "failed" and "exit 3" in held["summary"]

    line = 'FITNESS_RESULT ' + json.dumps({"run_id": "r2", "measured": 0, "rows": 7,
                                           "not_measured": ["a: skipped"]})
    _fake_cli(monkeypatch, 1, line)
    none = asyncio.run(src.execute(_item()))
    assert none["status"] == "failed" and "0 of 7" in none["summary"]


def test_execute_returns_a_failure_on_a_timeout_instead_of_raising(monkeypatch):
    src = _source()

    async def slow(argv, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr(src, "_run_cli", slow)
    monkeypatch.setattr(src, "_cap_seconds", lambda: 4500)
    result = asyncio.run(src.execute(_item()))
    assert result["status"] == "failed" and "4440 s" in result["summary"]


def test_the_cli_prints_the_line_the_source_parses(tmp_path, monkeypatch, capsys):
    src = _source()
    deps = _deps(tmp_path, {"alpha": [(3, 4)]})
    monkeypatch.setattr(F, "load_manifest", lambda path=None: [_bench("alpha", engine=False)])
    real = F.execute_run
    monkeypatch.setattr(F, "execute_run", lambda benches, **kw: real(
        benches, **{**kw, "deps": deps, "now": NOW}))
    assert F.main(["run", "--once"]) == F.EXIT_OK
    doc = src.parse_result(capsys.readouterr().out)
    assert doc["measured"] == 1 and doc["rows"] == 1 and doc["not_measured"] == []


def test_a_noise_run_with_too_few_repeats_is_refused(capsys):
    assert F.main(["noise", "--repeat", "2"]) == F.EXIT_REFUSED
    assert "at least 3 repeats" in capsys.readouterr().err
