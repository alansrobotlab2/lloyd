"""The frontend-probe canary has a scheduled runner (#1981).

`workers/sources/frontend_probe_canary.py` is the schedule; the measurement stays in
`scripts/automod/frontend_probe_canary.py`. Nothing here runs the real CLI or
writes the live state dir: `_run_cli` is faked and `scripts.automod.state.STATE_DIR`
points at `tmp_path` in every node that reads an artifact.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from workers.queue import WorkQueue, QueueItem

ROOT = Path(__file__).resolve().parent.parent


def _source():
    import workers.sources as sources
    return sources.SOURCE_REGISTRY["frontend-probe-canary"]


def _queued(q: WorkQueue) -> list[dict]:
    with q._connect() as conn:
        rows = conn.execute(
            "SELECT id, source, kind, payload_json, state FROM queue WHERE source=?",
            ("frontend-probe-canary",)).fetchall()
    return [dict(r) for r in rows]


def _item(payload=None) -> QueueItem:
    src = _source()
    return QueueItem(id=1, source=src.NAME, kind="canary", priority=80,
                     payload=payload or {"command": src.COMMAND}, state="running",
                     attempts=1, enqueued_at="", claimed_at=None,
                     claimed_by=None, not_before=None, dedup_key=None,
                     completed_at=None, error=None)


@pytest.fixture
def state_dir(monkeypatch, tmp_path):
    from scripts.automod import state as S
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setattr(S, "STATE_DIR", d)
    return d


def _fake_cli(monkeypatch, code: int, output: str, artifact: dict | None, state_dir: Path):
    """Replace the subprocess with one that exits `code` and (optionally) writes the
    artifact where the real CLI would — under the redirected STATE_DIR."""
    src = _source()
    seen = {}

    async def fake(argv, timeout):
        seen["argv"] = argv
        if artifact is not None:
            d = state_dir / "frontend_probe_canary"
            d.mkdir(parents=True, exist_ok=True)
            (d / "latest.json").write_text(json.dumps(artifact), encoding="utf-8")
        return code, output

    monkeypatch.setattr(src, "_run_cli", fake)
    return seen


PASSING = {"detected": 12, "seeds_measured": 12, "rate": 1.0,
           "rate_counting_blind_as_misses": 0.857, "passes": True,
           "run_at": "2026-10-01T12:00:00+00:00"}
BELOW = {"detected": 10, "seeds_measured": 12, "rate": 0.8333,
         "rate_counting_blind_as_misses": 0.714, "passes": False,
         "run_at": "2026-10-01T12:00:00+00:00"}


# ── clause 1: registered, one item per interval, the command in the payload ──

def test_the_source_is_registered_by_importing_the_package():
    src = _source()
    assert src.NAME == "frontend-probe-canary"
    assert callable(src.enqueue_if_due) and callable(src.execute)


def test_one_item_per_declared_interval(tmp_path):
    src = _source()
    q = WorkQueue(tmp_path / "canary.db")
    asyncio.run(src.enqueue_if_due(q, {}))
    rows = _queued(q)
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload_json"])
    assert payload["command"] == "python -m scripts.automod.frontend_probe_canary --seeds 12"

    # Inside the interval, with the first item still queued: nothing.
    asyncio.run(src.enqueue_if_due(q, {}))
    assert len(_queued(q)) == 1

    # Inside the interval, with the first item FINISHED (its dedup key released):
    # still nothing — the watermark, not the dedup key, is what holds the interval.
    q.mark_completed(rows[0]["id"])
    asyncio.run(src.enqueue_if_due(q, {}))
    assert len(_queued(q)) == 1, "a finished run let a second one in inside the interval"

    # Past the interval: one more.
    asyncio.run(src.enqueue_if_due(q, {"min_interval_seconds": 0}))
    assert len(_queued(q)) == 2


def test_the_declared_interval_is_a_day_and_the_seed_floor_is_the_whole_table():
    from scripts.automod import frontend_probe_canary as C
    src = _source()
    assert src.DEFAULT_INTERVAL_SECONDS == 86400
    assert src.SEEDS == len(C.must_detect_seeds()), (
        "the scheduled floor no longer equals the must-detect table, so a seed can "
        "drop out of n without failing the run")


def test_config_enables_the_source():
    """`workers/pool.py` skips a source whose config block does not say
    `enabled: true`, so a source registered with no block never runs."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    block = cfg["workers"]["sources"]["frontend-probe-canary"]
    assert block["enabled"] is True
    assert block["max_duration_seconds"] > block["timeout_seconds"]
    assert "inner_voice" not in block


# ── clause 2: the default state dir ──

def test_the_command_carries_no_state_dir_override():
    src = _source()
    argv = src.build_argv()
    assert argv[1:] == ["-m", "scripts.automod.frontend_probe_canary", "--seeds", "12"]
    assert "LLOYD_AUTOMOD_STATE" not in src.COMMAND
    assert not any("LLOYD_AUTOMOD_STATE" in a for a in argv)
    # ...and the child is spawned with no env= at all, so it inherits the pool's.
    import ast
    tree = ast.parse(Path(src.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "create_subprocess_exec"]
    assert len(calls) == 1
    assert "env" not in {kw.arg for kw in calls[0].keywords}


def test_the_run_reports_the_artifact_under_the_state_dir(monkeypatch, state_dir):
    src = _source()
    _fake_cli(monkeypatch, 0, "detected 12/12\n", PASSING, state_dir)
    result = asyncio.run(src.execute(_item()))
    assert result["artifact_path"] == str(state_dir / "frontend_probe_canary" / "latest.json")
    assert not (state_dir / "frontend_probe").exists()


def test_the_existing_canary_tests_still_redirect_the_state_dir():
    """This item adds a runner; it must not make any test write the live dir."""
    text = (ROOT / "tests" / "test_automod_frontend_probe_canary.py").read_text(encoding="utf-8")
    assert text.count('monkeypatch.setattr(S, "STATE_DIR"') >= 3, text.count("STATE_DIR")
    assert "LLOYD_AUTOMOD_STATE" in text


# ── clause 3: not measured / below the bar is never a success ──

def test_exit_2_is_a_failure_naming_not_measured_and_carries_no_rate(monkeypatch, state_dir):
    src = _source()
    # An OLDER passing artifact is on disk. Exit 2 must not quote it.
    d = state_dir / "frontend_probe_canary"
    d.mkdir()
    (d / "latest.json").write_text(json.dumps(PASSING), encoding="utf-8")
    _fake_cli(monkeypatch, 2, "not measured: the probe could not run (no chromium)\n",
              None, state_dir)
    result = asyncio.run(src.execute(_item()))
    assert result["status"] == "failed"
    assert "not measured" in result["summary"]
    for key in ("rate", "detected", "seeds_measured", "rate_counting_blind_as_misses", "meta"):
        assert key not in result, f"{key} reported by a run that measured nothing"


def test_exit_1_is_a_failure_carrying_the_measured_rate(monkeypatch, state_dir):
    src = _source()
    _fake_cli(monkeypatch, 1, "detected 10/12 (83.3%)\n", BELOW, state_dir)
    result = asyncio.run(src.execute(_item()))
    assert result["status"] == "failed"
    assert result["rate"] == BELOW["rate"]
    assert result["detected"] == 10 and result["seeds_measured"] == 12
    assert "below the bar" in result["summary"]


def test_exit_0_without_this_runs_artifact_is_a_failure(monkeypatch, state_dir):
    src = _source()
    d = state_dir / "frontend_probe_canary"
    d.mkdir()
    stale = d / "latest.json"
    stale.write_text(json.dumps(PASSING), encoding="utf-8")
    old = time.time() - 3600
    import os
    os.utime(stale, (old, old))
    _fake_cli(monkeypatch, 0, "", None, state_dir)
    result = asyncio.run(src.execute(_item()))
    assert result["status"] == "failed"
    assert "not measured" in result["summary"] and "rate" not in result


def test_a_crash_or_a_timeout_is_not_measured(monkeypatch, state_dir):
    src = _source()
    _fake_cli(monkeypatch, 139, "Segmentation fault\n", None, state_dir)
    assert asyncio.run(src.execute(_item()))["status"] == "failed"

    async def slow(argv, timeout):
        raise asyncio.TimeoutError()
    monkeypatch.setattr(src, "_run_cli", slow)
    result = asyncio.run(src.execute(_item()))
    assert result["status"] == "failed" and "not measured" in result["summary"]


def test_a_failure_survives_the_pools_normaliser(monkeypatch, state_dir):
    """The pool records `normalize_result(...)`, not the raw dict."""
    from workers.pool import normalize_result
    src = _source()
    _fake_cli(monkeypatch, 2, "not measured: x\n", None, state_dir)
    item = _item()
    norm = normalize_result(item, asyncio.run(src.execute(item)))
    assert norm["status"] == "failed" and "not measured" in norm["summary"]


# ── clause 4: a passing run exposes the numbers ──

def test_a_passing_run_exposes_the_artifact_fields(monkeypatch, state_dir):
    from workers.pool import normalize_result
    src = _source()
    _fake_cli(monkeypatch, 0, "detected 12/12 (100.0%)\n", PASSING, state_dir)
    item = _item()
    result = asyncio.run(src.execute(item))
    assert result["status"] == "success"
    for key in ("detected", "seeds_measured", "rate", "rate_counting_blind_as_misses", "run_at"):
        assert result[key] == PASSING[key], key
    # ...and they reach the run record, so a later job reads them from the runs table.
    meta = normalize_result(item, result)["meta"]
    for key in ("detected", "seeds_measured", "rate", "rate_counting_blind_as_misses", "run_at"):
        assert meta[key] == PASSING[key], key
