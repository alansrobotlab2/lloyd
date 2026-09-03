"""Groundskeeper retention sweep (scripts/groundskeeper/retention-sweep.py).

Pins the retention contract: old task logs are deleted, stale sessions are
gzipped (round-trip-validated, original removed), and anything younger than
the thresholds is untouched in both dry-run and apply modes.
"""
import gzip
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "groundskeeper" / "retention-sweep.py"


@pytest.fixture
def rs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("retention_sweep", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Redirect EVERY path this module can delete from. Patching only some of
    # them means a test that touches an unpatched root silently operates on real
    # data — sweep_autonomy_runs(apply=True) did exactly that and removed 844
    # live run records during development.
    for attr, sub in (
        ("TASKS_DIR", "tasks"),
        ("SESSIONS_DIR", "sessions"),
        ("AUTONOMY_RUNS_DIR", "autonomy-runs"),
        ("AUTONOMY_TASKS_DIR", "autonomy"),
        ("CANDIDATES_DIR", "skill-candidates"),
    ):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, tmp_path / sub)
            (tmp_path / sub).mkdir(exist_ok=True)
    # Fail loudly if the module grows a new destructive root that this fixture
    # does not cover, rather than letting it reach the real filesystem.
    for name, value in list(vars(mod).items()):
        if name.endswith("_DIR") and isinstance(value, Path):
            assert tmp_path in value.parents or value == tmp_path, (
                f"{name} is not redirected into tmp_path (points at {value}) — "
                f"add it to the fixture before writing tests that touch it")
    return mod


def _backdate(path: Path, days: float) -> None:
    ts = time.time() - days * 86400
    os.utime(path, (ts, ts))


def _write_session(dir: Path, name: str, last_active_days: float) -> Path:
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(days=last_active_days)).isoformat()
    p = dir / f"{name}.json"
    p.write_text(json.dumps({
        "session_id": name,
        "last_active": ts,
        "messages": [{"role": "user", "content": "hi"}],
    }, indent=2))
    return p


def test_old_task_logs_deleted_recent_kept(rs):
    now = time.time()
    old = rs.TASKS_DIR / "bg-old.log"
    old.write_text("x" * 100)
    _backdate(old, 45)
    recent = rs.TASKS_DIR / "bg-recent.log"
    recent.write_text("y" * 100)

    n, freed = rs.sweep_task_logs(apply=False, now=now)
    assert (n, freed) == (1, 100)
    assert old.exists()  # dry run touches nothing

    n, _ = rs.sweep_task_logs(apply=True, now=now)
    assert n == 1
    assert not old.exists()
    assert recent.exists()


def test_stale_session_gzipped_and_roundtrips(rs):
    now = time.time()
    stale = _write_session(rs.SESSIONS_DIR, "stale", last_active_days=120)
    fresh = _write_session(rs.SESSIONS_DIR, "fresh", last_active_days=5)

    n, _ = rs.sweep_sessions(apply=False, now=now)
    assert n == 1
    assert stale.exists()  # dry run touches nothing

    n, _ = rs.sweep_sessions(apply=True, now=now)
    assert n == 1
    assert not stale.exists()
    assert fresh.exists() and not fresh.with_suffix(".json.gz").exists()
    with gzip.open(rs.SESSIONS_DIR / "stale.json.gz", "rt") as fh:
        assert json.load(fh)["session_id"] == "stale"


def test_session_age_prefers_last_active_over_mtime(rs):
    # Recently-rewritten file (fresh mtime) with an old last_active must
    # still count as stale — mtime lies after reprocessing.
    p = _write_session(rs.SESSIONS_DIR, "rewritten", last_active_days=200)
    assert rs._session_age_days(p, time.time()) > 190


def test_run_records_age_by_frontmatter_not_mtime(rs, tmp_path):
    """A bulk operation on 2026-08-22 reset every run record's mtime, which made
    this sweep silently inert — 0 of 3,350 files matched. Age must come from the
    record's own frontmatter."""
    runs = tmp_path / "autonomy-runs" / "24"
    runs.mkdir(parents=True)
    monkey = rs.AUTONOMY_RUNS_DIR
    assert monkey  # fixture wired it

    old = runs / "run_24_20260329_120000.md"
    old.write_text("---\nrun_id: x\ncompleted_at: '2026-03-29T12:00:00+00:00'\n"
                   "status: success\n---\n\nbody\n")
    fresh = runs / "run_24_20260903_120000.md"
    fresh.write_text("---\nrun_id: y\ncompleted_at: '2026-09-03T12:00:00+00:00'\n"
                     "status: success\n---\n\nbody\n")
    # Both look brand-new on disk, exactly like the post-bulk-operation state.
    now = time.time()
    for p in (old, fresh):
        os.utime(p, (now, now))

    count, _ = rs.sweep_autonomy_runs(apply=False, now=now)
    assert count == 1, "the March record should be selected despite a fresh mtime"

    rs.sweep_autonomy_runs(apply=True, now=now)
    assert not old.exists()
    assert fresh.exists()


def test_legacy_epoch_named_records_are_swept(rs, tmp_path):
    """844 records from the pre-2026-03 scheduler were named <epoch_ms>.md and
    were permanently exempt: the glob only matched run_*.md."""
    runs = tmp_path / "autonomy-runs" / "37"
    runs.mkdir(parents=True)
    legacy = runs / "1774837994780.md"
    legacy.write_text("---\nrun_id: 1774837994780\n"
                      "started_at: '2026-03-30T01:40:05'\nstatus: success\n---\n\nx\n")
    keep = runs / "wiki-sweep-latest.json"
    keep.write_text("{}")
    now = time.time()
    os.utime(legacy, (now, now))

    count, _ = rs.sweep_autonomy_runs(apply=True, now=now)
    assert count == 1
    assert not legacy.exists()
    assert keep.exists(), "non-record files must be left alone"
