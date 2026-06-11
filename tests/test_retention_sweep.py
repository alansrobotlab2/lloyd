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
    monkeypatch.setattr(mod, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(mod, "SESSIONS_DIR", tmp_path / "sessions")
    mod.TASKS_DIR.mkdir()
    mod.SESSIONS_DIR.mkdir()
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
