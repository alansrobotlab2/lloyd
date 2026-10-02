"""An empty, idle second copy of a runtime store is moved out of the code tree, not
alerted on (`datawatch.inert_residue` / `quarantine_inert`).

On 2026-10-02 a 0-byte `workers.db` — created by `sqlite3` opening a guessed path —
alerted hourly for seven hours and was parked on a human, because a file git ignores
gives an automod round no diff to land. Everything needed to rule on it was in one
`lstat`. Pinned here: the move happens only when every measured condition holds, each
condition that fails leaves the file where it is and alerting, and nothing is deleted
— the file is in the data root's quarantine afterwards, with a line saying what it was.

Each node builds its own tree and data root under `tmp_path`.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datawatch  # noqa: E402

NOW = 1_800_000_000.0
OLD = NOW - 2 * datawatch.INERT_MIN_AGE_SECONDS


@pytest.fixture
def roots(tmp_path):
    """A tree holding an old, empty `workers.db` and a data root holding the real one."""
    tree, data = tmp_path / "lloyd", tmp_path / "lloyd-data"
    tree.mkdir()
    data.mkdir()
    (data / "workers.db").write_bytes(b"SQLite format 3\x00" + b"x" * 100)
    stray = tree / "workers.db"
    stray.write_bytes(b"")
    os.utime(stray, (OLD, OLD))
    return tree, data, stray


def _inert(tree, data, names=("workers.db",)):
    return datawatch.inert_residue(str(tree), list(names), str(data), NOW)


def test_the_incident_file_is_inert_and_is_moved_not_deleted(roots):
    tree, data, stray = roots
    assert _inert(tree, data) == ["workers.db"]

    moved = datawatch.quarantine_inert(str(tree), ["workers.db"], str(data), NOW)

    assert [name for name, _ in moved] == ["workers.db"]
    dest = Path(moved[0][1])
    assert not stray.exists()
    assert dest.is_file() and dest.stat().st_size == 0
    assert dest.parent == data / datawatch.QUARANTINE_SUBDIR
    assert dest.stat().st_mtime == OLD, "the copy keeps the time the stray was written"
    rows = [json.loads(l) for l in (dest.parent / "log.jsonl").read_text().splitlines()]
    assert rows == [{**rows[0], "source": str(stray), "destination": str(dest), "size": 0}]
    assert (data / "workers.db").stat().st_size > 0, "the real store is untouched"


def test_one_byte_of_data_is_not_residue(roots):
    tree, data, stray = roots
    stray.write_bytes(b"x")
    os.utime(stray, (OLD, OLD))
    assert _inert(tree, data) == []
    assert datawatch.quarantine_inert(str(tree), ["workers.db"], str(data), NOW) == []
    assert stray.read_bytes() == b"x"


def test_a_file_just_created_is_left_for_the_next_check(roots):
    tree, data, stray = roots
    os.utime(stray, (NOW - 5, NOW - 5))
    assert _inert(tree, data) == []


def test_a_sqlite_sidecar_means_a_writer_has_it_open(roots):
    tree, data, stray = roots
    (tree / "workers.db-wal").write_bytes(b"")
    assert _inert(tree, data) == []


def test_a_store_with_no_copy_in_the_data_root_might_be_the_only_one(roots):
    tree, data, stray = roots
    (data / "workers.db").unlink()
    assert _inert(tree, data) == []


def test_a_name_nobody_listed_is_a_persons_call(roots):
    tree, data, _ = roots
    other = tree / "scratch.out"
    other.write_bytes(b"")
    os.utime(other, (OLD, OLD))
    (data / "scratch.out").write_bytes(b"x")
    assert _inert(tree, data, ["scratch.out"]) == []


def test_a_directory_and_a_link_are_never_residue(roots):
    tree, data, stray = roots
    stray.unlink()
    (tree / "sessions").mkdir()
    (data / "sessions").mkdir()
    os.symlink(data / "workers.db", tree / "workers.db")
    assert _inert(tree, data, ["sessions", "workers.db"]) == []


# ── through the guardian's hourly check ───────────────────────────────

def _guardian(tmp_path, monkeypatch, tree, data, strays):
    import guardian as gmod
    import notify

    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)
    monkeypatch.setattr(gmod.policy, "REPO", str(tree))
    monkeypatch.setattr(gmod.policy, "DATA_ROOT", str(data))
    monkeypatch.setattr(gmod.datawatch, "stray_in_tree", lambda _tree: list(strays))
    g = gmod.Guardian.__new__(gmod.Guardian)
    g.notifier = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                                 vault_root=str(vault),
                                 backend_url="http://127.0.0.1:1")
    g.data = types.SimpleNamespace(armed=True)
    g._alert_seen = {}
    g.last_alert = ""
    news: list[tuple[str, str]] = []
    g.notifier.announce = lambda title, body="", level="info": news.append((title, body)) or {}
    alerts: list[tuple[str, str]] = []
    real_alert = g.notifier.alert

    def _alert(level, title, body, **kw):
        alerts.append((title, body))
        return real_alert(level, title, body, **kw)

    g.notifier.alert = _alert
    return gmod, g, news, alerts


def test_the_hourly_check_moves_residue_and_raises_no_incident(roots, tmp_path, monkeypatch):
    tree, data, stray = roots
    gmod, g, news, alerts = _guardian(tmp_path, monkeypatch, tree, data, ["workers.db"])
    g._runtime_data_incident(NOW)

    assert not stray.exists()
    assert alerts == [], "an empty second copy is news, not an incident"
    assert len(news) == 1 and str(stray) in news[0][1]
    assert not (tmp_path / "l.jsonl").exists() or "workers.db" not in (
        tmp_path / "l.jsonl").read_text()


def test_what_is_not_residue_still_alerts_and_alone(roots, tmp_path, monkeypatch):
    tree, data, stray = roots
    (tree / ".t").mkdir()
    gmod, g, news, alerts = _guardian(tmp_path, monkeypatch, tree, data,
                                      ["workers.db", ".t"])
    g._runtime_data_incident(NOW)

    assert not stray.exists()
    assert len(alerts) == 1 and alerts[0][0] == gmod.RUNTIME_DATA_ALERT_TITLE
    assert os.path.join(str(tree), ".t") in alerts[0][1]
    assert os.path.join(str(tree), "workers.db") not in alerts[0][1]


def test_a_disarmed_guardian_moves_nothing(roots, tmp_path, monkeypatch):
    tree, data, stray = roots
    gmod, g, news, alerts = _guardian(tmp_path, monkeypatch, tree, data, ["workers.db"])
    g.data = types.SimpleNamespace(armed=False)
    g._runtime_data_incident(NOW)
    assert stray.exists() and news == [] and alerts == []
