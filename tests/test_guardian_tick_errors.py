"""A tick that raises must not be silent, and a notifier that raises must not
raise out of the tick (F5 of the 2026-11 guardian review).

`check_pool` called `notifier.resolve` outside its `try` on every healthy tick,
and `resolve` read the recent daily notes with strict UTF-8 catching only
OSError: one invalid byte in a note raised out of every tick, ahead of
liveness, rollback requests and settle, while the unit kept sending
WATCHDOG=1 and the only trace was a heartbeat state nothing reads.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
if str(GUARDIAN_DIR) not in sys.path:
    sys.path.insert(0, str(GUARDIAN_DIR))

import notify  # noqa: E402
import policy  # noqa: E402

TITLE = "Worker pool is not running"


def _notifier(tmp_path: Path) -> notify.Notifier:
    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    return notify.Notifier(ledger=tmp_path / "ledger.jsonl", state_dir=tmp_path,
                           vault_root=str(vault), backend_url="http://127.0.0.1:1",
                           external=True)


def test_resolve_survives_an_invalid_byte_and_keeps_it(tmp_path):
    n = _notifier(tmp_path)
    n.alert("error", TITLE, "the pool is stopped", coalesce=True)
    note = next((tmp_path / "obsidian" / "memory").glob("*.md"))
    note.write_bytes(note.read_bytes().replace(b"\n\n## Self-mod",
                                               b"\n\nbad \xff byte\n\n## Self-mod", 1))

    n.resolve(TITLE, "the pool reports running")

    raw = note.read_bytes()
    assert b"bad \xff byte" in raw, "the undecodable byte is written back unchanged"
    assert notify.DAILY_STILL_OPEN.encode() not in raw, "the section was sealed"


def _guardian():
    import guardian as G

    g = G.Guardian.__new__(G.Guardian)
    g.alerts = []
    g.alert = lambda level, title, body, **kw: g.alerts.append((level, title))
    return G, g


def test_a_raising_resolve_does_not_raise_out_of_check_pool():
    G, g = _guardian()

    def boom(*a, **k):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    g.notifier = types.SimpleNamespace(resolve=boom)
    g.pool = types.SimpleNamespace(tick=lambda health: {"reason": "running"})

    g.check_pool({"backend_health": {"ok": True}})   # must not raise


def test_consecutive_tick_errors_alert_once_and_retract_on_recovery():
    G, g = _guardian()
    resolved = []
    g.notifier = types.SimpleNamespace(resolve=lambda t, note, **k: resolved.append(t))
    err = RuntimeError("tick blew up")

    for _ in range(policy.TICK_ERROR_ALERT_AFTER - 1):
        g._tick_outcome(err)
    assert g.alerts == []
    g._tick_outcome(err)
    assert g.alerts == [("critical", G.TICK_ERROR_ALERT_TITLE)]
    for _ in range(10):
        g._tick_outcome(err)
    assert len(g.alerts) == 1, "one alert per streak, not one per tick"

    g._tick_outcome(None)
    assert resolved == [G.TICK_ERROR_ALERT_TITLE]
    g._tick_outcome(None)
    assert resolved == [G.TICK_ERROR_ALERT_TITLE], "a healthy streak retracts once"
