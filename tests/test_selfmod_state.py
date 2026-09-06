"""On-disk state for the self-modification loop.

The centrepiece is `test_a_rollback_point_that_did_not_land_is_refused`. It is
the strict (not xfail) analogue of the documented defect at
`tests/test_autoresearch_promotion.py:362`: `snapshot_current_prompts` mkdirs
unconditionally, never verifies the copy landed, and `promote()` overwrites
live state anyway — which is why 26 of 83 ledger promotions have no rollback
point at all. Nothing in this package may mutate the live tree until its
rollback point has been read back from disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from scripts.selfmod import state as S


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """No test touches the real ~/.local/state/lloyd-selfmod."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    monkeypatch.setattr(S, "LKG_PATH", tmp_path / "last_known_good.json")
    monkeypatch.setattr(S, "CURRENT_PATH", tmp_path / "current.json")
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(S, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(S, "PAUSE_PATH", tmp_path / "pause")
    monkeypatch.setattr(S, "HALTED_PATH", tmp_path / "promotions-halted")
    monkeypatch.setattr(S, "BROKEN_PATH", tmp_path / "BROKEN")
    monkeypatch.setattr(S, "DENIED_PATH", tmp_path / "denied.json")
    monkeypatch.setattr(S, "BROKEN_DIR", tmp_path / "broken")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    return tmp_path


SHA_A = "a" * 40
SHA_B = "b" * 40


# ---------------------------------------------------------------------------
# The defect class this whole module exists to avoid
# ---------------------------------------------------------------------------

def test_a_rollback_point_that_did_not_land_is_refused(isolated_state, monkeypatch):
    """A silently no-op write must raise, not return a usable-looking path."""
    monkeypatch.setattr(S, "write_json", lambda path, payload: None)
    with pytest.raises(RuntimeError, match="did not land"):
        S.write_verified(isolated_state / "x.json", {"commit": SHA_A})


def test_a_rollback_point_that_did_not_round_trip_is_refused(isolated_state, monkeypatch):
    """Wrote something, but not what we asked for — equally unusable."""
    real_write = S.write_json  # capture before patching, or `wrong` recurses

    def wrong(path, payload):
        real_write(path, {**payload, "commit": SHA_B})
    monkeypatch.setattr(S, "write_json", wrong)
    with pytest.raises(RuntimeError, match="did not round-trip"):
        S.write_verified(isolated_state / "x.json", {"commit": SHA_A})


def test_a_good_write_round_trips(isolated_state):
    back = S.write_verified(isolated_state / "x.json", {"commit": SHA_A, "n": 1})
    assert back["commit"] == SHA_A and back["n"] == 1


def test_write_json_is_atomic_and_leaves_no_temp_files(isolated_state):
    target = isolated_state / "x.json"
    S.write_json(target, {"a": 1})
    S.write_json(target, {"a": 2})
    assert S.read_json(target) == {"a": 2}
    assert [p.name for p in isolated_state.iterdir() if ".tmp" in p.name] == []


def test_read_json_on_garbage_returns_none_rather_than_raising(isolated_state):
    p = isolated_state / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert S.read_json(p) is None
    assert S.read_json(isolated_state / "missing.json") is None


# ---------------------------------------------------------------------------
# Ledger — deliberately NOT autoresearch's best-effort append
# ---------------------------------------------------------------------------

def test_the_ledger_raises_where_autoresearchs_swallows(isolated_state, tmp_path):
    """Documented divergence, asserted so nobody refactors them together.

    `scripts.autoresearch.common.ledger_append` is contractually
    "never raises" — right for a research ledger, wrong for the audit record of
    what code is running in production, where a dropped line means you cannot
    reconstruct what landed.
    """
    from scripts.autoresearch.common import ledger_append

    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    impossible = blocker / "nested" / "ledger.jsonl"

    ledger_append(impossible, {"event": "x"})          # swallows

    with pytest.raises(OSError):                        # ours does not
        S.append_event({"event": "x"}, path=impossible)


def test_append_event_is_one_json_line_per_entry(isolated_state):
    S.append_event({"event": "promoted", "commit": SHA_A})
    S.append_event({"event": "settled", "commit": SHA_A})
    lines = S.LEDGER_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["commit"] == SHA_A for line in lines)


def test_append_event_stamps_time(isolated_state):
    S.append_event({"event": "promoted"})
    entry = json.loads(S.LEDGER_PATH.read_text(encoding="utf-8").strip())
    assert entry["created_at"].endswith("Z")
    assert abs(entry["ts"] - time.time()) < 5


def test_read_events_tolerates_a_corrupt_line(isolated_state):
    S.append_event({"event": "a"})
    with open(S.LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write("this is not json\n")
    S.append_event({"event": "b"})
    events = S.read_events()
    assert [e["event"] for e in events] == ["a", "b"]


def test_read_events_on_a_missing_ledger_is_empty(isolated_state):
    assert S.read_events() == []


# ---------------------------------------------------------------------------
# Last known good
# ---------------------------------------------------------------------------

def test_lkg_round_trips_and_pins_the_floor(isolated_state):
    S.write_lkg(SHA_A, floor=SHA_A)
    S.write_lkg(SHA_B)
    lkg = S.read_lkg()
    assert lkg["commit"] == SHA_B
    assert lkg["floor"] == SHA_A, "the floor must never move once set"


def test_lkg_preserves_health_and_eval_when_omitted(isolated_state):
    S.write_lkg(SHA_A, health={"mcp_degraded_modules": ["thunderbird"]},
                eval_baseline={"entity_hit_rate": 0.6})
    S.write_lkg(SHA_B)
    lkg = S.read_lkg()
    assert lkg["health"]["mcp_degraded_modules"] == ["thunderbird"]
    assert lkg["eval"]["entity_hit_rate"] == 0.6


def test_reading_a_missing_lkg_is_none(isolated_state):
    assert S.read_lkg() is None


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def test_halted_flag_round_trips(isolated_state):
    assert not S.is_halted()
    S.set_halted("2 rollbacks in 6h")
    assert S.is_halted()
    assert "2 rollbacks" in S.HALTED_PATH.read_text()
    S.clear_halted()
    assert not S.is_halted()


def test_pause_is_capped(isolated_state):
    S.set_pause(10 * 24 * 3600, cap=1800.0)
    assert S.pause_remaining() == pytest.approx(1800.0, abs=2.0)


def test_pause_expires_on_its_own(isolated_state):
    S.PAUSE_PATH.write_text(str(time.time() - 1), encoding="utf-8")
    assert S.pause_remaining() == 0.0


def test_a_missing_or_garbage_pause_reads_as_zero(isolated_state):
    assert S.pause_remaining() == 0.0
    S.PAUSE_PATH.write_text("not a number", encoding="utf-8")
    assert S.pause_remaining() == 0.0


def test_denylist_records_commits_and_trees(isolated_state):
    S.deny(SHA_A, "tree123")
    assert S.is_denied(commit=SHA_A)
    assert S.is_denied(tree_hash="tree123")
    assert not S.is_denied(commit=SHA_B)
    S.deny(SHA_A)  # idempotent
    assert S.read_denied()["commits"] == [SHA_A]


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

def test_the_lock_excludes_a_second_holder(isolated_state):
    with S.Lock(owner="round-1"):
        with pytest.raises(S.LockHeld, match="round-1"):
            S.Lock(owner="round-2").acquire()


def test_the_lock_is_released_on_exit(isolated_state):
    with S.Lock(owner="a"):
        pass
    with S.Lock(owner="b"):
        pass  # would raise if the first were still held


def test_the_lock_records_its_holder(isolated_state):
    with S.Lock(owner="round-xyz"):
        payload = json.loads(S.LOCK_PATH.read_text(encoding="utf-8"))
        assert payload["owner"] == "round-xyz"
        assert payload["pid"] == os.getpid()


def test_a_lock_whose_holder_died_is_available_again(isolated_state):
    """flock is released by the kernel when the holder exits, so a crashed
    round must not wedge the loop forever."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, %r);"
        "from scripts.selfmod import state as S;"
        "S.LOCK_PATH = __import__('pathlib').Path(%r);"
        "S.Lock(owner='dead').acquire()"
        % (str(Path(__file__).resolve().parent.parent), str(S.LOCK_PATH))
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    with S.Lock(owner="survivor"):
        pass
