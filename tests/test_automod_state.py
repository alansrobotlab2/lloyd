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
import re
import time
from pathlib import Path

import pytest

from scripts.automod import state as S


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """No test touches the real ~/.local/state/lloyd-automod."""
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

def _ledger_rows(path: Path) -> list[dict]:
    """Decode the ledger fixture, [] when nothing has been appended yet."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# The halt tests below are #1365: clause 1 is
# `test_setting_the_halt_appends_a_halt_set_row_carrying_its_reason`, clause 3
# is `test_clearing_the_halt_appends_a_halt_clear_row_naming_the_clearer`,
# `test_a_clear_that_lifted_nothing_records_no_transition`,
# `test_re_asserting_an_already_set_halt_leaves_the_first_set_as_the_start`,
# `test_recover_records_the_halt_clear_and_keeps_the_recovered_row` and
# `test_the_recover_cli_forwards_by_and_the_clear_precedes_the_restarts` — the
# last one driving `main(["recover", "--by", …])`, since only the CLI consumes
# the flag. Clause 2, the guardian's writer, is in
# tests/test_guardian_rollback.py.

def test_halted_flag_round_trips(isolated_state):
    assert not S.is_halted()
    S.set_halted("2 rollbacks in 6h")
    assert S.is_halted()
    assert "2 rollbacks" in S.HALTED_PATH.read_text()
    # `by` became required on the clear (#1365): a clear with no actor is the
    # gap, so this call names one rather than the function weakening to allow
    # the anonymous clear the ledger used to record.
    S.clear_halted(by="test round trip")
    assert not S.is_halted()


def test_setting_the_halt_appends_a_halt_set_row_carrying_its_reason(isolated_state):
    """The file says a freeze is on; the ledger says why it started.

    Before this, `set_halted` wrote `HALTED_PATH` and nothing else, so the
    2026-09-21 21:44Z freeze — 3 h 36 m of refused promotions — existed in the
    audit trail only as prose inside a guardian `alert` row.
    """
    S.set_halted("3 rollbacks in 6h", by="guardian flap quarantine")
    sets = [r for r in _ledger_rows(S.LEDGER_PATH) if r.get("event") == S.HALT_SET_EVENT]
    assert len(sets) == 1, "exactly one row whose subject is the halt being set"
    assert sets[0]["reason"] == "3 rollbacks in 6h"
    assert sets[0]["by"] == "guardian flap quarantine"
    assert sets[0]["already_halted"] is False, "this one is the start of the freeze"
    assert sets[0]["path"] == str(S.HALTED_PATH), "so a reader can name what to clear"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                        sets[0]["created_at"]), "the start time, offset-bearing UTC"


def test_clearing_the_halt_appends_a_halt_clear_row_naming_the_clearer(isolated_state):
    S.set_halted("vault tripwire: mass deletion", by="guardian vault check")
    assert S.clear_halted(by="alan, after checking the snapshot") is True
    assert not S.is_halted()
    clears = [r for r in _ledger_rows(S.LEDGER_PATH) if r.get("event") == S.HALT_CLEAR_EVENT]
    assert len(clears) == 1, "exactly one row whose subject is the halt being cleared"
    assert clears[0]["by"] == "alan, after checking the snapshot"
    assert clears[0]["by"].strip(), "clause 3: the field names what cleared it"
    with pytest.raises(TypeError):
        S.clear_halted()          # an anonymous clear is a TypeError, not ""


def test_a_clear_that_lifted_nothing_records_no_transition(isolated_state):
    """No freeze, no clear, no row — the ledger records transitions, not calls.

    The counterpart to the clause: an unconditional append here would make
    every no-op clear look like the end of a freeze, and the first set row's
    value as the start time depends on clears meaning something.
    """
    assert S.clear_halted(by="anyone") is False
    assert _ledger_rows(S.LEDGER_PATH) == []


def test_re_asserting_an_already_set_halt_leaves_the_first_set_as_the_start(isolated_state):
    """The flap quarantine re-sets on every rollback past the threshold."""
    S.set_halted("2 rollbacks in 6h", by="guardian flap quarantine")
    S.set_halted("3 rollbacks in 6h", by="guardian flap quarantine")
    sets = [r for r in _ledger_rows(S.LEDGER_PATH) if r.get("event") == S.HALT_SET_EVENT]
    assert [r["already_halted"] for r in sets] == [False, True]
    assert sets[-1]["reason"] == "3 rollbacks in 6h", "the latest reason wins"
    assert S.HALTED_PATH.read_text().count("3 rollbacks") == 1


def test_recover_records_the_halt_clear_and_keeps_the_recovered_row(isolated_state,
                                                                    monkeypatch):
    """`recover` is the one production route that lifts a halt.

    Its `recovered` row carries no actor and no round_id, which is why the
    2026-09-22 01:21:08Z clear was reconstructable only as a `cleared` field
    inside another event's subject. The clear now has a row of its own; the
    `recovered` row keeps its `cleared` list exactly as it was, because readers
    already key on that field.
    """
    import app.supervisor_client as SC
    from scripts.automod import round as R

    monkeypatch.setattr(SC, "start_process", lambda name: (True, "started"))
    S.BROKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    S.BROKEN_PATH.write_text("escalated: probe dead", encoding="utf-8")
    S.set_halted("2 rollbacks in 6h", by="guardian flap quarantine")

    out = R.recover()

    assert out["cleared"] == ["BROKEN", "promotions-halted"]
    rows = _ledger_rows(S.LEDGER_PATH)
    recovered = [r for r in rows if r.get("event") == "recovered"]
    assert len(recovered) == 1
    assert recovered[0]["cleared"] == ["BROKEN", "promotions-halted"], "unchanged"
    clears = [r for r in rows if r.get("event") == S.HALT_CLEAR_EVENT]
    assert len(clears) == 1, "the clear is on the record under its own subject"
    # The whole string, not a prefix: `by` is the answer to "who owns clearing
    # it", so a test that accepted any prefix would pass on an empty actor.
    assert clears[0]["by"] == f"round recover pid {os.getpid()}"
    assert not S.is_halted()


def test_the_recover_cli_forwards_by_and_the_clear_precedes_the_restarts(
        monkeypatch, isolated_state, capsys):
    """Drives `main(["recover", "--by", …])`, because `recover(by=…)` cannot fail here.

    The flap alert tells a human to lift the freeze, and the signed route is
    `python -m scripts.automod.round recover --by NAME`. Two seams on that one
    command, both invisible to a test that calls `recover(by=…)` itself:
    (1) argparse stores `--by` and main's dispatch has to hand it over — the
    commit this replaces parsed the flag and dispatched `recover()` with no
    arguments, so the name a human typed never reached the row and the pid
    fallback was recorded instead; (2) the halt-clear row is appended before
    `start_process` runs, so a restart that dies mid-flight cannot leave the
    freeze ending unrecorded. The fake below reads the ledger at each start to
    pin that order.

    What this does NOT cover, said plainly: `start_process` is a fake, so the
    real supervisor calls against lloyd-mcp and lloyd-backend, and the guardian
    daemon running this new code from a restarted process, are exercised only
    in production.
    """
    import app.supervisor_client as SC
    from scripts.automod import round as R

    starts: list[tuple[str, int]] = []

    def fake_start(name: str):
        clears = [r for r in _ledger_rows(S.LEDGER_PATH)
                  if r.get("event") == S.HALT_CLEAR_EVENT]
        starts.append((name, len(clears)))
        return True, "started"

    monkeypatch.setattr(SC, "start_process", fake_start)
    S.set_halted("vault tripwire: 3 tracked vault files vanished")

    rc = R.main(["recover", "--by", "  alan, snapshot checked  "])
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["cleared"] == ["promotions-halted"], "no BROKEN flag this time"
    clears = [r for r in _ledger_rows(S.LEDGER_PATH)
              if r.get("event") == S.HALT_CLEAR_EVENT]
    assert [c["by"] for c in clears] == ["alan, snapshot checked"], \
        "the name as typed at the prompt, whitespace stripped, not the pid"
    assert [name for (name, _) in starts] == ["lloyd-mcp", "lloyd-backend"], \
        "both programs, mcp first"
    assert [rows for (_, rows) in starts] == [1, 1], \
        "the halt-clear row was already in the ledger when each start ran"


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
        "from scripts.automod import state as S;"
        "S.LOCK_PATH = __import__('pathlib').Path(%r);"
        "S.Lock(owner='dead').acquire()"
        % (str(Path(__file__).resolve().parent.parent), str(S.LOCK_PATH))
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    with S.Lock(owner="survivor"):
        pass
