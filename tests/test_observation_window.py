"""The observation window is two numbers now, and both come from config.

A promotion is watched by the guardian for `errors_until_ts - landed_ts`, and
every other landing queues behind that window (`promote.wait_for_settle`), so
the one constant decides both how much a bad build is judged and how much of
the loop's day is spent serialized. It was 900 s for every promotion until
2026-09-20, when a day's measurement put it at 13.5 h of a 24 h window across
54 promotions while every rollback the window has EVER caused had fired within
5.5 minutes of the landing — 4 s, 147 s, 262 s, 327 s, all four false positives
or misattributions.

Two changes, pinned here:

  * a landing that restarted no service gets `errors_window_unrestarted_s`.
    The guardian already skips liveness and the error rate for such a promotion
    (`unrestarted` in `guardian.tick`) because the code that could crash is the
    code that was already running; what it still judges is data damage, so the
    window is shortened rather than removed;
  * both windows are read from `automod.landing`, where `errors_window_s` had
    sat unread since the block was written — the same defect as the two idle
    keys found dead on 2026-09-17, and the reason `promotion_announcement` now
    states the window it was given rather than formatting a constant.
"""

from __future__ import annotations

import time

import pytest

from scripts.automod import promote as P, state as S


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No live state and no live config.yaml: `landing_cfg` is the input under
    test, so every case states its own."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(S, "CURRENT_PATH", tmp_path / "current.json")
    monkeypatch.setattr(S, "ROLLBACK_REQUEST_PATH", tmp_path / "rollback_request.json")
    monkeypatch.setattr(S, "HALTED_PATH", tmp_path / "promotions-halted")
    monkeypatch.setattr(S, "BROKEN_PATH", tmp_path / "BROKEN")
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: {})
    return tmp_path


def _cfg(monkeypatch, **keys):
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: dict(keys))


# ── the two windows ───────────────────────────────────────────────────────

def test_an_unrestarted_landing_is_observed_for_less_than_a_restarted_one():
    """The whole point of the split. A promotion that replaced no process
    cannot crash one, and its error log belongs to a build that never booted."""
    assert P.errors_window(restart=False) < P.errors_window(restart=True)
    assert P.errors_window(restart=True) == P.ERRORS_WINDOW
    assert P.errors_window(restart=False) == P.ERRORS_WINDOW_UNRESTARTED


def test_both_windows_come_from_config(monkeypatch):
    _cfg(monkeypatch, errors_window_s=300, errors_window_unrestarted_s=90)
    assert P.errors_window(restart=True) == 300
    assert P.errors_window(restart=False) == 90


def test_a_missing_key_falls_back_to_its_own_constant(monkeypatch):
    """Each key falls back independently: setting one must not silently move
    the other, which is what a single shared default would do."""
    _cfg(monkeypatch, errors_window_s=300)
    assert P.errors_window(restart=True) == 300
    assert P.errors_window(restart=False) == P.ERRORS_WINDOW_UNRESTARTED


@pytest.mark.parametrize("bad", ["", "fifteen", None, [], {}, "900s"])
def test_an_unreadable_value_falls_back_rather_than_raising(monkeypatch, bad):
    """config.yaml is hand-edited and the promoter runs detached with its
    output going to a log nobody reads live. A typo must cost the default, not
    the landing."""
    _cfg(monkeypatch, errors_window_s=bad, errors_window_unrestarted_s=bad)
    assert P.errors_window(restart=True) == P.ERRORS_WINDOW
    assert P.errors_window(restart=False) == P.ERRORS_WINDOW_UNRESTARTED


@pytest.mark.parametrize("value,expected", [
    (0, P.ERRORS_WINDOW_FLOOR),
    (-60, P.ERRORS_WINDOW_FLOOR),
    (1, P.ERRORS_WINDOW_FLOOR),
    (86400, P.ERRORS_WINDOW_CEILING),
])
def test_a_window_outside_the_bounds_is_clamped_not_obeyed(monkeypatch, value, expected):
    """Both ends fail in a direction the guardian cannot see. A zero-length
    window settles a promotion nothing ever judged and advances the LKG anyway,
    so `last_known_good` would come to mean "last landed" while every record
    still looked healthy; above the ceiling the loop simply stops landing."""
    _cfg(monkeypatch, errors_window_s=value, errors_window_unrestarted_s=value)
    assert P.errors_window(restart=True) == expected
    assert P.errors_window(restart=False) == expected


def test_the_floor_leaves_room_for_the_guardian_to_actually_judge():
    """The guardian ticks every 5 s and `crash` needs three consecutive failed
    probes, so a window under ~20 s cannot produce the verdict it exists for.
    A floor below that would be a window in name only."""
    assert P.ERRORS_WINDOW_FLOOR >= 20


def test_the_live_config_sets_both_keys_inside_the_bounds():
    """config.yaml is the state a rebuild boots into (there is no override file
    for `automod`), so the tracked value has to be one the promoter will use
    unchanged — a clamped one would mean the file and the behaviour disagree."""
    cfg = S.landing_cfg.__wrapped__() if hasattr(S.landing_cfg, "__wrapped__") else None
    del cfg  # the fixture stubs landing_cfg; read the tracked file directly
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    landing = ((raw.get("automod") or {}).get("landing") or {})
    for key in ("errors_window_s", "errors_window_unrestarted_s"):
        assert key in landing, f"{key} is read by promote.errors_window and must be stated"
        value = float(landing[key])
        assert P.ERRORS_WINDOW_FLOOR <= value <= P.ERRORS_WINDOW_CEILING, (
            f"{key}={value} would be clamped, so config.yaml would not describe "
            "what is served")
    assert float(landing["errors_window_unrestarted_s"]) <= float(landing["errors_window_s"])


def test_the_dead_liveness_key_is_gone():
    """`liveness_window_s` configured a `LIVENESS_WINDOW` constant that was
    deleted as dead — the guardian applies liveness for the whole window. A key
    naming a mechanism that no longer exists is the defect `errors_window_s`
    had, one file over."""
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    landing = ((raw.get("automod") or {}).get("landing") or {})
    assert "liveness_window_s" not in landing
    assert not hasattr(P, "LIVENESS_WINDOW")


# ── the settle wait derives from the larger window ────────────────────────

def test_the_settle_wait_takes_the_larger_window_never_the_waiter_s_own(monkeypatch):
    """What a landing waits on is somebody else's promotion, and the window
    belongs to that one. Deriving from the caller's own window would hold a
    no-restart landing to 120 s while waiting out a restarted promotion's 450
    and refuse it with six minutes still to run."""
    _cfg(monkeypatch, errors_window_s=450, errors_window_unrestarted_s=120)
    assert P.settle_max_wait() == 450 + P.SETTLE_SLACK
    # ...and it follows whichever is larger, not the restarted one by name.
    _cfg(monkeypatch, errors_window_s=120, errors_window_unrestarted_s=450)
    assert P.settle_max_wait() == 450 + P.SETTLE_SLACK


def test_the_settle_wait_keeps_slack_above_the_window(monkeypatch):
    """A landing that gave up at exactly the window would race the guardian's
    own settle, which runs on a 5 s tick after the window expires."""
    _cfg(monkeypatch, errors_window_s=450, errors_window_unrestarted_s=120)
    assert P.settle_max_wait() > P.errors_window(True)


# ── the wait is recorded ──────────────────────────────────────────────────

def _rows(event):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == event]


def test_a_successful_settle_wait_is_recorded(monkeypatch):
    """It recorded nothing until 2026-09-20, so the cost this change is about
    was invisible: `land_wait_rounds` had a row and `wait_for_settle` had none,
    and the only way to see the window serializing landings was to difference
    `promoted` stamps by hand."""
    monkeypatch.setattr(P, "SETTLE_POLL_SECONDS", 0.01)
    seq = [{"commit": "c" * 40, "state": "observing"}, None]
    monkeypatch.setattr(S, "read_current", lambda: seq.pop(0) if seq else None)
    S.append_event({"event": "settled", "commit": "c" * 40}, path=S.LEDGER_PATH)

    P.wait_for_settle(max_wait=1.0, round_id="SM_W")

    row = _rows("land_wait_settle")[-1]
    assert row["round_id"] == "SM_W" and row["ok"] is True
    assert row["behind"] == 1 and row["waited_s"] >= 0
    assert row["windows"]["restarted"] == P.errors_window(True)
    assert row["windows"]["unrestarted"] == P.errors_window(False)


def test_a_settle_wait_that_times_out_records_what_it_waited(monkeypatch):
    """The failure already wrote `land_failed`; what it did not carry was how
    long it had waited, which is the number that says whether the window or the
    queue depth was the problem."""
    monkeypatch.setattr(P, "SETTLE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(S, "read_current",
                        lambda: {"commit": "c" * 40, "state": "observing"})

    with pytest.raises(P.PromoteError, match="still under observation"):
        P.wait_for_settle(max_wait=0.05, round_id="SM_W")

    row = _rows("land_failed")[-1]
    assert row["waited_for_settle"] is True and row["external_blocker"] is True
    assert row["waited_s"] >= 0 and row["behind"] == 1
    assert not _rows("land_wait_settle"), "a refusal is not a completed wait"


def test_no_round_id_writes_no_row(monkeypatch):
    """`wait_for_settle` is callable without a round — `promote` keeps its own
    refusal behind it — and a row with no round id is one the scorecard cannot
    attribute."""
    monkeypatch.setattr(P, "SETTLE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(S, "read_current", lambda: None)
    P.wait_for_settle(max_wait=0.05)
    assert not _rows("land_wait_settle")


# ── the promotion record carries the window it got ────────────────────────

# The end-to-end pin — that `promote` stamps the window its own
# `restart_needed` verdict chose — lives in `test_landing_without_restart.py`,
# beside the `tree`/`_servers` harness that drives a real promotion and the
# rest of the restart-decision suite:
#   test_the_window_follows_the_restart_decision


def test_the_guardian_reads_the_window_off_the_record_not_a_constant(tmp_path, monkeypatch):
    """The guardian is a pinned stdlib-only snapshot that never imports from
    `scripts/`, so a shorter window reaches it ONLY as a smaller
    `errors_until_ts`. If it ever grew its own copy of the number, changing
    config.yaml would move the promoter and leave the watchdog on the old one.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                           "agent-services" / "guardian"))
    import gstate

    guardian_src = (Path(__file__).resolve().parent.parent /
                    "agent-services" / "guardian" / "guardian.py").read_text(encoding="utf-8")
    assert "ERRORS_WINDOW" not in guardian_src, (
        "the guardian must take the window from errors_until_ts on the record")

    st = gstate.AutomodState(tmp_path)
    landed = time.time()
    gstate.write_json_atomic(st.current_path,
                             {"state": "observing", "commit": "b" * 40,
                              "landed_ts": landed, "errors_until_ts": landed + 120})
    assert st.current()["errors_until_ts"] - landed == 120
