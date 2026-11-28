"""A vault land announces itself while it runs, and a second call is told so (#4583).

`automod_vault_land` is synchronous inside lloyd-mcp and can outlast the 660 s
tool-call ceiling: #4429's land measured `seconds: 1529` against `budget: 900`, with
`lock_wait_s: 629` queued behind a running gate's tests lock. Its caller saw a
transport timeout on a healthy land, and nothing it could read said "in flight".

Pinned here, against a scratch state dir:

* `vault_round.land` writes a marker before any work and removes it whatever the
  outcome (returned or raised);
* the marker names item, session, paths and start, and its reader derives the stage
  (`review` until a `vault_review` row for the item lands after the start, then
  `committing`) and the live holder of the gate's tests lock;
* a second call for the same item while the first is live is refused with that state
  and says not to call again — never run twice;
* a marker whose pid is gone (lloyd-mcp restarted under the land) is not in flight;
* the marker lives beside the ledger, so isolating `LEDGER_PATH` isolates it;
* and, from #4628 below, that a caller is not made to wait for the land at all: past
  `agent_mcp.automod.VAULT_LAND_HANDBACK_SECONDS` the handler answers with this
  marker and the land runs on to its own ledger row.

The finalizer's half is in `tests/test_automod_review.py` (#4583 section), and the
second-call refusal across the MCP dispatch seam is in `tests/test_automod_hardening.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import pytest

from scripts.automod import state as S
from scripts.automod import vault_round as V


@pytest.fixture(autouse=True)
def _scratch_state(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "VAULT_LAND_DIR", tmp_path / "vault_lands")
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "GATE_TESTS_LOCK_PATH", tmp_path / "gate-tests.lock")
    return tmp_path


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_land_holds_a_marker_while_it_runs_and_drops_it_after(monkeypatch):
    seen = {}

    def fake_land(paths, message, **kw):
        seen["rec"] = S.vault_land_in_progress(item_id=4429)
        return {"ok": True}

    monkeypatch.setattr(V, "_land", fake_land)
    assert V.land(["backlog/a.md"], "msg", item_id=4429, session_id="s1") == {"ok": True}
    rec = seen["rec"]
    assert rec is not None, "no marker was live while the land ran"
    assert rec["item_id"] == 4429 and rec["paths"] == ["backlog/a.md"]
    assert rec["session_id"] == "s1" and rec["pid"] == os.getpid()
    assert rec["stage"] == "review"
    assert S.read_vault_land_marker(item_id=4429) is None, "the marker outlived the land"


def test_a_land_that_raises_drops_its_marker_too(monkeypatch):
    def boom(paths, message, **kw):
        raise V.VaultRoundError("validation failed")

    monkeypatch.setattr(V, "_land", boom)
    with pytest.raises(V.VaultRoundError):
        V.land(["backlog/a.md"], "msg", item_id=7)
    assert S.read_vault_land_marker(item_id=7) is None


def test_a_second_call_while_the_first_is_live_is_refused_with_its_state(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_land(paths, message, **kw):
        calls.append(paths)
        started.set()
        release.wait(10)
        return {"ok": True}

    monkeypatch.setattr(V, "_land", slow_land)
    t = threading.Thread(target=V.land, args=(["backlog/a.md"], "m"),
                         kwargs={"item_id": 4429, "session_id": "s1"})
    t.start()
    try:
        assert started.wait(10)
        with pytest.raises(V.VaultRoundError) as exc:
            V.land(["backlog/a.md"], "m", item_id=4429, session_id="s2")
    finally:
        release.set()
        t.join(10)
    msg = str(exc.value)
    assert "already in flight" in msg and "item #4429" in msg, msg
    assert "Do not call it again" in msg and "backlog/a.md" in msg, msg
    assert calls == [["backlog/a.md"]], "the second call ran the land again"
    # Once the first returns, a call for the item proceeds normally.
    assert V.land(["backlog/a.md"], "m", item_id=4429) == {"ok": True}


def test_a_marker_whose_process_is_gone_is_not_in_flight(monkeypatch):
    S.write_vault_land_marker(item_id=11, paths=["a.md"], pid=_dead_pid())
    assert S.read_vault_land_marker(item_id=11) is not None
    assert S.vault_land_in_progress(item_id=11) is None
    calls = []
    monkeypatch.setattr(V, "_land", lambda paths, message, **kw: calls.append(1) or {})
    V.land(["a.md"], "m", item_id=11)
    assert calls == [1], "a dead marker blocked a fresh land"


def test_the_stage_moves_to_committing_once_the_review_row_lands():
    rec = S.write_vault_land_marker(item_id=4599, paths=["autonomy/97-x.md"])
    assert S.vault_land_in_progress(item_id=4599)["stage"] == "review"
    S.append_event({"event": "vault_review", "item_id": 4598, "kind": "pass"})
    assert S.vault_land_in_progress(item_id=4599)["stage"] == "review", \
        "another item's review moved this land's stage"
    S.append_event({"event": "vault_review", "item_id": 4599, "kind": "pass"})
    live = S.vault_land_in_progress(item_id=4599)
    assert live["stage"] == "committing (review pass)", live
    assert live["started_at"] == rec["started_at"]
    assert live["elapsed_s"] >= 0


def test_a_live_tests_lock_holder_is_named_and_a_stale_payload_is_not(_scratch_state):
    S.write_vault_land_marker(item_id=5, paths=["a.md"])
    lock = S.GATE_TESTS_LOCK_PATH
    lock.write_text(json.dumps({"pid": _dead_pid(), "owner": "gate-SM_old",
                                "since": time.time()}))
    assert "tests_lock_holder" not in S.vault_land_in_progress(item_id=5)
    # A live automod-shaped holder: this process, made to read as one.
    lock.write_text(json.dumps({"pid": os.getpid(), "owner": "gate-SM_live",
                                "since": time.time()}))
    import scripts.automod.state as ST
    orig = ST.marker_pid_is_live
    try:
        ST.marker_pid_is_live = lambda pid, since=None: pid == os.getpid()
        live = S.vault_land_in_progress(item_id=5)
    finally:
        ST.marker_pid_is_live = orig
    assert live["tests_lock_holder"]["owner"] == "gate-SM_live", live
    assert "gate-SM_live" in S.vault_land_in_flight_text(live)


def test_an_item_less_land_is_keyed_by_session_and_one_without_either_writes_nothing(
        monkeypatch):
    seen = {}

    def fake(paths, message, **kw):
        seen["by_session"] = S.vault_land_in_progress(session_id="s9")
        return {}

    monkeypatch.setattr(V, "_land", fake)
    V.land(["a.md"], "m", session_id="s9")
    assert seen["by_session"] is not None
    assert S.vault_land_marker_path() is None  # neither item nor session: no key
    S.VAULT_LAND_DIR.mkdir(parents=True, exist_ok=True)
    V.land(["a.md"], "m")
    assert not list((S.VAULT_LAND_DIR).glob("*.json"))


def test_the_marker_lives_beside_the_ledger_unless_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "VAULT_LAND_DIR", None)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "elsewhere" / "promotions.jsonl")
    assert S.vault_land_marker_path(item_id=3) == tmp_path / "elsewhere" / "vault_lands" / "item-3.json"


# ─── #4628: the land is handed back, not waited out ───────────────────────────
#
# `automod_vault_land` used to hold its caller's turn for the whole land, and the land
# costs far more than any caller may wait: of the 165 `vault_land` rows in
# `~/.local/state/lloyd-automod/promotions.jsonl` dated November 2026, the median
# `guards.seconds` is 900.0 and the maximum 2389.3, and 12 of the 17 autonomy runs that
# month died at their `timeout_seconds` with this one call in flight (`idle_seconds`
# 133.1-659.8, so idle, never busy). Past the wait the handler now answers with the
# marker this file already pins, and the land continues to its own ledger row in
# lloyd-mcp.
#
# The nodes cross the seam the fix lives on — `agent_mcp.automod.call_tool`, which
# dispatches its body through `asyncio.to_thread` and now spawns the land's own thread —
# and hold the REAL `vault_round.land` and the real vault writer: the land is slowed
# inside `validate()`, the leg that runs before any `git add`, so the handback is
# observed while the land is genuinely still in flight rather than against a stand-in
# that could not have committed anything anyway.
#
# Every node releases the held land and waits for its thread before returning. pytest's
# `monkeypatch` puts `V.VAULT` back at the LIVE vault at teardown, and a land still
# running at that instant would finish its commit against the real vault.

BOUND = 0.4
LAND_ARGS = {"paths": ["backlog/4628-item.md"], "message": "item: promote #4628"}
# The bytes the land is asked to commit: a status flip `validate()` accepts…
GOOD_BODY = "---\nstatus: done\n---\n# item\n"
# …and one it refuses for real, by YAML. A `backlog/*.md` carries no loader, so
# `validate()` reaches `loader_errors` only for files it can load and never for these:
# the check that actually fires on this path is the front-matter parse, so the refusal
# node breaks that rather than stubbing a rail that would not have run.
BAD_BODY = "---\nstatus: done\nreview: [unclosed\n---\n# item\n"


def _wait_for(pred, timeout=30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _git(vault, *args) -> str:
    return subprocess.run(["git", *args], cwd=str(vault),
                          capture_output=True, text=True).stdout.strip()


def _scratch_vault(dir_, monkeypatch, *, body=GOOD_BODY):
    """A git vault holding one tracked backlog item, modified but not committed: the
    smallest thing a real `land()` can be asked to commit, and the same shape
    `test_a_vault_landing_with_no_item_id_is_attributed_to_the_turn_that_made_it` uses."""
    vault = dir_ / "obsidian"
    (vault / "backlog").mkdir(parents=True)
    _git(vault, "init", "-q", "-b", "main")
    _git(vault, "config", "user.email", "t@e.com")
    _git(vault, "config", "user.name", "t")
    (vault / "backlog" / "4628-item.md").write_text("---\nstatus: draft\n---\n# item\n")
    _git(vault, "add", "-A")
    _git(vault, "commit", "-q", "-m", "base")
    base = _git(vault, "rev-parse", "HEAD")
    (vault / "backlog" / "4628-item.md").write_text(body)
    monkeypatch.setattr(V, "VAULT", vault)
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    return vault, base


async def _hand_back_a_blocked_land(dir_, monkeypatch, *, body=GOOD_BODY):
    """Ask the real handler for a land and hold that land inside `validate()`.

    Returns `(payload, waited_s, is_error, vault, base_sha, release)`. `release` is
    unset on return, so the land is still in flight when the handler answers and the
    node decides when, and how, it is allowed to finish.
    """
    import agent_mcp.automod as AM
    from agent_mcp import _task_registry

    vault, base = _scratch_vault(dir_, monkeypatch, body=body)
    monkeypatch.setattr(AM, "VAULT_LAND_HANDBACK_SECONDS", BOUND)
    release = threading.Event()
    real_validate = V.validate

    def held_validate(paths, session_id=None):
        release.wait(30)
        return real_validate(paths, session_id=session_id)

    monkeypatch.setattr(V, "validate", held_validate)
    token = _task_registry.current_session_id.set("s4628")
    try:
        began = time.monotonic()
        res = await AM.call_tool("automod_vault_land", LAND_ARGS)
        waited = time.monotonic() - began
    finally:
        _task_registry.current_session_id.reset(token)
    return json.loads(res.content[0].text), waited, res.is_error, vault, base, release


def _finish_the_land(release):
    """Let the held land run to its own end and wait for it — see the section note."""
    release.set()
    assert _wait_for(lambda: S.read_vault_land_marker(session_id="s4628") is None), \
        "the land's thread never reached its finally"


def _land_rows():
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "vault_land"]


def test_the_handback_bound_is_under_the_ceiling_it_exists_to_beat():
    """#4628 clause 1's number, pinned where it is written.

    The bound exists to precede `CALL_TIMEOUT_SECONDS` (660.0, `app/harness/mcp_pool.py`),
    past which the pool reclaims the call, the run sits with no harness event, and the
    autonomy cap reads as the cause of a death the land caused. 120 s sits under that
    ceiling and under the tightest `timeout_seconds` of the tasks that still call the
    tool (#78's 1200 s, since #4563 took the land off #86), while the November median
    land cost 900 s — so an ordinary land is always the one handed back, never the
    caller.
    """
    import agent_mcp.automod as AM
    from app.harness.mcp_pool import CALL_TIMEOUT_SECONDS

    assert 0 < AM.VAULT_LAND_HANDBACK_SECONDS <= 120.0, AM.VAULT_LAND_HANDBACK_SECONDS
    assert AM.VAULT_LAND_HANDBACK_SECONDS < CALL_TIMEOUT_SECONDS


async def test_a_land_still_running_when_the_bound_expires_hands_the_caller_back(
        _scratch_state, monkeypatch):
    """Clause 1, across the seam the fix lives on: `agent_mcp.automod.call_tool`.

    The land is held before any `git add`, so at the instant the handler answers nothing
    is committed and the marker is live: the answer has to say all of that, name the
    marker's own fields, and tell the caller not to call again.
    """
    out, waited, is_error, vault, base, release = await _hand_back_a_blocked_land(
        _scratch_state, monkeypatch)
    try:
        assert not is_error, out
        assert waited < BOUND + 2, \
            f"the caller waited {waited:.1f} s for a land that was still blocked"
        assert out["vault_land_in_flight"] is True and out["landed"] is False, out
        assert out["commit"] is None, "an in-flight land answered with a commit sha"
        assert out["session_id"] == "s4628" and out["item_id"] is None, out
        assert out["stage"] == "review" and out["elapsed_s"] >= 0, out
        assert out["paths"] == ["backlog/4628-item.md"], out
        assert "no vault commit has happened yet" in out["no_commit_yet"], out
        assert "Do not call automod_vault_land again" in out["next"], out
        assert _git(vault, "rev-parse", "HEAD") == base, \
            "the land committed on its way to handing the caller back"
        assert S.vault_land_in_progress(session_id="s4628") is not None
    finally:
        _finish_the_land(release)


async def test_the_handed_back_land_still_writes_its_row_and_clears_its_marker(
        _scratch_state, monkeypatch):
    """Clause 3: the early answer is not an early exit.

    Once released, the detached land runs the real writer: exactly one `vault_land`
    row, carrying the commit sha and the paths a synchronous call would have written,
    and then the marker is gone so a later land is not refused.
    """
    out, waited, is_error, vault, base, release = await _hand_back_a_blocked_land(
        _scratch_state, monkeypatch)
    assert out["vault_land_in_flight"] is True, out
    assert _land_rows() == [], "the land ledgered a row before its caller was handed back"
    _finish_the_land(release)

    rows = _land_rows()
    assert len(rows) == 1, f"expected exactly one vault_land row, got {rows}"
    head = _git(vault, "rev-parse", "HEAD")
    assert head != base, "the detached land never committed"
    assert rows[0]["ok"] is True and rows[0]["commit"] == head, rows[0]
    assert rows[0]["paths"] == ["backlog/4628-item.md"], rows[0]
    assert rows[0]["session_id"] == "s4628", rows[0]
    assert S.vault_land_in_progress(session_id="s4628") is None, \
        "the marker outlived the land, so the next call would be refused"


async def test_a_land_that_fails_validation_after_the_handback_commits_nothing(
        _scratch_state, monkeypatch):
    """Clause 4: handing the caller back must not launder a failure.

    The refusal arrives after the early answer, in the land's own thread. Nothing may
    be committed, the vault tree has to be back as it was, and the ledger still has to
    carry the refusal naming the paths it refused — the record is the only thing the
    caller is still connected to.
    """
    out, waited, is_error, vault, base, release = await _hand_back_a_blocked_land(
        _scratch_state, monkeypatch, body=BAD_BODY)
    assert out["vault_land_in_flight"] is True, out
    _finish_the_land(release)

    rows = _land_rows()
    assert len(rows) == 1, f"expected exactly one vault_land row, got {rows}"
    assert rows[0]["ok"] is False, rows[0]
    assert rows[0]["paths"] == ["backlog/4628-item.md"], rows[0]
    assert rows[0]["errors"] == [
        "backlog/4628-item.md: front matter is not valid YAML: while parsing a flow sequence"], \
        rows[0]
    assert _git(vault, "rev-parse", "HEAD") == base, "a refused land committed a vault change"
    assert _git(vault, "status", "--porcelain") == "", "the vault tree was left changed"
    assert "status: draft" in (vault / "backlog" / "4628-item.md").read_text()
    assert S.vault_land_in_progress(session_id="s4628") is None


async def test_a_land_that_finishes_inside_the_bound_answers_exactly_as_the_lander_dict(
        _scratch_state, monkeypatch):
    """Clause 2, on the hop the lander's own dict does not cross by itself.

    A land that comes back within the bound is not re-projected: the caller's JSON is
    `land()`'s return plus `note`, keys and all. `commit`, `paths` and `guards` are what
    the attribution and seam guards read, so a fast path that grew a wrapper shape
    would break them while still looking green here.
    """
    import agent_mcp.automod as AM
    from agent_mcp import _task_registry

    monkeypatch.setattr(AM, "VAULT_LAND_HANDBACK_SECONDS", 30.0)
    returned = {"ok": True, "commit": "a15aa2ebf", "paths": ["backlog/4628-item.md"],
                "guards": {"state": "skipped", "refuse": False}}
    monkeypatch.setattr(V, "_land", lambda paths, message, **kw: dict(returned))
    token = _task_registry.current_session_id.set("s-fast")
    try:
        began = time.monotonic()
        res = await AM.call_tool("automod_vault_land", LAND_ARGS)
    finally:
        _task_registry.current_session_id.reset(token)
    out = json.loads(res.content[0].text)
    assert not res.is_error, out
    assert time.monotonic() - began < BOUND, "a land that finished fast was not fast"
    assert {k: v for k, v in out.items() if k != "note"} == returned, out
    assert "vault_land_in_flight" not in out, out


async def test_a_land_with_no_marker_key_is_not_detached(_scratch_state, monkeypatch):
    """The bound pays for itself only while a marker proves the land is single.

    `state.vault_land_marker_path` has no key when neither an item nor a session is
    present, so a detached call there could be followed by a second one that commits
    the same paths twice — the failure #4583's refusal exists to stop. Such a caller is
    handed the synchronous land it always had, and the wait is its own witness.
    """
    import agent_mcp.automod as AM

    monkeypatch.setattr(AM, "VAULT_LAND_HANDBACK_SECONDS", BOUND)
    S.VAULT_LAND_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(V, "_land", lambda paths, message, **kw: time.sleep(1.0)
                        or {"ok": True, "commit": "deadbeef"})
    began = time.monotonic()
    res = await AM.call_tool("automod_vault_land", LAND_ARGS)   # no session, no item
    waited = time.monotonic() - began
    out = json.loads(res.content[0].text)
    assert waited > BOUND, "a land with no marker key was detached from its caller"
    assert out["ok"] is True and out["commit"] == "deadbeef", out
    assert not list(S.VAULT_LAND_DIR.glob("*.json")), "a keyless land wrote a marker"
