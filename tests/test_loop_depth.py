"""Two rounds and two triage turns at once, and the three things that cost
#1199 and #1204 their night (2026-09-17).

#1204 passed all nine rungs on its fifth attempt. Three earlier gates had
every clause graded `met` and were refused by a regex that reads any new
`pytest.skip` as a test that cannot fail; the landing then gave up at 900 s,
112 seconds before the 37-minute scheduled task it was waiting on finished;
and because the item's attempts were already spent on those refusals, the
external landing failure read as `spent` and a finished change went back to
triage. The depth work is pinned here too because it leans on the same
landing wait: with two rounds, a landing's idle wait IS the other round's turn.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, gate as G, promote as P, review as RV, round as R
from scripts.automod import state as S
from workers.sources import autocode as I
from workers.sources import autotriage as M

ROOT = Path(__file__).resolve().parent.parent


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    B._TRIAGE_CLAIMS.clear()
    yield d
    B._TRIAGE_CLAIMS.clear()


def write_item(d: Path, item_id, *, status="draft", days_old=100, tags=()):
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created, "board": "lloyd",
          "id": item_id, "name": f"Thing {item_id}", "tags": list(tags)}
    path = d / f"{item_id}-thing.md"
    path.write_text("---\n" + yaml.safe_dump(fm) + "---\nDo the thing.\n", encoding="utf-8")
    return path


# ── the skip precheck ──────────────────────────────────────────────────────

@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    (r / "tests").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "tests" / "test_a.py").write_text("def test_old():\n    assert 1\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    return r, git(r, "rev-parse", "HEAD").stdout.strip()


def _prechecks(repo, body):
    r, base = repo
    (r / "tests" / "test_a.py").write_text(body)
    git(r, "commit", "-qam", "round")
    return RV.honesty_prechecks(r, base, ["tests/test_a.py"])


def test_a_skip_behind_a_condition_is_advisory(repo):
    """#1204's shape: the test skips when the machine lacks what it measures."""
    out = _prechecks(repo, (
        "import pytest\n\ndef test_new(tmp_path):\n"
        "    if not (tmp_path / 'vault').exists():\n"
        "        pytest.skip('no vault on this box')\n"
        "    assert 1\n"))
    assert [o["severity"] for o in out] == ["advisory"], out
    assert "conditional" in out[0]["problem"] and out[0]["line"] == 5


def test_a_skipif_marker_is_advisory_and_a_bare_skip_marker_still_blocks(repo):
    out = _prechecks(repo, (
        "import pytest, yaml\n\n"
        "@pytest.mark.skipif(not hasattr(yaml, 'CSafeLoader'), reason='no libyaml')\n"
        "def test_new():\n    assert 1\n"))
    assert [o["severity"] for o in out] == ["advisory"], out


def test_an_unconditional_skip_still_blocks(repo):
    """The pattern's reason for existing: a test that cannot fail."""
    marker = _prechecks(repo, "import pytest\n\n@pytest.mark.skip(reason='later')\ndef test_new():\n    assert 0\n")
    assert [o["severity"] for o in marker] == ["blocking"], marker


def test_a_skip_as_the_first_statement_of_a_test_blocks(repo):
    out = _prechecks(repo, "import pytest\n\ndef test_new():\n    pytest.skip('todo')\n    assert 0\n")
    assert [o["severity"] for o in out] == ["blocking"], out


def test_an_advisory_precheck_does_not_refuse_under_either_policy():
    parsed = {"premise": "sound", "summary": "", "clauses": [
        {"clause": 1, "verdict": "met", "note": "", "downgraded": []}],
        "test_honesty": [], "seams_unverified": [], "amendments_ok": True}
    pre = [{"file": "tests/t.py", "line": 5, "problem": "a new pytest.skip (conditional)",
            "severity": "advisory"}]
    assert RV.decide_by_grader(parsed, pre)[0] != "retry"
    assert RV.decide(parsed, pre)[0] != "retry"
    pre[0]["severity"] = "blocking"
    assert RV.decide_by_grader(parsed, pre)[0] == "retry"


# ── the landing's idle wait ────────────────────────────────────────────────

def _backend(monkeypatch, *, turns, jobs):
    """`turns` and `jobs` are callables of the poll number."""
    n = {"health": 0}

    def fake_get(url, timeout=5.0):
        if url.endswith("/api/workers/status"):
            return 200, {"pool": {"paused": True, "in_flight": jobs(n["health"])}}
        n["health"] += 1
        return 200, {"turns": turns(n["health"])}
    monkeypatch.setattr(P, "_get", fake_get)
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: True)
    monkeypatch.setattr(P, "pool_paused", lambda: True)
    return n


def test_the_idle_budget_does_not_burn_while_a_pool_job_is_what_holds_the_backend(monkeypatch):
    """#1204: scheduled task #74 outlasted a 900 s budget by 112 s. With the
    pool paused that job is a bounded wait, so the budget stands still."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(P.time, "time", lambda: clock["t"])
    monkeypatch.setattr(P.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 60))
    busy_until = 40   # polls: 40 min, far past the 900 s budget
    _backend(monkeypatch,
             turns=lambda i: {"active": 0, "queued": 0, "harness_runs": 1 if i <= busy_until else 0},
             jobs=lambda i: {"7": {"source": "scheduled-task"}} if i <= busy_until else {})
    ok, why = P.wait_idle(max_wait=900)
    assert ok, why


def test_the_idle_budget_still_burns_when_nothing_of_the_pools_is_in_flight(monkeypatch):
    """A chat turn or a leaked counter: nothing bounds it, so the budget does."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(P.time, "time", lambda: clock["t"])
    monkeypatch.setattr(P.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 60))
    _backend(monkeypatch, turns=lambda i: {"active": 0, "queued": 0, "harness_runs": 1},
             jobs=lambda i: {})
    ok, why = P.wait_idle(max_wait=900)
    assert not ok and "never went idle" in why
    assert clock["t"] - 1000.0 < 1200, "gave up at the budget, not the ceiling"


def test_the_hard_ceiling_bounds_a_job_that_never_ends(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(P.time, "time", lambda: clock["t"])
    monkeypatch.setattr(P.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 60))
    monkeypatch.setattr(P.S, "landing_cfg", lambda repo=None: {"idle_hard_max_wait_s": 3000})
    _backend(monkeypatch, turns=lambda i: {"active": 0, "queued": 0, "harness_runs": 1},
             jobs=lambda i: {"7": {"source": "scheduled-task"}})
    ok, why = P.wait_idle(max_wait=900)
    assert not ok and 2900 <= clock["t"] - 1000.0 <= 3200


def test_a_landing_waits_for_the_other_rounds_turn_without_pausing_anything(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(P.time, "time", lambda: clock["t"])
    monkeypatch.setattr(P.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    paused = []
    monkeypatch.setattr(P, "set_pool_paused", lambda on: paused.append(on) or True)
    polls = {"n": 0}

    def fake_get(url, timeout=5.0):
        polls["n"] += 1
        jobs = {"1": {"source": "autocode"}, "2": {"source": "autotriage"}} if polls["n"] < 5 \
            else {"2": {"source": "autotriage"}}
        return 200, {"pool": {"in_flight": jobs}}
    monkeypatch.setattr(P, "_get", fake_get)
    ok, why = P.wait_for_rounds(4500)
    assert ok and polls["n"] == 5 and paused == [], "triage keeps running; only the round is waited on"
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {"pool": {"in_flight": {"1": {"source": "autocode"}}}}))
    ok, why = P.wait_for_rounds(120)
    assert not ok and "still in flight" in why
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (None, None))
    assert P.wait_for_rounds(120)[0], "an unreadable pool goes straight to the drain"


def test_the_landing_config_is_read_and_promote_waits_for_rounds_before_the_drain():
    cfg = S.landing_cfg(ROOT)
    assert float(cfg["idle_hard_max_wait_s"]) >= 3600 + 600, "above the longest worker max_duration"
    # The wait for the other round's turn happens in `round.land`, BEFORE the
    # automod lock: under it, the turn being waited for gets `LockHeld` from
    # `automod_start` and the two block each other (#1215, the first
    # concurrent landing).
    land = (ROOT / "scripts/automod/round.py").read_text()
    assert land.index("P.wait_for_rounds(") < land.index("lock = _land_lock(round_id")
    assert "wait_for_rounds(" not in (ROOT / "scripts/automod/promote.py").read_text().split(
        "def promote(")[1], "never inside promote, which runs under the lock"


# ── an external landing failure keeps its own count ────────────────────────

def _implement(item_id, rid, phase="finished", **kw):
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": phase,
                    "round_id": rid, "stop_reason": "stop", "num_turns": 50, **kw}, path=S.LEDGER_PATH)


def test_a_gate_passed_round_whose_landing_failed_externally_is_not_spent_by_earlier_attempts(isolated):
    """#1204's row: five attempts in, all nine rungs green, landing lost to
    someone else's job. `attempts > EXTERNAL_RETRY_CAP` used to say spent."""
    write_item(isolated, 7, status="up_next")
    S.append_event({"event": "backlog_triage", "item_id": 7, "verdict": "confirmed",
                    "acceptance": "it works"}, path=S.LEDGER_PATH)
    for n in range(B.EXTERNAL_RETRY_CAP + 1):
        _implement(7, f"SM_{n}", phase="started")
        _implement(7, f"SM_{n}")
    rid = "SM_LAST"
    _implement(7, rid, phase="started")
    S.append_event({"event": "gate", "round_id": rid, "rung": "drill", "ok": True}, path=S.LEDGER_PATH)
    _implement(7, rid)
    S.append_event({"event": "land_failed", "round_id": rid, "ok": False, "external_blocker": True,
                    "detail": "backend never went idle"}, path=S.LEDGER_PATH)
    verdict, why = B.implement_outcomes(S.LEDGER_PATH)[7]
    assert verdict == "external" and rid in why

    # …and it is still bounded: the landing's OWN failures count.
    for n in range(B.EXTERNAL_RETRY_CAP):
        r2 = f"SM_L{n}"
        _implement(7, r2, phase="started"); _implement(7, r2)
        S.append_event({"event": "land_failed", "round_id": r2, "ok": False,
                        "external_blocker": True}, path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[7][0] == "spent"


def test_a_red_tree_is_still_capped_on_attempts(isolated):
    write_item(isolated, 8, status="up_next")
    for n in range(B.EXTERNAL_RETRY_CAP + 1):
        rid = f"SM_T{n}"
        _implement(8, rid, phase="started")
        S.append_event({"event": "gate", "round_id": rid, "rung": "tests", "ok": False,
                        "external_blocker": True}, path=S.LEDGER_PATH)
        _implement(8, rid)
    assert B.implement_outcomes(S.LEDGER_PATH)[8][0] == "spent"


# ── depth: autocode ────────────────────────────────────────────────────────

def test_depth_comes_from_max_inflight_and_defaults_to_one():
    assert I.round_depth({}) == 1 and I.round_depth({"max_inflight": 2}) == 2
    assert I.round_depth({"max_inflight": "junk"}) == 1 and I.round_depth({"max_inflight": 0}) == 1
    assert M.triage_depth({}) == 1 and M.triage_depth({"max_inflight": 2}) == 2
    live = yaml.safe_load((ROOT / "config.yaml").read_text())["workers"]
    rounds = int(live["sources"]["autocode"]["max_inflight"])
    triages = int(live["sources"]["autotriage"]["max_inflight"])
    assert int(live["slots"]) >= rounds + triages + 1, \
        "a slot beyond the long-lived turns, or a scheduled task queues behind them"


def _free_loop(monkeypatch, worktrees):
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: True)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_rollback_request", lambda: None)
    from scripts.automod import worktree as W
    monkeypatch.setattr(W, "prune_orphans", lambda repo: list(worktrees))


def test_the_loop_is_free_until_depth_rounds_are_open(monkeypatch):
    one = [str(I._LOOP_WORKTREE_ROOT / "SM_A" / "home" / "lloyd")]
    _free_loop(monkeypatch, one)
    assert I._loop_is_free(1)[0] is False
    assert I._loop_is_free(2) == (True, "free")
    _free_loop(monkeypatch, one + [str(I._LOOP_WORKTREE_ROOT / "SM_B" / "home" / "lloyd")])
    free, why = I._loop_is_free(2)
    assert free is False and "depth 2" in why


def test_a_landing_promotion_holds_every_new_round_whatever_the_depth(monkeypatch):
    """What lets a landing wait out the other round's turn: nothing new starts."""
    _free_loop(monkeypatch, [])
    monkeypatch.setattr(S, "read_current", lambda: {"state": "landing", "commit": "abc"})
    monkeypatch.setattr(S, "chamber_enabled", lambda repo=None: True)
    assert I._loop_is_free(2)[0] is False


def test_two_slots_queue_two_rows_and_no_more(isolated, monkeypatch, tmp_path):
    from workers.queue import WorkQueue
    from workers.sources import DECLINED
    q = WorkQueue(tmp_path / "workers.db")
    monkeypatch.setattr(I, "_housekeeping_due", lambda queue, cfg: False)
    monkeypatch.setattr(I, "_boot_settled", {"done": True})
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    monkeypatch.setattr(B, "select_confirmed", lambda ledger: (object(), {}))
    cfg = {"interval_seconds": 900, "retry_seconds": 60, "max_inflight": 2}
    assert asyncio.run(I.enqueue_if_due(q, cfg)) == DECLINED, "a slot is still empty: look again soon"
    assert asyncio.run(I.enqueue_if_due(q, cfg)) is None
    assert asyncio.run(I.enqueue_if_due(q, cfg)) == DECLINED
    keys = sorted(i.dedup_key for i in q.list_items(source=I.NAME))
    assert keys == [I.DEDUP_KEY, f"{I.DEDUP_KEY}:1"]
    # Depth 1 is the old behaviour exactly, on the old key.
    q1 = WorkQueue(tmp_path / "w1.db")
    assert asyncio.run(I.enqueue_if_due(q1, {"interval_seconds": 900})) is None
    assert asyncio.run(I.enqueue_if_due(q1, {"interval_seconds": 900, "retry_seconds": 60})) == DECLINED
    assert [i.dedup_key for i in q1.list_items(source=I.NAME)] == [I.DEDUP_KEY]


def test_a_turn_is_credited_with_its_own_round_not_the_latest_one():
    """By time alone, turn A's `finished` row names round B, and the reaper is
    handed a round that is still being worked on."""
    ev = [{"event": "round_start", "ts": 10.0, "round_id": "SM_A", "item_id": 1, "session_id": "sA"},
          {"event": "round_start", "ts": 20.0, "round_id": "SM_B", "item_id": 2, "session_id": "sB"}]
    assert I._round_opened_since(ev, 5.0) == "SM_B", "unfiltered: the old rule"
    assert I._round_opened_since(ev, 5.0, item_id=1, session_id="sA") == "SM_A"
    assert I._round_opened_since(ev, 5.0, item_id=1) == "SM_A", "the timeout path has no session"
    assert I._round_opened_since(ev, 5.0, item_id=3, session_id="sC") is None
    old = [{"event": "round_start", "ts": 10.0, "round_id": "SM_OLD"}]
    assert I._round_opened_since(old, 5.0, item_id=1, session_id="sA") == "SM_OLD", \
        "a row from before the fields existed matches as it always did"


# ── depth: triage claims ───────────────────────────────────────────────────

def test_a_claimed_item_leaves_every_triage_pool_until_released(isolated):
    for i in (1, 2, 3):
        write_item(isolated, i)
    assert B.select_candidate(S.LEDGER_PATH).id == 1
    B.claim_for_triage("run-a", [1])
    assert B.select_candidate(S.LEDGER_PATH).id == 2
    assert 1 not in {i.id for i in B.sweep_pool(S.LEDGER_PATH)}
    B.claim_for_triage("run-b", [2])
    assert B.select_candidate(S.LEDGER_PATH).id == 3
    B.release_triage_claim("run-a")
    assert B.select_candidate(S.LEDGER_PATH).id == 1 and B.triage_claimed_ids() == {2}


def test_a_cluster_with_a_claimed_member_is_skipped_whole(isolated):
    for i in (1, 2, 3, 4, 5):
        write_item(isolated, i)
    clusters = {"clusters": [{"id": "c1", "item_ids": [1, 2, 3]}, {"id": "c2", "item_ids": [4, 5]}]}
    assert B.select_cluster(S.LEDGER_PATH, clusters, min_size=2)[0]["id"] == "c1"
    B.claim_for_triage("run-a", [1])
    assert B.select_cluster(S.LEDGER_PATH, clusters, min_size=2)[0]["id"] == "c2", \
        "not a second group over what is left of c1"


def test_two_triage_runs_started_together_take_different_items(isolated, monkeypatch):
    for i in (1, 2):
        write_item(isolated, i)
    taken: list[int] = []
    both_in = asyncio.Event()

    async def fake_turn(prompt, **kw):
        taken.append(int(kw["title"].split("#")[1].split(":")[0]))
        if len(taken) == 2:
            both_in.set()
        await asyncio.wait_for(both_in.wait(), 5)   # both turns are in flight at once
        return {"text": "", "session_id": f"s{len(taken)}", "stop_reason": "max_turns",
                "num_turns": 1, "errors": [], "structured": None, "structured_error": ""}
    from workers.sources import _common as C
    monkeypatch.setattr(C, "run_prompt_in_session", fake_turn)
    monkeypatch.setattr(B, "implement_pool_full", lambda ledger, floor=20: {
        "full": False, "ready": 0, "bound": 20, "landed_items_7d": 0, "floor": 20})

    class _Q:
        def __init__(self, i): self.id, self.payload = i, {"group_triage": False}

    async def both():
        return await asyncio.gather(M.execute(_Q(101)), M.execute(_Q(102)))
    asyncio.run(both())
    assert sorted(taken) == [1, 2]
    assert B.triage_claimed_ids() == set(), "released however the turn ended"


def test_triage_queues_a_row_per_free_slot(tmp_path):
    from workers.queue import WorkQueue
    q = WorkQueue(tmp_path / "workers.db")
    asyncio.run(M.enqueue_if_due(q, {"max_inflight": 2}))
    asyncio.run(M.enqueue_if_due(q, {"max_inflight": 2}))
    assert sorted(i.dedup_key for i in q.list_items(source=M.NAME)) == [M.DEDUP_KEY, f"{M.DEDUP_KEY}:1"]
    q1 = WorkQueue(tmp_path / "w1.db")
    asyncio.run(M.enqueue_if_due(q1, {}))
    asyncio.run(M.enqueue_if_due(q1, {}))
    assert [i.dedup_key for i in q1.list_items(source=M.NAME)] == [M.DEDUP_KEY]


# ── two gates, one machine ─────────────────────────────────────────────────

def test_a_second_gate_queues_for_the_suite_and_the_canary_ports(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "GATE_TESTS_LOCK_PATH", tmp_path / "t.lock")
    monkeypatch.setattr(S, "GATE_CANARY_LOCK_PATH", tmp_path / "c.lock")
    order: list[str] = []

    def gate(name, hold):
        g = G.Gate.__new__(G.Gate)
        g.round_id, g._canary_lock = name, None
        g.SERIAL_MAX_WAIT = 10.0

        def tests():
            order.append(f"{name}:in"); time.sleep(hold); order.append(f"{name}:out")
            return True, "ok", {}
        ok, _, data = g._serialized("tests", tests)()
        assert ok and "lock_wait_s" in data
        boot = g._serialized("canary_boot", lambda: (True, "booted", {}))
        assert boot()[0] and g._canary_lock is not None
        # smoke and the drill reuse the lock the boot took; a reused boot never ran.
        assert g._serialized("drill", lambda: (True, "ok", {}))()[0]
        g._canary_lock.release()

    a = threading.Thread(target=gate, args=("A", 0.4)); b = threading.Thread(target=gate, args=("B", 0.0))
    a.start(); time.sleep(0.1); b.start(); a.join(); b.join()
    assert order == ["A:in", "A:out", "B:in", "B:out"], order
    assert G.Gate._serialized(G.Gate.__new__(G.Gate), "review", len) is len, "the long rung overlaps"


def test_a_landing_queues_for_the_lock_and_rechecks_the_chamber_after_taking_it(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(S, "chamber_enabled", lambda repo=None: True)
    monkeypatch.setattr(R, "LAND_LOCK_POLL", 0.02)
    # The winner's promotion is written while we queue, and clears when waited out.
    world = {"released": False, "settled": False}

    def current():
        if world["released"] and not world["settled"]:
            return {"state": "observing", "commit": "abc"}
        return None
    monkeypatch.setattr(S, "read_current", current)
    waited: list = []

    def settle(**kw):
        waited.append(kw["observed"]["commit"])
        world["settled"] = True
    monkeypatch.setattr(P, "wait_for_settle", settle)
    winner = S.Lock(owner="land-SM_A").acquire()
    threading.Timer(0.1, lambda: (world.__setitem__("released", True), winner.release())).start()
    lock = R._land_lock("SM_B")
    try:
        assert waited == ["abc"], "released and waited out the window written while it queued"
    finally:
        lock.release()
    held = S.Lock(owner="x").acquire()
    try:
        with pytest.raises(S.LockHeld):
            R._land_lock("SM_C", dry_run=True)
    finally:
        held.release()


def test_a_round_that_passed_its_gate_holds_the_other_slot(monkeypatch, tmp_path):
    """#1204's gate passed at 17:03:34 and the free slot was claimed at
    17:03:53, before `current.json` read `landing`: the landing then had a
    whole fresh turn to wait out."""
    import json as _json
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path)
    wt = [str(I._LOOP_WORKTREE_ROOT / "SM_A" / "home" / "lloyd")]
    _free_loop(monkeypatch, wt)
    assert I._loop_is_free(2) == (True, "free"), "an open round that has not gated holds nothing"
    (tmp_path / "SM_A").mkdir()
    (tmp_path / "SM_A" / "gate.json").write_text(_json.dumps({"ok": False}))
    assert I._loop_is_free(2)[0] is True, "a refused gate is a round still being worked on"
    (tmp_path / "SM_A" / "gate.json").write_text(_json.dumps({"ok": True}))
    free, why = I._loop_is_free(2)
    assert free is False and "SM_A passed its gate" in why
    monkeypatch.setattr(S, "gate_in_progress", lambda rid: {"pid": 1})
    assert I._loop_is_free(2)[0] is True, "re-gating after an edit: the old pass no longer speaks"


def test_preflight_does_not_call_the_other_gates_canary_a_stale_one(tmp_path, monkeypatch):
    """SM_20260917_184334 was refused at preflight, two seconds in, because the
    other round's canary was up: the port check ran before anything queued
    for the canary lock."""
    monkeypatch.setattr(S, "GATE_CANARY_LOCK_PATH", tmp_path / "c.lock")
    assert G._canary_lock_held() is False
    other = S.Lock(S.GATE_CANARY_LOCK_PATH, owner="gate-SM_A").acquire()
    try:
        assert G._canary_lock_held() is True
    finally:
        other.release()
    assert G._canary_lock_held() is False, "the probe itself must not leave the lock taken"
    src = (ROOT / "scripts/automod/gate.py").read_text()
    assert src.index("if not _canary_lock_held():") < src.index("is in use (stale canary?)")
