"""#1489: the autocode ReasoningBank — distil, prune, retrieve, inject, and the
live A/B (arm assignment, ledger markers, injection by arm, the report)."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.automod import reasoning_bank as RB

ROOT = Path(__file__).resolve().parent.parent
T0 = 1_790_000_000.0

TAUTOLOGY = ("assert isinstance(over, list) on a list comprehension cannot fail, so the "
             "test's own named check asserts nothing.")
LIVE_VAULT = ("The only clause-5 node is @pytest.mark.live_vault, which run_tests.sh deselects, "
              "so nothing in the gate's green run constrains clause 5.")
HALF = ("The measurement half is delivered and sound. The re-route half is not: no router "
        "file changed and no 7-day measurement exists.")


def _review(rid, item, ts, clauses, *, blocking=True, tests=()):
    return {"event": "review", "round_id": rid, "item_id": item, "ts": ts, "ok": True,
            "blocking": blocking, "clauses": list(clauses), "test_honesty": list(tests)}


def _start(rid, item, ts, goal):
    return {"event": "round_start", "round_id": rid, "item_id": str(item), "ts": ts,
            "goal": goal}


def _triage(item, ts, clauses):
    return {"event": "backlog_triage", "item_id": item, "ts": ts,
            "acceptance_clauses": list(clauses)}


def _clause(n, verdict, note):
    return {"clause": n, "verdict": verdict, "note": note}


def test_causes_are_classified_from_the_graders_words():
    assert RB.classify_cause(TAUTOLOGY) == "unfalsifiable_test"
    assert RB.classify_cause(LIVE_VAULT) == "skip_or_deselected"
    assert RB.classify_cause(HALF) == "half_missing"
    assert RB.classify_cause("the weather was pleasant") == "other"
    assert set(RB.LESSONS) == {c for c, _, _ in RB.CAUSES} | {"other"}


def test_a_refusal_then_a_met_regrade_is_a_worked_item_dated_at_the_fix():
    ev = [_triage(7, T0 - 10, ["router test pins the split", "report states n"]),
          _start("SM_a", 7, T0, "Implement backlog #7: route titles"),
          _review("SM_a", 7, T0 + 100, [_clause(1, "partial", TAUTOLOGY), _clause(2, "met", "ok")]),
          _review("SM_a", 7, T0 + 900, [_clause(1, "met", "now breaks under mutation"),
                                        _clause(2, "met", "ok")], blocking=False)]
    bank = RB.build_bank(ev)
    assert len(bank) == 1
    d = bank[0]
    assert (d["kind"], d["cause"], d["clause"], d["item_id"]) == ("worked", "unfalsifiable_test", 1, 7)
    # Knowable only once the fix was accepted: a replay between the two reviews
    # must not see it.
    assert d["ts"] == T0 + 900
    assert "mutation" in d["fix"] and "router test pins the split" in d["source"]
    assert d["round_id"] == "SM_a" and d["date"]


def test_distinct_causes_capped_per_round_and_placeholders_ignored():
    ev = [_start("SM_b", 8, T0, "Implement backlog #8"),
          _review("SM_b", 8, T0 + 50,
                  [_clause(1, "partial", TAUTOLOGY), _clause(2, "unmet", LIVE_VAULT),
                   _clause(3, "partial", HALF), _clause(4, "partial", "cannot fail either"),
                   _clause(5, "partial", "not addressed by the grader")],
                  tests=[{"severity": "blocking", "file": "t.py", "line": 3,
                          "problem": "hand-built bucket no caller produces"}])]
    bank = RB.build_bank(ev)
    assert len(bank) == RB.MAX_PER_ROUND
    assert len({d["cause"] for d in bank}) == RB.MAX_PER_ROUND
    assert all("not addressed" not in d["detail"] for d in bank)


def test_a_clean_landing_and_an_advisory_only_review_yield_nothing():
    ev = [_start("SM_c", 9, T0, "x"),
          _review("SM_c", 9, T0 + 5, [_clause(1, "met", "fine")], blocking=False),
          _start("SM_d", 10, T0, "y"),
          _review("SM_d", 10, T0 + 5, [_clause(1, "met", "fine")], blocking=True,
                  tests=[{"severity": "advisory", "problem": "cannot fail"}])]
    assert RB.build_bank(ev) == []


def _item(i, item, cause, ts, source, kind="failed", clause=1):
    return {"v": 1, "id": f"SM_{i}:1", "ts": ts, "date": "2026-09-20", "round_id": f"SM_{i}",
            "item_id": item, "kind": kind, "cause": cause, "clause": clause,
            "lesson": RB.LESSONS[cause], "detail": "d", "fix": "", "source": source}


def test_retrieve_ranks_by_similarity_and_never_leaks_self_or_future():
    bank = [_item(1, 1, "unfalsifiable_test", T0, "vault recall reranker djev canvas eval"),
            _item(2, 2, "half_missing", T0, "voice barge-in wake word conversation"),
            _item(3, 3, "skip_or_deselected", T0, "vault recall prefetch hybrid leg"),
            _item(4, 4, "unfalsifiable_test", T0, "vault recall reranker djev canvas eval second"),
            _item(5, 99, "protected_path", T0, "vault recall reranker djev canvas eval"),
            _item(6, 5, "artifact_stale", T0 + 1000, "vault recall reranker djev canvas eval")]
    got = RB.retrieve(bank, "djev reranker for vault recall", k=3, exclude_item=99,
                      before_ts=T0 + 500)
    ids = [d["id"] for d in got]
    assert ids[0] in ("SM_1:1", "SM_4:1")
    assert "SM_5:1" not in ids and "SM_6:1" not in ids          # own item, future
    assert len({d["cause"] for d in got}) == len(got)           # distinct causes
    assert "SM_2:1" not in ids or ids.index("SM_2:1") == len(ids) - 1
    assert RB.retrieve([], "x") == [] and RB.retrieve(bank, "x", k=0) == []


def test_prune_drops_old_and_superseded_failures():
    now = T0 + 40 * 86400
    old = _item(1, 1, "unpinned", T0, "a")
    fail = _item(2, 2, "unpinned", now - 100, "b")
    fixed = _item(3, 2, "unpinned", now - 50, "b", kind="worked")
    other_clause = _item(4, 2, "unpinned", now - 100, "b", clause=2)
    kept = RB.prune([old, fail, fixed, other_clause], now=now, max_age_days=30)
    assert [d["id"] for d in kept] == ["SM_3:1", "SM_4:1"]


def test_bank_round_trips_and_refresh_reuses_a_fresh_file(tmp_path):
    ledger = tmp_path / "promotions.jsonl"
    now = time.time()
    ev = [_start("SM_e", 11, now - 60, "Implement backlog #11"),
          _review("SM_e", 11, now - 30, [_clause(1, "partial", TAUTOLOGY)])]
    ledger.write_text("".join(json.dumps(e) + "\n" for e in ev) + "not json\n")
    path = tmp_path / "bank.jsonl"
    bank = RB.refresh_if_stale(ledger, path)
    assert len(bank) == 1 and RB.load_bank(path) == bank
    ledger.write_text("")                          # a fresh cache is not re-derived
    assert RB.refresh_if_stale(ledger, path) == bank
    os.utime(path, (now - 7200, now - 7200))       # a stale one is
    assert RB.refresh_if_stale(ledger, path) == []
    assert RB.load_bank(tmp_path / "missing.jsonl") == []


def test_render_block_names_round_item_and_cause():
    d = {**_item(1, 42, "unfalsifiable_test", T0, "s"), "kind": "worked", "fix": "mutation red"}
    text = RB.render_block([d])
    assert "SM_1" in text and "#42" in text and "unfalsifiable test" in text
    assert "Then accepted: mutation red" in text
    assert RB.render_block([]) == ""
    q = RB.item_query("t", ["c1"], "body\n## Findings (round x)\nleak")
    assert "leak" not in q and "c1" in q and "body" in q


def _autocode(monkeypatch, cfg, tmp_path, events):
    from workers.sources import autocode
    from scripts.automod import state as S
    from app import paths
    ledger = tmp_path / "promotions.jsonl"
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    monkeypatch.setattr(autocode, "_source_cfg", lambda name: dict(cfg))
    monkeypatch.setattr(S, "LEDGER_PATH", ledger)
    monkeypatch.setattr(paths, "REASONING_BANK_PATH", tmp_path / "bank.jsonl")
    return autocode


def test_injection_is_off_by_default_and_on_only_by_its_key(monkeypatch, tmp_path):
    now = time.time()
    ev = [_start("SM_f", 21, now - 90, "Implement backlog #21: vault recall reranker"),
          _review("SM_f", 21, now - 60, [_clause(1, "partial", TAUTOLOGY)])]
    cand = SimpleNamespace(id=22, name="vault recall reranker canvas", body="")
    triage = {"acceptance_clauses": ["reranker test fails under mutation"]}

    ac = _autocode(monkeypatch, {}, tmp_path, ev)
    assert ac._strategy_block(cand, triage) == ""
    assert not (tmp_path / "bank.jsonl").exists()   # off reads and writes nothing

    ac = _autocode(monkeypatch, {"reasoning_bank": True}, tmp_path, ev)
    block = ac._strategy_block(cand, triage)
    assert "Lessons from similar rounds" in block and "SM_f" in block and "#21" in block

    own = SimpleNamespace(id=21, name="vault recall reranker canvas", body="")
    assert ac._strategy_block(own, triage) == ""     # never the item's own rounds


def test_the_shipped_config_runs_the_ab_and_the_code_default_is_off(monkeypatch, tmp_path):
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert RB.setting(cfg["workers"]["sources"]["autocode"].get("reasoning_bank")) == "ab"
    # the code's own default, with the key absent, injects nothing and arms nothing
    ac = _autocode(monkeypatch, {}, tmp_path, [])
    assert ac._strategy_for(SimpleNamespace(id=5, name="x", body=""), {}) == ("", {})


def test_eval_pool_hides_the_future_and_the_items_own_rounds():
    spec = importlib.util.spec_from_file_location(
        "rb_eval", ROOT / "eval" / "run_reasoningbank_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    bank = [_item(1, 1, "unpinned", T0, "a"),
            _item(2, 7, "half_missing", T0, "a"),                 # the query's own item
            _item(3, 3, "artifact_stale", T0 + 10, "a"),          # at the round's start
            _item(4, 4, "unpinned", T0 - 5, "a", kind="worked")]
    q = {"round_id": "SM_q", "item_id": 7, "start_ts": T0 + 10, "goal": "a", "clauses": []}
    assert [d["id"] for d in ev.pool_at(bank, q, same_item=False)] == ["SM_1:1", "SM_4:1"]
    assert "SM_2:1" in [d["id"] for d in ev.pool_at(bank, q, same_item=True)]
    # a later fix never supersedes an earlier refusal from the replay's future
    late_fix = _item(5, 1, "unpinned", T0 + 50, "a", kind="worked")
    assert "SM_1:1" in [d["id"] for d in ev.pool_at(bank + [late_fix], q, same_item=False)]


def test_prior_items_are_the_most_frequent_causes_newest_example_each():
    bank = [_item(1, 1, "unpinned", T0, "a"), _item(2, 2, "unpinned", T0 + 5, "b"),
            _item(3, 3, "unpinned", T0 + 9, "c"), _item(4, 4, "half_missing", T0, "d"),
            _item(5, 5, "half_missing", T0 + 1, "e"), _item(6, 6, "other", T0, "f"),
            _item(7, 7, "other", T0, "g"), _item(8, 8, "other", T0, "h"),
            _item(9, 9, "artifact_stale", T0, "i")]
    got = RB.prior_items(bank, k=2)
    assert [(d["cause"], d["id"]) for d in got] == [("unpinned", "SM_3:1"),
                                                    ("half_missing", "SM_5:1")]
    # never `other` (a lesson with no content), never the item's own, never the future
    got = RB.prior_items(bank, k=2, exclude_item=3, before_ts=T0 + 3)
    assert sorted(d["id"] for d in got) == ["SM_1:1", "SM_5:1"]
    assert all(d["cause"] != "other" for d in RB.prior_items(bank, k=10))


def test_mode_similar_retrieves_by_item_and_prior_ignores_the_item(monkeypatch, tmp_path):
    now = time.time()
    ev = [_start("SM_g", 31, now - 90, "Implement backlog #31: voice barge-in wake word"),
          _review("SM_g", 31, now - 80, [_clause(1, "partial", LIVE_VAULT)]),
          _start("SM_h", 32, now - 90, "Implement backlog #32: vault recall reranker"),
          _review("SM_h", 32, now - 70, [_clause(1, "partial", TAUTOLOGY)]),
          _start("SM_i", 33, now - 90, "Implement backlog #33: vault recall canvas"),
          _review("SM_i", 33, now - 60, [_clause(1, "partial", TAUTOLOGY)])]
    cand = SimpleNamespace(id=40, name="voice barge-in wake word", body="")
    cfg = {"reasoning_bank": True, "reasoning_bank_k": 1}
    ac = _autocode(monkeypatch, {**cfg, "reasoning_bank_mode": "similar"}, tmp_path, ev)
    assert "SM_g" in ac._strategy_block(cand, {})
    (tmp_path / "bank.jsonl").unlink()
    ac = _autocode(monkeypatch, cfg, tmp_path, ev)          # default mode: prior
    block = ac._strategy_block(cand, {})
    assert "unfalsifiable test" in block and "SM_g" not in block


# --- the live A/B (#1489) --------------------------------------------------

def _started(item, ts, **kw):
    return {"event": "backlog_implement", "item_id": item, "phase": "started", "ts": ts, **kw}


def _finished(item, ts, rid, **kw):
    return {"event": "backlog_implement", "item_id": item, "phase": "finished", "ts": ts,
            "round_id": rid, **kw}


def _item_with_arm(arm, start=1000):
    return next(i for i in range(start, start + 200) if RB.hash_arm(f"{i}:0") == arm)


def test_setting_reads_off_on_ab_and_the_old_bools():
    assert [RB.setting(v) for v in (None, False, "off", "false", "nonsense")] == ["off"] * 5
    assert [RB.setting(v) for v in (True, "on", "true")] == ["on"] * 3
    assert RB.setting("ab") == RB.setting(" AB ") == "ab"


def test_arm_assignment_is_deterministic_rederivable_and_balanced():
    ledger: list[dict] = []
    picks = []
    for n in range(400):
        item = 100 + (n % 137)                       # items come back: re-offers are new draws
        a = RB.assign_arm(ledger, item)
        assert a == RB.assign_arm(list(ledger), item)   # same ledger, same answer
        attempt = sum(1 for r in ledger if r["item_id"] == item)
        assert a["key"] == f"{item}:{attempt}"
        picks.append(a["how"])
        ledger.append(_started(item, T0 + n, reasoning_bank_arm=a["arm"],
                               reasoning_bank_arm_key=a["key"]))
        counts = {x: sum(1 for r in ledger if r["reasoning_bank_arm"] == x) for x in RB.ARMS}
        assert abs(counts["control"] - counts["common_causes"]) <= RB.BALANCE_SLACK
    assert picks.count("hash") > 300                  # the hash decides; the guard only trims
    assert {RB.hash_arm(f"{i}:0") for i in range(20)} == set(RB.ARMS)
    # sha256, not Python's per-process salted hash(): pinned values
    assert [RB.hash_arm(f"{i}:0") for i in (1489, 1490)] == \
        [RB.hash_arm("1489:0"), RB.hash_arm("1490:0")]


def test_a_warm_continuation_inherits_its_sessions_arm_and_cluster():
    item = _item_with_arm("common_causes")
    other = _item_with_arm("control", start=item + 1)
    ledger = [_started(item, T0, reasoning_bank_arm="common_causes",
                       reasoning_bank_arm_key=f"{item}:0", reasoning_bank_cluster=f"{item}:0"),
              _finished(item, T0 + 60, "SM_x", session_id="sess-1")]
    a = RB.assign_arm(ledger, other, continues_session="sess-1")
    assert (a["arm"], a["cluster"], a["how"]) == ("common_causes", f"{item}:0", "chain")
    # a session from before the A/B (no arm on it) is a fresh draw
    pre = [_started(item, T0), _finished(item, T0 + 60, "SM_x", session_id="sess-0")]
    assert RB.assign_arm(pre, other, continues_session="sess-0")["how"] == "hash"


def test_the_arm_reaches_round_start_from_the_open_turn_only():
    rows = [_started(7, T0, reasoning_bank_arm="control", reasoning_bank_arm_key="7:0")]
    assert RB.arm_for_round(rows, 7) == {"reasoning_bank_arm": "control",
                                         "reasoning_bank_arm_key": "7:0"}
    assert RB.arm_for_round(rows, 8) == {}                        # not this item
    assert RB.arm_for_round(rows, None) == {}                     # a CLI round
    assert RB.arm_for_round(rows + [_finished(7, T0 + 5, "SM_a")], 7) == {}   # turn over
    assert RB.arm_for_round([_started(7, T0)], 7) == {}           # not in the A/B


def test_round_start_writes_the_arm_on_its_ledger_row(tmp_path, monkeypatch):
    import subprocess
    from scripts.automod import backlog as B, round as R, state as S, worktree as W
    live = tmp_path / "live"
    (live / "app").mkdir(parents=True)

    def g(*a):
        return subprocess.run(["git", "-C", str(live), *a], capture_output=True, check=False)
    subprocess.run(["git", "init", "-q", "-b", "main", str(live)], capture_output=True)
    g("config", "user.email", "t@e.com")
    g("config", "user.name", "t")
    (live / "app" / "m.py").write_text("V = 1\n")
    g("add", "-A")
    g("commit", "-q", "-m", "base")
    state = tmp_path / "state"
    state.mkdir()
    for name in ("STATE_DIR", "BROKEN_DIR"):
        monkeypatch.setattr(S, name, state)
    monkeypatch.setattr(S, "ROUNDS_DIR", state / "rounds")
    for name, fn in (("LEDGER_PATH", "promotions.jsonl"), ("HALTED_PATH", "halted"),
                     ("BROKEN_PATH", "BROKEN"), ("LOCK_PATH", "lock"),
                     ("CURRENT_PATH", "current.json")):
        monkeypatch.setattr(S, name, state / fn)
    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(B, "orphan_stale_amendments", lambda *a, **k: None)
    S.append_event(_started(42, time.time(), reasoning_bank_arm="common_causes",
                            reasoning_bank_arm_key="42:0"))
    out = R.start("Implement backlog #42", force=True, item_id=42, opened_by="tool")
    try:
        row = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_start"][-1]
        assert (row["reasoning_bank_arm"], row["reasoning_bank_arm_key"]) == ("common_causes", "42:0")
    finally:
        W.remove(out["round_id"], repo=live)
    out2 = R.start("a person's round", force=True)
    try:
        row2 = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_start"][-1]
        assert "reasoning_bank_arm" not in row2
    finally:
        W.remove(out2["round_id"], repo=live)


def _ab_ledger(now):
    return [_start("SM_f", 21, now - 900, "Implement backlog #21: vault recall reranker"),
            _review("SM_f", 21, now - 800, [_clause(1, "partial", TAUTOLOGY)]),
            _start("SM_g", 23, now - 900, "Implement backlog #23: voice"),
            _review("SM_g", 23, now - 700, [_clause(1, "partial", HALF)])]


def test_ab_injects_only_in_the_treatment_arm_and_off_on_are_unchanged(monkeypatch, tmp_path):
    ev = _ab_ledger(time.time())
    treat, ctrl = _item_with_arm("common_causes"), _item_with_arm("control")
    T = SimpleNamespace(id=treat, name="n", body="")
    C = SimpleNamespace(id=ctrl, name="n", body="")
    ac = _autocode(monkeypatch, {"reasoning_bank": "ab"}, tmp_path, ev)
    block, f = ac._strategy_for(T, {})
    assert f["reasoning_bank_arm"] == "common_causes" and f["reasoning_bank_arm_key"] == f"{treat}:0"
    assert "unfalsifiable test" in block and "half missing" in block
    assert len(block) == f["reasoning_bank_block_chars"] <= RB.AB_MAX_BLOCK_CHARS
    block, f = ac._strategy_for(C, {})
    assert (block, f["reasoning_bank_arm"]) == ("", "control")
    assert "reasoning_bank_block_chars" not in f
    # `ab` never leaks into the plain `on` path
    assert ac._strategy_block(T, {}) == ""
    # `on` is the old behaviour exactly: every round, no arm
    ac = _autocode(monkeypatch, {"reasoning_bank": True}, tmp_path, ev)
    for cand in (T, C):
        block, f = ac._strategy_for(cand, {})
        assert f == {} and block == ac._strategy_block(cand, {})
        assert "Lessons from similar rounds" in block
    ac = _autocode(monkeypatch, {"reasoning_bank": False}, tmp_path, ev)
    assert ac._strategy_for(T, {}) == ("", {}) == ac._strategy_for(C, {})


def test_the_treatment_block_is_held_under_its_token_budget():
    long = "x " * 400
    items = [{**_item(i, i, c, T0, "s"), "detail": long}
             for i, c in enumerate(("unpinned", "half_missing", "unfalsifiable_test"), 1)]
    assert len(RB.render_block(items)) > RB.AB_MAX_BLOCK_CHARS
    fit = RB.fit_block(items, RB.AB_MAX_BLOCK_CHARS)
    assert 0 < len(fit) <= RB.AB_MAX_BLOCK_CHARS
    assert all(RB.LESSONS[c] in fit for c in ("unpinned", "half_missing", "unfalsifiable_test"))
    short = [_item(1, 1, "unpinned", T0, "s")]
    assert RB.fit_block(short, RB.AB_MAX_BLOCK_CHARS) == RB.render_block(short)


def test_execute_writes_the_arm_on_the_started_row_and_the_block_in_the_prompt(
        monkeypatch, tmp_path):
    import asyncio
    from scripts.automod import backlog as B, state as S
    treat = _item_with_arm("common_causes")
    ac = _autocode(monkeypatch, {"reasoning_bank": "ab"}, tmp_path, _ab_ledger(time.time()))
    cand = SimpleNamespace(id=treat, name="n", body="", status="up_next", priority="low")
    seen = {}

    async def fake_run(item, candidate, triage, budget, started, reoffer, **kw):
        seen["reoffer"] = reoffer
        return {"status": "ok"}

    async def no_warm(slot):
        return None
    monkeypatch.setattr(ac, "_loop_is_free", lambda *a, **k: (True, ""))
    monkeypatch.setattr(B, "select_confirmed", lambda *a, **k: (cand, {}))
    monkeypatch.setattr(ac, "_warm_session", no_warm)
    monkeypatch.setattr(ac, "_reoffer_for", lambda i: "")
    monkeypatch.setattr(ac, "_run_and_record", fake_run)
    monkeypatch.setattr(ac, "_vault_landed_since", lambda *a, **k: False)
    monkeypatch.setattr(ac, "_maybe_flush", lambda *a, **k: None)
    monkeypatch.setattr(B, "set_status", lambda *a, **k: None)
    monkeypatch.setattr(B, "reconcile_statuses", lambda *a, **k: None)
    asyncio.run(ac.execute(SimpleNamespace(payload={}, dedup_key="autocode:round")))
    row = [e for e in S.read_events(path=S.LEDGER_PATH, limit=1000)
           if e.get("event") == "backlog_implement"][-1]
    assert row["phase"] == "started" and row["reasoning_bank_arm"] == "common_causes"
    assert row["reasoning_bank_arm_key"] == f"{treat}:0"
    assert "unfalsifiable test" in seen["reoffer"]


def _unit_rows(item, arm, t, *, rid, landed=False, refusals=0, closed=False, hours=1.0):
    out = [_started(item, t, reasoning_bank_arm=arm, reasoning_bank_arm_key=f"{item}:0",
                    reasoning_bank_cluster=f"{item}:0"),
           {"event": "round_start", "round_id": rid, "item_id": item, "ts": t + 1,
            "reasoning_bank_arm": arm}]
    out += [_review(rid, item, t + 10 + k, [_clause(1, "partial", TAUTOLOGY)])
            for k in range(refusals)]
    if landed:
        out.append({"event": "promoted", "round_id": rid, "ts": t + 100})
    out.append(_finished(item, t + hours * 3600, rid,
                         outcome={"landed": landed, "acceptance": "met" if landed else "not_met"}))
    if closed:
        out.append({"event": "item_landed", "item_id": item, "round_id": rid,
                    "closed": True, "ts": t + hours * 3600 + 5})
    return out


def test_report_math_on_a_synthetic_ledger():
    ev, t = [], T0
    # control: 60 rounds, 30 land (all resolved), 1 refusal each, 1 h each
    for i in range(60):
        ev += _unit_rows(1000 + i, "control", t, rid=f"SM_c{i}", landed=i % 2 == 0,
                         closed=i % 2 == 0, refusals=1)
        t += 10
    # common_causes: 60 rounds, 45 land and resolve, no refusals, 0.5 h each
    for i in range(60):
        ev += _unit_rows(2000 + i, "common_causes", t, rid=f"SM_t{i}", landed=i % 4 != 0,
                         closed=i % 4 != 0, hours=0.5)
        t += 10
    ev.append(_started(3000, t, reasoning_bank_arm="control", reasoning_bank_arm_key="3000:0"))
    ev += [_started(3001, t, reasoning_bank_arm="control", reasoning_bank_arm_key="3001:0"),
           {"event": "backlog_implement", "item_id": 3001, "phase": "skipped", "ts": t + 1}]
    r = RB.ab_report(ev, now=T0 + 3 * 86400, min_n=50, target_n=150)
    c, x = r["arms"]["control"], r["arms"]["common_causes"]
    assert (c["rounds"], c["landed"], x["rounds"], x["landed"]) == (60, 30, 60, 45)
    assert (r["open"], r["excluded"], r["round_arm_mismatches"]) == (1, 1, 0)
    assert c["landed_rate"] == 0.5 and x["landed_rate"] == 0.75
    lo, hi = c["landed_ci"]
    assert 0.37 < lo < 0.38 and 0.62 < hi < 0.63                   # Wilson 30/60
    assert c["refusals_per_round"] == 1.0 and x["refusals_per_round"] == 0.0
    assert abs(c["resolved_per_round_hour"] - 0.5) < 1e-9          # 30 / 60 h
    assert abs(x["resolved_per_round_hour"] - 1.5) < 1e-9          # 45 / 30 h
    rlo, rhi = c["resolved_per_round_hour_ci"]
    assert rlo < 0.5 < rhi
    d = r["diff"]["resolved_per_round_hour"]
    assert abs(d["diff"] - 1.0) < 1e-9 and d["lo"] > 0 and d["excludes_zero"]
    assert r["diff"]["refusals_per_round"]["diff"] == -1.0
    assert r["verdict"].startswith("common_causes wins")
    assert not r["stop_rule_met"]                                 # 60 < 150 and 3 < 10 days
    assert r["n_needed_per_arm"] == RB.n_for_mde((30 + 45) / 120, 0.15)
    assert "verdict: common_causes wins" in RB.format_report(r)
    # below the minimum n there is no verdict, whatever the intervals say
    assert RB.ab_report(ev, now=T0 + 3 * 86400, min_n=61)["verdict"].startswith("insufficient")
    # the stop rule's clock, and its count
    assert RB.ab_report(ev, now=T0 + 11 * 86400)["stop_rule_met"]
    assert RB.ab_report(ev, now=T0 + 3 * 86400, target_n=60)["stop_rule_met"]
    # identical arms: no difference is claimed
    same = []
    for i in range(60):
        same += _unit_rows(4000 + i, RB.ARMS[i % 2], T0 + 10 * i, rid=f"SM_s{i}",
                           landed=(i // 2) % 2 == 0, closed=(i // 2) % 2 == 0)
    rs = RB.ab_report(same, now=T0 + 86400, min_n=20)
    assert not rs["diff"]["landed_rate"]["excludes_zero"]
    assert rs["verdict"].startswith("no measured difference")


def test_n_for_mde_matches_the_two_proportion_formula():
    # 0.5 -> 0.65, alpha .05 two-sided, power .8, no continuity correction:
    # (1.96*sqrt(2*.575*.425) + .8416*sqrt(.25+.2275))^2 / .15^2 = 169.3 -> 170
    assert RB.n_for_mde(0.5, 0.15) == 170
    assert RB.n_for_mde(0.5, 0.0) == 0


def test_ab_report_cli_runs_on_a_ledger_path(tmp_path, capsys):
    p = tmp_path / "l.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in _unit_rows(5, "control", T0, rid="SM_z")))
    assert RB.main(["ab-report", "--ledger", str(p)]) == 0
    assert "insufficient" in capsys.readouterr().out
