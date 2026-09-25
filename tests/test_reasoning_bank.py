"""#1489: the autocode ReasoningBank — distil, prune, retrieve, inject (off)."""
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


def test_the_shipped_config_leaves_it_off():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["workers"]["sources"]["autocode"].get("reasoning_bank") is False


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
