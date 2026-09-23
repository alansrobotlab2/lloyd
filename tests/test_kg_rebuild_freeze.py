"""kg_rebuild.py — the freeze has to be able to end by itself.

The 2026-09-03 rebuild paused #24, #48 and #74 and its gate never passed, so
`swap` never ran. Un-pausing lived only in a line `swap` prints, hardcoded to
those three ids, on the one outcome that did not happen. #24 and #74 stayed
paused for four days and neither task file said why. These pin the way out.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import kg_rebuild  # noqa: E402


def _task_file(tmp_path: Path, task_id: int, status: str) -> Path:
    p = tmp_path / f"{task_id}-task.md"
    p.write_text(
        f"---\nid: {task_id}\nname: Task {task_id}\nstatus: {status}\n"
        f"segment: autonomy\n---\n\n## Activity Log\n", encoding="utf-8")
    return p


def test_unfreeze_restores_each_task_to_its_recorded_status(tmp_path):
    a = _task_file(tmp_path, 24, "paused")
    b = _task_file(tmp_path, 74, "paused")
    state = {"paused": [
        {"task": 24, "path": str(a), "was": "up_next"},
        {"task": 74, "path": str(b), "was": "in_progress"},
    ]}

    restored = kg_rebuild._unfreeze(state)

    assert {r["task"] for r in restored} == {24, 74}
    assert "\nstatus: up_next\n" in a.read_text()
    assert "\nstatus: in_progress\n" in b.read_text()


def test_unfreeze_leaves_a_task_freeze_found_already_paused(tmp_path):
    """`freeze` records those so it can leave them paused, not restore them."""
    p = _task_file(tmp_path, 48, "paused")
    state = {"paused": [{"task": 48, "path": str(p), "was": "paused"}]}

    assert kg_rebuild._unfreeze(state) == []
    assert "\nstatus: paused\n" in p.read_text()


def test_unfreeze_is_idempotent_and_survives_a_missing_task_file(tmp_path):
    p = _task_file(tmp_path, 24, "paused")
    gone = tmp_path / "deleted.md"
    state = {"paused": [{"task": 24, "path": str(p), "was": "up_next"},
                        {"task": 99, "path": str(gone), "was": "up_next"}]}

    assert len(kg_rebuild._unfreeze(state)) == 1
    # A second run finds nothing left to move rather than rewriting the file.
    assert kg_rebuild._unfreeze(state) == []


def test_unfreeze_only_rewrites_the_status_line(tmp_path):
    p = _task_file(tmp_path, 24, "paused")
    p.write_text(p.read_text() + "- a log line mentioning status: paused\n")
    state = {"paused": [{"task": 24, "path": str(p), "was": "up_next"}]}

    kg_rebuild._unfreeze(state)
    text = p.read_text()

    assert "\nstatus: up_next\n" in text
    assert "- a log line mentioning status: paused\n" in text


def test_swap_and_rollback_both_call_unfreeze():
    """Neither may go back to printing a reminder for a human to act on."""
    src = (ROOT / "scripts" / "memory" / "kg_rebuild.py").read_text()
    swap = src.split("def cmd_swap")[1].split("def cmd_rollback")[0]
    rollback = src.split("def cmd_rollback")[1].split("def _unfreeze")[0]
    assert "_unfreeze(state)" in swap
    assert "_unfreeze(state)" in rollback
    # The docstrings quote the old line, so match the statement, not the words.
    assert 'print("Then un-pause' not in src, "the reminder was hardcoded to one run's ids"


class _FakeStore:
    """Records the query `_carryover_present` builds, and answers it."""

    def __init__(self, rows: list[tuple[str, str]]):
        # (entity, text_hash) pairs the store "holds"
        self.rows = rows
        self.aliases = self

    def resolve(self, entity):        # no alias table in a fresh rebuild
        return entity

    def _query(self, sql, params):
        text_hash, *names = params
        wanted = set(names)
        return [1 for e, h in self.rows if h == text_hash and e.lower() in wanted]


def test_carryover_present_folds_entity_case():
    """The rebuild builds its own registry, and `export` skips case aliases.

    So the fact exported under `AutoResearch` lands under `autoresearch`. On
    the 2026-09-03 rebuild that read as 21 carried-over facts missing from an
    import that dropped none.
    """
    fact = {"entity": "AutoResearch", "fact": "a carried-over sentence"}
    held = [("autoresearch", kg_rebuild._text_hash(fact["fact"]))]

    class Store(_FakeStore):
        def _query(self, sql, params):
            assert "LOWER(entity)" in sql, "the entity side must be folded"
            text_hash, *names = params
            assert all(n == n.lower() for n in names), "the value side too"
            return [1 for e, h in self.rows if h == text_hash and e.lower() in set(names)]

    assert kg_rebuild._carryover_present(Store(held), fact) is True


def test_carryover_present_still_needs_the_same_text():
    fact = {"entity": "AutoResearch", "fact": "a carried-over sentence"}

    class Store(_FakeStore):
        def _query(self, sql, params):
            text_hash, *names = params
            return [1 for e, h in self.rows if h == text_hash and e.lower() in set(names)]

    other = [("autoresearch", kg_rebuild._text_hash("a different sentence"))]
    assert kg_rebuild._carryover_present(Store(other), fact) is False


def test_skip_eval_cannot_authorise_a_swap():
    """`swap` reads state['gate']; a run that skipped the retrieval checks
    must not be able to set it. Skipping a check may not grant what running
    it would have had to earn."""
    src = (ROOT / "scripts" / "memory" / "kg_rebuild.py").read_text()
    body = src.split("def cmd_gate")[1].split("def _facts_written_since_export")[0]
    assert "authorises_swap = results[\"pass\"] and not args.skip_eval" in body
    assert "save_state(gate=authorises_swap" in body


# ── the baseline after the 2026-09-22 deletion ────────────────────────────────
#
# The live store held 30 entities and no edges. The eval refuses an empty corpus
# by default, `freeze` recorded an empty baseline, and `gate` records "eval: no
# result" on an empty baseline, so the rebuild that would have repopulated the
# graph could never swap in.


def _freeze_in(tmp_path, monkeypatch, *, dry_run: bool):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(kg_rebuild, "RUN_ROOT", tmp_path / "backups")
    monkeypatch.setattr(kg_rebuild, "STATE_PATH", tmp_path / "rebuild-state.json")
    monkeypatch.setattr(kg_rebuild, "VAULT_KG_DB", tmp_path / "kg.sqlite")
    monkeypatch.setattr(kg_rebuild, "PAUSED_TASKS", ())
    monkeypatch.setattr(kg_rebuild, "_set_write_enabled", lambda enabled: None)
    monkeypatch.setattr(kg_rebuild, "invocation_ledger", lambda: {})

    def fake_eval(label, run_dir, **kw):
        calls.append((label, kw))
        return {"mrr_doc": 0.1, "ndcg10": 0.1}
    monkeypatch.setattr(kg_rebuild, "_run_eval", fake_eval)
    rc = kg_rebuild.cmd_freeze(SimpleNamespace(dry_run=dry_run, keep_writes=True))
    return rc, calls


def test_freeze_scores_an_empty_graph_on_purpose(tmp_path, monkeypatch):
    rc, calls = _freeze_in(tmp_path, monkeypatch, dry_run=False)
    assert rc == 0
    assert calls == [("rebuild-before", {"allow_empty_corpus": True})]
    import json
    state = json.loads((tmp_path / "rebuild-state.json").read_text())
    assert state["eval_before"]["mrr_doc"] == 0.1, "the gate needs a baseline to compare against"


def test_the_after_eval_never_gets_the_empty_corpus_allowance():
    import inspect
    src = inspect.getsource(kg_rebuild.cmd_gate)
    assert '_run_eval("rebuild-after", run_dir)' in src
    assert "allow_empty_corpus" not in src


def test_a_dry_run_writes_no_state_that_claims_a_pause(tmp_path, monkeypatch):
    rc, calls = _freeze_in(tmp_path, monkeypatch, dry_run=True)
    assert rc == 0
    assert not (tmp_path / "rebuild-state.json").exists(), (
        "unfreeze/rollback/swap read the paused list back and act on it")
    assert calls == [], "a dry run does not need a baseline"
