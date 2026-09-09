"""A mixed round's two halves are joined, so a rollback can undo both.

Why this file exists
--------------------
A `mixed` backlog item lands its vault half through `autoimplement_vault_land` and
its code half through the promoter. Until now nothing joined them: the vault
sha went into a `vault_land` ledger event keyed on the backlog `item_id`, the
code commit went into `current.json` keyed on the round, and a rollback saw
only the second.

Backlog #377 is the case that shows the cost. Reverting its code commit alone
restores `prompt_builder.ANTICOMPLIANCE_DIRECTIVE` while `SOUL.md` keeps the
condensed `Anti-Compliance Directive` the round wrote — recreating the exact
doubled-frame state (#465) the round existed to remove, silently, at the
moment the system is least supervised.

The join is on time, because the ledger's `round_start` is the only anchor the
two halves share. The window is closed at the next round rather than left
running to now: only one round holds the lock at a time, so an unbounded
window makes an aborted round claim its successor's vault commit.
"""
from __future__ import annotations

import json

import pytest

from scripts.autoimplement import promote as P
from scripts.autoimplement import state as S


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A scratch ledger, so nothing here reads or writes the real one."""
    path = tmp_path / "promotions.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", path)
    clock = {"t": 1000.0}

    def add(event: dict, *, dt: float = 1.0):
        clock["t"] += dt
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": clock["t"], **event}) + "\n")

    return add


def test_a_round_claims_the_vault_commit_it_landed(ledger):
    ledger({"event": "round_start", "round_id": "SM_A"})
    ledger({"event": "vault_land", "ok": True, "commit": "aaa111", "item_id": 377})
    ledger({"event": "promoted", "round_id": "SM_A"})
    assert P.vault_commits_for("SM_A") == ["aaa111"]


def test_a_round_that_landed_no_vault_change_claims_nothing(ledger):
    ledger({"event": "round_start", "round_id": "SM_A"})
    ledger({"event": "promoted", "round_id": "SM_A"})
    assert P.vault_commits_for("SM_A") == []


def test_an_aborted_round_does_not_claim_its_successors_commit(ledger):
    """Unbounded, this is the failure: the abort has no end event of its own."""
    ledger({"event": "round_start", "round_id": "SM_A"})
    ledger({"event": "round_aborted", "round_id": "SM_A"})
    ledger({"event": "round_start", "round_id": "SM_B"})
    ledger({"event": "vault_land", "ok": True, "commit": "bbb222", "item_id": 9})
    assert P.vault_commits_for("SM_A") == []
    assert P.vault_commits_for("SM_B") == ["bbb222"]


def test_a_failed_vault_land_is_not_claimed(ledger):
    """`ok: False` means the paths were reverted; there is no sha to undo."""
    ledger({"event": "round_start", "round_id": "SM_A"})
    ledger({"event": "vault_land", "ok": False, "errors": ["nope"], "item_id": 1})
    assert P.vault_commits_for("SM_A") == []


def test_several_vault_commits_come_back_in_order(ledger):
    ledger({"event": "round_start", "round_id": "SM_A"})
    ledger({"event": "vault_land", "ok": True, "commit": "aaa111"})
    ledger({"event": "vault_land", "ok": True, "commit": "bbb222"})
    assert P.vault_commits_for("SM_A") == ["aaa111", "bbb222"]


def test_an_unknown_round_claims_nothing(ledger):
    ledger({"event": "round_start", "round_id": "SM_A"})
    ledger({"event": "vault_land", "ok": True, "commit": "aaa111"})
    assert P.vault_commits_for("SM_NOPE") == []


def test_a_vault_land_before_the_round_opened_is_not_claimed(ledger):
    ledger({"event": "vault_land", "ok": True, "commit": "old000"})
    ledger({"event": "round_start", "round_id": "SM_A"})
    assert P.vault_commits_for("SM_A") == []


def test_the_promotion_record_carries_the_field():
    """Static: the promoter writes it, or the rollback has nothing to read."""
    from pathlib import Path

    src = (Path(P.__file__)).read_text(encoding="utf-8")
    assert '"vault_commits": vault_commits_for(round_id)' in src
    assert '"vault_commits": current.get("vault_commits")' in src, (
        "the promoted ledger event must carry it too — `clear_current` deletes "
        "current.json when the promotion settles"
    )


# ── reverting them ──────────────────────────────────────────────────────────

def test_revert_many_goes_newest_first(monkeypatch):
    """Oldest-first conflicts by construction when both touch the same file."""
    from scripts.autoimplement import vault_round as VR

    order: list[str] = []
    monkeypatch.setattr(VR, "revert", lambda sha, reason="": order.append(sha))
    out = VR.revert_many(["aaa", "bbb", "ccc"])
    assert order == ["ccc", "bbb", "aaa"]
    assert out["ok"] and out["reverted"] == ["ccc", "bbb", "aaa"]


def test_revert_many_reports_partial_progress(monkeypatch):
    """A rollback that undid two of three has still changed the tree."""
    from scripts.autoimplement import vault_round as VR

    def flaky(sha, reason=""):
        if sha == "bbb":
            raise VR.VaultRoundError("conflict")

    monkeypatch.setattr(VR, "revert", flaky)
    out = VR.revert_many(["aaa", "bbb", "ccc"])
    assert out["ok"] is False
    assert out["reverted"] == ["ccc"] and out["failed"] == "bbb"


def test_revert_many_on_an_empty_list_is_a_no_op(monkeypatch):
    from scripts.autoimplement import vault_round as VR

    monkeypatch.setattr(VR, "revert", lambda sha, reason="": pytest.fail("called"))
    assert VR.revert_many([]) == {"ok": True, "reverted": []}
