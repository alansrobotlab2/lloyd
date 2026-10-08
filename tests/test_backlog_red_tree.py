"""The red tree heals itself: the gate files it, the next round takes it, the gate closes it.

Since 2026-09-24 the `tests` rung passes on failures that reproduce at the
round's base. Passing alone would leave `main` red indefinitely — nothing in
the loop fixed a red tree, and episodes ran 20-59 h — so the gate files the
breakage as one `high`, pre-confirmed `red-tree` item and closes it on the
first full green run at a descendant base.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from scripts.automod import backlog as B
from scripts.automod import state as S
from tests.test_backlog_unattended import _confirm, isolated, write_item  # noqa: F401

RED = "tests/test_uptake.py::test_red"
RED2 = "tests/test_uptake.py::test_red_two"
OTHER = "tests/test_trajectory_extraction.py::test_also_red"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    """c1 -> c2 on main, c3 on a side branch off c1: an ancestor, a descendant
    and an unrelated base."""
    r = tmp_path / "live"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    shas = {}
    for name in ("c1", "c2"):
        (r / f"{name}.txt").write_text(name)
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", name)
        shas[name] = git(r, "rev-parse", "HEAD")
    git(r, "checkout", "-q", "-b", "side", shas["c1"])
    (r / "c3.txt").write_text("c3")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "c3")
    shas["c3"] = git(r, "rev-parse", "HEAD")
    return r, shas


def _events(name):
    if not S.LEDGER_PATH.exists():
        return []
    return [json.loads(line) for line in S.LEDGER_PATH.read_text().splitlines()
            if line.strip() and json.loads(line).get("event") == name]


def _fm(item):
    return B._split_frontmatter(item.path.read_text(encoding="utf-8"))[0]


def test_new_item_writes_the_boards_front_matter_and_the_next_id(isolated):
    write_item(isolated, 41)
    item = B.new_item("A new one", "body text", priority="high", tags=("red-tree",))
    assert item.id == 42 and item.name == "A new one" and item.priority == "high"
    fm = _fm(item)
    assert fm["type"] == "backlog" and fm["segment"] == "backlog"
    assert fm["status"] == "draft" and fm["board"] == "lloyd" and fm["position"] == 42000
    assert fm["tags"] == ["red-tree"] and "created" in fm
    assert "body text" in item.body


def test_a_red_tree_is_filed_once_high_and_confirmed(isolated, repo):
    r, c = repo
    res = B.file_red_tree_item(c["c1"], [RED, OTHER], "SM_A", [], live_root=r)
    assert res["action"] == "created"
    item = B.item_by_id(res["item_id"])
    assert item.priority == "high" and B.RED_TREE_TAG in item.tags
    assert item.status == "up_next", "pre-confirmed: straight into the implement pool"
    assert "main is red" in item.name
    fm = _fm(item)
    assert fm["red_tree_base"] == c["c1"] and fm["red_tree_nodes"] == sorted([RED, OTHER])
    clauses = fm["acceptance_clauses"]
    assert any("tests/test_uptake.py" in cl for cl in clauses)
    assert any("tests/test_trajectory_extraction.py" in cl for cl in clauses)
    assert "no test is skipped" in clauses[-1]
    # Not a spawn tag: expiry and the write-time merge leave it alone.
    assert not any(t.startswith("spawned-by-") for t in item.tags)
    triage = [e for e in _events("backlog_triage") if e["item_id"] == item.id]
    assert triage and triage[-1]["auto"] is True and triage[-1]["red_tree"] is True
    assert triage[-1]["verdict"] == "confirmed"
    filed = _events("red_tree_filed")
    assert filed == [{**filed[0], "event": "red_tree_filed", "item_id": item.id, "base": c["c1"],
                      "node_ids": sorted([RED, OTHER]), "round_id": "SM_A", "action": "created"}]
    # Same base, same (or fewer) nodes: nothing written.
    assert B.file_red_tree_item(c["c1"], [RED], "SM_B", [], live_root=r) is None
    assert len(B.open_red_tree_items()) == 1 and len(_events("red_tree_filed")) == 1


def test_the_next_round_takes_it_first(isolated, repo):
    r, c = repo
    write_item(isolated, 10, days_old=300)
    _confirm(10)
    write_item(isolated, 11, days_old=200, priority="high")
    _confirm(11)
    res = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)
    item, ev = B.select_confirmed(S.LEDGER_PATH)
    assert item.id == res["item_id"], "a red tree is the newest high: it goes first"
    assert ev.get("red_tree") is True


def test_a_superset_on_the_same_base_appends_clauses(isolated, repo):
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)["item_id"]
    res = B.file_red_tree_item(c["c1"], [RED, RED2, OTHER], "SM_B", [], live_root=r)
    assert res == {"item_id": iid, "action": "merged"}
    fm = _fm(B.item_by_id(iid))
    assert fm["red_tree_nodes"] == sorted([RED, RED2, OTHER])
    assert any("tests/test_trajectory_extraction.py" in cl for cl in fm["acceptance_clauses"])
    assert len(fm["acceptance_clauses"]) <= B.MAX_CLAUSES
    assert _events("red_tree_filed")[-1]["action"] == "merged"
    assert len(B.open_red_tree_items()) == 1


def test_a_newer_base_replaces_and_an_older_one_is_ignored(isolated, repo):
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED, OTHER], "SM_A", [], live_root=r)["item_id"]
    res = B.file_red_tree_item(c["c2"], [OTHER], "SM_B", [], live_root=r)
    assert res == {"item_id": iid, "action": "replaced"}
    fm = _fm(B.item_by_id(iid))
    assert fm["red_tree_base"] == c["c2"] and fm["red_tree_nodes"] == [OTHER]
    # A round still cut from c1 sees RED too; the item already describes c2.
    assert B.file_red_tree_item(c["c1"], [RED], "SM_C", [], live_root=r) is None
    assert _fm(B.item_by_id(iid))["red_tree_nodes"] == [OTHER]


def test_the_rounds_own_files_are_never_filed(isolated, repo):
    r, c = repo
    assert B.file_red_tree_item(c["c1"], [RED], "SM_A", ["tests/test_uptake.py"],
                                live_root=r) is None
    res = B.file_red_tree_item(c["c1"], [RED, OTHER], "SM_A", ["tests/test_uptake.py"],
                               live_root=r)
    assert _fm(B.item_by_id(res["item_id"]))["red_tree_nodes"] == [OTHER]


def test_healing_closes_on_a_descendant_base_only(isolated, repo):
    r, c = repo
    iid = B.file_red_tree_item(c["c2"], [RED], "SM_A", [], live_root=r)["item_id"]
    assert B.close_healed_red_tree(c["c1"], "SM_OLD", r) == [], "an ancestor proves nothing"
    assert B.close_healed_red_tree(c["c3"], "SM_SIDE", r) == [], "an unrelated base proves nothing"
    assert B.close_healed_red_tree(c["c2"], "SM_T", r, touched=["tests/test_uptake.py"]) == [], \
        "the round fixing the file closes it by landing, not by its own green run"
    assert B.item_by_id(iid).status == "up_next"
    assert B.close_healed_red_tree(c["c2"], "SM_G", r) == [iid]
    item = B.item_by_id(iid)
    assert item.status == "done" and _fm(item)["autotriage_retired"] == "already_done"
    assert _events("red_tree_closed") == [{**_events("red_tree_closed")[0], "event": "red_tree_closed",
                                           "item_id": iid, "base": c["c2"], "round_id": "SM_G"}]
    assert B.open_red_tree_items() == [], "a done red-tree item is not open"


def test_a_heal_is_not_undone_by_an_older_view_but_a_wrong_heal_is(isolated, repo):
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)["item_id"]
    assert B.close_healed_red_tree(c["c2"], "SM_G", r) == [iid]
    # A round still on c1 sees the old failure: within the cooldown, not refiled.
    assert B.file_red_tree_item(c["c1"], [RED], "SM_OLDER", [], live_root=r) is None
    # The same failure reproducing AT the heal's base says the heal was wrong
    # (the green run's diff fixed the test and never landed): filed again.
    res = B.file_red_tree_item(c["c2"], [RED], "SM_AGAIN", [], live_root=r)
    assert res["action"] == "created" and res["item_id"] != iid


def test_the_cooldown_ends(isolated, repo):
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)["item_id"]
    B.close_healed_red_tree(c["c2"], "SM_G", r)
    path = B.item_by_id(iid).path
    fm, body = B._split_frontmatter(path.read_text(encoding="utf-8"))
    fm["completed"] = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")
    path.write_text(f"---\n{yaml.dump(fm)}---\n{body}", encoding="utf-8")
    res = B.file_red_tree_item(c["c1"], [RED], "SM_LATER", [], live_root=r)
    assert res["action"] == "created"


def _working(iid):
    """What autocode does the moment a round's turn starts on the item."""
    B.update_frontmatter(B.item_by_id(iid).path, {"status": "in_progress"})


def test_an_item_a_round_is_working_keeps_its_contract(isolated, repo):
    """#1454, 2026-09-24: a sibling round's gate found three more failures at the
    same base 20 minutes into #1454's own turn and appended a clause to it — the
    review would have graded that round on work it was never given."""
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)["item_id"]
    _working(iid)
    before = _fm(B.item_by_id(iid))["acceptance_clauses"]
    res = B.file_red_tree_item(c["c1"], [RED, OTHER], "SM_B", [], live_root=r)
    assert res["action"] == "created" and res["item_id"] != iid
    assert _fm(B.item_by_id(iid))["acceptance_clauses"] == before
    assert _fm(B.item_by_id(iid))["red_tree_nodes"] == [RED]
    assert _fm(B.item_by_id(res["item_id"]))["red_tree_nodes"] == [OTHER], \
        "only what the worked item does not already cover moves"


def test_what_a_worked_item_covers_is_a_no_op(isolated, repo):
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED, RED2], "SM_A", [], live_root=r)["item_id"]
    _working(iid)
    assert B.file_red_tree_item(c["c1"], [RED], "SM_B", [], live_root=r) is None
    assert B.file_red_tree_item(c["c2"], [RED2], "SM_C", [], live_root=r) is None, \
        "not replaced on a newer base either while its round runs"
    assert len(B.open_red_tree_items()) == 1


def test_the_overflow_merges_into_a_second_open_item_that_is_not_worked(isolated, repo):
    r, c = repo
    first = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)["item_id"]
    _working(first)
    second = B.file_red_tree_item(c["c1"], [OTHER], "SM_B", [], live_root=r)["item_id"]
    res = B.file_red_tree_item(c["c1"], [RED, OTHER, RED2], "SM_C", [], live_root=r)
    assert res == {"item_id": second, "action": "merged"}
    assert _fm(B.item_by_id(second))["red_tree_nodes"] == sorted([OTHER, RED2])
    assert _fm(B.item_by_id(first))["red_tree_nodes"] == [RED]


# ── the implement loop's pickup probe ───────────────────────────────────────
# `close_healed_red_tree` above is reachable only from a FULL green `tests`
# rung, so an item whose red was healed by a vault commit — or by anything at
# all that the next round happens not to re-run — stayed open and was picked up:
# #2384 spent 22 turns and opened no round, and #1846 nine days earlier spent 14
# and was closed `unnecessary`. These tests drive the decision with the pytest
# subprocess injected, which is the seam the decision actually lives at; the
# probe itself, aimed at a real pytest in a throwaway worktree, is
# `tests/test_automod_gate.py::test_a_pickup_probe_reruns_the_nodes_it_is_given`.

def _probe_result(head, unresolved=(), *, conclusive=True, note=None):
    """A stand-in for `gate.red_tree_state_at_head`, closing over one answer."""
    def probe(nodes):
        return {"head": head, "probed": list(nodes), "unresolved": list(unresolved),
                "conclusive": conclusive,
                "note": note if note is not None
                        else f"probed {len(nodes)} node(s) at base {head[:8]}: "
                             f"0 already failing"}
    return probe


def _pickup(isolated, item_id, r, probe):
    """The default ledger is `S.LEDGER_PATH`, which `isolated` points at tmp."""
    return B.close_healed_red_tree_at_pickup(B.item_by_id(item_id), probe, live_root=r)


def test_a_pickup_probe_that_finds_every_node_green_closes_the_item(isolated, repo):
    """Clause 2: the close goes through `already_done` and leaves a
    `red_tree_closed` row naming the item, the probed head and who closed it."""
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED, RED2], "SM_A", [], live_root=r)["item_id"]
    res = _pickup(isolated, iid, r, _probe_result(c["c2"]))
    assert res["action"] == "closed" and res["head"] == c["c2"], res

    item = B.item_by_id(iid)
    assert item.status == "done", "closed through the verdict path, not parked"
    assert _fm(item)["autotriage_retired"] == "already_done"
    assert _fm(item)["red_tree_healed_base"] == c["c2"], \
        "the heal is recorded on the item, as the full-run closer records its base"

    row = [e for e in _events("red_tree_closed") if e["item_id"] == iid][-1]
    assert row["base"] == c["c2"] and row["head"] == c["c2"], "the probed head"
    assert row["closed_by"] == B.PICKUP_CLOSER, "who closed it"
    assert row["nodes"] == sorted([RED, RED2])
    triage = [e for e in _events("backlog_triage")
              if e["item_id"] == iid and e["verdict"] == "already_done"]
    assert triage and triage[-1]["closed"] is True and triage[-1]["auto"] is True
    assert _events("backlog_implement") == [], \
        "the close is the alternative to an attempt, not a note beside one"


def test_a_pickup_probe_that_finds_a_node_still_red_credits_nothing(isolated, repo):
    """Clause 3, the negative control: one listed node still failing at the
    probed head (of the two this item lists) means the item is picked up exactly
    as it is today — the decision half here, and `tests/test_autocode_continuation
    .py::test_a_red_tree_whose_nodes_still_fail_is_started_exactly_as_today` pins
    the `started` row and the turn."""
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED, RED2], "SM_A", [], live_root=r)["item_id"]
    res = _pickup(isolated, iid, r, _probe_result(c["c2"], unresolved=[RED2]))
    assert res["action"] == "attempt", res
    assert res["reason"] == f"still red at {c['c2'][:12]}: {RED2}", \
        "one red node is enough to pick the item up, and the reason names only it"

    item = B.item_by_id(iid)
    assert item.status == "up_next" and item.priority == "high", "still queued for the loop"
    assert "red_tree_healed_base" not in _fm(item)
    assert _events("red_tree_closed") == [], \
        "the decision function may not spend a verdict it did not earn"


@pytest.mark.parametrize("cause,note", [
    ("worktree", "baseline worktree failed: fatal: bad object f321907091baa"),
    ("timeout", "baseline probe timed out after 300s"),
    ("nothing collected",
     "baseline probe INCONCLUSIVE at base f3219070 — pytest produced no summary "
     "(rc=4): ERROR: not found: tests/test_uptake.py::test_a_failing_test"),
])
def test_an_inconclusive_pickup_probe_closes_nothing_even_though_it_found_no_failures(
        isolated, repo, cause, note):
    """Clause 4. `_failures_at_base` fails CLOSED, so an unbuildable worktree, a
    timeout and a pytest that collected nothing all come back with an EMPTY set
    of failures — which at the gate means "not pre-existing" and here would mean
    "healed, close a live red tree" if the close keyed on the set. The close keys
    on `conclusive` instead, so all three leave the item exactly as they found
    it, and the `unresolved` list below is empty in every case: that is the trap.
    """
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)["item_id"]
    res = _pickup(isolated, iid, r, _probe_result(c["c2"], (), conclusive=False, note=note))
    assert res["action"] == "attempt", f"{cause}: an empty answer is not a heal"
    assert cause in res["reason"] or note[:40] in res["reason"]
    assert B.item_by_id(iid).status == "up_next"
    assert _events("red_tree_closed") == [], "nothing was credited to a run that did not happen"


def test_a_green_pickup_probe_at_a_tree_the_item_does_not_descend_from_closes_nothing(
        isolated, repo):
    """Clause 5: the ancestry rule the full-run closer obeys holds at pickup too.
    `c1` is the item's own base's ancestor and `c3` a sibling branch; a green run
    at either says nothing about the tree the item was filed against."""
    r, c = repo
    iid = B.file_red_tree_item(c["c2"], [RED], "SM_A", [], live_root=r)["item_id"]
    assert _pickup(isolated, iid, r, _probe_result(c["c1"]))["action"] == "attempt", \
        "an ancestor of the item's base proves nothing"
    assert _pickup(isolated, iid, r, _probe_result(c["c3"]))["action"] == "attempt", \
        "an unrelated branch proves nothing either"
    assert _events("red_tree_closed") == [] and B.item_by_id(iid).status == "up_next"
    assert _pickup(isolated, iid, r, _probe_result(c["c2"]))["action"] == "closed", \
        "the base the item was filed at is itself, and a green run there is the answer"


def test_an_item_that_is_not_a_red_tree_item_is_picked_up_without_any_probe(isolated, repo):
    """The probe is a red-tree-item thing: an ordinary item costs the loop nothing
    and its nodes are never run."""
    r, c = repo
    write_item(isolated, 41)
    asked: list = []
    res = B.close_healed_red_tree_at_pickup(
        B.item_by_id(41), lambda nodes: asked.append(list(nodes)) or {}, live_root=r)
    assert res["action"] == "attempt" and asked == []


def test_a_closed_red_tree_item_is_not_probed_or_resurrected(isolated, repo):
    """A green probe never re-writes a `done` item's close, and never starts a
    turn for it either — there is no attempt left to spend."""
    r, c = repo
    iid = B.file_red_tree_item(c["c1"], [RED], "SM_A", [], live_root=r)["item_id"]
    B.close_healed_red_tree_at_pickup(B.item_by_id(iid), _probe_result(c["c2"]), live_root=r)
    before = _events("red_tree_closed")
    res = B.close_healed_red_tree_at_pickup(B.item_by_id(iid), _probe_result(c["c3"]),
                                            live_root=r)
    assert res["action"] == "closed" and "not open" in res["reason"]
    assert _events("red_tree_closed") == before, "one close, one row"
