"""Red-tree items: a full-suite failure the gate did not cause, filed once and
closed by the loop itself when the tree heals.

Moved out of `scripts/automod/backlog.py` unchanged (2026-11-23, the first cut of
the split assessed in `architecture/backlog.md`). Every backlog name is read
through the module at call time (`B.all_items`, `B.update_frontmatter`, ...), so
a `monkeypatch.setattr(backlog, ...)` reaches this code exactly as it did when it
lived there. `backlog` re-exports every name defined here (`backlog.__getattr__`),
so `backlog.file_red_tree_item` and `from scripts.automod.backlog import ...`
keep working; the constants (`RED_TREE_TAG`, `RED_TREE_COOLDOWN_S`, ...) and
`new_item` stay in `backlog`, where the rest of the module reads them.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from app.backlog_status import OPEN_STATUSES

from scripts.automod import backlog as B


def _red_tree_file(node: str) -> str:
    return str(node).split("::", 1)[0]


def _red_tree_clauses(nodes: list[str]) -> list[str]:
    """One clause per failing file, then the no-escape clause — at most
    `MAX_CLAUSES` in all, the files past the budget folded into the last file
    clause so every node stays in the contract."""
    by_file: dict[str, list[str]] = {}
    for n in sorted(set(nodes)):
        by_file.setdefault(_red_tree_file(n), []).append(n)
    files = sorted(by_file)
    room = B.MAX_CLAUSES - 1
    head, rest = (files, []) if len(files) <= room else (files[:room - 1], files[room - 1:])

    def names(ns: list[str]) -> str:
        short = [n.split("::", 1)[1] if "::" in n else "(collection)" for n in ns]
        shown = ", ".join(short[:8])
        return shown + (f", +{len(short) - 8} more" if len(short) > 8 else "")

    out = [f'`pytest -m "{B._RED_TREE_MARK_EXPR}" {f}` exits 0; it failed at nodes: {names(by_file[f])}'
           for f in head]
    if rest:
        out.append(f'`pytest -m "{B._RED_TREE_MARK_EXPR}" {" ".join(rest)}` exits 0; '
                   f"{sum(len(by_file[f]) for f in rest)} node(s) failed across these files")
    out.append(B._RED_TREE_NO_ESCAPE)
    return [c[:B.CLAUSE_MAX_CHARS] for c in out]


def _red_tree_body(base: str, nodes: list[str], round_id: str) -> str:
    return (f"The live tree is red: these tests fail at base `{base}` with no round's "
            f"diff present. Found by the `tests` rung of round {round_id}, whose base "
            f"probe reproduced every one of them. The gate now PASSES rounds on these "
            f"failures (they are not the round's) — so nothing else will fix them.\n\n"
            + "\n".join(f"- `{n}`" for n in sorted(set(nodes))[:50])
            + (f"\n- … +{len(set(nodes)) - 50} more" if len(set(nodes)) > 50 else "")
            + "\n\nThe gate closes this item itself on the first full green run at a "
              "base that descends from the one above.")


def _red_tree_fm(item: B.Item) -> dict:
    try:
        fm, _ = B._split_frontmatter(item.path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    return fm


def open_red_tree_items(boards: tuple[str, ...] | None = B.DEFAULT_BOARDS, *,
                        backlog_dir: Path | None = None) -> list[B.Item]:
    """Open `red-tree` items, newest first. `done` is not open."""
    return sorted((i for i in B.open_items(boards, backlog_dir=backlog_dir)
                   if B.RED_TREE_TAG in i.tags), key=lambda i: -i.id)


def _is_ancestor(live_root: Path | None, older: str, newer: str) -> bool:
    """`git merge-base --is-ancestor older newer`. False when git cannot say."""
    import subprocess
    if not older or not newer:
        return False
    if older == newer:
        return True
    if live_root is None:
        from app.paths import LLOYD_HOME
        live_root = LLOYD_HOME
    try:
        r = subprocess.run(["git", "-C", str(live_root), "merge-base", "--is-ancestor",
                            older, newer], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 — no git, no claim
        return False
    return r.returncode == 0


def _closed_recently(fm: dict, now: float, window: float) -> bool:
    raw = str(fm.get("completed") or fm.get("updated") or "")
    if not raw:
        return False
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)   # this loop's closers stamp naive UTC
    return now - dt.timestamp() <= window


def file_red_tree_item(base: str, node_ids, round_id: str, changed_paths=(), *,
                       ledger: Path | None = None, live_root: Path | None = None,
                       boards: tuple[str, ...] | None = B.DEFAULT_BOARDS,
                       backlog_dir: Path | None = None,
                       now: float | None = None) -> dict | None:
    """File (or update) the one open `red-tree` item for failures that predate
    the round. Returns `{"item_id", "action"}` or None when nothing was written.

    * Nodes in the round's own diff are dropped: a round that edits a red test
      file owns it (the gate's touched-file rule), so it is not the tree's.
    * An open red-tree item on the SAME base gains the new nodes (a clause per
      new file) — action `merged`; a subset is a no-op. On a base that DESCENDS
      from the item's, the item is rewritten for the new base — `replaced`. A
      report from an older base than the item's is a no-op: the item already
      describes a newer tree.
    * An item a round is working (`in_progress`) is never rewritten: the
      nodes it covers are its round's, and the rest are filed elsewhere — on
      another open red-tree item, or a new one.
    * With none open, a `red-tree` item closed within `RED_TREE_COOLDOWN_S`
      whose nodes cover the report suppresses it.
    * Else a new item: priority `high`, tag `red-tree`, confirmed through
      `record_verdict` exactly as a group triage's umbrella is, with an
      `auto: true, red_tree: true` triage row — `created`.

    Every write is a `red_tree_filed` ledger row. Callers wrap this in
    try/except: a filing failure is never the gate's verdict.
    """
    from scripts.automod import state as S
    ledger = ledger or S.LEDGER_PATH
    now = time.time() if now is None else now
    changed = {str(p) for p in (changed_paths or ())}
    nodes = sorted({str(n) for n in (node_ids or ()) if _red_tree_file(n) not in changed})
    if not base or not nodes:
        return None

    def _event(item_id: int, action: str, item_nodes: list[str]) -> None:
        S.append_event({"event": "red_tree_filed", "item_id": int(item_id), "base": base,
                        "node_ids": sorted(item_nodes)[:50], "round_id": round_id,
                        "action": action}, path=ledger)

    open_items_ = open_red_tree_items(boards, backlog_dir=backlog_dir)
    # A round is working an `in_progress` item against the contract it was
    # given. Growing that contract mid-flight graded #1454's round on a clause
    # added 20 minutes into its turn (2026-09-24): what such an item already
    # covers is its round's, and only the rest is filed — before any other
    # item is looked at, or a newer open item would absorb the worked one's.
    for item in open_items_:
        if item.status == "in_progress":
            covered = {str(n) for n in (_red_tree_fm(item).get("red_tree_nodes") or [])}
            nodes = [n for n in nodes if n not in covered]
    if not nodes:
        return None
    for item in open_items_:
        if item.status == "in_progress":
            continue
        fm = _red_tree_fm(item)
        old_base = str(fm.get("red_tree_base") or "")
        old_nodes = [str(n) for n in (fm.get("red_tree_nodes") or [])]
        if old_base == base:
            if set(nodes) <= set(old_nodes):
                return None
            union = sorted(set(old_nodes) | set(nodes))
            action = "merged"
            new_base = base
        elif _is_ancestor(live_root, old_base, base):
            union = nodes
            action = "replaced"
            new_base = base
        elif old_base and _is_ancestor(live_root, base, old_base):
            return None           # an older view of a tree the item already describes
        else:
            if set(nodes) <= set(old_nodes):
                return None
            union = sorted(set(old_nodes) | set(nodes))
            action = "merged"
            new_base = old_base or base
        clauses, _dropped = B.cap_new_clauses(_red_tree_clauses(union))
        B.update_frontmatter(item.path, {"red_tree_base": new_base, "red_tree_nodes": union,
                                       "acceptance_clauses": clauses},
                           activity=(f"red tree {action} by round {round_id}: base "
                                     f"{new_base[:12]}, {len(union)} failing node(s)"))
        _event(item.id, action, union)
        return {"item_id": item.id, "action": action}

    for item in B.all_items(boards, backlog_dir=backlog_dir):
        if B.RED_TREE_TAG not in item.tags or item.status in OPEN_STATUSES:
            continue
        fm = _red_tree_fm(item)
        covered = {str(n) for n in (fm.get("red_tree_nodes") or [])}
        if not (set(nodes) <= covered and _closed_recently(fm, now, B.RED_TREE_COOLDOWN_S)):
            continue
        # Suppressed only as an OLDER view of the healed tree. A report at the
        # heal's own base (or a newer one) says the heal was wrong — the green
        # run that closed it carried a diff that fixed the test, and that diff
        # never landed — so it is filed again.
        healed = str(fm.get("red_tree_healed_base") or fm.get(B.LANDED_MARKER) or "")
        if not healed or (healed != base and _is_ancestor(live_root, base, healed)):
            return None

    files = sorted({_red_tree_file(n) for n in nodes})
    name = f"main is red: {len(nodes)} failing test(s) in {len(files)} file(s) at {base[:8]}"
    item = B.new_item(name, _red_tree_body(base, nodes, round_id), priority="high",
                    tags=(B.RED_TREE_TAG,), status="draft", board=B.RED_TREE_BOARD,
                    backlog_dir=backlog_dir)
    B.update_frontmatter(item.path, {"red_tree_base": base, "red_tree_nodes": nodes,
                                   "red_tree_round": round_id})
    item = B.load_item(item.path) or item
    clauses, dropped = B.cap_new_clauses(_red_tree_clauses(nodes))
    acceptance = (f"Every test listed on the item passes at live HEAD under the gate's "
                  f"selection (-m \"{B._RED_TREE_MARK_EXPR}\"), with no test skipped, xfailed, "
                  f"deleted or excluded to get there.")
    evidence = (f"filed by the gate: round {round_id}'s `tests` rung reproduced "
                f"{len(nodes)} failure(s) at base {base[:12]} with its diff absent")
    B.record_verdict(item, "confirmed", evidence, acceptance=acceptance,
                   acceptance_clauses=clauses, dropped_clauses=dropped)
    S.append_event({"event": "backlog_triage", "item_id": item.id, "verdict": "confirmed",
                    "held": False, "surface": "code", "check": "",
                    "evidence": evidence, "acceptance": acceptance,
                    "acceptance_clauses": clauses, "clauses_dropped": len(dropped),
                    "spawned": [], "closed": False, "auto": True, "red_tree": True,
                    "round_id": round_id, "verdict_source": "gate"}, path=ledger)
    _event(item.id, "created", nodes)
    return {"item_id": item.id, "action": "created"}


def close_healed_red_tree(base: str, round_id: str, live_root: Path | None = None, *,
                          touched=(), ledger: Path | None = None,
                          boards: tuple[str, ...] | None = B.DEFAULT_BOARDS,
                          backlog_dir: Path | None = None) -> list[int]:
    """Close every open `red-tree` item a full green run at `base` proves healed.

    Healed means the item's `red_tree_base` is an ancestor of (or equal to)
    `base`: the tree the item described has since become a tree whose whole
    suite passes. A green run at an OLDER or unrelated base proves nothing
    about the item's tree and closes nothing. Closed through `record_verdict`'s
    `already_done`, with a `red_tree_closed` ledger row. Returns the closed ids.

    A round whose own diff `touched` one of the item's failing files is the
    round fixing it: its green run is the fix, not the heal, and its landing
    closes the item through `close_settled_items`. That skip is the common
    case; a fix made in non-test code still closes here, and if it never
    lands the next report at this base files the item again (see the
    cooldown in `file_red_tree_item`).
    """
    from scripts.automod import state as S
    ledger = ledger or S.LEDGER_PATH
    touched_files = {str(p) for p in (touched or ())}
    closed: list[int] = []
    for item in open_red_tree_items(boards, backlog_dir=backlog_dir):
        fm = _red_tree_fm(item)
        item_base = str(fm.get("red_tree_base") or "")
        if not item_base or not _is_ancestor(live_root, item_base, base):
            continue
        if {_red_tree_file(n) for n in (fm.get("red_tree_nodes") or [])} & touched_files:
            continue
        B.update_frontmatter(item.path, {"red_tree_healed_base": base})
        item = B.load_item(item.path) or item
        evidence = (f"healed: round {round_id}'s `tests` rung ran the full suite green at "
                    f"base {base[:12]}, which descends from {item_base[:12]} where this "
                    f"item's tests failed")
        if B.record_verdict(item, "already_done", evidence, close=True) is None:
            continue
        S.append_event({"event": "backlog_triage", "item_id": item.id,
                        "verdict": "already_done", "closed": True, "auto": True,
                        "red_tree": True, "evidence": evidence, "spawned": [],
                        "round_id": round_id}, path=ledger)
        S.append_event({"event": "red_tree_closed", "item_id": item.id, "base": base,
                        "round_id": round_id}, path=ledger)
        closed.append(item.id)
    return closed


#: Who wrote a `red_tree_closed` row that the full-run closer did not write.
PICKUP_CLOSER = "pickup-probe"


def close_healed_red_tree_at_pickup(item: B.Item, probe, *, live_root: Path | None = None,
                                    ledger: Path | None = None,
                                    closer: str = PICKUP_CLOSER) -> dict:
    """Re-run one open red-tree item's OWN nodes at HEAD, and close it if they pass.

    The closer above is the only closer the loop had, and it is reachable only
    from a FULL green `tests` rung (`if ok and not only:` in `gate.rung_tests`).
    So an already-healed item stays open: a delta-only green run (any round that
    edits a test file) never reaches the closer, and a heal that arrived on the
    VAULT surface has no closer on its own route at all — #2384's red was undone
    by vault commit `1f5900e1`, which runs no pytest. The item then sits
    `draft`/`high` on the board and the implement loop picks it and discovers by
    hand, mid-turn, what this probe answers first: #2384 spent 22 turns and
    opened no round, and #1846 nine days earlier spent 14 the same way — the only
    two of the 50 red-tree items ever attempted that ended with no round.

    `probe(nodes) -> {"head", "unresolved", "conclusive", "note"}` is injected —
    in production `gate.red_tree_state_at_head` — so the decision is graded
    without a pytest subprocess, and the probe's polarity stays with the probe.
    `conclusive` is load-bearing here and nowhere else: `_failures_at_base` fails
    CLOSED, so a worktree that would not build, a timeout and a pytest that
    collected nothing all answer "no failures", which at the gate means "not
    pre-existing" and would mean HERE "healed — close a live red tree".

    The full-run closer's ancestry rule still holds, and it holds the same way:
    a green run at a base the item's `red_tree_base` does not descend from says
    nothing about the tree the item describes and closes nothing.

    Returns `{"action": "closed" | "attempt", "item_id", "head", "reason"}`.
    Every non-answer, refusal and mismatch is `attempt`, which is exactly today's
    behaviour: the cost of being wrong that way is one round that was going to be
    spent anyway, and the cost of being wrong the other way is a real red tree
    filed as healed.
    """
    from scripts.automod import state as S
    ledger = ledger or S.LEDGER_PATH
    if item.status not in OPEN_STATUSES:
        return {"action": "closed", "item_id": item.id, "head": "",
                "reason": f"the item is `{item.status}`, not open: no attempt to spend"}
    fm = _red_tree_fm(item)
    nodes = [str(n) for n in (fm.get("red_tree_nodes") or []) if str(n).strip()]
    if not nodes:
        return {"action": "attempt", "item_id": item.id, "head": "",
                "reason": "no `red_tree_nodes` in its front matter to probe"}
    res = probe(nodes) or {}
    head = str(res.get("head") or "")
    note = str(res.get("note") or "")
    if not res.get("conclusive"):
        return {"action": "attempt", "item_id": item.id, "head": head,
                "reason": f"pickup probe gave no answer: {note[:200]}"}
    unresolved = [str(n) for n in (res.get("unresolved") or [])]
    if unresolved:
        still = ", ".join(unresolved[:3]) + (f", +{len(unresolved) - 3} more"
                                             if len(unresolved) > 3 else "")
        return {"action": "attempt", "item_id": item.id, "head": head,
                "reason": f"still red at {head[:12]}: {still}"}
    item_base = str(fm.get("red_tree_base") or "")
    if not _is_ancestor(live_root, item_base, head):
        return {"action": "attempt", "item_id": item.id, "head": head,
                "reason": (f"the nodes pass at {head[:12] or '?'}, but the item was filed at "
                           f"{item_base[:12] or '?'} and that is not an ancestor of the tree "
                           f"the probe ran — a green run there says nothing about this item")}
    B.update_frontmatter(item.path, {"red_tree_healed_base": head})
    item = B.load_item(item.path) or item
    evidence = (f"healed at pickup: the implement loop re-ran this item's own {len(nodes)} "
                f"node(s) at HEAD {head[:12]}, which descends from {item_base[:12]} where they "
                f"failed, and every one passed ({note[:160]})")
    if B.record_verdict(item, "already_done", evidence, close=True) is None:
        return {"action": "attempt", "item_id": item.id, "head": head,
                "reason": "its front matter refused the close; the item stays as it is"}
    S.append_event({"event": "backlog_triage", "item_id": item.id,
                    "verdict": "already_done", "closed": True, "auto": True,
                    "red_tree": True, "evidence": evidence, "spawned": [],
                    "closed_by": closer}, path=ledger)
    # `base` keeps the row identical in shape to the full-run closer's, so the
    # scorecard counts a pickup close beside its closes; `head` names the same
    # sha for a reader who only knows this probe.
    S.append_event({"event": "red_tree_closed", "item_id": item.id, "base": head,
                    "head": head, "closed_by": closer, "nodes": sorted(nodes),
                    "probe": note}, path=ledger)
    return {"action": "closed", "item_id": item.id, "head": head,
            "reason": f"{len(nodes)} node(s) pass at {head[:12]}"}
