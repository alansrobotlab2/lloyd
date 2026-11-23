"""The sweep: every open item read once, retired or ranked.

The functions of the sweep (`sweep_pool`, `record_sweep_verdicts`,
`unfold_oversized_umbrellas`, ...), moved out of `scripts/automod/backlog.py`
(2026-11-23, the second cut of the split assessed in `architecture/backlog.md`).
Why the sweep exists and its constants (`SWEEP_VERDICTS`, `SWEEP_SCHEMA`,
`SWEEP_POOL_STATUSES`) stay in `backlog`, beside the comment that explains them.
The bodies are unchanged except that every reference to a backlog name — this
module's own names included — reads `B.<name>`, so a `monkeypatch.setattr(backlog,
"sweep_enabled", ...)` reaches this code as it did before. The foot of this
module binds every name it defines back onto `backlog`.
"""
from __future__ import annotations

from pathlib import Path

from app.backlog_tags import NEEDS_HUMAN_TAG
from app.backlog_status import OPEN_STATUSES

from scripts.automod import backlog as B


def sweep_enabled() -> bool:
    """`workers.sources.autotriage.sweep`, default off. Lazy and fail-closed:
    with no config there is no sweep, and nothing yields to one."""
    try:
        from app.config import CONFIG
        cfg = (((CONFIG or {}).get("workers") or {}).get("sources") or {}).get("autotriage") or {}
        return bool(cfg.get("sweep", False))
    except Exception:  # noqa: BLE001
        return False


def sweep_abandoned_ids(ledger: Path) -> set[int]:
    """Items in a batch that ran out of budget twice. Left unswept for the
    ordinary passes rather than retried forever."""
    out: set[int] = set()
    for d in B._ledger_events(ledger, "backlog_sweep", require_item=False, merged=True):
        if d.get("verdict") == "abandoned":
            out |= {int(i) for i in (d.get("item_ids") or [])}
    return out


def sweep_pool(ledger: Path, boards: tuple[str, ...] | None = B.DEFAULT_BOARDS, *,
               items: list[B.Item] | None = None) -> list[B.Item]:
    """Open items the sweep has not read, never-judged first, then by the
    human's priority, then oldest. A `high` item is not the sweep's to read:
    single triage takes it next, whatever else is unread.

    Quarantine does not apply — reading a self-filed item once is the whole
    point. `draft` and `up_next` both: a confirmed item is ranked so the
    implement order can prefer it, and can still be retired if its premise
    has gone. Out: grouped members (their umbrella is the unit), umbrellas
    (confirmed work with a contract), `needs-human`, `expired`, anything
    already swept or parked, and a batch that ran out of budget twice.
    """
    # `swept_ids` merged with the rotations: a sweep row is the record that the sweep
    # HAS read an item, and `LEDGER_ARCHIVE_AGE_DAYS = 14` cannot unread it. Left
    # live-only it did two things at once — the item went back into `unswept` to be
    # re-read every fortnight forever, and `.sweep.unswept` in `board_health` moved
    # across the fold, which is the refusal this whole change is about (#3612).
    done = B.swept_ids(ledger, merged=True) | B.sweep_abandoned_ids(ledger)
    seen = B.triaged_ids(ledger)
    claimed = B.triage_claimed_ids() | B.implement_claimed_ids()
    pool = [i for i in (items if items is not None else B.open_items(boards))
            if i.status in B.SWEEP_POOL_STATUSES and i.id not in done
            and i.id not in claimed
            and not B.has_standing_rank(i) and not B.is_parked(i) and not B.is_high(i)
            and not B.is_grouped(i) and not B.is_umbrella(i)
            and NEEDS_HUMAN_TAG not in i.tags and B.EXPIRED_TAG not in i.tags]
    return sorted(pool, key=lambda i: (i.id in seen, B.priority_key(i), i.created or "9999", i.id))


def select_sweep_batch(ledger: Path, n: int,
                       boards: tuple[str, ...] | None = B.DEFAULT_BOARDS) -> list[B.Item]:
    return B.sweep_pool(ledger, boards)[:max(1, int(n))]


def sweep_pending(ledger: Path, boards: tuple[str, ...] | None = B.DEFAULT_BOARDS) -> int:
    """How many items a sweep still has to read; 0 when the sweep is off.
    Autocode yields to a positive answer (`yield_to_sweep`)."""
    if not B.sweep_enabled():
        return 0
    return len(B.sweep_pool(ledger, boards))


def sweep_batch_id(item_ids) -> str:
    import hashlib
    ids = sorted(int(i) for i in item_ids)
    return "sw-" + hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:10]


def sweep_incomplete_attempts(ledger: Path, batch_id: str) -> int:
    return sum(1 for d in B._ledger_events(ledger, "backlog_sweep", require_item=False)
               if d.get("batch_id") == batch_id and d.get("verdict") == B.INCOMPLETE)


def record_sweep_verdicts(batch_id: str, members: list[B.Item], verdicts: dict[int, dict], *,
                          session_id: str = "", extra: dict | None = None,
                          ledger: Path | None = None) -> dict:
    """Write a sweep's verdicts onto the items and the ledger.

    Retiring verdicts go through `record_verdict` and get a `backlog_triage`
    row, so the status pipeline, the scorecard and `triaged_ids` see an
    ordinary close. `duplicate_of` may point at any OPEN item, in the batch
    or not; a target that is closed, missing, or itself retired here becomes
    `keep`. A `keep` writes `worth`/`size` and the `swept` tag; a `low`
    draft is `parked` too. Only judged ids are recorded — an item the turn
    did not list stays unswept and is offered again.
    """
    from scripts.automod import state as S
    ledger = ledger or B.LEDGER_DEFAULT()
    by_id = {m.id: m for m in members}
    member_ids = set(by_id)
    verdicts = B.sweep_verdicts_not_mid_round(verdicts, member_ids, ledger)
    # A target outside the batch counts as a member for chain resolution
    # when it is open on disk; anything else `_resolve_duplicates` turns
    # into a keep.
    external = set()
    for i, v in verdicts.items():
        if v.get("verdict") != "duplicate_of":
            continue
        t = int(v.get("duplicate_of") or 0)
        if t and t not in member_ids:
            target = B.item_by_id(t)
            if target is not None and target.status in OPEN_STATUSES and not B.is_parked(target):
                external.add(t)
    verdicts = B._ranked_only(B._resolve_duplicates(verdicts, member_ids | external))
    counts = {"retired": 0, "duplicates": 0, "kept": 0, "parked": 0}
    judged: dict[str, str] = {}
    ranked: dict[str, str] = {}
    for i, v in verdicts.items():
        item = by_id[i]
        verdict = str(v.get("verdict") or "keep")
        evidence = " ".join(str(v.get("evidence") or "").split())[:600]
        worth = B._level(v.get("worth"), B.WORTH_LEVELS)
        size = B._level(v.get("size"), B.SIZE_LEVELS)
        judged[str(i)] = verdict
        if verdict == "duplicate_of":
            t = int(v["duplicate_of"])
            refused = B.record_verdict(item, "stale", f"duplicate of #{t}: {evidence}",
                                     close=True) is None
            B.update_frontmatter(item.path, {"duplicate_of": t})
            S.append_event({"event": "backlog_triage", "item_id": i, "verdict": "stale",
                            "closed": not refused, "duplicate_of": t, "sweep_batch": batch_id,
                            "evidence": evidence[:1000], "session_id": session_id,
                            "spawned": [], "auto": True,
                            **B.close_refusal_fields(item, refused)}, path=ledger)
            # The pair is recorded and `duplicate_of` is on the item whatever the close
            # did, which is what this counter counts; `closed` on the row is the item
            # (#4284, and the same reading as `apply_group_verdicts` above).
            counts["duplicates"] += 1
        elif verdict in B.RETIRING:
            refused = B.record_verdict(item, verdict, evidence, close=True) is None
            S.append_event({"event": "backlog_triage", "item_id": i, "verdict": verdict,
                            "closed": not refused, "sweep_batch": batch_id,
                            "evidence": evidence[:1000], "session_id": session_id,
                            "spawned": [], "auto": True,
                            **B.close_refusal_fields(item, refused)}, path=ledger)
            # Not a retirement until the close happened: the sweep docstring promises
            # these rows are "an ordinary close", and a held carrier is not one (#4284).
            if not refused:
                counts["retired"] += 1
        else:
            park = worth == "low" and item.status == B.TRIAGE_POOL_STATUS
            tags = (B.SWEPT_TAG,) + ((B.PARKED_TAG,) if park else ())
            B.update_frontmatter(item.path, {"worth": worth or None, "size": size or None},
                               activity=(f"swept ({batch_id}): worth={worth or '?'} "
                                         f"size={size or '?'}" + (" — parked: out of every pool "
                                         "until a person removes the tag" if park else "")
                                         + (f" — {evidence}" if evidence else ""))[:700],
                               add_tags=tags)
            ranked[str(i)] = f"{worth or '?'}/{size or '?'}"
            counts["kept"] += 1
            counts["parked"] += int(park)
    summary = {"event": "backlog_sweep", "batch_id": batch_id,
               "item_ids": sorted(member_ids), "judged": judged, "ranked": ranked, **counts,
               "session_id": session_id, **(extra or {})}
    S.append_event(summary, path=ledger)
    return {**counts, "judged": judged, "ranked": ranked}


def unfold_oversized_umbrellas(ledger: Path, boards: tuple[str, ...] | None = B.DEFAULT_BOARDS, *,
                               min_clauses: int = 8, dry_run: bool = False) -> list[dict]:
    """Unfold every never-attempted open umbrella whose contract carries at
    least `min_clauses` clauses, and close it `done` tagged `unfolded`.

    56 umbrellas sat in `up_next` on 2026-09-15, every one with 8–12
    clauses from the days before `group_max_items` fell to 4, and that shape
    landed 5 of 27 rounds against 52 of 151 for singles. Their 158 members
    were folded out of every pool. A member goes back to `draft` untriaged
    (its `folded` row is not a verdict), where the sweep reads and ranks it;
    it does not re-cluster (`group_triaged_ids`). An umbrella a round has
    already attempted, or that is landing or landed, is not touched — that
    is `unfold_spent_umbrellas`' or a person's call.
    """
    from scripts.automod import state as S
    history = B.implement_history(ledger)
    unfinished = B.items_with_unfinished_rounds(ledger, history=history)
    confirmed = B.confirmed_verdicts(ledger, with_archives=True)
    out: list[dict] = []
    for item in B.open_items(boards):
        if not B.is_umbrella(item) or item.id in history or item.id in unfinished:
            continue
        fm, _ = B._split_frontmatter(item.path.read_text(encoding="utf-8"))
        if fm.get(B.LANDED_MARKER) or any(fm.get(m) for m in B._LEGACY_LANDED_MARKERS):
            continue
        n = len(B.acceptance_clauses_of(confirmed.get(item.id), fm))
        if n < int(min_clauses):
            continue
        reason = (f"oversized: {n} clauses against a cap of {B.MAX_CLAUSES} and never attempted; "
                  f"members released to be swept and ranked on their own")
        row = {"umbrella_id": item.id, "clauses": n, "members": list(item.members), "reason": reason}
        if dry_run:
            out.append(row)
            continue
        res = B.unfold_umbrella(item.id, reason, ledger=ledger)
        B.set_status(item.id, "done", f"unfolded: {reason}",
                   add_tags=(B.UNFOLDED_TAG,), remove_tags=(NEEDS_HUMAN_TAG, B.HELD_TAG))
        S.append_event({"event": "backlog_umbrella_unfolded", "item_id": item.id,
                        "released": res["released"], "reason": reason[:400],
                        "oversized": True, "clauses": n, "auto": True}, path=ledger)
        out.append({**row, "released": res["released"]})
    return out


# Bound onto `backlog`, so `backlog.<name>`, `from scripts.automod.backlog import
# <name>`, a bare call from code still in backlog, and a monkeypatch on backlog
# all keep meaning what they meant before the move.
SWEEP_NAMES = (
    "sweep_enabled",
    "sweep_abandoned_ids",
    "sweep_pool",
    "select_sweep_batch",
    "sweep_pending",
    "sweep_batch_id",
    "sweep_incomplete_attempts",
    "record_sweep_verdicts",
    "unfold_oversized_umbrellas",
)
for _name in SWEEP_NAMES:
    setattr(B, _name, globals()[_name])
