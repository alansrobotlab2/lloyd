"""A reader for the board-steward disagreement signal — #3519.

The steward's rolling-7-day agreement is measured (`judgement_aggregate`,
`workers/dispatch_watch.py`) and published (`GET /api/workers/health`), and
owed-check settled the bar `apply` turns on: summed `judged` >= 50 AND agree
`rate` >= 0.90 over the window (#3089 asked, #3329 ruled 2026-10-25). That bar
has two ways to be unmet and from the route they look identical — a steward
that has judged nothing yet, and a steward whose judgment is measured as wrong.
Only the second is actionable, and until #3519 nothing read it: the disagreeing
item ids sat in `runs.meta_json` (`agreement()` writes them as bare ints,
`workers/sources/board_steward.py`) while `autotriage` triaged items,
`owed-check` answered owed entries and `arch-review` read docs. The fleet built
the instrument, set the bar, and left no reader in the region between "not yet"
and "clears" — which is where the live window sits (2026-10-27: `judged` 127,
`rate` 0.7953, 26 disagreements).

This module is that reader. At the end of a steward tick, when the window is
measured AND under the floor, it files — or adds to — ONE draft item on the
`lloyd` board naming each disagreeing item with **both sides**: the status the
steward proposed and the status `desired_statuses` wants, with the machine's
`why`. Then a person or `autotriage` can rule which side is wrong without
hand-running sqlite.

Two things this deliberately is not:

* **It does not move the flip.** `apply` is read once at the top of the tick and
  this runs after it; filing a question is not widening what the steward may
  write. The bar itself stays owed-check's to rule.
* **It is not in `workers/dispatch_watch.py`.** That module keeps the reader and
  nothing else precisely so the backlog writer is not reachable from
  `app.routers.workers`' import closure ("Read-only, and it decides nothing"),
  and a trigger there would undo that rail. The decision lives here, on the
  writer side, at the end of a tick — where the writer is already reachable.

**Keep the host imports function-local.** They are imported inside the functions
below, the way `board_steward.py` imports its own. Hoisting them to module level
would reshape what the worker pool's process imports at startup and re-grow the
`app` closure `tests/test_board_steward.py::test_worker_module_import_surface_
matches_what_was_shipped` guards (#3323).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-workers.board-disagreement")

#: The tag that identifies the one open disagreement item. Finding the item by
#: tag rather than by name is what lets a merge find an item whose name was
#: edited by hand afterwards.
TAG = "board-steward-disagreement"

#: The filing board — the loop's own board, where `autotriage` and the loop read.
BOARD = "lloyd"

#: The item's name, and so its slug, deliberately carries no count: a merge
#: adds rows to the same item, and a name that says how many rows there are
#: starts lying the moment one more tick disagrees.
NAME = ("board-steward disagreed with the state machine and nobody has ruled "
        "which side is wrong")

#: owed-check's flip bar (#3089 asked, #3329 ruled 2026-10-25) — restated here so
#: the trigger and the flip name one pair of numbers rather than two copies that
#: can drift apart. Restating is not owning: the floor is ruled in
# `architecture/workers-jobs.md` §board-steward and by owed-check, and
#: `tests/test_board_steward.py` pins these two constants against that section so
#: a re-ruled floor fails loudly instead of quietly moving the trigger.
FLIP_MIN_JUDGED = 50
FLIP_MIN_RATE = 0.90

#: Rows one filing may write. A tick is capped at `MOVES_MAX` (40) moves by
#: `parse_steward`, so this is not a truncation in practice; it is the bound that
#: keeps a malformed `disagree` list from writing an unbounded body.
ROWS_MAX = 40

#: Ledger event kind, read back with `B._ledger_events(ledger, EVENT)`.
EVENT = "board_steward_disagreement"

_ROW_RE = re.compile(r"^- #(\d+)[:\s]", re.MULTILINE)


def below_floor(window: dict) -> tuple[bool, str]:
    """Is this window a MEASURED disagreement with the flip bar? (and if not, why.)

    Both legs, in the order owed-check's rule states them: without the volume
    leg the rate is not evidence yet (`truncated` says so too — a capped read
    counts whole ticks, so it understates both sums), and at or above the floor
    the steward is agreeing, which needs no item.
    """
    if not window or not window.get("judged"):
        return False, "nothing judged in the window"
    if window.get("truncated"):
        return False, "window read truncated: sums understate, no verdict"
    if int(window["judged"]) < FLIP_MIN_JUDGED:
        return False, f"volume leg unmet: judged {window['judged']} < {FLIP_MIN_JUDGED}"
    rate = window.get("rate")
    if rate is None:
        return False, "rate is None: nothing judged, so nothing measured"
    if float(rate) >= FLIP_MIN_RATE:
        return False, f"rate {rate} is at or above the {FLIP_MIN_RATE} floor"
    return True, f"judged {window['judged']} with rate {rate} under the {FLIP_MIN_RATE} floor"


def disagreement_rows(ids: list[int], moves: list[dict],
                      expected: dict[int, tuple]) -> list[str]:
    """One body line per disagreeing item id, naming **both** sides.

    The steward's side comes from its own parsed `moves` (what it asked for that
    id), and the machine's side from `expected` — `desired_statuses`'
    `{id: (status, why[, needs_human])}`, the same object the tick already
    compared against at `board_steward.py:769`. Both halves are therefore in
    scope at the trigger site, which is why this needs no new persistence: the
    proposed move is what the store never kept (`response_json` is empty in all
    1,142 board-steward rows).
    """
    proposed: dict[int, str] = {}
    for m in moves or []:
        iid = m.get("item_id")
        if isinstance(iid, int) and m.get("status"):
            proposed[iid] = str(m["status"])
    out: list[str] = []
    for iid in list(ids)[:ROWS_MAX]:
        want = expected.get(iid) if expected else None
        machine = str(want[0]) if want else "(the machine abstains)"
        why = ""
        if want and len(want) > 1 and want[1]:
            why = f" — {want[1]}"
        if want and len(want) > 2 and want[2]:
            why += " (needs a human)"
        out.append(f"- #{iid}: steward proposed `{proposed.get(iid, '(not in its moves)')}`"
                   f"; the state machine wants `{machine}`{why}")
    return out


def file_or_update(agree: dict, moves: list[dict], expected: dict[int, tuple], *,
                   window: dict, backlog_dir: Path | None = None,
                   ledger: Path | None = None) -> dict | None:
    """File or add to the ONE open disagreement item; return what happened.

    `window` is `judgement_aggregate`'s dict for the same 7 days the health route
    serves, passed in rather than looked up here so the number that triggers the
    filing and the number on the route cannot be two readings of two windows.

    Returns `None` when nothing was filed (bar met, volume leg unmet, nothing
    disagreed, or an open item a round is working). Never raises on a missing
    store; raises on a real write failure, and its caller swallows — see
    `_file_quietly`.
    """
    ids = sorted({int(i) for i in (agree.get("disagree") or []) if isinstance(i, int)})
    if not ids:
        return None
    should, reason = below_floor(window)
    if not should:
        logger.debug("disagreement filing skipped: %s", reason)
        return None

    B = _backlog()
    openers = [it for it in B.open_items((BOARD,), backlog_dir=backlog_dir)
               if TAG in (it.tags or ())]

    # An item a round is working is never rewritten — same rule
    # `file_red_tree_item` keeps for a `blocked: red-tree` round — and nothing is
    # filed beside it either: the round working that item IS the reader this
    # window needs, and a second item naming the same disagreement would be the
    # duplicate this clause exists to prevent.
    if any(it.status == "in_progress" for it in openers):
        logger.info("disagreement filing skipped: #%s is being worked",
                    [it.id for it in openers if it.status == "in_progress"])
        return None

    stamp = _stamp()
    if openers:
        return _merge(openers[0], B, ids, moves, expected, window,
                      stamp, backlog_dir=backlog_dir, ledger=ledger)
    return _create(B, ids, moves, expected, window, stamp,
                   backlog_dir=backlog_dir, ledger=ledger)


def _create(B: Any, ids: list[int], moves: list[dict], expected: dict[int, tuple],
            window: dict, stamp: str, *, backlog_dir: Path | None,
            ledger: Path | None) -> dict:
    rows = disagreement_rows(ids, moves, expected)
    body = (
        f"# {NAME}\n\n"
        f"Filed by `workers/board_disagreement.py` at the end of a board-steward "
        f"tick whose trailing-7-day judgement was measured **below** owed-check's flip "
        f"floor: `judged={window.get('judged')}` (the leg it needs is "
        f"{FLIP_MIN_JUDGED}) with agree `rate={window.get('rate')}` against a floor of "
        f"{FLIP_MIN_RATE}, over the window since {window.get('window_since')}. The volume "
        f"leg is cleared, so this is a measured disagreement and not an unmeasured one — "
        f"which is the case nothing used to read.\n\n"
        f"Each row names BOTH sides, from the same comparison the tick made: what the "
        f"steward proposed for that item, and what `desired_statuses` "
        f"(`scripts/automod/backlog.py`) wants, with its reason. The proposed move is "
        f"stored nowhere else — `response_json` is empty in every board-steward "
        f"`runs` row — so this item is the record.\n\n"
        f"## Rows\n\n" + "\n".join(rows) + "\n\n"
        f"## What to decide\n\n"
        f"Per item, rule which side is wrong:\n\n"
        f"- the **steward's** prompt/grounding (`workers/sources/board_steward.py`) — then "
        f"say what it should have proposed, and its prompt or grounding changes; or\n"
        f"- the **state machine** (`scripts/automod/backlog.py::desired_statuses`) — then "
        f"the machine's rule changes.\n\n"
        f"`apply` does NOT turn on this item. It turns only on owed-check's bar "
        f"(`judged >= {FLIP_MIN_JUDGED}` AND `rate >= {FLIP_MIN_RATE}` over the rolling 7 "
        f"days, #3089 asked / #3329 ruled), and this filing never edits a status: it is "
        f"filed `draft` and left for a judgment.\n"
    )
    # `new_item` (host:scripts/automod/backlog.py:8573) returns the loaded Item and
    # takes no activity line — the create's record is this body plus the ledger
    # row below, which is exactly one write and no second chance to fail.
    it = B.new_item(NAME, body, priority="medium", tags=[TAG], status="draft",
                    board=BOARD, backlog_dir=backlog_dir)
    _event(ledger, item_id=it.id, action="filed", ids=ids, window=window,
           stamp=stamp, new_ids=ids)
    return {"action": "filed", "item_id": it.id, "path": str(it.path),
            "ids": ids, "rows": len(rows)}


def _merge(item: Any, B: Any, ids: list[int], moves: list[dict],
           expected: dict[int, tuple], window: dict, stamp: str, *,
           backlog_dir: Path | None, ledger: Path | None) -> dict | None:
    """Add the ids this item does not name yet, into the same file."""
    try:
        text = Path(item.path).read_text(encoding="utf-8")
    except OSError:
        text = ""
    fm, body = B._split_frontmatter(text)
    seen = {int(m) for m in _ROW_RE.findall(body or "")}
    fresh = [i for i in ids if i not in seen]
    if not fresh:
        logger.debug("disagreement item #%s already names %s", item.id, ids)
        return None
    rows = disagreement_rows(fresh, moves, expected)
    new_body = ((body or "").rstrip("\n") + "\n\n"
                f"### Added {stamp} (window `judged={window.get('judged')}`, "
                f"`rate={window.get('rate')}`)\n\n" + "\n".join(rows) + "\n")
    _write_body(Path(item.path), fm, new_body, B,
                activity=f"board-steward disagreement: {len(fresh)} row(s) added")
    _event(ledger, item_id=item.id, action="merged", ids=ids, window=window,
           stamp=stamp, new_ids=fresh)
    return {"action": "merged", "item_id": item.id, "path": str(item.path),
            "ids": ids, "new_ids": fresh, "rows": len(rows)}


def _write_body(path: Path, fm: dict, body: str, B: Any, *, activity: str) -> None:
    """Move the body and the front matter together, under one lock, one write.

    `update_frontmatter` is the public writer but it re-writes the body it just
    read, so a caller that changed the body cannot use it; the alternative —
    write the body, then call it — is two writes with a window in which another
    reader sees the new rows and the old `updated`. So this takes the same lock
    it takes and does the same tail: re-read under the lock (an edit that landed
    while we were computing is kept, not overwritten), refresh `updated`, append
    the activity line, write atomically. It never touches `status`, which is the
    invariant every backlog writer keeps.
    """
    from app.atomic_io import commit_lock
    from app import backlog_write as BW
    from app.backlog_move import now_stamp

    with commit_lock(path) as lock:
        current = path.read_text(encoding="utf-8")
        fm_now, _body_now = B._split_frontmatter(current)
        # A front matter that will not parse is not rewritten by this reader:
        # `update_frontmatter` refuses the same case through `_unparsed_guard`
        # rather than write a file whose front matter it could not read, and a
        # silent drop of `status`/`board` would be worse than an unmerged row.
        # `name` is deliberately NOT required here: `new_item` writes no `name:`
        # key at all (the title is the body's `# heading`), so requiring one
        # would refuse every item this module files itself.
        if not isinstance(fm_now, dict) or not fm_now.get("status"):
            raise ValueError(f"{path.name}: front matter does not parse; merge refused")
        fm2 = dict(fm_now)
        stamp = now_stamp()
        log = list(fm2.get("activity_log") or [])
        log.append(f"**{stamp}** — {activity}")
        fm2["activity_log"] = log
        fm2["updated"] = stamp
        # Whatever the file holds NOW is what our rows go on top of. The id-dedup
        # ran against the read taken before the lock, and this lock excludes every
        # other writer of it, so the only party that can have moved the body in
        # between is a person editing the item — and for them, keeping their text
        # and adding the rows is the safe answer.
        BW.write_item(path, fm2, body, lock=lock)


def _file_quietly(agree: dict, moves: list[dict], expected: dict[int, tuple],
                  *, backlog_dir: Path | None = None) -> dict | None:
    """`file_or_update` for the end of a tick: the window is read here, and no
    failure of any kind is allowed to reach the caller.

    The filing is a report on the tick, never its outcome: an unavailable store
    must not turn a `success` tick into a `failed` one, because the agreement
    record that decides the flip is written from tick *success* — a filing that
    could break the instrument it reports on would make the flip harder to earn,
    not easier. The exception is logged with its type, so the outage is visible
    on the same route that publishes the rate.
    """
    try:
        window = window_now()
        return file_or_update(agree, moves, expected, window=window,
                              backlog_dir=backlog_dir)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: see the docstring
        logger.warning("board-steward disagreement filing failed (%s: %s); "
                       "the tick is unaffected", type(exc).__name__, exc)
        return None


def window_now(now: float | None = None) -> dict:
    """The published rolling-7-day judgement, read from the store the route reads.

    One function with the health endpoint, so the trigger and the number a person
    curls are the same measurement (`judgement_aggregate` is what
    `app/routers/workers.py` serves). Opening the queue is done the way
    `workers/sources/autocode.py` opens it — the pool's singleton when this is
    the pool's process, the configured db otherwise.
    """
    from workers.dispatch_watch import judgement_aggregate
    from workers.queue import configured_db_path, get_queue
    try:
        queue = get_queue()
    except Exception:  # noqa: BLE001 — not the pool's process: no singleton yet
        queue = get_queue(configured_db_path())
    return judgement_aggregate(queue, now=now)


# ── small helpers, each importing its host inside itself ────────────────────

def _backlog() -> Any:
    from scripts.automod import backlog as B
    return B


def _stamp() -> str:
    from scripts.automod import state as S
    return S.now_iso()


def _event(ledger: Path | None, *, item_id: int | None, action: str,
           ids: list[int], window: dict, stamp: str,
           new_ids: list[int] | None = None) -> None:
    """One ledger row per file-or-merge, carrying the window's own numbers.

    The run's `summary` is capped at 500 characters (#1606) and 77 of the 154
    board-steward rows already stop at exactly 500, so the summary cannot hold
    this record. The ledger can: `judged`, `disagree` and `rate` are the window's,
    so a reader of the ledger sees the same figures the route published at the
    moment the artifact was written.
    """
    from scripts.automod import state as S
    S.append_event({
        "event": EVENT,
        "item_id": item_id,
        "action": action,
        "judged": window.get("judged"),
        "disagree": window.get("disagree"),
        "rate": window.get("rate"),
        "window_since": window.get("window_since"),
        "tick_item_ids": ids,
        "new_item_ids": new_ids if new_ids is not None else ids,
        "at": stamp,
    }, path=ledger)
