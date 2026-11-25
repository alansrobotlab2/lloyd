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

#: Where the model's raw reply actually lives, as ONE sentence held here so the
#: `disagreement_rows` docstring, the rendered filing body and
#: `architecture/workers-jobs.md` cannot drift into three separate claims (#4387).
#: Boundary-limited on purpose: `43cc6ba7` (2026-11-19) is the commit that started
#: writing the reply into `runs.response_json`, so the sentence this one replaced — that
#: the store never keeps the reply at all — read as a present-tense limitation while
#: being true only of rows an older pool wrote. It quotes no row count, and neither may
#: the two other surfaces: the figure the architecture page published beside the old
#: claim had rotted before the next reader checked it, which is what a count in prose
#: always does and why none of the three may carry one.
REPLY_PROVENANCE = (
    "The model's raw reply is kept in `runs.response_json` for any row a pool running "
    "`43cc6ba7` (2026-11-19) or later wrote; an older row is empty there, and any tick's "
    "reply is recoverable from its `meta_json.session_id` at "
    "`~/lloyd-data/sessions/<id>.json`."
)

#: owed-check's flip bar (#3089 asked, #3329 ruled 2026-10-25) — restated here so
#: the trigger and the flip name one pair of numbers rather than two copies that
#: can drift apart. Restating is not owning: the floor is ruled in
# `architecture/workers-jobs.md` §board-steward and by owed-check, and
#: `tests/test_board_steward.py` pins these two constants against that section so
#: a re-ruled floor fails loudly instead of quietly moving the trigger.
FLIP_MIN_JUDGED = 50
FLIP_MIN_RATE = 0.90

#: The two names one filing's direction takes — the ledger row's `leg` field AND the
#: wording of the created body's header. One pair of spellings, chosen so the string a
#: reader SELECTs out of `promotions.jsonl` is the same string that reader sees on the
#: item (#4251).
LEG_REVERT = "post-flip revert"
LEG_RATE = "pre-flip rate"

#: The prefix of the reason `below_floor` returns from its post-flip leg, and the ONLY key
#: `leg_from_trigger` reads. The reason is built FROM this constant so the two cannot drift
#: by editing one of them; a re-worded reason that moved the prefix would otherwise
#: silently re-label every revert filing as a rate filing, in both surfaces at once.
#: Pinned in `tests/test_board_steward.py` against the string `below_floor` really emits.
REVERT_TRIGGER_PREFIX = "steward_move applied"


def leg_from_trigger(trigger: str) -> str:
    """Which leg of `below_floor` filed this, derived from the trigger it returned.

    The ONE derivation of the direction, read by every surface that publishes it: the
    `leg` field of the `_event` ledger row, the created body's header sentence, and the
    `### Added` heading a widening writes. Before #4251 only the ledger row derived
    anything — the header was one unconditional f-string declaring the judgement "measured
    **below** owed-check's flip floor", so a revert-leg filing on a window that had CLEARED
    both floors printed the opposite of its own `rate`, beside a row that said
    `post-flip revert`. The body's reader is the reader of `config.yaml`'s 7-day stop
    condition, which is why both surfaces now answer through this one function.
    """
    return (LEG_REVERT if str(trigger or "").startswith(REVERT_TRIGGER_PREFIX)
            else LEG_RATE)

#: Rows one filing may write. A tick is capped at `MOVES_MAX` (40) moves by
#: `parse_steward`, so this is not a truncation in practice; it is the bound that
#: keeps a malformed `disagree` list from writing an unbounded body.
ROWS_MAX = 40

#: Ledger event kind, read back with `B._ledger_events(ledger, EVENT)`.
EVENT = "board_steward_disagreement"

#: The two kinds the post-flip leg pairs (`reverted_moves`). `steward_move` is the
#: steward's own applied write (#3467); `status_moved` is the state machine's, stamped by
#: `scripts/automod/backlog.py` on every status it changes — including one it puts back.
EVENT_KIND_APPLIED = "steward_move"
EVENT_KIND_REVERT = "status_moved"

#: The two halves of the reason the state machine stamps on its OWN abstention, and why
#: neither one of them is a revert. `scripts/automod/backlog.py:5738-5740`
#: (`desired_statuses`, the #3467 handover) writes, for an item whose latest
#: `steward_move` still stands: `the board steward moved it to {claim[1]}
#: (`steward_move`: {claim[2]}); the machine abstains until this item lands, is promoted or
#: is re-triaged`. That row is the machine SAYING THE STEWARD WON, stamped into the same
#: `status_moved` kind the leg reads for a put-back — so a reader that keys on the kind
#: alone scores the machine's loudest agreement as its own reversal.
#:
#: Both halves are required, not just the first: an item's own triage or park reason can
#: legitimately contain the phrase "the board steward moved it" (a park line quoting the
#: steward's claim does), and only the abstention half is the machine's first-person
#: sentence. The pair is checked case-insensitively because the reason crosses a process
#: boundary, and `test_the_abstention_echo_filter_is_read_from_the_writer_that_stamps_it`
#: re-reads both fragments out of `desired_statuses`' own source, so a writer that ever
#: re-words its sentence reddens here instead of silently re-arming the false pair.
ABSTENTION_ECHO_MARKS = ("the board steward moved it", "the machine abstains")


def _is_abstention_echo(ev: dict) -> bool:
    """Is this `status_moved` the machine echoing the steward's claim instead of moving?

    `bool(ABSTENTION_ECHO_MARKS) and` is not decoration: `all()` over an EMPTY iterable is
    True, so an emptied tuple would classify EVERY row as an abstention echo and silence
    the whole revert leg — the false negative, which is the direction that hides a real
    disagreement. An empty mark list has to read as "no filter", which is also what the
    mutation in `test_the_machines_abstention_echo_is_not_a_revert_even_when_its_to_differs`
    does to delete the filter.
    """
    reason = str(ev.get("reason") or "").lower()
    return bool(ABSTENTION_ECHO_MARKS) and all(mark in reason for mark in ABSTENTION_ECHO_MARKS)


def _supersedes(cur: dict, ev: dict) -> bool:
    """Does `ev` replace `cur` as the applied move the item currently stands on?

    The store is append-only, so the row stamped later is the one still in force, and a tie
    goes to the row read second for the same reason. Extracted from `reverted_moves`'
    selection loop so the pairing rule has one name and one mutation:
    `test_a_superseded_applied_move_is_not_a_revert_the_latest_one_is` replaces this with
    an always-False rule — which is the pairing that shipped until #4409, first row per item
    — and requires that node to redden. An earlier row that a later one overwrote is not
    evidence of anything the steward still believes, which is what the whole leg claims to
    measure.
    """
    return _iso_norm(str(ev.get("created_at") or "")) >= _iso_norm(
        str(cur.get("created_at") or ""))

_ROW_RE = re.compile(r"^- #(\d+)[:\s]", re.MULTILINE)


def reverted_moves(window: dict, ledger: Path) -> list[dict]:
    """Applied moves the MACHINE PUT BACK inside this window: the post-flip leg.

    Why this exists at all. `agreement()` scores a `disagree` only over the moves it
    proposes on the tick that asks, so once `apply: true` has written a wrong status the
    item sits at the steward's value and is never re-proposed: the divergence lands in
    `missed` and in the `judged` denominator only and NEVER arrives as a fresh
    `disagree` (`config.yaml`'s board-steward block, the pre-flip-only caveat). Post-flip
    the rate can therefore only drift down and can never score a fresh disagreement, so
    a window that holds `rate` >= 0.90 while the state machine silently reverts every
    move the steward applied would read as agreement by the only measure the old
    `below_floor` could see — the alarm the flip was required to keep alive would have
    gone blind exactly when the flip made it the only witness left.

    The event pair is what the flip itself writes. `workers/sources/board_steward.py`
    stamps a `steward_move` row for every status it APPLIES (#3467), and
    `desired_statuses` abstains from its own verdict on that item until a row of its own
    is stamped after it — when it does move, `scripts/automod/backlog.py` stamps a
    `status_moved` row naming the status it moved FROM and TO. A `status_moved` whose
    `to` differs from the status the steward wrote, stamped after that `steward_move` and
    inside the same window, IS the disagreement: the steward applied a change and the
    machine put it back.

    Returns `[{"item_id", "applied", "reverted_to", "at", "reason"}]`, one per item: the
    FIRST revert found AFTER that item's LATEST applied move. Two filters, both of which the
    live post-flip ledger proved necessary (#4409, measured 2026-11-21 over the rows the
    flip wrote):

    * **LATEST applied, not first.** An item the steward moves twice in one window has two
      `steward_move` rows, and the second supersedes the first — the machine putting the
      item at the status the steward's NEWEST row says is agreement, not a put-back. The
      shipped pairing took the item's FIRST row and stopped at the first revert, so #4240
      (`draft→up_next` at 2026-11-17T23:27:39Z, then `up_next→in_progress` at
      2026-11-18T00:50:10Z) paired the stale first row against a `to: in_progress` move and
      fired a disagreement that never happened; #4244 did the same with `up_next→draft` at
      2026-11-18T00:50:11Z. Pairing on the latest row leaves only the two genuine pairs in
      the window (#4172, #4151) — see
      `test_a_superseded_applied_move_is_not_a_revert_the_latest_one_is`.
    * **An abstention echo is not a revert.** `ABSTENTION_ECHO_MARKS` above; the machine's
      own `the machine abstains` row is stamped `status_moved`, and its `to` can differ from
      an applied status whenever the steward moved the item again before the machine echoed,
      which is a second way the same tick manufactures a pair out of an agreement.

    An applied row stamped BEFORE `window_since` is out too, which is what "inside the same
    window" has always claimed above: the shipped reader windowed only the revert, so one
    stale apply could keep pairing with every machine move for the rest of the store. On the
    rows the flip actually wrote this changes nothing — the six revert-leg firings' windows
    each still return exactly the pairs the un-windowed read returns — so the narrowing
    costs no real finding and removes a source of re-firing.

    THE FIELD NAMES ARE MEASURED, NOT ASSUMED (#4147's first round died here). The stamp
    is `created_at` on EVERY row because `scripts/automod/state.py:187` prepends
    `{"ts": …, "created_at": now_iso()}` before the entry's own keys — it is not a key any
    writer passes, so reading `at`, which this module's OWN `_event` does add for
    `board_steward_disagreement` rows, finds nothing on these two kinds and the leg is
    silently dead post-flip: the real row reads `{"created_at": "2026-11-15T23:58:40Z",
    "event": "status_moved", "item_id": 4155, "from": "up_next", "to": "draft", "reason":
    "its one unattended attempt is spent; …", "needs_human": false, "owed": false}`. And
    BOTH kinds write the status pair as `from`/`TO`: `steward_move` at
    `workers/sources/board_steward.py:1606` and all three `status_moved` writers
    (`scripts/automod/backlog.py:5994`, `:6046`, `:7202`); `from_status`/`to_status` appear
    on neither. The cause that put the wrong spellings here was reading this module's
    `_event` — the disagreement filer's own writer — and assuming it described the rows
    this leg has to read. `test_the_pair_is_found_only_in_the_field_names_the_writers_
    actually_stamp` is the instrument: it builds one ledger in the real shape and one in
    the invented shape and requires the second to find nothing, so a reader that ever
    goes back to `at`/`to_status` reddens instead of going quiet.

    Missing ledger, unreadable ledger and a window with no `window_since` all give `[]`,
    the way `read_events` fails for those cases — a store that is not there is no evidence
    of a revert, so the alarm stays silent rather than firing on an outage. The read is
    whole-file (`limit=0`) for the reason `_window_pairs` gives: a last-400 read silently
    saw two days of a seven-day window and understated coverage, which is the failure
    #3651 was filed for. `ledger=None` is not a guess at a path — `read_events` resolves
    None to the production store, so the alarm cannot be pointed at a different ledger
    than the one the writers write. Stamps are compared through `_iso_norm`, because
    `state.now_iso()` writes a `Z` suffix and `judgement_aggregate` publishes a `+00:00`
    offset, and a string `<` across those two formats answers confidently and wrongly.
    """
    since = _iso_norm(str(window.get("window_since") or ""))
    if not since:
        return []
    try:
        from scripts.automod import state as S
        events = S.read_events(ledger, 0)       # limit=0 is the whole file
    except Exception:  # noqa: BLE001 — no store is no evidence; see the docstring
        return []
    reverts = [e for e in events
               if e.get("event") == EVENT_KIND_REVERT
               and _iso_norm(str(e.get("created_at") or "")) >= since
               and not _is_abstention_echo(e)]   # the machine agreeing, stamped as a move
    out: list[dict] = []

    def _item_of(ev: dict):
        # `item_id` is a plain int on both kinds, and a hand-edited or half-written row
        # without one is skipped rather than fatal — same rule `_ledger_rows` applies to
        # every predicate it runs over the store.
        try:
            return int(ev.get("item_id"))
        except (TypeError, ValueError):
            return None

    #: The item's LATEST in-window applied move, keyed by item, kept in the store's own
    #: order of first appearance so `below_floor`'s reason stays deterministic.
    latest: dict[int, dict] = {}
    for ev in events:
        if ev.get("event") != EVENT_KIND_APPLIED:
            continue
        item = _item_of(ev)
        applied = str(ev.get("to") or "")
        stamp = _iso_norm(str(ev.get("created_at") or ""))
        if item is None or not applied or stamp < since:
            continue                     # nothing this window can pair: no item, no status,
            # or an apply stamped before the window opened
        cur = latest.get(item)
        if cur is None or _supersedes(cur, ev):
            latest[item] = ev

    for item, ev in latest.items():
        applied = str(ev.get("to") or "")    # `to` on BOTH kinds; there is no `to_status`
        at = _iso_norm(str(ev.get("created_at") or ""))
        for ev2 in reverts:
            if _item_of(ev2) != item:
                continue
            if _iso_lt(_iso_norm(ev2.get("created_at")), at):
                continue                     # the machine moved BEFORE the steward applied
            to2 = str(ev2.get("to") or "")   # `to` on BOTH kinds; there is no `to_status`
            if not to2 or to2 == applied:
                continue                     # the machine agreed; nothing was put back
            out.append({"item_id": item, "applied": applied, "reverted_to": to2,
                        "at": str(ev2.get("created_at") or ""),
                        "reason": str(ev2.get("reason") or "")})
            break
    return out


def _iso_norm(s: str) -> str:
    """One comparable form for the two ISO shapes this pair is stamped in.

    `state.now_iso()` (both writers' `at`) ends in `Z`; the window bound comes from
    `judgement_aggregate`, which publishes a `+00:00` offset. Lexicographic order on the
    raw strings puts every `Z` row before every `+` row regardless of the date, so the
    window filter would keep nothing or everything depending on a character.
    """
    return str(s or "").strip().replace("Z", "+00:00")


def _iso_lt(a: str, b: str) -> bool:
    """Stamp compare, tolerant of the `Z` suffix and of a missing stamp.

    Both kinds come from the same `_event` helper today, so they share a format, but a
    `status_moved` row is written by a second producer (`scripts/automod/backlog.py`)
    across a process boundary, and a naive `<` on two different formats is a comparison
    that returns an answer either way. No stamp on the revert is no evidence of an
    order: not less-than, so the pair is left alone here rather than guessed at.
    """
    x, y = str(a or ""), str(b or "")
    if not x or not y:
        return False
    return x.replace("Z", "+00:00") < y.replace("Z", "+00:00")


def below_floor(window: dict, ledger: Path | None = None) -> tuple[bool, str, list[dict]]:
    """Is this window a MEASURED disagreement with the steward? (and if not, why, and
    which reverted pairs the verdict keyed on.)

    Two legs, and the second one is why this function survived the flip.

    The rate leg is the pre-flip one, both thresholds in the order owed-check's rule
    states them: without the volume leg the rate is not evidence yet (`truncated` says so
    too — a capped read counts whole ticks, so it understates both sums), and at or above
    the floor the steward was agreeing, which needs no item.

    The revert leg is the post-flip one, and it is what keeps this alarm measuring after
    `apply: true`: `agreement()` scores a `disagree` only on the tick that asks, so an
    applied wrong status is never re-proposed and can never arrive as a fresh `disagree` —
    see `reverted_moves`. It is evaluated AFTER the volume floor and the rate floor, on
    purpose (round-1 review finding 2): judged below the floor is not a measurement this
    alarm may dress up as disagreement, and a sub-floor rate already files on its own
    reading — so the leg can only ADD a firing, never re-word one, and every pre-flip
    reason string keeps its exact text. A CLEARED window with a revert in it is the case
    this alarm was kept alive for, and it fires with the item and both statuses named.
    Both legs silent means the steward applied nothing the machine argued with and is
    still agreeing: no item, and the reason says which half is quiet.

    The THIRD element is every pair `reverted_moves` returned for this window (#4514), handed
    to the caller so the filing can put them on its ledger row. A row's own `item_id` is the
    disagreement item and its `tick_item_ids` are the tick's disagree ids, so the item the leg
    keyed on was readable only by replaying the store — and the replay bounds BELOW only
    (`window_since`), so replaying an old firing today returns pairs stamped years later. The
    list is `[]` on every early return here, INCLUDING the sub-floor rate return that comes
    BEFORE the leg is evaluated, so an empty list reaching a row means "this leg never ran",
    never "nothing was put back"; `leg_from_trigger` on the reason is the discriminator. The
    reason text is unchanged by all of this — the trigger still names the ONE pair it named
    before, which is #4381's asymmetry and exactly why the whole list belongs on the row.
    """
    if not window or not window.get("judged"):
        return False, "nothing judged in the window", []
    if window.get("truncated"):
        return False, "window read truncated: sums understate, no verdict", []
    if int(window["judged"]) < FLIP_MIN_JUDGED:
        return False, f"volume leg unmet: judged {window['judged']} < {FLIP_MIN_JUDGED}", []
    rate = window.get("rate")
    if rate is None:
        return False, "rate is None: nothing judged, so nothing measured", []
    if float(rate) < FLIP_MIN_RATE:
        # The pre-flip leg is evaluated FIRST and still wins: a sub-floor rate files on its
        # own reading, so the post-flip leg below can only ADD a firing, never re-word one.
        return True, (f"judged {window['judged']} with rate {rate} under the "
                      f"{FLIP_MIN_RATE} floor"), []
    pairs = reverted_moves(window, ledger)
    if pairs:
        rev = pairs[0]
        # Built FROM `REVERT_TRIGGER_PREFIX`: the same string is what `leg_from_trigger`
        # keys on, and a drift between the two would mislabel the direction in the body,
        # the ledger row and the widening heading together. The rendered text is unchanged,
        # and it still names only this FIRST pair — the pair set goes to the caller, which
        # writes all of them onto the row (#4514).
        return True, (
            f"{REVERT_TRIGGER_PREFIX} to #{rev['item_id']} as `{rev['applied']}` was moved "
            f"back to `{rev['reverted_to']}` by the state machine on {rev['at']} "
            f"(reason: {rev['reason'] or 'unrecorded'}): the steward's applied move was "
            f"undone inside the window, which is the disagreement `rate` can no longer "
            f"score post-flip"), pairs
    return False, (f"rate {rate} is at or above the {FLIP_MIN_RATE} floor — a PRE-FLIP "
                   f"only figure post-flip; `reverted_moves` found nothing to undo"), []


def disagreement_rows(ids: list[int], moves: list[dict],
                      sides: dict[int, tuple]) -> list[str]:
    """One body line per disagreeing item id, naming **both** sides.

    The steward's side comes from its own parsed `moves` (what it asked for that id);
    the machine's side from `sides` — `{id: (status, why[, needs_human])}` **as the
    disagreeing tick recorded it**, which is `agreement()["disagree_sides"]` for this
    tick and `_window_sides`' rebuild of earlier ticks' for the ids a merge names
    (#3677). It is deliberately not a fresh `desired_statuses` read: the old call at
    `_widen` answered abstention for ids whose own tick had scored a conflict, and
    `agreement()` files an id as `disagree` only when the machine HAD an opinion
    (`workers/sources/board_steward.py:920`, abstention being the separate `no_opinion`
    bucket at `:921`) — so every one of those `(the machine abstains)` rows was a
    sentence the classifier had already ruled impossible. #3677 counted 12 of its own 22
    rows that way, and 2 more printed as agreement inside the disagreement list.

    "Recorded at the tick" is not "kept by luck": the same tick's `board_steward` ledger
    event carries `moves`, each with its `item_id`, `status` and the `note` the steward
    wrote, and now carries the machine's half in `agreement.disagree_sides`. The model's
    raw reply is kept in `runs.response_json` for any row a pool running `43cc6ba7`
    (2026-11-19) or later wrote; an older row is empty there, and any tick's reply is
    recoverable from its `meta_json.session_id` at `~/lloyd-data/sessions/<id>.json`.
    (`REPLY_PROVENANCE` holds this sentence; the node that reads this docstring compares
    them, so keep the two in step or the docstring goes red, not merely stale.)
    """
    proposed: dict[int, str] = {}
    for m in moves or []:
        iid = m.get("item_id")
        if isinstance(iid, int) and m.get("status"):
            proposed[iid] = str(m["status"])
    out: list[str] = []
    for iid in list(ids)[:ROWS_MAX]:
        want = sides.get(iid) if sides else None
        if want is None:
            # A tick predating `disagree_sides`. The honest reading is that the artifact
            # does not say, which is not the same claim as "the machine abstained" — and
            # an id this filer added from an old event may not have been a disagreement
            # at all, so abstention here would be a guess wearing the shape of a fact.
            out.append(f"- #{iid}: steward proposed `{proposed.get(iid, '(not in its moves)')}`"
                       f"; the state machine's side for this tick was **not recorded**")
            continue
        machine = str(want[0])
        why = ""
        if len(want) > 1 and want[1]:
            why = f" — {want[1]}"
        if len(want) > 2 and want[2]:
            why += " (needs a human)"
        out.append(f"- #{iid}: steward proposed `{proposed.get(iid, '(not in its moves)')}`"
                   f"; the state machine wants `{machine}`{why}")
    return out


#: The phrase `disagreement_rows` writes when a tick recorded a machine side. Recognised
#: by its exact text on purpose: the count that uses it is a claim about what the FILED
#: ROWS SAY, so it must be the same string the renderer emits and nothing else.
_RECORDED_SIDE = "; the state machine wants "


def _recorded_row_ids(body: str) -> set[int]:
    """Which ids a filed body already prints a recorded machine side for.

    The merge needs this because the merged Coverage figure has to be counted over the
    rows the body will hold, and half of those rows were written by an earlier filing
    whose `expected` is long gone (#3680, the sentence the review rung refused #3677
    for). The store is not consulted: an id's row in the file is the record of what that
    filing knew, and re-deriving it now is the exact mistake clause 2 retired.

    Only `_RECORDED_SIDE` counts. A legacy `(the machine abstains)` row and a
    `**not recorded**` row both carry nothing a ruler can rule against — the first was a
    filing-time recompute the tick never scored, the second says so out loud — so neither
    may inflate the covered count.
    """
    out: set[int] = set()
    for line in (body or "").splitlines():
        m = _ROW_RE.match(line)
        if m and _RECORDED_SIDE in line:
            out.add(int(m.group(1)))
    return out


def _window_pairs(window: dict, ledger: Path) -> dict[int, dict]:
    """`{item_id: {"tick", "proposed", "side"}}` — one disagreement as ONE tick scored it.

    The unit a ruling needs is a PAIR: the status the steward proposed and the status the
    machine wanted, both written by the same tick at the same instant. `_window_moves` and
    `_window_sides` used to walk the ledger separately and keep separate winners — newest
    tick for the proposal, oldest for the side — and the two winners were different ticks.
    #3752's measured case is #3433: the tick that disagreed about it
    (`2026-10-28T06:26:33Z`) proposed `draft` and recorded no side, and the row printed
    `steward proposed up_next` — the proposal of an AGREEING tick at `2026-11-03T23:43:23Z` —
    beside a machine side that agreed with it. An agreement printed inside a list of
    disagreements is the same class of lie as the abstentions #3677 was filed on, and
    pairing is what makes it unrepresentable: one traversal, one chosen tick per id, and
    `proposed` and `side` read out of that tick together.

    Which tick, per id: the NEWEST window tick that disagreed about the id AND recorded its
    side (`agreement.disagree_sides`, shipped by #3680 in `7fdb8c69`); if none recorded,
    the NEWEST window tick that disagreed, with `side` None. A recorded side outranks a
    newer unrecorded one because a tick that recorded left BOTH halves behind, and that is
    the row a reader can rule on — which is what
    `test_the_recorded_side_survives_the_ledger_line_the_filer_reads` holds: a later tick
    that disagreed and recorded nothing must not blank it.

    Which ticks count as disagreeing: an event whose `agreement` block is a dict counts
    ONLY for the ids that block's `disagree` names — an event that scored the id as `agree`
    is exactly the source this closes out. An event with no `agreement` block predates
    #1688 and cannot be attributed either way, so its moves are still offered (which is
    what #3651's recovery drives) with `side` None, and its row prints
    `(the machine's side for this tick was not recorded)` rather than an answer.

    An id with no move anywhere is absent, which is the honest answer: the tick that
    judged it wrote no `moves` row for it, so there is nothing to name as its proposed
    side, and `disagreement_rows` renders those as `(not in its moves)`.

    Missing ledger, unreadable ledger and an empty window all give `{}`. The read is
    whole-file (`limit=0`) because a seven-day window has 116 of its 185 steward events
    outside a last-400 read; a reader that silently saw two days of a seven-day window
    would understate coverage, which is the failure #3651 was filed for.
    """
    proposed: dict[int, dict] = {}        # newest disagreeing tick, side or not
    recorded: dict[int, dict] = {}        # newest disagreeing tick that recorded a side
    since = str(window.get("window_since") or "")
    if not since:
        return {}
    try:
        from scripts.automod import state as S
        events = S.read_events(ledger, 0)   # limit=0 is the whole file
    except Exception:
        # FileNotFoundError for a store that does not exist yet, and nothing here may
        # break a tick. The cost of swallowing is that a READER BUG lands here too: a
        # call into `state` for a function that was never there raises AttributeError,
        # is caught, and the window silently has no pairs — every row then reads
        # `(not in its moves)` and the filing still files. It happened while writing
        # #3651: `state.read_limit` does not exist, and only
        # `test_the_filing_names_every_window_id_whose_move_is_in_the_window` caught
        # the silent reader. Hence the warning rather than a bare `return`.
        logger.warning("disagreement widening read no pairs from %s", ledger,
                       exc_info=True)
        return {}
    for ev in events:
        if not isinstance(ev, dict) or ev.get("event") != "board_steward":
            continue
        created = str(ev.get("created_at") or "")
        if created < since:
            continue            # rows are appended in order, but this window is the bound
        block = ev.get("agreement")
        scored = isinstance(block, dict)
        disagreed: set[int] = set()
        sides: dict[int, tuple[str, str]] = {}
        if scored:
            disagreed = {i for i in (block.get("disagree") or [])
                         if isinstance(i, int) and not isinstance(i, bool)}
            for rec in (block.get("disagree_sides") or []):
                if not isinstance(rec, dict):
                    continue
                sid = rec.get("item_id")
                try:
                    sid = int(sid) if not isinstance(sid, bool) else None
                except (TypeError, ValueError):
                    continue    # an id the JSON round trip mangled is not this row's
                if sid is None or not rec.get("status"):
                    continue
                sides[sid] = (str(rec["status"]), str(rec.get("why") or ""))
        for m in (ev.get("moves") or []):
            iid = m.get("item_id") if isinstance(m, dict) else None
            if not isinstance(iid, int) or isinstance(iid, bool) or not m.get("status"):
                continue
            if scored and iid not in disagreed:
                continue        # this tick AGREED about it: not a side of a disagreement
            pair = {"tick": created, "proposed": str(m["status"]),
                    "side": sides.get(iid)}
            proposed[iid] = pair                # forward walk: newest candidate wins
            if pair["side"]:
                recorded[iid] = pair
    return {iid: recorded.get(iid, pair) for iid, pair in proposed.items()}


def _window_moves(window: dict, ledger: Path) -> dict[int, str]:
    """The proposals of the paired ticks: `{item_id: proposed_status}`.

    A view over :func:`_window_pairs`, kept under its own name because a proposal is what
    a row's first half is. It used to walk the ledger on its own and keep the NEWEST tick's
    proposal while `_window_sides` kept the OLDEST tick's side, and that split is what let
    one tick's proposal be printed beside another tick's answer (#3752).
    """
    return {iid: pair["proposed"]
            for iid, pair in _window_pairs(window, ledger).items() if pair["proposed"]}


def _window_sides(window: dict, ledger: Path) -> dict[int, tuple[str, str]]:
    """The machine sides of the paired ticks: `{item_id: (status, why)}`.

    The other view over :func:`_window_pairs`. Every entry is a side a tick that ALSO
    disagreed recorded — `agreement()["disagree_sides"]` writes it for exactly the ids in
    that tick's `disagree`, and the pairing above refuses a side from a tick that scored
    the id as `agree` — so a row can print one as fact instead of asserting abstention
    (#3677). An id whose disagreeing tick recorded nothing is absent, and renders
    **not recorded**.
    """
    return {iid: pair["side"]
            for iid, pair in _window_pairs(window, ledger).items() if pair["side"]}


#: The on-disk status that is terminal in intent but off-vocabulary in
#: `app/backlog_status.py`, where every non-terminal word maps to `draft`. A file
#: a person put out of reach for good is therefore invisible to `open_items`, so
#: a disagreement about it is one nobody can act on — and `CLOSED_STATUSES` alone
#: would not see it. Added here rather than to the host's vocabulary: that
#: mapping is deliberately lopsided the other way (#287).
ARCHIVED_STATUS = "archived"


def ruled_ids(ids: list[int], *, backlog_dir: Path | None = None,
              ledger: Path | None = None) -> dict[int, str]:
    """Which of these WINDOW ids the board has already answered, and why.

    #3914. The widening takes its ids from the window's per-tick history
    (`_window_pairs`) and nothing on that path asks whether the item still
    disagrees, so a filing keeps printing rows whose ruling landed days ago:
    item #3858 (filed 2026-11-06T17:03:51Z) and #3913 (2026-11-07T20:45:17Z) both
    still printed `- #3433: steward proposed draft; the state machine wants
    up_next` more than a day after commit `6119589c` resolved those four ids, and
    every re-filing spends a sweep and a triage pass on a question nobody is
    asking. `below_floor` cannot see this either: it reads only `judged`/`rate`,
    which the aggregate sums out of `runs.meta_json` with no board read at all, so
    the trigger stays armed until the sub-floor ticks age out by the clock.

    An id is ruled when EITHER live fact holds about it:

    * its item is off the board — `done`, the host's other terminal statuses, or
      `archived`. Deliberately NOT "anything outside `OPEN_STATUSES`": #287 has
      sat on disk as `review` since April, and a stranded item is a stuck one, not
      a ruled one. Keeping a stale row is the cheaper error, which is the same
      lopsided reading `app/backlog_status.py` documents for the machine's side.
    * its live on-disk status is what `desired_statuses` answers for it NOW. This
      is the arm that catches the four ids above: `6119589c` settled them by
      changing what the MACHINE answers (`draft` — its hand_run is still
      unexecuted), not by moving the board to the `up_next` the row recorded, so
      an equality against the RECORDED side leaves all four printing. Only the
      live table sees them.

    **This read may only ever REMOVE an id.** It answers nothing about a row's
    content: both halves of every surviving row still come from the tick that
    scored them, exactly as #3677 left them, and no status or reason from the live
    table reaches the body — which is why the result is ids with a reason phrase
    and not a table of sides. That is the whole difference from the filing-time
    recompute #3677 removed: that one SUPPLIED halves about an instant that had not
    scored the id; this one only asks a yes/no of the same instant and can lose a
    row, never invent one.

    Fails OPEN — a board or ledger that cannot be read returns `{}` and every row
    prints, which is this file's behaviour before #3914: a filing that silently
    loses rows is worse than one that re-names a settled id. Same stance as
    `moved_item_ids` (`workers/sources/board_steward.py:1242`), which #3752 shipped
    for a board that moved MID-tick; this is the between-ticks half of the same
    fact, and it is unreachable from there because that guard runs inside a tick's
    own scoring and never sees the window.
    """
    if not ids:
        return {}
    from app.backlog_status import CLOSED_STATUSES
    from scripts.automod import state as S
    B = _backlog()
    try:
        on_disk = {int(i.id): str(i.status or "")
                   for i in B.all_items((BOARD,), backlog_dir=backlog_dir)}
        want = B.desired_statuses(Path(ledger) if ledger else S.LEDGER_PATH) or {}
    except Exception as exc:                       # fail OPEN, per the docstring
        logger.warning("disagreement abstention read failed (%s: %s); printing every "
                       "window row", type(exc).__name__, str(exc)[:200])
        return {}
    closed = CLOSED_STATUSES | {ARCHIVED_STATUS}
    out: dict[int, str] = {}
    for iid in {int(i) for i in ids}:
        status = on_disk.get(iid)
        if status is None:
            # No board file at all: the live read proves nothing in either
            # direction, and an id with nothing to remove cannot be a phantom.
            continue
        if status in closed:
            out[iid] = f"its item is off the board (`{status}`)"
            continue
        entry = want.get(iid)
        if entry and str(entry[0]) == status:
            out[iid] = (f"its status on disk (`{status}`) is what the state machine "
                        f"answers for it now")
    return out


#: The event the persistence guard below reads: the tick's own ledger row.
_STEWARD_EVENT = "board_steward"


def _block_disagree_ids(agree: dict) -> list[int]:
    """The ids one tick's `agreement` block scored as disagreements.

    `disagree` is the published list; `disagree_sides` (#3680) is the same ids with the
    machine's half attached. A row can carry one and not the other — the list is what
    `agreement()` builds first — so the record is the fallback, never the reverse.
    """
    ids = [i for i in (agree.get("disagree") or []) if isinstance(i, int)]
    if ids:
        return ids
    sides = agree.get("disagree_sides")
    if isinstance(sides, list):
        return [s.get("item_id") for s in sides
                if isinstance(s, dict) and isinstance(s.get("item_id"), int)]
    return []


def prior_tick_disagree(ledger: Path | None = None,
                        session_id: str | None = None) -> set[int] | None:
    """The disagreement of the steward tick BEFORE this one; None where there is nothing to compare.

    #4378's bar, and the reason for it: one tick's disagreement is ONE reading of ONE board
    by ONE model, and it was already enough to put a row on the lloyd board and a question to
    owed-check. Walked 2026-11-24 over the live ledger's 16,632 rows — each of its 270
    `board_steward` events compared with ITS OWN immediately-preceding steward row, as this
    function does at filing time — 17 ticks scored a non-empty disagreement, naming 21
    id-ticks over 14 ids: 4 id-ticks earn (all of them #4276, disagreeing twice in a row), 1
    is withheld (#4548, new against its preceding tick), and 16 fall through to the
    no-comparison shape below because the steward tick before them recorded no disagreement
    at all. That ratio is the honest size of this guard on current traffic, and it is the
    clause-3 fallback doing the work: the bar bites when the tick before this one was also
    scoring, which is what "consecutive" was ruled to mean.

    The reader is the store the tick already writes, so no second copy of the answer exists
    to disagree with the first: the `board_steward` event carries the tick's own `agreement`
    block, whose `disagree` is the id list and whose `disagree_sides` is the machine's half
    (#3680). `_window_pairs` reads that same block for the same purpose.

    Three things about the read, each already learned elsewhere in this file:
      - `limit=0`, the whole file, not `read_events`' default 100. The ledger holds far more
        events than steward ticks, so a bounded newest-N window can end before the previous
        tick and there is nothing to compare against — the trap #3651 documents at
        `_window_pairs`, where the same fix is applied.
      - THIS tick's row is excluded by `session_id`, not by position, because the tick
        appends its own `board_steward` event (`workers/sources/board_steward.py:1903`) and
        files only afterwards. Read by position the newest row IS the current tick, so the
        comparison would be the tick against its own answer and every id would pass. A row
        whose session cannot be read is never taken for the current one: an unreadable head
        makes the row eligible, it does not end the filing.
      - It consults exactly ONE row: the steward tick immediately before this one. Walking
        back to the last tick that DID record a disagreement would score "disagreed twice,
        with ten agreeing ticks in between" as persistence, and the ruling is about a
        CONSECUTIVE pair — #4240 disagreed on two ticks 26 minutes apart with an agreeing tick
        between them, and the item rules that withheld. So a preceding row whose `agreement`
        block is missing (it predates #1688), is not a dict, or names no disagreement returns
        None: there is nothing to compare, and the filing goes unguarded — the shape that
        files this tick exactly as before #4378, because a guard that cannot answer must not
        also mute the alarm.
    """
    from scripts.automod import state as S   # function-local: see the module docstring
    try:
        events = S.read_events(ledger, 0)   # limit=0 is the whole file
    except Exception as exc:  # noqa: BLE001 — the rail `_window_pairs` documents verbatim
        logger.warning("board-steward persistence guard: reading %s failed (%s: %s); the "
                       "filing proceeds unguarded, which files this tick as it did before "
                       "the guard existed", ledger or S.LEDGER_PATH,
                       type(exc).__name__, exc)
        return None

    for ev in reversed(events):            # newest first: the tick immediately before this
        if not isinstance(ev, dict) or ev.get("event") != _STEWARD_EVENT:
            continue
        if session_id is not None and ev.get("session_id") == session_id:
            continue                       # this tick's own row: never its own evidence
        agree = ev.get("agreement")
        if not isinstance(agree, dict):
            # The row is the preceding tick and it records nothing: an event from before
            # #1688, or one a diverged or dead turn left without a block. No comparison
            # exists, so the filing is today's — and NOTHING is looked for further back.
            return None
        ids = _block_disagree_ids(agree)
        if not ids:
            # A preceding tick that named no disagreement — it agreed, or judged nothing — is
            # not evidence that anything persisted, and clause 3 rules the fallback for it:
            # this tick files as it did before the guard. It is never read as "disagreed
            # about nothing", which would withhold every id this tick just named.
            if isinstance(ev.get("moves"), list) and ev["moves"]:
                # The same absence inside a block that should have named the tick's own
                # disagreements: said out loud, because a broken record and a tick that
                # agreed read the same way to the filing and only one of them is a fault.
                logger.warning("board-steward persistence guard: the immediately preceding "
                               "tick's agreement block names no disagreement while its "
                               "`moves` propose %d; its side cannot be recovered, so this "
                               "filing proceeds unguarded", len(ev["moves"]))
        return set(ids) if ids else None
    return None


def _widen(agree: dict, moves: list[dict], expected: dict[int, tuple],
           window: dict, ledger: Path) -> tuple[list[int], list[dict], dict]:
    """The rows a ruling can actually be made on: the window's, not the tick's.

    Returns `(ids, moves, expected)`. `ids` is the tick's disagree set (unchanged,
    in order, first) followed by every other id the window's aggregate names.

    Where each half of a row comes from, and it is the whole design of this function:
      - an id THIS tick scored as a disagreement: its proposal is this tick's move and its
        side is this tick's `disagree_sides`, falling back to `expected` — the same object
        the tick scored against, so it is that tick's answer and not a later reading;
      - any other id: BOTH halves from the one window tick `_window_pairs` chose, or the
        row prints `(not in its moves)` / `(the machine's side for this tick was not
        recorded)`.
    `expected` is never used for the second group. That seed is what made #3433's row
    print a machine side — `('up_next', 'triage confirmed it with an acceptance check')`,
    the filing tick's live `desired_statuses` for an id it had not judged — beside an
    `up_next` proposal lifted off an agreeing tick, an agreement inside the disagreement
    list. #3677 removed the re-run of `desired_statuses` for window ids and #3680 shipped
    the per-tick record; this removes the last way a row can speak from an instant that
    did not score it.

    The last loop drops any pair whose two halves name the SAME status. A pair like that
    cannot come from an honest tick — `disagree_sides` is written for exactly the ids whose
    status differs from the proposal — so it means the halves came from different places,
    and rendering it is the lie the clause forbids. The row stays named and renders
    **not recorded**, which is also how Coverage counts it.

    `stats`, from `row_stats`, carries what the widening could NOT do for the Coverage
    bullet: how many named ids have no proposal in any ledger move, and how many ids the
    row cap dropped.
    """
    tick_ids = [int(i) for i in (agree.get("disagree") or []) if isinstance(i, int)]
    win_ids = [i for i in (window.get("disagree_ids") or []) if isinstance(i, int)]
    ids = tick_ids + [i for i in win_ids if i not in set(tick_ids)]

    pairs = _window_pairs(window, ledger)
    proposed = {m.get("item_id"): m.get("status") for m in (moves or []) if isinstance(m, dict)}

    wide_moves: dict[int, Any] = dict.fromkeys(ids)   # None where nothing proposed it
    # This tick's move is authoritative for the ids IT scored as disagreements, and for
    # no others: this tick's `moves` list also carries every item it happened to agree
    # with, and putting one of those statuses in a window id's row is half of #3433's lie.
    for iid in tick_ids:
        status = proposed.get(iid)
        if isinstance(status, str) and status:
            wide_moves[iid] = status
    if any(v is None for v in wide_moves.values()):
        for iid, pair in pairs.items():
            if iid in wide_moves and wide_moves[iid] is None and pair["proposed"]:
                wide_moves[iid] = pair["proposed"]

    # The machine's halves, each one from the tick that scored it. For an id this tick
    # judged, its own record wins even over a ledger pair — that pair can only be an
    # EARLIER tick's disagreement about the same item, and the row being written now is
    # this tick's.
    wide_expected: dict[int, tuple[str, str]] = {
        iid: pair["side"] for iid, pair in pairs.items() if pair["side"]}
    recorded_sides = {}
    for side in (agree.get("disagree_sides") or []):
        if not isinstance(side, dict) or not isinstance(side.get("item_id"), int):
            continue
        if not side.get("status"):
            continue
        recorded_sides[side["item_id"]] = (str(side["status"]), str(side.get("why") or ""))
    for iid in tick_ids:
        side = recorded_sides.get(iid)
        if side is None:
            # The same table the id was scored against, held by reference from the top of
            # the tick — the tick's own answer, which is what `disagree_sides` would have
            # written had this tick been running the build that records it (#3677). Not a
            # re-read: `execute` reads `desired_statuses` once and passes it down.
            entry = (expected or {}).get(iid)
            side = ((str(entry[0]), str(entry[1]) if len(entry) > 1 else "")
                    if entry else None)
        if side:
            wide_expected[iid] = side

    # A pair whose two halves name the same status is not a disagreement, so it gets the
    # rendering of a side nobody recorded rather than an agreement printed inside the list.
    for iid in [k for k, want in wide_expected.items()
                if want[0] and wide_moves.get(k) == want[0]]:
        wide_expected.pop(iid)

    rows = [{"item_id": i, "status": s} for i, s in wide_moves.items() if s]
    return ids, rows, wide_expected


def row_stats(ids: list[int], rows: list[dict], window: dict) -> dict:
    """What a set of rows could not say, in the three figures Coverage reports separately.

    `unrecoverable` counts the NAMED ids with no proposal in any move; `unruled` counts the
    same fault among the ids the row cap dropped, reported apart because those ids are not
    in the body at all and an id counted in both figures would be reporting the same gap
    twice. `distinct` is None whenever the window gives no id list — the count alone cannot
    tell ids from rows, and a missing key is a missing key.
    """
    have = {m.get("item_id") for m in (rows or [])}
    named = min(len(ids), ROWS_MAX)
    win = [i for i in (window.get("disagree_ids") or []) if isinstance(i, int)]
    return {"unrecoverable": sum(1 for i in ids[:named] if i not in have),
            "unruled": sum(1 for i in ids[named:] if i not in have),
            "dropped": len(ids) - named,
            "distinct": len(win) if "disagree_ids" in window else None}


def file_or_update(agree: dict, moves: list[dict], expected: dict[int, tuple], *,
                   window: dict, backlog_dir: Path | None = None,
                   ledger: Path | None = None,
                   rules_note: str = "",
                   session_id: str | None = None) -> dict | None:
    """File or add to the ONE open disagreement item; return what happened.

    `window` is `judgement_aggregate`'s dict for the same 7 days the health route
    serves, passed in rather than looked up here so the number that triggers the
    filing and the number on the route cannot be two readings of two windows.

    `ledger` is the promotions store BOTH sides read: the trigger's post-flip leg and the
    `board_steward_disagreement` rows this writes. It is passed to `below_floor` for
    exactly that reason — a leg that read the default store while this wrote a named one
    would let a test drive one file and the trigger another, which is the seam this
    module keeps being filed over. None means the production store, resolved by
    `read_events` itself.

    `session_id` is the filing tick's OWN session, which its `board_steward` ledger row
    already carries. The tick appends that row and files afterwards, so it is how the
    persistence guard (#4378, `prior_tick_disagree`) tells the tick's own answer apart from
    the tick before it. None means the caller did not name itself: the newest steward row is
    then read as the preceding tick, which for a live tick is its own row — the guard is
    consulted, passes everything, and the filing is today's. `_file_quietly` forwards it.

    `rules_note` is the block's OWN line about the flag (e.g. "`apply: true` since
    2026-11-16T00:06Z") carried down onto the created item's body, so a reader of the
    alarm can see whether the flip was live when it fired — the reading that decides
    `config.yaml`'s 7-day stop condition. Absent (the pre-flip default) prints `absent`.

    Returns `None` when nothing was filed (bar met, volume leg unmet, nothing
    disagreed, an open item a round is working, or every id this tick named being
    withheld by the persistence guard as not yet earned — which writes no item, no body
    section and no `board_steward_disagreement` row). Never raises on a missing
    store; raises on a real write failure, and its caller swallows — see
    `_file_quietly`.
    """
    tick_ids = sorted({int(i) for i in (agree.get("disagree") or []) if isinstance(i, int)})
    if not tick_ids:
        return None
    should, reason, rev_pairs = below_floor(window, ledger)
    if not should:
        logger.debug("disagreement filing skipped: %s", reason)
        return None

    # The window's ids, not this tick's. A filing that named one tick's rows was a report on
    # the tick that happened to trip the trigger while the figure beside it (`judged`,
    # `rate`) was the seven-day one, so a ruler was handed 1 of 25 rows and told to judge a
    # week (#3651). `agree`/`moves`/`expected` in scope here are this tick's board subset
    # only; `_widen` pulls the rest from the store the tick wrote to: the window's
    # `board_steward` ledger moves for the steward's half, and the same events' recorded
    # machine sides for the other. NOT `desired_statuses` — that is the board's state NOW,
    # and a row about a tick cannot be answered from a later instant (#3677).
    # This reads; it does not write, so widening cannot settle a disagreement by moving it.
    from scripts.automod import state as S
    led = Path(ledger) if ledger else S.LEDGER_PATH
    ids, wide_moves, wide_expected = _widen(agree, moves, expected, window, led)
    if not ids:
        return None

    # An id the WINDOW remembers that the board has already answered is not an open
    # question, and printing it as one is what re-files the same eight ids across six
    # consecutive items (#3752 … #3913). See `ruled_ids`: the live read may only REMOVE,
    # so it is applied to the widened ids ALONE — `tick_ids` are the ids THIS tick
    # scored as disagreeing, and dropping one of those would be the filer disagreeing
    # with its own measurement, which is #3752's job inside the tick and nobody's here.
    # What it removes is reported in Coverage, never subtracted in silence.
    abstained = ruled_ids([i for i in ids if i not in set(tick_ids)],
                          backlog_dir=backlog_dir, ledger=led)
    if abstained:
        # Empties nothing on its own: `tick_ids` is non-empty past its own leg above and is
        # never a candidate here, so at least this tick's disagreement survives the
        # abstention. It does NOT survive the persistence guard below, which can withhold
        # every id this tick named — that is the one path that can leave `ids` empty, and it
        # returns `None` rather than filing an empty body (#4378).
        ids = [i for i in ids if i not in abstained]

    # #4378: two consecutive disagreeing ticks, not one, earn a row. Applied AFTER the
    # widening above and to `ids` as a whole — both the tick's ids and the window's — because
    # the bar is "did the preceding tick disagree about this id", which is exactly as true of
    # an id the window remembers as of one this tick just named. A gate on the tick's ids
    # alone would leave every single-tick window row in place and not achieve the bar at all.
    # `prior_tick_disagree` returns None where there is nothing to compare (no steward row,
    # no `agreement` block, or a block that names no disagreement), and then nothing is
    # withheld: a guard that cannot answer files this tick as it did before #4378 rather than
    # muting the instrument that decides the flip.
    prior = prior_tick_disagree(led, session_id=session_id)
    withheld: list[int] = []
    if prior is not None:
        withheld = [i for i in ids if i not in prior]
        ids = [i for i in ids if i in prior]
        if withheld and not ids:
            # Everything this tick named is a single tick's reading. There is no row to
            # write, so there is nothing to merge into an open item either: a body section
            # naming no id and a ledger row naming no `new_item_ids` would be an event that
            # reads as a filing and decides nothing.
            logger.info("board-steward persistence guard (#4378): %s was each scored a "
                        "disagreement by this tick and by none before it, so nothing is "
                        "filed and no %s row is written", withheld, EVENT)
            return None
    stats = row_stats(ids, wide_moves, window)
    # Of the rows this filing actually WRITES, how many carry the machine's side as the
    # disagreeing tick recorded it. Ruleability needs both halves of a disagreement, and
    # `unrecoverable` counts only the steward's: #3677's Coverage bullet claimed "every
    # row here can be ruled from the item alone" on the strength of proposals alone,
    # while 12 of its 22 rows had no machine side to rule against.
    stats["recorded"] = sum(1 for i in ids[:min(len(ids), ROWS_MAX)]
                            if i in wide_expected)
    # Carried as the ids + reasons themselves, not a count: Coverage names them, and a
    # figure without the names is exactly the silent subtraction this filing swears
    # against — a reader cannot tell "the ruling landed" from "the read failed".
    stats["abstained"] = abstained
    # Named individually like the abstentions beside them: the count says how many rows the
    # guard withheld, the ids say WHICH questions are waiting for their second tick, and a
    # bare count is the silent subtraction this filing swears against.
    stats["withheld"] = withheld

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
        # `recorded` is withheld: it counts `ids`, and the merge publishes it beside a
        # denominator counting the whole file. `_merge` derives the merged figure from
        # the rows the body will hold (#3680); the create path keeps this one, whose
        # population and denominator are the same list by construction.
        return _merge(openers[0], B, ids, wide_moves, wide_expected, window, stamp,
                      backlog_dir=backlog_dir, ledger=ledger, trigger=reason,
                      revert_pairs=rev_pairs,
                      **{k: v for k, v in stats.items() if k != "recorded"})
    return _create(B, ids, wide_moves, wide_expected, window, stamp,
                   backlog_dir=backlog_dir, ledger=ledger, rules_note=rules_note,
                   trigger=reason, revert_pairs=rev_pairs, **stats)


def _coverage(named: int, window: dict, *, merged: bool = False, unrecoverable: int = 0,
              dropped: int = 0, distinct: int | None = None, unruled: int = 0,
              recorded: int | None = None,
              abstained: dict[int, str] | None = None,
              withheld: list[int] | None = None) -> str:
    """What these rows do NOT cover, in numbers.

    Before #3651 the ids were the CURRENT tick's (`file_or_update` took them from
    `agreement()["disagree"]`, one tick) while every figure the body otherwise cites is
    the rolling window's: the item filed at 2026-10-29T17:27:13Z named 2 rows against a
    window holding 28 disagreements over 25 distinct items. The ids are the window's now,
    so what is left unreported is a different set of things, and each gets its own figure
    rather than one subtraction that would blur them:

    * **rows vs ids** — `disagree` is a sum of per-tick lengths, so one item that
      disagreed in three ticks is three rows and one ruling; `distinct` says how many
      items the window actually names.
    * **unrecoverable** — ids whose proposed side is in no ledger move, so no row can say
      what the steward asked for and no ruling can be made from the item alone.
    * **dropped** — ids past `ROWS_MAX`, named nowhere in the body at all.
    * **abstained** (#3914) — ids the window still remembers whose ruling has already
      landed, so no row is written for them. Named individually, because this figure
      moves `un` in the line above: without the names a reader cannot tell a question
      that stopped being open from a read that failed, which is the difference between
      a filing that abstains and one that silently loses rows.
    * **withheld** (#4378) — ids scored a disagreement that have NOT yet earned a row,
      because the steward tick immediately before this one did not also score them as one.
      Named individually for the same reason as the abstentions, and kept in its own line
      because the two are different states: an abstained id's ruling has landed and will
      never be written, a withheld id is one more disagreeing tick from being written.

    State the zero for each. An item that covers its window has to say so, or a reader
    cannot tell a genuinely small case from an unstated large one — and a merge moves both
    sides of every subtraction, so all of it is re-derived, never kept.
    """
    total = int(window.get("disagree") or 0)
    un = max(0, total - int(named))
    if merged:
        first = (f"- this item now names {named} of the window's {total} disagreement "
                 f"rows; {un} are not represented here.")
    elif un:
        first = (f"- **{un} of the window's {total} disagreement rows are not "
                 f"represented here** (rows and ids differ: `disagree` counts one item "
                 f"once per tick it disagreed in, and a ruling is per item).")
    else:
        first = (f"- all {named} of the window's {total} disagreement rows are "
                 f"represented here; nothing in the window is unruled by omission.")
    # First of the extras: this figure moves the `un` in `first`, so the reader has to be
    # told which part of that gap is a row nobody could recover and which is a question
    # that stopped being open (#3914). Zero is stated, like every other figure here.
    ab = {int(k): str(v) for k, v in (abstained or {}).items()}
    if ab:
        extra = ("- " + str(len(ab)) + " of the window's ids were **abstained, not "
                 "dropped**: their ruling has already landed — "
                 + "; ".join(f"#{i} ({ab[i]})" for i in sorted(ab))
                 + ". A live board read removed these rows; it supplied neither half of "
                   "any row above. `judged` and `rate` still count the ticks these ids "
                   "were scored on, and whether they should is owed-check's to rule "
                   "(#3089 asked, #3329 set the bar), not this filing's.\n")
    else:
        extra = "- 0 of the window's ids were abstained as already ruled.\n"
    # The persistence guard's share, in the same shape as the abstentions beside it and with
    # its zero stated: a reader has to be able to tell "nothing was withheld" from
    # "the guard never ran", which is why the line is printed unconditionally.
    wh = [int(i) for i in (withheld or [])]
    if wh:
        extra += ("- " + str(len(wh)) + " of this tick's ids were withheld as "
                  "**not-yet-earned**: the steward tick immediately before this one did not "
                  "score them as a disagreement, so #4378 holds them back until one does — "
                  + ", ".join(f"#{i}" for i in sorted(wh))
                  + ". `judged` and `rate` still count the ticks they were scored on; only "
                    "the row waits.\n")
    else:
        extra += "- 0 of this tick's ids were withheld as **not-yet-earned**.\n"
    if distinct is not None:
        extra += (f"- the window's {total} disagreement rows are {distinct} distinct "
                  f"items, and this item names {int(named)} of them: a row is a tick's "
                  f"disagreement and a ruling is per item, so one item disagreeing in two "
                  f"ticks is two rows and one decision.\n")
    if unrecoverable:
        extra += (f"- {int(unrecoverable)} of the named ids carry `(not in its moves)`: no "
                  f"`board_steward` event in the window proposed a status for them, so their "
                  f"steward side is unrecoverable from the store and a ruling on those ids "
                  f"needs the tick's session, not this item.\n")
    else:
        extra += ("- 0 of the named ids are unrecoverable: every one carries the status "
                  "a `board_steward` event in this window proposed" +
                  (" for its steward half" if recorded is not None else
                   ", so every row here can be ruled from the item alone") + ".\n")
    if recorded is not None:
        # The other half of the same claim, named as its own number because the two are
        # not the same count: an id can have its proposal in the window and still have no
        # recorded machine side, which is the one state a row cannot be ruled in.
        shortfall = int(named) - int(recorded)
        extra += (f"- {int(recorded)} of the {int(named)} named rows carry the machine's "
                  "side as its own tick recorded it" +
                  (f", and {shortfall} do not: those ticks predate the record, so their "
                   "rows say **not recorded** and only the steward's half of them can be "
                   "ruled from the item." if shortfall else
                   ". Both halves of every row are in the artifact, so every row here can "
                   "be ruled from the item alone.") + "\n")
    if dropped:
        extra += (f"- **the rows are capped at {ROWS_MAX}, so {int(named)} of "
                  f"{int(named) + int(dropped)} named ids are written and {int(dropped)} "
                  f"are not**: {int(unruled)} of those {int(dropped)} are also unrecoverable "
                  f"from the store. The cap is what keeps one merge cycle from writing a body "
                  f"nobody reads; the truncation is stated here with both figures rather than "
                  f"the ids being silently dropped, and the next tick re-derives the whole "
                  f"window, so nothing is lost permanently.\n")
    return ("## Coverage\n\n" + first + "\n" + extra +
            f"- the window itself: {window.get('judged')} judged, `rate="
            f"{window.get('rate')}` against owed-check's {FLIP_MIN_RATE} floor, since "
            f"{window.get('window_since')}.\n\n")


def _create(B: Any, ids: list[int], moves: list[dict], expected: dict[int, tuple],
            window: dict, stamp: str, *, backlog_dir: Path | None,
            ledger: Path | None, unrecoverable: int = 0, dropped: int = 0,
            distinct: int | None = None, unruled: int = 0,
            recorded: int | None = None, rules_note: str = "",
            trigger: str = "",
            revert_pairs: list[dict] | None = None,
            abstained: dict[int, str] | None = None,
            withheld: list[int] | None = None) -> dict:
    rows = disagreement_rows(ids, moves, expected)
    # The direction sentence, derived (#4251). This used to be one unconditional f-string
    # declaring the window "measured **below** owed-check's flip floor", which is FALSE of
    # every revert-leg filing: `below_floor` reaches that leg only after the volume floor and
    # the rate floor have both PASSED, so #4184 (`rate` 0.968) and #4247 (`rate` 0.9691) each
    # printed "below … a floor of 0.9" against their own above-floor figures, while the ledger
    # row written in the same call said `post-flip revert`. Only the direction moves: the
    # volume clause after this sentence is true on both legs, and the rate leg's bytes are
    # untouched — `tests/test_board_steward.py` compares the sub-floor header against the
    # exact text this file shipped at HEAD.
    leg = leg_from_trigger(trigger)
    if leg == LEG_REVERT:
        head = ("tick whose trailing-7-day judgement **cleared both** of owed-check's flip "
                "floors — the volume leg and the agree-rate leg —")
        fired = (f"What fired is the **{LEG_REVERT}** leg: inside that window the state "
                 f"machine moved back a status the steward had applied, which is the "
                 f"disagreement the agree `rate` can no longer score post-flip. ")
    else:
        head = ("tick whose trailing-7-day judgement was measured **below** owed-check's "
                "flip floor:")
        fired = ""
    # Where the raw reply lives, then the caveat that the runs `source` is hyphenated:
    # the underscored spelling is the ledger's event kind, so a reader who filters `runs`
    # with it sees 0 rows and concludes the store keeps nothing when it keeps everything.
    # `REPLY_PROVENANCE` is the one sentence the `disagreement_rows` docstring and
    # `architecture/workers-jobs.md` carry too, so the three cannot disagree (#4387).
    reply_here = REPLY_PROVENANCE + (
        " A `runs` lookup for the ticks these rows come from needs the hyphenated "
        "`source`: `board-steward` is the runs spelling, while the underscored "
        "`board_steward` is the ledger's event kind and matches no runs row at all.")
    body = (
        f"# {NAME}\n\n"
        f"Filed by `workers/board_disagreement.py` at the end of a board-steward "
        f"{head} `judged={window.get('judged')}` (the leg it needs is "
        f"{FLIP_MIN_JUDGED}) with agree `rate={window.get('rate')}` against a floor of "
        f"{FLIP_MIN_RATE}, over the window since {window.get('window_since')}. {fired}"
        f"The volume leg is cleared, so this is a measured disagreement and not an "
        f"unmeasured one — which is the case nothing used to read.\n\n"
        f"Each row names BOTH sides of one disagreement: what the "
        f"steward proposed for that item, and what `desired_statuses` "
        f"(`scripts/automod/backlog.py`) wants, with its reason. Both sides outlive the "
        f"tick elsewhere: the proposal itself is in the ledger's `board_steward` event "
        f"for that tick, whose `moves` list carries the `item_id`, the `status` and the "
        f"`note` the steward wrote for it, and the machine's side is in that same event's "
        f"`agreement.disagree_sides`, recorded by the tick that scored it (#3677). "
        f"{reply_here}\n\n"
        f"## Rows\n\n" + "\n".join(rows) + "\n\n"
        + _coverage(min(len(ids), ROWS_MAX), window, withheld=withheld,
                    unrecoverable=unrecoverable,
                    dropped=dropped, distinct=distinct, unruled=unruled,
                    recorded=recorded, abstained=abstained) +
        f"## What to decide\n\n"
        f"Per item, rule which side is wrong:\n\n"
        f"- the **steward's** prompt/grounding (`workers/sources/board_steward.py`) — then "
        f"say what it should have proposed, and its prompt or grounding changes; or\n"
        f"- the **state machine** (`scripts/automod/backlog.py::desired_statuses`) — then "
        f"the machine's rule changes.\n\n"
        f"`apply` does NOT turn on this item. The bar that turned it on — owed-check's "
        f"(`judged >= {FLIP_MIN_JUDGED}` AND `rate >= {FLIP_MIN_RATE}` over the rolling 7 "
        f"days, #3089 asked / #3329 ruled) — is a PRE-FLIP measure: after the flip the "
        f"rate stops scoring agreement, so this alarm's post-flip leg "
        f"(`reverted_moves`) is what measures the steward. This filing never edits a "
        f"status: it is filed `draft` and left for a judgment.\n"
        f"\n"
        f"**Flag state when this was filed**: {rules_note or 'absent'} — the value and "
        f"the moment from `config.yaml`'s own board-steward block, carried here because "
        f"the stop condition that reverses the flip is watched from the flip's minute, "
        f"not from this item's.\n"
        f"\n"
        f"**What tripped it**: {trigger or 'absent'} — which leg of `below_floor` fired, "
        f"in that function's own words. It is on the body because the two legs are read "
        f"differently: a rate-leg filing is an opinion about a window that has ALREADY "
        f"aged into the pre-flip measure, and a revert-leg filing is a specific applied "
        f"move the machine put back — the second is the one the 7-day stop condition is "
        f"watched for, and a reader cannot tell them apart from the counts alone.\n"
    )
    # `new_item` (host:scripts/automod/backlog.py:8488) returns the loaded Item and
    # takes no activity line — the create's record is this body plus the ledger
    # row below, which is exactly one write and no second chance to fail.
    it = B.new_item(NAME, body, priority="medium", tags=[TAG], status="draft",
                    board=BOARD, backlog_dir=backlog_dir)
    _event(ledger, item_id=it.id, action="filed", ids=ids, window=window,
           stamp=stamp, new_ids=ids, trigger=trigger, revert_pairs=revert_pairs)
    return {"action": "filed", "item_id": it.id, "path": str(it.path),
            "ids": ids, "rows": len(rows)}


def _strip_coverage(body: str) -> str:
    """Drop the `## Coverage` section, which `_coverage` re-derives on every write.

    A merge moves both sides of the subtraction — the row count grew and the
    window may have moved — so keeping the filed body's line beside the new one is
    the exact unreadable the section exists to prevent: "#21 are not represented"
    and "#6 of 26 now named" in one file, with nothing saying which is current.
    Heading-scoped, because the section is prose and its own bounds are its
    headings; a row of `## Rows` is untouched, and so is an `### Added` section,
    which is a sub-heading.
    """
    out: list[str] = []
    skipping = False
    for line in (body or "").splitlines():
        if line.startswith("## Coverage"):
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def _merge(item: Any, B: Any, ids: list[int], moves: list[dict],
           expected: dict[int, tuple], window: dict, stamp: str, *,
           backlog_dir: Path | None, ledger: Path | None, trigger: str = "",
           unrecoverable: int = 0,
           dropped: int = 0, distinct: int | None = None,
           revert_pairs: list[dict] | None = None,
           unruled: int = 0, abstained: dict[int, str] | None = None,
           withheld: list[int] | None = None) -> dict | None:
    """Add the ids this item does not name yet, into the same file.

    `recorded` is deliberately NOT a parameter: the caller's figure counts the ids THIS
    filing widened, and a merge publishes it beside a denominator that counts the whole
    file. #3680 refused #3677's round on that mismatch — the bullet wrote "0 of the 6
    named rows carry the machine's side" over a body whose five earlier rows each print
    one — so the count is recomputed here, over `pop`, the same list the denominator is
    measured from.

    `revert_pairs` is CARRIED through to the ledger row, never re-derived here (#4514). It is
    the pair set the trigger fired on — a fact about the window at the instant this filing was
    decided — and `reverted_moves` bounds only BELOW by `window_since`, so re-reading the
    store from inside a widening would append pairs the machine put back AFTER the trigger
    spoke and credit this firing with reverts it never saw.
    """
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
    # The population the merged body names, in the order it names it: the rows already
    # filed, then this filing's. `len(pop) == len(seen | set(fresh))`, so the denominator
    # is the number it always was — what changed is that the numerator is now measured
    # over this same list (#3680).
    pop: list[int] = []
    for m in _ROW_RE.findall(body or ""):
        if (iid := int(m)) not in pop:
            pop.append(iid)
    pop += [i for i in fresh if i not in pop]
    named = min(len(pop), ROWS_MAX)
    carry = _recorded_row_ids(body)
    # Two disjoint sums because the two halves have two different sources of truth: a row
    # already in the body is never re-rendered, so what it PRINTS is all the artifact will
    # ever say about its side; a fresh row's side is `expected`, which is what renders it.
    recorded = (sum(1 for i in pop[:named] if i in seen and i in carry)
                + sum(1 for i in pop[:named] if i not in seen and i in expected))
    # Coverage moves with the merge: `pop` is what the file names the moment this write
    # lands, and the window's count is the denominator. Both sides of the subtraction
    # moved, so the filed body's line is dropped rather than left beside this one —
    # see `_strip_coverage`.
    # Handed to `_write_body` UNCOMPOSED, because `body` here is what the file said
    # BEFORE the widening — seconds of ledger reads ago — and this is the one place a
    # human's ruling on a disagreement is ever rewritten. Composing it here is what made
    # the "kept their text" claim false: `_write_body` re-reads under the lock and appends
    # to THAT read instead. #3677.
    # The leg is named in the heading (`leg_from_trigger`, the same call `_event` makes for
    # its `leg` field, #4251). `_merge` already forwarded `trigger` to the ledger row, so the
    # widening's leg was recorded and never published: a reader of a merged body saw the
    # ORIGINAL create's direction and nothing for the widening, which is how a revert-leg
    # widening under a rate-leg create reads as more of the same. The figures keep their
    # exact position and format; only the trailing clause is new.
    appended = (f"### Added {stamp} (window `judged={window.get('judged')}`, "
                f"`rate={window.get('rate')}`) — widened by the "
                f"**{leg_from_trigger(trigger)}** leg\n\n" + "\n".join(rows) + "\n\n"
                + _coverage(named, window, merged=True, withheld=withheld,
                            unrecoverable=unrecoverable, dropped=dropped,
                            distinct=distinct, unruled=unruled, recorded=recorded,
                            abstained=abstained))
    _write_body(Path(item.path), fm, appended, B,
                activity=f"board-steward disagreement: {len(fresh)} row(s) added")
    _event(ledger, item_id=item.id, action="merged", ids=ids, window=window,
           stamp=stamp, new_ids=fresh, trigger=trigger, revert_pairs=revert_pairs)
    return {"action": "merged", "item_id": item.id, "path": str(item.path),
            "ids": ids, "new_ids": fresh, "rows": len(rows)}


def _write_body(path: Path, fm: dict, appended: str, B: Any, *,
                activity: str) -> None:
    """Append this filing's text to the body AS THE FILE SAYS UNDER ITS OWN LOCK.

    `update_frontmatter` is the public writer but it re-writes the body it just
    read, so a caller that changed the body cannot use it; the alternative —
    write the body, then call it — is two writes with a window in which another
    reader sees the new rows and the old `updated`. So this takes the same lock
    it takes and does the same tail: re-read under the lock (an edit that landed
    while we were computing is kept, not overwritten), refresh `updated`, append
    the activity line, write atomically. It never touches `status`, which is the
    invariant every backlog writer keeps.

    The parameter is therefore the text to APPEND, not the body to write (#3677).
    `commit_lock` is reentrant, so it excludes every OTHER writer and not the
    caller's own earlier read: `_merge` read the body before widening it, and the
    gap to this write is the whole widening — `_window_moves` over an 18,479-row
    ledger plus the window read. A ruling posted inside that gap used to be replaced
    by `_merge`'s stale string while the activity log claimed it was kept, and what
    it erased was a judgment about which side of a disagreement was wrong — the one
    thing on the item that cannot be regenerated. The stale `## Coverage` section is
    stripped from the fresh read here, because it is this worker's own section to
    replace and nobody else's, and `appended` carries the current one.
    """
    from app.atomic_io import commit_lock
    from app import backlog_write as BW
    from app.backlog_move import now_stamp

    with commit_lock(path) as lock:
        current = path.read_text(encoding="utf-8")
        fm_now, body_now = B._split_frontmatter(current)
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
        # Whatever the file holds NOW is what our rows go on top of — read here, under
        # the lock, not carried in from before it. The id-dedup ran against the earlier
        # read, and this lock excludes every other writer of the file, so the only party
        # that can have moved the body in between is a person editing the item, and for
        # them keeping their text and adding the rows is the safe answer. Composing the
        # two above this function is what made that sentence a comment and not a
        # behaviour (#3677); the composition is here now.
        BW.write_item(path, fm2, _strip_coverage(body_now).rstrip("\n") + "\n\n"
                      + appended, lock=lock)


def _file_quietly(agree: dict, moves: list[dict], expected: dict[int, tuple],
                  *, backlog_dir: Path | None = None,
                  ledger: Path | None = None,
                  rules_note: str = "",
                  session_id: str | None = None) -> dict | None:
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
        # `ledger` is forwarded to the trigger AND to the rows this writes (round-1 review
        # finding 3): before that, `below_floor` always read the production store while the
        # rows went wherever `file_or_update` was pointed, so no test could put a revert and
        # its alarm in one file and the seam went ungraded. None still means the production
        # store, resolved by `read_events`/`append_event` themselves — never by a guess here.
        return file_or_update(agree, moves, expected, window=window,
                              backlog_dir=backlog_dir, ledger=ledger,
                              rules_note=rules_note, session_id=session_id)
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
           new_ids: list[int] | None = None,
           trigger: str = "",
           revert_pairs: list[dict] | None = None) -> None:
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
        # Which leg fired, as a FIELD rather than prose: the 7-day stop condition is read
        # off this store by a later job, which must be able to SELECT the revert-leg rows
        # without parsing a sentence or re-measuring a window that has since aged. The value
        # comes from `leg_from_trigger`, the one derivation the created body's direction and
        # a widening's heading read too (#4251), so no filing can print "below" while its own
        # row says "post-flip revert".
        "leg": leg_from_trigger(trigger),
        # Every pair `reverted_moves` returned for this window — the items the revert leg
        # actually keyed on (#4514). A row's `item_id` is the disagreement item and its
        # `tick_item_ids` are the tick's disagree ids, so for each of the six post-flip
        # firings of 2026-11-16..20 the keyed item sat in NO field of its own row, and the
        # only way to recover it was replaying `reverted_moves` over the store — a replay
        # bounded BELOW only, so it also returns pairs stamped after the firing. A row
        # written at the firing's own instant is the only record there is. These four keys
        # are all an audit has to ask; `reverted_moves`' own `reason` stays off the row
        # because the body's trigger sentence already quotes it verbatim.
        # An EMPTY list is NOT a finding that nothing was put back: `below_floor` answers a
        # sub-floor rate before the leg is evaluated, so on a `pre-flip rate` row this field
        # means "the leg never ran". `leg` above is the only discriminator — do not SELECT
        # on the empty list.
        "revert_pairs": [{k: p.get(k) for k in ("item_id", "applied", "reverted_to", "at")}
                         for p in (revert_pairs or ())],
        "at": stamp,
    }, path=ledger)
