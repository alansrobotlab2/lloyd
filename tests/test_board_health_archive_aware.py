"""`board_health()` over a ledger and over a fold of it (#3612).

The retention sweep's neutrality proof asks this function the same question twice —
once over the store as it is, once over the store with the >14-day rows removed — and
refuses the fold if any answer moves. That made the fold unreachable from 2026-10-30
onward, and the reason is one word in the readers `board_health` is built from:
`triaged_ids` decided `triaged` versus `pool` for every open draft from rows the live
file held, so archiving an open item's ONLY triage row reclassified it. Measured on the
live store: `.draft.pool` 0 → 1, `.draft.triaged` 14 → 13, `triaged_ids` 861 → 725, with
exactly ONE open item among the 136 ids that lost their verdict — #2796, whose single
`backlog_triage` row is dated 2026-10-16T00:08:30Z and crossed the 14-day window at
2026-10-30T00:08:30Z, about five hours before the first refusing run.

#1975's owed ruling (2026-10-02T02:24:36Z) licensed the 14-day window on the measurement
"Open items reached: 0 for every set at 7, 14, 21 and 30 days" and left the other branch
open — archive-aware readers, or an accepted flip. #2796 is the first item to live past
the window, so the branch is now ruled: the readers answer from the live ledger PLUS the
`promotions-archive-<YYYYMM>.jsonl.gz` rotations beside it, and a fold that files an
item's evidence rather than destroying it moves nothing.

These nodes pin the WHOLE payload, not the two draft counts, because the proof compares
the payload: one moving figure and the fold is refused for every fold that follows.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from scripts.automod import backlog as B
from scripts.automod import board_decisions as BD
from scripts.automod import state as S

# One clock for both layouts, and it is deliberately NOT `time.time()`:
# `board_health` stamps `.decisions.window.since/.until` from its `now`, so a per-call
# clock disagrees across two calls over byte-identical bytes (#1975 review) and this
# file would go red for the wrong reason.
NOW = 1_793_000_000.0
MONTH = time.strftime("%Y%m", time.gmtime(NOW - 20 * 86400))
# #2796's shape: one open `draft`, one terminal triage row, 20 days old — past the
# sweep's `LEDGER_ARCHIVE_AGE_DAYS = 14`, so the fold chooses it every day from here.
OPEN_DRAFT_ID = 2796
TRIAGE_TS = NOW - 20 * 86400

TRIAGE_ROW = {"event": "backlog_triage", "item_id": OPEN_DRAFT_ID,
              "verdict": "confirmed", "verdict_source": "structured", "ts": TRIAGE_TS,
              "surface": "code", "acceptance": "The clause the round must satisfy."}


def _clear_caches():
    S._rows_cache.clear()
    S._rows_reads.clear()
    if hasattr(B, "_clear_caches"):
        B._clear_caches()


def _board(tmp_path: Path) -> Path:
    """A board holding exactly one open `draft` item — the only board shape the flip
    can reach. A `draft` with no triage row is `.draft.pool`; the same draft WITH a
    terminal triage row is `.draft.triaged` (`triaged_ids` is what tells them apart)."""
    board = tmp_path / "board"
    board.mkdir()
    (board / f"{OPEN_DRAFT_ID}-an-open-draft.md").write_text(
        "---\ntype: backlog\nsegment: backlog\nstatus: draft\npriority: low\n"
        "board: lloyd\nblocked: false\nassigned: false\n---\n\n# An open draft\n\n"
        "Body.\n", encoding="utf-8")
    return board


def _layout(tmp_path: Path, name: str, *, rows_live: bool, rows_archived: bool) -> Path:
    """One store directory: `rows_live` keeps the triage row in `promotions.jsonl`,
    `rows_archived` files it in the month rotation the sweep would write."""
    store = tmp_path / name
    store.mkdir()
    line = json.dumps(TRIAGE_ROW) + "\n"
    (store / "promotions.jsonl").write_text(line if rows_live else "", encoding="utf-8")
    if rows_archived:
        (store / f"promotions-archive-{MONTH}.jsonl.gz").write_bytes(
            gzip.compress(line.encode("utf-8")))
    return store


def _health(store: Path, board: Path) -> dict:
    _clear_caches()
    payload = B.board_health(store / "promotions.jsonl", backlog_dir=board, now=NOW)
    assert payload.get("error") is None, payload
    return payload


def _moved_keys(before: dict, after: dict) -> list[str]:
    """Dotted keys whose value differs — the same shape the sweep reports in its
    `REFUSED (board_health() changed at …)` line."""
    out: list[str] = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    out.append(f"{path}.{key}")
                else:
                    walk(a[key], b[key], f"{path}.{key}")
        elif a != b:
            out.append(path)

    walk(before, after, "")
    return out


def test_board_health_is_identical_over_a_fold_of_an_open_items_only_triage_row(tmp_path):
    """Clause 3: the payload does not move when the row that makes a draft `triaged`
    is FILED rather than destroyed.

    Two layouts of one store — the row live, and the row in the month rotation — over
    one board and one pinned `now`. Every figure has to agree, `.draft.triaged` and
    `.draft.pool` in particular: those two are the keys the live proof named in its
    refusal on 2026-10-30 and 2026-10-31, and the sweep compares the payload, so a
    single moving key keeps the fold refused for good.
    """
    board = _board(tmp_path)
    live = _health(_layout(tmp_path, "live", rows_live=True, rows_archived=False), board)
    folded = _health(_layout(tmp_path, "folded", rows_live=False, rows_archived=True),
                     board)

    assert _moved_keys(live, folded) == [], (
        "a fold that only MOVED the row changed the board; that is the refusal that "
        "made the retention sweep unreachable from 2026-10-30"
    )
    # The fixture must not be vacuous: if both layouts read the item as `pool`, the
    # equality above proves nothing. A folded, still-read verdict is what is asserted.
    assert live["draft"]["triaged"] == 1 and live["draft"]["pool"] == 0, live["draft"]
    assert folded["draft"]["triaged"] == 1 and folded["draft"]["pool"] == 0, (
        folded["draft"])


def test_the_flip_this_change_prevents_is_visible_when_the_row_is_only_dropped(tmp_path):
    """The negative control, and the number #1975's owed clause 3 asked for.

    Same fixture, same board, but the triage row is DELETED instead of archived — the
    pre-#3612 reality for a store whose reader sees only the live file. The two draft
    counts have to move, which is what makes the node above a claim about archiving
    rather than a claim about an inert fixture. It also says, in one place, how far the
    accepted-flip branch (#3612 option (b)) would have reached: on the live store that
    is ONE open item — #2796 — of the 136 ids whose only triage row folded on
    2026-10-31, the other 135 being `status: done`.
    """
    board = _board(tmp_path)
    before = _health(_layout(tmp_path, "before", rows_live=True, rows_archived=False),
                     board)
    after = _health(_layout(tmp_path, "dropped", rows_live=False, rows_archived=False),
                    board)

    moved = _moved_keys(before, after)
    assert ".draft.triaged" in moved and ".draft.pool" in moved, moved
    assert (before["draft"]["triaged"], before["draft"]["pool"]) == (1, 0)
    assert (after["draft"]["triaged"], after["draft"]["pool"]) == (0, 1), after["draft"]


def test_a_folded_held_confirmation_and_release_leave_the_board_where_they_were(tmp_path):
    """`.held` is the other ledger-derived draft figure in the same payload, and it
    answers through `held_confirmations` — a confirmation waiting for room, plus the
    release rows that ended the wait.

    Held here means BOTH rows are archived: the `confirmed` verdict carrying `held:
    true` and nothing released. If the confirmation folded and the release did not, the
    item would be counted held; if the release folded and the confirmation did not, held
    again. Either way `.held` moves and the fold is refused, so the pair is read from one
    merged population and cannot be split by a rotation.
    """
    board = _board(tmp_path)
    held_rows = [{**TRIAGE_ROW, "held": True},
                 {"event": "backlog_confirm_released", "item_id": OPEN_DRAFT_ID,
                  "ts": TRIAGE_TS + 3600, "round_id": "SM_REL_1"}]

    def store(name: str, *, live: bool) -> Path:
        d = tmp_path / name
        d.mkdir()
        text = "".join(json.dumps(r) + "\n" for r in held_rows)
        (d / "promotions.jsonl").write_text(text if live else "", encoding="utf-8")
        if not live:
            (d / f"promotions-archive-{MONTH}.jsonl.gz").write_bytes(
                gzip.compress(text.encode("utf-8")))
        return d

    live = _health(store("held-live", live=True), board)
    folded = _health(store("held-folded", live=False), board)
    assert _moved_keys(live, folded) == [], _moved_keys(live, folded)
    # And the pair is genuinely being read: the release resolves the confirmation in
    # BOTH layouts, so nobody is held. Read the confirmation from the archive while the
    # release stayed live-only — or the other way round — and `.held` is 1 here, which
    # is a figure that moves across a fold and therefore a refused fold.
    assert live["draft"]["held"] == 0, live["draft"]


def test_the_quarantine_figure_does_not_move_when_the_release_row_folds(tmp_path):
    """`.quarantined` is self-filed items MINUS `released_ids`, so an archived release
    row that stops answering ADDS a quarantine — and the item behind it goes back into
    the pool the sweep re-triages.

    A `backlog_sweep` row is the release #3612 names: the item was ranked by a sweep, so
    it may be triaged singly. Here the release is 20 days old and the board item is
    self-filed (`spawned-by-*`), which is the shape the figure is computed over.
    """
    board = tmp_path / "board"
    board.mkdir()
    (board / f"{OPEN_DRAFT_ID}-a-self-filed-draft.md").write_text(
        "---\ntype: backlog\nsegment: backlog\nstatus: draft\npriority: low\n"
        "board: lloyd\nblocked: false\nassigned: false\n"
        "tags: [spawned-by-triage]\n---\n\n# A self-filed draft\n\nBody.\n",
        encoding="utf-8")
    release = {"event": "backlog_sweep", "judged": {str(OPEN_DRAFT_ID): "keep"},
               "ts": TRIAGE_TS}

    def store(name: str, *, live: bool) -> Path:
        d = tmp_path / name
        d.mkdir()
        text = json.dumps(release) + "\n"
        (d / "promotions.jsonl").write_text(text if live else "", encoding="utf-8")
        if not live:
            (d / f"promotions-archive-{MONTH}.jsonl.gz").write_bytes(
                gzip.compress(text.encode("utf-8")))
        return d

    live = _health(store("rel-live", live=True), board)
    folded = _health(store("rel-folded", live=False), board)
    assert _moved_keys(live, folded) == [], _moved_keys(live, folded)
    assert live["draft"]["quarantined"] == 0, (
        "the release row was not read at all, so this fixture proves nothing about folds")


# ── #3725 clause 1: `.up_next.never_attempted` is the same failure one partition over ──
#
# #3612 ruled the draft arms and #3650 (done) shipped `LEDGER_NEUTRALITY_TOLERATED_KEYS`
# as a frozenset of exactly those two paths; the `up_next` arm was in neither scope. On
# 2026-11-03 the sweep refused its fold there: `REFUSED (board_health() changed at
# .up_next.never_attempted) — live file untouched, 31251370 bytes`, because `attempted`
# was `{int(d["item_id"]) for d in _ledger_events(ledger, "backlog_implement")}` — a
# one-file read. Over before/after copies of the live store that set went 755 → 717, and
# exactly ONE of the 38 losers is open in `up_next`: #3024, whose only two
# `backlog_implement` rows are dated 2026-10-19 and both carry `round_id: None`, so
# `_archive_plan`'s newest-of-round keeper (`retention-sweep.py:2392`, which skips a row
# whose `round_id` is not a non-empty string) never spares them. The nodes below are that
# item's shape; #3725 fixes the figure rather than widening the tolerance set, because
# `never_attempted` is a prioritisation input the board steward re-offers on.

OPEN_UP_NEXT_ID = 3024
_ATTEMPT_TS = NOW - 20 * 86400          # MONTH's bucket, and outside `WINDOW_DAYS = 7`
ATTEMPT_ROWS = [
    {"event": "backlog_implement", "item_id": OPEN_UP_NEXT_ID, "round_id": None,
     "phase": "started", "ts": _ATTEMPT_TS},
    {"event": "backlog_implement", "item_id": OPEN_UP_NEXT_ID, "round_id": None,
     "phase": "finished", "ts": _ATTEMPT_TS + 128},
]
#: One row that stays live in every layout, so the folded store is not empty and the
#: young-row keeper has something to keep — an empty live file would let a reader that
#: globbed nothing look identical to one that read the archive.
_YOUNG_ROW = {"event": "gate", "round_id": "SM_YOUNG", "rung": "tests",
              "ts": NOW - 2 * 86400}


def _up_next_board(tmp_path: Path) -> Path:
    """A board holding exactly one open `up_next` item, #3024's own front matter shape
    (`priority: low`, nothing else marking it). It is the only board shape the arm
    counts: `up = [i for i in items if i.status == IMPLEMENT_POOL_STATUS]`."""
    board = tmp_path / "board"
    board.mkdir()
    (board / f"{OPEN_UP_NEXT_ID}-an-open-up-next-item.md").write_text(
        "---\ntype: backlog\nsegment: backlog\nstatus: up_next\npriority: low\n"
        "board: lloyd\nblocked: false\nassigned: false\n---\n\n"
        "# An open up_next item\n\nBody.\n", encoding="utf-8")
    return board


def _attempt_layout(tmp_path: Path, name: str, *, rows_live: bool,
                    rows_archived: bool) -> Path:
    """One store directory holding #3024's two attempt rows either live or filed, plus
    the young row the fold always keeps."""
    store = tmp_path / name
    store.mkdir()
    text = "".join(json.dumps(r) + "\n" for r in ATTEMPT_ROWS)
    (store / "promotions.jsonl").write_text(
        (text if rows_live else "") + json.dumps(_YOUNG_ROW) + "\n", encoding="utf-8")
    if rows_archived:
        (store / f"promotions-archive-{MONTH}.jsonl.gz").write_bytes(
            gzip.compress(text.encode("utf-8")))
    return store


def test_an_up_next_item_whose_only_attempt_rows_are_filed_is_still_attempted(tmp_path):
    """Clause 1 of #3725: `attempted` is read over the live ledger PLUS the rotations, so
    `.up_next.never_attempted` does not step when the rows that make an item attempted are
    FILED rather than destroyed.

    Two layouts of one store over one board and one pinned `now`, exactly like the draft
    nodes above, but the rows are `backlog_implement` and the arm is `up_next`'s. Both
    figures have to be 0 — the folded side reading 1 is the +1 that made the live fold
    refuse on 2026-11-03 — and the whole payload has to agree, because the sweep compares
    payloads, not this one key.
    """
    board = _up_next_board(tmp_path)
    live = _health(_attempt_layout(tmp_path, "live", rows_live=True, rows_archived=False),
                   board)
    folded = _health(_attempt_layout(tmp_path, "folded", rows_live=False,
                                     rows_archived=True), board)

    assert live["up_next"]["total"] == 1 == folded["up_next"]["total"], (
        f"the board stopped holding one open up_next item, so the arm below is a count over "
        f"an empty partition: {live['up_next']} / {folded['up_next']}")
    assert live["up_next"]["never_attempted"] == 0, live["up_next"]
    assert folded["up_next"]["never_attempted"] == 0, folded["up_next"]
    assert _moved_keys(live, folded) == [], _moved_keys(live, folded)


def test_the_up_next_attempt_arm_steps_when_the_rows_are_destroyed_not_filed(tmp_path):
    """The control under clause 1, and the reason its 0 is not an artefact.

    Same board, same young row, minus the two attempt rows entirely — the state a store
    whose rows were DELETED leaves, which is what the guard's `drops evidence` branch is
    for. `never_attempted` has to answer 1 here: if it answered 0 in this layout too, the
    item was never counted as attempted in the first place and the node above would be
    pinning nothing.
    """
    board = _up_next_board(tmp_path)
    before = _health(_attempt_layout(tmp_path, "before", rows_live=True,
                                     rows_archived=False), board)
    after = _health(_attempt_layout(tmp_path, "dropped", rows_live=False,
                                    rows_archived=False), board)

    assert (before["up_next"]["never_attempted"],
            after["up_next"]["never_attempted"]) == (0, 1), (
        f"{before['up_next']} / {after['up_next']}")
    assert ".up_next.never_attempted" in _moved_keys(before, after), _moved_keys(
        before, after)


# ── #4039: `.decisions` is the third arm of that one class ────────────────────────
#
# #3612 widened the readers `board_health` is built from and #3650 ruled the two `draft`
# arms tolerated; #3725 made the `up_next` arm fold-invariant the same way. The `decisions`
# partition was the third and last reader with no archive arm: `board_decisions` replayed
# `B._ledger_rows(ledger)` — ONE file — while `triaged_ids`, `implement_outcomes`,
# `retriage_marks` and `implement_history` all take a `with_archives` arm. So on
# 2026-11-11 the retention sweep refused its own fold over a 31,035,309-byte store: seven
# moved keys, every one under `.decisions`, `.decisions.promotions.count` 292 → 295 with
# `by_decider.unrecorded` ABSENT → 3 and `.decisions.window` byte-identical on both sides.
#
# The shape that moves is an item whose pool ENTRY row folds while its later, in-window
# RE-ENTRY row stays live. `enter()` is a no-op only while the item is in `in_pool`, so the
# row that is filed is what kept the open entry remembered: with it gone the re-entry reads
# as a FRESH promotion, and a `backlog_implement`/`phase: started` row is the sole producer
# of the `unrecorded` decider — a fold would re-attribute somebody's decision to "a person
# in Mission Control, or autonomy #35", which is a different claim from the draft-arm step
# #3650 accepted. So these nodes pin invariance from evidence being FILED, never the
# tolerance.
POOL_ITEM = 9904039           # the item whose pool-entry row the fold reaches
OPEN_PROMO_IDS = (9904101, 9904102, 9904103, 9904104, 9904105)
RETIRED_ID = 9904106
_ENTRY_TS = NOW - 20 * 86400          # MONTH's bucket, outside `WINDOW_DAYS = 7`
_REENTRY_TS = NOW - 2 * 86400         # inside the window, and young enough to stay live
_OTHER_TS = NOW - 3 * 86400
_REENTRY_DAY = time.strftime("%Y-%m-%d", time.gmtime(_REENTRY_TS))


def _confirm(iid: int, ts: float) -> dict:
    """A terminal `backlog_triage` confirm: the event that enters an item into the pool."""
    return {"event": "backlog_triage", "item_id": iid, "verdict": "confirmed",
            "verdict_source": "structured", "ts": ts, "surface": "code",
            "acceptance": "The clause the round must satisfy."}


def _decision_rows() -> list[dict]:
    """One ledger whose `.decisions` figures depend on a single row's location.

    Row 0 is the pool ENTRY the fold reaches: 20 days old and carrying no `round_id`, so
    `_archive_plan`'s newest-of-round keeper spares nothing and it is a fold candidate every
    day. Row 1 is the in-window re-entry that always stays live. Rows 2-6 are five
    promotions that sit in the same place in every layout, which is what makes `count` a
    number rather than 0 and clears `MIN_PROMOTIONS = 5` so `outcome` answers `measured`;
    two of them reach done, so an extra OPEN promotion moves `done_rate` and the terminal
    counts as well as `count` — the seven-key shape the live diff has. Rows 7-8 are a
    re-triage mark and a retirement for a third item: `retire()` is the accumulator that is
    NOT idempotent in a duplicate, and those two are clause 3's duplicate candidates.
    """
    return [
        _confirm(POOL_ITEM, _ENTRY_TS),
        {"event": "backlog_implement", "item_id": POOL_ITEM, "round_id": "SM_4039",
         "phase": "started", "ts": _REENTRY_TS},
        *[_confirm(i, _OTHER_TS) for i in OPEN_PROMO_IDS],
        {"event": "item_landed", "item_id": OPEN_PROMO_IDS[0], "closed": True,
         "ts": _OTHER_TS + 3600},
        {"event": "status_moved", "item_id": OPEN_PROMO_IDS[1], "from": "up_next",
         "to": "draft", "reason": "returned by review — round refused",
         "ts": _OTHER_TS + 7200},
        {"event": "backlog_retriage", "item_id": RETIRED_ID, "ts": _OTHER_TS + 60},
        {"event": "status_moved", "item_id": RETIRED_ID, "from": "up_next",
         "to": "draft", "reason": "needs_human — person still owes",
         "ts": _OTHER_TS + 120},
        _YOUNG_ROW,
    ]


def _ndjson(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _dec_store(tmp_path: Path, name: str, live_rows: list[dict],
               archived_rows: list[dict]) -> Path:
    """One store directory: `live_rows` in `promotions.jsonl`, `archived_rows` in the
    rotation for the MONTH their own `ts` names, which is how `_month_buckets` groups."""
    store = tmp_path / name
    store.mkdir()
    (store / "promotions.jsonl").write_text(_ndjson(live_rows), encoding="utf-8")
    months: dict[str, list[dict]] = {}
    for row in archived_rows:
        months.setdefault(time.strftime("%Y-%m", time.gmtime(float(row["ts"]))),
                          []).append(row)
    for month, rows_in in months.items():
        (store / f"promotions-archive-{month}.jsonl.gz").write_bytes(
            gzip.compress(_ndjson(rows_in).encode("utf-8")))
    return store


def _empty_board(tmp_path: Path, name: str) -> Path:
    """An empty board: the `.decisions` partition is counted from ledger rows alone, so
    the fixture needs no item files and reads no real board."""
    board = tmp_path / f"{name}-board"
    board.mkdir()
    return board


def test_a_folded_pool_entry_row_leaves_the_decisions_figures_where_they_were(tmp_path):
    """Clause 1 of #4039: filing an item's pool-entry row does not re-count its re-entry.

    Two layouts of one store over one board and one pinned `now`, the same pair the draft
    and up_next nodes above use: the entry row live, and the entry row in the rotation for
    its own month. `.decisions.promotions.count` and `.decisions.per_day` have to be equal —
    five, the five promotions every layout holds — and so does the whole `.decisions`
    payload, because the sweep's neutrality proof compares payloads and one moving key
    refuses the fold for good.

    The fixture's bite is pinned next door, not here:
    `test_deleting_the_pool_entry_row_re_counts_a_promotion_attributed_to_unrecorded` shows
    the same seven rows with that one row in neither file counting a sixth promotion under
    `unrecorded`, which is what proves the re-entry row is being read and counted at all.
    """
    rows = _decision_rows()
    entry = rows[0]
    board = _empty_board(tmp_path, "dec")

    live = _health(_dec_store(tmp_path, "dec-live", rows, []), board)
    folded = _health(_dec_store(tmp_path, "dec-folded",
                                [r for r in rows if r is not entry], [entry]), board)

    assert live["decisions"]["promotions"]["count"] == 5, live["decisions"]["promotions"]
    assert folded["decisions"]["promotions"]["count"] == 5, (
        f"a fold that only MOVED the entry row re-counted this item's re-entry as a fresh "
        f"promotion — the refusal that has this store unbounded: "
        f"{folded['decisions']['promotions']}")
    assert folded["decisions"]["per_day"] == live["decisions"]["per_day"], (
        folded["decisions"]["per_day"])
    assert _moved_keys(live["decisions"], folded["decisions"]) == [], (
        _moved_keys(live["decisions"], folded["decisions"]))
    for side in (live, folded):
        assert "unrecorded" not in side["decisions"]["promotions"]["by_decider"], (
            f"the re-entry was counted as a promotion nobody decided: "
            f"{side['decisions']['promotions']['by_decider']}")


def test_deleting_the_pool_entry_row_re_counts_a_promotion_attributed_to_unrecorded(tmp_path):
    """Clause 2 of #4039: the invariance comes from evidence being FILED, not ignored.

    Same seven-plus rows, same board, but the entry row is in NEITHER file — the state a
    store is in when the row is destroyed rather than rotated. `.decisions.promotions.count`
    has to step 5 → 6 and the extra promotion has to carry the `unrecorded` decider, on
    `_REENTRY_DAY`, which is the attribution #4039 refuses to let a fold make: `enter(iid,
    ts, "unrecorded", "implement_started", None)` is the only producer of that decider, and
    `health_line` prints it to the steward as "a person in Mission Control, or autonomy #35".

    This is also the node that says the fixture is not inert: if the re-entry row were never
    read, clause 1's two `count == 5` assertions would pass for the wrong reason.
    """
    rows = _decision_rows()
    entry = rows[0]
    board = _empty_board(tmp_path, "dec-del")

    before = _health(_dec_store(tmp_path, "del-before", rows, []), board)
    after = _health(_dec_store(tmp_path, "del-dropped",
                               [r for r in rows if r is not entry], []), board)

    assert (before["decisions"]["promotions"]["count"],
            after["decisions"]["promotions"]["count"]) == (5, 6), (
        f"{before['decisions']['promotions']} / {after['decisions']['promotions']}")
    assert after["decisions"]["promotions"]["by_decider"].get("unrecorded") == 1, (
        after["decisions"]["promotions"]["by_decider"])
    assert "unrecorded" not in before["decisions"]["promotions"]["by_decider"]

    def day(payload: dict) -> dict:
        rows_in = [d for d in payload["decisions"]["per_day"] if d["day"] == _REENTRY_DAY]
        assert len(rows_in) == 1, payload["decisions"]["per_day"]
        return rows_in[0]

    assert (day(before)["promotions"], day(before)["by_decider"]) == (0, {}), day(before)
    assert (day(after)["promotions"], day(after)["by_decider"]) == (1, {"unrecorded": 1}), (
        day(after))
    assert ".decisions.promotions.count" in _moved_keys(before, after), (
        _moved_keys(before, after))


def test_a_row_in_both_the_live_ledger_and_a_rotation_is_replayed_once(tmp_path):
    """Clause 3 of #4039: a row present in BOTH files is one event to the replay.

    `state.ledger_rows_merged` documents that it does not dedupe because its consumers
    "resolve by last-wins or by `max(ts)`, which are both idempotent in a duplicate". A
    replay is neither: `retire()` appends to `retirements` unconditionally, so the same
    row seen twice double-counts a retirement and steps `.retire_reopen.retired`. Every
    figure of `.decisions` therefore has to come back equal to the layout where each row is
    in one file only, and the equality is pinned to a number beside it: `retired` is 2 on
    the single-copy side (one landed, one needs_human), so the side carrying three of those
    rows in BOTH files has to answer 2 as well and not 4, and `count` stays 5.

    The merged reader is asserted underneath the payload, because the payload equality
    alone would also be true of a reader that never opened the rotation at all — which is
    exactly how this module behaved at base, and is why the three count assertions above
    were green there. `_ledger_rows_merged` hands back every duplicated row twice;
    `_replay_rows` is the pass that collapses it, and it is a new seam, so it is measured.

    The rows are duplicated IN window on purpose. A fold only ever leaves rows ≥
    `LEDGER_ARCHIVE_AGE_DAYS` = 14 in both files and `WINDOW_DAYS` = 7, so a fold-produced
    duplicate is out of window by construction and moves no windowed figure — that
    boundedness is measured, not assumed, and it is also why this arm cannot be checked at
    all unless the duplicated rows are put inside the window.
    """
    rows = _decision_rows()
    duplicated = [rows[0], rows[6], rows[7]]        # entry, re-triage mark, retirement
    board = _empty_board(tmp_path, "dec-dup")
    store = _dec_store(tmp_path, "dup-both", rows, duplicated)

    single = _health(_dec_store(tmp_path, "dup-single", rows, []), board)
    both = _health(store, board)

    # The LEDGER FILE, not the store directory: `ledger_rows` answers `[]` for a path it
    # cannot read, and a directory stats fine, so a directory handed to these two reads is
    # a silent zero rows rather than an error — which is exactly how this assertion first
    # read as a failing product claim.
    led = store / "promotions.jsonl"
    assert len(B._ledger_rows_merged(led)) == len(rows) + len(duplicated), (
        "the merged reader is documented NOT to dedupe; if it starts to, `_replay_rows`'s "
        f"own pass is dead code and this node no longer tests what it says: "
        f"{len(B._ledger_rows_merged(led))} rows for {len(rows)} live + "
        f"{len(duplicated)} rotated")
    assert len(BD._replay_rows(led)) == len(rows), (
        f"the replay has to be handed each distinct row exactly once: "
        f"{len(BD._replay_rows(led))} rows back for {len(rows)} distinct ones")

    assert single["decisions"]["retire_reopen"]["retired"] == 2, (
        f"the fixture holds two in-window retirements (one landed, one needs_human); with "
        f"fewer, `.retire_reopen` is a count over an empty set and the equality below proves "
        f"nothing: {single['decisions']['retire_reopen']}")
    assert _moved_keys(single["decisions"], both["decisions"]) == [], (
        f"a row the live ledger and a rotation BOTH hold was replayed twice: "
        f"{_moved_keys(single['decisions'], both['decisions'])}")
    assert both["decisions"]["promotions"]["count"] == 5, (
        both["decisions"]["promotions"])
