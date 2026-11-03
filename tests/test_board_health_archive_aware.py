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
