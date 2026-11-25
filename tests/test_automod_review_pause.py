"""#4562: a person who pauses the pool through mission control stops review grading.

The incident this file exists over, read off the live dispatch store: `pause_journal`
id 75 opened at `2026-11-22T21:17:25.840615+00:00` with `holder_kind=person` and
`actor=mission-control`, and id 76 released it 27 h 21 m 24 s later at
`2026-11-24T00:38:49.696701+00:00` (`hold_id=75`). Across that window
`select count(*) from runs where started_at between '2026-11-22T21:17:25' and
'2026-11-24T00:38:49'` returned **0** — the store said the machine was idle — while
47 session files appeared under it, 46 with `source: automod-review`: 45 from the
offline calibration backfill (titles like `review #3981
(backfill:cal_3981-gh-token-fallback)`, the id built at `review_tools.py:150`) and 1
an in-flight round (`20261122_133146_review_16a0`, 14 min after the hold opened).

Why the pause did not stop it: the hold is a queue watermark
(`workers/pool.py`: `PAUSE_WM_SOURCE`/`PAUSE_WM_KEY`/`PAUSE_WM_HOLDER_KEY`), and a
watermark gates *claims from the pool*. The reviewer claims nothing — it POSTs its own
`/api/message/stream` turn — so `grep -E 'pause|halt|wm_get|operator_paused'
scripts/automod/review.py` returned rc=1 over the whole module while the grader ran 46
times on a machine somebody had asked to be free.

Every node here seeds the hold through the pool's OWN constants and reads the store
the pool writes, so these are crossings of that boundary rather than a second flag
checked against itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.automod import review as RP
from workers.pool import PAUSE_WM_HOLDER_KEY, PAUSE_WM_KEY, PAUSE_WM_SOURCE
from workers.queue import WorkQueue

#: The three routes that reach `write_session`, named as `review_route` names them,
#: with the `round_id` each one actually passes. A live round's is its real id; the
#: vault grade passes the literal `vault` (`review.py`, `grade_vault` -> `run_grader`);
#: the backfill's is built as `backfill:cal_<case>` at `review_tools.py:150` from
#: `calibration_cases.py:266`.
ROUTES = [
    pytest.param("round", "SM_20261125_100809", id="code-round"),
    pytest.param("vault", "vault", id="vault-grade"),
    pytest.param("calibration", "backfill:cal_3981-gh-token-fallback", id="calibration-backfill"),
]

BACKEND = "http://127.0.0.1:9"  # nothing listens; never reached under a hold


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The dispatch store the reviewer reads, redirected off the live `workers.db`.

    One indirection (`_queue_store_path`) is redirected, because that is the only
    path the review's two store touches take: `operator_hold_armed` and
    `record_review_run` both go through `_dispatch_store`.
    """
    path = tmp_path / "workers.db"
    q = WorkQueue(path)

    monkeypatch.setattr(RP, "_queue_store_path", lambda: path)
    return q


@pytest.fixture
def sessions(tmp_path):
    return tmp_path / "sessions"


def _hold(store, holder: str = "mission-control") -> None:
    """Arm the hold exactly as `WorkerPool.pause` persists it for a person."""
    store.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "1")
    store.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_HOLDER_KEY, holder)


def _transport(monkeypatch) -> list:
    """Replace the one transport call and record every turn that is posted.

    Returns the list the test asserts on: `[]` means no turn left the process, which
    is the whole claim under a hold.
    """
    posted: list = []

    def fake(url, payload, timeout):
        posted.append((url, payload))
        yield ("done", {"response": "graded", "stop_reason": "final",
                        "structured": {"premise": "sound", "clauses": [],
                                       "test_honesty": [], "seams_unverified": []},
                        "structured_error": ""})

    monkeypatch.setattr(RP, "_post_stream", fake)
    return posted


def _dispatch(round_id: str, sessions) -> dict:
    """One grading dispatch, the way every route ends up dispatched."""
    return RP.run_grader(prompt="grade this", item_id=4562, round_id=round_id,
                         backend=BACKEND, sessions_dir=sessions)


def _session_files(sessions: Path) -> list[Path]:
    return sorted(sessions.glob("*.json")) if sessions.exists() else []


# ── clause 1: under the hold, no grader session is minted, on any route ──────

@pytest.mark.parametrize("kind,round_id", ROUTES)
def test_no_grader_session_is_minted_while_the_operator_hold_is_armed(
        tmp_path, monkeypatch, store, sessions, kind, round_id):
    """The hold stops every route, and mints nothing on the way.

    `run_grader` is the single dispatch all three reach — the code-round grade
    through `grade`, the vault grade through `grade_vault`, the offline calibration
    backfill through `review_tools.grade_commit` — so the check that covers 46
    sessions is the one before `write_session`. Asserted per route because the claim
    is about the fleet, not about a function: 45 of those 46 sessions were the
    backfill, and a rail that only covered the round route would leave the fleet.
    """
    _hold(store)
    posted = _transport(monkeypatch)

    res = _dispatch(round_id, sessions)

    assert res["ok"] is False, res
    assert RP.OPERATOR_HOLD_ERROR in res["error"], res["error"]
    assert _session_files(sessions) == [], "a refused dispatch must not leave a session behind"
    assert posted == [], "the turn must never be posted, not merely its answer discarded"
    assert res["session_id"] == "", "nothing to cancel, so nothing to name"


def test_the_check_reads_the_same_watermark_the_pool_honours(tmp_path, monkeypatch, store,
                                                            sessions):
    """One fact, two readers: the reviewer asks the pool's own keys.

    The hold is armed here through `PAUSE_WM_SOURCE`/`PAUSE_WM_KEY` imported from
    `workers.pool`, the constants `WorkerPool.pause` writes and
    `workers/pool.py:925` reads. A second flag that `review.py` kept for itself would
    drift exactly as quietly as the missing one did, and this node would stay green.
    """
    _hold(store)
    assert RP.operator_hold_armed() == "mission-control"
    # Released the way mission control does it: the value goes to 0, the key stays.
    store.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "0")
    assert RP.operator_hold_armed() == ""
    posted = _transport(monkeypatch)
    res = _dispatch("SM_20261125_100809", sessions)
    assert res["ok"] is True, res
    assert len(posted) == 1


# ── clause 2: refused by return value, and the wait is named ─────────────────

def test_a_held_grade_is_refused_by_return_value_and_names_who_was_waited_for(
        tmp_path, monkeypatch, store, sessions):
    """No exception path is added, and the error says whom the round is waiting on.

    `run_grader` has never raised — a grader that throws takes the rung down with it
    and spends an attempt on a machine nobody is using. The holder is in the text
    because "engine unreachable" and "mission control has the machine" have opposite
    fixes, and the 27 h 21 m hold in this file's docstring is the difference between
    waiting and re-gating 800 times.
    """
    _hold(store, holder="mission-control")
    _transport(monkeypatch)

    res = _dispatch("SM_20261125_100809", sessions)   # raises nothing

    assert res["ok"] is False
    assert "operator pool hold is armed" in res["error"]
    assert "mission-control" in res["error"]
    assert res["operator_hold"] == "mission-control"
    # The keys a caller reads unconditionally are all present, refusal included.
    for key in ("structured", "structured_error", "text", "stop_reason",
                "session_id", "duration_s", "retries", "waited_s"):
        assert key in res, f"a refused dispatch must answer every key an answered one does: {key}"
    # An operator hold is NOT an engine outage: `grader_outage` keys on transport
    # exception names, and reading the hold as an outage would book it as the
    # model server's fault in every instrument that asks.
    assert RP.grader_outage(res["error"]) is False
    assert RP.review_run_status(res) == "cancelled"


def test_a_hold_with_no_holder_key_still_refuses_as_unattributed(tmp_path, monkeypatch, store,
                                                                sessions):
    """The pause's own bookkeeping can lose the holder; the hold may not be missed.

    `pause_journal` id 76 released with `actor=unattributed`, so an unclaimed actor is
    a real state on this store. Falling back to `unattributed` keeps the refusal (the
    machine is still held) instead of treating a missing name as no hold — the one
    reading that would put 46 sessions back on a paused machine.
    """
    store.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "1")
    _transport(monkeypatch)

    assert RP.operator_hold_armed() == "unattributed"
    res = _dispatch("vault", sessions)
    assert res["ok"] is False and res["operator_hold"] == "unattributed"


def test_an_unreadable_store_is_not_read_as_a_hold(tmp_path, monkeypatch, store, sessions):
    """Fail-open where the hold cannot be read, so clause 3's world is unchanged.

    Pointing the store at a directory makes every queue call raise. The shipped
    answer is "no hold", which is what grading did before this file existed: a broken
    dispatch store must not become a second reason a round cannot be graded, and it
    must not silently refuse a grade that today runs.
    """
    (tmp_path / "unreadable.db").mkdir()   # a directory: no database can open there
    monkeypatch.setattr(RP, "_queue_store_path", lambda: tmp_path / "unreadable.db")
    posted = _transport(monkeypatch)

    assert RP.operator_hold_armed() == ""
    res = _dispatch("SM_20261125_100809", sessions)
    assert res["ok"] is True, res
    assert len(posted) == 1 and len(_session_files(sessions)) == 1


# ── the vault route's own reported outcome ───────────────────────────────────

@pytest.fixture
def vault_tree(tmp_path, monkeypatch):
    """A backlog and a ledger of its own, so a vault grade can be driven for real.

    `grade_vault` reads the item's acceptance from the backlog and its triage from the
    ledger; both are redirected to `tmp_path`, the same pair of seams
    `tests/test_automod_review.py` patches.
    """
    import yaml

    from scripts.automod import backlog as B
    from scripts.automod import state as S

    backlog = tmp_path / "backlog"
    backlog.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    tree = tmp_path / "vault"
    (tree / "skills" / "x").mkdir(parents=True)
    (tree / "skills" / "x" / "SKILL.md").write_text("---\nname: x\n---\n# x\n", encoding="utf-8")
    (backlog / "4562-a-thing.md").write_text(
        "---\n" + yaml.dump({"status": "up_next", "priority": "medium", "board": "lloyd",
                             "acceptance_clauses": ["the skill names the retry rule"]},
                            default_flow_style=False)
        + "---\n\n# A thing\n\nDo it.\n", encoding="utf-8")
    S.append_event({"event": "backlog_triage", "item_id": 4562, "verdict": "confirmed",
                    "acceptance": "the skill names the retry rule",
                    "acceptance_clauses": ["the skill names the retry rule"],
                    "surface": "vault"}, path=S.LEDGER_PATH)
    return tree


def test_the_vault_grade_reports_a_held_grader_as_an_abstention_not_a_verdict(
        tmp_path, monkeypatch, store, sessions, vault_tree):
    """The vault route never invents a diff verdict for a machine it could not use.

    `grade_vault` classifies an unusable grade into labelled kinds, and a hold is
    neither a divergence (`GRADER_DIVERGED`, a token budget) nor an outage
    (`GRADER_OUTAGE`, a transport name): it reaches the catch-all `skipped`, with the
    hold in the findings a person reads. `vault_round` records `skipped` as
    non-blocking, so a round held mid-flight is not charged a failed landing.
    """
    _hold(store)
    posted = _transport(monkeypatch)

    kind, findings, clauses = RP.grade_vault(item_id=4562, paths=["skills/x/SKILL.md"],
                                             diff="+x", vault=vault_tree,
                                             backend=BACKEND, sessions_dir=sessions)

    assert kind == "skipped", findings
    assert "operator pool hold is armed" in findings, findings
    assert kind not in RP.GRADER_FAILURE_KINDS, "a hold is not a grading fault of this item"
    assert clauses == []
    assert _session_files(sessions) == [] and posted == []


# ── the record the operator reads while the machine is free ──────────────────

def test_the_review_store_is_the_one_the_queue_config_names(tmp_path):
    """So the row lands in `workers.db` and not beside the round.

    The gate is a separate process from the backend, so the queue singleton is never
    initialised there and a review that resolved its own path would write a store
    nobody queries — which is half of why `count(*)` stayed 0. It asks
    `workers.queue.configured_db_path`, the accessor the aggregator uses for the same
    reason, and this node fails the day the indirection is repointed at a local path.
    """
    from workers import queue as Q

    resolved = RP._queue_store_path()
    assert resolved == Q.configured_db_path()
    assert resolved.name == "workers.db", (
        f"the review's rows must be countable in the store every source writes, not {resolved}")


def test_the_hold_watermark_literals_are_the_pools_own():
    """The review reads the pool's hold, not a second flag that can fall out of step.

    `scripts/automod/review.py` mirrors the three watermark parts instead of
    importing `workers.pool`, because that module pulls in `app.cost_outlier`,
    `app.engine_pressure`, `app.harness.policy` and `app.sessions_io` at module
    scope and the reviewer must not import the application it judges — the gate runs
    as a separate process from the backend. A mirror with nothing watching it is how
    a reader ends up consulting a key nobody writes: change either side and this is
    the node that says so.
    """
    assert (RP.PAUSE_WM_SOURCE, RP.PAUSE_WM_KEY, RP.PAUSE_WM_HOLDER_KEY) == (
        PAUSE_WM_SOURCE, PAUSE_WM_KEY, PAUSE_WM_HOLDER_KEY), (
        "the review reads a hold watermark the pool does not write, so a mission-control "
        "pause would stop nothing again")
