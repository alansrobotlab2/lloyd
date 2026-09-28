"""#1636: most-specific-wins grant resolution for one (scope, tool) pair.

A scope may hold several live grants for the same (scope, tool) — `mint()`
inserts with no uniqueness on the pair (`app/harness/policy.py:543-601`), and
`sync_task_grants` matches on the triple (scope, tool, predicate), which
presumes more than one predicate per pair. Before this change resolution was
first-match in `live()`'s `ORDER BY expires_at ASC`, so a predicate-bearing row
could never shadow a predicate-less one: the loop fell through the mismatch and
the broad grant paid. Which row's quota an in-bounds call consumed was decided
by expiry date, not by specificity.

The rule now: for a given (scope, tool), a live predicate-bearing grant makes
its predicate the governing condition for that pair. A call outside the
predicate is denied even while the predicate-less grant for the same pair is
live, and a call inside it is paid by the bounded grant. A pair holding only a
predicate-less grant behaves exactly as it did before, so nothing that exists
today moves: the live store holds one grant and no pair holds two, and
`autonomy/40-nightly-reflection-config.md` is the only task declaring a `grants:`
block, with one entry.

The process boundary this change crosses is the shared grant database:
`check_grants` runs in the backend's dispatch hook, and the same function runs
in the `agent-stdio` child through `agent_mcp.egress.grant_covers`, both opening
one sqlite file. Every test here uses a real `GrantStore` on a scratch file, so
the row it asserts on is a row that was written and read back;
`test_the_egress_guard_answering_from_the_same_pair_is_unchanged` crosses the
production caller itself, on the destination axis #628 owns.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.harness.policy import GrantStore, check_grants

#: The wall clock, not a fixed date: expiry is compared against `_now()`, so a
#: hard-coded date would leave every grant minted below expired.
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

#: The pair the item's reproduction is written against.
SCOPE = "autonomy-task:9"
TOOL = "email_update"
#: The bounded grant's own bound, and the two call sizes either side of it.
BOUND = 50
OUT_OF_BOUNDS = [f"msg-{i}" for i in range(1200)]
IN_BOUNDS = [f"msg-{i}" for i in range(10)]
PREDICATE = f"len(messageIds)<={BOUND}"


def _store(tmp_path) -> GrantStore:
    """A real store on a scratch file — never the live `workers.db`."""
    return GrantStore(str(tmp_path / "grants.sqlite"))


def _pair(store: GrantStore, *, tight_first: bool, quota: int | None = None,
          predicate: str = PREDICATE, tool: str = TOOL,
          destination: str | None = None) -> tuple[dict, dict]:
    """Mint the (broad, bounded) pair for one (scope, tool).

    `tight_first` flips which row `live()`'s `ORDER BY expires_at ASC` puts
    first, which before this change was the only thing deciding the answer.
    """
    tight_expiry = NOW + dt.timedelta(hours=1 if tight_first else 6)
    loose_expiry = NOW + dt.timedelta(hours=6 if tight_first else 1)
    broad = store.mint(scope=SCOPE, tool_pattern=tool, arg_predicate="",
                       issued_by="Alan", expires_at=loose_expiry)
    bounded = store.mint(scope=SCOPE, tool_pattern=tool,
                         arg_predicate=predicate, quota=quota,
                         issued_by="Alan", expires_at=tight_expiry,
                         destination=destination)
    return broad, bounded


def _call(store: GrantStore, ids: list[str], tool: str = TOOL):
    return check_grants(store, scope=SCOPE, tool_name=tool,
                        tool_input={"messageIds": ids}, record=False, now=NOW)


# ── clause 1: outside the predicate, the call is denied whichever row expires
# first, and the reason says which bounded grant refused it ──────────────────

@pytest.mark.parametrize("tight_first", [True, False],
                         ids=["bounded_expires_first", "broad_expires_first"])
def test_call_outside_the_predicate_is_denied_in_either_expiry_order(
        tmp_path, tight_first):
    """1200 ids against a live `len(messageIds)<=50` grant: denied, both orders.

    Before the change this returned `allowed=True grant_id=<the broad row>` for
    `tight_first=True` (the bounded row was skipped on predicate mismatch and
    the broad row paid) and `allowed=True grant_id=<the broad row>` immediately
    for `tight_first=False`. The expiry order no longer decides anything.
    """
    store = _store(tmp_path)
    broad, bounded = _pair(store, tight_first=tight_first)

    d = _call(store, OUT_OF_BOUNDS)

    assert d.allowed is False, (
        f"1200 messageIds must not be paid for by the predicate-less grant "
        f"#{broad['id']} while the bounded grant #{bounded['id']} "
        f"({PREDICATE}) is live; got allowed={d.allowed} "
        f"grant_id={d.grant_id}")
    assert d.grant_id is None, "a denial must not name a grant that paid"
    # The operator has to see WHY the live broad grant did not pay: the deny
    # reason names the bounded grant's predicate, and both rows by id.
    assert PREDICATE in d.reason, (
        f"deny reason must name the bounded grant's predicate, got: {d.reason}")
    assert str(bounded["id"]) in d.reason, (
        f"deny reason must name the bounded grant #{bounded['id']}, "
        f"got: {d.reason}")
    assert str(broad["id"]) in d.reason, (
        f"deny reason must name the shadowed predicate-less grant "
        f"#{broad['id']}, got: {d.reason}")


# ── clause 2: inside the predicate, the bounded grant pays and the broad
# grant's quota is untouched ────────────────────────────────────────────────

@pytest.mark.parametrize("tight_first", [True, False],
                         ids=["bounded_expires_first", "broad_expires_first"])
def test_call_inside_the_predicate_is_paid_by_the_bounded_grant(
        tmp_path, tight_first):
    """10 ids: `grant_id` is the bounded row, and only its `consumed` moves."""
    store = _store(tmp_path)
    broad, bounded = _pair(store, tight_first=tight_first)

    d = _call(store, IN_BOUNDS)

    assert d.allowed is True, f"10 ids is inside {PREDICATE}: {d.reason}"
    assert d.grant_id == bounded["id"], (
        f"the call must be paid by the bounded grant #{bounded['id']}, not by "
        f"the predicate-less #{broad['id']}; got grant_id={d.grant_id} "
        f"(expiry order no longer picks the payer)")
    assert store.get(bounded["id"])["consumed"] == 1
    assert store.get(broad["id"])["consumed"] == 0, (
        "the shadowed grant's quota must not be consumed by a call the bounded "
        "grant paid for")


# ── clause 3: the rule is per (scope, tool), and changes nothing else ───────

def test_a_lone_predicate_less_grant_still_pays_a_1200_id_call(tmp_path):
    """No predicate-bearing row for the pair → today's behaviour, unchanged."""
    store = _store(tmp_path)
    broad = store.mint(scope=SCOPE, tool_pattern=TOOL, arg_predicate="",
                       issued_by="Alan", expires_at=NOW + dt.timedelta(hours=6))

    d = _call(store, OUT_OF_BOUNDS)

    assert d.allowed is True, (
        "a scope holding only a predicate-less grant must be unaffected by the "
        f"shadow rule; got: {d.reason}")
    assert d.grant_id == broad["id"]
    assert store.get(broad["id"])["consumed"] == 1


def test_a_bounded_grant_for_another_tool_does_not_shadow_this_one(tmp_path):
    """The shadow is scoped to one (scope, tool) pair, not to the scope."""
    store = _store(tmp_path)
    broad = store.mint(scope=SCOPE, tool_pattern=TOOL, arg_predicate="",
                       issued_by="Alan", expires_at=NOW + dt.timedelta(hours=6))
    store.mint(scope=SCOPE, tool_pattern="email_delete",
               arg_predicate=PREDICATE, issued_by="Alan",
               expires_at=NOW + dt.timedelta(hours=1))

    d = _call(store, OUT_OF_BOUNDS)

    assert d.allowed is True, (
        "a predicate-bearing row for `email_delete` must not shadow the "
        f"predicate-less row for `email_update`; got: {d.reason}")
    assert d.grant_id == broad["id"]
    assert store.get(broad["id"])["consumed"] == 1


# ── the semantics the rule has to state, pinned so the next reader cannot
# guess which way it went ────────────────────────────────────────────────────

def test_a_spent_bounded_grant_does_not_hand_the_call_back_to_the_broad_one(
        tmp_path):
    """Quota exhaustion narrows nothing: the predicate keeps governing the pair.

    The bounded grant carries `quota=1`. Its one call is spent, and the pair is
    left holding a bounded row that cannot pay and a broad row it shadows. The
    broad row must NOT spring back to life: falling back to it would let
    spending a quota *widen* the capability — every call, in or out of the
    predicate, is denied while the narrowing row stands.
    """
    store = _store(tmp_path)
    broad, bounded = _pair(store, tight_first=True, quota=1)

    first = _call(store, IN_BOUNDS)
    assert first.allowed is True and first.grant_id == bounded["id"]

    second = _call(store, IN_BOUNDS)
    assert second.allowed is False, (
        "with the bounded grant's quota spent, the predicate-less grant must "
        f"not resume paying; got grant_id={second.grant_id}")
    assert str(bounded["id"]) in second.reason
    assert store.get(broad["id"])["consumed"] == 0

    out_of_bounds = _call(store, OUT_OF_BOUNDS)
    assert out_of_bounds.allowed is False
    assert PREDICATE in out_of_bounds.reason


def test_a_destination_scoped_bounded_grant_stays_outside_the_shadow_rule(
        tmp_path):
    """#628's axis is answered by `_decide_destination`, and is not a shadow.

    A grant minted *for a destination* is not a licence for the tool generally
    (`check_grants` skips destination rows in the tier-2 match loop), so it must
    not narrow the tool generally either: the predicate-less row still pays an
    ordinary tier-2 call with no destination.
    """
    store = _store(tmp_path)
    broad, dest_bounded = _pair(store, tight_first=True,
                                destination="example.com")

    d = _call(store, OUT_OF_BOUNDS)

    assert d.allowed is True, (
        f"the destination-scoped grant #{dest_bounded['id']} must not shadow "
        f"the predicate-less #{broad['id']}; got: {d.reason}")
    assert d.grant_id == broad["id"]


def test_the_egress_guard_answering_from_the_same_pair_is_unchanged(
        tmp_path, monkeypatch):
    """The child-process caller is unaffected: it always names a destination.

    `agent_mcp.egress.grant_covers` is the second process's route into
    `check_grants` — same function, same sqlite file, `destination=<host>` and
    `tool_input={}` — and it answers on the destination axis alone. A
    predicate-bearing destination grant and a predicate-less one for the same
    pair must not change what the guard says about a host it does not cover.
    """
    from agent_mcp import egress

    path = tmp_path / "workers.db"
    monkeypatch.setenv("LLOYD_GRANT_DB", str(path))
    store = GrantStore(str(path))
    _pair(store, tight_first=True, tool="http_fetch",
          destination="docs.example.org")

    authorized, reason = egress.grant_covers(
        scope=SCOPE, tool="http_fetch", host="collector.evil.example", at=NOW)

    assert authorized is False, (
        "a grant for docs.example.org must not authorize a fetch to "
        f"collector.evil.example: {reason}")
    authorized_ok, reason_ok = egress.grant_covers(
        scope=SCOPE, tool="http_fetch", host="docs.example.org", at=NOW)
    assert authorized_ok is True, (
        f"the covered host must still be authorized: {reason_ok}")
