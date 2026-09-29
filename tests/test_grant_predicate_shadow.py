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

The nodes below `# ── #1834` pin the other half of the same rule: what it says
at mint, when the person who created the shadow is present to read it. Those
cross two further boundaries — the `grant_create` tool result over MCP, and the
`lloyd-harness-policy` log line an autonomy task's `grants:` sync writes — each
by its own production caller, with the interactive-session check satisfied by a
real session file rather than stubbed away.
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


# ── #1834: the same sentence, said at mint ──────────────────────────────────
#
# The nine nodes above pin what a shadow does to a call. These pin what it says
# to the person who created it, because minting is the only moment one of them
# is standing there: `check_grants` answers a dispatch, and a dispatch for that
# tool may not come for a week, or may come and be paid by the bounded row so
# the denial never fires at all. Two surfaces can carry it — the `grant_create`
# tool result and the autonomy task's `grants:` sync — and both are crossed
# below by their own production caller, not by the helper underneath.

def _mint_via_tool(tmp_path, monkeypatch, store: GrantStore,
                   **fields) -> dict:
    """Mint through the real MCP seam and hand back the parsed tool result.

    `agent_mcp.builtin_grants.call_tool` is the process boundary: a human's
    `grant_create` reaches the `agent-stdio` child as JSON and the answer comes
    back as one text block, so the dict parsed here is the text that person
    reads. Nothing in the mint path is stubbed. The session-presence check is
    *satisfied* rather than bypassed — a `mission-control` session file for the
    id the aggregator binds — so `_refusal_if_not_human` runs for real and this
    stays the interactive path the tool's whole design rests on.
    """
    import asyncio
    import json

    from agent_mcp import _task_registry, builtin_grants
    from app import sessions_io

    session = "sess-mint-1834"
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    (sessions / f"{session}.json").write_text(
        json.dumps({"session_id": session, "platform": "mission-control"}),
        encoding="utf-8")
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", sessions)
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))

    token = _task_registry.current_session_id.set(session)
    try:
        result = asyncio.run(builtin_grants.call_tool("grant_create", {
            "scope": SCOPE, "tool": TOOL, "issued_by": "Alan",
            "expires_at": (NOW + dt.timedelta(hours=6)).isoformat(),
            **fields}))
    finally:
        _task_registry.current_session_id.reset(token)
    assert result.is_error is not True, result.content[0].text
    return json.loads(result.content[0].text)


def test_minting_a_bounded_grant_over_a_live_broad_one_warns_the_issuer(
        tmp_path, monkeypatch):
    """Clause 1: the result that says `granted: true` also says what it narrowed.

    One live predicate-less row, then a bounded mint. The mint is not refused —
    bounding a tool is exactly what a human should be able to do — but a human
    who does not see that their predicate now governs the pair will keep
    believing the broad grant they minted last week is what their task runs on.
    """
    from app.harness.policy import shadow_warning_for

    store = _store(tmp_path)
    broad = store.mint(scope=SCOPE, tool_pattern=TOOL, arg_predicate="",
                       issued_by="Alan", expires_at=NOW + dt.timedelta(hours=6))

    out = _mint_via_tool(tmp_path, monkeypatch, store, predicate=PREDICATE)

    assert out.get("granted") is True, out
    new_id = out["grant_id"]
    written = store.get(new_id)
    assert written is not None and written["arg_predicate"] == PREDICATE, (
        "the warning must not stop the write; the bounded row has to be in the "
        f"store; got {written}")
    warn = out.get("shadow_warning")
    assert warn, f"minting a narrowing row over a live broad one must warn: {out}"
    # Names the row the human did not just mint — the one that stopped paying —
    # and the row and bound that took over, so the echo is actionable alone.
    assert f"#{broad['id']}" in warn, warn
    assert f"#{new_id}" in warn and PREDICATE in warn, warn
    assert shadow_warning_for(store, scope=SCOPE, tool_pattern=TOOL) == warn


def test_minting_a_broad_grant_under_a_live_bounded_one_warns_and_is_written(
        tmp_path, monkeypatch):
    """Clause 2: the mint that changes nothing says so, and still writes.

    The denial this human is answering was produced by the bounded row, so
    minting the predicate-less row the denial seems to ask for leaves the
    predicate governing the pair — before and after their mint, byte for byte the
    same answer. `granted: true` alone reports that as a fix. It is not refused:
    revoking or widening the bounded row is the visible act, and refusing the
    mint would be a second, unwritten uniqueness rule on the pair.
    """
    store = _store(tmp_path)
    bounded = store.mint(scope=SCOPE, tool_pattern=TOOL,
                         arg_predicate=PREDICATE, issued_by="Alan",
                         expires_at=NOW + dt.timedelta(hours=6))

    out = _mint_via_tool(tmp_path, monkeypatch, store)

    assert out.get("granted") is True, out
    assert store.get(out["grant_id"]) is not None, (
        "the predicate-less row must still be written — this is a note, not a "
        "refusal")
    warn = out.get("shadow_warning")
    assert warn, (
        f"a predicate-less mint under a live bounded grant #{bounded['id']} "
        f"must say the predicate governs the pair; got {out}")
    assert f"#{bounded['id']}" in warn, warn
    assert PREDICATE in warn, (
        f"the warning must carry the bound that governs the pair, so the reader "
        f"knows what to widen; got: {warn}")


@pytest.mark.parametrize("side", ["bounded", "broad"],
                         ids=["minting_bounded_beside_bounded",
                              "minting_broad_beside_broad"])
def test_a_mint_with_nothing_on_the_other_side_of_the_split_warns_nothing(
        tmp_path, monkeypatch, side):
    """Clause 3: no shadow standing, no warning — in either direction.

    The silence is the point: a warning that fires on every second grant for a
    pair is noise, and noise at mint time is training the reader to ignore the
    one case that matters. A pair holding two bounded rows resolves on the
    shipped expiry-order tie-break with no predicate-less row harmed, and two
    predicate-less rows were always paid in expiry order; neither is a narrowing.

    The check is made twice per case: the tool result for the row just minted,
    and the pair measured before the mint as well, so the existing row's own
    side of the split is seen to say nothing too.
    """
    from app.harness.policy import shadow_warning_for

    store = _store(tmp_path)
    first = (store.mint(scope=SCOPE, tool_pattern=TOOL,
                        arg_predicate="len(messageIds)<=20", issued_by="Alan",
                        expires_at=NOW + dt.timedelta(hours=6))
             if side == "bounded" else
             store.mint(scope=SCOPE, tool_pattern=TOOL, arg_predicate="",
                        issued_by="Alan", expires_at=NOW + dt.timedelta(hours=6)))
    before = shadow_warning_for(store, scope=SCOPE, tool_pattern=TOOL)

    out = _mint_via_tool(tmp_path, monkeypatch, store,
                         **({"predicate": PREDICATE} if side == "bounded"
                            else {}))

    assert out.get("granted") is True, out
    assert store.get(out["grant_id"]) is not None
    assert "shadow_warning" not in out, (
        f"a {side} mint beside a live {side} row has no shadow to name; got: "
        f"{out.get('shadow_warning')}")
    assert before == "", (
        f"the pair must measure as unshadowed from the existing row #{first['id']} "
        f"too; got: {before}")
    assert shadow_warning_for(store, scope=SCOPE, tool_pattern=TOOL) == ""


def test_a_destination_scoped_row_warns_nothing_from_either_side(
        tmp_path, monkeypatch):
    """Clause 4's exclusion, pinned rather than intended: #628 rows are off both lists.

    `store.live()` filters on revocation and expiry only — the destination
    exclusion is `check_grants`' own, applied to its `eligible` list after the
    read. A mint-side detector that scanned `live()` and split on predicate
    alone would name a destination row as narrowing the tool, which is the exact
    claim the shipped rule refuses to make (`#1636`'s
    `test_a_destination_scoped_bounded_grant_stays_outside_the_shadow_rule` says
    that row does not shadow a call). The warning would then be a paraphrase of
    something the gate has already denied.
    """
    from app.harness.policy import shadow_warning_for

    store = _store(tmp_path)
    # A bounded row for one host beside a fresh predicate-less mint.
    dest_bounded = store.mint(scope=SCOPE, tool_pattern=TOOL,
                              arg_predicate=PREDICATE, issued_by="Alan",
                              expires_at=NOW + dt.timedelta(hours=6),
                              destination="example.com")
    out = _mint_via_tool(tmp_path, monkeypatch, store)
    assert out.get("granted") is True, out
    assert "shadow_warning" not in out, (
        f"the destination-bound grant #{dest_bounded['id']} must not be named "
        f"as shadowing a predicate-less row; got: {out.get('shadow_warning')}")

    # And the other way round: a bounded mint beside a destination-bound broad row.
    other = tmp_path / "second"
    other.mkdir(exist_ok=True)
    store2 = _store(other)
    dest_broad = store2.mint(scope=SCOPE, tool_pattern=TOOL, arg_predicate="",
                             issued_by="Alan",
                             expires_at=NOW + dt.timedelta(hours=6),
                             destination="example.com")
    out2 = _mint_via_tool(other, monkeypatch, store2,
                          predicate=PREDICATE)
    assert out2.get("granted") is True, out2
    assert "shadow_warning" not in out2, (
        f"a bounded mint beside the destination-bound #{dest_broad['id']} warns "
        f"nothing; got: {out2.get('shadow_warning')}")

    # A destination-bearing mint itself: the tool has no destination input, so
    # this is the frontmatter/store path asking about its own row.
    dest_mint = store2.mint(scope=SCOPE, tool_pattern=TOOL,
                           arg_predicate="len(messageIds)<=20",
                           issued_by="Alan",
                           expires_at=NOW + dt.timedelta(hours=3),
                           destination="other.example")
    assert dest_mint["destination"] == "other.example"
    assert shadow_warning_for(store2, scope=SCOPE, tool_pattern=TOOL) == "", (
        "a destination row participates on neither side, so a pair holding "
        "nothing but destination rows says nothing")


def test_the_mint_echo_and_a_later_denial_for_one_pair_share_one_sentence(
        tmp_path, monkeypatch):
    """Clause 4: one sentence, not two paraphrases that drift apart.

    The mint echo and the denial are read by the same person at different times,
    and `_explain_shadow` cannot simply be called at mint: it takes the *call*,
    and with `args={}` every bounded row 'could have paid', its swallow guard
    fires, and the text comes back empty. So the sentence was extracted and both
    paths render it. What each adds is only what its own moment knows — the
    denial gets a per-row reason, the echo gets none — and the shared half is
    checked here against the renderer itself, so a reworded copy on one side
    fails rather than reading as a near miss.
    """
    from app.harness.policy import _SHADOW_RULE, _shadow_sentence

    store = _store(tmp_path)
    broad = store.mint(scope=SCOPE, tool_pattern=TOOL, arg_predicate="",
                       issued_by="Alan", expires_at=NOW + dt.timedelta(hours=6))
    out = _mint_via_tool(tmp_path, monkeypatch, store, predicate=PREDICATE)
    warn = out["shadow_warning"]
    bounded = store.get(out["grant_id"])

    d = _call(store, OUT_OF_BOUNDS)
    assert d.allowed is False, (
        f"1200 ids against the row just minted must be denied: {d.reason}")

    named = f"#{bounded['id']} predicate={PREDICATE!r}"
    assert named in warn, warn
    assert named in d.reason, d.reason
    assert f"#{broad['id']}" in warn and f"#{broad['id']}" in d.reason, (
        "both texts must name the same shadowed predicate-less row: "
        f"echo={warn!r} denial={d.reason!r}")
    assert _SHADOW_RULE in warn and _SHADOW_RULE in d.reason, (
        "the rule half must be one string in one place, not two prose copies")
    assert warn == _shadow_sentence(scope=SCOPE, tool=TOOL, bounded=[bounded],
                                   broad=[store.get(broad['id'])]), (
        "the echo is the shared sentence for this pair and nothing else; got "
        f"{warn!r}")
    # The denial carries everything the echo does plus the reason it has: the
    # bracket is the only place the two texts are allowed to differ.
    assert _shadow_sentence(scope=SCOPE, tool=TOOL, bounded=[bounded],
                            broad=[store.get(broad['id'])],
                            reasons={bounded['id']: 'does not match this call'}
                            ) in d.reason, d.reason


def test_sync_task_grants_logs_the_shadow_with_the_task_id_and_still_mints(
        tmp_path, caplog):
    """Clause 5: the frontmatter path has no human present, so server.err does.

    A task declaring a predicate-less `grants:` entry while a bounded row for the
    same pair stands mints a row that changes nothing — `sync_task_grants`
    dedupes on (tool, predicate), so it does not even notice the other row — and
    until now the only thing it ever said was the `not live` warning about an
    expiry in the past. Same sentence as the tool echo, on the logger that
    reaches `server.err`, carrying the task id so a reader of a nightly's log
    can tell which task narrowed what.
    """
    import logging

    from app.harness.policy import _SHADOW_RULE, sync_task_grants

    store = _store(tmp_path)
    bounded = store.mint(scope=SCOPE, tool_pattern=TOOL,
                         arg_predicate=PREDICATE, issued_by="Alan",
                         expires_at=NOW + dt.timedelta(hours=6))
    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")

    minted = sync_task_grants(
        store, task_id=9, scope=SCOPE,
        grants=[{"tool": TOOL, "expires_at":
                 (NOW + dt.timedelta(hours=3)).isoformat(),
                 "issued_by": "Alan", "note": "declared broad"}])

    assert minted == 1, "the declared row must still be minted despite the warning"
    declared = [r for r in store.live(scope=SCOPE, now=NOW)
                if r["tool_pattern"] == TOOL and not r["arg_predicate"]]
    assert len(declared) == 1, (
        f"the predicate-less declared row must be in the store; live rows: "
        f"{[(r['id'], r['arg_predicate']) for r in store.live(scope=SCOPE, now=NOW)]}")

    logged = "\n".join(rec.getMessage() for rec in caplog.records
                       if rec.name == "lloyd-harness-policy"
                       and rec.levelno >= logging.WARNING)
    assert _SHADOW_RULE in logged, (
        f"the task path must log the shared sentence, not a fresh paraphrase; "
        f"warnings were: {logged!r}")
    assert "task #9" in logged, f"the log line must carry the task id: {logged!r}"
    assert f"#{bounded['id']}" in logged, (
        f"the log line must name the bounded row that governs the pair: "
        f"{logged!r}")
    assert f"#{declared[0]['id']}" in logged, (
        f"and the row it just minted, so the two are not guessed apart: "
        f"{logged!r}")
    assert PREDICATE in logged, f"the bound itself has to be in the line: {logged!r}"
