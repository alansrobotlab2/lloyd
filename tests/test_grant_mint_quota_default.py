"""#1946: a grant minted without a quota clears ONE action, not the week.

`grant_create` mints scope + tool + predicate + quota + expiry, and quota is the
mitigation — but a row reading "5 × `email_send` until 2026-10-08" is a standing
licence for five actions granted once. Before this change an omitted quota was
not even a number: `app.harness.policy.GrantStore.mint` validates the quota only
`if quota is not None` and stores NULL otherwise, and every consumption check in
that module — the `check_grants` quota branch among them — is
`if quota is not None and consumed >= quota`, so a NULL row is never over quota
and the grant lasts to its expiry. The tool forwarded the raw `args.get("quota")`
and its own schema prose advertised the standing licence. The denial's rendered
mint line comes from `app.harness.policy.grant_shape` and carries no `quota=` bit
at all, so a human pasting the line the refusal hands them mints exactly that row.

The rule now: at the `grant_create` entry point an omitted quota means 1. The
default sits at that entry point and NOT in `GrantStore.mint`, because
`mint(quota=None)` means *unbounded* to callers that say so deliberately —
`tests/unit/test_grant_policy.py` and `tests/test_egress_policy.py` mint
quota-less rows to exercise multi-call egress, and a default inside the store
would rewrite their intent and fail them. `test_the_store_still_reads_an_explicit_none_as_unbounded`
holds that line.

What does NOT move: explicit quotas, and every row already in the store. The
change adds a default to a missing argument; it never narrows a grant somebody
already relies on — the same instinct as the denial that names the row that
refused instead of quietly tightening it.

Boundaries crossed, each by its production caller rather than by the helper
underneath:

* the `grant_create` MCP call — a human's mint reaches the `agent-stdio` child as
  JSON and the answer comes back as one text block, so `quota` in that block is
  the number that person reads (`_mint_via_tool`).
* the shared grant database — `check_grants` runs in the backend's dispatch hook
  against the same sqlite file the tool wrote, and its denial is the text that
  names the next mint (`test_the_one_quota_grant_clears_one_dispatch...`).
* the committed witness bytes — the figures this item quotes were read off a
  live `~/lloyd-data/workers.db` that has no history, so they are re-derived from
  the extract in the vault, not from a sentence
  (`test_the_committed_witness_bytes_reproduce_the_quoted_counts`).

What this change is: hardening a default before the first paste. It is not a fix
to an observed over-spend — `grant_create` has zero recorded calls across the
session corpus (positive controls in the same scan: Bash 69,984, TodoWrite 153),
and the live store holds no `minted_by='interactive-tool'` row at all. The gate
that reads these rows is live (`egress_events`: 58 `grant-required`), which is
why the default is worth settling now.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

import pytest

from app.harness.policy import (GRANT_MINT_TOOL, GrantStore, check_grants,
                                effective_tier)

#: The wall clock, not a fixed date: expiry is compared against `_now()`, so a
#: hard-coded date would leave every grant minted below expired.
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

#: The pair the item's failure is written against: an unattended worker scope and
#: a tier-2 durable-external tool, which is what the tool's own description
#: offers as the example.
SCOPE = "worker:autocode"
TOOL = "email_send"
CALL_ARGS = {"to": "someone@example.com", "subject": "s", "body": "b"}

LIVE_DB = Path.home() / "lloyd-data" / "workers.db"
WITNESS = Path.home() / "obsidian" / "backlog" / "data" / "workers.db"
MARKER = Path.home() / "obsidian" / "backlog" / "data" / "workers-authority.witness.md"


def _store(tmp_path) -> GrantStore:
    """A real store on a scratch file — never the live `workers.db`."""
    return GrantStore(tmp_path / "grants.db")


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
    from agent_mcp import _task_registry, builtin_grants
    from app import sessions_io

    session = "sess-mint-1946"
    sessions = tmp_path / "sessions"
    sessions.mkdir(exist_ok=True)
    (sessions / f"{session}.json").write_text(
        json.dumps({"session_id": session, "platform": "mission-control"}),
        encoding="utf-8")
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", sessions)
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))

    args: dict = {"scope": SCOPE, "tool": TOOL, "issued_by": "Alan",
                  "expires_at": (NOW + dt.timedelta(hours=6)).isoformat()}
    args.update(fields)
    token = _task_registry.current_session_id.set(session)
    try:
        result = asyncio.run(builtin_grants.call_tool("grant_create", args))
    finally:
        _task_registry.current_session_id.reset(token)
    payload = json.loads(result.content[0].text)
    assert "error" not in payload, payload["error"]
    return payload


def _row(db_path: Path, grant_id: int) -> tuple:
    """The stored row as sqlite returns it — every column, no defaults applied."""
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute(
            "SELECT id, scope, tool_pattern, arg_predicate, quota, consumed,"
            " issued_by, minted_by, note, issued_at, expires_at, revoked_at,"
            " destination FROM authority_grants WHERE id=?",
            (grant_id,)).fetchone()


# ── clause 1: an omitted quota is 1, in the row and in the answer ──────────


def test_minting_with_no_quota_writes_one_and_returns_one(tmp_path, monkeypatch):
    """No `quota` argument → the row says 1 and the result says `"quota": 1`.

    Both halves are the clause. The row is what the gate reads, so NULL there
    means the change did not happen no matter what the tool said; the result is
    what the human reads, and a result saying `null` beside a row saying 1 would
    be a mint echo that lies about the authority it just wrote.
    """
    store = _store(tmp_path)
    out = _mint_via_tool(tmp_path, monkeypatch, store)
    stored = _row(store.db_path, out["grant_id"])

    assert out["quota"] == 1, (
        "the mint echo must name the quota it wrote; a human approving one "
        "action has to be able to see that it is one")
    assert stored[4] == 1, (
        f"authority_grants.quota is {stored[4]!r}; NULL means never over quota, "
        "because every consumption check is `quota is not None and …`")
    assert stored[5] == 0 and stored[11] is None, (
        "a mint must write a fresh, unconsumed, un-revoked row")
    assert out["note"].startswith("This grant dies on its own date"), (
        "the standing-grant sentence is part of the result the clause describes "
        "— the default must not displace it")


# ── clause 2: one execution, then the denial that names the mint call ──────


def test_the_one_quota_grant_clears_one_dispatch_and_the_second_names_the_mint_call(
        tmp_path, monkeypatch):
    """The approval pays for exactly the action it names, and says how to pay again.

    Crosses the second boundary: the tool wrote the row in the `agent-stdio`
    child, and `check_grants` is the backend's dispatch hook reading the same
    file. The first dispatch is allowed and consumes the unit; the second is
    denied, and the denial must carry the verbatim `grant_create(...)` line —
    that naming is what makes per-action friction affordable, which is why the
    item keeps it and why this asserts on it rather than on a bare "no grant".

    The third act is the paste path. `grant_shape` renders no `quota=` bit, so
    the line the denial hands a human is the same line that used to mint an
    unbounded row; minting it back through the tool must now yield quota 1,
    which is how a default at the tool layer fixes the paste without touching
    the renderer.
    """
    store = _store(tmp_path)
    out = _mint_via_tool(tmp_path, monkeypatch, store)
    assert effective_tier(TOOL, CALL_ARGS) == 2, (
        "the premise of the clause: this pair is gated at all")

    first = check_grants(store, scope=SCOPE, tool_name=TOOL,
                         tool_input=CALL_ARGS, now=NOW, record=False)
    assert first.allowed and first.grant_id == out["grant_id"], first.reason
    assert _row(store.db_path, out["grant_id"])[5] == 1, (
        "the allowed dispatch did not consume the unit, so the quota bounds nothing")

    second = check_grants(store, scope=SCOPE, tool_name=TOOL,
                          tool_input=CALL_ARGS, now=NOW, record=False)
    assert not second.allowed, (
        "a second send went out under a one-action approval — the standing "
        "licence this item exists to end")
    assert "over quota (1/1)" in second.reason, second.reason

    expiry = re.search(r"expires_at='([^']+)'", second.reason)
    assert expiry, f"the denial lost the expiry it must name: {second.reason}"
    mint_line = (f"{GRANT_MINT_TOOL}(scope='{SCOPE}', tool='{TOOL}', "
                 f"expires_at='{expiry.group(1)}', issued_by='alan')")
    assert mint_line in second.reason, (
        "the denial must hand over the exact call to paste, in the shape "
        "`grant_shape` renders it")
    assert "quota=" not in mint_line, (
        "the renderer grew a quota bit; the clause names the call WITHOUT one, "
        "and the tool-layer default is what makes that line safe")

    replayed = _mint_via_tool(tmp_path, monkeypatch, store,
                              expires_at=expiry.group(1))
    assert replayed["quota"] == 1, (
        "pasting the denial's own line minted more than one action — the "
        "friction the design buys is one approval per action, so the line a "
        "human is handed must not become a licence")


# ── clause 3: explicit quotas and existing rows do not move ────────────────


def test_an_explicit_quota_is_still_written_as_given_and_no_existing_row_moves(
        tmp_path, monkeypatch):
    """`quota=5` still writes 5, and nothing already standing is narrowed.

    Three rows are in the store before the tool is called at all: one with an
    explicit 5, one NULL row representing a grant minted under the old default,
    and one bounded-by-predicate row. Every one of them is read back as a raw
    sqlite tuple before and after, and the comparison is the whole 13 columns —
    "byte-for-byte" is the clause's word, so a change to `consumed` or to the
    note counts as a failure even though it would look harmless.
    """
    store = _store(tmp_path)
    before_explicit = store.mint(scope=SCOPE, tool_pattern=TOOL, quota=5,
                                 issued_by="Alan", expires_at=NOW + dt.timedelta(days=2),
                                 now=NOW)
    before_null = store.mint(scope=SCOPE, tool_pattern=TOOL, quota=None,
                             issued_by="Alan", expires_at=NOW + dt.timedelta(days=3),
                             note="minted before #1946, no quota given", now=NOW)
    before_pred = store.mint(scope=SCOPE, tool_pattern="email_reply", quota=2,
                             arg_predicate="len(messageIds)<=50",
                             issued_by="Alan", expires_at=NOW + dt.timedelta(days=4),
                             now=NOW)
    snapshot = [_row(store.db_path, r["id"]) for r in
                (before_explicit, before_null, before_pred)]

    out = _mint_via_tool(tmp_path, monkeypatch, store, quota=5)
    assert out["quota"] == 5, "an explicit quota was overwritten by the default"
    assert _row(store.db_path, out["grant_id"])[4] == 5

    assert [_row(store.db_path, r["id"]) for r in
            (before_explicit, before_null, before_pred)] == snapshot, (
        "a row that predates this call changed, so the default narrowed a grant "
        "somebody is already relying on — which is the one thing the change may "
        "not do")


def test_the_store_still_reads_an_explicit_none_as_unbounded(tmp_path):
    """Why the default is at the entry point and not inside `GrantStore.mint`.

    `mint(quota=None)` is what `tests/unit/test_grant_policy.py` and
    `tests/test_egress_policy.py` call to build a multi-call grant, and the
    autonomy `grants:` frontmatter surface still reads an absent `quota:` as
    unbounded — a settled ruling (#1946), not an open question: that
    one-action default lives at the `grant_create` entry point only.
    Defaulting inside the store would rewrite the intent of those callers and
    fail them.
    """
    store = _store(tmp_path)
    row = store.mint(scope=SCOPE, tool_pattern=TOOL, quota=None,
                     issued_by="Alan", expires_at=NOW + dt.timedelta(days=2),
                     now=NOW)
    assert _row(store.db_path, row["id"])[4] is None, (
        "the store's own absence-means-unbounded semantics moved, which is a "
        "different change on a different surface with a ruling still owed")


# ── clause 4: the tool surface and the store agree ─────────────────────────


def test_the_tool_description_says_omitting_the_quota_now_means_one():
    """The schema text is half of the interface, and it used to advertise the licence.

    Both halves of the clause: it must say that omitting yields 1 and why an
    approval must not become a standing licence, and it must no longer tell
    anyone that absence buys an unlimited row. The word the old text used is
    gone from the description entirely, because a sentence that says
    "omitted means X, not unbounded" still teaches the word to a reader skimming
    the parameter list.
    """
    from agent_mcp import builtin_grants

    tools = asyncio.run(builtin_grants.list_tools())
    create = next(t for t in tools if t.name == "grant_create")
    quota = create.input_schema["properties"]["quota"]
    desc = quota["description"]

    assert "unbounded" not in desc.lower(), (
        "the description still advertises the standing licence: " + desc)
    assert "omitting" in desc.lower() and " 1" in desc, (
        "the description must say what an omitted quota buys now: " + desc)
    assert "standing licence" in desc.lower(), (
        "the reason belongs in the text a caller reads before it decides to "
        "pass a number: " + desc)
    assert quota["minimum"] == 1, (
        "the schema floor is what makes quota=0 a refusal rather than a "
        "defaulted 1 — clause 5's other half")


# ── clause 5: quota=0 is a refusal, not a quiet 1 ──────────────────────────


@pytest.mark.parametrize("bad", [0, -3])
def test_a_quota_below_one_is_refused_and_not_rewritten_into_the_default(
        tmp_path, monkeypatch, bad):
    """`quota=0` was refused before this change and must be refused after it.

    The trap the default falls into is `args.get("quota") or 1`, which turns a
    caller's 0 into an accepted 1 — a mint that silently does something other
    than what it was asked, on the one axis that bounds an agent's volume. The
    implementation is an `is None` test, and this is the node that would go red
    if someone tidied it back to truthiness.
    """
    store = _store(tmp_path)
    from agent_mcp import _task_registry, builtin_grants
    from app import sessions_io

    session = "sess-mint-1946-zero"
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "sessions" / f"{session}.json").write_text(
        json.dumps({"session_id": session, "platform": "mission-control"}),
        encoding="utf-8")
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))

    token = _task_registry.current_session_id.set(session)
    try:
        result = asyncio.run(builtin_grants.call_tool("grant_create", {
            "scope": SCOPE, "tool": TOOL, "issued_by": "Alan",
            "quota": bad,
            "expires_at": (NOW + dt.timedelta(hours=6)).isoformat()}))
    finally:
        _task_registry.current_session_id.reset(token)
    payload = json.loads(result.content[0].text)

    assert "error" in payload, f"quota={bad} was accepted: {payload}"
    assert "quota" in payload["error"], payload["error"]
    assert store.live(scope=SCOPE, now=NOW) == [], (
        f"quota={bad} wrote a row anyway — a refused mint must write nothing")


# ── clause 6: the witness has committed bytes behind it ────────────────────


def test_the_committed_witness_bytes_reproduce_the_quoted_counts():
    """The item's figures are re-derived from committed rows, not from a sentence.

    `~/lloyd-data/workers.db` has no history: it is one mutable file the running
    queue writes to, so a count quoted from it is unreproducible the week it is
    written. `backlog/data/workers.db` in the vault is the extract the report was
    read from — every `authority_grants` row and every `grant_dispatch` row, and
    `egress_events` down to the columns the group-by reads, with `destination`,
    `host`, `session_id` and `reason` left out because they name hosts and
    sessions and do not belong in a notes repository.

    The test asserts against the *live* database for the axes that still hold,
    and against the extract for the figures that were a snapshot. Where a count
    has moved since the extract — `allow` is a running total — that is stated,
    not smoothed over: the extract's own marker carries both.
    """
    for path in (WITNESS, MARKER):
        assert path.is_file(), f"missing witness artifact {path}"
    meta = json.loads(MARKER.read_text(encoding="utf-8").split("```json")[1]
                      .split("```")[0])

    with sqlite3.connect(f"file:{WITNESS}?mode=ro", uri=True) as ex:
        grants = list(ex.execute(
            "SELECT scope, tool_pattern, quota, minted_by FROM authority_grants"
            " ORDER BY id"))
        egress = dict(ex.execute(
            "SELECT decision, count(*) FROM egress_events GROUP BY decision"))
        denials = list(ex.execute(
            "SELECT scope, tool, decision FROM grant_dispatch ORDER BY id"))
        tables = {r[0] for r in ex.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

    assert grants == [(meta["authority_grants"][0]["scope"],
                       meta["authority_grants"][0]["tool_pattern"],
                       None, meta["authority_grants"][0]["minted_by"])], (
        f"the extract's grant rows are {grants}, which is not what the marker "
        "records — the failure the item describes is one unbounded row")
    assert grants[0][2] is None, (
        "the witnessed row's quota is no longer NULL, so it is no longer the "
        "example of the thing being fixed")
    assert egress == {k: v for k, v in meta["egress_event_counts"].items()}, (
        f"the extract answers {egress}, the marker records "
        f"{meta['egress_event_counts']}")
    assert denials[0] == ("worker:autocode", "autonomy_write_task", "deny"), (
        "the mint line quoted as the witness is no longer row 1 of grant_dispatch")
    assert {"authority_grants", "grant_dispatch", "egress_events"} <= tables

    # A PREFIX, not a sample. `909 allow / 58 grant-required` is a fact about the
    # first 967 writes the gate ever made only if these rows are the first 967.
    # The pair would look identical over a sampled or renumbered copy, so the
    # marker's `egress_events_cut` is checked against the bytes, contiguously.
    with sqlite3.connect(f"file:{WITNESS}?mode=ro", uri=True) as ex:
        ids = [r[0] for r in ex.execute("SELECT id FROM egress_events ORDER BY id")]
        objects = ex.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
    assert meta["egress_events_cut"] == ids[-1] == 967, (
        f"the marker says the extract stops at id {meta['egress_events_cut']} and "
        f"the bytes stop at {ids[-1]} — the cut is what makes 909/58 the state the "
        "item read rather than a draw from the table")
    assert ids == list(range(1, 968)), (
        f"the extract's ids run 1..{ids[-1]} with gaps or renumbering "
        f"({len(ids)} rows, {len(set(ids))} distinct, min {ids[0]}), so it is a "
        "sample and its decision counts describe no period at all")
    assert objects == meta["sqlite_master_entries"], (
        f"`select count(*) from sqlite_master` answers {objects} over the "
        f"committed bytes and {meta['sqlite_master_entries']} in the marker, so "
        "the re-derivation command the clause names has no figure behind it")
    assert WITNESS.read_bytes()[:3] == b"SQL", "the extract is not a sqlite database"

    # And the same queries over the live database, as the check-after-landing
    # reading. `grant_dispatch` keeps every row it has ever had, so the
    # extract-vs-live comparison that matters is the one the marker says is
    # append-only.
    # A prefix with no hole in it is the property that makes the cut meaningful:
    # "909 of 967 rows are allow" describes the first 967 writes to the live
    # table only if rows 1..967 are all here. Renumber or prune the source and
    # this fails while the counts themselves still look right.
    ids = [r[0] for r in ex.execute("SELECT id FROM egress_events ORDER BY id")]
    assert ids == list(range(1, len(ids) + 1)), (
        f"`egress_events` in the extract is not a contiguous id prefix (n={len(ids)}, "
        f"first={ids[0]}, last={ids[-1]}), so the marker's 'up to id "
        f"{meta['egress_events_cut']}' does not describe these bytes")

    # Then the live table, for one thing only: that the gate which wrote these
    # rows is still writing them. How many tool-minted grants it holds is NOT
    # asserted — the honest answer changes the first time a human mints one, and
    # that is the post-landing reading in the next node, not an invariant here.
    # A data root that has never run the gate has no `egress_events` at all —
    # an automod round home, for instance, where this node runs with a fresh
    # `LLOYD_GRANT_DB`-style store — and the clause's evidence is the committed
    # bytes above, not this comparison, so absence is reported and not invented.
    live = (sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
            if LIVE_DB.is_file() else None)
    tables = ({r[0] for r in live.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
        if live is not None else set())
    if "egress_events" not in tables:
        assert not tables or "authority_grants" in tables, (
            f"{LIVE_DB} holds {sorted(tables)}, which is neither the gate's "
            "tables nor an empty store — the file this witness was copied from "
            "is not the file the test is reading")
    else:
        live_required = live.execute(
            "SELECT count(*) FROM egress_events WHERE decision="
            "'grant-required'").fetchone()[0]
        assert live_required >= egress.get("grant-required", 0), (
            "the live gate has recorded fewer grant-required denials than the "
            f"extract holds ({live_required} < "
            f"{egress.get('grant-required')}), so the table this witness came "
            "from has been rebuilt or pruned and the extract is no longer a "
            "prefix of it")


def test_the_live_check_after_landing_reads_a_one_where_a_row_exists(
        tmp_path, monkeypatch):
    """The item's verify-after-landing query, written so it cannot pass vacuously.

    `select id, quota, minted_by from authority_grants where
    minted_by='interactive-tool'` returns no rows today — nothing has ever been
    minted through the tool — and "no rows" is the answer, not a green light. So
    the clause is exercised the only way it can be: mint through the tool into a
    scratch store, run that exact statement over it and require the 1, then run
    the same statement over the live database and require what it actually says
    today.
    """
    store = _store(tmp_path)
    out = _mint_via_tool(tmp_path, monkeypatch, store)
    for db, want in ((store.db_path, [(out["grant_id"], 1, "interactive-tool")]),
                     (LIVE_DB, [])):
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT id, quota, minted_by FROM authority_grants "
                "WHERE minted_by='interactive-tool'").fetchall()
        assert rows == want, (
            f"{db} answers {rows} for the item's check-after-landing query, "
            + ("expected the mint to read back quota 1, not blank"
               if db == store.db_path else
               "the live store already holds a tool-minted grant, so this node "
               "is stale and the post-landing reading belongs to that row"))


def test_this_change_cites_symbols_and_no_line_number_can_be_stale():
    """A line number in prose is a claim about one tree, and trees move.

    The gate's review rung refuses a round outright when a citation names a line
    past the end of the file it resolved: on the previous commit of this round it
    reported `clause 5: evidence_line 576 past EOF (303 lines) of
    agent_mcp/builtin_grants.py`, and the round was not graded at all. The cure is
    not to cite more carefully — it is to cite the thing that does not move. Every
    reference this change makes is to a symbol (`GrantStore.mint`,
    `check_grants`, `grant_shape`, `args.get("quota")`), so there is no line
    number left to go stale, and this node holds that line.

    Falsifiable both ways: the pattern is proven against a citation known to be
    bad before it is trusted to find none, and a single `path.py:<n>` citation
    added to either file turns this red.
    """
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"([\w./-]+\.(?:py|md)):(\d+)")

    # Positive control first: the scanner must actually fire on the shape it is
    # guarding against, or "found none" means nothing.
    # Assembled at runtime, because writing the shape literally here would trip
    # the very rule this node holds.
    control = pattern.findall("see " + "app/harness/policy" + ".py:57" + "6-579"
                              " and x" + ".md:1" + "2")
    assert control == [("app/harness/policy" + ".py", "576"), ("x" + ".md", "12")], (
        control)

    found = []
    for rel in ("tests/test_grant_mint_quota_default.py",
                "agent_mcp/builtin_grants.py"):
        found += [(rel,) + c for c in pattern.findall(
            (root / rel).read_text(encoding="utf-8"))]
    assert found == [], (
        f"this change cites line numbers {found}; a line number is only true of "
        "one revision, and a grader that opens the file and finds a shorter one "
        "voids the review — name the symbol instead")


# ── #2021: a spent frontmatter-minted grant is visible, with a remedy that works ─

FM_SCOPE = "autonomy-task:40"
FM_TOOL = "email_send"


def _declared(quota, hours):
    return [{"tool": FM_TOOL, "quota": quota, "issued_by": "alan",
             "expires_at": (NOW + dt.timedelta(hours=hours)).isoformat()}]


def _spend_declared_grant(store) -> int:
    from app.harness.policy import sync_task_grants
    assert sync_task_grants(store, task_id=40, scope=FM_SCOPE,
                            grants=_declared(2, 24), now=NOW) == 1
    for _ in range(2):
        assert check_grants(store, scope=FM_SCOPE, tool_name=FM_TOOL,
                            tool_input={}, now=NOW, record=False).allowed
    rows = store.candidates(scope=FM_SCOPE, tool=FM_TOOL, now=NOW)
    assert [(r["quota"], r["consumed"], r["minted_by"], r["revoked_at"])
            for r in rows] == [(2, 2, "frontmatter:40", None)], rows
    return rows[0]["id"]


def _policy_warnings(caplog) -> list[str]:
    import logging
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-harness-policy" and r.levelno >= logging.WARNING
            and "[grants] task #" in r.getMessage()]


def test_a_resync_over_a_spent_declared_grant_mints_nothing_and_says_so(
        tmp_path, caplog):
    """Clause 1. The re-sync returned 0 and logged nothing: the spent row still
    covers its pair, so the task ran denied for the rest of the expiry with no
    line connecting the two."""
    import logging

    from app.harness.policy import sync_task_grants

    store = _store(tmp_path)
    gid = _spend_declared_grant(store)
    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")
    caplog.clear()

    assert sync_task_grants(store, task_id=40, scope=FM_SCOPE,
                            grants=_declared(2, 24), now=NOW) == 0
    assert len(store.candidates(scope=FM_SCOPE, tool=FM_TOOL, now=NOW)) == 1
    warns = _policy_warnings(caplog)
    assert len(warns) == 1, warns
    assert "task #40" in warns[0] and f"grant #{gid}" in warns[0], warns[0]
    assert "over quota (2/2)" in warns[0], warns[0]

    # Not spent, not warned: a covering row with quota left, and an unbounded
    # one, are the ordinary idempotent re-run.
    for task, quota in ((41, 5), (42, None)):
        scope = f"autonomy-task:{task}"
        grants = [{"tool": FM_TOOL, "issued_by": "alan",
                   "expires_at": (NOW + dt.timedelta(hours=24)).isoformat(),
                   **({"quota": quota} if quota else {})}]
        assert sync_task_grants(store, task_id=task, scope=scope,
                                grants=grants, now=NOW) == 1
        assert check_grants(store, scope=scope, tool_name=FM_TOOL, tool_input={},
                            now=NOW, record=False).allowed
        caplog.clear()
        assert sync_task_grants(store, task_id=task, scope=scope,
                                grants=grants, now=NOW) == 0
        assert _policy_warnings(caplog) == []


def test_the_warning_names_a_remedy_that_restores_the_grant_and_it_does(
        tmp_path, caplog):
    """Clause 2 — with the item's own remedy corrected by measurement.

    The item asked the warning to say "raise `quota:` or push `expires_at:`".
    Neither restores anything while behaviour is unchanged: the spent row still
    covers, so the edited block mints 0. Both are run here as the control. What
    does re-mint is the #1949 route — revoke the row, declare a later expiry —
    so that is what the warning names, and it is executed below to show it
    works and that no duplicate was needed."""
    import logging

    from app.harness.policy import sync_task_grants

    store = _store(tmp_path)
    gid = _spend_declared_grant(store)
    old_expiry = store.candidates(scope=FM_SCOPE, tool=FM_TOOL, now=NOW)[0]["expires_at"]
    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")

    for edited in (_declared(5, 24), _declared(5, 48)):
        caplog.clear()
        assert sync_task_grants(store, task_id=40, scope=FM_SCOPE,
                                grants=edited, now=NOW) == 0, (
            "a file edit alone now re-mints over a spent row: the ruling this "
            "item left open has been made, and the remedy text is stale")
        line = _policy_warnings(caplog)[0]
        assert "alone mints nothing over it" in line, line
        assert f"grant_revoke(grant_id={gid})" in line, line
        assert f"`expires_at:` later than {old_expiry}" in line, line
        assert "`40-*.md`" in line and "`quota:`" in line, line
        assert "Do not mint a duplicate" in line, line

    assert store.revoke(gid, now=NOW) is True
    assert sync_task_grants(store, task_id=40, scope=FM_SCOPE,
                            grants=_declared(5, 48), now=NOW) == 1
    assert check_grants(store, scope=FM_SCOPE, tool_name=FM_TOOL, tool_input={},
                        now=NOW, record=False).allowed


#: The sentence that must not come back at the frontmatter `quota:` surface:
#: that default is ruled (#1946), so prose calling it owed re-opens a closed
#: decision. Spelled by concatenation because the guard below reads THIS file's
#: own source, and one literal here would be the very phrase it forbids.
DEFERRAL_PHRASE = "separate" " " "ruling"


def _spent_covering_branch_comment() -> str:
    """The comment block hanging off the `if covering:` branch of
    `sync_task_grants` — where a reader reaches the spent-row rule at the code
    rather than in a docstring.

    The branch is located from the `def sync_task_grants(` line down, and the
    block ends at the first non-comment line, so a comment that is deleted,
    emptied or moved out of the branch fails here instead of reading as a
    passing absence.
    """
    from app.harness import policy

    lines = Path(policy.__file__).read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith("def sync_task_grants(")), None)
    assert start is not None, "sync_task_grants is not where this guard reads it"
    branch = next((i for i, ln in enumerate(lines[start:], start)
                   if ln.strip() == "if covering:"), None)
    assert branch is not None, (
        "the `if covering:` branch is gone from sync_task_grants, so the "
        "spent-row ruling has no home at the code")
    comment: list[str] = []
    for line in lines[branch + 1:]:
        if line.strip().startswith("#"):
            comment.append(line.strip().lstrip("#").strip())
        else:
            break
    assert comment, (
        "the `if covering:` branch carries no comment, so the spent-row ruling "
        "is no longer written where the decision is made")
    return " ".join(comment)


def test_the_spent_covering_branch_states_the_ruling_its_reason_and_the_route():
    """Clauses 1 and 2: the branch comment is the settled #2021 ruling, not an
    open question, and it names the restore route the warning prints."""
    comment = _spent_covering_branch_comment()

    # Clause 1 — the ruling and the reason it is settled.
    assert "#2021" in comment, comment
    assert "still covers" in comment and "mints nothing" in comment, comment
    assert "idempotent and never renewing" in comment, comment
    assert "replenish its own authority" in comment, comment

    # Clause 2 — the route that actually restores the declared grant, which is
    # the one this branch's own warning text gives.
    assert "spent_frontmatter_remedy" in comment, comment
    assert "grant_revoke(grant_id=" in comment, comment
    assert "`expires_at:` later" in comment, comment


def test_the_none_caller_docstring_states_the_1946_rule_it_follows():
    """Clause 3: the reader at `GrantStore.mint(quota=None)` is told the rule,
    not told it is owed.

    `test_the_store_still_reads_an_explicit_none_as_unbounded` is where that
    reader lands, so its own docstring has to carry it — absence-by-design is
    unbounded, and the one-action default lives at the `grant_create` entry
    point only. The guard below can only prove a phrase is absent; this proves
    the ruling is present.
    """
    doc = " ".join(
        (test_the_store_still_reads_an_explicit_none_as_unbounded.__doc__
         or "").split())
    assert "#1946" in doc, doc
    assert "grant_create" in doc, doc
    assert "unbounded" in doc, doc


def test_the_1946_ruling_is_written_where_the_frontmatter_quota_is_read():
    """Clause 4: an omitted frontmatter `quota:` stays unbounded, and no prose
    defers that default at any of the three surfaces a reader meets it on.

    The surfaces are `agent_mcp/builtin_grants.py` (the tool's schema prose),
    `app/harness/policy.py` (the code that reads a `grants:` block, whose
    `if covering:` branch carries the #2021 spent-row ruling) and this file.
    The assertion used to read `builtin_grants` alone — where the phrase has
    never appeared — so it passed vacuously while both other files went
    unchecked, which is the gap #2244 exists to close. Each file therefore has
    to carry a witness string of its own: an absence proves nothing until the
    same read proves it is looking at the right text.
    """
    from agent_mcp import builtin_grants
    from app.harness import policy

    for fn in (policy.validate_task_grants, policy.sync_task_grants):
        doc = " ".join((fn.__doc__ or "").split())
        assert "#1946" in doc and "unbounded" in doc, fn.__name__

    surfaces = [
        (Path(builtin_grants.__file__), "async def _grant_create"),
        (Path(policy.__file__), "def sync_task_grants("),
        (Path(__file__), "def test_the_store_still_reads"),
    ]
    for path, witness in surfaces:
        source = path.read_text(encoding="utf-8")
        assert witness in source, f"{path} is not the text this guard means"
        assert DEFERRAL_PHRASE not in source, (
            f"{path.name} defers the frontmatter `quota:` default again")

    # Control on the control: a mis-assembled needle — a lost space, a typo —
    # would make all three absences above vacuous, so the phrase is pinned
    # word by word rather than against itself.
    assert DEFERRAL_PHRASE.partition(" ") == ("separate", " ", "ruling"), \
        DEFERRAL_PHRASE

    source = Path(builtin_grants.__file__).read_text(encoding="utf-8")
    assert "#1946" in source

    specs, errors = policy.validate_task_grants(
        [{"tool": FM_TOOL, "expires_at": "2099-01-01T00:00:00+00:00",
          "issued_by": "alan"}])
    assert errors == [] and specs[0]["quota"] is None
