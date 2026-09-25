"""#783: the usage counter's skill dimension.

Before this, `usage.db` could say what a *model* cost and nothing about which
skill body was in front of the model when it was spent: the `usage` table had
fourteen columns and none of them was skill- or route-related, `model_breakdown`
grouped by model alone, and the dashboard published only `by_model_24h`. Per-skill
cost existed in no artifact at all — the bench-side axis #783's framing assumed
(`body_delivered_by`) has never been in the tree.

Three shapes decide whether the new column means anything, and each is pinned
here rather than asserted in prose:

* **NULL, not an empty list, for "no skill".** A turn that delivered no skill and
  a turn whose writer never learned to record one have to stay distinguishable,
  the same rule `prefix_misses` follows (`usage_store.py:184-190`).
* **Token-denominated, never dollars.** `cost_usd` is written `0.0` at every
  write site (`app/routers/messages.py` 1199/2287 as read at triage, and
  `app/run_recorder.py`), so a per-skill cost in dollars would read 0 for every
  skill and look like a clean bill.
* **One parser.** The names come from `skill_dispatch`'s existing skill-tag walk,
  so a second regex cannot drift from the one that decides IV de-duplication.
  Pinned by muting that module's regex and requiring both readers to go blind.

Not instrumented by this change, and named so the next reader does not assume it:
`app/run_recorder.py`'s background-run row (post-landing on #783), the
`skills_read` tool path, and the `dispatch` route, whose
`harness.skill_dispatch.enabled` key is absent from `config.yaml` and is a
human edit.
"""

from __future__ import annotations

import ast
import json
import sqlite3

import usage_store
from app.harness import skill_dispatch as sd

#: The `usage` table as it stood before #783: the 14 columns of
#: `usage_store._init_schema` including the prefix-miss pair, no skill column.
#: Written here as DDL so the migration test starts from a real old file rather
#: than from the current schema with a column dropped by hand.
LEGACY_USAGE_DDL = """
CREATE TABLE usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    session_id      TEXT,
    model           TEXT,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_create    INTEGER NOT NULL DEFAULT 0,
    cache_read      INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL,
    duration_ms     INTEGER,
    duration_api_ms INTEGER,
    num_turns       INTEGER,
    reprefill_tokens INTEGER,
    prefix_misses    INTEGER
);
"""

#: The two skills a rendered turn's context carries: the top hit, whose body is
#: injected whole, and the runner-up, which arrives as an excerpt.
FULL_SKILL = "web-search-and-fetch"
EXCERPT_SKILL = "youtube-transcript"


def _real_prefetch_text(first_body: str = "body\n",
                        second_body: str = "excerpt\n") -> str:
    """The turn's context, rendered by the renderer that actually builds it.

    Every route assertion below reads this, so the markup is `prefetch`'s and not
    a copy: the attribute that separates `prefetch` from `prefetch_excerpt` is
    written at `prefetch.py:941` and nowhere else in this file. A hand-written
    fixture would keep passing after a renderer change that dropped or renamed
    `excerpt="true"` — the recorded route would silently call every excerpt a
    full body, which is the mislabel this column exists to remove.
    """
    import prefetch

    return prefetch._format_context(
        [
            (29.7, {"name": FULL_SKILL, "raw": first_body}),
            (4.1, {"name": EXCERPT_SKILL, "raw": second_body}),
        ],
        [],
        vault_results=[], session_results=[],
        ambient_entries=[], backlog_refs=[],
    )


PREFETCH_TEXT = _real_prefetch_text()


def test_the_fixture_is_the_real_render_and_labels_the_two_routes():
    """Positive control for deriving the fixture: `_format_context` still emits
    both tag forms, so a `prefetch_excerpt` row below is evidence about the
    product and not a tautology over this file's own string."""
    assert '<context>' in PREFETCH_TEXT and PREFETCH_TEXT.rstrip().endswith('</context>')
    assert f'<skill name="{FULL_SKILL}" score="29.7">' in PREFETCH_TEXT, PREFETCH_TEXT
    assert f'<skill name="{EXCERPT_SKILL}" score="4.1" excerpt="true">' in PREFETCH_TEXT, (
        PREFETCH_TEXT
    )


def _row_skills(session_id: str) -> list[dict]:
    """The stored `skills` value for one session, decoded."""
    row = usage_store._conn().execute(
        "SELECT skills FROM usage WHERE session_id = ?", (session_id,)
    ).fetchone()
    raw = row["skills"]
    return [] if raw is None else json.loads(raw)


# ── clause 1: the column, written and read back ─────────────────────────


def test_a_row_written_with_one_skill_reads_back_that_skill_and_its_route():
    usage_store.record_usage(
        session_id="s-one", model="primary", input_tokens=10, output_tokens=1,
        skills=[{"name": "youtube-transcript", "route": "prefetch"}],
    )
    assert _row_skills("s-one") == [
        {"name": "youtube-transcript", "route": "prefetch"}
    ]


def test_a_turn_that_delivered_no_skill_stores_null_not_an_empty_list():
    """NULL is the difference between "no skill was delivered" and "nobody
    recorded". The positive control is the sibling row in the same file: the
    NULL below cannot be a writer that never persists the column."""
    usage_store.record_usage(session_id="s-none", model="primary",
                             input_tokens=10, output_tokens=1)
    usage_store.record_usage(session_id="s-empty", model="primary",
                             input_tokens=10, output_tokens=1, skills=[])
    usage_store.record_usage(session_id="s-skill", model="primary",
                             input_tokens=10, output_tokens=1,
                             skills=[{"name": "youtube-transcript",
                                      "route": "prefetch"}])
    rows = usage_store._conn().execute(
        "SELECT session_id, skills FROM usage ORDER BY session_id"
    ).fetchall()
    assert {r["session_id"]: r["skills"] for r in rows} == {
        "s-empty": None,
        "s-none": None,
        "s-skill": json.dumps([{"name": "youtube-transcript",
                               "route": "prefetch"}], sort_keys=True),
    }


def test_a_bare_skill_name_is_refused_rather_than_stored_as_garbage():
    """`injected_skill_names()` returns a set of names, and a caller that hands
    *that* to `record_usage` must not have `"alpha"` stored as
    `{"name": "a", "route": "l"}` — a `str` iterates its own characters. The
    pair form and a name without a route are both real records, so both stay."""
    usage_store.record_usage(session_id="bad-set", model="primary",
                             input_tokens=5, skills={"alpha", "beta"})
    usage_store.record_usage(session_id="bad-pairs", model="primary",
                             input_tokens=5, skills=[("alpha", "prefetch")])
    usage_store.record_usage(session_id="bad-routeless", model="primary",
                             input_tokens=5, skills=[{"name": "gamma"}])
    rows = {r["session_id"]: r["skills"] for r in usage_store._conn().execute(
        "SELECT session_id, skills FROM usage").fetchall()}
    assert rows["bad-set"] is None, (
        f"a set of bare names must store nothing, not a mangled name: {rows['bad-set']}"
    )
    assert json.loads(rows["bad-pairs"]) == [
        {"name": "alpha", "route": "prefetch"}]
    assert json.loads(rows["bad-routeless"]) == [{"name": "gamma", "route": ""}]


def test_an_existing_usage_db_keeps_every_row_and_gains_nulls(tmp_path, monkeypatch):
    """The column arrives through the additive `ALTER TABLE` path
    (`usage_store.py:103-106`), so a live `usage.db` is never recreated.
    Reproduced from the real file: 3 rows written under the old DDL by plain
    sqlite3, then handed to the store, which must keep them and add the column."""
    db = tmp_path / "usage-legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(LEGACY_USAGE_DDL)
    for i in range(3):
        conn.execute(
            "INSERT INTO usage (session_id, model, input_tokens, output_tokens)"
            " VALUES (?, 'primary', ?, ?)", (f"legacy-{i}", 100 + i, 5 + i),
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr(usage_store, "DB_PATH", db)

    total = usage_store.summary()
    assert total["requests"] == 3, (
        "an old usage.db must keep every row across the column add; summary "
        f"counts {total['requests']}"
    )
    cols = {r["name"] for r in usage_store._conn().execute(
        "PRAGMA table_info(usage)").fetchall()}
    assert "skills" in cols, f"the additive ALTER added nothing: {sorted(cols)}"
    legacy = usage_store._conn().execute(
        "SELECT skills FROM usage WHERE session_id LIKE 'legacy-%'").fetchall()
    assert [r["skills"] for r in legacy] == [None, None, None], (
        "pre-column rows must read NULL, not an empty list or a forged zero"
    )

    # And the migrated file accepts a write on the new column.
    usage_store.record_usage(session_id="post-migration", model="primary",
                             input_tokens=1, skills=[{"name": "youtube-transcript",
                                                      "route": "prefetch"}])
    assert _row_skills("post-migration") == [
        {"name": "youtube-transcript", "route": "prefetch"}
    ]


# ── clause 2: the per-skill breakdown ───────────────────────────────────


def _seed_window_rows():
    """Four rows: two skills on one turn, the same skill on a second turn, a
    skill on another model, and an out-of-window row that must not count.

    Tokens are the only magnitude the store carries, so every number below is
    tokens (`cost_usd` is written 0.0 at the call sites).
    """
    usage_store.record_usage(session_id="w1", model="primary",
                             input_tokens=100, output_tokens=10,
                             cache_create=5, cache_read=50,
                             skills=[{"name": "alpha", "route": "prefetch"}])
    usage_store.record_usage(session_id="w2", model="primary",
                             input_tokens=200, output_tokens=20,
                             cache_create=7, cache_read=80,
                             skills=[{"name": "alpha", "route": "prefetch"},
                                     {"name": "beta", "route": "prefetch_excerpt"}])
    usage_store.record_usage(session_id="w3", model="eco",
                             input_tokens=400, output_tokens=40,
                             cache_create=0, cache_read=0,
                             skills=[{"name": "alpha", "route": "prefetch"}])
    usage_store.record_usage(session_id="w4", model="primary",
                             input_tokens=999, output_tokens=99,
                             cache_create=9, cache_read=9)  # no skill delivered
    conn = usage_store._conn()
    conn.execute(
        "INSERT INTO usage (ts, session_id, model, input_tokens, output_tokens,"
        " skills) VALUES ('2020-01-01T00:00:00', 'w5-out-of-window', 'primary',"
        " 1234, 12, ?)",
        (json.dumps([{"name": "alpha", "route": "prefetch"}], sort_keys=True),),
    )
    conn.commit()


def test_skill_breakdown_sums_tokens_per_skill_and_route():
    _seed_window_rows()
    rows = usage_store.skill_breakdown(hours=24)
    assert rows == [
        {"skill": "alpha", "route": "prefetch", "requests": 3,
         "input_tokens": 700, "output_tokens": 70,
         "cache_create": 12, "cache_read": 130},
        {"skill": "beta", "route": "prefetch_excerpt", "requests": 1,
         "input_tokens": 200, "output_tokens": 20,
         "cache_create": 7, "cache_read": 80},
    ], rows


def test_skill_breakdown_carries_no_dollar_figure():
    """`cost_usd` is 0.0 at every write site, so a dollar column would be a
    column of zeros that reads as "these skills cost nothing".

    The row set is asserted non-empty before its keys are swept: a `for` loop
    over an empty list passes, and an empty breakdown would otherwise be reported
    here as "no dollar column", which is a verdict this test cannot make.
    """
    _seed_window_rows()
    rows = usage_store.skill_breakdown(hours=24)
    assert rows, "the seeded window must produce skill rows before keys are swept"
    for row in rows:
        assert set(row) == {"skill", "route", "requests", "input_tokens",
                            "output_tokens", "cache_create", "cache_read"}


def test_skill_breakdown_counts_the_same_rows_as_model_breakdown():
    """Same window, same exclusions: the two breakdowns must disagree about the
    dimension and never about which usage rows exist."""
    _seed_window_rows()
    models = usage_store.model_breakdown(hours=24)
    assert sum(r["requests"] for r in models) == 4, (
        "the window holds 4 rows (w1-w4); w5 is dated 2020 and must be out of "
        f"both breakdowns: {models}"
    )
    assert usage_store.summary(hours=24)["requests"] == 4

    assert {r["model"] for r in models} == {"primary", "eco"}
    assert {r["model"] for r in usage_store.model_breakdown(
        hours=24, exclude_models=["eco"])} == {"primary"}
    # The same exclusion moves the skill breakdown: w3's 400 input tokens are
    # the only ones `alpha` loses, so the two views cannot drift apart.
    assert [r for r in usage_store.skill_breakdown(hours=24,
                                                   exclude_models=["eco"])] == [
        {"skill": "alpha", "route": "prefetch", "requests": 2,
         "input_tokens": 300, "output_tokens": 30,
         "cache_create": 12, "cache_read": 130},
        {"skill": "beta", "route": "prefetch_excerpt", "requests": 1,
         "input_tokens": 200, "output_tokens": 20,
         "cache_create": 7, "cache_read": 80},
    ]
    # The out-of-window row names `alpha` with 1,234 input tokens; if the skill
    # breakdown used its own window it would read 1,934 here. The membership
    # check first: an empty breakdown would satisfy `all(...)` vacuously, which
    # is an empty verdict rather than a passing one.
    windowed = usage_store.skill_breakdown(hours=24)
    assert "alpha" in {r["skill"] for r in windowed}, windowed
    assert all(r["input_tokens"] < 1900 for r in windowed), windowed


def test_an_unmeasured_turn_is_absent_from_the_breakdown_not_zeroed():
    """w4 delivered no skill. It stays a row in `model_breakdown` and appears in
    no skill row — a `NULL`-skill turn must not become a skill named ""."""
    _seed_window_rows()
    skills = {r["skill"] for r in usage_store.skill_breakdown(hours=24)}
    assert "" not in skills and None not in skills
    assert sum(r["requests"] for r in usage_store.model_breakdown(hours=24)) == 4


# ── clause 3: one parser, and the route it records ──────────────────────


def test_deliveries_come_from_the_one_skill_tag_walk(monkeypatch):
    """`injected_skill_names` is a projection of the route-aware walk, not a
    sibling regex. Muting the module's single skill-tag regex must blind both:
    a second parser would keep reporting skills here."""
    assert sd.skill_deliveries(PREFETCH_TEXT) == [
        {"name": "web-search-and-fetch", "route": "prefetch"},
        {"name": "youtube-transcript", "route": "prefetch_excerpt"},
    ]
    assert sd.injected_skill_names(PREFETCH_TEXT) == {
        "web-search-and-fetch", "youtube-transcript"
    }
    assert sd.skill_deliveries("") == []
    assert sd.injected_skill_names("") == set()

    class _Blind:
        @staticmethod
        def finditer(_text):
            return iter(())

    monkeypatch.setattr(sd, "_SKILL_TAG_RE", _Blind())
    assert sd.skill_deliveries(PREFETCH_TEXT) == []
    assert sd.injected_skill_names(PREFETCH_TEXT) == set()


def test_an_excerpt_render_is_a_different_route_from_a_full_body():
    """`excerpt="true"` is one line of a protocol, not the protocol: a turn that
    saw the excerpt is not evidence the model had the procedure."""
    routes = {d["name"]: d["route"] for d in sd.skill_deliveries(PREFETCH_TEXT)}
    assert routes == {"web-search-and-fetch": "prefetch",
                      "youtube-transcript": "prefetch_excerpt"}
    assert sd.ROUTE_PREFETCH == "prefetch"
    assert sd.ROUTE_PREFETCH_EXCERPT == "prefetch_excerpt"


def test_a_skill_named_in_both_renders_is_recorded_once_per_route():
    """A repeated name/route pair would double the `requests` count for a turn
    that only spent its tokens once."""
    text = ('<skill name="alpha" score="1.0">\nb\n</skill>\n'
            '<skill name="alpha" score="1.0">\nb again\n</skill>')
    assert sd.skill_deliveries(text) == [{"name": "alpha", "route": "prefetch"}]


def test_a_prefetched_turn_produces_a_usage_row_naming_skill_and_route():
    """The acceptance check in miniature: render → parse → store → breakdown,
    through the real SQLite file conftest points at scratch."""
    usage_store.record_usage(session_id="live-turn", model="primary",
                             input_tokens=500, output_tokens=25,
                             cache_create=0, cache_read=400,
                             skills=sd.skill_deliveries(PREFETCH_TEXT))
    assert usage_store.skill_breakdown(hours=24) == [
        {"skill": "web-search-and-fetch", "route": "prefetch", "requests": 1,
         "input_tokens": 500, "output_tokens": 25,
         "cache_create": 0, "cache_read": 400},
        {"skill": "youtube-transcript", "route": "prefetch_excerpt", "requests": 1,
         "input_tokens": 500, "output_tokens": 25,
         "cache_create": 0, "cache_read": 400},
    ]


# ── the chat-turn write sites actually pass the deliveries ───────────────


def _messages_tree():
    """The router's own source as an AST. Read off the file rather than the live
    module attributes, because `tests/test_session_queue.py` rebinds
    `msg_mod._run_turn` to a stub and a full-suite run would otherwise grade that
    fake; and parsed rather than searched as text, because a parser cannot be
    fooled by a commented-out writer."""
    import inspect as _inspect

    import app.routers.messages as msg

    return ast.parse(open(_inspect.getsourcefile(msg), encoding="utf-8").read())


def _functions(tree) -> dict:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _record_usage_calls(node) -> list:
    """Every live `usage_store.record_usage(...)` call inside `node`."""
    found = []
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "record_usage"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "usage_store"):
            found.append(sub)
    return found


def _skill_argument(call) -> str:
    """The rendered `skills=` argument of one call, or "" when it has none."""
    for kw in call.keywords:
        if kw.arg == "skills":
            return ast.unparse(kw.value)
    return ""


def test_every_chat_turn_usage_write_site_names_the_skill_deliveries():
    """`_run_turn` and `post_message` are the two chat paths that write a usage
    row, and each must hand its turn's prefetched text to the parser. The count
    is pinned too: a new `record_usage` call added in either function without
    `skills=` fails here, as does dropping it from one of the three.

    What this test can prove and what it cannot are both worth stating: it grades
    the call's shape — that the argument is present, spelled so, and not inside a
    comment — and it cannot prove the names it reads resolve at runtime. An
    undefined name there is swallowed by the turn's own `except Exception`, which
    logs a warning and drops the row, so the shape check is paired with the
    name-resolution check below and with the driven turns further down this file.

    `app/run_recorder.py`'s background-run row is deliberately not covered —
    #783 lists it as post-landing work, so its absence is stated, not asserted.
    """
    tree = _messages_tree()
    functions = _functions(tree)

    # The parser must be reachable by name in the router's own namespace.
    imported = {
        (alias.asname or alias.name.split(".")[0])
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "skill_deliveries" in imported, (
        "`skill_deliveries` is not imported in messages.py, so every writer "
        "below raises NameError into its own `except` and books nothing"
    )

    total_calls = 0
    for name in ("_run_turn", "post_message"):
        assert name in functions, f"{name} is gone from the router — test is stale"
        # The turn's prefetched text must be bound in the function that writes the
        # row: each writer reads a local, so a rename or a moved assignment makes
        # the argument a NameError and the turn silently skill-less.
        bound = {
            node.arg
            for a in (functions[name].args.posonlyargs + functions[name].args.args
                      + functions[name].args.kwonlyargs)
            for node in [a]
        } | {
            t.id
            for sub in ast.walk(functions[name])
            for t in ([sub] if isinstance(sub, ast.Name)
                      and isinstance(sub.ctx, ast.Store) else [])
        }
        assert "prefetched_text" in bound, (
            f"{name} passes `prefetched_text` to the parser but never binds that "
            "name in this function"
        )
        calls = _record_usage_calls(functions[name])
        assert calls, f"{name} writes no usage row — the test is stale"
        for call in calls:
            assert _skill_argument(call) == "skill_deliveries(prefetched_text)", (
                f"{name} line {call.lineno} has a record_usage call that names no "
                f"skill deliveries (skills={_skill_argument(call)!r})"
            )
        total_calls += len(calls)
    assert total_calls == 3, (
        f"expected the 3 known chat-turn write sites (two in _run_turn: the "
        f"result path and the error path; one in post_message), found "
        f"{total_calls} — update this test and #783 together"
    )


# ---------------------------------------------------------------------------
# The seam the writer test cannot prove: driving the real turn function.
# ---------------------------------------------------------------------------

def _fake_result_event(**over):
    """The `result` event the harness loop ends a turn with, in the shape
    `_run_turn` reads: `usage` for the tokens, `response_text` for the answer."""
    evt = {
        "type": "result",
        "usage": {"input_tokens": 4000, "output_tokens": 50,
                  "cache_read": 3000, "cache_create": 100},
        "stop_reason": "stop",
        "duration_ms": 1200,
        "num_turns": 2,
        "response_text": "done",
    }
    evt.update(over)
    return evt


def _pristine_messages_module(monkeypatch):
    """A private second copy of `app/routers/messages.py` to drive.

    `tests/test_session_queue.py` rebinds `messages._run_turn` to its own recording
    stub at import — that file's comment calls a fake `messages.py` "the only other
    way to exercise the consumer" — so in a full-suite run the live attribute is
    somebody else's fake, which writes no usage row whatsoever. Driving it would
    fail these tests for a reason that has nothing to do with this change, and
    quietly skipping the stub would make them vacuous.

    The mechanism lives in `tests/_messages_copy.py` now, shared with
    `tests/test_compaction_record.py`, which drives the same router for #1078. The
    review of that round refused on exactly this point: the copy's `__globals__`
    subtlety is the thing a test gets wrong silently, and it existed twice.
    """
    from tests._messages_copy import load_messages_copy

    return load_messages_copy(monkeypatch, name="messages_under_skill_test")


async def _drive_run_turn(tmp_path, monkeypatch, *, prefetched, events,
                          fail_first_append=False, raise_after=False,
                          session_id="20260924_090000_webchat"):
    """Run the real `_run_turn` over a stubbed harness `run_query`.

    Everything above the loop is the route's own payload; everything below it is
    the real persistence. Only `run_query` — the generator the loop consumes — is
    replaced, so the `skills=skill_deliveries(prefetched_text)` line is executed
    by the function under test rather than quoted at it, and a name that is not in
    scope there raises instead of passing.

    `fail_first_append` raises out of the answer's persistence call, which lands
    a turn that has already booked its row in the `except` arm; `raise_after`
    makes the harness raise once `events` are spent, before any `result`.
    """
    import asyncio

    from app.harness.options import RunOptions
    from app.sessions_io import SessionQueue, SessionTurn

    msg = _pristine_messages_module(monkeypatch)

    monkeypatch.setattr(msg, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(msg._event_log, "log_event", lambda *a, **k: None)

    async def _no_observer(*a, **k):
        return None

    monkeypatch.setattr(msg, "attach_observer_for_turn", _no_observer)
    monkeypatch.setattr(msg, "_build_state_anchor", lambda *a, **k: "")

    seen: dict[str, object] = {}

    async def _fake_run_query(harness_messages, options):
        seen["harness_messages"] = harness_messages
        seen["options"] = options
        for evt in events:
            yield evt
        if raise_after:
            raise RuntimeError("simulated harness failure mid-turn")

    monkeypatch.setattr(msg, "run_query", _fake_run_query)

    if fail_first_append:
        real_append = msg._append_messages
        state = {"failed": False}

        async def _flaky_append(session_id_, entries):
            # The turn's earlier append is the user message, before the loop. The
            # one to break is the answer's: that is where the result path stands
            # with `final_persisted` still false and its usage row already booked,
            # which is what lands the turn in the `except` arm.
            if not state["failed"] and any(
                    e.get("role") == "assistant" for e in entries):
                state["failed"] = True
                raise OSError("simulated persistence fault")
            return await real_append(session_id_, entries)

        monkeypatch.setattr(msg, "_append_messages", _flaky_append)

    meta_path = tmp_path / f"{session_id}.json"
    meta_path.write_text(json.dumps({"messages": [], "model": "primary"}))
    turn = SessionTurn(
        turn_id="1",
        source="user",
        payload={
            "text": "what does the skill say",
            "prefetched_text": prefetched,
            "model": "primary",
            "options": RunOptions(model="primary", max_turns=60),
            "meta_path": meta_path,
            "deadline_seconds": 0.0,
        },
        enqueued_at=None,
    )
    q = SessionQueue()
    q.cancel_event = asyncio.Event()
    await msg._run_turn(session_id, turn, q)
    return seen


def _usage_rows():
    return usage_store._conn().execute(
        "SELECT session_id, skills, input_tokens, output_tokens, cache_create, "
        "cache_read FROM usage ORDER BY id").fetchall()


def test_a_real_turn_through_the_stream_writer_records_the_skill_dimension(
        tmp_path, monkeypatch):
    """#783 clause 3, driven rather than quoted. `post_message_stream` prefetches
    (`messages.py:1931`), puts `prefetched_text` in the turn payload
    (`messages.py:2055`), and `_session_consumer` runs `_run_turn`, whose harness
    loop consumes `run_query` (`messages.py:1029`) and books the turn on the
    `result` event.

    Driving that function is what proves the argument is in scope where it is
    used: the AST test grades the call's shape, and a name error at the end of a
    turn is swallowed by the broad `except` that only logs a warning, so a broken
    writer would leave every live turn silently skill-less.

    Asserted: the row `usage.db` holds after a turn whose context carried one full
    body and one excerpt; that the text the writer parsed is the text handed to
    the harness as the user message; and that exactly one row was written, so this
    is the result path's row and not the error path's.
    """
    import asyncio

    prefetched = _real_prefetch_text("full protocol body\n", "one excerpt line\n")
    seen = asyncio.run(_drive_run_turn(
        tmp_path, monkeypatch, prefetched=prefetched,
        events=[{"type": "text_delta", "text": "working"}, _fake_result_event()]))

    # Positive control: the prompt the writer read is the prompt the model got.
    user_contents = [m["content"] for m in seen["harness_messages"]
                     if m.get("role") == "user"]
    assert prefetched in user_contents, (
        "the prefetched text the writer parses is not the text the harness was "
        "handed, so the recorded route would not describe this turn"
    )

    rows = _usage_rows()
    assert len(rows) == 1, (
        f"a clean turn must write exactly one usage row; got {len(rows)} — more "
        "than one means the turn fell into the error path, whose row is written "
        "by different code"
    )
    row = rows[0]
    assert row["session_id"].startswith("20260924_090000_webchat"), row["session_id"]
    assert json.loads(row["skills"]) == [
        {"name": FULL_SKILL, "route": "prefetch"},
        {"name": EXCERPT_SKILL, "route": "prefetch_excerpt"},
    ], row["skills"]
    assert (row["input_tokens"], row["output_tokens"]) == (4000, 50), dict(row)
    assert (row["cache_create"], row["cache_read"]) == (100, 3000), dict(row)
    assert usage_store.skill_breakdown(hours=24) == [
        {"skill": FULL_SKILL, "route": "prefetch", "requests": 1,
         "input_tokens": 4000, "output_tokens": 50,
         "cache_create": 100, "cache_read": 3000},
        {"skill": EXCERPT_SKILL, "route": "prefetch_excerpt", "requests": 1,
         "input_tokens": 4000, "output_tokens": 50,
         "cache_create": 100, "cache_read": 3000},
    ], "the live row a turn wrote must appear in the published breakdown"


def test_the_stream_error_path_still_names_the_skill_deliveries(tmp_path, monkeypatch):
    """A turn that dies before its `result` event is booked by a different
    `record_usage` call (`_run_turn`'s `_book_usage`, D12) than the result
    path's, and that call has the same argument in scope only by coincidence of
    editing. The harness here streams one iteration and then raises, which is
    the shape of a turn that died mid-flight.

    Without the dimension there the skill is invisible in exactly the case an
    operator investigates: the turns that broke.
    """
    import asyncio

    prefetched = _real_prefetch_text("full protocol body\n", "one excerpt line\n")
    asyncio.run(_drive_run_turn(
        tmp_path, monkeypatch, prefetched=prefetched, raise_after=True,
        events=[{"type": "text_delta", "text": "a partial answer"},
                {"type": "assistant_message", "iteration": 1,
                 "usage": {"input_tokens": 4000, "output_tokens": 50}}]))

    rows = _usage_rows()
    assert len(rows) == 1, (
        "expected the error path's row; the fault did not reach the `except` "
        f"arm and this test proved nothing: {len(rows)}"
    )
    assert json.loads(rows[0]["skills"]) == [
        {"name": FULL_SKILL, "route": "prefetch"},
        {"name": EXCERPT_SKILL, "route": "prefetch_excerpt"},
    ], f"the error writer dropped the skill dimension: {rows[0]['skills']}"


def test_a_fault_after_the_result_row_does_not_book_the_turn_twice(tmp_path, monkeypatch):
    """A persistence fault after the `result` handler has booked its row lands
    the turn in the `except` arm with `stream_stats` already full. That used to
    write a second row for the same turn (this file pinned it as two); D12 makes
    the booking once-only, so the result row is the only one."""
    import asyncio

    prefetched = _real_prefetch_text("full protocol body\n", "one excerpt line\n")
    asyncio.run(_drive_run_turn(
        tmp_path, monkeypatch, prefetched=prefetched, fail_first_append=True,
        events=[{"type": "text_delta", "text": "a partial answer"},
                _fake_result_event()]))

    rows = _usage_rows()
    assert len(rows) == 1, f"one turn, one usage row; got {len(rows)}"
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (4000, 50)
